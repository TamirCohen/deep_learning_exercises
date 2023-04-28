import torch.nn as nn
import torch
from collections import Counter
from torch.utils.data import DataLoader
from typing import List, Tuple
# set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#TODO amazing reference https://github.com/pytorch/examples/blob/e11e0796fc02cc2cd5b6ec2ad7cea21f77e25402/word_language_model/main.py#L104
#TODO the model is unrolled 35 times
#TODO (successive minibatches sequentially traverse the training set
#TODO its parameters are initialized uniformly in [−0.05, 0.05]
#TODO should add <eos> token to the end of each sentence?
#TODO should I remove the <unk> token? and the , and .?

EMBEDDING_SIZE = 100
# number of layers in the LSTM - as specified in the paper
NUM_LAYERS = 2
# The size of the hidden state of the LSTM - as specified in the paper
HIDDEN_SIZE = 200
# not sure
INPUT_SIZE = 1
# Size of the minibatch as specified in the paper
BATCH_SIZE = 20
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
DROPOUT = 0.2

def load_data():
    train_data = open('PTB/ptb.train.txt', 'r').read()
    valid_data = open('PTB/ptb.valid.txt', 'r').read()
    test_data = open('PTB/ptb.test.txt', 'r').read()
    return train_data, valid_data, test_data 

# define tokenizer and build vocabulary

def tokenize(text_data: str):
    """
    Tokenize the data
    """
    return text_data.split()

def build_vocab(text_data: str):
    """
    Build a vocabulary from the data
    ie: mapping between a word and an index
    """
    words = tokenize(text_data)
    counter = Counter()
    counter.update(words)
    return {word: index for index, word in enumerate(counter.keys())}

class PennTreeBankDataset(torch.utils.data.Dataset):
    def __init__(self, raw_data: str, vocab: dict):
        self.raw_data = raw_data
        self.vocab = vocab
        self.sentences = self.raw_data.splitlines()
        
    def __getitem__(self, index):
        """
        get the sentence at the index, then shift it by one word to the right to get the target sentence
        """
        sentence_vocab = self.sentence_to_vocab(tokenize(self.sentences[index]), self.vocab)
        sentence_tensor = torch.tensor(sentence_vocab, dtype=torch.long, device=device)
        # shift sentence by one word
        target_vocab = sentence_vocab[1:] + [0]
        target_tensor = torch.tensor(target_vocab, dtype=torch.long, device=device)
        return sentence_tensor, target_tensor
        
    def __len__(self):
        return len(self.sentences)

    def sentence_to_vocab(self, words: str, vocab: dict):
        """
        Convert the data to a vector of indices
        """
        return [vocab[word] for word in words]

def pad_sentence_and_target(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    get tuples of sentence and target, then pad each of them to the maximum length
    """
    
    sentences = [item[0] for item in batch]
    target_sentences = [item[1] for item in batch]

    lengths = torch.LongTensor([len(seq) for seq in sentences])
    # pad sequences
    padded_sentences = torch.nn.utils.rnn.pad_sequence(sentences, batch_first=True)
    padded_targets = torch.nn.utils.rnn.pad_sequence(target_sentences, batch_first=True)
    #TODO should I return this as a tuple? 
    return padded_sentences, padded_targets, lengths

def create_data_loaders(train_data: str, validation_data: str, test_data: str, batch_size: int, vocab: dict):
    train_loader = DataLoader(PennTreeBankDataset(train_data, vocab), batch_size=batch_size, shuffle=True, collate_fn=pad_sentence_and_target)
    val_loader = DataLoader(PennTreeBankDataset(validation_data, vocab), batch_size=batch_size, shuffle=False, collate_fn=pad_sentence_and_target)
    test_loader = DataLoader(PennTreeBankDataset(test_data, vocab), batch_size=batch_size, shuffle=False, collate_fn=pad_sentence_and_target)
    return train_loader, val_loader, test_loader

class LstmRegularized(nn.Module):
    #TODO not tested yet :( - and not working probably
    def __init__(self, embedding_size, vocab_size, hidden_size, output_size, num_layers, dropout, batch_size):
        super(LstmRegularized, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers  = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size)
        self.lstm = torch.nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.loss_function = torch.nn.functional.nll_loss
        # Cast the LSTM weights and biases to long data type
        self.optimizer = torch.optim.SGD(self.parameters(), lr=LEARNING_RATE)
    
    def forward(self, input, hidden):
        """
        The input is a long tensor of size (batch_size, seq_len) which contains the indices of the words in the sentence
        """
        #TODO init word2vec embedding
        embedding = self.embedding(input)
        output, hidden = self.lstm(embedding, hidden)
        output = self.linear(output)
        output = self.softmax(output)
        return output, hidden
    
    def evaluate_model(self, data_loader: DataLoader) -> float:
        correct = 0
        total = 0
        #TODO consider testing all the shiftings of the sentence
        with torch.no_grad():
            for sentence, target_sentence, _ in data_loader:
                try:
                    self.lstm.eval()
                    
                    word_output_probabilities, _ = self.forward(sentence, None)
                    # get the index of the most probable word
                    predicted_words = torch.argmax(word_output_probabilities, dim=2)
                    correct += (predicted_words == target_sentence).sum().item()
                    # increase the total number of words in the target sentence
                    total += torch.numel(target_sentence)
                finally:
                    self.lstm.train()

        return correct / total

    def train(self, train_loader, valid_loader, test_loader, num_epochs):
        # Epoch iterations
        for epoch in range(num_epochs):
            # Initialize hidden state
            hidden_states = None
            # Batch iterations
            for batch_number, (sentence, target_sentence, lengths) in enumerate(train_loader):
                # Added the dimensiton of the word embedding (Which is one in this case)
                # Transpose so it will contain the correct dimensions for the LSTM
                # sentence = sentence.unsqueeze(-1)
                # Model unrolling iterations
                #TODO it should be 35 - validate it
                
                # Skip partial batches
                if sentence.shape[0] != self.batch_size:
                    continue
                self.optimizer.zero_grad()

                word_output_probabilities, hidden_states = self.forward(sentence, hidden_states)
                word_output_probabilities = word_output_probabilities.view(-1, self.output_size)
                target_sentence = target_sentence.view(-1)

                #TODO validate the loss calculation
                loss = self.loss_function(word_output_probabilities, target_sentence)
                if batch_number % 100 == 0:
                    print(f"Training loss for batch {batch_number} epoch {epoch} is {loss}")
                loss.backward()
                self.optimizer.step()

                # Using the hidden states of the last batch as the intializor
                # We need to detach the hidden_states so the it wont be traersed by the backward
                hidden_states = (hidden_states[0].detach(), hidden_states[1].detach())

            print(f"Epoch {epoch} is done, accuracy on validation set is: {self.evaluate_model(valid_loader)}") 

def main():
    # The vocab is built from the training data
    # If a word is missing from the training data, it will be replaced with <unk>
    train_data, valid_data, test_data = load_data()
    vocab = build_vocab(train_data)
    
    train_loader, valid_loader, test_loader = create_data_loaders(train_data, valid_data, test_data, BATCH_SIZE, vocab)
    lstm_model = LstmRegularized(EMBEDDING_SIZE, len(vocab), HIDDEN_SIZE, len(vocab), NUM_LAYERS, DROPOUT, BATCH_SIZE)
    lstm_model.to(device)
    lstm_model.train(train_loader, valid_loader, test_loader, NUM_EPOCHS)

if __name__ == "__main__":
    main()