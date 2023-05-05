import torch.nn as nn
import torch
from collections import Counter
from torch.utils.data import DataLoader
from typing import List, Tuple, Any
from torch.utils.tensorboard import SummaryWriter
import itertools

# set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#TODO the model is unrolled 35 times
#TODO (successive minibatches sequentially traverse the training set
#TODO its parameters are initialized uniformly in [−0.05, 0.05]
#TODO should add <eos> token to the end of each sentence?
#TODO should I remove the <unk> token? and the , and .?

# TODO PARAMETERS TO PLAY WITH - LEARNING RATE, EMBEDDING SIZE, HIDDEN SIZE 

# Chat GPT said thath 300 is a good embedding size
EMBEDDING_SIZE = 300
# number of layers in the LSTM - as specified in the paper
NUM_LAYERS = 2
# The size of the hidden state of the LSTM - as specified in the paper
HIDDEN_SIZE = 200
# Size of the minibatch as specified in the paper
BATCH_SIZE = 20

# Change the learning rate and perhaps the optimazation method to match 
LEARNING_RATE = 0.005
NUM_EPOCHS = 20
# Dropout From the paper: We apply dropout on non-recurrent connections of the LSTM
DROPOUT = 0.5

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
    #TODO not sure - start from 1, 0 is reserved for padding - when I tried it it failed on some Error
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

class RnnRegularized(nn.Module):
    def __init__(self, embedding_size, vocab_size, hidden_size, output_size, num_layers, dropout, batch_size, model):
        super(RnnRegularized, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers  = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size)
        if model == "lstm":
            self.rnn_cell = torch.nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        elif model == "gru":
            self.rnn_cell = torch.nn.GRU(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True)

        self.linear = torch.nn.Linear(hidden_size, output_size)
        # Dim should be equal 2 because dim 2 is the words dimension!
        self.logsoftmax = torch.nn.LogSoftmax(dim=2)
        self.loss_function = torch.nn.functional.nll_loss
        # Cast the LSTM weights and biases to long data type
        self.optimizer = torch.optim.SGD(self.parameters(), lr=LEARNING_RATE)
        self.description = f"{model}_Model_learning_{LEARNING_RATE}_dropout_{dropout}"
        self.writer = SummaryWriter(self.description)

    
    def forward(self, input, hidden, lengths):
        """
        The input is a long tensor of size (batch_size, seq_len) which contains the indices of the words in the sentence
        """
        #TODO init word2vec embedding

        embedding = self.embedding(input)
        embedding = nn.utils.rnn.pack_padded_sequence(embedding, lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn_cell(embedding, hidden)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = self.linear(output)
        # Using the logsoftmax instead of softmax - like the paper
        # To get the probabilities of the words: torch.exp(output)
        output = self.logsoftmax(output)
        return output, hidden, output_lengths
    
    def calculate_perplexity(self, data_loader: DataLoader) -> float:
        total_perplexity = 0
        self.rnn_cell.eval()
        with torch.no_grad():
            for sentence, target_sentence, lengths in data_loader:
                word_log_probabilities, _ , _= self.forward(sentence, None, lengths)
                _, perplexity = self.calculate_perplexity_of_sentence(word_log_probabilities, target_sentence)
                total_perplexity += perplexity
        self.rnn_cell.train()
        return total_perplexity / len(data_loader)

    def calculate_perplexity_of_sentence(self, word_log_probabilities, target_sentence) -> Tuple[Any, float]:
        
            word_log_probabilities = word_log_probabilities.transpose(1, 2)
            
            #TODO what to do if the sentence is short?

            #TODO not sure if to keep the ignore index
            loss = self.loss_function(word_log_probabilities, target_sentence, ignore_index=0)
            return loss, torch.exp(loss)

    def train(self, train_loader, valid_loader, test_loader, num_epochs):
        # Epoch iterations
        for epoch in range(num_epochs):
            # Initialize hidden state
            hidden_states = None
            # Batch iterations
            for batch_number, (sentence, target_sentence, lengths) in enumerate(train_loader):
                # Added the dimensiton of the word embedding (Which is one in this case)
                # Transpose so it will contain the correct dimensions for the LSTM
                # Model unrolling iterations
                #TODO it should be 35 - validate it
                
                # Skip partial batches
                if sentence.shape[0] != self.batch_size:
                    continue
                self.optimizer.zero_grad()

                word_log_probabilities, hidden_states, output_length = self.forward(sentence, hidden_states, lengths)

                #TODO use pack_padded_sequence to ignore the padding

                # shape: (batch_size, seq_len, vocab_size)
                # transpose to (batch_size, vocab_size, seq_len)
                
                loss, perplexity =  self.calculate_perplexity_of_sentence(word_log_probabilities, target_sentence)

                if batch_number % 200 == 0:
                    #TODO graph should be in perplexity
                    print(f"Training perplexity for batch {batch_number} epoch {epoch} is {perplexity}")
                loss.backward()
                self.optimizer.step()

                # Using the hidden states of the last batch as the intializor
                # We need to detach the hidden_states so the it wont be traersed by the backward
                hidden_states = (hidden_states[0].detach(), hidden_states[1].detach())

            test_perplexity = self.calculate_perplexity(test_loader)
            train_perplexity = self.calculate_perplexity(train_loader)
            self.writer.add_scalars("perplexity", {"train": train_perplexity, "test": test_perplexity}, epoch)
            
            print(f"Epoch {epoch} is done, perplexity on test set is: {test_perplexity}") 

def main():
    # The vocab is built from the training data
    # If a word is missing from the training data, it will be replaced with <unk>
    train_data, valid_data, test_data = load_data()
    vocab = build_vocab(train_data)
    train_loader, valid_loader, test_loader = create_data_loaders(train_data, valid_data, test_data, BATCH_SIZE, vocab)

    # models = ["lstm", "gru"]
    # dropouts = [0, DROPOUT]
    models = ["lstm"]
    dropouts = [0]
    for model_name, dropout in itertools.product(models, dropouts):
        model = RnnRegularized(EMBEDDING_SIZE, len(vocab), HIDDEN_SIZE, len(vocab), NUM_LAYERS, dropout, BATCH_SIZE, model_name).to(device)
        model.train(train_loader, valid_loader, test_loader, NUM_EPOCHS)

if __name__ == "__main__":
    main()



