import torch.nn as nn
import torch
from collections import Counter
from torch.utils.data import DataLoader
from typing import List, Tuple, Any
from torch.utils.tensorboard import SummaryWriter
import itertools
from pathlib import Path
from torch.optim.lr_scheduler import StepLR

# set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#TODO rename word_log_probs to output
#TODO the model is unrolled 35 times
#TODO (successive minibatches sequentially traverse the training set
#TODO its parameters are initialized uniformly in [−0.05, 0.05]
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
NUM_EPOCHS = 2
NUM_BATCHES = 10000
# Dropout From the paper: We apply dropout on non-recurrent connections of the LSTM
DROPOUT = 0.5

# Not sure about this one
SEQUENCE_LENGTH = 35

# Like in the paper
CLIP_GRADIENT_VALUE = 5

SAVED_MODELS_DIR = 'saved_models'

LOG_BATCH_INTERVAL = 600
NUMBER_OF_BATCHES_FOR_LOSS = 1000

def load_data():
    train_data = open('PTB/ptb.train.txt', 'r').read()
    valid_data = open('PTB/ptb.valid.txt', 'r').read()
    test_data = open('PTB/ptb.test.txt', 'r').read()
    return preprocess_data(train_data), preprocess_data(valid_data), preprocess_data(test_data) 

# define tokenizer and build vocabulary

def preprocess_data(text_data: str):
    """ Remove newlines and replace them with <eos> token """
    # Not sure if I should do this
    text_data = text_data.lower()
    text_data = text_data.replace('\r\n', '<eos>')
    text_data = text_data.replace('\n', '<eos>')
    return text_data

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
    print("The total number of words is {}".format(len(words)))
    print("found {} unique tokens in training data".format(len(counter)))
    print(f"The 30 most common words are {counter.most_common(30)}")
    
    return {word: index for index, word in enumerate(counter.keys())}

class PennTreeBankDataset(torch.utils.data.Dataset):
    def __init__(self, raw_data: str, vocab: dict, seq_length: int):
        self.raw_data = raw_data
        self.vocab = vocab
        self.encoded_text = self.text_to_vocab(raw_data, vocab)
        self.seq_length = seq_length
    
    def text_to_vocab(self, text: str, vocab: dict):
        """
        Convert the data to a vector of indices
        """
        return [vocab[word] for word in tokenize(text)]

    def __getitem__(self, index):
        """
        get the sentence at the index, then shift it by one word to the right to get the target sentence
        """
        return torch.tensor(self.encoded_text[index:index+self.seq_length] ,dtype=torch.long, device=device), \
            torch.tensor(self.encoded_text[index+1:index+self.seq_length+1] ,dtype=torch.long, device=device)
        
    def __len__(self):
        return len(self.encoded_text) - self.seq_length + 1


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
    train_loader = DataLoader(PennTreeBankDataset(train_data, vocab, SEQUENCE_LENGTH), batch_size=batch_size, shuffle=True, collate_fn=pad_sentence_and_target)
    val_loader = DataLoader(PennTreeBankDataset(validation_data, vocab, SEQUENCE_LENGTH), batch_size=batch_size, shuffle=True, collate_fn=pad_sentence_and_target)
    test_loader = DataLoader(PennTreeBankDataset(test_data, vocab, SEQUENCE_LENGTH), batch_size=batch_size, shuffle=True, collate_fn=pad_sentence_and_target)
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
        self.loss_function = nn.CrossEntropyLoss()
        # Cast the LSTM weights and biases to long data type
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.description = f"{model}_Model_learning_{LEARNING_RATE}_dropout_{dropout}"
        self.writer = SummaryWriter(self.description)
        # self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.1)
        #TODO consider adding nn.Dropout - maybe the dropout in LSTM is not like the one in the paper
    
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
        return output, hidden, output_lengths
    
    def calculate_perplexity(self, data_loader: DataLoader) -> float:
        total_perplexity = 0
        self.rnn_cell.eval()
        # Sampling batches to calculate the perplexity faster
        with torch.no_grad():
            for i, (sentence, target_sentence, lengths) in enumerate(data_loader):
                word_log_probabilities, _ , _= self.forward(sentence, None, lengths)
                _, perplexity = self.calculate_perplexity_of_sentence(word_log_probabilities, target_sentence)
                total_perplexity += perplexity
                if i == NUMBER_OF_BATCHES_FOR_LOSS:
                    break
        self.rnn_cell.train()
        return total_perplexity / NUMBER_OF_BATCHES_FOR_LOSS

    def calculate_perplexity_of_sentence(self, word_log_probabilities, target_sentence) -> Tuple[Any, float]:
        
        word_log_probabilities = word_log_probabilities.transpose(1, 2)
        
        loss = self.loss_function(word_log_probabilities, target_sentence)
        return loss, torch.exp(loss)

    def train(self, train_loader, valid_loader, test_loader, num_epochs):
        # Epoch iterations
        prev_perplexity = float("inf")
        for epoch in range(num_epochs):
            # Initialize hidden state
            hidden_states = None
            # Batch iterations
            print(f"Starting to Traing Epoch: {epoch} for {len(train_loader)} batches")
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
                if batch_number % LOG_BATCH_INTERVAL == 0:
                    test_perplexity = self.calculate_perplexity(test_loader)
                    train_perplexity = self.calculate_perplexity(train_loader)
                    print("<LOGGING> Train perplexity {}, test perplexity {}, batch number {}".format(train_perplexity, test_perplexity, batch_number))
                    self.writer.add_scalars("perplexity", {"train": train_perplexity, "test": test_perplexity}, epoch * len(train_loader) + batch_number)
                    if test_perplexity < prev_perplexity:
                        print("Saving model state because test perplexity is lower than previous one")
                        torch.save(self.state_dict(),  Path(SAVED_MODELS_DIR) / Path(self.description + ".pth"))
                        prev_perplexity = test_perplexity
                        return
                
                if batch_number >= NUM_BATCHES:
                    print("Finished training")
                    break
                loss.backward()
                nn.utils.clip_grad_value_(self.parameters(), clip_value=CLIP_GRADIENT_VALUE)
                self.optimizer.step()

                # Using the hidden states of the last batch as the intializor
                # We need to detach the hidden_states so the it wont be traersed by the backward
                hidden_states = (hidden_states[0].detach(), hidden_states[1].detach())

            print(f"Epoch {epoch} is done, perplexity on test set is: {test_perplexity}, perplexity on train set is: {train_perplexity}") 

def main():
    # The vocab is built from the training data
    # If a word is missing from the training data, it will be replaced with <unk>
    train_data, valid_data, test_data = load_data()
    vocab = build_vocab(train_data)
    train_loader, valid_loader, test_loader = create_data_loaders(train_data, valid_data, test_data, BATCH_SIZE, vocab)

    models = ["lstm", "gru"]
    dropouts = [0, DROPOUT]
    for model_name, dropout in itertools.product(models, dropouts):
        model = RnnRegularized(EMBEDDING_SIZE, len(vocab), HIDDEN_SIZE, len(vocab), NUM_LAYERS, dropout, BATCH_SIZE, model_name).to(device)
        print(f"Starting to train model {model.description}")
        model.train(train_loader, valid_loader, test_loader, NUM_EPOCHS)

if __name__ == "__main__":
    main()



