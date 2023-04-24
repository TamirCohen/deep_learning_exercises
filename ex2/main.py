import torch.nn as nn
import torch
from collections import Counter
from torch.utils.data import DataLoader
from functools import partial
from typing import List
# set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#TODO the model is unrolled 35 times
#TODO (successive minibatches sequentially traverse the training set
#TODO its parameters are initialized uniformly in [−0.05, 0.05]
NUM_LAYERS = 2
HIDDEN_SIZE = 200
INPUT_SIZE = 100
OUTPUT_SIZE = 100
DROPOUT = 0.2
BATCH_SIZE = 20

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
        
    def __get_item__(self, index):
        """
        get the sentence at the index, then shift it by one word to the right to get the target sentence
        """
        sentence_vocab = self.sentence_to_vocab(tokenize(self.sentences[index]), self.vocab)
        sentence_tensor = torch.tensor(sentence_vocab, dtype=torch.long)
        # shift sentence by one word
        target_vocab = sentence_vocab[1:] + [0]
        target_tensor = torch.tensor(target_vocab, dtype=torch.long)
        return sentence_tensor, target_tensor
        
    def __len__(self):
        return len(self.sentences)

    def sentence_to_vocab(self, words: str, vocab: dict):
        """
        Convert the data to a vector of indices
        """
        return [vocab[word] for word in words]

def create_data_loaders(train_data: str, validation_data: str, test_data: str, batch_size: int, vocab: dict):
    train_loader = DataLoader(PennTreeBankDataset(train_data, vocab), batch_size=batch_size, shuffle=True, collate_fn=nn.utils.rnn.pad_sequence)
    val_loader = DataLoader(PennTreeBankDataset(validation_data, vocab), batch_size=batch_size, shuffle=False, collate_fn=nn.utils.rnn.pad_sequence)
    test_loader = DataLoader(PennTreeBankDataset(test_data, vocab), batch_size=batch_size, shuffle=False, collate_fn=nn.utils.rnn.pad_sequence)
    return train_loader, val_loader, test_loader

class LstmRegularized(nn.Module):
    #TODO not tested yet :( - and not working probably
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(LstmRegularized, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers  = num_layers
        self.dropout = dropout
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout)
        self.linear = torch.nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)
    
    def forward(self, input, hidden):
        output, hidden = self.lstm(input, hidden)
        output = self.linear(output)
        output = self.softmax(output)
        return output, hidden
        
def main():
    # The vocab is built from the training data
    # If a word is missing from the training data, it will be replaced with <unk>
    train_data, valid_data, test_data = load_data()
    vocab = build_vocab(train_data)
    
    train_loader, valid_loader, test_loader = create_data_loaders(train_data, valid_data, test_data, BATCH_SIZE, vocab)

if __name__ == "__main__":
    main()