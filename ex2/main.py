# Load data from zip PTB.zip
from zipfile import ZipFile

import torch.nn as nn
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from collections import Counter

# set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_LAYERS = 2
HIDDEN_SIZE = 200
INPUT_SIZE = 100
OUTPUT_SIZE = 100
DROPOUT = 0.2

def load_data():
    train_data = open('PTB/ptb.train.txt', 'r').read()
    valid_data = open('PTB/ptb.valid.txt', 'r').read()
    test_data = open('PTB/ptb.test.txt', 'r').read()
    return train_data, valid_data, test_data 

# define tokenizer and build vocabulary

def build_vocab(data) -> Vocab:
    """
    Build a vocabulary from the data
    """
    tokenizer = get_tokenizer('basic_english')
    counter = Counter()
    for line in data.splitlines():
        counter.update(tokenizer(line))
    vocab = Vocab(counter)
    return vocab

# dropout is between the layers - not the recurrent connections

class LstmRegularized(nn.Module):
    
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
    a, b, c =  load_data()
    vocab = build_vocab(a)
    model = LstmRegularized(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS, DROPOUT)

if __name__ == "__main__":
    main()