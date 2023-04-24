import torch.nn as nn
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from collections import Counter
from torch.utils.data import DataLoader
from functools import partial

# set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#TODO the model is unrolled 35 times

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

def collect_fn(data, vocab, tokenizer):
    text = [torch.tensor([vocab[token] for token in tokenizer(line)], dtype=torch.long) for line in data.splitlines()] 
    # Create a tensor of the same length as the longest sentence in the batch
    # each row is a sentence, each column is a word
    return nn.utils.rnn.pad_sequence(text, batch_first=True)

def create_data_loaders(train_data, validation_data, test_data, batch_size, vocab):
    collect_fn_with_params = partial(collect_fn, vocab=vocab, tokenizer=get_tokenizer('basic_english'))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collect_fn_with_params)
    val_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, collate_fn=collect_fn_with_params)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collect_fn_with_params)
    return train_loader, val_loader, test_loader

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
    # The vocab is built from the training data
    # If a word is missing from the training data, it will be replaced with <unk>
    train_data, valid_data, test_data =  load_data()
    vocab = build_vocab(train_data)
    train_loader, valid_loader, test_loader = create_data_loaders(train_data, valid_data, test_data, vocab, BATCH_SIZE)
    model = LstmRegularized(input_size=len(vocab), hidden_size=HIDDEN_SIZE, output_size=len(vocab), num_layers=NUM_LAYERS, dropout=DROPOUT)

if __name__ == "__main__":
    main()