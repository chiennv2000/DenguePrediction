import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Stacked_LSTM(nn.Module):
    def __init__(self, args):
        super(Stacked_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=args.n_features, hidden_size=args.hidden_size, num_layers=args.n_layers, batch_first=True)
        self.linear = nn.Linear(args.hidden_size, 1)
    
    def forward(self, X):
        output, (last_hidden, _) = self.lstm(X)
        last_hidden_vector = output[:, -1, :]

        return self.linear(last_hidden_vector)
    
    def predict(self, X):
        X = torch.tensor(X)
        return self.forward(X).squeeze()
    

class LSTM_and_Attention(nn.Module):
    def __init__(self, args):
        super(LSTM_and_Attention, self).__init__()
        self.lstm = nn.LSTM(input_size=args.n_features, hidden_size=args.hidden_size, num_layers=args.n_layers, batch_first=True)
        self.attention_linear = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear = nn.Linear(args.hidden_size*2, 1)
    
    def forward(self, X):
        output, (last_hidden, _) = self.lstm(X)
        last_hidden_vector = output[:, -1, :]
        # Attention
        remain_hidden_vector = output[:, :-1, :]
        e_t = remain_hidden_vector.bmm(self.attention_linear(last_hidden_vector).unsqueeze(2)).squeeze()

        alpha_t = F.softmax(e_t, dim=1)
        attenion_vector = remain_hidden_vector.transpose(2, 1).bmm(alpha_t.unsqueeze(2)).squeeze()

        combine_vector = torch.cat((last_hidden_vector, attenion_vector), dim=1)
        return self.linear(combine_vector)
    
    def predict(self, X):
        with torch.no_grad():
            X = torch.tensor(X)
        return self.forward(X).squeeze()

class CNN(nn.Module):
    def __init__(self, n_out=6,dropout=0.1):
        super(CNN, self).__init__()
        self.flat = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv2d(1, n_out, 2)
        self.fc = nn.Linear(n_out*2*2, 1)
    
    def forward(self, input):
        x = input.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.dropout(self.flat(x))
        output = self.fc(x)
        return output
    
    def predict(self, X):
        with torch.no_grad():
            X = torch.tensor(X)
        return self.forward(X).squeeze()
    

class PositionalEncoder(nn.Module):
    def __init__(self, d_model=3, n_feature=3, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(n_feature, d_model)
        for pos in range(n_feature):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos/(10000**(2*i/d_model)))
                if i + 1 < d_model:
                    pe[pos, i+1] = math.cos(pos/(10000**((2*i+1)/d_model)))
        pe = pe.unsqueeze(0)        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x*math.sqrt(self.d_model)
        length = x.size(1)
        pe = Variable(self.pe[:, :length], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        x = self.dropout(x)
        
        return x

class TransformerModel(nn.Module):
    def __init__(self, d_input, n_head=3, hidden_size=384, n_layers=2, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.hidden_size = hidden_size
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_input, nhead=n_head, dim_feedforward=hidden_size, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.decoder = nn.Linear(d_input, 1)
        self.pe = PositionalEncoder(dropout=dropout)
    
    def forward(self, input):
        #input = self.pos_encoder(input)
        input = self.pe(input)
        output = self.transformer_encoder(input)
        output = output.mean(dim=-2)
        output = self.decoder(output)
        return output
    
    def predict(self, X):
        with torch.no_grad():
            X = torch.tensor(X)
        return self.forward(X).squeeze()
    





