from __future__ import absolute_import, division, print_function

import argparse
import pandas as pd

import torch
from torch.utils.data import DataLoader, TensorDataset

from preprocess import preprocess_data
from training import initialize_model, train
from utils import get_data


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_path",
                        type=str,
                        default='./data/full_cleaned_dataset.xlsx')
    
    parser.add_argument("--city_name",
                        type=str,
                        default='Hà Nội')
    
    parser.add_argument("--look_back",
                        type=int,
                        default=3)
    
    parser.add_argument("--n_features",
                        type=int,
                        default=3)
    
    parser.add_argument("--batch_size",
                        type=int,
                        default=16)
    
    parser.add_argument("--epoch",
                        type=int,
                        default=100)
    
    parser.add_argument("--hidden_size",
                        type=int,
                        default=384)
    
    parser.add_argument("--n_layers",
                        type=int,
                        default=16)
    
    parser.add_argument("--learning_rate",
                        type=float,
                        default=1e-3)
    
    parser.add_argument("--type_model",
                        type=str,
                        default="attention_lstm",
                        help="lstm, lstm_attention, cnn or transformer")
    
    
    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    
    full_data = pd.read_excel(args.data_path)
    data = preprocess_data(full_data, city=args.city_name, disease='Dengue_fever')
    train_X, train_y, test_X, test_y = get_data(data, look_back=args.look_back, k_feature=args.n_features)

    train_tensor = TensorDataset(torch.tensor(train_X), torch.tensor(train_y))
    test_tensor = TensorDataset(torch.tensor(test_X), torch.tensor(test_y))

    train_loader = DataLoader(train_tensor, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_tensor, batch_size=args.batch_size, shuffle=False)
    
    model_lstm, optimizer, loss_fn = initialize_model(type_model=args.type_model, args=args)
    model_lstm.to(device)
    print("Starting training...")
    best_lstm_model = train(model_lstm, optimizer, loss_fn, train_loader, test_loader, epochs=args.epoch)
    #plot_entire_chart(best_lstm_model, train_X, train_y, test_X, test_y, 'LSTM/' + city_name)

main()