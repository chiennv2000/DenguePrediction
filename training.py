
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from models import Stacked_LSTM, LSTM_and_Attention, CNN, TransformerModel

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

def initialize_model(type_model, args):
    if type_model.lower() == 'lstm':
        model = Stacked_LSTM(args)
    elif type_model.lower() == 'attention_lstm':
        model = LSTM_and_Attention(args)
    elif type_model.lower() == 'transformer':
        model = TransformerModel(d_input=3)
    elif type_model.lower() == 'cnn':
        model = CNN(n_out=36, dropout=0.01)
    else:
        raise ValueError("Invalid name!")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    return model, optimizer, loss_fn

def step(model, optimizer, loss_fn, batch):
    model.train()
    X_batch, y_batch = tuple(t.to(device) for t in batch)

    optimizer.zero_grad()
    y_pred = model.forward(X_batch)
    loss = loss_fn(y_pred.squeeze(), y_batch)
    
    loss.backward()
    optimizer.step()

    return loss.item()

def validation(model, loss_fn, test_loader):
    model.eval()
    eval_loss = 0.0

    result = {}

    y_true = np.array([])
    y_pred = np.array([])

    for batch in test_loader:
        X_batch, y_batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            outputs = model.forward(X_batch)
            loss = loss_fn(outputs.squeeze(), y_batch)
            eval_loss += loss.mean().item()

            y_pred = np.concatenate((y_pred, outputs.squeeze().cpu().numpy()), axis=0)
            y_true = np.concatenate((y_true, y_batch.cpu().numpy()), axis=0)

    #print(y_pred.shape, y_true.shape)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r_squard = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    result['mse'] = mse
    result['rmse'] = rmse
    result['r_squard'] = r_squard
    result['mae'] = mae

    return eval_loss/len(test_loader), result


def train(model, optimizer, loss_fn, train_loader, test_loader, epochs=20):
    best_lost = float("inf")
    best_model = None
    best_result = None
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            loss = step(model, optimizer, loss_fn, batch)
            total_loss += loss
            
        train_loss = total_loss/len(train_loader)
        eval_loss, result = validation(model, loss_fn, test_loader)
        
        if eval_loss < best_lost:
            best_lost = eval_loss
            best_model = copy.deepcopy(model)
            best_result = result
        if (epoch + 1) == epochs or (epoch + 1) in [c + 1 for c in range(epochs) if c % int(epochs/10) == 0]:
            print(f"Epoch: {epoch+1:2}/{epochs:2} - train_loss: {train_loss:.4f} - test_loss: {eval_loss:4f}")
            
    print("MSE: ", result['mse'])
    print("RMSE: ", result['rmse'])
    print("MAE: ", result['mae'])
    print("R-squard: ", result['r_squard'])
    return best_model