import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd

import os
from tqdm.notebook import tqdm

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score as roc_score
from sklearn.metrics import precision_recall_curve, auc

from sklearn.utils import class_weight as class_weight_f
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader


def get_loaders(data_train, y_train, data_valid, y_valid, batch_size=4096, use_scaling=True):
    if use_scaling:
        sc = StandardScaler().fit(data_train)
        data_train_scaled = sc.transform(data_train)
        data_valid_scaled = sc.transform(data_valid)
    else:
        data_train_scaled = data_train.copy()
        data_valid_scaled = data_valid.copy()

    class_weight = class_weight_f.compute_class_weight(class_weight='balanced',
                                                       classes=np.unique(y_train), y=y_train)
    print(f'class_weight: {class_weight}')

    def to_tensor(input):
        return torch.from_numpy(input).to(torch.float32)

    train_pt = to_tensor(data_train_scaled)
    valid_pt = to_tensor(data_valid_scaled)

    y_train_pt = to_tensor(y_train).reshape(-1, 1)
    y_valid_pt = to_tensor(y_valid).reshape(-1, 1)

    train_dataset = TensorDataset(train_pt, y_train_pt)
    valid_dataset = TensorDataset(valid_pt, y_valid_pt)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    loaders = {'train': train_loader, 'valid': valid_loader}
    return loaders, class_weight


class Classifier(nn.Module):
    def __init__(self, n_inputs=5):
        super().__init__()
        self.layers = [
            nn.Linear(in_features=n_inputs, out_features=64, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1, bias=True),
            nn.Sigmoid(),
        ]
        self.model_stack = nn.Sequential(*self.layers)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        return self.model_stack(x.to(self.device))

    def predict(self, x, numpy_input=False):
        with torch.no_grad():
            self.eval()
            if numpy_input:
                x = torch.tensor(x)
            prediction = self.forward(x).detach().cpu().numpy()
        return prediction


def init_model(n_inputs=5):
    model = Classifier(n_inputs)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return model.to(model.device), optimizer


def get_batch_weights(batch, class_weights):
    batch_labels = batch[1]
    if class_weights is not None:
        id_matrix = torch.ones(batch_labels.shape)
        batch_weights = (id_matrix - batch_labels) * class_weights[0] \
                                    + batch_labels * class_weights[1]
    else:
        batch_weights = None
    return batch_weights


def train_model(model, optimizer, loaders: dict, n_epochs=100, class_weights=None, verbose=False, save_dir=None):
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    train_loss = np.zeros(n_epochs)
    valid_loss = np.zeros(n_epochs)
    loss_func = F.binary_cross_entropy
    for epoch in tqdm(range(n_epochs), 'epoch'):
        if verbose:
            print(f'Epoch: {epoch + 1}')
        epoch_train_loss = 0.
        epoch_valid_loss = 0.

        model.train()
        for i, batch in enumerate(loaders['train']):
            if verbose:
                print("\t...batch num", i)

            optimizer.zero_grad()

            batch_inputs, batch_labels = batch[0], batch[1]
            batch_outputs = model(batch_inputs.to(model.device)).cpu()
            batch_weights = get_batch_weights(batch, class_weights)
            batch_loss = loss_func(batch_outputs, batch_labels, weight=batch_weights)

            batch_loss.backward()
            optimizer.step()

            epoch_train_loss += batch_loss.item()
            if verbose:
                print("\t...batch training loss:", batch_loss.item())

        epoch_train_loss /= (i + 1)
        if verbose:
            print("training loss:", epoch_train_loss)

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(loaders['valid']):
                batch_inputs, batch_labels = batch[0], batch[1]
                batch_outputs = model(batch_inputs.to(model.device)).cpu()
                batch_weights = get_batch_weights(batch, class_weights)
                batch_loss = loss_func(batch_outputs, batch_labels, weight=batch_weights)
                epoch_valid_loss += batch_loss.item()

            epoch_valid_loss /= (i + 1)
            if verbose:
                print("validation loss:", epoch_valid_loss)

        train_loss[epoch] = epoch_train_loss
        valid_loss[epoch] = epoch_valid_loss

        if save_dir is not None:
            torch.save(model, os.path.join(save_dir, f"ep{epoch + 1}_model.pt"))
        if verbose:
            print()

    return train_loss, valid_loss


def minimum_validation_loss_models(clf_dir, n_epochs):
    print(n_epochs)
    valid_loss_matrix = np.load(os.path.join(clf_dir, 'valid_loss_matrix.npy'))
    n_runs = valid_loss_matrix.shape[0]
    
    model_paths = []
    for i in range(n_runs):
        min_val_loss_epochs = np.argpartition(valid_loss_matrix[i, :], n_epochs)[:n_epochs]
        print(f"Run {i + 1} minimum validation loss epochs: {min_val_loss_epochs}")
        model_paths.append(
            [os.path.join(clf_dir, 'runs', f'run_{i + 1}', f'ep{ep + 1}_model.pt')
                    for ep in min_val_loss_epochs]
            )
    return model_paths


def train_classifier(loaders, class_weight, clf_dir, n_epochs=100, n_runs=5):
    clf_dir = os.path.join(os.getcwd(), clf_dir)

    train_run_losses = {}
    valid_run_losses = {}
    for run in range(n_runs):
        print(f'Run {run + 1}...')
        model, optimizer = init_model(n_inputs=4)
        save_dir = os.path.join(clf_dir, 'runs', f'run_{run + 1}')
        train_loss, valid_loss = train_model(model, optimizer, loaders, n_epochs, class_weight, save_dir=save_dir)
        train_run_losses[run + 1] = train_loss
        valid_run_losses[run + 1] = valid_loss

    def get_loss_mat(runs_losses_dict):
        runs_losses = list(runs_losses_dict.values())
        return np.asarray(runs_losses)

    train_loss_matrix = get_loss_mat(train_run_losses)
    valid_loss_matrix = get_loss_mat(valid_run_losses)

    np.save(os.path.join(clf_dir, 'train_loss_matrix.npy'), train_loss_matrix)
    np.save(os.path.join(clf_dir, 'valid_loss_matrix.npy'), valid_loss_matrix)

    model_paths = minimum_validation_loss_models(clf_dir, n_epochs=10)
    return model_paths


def preds_from_model(X_test, scaler, model_path_list, save_dir=None, take_mean=True):
    X_test = torch.from_numpy(scaler.transform(X_test[:, 1:-1])).to(torch.float32)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    run_predictions = []
    for i, model_paths in enumerate(model_path_list):  # looping over runs
        model_list = [torch.load(model_path, map_location=device) for model_path in model_paths]
        epoch_predictions = []

        for model in model_list:  # looping over epochs
            model.eval()
            preds = model.predict(X_test, numpy_input=False).flatten()
            epoch_predictions.append(preds)

        run_predictions.append(np.stack(epoch_predictions))

    preds_matrix = np.stack(run_predictions)  # (n_runs, n_epochs, N)

    if take_mean:  # mean by epochs
        preds_matrix = np.mean(preds_matrix, axis=1, keepdims=True)  # (n_runs, 1, N)

    if save_dir is not None:
        np.save(os.path.join(save_dir, 'preds_matrix.npy'), preds_matrix)
    return preds_matrix

