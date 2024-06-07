import os
import torch
import shutil
import torchinfo
import numpy as np
from tqdm import tqdm
from typing import Union
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


class TrainingAssistant(object):
    def __init__(self, batch_size=256, checkpoint_dir=None):
        self.batch_size = batch_size
        self.optim = None
        self.scheduler = None

        self.checkpoint_dir = checkpoint_dir
        if self.checkpoint_dir is not None:
            try:
                os.makedirs(self.checkpoint_dir)
                print(f'Created directory {self.checkpoint_dir}')
            except:
                print(f'Directory {self.checkpoint_dir} already exists or can not be created')
                pass

        self.train_losses = []
        self.valid_losses = []
        self.min_epoch_loss = torch.inf

    def set_optimizer(self, **optim_kwargs):
        self.optim = torch.optim.Adam(self.model.parameters(), **optim_kwargs)

    def set_scheduler(self, sched_type, **sched_kwargs):
        self.scheduler = sched_type(self.optim, **sched_kwargs)

    def param_summary(self, **kwargs):
        print(torchinfo.summary(self.model, **kwargs))

    def save_components(self, checkpoint_dir):
        '''Saves state dicts of model, optimizer and scheduler'''
        torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, f'model.pt'))
        torch.save(self.optim.state_dict(), os.path.join(checkpoint_dir, f'optim.pt'))
        if self.scheduler is not None:
            torch.save(self.scheduler.state_dict(), os.path.join(checkpoint_dir, f'sched.pt'))

    def checkpoint(self, epoch):
        assert self.checkpoint_dir is not None
        losses = self.valid_losses if len(self.valid_losses) else self.train_losses
        if losses[-1] < self.min_epoch_loss:
            self.min_epoch_loss = losses[-1]

            checkpoint_dir_best = os.path.join(self.checkpoint_dir, 'best')
            os.makedirs(checkpoint_dir_best, exist_ok=True)
            self.save_components(checkpoint_dir_best)

        checkpoint_dir_i = os.path.join(self.checkpoint_dir, f"{epoch}")
        os.makedirs(checkpoint_dir_i, exist_ok=True)
        self.save_components(checkpoint_dir_i)

    def load_from_checkpoint(self, postfix='best', strict=True):
        assert self.checkpoint_dir is not None
        checkpoint_dir = os.path.join(self.checkpoint_dir, str(postfix))
        print(checkpoint_dir)
        try:
            self.model.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'model.pt')), strict=strict)
            self.optim.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'optim.pt')))
            if self.scheduler is not None:
                self.scheduler.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'sched.pt')))
        except Exception as e:
            print(f'Failed to load weights with strict={strict}')
            print(e)

    def clear_checkpoints(self):
        for root, dirs, files in os.walk(self.checkpoint_dir):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))
        print(f'Directory {self.checkpoint_dir} has been cleared!')

    def _cast_to_tensor(self, input):
        if input is None:
            return None
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input).to(torch.float32)
        return input

    def _get_loaders(self, Y_train, X_cond_train=None, Y_valid=None, X_cond_valid=None):
        Y_train = self._cast_to_tensor(Y_train)
        Y_valid = self._cast_to_tensor(Y_valid)
        X_cond_train = self._cast_to_tensor(X_cond_train)
        X_cond_valid = self._cast_to_tensor(X_cond_valid)

        loaders = {}
        if X_cond_train is not None:
            train_dataset = TensorDataset(Y_train, X_cond_train)
        else:
            train_dataset = TensorDataset(Y_train)
        loaders["train"] = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        if Y_valid is not None:
            if X_cond_valid is not None:
                valid_dataset = TensorDataset(Y_valid, X_cond_valid)
            else:
                valid_dataset = TensorDataset(Y_valid)
            loaders["valid"] = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True)
        return loaders

    def run_epoch(self, loader, is_train=True):
        # To implement in relative class
        # Must provide train and valid epoch run
        raise NotImplemented()

    def sample(self, input: Union[torch.Tensor, np.ndarray, int], batch_size=None):
        # To implement in relative class
        # Must provide sampling strategy for model
        raise NotImplemented()

    def fit(self, Y_train: Union[torch.Tensor, np.ndarray],
                  X_cond_train: Union[torch.Tensor, np.ndarray] = None,
                  Y_valid: Union[torch.Tensor, np.ndarray] = None,
                  X_cond_valid: Union[torch.Tensor, np.ndarray] = None,
                  strict=False):
        if self.optim is None:
            raise ValueError('Model optimizer has to be set. Use `set_optimizer` method.')

        loaders = self._get_loaders(Y_train, X_cond_train, Y_valid, X_cond_valid)
        self.train_losses = []
        self.valid_losses = []
        for epoch in tqdm(range(self.n_epochs), desc=f"Epoch"):
            epoch_train_loss = self.run_epoch(loaders["train"])
            self.train_losses.append(epoch_train_loss)
            if "valid" in loaders:
                with torch.no_grad():
                    epoch_valid_loss = self.run_epoch(loaders["valid"], is_train=False)
                self.valid_losses.append(epoch_valid_loss)
            self.checkpoint(epoch + 1)

        self.load_from_checkpoint('best', strict=strict)
        return self

    def sample_from_N_checkpoints(self, input: Union[torch.tensor, np.ndarray, int], N=10, **kwargs):
        losses = self.valid_losses if len(self.valid_losses) else self.train_losses
        indx = np.argsort(losses)[:N]
        Ys = []
        for i in indx:
            self.load_from_checkpoint(postfix=i + 1)
            Ys.append(self.sample(input, **kwargs))
        Y = np.concatenate(Ys, axis=0)
        np.random.shuffle(Y)
        return Y
