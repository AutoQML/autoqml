import os
from collections.abc import Callable

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from autoqml.constants import InputData, TargetData
from autoqml.search_space import Configuration
from autoqml.search_space.base import TunableMixin
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from optuna import Trial
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from typing_extensions import Self

CHECKPOINT_PATH = 'autoqml/saved_models/'


class ClassicNNRImplementationError(RuntimeError):
    pass


class _NNRegressor(L.LightningModule):
    def __init__(
        self,
        in_features: int,
        n_hidden_neurons: int,  # between 5 and 50
        act_fn: torch.nn.Module,
        loss_fn: Callable,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.loss_fn = loss_fn
        self.net = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=n_hidden_neurons),
            act_fn(),
            nn.Linear(
                in_features=n_hidden_neurons,
                out_features=n_hidden_neurons // 2
            ),
            act_fn(),
            nn.Linear(in_features=n_hidden_neurons // 2, out_features=1),
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x)
        x = x.to(torch.float)
        return self.net(x)

    def _calc_loss(self, batch):
        X, y = batch
        if isinstance(X, np.ndarray):
            X = torch.tensor(X)
        X = X.to(torch.float)
        y_pred = self.forward(X).squeeze(1)
        loss = self.loss_fn(y, y_pred)
        # loss = loss.sum(dim=[1]).mean(dim=[0])
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-6
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

    def training_step(self, batch, batch_idx):
        loss = self._calc_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calc_loss(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self._calc_loss(batch)
        self.log('test_loss', loss)


act_fn_choice = {
    'ReLU': nn.ReLU,
    'LeakyReLU': nn.LeakyReLU,
    'GELU': nn.GELU,
    'Sigmoid': nn.Sigmoid,
    'Tanh': nn.Tanh,
}
loss_fn_choice = {
    'mse_loss': F.mse_loss,
    'mae_loss': F.l1_loss,
}
output_act_fn_choice = {
    'Sigmoid': nn.Sigmoid,
    'Tanh': nn.Tanh,
}


class NNRegressor(BaseEstimator, TransformerMixin, TunableMixin):
    def __init__(
        self,
        n_hidden_neurons=128,
        act_fn='GELU',
        loss_fn='mse_loss',
        max_epochs=200,
    ):
        self.n_hidden_neurons = n_hidden_neurons
        self.act_fn = act_fn_choice[act_fn]
        self.loss_fn = loss_fn_choice[loss_fn]
        self.max_epochs = max_epochs

    def _init_environment(self):
        L.seed_everything(777)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.device = (
            torch.device('cuda:0')
            if torch.cuda.is_available() else torch.device('cpu')
        )
        self.CHECKPOINT_PATH = os.environ.get(
            "PATH_CHECKPOINT", "saved_models"
        )

    def _trainer_factory(self) -> L.Trainer:
        # checkpoint file name is derived from in_features and n_hidden_neurons
        trainer = L.Trainer(
            default_root_dir=os.path.join(
                CHECKPOINT_PATH, 'nn_regressor_%s_x_%s_x_%i' %
                (self.act_fn, self.loss_fn, self.n_hidden_neurons)
            ),
            accelerator='auto',
            devices=1,
            max_epochs=self.max_epochs,
            log_every_n_steps=1,
            callbacks=[
                ModelCheckpoint(save_weights_only=True),
                LearningRateMonitor('epoch'),
                EarlyStopping(monitor="val_loss", mode="min"),
            ],
        )
        trainer.logger._log_graph = True
        trainer.logger._default_hp_metric = None
        return trainer

    def _train(
        self,
        train_loader,
        val_loader,
        trainer,
        input_dim,
    ) -> tuple[_NNRegressor, dict]:
        # Uncomment to use pretrained model:
        # pretrained_filename = os.path.join(CHECKPOINT_PATH, 'autoencoder_%i_x_%i' % (input_dim, latent_dim))
        # if os.path.isfile(pretrained_filename):
        #     print("Found pretrained model, loading...")
        #     model = _Autoencoder.load_from_checkpoint(pretrained_filename)
        # else:
        model = _NNRegressor(
            in_features=input_dim,
            n_hidden_neurons=self.n_hidden_neurons,
            act_fn=self.act_fn,
            loss_fn=self.loss_fn,
        )
        trainer.fit(model, train_loader, val_loader)
        train_result = trainer.test(
            model, dataloaders=train_loader, verbose=False
        )
        val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
        result = {"train": train_result, "val": val_result}
        return model, result

    def fit(self, X: InputData, y: TargetData) -> Self:
        self._init_environment()
        input_dim = X.shape[1]  # the length of one data point
        trainer = self._trainer_factory()
        # Check for Pandas Dataframes or Series and convert to numpy arrays
        if isinstance(X, pd.core.frame.DataFrame
                     ) or isinstance(X, pd.core.series.Series):
            X = X.values
        if isinstance(y, pd.core.series.Series):
            y = y.values
        # train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=777
        )
        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        # Create a TensorDataset from the tensors
        train_dataset = data.TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = data.TensorDataset(X_test_tensor, y_test_tensor)

        # Create a DataLoader from the TensorDataset
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=10,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=4
        )
        val_loader = data.DataLoader(
            test_dataset,
            batch_size=10,
            shuffle=False,
            drop_last=False,
            num_workers=4
        )
        model, train_result = self._train(
            train_loader=train_loader,
            val_loader=val_loader,
            trainer=trainer,
            input_dim=input_dim,
        )
        self.estimator = model
        self.train_result = train_result
        return self

    def predict(self, X: InputData) -> TargetData:
        if not hasattr(self, 'estimator'):
            raise ClassicNNRImplementationError()
        y_pred = self.estimator(X)
        # detach the tensor and convert to numpy array,
        # so that following components in the pipeline can use it.
        y_pred = y_pred.detach().numpy()
        return y_pred

    def sample_configuration(
        self, trial: Trial, defaults: Configuration
    ) -> Configuration:
        return {
            'n_hidden_neurons':
                (
                    self._get_default_values(trial, 'max_epochs', defaults)
                    if self._fullname('max_epochs') in defaults else trial.
                    suggest_int(self._fullname('max_epochs'), 2**5, 2**9)
                ),
            'act_fn':
                (
                    self._get_default_values(trial, 'act_fn', defaults)
                    if self._fullname('act_fn') in defaults else
                    trial.suggest_categorical(
                        self._fullname('act_fn'),
                        ['ReLU', 'LeakyReLU', 'GELU', 'Sigmoid', 'Tanh']
                    )
                ),
            'loss_fn':
                (
                    self._get_default_values(trial, 'loss_fn', defaults)
                    if self._fullname('loss_fn') in defaults else
                    trial.suggest_categorical(
                        self._fullname('loss_fn'), ['mse_loss', 'mae_loss']
                    )
                ),
        }
