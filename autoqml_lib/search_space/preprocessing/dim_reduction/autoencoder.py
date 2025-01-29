import os

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from autoqml_lib.constants import InputData, TargetData
from autoqml_lib.meta_learning.datastatistics import DataStatistics
from autoqml_lib.search_space import Configuration
from autoqml_lib.search_space.base import TunableMixin
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from optuna import Trial
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from typing_extensions import Self

CHECKPOINT_PATH = 'autoqml_lib/saved_models/'


class AutoencoderImplementationError(RuntimeError):
    pass


class _Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        act_fn: object = nn.GELU,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=latent_dim),
            act_fn(),
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x)
        x = x.to(torch.float)
        return self.net(x)


class _Decoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        output_dim,
        output_act_fn: object = nn.Tanh,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=output_dim),
            output_act_fn(),
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x)
        x = x.to(torch.float)
        return self.net(x)


class _Autoencoder(L.LightningModule):
    def __init__(
        self,
        input_dim,
        latent_dim,
        encoder_class: object = _Encoder,
        decoder_class: object = _Decoder,
        act_fn: object = nn.GELU,
        output_act_fn: object = nn.Tanh,
        loss_fn: object = F.mse_loss,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.loss_fn = loss_fn
        self.encoder = encoder_class(
            input_dim=input_dim, latent_dim=latent_dim, act_fn=act_fn
        )
        self.decoder = decoder_class(
            latent_dim=latent_dim,
            output_dim=input_dim,
            output_act_fn=output_act_fn
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x)
        x = x.to(torch.float)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def _get_reconstruction_loss(self, batch):
        x = batch
        if isinstance(x, np.ndarray):
            x = torch.tensor(x)
        x = x.to(torch.float)
        x_hat = self.forward(x)
        loss = self.loss_fn(x, x_hat, reduction='none')
        loss = loss.sum(dim=[1]).mean(dim=[0])
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, patience=20, min_lr=1e-5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
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


class Autoencoder(BaseEstimator, TransformerMixin, TunableMixin):
    def __init__(
        self,
        latent_dim=10,
        act_fn='GELU',
        output_act_fn='Tanh',
        loss_fn='mse_loss',
        max_epochs=50,
    ):
        self.latent_dim = latent_dim
        self.act_fn = act_fn_choice[act_fn]
        self.loss_fn = loss_fn_choice[loss_fn]
        self.output_act_fn = output_act_fn_choice[output_act_fn]
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

    def _trainer_factory(self, input_dim: int) -> L.Trainer:
        """input_dim is only used to name the checkpoint file."""
        trainer = L.Trainer(
            default_root_dir=os.path.join(
                CHECKPOINT_PATH,
                'autoencoder_%i_x_%i' % (input_dim, self.latent_dim)
            ),
            accelerator='auto',
            devices=1,
            max_epochs=self.max_epochs,
            log_every_n_steps=1,
            callbacks=[
                ModelCheckpoint(save_weights_only=False),
                LearningRateMonitor('epoch'),
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
    ) -> tuple[_Autoencoder, dict]:
        # pretrained_filename = os.path.join(CHECKPOINT_PATH, 'autoencoder_%i_x_%i' % (input_dim, latent_dim))
        # if os.path.isfile(pretrained_filename):
        #     print("Found pretrained model, loading...")
        #     model = _Autoencoder.load_from_checkpoint(pretrained_filename)
        # else:
        model = _Autoencoder(
            input_dim=input_dim,
            latent_dim=self.latent_dim,
            act_fn=self.act_fn,
            output_act_fn=self.output_act_fn,
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
        trainer = self._trainer_factory(input_dim=input_dim)
        X_train, X_test = train_test_split(X, test_size=0.25)
        train_loader = data.DataLoader(
            X_train,
            batch_size=10,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=4
        )
        val_loader = data.DataLoader(
            X_test,
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
        self.autoencoder = model
        self.train_result = train_result
        return self

    def transform(self, X: InputData) -> InputData:
        if not hasattr(self, 'autoencoder'):
            raise AutoencoderImplementationError()
        X = self.autoencoder.encoder(X)
        # detach the tensor and convert to numpy array,
        # so that following components in the pipeline can use it.
        X = X.detach().numpy()
        return X

    def sample_configuration(
        self, trial: Trial, defaults: Configuration,
        dataset_statistics: DataStatistics
    ) -> Configuration:
        return {
            'latent_dim':
                (
                    self._get_default_values(trial, 'latent_dim', defaults)
                    if self._fullname('latent_dim') in defaults else
                    trial.suggest_int(self._fullname('latent_dim'), 5, 10)
                ),
            'max_epochs':
                (
                    self._get_default_values(trial, 'max_epochs', defaults)
                    if self._fullname('max_epochs') in defaults else
                    trial.suggest_int(self._fullname('max_epochs'), 20, 50)
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
            'output_act_fn':
                (
                    self._get_default_values(trial, 'output_act_fn', defaults)
                    if self._fullname('output_act_fn') in defaults else
                    trial.suggest_categorical(
                        self._fullname('output_act_fn'), ['Sigmoid', 'Tanh']
                    )
                ),
        }
