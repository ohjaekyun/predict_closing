from typing import List, Callable, Union, Any, TypeVar, Tuple
from torch import nn
import torch
from torch.nn import functional as F
import numpy as np
import pandas as pd
from .rnn import CMV_LSTM, MV_LSTM
from .vae import ConditionalVAE, VanillaVAEEncoder, VanillaVAEDecoder
from torch import tensor as Tensor
torch.autograd.set_detect_anomaly(True)

class UnitRVAE(nn.Module):
    def __init__(
        self,
        n_features,
        seq_length,
        latent_dim,
        n_hidden=20,
        n_layers=2,
        hidden_dims=[16, 32, 64]
        ):
        super().__init__()

        self.latent_dim = latent_dim
        #in_channels += 1 # To account for the extra label channel
        self.vae_encoder = VanillaVAEEncoder(
            n_features,
            latent_dim,
            hidden_dims
        )
        self.vae_decoder = VanillaVAEDecoder(
            latent_dim,
            n_features,
            self.vae_encoder.hidden_dims
        )
        self.rnn = MV_LSTM(
            latent_dim,
            seq_length,
            n_hidden=n_hidden,
            n_layers=n_layers
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        return self.vae_encoder(input)

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """

        return self.vae_decoder(z)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        z = self.rnn(z)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(
        self,
        label,
        *args,
        **kwargs
        ) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, label)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1))

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int,
               **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)
        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


class RVAE(nn.Module):
    def __init__(
        self,
        n_features1,
        n_features2,
        n_features3,
        n_latent_features1,
        n_latent_features2,
        n_latent_features3,
        seq_length,
        n_hidden=20,
        n_layers=2,
        hidden_dims=[16, 32, 64],
        ):
        super().__init__()

        self.latent_dim1 = n_latent_features1
        #in_channels += 1 # To account for the extra label channel
        self.vae_encoder1 = VanillaVAEEncoder(
            n_features1,
            n_latent_features1,
            hidden_dims
        )
        self.vae_decoder1 = VanillaVAEDecoder(
            n_latent_features1,
            n_features1,
            self.vae_encoder1.hidden_dims
        )

        self.latent_dim2 = n_latent_features2

        self.vae_encoder2 = VanillaVAEEncoder(
            n_features2,
            n_latent_features2,
            hidden_dims
        )
        self.vae_decoder2 = VanillaVAEDecoder(
            n_latent_features2,
            n_features2,
            self.vae_encoder2.hidden_dims
        )

        self.latent_dim3 = n_latent_features3

        self.vae_encoder3 = VanillaVAEEncoder(
            n_features3,
            n_latent_features3,
            hidden_dims
        )
        self.vae_decoder3 = VanillaVAEDecoder(
            n_latent_features3,
            n_features3,
            self.vae_encoder3.hidden_dims
        )

        self.rnn = CMV_LSTM(
            n_latent_features1,
            n_latent_features2,
            n_latent_features3,
            seq_length,
            n_hidden=n_hidden,
            n_layers=n_layers
        )

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input1: Tensor, input2: Tensor, input3: Tensor, **kwargs) -> List[Tensor]:

        mu1, log_var1 = self.vae_encoder1(input1)
        z1 = self.reparameterize(mu1, log_var1)

        mu2, log_var2 = self.vae_encoder2(input2)
        z2 = self.reparameterize(mu2, log_var2)

        mu3, log_var3 = self.vae_encoder3(input3)
        z3 = self.reparameterize(mu3, log_var3)

        z1, z2, z3 = self.rnn(z1, z2, z3)
        return  [
            [self.vae_decoder1(z1), input1, mu1, log_var1],
            [self.vae_decoder2(z2), input2, mu2, log_var2],
            [self.vae_decoder3(z3), input3, mu3, log_var3]
        ]

    def loss_function(self,
                      labels,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        #list_recons = []
        #list_inputs = []
        #list_mus = []
        #list_log_vars = []
        #for arg in args:
        #    list_recons.append(arg[0])
        #    list_inputs.append(arg[1])
        #    list_mus.append(arg[2])
        #    list_log_vars.append(arg[3])

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss = kld_loss = 0
        for label, arg in zip(labels, args):
            recons = arg[0]
            input = arg[1]
            mu = arg[2]
            log_var = arg[3]
            recons_loss = recons_loss + F.mse_loss(recons, label)
            kld_loss = kld_loss + torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1))

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int,
               idx: int
               ) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        if idx == 1:
            latent_dim = self.vae_encoder1.latent_dim
            decoder = self.vae_decoder1
        elif idx == 2:
            latent_dim = self.vae_encoder2.latent_dim
            decoder = self.vae_decoder2
        elif idx == 3:
            latent_dim = self.vae_encoder3.latent_dim
            decoder = self.vae_decoder3
        z = torch.randn(num_samples, latent_dim)
        z = z.to(current_device)

        samples = decoder(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        results = self.forward(x)
        return [
            results[0][0],
            results[1][0],
            results[2][0]
        ]