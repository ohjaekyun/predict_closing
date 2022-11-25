from torch import nn
import torch


class VAE_Loss(nn.Module):

  def __init__(self):
    super().__init__()

    self.nlloss = nn.MSELoss()
  
  def KL_loss (self, mu, log_var):

    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    kl = kl.sum(-1)  # to go from multi-dimensional z to single dimensional z : (batch_size x latent_size) ---> (batch_size) 
                                                                      # i.e Z = [ [z1_1, z1_2 , ...., z1_lt] ] ------> z = [ z1] 
                                                                      #         [ [z2_1, z2_2, ....., z2_lt] ]             [ z2]
                                                                      #                   .                                [ . ]
                                                                      #                   .                                [ . ]
                                                                      #         [[zn_1, zn_2, ....., zn_lt] ]              [ zn]
                                                                      
                                                                      #        lt=latent_size 
    kl = kl.mean()                                                     
    return kl

  def reconstruction_loss(self, x_hat_param, x):


    x = x.view(-1).contiguous()
    x_hat_param = x_hat_param.view(-1).contiguous()#, x_hat_param.size(2))

    recon = self.nlloss(x_hat_param, x)

    return recon
  

  def forward(self, mu, log_var, x_hat_param, x):
    kl_loss = self.KL_loss(mu, log_var)
    recon_loss = self.reconstruction_loss(x_hat_param, x)


    elbo = kl_loss + recon_loss # we use + because recon loss is a NLLoss (cross entropy) and it's negative in its own, and in the ELBO equation we have
                              # elbo = KL_loss - recon_loss, therefore, ELBO = KL_loss - (NLLoss) = KL_loss + NLLoss

    return elbo#, kl_loss, recon_loss
