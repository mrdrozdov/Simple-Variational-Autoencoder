import argparse
import numpy as np
import os
from utils_vae import img_tile, mnist_reader

import torch
import torch.nn as nn
import torch.optim as optim


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=40)
    parser.add_argument("--nz", type=int, default=20)
    parser.add_argument("--layersize", type=int, default=400)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--b1", type=float, default=0.9)
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--e", type=float, default=1e-8)
    parser.add_argument("--bsize", type=int, default=64)
    return parser.parse_args()

args = parse_args()


np.random.seed(111)

class VAE(nn.Module):
    def __init__(self, numbers):
        super().__init__()

        self.numbers = numbers

        self.epochs = args.epoch
        self.batch_size = args.bsize
        self.learning_rate = args.lr
        self.decay = 0.001
        self.nz = args.nz
        self.layersize = args.layersize

        self.img_path = "./images"
        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)

        # Xavier initialization is used to initialize the weights
        # init encoder weights
        self._e_W0 = np.random.randn(784, self.layersize).astype(np.float32) * np.sqrt(2.0/(784))
        self._e_b0 = np.zeros(self.layersize).astype(np.float32)

        self._e_W_mu = np.random.randn(self.layersize, self.nz).astype(np.float32) * np.sqrt(2.0/(self.layersize))
        self._e_b_mu = np.zeros(self.nz).astype(np.float32)

        self._e_W_logvar = np.random.randn(self.layersize, self.nz).astype(np.float32) * np.sqrt(2.0/(self.layersize))
        self._e_b_logvar = np.zeros(self.nz).astype(np.float32)

        # init decoder weights
        self._d_W0 = np.random.randn(self.nz, self.layersize).astype(np.float32) * np.sqrt(2.0/(self.nz))
        self._d_b0 = np.zeros(self.layersize).astype(np.float32)

        self._d_W1 = np.random.randn(self.layersize, 784).astype(np.float32) * np.sqrt(2.0/(self.layersize))
        self._d_b1 = np.zeros(784).astype(np.float32)

        #
        self.e_W0 = nn.Parameter(torch.from_numpy(self._e_W0).float())
        self.e_b0 = nn.Parameter(torch.from_numpy(self._e_b0).float())
        self.e_W_mu = nn.Parameter(torch.from_numpy(self._e_W_mu).float())
        self.e_b_mu = nn.Parameter(torch.from_numpy(self._e_b_mu).float())
        self.e_W_logvar = nn.Parameter(torch.from_numpy(self._e_W_logvar).float())
        self.e_b_logvar = nn.Parameter(torch.from_numpy(self._e_b_logvar).float())

        self.d_W0 = nn.Parameter(torch.from_numpy(self._d_W0).float())
        self.d_b0 = nn.Parameter(torch.from_numpy(self._d_b0).float())
        self.d_W1 = nn.Parameter(torch.from_numpy(self._d_W1).float())
        self.d_b1 = nn.Parameter(torch.from_numpy(self._d_b1).float())

        # init Adam optimizer
        self.b1 = args.b1
        self.b2 = args.b2
        self.e = args.e
        self.m = [0] * 10
        self.v = [0] * 10
        self.t = 0

    def encoder(self, img):
        #self.e_logvar : log variance
        #self.e_mean : mean

        e_input = np.reshape(img, (self.batch_size,-1))
        e_input = torch.from_numpy(e_input).float()

        e_h0_l = torch.matmul(e_input, self.e_W0) + self.e_b0
        e_h0_a = nn.LeakyReLU(negative_slope=0.01)(e_h0_l)

        e_logvar = torch.matmul(e_h0_a, self.e_W_logvar) + self.e_b_logvar
        e_mu = torch.matmul(e_h0_a, self.e_W_mu) + self.e_b_mu

        return e_mu, e_logvar

    def decoder(self, z):
        #self.d_out : reconstruction image 28x28

        z = z.view(self.batch_size, self.nz)

        d_h0_l = torch.matmul(z, self.d_W0) + self.d_b0
        d_h0_a = torch.relu(d_h0_l)

        d_h1_l = torch.matmul(d_h0_a, self.d_W1) + self.d_b1
        d_h1_a = torch.sigmoid(d_h1_l)

        d_out = d_h1_a.view(self.batch_size, 28, 28, 1)

        return d_out

    def forward(self, x):
        #Encode
        mu, logvar = self.encoder(x)

        #use reparameterization trick to sample from gaussian
        sample_z = mu + torch.exp(logvar * .5) * torch.from_numpy(np.random.standard_normal(size=(self.batch_size, self.nz))).float()

        decode = self.decoder(sample_z)

        return decode, mu, logvar, None, sample_z

    def train(self, optimizer):

        #Read in training data
        trainX, _, train_size = mnist_reader(self.numbers)

        np.random.shuffle(trainX)

        #set batch indices
        batch_idx = train_size//self.batch_size
        batches_per_epoch = min(10, batch_idx)
        # batches_per_epoch = batch_idx
        del batch_idx

        total_loss = 0
        total_kl = 0
        total = 0

        for epoch in range(self.epochs):
            for idx in range(batches_per_epoch):
                # prepare batch and input vector z
                train_batch = trainX[idx*self.batch_size:idx*self.batch_size + self.batch_size]
                #ignore batch if there are insufficient elements
                if train_batch.shape[0] != self.batch_size:
                    break

                ################################
                #       Forward Pass
                ################################

                out, mu, logvar, _, sample_z = self(train_batch)

                # Reconstruction Loss
                rec_loss = nn.BCELoss(reduction='sum')(out, torch.from_numpy(train_batch).float())

                #K-L Divergence
                # kl = -0.5 * np.sum(1 + logvar - np.power(mu, 2) - np.exp(logvar))
                kl = -0.5 * torch.sum(1 + logvar - mu ** 2 - torch.exp(logvar))

                loss = rec_loss + kl
                loss = loss / self.batch_size

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                rec_loss = rec_loss.item()
                kl = kl.item()

                #Loss Recordkeeping
                total_loss += rec_loss / self.batch_size
                total_kl += kl / self.batch_size
                total += 1

                self.img = np.squeeze(out.data.numpy(), axis=3) * 2 - 1

                print("Epoch [%d] Step [%d/%d]  RC Loss:%.4f  KL Loss:%.4f  lr: %.4f"%(
                        epoch, idx, batches_per_epoch, rec_loss / self.batch_size, kl / self.batch_size, self.learning_rate))

            sample = np.array(self.img)

            #save image result every epoch
            img_tile(sample, self.img_path, epoch, idx, "res", True)


if __name__ == '__main__':

    # Adjust the numbers that appear in the training data. Less numbers helps
    # run the program to see faster results
    numbers = [1, 2, 3]
    model = VAE(numbers)

    for name, p in model.named_parameters():
        print(name, p.shape)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.b1, args.b2), eps=args.e)
    model.train(optimizer)
