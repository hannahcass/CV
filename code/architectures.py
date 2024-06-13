import numpy as np
import torch
from torch import nn


### Autoencoder Original
class ConvAutoEncoder(nn.Module):
    def __init__(self, in_channels: int, hid_channels: int, code_channels: int,
                 kernel_size: int, stride: int, pooling: (int, int),
                 activation: nn.Module = nn.ReLU()):

        super().__init__()

        self.encoder_hidden = nn.Conv2d(in_channels=in_channels,
                                        out_channels=hid_channels,
                                        kernel_size=kernel_size,
                                        stride=stride)
        self.encoder_pool = nn.AvgPool2d(kernel_size=pooling)
        self.encoder_output = nn.Conv2d(in_channels=hid_channels,
                                        out_channels=code_channels,
                                        kernel_size=kernel_size,
                                        stride=stride)

        self.act = activation

        self.decoder_hidden = nn.ConvTranspose2d(in_channels=code_channels,
                                                 out_channels=hid_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride)
        self.decoder_unpool = nn.ConvTranspose2d(in_channels=hid_channels,
                                                 out_channels=hid_channels,
                                                 kernel_size=pooling,
                                                 stride=pooling)

        self.decoder_unpool.weight.data.fill_(1 / np.multiply(*pooling))
        self.decoder_unpool.weight.requires_grad = False
        self.decoder_unpool.bias.requires_grad = False

        self.decoder_output = nn.ConvTranspose2d(in_channels=hid_channels,
                                                 out_channels=in_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride)

    def forward(self, x):
        x = self.encoder_hidden(x)
        x = self.act(x)
        x = self.encoder_pool(x)
        x = self.encoder_output(x)
        x = self.act(x)
        x = self.decoder_hidden(x)
        x = self.decoder_unpool(x)
        x = self.act(x)
        x = self.decoder_output(x)
        return x

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : torch.Tensor
            Inputs to be reconstructed.
        y : torch.Tensor
            Result of reconstruction, with values
            in the same range as the targets.
        """
        logits = self.forward(x)
        return torch.clamp(logits, x.min(), x.max())


### Experimental version of above model
class ConvAutoEncoder2(nn.Module):
    def __init__(self, in_channels: int = 3, hid_channels: int = 64,
                 code_channels: int = 128, kernel_size: int = 5,
                 stride: int = 1, pooling: (int, int) = (2, 2),
                 dropout: float = 0.2, activation: nn.Module = nn.ReLU()):

        super().__init__()

        self.encoder_hidden = nn.Conv2d(in_channels=in_channels,
                                        out_channels=hid_channels,
                                        kernel_size=kernel_size,
                                        stride=stride)
        self.encoder_pool = nn.AvgPool2d(kernel_size=pooling)
        self.encoder_output1 = nn.Conv2d(in_channels=hid_channels,
                                         out_channels=code_channels,
                                         kernel_size=kernel_size,
                                         stride=2*stride, padding=2)
        self.encoder_output2 = nn.Conv2d(in_channels=code_channels,
                                         out_channels=code_channels,
                                         kernel_size=kernel_size,
                                         stride=2*stride)

        self.act = activation
        self.dropout = nn.Dropout2d(p=dropout)

        self.decoder_hidden2 = nn.ConvTranspose2d(in_channels=code_channels,
                                                  out_channels=code_channels,
                                                  kernel_size=kernel_size,
                                                  stride=2*stride)
        self.decoder_hidden1 = nn.ConvTranspose2d(in_channels=code_channels,
                                                  out_channels=hid_channels,
                                                  kernel_size=kernel_size,
                                                  stride=2*stride, padding=2,
                                                  output_padding=1)
        self.decoder_unpool = nn.ConvTranspose2d(in_channels=hid_channels,
                                                 out_channels=hid_channels,
                                                 kernel_size=pooling,
                                                 stride=pooling)

        self.decoder_unpool.weight.data.fill_(1 / np.multiply(*pooling))
        self.decoder_unpool.weight.requires_grad = False
        self.decoder_unpool.bias.requires_grad = False

        self.decoder_output = nn.ConvTranspose2d(in_channels=hid_channels,
                                                 out_channels=in_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride)

    def forward(self, x):
        x = self.encoder_hidden(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.encoder_pool(x)
        x = self.encoder_output1(x)
        x = self.encoder_output2(x)
        x = self.act(x)
        x = self.decoder_hidden2(x)
        x = self.decoder_hidden1(x)
        x = self.decoder_unpool(x)
        x = self.dropout(x)
        x = self.act(x)
        x = self.decoder_output(x)

        return x

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : torch.Tensor
            Inputs to be reconstructed.
        y : torch.Tensor
            Result of reconstruction, with values
            in the same range as the targets.
        """
        logits = self.forward(x)
        return torch.clamp(logits, x.min(), x.max())


### Autoencoder Test 1
class ConvAutoEncoderTest(nn.Module):
    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.Dropout2d(0.5),  # b, 16, 5, 5
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.Dropout2d(0.5),  # b, 8, 15, 15
            nn.ConvTranspose2d(64, 3, 2, stride=2),  # b, 1, 28, 28
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class NewNorm(nn.Module):
    def __init__(self, dropout: float = 0.25):

        super().__init__()
        self.p_dropout = dropout

        self.enc_conv1 = nn.Conv2d(3, 64, 3, padding=1, stride=2)
        self.enc_conv2 = nn.Conv2d(64, 128, 3, padding=1, stride=2)

        self.dec_conv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec_conv2 = nn.ConvTranspose2d(64, 3, 2, stride=2)

        self.dropout = nn.Dropout2d(p=dropout)
        self.activation = nn.ReLU()

    def forward(self, x):

        x = self.enc_conv1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.enc_conv2(x)
        x = self.dropout(x)
        x = self.activation(x)

        x = self.dec_conv1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dec_conv2(x)
        x = self.dropout(x)
        x = torch.sigmoid(x)

        return x

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : torch.Tensor
            Inputs to be reconstructed.
        y : torch.Tensor
            Result of reconstruction, with values
            in the same range as the targets.
        """
        logits = self.forward(x)
        return torch.clamp(logits, x.min(), x.max())


class NewNorm2b(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=1),
            nn.Dropout2d(0.25),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.Dropout2d(0.25)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3),
            nn.Dropout2d(0.25),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 5, stride=2, padding=1, output_padding=1),
            nn.Dropout2d(0.25),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x