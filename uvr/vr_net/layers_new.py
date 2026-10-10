import torch
from torch import nn
import torch.nn.functional as F

from uvr import spectral as spec_utils


class Conv2DBNActiv(nn.Module):
    """
    Conv2DBNActiv Class:
    This class implements a convolutional layer followed by batch normalization and an activation function.
    It is a fundamental building block for constructing neural networks, especially useful in image and audio processing tasks.
    The class encapsulates the pattern of applying a convolution, normalizing the output, and then applying a non-linear activation.
    """

    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, dilation=1, activ=nn.ReLU):
        super(Conv2DBNActiv, self).__init__()

        # Sequential model combining Conv2D, BatchNorm, and activation function into a single module
        self.conv = nn.Sequential(
            nn.Conv2d(
                nin,
                nout,
                kernel_size=ksize,
                stride=stride,
                padding=pad,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(nout),
            activ(),
        )

    def __call__(self, input_tensor):
        # Forward pass through the sequential model
        return self.conv(input_tensor)


class Encoder(nn.Module):
    """
    Encoder Class:
    This class defines an encoder module typically used in autoencoder architectures.
    It consists of two convolutional layers, each followed by batch normalization and an activation function.
    """

    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.LeakyReLU):
        super(Encoder, self).__init__()

        # First convolutional layer of the encoder
        self.conv1 = Conv2DBNActiv(nin, nout, ksize, stride, pad, activ=activ)
        # Second convolutional layer of the encoder
        self.conv2 = Conv2DBNActiv(nout, nout, ksize, 1, pad, activ=activ)

    def __call__(self, input_tensor):
        # Applying the first and then the second convolutional layers
        hidden = self.conv1(input_tensor)
        hidden = self.conv2(hidden)

        return hidden


class Decoder(nn.Module):
    """
    Decoder Class:
    This class defines a decoder module, which is the counterpart of the Encoder class in autoencoder architectures.
    It applies a convolutional layer followed by batch normalization and an activation function, with an optional dropout layer for regularization.
    """

    def __init__(
        self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.ReLU, dropout=False
    ):
        super(Decoder, self).__init__()
        # Convolutional layer with optional dropout for regularization
        self.conv1 = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ)
        # self.conv2 = Conv2DBNActiv(nout, nout, ksize, 1, pad, activ=activ)
        self.dropout = nn.Dropout2d(0.1) if dropout else None

    def __call__(self, input_tensor, skip=None):
        # Forward pass through the convolutional layer and optional dropout
        input_tensor = F.interpolate(
            input_tensor, scale_factor=2, mode="bilinear", align_corners=True
        )

        if skip is not None:
            skip = spec_utils.crop_center(skip, input_tensor)
            input_tensor = torch.cat([input_tensor, skip], dim=1)

        hidden = self.conv1(input_tensor)
        # hidden = self.conv2(hidden)

        if self.dropout is not None:
            hidden = self.dropout(hidden)

        return hidden


class ASPPModule(nn.Module):
    """
    ASPPModule Class:
    This class implements the Atrous Spatial Pyramid Pooling (ASPP) module, which is useful for semantic image segmentation tasks.
    It captures multi-scale contextual information by applying convolutions at multiple dilation rates.
    """

    def __init__(self, nin, nout, dilations=(4, 8, 12), activ=nn.ReLU, dropout=False):
        super(ASPPModule, self).__init__()

        # Global context convolution captures the overall context
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            Conv2DBNActiv(nin, nout, 1, 1, 0, activ=activ),
        )
        self.conv2 = Conv2DBNActiv(nin, nout, 1, 1, 0, activ=activ)
        self.conv3 = Conv2DBNActiv(
            nin, nout, 3, 1, dilations[0], dilations[0], activ=activ
        )
        self.conv4 = Conv2DBNActiv(
            nin, nout, 3, 1, dilations[1], dilations[1], activ=activ
        )
        self.conv5 = Conv2DBNActiv(
            nin, nout, 3, 1, dilations[2], dilations[2], activ=activ
        )
        self.bottleneck = Conv2DBNActiv(nout * 5, nout, 1, 1, 0, activ=activ)
        self.dropout = nn.Dropout2d(0.1) if dropout else None

    def forward(self, input_tensor):
        _, _, h, w = input_tensor.size()

        # Upsample global context to match input size and combine with local and multi-scale features
        feat1 = F.interpolate(
            self.conv1(input_tensor), size=(h, w), mode="bilinear", align_corners=True
        )
        feat2 = self.conv2(input_tensor)
        feat3 = self.conv3(input_tensor)
        feat4 = self.conv4(input_tensor)
        feat5 = self.conv5(input_tensor)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        out = self.bottleneck(out)

        if self.dropout is not None:
            out = self.dropout(out)

        return out


class LSTMModule(nn.Module):
    """
    LSTMModule Class:
    This class defines a module that combines convolutional feature extraction with a bidirectional LSTM for sequence modeling.
    It is useful for tasks that require understanding temporal dynamics in data, such as speech and audio processing.
    """

    def __init__(self, nin_conv, nin_lstm, nout_lstm):
        super(LSTMModule, self).__init__()
        # Convolutional layer for initial feature extraction
        self.conv = Conv2DBNActiv(nin_conv, 1, 1, 1, 0)

        # Bidirectional LSTM for capturing temporal dynamics
        self.lstm = nn.LSTM(
            input_size=nin_lstm, hidden_size=nout_lstm // 2, bidirectional=True
        )

        # Dense layer for output dimensionality matching
        self.dense = nn.Sequential(
            nn.Linear(nout_lstm, nin_lstm), nn.BatchNorm1d(nin_lstm), nn.ReLU()
        )

    def forward(self, input_tensor):
        N, _, nbins, nframes = input_tensor.size()

        # Extract features and prepare for LSTM
        hidden = self.conv(input_tensor)[:, 0]  # N, nbins, nframes
        hidden = hidden.permute(2, 0, 1)  # nframes, N, nbins
        hidden, _ = self.lstm(hidden)

        # Apply dense layer and reshape to match expected output format
        hidden = self.dense(hidden.reshape(-1, hidden.size()[-1]))  # nframes * N, nbins
        hidden = hidden.reshape(nframes, N, 1, nbins)
        hidden = hidden.permute(1, 2, 3, 0)

        return hidden
