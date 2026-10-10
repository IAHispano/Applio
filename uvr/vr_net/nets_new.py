import torch
from torch import nn
import torch.nn.functional as F
from . import layers_new as layers


class BaseNet(nn.Module):
    """
    BaseNet Class:
    This class defines the base network architecture for vocal removal. It includes a series of encoders for feature extraction,
    an ASPP module for capturing multi-scale context, and a series of decoders for reconstructing the output. Additionally,
    it incorporates an LSTM module for capturing temporal dependencies.
    """

    def __init__(
        self, nin, nout, nin_lstm, nout_lstm, dilations=((4, 2), (8, 4), (12, 6))
    ):
        super(BaseNet, self).__init__()
        # Initialize the encoder layers with increasing output channels for hierarchical feature extraction.
        self.enc1 = layers.Conv2DBNActiv(nin, nout, 3, 1, 1)
        self.enc2 = layers.Encoder(nout, nout * 2, 3, 2, 1)
        self.enc3 = layers.Encoder(nout * 2, nout * 4, 3, 2, 1)
        self.enc4 = layers.Encoder(nout * 4, nout * 6, 3, 2, 1)
        self.enc5 = layers.Encoder(nout * 6, nout * 8, 3, 2, 1)

        # ASPP module for capturing multi-scale features with different dilation rates.
        self.aspp = layers.ASPPModule(nout * 8, nout * 8, dilations, dropout=True)

        # Decoder layers for upscaling and merging features from different levels of the encoder and ASPP module.
        self.dec4 = layers.Decoder(nout * (6 + 8), nout * 6, 3, 1, 1)
        self.dec3 = layers.Decoder(nout * (4 + 6), nout * 4, 3, 1, 1)
        self.dec2 = layers.Decoder(nout * (2 + 4), nout * 2, 3, 1, 1)

        # LSTM module for capturing temporal dependencies in the sequence of features.
        self.lstm_dec2 = layers.LSTMModule(nout * 2, nin_lstm, nout_lstm)
        self.dec1 = layers.Decoder(nout * (1 + 2) + 1, nout * 1, 3, 1, 1)

    def __call__(self, input_tensor):
        # Sequentially pass the input through the encoder layers.
        encoded1 = self.enc1(input_tensor)
        encoded2 = self.enc2(encoded1)
        encoded3 = self.enc3(encoded2)
        encoded4 = self.enc4(encoded3)
        encoded5 = self.enc5(encoded4)

        # Pass the deepest encoder output through the ASPP module.
        bottleneck = self.aspp(encoded5)

        # Sequentially upscale and merge the features using the decoder layers.
        bottleneck = self.dec4(bottleneck, encoded4)
        bottleneck = self.dec3(bottleneck, encoded3)
        bottleneck = self.dec2(bottleneck, encoded2)
        # Concatenate the LSTM module output for temporal feature enhancement.
        bottleneck = torch.cat([bottleneck, self.lstm_dec2(bottleneck)], dim=1)
        bottleneck = self.dec1(bottleneck, encoded1)

        return bottleneck


class CascadedNet(nn.Module):
    """
    CascadedNet Class:
    This class defines a cascaded network architecture that processes input in multiple stages, each stage focusing on different frequency bands.
    It utilizes the BaseNet for processing, and combines outputs from different stages to produce the final mask for vocal removal.
    """

    def __init__(self, n_fft, nn_arch_size=51000, nout=32, nout_lstm=128):
        super(CascadedNet, self).__init__()
        # Calculate frequency bins based on FFT size.
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.nin_lstm = self.max_bin // 2
        self.offset = 64
        # Adjust output channels based on the architecture size.
        nout = 64 if nn_arch_size == 218409 else nout

        # print(nout, nout_lstm, n_fft)

        # Initialize the network stages, each focusing on different frequency bands and progressively refining the output.
        self.stg1_low_band_net = nn.Sequential(
            BaseNet(2, nout // 2, self.nin_lstm // 2, nout_lstm),
            layers.Conv2DBNActiv(nout // 2, nout // 4, 1, 1, 0),
        )
        self.stg1_high_band_net = BaseNet(
            2, nout // 4, self.nin_lstm // 2, nout_lstm // 2
        )

        self.stg2_low_band_net = nn.Sequential(
            BaseNet(nout // 4 + 2, nout, self.nin_lstm // 2, nout_lstm),
            layers.Conv2DBNActiv(nout, nout // 2, 1, 1, 0),
        )
        self.stg2_high_band_net = BaseNet(
            nout // 4 + 2, nout // 2, self.nin_lstm // 2, nout_lstm // 2
        )

        self.stg3_full_band_net = BaseNet(
            3 * nout // 4 + 2, nout, self.nin_lstm, nout_lstm
        )

        # Output layer for generating the final mask.
        self.out = nn.Conv2d(nout, 2, 1, bias=False)
        # Auxiliary output layer for intermediate supervision during training.
        self.aux_out = nn.Conv2d(3 * nout // 4, 2, 1, bias=False)

    def forward(self, input_tensor):
        # Preprocess input tensor to match the maximum frequency bin.
        input_tensor = input_tensor[:, :, : self.max_bin]

        # Split the input into low and high frequency bands.
        bandw = input_tensor.size()[2] // 2
        l1_in = input_tensor[:, :, :bandw]
        h1_in = input_tensor[:, :, bandw:]

        # Process each band through the first stage networks.
        l1 = self.stg1_low_band_net(l1_in)
        h1 = self.stg1_high_band_net(h1_in)

        # Combine the outputs for auxiliary supervision.
        aux1 = torch.cat([l1, h1], dim=2)

        # Prepare inputs for the second stage by concatenating the original and processed bands.
        l2_in = torch.cat([l1_in, l1], dim=1)
        h2_in = torch.cat([h1_in, h1], dim=1)

        # Process through the second stage networks.
        l2 = self.stg2_low_band_net(l2_in)
        h2 = self.stg2_high_band_net(h2_in)

        # Combine the outputs for auxiliary supervision.
        aux2 = torch.cat([l2, h2], dim=2)

        # Prepare input for the third stage by concatenating all previous outputs with the original input.
        f3_in = torch.cat([input_tensor, aux1, aux2], dim=1)

        # Process through the third stage network.
        f3 = self.stg3_full_band_net(f3_in)

        # Apply the output layer to generate the final mask and apply sigmoid for normalization.
        mask = torch.sigmoid(self.out(f3))

        # Pad the mask to match the output frequency bin size.
        mask = F.pad(
            input=mask,
            pad=(0, 0, 0, self.output_bin - mask.size()[2]),
            mode="replicate",
        )

        # During training, generate and pad the auxiliary output for additional supervision.
        if self.training:
            aux = torch.cat([aux1, aux2], dim=1)
            aux = torch.sigmoid(self.aux_out(aux))
            aux = F.pad(
                input=aux,
                pad=(0, 0, 0, self.output_bin - aux.size()[2]),
                mode="replicate",
            )
            return mask, aux
        else:
            return mask

    # Method for predicting the mask given an input tensor.
    def predict_mask(self, input_tensor):
        mask = self.forward(input_tensor)

        # If an offset is specified, crop the mask to remove edge artifacts.
        if self.offset > 0:
            mask = mask[:, :, :, self.offset : -self.offset]
            assert mask.size()[3] > 0

        return mask

    # Method for applying the predicted mask to the input tensor to obtain the predicted magnitude.
    def predict(self, input_tensor):
        mask = self.forward(input_tensor)
        pred_mag = input_tensor * mask

        # If an offset is specified, crop the predicted magnitude to remove edge artifacts.
        if self.offset > 0:
            pred_mag = pred_mag[:, :, :, self.offset : -self.offset]
            assert pred_mag.size()[3] > 0

        return pred_mag
