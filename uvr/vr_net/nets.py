import torch
from torch import nn
import torch.nn.functional as F

from . import layers


class BaseASPPNet(nn.Module):
    """
    BaseASPPNet Class:
    This class defines the base architecture for an Atrous Spatial Pyramid Pooling (ASPP) network.
    It is designed to extract features from input data at multiple scales by using dilated convolutions.
    This is particularly useful for tasks that benefit from understanding context at different resolutions,
    such as semantic segmentation. The network consists of a series of encoder layers for downsampling and feature extraction,
    followed by an ASPP module for multi-scale feature extraction, and finally a series of decoder layers for upsampling.
    """

    def __init__(self, nn_architecture, nin, ch, dilations=(4, 8, 16)):
        super(BaseASPPNet, self).__init__()
        self.nn_architecture = nn_architecture

        # Encoder layers progressively increase the number of channels while reducing spatial dimensions.
        self.enc1 = layers.Encoder(nin, ch, 3, 2, 1)
        self.enc2 = layers.Encoder(ch, ch * 2, 3, 2, 1)
        self.enc3 = layers.Encoder(ch * 2, ch * 4, 3, 2, 1)
        self.enc4 = layers.Encoder(ch * 4, ch * 8, 3, 2, 1)

        # Depending on the network architecture, an additional encoder layer and a specific ASPP module are initialized.
        if self.nn_architecture == 129605:
            self.enc5 = layers.Encoder(ch * 8, ch * 16, 3, 2, 1)
            self.aspp = layers.ASPPModule(nn_architecture, ch * 16, ch * 32, dilations)
            self.dec5 = layers.Decoder(ch * (16 + 32), ch * 16, 3, 1, 1)
        else:
            self.aspp = layers.ASPPModule(nn_architecture, ch * 8, ch * 16, dilations)

        # Decoder layers progressively decrease the number of channels while increasing spatial dimensions.
        self.dec4 = layers.Decoder(ch * (8 + 16), ch * 8, 3, 1, 1)
        self.dec3 = layers.Decoder(ch * (4 + 8), ch * 4, 3, 1, 1)
        self.dec2 = layers.Decoder(ch * (2 + 4), ch * 2, 3, 1, 1)
        self.dec1 = layers.Decoder(ch * (1 + 2), ch, 3, 1, 1)

    def __call__(self, input_tensor):
        # The input tensor is passed through a series of encoder layers.
        hidden_state, encoder_output1 = self.enc1(input_tensor)
        hidden_state, encoder_output2 = self.enc2(hidden_state)
        hidden_state, encoder_output3 = self.enc3(hidden_state)
        hidden_state, encoder_output4 = self.enc4(hidden_state)

        # Depending on the network architecture, the hidden state is processed by an additional encoder layer and the ASPP module.
        if self.nn_architecture == 129605:
            hidden_state, encoder_output5 = self.enc5(hidden_state)
            hidden_state = self.aspp(hidden_state)
            # The decoder layers use skip connections from the encoder layers for better feature integration.
            hidden_state = self.dec5(hidden_state, encoder_output5)
        else:
            hidden_state = self.aspp(hidden_state)

        # The hidden state is further processed by the decoder layers, using skip connections for feature integration.
        hidden_state = self.dec4(hidden_state, encoder_output4)
        hidden_state = self.dec3(hidden_state, encoder_output3)
        hidden_state = self.dec2(hidden_state, encoder_output2)
        hidden_state = self.dec1(hidden_state, encoder_output1)

        return hidden_state


def determine_model_capacity(n_fft_bins, nn_architecture):
    """
    The determine_model_capacity function is designed to select the appropriate model configuration
    based on the frequency bins and network architecture. It maps specific architectures to predefined
    model capacities, which dictate the structure and parameters of the CascadedASPPNet model.
    """

    # Predefined model architectures categorized by their precision level.
    sp_model_arch = [31191, 33966, 129605]
    hp_model_arch = [123821, 123812]
    hp2_model_arch = [537238, 537227]

    # Mapping network architectures to their corresponding model capacity data.
    if nn_architecture in sp_model_arch:
        model_capacity_data = [
            (2, 16),
            (2, 16),
            (18, 8, 1, 1, 0),
            (8, 16),
            (34, 16, 1, 1, 0),
            (16, 32),
            (32, 2, 1),
            (16, 2, 1),
            (16, 2, 1),
        ]

    if nn_architecture in hp_model_arch:
        model_capacity_data = [
            (2, 32),
            (2, 32),
            (34, 16, 1, 1, 0),
            (16, 32),
            (66, 32, 1, 1, 0),
            (32, 64),
            (64, 2, 1),
            (32, 2, 1),
            (32, 2, 1),
        ]

    if nn_architecture in hp2_model_arch:
        model_capacity_data = [
            (2, 64),
            (2, 64),
            (66, 32, 1, 1, 0),
            (32, 64),
            (130, 64, 1, 1, 0),
            (64, 128),
            (128, 2, 1),
            (64, 2, 1),
            (64, 2, 1),
        ]

    # Initializing the CascadedASPPNet model with the selected model capacity data.
    cascaded = CascadedASPPNet
    model = cascaded(n_fft_bins, model_capacity_data, nn_architecture)

    return model


class CascadedASPPNet(nn.Module):
    """
    CascadedASPPNet Class:
    This class implements a cascaded version of the ASPP network, designed for processing audio signals
    for tasks such as vocal removal. It consists of multiple stages, each with its own ASPP network,
    to process different frequency bands of the input signal. This allows the model to effectively
    handle the full spectrum of audio frequencies by focusing on different frequency bands separately.
    """

    def __init__(self, n_fft, model_capacity_data, nn_architecture):
        super(CascadedASPPNet, self).__init__()
        # The first stage processes the low and high frequency bands separately.
        self.stg1_low_band_net = BaseASPPNet(nn_architecture, *model_capacity_data[0])
        self.stg1_high_band_net = BaseASPPNet(nn_architecture, *model_capacity_data[1])

        # Bridge layers connect different stages of the network.
        self.stg2_bridge = layers.Conv2DBNActiv(*model_capacity_data[2])
        self.stg2_full_band_net = BaseASPPNet(nn_architecture, *model_capacity_data[3])

        self.stg3_bridge = layers.Conv2DBNActiv(*model_capacity_data[4])
        self.stg3_full_band_net = BaseASPPNet(nn_architecture, *model_capacity_data[5])

        # Output layers for the final mask prediction and auxiliary outputs.
        self.out = nn.Conv2d(*model_capacity_data[6], bias=False)
        self.aux1_out = nn.Conv2d(*model_capacity_data[7], bias=False)
        self.aux2_out = nn.Conv2d(*model_capacity_data[8], bias=False)

        # Parameters for handling the frequency bins of the input signal.
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        self.offset = 128

    def forward(self, input_tensor):
        # The forward pass processes the input tensor through each stage of the network,
        # combining the outputs of different frequency bands and stages to produce the final mask.
        mix = input_tensor.detach()
        input_tensor = input_tensor.clone()

        # Preparing the input tensor by selecting the mainput_tensorimum frequency bin.
        input_tensor = input_tensor[:, :, : self.max_bin]

        # Processing the low and high frequency bands separately in the first stage.
        bandwidth = input_tensor.size()[2] // 2
        aux1 = torch.cat(
            [
                self.stg1_low_band_net(input_tensor[:, :, :bandwidth]),
                self.stg1_high_band_net(input_tensor[:, :, bandwidth:]),
            ],
            dim=2,
        )

        # Combining the outputs of the first stage and passing through the second stage.
        hidden_state = torch.cat([input_tensor, aux1], dim=1)
        aux2 = self.stg2_full_band_net(self.stg2_bridge(hidden_state))

        # Further processing the combined outputs through the third stage.
        hidden_state = torch.cat([input_tensor, aux1, aux2], dim=1)
        hidden_state = self.stg3_full_band_net(self.stg3_bridge(hidden_state))

        # Applying the final output layer to produce the mask.
        mask = torch.sigmoid(self.out(hidden_state))

        # Padding the mask to match the output frequency bin size.
        mask = F.pad(
            input=mask,
            pad=(0, 0, 0, self.output_bin - mask.size()[2]),
            mode="replicate",
        )

        # During training, auxiliary outputs are also produced and padded accordingly.
        if self.training:
            aux1 = torch.sigmoid(self.aux1_out(aux1))
            aux1 = F.pad(
                input=aux1,
                pad=(0, 0, 0, self.output_bin - aux1.size()[2]),
                mode="replicate",
            )
            aux2 = torch.sigmoid(self.aux2_out(aux2))
            aux2 = F.pad(
                input=aux2,
                pad=(0, 0, 0, self.output_bin - aux2.size()[2]),
                mode="replicate",
            )
            return mask * mix, aux1 * mix, aux2 * mix
        else:
            return mask  # * mix

    def predict_mask(self, input_tensor):
        # This method predicts the mask for the input tensor by calling the forward method
        # and applying any necessary padding adjustments.
        mask = self.forward(input_tensor)

        # Adjusting the mask by removing padding offsets if present.
        if self.offset > 0:
            mask = mask[:, :, :, self.offset : -self.offset]

        return mask
