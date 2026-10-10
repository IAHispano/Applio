import torch
from torch import nn
import torch.nn.functional as F

from uvr import spectral as spec_utils


class Conv2DBNActiv(nn.Module):
    """
    This class implements a convolutional layer followed by batch normalization and an activation function.
    It is a common pattern in deep learning for processing images or feature maps. The convolutional layer
    applies a set of learnable filters to the input. Batch normalization then normalizes the output of the
    convolution, and finally, an activation function introduces non-linearity to the model, allowing it to
    learn more complex patterns.

    Attributes:
        conv (nn.Sequential): A sequential container of Conv2d, BatchNorm2d, and an activation layer.

    Args:
        num_input_channels (int): Number of input channels.
        num_output_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the kernel. Defaults to 3.
        stride_length (int, optional): Stride of the convolution. Defaults to 1.
        padding_size (int, optional): Padding added to all sides of the input. Defaults to 1.
        dilation_rate (int, optional): Spacing between kernel elements. Defaults to 1.
        activation_function (callable, optional): The activation function to use. Defaults to nn.ReLU.
    """

    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, dilation=1, activ=nn.ReLU):
        super(Conv2DBNActiv, self).__init__()

        # The nn.Sequential container allows us to stack the Conv2d, BatchNorm2d, and activation layers
        # into a single module, simplifying the forward pass.
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
        # Defines the computation performed at every call.
        # Simply passes the input through the sequential container.
        return self.conv(input_tensor)


class SeperableConv2DBNActiv(nn.Module):
    """
    This class implements a separable convolutional layer followed by batch normalization and an activation function.
    Separable convolutions are a type of convolution that splits the convolution operation into two simpler operations:
    a depthwise convolution and a pointwise convolution. This can reduce the number of parameters and computational cost,
    making the network more efficient while maintaining similar performance.

    The depthwise convolution applies a single filter per input channel (input depth). The pointwise convolution,
    which follows, applies a 1x1 convolution to combine the outputs of the depthwise convolution across channels.
    Batch normalization is then applied to stabilize learning and reduce internal covariate shift. Finally,
    an activation function introduces non-linearity, allowing the network to learn complex patterns.
    Attributes:
        conv (nn.Sequential): A sequential container of depthwise Conv2d, pointwise Conv2d, BatchNorm2d, and an activation layer.

    Args:
        num_input_channels (int): Number of input channels.
        num_output_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the kernel for the depthwise convolution. Defaults to 3.
        stride_length (int, optional): Stride of the convolution. Defaults to 1.
        padding_size (int, optional): Padding added to all sides of the input for the depthwise convolution. Defaults to 1.
        dilation_rate (int, optional): Spacing between kernel elements for the depthwise convolution. Defaults to 1.
        activation_function (callable, optional): The activation function to use. Defaults to nn.ReLU.
    """

    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, dilation=1, activ=nn.ReLU):
        super(SeperableConv2DBNActiv, self).__init__()

        # Initialize the sequential container with the depthwise convolution.
        # The number of groups in the depthwise convolution is set to num_input_channels, which means each input channel is treated separately.
        # The pointwise convolution then combines these separate channels into num_output_channels channels.
        # Batch normalization is applied to the output of the pointwise convolution.
        # Finally, the activation function is applied to introduce non-linearity.
        self.conv = nn.Sequential(
            nn.Conv2d(
                nin,
                nin,  # For depthwise convolution, in_channels = out_channels = num_input_channels
                kernel_size=ksize,
                stride=stride,
                padding=pad,
                dilation=dilation,
                groups=nin,  # This makes it a depthwise convolution
                bias=False,  # Bias is not used because it will be handled by BatchNorm2d
            ),
            nn.Conv2d(
                nin,
                nout,  # Pointwise convolution to combine channels
                kernel_size=1,  # Kernel size of 1 for pointwise convolution
                bias=False,  # Bias is not used because it will be handled by BatchNorm2d
            ),
            nn.BatchNorm2d(nout),  # Normalize the output of the pointwise convolution
            activ(),  # Apply the activation function
        )

    def __call__(self, input_tensor):
        # Pass the input through the sequential container.
        # This performs the depthwise convolution, followed by the pointwise convolution,
        # batch normalization, and finally applies the activation function.
        return self.conv(input_tensor)


class Encoder(nn.Module):
    """
    The Encoder class is a part of the neural network architecture that is responsible for processing the input data.
    It consists of two convolutional layers, each followed by batch normalization and an activation function.
    The purpose of the Encoder is to transform the input data into a higher-level, abstract representation.
    This is achieved by applying filters (through convolutions) that can capture patterns or features in the data.
    The Encoder can be thought of as a feature extractor that prepares the data for further processing by the network.
    Attributes:
        conv1 (Conv2DBNActiv): The first convolutional layer in the encoder.
        conv2 (Conv2DBNActiv): The second convolutional layer in the encoder.

    Args:
        number_of_input_channels (int): Number of input channels for the first convolutional layer.
        number_of_output_channels (int): Number of output channels for the convolutional layers.
        kernel_size (int): Kernel size for the convolutional layers.
        stride_length (int): Stride for the convolutional operations.
        padding_size (int): Padding added to all sides of the input for the convolutional layers.
        activation_function (callable): The activation function to use after each convolutional layer.
    """

    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.LeakyReLU):
        super(Encoder, self).__init__()

        # The first convolutional layer takes the input and applies a convolution,
        # followed by batch normalization and an activation function specified by `activation_function`.
        # This layer is responsible for capturing the initial set of features from the input data.
        self.conv1 = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ)

        # The second convolutional layer further processes the output from the first layer,
        # applying another set of convolution, batch normalization, and activation.
        # This layer helps in capturing more complex patterns in the data by building upon the initial features extracted by conv1.
        self.conv2 = Conv2DBNActiv(nout, nout, ksize, stride, pad, activ=activ)

    def __call__(self, input_tensor):
        # The input data `input_tensor` is passed through the first convolutional layer.
        # The output of this layer serves as a 'skip connection' that can be used later in the network to preserve spatial information.
        skip = self.conv1(input_tensor)

        # The output from the first layer is then passed through the second convolutional layer.
        # This processed data `hidden` is the final output of the Encoder, representing the abstracted features of the input.
        hidden = self.conv2(skip)

        # The Encoder returns two outputs: `hidden`, the abstracted feature representation, and `skip`, the intermediate representation from conv1.
        return hidden, skip


class Decoder(nn.Module):
    """
    The Decoder class is part of the neural network architecture, specifically designed to perform the inverse operation of an encoder.
    Its main role is to reconstruct or generate data from encoded representations, which is crucial in tasks like image segmentation or audio processing.
    This class uses upsampling, convolution, optional dropout for regularization, and concatenation of skip connections to achieve its goal.

    Attributes:
        convolution (Conv2DBNActiv): A convolutional layer with batch normalization and activation function.
        dropout_layer (nn.Dropout2d): An optional dropout layer for regularization to prevent overfitting.

    Args:
        input_channels (int): Number of input channels for the convolutional layer.
        output_channels (int): Number of output channels for the convolutional layer.
        kernel_size (int): Kernel size for the convolutional layer.
        stride (int): Stride for the convolutional operations.
        padding (int): Padding added to all sides of the input for the convolutional layer.
        activation_function (callable): The activation function to use after the convolutional layer.
        include_dropout (bool): Whether to include a dropout layer for regularization.
    """

    def __init__(
        self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.ReLU, dropout=False
    ):
        super(Decoder, self).__init__()

        # Initialize the convolutional layer with specified parameters.
        self.conv = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ)

        # Initialize the dropout layer if include_dropout is set to True
        self.dropout = nn.Dropout2d(0.1) if dropout else None

    def __call__(self, input_tensor, skip=None):
        # Upsample the input tensor to a higher resolution using bilinear interpolation.
        input_tensor = F.interpolate(
            input_tensor, scale_factor=2, mode="bilinear", align_corners=True
        )
        # If a skip connection is provided, crop it to match the size of input_tensor and concatenate them along the channel dimension.
        if skip is not None:
            skip = spec_utils.crop_center(
                skip, input_tensor
            )  # Crop skip_connection to match input_tensor's dimensions.
            input_tensor = torch.cat(
                [input_tensor, skip], dim=1
            )  # Concatenate input_tensor and skip_connection along the channel dimension.

        # Pass the concatenated tensor (or just input_tensor if no skip_connection is provided) through the convolutional layer.
        output_tensor = self.conv(input_tensor)

        # If dropout is enabled, apply it to the output of the convolutional layer.
        if self.dropout is not None:
            output_tensor = self.dropout(output_tensor)

        # Return the final output tensor.
        return output_tensor


class ASPPModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) Module is designed for capturing multi-scale context by applying
    atrous convolution at multiple rates. This is particularly useful in segmentation tasks where capturing
    objects at various scales is beneficial. The module applies several parallel dilated convolutions with
    different dilation rates to the input feature map, allowing it to efficiently capture information at
    multiple scales.

    Attributes:
        conv1 (nn.Sequential): Applies adaptive average pooling followed by a 1x1 convolution.
        nn_architecture (int): Identifier for the neural network architecture being used.
        six_layer (list): List containing architecture identifiers that require six layers.
        seven_layer (list): List containing architecture identifiers that require seven layers.
        conv2-conv7 (nn.Module): Convolutional layers with varying dilation rates for multi-scale feature extraction.
        bottleneck (nn.Sequential): A 1x1 convolutional layer that combines all features followed by dropout for regularization.
    """

    def __init__(self, nn_architecture, nin, nout, dilations=(4, 8, 16), activ=nn.ReLU):
        """
        Initializes the ASPP module with specified parameters.

        Args:
            nn_architecture (int): Identifier for the neural network architecture.
            input_channels (int): Number of input channels.
            output_channels (int): Number of output channels.
            dilations (tuple): Tuple of dilation rates for the atrous convolutions.
            activation (callable): Activation function to use after convolutional layers.
        """
        super(ASPPModule, self).__init__()

        # Adaptive average pooling reduces the spatial dimensions to 1x1, focusing on global context,
        # followed by a 1x1 convolution to project back to the desired channel dimension.
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            Conv2DBNActiv(nin, nin, 1, 1, 0, activ=activ),
        )

        self.nn_architecture = nn_architecture
        # Architecture identifiers for models requiring additional layers.
        self.six_layer = [129605]
        self.seven_layer = [537238, 537227, 33966]

        # Extra convolutional layer used for six and seven layer configurations.
        extra_conv = SeperableConv2DBNActiv(
            nin, nin, 3, 1, dilations[2], dilations[2], activ=activ
        )

        # Standard 1x1 convolution for channel reduction.
        self.conv2 = Conv2DBNActiv(nin, nin, 1, 1, 0, activ=activ)

        # Separable convolutions with different dilation rates for multi-scale feature extraction.
        self.conv3 = SeperableConv2DBNActiv(
            nin, nin, 3, 1, dilations[0], dilations[0], activ=activ
        )
        self.conv4 = SeperableConv2DBNActiv(
            nin, nin, 3, 1, dilations[1], dilations[1], activ=activ
        )
        self.conv5 = SeperableConv2DBNActiv(
            nin, nin, 3, 1, dilations[2], dilations[2], activ=activ
        )

        # Depending on the architecture, include the extra convolutional layers.
        if self.nn_architecture in self.six_layer:
            self.conv6 = extra_conv
            nin_x = 6
        elif self.nn_architecture in self.seven_layer:
            self.conv6 = extra_conv
            self.conv7 = extra_conv
            nin_x = 7
        else:
            nin_x = 5

        # Bottleneck layer combines all the multi-scale features into the desired number of output channels.
        self.bottleneck = nn.Sequential(
            Conv2DBNActiv(nin * nin_x, nout, 1, 1, 0, activ=activ), nn.Dropout2d(0.1)
        )

    def forward(self, input_tensor):
        """
        Forward pass of the ASPP module.

        Args:
            input_tensor (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying ASPP.
        """
        _, _, h, w = input_tensor.size()

        # Apply the first convolutional sequence and upsample to the original resolution.
        feat1 = F.interpolate(
            self.conv1(input_tensor), size=(h, w), mode="bilinear", align_corners=True
        )

        # Apply the remaining convolutions directly on the input.
        feat2 = self.conv2(input_tensor)
        feat3 = self.conv3(input_tensor)
        feat4 = self.conv4(input_tensor)
        feat5 = self.conv5(input_tensor)

        # Concatenate features from all layers. Depending on the architecture, include the extra features.
        if self.nn_architecture in self.six_layer:
            feat6 = self.conv6(input_tensor)
            out = torch.cat((feat1, feat2, feat3, feat4, feat5, feat6), dim=1)
        elif self.nn_architecture in self.seven_layer:
            feat6 = self.conv6(input_tensor)
            feat7 = self.conv7(input_tensor)
            out = torch.cat((feat1, feat2, feat3, feat4, feat5, feat6, feat7), dim=1)
        else:
            out = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)

        # Apply the bottleneck layer to combine and reduce the channel dimensions.
        bottleneck_output = self.bottleneck(out)
        return bottleneck_output
