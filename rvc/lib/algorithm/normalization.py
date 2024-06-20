import torch


class LayerNorm(torch.nn.Module):
    """Layer normalization module.

    Args:
        channels (int): Number of channels.
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-5.

    """

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, time_steps).

        Returns:
            torch.Tensor: Layer-normalized tensor of shape (batch_size, channels, time_steps).

        """
        x = x.transpose(1, -1)
        x = torch.nn.functional.layer_norm(
            x, (self.channels,), self.gamma, self.beta, self.eps
        )
        return x.transpose(1, -1)
