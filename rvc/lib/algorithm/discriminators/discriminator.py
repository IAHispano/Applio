import torch.nn as nn
class CombinedDiscriminator(nn.Module):
    def __init__(self, discriminators):
        super().__init__()
        self.discriminators = nn.ModuleList(discriminators)

    def forward(self, y, y_hat):
        os = [d(y, y_hat) for d in self.discriminators]
        return [sum(o, []) for o in zip(*os)]

