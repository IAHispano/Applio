"""Slicing Adversarial Network heads (arXiv 2301.12811, ICLR 2024).

A branch's last projection is split into a **direction** on the unit sphere and
a **scale**.  The direction is trained only by its own term, reading a detached
input; the scale and the trunk are trained by the ordinary adversarial term
against a detached direction.  The claim is that this makes the discriminator
induce a metric between the two distributions rather than a decision boundary,
so the generator receives a direction to move in rather than a verdict.

Part of the ``v4`` discriminator, which is RefineGAN2's, and of nothing else.
It is not a switch: ``v2`` and ``v3`` are the layouts every existing Applio
checkpoint was trained against, and SAN changes ``conv_post``'s state-dict keys,
so putting it behind a flag on those would only create a way to make an old
discriminator unloadable.

Three things a caller has to get right:

* No ``weight_norm`` on top -- these layers normalise their own weight.
* ``normalize_weight()`` after every optimizer step: the direction is on the
  sphere only because it is put back there, and an Adam step moves it off.
* ``san_training=True`` in the discriminator update only.  The generator may
  not move the direction, so asking for it there builds a graph nothing reads.

The float32 island is deliberate: the head is a handful of parameters against
the branch's millions, and a normalised projection under FP16 autocast loses the
small differences it exists to measure.
"""

import torch
from torch import nn
from torch.nn import functional as F


#: Weight on the direction term.  It trains the unit-norm projection only, a
#: quantity the generator never sees; at 1.0 it doubles ``loss_disc`` for that.
SAN_DIRECTION_WEIGHT = 0.25


def normalize_weight(weight):
    """Project each output channel's kernel onto the unit sphere."""
    shape = (weight.shape[0],) + (1,) * (weight.ndim - 1)
    norm = weight.flatten(1).norm(p=2, dim=1).clamp_min(1e-12)
    return weight / norm.view(shape)


class _SANConvMixin:
    #: ``scale`` multiplies a unit-norm projection, so it *is* the branch's
    #: output gain: unbounded it drifts and takes the loss with it, and at zero
    #: the direction stops receiving any gradient.
    SCALE_MIN = 1e-4
    SCALE_MAX = 4.0

    def _split_weight(self, view_shape):
        scale = self.weight.detach().flatten(1).norm(p=2, dim=1).clamp_min(1e-12)
        self.weight = nn.Parameter(self.weight.detach() / scale.view(view_shape))
        self.scale = nn.Parameter(scale)
        if self.bias is not None:
            # The bias moves to the input side, in input channels: the output is
            # a normalised projection times a scale, and an additive term after
            # it would sit outside the geometry the method is about.
            self.bias = nn.Parameter(
                torch.zeros(
                    self.in_channels,
                    device=self.weight.device,
                    dtype=self.weight.dtype,
                )
            )

    def _san_forward(self, conv, input, san_training, weight_view, bias_view):
        with torch.autocast(device_type=input.device.type, enabled=False):
            input = input.float()
            direction = normalize_weight(self.weight.float())
            scale = self.scale.float().clamp(self.SCALE_MIN, self.SCALE_MAX)
            scale = scale.view(weight_view)
            if self.bias is not None:
                input = input + self.bias.float().view(bias_view)
            if not san_training:
                return conv(input, direction) * scale
            # The detaches are the method: ``function`` trains the trunk and the
            # scale without moving the direction, ``direction`` trains the
            # projection alone.
            function_output = conv(input, direction.detach()) * scale
            direction_output = conv(input.detach(), direction) * scale.detach()
            return function_output, direction_output

    @torch.no_grad()
    def normalize_weight(self):
        """Put the direction back on the sphere after an optimizer step."""
        self.weight.copy_(normalize_weight(self.weight))


class SANConv1d(_SANConvMixin, nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._split_weight((-1, 1, 1))

    def forward(self, input, san_training: bool = False):
        def conv(x, weight):
            return F.conv1d(
                x, weight, None, self.stride, self.padding, self.dilation, self.groups
            )

        return self._san_forward(
            conv, input, san_training, (1, self.out_channels, 1), (1, self.in_channels, 1)
        )


class SANConv2d(_SANConvMixin, nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._split_weight((-1, 1, 1, 1))

    def forward(self, input, san_training: bool = False):
        def conv(x, weight):
            return F.conv2d(
                x, weight, None, self.stride, self.padding, self.dilation, self.groups
            )

        return self._san_forward(
            conv,
            input,
            san_training,
            (1, self.out_channels, 1, 1),
            (1, self.in_channels, 1, 1),
        )


def san_tail(module, x, fmap, san_training):
    """``conv_post`` plus the flatten, in the one shape SAN can change.

    Every branch ends the same way; factoring the ending keeps ``use_san`` from
    being four near-identical edits that drift apart.
    """
    if not getattr(module, "use_san", False):
        x = module.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap

    out = module.conv_post(x, san_training=san_training)
    if not san_training:
        fmap.append(out)
        return torch.flatten(out, 1, -1), fmap

    function_output, direction_output = out
    # The *function* output is the feature map: it is what feature matching
    # reads, and the direction output is not something the generator may see.
    fmap.append(function_output)
    return [
        torch.flatten(function_output, 1, -1),
        torch.flatten(direction_output, 1, -1),
    ], fmap


def normalize_san_weights(net_d):
    """Reproject every SAN direction after ``optim_d.step()``.

    A no-op on a discriminator without SAN heads.  Not optional on one with
    them: nothing raises when the direction drifts off the sphere, it simply
    stops being a projection.
    """
    model = net_d.module if hasattr(net_d, "module") else net_d
    for module in model.modules():
        normalize = getattr(module, "normalize_weight", None)
        if normalize is not None:
            normalize()
