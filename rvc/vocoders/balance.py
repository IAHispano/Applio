"""Discriminator learning rate driven by how well each family of heads separates."""

import torch


def head_family(label):
    """The family a ``MPD_MSD_Combined.branch_labels`` entry belongs to."""
    if label.startswith("period_"):
        return "mpd"
    if label.startswith("resolution_"):
        return "mrd"
    return label


def head_accuracies(real_outputs, fake_outputs):
    """Per-head fraction of logits on the right side of 0.5 -- the equilibrium
    of both the LSGAN and the SAN softplus losses.

    0.5 is chance (the generator fools the head), 1.0 is a head that is never
    wrong.  Returns a detached ``(heads,)`` device tensor; under SAN the
    function output is the one read.
    """
    accuracies = []
    for dr, dg in zip(real_outputs, fake_outputs):
        if isinstance(dr, (list, tuple)):
            dr, dg = dr[0], dg[0]
        dr, dg = dr.detach().float(), dg.detach().float()
        accuracies.append(
            0.5 * ((dr > 0.5).float().mean() + (dg < 0.5).float().mean())
        )
    return torch.stack(accuracies)


class FamilyReducer:
    """Reduces per-head accuracies to one weighted mean per family.

    Built once per epoch so the step only pays one matrix product.
    """

    def __init__(self, labels, weights, device):
        self.names = []
        for label in labels:
            family = head_family(label)
            if family not in self.names:
                self.names.append(family)
        weights = [1.0] * len(labels) if weights is None else [float(w) for w in weights]
        matrix = torch.zeros(len(self.names), len(labels))
        for index, (label, weight) in enumerate(zip(labels, weights)):
            matrix[self.names.index(head_family(label)), index] = weight
        self.matrix = (matrix / matrix.sum(dim=1, keepdim=True)).to(device)

    def __call__(self, accuracies):
        return self.matrix @ accuracies
