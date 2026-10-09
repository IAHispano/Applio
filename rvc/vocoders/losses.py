import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple

def _branch_weight(branch_weights, index):
    """``branch_weights[index]`` as a float, or 1.0 when no weighting is asked for.

    A plain Python number rather than a tensor: these are per-branch constants
    the caller reads once off the discriminator, and keeping them out of the
    graph is what makes an unweighted call byte-identical to the code that had
    no weighting at all.
    """

    if branch_weights is None:
        return 1.0
    return float(branch_weights[index])


def feature_loss(fmap_r, fmap_g, normalize=False, branch_weights=None):
    """Feature matching, optionally weighted per discriminator branch.

    ``branch_weights`` is one number per entry of ``fmap_r``, in branch order --
    ``MPD_MSD_Combined.branch_weights`` is exactly that.  It scales the whole
    branch, every layer of it, because a branch's feature-matching pull and its
    adversarial pull are the same branch's opinion and down-weighting only one
    of them would leave the generator chasing features from a head whose score
    it was told to discount.
    """

    def l1(r, g):
        # ``g`` is promoted inside the subtraction rather than copied to FP32
        # first, and the norm is ``sum |r - g|`` in one pass instead of two.
        return torch.linalg.vector_norm(r.float() - g, ord=1)

    terms = []
    weights = []
    for index, (dr, dg) in enumerate(zip(fmap_r, fmap_g)):
        weight = _branch_weight(branch_weights, index)
        for rl, gl in zip(dr, dg):
            # A tuple is one map split into bands: the mean over all its elements.
            if isinstance(rl, tuple):
                total = torch.stack([l1(r, g) for r, g in zip(rl, gl)]).sum()
                count = sum(r.numel() for r in rl)
            else:
                total, count = l1(rl, gl), rl.numel()
            terms.append(total * (weight / count))
            weights.append(weight)
    if not terms:
        first = fmap_r[0][0]
        device = (first[0] if isinstance(first, tuple) else first).device
        return torch.zeros((), device=device)
    # One reduction, not a chain of scalar adds -- there is a term per map.
    loss = torch.stack(terms).sum()
    if not normalize:
        return loss
    # The weighted analogue of dividing by the term count: with the default
    # weights this is ``len(terms)`` and the two agree exactly.
    total = sum(weights)
    return loss / total if total else loss


def loud_crop(real, count, segment, cond=None, hop=1):
    """One random ``segment``-sample window of ``real``, then its ``count``
    loudest clips.

    For the extra discriminator passes (R1, the HF floor negative): a shorter
    window costs proportionally less, and the loudest clips skip mutes, which
    cost the same and say nothing.  ``segment`` 0 keeps the whole clip.
    Neither choice touches the host.

    With ``cond`` (frame-rate features, ``hop`` samples per frame) the window
    snaps to whole frames, the same window and clips are taken from it, and
    ``(real, cond)`` is returned.
    """
    real = real.detach()
    length = real.shape[-1]
    if cond is not None:
        segment = segment // hop * hop
    if 0 < segment < length:
        start = int(torch.randint(0, (length - segment) // hop + 1, ())) * hop
        real = real[..., start : start + segment]
        if cond is not None:
            cond = cond[..., start // hop : (start + segment) // hop]
    if count < real.shape[0]:
        loudness = real.float().square().mean(dim=tuple(range(1, real.dim())))
        indices = loudness.topk(count).indices
        real = real.index_select(0, indices)
        if cond is not None:
            cond = cond.index_select(0, indices)
    return real if cond is None else (real, cond.detach())


def r1_penalty(discriminator, real, branch, dtype=None, cond=None):
    """R1 for one branch: batch mean of ``||d score / d real||^2``,
    differentiable in D's weights.

    ``discriminator`` is the unwrapped ``MPD_MSD_Combined``.  One branch per
    call keeps the double-backward graph, and the VRAM it peaks at, to that
    branch.  ``dtype`` is the autocast type to run under; pass ``bfloat16``
    only -- in FP16 the squared input gradient can underflow, so anything else
    runs in FP32.  ``cond`` goes to a mel-conditioned branch.
    """
    real = real.float().requires_grad_(True)
    enabled = dtype == torch.bfloat16
    with torch.autocast(real.device.type, dtype=torch.bfloat16, enabled=enabled):
        score = discriminator.real_score(real, branch, cond=cond)
        (grad,) = torch.autograd.grad(score.sum(), real, create_graph=True)
    return grad.float().square().flatten(1).sum(1).mean()


def discriminator_loss(
    disc_real_outputs,
    disc_generated_outputs,
    san_direction_weight=1.0,
    normalize=False,
    per_branch=False,
    branch_weights=None,
):
    """Discriminator loss, aggregated across all MPD/MSD heads.

    With ``per_branch``, a fourth element is appended: a detached
    ``(heads, 2)`` tensor of each head's ``(real, fake)`` contribution before
    the ``normalize`` division -- the aggregate alone hides which head (e.g.
    a period vs. a spectrogram branch) is collapsing.  That tensor is reported
    *unweighted*: it exists to say what a head is doing, and scaling it by the
    weight the head was given would hide exactly the state the weight is there
    to manage.

    ``branch_weights`` scales each head's contribution to the trained
    objective, in branch order.  Weighting the discriminator's own loss as well
    as the generator's is deliberate: a head the generator is told to discount
    but that still trains at full rate keeps pulling away, and the gap it opens
    is what the weight was meant to close.
    """
    loss = 0
    loss_real = 0
    loss_fake = 0
    branch_losses = [] if per_branch else None
    branch_count = 0
    weight_total = 0.0
    for index, (dr, dg) in enumerate(zip(disc_real_outputs, disc_generated_outputs)):
        branch_count += 1
        weight = _branch_weight(branch_weights, index)
        weight_total += weight
        if isinstance(dr, (list, tuple)):
            dr_fun, dr_dir = dr
            dg_fun, dg_dir = dg
            # SAN splits every head into a *function* output (trains scale and
            # trunk) and a *direction* output (trains the unit-norm projection
            # only); both need the same one-sided, bounded surrogate. Mirroring
            # the function term here (rather than `-w * softplus(1-dg_dir)**2`)
            # keeps the fake-direction term bounded below and saturating --
            # the unbounded form let the discriminator win by pushing the
            # direction output on fakes negative without discriminating at all.
            r_loss = (
                torch.mean(F.softplus(1 - dr_fun.float()) ** 2)
                + float(san_direction_weight) * torch.mean(F.softplus(1 - dr_dir.float()) ** 2)
            )
            g_loss = (
                torch.mean(F.softplus(dg_fun.float()) ** 2)
                + float(san_direction_weight) * torch.mean(F.softplus(dg_dir.float()) ** 2)
            )
        else:
            r_loss = torch.mean((1 - dr.float()) ** 2)
            g_loss = torch.mean(dg.float() ** 2)
        if branch_losses is not None:
            branch_losses.append(
                torch.stack((r_loss.detach(), g_loss.detach()))
            )
        if weight != 1.0:
            r_loss = weight * r_loss
            g_loss = weight * g_loss
        loss += r_loss + g_loss
        loss_real += r_loss
        loss_fake += g_loss

    if normalize and branch_count:
        divisor = weight_total if branch_weights is not None else branch_count
        if divisor:
            loss = loss / divisor
            loss_real = loss_real / divisor
            loss_fake = loss_fake / divisor
    if branch_losses is not None:
        return loss, loss_real, loss_fake, torch.stack(branch_losses)
    return loss, loss_real, loss_fake


def generator_loss(
    disc_outputs,
    normalize=False,
    san_direction_weight=1.0,
    use_softplus=False,
    branch_weights=None,
    per_branch=False,
):
    """
    Generator loss with LSGAN as the default and optional SAN softplus loss.

    ``branch_weights`` scales each head's term, in branch order; see
    ``MPD_MSD_Combined.branch_weights`` for where they come from and why one
    head needs them.

    With ``per_branch``, a second element is appended: a detached ``(heads,)``
    tensor of each head's contribution, **after** its weight and before the
    ``normalize`` division.  Weighted, unlike ``discriminator_loss``'s
    per-branch tensor, and the difference is the point of each.  That one
    reports what a head *is* -- a state a weight must not disguise.  This one
    reports what a head *costs the generator*, which is the quantity a weight
    is chosen against, so reading it post-weight is what makes it an answer
    rather than an input to a mental multiplication.

    Why it is not derivable from ``disc_sep``: separation is a logit gap and
    this is a saturating function of it, so a head separating 50x better than
    another does not contribute 50x the term.  Judging a weight off the
    separations alone is exactly the arithmetic this series exists to remove.
    """
    losses = []
    weights = []
    branch_terms = [] if per_branch else None
    for index, dg in enumerate(disc_outputs):
        weight = _branch_weight(branch_weights, index)
        weights.append(weight)
        if isinstance(dg, (list, tuple)):
            if use_softplus:
                dg = dg[0]
            else:
                l = torch.mean((1 - dg[0].float()) ** 2)
                if len(dg) > 1:
                    l = l + float(san_direction_weight) * torch.mean(
                        (1 - dg[1].float()) ** 2
                    )
                l = weight * l if weight != 1.0 else l
                losses.append(l)
                if branch_terms is not None:
                    branch_terms.append(l.detach())
                continue
        if use_softplus:
            l = torch.mean(F.softplus(1.0 - dg.float()).square())
        else:
            l = torch.mean((1 - dg.float()) ** 2)
        l = weight * l if weight != 1.0 else l
        losses.append(l)
        if branch_terms is not None:
            branch_terms.append(l.detach())

    if not losses:
        empty = torch.zeros(())
        return (empty, empty.reshape(0)) if per_branch else empty
    loss = sum(losses)
    if normalize:
        total = sum(weights)
        if total:
            loss = loss / total
    if branch_terms is not None:
        return loss, torch.stack(branch_terms)
    return loss


class MultiScaleSTFTLoss(nn.Module):
    """Spectral convergence and log-magnitude loss at multiple STFT resolutions."""

    def __init__(
        self,
        fft_sizes: Tuple[int, ...] = (512, 1024, 2048),
        hop_sizes: Tuple[int, ...] = (128, 256, 512),
        win_sizes: Tuple[int, ...] = (512, 1024, 2048),
        log_scale: float = 1000.0,
        spectral_convergence: bool = False,
    ):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes
        #: Compression knee, matching ``wave_to_mel(for_loss=True)``.  See
        #: :meth:`forward` for why the compression is ``log1p`` and not ``log``.
        self.log_scale = float(log_scale)
        #: Spectral convergence, off by default. ``||X - X̂||_F / ||X||_F``'s
        #: Frobenius norm is dominated by the loudest bins (measured: top 1%
        #: above 7.1 vs. median 0.013), which duplicates what the low-frequency
        #: mel term already covers and works against MS-STFT's actual value:
        #: linear frequency resolution at the top of the band.
        self.spectral_convergence = bool(spectral_convergence)

    def _stft(self, x: torch.Tensor, fft_size: int, hop_size: int, win_size: int) -> torch.Tensor:
        x = x.float().squeeze(1)
        x = F.pad(x, (win_size // 2, win_size // 2), mode='reflect')

        window = torch.hann_window(win_size, device=x.device, dtype=x.dtype)
        stft = torch.stft(
            x, fft_size, hop_size, win_size, window,
            return_complex=True, center=False
        )
        return stft.abs()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute multi-scale STFT loss over ``(B, T)`` audio.

        Uses ``log1p(mag * log_scale)``, matching ``wave_to_mel(for_loss=True)``,
        instead of ``log(mag.clamp(1e-5))``: on this dataset 8% of bins are
        real digital silence, and the clamp's fixed floor gives those bins a
        gradient ~13x larger than audible ones, pushing silence toward more
        silence. ``log1p`` agrees with ``log`` at audible levels but turns
        linear below ``1 / log_scale``, so near-zero bins are scored on
        distance from zero instead of a ratio between two inaudible numbers.
        """
        sc_loss = 0.0
        mag_loss = 0.0

        for fft_size, hop_size, win_size in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            pred_mag = self._stft(pred, fft_size, hop_size, win_size)
            target_mag = self._stft(target, fft_size, hop_size, win_size)

            if self.spectral_convergence:
                flat_target = target_mag.reshape(target_mag.size(0), -1)
                flat_diff = (target_mag - pred_mag).reshape(target_mag.size(0), -1)
                target_nrg = torch.norm(flat_target, p=2, dim=1)
                diff_nrg = torch.norm(flat_diff, p=2, dim=1)

                # SC is undefined for zero-energy targets
                mask = target_nrg > 1e-4
                if mask.any():
                    sc_loss += (diff_nrg[mask] / target_nrg[mask]).mean()

            mag_loss += F.l1_loss(
                torch.log1p(pred_mag * self.log_scale),
                torch.log1p(target_mag * self.log_scale),
            )

        if self.spectral_convergence and sc_loss != 0.0:
            sc_loss = sc_loss / len(self.fft_sizes)
        mag_loss = mag_loss / len(self.fft_sizes)
        return sc_loss + mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """Spectral convergence and log-magnitude losses, averaged over the resolutions."""

    def __init__(self, fft_sizes, hop_sizes, win_lengths):
        super().__init__()
        self.resolutions = list(zip(fft_sizes, hop_sizes, win_lengths))
        for index, (_, _, win_length) in enumerate(self.resolutions):
            self.register_buffer(f"window_{index}", torch.hann_window(win_length), persistent=False)

    def _magnitude(self, x, index):
        fft_size, hop_size, win_length = self.resolutions[index]
        window = getattr(self, f"window_{index}")
        spec = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=True)
        return torch.clamp(spec.abs(), min=10**-3.5)

    def forward(self, x, y):
        """Predicted ``x`` and target ``y``, [B, T] each."""
        x, y = x.float(), y.float()
        sc_loss = mag_loss = 0.0
        for index in range(len(self.resolutions)):
            x_mag, y_mag = self._magnitude(x, index), self._magnitude(y, index)
            sc_loss = sc_loss + torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")
            mag_loss = mag_loss + F.l1_loss(torch.log(y_mag), torch.log(x_mag))
        return sc_loss / len(self.resolutions), mag_loss / len(self.resolutions)
