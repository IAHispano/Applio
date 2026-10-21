"""Text frontends that emit 100 Hz RVC inputs directly."""

import math

import torch
from torch import nn
import torch.nn.functional as functional


def positions(length, hidden, device):
    steps = torch.arange(length, device=device).float().unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, hidden, 2, device=device) * (-math.log(10000) / hidden)
    )
    values = torch.zeros(length, hidden, device=device)
    values[:, 0::2] = torch.sin(steps * frequencies)
    values[:, 1::2] = torch.cos(steps * frequencies)
    return values


class TextFrontend(nn.Module):
    @staticmethod
    def tokenize(text):
        from rvc.lib.tools.text_rvc.data import tokenize

        return tokenize(text)

    def __init__(self, feature_dim=768, hidden=128, layers=2, reduction=4):
        super().__init__()
        if (
            feature_dim not in (256, 768)
            or hidden < 16
            or hidden % 4
            or not 1 <= layers <= 8
            or not 1 <= reduction <= 8
        ):
            raise ValueError("Invalid frontend dimensions")
        self.config = {
            "feature_dim": feature_dim,
            "hidden": hidden,
            "layers": layers,
            "reduction": reduction,
        }
        self.feature_dim = feature_dim
        self.reduction = reduction
        self.channels = feature_dim + 2
        self.hidden = hidden
        self.duration_scale = 1.0
        self.embedding = nn.Embedding(258, hidden, padding_idx=0)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                hidden, 4, hidden * 4, dropout=0.1, batch_first=True
            ),
            layers,
            enable_nested_tensor=False,
        )
        self.input_projection = nn.Linear(self.channels * reduction, hidden)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                hidden, 4, hidden * 4, dropout=0.1, batch_first=True
            ),
            layers,
        )
        self.output_projection = nn.Linear(hidden, self.channels * reduction)
        self.duration = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )

    def encode(self, tokens):
        padding = tokens == 0
        embedded = self.embedding(tokens) * math.sqrt(self.hidden)
        memory = self.encoder(
            embedded + positions(tokens.size(1), self.hidden, tokens.device),
            src_key_padding_mask=padding,
        )
        pooled = (memory * (~padding).unsqueeze(-1)).sum(1) / (~padding).sum(
            1
        ).unsqueeze(-1)
        return memory, padding, self.duration(pooled).squeeze(-1)

    def decode(self, memory, text_padding, previous):
        embedded = self.input_projection(previous)
        embedded = embedded + positions(previous.size(1), self.hidden, previous.device)
        causal = torch.triu(
            torch.ones(
                previous.size(1),
                previous.size(1),
                device=previous.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        result = self.decoder(
            embedded, memory, tgt_mask=causal, memory_key_padding_mask=text_padding
        )
        return self.output_projection(result)

    def forward(self, tokens, groups):
        memory, padding, duration = self.encode(tokens)
        shifted = functional.pad(groups[:, :-1], (0, 0, 1, 0))
        return self.decode(memory, padding, shifted), duration

    @torch.inference_mode()
    def generate(self, tokens, max_frames=2000):
        self.eval()
        if tokens.ndim != 2 or tokens.size(0) != 1 or not tokens.size(1):
            raise ValueError("Generate accepts one nonempty text sequence")
        memory, padding, log_steps = self.encode(tokens)
        if not torch.isfinite(log_steps).all():
            raise ValueError("Non-finite duration prediction")
        requested = math.exp(min(20, max(0, float(log_steps[0])))) * self.duration_scale
        steps = max(1, round(requested))
        if steps * self.reduction > max_frames:
            raise ValueError("Predicted speech exceeds the configured frame limit")
        previous = torch.zeros(
            1, 1, self.channels * self.reduction, device=tokens.device
        )
        generated = []
        for _ in range(steps):
            group = self.decode(memory, padding, previous)[:, -1]
            if not torch.isfinite(group).all():
                raise ValueError("Non-finite acoustic prediction")
            frames = group.reshape(1, self.reduction, self.channels).clone()
            # Teacher forcing supplies binary voicing, not its unbounded logits.
            frames[..., -1] = torch.sigmoid(frames[..., -1])
            generated.append(frames)
            previous = torch.cat((previous, frames.reshape(1, 1, -1)), dim=1)
        return torch.cat(generated, dim=1)[0]


class DurationFrontend(nn.Module):
    """Text-only length regulation avoids acoustic teacher-forcing exposure bias."""

    def __init__(
        self,
        feature_dim=768,
        hidden=128,
        layers=2,
        reduction=1,
        phonetic=False,
        phoneme_vocabulary=None,
        language="en",
    ):
        super().__init__()
        if (
            feature_dim not in (256, 768)
            or hidden < 16
            or hidden % 4
            or not 1 <= layers <= 8
            or reduction != 1
        ):
            raise ValueError("Invalid duration frontend dimensions")
        self.config = dict(
            feature_dim=feature_dim,
            hidden=hidden,
            layers=layers,
            reduction=1,
            architecture="duration",
            phonetic=phonetic,
            phoneme_vocabulary=phoneme_vocabulary,
            language=language,
        )
        self.feature_dim, self.hidden, self.reduction = feature_dim, hidden, 1
        self.channels = feature_dim + 2
        self.duration_scale = 1.0
        self.phoneme_vocabulary = phoneme_vocabulary
        self.language = language
        self.vocab_size = 512 if phoneme_vocabulary else 258
        self.eos = 511 if phoneme_vocabulary else 257
        self.embedding = nn.Embedding(self.vocab_size, hidden, padding_idx=0)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                hidden, 4, hidden * 4, dropout=0.1, batch_first=True
            ),
            layers,
            enable_nested_tensor=False,
        )
        self.duration = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.phase = nn.Linear(2, hidden)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(hidden, hidden, 5, padding=2 * dilation, dilation=dilation)
            for dilation in (1, 2, 4, 1, 2, 4)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in self.convolutions)
        self.output_projection = nn.Linear(hidden, self.channels)
        self.phonetic_head = (
            phonetic_head(feature_dim, self.vocab_size) if phonetic else None
        )

    def tokenize(self, text):
        from rvc.lib.tools.text_rvc.alignment import phoneme_labels
        from rvc.lib.tools.text_rvc.data import tokenize

        return (
            [
                label + 1
                for label in phoneme_labels(
                    text, self.phoneme_vocabulary, self.language
                )
            ]
            + [self.eos]
            if self.phoneme_vocabulary
            else tokenize(text)
        )

    def encode(self, tokens):
        padding = tokens == 0
        memory = self.encoder(
            self.embedding(tokens)
            + positions(tokens.size(1), self.hidden, tokens.device),
            src_key_padding_mask=padding,
        )
        return memory, self.duration(memory).squeeze(-1)

    def forward(self, tokens, groups=None, durations=None):
        memory, log_duration = self.encode(tokens)
        if durations is None:
            durations = (
                (log_duration.clamp(0, math.log(301)).expm1() * self.duration_scale)
                .round()
                .long()
            )
            durations = durations.clamp_min(1)
            durations = durations.masked_fill((tokens == 0) | (tokens == self.eos), 0)
        lengths = durations.sum(1)
        if int(lengths.max()) < 1 or int(lengths.max()) > 3000:
            raise ValueError("Predicted speech has an invalid duration")
        expanded = []
        for encoded, timing in zip(memory, durations):
            counts = timing.detach().cpu().tolist()
            sequence = torch.repeat_interleave(
                encoded, timing, dim=0, output_size=sum(counts)
            )
            phases = torch.cat(
                [torch.arange(count) / max(1, count) for count in counts]
            ).to(tokens.device)
            sequence = sequence + self.phase(torch.stack((phases, 1 - phases), -1))
            expanded.append(sequence)
        values = nn.utils.rnn.pad_sequence(expanded, batch_first=True)
        valid = (
            torch.arange(values.size(1), device=tokens.device)[None] < lengths[:, None]
        )
        for convolution, norm in zip(self.convolutions, self.norms):
            values = norm(
                values
                + functional.gelu(convolution(values.transpose(1, 2)).transpose(1, 2))
            )
            values = values * valid.unsqueeze(-1)
        return self.output_projection(values), log_duration

    @torch.inference_mode()
    def generate(self, tokens, max_frames=2000):
        from rvc.lib.tools.text_rvc.alignment import alignment_text
        from rvc.lib.tools.text_rvc.data import tokenize

        if tokens.ndim != 2 or tokens.size(0) != 1:
            raise ValueError("Generate accepts one text sequence")
        # Training and inference share punctuation removal; checkpoint token IDs stay bytes.
        if not self.phoneme_vocabulary:
            text = bytes(
                int(value) - 1 for value in tokens[0] if 1 <= int(value) <= 256
            ).decode("utf-8")
            tokens = torch.tensor(
                [tokenize(alignment_text(text))], device=tokens.device
            )
        self.eval()
        prediction, _ = self(tokens)
        if prediction.size(1) > max_frames or not torch.isfinite(prediction).all():
            raise ValueError("Invalid or oversized acoustic prediction")
        result = prediction[0].clone()
        result[:, -1] = result[:, -1].sigmoid()
        return result


def transfer_frontend(target, source, reset_tokenizer=False):
    """Retain acoustic layers and remap shared phonemes by symbol, never token ID."""
    weights = {
        key: value
        for key, value in source.state_dict().items()
        if not key.startswith(("embedding.", "phonetic_head."))
    }
    missing, unexpected = target.load_state_dict(weights, strict=False)
    if missing != ["embedding.weight"] or unexpected:
        raise ValueError("Acoustic transfer architecture differs")
    shared = 0
    if not reset_tokenizer:
        with torch.no_grad():
            old = getattr(source, "phoneme_vocabulary", None)
            new = getattr(target, "phoneme_vocabulary", None)
            if old and new:
                for phone in old.keys() & new.keys():
                    target.embedding.weight[new[phone] + 1].copy_(
                        source.embedding.weight[old[phone] + 1]
                    )
                    shared += not phone.startswith("<")
            elif (
                old == new
                and source.embedding.weight.shape == target.embedding.weight.shape
            ):
                target.embedding.weight.copy_(source.embedding.weight)
            target.embedding.weight[0].copy_(source.embedding.weight[0])
            target.embedding.weight[target.eos].copy_(
                source.embedding.weight[source.eos]
            )
    return shared


def phonetic_head(dimensions, vocabulary_size=258):
    return nn.Sequential(
        nn.Linear(dimensions, 128), nn.GELU(), nn.Linear(128, vocabulary_size)
    )


def fit_phonetic_head(model, examples, mean, scale, steps=300):
    """Fit a frozen phoneme discriminator with a bounded training-only frame pool."""
    from itertools import islice
    from rvc.lib.tools.text_rvc.data import tokenize

    features, labels = [], []
    # Loading every frame at once makes ordinary multi-hour datasets exhaust RAM.
    for row in islice(examples, 1024):
        timing = torch.repeat_interleave(
            torch.tensor(row.get("tokens") or tokenize(row["text"])),
            torch.tensor(row["durations"]),
        )
        if len(timing) != len(row["features"]):
            raise ValueError("Phoneme durations must cover the acoustic frames")
        selected = torch.randperm(len(timing))[:128]
        features.append((row["features"][selected] - mean) / scale)
        labels.append(timing[selected])
    features, labels = torch.cat(features), torch.cat(labels)
    head = phonetic_head(model.feature_dim, model.vocab_size)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.001)
    for _ in range(steps):
        selection = torch.randint(len(features), (min(2048, len(features)),))
        optimizer.zero_grad(set_to_none=True)
        objective = functional.cross_entropy(
            head(features[selection]), labels[selection], label_smoothing=0.05
        )
        objective.backward()
        optimizer.step()
    model.phonetic_head = head.eval().requires_grad_(False)
    model.config["phonetic"] = True


def batch(examples, mean, scale, reduction):
    from rvc.lib.tools.text_rvc.data import tokenize

    channels = len(mean) + 2
    token_sequences = [
        torch.tensor(row.get("tokens") or tokenize(row["text"]), dtype=torch.long)
        for row in examples
    ]
    tokens = nn.utils.rnn.pad_sequence(token_sequences, batch_first=True)
    lengths = torch.tensor([len(row["features"]) for row in examples])
    frames = math.ceil(int(lengths.max()) / reduction) * reduction
    values = torch.zeros(len(examples), frames, channels)
    valid = torch.arange(frames).unsqueeze(0) < lengths.unsqueeze(1)
    for index, row in enumerate(examples):
        count = int(lengths[index])
        f0 = row["f0"]
        values[index, :count, :-2] = (row["features"] - mean) / scale
        values[index, :count, -2] = torch.where(
            f0 > 0, (f0.clamp_min(1).log() - 5.3) / 0.5, 0
        )
        values[index, :count, -1] = (f0 > 0).float()
    result = (
        tokens,
        values.reshape(len(examples), -1, channels * reduction),
        valid,
        lengths,
    )
    if all("durations" in row for row in examples):
        timing = nn.utils.rnn.pad_sequence(
            [torch.tensor(row["durations"], dtype=torch.long) for row in examples],
            batch_first=True,
        )
        if (
            timing.shape != tokens.shape
            or not torch.equal(timing.sum(1), lengths)
            or (timing < 0).any()
        ):
            raise ValueError("Token durations must cover the acoustic frames")
        return (*result, timing)
    return result


def loss(model, tokens, targets, valid, lengths, durations=None):
    prediction, duration = (
        model(tokens, targets, durations)
        if durations is not None
        else model(tokens, targets)
    )
    prediction = prediction.reshape(*valid.shape, model.channels)
    truth = targets.reshape_as(prediction)
    features = (prediction[..., :-2] - truth[..., :-2]).square().mean(-1)[valid].mean()
    voiced = valid & (truth[..., -1] > 0)
    pitch = (
        functional.smooth_l1_loss(prediction[..., -2][voiced], truth[..., -2][voiced])
        if voiced.any()
        else prediction[..., -2].sum() * 0
    )
    voicing = functional.binary_cross_entropy_with_logits(
        prediction[..., -1][valid], truth[..., -1][valid]
    )
    steps = torch.ceil(lengths.float() / model.reduction)
    duration_loss = (
        functional.smooth_l1_loss(
            duration[tokens != 0], durations.float().log1p()[tokens != 0]
        )
        if durations is not None
        else functional.smooth_l1_loss(duration, steps.log())
    )
    total = features + 0.2 * pitch + 0.2 * voicing + 0.2 * duration_loss
    metrics = {
        "feature_mse": float(features.detach()),
        "pitch_loss": float(pitch.detach()),
        "voicing_loss": float(voicing.detach()),
        "duration_loss": float(duration_loss.detach()),
    }
    if durations is not None and model.phonetic_head is not None:
        labels = nn.utils.rnn.pad_sequence(
            [
                torch.repeat_interleave(text, timing)
                for text, timing in zip(tokens, durations)
            ],
            batch_first=True,
        )
        phonetic = functional.cross_entropy(
            model.phonetic_head(prediction[..., :-2][valid]), labels[valid]
        )
        total = total + 0.15 * phonetic
        metrics["phonetic_loss"] = float(phonetic.detach())
    return total, metrics


def denormalize(values, mean, scale):
    features = values[:, :-2] * scale + mean
    f0 = (values[:, -2] * 0.5 + 5.3).clamp(math.log(50), math.log(1100)).exp()
    f0 = torch.where(values[:, -1] >= 0.5, f0, 0)
    mel = 1127 * torch.log1p(f0 / 700)
    low, high = 1127 * math.log1p(50 / 700), 1127 * math.log1p(1100 / 700)
    coarse = (
        torch.where(mel > 0, (mel - low) * 254 / (high - low) + 1, mel)
        .clamp(1, 255)
        .round()
        .long()
    )
    return features, f0, coarse
