"""Deterministic corpus sampling without letting the longest domain dominate.

The optional mixed-corpus sampler weights domains by square-root audio duration,
chooses a voice uniformly inside that domain, then a segment uniformly inside
that voice. This increases singing/expressive exposure without assigning the
smallest domain the same total weight as the largest. Rank partitioning happens
after the shared global draw, preserving exact-resume/DDP determinism.
"""

from collections import defaultdict
import math

import torch


class CorpusSampler:
    def __init__(self, dataset, mode="segments"):
        if mode not in {"segments", "domain-speaker"}:
            raise ValueError("Unknown corpus sampling mode")
        self.mode, self.size = mode, len(dataset)
        if self.size < 1:
            raise ValueError("Cannot sample an empty corpus")
        domains = defaultdict(lambda: defaultdict(list))
        duration = defaultdict(float)
        for index, entry in enumerate(dataset.entries):
            speaker = dataset.manifest["speakers"][entry["speaker"]]
            domain = speaker.split("/", 1)[0] if "/" in speaker else "default"
            domains[domain][speaker].append(index)
            duration[domain] += entry["samples"] / dataset.config.sample_rate
        self.domains = sorted(domains)
        self.groups = [[domains[domain][speaker] for speaker in sorted(domains[domain])]
                       for domain in self.domains]
        self.weights = torch.tensor([math.sqrt(duration[d]) for d in self.domains], dtype=torch.float64)
        self.weights /= self.weights.sum()
        self.summary = dict(mode=mode, domains=[dict(domain=domain, speakers=len(domains[domain]),
            seconds=duration[domain], probability=float(self.weights[i])) for i, domain in enumerate(self.domains)])

    def draw(self, count, generator):
        if count < 1:
            raise ValueError("Sampling count must be positive")
        if self.mode == "segments":
            # Match the original draw exactly, including its RNG consumption.
            return torch.randint(self.size, (count,), generator=generator)
        selected = torch.multinomial(self.weights, count, replacement=True, generator=generator).tolist()
        indices = []
        for domain in selected:
            voices = self.groups[domain]
            voice = int(torch.randint(len(voices), (), generator=generator))
            segments = voices[voice]
            position = int(torch.randint(len(segments), (), generator=generator))
            indices.append(segments[position])
        return torch.tensor(indices, dtype=torch.long)


def validation_indices(dataset, limit=0):
    """Fixed voice-covered panel, preferring different recordings per voice.

    Zero preserves the original full validation order. A bounded panel changes
    checkpoint selection, so the trainer records both its version and indices.
    Final held-out evaluation remains a separate, explicitly reported scope.
    """
    if limit < 0:
        raise ValueError("Validation limit cannot be negative")
    if not limit or limit >= len(dataset):
        return list(range(len(dataset)))
    voices = defaultdict(lambda: defaultdict(list))
    for index, entry in enumerate(dataset.entries):
        voices[entry["speaker"]][entry["source_hash"]].append(index)
    if limit < len(voices):
        raise ValueError("Validation panel must cover every held-out voice")
    groups = []
    for speaker in sorted(voices):
        records = voices[speaker]
        ordered = [sorted(records[source], key=lambda i: dataset.entries[i]["start_sample"])
                   for source in sorted(records)]
        # First segment from each recording before any second segment.
        indices = [record[offset] for offset in range(max(map(len, ordered)))
                   for record in ordered if offset < len(record)]
        groups.append(indices)
    return [group[offset] for offset in range(max(map(len, groups)))
            for group in groups if offset < len(group)][:limit]
