"""Compatibility entry point for callers using the standalone selector."""


def pretrained_selector(vocoder, sample_rate):
    from rvc.lib.tools.prerequisites_download import ensure_pretrained

    return ensure_pretrained(vocoder, int(sample_rate))
