import math


def automatic_block_frame(processing_ms, sample_rate=48000):
    """Keep speech context and 50% timing headroom, rounded to 10 ms."""
    duration_ms = min(1000, max(100, math.ceil(processing_ms * 1.5 / 10) * 10))
    return round(duration_ms * sample_rate / 1000)
