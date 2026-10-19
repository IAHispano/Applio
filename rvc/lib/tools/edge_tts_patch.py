"""Runtime fix for edge-tts 403 (rany2/edge-tts#490).

Since October 2026 ``speech.platform.bing.com`` answers the synthesis
WebSocket with ``403 Invalid response status`` in many regions. Microsoft
Edge 155 synthesizes through ``api.msedgeservices.com``, which still
accepts the legacy token given as ``Ocp-Apim-Subscription-Key`` with the
legacy ``Sec-MS-GEC`` / ``Sec-MS-GEC-Version``.

Upstream fix (unmerged PR rany2/edge-tts#491) changes ``WSS_URL`` in
``edge_tts.constants`` from::

    wss://speech.platform.bing.com/consumer/speech/synthesize/readaloud/edge/v1?TrustedClientToken=...

to::

    wss://api.msedgeservices.com/tts/cognitiveservices/websocket/v1?Ocp-Apim-Subscription-Key=...

The voice list endpoint still works on the legacy host, so ``VOICE_LIST``
is left unchanged.

Because ``edge-tts==7.2.8`` is pinned and the upstream release with the
fix is not out yet, we monkey-patch both ``edge_tts.constants.WSS_URL``
and ``edge_tts.communicate.WSS_URL`` (``communicate`` does
``from .constants import WSS_URL``, so patching ``constants`` alone is
not enough) plus the locale-independent ``Date`` header parsing from the
same PR.
"""

from datetime import timezone as _tz
from email.utils import parsedate_to_datetime as _parsedate_to_datetime
from typing import Optional

_APPLIED = False

NEW_WSS_HOST = "api.msedgeservices.com/tts/cognitiveservices/websocket/v1"


def apply_edge_tts_fix() -> str:
    """Patch the installed ``edge_tts`` package in place. Idempotent.

    Returns:
        The new ``WSS_URL`` that was applied.
    """
    global _APPLIED

    import edge_tts.communicate as _communicate
    import edge_tts.constants as _constants

    new_wss_url = (
        f"wss://{NEW_WSS_HOST}"
        f"?Ocp-Apim-Subscription-Key={_constants.TRUSTED_CLIENT_TOKEN}"
    )
    _constants.WSS_URL = new_wss_url
    _communicate.WSS_URL = new_wss_url

    # Upstream PR #491 (1/2): parse the server's Date header in any locale.
    # strptime("%a, ... %b ...") follows the process locale, so e.g. pt_BR /
    # fr_FR / de_DE fail to parse "Mon, 12 Oct 2026 ..." and a 403 ends in
    # SkewAdjustmentError instead of the clock-skew retry.
    try:
        from edge_tts.drm import DRM as _DRM

        def _parse_rfc2616_date(date: str) -> Optional[float]:
            try:
                parsed = _parsedate_to_datetime(date)
            except (TypeError, ValueError, IndexError):
                return None
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=_tz.utc)
            return parsed.timestamp()

        _parse_rfc2616_date.__doc__ = _DRM.parse_rfc2616_date.__doc__
        _DRM.parse_rfc2616_date = staticmethod(_parse_rfc2616_date)
    except Exception:
        # DRM fix is best-effort; the endpoint fix above is the critical one.
        pass

    _APPLIED = True
    return new_wss_url


# Apply on import so every ``import edge_tts_patch`` fixes the library.
try:
    apply_edge_tts_fix()
except ImportError:
    # edge-tts not installed (e.g. docs / lint envs); callers handle it.
    pass
