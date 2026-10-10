"""Vendored Demucs v4 inference stack (torch-only, no external demucs install).

Some checkpoints were pickled with ``demucs.*`` module paths, so this
package aliases itself as ``demucs`` for unpickling compatibility.
"""

import sys as _sys

_sys.modules.setdefault("demucs", _sys.modules[__name__])
