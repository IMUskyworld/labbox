"""Dev entry point: run the CLI from a source checkout without an editable install.

Why this exists is a real Windows finding, recorded in DEVELOPMENT.md §16: an editable
install writes ``_editable_impl_labbox.pth`` containing the absolute source path, and
``site.py`` decodes ``.pth`` files with the *locale* encoding. On a cp936 machine whose
project path contains non-ASCII characters (``C:\\code\\github小项目``) that path comes out
mojibake and ``import labbox`` fails, so ``labbox.exe`` cannot start.

The workaround for users is to install LabBox under an ASCII path (pipx/uv tool install);
for the test suite this shim injects ``src`` into ``sys.path`` in-process, which keeps the
child training processes' environment clean.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from labbox.cli.main import main

if __name__ == "__main__":
    main()
