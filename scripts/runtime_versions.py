"""Print the serving runtime's package versions as serve-manifest lines
(vllm, torch, transformers, compressed_tensors), through the same collector
the freeze uses for --runtime auto (qflip.versions)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qflip.versions import main  # noqa: E402

if __name__ == "__main__":
    main()
