"""The one collector of the serving runtime's package versions, used by
scripts/01_serve.sh (serve manifest lines) and by the freeze (`--runtime
auto`), so both report the same string: the module's `__version__`, local
build tag included (torch reports 2.14.0+cu130 where the distribution
metadata says 2.14.0)."""

import importlib

# serve-manifest line -> (import name, distribution name)
SERVING_MODULES = {
    "vllm": ("vllm", "vllm"),
    "torch": ("torch", "torch"),
    "transformers": ("transformers", "transformers"),
    "compressed_tensors": ("compressed_tensors", "compressed-tensors"),
}


def module_version(import_name: str) -> str | None:
    try:
        mod = importlib.import_module(import_name)
    except ImportError:
        return None
    v = getattr(mod, "__version__", None)
    return str(v) if v else None


def collect() -> dict[str, str | None]:
    """{manifest line name: module __version__ or None when not importable}."""
    return {key: module_version(imp) for key, (imp, _dist) in SERVING_MODULES.items()}


def manifest_lines(versions: dict[str, str | None]) -> list[str]:
    return [f"{key}: {versions.get(key) or 'unknown'}" for key in SERVING_MODULES]


def main() -> None:
    print("\n".join(manifest_lines(collect())))


if __name__ == "__main__":
    main()
