"""Run provenance: a metadata sidecar next to every result file, and the
checks that keep one result file to one population and one server.

Compatibility fields decide whether a run may resume into an existing file;
history fields are recorded but never compared (a server restart changes the
serve manifest's timestamp and must not block a legitimate resume).

The core configuration hash covers only the scientific sections of the config
(CORE_SECTIONS); judge settings are frozen separately and must not break
resume compatibility. The full-file hash is kept as history.
"""

import hashlib
import json
import os
import socket
import struct
import subprocess
import urllib.request
from urllib.parse import urlsplit
from datetime import datetime, timezone
from pathlib import Path

import yaml

CORE_SECTIONS = ("server", "models", "data", "prompt", "images", "generation", "serving", "quantize", "analysis")

COMPAT_FIELDS = (
    "split",
    "split_file_sha256",
    "exclusions_sha256",
    "annotations_sha256",
    "core_config_sha256",
    "prompt_sha256",
    "precision",
    "condition",
    "served_model_name",
    "model_path_or_id",
    "model_revision",
    "checkpoint_fingerprint",
    "quantization",
    "dtype",
    "max_pixels",
    "max_model_len",
    "kv_args",
    "seed",
    "max_num_seqs",
    "prefix_caching",
    "vllm_version",
)
# Fields that legitimately differ between the two precisions of one comparison.
PRECISION_FIELDS = ("precision", "served_model_name", "model_path_or_id", "model_revision", "checkpoint_fingerprint", "quantization", "dtype")
HISTORY_FIELDS = ("created", "config_sha256", "code_revision", "code_dirty", "host", "serve_manifest", "serve_date", "server_pid", "server_port")
# Serving settings every manifest must record (scripts/01_serve.sh writes them).
SERVING_MANIFEST_KEYS = ("max_pixels", "max_model_len", "kv_args", "seed", "max_num_seqs", "prefix_caching")


class ProvenanceError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def core_config(cfg: dict) -> dict:
    return {k: cfg[k] for k in CORE_SECTIONS if k in cfg}


def core_config_sha256(cfg: dict) -> str:
    """Canonical hash of the core sections only (sorted keys, no whitespace),
    so a judge-only edit of the YAML leaves it unchanged."""
    return sha256_text(json.dumps(core_config(cfg), sort_keys=True, separators=(",", ":"), default=str))


def git_revision(root: str | Path = ".") -> tuple[str, bool]:
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True).stdout.strip())
        return rev, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", True


def parse_serve_manifest(path: str | Path) -> dict:
    """`key: value` lines written by scripts/01_serve.sh."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def checkpoint_fingerprint(model_path_or_id: str, revision: str | None = None) -> str:
    """Local checkpoint: sha256 over the sha256s of config.json and every
    *.safetensors file (sorted), so an overwritten directory changes it.
    Hub model: hf:<id>@<revision>."""
    p = Path(model_path_or_id)
    if p.is_dir():
        files = sorted([p / "config.json"] + list(p.glob("*.safetensors")))
        if not (p / "config.json").exists():
            raise ProvenanceError(f"{p} has no config.json")
        h = hashlib.sha256()
        for f in files:
            h.update(f.name.encode())
            h.update(sha256_file(f).encode())
        return "sha256:" + h.hexdigest()
    return f"hf:{model_path_or_id}@{revision or 'unknown'}"


def quant_and_dtype(extra_args: str) -> tuple[str, str]:
    toks = extra_args.split()
    quant = toks[toks.index("--quantization") + 1] if "--quantization" in toks else "none"
    dtype = toks[toks.index("--dtype") + 1] if "--dtype" in toks else "auto"
    return quant, dtype


def build_run_meta(
    *,
    split: str,
    precision: str,
    condition: str,
    config_path: str | Path,
    annotations_path: str | Path,
    prompt_path: str | Path,
    split_file: str | Path,
    exclusions_file: str | Path | None,
    serve_manifest_path: str | Path,
    served_model_name: str,
    repo_root: str | Path = ".",
) -> dict:
    m = parse_serve_manifest(serve_manifest_path)
    quant, dtype = quant_and_dtype(m.get("extra_args", ""))
    model = m.get("model", "")
    revision = m.get("revision") or None
    rev, dirty = git_revision(repo_root)
    excl = Path(exclusions_file) if exclusions_file else None
    cfg = yaml.safe_load(Path(config_path).read_text())
    return {
        "split": split,
        "split_file_sha256": sha256_file(split_file),
        "exclusions_sha256": sha256_file(excl) if excl and excl.exists() else None,
        "annotations_sha256": sha256_file(annotations_path),
        "core_config_sha256": core_config_sha256(cfg),
        "prompt_sha256": sha256_file(prompt_path),
        "precision": precision,
        "condition": condition,
        "served_model_name": served_model_name,
        "model_path_or_id": model,
        "model_revision": revision,
        "checkpoint_fingerprint": checkpoint_fingerprint(model, revision) if model else None,
        "quantization": quant,
        "dtype": dtype,
        "max_pixels": m.get("max_pixels"),
        "max_model_len": m.get("max_model_len"),
        "kv_args": m.get("kv_args"),
        "seed": m.get("seed"),
        "max_num_seqs": m.get("max_num_seqs"),
        "prefix_caching": m.get("prefix_caching"),
        "vllm_version": m.get("vllm"),
        # history
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_sha256": sha256_file(config_path),
        "code_revision": rev,
        "code_dirty": dirty,
        "host": m.get("gpu"),
        "serve_date": m.get("date"),
        "server_pid": m.get("pid"),
        "server_port": m.get("port"),
        "serve_manifest": m,
    }


def meta_path(result_file: str | Path) -> Path:
    p = Path(result_file)
    return p.with_name(p.name + ".meta.json")


def load_meta(result_file: str | Path) -> dict | None:
    mp = meta_path(result_file)
    return json.loads(mp.read_text()) if mp.exists() else None


def compat_diff(a: dict, b: dict, ignore: tuple[str, ...] = ()) -> list[str]:
    return [f for f in COMPAT_FIELDS if f not in ignore and a.get(f) != b.get(f)]


def check_records_against_meta(result_file: str | Path, meta: dict, allowed_scenes: set[str] | None) -> int:
    """Every raw row of the file (retries included, before deduplication) must
    carry the sidecar's precision and condition and, when the population is
    bounded, a scene inside it. Shared by the resume path and the analysis
    path. Returns the raw row count."""
    from qflip.records import iter_raw_records

    n = 0
    for r in iter_raw_records(result_file):
        n += 1
        for k in ("precision", "condition"):
            if r.get(k) != meta[k]:
                raise ProvenanceError(f"{result_file}: row {n} ({r.get('question_id')!r}) has {k}={r.get(k)!r}, sidecar says {meta[k]!r}")
        if allowed_scenes is not None and r.get("scene_token") not in allowed_scenes:
            raise ProvenanceError(f"{result_file}: row {n} ({r.get('question_id')!r}) is from scene {r.get('scene_token')!r}, outside the {meta['split']} population")
    return n


def ensure_meta(result_file: str | Path, meta: dict, allowed_scenes: set[str] | None = None) -> None:
    """Write the sidecar for a new result file; for an existing file, refuse
    unless the sidecar exists, every compatibility field matches, and the
    records already present belong to the declared population."""
    rf, mp = Path(result_file), meta_path(result_file)
    if rf.exists() and rf.stat().st_size > 0:
        if not mp.exists():
            raise ProvenanceError(f"{rf} exists without {mp.name}; refusing to append to a file of unknown provenance")
        old = json.loads(mp.read_text())
        diff = compat_diff(old, meta)
        if diff:
            raise ProvenanceError(f"{rf} was produced under different settings; differing fields: {diff}. Use another --out-dir.")
        check_records_against_meta(rf, meta, allowed_scenes)
        return
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(meta, indent=2))


def _same_model(a: str, b: str) -> bool:
    if a == b:
        return True
    try:
        pa, pb = Path(a), Path(b)
        return pa.exists() and pb.exists() and pa.resolve() == pb.resolve()
    except (OSError, ValueError):  # argv entries can be arbitrary text, not paths
        return False


def check_server(base_url: str, served_model_name: str, expect_model: str | None = None, timeout_s: float = 10) -> list[dict]:
    """The server at base_url must list served_model_name under /v1/models.
    With expect_model, the model card's `root` (vLLM reports the model path or
    id it loaded) must name that checkpoint: the official and the own INT4
    checkpoints share the served alias, so the alias alone is not identity."""
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as r:  # noqa: S310 - local server
            cards = json.loads(r.read()).get("data", [])
    except Exception as e:  # noqa: BLE001
        raise ProvenanceError(f"cannot query {url}: {e!r}") from e
    ids = [m.get("id") for m in cards]
    if served_model_name not in ids:
        raise ProvenanceError(f"server at {base_url} serves {ids}, expected {served_model_name!r}; wrong server running")
    if expect_model:
        card = next(m for m in cards if m.get("id") == served_model_name)
        root = card.get("root")
        if root and not _same_model(str(root), expect_model):
            raise ProvenanceError(f"server at {base_url} loaded {root!r}, the manifest says {expect_model!r}; the manifest does not describe the running server")
    return cards


def url_port(base_url: str) -> int:
    u = urlsplit(base_url)
    if u.port is not None:
        return u.port
    return 443 if u.scheme == "https" else 80


LOCAL_HOSTS = {"localhost": ("127.0.0.1", "::1"), "127.0.0.1": ("127.0.0.1",), "::1": ("::1",)}


def url_local_addresses(base_url: str) -> tuple[str, ...]:
    """The loopback addresses an endpoint may be bound to. The guard runs on
    the serving host and the client talks to it over loopback, so any other
    hostname is refused rather than resolved."""
    host = (urlsplit(base_url).hostname or "").lower()
    if host in LOCAL_HOSTS:
        return LOCAL_HOSTS[host]
    try:
        ip = socket.inet_pton(socket.AF_INET, host)
        if ip[0] == 127:
            return (host,)
    except OSError:
        pass
    raise ProvenanceError(f"endpoint {base_url} is not a loopback address; the run guard only binds servers on this host (use localhost or 127.0.0.1)")


def _decode_proc_addr(hex_addr: str) -> str:
    """/proc/net/tcp{,6} local_address field (host byte order per 32-bit word) to text."""
    if len(hex_addr) == 8:
        return socket.inet_ntop(socket.AF_INET, struct.pack("<I", int(hex_addr, 16)))
    raw = b"".join(struct.pack("<I", int(hex_addr[i:i + 8], 16)) for i in range(0, 32, 8))
    return socket.inet_ntop(socket.AF_INET6, raw)


def _is_v4(addr: str) -> bool:
    return ":" not in addr


def _addr_matches(bound: str, wanted: tuple[str, ...]) -> bool:
    """A bound address serves a wanted one when equal, or when it is the
    wildcard of that family: 0.0.0.0 serves IPv4 addresses only; :: (an IPv6
    dual-stack wildcard, Linux default) serves both families."""
    if bound == "0.0.0.0":
        return any(_is_v4(w) for w in wanted)
    if bound == "::":
        return bool(wanted)
    if bound.startswith("::ffff:"):
        bound = bound[7:]
    return bound in wanted


def listening_sockets(port: int, addresses: tuple[str, ...], proc_root: str | Path = "/proc") -> dict[str, str]:
    """inode -> bound address for LISTEN sockets on the port whose binding
    serves one of the addresses."""
    out = {}
    for table in ("net/tcp", "net/tcp6"):
        try:
            lines = (Path(proc_root) / table).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 10:
                continue
            local, state, inode = parts[1], parts[3], parts[9]
            addr_hex, port_hex = local.rsplit(":", 1)
            if state == "0A" and int(port_hex, 16) == port:
                bound = _decode_proc_addr(addr_hex)
                if _addr_matches(bound, addresses):
                    out[inode] = bound
    return out


def listener_pids(port: int, addresses: tuple[str, ...], proc_root: str | Path = "/proc") -> tuple[dict[int, str], dict[str, str]]:
    """(pid -> bound address) for every process holding a matching LISTEN
    socket, and (inode -> bound address) for matching sockets whose owner is
    not visible from this process."""
    inodes = listening_sockets(port, addresses, proc_root)
    if not inodes:
        return {}, {}
    targets = {f"socket:[{i}]": i for i in inodes}
    owners: dict[int, str] = {}
    seen: set[str] = set()
    for proc in Path(proc_root).iterdir():
        if not proc.name.isdigit():
            continue
        try:
            for fd in (proc / "fd").iterdir():
                try:
                    link = os.readlink(fd)
                except OSError:
                    continue
                if link in targets:
                    owners[int(proc.name)] = inodes[targets[link]]
                    seen.add(targets[link])
        except OSError:
            continue
    unresolved = {i: bound for i, bound in inodes.items() if i not in seen}
    return owners, unresolved


def listener_pid(port: int, base_url: str = "http://localhost", proc_root: str | Path = "/proc") -> int | None:
    """The single pid serving the endpoint's address and port; None when no
    matching listener exists. Raises when a matching socket's owner cannot be
    seen (ownership unresolved) or when several pids own matching sockets."""
    owners, unresolved = listener_pids(port, url_local_addresses(base_url), proc_root)
    if unresolved:
        raise ProvenanceError(f"port {port} has {len(unresolved)} listener(s) for {base_url} whose owning process is not visible from here ({', '.join(unresolved.values())}); ownership unresolved, refusing")
    if len(owners) > 1:
        raise ProvenanceError(f"port {port} has several listeners for {base_url}: {owners}; ownership is ambiguous")
    return next(iter(owners), None)


def _ancestors(pid: int, depth: int = 6, proc_root: str | Path = "/proc") -> list[int]:
    chain = []
    for _ in range(depth):
        try:
            status = (Path(proc_root) / str(pid) / "status").read_text()
        except OSError:
            break
        ppid = next((int(line.split()[1]) for line in status.splitlines() if line.startswith("PPid:")), 0)
        if ppid <= 1:
            break
        chain.append(ppid)
        pid = ppid
    return chain


def check_server_process(manifest: dict, base_url: str, proc_root: str | Path = "/proc") -> int:
    """Bind the manifest to the server the client talks to: the endpoint must
    be a loopback address; the manifest pid must be alive on this host with a
    command line naming the manifest's model and port; the manifest port must
    be the client's endpoint port exactly; and the process holding the LISTEN
    socket serving that address and port (wildcard bindings included) must be
    that pid or one of its descendants. Unverifiable or ambiguous ownership is
    refused."""
    pid_s, port_s = manifest.get("pid"), manifest.get("port")
    if not pid_s or not port_s:
        raise ProvenanceError("serve manifest has no pid/port lines; restart the server with the current scripts/01_serve.sh")
    pid, port = int(pid_s), int(port_s)
    url_local_addresses(base_url)
    client_port = url_port(base_url)
    if client_port != port:
        raise ProvenanceError(f"client endpoint {base_url} is port {client_port}, the manifest server is on port {port}")
    cmdline = Path(proc_root) / str(pid) / "cmdline"
    if not cmdline.exists():
        raise ProvenanceError(f"serve manifest names pid {pid}, which is not running on this host; the manifest is stale")
    argv = [a.decode(errors="replace") for a in cmdline.read_bytes().split(b"\0") if a]
    model = manifest.get("model", "")
    if not any(_same_model(a, model) for a in argv):
        raise ProvenanceError(f"pid {pid} command line does not name the manifest model {model!r}: {' '.join(argv)[:200]}")
    ports = [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "--port"]
    if ports != [str(port)]:
        raise ProvenanceError(f"pid {pid} command line port {ports} is not the manifest port {port}")
    owner = listener_pid(port, base_url, proc_root)
    if owner is None:
        raise ProvenanceError(f"no process holding a LISTEN socket for {base_url} is visible from here; cannot verify that pid {pid} serves it")
    if owner != pid and pid not in _ancestors(owner, proc_root=proc_root):
        raise ProvenanceError(f"port {port} is served by pid {owner}, which is not the manifest server {pid} or its child")
    return pid


def effective_base_url(cfg: dict) -> str:
    """The endpoint the inference client uses (qflip.client honors QFLIP_BASE_URL)."""
    return os.environ.get("QFLIP_BASE_URL", cfg["server"]["base_url"])
