"""Organism states: the base model plus an ordered list of LoRA weight deltas.

A state is identified by an opaque id, the hash of its manifest. The manifest records lineage and
adapters but no behavior name; the analyst index (``records.StateRecord``) maps ids to behaviors and
is never read by audit code (invariant 3).

Adapters are stored in PEFT format. Three operations work on their tensors directly:
- the effective delta from the base, Σᵢ sᵢ·BᵢAᵢ with sᵢ = αᵢ/rᵢ, for weight-based tools;
- its squared Frobenius norm in low-rank form, for the erasing penalty and normalizations;
- concatenation into one LoRA of rank Σᵢ rᵢ with unit scaling, for serving a stack with vLLM, which
  applies one adapter per request. It is built by hand: PEFT's ``add_weighted_adapter(..., "cat")``
  copies the first adapter's config (``use_rslora``, ``lora_dropout``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
# The served file: ``concatenate`` (unit scaling, scaled B) saved by ``write_lora`` in fp32 with
# dropout 0. Part of the cache key of served adapters; change it when that output changes.
SERVED_FORMAT = "concat-unit-scaling-fp32-v1"
# vLLM 0.30 accepts these values of max_lora_rank.
VLLM_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)


@dataclass
class Lora:
    """A LoRA adapter's tensors: module prefix → (A [r, in], B [out, r]), with its scaling."""

    rank: int
    alpha: float
    modules: dict[str, tuple[torch.Tensor, torch.Tensor]]
    target_modules: tuple[str, ...]
    base_model: str | None = None

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank


def read_lora(path: Path, *, device: str = "cpu") -> Lora:
    """Load a PEFT LoRA directory; reject features the composition math does not cover."""
    config = json.loads((path / ADAPTER_CONFIG).read_text())
    if config.get("peft_type", "LORA") != "LORA":
        raise ValueError(f"{path}: not a LoRA adapter ({config.get('peft_type')})")
    for key in ("use_rslora", "use_dora"):
        if config.get(key):
            raise ValueError(f"{path}: {key} is not supported")
    if config.get("modules_to_save") or config.get("bias", "none") != "none":
        raise ValueError(f"{path}: modules_to_save and LoRA biases are not supported")
    if config.get("rank_pattern") or config.get("alpha_pattern"):
        raise ValueError(f"{path}: rank or alpha patterns are not supported")
    tensors = load_file(str(path / ADAPTER_WEIGHTS), device=device)
    modules: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, tensor in tensors.items():
        if name.endswith(".lora_A.weight"):
            prefix = name.removesuffix(".lora_A.weight")
            b = tensors[f"{prefix}.lora_B.weight"]
            modules[prefix] = (tensor.float(), b.float())
        elif not name.endswith(".lora_B.weight"):
            raise ValueError(f"{path}: unexpected tensor {name}")
    rank = int(config["r"])
    for prefix, (a, b) in modules.items():
        if a.shape[0] != rank or b.shape[1] != rank:
            raise ValueError(f"{path}: {prefix} has rank {a.shape[0]}, config says {rank}")
    return Lora(
        rank=rank,
        alpha=float(config["lora_alpha"]),
        modules=modules,
        target_modules=tuple(sorted(config["target_modules"])),
        base_model=config.get("base_model_name_or_path"),
    )


def write_lora(
    path: Path, lora: Lora, *, dtype: torch.dtype = torch.float32, dropout: float = 0.0
) -> None:
    """Save in PEFT format with no extras, as vLLM and PEFT both load it.

    ``dropout`` only matters when PEFT trains the adapter further (continued mode); stored and
    served adapters keep 0.
    """
    path.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for prefix, (a, b) in sorted(lora.modules.items()):
        tensors[f"{prefix}.lora_A.weight"] = a.to(dtype).contiguous()
        tensors[f"{prefix}.lora_B.weight"] = b.to(dtype).contiguous()
    save_file(tensors, str(path / ADAPTER_WEIGHTS))
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": lora.base_model,
        "r": lora.rank,
        "lora_alpha": lora.alpha,
        "lora_dropout": float(dropout),
        "target_modules": list(lora.target_modules),
        "bias": "none",
        "use_rslora": False,
        "use_dora": False,
        "modules_to_save": None,
        "fan_in_fan_out": False,
        "init_lora_weights": True,
        "rank_pattern": {},
        "alpha_pattern": {},
    }
    (path / ADAPTER_CONFIG).write_text(json.dumps(config, indent=2) + "\n")


def concatenate(loras: Sequence[Lora]) -> Lora:
    """One LoRA equal to the sum of ``loras``: A stacked, scaled B side by side, unit scaling.

    Modules an adapter does not target get zero blocks, so every module has the full rank.
    """
    if not loras:
        raise ValueError("Nothing to concatenate")
    bases = {lora.base_model for lora in loras} - {None}
    if len(bases) > 1:
        raise ValueError(f"Adapters for different base models: {sorted(bases)}")
    shapes: dict[str, tuple[int, int]] = {}
    for lora in loras:
        for prefix, (a, b) in lora.modules.items():
            shape = (b.shape[0], a.shape[1])
            if shapes.setdefault(prefix, shape) != shape:
                raise ValueError(f"{prefix}: shapes {shapes[prefix]} and {shape} disagree")
    total = sum(lora.rank for lora in loras)
    modules = {}
    for prefix, (out_dim, in_dim) in sorted(shapes.items()):
        a_blocks, b_blocks = [], []
        for lora in loras:
            if prefix in lora.modules:
                a, b = lora.modules[prefix]
                a_blocks.append(a.float())
                b_blocks.append(b.float() * lora.scaling)
            else:
                a_blocks.append(torch.zeros(lora.rank, in_dim))
                b_blocks.append(torch.zeros(out_dim, lora.rank))
        modules[prefix] = (torch.cat(a_blocks, dim=0), torch.cat(b_blocks, dim=1))
    targets = sorted({t for lora in loras for t in lora.target_modules})
    return Lora(
        rank=total,
        alpha=float(total),
        modules=modules,
        target_modules=tuple(targets),
        base_model=next(iter(bases), None),
    )


def serving_rank(rank: int) -> int:
    """The smallest vLLM max_lora_rank that admits ``rank``."""
    for allowed in VLLM_LORA_RANKS:
        if allowed >= rank:
            return allowed
    raise ValueError(f"Total LoRA rank {rank} exceeds vLLM's maximum {VLLM_LORA_RANKS[-1]}")


def _factors(loras: Sequence[Lora], prefix: str) -> tuple[torch.Tensor, torch.Tensor] | None:
    parts = [(lora.modules[prefix], lora.scaling) for lora in loras if prefix in lora.modules]
    if not parts:
        return None
    a = torch.cat([ab[0].float() for ab, _ in parts], dim=0)
    b = torch.cat([ab[1].float() * s for ab, s in parts], dim=1)
    return a, b


def module_prefixes(loras: Iterable[Lora]) -> list[str]:
    return sorted({p for lora in loras for p in lora.modules})


def effective_delta(loras: Sequence[Lora], prefix: str) -> torch.Tensor:
    """Σᵢ sᵢ·BᵢAᵢ for one module, materialized [out, in] in fp32."""
    ab = _factors(loras, prefix)
    if ab is None:
        raise KeyError(prefix)
    a, b = ab
    return b @ a


def edit_norms(state: Sequence[Lora], parent: Sequence[Lora]) -> dict[str, Any]:
    """The Frobenius norm of the effective weight edit from ``parent``'s adapters to ``state``'s,
    Σ_state s·BA − Σ_parent s·BA, per module and over all ("total"), in float64 without
    materializing it (as ``delta_sq_norm``, on the state's factors and the parent's with B
    negated): the added adapter's edit in stack mode, the trained implant's change in continued
    mode. Stage 5's comparison reads these from A and H, not LoRA-factor norms (C15, review s5b
    4)."""
    signed = [(x, 1.0) for x in state] + [(x, -1.0) for x in parent]
    per = {}
    with torch.no_grad():
        for prefix in module_prefixes(x for x, _ in signed):
            parts = [(x.modules[prefix], sign * x.scaling) for x, sign in signed
                     if prefix in x.modules]  # fmt: skip
            a = torch.cat([ab[0].double() for ab, _ in parts], dim=0)
            b = torch.cat([ab[1].double() * s for ab, s in parts], dim=1)
            per[prefix] = float(((b.T @ b) * (a @ a.T)).sum().clamp_min(0.0).sqrt())
    return {"modules": per, "total": float(sum(v * v for v in per.values()) ** 0.5)}


def delta_sq_norm(loras: Sequence[Lora], prefixes: Iterable[str] | None = None) -> torch.Tensor:
    """Σ over modules of ‖Σᵢ sᵢ·BᵢAᵢ‖²_F without materializing the deltas.

    With stacked factors A [R, in] and B [out, R], ‖BA‖²_F = tr(BᵀB·AAᵀ) = Σ (BᵀB ⊙ AAᵀ), so the
    cost is O(R²·(in + out)) per module. Differentiable in the factors.
    """
    total = torch.zeros(())
    for prefix in module_prefixes(loras) if prefixes is None else prefixes:
        ab = _factors(loras, prefix)
        if ab is None:
            continue
        a, b = ab
        term = ((b.T @ b) * (a @ a.T)).sum()
        total = total.to(term.device) + term
    return total


def tensors_sha256(tensors: Mapping[str, torch.Tensor]) -> str:
    """Hash of names, dtypes, shapes and bytes, independent of the file's serialization."""
    digest = hashlib.sha256()
    for name in sorted(tensors):
        t = tensors[name].detach().contiguous().cpu()
        digest.update(f"{name}|{t.dtype}|{tuple(t.shape)}|".encode())
        digest.update(t.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def adapter_sha256(path: Path) -> str:
    return tensors_sha256(load_file(str(path / ADAPTER_WEIGHTS)))


@dataclass(frozen=True)
class AdapterRef:
    """One adapter of a state.

    ``location`` is ``self:<name>`` for an adapter stored with this state (its id cannot appear in
    its own manifest, which it hashes), ``state:<state id>/<name>`` for an adapter of an ancestor,
    or ``hf:<repo>@<revision>`` for a pinned Hub adapter.
    """

    name: str
    location: str
    sha256: str
    rank: int
    alpha: float
    role: str  # implant, suppress, recipe, relearn, ia, ia_control, external
    trained: Mapping[str, Any] = field(default_factory=dict)  # data hash, recipe, seed, steps


@dataclass(frozen=True)
class StateManifest:
    base_model: str
    base_revision: str
    kind: str
    parent_id: str | None
    mode: str | None
    method: str | None
    strength: str | None
    dose: int | None
    seed: int
    adapters: tuple[AdapterRef, ...]
    chat_template_kwargs: Mapping[str, Any]
    provenance: Mapping[str, Any]
    merge: Mapping[str, Any] | None = None  # merge path and tensor hash of the precision check

    def to_json(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["adapters"] = [dataclasses.asdict(a) for a in self.adapters]
        return json.loads(json.dumps(data, sort_keys=True))

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> StateManifest:
        fields = dict(data)
        fields["adapters"] = tuple(AdapterRef(**a) for a in data["adapters"])
        return cls(**fields)

    @property
    def state_id(self) -> str:
        blob = json.dumps(self.to_json(), sort_keys=True, separators=(",", ":")).encode()
        return "s" + hashlib.sha256(blob).hexdigest()[:16]


class StateStore:
    """``<root>/<state id>/manifest.json`` plus the adapters a state introduced."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def adapter_dir(self, state_id: str, name: str) -> Path:
        return self.root / state_id / "adapters" / name

    def save(self, manifest: StateManifest, new_adapters: Mapping[str, Path]) -> str:
        """Store a state; ``new_adapters`` maps its ``self:`` adapters to PEFT dirs to copy in.

        Every adapter of the manifest must exist afterwards with the recorded hash.
        """
        state_id = manifest.state_id
        target = self.root / state_id
        if (target / "manifest.json").exists():
            raise FileExistsError(f"State {state_id} already exists")
        own = {
            ref.location.removeprefix("self:")
            for ref in manifest.adapters
            if ref.location.startswith("self:")
        }
        if own != set(new_adapters):
            raise ValueError(
                f"Adapters to store {sorted(new_adapters)} != self adapters {sorted(own)}"
            )
        for name, src in new_adapters.items():
            dst = self.adapter_dir(state_id, name)
            dst.mkdir(parents=True, exist_ok=True)
            for filename in (ADAPTER_CONFIG, ADAPTER_WEIGHTS):
                shutil.copyfile(src / filename, dst / filename)
        target.mkdir(parents=True, exist_ok=True)
        (target / "manifest.json").write_text(json.dumps(manifest.to_json(), indent=2) + "\n")
        self.verify(state_id)
        return state_id

    def load(self, state_id: str) -> StateManifest:
        manifest = StateManifest.from_json(
            json.loads((self.root / state_id / "manifest.json").read_text())
        )
        if manifest.state_id != state_id:
            raise ValueError(f"Manifest of {state_id} hashes to {manifest.state_id}")
        return manifest

    def resolve(self, ref: AdapterRef, owner_id: str) -> Path:
        """Local PEFT directory of an adapter of state ``owner_id`` (downloading Hub adapters)."""
        kind, _, rest = ref.location.partition(":")
        if kind == "self":
            return self.adapter_dir(owner_id, rest)
        if kind == "state":
            state_id, _, name = rest.partition("/")
            return self.adapter_dir(state_id, name)
        if kind == "hf":
            from huggingface_hub import snapshot_download

            repo, _, revision = rest.partition("@")
            return Path(
                snapshot_download(
                    repo_id=repo,
                    revision=revision,
                    allow_patterns=[ADAPTER_CONFIG, ADAPTER_WEIGHTS],
                )
            )
        raise ValueError(f"Unknown adapter location {ref.location!r}")

    def verify(self, state_id: str) -> None:
        """Every adapter exists and matches its recorded tensor hash."""
        for ref in self.load(state_id).adapters:
            actual = adapter_sha256(self.resolve(ref, state_id))
            if actual != ref.sha256:
                raise ValueError(f"{state_id}: adapter {ref.name} hash {actual} != {ref.sha256}")

    def loras(self, state_id: str) -> list[Lora]:
        return [read_lora(self.resolve(ref, state_id)) for ref in self.load(state_id).adapters]


def served_adapter(
    store: StateStore,
    state_id: str,
    cache_dir: Path,
    readout: Sequence[tuple[AdapterRef, Path]] = (),
) -> tuple[Path, int] | None:
    """A single concatenated adapter for a state, optionally with readout adapters on top.

    Returns its directory and rank, or None for a state without adapters and no readout. The
    directory name hashes everything the file depends on: each component's tensor hash, rank and
    alpha (the tensor hash does not cover the scaling) in order, and the output format, so equal
    stacks reuse one file and nothing else does.
    """
    refs = list(store.load(state_id).adapters) + [ref for ref, _ in readout]
    if not refs:
        return None
    own = [store.resolve(r, state_id) for r in store.load(state_id).adapters]
    paths = own + [p for _, p in readout]
    parts = [SERVED_FORMAT]
    for ref, path in zip(refs, paths, strict=True):
        config = json.loads((path / ADAPTER_CONFIG).read_text())
        rank, alpha = int(config["r"]), float(config["lora_alpha"])
        if (rank, alpha) != (ref.rank, float(ref.alpha)):
            raise ValueError(
                f"{path}: rank {rank}, alpha {alpha} on disk; the reference records "
                f"{ref.rank}, {ref.alpha}"
            )
        parts.append(f"{ref.sha256}:{rank}:{alpha!r}")
    key = hashlib.sha256("|".join(parts).encode()).hexdigest()[:20]
    out = cache_dir / key
    if not (out / ADAPTER_WEIGHTS).exists():
        merged = concatenate([read_lora(p) for p in paths])
        tmp = cache_dir / f".{key}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        write_lora(tmp, merged)
        tmp.replace(out)
    rank = sum(r.rank for r in refs)
    return out, rank
