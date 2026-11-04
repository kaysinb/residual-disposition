"""Build the follow-up's stack-branch validation suites on CPU (preregistration §13.7, barrier B4):
V4 for PD and ADL and O4 for the observer, by §12.4.10's procedure, saved create-only under
``data/strong_suites/<kind>-<name>/`` with their manifests; prints their hashes for
``configs/continuation/follow_b4.yaml``.

    uv run python scripts/make_follow_suites.py --seed SEED [--only pd|adl|observer]

- **Populations:** §12.4.10's (``make_strong_suites.py``), each rebuilt and checked against the
  population sha256 the frozen suites record.
- **Exclusions:** PD the frozen pool's prefills and V1–V3's; ADL the screening fragments'
  documents and V1–V3's; each frozen suite loaded only at its hash (``strong.yaml``).
- **RNG with new keys:** ``random.Random(derive_seed(seed, *keys)).shuffle``, PD ``"follow_pd"``;
  ADL the stratum by ``"follow_adl_stratum"``, then ``"follow_adl", <length>`` per stratum.
- **O4:** the pool's slice after O1–O3 ([9600:10200]) in its one seeded shuffle; O1–O3 rebuilt
  and checked at their hashes; O4 checked disjoint from every saved set of the pool (the
  exploratory, implant and other training sets), the confirmation set and O1–O3.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_strong_suites as M  # noqa: E402

KEYS = {"pd": "follow_pd", "adl": ("follow_adl_stratum", "follow_adl")}
EARLIER = ("V1", "V2", "V3")
OBSERVER_SLICE = (9600, 10200)  # §13.7


def frozen_suites(root: Path, kind: str, names) -> list:
    from residual_disposition import continuation_strong as CS

    return [CS.load_suite(root, kind, n) for n in names]


def main(argv: list[str] | None = None) -> int:
    from residual_disposition import adl
    from residual_disposition import strong_suites as S
    from residual_disposition.paths import data_root

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, required=True, help="the suites' base seed")
    parser.add_argument("--only", choices=("pd", "adl", "observer"), default=None)
    args = parser.parse_args(argv)
    root = data_root()
    tok, model = M.tokenizer()
    printed = {}
    if args.only in (None, "pd"):
        earlier = frozen_suites(root, "pd", EARLIER)
        frozen = M.frozen_prefills(tok)
        population = S.pd_population(tok, M.c4_validation_texts(), exclude=frozen)
        recorded = earlier[0].provenance["population_sha256"]
        if S._sha256([list(p) for p in population]) != recorded:
            raise SystemExit("the PD population is not the frozen suites' population")
        taken = {tuple(r) for s in earlier for r in s.records.values()}
        pop = [p for p in population if p not in taken]
        suites = S.draw_pd_suites(pop, args.seed, names=("V4",), key=KEYS["pd"])
        S.check_disjoint_suites([*earlier, *suites.values()], frozen=frozen)
        extra = {"source": M.C4_VALIDATION, "tokenizer": model,
                 "excluded": "the frozen PD pool's 9,194 prefills and V1–V3's 27,582",
                 "base_population": len(population),
                 "base_population_sha256": recorded}  # fmt: skip
        for s in suites.values():
            printed[f"pd-{s.name}"] = M.save(root / "strong_suites", s, extra)
    if args.only in (None, "adl"):
        from residual_disposition.adl import corpus_texts

        earlier = frozen_suites(root, "adl", EARLIER)
        screened, spec = M.screening_documents(tok, adl.SEQUENCE_TOKENS, adl.SEQUENCES)
        texts = corpus_texts({**spec, "files": M.FINEWEB_UNREAD})
        population = [(d, t) for d, t in S.adl_population(tok, texts) if d not in screened]
        recorded = earlier[0].provenance["population_sha256"]
        if S._sha256([d for d, _ in population]) != recorded:
            raise SystemExit("the ADL population is not the frozen suites' population")
        taken = {r["document"] for s in earlier for r in s.records.values()}
        pop = [(d, t) for d, t in population if d not in taken]
        suites = S.draw_adl_suites(pop, args.seed, names=("V4",), keys=KEYS["adl"])
        S.check_disjoint_suites([*earlier, *suites.values()],
                                frozen=[{"document": d} for d in screened])  # fmt: skip
        extra = {"source": {**spec, "files": M.FINEWEB_UNREAD}, "tokenizer": model,
                 "excluded": "the screening fragments' documents and V1–V3's",
                 "base_population": len(population),
                 "base_population_sha256": recorded}  # fmt: skip
        for s in suites.values():
            printed[f"adl-{s.name}"] = M.save(root / "strong_suites", s, extra)
    if args.only in (None, "observer"):
        from omegaconf import OmegaConf

        from residual_disposition import continuation_confirm as C
        from residual_disposition.config import load_config
        from residual_disposition.continuation_confirm import FROZEN as CONF
        from residual_disposition.prompts import (
            Prompt,
            build_pool,
            check_disjoint,
            download_alpaca,
            set_sha256,
            split_pool,
        )

        spec = OmegaConf.to_container(load_config("prompts/default").prompts, resolve=True)
        pool = build_pool(download_alpaca(spec["source"]), min_chars=spec["min_chars"],
                          max_chars=spec["max_chars"])  # fmt: skip
        saved = json.loads((root / "prompts" / CONF["source_pool"] / "manifest.json").read_text())
        sets = split_pool(pool, spec["sets"], spec["seed"])
        for name, prompts in sets.items():
            if set_sha256(prompts) != saved["sets"][name]["sha256"]:
                raise SystemExit(f"{name}: the pool does not give the saved set")
        sizes = {**spec["sets"], "confirmation": CONF["prompts"]}
        earlier = S.observer_suites(pool, sizes, spec["seed"])
        for s, frozen in zip(earlier.values(), frozen_suites(root, "observer", ("O1", "O2", "O3")),
                             strict=True):  # fmt: skip
            if s.sha256 != frozen.sha256:
                raise SystemExit(f"{s.name}: the pool does not give the frozen suite")
        sizes.update({name: len(s.records) for name, s in earlier.items()})
        suites = S.observer_suites(pool, sizes, spec["seed"], names=("O4",))
        o4 = suites["O4"]
        if tuple(o4.provenance["slice"]) != OBSERVER_SLICE:
            raise SystemExit(f"O4 is the slice {o4.provenance['slice']}, not {OBSERVER_SLICE}")
        confirmation = {"confirmation": C.prompts(root)}
        check_disjoint({**sets, **confirmation,
                        **{n: [Prompt(k, t) for k, t in s.records.items()]
                           for n, s in {**earlier, **suites}.items()}})  # fmt: skip
        extra = {"source": spec["source"], "pool_id": CONF["source_pool"],
                 "disjoint_from": [*sets, *confirmation, *earlier]}  # fmt: skip
        printed["observer-O4"] = M.save(root / "strong_suites", o4, extra)
    print(json.dumps({k: {"sha256": v["sha256"], "inventory_sha256": v["inventory_sha256"],
                          "records": v["records"],
                          "population": v.get("population", v.get("pool"))}
                      for k, v in printed.items()}, indent=1))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
