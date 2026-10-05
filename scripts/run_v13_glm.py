"""Run the v13 GLM packets through the API on GLM-5.2 (docs/v13_plan.md §3).

Same never-stored-key policy as glm_client.py: export GLM_API_KEY in this
shell. Default model is glm-5.2 per the v13 plan; --model overrides (e.g.
the free glm-4.5-flash if the account rate-limits, or the GLM-5.2 chat UI
route: paste data/glm_packets_v13/MASTER_PROMPT_v13.md + ORDERS_v13.txt and
save replies to data/glm_raw_v13/ -- ingest does not care which route a
reply came from).

    export GLM_API_KEY='...'
    venv/bin/python scripts/run_v13_glm.py
    venv/bin/python scripts/run_v13_glm.py --only data/glm_packets_v13/001_relation.md
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from glm_client import GLMClient, NoBalance, RateLimit  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packets", default="data/glm_packets_v13",
                    help="dir with packet .md files (or pass files via --only)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these packet files")
    ap.add_argument("--out", default="data/glm_raw_v13")
    ap.add_argument("--model", default="glm-5.2",
                    help="v13 default; falls back manually with e.g. "
                         "--model glm-4.5-flash")
    ap.add_argument("--min-gap", type=float, default=6.0)
    ap.add_argument("--max-tokens", type=int, default=8192)
    args = ap.parse_args()

    src = PROJECT_ROOT / args.packets
    packets = ([Path(p) for p in args.only] if args.only
               else sorted(src.glob("*.md")))
    packets = [p for p in packets if not p.name.startswith("MASTER")]
    if not packets:
        print(f"no packets under {src} -- run "
              f"scripts/make_v13_glm_orders.py --packets first")
        return 1
    out = PROJECT_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    todo = [p for p in packets
            if not (out / (p.stem + ".txt")).exists()
            or (out / (p.stem + ".txt")).stat().st_size == 0]
    print(f"{len(packets)} packets, {len(todo)} to run on {args.model}")
    cli = GLMClient(model=args.model, min_gap=args.min_gap)
    done = fail = 0
    t0 = time.time()
    for p in todo:
        print(f"  -> {p.name} ...", flush=True)
        try:
            reply = cli.chat(p.read_text(encoding="utf-8"),
                             max_tokens=args.max_tokens)
        except NoBalance as e:
            print(f"\nSTOPPING: {e}\nTop up, or use the chat-UI route "
                  f"(MASTER_PROMPT_v13.md + ORDERS_v13.txt).")
            return 2
        except RateLimit as e:
            print(f"     gave up on this packet: {e}")
            fail += 1
            continue
        (out / (p.stem + ".txt")).write_text(reply, encoding="utf-8")
        done += 1
        print(f"     saved {len(reply)} chars", flush=True)
    print(f"\n{done} replies, {fail} failed, {(time.time() - t0) / 60:.1f} min, "
          f"{cli.calls} calls, {cli.retries} retries, "
          f"{cli.tokens_out} completion tokens")
    return 0 if fail == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
