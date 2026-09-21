"""Send packet files to the GLM API and save the raw replies.

Optional convenience for the chat workflow: identical prompts, so a packet can
be run here OR pasted into the GLM chat UI, and ingest_glm_batch.py does not
care which. Rate limits on the supplied account make the chat route the
practical one for volume; this path is for pilots.

    export GLM_API_KEY='...'
    venv/bin/python scripts/run_glm_packets.py data/glm_pilot_packets/*.md
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("packets", nargs="+")
    ap.add_argument("--out", default="data/glm_raw")
    ap.add_argument("--model", default="glm-4.5-flash")
    ap.add_argument("--min-gap", type=float, default=12.0,
                    help="seconds between requests; the free tier trips code "
                         "1302 well below one request per second")
    ap.add_argument("--max-tokens", type=int, default=8192)
    args = ap.parse_args()

    out = PROJECT_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    cli = GLMClient(model=args.model, min_gap=args.min_gap)
    done = fail = 0
    t0 = time.time()
    for p in args.packets:
        pp = Path(p)
        dest = out / (pp.stem + ".txt")
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  skip {pp.name} (already have a reply)")
            continue
        print(f"  -> {pp.name} ...", flush=True)
        try:
            reply = cli.chat(pp.read_text(encoding="utf-8"),
                             max_tokens=args.max_tokens)
        except NoBalance as e:
            print(f"\nSTOPPING: {e}\nTop up the account, or use the chat UI.")
            return 2
        except RateLimit as e:
            print(f"     gave up on this packet: {e}")
            fail += 1
            continue
        dest.write_text(reply, encoding="utf-8")
        done += 1
        print(f"     saved {len(reply)} chars -> {dest}")
    dt = time.time() - t0
    print(f"\n{done} replies, {fail} failed, {dt/60:.1f} min, "
          f"{cli.calls} calls, {cli.retries} retries, "
          f"{cli.tokens_out} completion tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
