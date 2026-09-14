# data/ — taxonomy

| Dir | What it is | Status |
|---|---|---|
| `benchmarks/` | TIME (canonical `TIME_Newest.json`, 104,951), TimeBench (21,188), TRAM (980,918) | **eval sets — never train on these** |
| `combined_80_20_split` | v1 training data (original mixture) | frozen |
| `combined_80_20_v2 … v7` | training data per cycle | frozen; v7 in use |
| `training_versions/` | **per-version folders: verified data copy + measured manifest + story.** Originals remain canonical in `combined_80_20_*` (configs + in-flight v7 point there). | start here |
| `corpus/` | CC-News 2023-24 + CNN stories — source corpus for the v8a news generator | v8a fuel |
| `synthetic/` | generated news items (v8a's audited 12,000 + the 300-row sample) | waiting for v8a |
| `rehearsal/` | dolly / HotpotQA / DROP / CoQA pools used by the builders | builder input |
| `prepared_v4/` | large external pools (cotcoll 575k, slimorca, squad) | mostly unused; v8b resource |
| `cot/` | STaR pilot traces + manual seeds | v9 resource |
| `raw/` | original dataset downloads | provenance only |
