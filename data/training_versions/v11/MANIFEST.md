# Training data — v11

Dir: `data/combined_80_20_v11/` (85.6 MB) — train 11,446 / val 2,869 | synthetic 53.2% | general rehearsal 0.0%

Prompt-parity arm: v10-glm's mixture minus 561 contaminated TimeQA rows and 124 relation/ordering rows outside TRAM's gold label space; trained with prompt_format eval_parity (the exact evaluation prompt, date pinned); hyperparameters chosen on a held-out dev split (winner t09: lr 1.77e-4, 1 epoch, LoRA r16). Parity tokenisation drops a further 306 gold-truncated TimeQA rows.

Results: seed 42, test-minus-dev vs zs-vllm-pinned: TIME 46.80 (+5.65; +5.05 no-abstain), TimeBench 46.48 (+1.65), TRAM 53.74 (+7.19 micro / +4.99 macro) — first arm to beat zero-shot on all three. Seeds 43/44 pending. docs/results_and_methodology.md §10.

Full composition in `manifest.json`. A sha256-verified copy of the data is in `data/`; the canonical dir is `data/combined_80_20_v11/` (configs point there — if the copy ever diverges, the original wins).
