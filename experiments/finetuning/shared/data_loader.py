"""
experiments/finetuning/shared/data_loader.py
==================================
Shared base data loader kept for backward compatibility with existing
LLaMA / Qwen / Mistral data_loader modules that import CurriculumDataLoader.

For the current flat-training pipeline, most logic lives in each model's
own ``data_loader.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from datasets import Dataset
from transformers import AutoTokenizer

from experiments.finetuning.shared.utils import setup_logger, read_jsonl

DEFAULT_STAGE_PATHS = {
    "stage1_explicit": "data/training/stage1_explicit.jsonl",
    "stage2_implicit": "data/training/stage2_implicit.jsonl",
    "stage3_complex":  "data/training/stage3_complex.jsonl",
}


class CurriculumDataLoader:
    """Base curriculum data loader (backward-compat stub).

    The flat-training pipeline does **not** use this class; it exists so
    that ``from experiments.finetuning.shared.data_loader import CurriculumDataLoader``
    does not break in files that still reference it.
    """

    def __init__(
        self,
        model_key: str = "llama",
        data_root: str | Path = ".",
        include_system_prompt: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        self.model_key = model_key
        self.data_root = Path(data_root)
        self.include_system_prompt = include_system_prompt
        self.log = logger or setup_logger(f"{model_key}.data_loader")

    # kept for API compat — callers should migrate to flat loading
    def load_stage(
        self,
        stage_name: str,
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        eval_ratio: float = 0.02,
        seed: int = 42,
        stage_path: Optional[str | Path] = None,
    ) -> tuple[Dataset, Optional[Dataset]]:
        path = Path(stage_path) if stage_path else (self.data_root / DEFAULT_STAGE_PATHS.get(stage_name, ""))
        if not path.exists():
            raise FileNotFoundError(f"Stage data not found: {path}")

        records = read_jsonl(path)
        self.log.info("Loaded %d records from %s", len(records), path)

        from experiments.finetuning.LLaMA.data_loader import normalize_and_tokenize
        dataset = normalize_and_tokenize(records, tokenizer, max_length, self.log)

        if eval_ratio > 0 and len(dataset) > 10:
            split = dataset.train_test_split(test_size=eval_ratio, seed=seed)
            return split["train"], split["test"]
        return dataset, None

    def load_all_stages(
        self,
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        eval_ratio: float = 0.02,
        seed: int = 42,
    ) -> dict:
        result = {}
        for stage_name in DEFAULT_STAGE_PATHS:
            try:
                result[stage_name] = self.load_stage(
                    stage_name, tokenizer, max_length, eval_ratio, seed,
                )
            except FileNotFoundError:
                self.log.warning("Stage data not found: %s", stage_name)
        return result
