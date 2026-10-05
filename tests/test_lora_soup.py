"""scripts/lora_soup.py: the concat soup is the exact mean of merged deltas."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lora_soup import check_config, concat_soup  # noqa: E402

KA, KB = "m.lora_A.weight", "m.lora_B.weight"


def _seeds(n=3, r=4, d_in=12, d_out=10):
    g = torch.Generator().manual_seed(0)
    return [{KA: torch.randn(r, d_in, generator=g),
             KB: torch.randn(d_out, r, generator=g)} for _ in range(n)]


def test_concat_equals_mean_of_merged_deltas():
    sds = _seeds()
    soup = concat_soup(sds)
    target = sum(sd[KB] @ sd[KA] for sd in sds) / len(sds)
    assert soup[KA].shape == (12, 12) and soup[KB].shape == (10, 12)
    assert torch.allclose(soup[KB] @ soup[KA], target, atol=1e-5)


def test_naive_ab_mean_is_not_the_soup():
    sds = _seeds()
    a = sum(sd[KA] for sd in sds) / 3
    b = sum(sd[KB] for sd in sds) / 3
    target = sum(sd[KB] @ sd[KA] for sd in sds) / 3
    assert not torch.allclose(b @ a, target, atol=1e-2)


def test_single_adapter_is_identity():
    sd = _seeds(n=1)
    soup = concat_soup(sd)
    assert torch.equal(soup[KA], sd[0][KA]) and torch.equal(soup[KB], sd[0][KB])


@pytest.mark.parametrize("bad", [
    {"use_rslora": True}, {"use_dora": True}, {"bias": "all"},
    {"modules_to_save": ["lm_head"]}, {"rank_pattern": {"q_proj": 8}},
])
def test_refuses_nonlinear_configs(bad):
    assert check_config(dict({"peft_type": "LORA"}, **bad)) is not None


def test_accepts_plain_lora():
    assert check_config({"peft_type": "LORA", "bias": "none"}) is None
