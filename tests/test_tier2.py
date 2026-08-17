"""Tests for the tier-2 squad-encoder experiment."""

import numpy as np
import pytest
import torch

from src.data.sportmonks_parse import MATCHES_PARQUET
from src.models.net import NetConfig, SquadEncoder

needs_data = pytest.mark.skipif(not MATCHES_PARQUET.exists(),
                                reason="SportMonks not parsed yet")


def test_squad_encoder_is_permutation_invariant():
    """The whole requirement. A squad is a SET -- the order players appear in
    a lineup feed carries no information about the team, so an encoder that
    could read the order would learn the feed's conventions instead."""
    enc = SquadEncoder(30, 16).eval()
    sq = torch.randn(4, 2, 11, 30)
    mk = torch.ones(4, 2, 11, dtype=torch.bool)
    a = enc(sq, mk)
    for seed in range(3):
        perm = torch.randperm(11, generator=torch.Generator().manual_seed(seed))
        assert torch.allclose(a, enc(sq[:, :, perm, :], mk), atol=1e-5)


def test_squad_encoder_ignores_masked_slots_entirely():
    """A masked slot must not influence the output whatever is stored there --
    otherwise an absent player becomes a statement about the squad."""
    enc = SquadEncoder(30, 16).eval()
    sq = torch.randn(3, 2, 11, 30)
    mk = torch.ones(3, 2, 11, dtype=torch.bool)
    mk[:, :, 8:] = False
    poisoned = sq.clone()
    poisoned[:, :, 8:] = 999.0
    assert torch.allclose(enc(sq, mk), enc(poisoned, mk), atol=1e-4)


def test_squad_encoder_max_pool_is_not_fooled_by_empty_slots():
    """Masked slots are filled with -inf before the max, not zero. A zero fill
    would let an absent player beat any genuinely negative feature."""
    enc = SquadEncoder(4, 8).eval()
    sq = torch.full((1, 2, 11, 4), -5.0)
    mk = torch.zeros(1, 2, 11, dtype=torch.bool)
    mk[:, :, 0] = True
    out = enc(sq, mk)
    assert torch.isfinite(out).all(), "-inf must not leak into the output"


def test_squad_encoder_output_dimension():
    enc = SquadEncoder(30, 24)
    assert enc.out_dim == 4 * 24          # (mean + max) x (home + away)
    assert enc(torch.randn(5, 2, 11, 30),
               torch.ones(5, 2, 11, dtype=torch.bool)).shape == (5, 96)


def test_squad_encoder_responds_to_squad_quality():
    """Sanity floor: a squad of better players must produce a different vector
    from a squad of worse ones."""
    enc = SquadEncoder(6, 12).eval()
    mk = torch.ones(2, 2, 11, dtype=torch.bool)
    weak = torch.zeros(2, 2, 11, 6)
    strong = torch.ones(2, 2, 11, 6) * 3.0
    assert not torch.allclose(enc(weak, mk), enc(strong, mk), atol=1e-3)


def test_config_requires_a_squad_tensor_when_the_encoder_is_on():
    from src.models.net import build_vocab, train_net
    import pandas as pd
    df = pd.DataFrame({
        "home_key": ["a"] * 40, "away_key": ["b"] * 40,
        "fthg": 1, "ftag": 0, "result": "H", "country": "X", "div": "E0",
        "season": "2020-21",
        "kickoff": pd.date_range("2020-08-01", periods=40, freq="7D"),
    })
    with pytest.raises(ValueError, match="squad_hidden"):
        train_net(df, np.zeros((40, 2)), build_vocab(df),
                  NetConfig(seq_hidden=0, squad_hidden=8, max_epochs=1, patience=1))


@needs_data
def test_tier2_panel_aligns_squads_with_matches():
    from src.tier2 import build_tier2_panel
    panel, squads, mask = build_tier2_panel()
    assert len(panel) == len(squads) == len(mask)
    assert squads.shape[1:3] == (2, 11)
    assert panel["kickoff"].is_monotonic_increasing
    assert mask.mean() > 0.90
    # Every feature the net needs must have survived the join.
    from src.models.baselines import ALL_FEATURES
    assert not [c for c in ALL_FEATURES if c not in panel.columns]
