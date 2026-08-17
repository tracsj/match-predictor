"""The network.

A multi-layer net trained by backpropagation on pre-match inputs, with team
and league embeddings and two heads sharing one trunk.

    inputs                     trunk                      heads
    ------                     -----                      -----
    rating features  ----\
    team_home  -> emb ----+--> [Linear -> GELU -> Drop]   1X2 softmax (3)
    team_away  -> emb ----+--> [Linear -> GELU -> Drop]   goals: log lambda x2
    league     -> emb ----/     x k parallel members       -> Poisson grid -> 1X2

**Why embeddings are the point.** A rating compresses a team into one number
that only moves when results move. An embedding is a learned vector, free to
encode whatever actually predicts outcomes -- style, volatility, home/away
asymmetry -- and it is the natural thing a net can do that a gradient-boosted
tree on ratings cannot. It is also the obvious place to overfit, so the
embeddings are weight-decayed hard and kept small (8-16d over ~1,300 teams).

**Why parallel members rather than one deep stack.** TabM (Gorishniy et al.,
ICLR 2025) found that a parameter-efficient implicit ensemble of shallow MLPs
is the strongest general-purpose tabular neural architecture. At ~50k training
rows with roughly 0.1 nats of learnable signal, depth buys overfitting and
ensembling buys stability. Yeung et al.'s transformer beat their LSTM on this
task mainly by having lower variance, not higher accuracy.

**Why two heads.** 1X2 is what gets bet. The goal head predicts a Poisson rate
per side, which yields a scoreline grid, over/under and both-teams-to-score
for free, and an *implied* 1X2 that can be cross-checked against the softmax
head. Multi-task heads over a shared trunk are well demonstrated in-game
(Horton & Lucey 2025) and, per the literature sweep, have never been ablated
pre-match. That ablation is cheap here and worth having.

Everything returns probabilities in (H, D, A) order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.eval.metrics import OUTCOMES

__all__ = ["NetConfig", "MatchNet", "SquadEncoder", "Vocab", "build_vocab",
           "train_net", "predict", "TemperatureScaler", "poisson_to_hda"]


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------

@dataclass
class Vocab:
    """Stable integer ids for teams and leagues, with index 0 reserved.

    Built from the whole corpus rather than per split. Team identity is known
    before kickoff, so this carries no information about the result -- it is
    not the same kind of object as a fitted parameter. Index 0 is <unk> so a
    team absent at fit time still predicts instead of raising, which is the
    failure that makes Dixon-Coles awkward in a walk-forward loop.
    """
    teams: dict[str, int] = field(default_factory=dict)
    leagues: dict[str, int] = field(default_factory=dict)

    @property
    def n_teams(self) -> int:
        return len(self.teams) + 1

    @property
    def n_leagues(self) -> int:
        return len(self.leagues) + 1

    def team_ids(self, country, key) -> np.ndarray:
        return np.array([self.teams.get(f"{c}|{k}", 0) for c, k in zip(country, key)],
                        dtype=np.int64)

    def league_ids(self, div) -> np.ndarray:
        return np.array([self.leagues.get(str(d), 0) for d in div], dtype=np.int64)


def build_vocab(df: pd.DataFrame) -> Vocab:
    keys = pd.unique(pd.concat([
        df["country"].astype(str) + "|" + df["home_key"].astype(str),
        df["country"].astype(str) + "|" + df["away_key"].astype(str),
    ]))
    divs = pd.unique(df["div"].astype(str))
    return Vocab(teams={k: i + 1 for i, k in enumerate(sorted(keys))},
                 leagues={d: i + 1 for i, d in enumerate(sorted(divs))})


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

@dataclass
class NetConfig:
    """Defaults are the configuration the ablation actually supports, not the
    one the architecture section argues for. Measured over 45,629 out-of-sample
    matches, three seeds:

        full (team 12, league 6, 8 members)   RPS 0.20813
        no team embedding                         0.20795   t -2.81 vs full
        no league embedding                       0.20797   t -2.72 vs full
        single trunk member                       0.20799   t -1.92 vs full
        wide, hidden 256                          0.20808   (and 161s vs 60s)

    So the embeddings and the extra capacity both HURT at this scale, and the
    honest default is the small one. Set team_dim/league_dim above zero to
    turn them back on -- worth retrying if the corpus ever grows toward the
    100k-300k range where the literature finds deep models competitive.

    The goals head is the one component that earns its place: removing it
    costs 0.00007 RPS and, obviously, the entire scoreline/over-under output.
    """
    team_dim: int = 0           # measured: embeddings overfit at this scale
    league_dim: int = 0
    hidden: int = 96
    members: int = 1            # 8 members cost 3x the time for nothing
    dropout: float = 0.2
    lr: float = 1e-3            # 3e-3 overfits within one epoch
    weight_decay: float = 1e-4
    emb_weight_decay: float = 1e-3   # embeddings overfit first, so decay harder
    batch_size: int = 1024
    max_epochs: int = 120
    patience: int = 10
    goal_loss_weight: float = 0.3    # 0.0 disables the goals head (for ablation)
    max_goals: int = 10
    seed: int = 0
    # Recurrent branch over each team's last 10 matches. The rolling features
    # are means over the same window and therefore order-blind: a team that
    # lost four then won six looks identical to one that won six then lost
    # four. This asks whether the ORDER carries anything the mean does not.
    #
    # It does, and it is the only component in this model that beats the
    # baseline rather than matching it. Measured over 45,629 out-of-sample
    # matches against an ordered logit at 0.20789:
    #
    #     no sequence branch    RPS 0.20784   t +0.57 vs logit   (a tie)
    #     GRU(32)                   0.20765   t +2.50 vs logit   (a win)
    #                                         t +2.54 vs no-sequence
    #     GRU(64)                   0.20777   t +1.24 vs logit
    #
    # 64 units is worse than 32, which is the same overfitting pattern as the
    # embeddings and the wide trunk. 32 is the measured optimum, not a guess.
    seq_hidden: int = 32
    # Deep-Sets encoder over the starting XI. 0 disables it. Off by default
    # because it only exists for the two SportMonks leagues; the tier-2
    # experiment turns it on.
    squad_hidden: int = 0


class SquadEncoder(nn.Module):
    """Deep Sets over a starting XI: shared per-player MLP, then masked pooling.

    Permutation invariance is the whole requirement. A squad is a SET -- the
    order players appear in a lineup feed carries no information about the
    team, so an encoder that could read the order would learn the feed's
    conventions rather than the football. Sum, mean and max pooling are
    invariant by construction, which is why this is Deep Sets rather than
    anything that flattens the player axis.

    Mean and max are both kept: mean carries "how good is this team on
    average", max carries "does it contain a superstar", and those are
    different questions a squad vector should be able to answer.
    """

    def __init__(self, n_features: int, hidden: int):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(n_features, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.out_dim = 4 * hidden      # (mean + max) x (home + away)

    def forward(self, squads, mask):
        # squads (b, 2, S, F); mask (b, 2, S)
        h = self.phi(squads)                                  # (b, 2, S, H)
        m = mask.unsqueeze(-1).to(h.dtype)
        counts = m.sum(dim=2).clamp(min=1.0)
        pooled_mean = (h * m).sum(dim=2) / counts             # (b, 2, H)
        # -inf on masked slots so an empty slot can never win the max. A zero
        # fill would make "absent player" beat any genuinely negative feature.
        pooled_max = h.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=2).values
        pooled_max = torch.nan_to_num(pooled_max, neginf=0.0)
        return torch.cat([pooled_mean, pooled_max], dim=-1).flatten(1)


class MatchNet(nn.Module):
    def __init__(self, n_cont: int, vocab: Vocab, cfg: NetConfig,
                 n_seq_features: int = 0, n_squad_features: int = 0):
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(cfg.seed)

        # dim 0 disables an embedding entirely, which is how the ablation
        # ("do team embeddings earn their place?") is run.
        self.team_emb = (nn.Embedding(vocab.n_teams, cfg.team_dim, padding_idx=0)
                         if cfg.team_dim > 0 else None)
        self.league_emb = (nn.Embedding(vocab.n_leagues, cfg.league_dim, padding_idx=0)
                           if cfg.league_dim > 0 else None)
        for emb in (self.team_emb, self.league_emb):
            if emb is not None:
                nn.init.normal_(emb.weight, std=0.05)
                with torch.no_grad():
                    emb.weight[0].zero_()

        self.squad_enc = (SquadEncoder(n_squad_features, cfg.squad_hidden)
                          if cfg.squad_hidden > 0 else None)

        d_in = n_cont + 2 * cfg.team_dim + cfg.league_dim
        if cfg.seq_hidden > 0:
            d_in += 2 * cfg.seq_hidden          # one final state per side
        if self.squad_enc is not None:
            d_in += self.squad_enc.out_dim
        k, h = cfg.members, cfg.hidden

        # k parallel two-layer MLPs, expressed as batched weights so the whole
        # ensemble runs in two einsums rather than a python loop.
        self.w1 = nn.Parameter(torch.empty(k, d_in, h))
        self.b1 = nn.Parameter(torch.zeros(k, h))
        self.w2 = nn.Parameter(torch.empty(k, h, h))
        self.b2 = nn.Parameter(torch.zeros(k, h))
        for w in (self.w1, self.w2):
            for i in range(k):
                nn.init.kaiming_uniform_(w[i], a=math.sqrt(5))

        self.drop = nn.Dropout(cfg.dropout)
        self.head_hda = nn.Linear(h, 3)
        self.head_goals = nn.Linear(h, 2)

        # One GRU shared by both sides, so "a run of good form" means the same
        # thing whichever team it belongs to. Two separate encoders would have
        # to learn the concept twice from half the data each.
        self.gru = (nn.GRU(n_seq_features, cfg.seq_hidden, batch_first=True)
                    if cfg.seq_hidden > 0 else None)

    def encode_sequences(self, seq) -> torch.Tensor:
        """(batch, 2, L, F) -> (batch, 2 * seq_hidden), home state then away."""
        b, sides, L, F = seq.shape
        flat = seq.reshape(b * sides, L, F)
        _, h_n = self.gru(flat)
        return h_n[-1].reshape(b, sides * self.cfg.seq_hidden)

    def trunk(self, x_cont, home_id, away_id, league_id, seq=None,
              squads=None, squad_mask=None) -> torch.Tensor:
        parts = [x_cont]
        if self.team_emb is not None:
            parts += [self.team_emb(home_id), self.team_emb(away_id)]
        if self.league_emb is not None:
            parts.append(self.league_emb(league_id))
        if self.gru is not None:
            if seq is None:
                raise ValueError("seq_hidden > 0 but no sequence tensor was passed")
            parts.append(self.encode_sequences(seq))
        if self.squad_enc is not None:
            if squads is None or squad_mask is None:
                raise ValueError("squad_hidden > 0 but no squad tensor was passed")
            parts.append(self.squad_enc(squads, squad_mask))
        x = torch.cat(parts, dim=-1)
        # (batch, d_in) -> (batch, k, h)
        z = torch.einsum("bd,kdh->bkh", x, self.w1) + self.b1
        z = self.drop(F.gelu(z))
        z = torch.einsum("bkh,khg->bkg", z, self.w2) + self.b2
        return self.drop(F.gelu(z))

    def forward(self, x_cont, home_id, away_id, league_id, seq=None,
                squads=None, squad_mask=None):
        z = self.trunk(x_cont, home_id, away_id, league_id, seq, squads, squad_mask)
        logits = self.head_hda(z).mean(dim=1)            # average the ensemble
        # Softplus keeps the rate positive; the offset centres it near a
        # realistic 1.35 goals rather than starting the model at zero.
        log_rates = self.head_goals(z).mean(dim=1)
        rates = F.softplus(log_rates) + 1e-4
        return logits, rates


def poisson_to_hda(rates: np.ndarray, max_goals: int = 10) -> np.ndarray:
    """Turn (lambda_home, lambda_away) into H/D/A by summing a scoreline grid.

    Independent Poisson, which is the classical starting point. Dixon-Coles
    adds a low-score correlation correction; that is a natural extension once
    the head is shown to carry its weight.
    """
    rates = np.asarray(rates, dtype=float)
    k = np.arange(max_goals + 1)
    logfact = np.cumsum(np.concatenate([[0.0], np.log(k[1:])]))

    def pmf(lam):
        lam = lam[:, None]
        return np.exp(k * np.log(lam) - lam - logfact)

    ph, pa = pmf(rates[:, 0]), pmf(rates[:, 1])
    grid = ph[:, :, None] * pa[:, None, :]
    idx = np.arange(max_goals + 1)
    home = grid[:, idx[:, None] > idx[None, :]].sum(axis=1)
    draw = grid[:, idx, idx].sum(axis=1)
    away = grid[:, idx[:, None] < idx[None, :]].sum(axis=1)
    out = np.column_stack([home, draw, away])
    return out / out.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def _tensors(df, X, vocab, device):
    return (
        torch.tensor(X, dtype=torch.float32, device=device),
        torch.tensor(vocab.team_ids(df["country"], df["home_key"]), device=device),
        torch.tensor(vocab.team_ids(df["country"], df["away_key"]), device=device),
        torch.tensor(vocab.league_ids(df["div"]), device=device),
    )


def train_net(
    train_df: pd.DataFrame,
    X_train: np.ndarray,
    vocab: Vocab,
    cfg: NetConfig = NetConfig(),
    val_fraction: float = 0.15,
    device: str = "cpu",
    verbose: bool = False,
    seq_train: np.ndarray | None = None,
    squads_train: np.ndarray | None = None,
    squad_mask_train: np.ndarray | None = None,
    init_from: "MatchNet | None" = None,
) -> tuple[MatchNet, dict]:
    """Hand-written training loop, deliberately readable.

    The validation split is the LAST `val_fraction` of the training window in
    time, never a random slice. Early stopping on a random validation set
    would select the epoch that best predicts the past, which is the same
    leak the walk-forward splitter exists to prevent -- one level down.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    n = len(train_df)
    cut = int(n * (1 - val_fraction))
    tr, va = slice(0, cut), slice(cut, n)

    mu = np.nanmean(X_train[tr], axis=0)
    sd = np.nanstd(X_train[tr], axis=0)
    sd[sd < 1e-9] = 1.0
    Xs = np.nan_to_num((X_train - mu) / sd)

    xc, hid, aid, lid = _tensors(train_df, Xs, vocab, device)
    y = torch.tensor([OUTCOMES.index(v) for v in train_df["result"]], device=device)
    gh = torch.tensor(train_df["fthg"].to_numpy(), dtype=torch.float32, device=device)
    ga = torch.tensor(train_df["ftag"].to_numpy(), dtype=torch.float32, device=device)

    n_seq_f = 0 if seq_train is None else seq_train.shape[-1]
    if cfg.seq_hidden > 0 and seq_train is None:
        raise ValueError("cfg.seq_hidden > 0 requires seq_train")
    seq_t = (torch.tensor(seq_train, dtype=torch.float32, device=device)
             if cfg.seq_hidden > 0 else None)

    n_squad_f = 0 if squads_train is None else squads_train.shape[-1]
    if cfg.squad_hidden > 0 and squads_train is None:
        raise ValueError("cfg.squad_hidden > 0 requires squads_train")
    squad_t = (torch.tensor(squads_train, dtype=torch.float32, device=device)
               if cfg.squad_hidden > 0 else None)
    squad_m = (torch.tensor(squad_mask_train, dtype=torch.bool, device=device)
               if cfg.squad_hidden > 0 else None)

    model = MatchNet(X_train.shape[1], vocab, cfg, n_seq_features=n_seq_f,
                     n_squad_features=n_squad_f).to(device)

    if init_from is not None:
        # Warm-start from a tier-1 pretrained model. Only the parameters whose
        # shapes still match are copied -- the trunk's first layer changes
        # width when a squad encoder is bolted on, so it is reinitialised while
        # everything downstream of it carries over. Both arms of the tier-2 A/B
        # start from the same pretrained weights, or the comparison would be
        # about small-data optimisation rather than about players.
        src = init_from.state_dict()
        dst = model.state_dict()
        carried = {k: v for k, v in src.items()
                   if k in dst and dst[k].shape == v.shape}
        dst.update(carried)
        model.load_state_dict(dst)

    emb_params = [p for emb in (model.team_emb, model.league_emb)
                  if emb is not None for p in emb.parameters()]
    emb_ids = {id(p) for p in emb_params}
    other = [p for p in model.parameters() if id(p) not in emb_ids]
    groups = [{"params": other, "weight_decay": cfg.weight_decay}]
    if emb_params:
        groups.insert(0, {"params": emb_params, "weight_decay": cfg.emb_weight_decay})
    opt = torch.optim.AdamW(groups, lr=cfg.lr)

    def losses(idx):
        sq = None if seq_t is None else seq_t[idx]
        sqd = None if squad_t is None else squad_t[idx]
        sqm = None if squad_m is None else squad_m[idx]
        logits, rates = model(xc[idx], hid[idx], aid[idx], lid[idx], sq, sqd, sqm)
        ce = F.cross_entropy(logits, y[idx])
        if cfg.goal_loss_weight > 0:
            pois = (F.poisson_nll_loss(rates[:, 0], gh[idx], log_input=False, full=False)
                    + F.poisson_nll_loss(rates[:, 1], ga[idx], log_input=False, full=False))
        else:
            pois = torch.zeros((), device=device)
        return ce, pois

    tr_idx = torch.arange(cut, device=device)
    va_idx = torch.arange(cut, n, device=device)

    best = {"val": float("inf"), "epoch": -1, "state": None}
    history = []

    for epoch in range(cfg.max_epochs):
        model.train()
        perm = tr_idx[torch.randperm(len(tr_idx), device=device)]
        for i in range(0, len(perm), cfg.batch_size):
            batch = perm[i:i + cfg.batch_size]
            ce, pois = losses(batch)
            loss = ce + cfg.goal_loss_weight * pois
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            ce, pois = losses(va_idx)
            val = float(ce)                # select on the 1X2 loss, not the sum
        history.append({"epoch": epoch, "val_ce": val, "val_poisson": float(pois)})

        if val < best["val"] - 1e-5:
            best = {"val": val, "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
        elif epoch - best["epoch"] >= cfg.patience:
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()

    meta = {"mu": mu, "sd": sd, "best_epoch": best["epoch"],
            "warm_started": init_from is not None,
            "best_val_ce": best["val"], "epochs_run": len(history),
            "history": history}
    if verbose:
        print(f"  stopped at epoch {len(history)}, best {best['epoch']} "
              f"(val CE {best['val']:.4f})")
    return model, meta


@torch.no_grad()
def predict(model: MatchNet, df: pd.DataFrame, X: np.ndarray, vocab: Vocab,
            meta: dict, device: str = "cpu", seq: np.ndarray | None = None,
            squads: np.ndarray | None = None,
            squad_mask: np.ndarray | None = None) -> dict:
    """Returns the softmax 1X2, the Poisson-implied 1X2, and the goal rates."""
    Xs = np.nan_to_num((X - meta["mu"]) / meta["sd"])
    xc, hid, aid, lid = _tensors(df, Xs, vocab, device)
    sq = (torch.tensor(seq, dtype=torch.float32, device=device)
          if model.cfg.seq_hidden > 0 else None)
    sqd = (torch.tensor(squads, dtype=torch.float32, device=device)
           if model.cfg.squad_hidden > 0 else None)
    sqm = (torch.tensor(squad_mask, dtype=torch.bool, device=device)
           if model.cfg.squad_hidden > 0 else None)
    model.eval()
    logits, rates = model(xc, hid, aid, lid, sq, sqd, sqm)
    p = torch.softmax(logits, dim=-1).cpu().numpy()
    r = rates.cpu().numpy()
    return {"hda": p / p.sum(axis=1, keepdims=True),
            "goal_rates": r,
            "hda_from_goals": poisson_to_hda(r, model.cfg.max_goals),
            "logits": logits.cpu().numpy()}


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

class TemperatureScaler:
    """Single-parameter temperature scaling (Guo et al., 2017).

    Divides the logits by one learned scalar fitted on held-out data. One
    parameter, preserves the argmax, essentially cannot overfit -- the right
    default for a net. Isotonic regression is more flexible and needs far more
    held-out data; a study that moved ROI from ~1% to ~10% on the calibration
    step alone is the reason this is fitted strictly out of sample.
    """

    def __init__(self):
        self.log_t = 0.0

    @property
    def temperature(self) -> float:
        return float(np.exp(self.log_t))

    def fit(self, logits: np.ndarray, y) -> "TemperatureScaler":
        z = torch.tensor(np.asarray(logits), dtype=torch.float32)
        t = torch.tensor([OUTCOMES.index(str(v).strip().upper()) for v in y])
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.2, max_iter=80)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(z / torch.exp(log_t), t)
            loss.backward()
            return loss

        opt.step(closure)
        self.log_t = float(log_t.detach())
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        z = np.asarray(logits, dtype=float) / self.temperature
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)
