"""Modello MLP per Behavioral Cloning (Section II-C del paper).

Implementa una policy pi_theta : s_t -> a_t come MLP a 4 hidden layers (in
[32, 256] neuroni), attivazione ReLU o tanh, allenata con MSE su Adam.

Le statistiche di z-score (i_mean, i_std, o_mean, o_std) sono memorizzate
all'interno del modulo come buffer non-trainabili: in questo modo, una volta
caricato un checkpoint con ``BCPolicy.load(...)``, ``predict_action`` puo'
essere chiamato direttamente con stati raw senza dover ri-trasportare le
statistiche manualmente.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}


@dataclass
class BCConfig:
    state_dim: int
    action_dim: int
    hidden: tuple[int, ...]
    activation: str = "relu"

    def validate(self) -> None:
        if self.activation not in _ACTIVATIONS:
            raise ValueError(
                f"Activation '{self.activation}' non supportata. "
                f"Scegli tra {list(_ACTIVATIONS)}."
            )
        if not self.hidden:
            raise ValueError("hidden deve contenere almeno un layer.")


class BCPolicy(nn.Module):
    """MLP fully-connected per BC con normalizzazione z-score integrata."""

    def __init__(self, cfg: BCConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg

        Act = _ACTIVATIONS[cfg.activation]
        layers: list[nn.Module] = []
        in_dim = cfg.state_dim
        for h in cfg.hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(Act())
            in_dim = h
        layers.append(nn.Linear(in_dim, cfg.action_dim))
        self.net = nn.Sequential(*layers)

        # Buffer per z-score (popolati in fit() o load())
        self.register_buffer("i_mean", torch.zeros(cfg.state_dim))
        self.register_buffer("i_std", torch.ones(cfg.state_dim))
        self.register_buffer("o_mean", torch.zeros(cfg.action_dim))
        self.register_buffer("o_std", torch.ones(cfg.action_dim))

    # ------------------------------------------------------------------
    # Forward / prediction
    # ------------------------------------------------------------------
    def forward(self, s_norm: torch.Tensor) -> torch.Tensor:
        """Forward pass su input gia' normalizzato."""
        return self.net(s_norm)

    @torch.no_grad()
    def predict_action(self, s: np.ndarray) -> np.ndarray:
        """Predice a_t a partire da uno stato raw s_t (1D o 2D)."""
        single = (s.ndim == 1)
        s_t = torch.as_tensor(np.atleast_2d(s), dtype=torch.float32,
                              device=self.i_mean.device)
        s_n = (s_t - self.i_mean) / self.i_std
        a_n = self.net(s_n)
        a = a_n * self.o_std + self.o_mean
        a_np = a.cpu().numpy()
        return a_np[0] if single else a_np

    # ------------------------------------------------------------------
    # Set normalization stats
    # ------------------------------------------------------------------
    def set_norm_stats(self, i_mean: np.ndarray, i_std: np.ndarray,
                       o_mean: np.ndarray, o_std: np.ndarray) -> None:
        self.i_mean.copy_(torch.as_tensor(i_mean, dtype=torch.float32))
        self.i_std.copy_(torch.as_tensor(i_std, dtype=torch.float32))
        self.o_mean.copy_(torch.as_tensor(o_mean, dtype=torch.float32))
        self.o_std.copy_(torch.as_tensor(o_std, dtype=torch.float32))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(
        self,
        I_norm: np.ndarray,
        O_norm: np.ndarray,
        epochs: int,
        batch_size: int,
        lr: float,
        weight_decay: float = 0.0,
        device: str | torch.device = "cpu",
        random_state: int | None = 0,
        log_every: int = 25,
        val_split: float = 0.0,
        noise_std: float = 0.0,
    ) -> dict:
        """Training MSE + Adam su (I_norm, O_norm) gia' z-scored.

        ``noise_std`` (>=0) aggiunge rumore gaussiano allo stato di input ad
        ogni minibatch (data augmentation per mitigare il covariate shift
        tipico del BC con poche demo, vedi DART, Laskey 2017). Poiche'
        l'input e' gia' z-scored, ``noise_std`` e' interpretabile come
        frazione della std per dimensione (es. 0.02 = ~2% della std).
        Lasciato a 0.0 disabilita l'augmentation.

        Ritorna un dizionario con la storia (train/val loss per epoca).
        """
        if noise_std < 0.0:
            raise ValueError(f"noise_std deve essere >= 0, ricevuto {noise_std}.")
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        device = torch.device(device)
        self.to(device)

        I_t = torch.as_tensor(I_norm, dtype=torch.float32)
        O_t = torch.as_tensor(O_norm, dtype=torch.float32)

        n = I_t.shape[0]
        n_val = int(round(val_split * n)) if val_split > 0 else 0
        if n_val > 0:
            perm = torch.randperm(n)
            val_idx = perm[:n_val]
            tr_idx = perm[n_val:]
            I_tr, O_tr = I_t[tr_idx], O_t[tr_idx]
            I_val, O_val = I_t[val_idx].to(device), O_t[val_idx].to(device)
        else:
            I_tr, O_tr = I_t, O_t
            I_val = O_val = None

        ds = torch.utils.data.TensorDataset(I_tr, O_tr)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=True, drop_last=False,
        )

        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, epochs + 1):
            self.train()
            running = 0.0
            count = 0
            for xb, yb in loader:
                xb = xb.to(device); yb = yb.to(device)
                if noise_std > 0.0:
                    xb = xb + torch.randn_like(xb) * noise_std
                opt.zero_grad(set_to_none=True)
                pred = self.net(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                running += float(loss.item()) * xb.shape[0]
                count += xb.shape[0]
            train_loss = running / max(count, 1)
            history["train_loss"].append(train_loss)

            val_loss = float("nan")
            if I_val is not None:
                self.eval()
                with torch.no_grad():
                    val_loss = float(loss_fn(self.net(I_val), O_val).item())
            history["val_loss"].append(val_loss)

            if log_every > 0 and (epoch == 1 or epoch % log_every == 0 or epoch == epochs):
                msg = f"  epoch {epoch:4d}/{epochs}  train_mse={train_loss:.6e}"
                if I_val is not None:
                    msg += f"  val_mse={val_loss:.6e}"
                print(msg)

        return history

    # ------------------------------------------------------------------
    # Persistenza (.pt)
    # ------------------------------------------------------------------
    def save(self, path: Path | str, extra: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.state_dict(),
            "config": {
                "state_dim": self.cfg.state_dim,
                "action_dim": self.cfg.action_dim,
                "hidden": list(self.cfg.hidden),
                "activation": self.cfg.activation,
            },
        }
        if extra:
            payload["extra"] = extra
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path | str, map_location: str | torch.device = "cpu") -> "BCPolicy":
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        cfg_d = payload["config"]
        cfg = BCConfig(
            state_dim=int(cfg_d["state_dim"]),
            action_dim=int(cfg_d["action_dim"]),
            hidden=tuple(int(h) for h in cfg_d["hidden"]),
            activation=str(cfg_d["activation"]),
        )
        model = cls(cfg)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model

    @classmethod
    def load_with_extra(cls, path: Path | str,
                        map_location: str | torch.device = "cpu"):
        """Come ``load`` ma ritorna anche il dizionario ``extra`` salvato."""
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        cfg_d = payload["config"]
        cfg = BCConfig(
            state_dim=int(cfg_d["state_dim"]),
            action_dim=int(cfg_d["action_dim"]),
            hidden=tuple(int(h) for h in cfg_d["hidden"]),
            activation=str(cfg_d["activation"]),
        )
        model = cls(cfg)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model, payload.get("extra", {})


def parse_hidden(spec: str | Iterable[int]) -> tuple[int, ...]:
    """Parsa una spec del tipo '128,128,128,64' in una tupla di int."""
    if isinstance(spec, str):
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        return tuple(int(p) for p in parts)
    return tuple(int(x) for x in spec)
