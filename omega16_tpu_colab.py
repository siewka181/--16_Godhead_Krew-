"""
Ω-16 Swarm (server-side logic) dla Google Colab + JAX (TPU-ready).

Cel:
- Symulacja 1.2M agentów z API po stronie "serwera" (kontroler sesji).
- Praca na TPU przez `jax.pmap` (gdy dostępne), z fallbackiem na CPU/GPU.
- Kod gotowy do użycia w Colab jako backend do notebookowych komórek klienckich.

Szybki start (Colab TPU):
1) Runtime -> Change runtime type -> TPU
2) !pip -q install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
3) from omega16_tpu_colab import Omega16Config, Omega16Server
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jax import lax


# =========================
# Konfiguracja i stan modelu
# =========================


@dataclass(frozen=True)
class Omega16Config:
    total_agents: int = 1_200_000
    world_size: float = 1.0
    dt: float = 0.025
    max_speed: float = 0.85

    # Wagi dynamiki Ω-16 (uproszczony model stabilny numerycznie)
    w_cohesion: float = 0.12
    w_alignment: float = 0.48
    w_separation: float = 0.08
    w_noise: float = 0.004

    # "Promień" separacji w funkcji odległości od środka (aproksymacja)
    avoid_radius: float = 0.02


@dataclass
class SwarmState:
    pos: jnp.ndarray  # [devices, local_agents, 2]
    vel: jnp.ndarray  # [devices, local_agents, 2]
    step: jnp.ndarray  # scalar int32


# =========================
# Niskopoziomowa fizyka roju
# =========================


def _init_local_agents(local_agents: int, key: jnp.ndarray, world_size: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Inicjalizacja shardu agentów."""
    k1, k2 = jax.random.split(key)
    pos = jax.random.uniform(
        k1,
        (local_agents, 2),
        minval=-world_size,
        maxval=world_size,
        dtype=jnp.float32,
    )
    vel = jax.random.normal(k2, (local_agents, 2), dtype=jnp.float32) * 0.03
    return pos, vel


def _omega16_local_dynamics(
    pos: jnp.ndarray,
    vel: jnp.ndarray,
    key: jnp.ndarray,
    cfg: Omega16Config,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Uproszczona dynamika Ω-16 (skalowalna):
    - cohesion/alignment względem średniej shardu,
    - separation jako odpychanie od centrum świata,
    - szum gaussowski.

    Uwaga: to model przybliżony, zaprojektowany pod dużą skalę (1.2M).
    """
    center = jnp.mean(pos, axis=0, keepdims=True)
    avg_vel = jnp.mean(vel, axis=0, keepdims=True)

    cohesion = center - pos
    alignment = avg_vel - vel
    separation = -jnp.tanh(pos / cfg.avoid_radius)
    noise = jax.random.normal(key, pos.shape, dtype=jnp.float32)

    acc = (
        cfg.w_cohesion * cohesion
        + cfg.w_alignment * alignment
        + cfg.w_separation * separation
        + cfg.w_noise * noise
    )

    new_vel = vel + acc * cfg.dt
    speed = jnp.linalg.norm(new_vel, axis=1, keepdims=True) + 1e-7
    new_vel = jnp.where(speed > cfg.max_speed, new_vel * (cfg.max_speed / speed), new_vel)

    new_pos = pos + new_vel * cfg.dt
    ws = cfg.world_size
    new_pos = jnp.where(new_pos > ws, -ws, new_pos)
    new_pos = jnp.where(new_pos < -ws, ws, new_pos)
    return new_pos, new_vel


def _pmap_step_fn(local_pos: jnp.ndarray, local_vel: jnp.ndarray, key: jnp.ndarray, cfg: Omega16Config):
    """Krok symulacji wykonywany równolegle per urządzenie."""
    new_pos, new_vel = _omega16_local_dynamics(local_pos, local_vel, key, cfg)

    # Cross-shard globalne centrum (all-reduce)
    local_center = jnp.mean(new_pos, axis=0)
    global_center = lax.pmean(local_center, axis_name="devices")
    new_pos = new_pos + 0.01 * (global_center[None, :] - new_pos)
    return new_pos, new_vel


# Statyczny pmap kompilowany raz na proces
_PMAP_STEP = jax.pmap(_pmap_step_fn, axis_name="devices", static_broadcasted_argnums=(3,))


@jax.jit
def _single_device_step(pos: jnp.ndarray, vel: jnp.ndarray, key: jnp.ndarray, cfg: Omega16Config):
    """Fallback dla 1 urządzenia (CPU/GPU) bez pmap."""
    return _omega16_local_dynamics(pos, vel, key, cfg)


# =========================
# Warstwa serwerowa (API backendu)
# =========================


class Omega16Server:
    """
    Serwer logiki roju Ω-16 dla Colab.

    Dostępne operacje:
    - initialize(seed)
    - tick(n_steps)
    - metrics()
    - snapshot(sample)
    """

    def __init__(self, config: Omega16Config = Omega16Config()):
        self.cfg = config
        self.device_count = jax.device_count()
        if self.cfg.total_agents % self.device_count != 0:
            raise ValueError(
                f"total_agents={self.cfg.total_agents} musi dzielić się przez device_count={self.device_count}"
            )
        self.local_agents = self.cfg.total_agents // self.device_count
        self.initialized = False
        self.state: SwarmState | None = None
        self._master_key = jax.random.PRNGKey(0)

    def initialize(self, seed: int = 2026) -> Dict[str, int | float]:
        """Inicjalizacja świata i agentów."""
        self._master_key = jax.random.PRNGKey(seed)

        keys = jax.random.split(self._master_key, self.device_count + 1)
        self._master_key = keys[0]
        init_keys = keys[1:]

        if self.device_count > 1:
            def build_for_device(k):
                return _init_local_agents(self.local_agents, k, self.cfg.world_size)

            local = [build_for_device(k) for k in init_keys]
            pos = jnp.stack([x[0] for x in local], axis=0)
            vel = jnp.stack([x[1] for x in local], axis=0)
        else:
            pos1, vel1 = _init_local_agents(self.local_agents, init_keys[0], self.cfg.world_size)
            pos = pos1[None, ...]
            vel = vel1[None, ...]

        self.state = SwarmState(pos=pos, vel=vel, step=jnp.array(0, dtype=jnp.int32))
        self.initialized = True

        return {
            "total_agents": self.cfg.total_agents,
            "device_count": self.device_count,
            "agents_per_device": self.local_agents,
        }

    def tick(self, n_steps: int = 1) -> Dict[str, float | int]:
        """Wykonuje n kroków symulacji i zwraca telemetry."""
        if not self.initialized or self.state is None:
            raise RuntimeError("Server nie został zainicjalizowany. Wywołaj initialize().")
        if n_steps <= 0:
            raise ValueError("n_steps musi być > 0")

        t0 = time.time()
        state = self.state

        for _ in range(n_steps):
            k_main, k_step = jax.random.split(self._master_key)
            self._master_key = k_main

            if self.device_count > 1:
                step_keys = jax.random.split(k_step, self.device_count)
                pos, vel = _PMAP_STEP(state.pos, state.vel, step_keys, self.cfg)
            else:
                pos0, vel0 = _single_device_step(state.pos[0], state.vel[0], k_step, self.cfg)
                pos, vel = pos0[None, ...], vel0[None, ...]

            state = SwarmState(pos=pos, vel=vel, step=state.step + 1)

        # synchronizacja hosta
        state.pos.block_until_ready()
        elapsed = time.time() - t0
        self.state = state

        return {
            "step": int(jax.device_get(state.step)),
            "elapsed_sec": float(elapsed),
            "agent_steps_per_sec": float((self.cfg.total_agents * n_steps) / max(elapsed, 1e-9)),
        }

    def metrics(self) -> Dict[str, float | int]:
        """Szybkie metryki globalne (serwer-side)."""
        if self.state is None:
            raise RuntimeError("Brak stanu. Wywołaj initialize().")

        pos = self.state.pos.reshape(-1, 2)
        vel = self.state.vel.reshape(-1, 2)
        mean_pos = jnp.mean(pos, axis=0)
        mean_speed = jnp.mean(jnp.linalg.norm(vel, axis=1))
        std_pos = jnp.std(pos, axis=0)

        return {
            "step": int(jax.device_get(self.state.step)),
            "mean_pos_x": float(jax.device_get(mean_pos[0])),
            "mean_pos_y": float(jax.device_get(mean_pos[1])),
            "mean_speed": float(jax.device_get(mean_speed)),
            "std_pos_x": float(jax.device_get(std_pos[0])),
            "std_pos_y": float(jax.device_get(std_pos[1])),
        }

    def snapshot(self, sample: int = 8192) -> Dict[str, jnp.ndarray | int]:
        """
        Zwraca mały wycinek danych do wizualizacji po stronie klienta notebooka.
        """
        if self.state is None:
            raise RuntimeError("Brak stanu. Wywołaj initialize().")
        if sample <= 0:
            raise ValueError("sample musi być > 0")

        pos = self.state.pos.reshape(-1, 2)
        vel = self.state.vel.reshape(-1, 2)
        n = pos.shape[0]
        sample = min(sample, int(n))

        idx = jnp.linspace(0, n - 1, sample, dtype=jnp.int32)
        return {
            "step": int(jax.device_get(self.state.step)),
            "pos": jax.device_get(pos[idx]),
            "vel": jax.device_get(vel[idx]),
        }


# =========================
# Przykład użycia w skrypcie
# =========================


def main() -> None:
    cfg = Omega16Config(total_agents=1_200_000)
    server = Omega16Server(cfg)

    meta = server.initialize(seed=2026)
    print("[init]", meta)

    # Wykonaj partię kroków (server-side)
    telem = server.tick(n_steps=250)
    print("[tick]", telem)

    m = server.metrics()
    print("[metrics]", m)

    snap = server.snapshot(sample=4096)
    print("[snapshot] step=", snap["step"], "sample=", snap["pos"].shape[0])


if __name__ == "__main__":
    main()
