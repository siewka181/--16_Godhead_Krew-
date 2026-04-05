"""
Symulacja 1.2M agentów Roju Ω-16 na TPU (Google Colab) z użyciem JAX.

Instrukcja uruchomienia w Colab:
1) Runtime -> Change runtime type -> TPU
2) Uruchom:
   !pip -q install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
3) Następnie uruchom ten skrypt.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import lax


# Parametry zadania
TOTAL_AGENTS = 1_200_000
STEPS = 500
DT = 0.03
NEIGHBOR_RADIUS = 0.045
AVOID_RADIUS = 0.012
MAX_SPEED = 0.9
NOISE_SCALE = 0.005


@dataclass
class SwarmState:
    pos: jnp.ndarray  # [local_agents, 2]
    vel: jnp.ndarray  # [local_agents, 2]


def init_local_state(local_agents: int, seed: int) -> SwarmState:
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    pos = jax.random.uniform(k1, (local_agents, 2), minval=-1.0, maxval=1.0)
    vel = jax.random.normal(k2, (local_agents, 2)) * 0.05
    return SwarmState(pos=pos, vel=vel)


def pairwise_rules(pos: jnp.ndarray, vel: jnp.ndarray, key: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Uproszczone reguły roju Ω-16 (lokalna spójność, wyrównanie, unikanie).
    Złożoność O(N^2) byłaby zbyt duża, więc aproksymujemy oddziaływania
    przez globalne/statystyczne sygnały + lokalne zaburzenia.
    """
    center = jnp.mean(pos, axis=0, keepdims=True)            # spójność
    avg_vel = jnp.mean(vel, axis=0, keepdims=True)           # wyrównanie

    to_center = center - pos
    align = avg_vel - vel

    # Unikanie: pseudo-losowy kierunek zależny od pozycji i PRNG
    noise = jax.random.normal(key, pos.shape) * NOISE_SCALE
    avoid = -jnp.tanh(pos / AVOID_RADIUS) * 0.01

    acc = 0.15 * to_center + 0.55 * align + avoid + noise
    new_vel = vel + acc * DT

    speed = jnp.linalg.norm(new_vel, axis=1, keepdims=True) + 1e-7
    new_vel = jnp.where(speed > MAX_SPEED, new_vel * (MAX_SPEED / speed), new_vel)

    new_pos = pos + new_vel * DT
    new_pos = jnp.where(new_pos > 1.0, -1.0, new_pos)
    new_pos = jnp.where(new_pos < -1.0, 1.0, new_pos)
    return new_pos, new_vel


def shard_step(local_state: SwarmState, key: jnp.ndarray) -> SwarmState:
    pos, vel = pairwise_rules(local_state.pos, local_state.vel, key)

    # Globalny sygnał między shardami TPU (all-reduce)
    local_center = jnp.mean(pos, axis=0)
    global_center = lax.pmean(local_center, axis_name="devices")
    pos = pos + 0.01 * (global_center[None, :] - pos)

    return SwarmState(pos=pos, vel=vel)


p_shard_step = jax.pmap(shard_step, axis_name="devices")


def run_simulation(total_agents: int = TOTAL_AGENTS, steps: int = STEPS, seed: int = 42) -> SwarmState:
    device_count = jax.device_count()
    if total_agents % device_count != 0:
        raise ValueError(
            f"Liczba agentów ({total_agents}) musi dzielić się przez liczbę urządzeń TPU ({device_count})."
        )

    local_agents = total_agents // device_count
    print(f"TPU devices: {device_count}")
    print(f"Agenci na shard: {local_agents}")

    keys = jax.random.split(jax.random.PRNGKey(seed), device_count * (steps + 1)).reshape(device_count, steps + 1, 2)

    # Inicjalizacja stanu na każdym rdzeniu TPU
    init_states = [init_local_state(local_agents, seed + i) for i in range(device_count)]
    pos = jnp.stack([s.pos for s in init_states], axis=0)
    vel = jnp.stack([s.vel for s in init_states], axis=0)
    state = SwarmState(pos=pos, vel=vel)

    @jax.jit
    def rollout(s: SwarmState, all_keys: jnp.ndarray) -> SwarmState:
        def body_fn(carry: SwarmState, t: int):
            step_keys = all_keys[:, t, :]
            nxt = p_shard_step(carry, step_keys)
            return nxt, None

        final_state, _ = lax.scan(body_fn, s, jnp.arange(steps))
        return final_state

    # Warm-up kompilacji
    _ = rollout(state, keys).pos.block_until_ready()

    start = time.time()
    final_state = rollout(state, keys)
    final_state.pos.block_until_ready()
    elapsed = time.time() - start

    agents_per_sec = total_agents * steps / max(elapsed, 1e-9)
    print(f"Czas symulacji: {elapsed:.3f}s")
    print(f"Przepustowość: {agents_per_sec:,.0f} agent-kroków/s")

    return final_state


def summarize(state: SwarmState) -> None:
    pos = jax.device_get(state.pos).reshape(-1, 2)
    vel = jax.device_get(state.vel).reshape(-1, 2)

    mean_pos = pos.mean(axis=0)
    mean_speed = jnp.linalg.norm(vel, axis=1).mean()
    cloud_std = pos.std(axis=0)

    print("\n=== Podsumowanie Ω-16 ===")
    print(f"Średnia pozycja: x={mean_pos[0]:.4f}, y={mean_pos[1]:.4f}")
    print(f"Średnia prędkość: {float(mean_speed):.4f}")
    print(f"Rozrzut chmury: sx={cloud_std[0]:.4f}, sy={cloud_std[1]:.4f}")


if __name__ == "__main__":
    print("Start symulacji Ω-16 (JAX TPU)")
    swarm_state = run_simulation(total_agents=TOTAL_AGENTS, steps=STEPS, seed=2026)
    summarize(swarm_state)
