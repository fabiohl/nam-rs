#!/usr/bin/env python3
"""
Deterministic A2-Full (CH=8) and A2-Lite (CH=3) fixture generator.

Produces wavenet_a2_full.nam and wavenet_a2_lite.nam with the fixed
A2 skeleton (23 layers, canonical kernels/dilations, LeakyReLU, head_scale).

Source of truth: NAM/wavenet/a2_fast.h:30-43
Weight stream order mirrors a2_fast.cpp:196-282 as consumed by
the Rust WaveNetA2::set_weights() in src/models/a2/model.rs.
"""

import json
import random
from pathlib import Path
from typing import List

OUTPUT_DIR = Path(__file__).resolve().parent / "models"

KERNEL_SIZES = [
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    15, 15,
    6, 6, 6, 6, 6, 6, 6,
]

DILATIONS = [
    1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239,
    1, 13,
    1, 3, 7, 17, 41, 101, 239,
]

HEAD_KERNEL_SIZE = 16
HEAD_SCALE = 0.02
SEED = 42
NUM_LAYERS = 23


def gen_weights(n: int, rng: random.Random, scale: float = 0.05) -> List[float]:
    return [rng.uniform(-1.0, 1.0) * scale for _ in range(n)]


def count_weights(ch: int) -> int:
    count = 0
    count += ch          # rechannel_w (no bias — matches C++ A2FastModel)
    for k in KERNEL_SIZES:
        count += ch * ch * k  # conv_w (CH × CH × K)
        count += ch           # conv_b
        count += ch           # mixin_w
        count += ch * ch      # l1x1_w
        count += ch           # l1x1_b
    count += HEAD_KERNEL_SIZE * ch  # head_w
    count += 1                      # head_b
    count += 1                      # head_scale
    return count


def generate_weights(ch: int, rng: random.Random) -> List[float]:
    weights: List[float] = []

    # 1. Rechannel: weights (CH) — no bias (matches C++ A2FastModel)
    weights.extend(gen_weights(ch, rng, scale=0.05))

    # 2. Per-layer
    for k in KERNEL_SIZES:
        # conv_w: CH × CH × K
        weights.extend(gen_weights(ch * ch * k, rng, scale=0.05))
        # conv_b: CH
        weights.extend(gen_weights(ch, rng, scale=0.01))
        # mixin_w: CH
        weights.extend(gen_weights(ch, rng, scale=0.05))
        # l1x1_w: CH × CH
        weights.extend(gen_weights(ch * ch, rng, scale=0.05))
        # l1x1_b: CH
        weights.extend(gen_weights(ch, rng, scale=0.01))

    # 3. Head rechannel: 16*CH weights + 1 bias
    weights.extend(gen_weights(HEAD_KERNEL_SIZE * ch, rng, scale=0.05))
    weights.extend(gen_weights(1, rng, scale=0.01))

    # 4. Head scale: 1 float (should be positive and small, like 0.02)
    weights.extend([0.02])

    return weights


def build_layer_config(ch: int) -> dict:
    return {
        "input_size": 1,
        "condition_size": 1,
        "channels": ch,
        "bottleneck": ch,
        "head": {
            "out_channels": 1,
            "kernel_size": HEAD_KERNEL_SIZE,
            "bias": True,
        },
        "kernel_sizes": list(KERNEL_SIZES),
        "dilations": list(DILATIONS),
        "activation": [{"type": "LeakyReLU", "negative_slope": 0.01}] * NUM_LAYERS,
        "gating_mode": ["none"] * NUM_LAYERS,
        "head1x1": {"active": False, "out_channels": ch, "groups": 1},
        "layer1x1": {"active": True, "groups": 1},
        "groups_input": 1,
        "groups_input_mixin": 1,
    }


def build_nam(ch: int, weights: List[float], label: str) -> dict:
    return {
        "version": "0.6.0",
        "architecture": "WaveNet",
        "config": {
            "in_channels": 1,
            "head_scale": HEAD_SCALE,
            "head": None,
            "layers": [build_layer_config(ch)],
        },
        "weights": weights,
        "metadata": {
            "name": f"A2-{label} Fixture (CH={ch})",
            "modeled_by": "tests/fixtures/generate_a2_fixtures.py",
        },
        "sample_rate": 48000,
    }


def main() -> None:
    rng = random.Random(SEED)
    models = {}

    for ch, label, fname in [
        (3, "Lite", "wavenet_a2_lite.nam"),
        (8, "Full", "wavenet_a2_full.nam"),
    ]:
        expected = count_weights(ch)
        weights = generate_weights(ch, rng)
        assert len(weights) == expected, (
            f"CH={ch}: got {len(weights)} weights, expected {expected}"
        )
        doc = build_nam(ch, weights, label)
        models[label] = doc
        out_path = OUTPUT_DIR / fname
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(doc, f, indent=2)
        print(f"Written {out_path}  ({len(weights)} weights)")

    # Generate the container model
    container_doc = {
        "version": "0.7.0",
        "architecture": "SlimmableContainer",
        "config": {
            "submodels": [
                {
                    "max_value": 0.5,
                    "model": models["Lite"]
                },
                {
                    "max_value": 1.0,
                    "model": models["Full"]
                }
            ]
        },
        "weights": [],
        "sample_rate": 48000,
        "metadata": {
            "name": "A2-Container Fixture",
            "modeled_by": "tests/fixtures/generate_a2_fixtures.py",
        }
    }
    out_path = OUTPUT_DIR / "wavenet_a2_container.nam"
    with open(out_path, "w") as f:
        json.dump(container_doc, f, indent=2)
    print(f"Written {out_path} (Container)")

    print("Done.")


if __name__ == "__main__":
    main()
