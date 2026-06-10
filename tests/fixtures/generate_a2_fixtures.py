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
import math
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


def gen_weights(n: int, rng: random.Random) -> List[float]:
    return [rng.uniform(-1.0, 1.0) for _ in range(n)]


def num_blocks(ch: int) -> int:
    return math.ceil(ch / 4)


def count_weights(ch: int) -> int:
    nblk = num_blocks(ch)
    count = 0
    count += ch          # rechannel_w
    count += ch          # rechannel_b
    for k in KERNEL_SIZES:
        count += nblk * 4 * ch * k  # conv_w (padded interleaved 4-wide)
        count += ch                 # conv_b
        count += ch                 # mixin_w
        count += ch * ch            # l1x1_w
        count += ch                 # l1x1_b
    count += HEAD_KERNEL_SIZE * ch  # head_w
    count += 1                      # head_b
    count += 1                      # head_scale
    return count


def generate_weights(ch: int, rng: random.Random) -> List[float]:
    nblk = num_blocks(ch)
    weights: List[float] = []

    # 1. Rechannel: weights (CH) + bias (CH)
    weights.extend(gen_weights(ch, rng))
    weights.extend(gen_weights(ch, rng))

    # 2. Per-layer
    for k in KERNEL_SIZES:
        # conv_w: ceil(CH/4) * 4 * CH * K
        weights.extend(gen_weights(nblk * 4 * ch * k, rng))
        # conv_b: CH
        weights.extend(gen_weights(ch, rng))
        # mixin_w: CH
        weights.extend(gen_weights(ch, rng))
        # l1x1_w: CH * CH
        weights.extend(gen_weights(ch * ch, rng))
        # l1x1_b: CH
        weights.extend(gen_weights(ch, rng))

    # 3. Head rechannel: 16*CH weights + 1 bias
    weights.extend(gen_weights(HEAD_KERNEL_SIZE * ch, rng))
    weights.extend(gen_weights(1, rng))

    # 4. Head scale: 1 float
    weights.extend(gen_weights(1, rng))

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
        "activation": "LeakyReLU",
        "gating_mode": ["none"] * NUM_LAYERS,
        "head1x1": {"active": False},
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
    }


def main() -> None:
    rng = random.Random(SEED)

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
        out_path = OUTPUT_DIR / fname
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(doc, f, indent=2)
        print(f"Written {out_path}  ({len(weights)} weights)")

    print("Done.")


if __name__ == "__main__":
    main()
