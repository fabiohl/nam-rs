#!/usr/bin/env python3
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
"""
Deterministic A2-Full (CH=8) and A2-Lite (CH=3) fixture generator.

Produces wavenet_a2_full.nam and wavenet_a2_lite.nam with the fixed
A2 skeleton (23 layers, canonical kernels/dilations, LeakyReLU, head_scale).

Also produces dynamic A2 models with gating/blending for WaveNetA2Dyn parity
validation (Task 3.3: Golden Vectors e C++ Parity).

Source of truth: NAM/wavenet/a2_fast.h:30-43
Weight stream order mirrors C++ WaveNet::set_weights_() as consumed by
both C++ generic WaveNet and Rust WaveNetA2Dyn::set_weights().
"""

import json
import random
from pathlib import Path
from typing import List, Optional

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
SEED_DYNAMIC = 123
NUM_LAYERS = 23


# T2.5: Weight scales tuned so that A2 output lands in realistic audio regime
# (pico ≈ 0.3, LUFS ≈ −18 to −23) instead of near-silence (pico ≈ 2e-3, LUFS ≈ −68).
# Output grows super-linearly with CH count (more internal channels = more gain
# accumulation across 23 layers). Lite (CH=3) needs higher scale than Full (CH=8).
SCALES = {
    3: {"weight": 0.45, "bias": 0.09},
    8: {"weight": 0.28, "bias": 0.065},
}

# Dynamic model scales (gentler to avoid saturation with gating)
SCALES_DYNAMIC = {
    3: {"weight": 0.30, "bias": 0.06},
    8: {"weight": 0.18, "bias": 0.04},
}


def gen_weights(n: int, rng: random.Random, scale: float) -> List[float]:
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


def count_weights_film(ch: int, num_film_keys: int) -> int:
    base = count_weights(ch)
    for _ in range(NUM_LAYERS):
        for _ in range(num_film_keys):
            base += ch * 2      # film_w (cond_size=1: ch*2*1)
            base += ch * 2      # film_b
    return base


def count_weights_dynamic(ch: int, bottleneck: int, gating_modes: List[str]) -> int:
    count = 0
    count += ch  # rechannel_w
    for k, gm in zip(KERNEL_SIZES, gating_modes):
        out_ch = bottleneck * 2 if gm in ("gated", "blended") else bottleneck
        count += ch * out_ch * k  # conv_w
        count += out_ch           # conv_b
        count += out_ch           # mixin_w
        count += bottleneck * ch  # l1x1_w
        count += ch               # l1x1_b
    count += HEAD_KERNEL_SIZE * ch  # head_w
    count += 1                      # head_b
    count += 1                      # head_scale
    return count


def generate_weights(ch: int, rng: random.Random) -> List[float]:
    scales = SCALES[ch]
    ws = scales["weight"]
    bs = scales["bias"]
    weights: List[float] = []

    # 1. Rechannel: weights (CH) — no bias (matches C++ A2FastModel)
    weights.extend(gen_weights(ch, rng, scale=ws))

    # 2. Per-layer
    for k in KERNEL_SIZES:
        # conv_w: CH × CH × K
        weights.extend(gen_weights(ch * ch * k, rng, scale=ws))
        # conv_b: CH
        weights.extend(gen_weights(ch, rng, scale=bs))
        # mixin_w: CH
        weights.extend(gen_weights(ch, rng, scale=ws))
        # l1x1_w: CH × CH
        weights.extend(gen_weights(ch * ch, rng, scale=ws))
        # l1x1_b: CH
        weights.extend(gen_weights(ch, rng, scale=bs))

    # 3. Head rechannel: 16*CH weights + 1 bias
    weights.extend(gen_weights(HEAD_KERNEL_SIZE * ch, rng, scale=ws))
    weights.extend(gen_weights(1, rng, scale=bs))

    # 4. Head scale: 1 float (should be positive and small, like 0.02)
    weights.extend([0.02])

    return weights


def generate_weights_film(ch: int, num_film_keys: int, rng: random.Random) -> List[float]:
    scales = SCALES[ch]
    ws = scales["weight"]
    bs = scales["bias"]
    weights: List[float] = []

    weights.extend(gen_weights(ch, rng, scale=ws))
    for k in KERNEL_SIZES:
        weights.extend(gen_weights(ch * ch * k, rng, scale=ws))
        weights.extend(gen_weights(ch, rng, scale=bs))
        weights.extend(gen_weights(ch, rng, scale=ws))
        weights.extend(gen_weights(ch * ch, rng, scale=ws))
        weights.extend(gen_weights(ch, rng, scale=bs))
        # FiLM weights per layer (ch*2 for w, ch*2 for b)
        for _ in range(num_film_keys):
            weights.extend(gen_weights(ch * 2, rng, scale=ws))
            scale_bias = [1.0 + v for v in gen_weights(ch, rng, scale=bs)]
            shift_bias = gen_weights(ch, rng, scale=bs)
            weights.extend(scale_bias + shift_bias)

    weights.extend(gen_weights(HEAD_KERNEL_SIZE * ch, rng, scale=ws))
    weights.extend(gen_weights(1, rng, scale=bs))
    weights.extend([0.02])
    return weights


def generate_weights_dynamic(
    ch: int, bottleneck: int, gating_modes: List[str], rng: random.Random
) -> List[float]:
    scales = SCALES_DYNAMIC.get(ch, SCALES.get(ch, {"weight": 0.2, "bias": 0.05}))
    ws = scales["weight"]
    bs = scales["bias"]
    weights: List[float] = []

    # 1. Rechannel: weights (CH)
    weights.extend(gen_weights(ch, rng, scale=ws))

    # 2. Per-layer
    for k, gm in zip(KERNEL_SIZES, gating_modes):
        out_ch = bottleneck * 2 if gm in ("gated", "blended") else bottleneck
        # conv_w: CH × out_ch × K
        weights.extend(gen_weights(ch * out_ch * k, rng, scale=ws))
        # conv_b: out_ch
        weights.extend(gen_weights(out_ch, rng, scale=bs))
        # mixin_w: out_ch
        weights.extend(gen_weights(out_ch, rng, scale=ws))
        # l1x1_w: bottleneck × CH (l1x1 always from bottleneck dimension)
        weights.extend(gen_weights(bottleneck * ch, rng, scale=ws))
        # l1x1_b: CH
        weights.extend(gen_weights(ch, rng, scale=bs))

    # 3. Head rechannel: 16*CH weights + 1 bias
    weights.extend(gen_weights(HEAD_KERNEL_SIZE * ch, rng, scale=ws))
    weights.extend(gen_weights(1, rng, scale=bs))

    # 4. Head scale
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


def build_layer_config_film(ch: int, film_keys: list) -> dict:
    cfg = build_layer_config(ch)
    for key in film_keys:
        cfg[key] = {"active": True, "shift": True, "groups": 1}
    return cfg


def build_layer_config_dynamic(
    ch: int,
    bottleneck: int,
    activations: List[dict],
    gating_modes: List[str],
    secondary_activations: List[Optional[dict]],
) -> dict:
    return {
        "input_size": 1,
        "condition_size": 1,
        "channels": ch,
        "bottleneck": bottleneck,
        "head": {
            "out_channels": 1,
            "kernel_size": HEAD_KERNEL_SIZE,
            "bias": True,
        },
        "kernel_sizes": list(KERNEL_SIZES),
        "dilations": list(DILATIONS),
        "activation": activations,
        "gating_mode": gating_modes,
        "secondary_activation": secondary_activations,
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


def build_nam_film(ch: int, weights: List[float], label: str, film_keys: list) -> dict:
    return {
        "version": "0.6.0",
        "architecture": "WaveNet",
        "config": {
            "in_channels": 1,
            "head_scale": HEAD_SCALE,
            "head": None,
            "layers": [build_layer_config_film(ch, film_keys)],
        },
        "weights": weights,
        "metadata": {
            "name": f"A2-{label} FiLM Fixture (CH={ch})",
            "modeled_by": "tests/fixtures/generate_a2_fixtures.py",
        },
        "sample_rate": 48000,
    }


def build_nam_dynamic(
    ch: int,
    bottleneck: int,
    weights: List[float],
    activations: List[dict],
    gating_modes: List[str],
    secondary_activations: List[Optional[dict]],
    label: str,
) -> dict:
    return {
        "version": "0.7.0",
        "architecture": "WaveNet",
        "config": {
            "in_channels": 1,
            "head_scale": HEAD_SCALE,
            "head": None,
            "layers": [
                build_layer_config_dynamic(
                    ch, bottleneck, activations, gating_modes, secondary_activations
                )
            ],
        },
        "weights": weights,
        "metadata": {
            "name": f"A2-{label} Fixture (CH={ch}, BN={bottleneck})",
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

    # ── Dynamic A2 models (Task 3.3: Golden Vectors e C++ Parity) ──────────
    rng_d = random.Random(SEED_DYNAMIC)

    # Model 1: A2-Dynamic-Gated CH=8 — gating on 3 layers (early, mid, late)
    ch, bn = 8, 8
    gating_modes_gated = ["none"] * NUM_LAYERS
    gating_modes_gated[2] = "gated"
    gating_modes_gated[10] = "gated"
    gating_modes_gated[18] = "gated"
    sec_activations_gated: List[Optional[dict]] = [None] * NUM_LAYERS
    sec_activations_gated[2] = {"type": "Sigmoid"}
    sec_activations_gated[10] = {"type": "Sigmoid"}
    sec_activations_gated[18] = {"type": "Sigmoid"}
    activations_gated = [{"type": "LeakyReLU", "negative_slope": 0.01}] * NUM_LAYERS
    expected_gated = count_weights_dynamic(ch, bn, gating_modes_gated)
    w_gated = generate_weights_dynamic(ch, bn, gating_modes_gated, rng_d)
    assert len(w_gated) == expected_gated, (
        f"Dynamic-Gated: got {len(w_gated)} weights, expected {expected_gated}"
    )
    doc_gated = build_nam_dynamic(
        ch, bn, w_gated, activations_gated, gating_modes_gated,
        sec_activations_gated, "Dynamic-Gated"
    )
    out_path = OUTPUT_DIR / "a2_dynamic_gated_ch8.nam"
    with open(out_path, "w") as f:
        json.dump(doc_gated, f, indent=2)
    print(f"Written {out_path}  ({len(w_gated)} weights)")

    # Model 2: A2-Dynamic-Blended CH=3 — blending on 2 layers
    ch, bn = 3, 3
    gating_modes_blended = ["none"] * NUM_LAYERS
    gating_modes_blended[5] = "blended"
    gating_modes_blended[15] = "blended"
    sec_activations_blended: List[Optional[dict]] = [None] * NUM_LAYERS
    sec_activations_blended[5] = {"type": "Tanh"}
    sec_activations_blended[15] = {"type": "Tanh"}
    activations_blended = [{"type": "LeakyReLU", "negative_slope": 0.01}] * NUM_LAYERS
    expected_blended = count_weights_dynamic(ch, bn, gating_modes_blended)
    w_blended = generate_weights_dynamic(ch, bn, gating_modes_blended, rng_d)
    assert len(w_blended) == expected_blended, (
        f"Dynamic-Blended: got {len(w_blended)} weights, expected {expected_blended}"
    )
    doc_blended = build_nam_dynamic(
        ch, bn, w_blended, activations_blended, gating_modes_blended,
        sec_activations_blended, "Dynamic-Blended"
    )
    out_path = OUTPUT_DIR / "a2_dynamic_blended_ch3.nam"
    with open(out_path, "w") as f:
        json.dump(doc_blended, f, indent=2)
    print(f"Written {out_path}  ({len(w_blended)} weights)")

    # ── A2-FiLM model (Tarefa B.1.1: FiLM routing policy) ──────────────────
    FILM_KEYS_ACTIVE = [
        "conv_post_film",
        "input_mixin_post_film",
        "activation_post_film",
        "layer1x1_post_film",
    ]
    for ch, label, fname in [(3, "FiLM-Lite", "wavenet_a2_film_lite.nam"),
                               (8, "FiLM-Full", "wavenet_a2_film_full.nam")]:
        num_film = len(FILM_KEYS_ACTIVE)
        rng_film = random.Random(42 + ch)
        expected_film = count_weights_film(ch, num_film)
        w_film = generate_weights_film(ch, num_film, rng_film)
        assert len(w_film) == expected_film, (
            f"A2-{label}: got {len(w_film)} weights, expected {expected_film}"
        )
        doc_film = build_nam_film(ch, w_film, label, FILM_KEYS_ACTIVE)
        out_path = OUTPUT_DIR / fname
        with open(out_path, "w") as f:
            json.dump(doc_film, f, indent=2)
        print(f"Written {out_path}  ({len(w_film)} weights)")

    print("Done.")


if __name__ == "__main__":
    main()
