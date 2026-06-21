#!/usr/bin/env python3
"""
Deterministic fixture generator for Task B.1.2 — ConvNet, WaveNetDyn, LstmDyn.

Produces:
  convnet_test.nam     — 2-block ConvNet (CH=8,4; Tanh; no post-stack head)
  wavenet_dyn_free.nam — Free-shape WaveNet (CH=7,4; non-catalog geometry)
  lstm_dyn_test.nam    — LSTM 1x7 (uncatalogued hidden_size → dynamic path)

All models use deterministic PRNG seeds and calibrated weight scales
for realistic audio amplitude regime.

SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
"""

import json
import random
from pathlib import Path
from typing import List

OUTPUT_DIR = Path(__file__).resolve().parent / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Helpers
# =============================================================================


def gen_weights(n: int, rng: random.Random, scale: float) -> List[float]:
    return [rng.uniform(-1.0, 1.0) * scale for _ in range(n)]


# =============================================================================
# 1. ConvNet
#
# NOTE: C++ golden generation NOT currently possible for ConvNet.
# The NAM 0.5.4 ConvNet uses a multi-block architecture with per-block channels,
# kernel_size>2, and a `layers`-based JSON config. The C++ render tool (NAM Core
# v0.5.3) implements a different ConvNet: single `channels` shared across blocks,
# flat `dilations`, fixed kernel_size=2, and `batchnorm` flag. These are
# architecturally incompatible — the C++ render will crash with a JSON type_error.
#
# Golden vectors for ConvNet must be generated through a NAM 0.5.4+ render pipeline
# or via a future C++ Core upgrade. For now, this fixture serves Rust-only validation.
# =============================================================================

CONVNET_SEED = 77
CONVNET_CHANNELS = [8, 4]
CONVNET_KERNEL = 3
CONVNET_DILATIONS = [[1, 2, 4], [1, 2, 4]]
CONVNET_ACTIVATION = "Tanh"
CONVNET_HEAD_SCALE = 0.02

# Weight scales per block (conv_w, conv_b, bn_scale, bn_offset)
CONVNET_SCALES = {
    8: {"conv_w": 0.20, "conv_b": 0.04, "bn_s": 0.98, "bn_o": 0.02},
    4: {"conv_w": 0.25, "conv_b": 0.05, "bn_s": 0.98, "bn_o": 0.02},
}


def count_convnet_weights() -> int:
    count = 0
    for i, ch in enumerate(CONVNET_CHANNELS):
        in_ch = 1 if i == 0 else CONVNET_CHANNELS[i - 1]
        count += ch * in_ch * CONVNET_KERNEL  # conv_w
        count += ch  # conv_b
        count += ch  # bn_scale
        count += ch  # bn_offset
    count += 1  # head_scale
    return count


def generate_convnet_weights(rng: random.Random) -> List[float]:
    weights: List[float] = []
    for i, ch in enumerate(CONVNET_CHANNELS):
        in_ch = 1 if i == 0 else CONVNET_CHANNELS[i - 1]
        sc = CONVNET_SCALES[ch]
        # Conv1D weights: [OUT][KERNEL][IN] raw row-major (original layout)
        weights.extend(gen_weights(ch * in_ch * CONVNET_KERNEL, rng, sc["conv_w"]))
        # Conv1D bias
        weights.extend(gen_weights(ch, rng, sc["conv_b"]))
        # BatchNorm scale (near 1.0, low variance)
        weights.extend([max(0.9, min(1.1, rng.uniform(0.97, 1.03))) for _ in range(ch)])
        # BatchNorm offset (near 0.0)
        weights.extend(gen_weights(ch, rng, sc["bn_o"]))
    weights.append(CONVNET_HEAD_SCALE)
    return weights


def build_convnet_nam(weights: List[float]) -> dict:
    return {
        "version": "0.5.4",
        "architecture": "ConvNet",
        "config": {
            "layers": [
                {
                    "channels": CONVNET_CHANNELS[i],
                    "kernel_size": CONVNET_KERNEL,
                    "dilations": CONVNET_DILATIONS[i],
                    "activation": CONVNET_ACTIVATION,
                }
                for i in range(len(CONVNET_CHANNELS))
            ],
            "head_scale": CONVNET_HEAD_SCALE,
        },
        "weights": weights,
        "metadata": {
            "name": "ConvNet Test Fixture (2 blocks, CH=8→4)",
            "modeled_by": "tests/fixtures/generate_b1_2_fixtures.py",
        },
        "sample_rate": 48000,
    }


# =============================================================================
# 2. WaveNetDyn (Free-shape, non-catalog geometry)
# =============================================================================

WAVENET_DYN_SEED = 123
WAVENET_DYN_ARRAY_CH = [7, 4]
WAVENET_DYN_KERNEL = 3
WAVENET_DYN_DILATIONS = [[1, 2, 4], [8, 16]]
WAVENET_DYN_HEAD_SIZES = [4, 1]
WAVENET_DYN_CONDITION_SIZE = 1
WAVENET_DYN_HEAD_SCALE = 0.02

WAVENET_DYN_SCALES = {
    7: {"weight": 0.22, "bias": 0.05},
    4: {"weight": 0.28, "bias": 0.06},
}


def count_wavenet_dyn_weights() -> int:
    count = 0
    for i, ch in enumerate(WAVENET_DYN_ARRAY_CH):
        in_ch = 1 if i == 0 else WAVENET_DYN_ARRAY_CH[i - 1]
        head = WAVENET_DYN_HEAD_SIZES[i]
        cond = WAVENET_DYN_CONDITION_SIZE
        dilations = WAVENET_DYN_DILATIONS[i]

        # rechannel.w (no bias)
        count += in_ch * ch
        for _ in dilations:
            count += ch * ch * WAVENET_DYN_KERNEL  # conv_w
            count += ch  # conv_b
            count += cond * ch  # input_mixin.w (no bias)
            count += ch * ch  # one_by_one.w
            count += ch  # one_by_one.b
        # head_rechannel.w
        count += ch * head
        # head_rechannel.b only for last array
        if i == len(WAVENET_DYN_ARRAY_CH) - 1:
            count += head
    count += 1  # head_scale
    return count


def generate_wavenet_dyn_weights(rng: random.Random) -> List[float]:
    weights: List[float] = []
    cond = WAVENET_DYN_CONDITION_SIZE

    for i, ch in enumerate(WAVENET_DYN_ARRAY_CH):
        in_ch = 1 if i == 0 else WAVENET_DYN_ARRAY_CH[i - 1]
        head = WAVENET_DYN_HEAD_SIZES[i]
        dilations = WAVENET_DYN_DILATIONS[i]
        sc = WAVENET_DYN_SCALES[ch]
        ws = sc["weight"]
        bs = sc["bias"]

        # rechannel.w: [OUT][IN] raw row-major → read as OUT*IN floats
        weights.extend(gen_weights(in_ch * ch, rng, scale=ws))

        for _ in dilations:
            # conv1d.w: [OUT][KERNEL][IN] raw row-major
            weights.extend(gen_weights(ch * ch * WAVENET_DYN_KERNEL, rng, scale=ws))
            # conv1d.b
            weights.extend(gen_weights(ch, rng, scale=bs))
            # input_mixin.w: [OUT][IN] raw row-major, IN=cond
            weights.extend(gen_weights(cond * ch, rng, scale=ws))
            # one_by_one.w: [OUT][IN] raw row-major
            weights.extend(gen_weights(ch * ch, rng, scale=ws))
            # one_by_one.b
            weights.extend(gen_weights(ch, rng, scale=bs))

        # head_rechannel.w: [OUT][IN] raw row-major
        weights.extend(gen_weights(ch * head, rng, scale=ws))
        # head_rechannel.b only for last array
        if i == len(WAVENET_DYN_ARRAY_CH) - 1:
            weights.extend(gen_weights(head, rng, scale=bs))

    weights.append(WAVENET_DYN_HEAD_SCALE)
    return weights


def build_wavenet_dyn_nam(weights: List[float]) -> dict:
    ch0 = WAVENET_DYN_ARRAY_CH[0]
    ch1 = WAVENET_DYN_ARRAY_CH[1]
    return {
        "version": "0.5.4",
        "architecture": "WaveNet",
        "config": {
            "layers": [
                {
                    "input_size": 1,
                    "condition_size": WAVENET_DYN_CONDITION_SIZE,
                    "head_size": WAVENET_DYN_HEAD_SIZES[0],
                    "channels": ch0,
                    "kernel_size": WAVENET_DYN_KERNEL,
                    "dilations": WAVENET_DYN_DILATIONS[0],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "input_size": ch0,
                    "condition_size": WAVENET_DYN_CONDITION_SIZE,
                    "head_size": WAVENET_DYN_HEAD_SIZES[1],
                    "channels": ch1,
                    "kernel_size": WAVENET_DYN_KERNEL,
                    "dilations": WAVENET_DYN_DILATIONS[1],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": True,
                },
            ],
            "head_scale": WAVENET_DYN_HEAD_SCALE,
        },
        "weights": weights,
        "metadata": {
            "name": f"WaveNetDyn Free-Shape Fixture (CH={ch0}/{ch1})",
            "modeled_by": "tests/fixtures/generate_b1_2_fixtures.py",
        },
        "sample_rate": 48000,
    }


# =============================================================================
# 3. LstmDyn (uncatalogued hidden_size → dynamic path)
# =============================================================================

LSTM_DYN_SEED = 456
LSTM_DYN_NUM_LAYERS = 1
LSTM_DYN_HIDDEN_SIZE = 7  # not in catalog: 3,8,12,16,24,40

LSTM_DYN_SCALE = 0.08


def count_lstm_dyn_weights() -> int:
    h = LSTM_DYN_HIDDEN_SIZE
    n = LSTM_DYN_NUM_LAYERS
    count = 0
    for i in range(n):
        inp = 1 if i == 0 else h
        ih = inp + h
        count += 4 * h * ih  # input_hidden_weights
        count += 4 * h  # bias
        count += h  # initial_hidden_state
        count += h  # initial_cell_state
    count += h  # head_weights
    count += 1  # head_bias
    return count


def generate_lstm_dyn_weights(rng: random.Random) -> List[float]:
    h = LSTM_DYN_HIDDEN_SIZE
    n = LSTM_DYN_NUM_LAYERS
    sc = LSTM_DYN_SCALE
    weights: List[float] = []

    for i in range(n):
        inp = 1 if i == 0 else h
        ih = inp + h
        # input_hidden_weights: [Gate][H][IH] original layout
        weights.extend(gen_weights(4 * h * ih, rng, scale=sc))
        # bias
        weights.extend(gen_weights(4 * h, rng, scale=sc))
        # initial_hidden_state
        weights.extend(gen_weights(h, rng, scale=sc * 0.1))
        # initial_cell_state
        weights.extend(gen_weights(h, rng, scale=sc * 0.1))

    # head_weights
    weights.extend(gen_weights(h, rng, scale=sc))
    # head_bias
    weights.extend(gen_weights(1, rng, scale=sc))

    return weights


def build_lstm_dyn_nam(weights: List[float]) -> dict:
    return {
        "version": "0.5.4",
        "architecture": "LSTM",
        "config": {
            "num_layers": LSTM_DYN_NUM_LAYERS,
            "hidden_size": LSTM_DYN_HIDDEN_SIZE,
            "input_size": 1,
        },
        "weights": weights,
        "metadata": {
            "name": f"LSTM-Dyn Test Fixture ({LSTM_DYN_NUM_LAYERS}x{LSTM_DYN_HIDDEN_SIZE})",
            "modeled_by": "tests/fixtures/generate_b1_2_fixtures.py",
        },
        "sample_rate": 48000,
    }


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    # --- ConvNet ---
    rng_convnet = random.Random(CONVNET_SEED)
    expected_convnet = count_convnet_weights()
    w_convnet = generate_convnet_weights(rng_convnet)
    assert len(w_convnet) == expected_convnet, (
        f"ConvNet: got {len(w_convnet)} weights, expected {expected_convnet}"
    )
    doc = build_convnet_nam(w_convnet)
    out = OUTPUT_DIR / "convnet_test.nam"
    with open(out, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"Written {out}  ({len(w_convnet)} weights)")

    # --- WaveNetDyn ---
    rng_wdyn = random.Random(WAVENET_DYN_SEED)
    expected_wdyn = count_wavenet_dyn_weights()
    w_wdyn = generate_wavenet_dyn_weights(rng_wdyn)
    assert len(w_wdyn) == expected_wdyn, (
        f"WaveNetDyn: got {len(w_wdyn)} weights, expected {expected_wdyn}"
    )
    doc = build_wavenet_dyn_nam(w_wdyn)
    out = OUTPUT_DIR / "wavenet_dyn_free.nam"
    with open(out, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"Written {out}  ({len(w_wdyn)} weights)")

    # --- LstmDyn ---
    rng_lstm = random.Random(LSTM_DYN_SEED)
    expected_lstm = count_lstm_dyn_weights()
    w_lstm = generate_lstm_dyn_weights(rng_lstm)
    assert len(w_lstm) == expected_lstm, (
        f"LSTM-Dyn: got {len(w_lstm)} weights, expected {expected_lstm}"
    )
    doc = build_lstm_dyn_nam(w_lstm)
    out = OUTPUT_DIR / "lstm_dyn_test.nam"
    with open(out, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"Written {out}  ({len(w_lstm)} weights)")

    print("Done. All B.1.2 fixtures generated.")


if __name__ == "__main__":
    main()
