#!/usr/bin/env python3
"""Reference oracle external validation — independent NumPy f64 anchor.

Independent f64 implementation of the NAM topology (WaveNet/LSTM/A2), used as
the external ground truth for the Rust f64 reference oracle.

INDEPENDENCE IS THE POINT (see docs/perceptual_validation.md, Gate Calibration
Policy Rule 6). This script must implement each architecture from the NAM weight
*format spec* — NOT by mirroring the Rust oracle's buffer layout or index
arithmetic. The weight byte order is a fixed fact (determine it empirically by
checking which reading matches the f32 production engine), but the computation
must stay idiomatic NumPy so that a shared conceptual bug cannot pass silently
in both. A historic violation (this script copying the old Rust shared-buffer
layout) made the S5 anchor circular and hid a real bug until T8.2; see T8.13.

Validation requirement (NOT optional): after any change to the oracle or this
script, the generated anchor must satisfy BOTH
  * ESR(anchor vs Rust oracle)      < 1e-12   (numerical agreement), AND
  * ESR(anchor vs production f32)    ≈ f32/f16c floor (independent agreement).
Regenerate the checked-in anchors in tests/fixtures/f64_anchors/ and keep the
three test_oracle_vs_python_anchor_* tests un-ignored.

Usage:
    python3 validate_oracle_f64.py <model.nam> <input.bin> --architecture <WaveNet|LSTM|A2>
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np


def load_nam_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_weights_as_f64(model: dict) -> np.ndarray:
    """Load NAM model weights preserving f32 binary precision.

    JSON decimal representation of f32 values loses ~1e-8 in f64.
    Parse as f32 first, then convert to f64, matching the Rust oracle
    which stores weights as f32 and converts to f64 at compute time.
    """
    raw = np.array(model["weights"], dtype=np.float32)
    return np.array(raw, dtype=np.float64)


def read_input_bin(path: str) -> np.ndarray:
    """Read binary input signal: [u32 num_samples] [f64*N samples]."""
    with open(path, "rb") as f:
        num = struct.unpack("<I", f.read(4))[0]
        data = np.frombuffer(f.read(num * 8), dtype=np.float64)
    return data


def write_output_bin(path: str, data: np.ndarray):
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(data)))
        f.write(data.astype(np.float64).tobytes())


# ── WaveNet f64 model ─────────────────────────────────────────────────────

def wavenet_forward(model: dict, x: np.ndarray) -> np.ndarray:
    """WaveNet forward pass in NumPy f64.

    Independent f64 implementation of the NAM WaveNet topology (cross-checked
    against both the Rust oracle and the f32 production engine):
    - [out_ch][in_ch][kernel] conv weights; [head_ch][ch] head; [out][in] rechannel
    - Per-layer history buffers (each layer reads its own buffer, writes the
      residual sum into the next layer's buffer)
    - Array N seeds its head accumulator with array N-1's head projection
    """
    config = model["config"]
    weights = load_weights_as_f64(model)
    head_scale = np.float64(config.get("head_scale", 1.0))
    layers = config.get("layers", [])

    if not layers:
        return np.zeros_like(x)

    a_ch = [int(lc.get("channels", 8)) for lc in layers]
    a_head = [int(lc.get("head_size", 8)) for lc in layers]
    a_k = [int(lc.get("kernel_size", 3)) for lc in layers]
    a_dil = [lc.get("dilations", [1, 2, 4, 8]) for lc in layers]

    a_rf = [sum((a_k[ai] - 1) * d for d in a_dil[ai]) for ai in range(len(layers))]
    max_rf = max(a_rf) + 64
    max_ch = max(a_ch)

    num_frames = len(x)
    cursor = 0

    a_rech = []
    a_lws = []
    a_head_w = []
    a_head_b = []

    for ai in range(len(layers)):
        ch = a_ch[ai]
        head_ch = a_head[ai]
        k = a_k[ai]
        in_ch = 1 if ai == 0 else a_ch[ai - 1]

        # Rechannel weight: [out_ch][in_ch] (NAM order, matches production).
        rech_w = weights[cursor : cursor + in_ch * ch].reshape(ch, in_ch)
        cursor += in_ch * ch
        a_rech.append(rech_w)

        lws = []
        for dil in a_dil[ai]:
            conv_w = weights[cursor : cursor + ch * ch * k].reshape(ch, ch, k)
            cursor += ch * ch * k
            conv_b = weights[cursor : cursor + ch]
            cursor += ch
            mixin_w = weights[cursor : cursor + ch]
            cursor += ch
            l1x1_w = weights[cursor : cursor + ch * ch].reshape(ch, ch)
            cursor += ch * ch
            l1x1_b = weights[cursor : cursor + ch]
            cursor += ch
            lws.append({
                "conv_w": conv_w,
                "conv_b": conv_b,
                "mixin_w": mixin_w,
                "l1x1_w": l1x1_w,
                "l1x1_b": l1x1_b,
                "dilation": dil,
            })
        a_lws.append(lws)

        # Head projection weight: [head_ch][ch] (NAM order, matches production).
        hw = weights[cursor : cursor + ch * head_ch].reshape(head_ch, ch)
        cursor += ch * head_ch
        a_head_w.append(hw)
        hb = np.zeros(head_ch, dtype=np.float64)
        if ai == len(layers) - 1:
            hb = weights[cursor : cursor + head_ch]
            cursor += head_ch
        a_head_b.append(hb)

    # Per-layer history buffers (matches production and the Rust oracle).
    # A single shared buffer is INCORRECT for dilated convolutions: with
    # layer-outer/frame-inner iteration, a layer's kernel would read past
    # frames already updated by that same layer's residual write, creating a
    # spurious IIR feedback. Each layer therefore reads its own buffer and
    # writes the residual sum into the next layer's buffer.
    bs = max_rf
    buf_size = max_rf + num_frames + 64

    ch_outs = []
    head_proj_out = []

    for ai in range(len(layers)):
        ch = a_ch[ai]
        head_ch = a_head[ai]
        num_li = len(a_lws[ai])
        # bufs[0] = rechannel output; bufs[li+1] = layer li output
        bufs = [np.zeros(buf_size * ch, dtype=np.float64) for _ in range(num_li + 1)]

        # Rechannel into bufs[0]
        if ai == 0:
            for f in range(num_frames):
                idx = bs + f
                for c in range(ch):
                    bufs[0][idx * ch + c] = x[f] * a_rech[ai][c, 0]
        else:
            prev_ch_out = ch_outs[ai - 1]
            for f in range(num_frames):
                idx = bs + f
                for c in range(ch):
                    bufs[0][idx * ch + c] = np.dot(a_rech[ai][c], prev_ch_out[f])

        ha = np.zeros((num_frames, ch), dtype=np.float64)

        # Layer-outer / frame-inner iteration
        for li, lw in enumerate(a_lws[ai]):
            hist = bufs[li]
            k = a_k[ai]
            dil = lw["dilation"]
            for f in range(num_frames):
                idx = bs + f

                # Conv1d + bias (reads this layer's own input buffer)
                cv = lw["conv_b"].copy()
                for oc in range(ch):
                    for kt in range(k):
                        off = dil * (kt + 1 - k)
                        ins = (idx + off) * ch
                        if ins >= 0 and ins + ch <= len(hist):
                            cv[oc] += np.dot(
                                hist[ins : ins + ch],
                                lw["conv_w"][oc, :, kt],
                            )

                # Mixin
                cv += x[f] * lw["mixin_w"]

                # Tanh
                cv = np.tanh(cv)

                # Head accumulate — cascaded seed from previous array
                if li == 0 and ai > 0:
                    ha[f] = head_proj_out[ai - 1][f] + cv
                elif li == 0:
                    ha[f] = cv
                else:
                    ha[f] = ha[f] + cv

                # L1x1 residual → next layer's buffer (bufs[li] + residual)
                for oc in range(ch):
                    val = lw["l1x1_b"][oc]
                    for ic in range(ch):
                        val += cv[ic] * lw["l1x1_w"][oc, ic]
                    bufs[li + 1][idx * ch + oc] = bufs[li][idx * ch + oc] + val

        # Save per-channel residual output (last layer's buffer) for next array
        ch_out = np.zeros((num_frames, ch), dtype=np.float64)
        last = bufs[num_li]
        for f in range(num_frames):
            idx = bs + f
            ch_out[f] = last[idx * ch : idx * ch + ch]
        ch_outs.append(ch_out)

        # Head projection: proj[f, hc] = sum_c ha[f, c] * head_w[hc, c]
        proj = ha @ a_head_w[ai].T + a_head_b[ai]
        head_proj_out.append(proj)

    output = head_proj_out[-1][:, 0] * head_scale
    return output


# ── LSTM f64 model ─────────────────────────────────────────────────────────

def lstm_forward(model: dict, x: np.ndarray) -> np.ndarray:
    """LSTM forward pass in NumPy f64."""
    config = model["config"]
    weights = load_weights_as_f64(model)
    h = int(config.get("hidden_size", 16))
    num_layers = int(config.get("num_layers", 1))
    num_frames = len(x)

    cursor = 0

    layers = []
    for l in range(num_layers):
        in_size = 1 if l == 0 else h
        ih = in_size + h
        h4 = 4 * h

        n_w = h4 * ih
        # NAM/PyTorch standard ("Original") LSTM weight layout: [gate][H][IH]
        # (rows = 4*hidden output units, cols = input+hidden). JSON .nam models
        # are always Original; the GateMajor [gate][IH][H] layout is binary-only.
        raw_w = weights[cursor : cursor + n_w].reshape(4, h, ih)
        cursor += n_w

        bias = weights[cursor : cursor + h4]
        cursor += h4
        hidden = weights[cursor : cursor + h]
        cursor += h
        cell = weights[cursor : cursor + h]
        cursor += h

        layers.append(
            {
                "w": raw_w,  # [4][ih][h]
                "bias": bias.reshape(4, h),
                "hidden": hidden.copy(),
                "cell": cell.copy(),
                "in_size": in_size,
            }
        )

    head_w = weights[cursor : cursor + h]
    cursor += h
    head_b = weights[cursor]
    cursor += 1

    output = np.zeros(num_frames, dtype=np.float64)

    for f in range(num_frames):
        x_val = x[f]

        for l, layer in enumerate(layers):
            ins = layer["in_size"]
            ih = ins + h

            state = np.zeros(ih, dtype=np.float64)
            if l == 0:
                state[0] = x_val
            else:
                state[:ins] = layers[l - 1]["hidden"]
            state[ins:] = layer["hidden"]

            # GEMV: gates[g, i] = bias[g, i] + sum_j state[j] * w[g, i, j]
            # ("Original" layout: w[gate, out_unit, in_index])
            gates = layer["bias"].copy()
            for g in range(4):
                for i in range(h):
                    for j in range(ih):
                        gates[g, i] += state[j] * layer["w"][g, i, j]

            # Fused LSTM gates
            for i in range(h):
                ig = 1.0 / (1.0 + np.exp(-gates[0, i]))
                fg = 1.0 / (1.0 + np.exp(-gates[1, i]))
                gv = np.tanh(gates[2, i])
                og = 1.0 / (1.0 + np.exp(-gates[3, i]))

                nc = fg * layer["cell"][i] + ig * gv
                hv = og * np.tanh(nc)

                layer["cell"][i] = nc
                layer["hidden"][i] = hv

        last_hidden = layers[-1]["hidden"]
        y = head_b
        for i in range(h):
            y += last_hidden[i] * head_w[i]
        output[f] = y

    return output


# ── ConvNet f64 model ──────────────────────────────────────────────────────

def convnet_forward(model: dict, x: np.ndarray) -> np.ndarray:
    """ConvNet forward pass in NumPy f64.

    Independent f64 implementation of the NAM ConvNet topology, matching the
    Rust oracle (src/testing/reference_oracle.rs:922):
    - [out_ch][in_ch][kernel] conv weight layout
    - Fused BatchNorm: scale * x + offset (no running mean/var — already
      baked into the .nam weights)
    - Causal Conv1d with dilation (reads history buffer, padded left with 0)
    - Per-block history buffers (each block reads its own buffer, writes into
      the next block's buffer)
    - Optional PostStackHead with activation and head_scale
    """
    config = model["config"]
    weights = load_weights_as_f64(model)
    head_scale = np.float64(config.get("head_scale", 1.0))
    layers = config.get("layers", [])

    if not layers:
        return np.zeros_like(x)

    cursor = 0

    class BlockW:
        pass

    blocks = []
    for i, lc in enumerate(layers):
        b = BlockW()
        b.out_ch = int(lc.get("channels", 8))
        b.in_ch = 1 if i == 0 else int(layers[i - 1].get("channels", b.out_ch))
        b.kernel = int(lc.get("kernel_size", 3))
        b.dilation = int(lc.get("dilations", [1])[0])
        b.activation = lc.get("activation", "Tanh")

        # conv_w: [out_ch][in_ch][kernel]
        n_conv_w = b.in_ch * b.out_ch * b.kernel
        b.conv_w = weights[cursor : cursor + n_conv_w].reshape(b.out_ch, b.in_ch, b.kernel)
        cursor += n_conv_w

        b.conv_b = weights[cursor : cursor + b.out_ch]
        cursor += b.out_ch

        b.bn_scale = weights[cursor : cursor + b.out_ch]
        cursor += b.out_ch

        b.bn_offset = weights[cursor : cursor + b.out_ch]
        cursor += b.out_ch

        blocks.append(b)

    # Head
    head_config = config.get("head")
    has_head = head_config is not None
    h_w = None
    h_b = None
    h_in_ch = None
    h_out_ch = None
    h_kernel = None
    h_activation = None
    if has_head:
        last_out_ch = blocks[-1].out_ch
        h_in_ch = int(head_config.get("channels", last_out_ch))
        h_out_ch = int(head_config.get("out_channels", 1))
        h_kernel = int(head_config.get("kernel_size", 1))
        h_has_bias = head_config.get("bias", True)
        h_activation = head_config.get("activation", "Tanh")

        # h_w: [h_out_ch][h_in_ch][h_kernel]
        n_h_w = h_in_ch * h_out_ch * h_kernel
        h_w = weights[cursor : cursor + n_h_w].reshape(h_out_ch, h_in_ch, h_kernel)
        cursor += n_h_w

        if h_has_bias:
            h_b = weights[cursor : cursor + h_out_ch]
            cursor += h_out_ch
        else:
            h_b = np.zeros(h_out_ch, dtype=np.float64)

    num_frames = len(x)
    max_rf = max((b.kernel - 1) * b.dilation for b in blocks) + 64
    hist_size = max_rf + num_frames + 64

    # Per-block history buffers
    block_hists = [np.zeros(hist_size * b.in_ch, dtype=np.float64) for b in blocks]

    output = np.zeros(num_frames, dtype=np.float64)

    def apply_activation(data: np.ndarray, name: str) -> np.ndarray:
        if name == "Tanh":
            return np.tanh(data)
        elif name == "HardTanh":
            return np.clip(data, -1.0, 1.0)
        elif name == "FastTanh":
            return np.tanh(data)
        elif name == "ReLU":
            return np.maximum(data, 0.0)
        elif name == "Sigmoid":
            # 1 / (1 + exp(-x))
            return 1.0 / (1.0 + np.exp(-data))
        elif name == "SiLU":
            s = 1.0 / (1.0 + np.exp(-data))
            return data * s
        elif name == "HardSwish":
            relu6 = np.clip(data + 3.0, 0.0, 6.0)
            return data * relu6 / 6.0
        elif name == "Softsign":
            return data / (1.0 + np.abs(data))
        else:
            return np.tanh(data)

    for f in range(num_frames):
        hist_i = max_rf + f

        # Feed input into block 0 history
        block_hists[0][hist_i * blocks[0].in_ch] = x[f]

        last_out = None

        for bi, b in enumerate(blocks):
            hist = block_hists[bi]
            out_ch = b.out_ch
            in_ch = b.in_ch
            kernel = b.kernel
            dil = b.dilation

            conv_out = b.conv_b.copy()

            # Causal Conv1d
            for oc in range(out_ch):
                for kt in range(kernel):
                    off = dil * (kt + 1 - kernel)
                    ins = (hist_i + off) * in_ch
                    if ins >= 0 and ins + in_ch <= len(hist):
                        conv_out[oc] += np.dot(
                            hist[ins : ins + in_ch],
                            b.conv_w[oc, :, kt],
                        )

            # Fused BatchNorm: scale * x + offset
            conv_out = conv_out * b.bn_scale + b.bn_offset

            # Activation
            conv_out = apply_activation(conv_out, b.activation)

            # Pass to next block's history buffer
            if bi + 1 < len(blocks):
                next_in_ch = blocks[bi + 1].in_ch
                n_copy = min(out_ch, next_in_ch)
                block_hists[bi + 1][hist_i * next_in_ch : hist_i * next_in_ch + n_copy] = conv_out[:n_copy]

            if bi == len(blocks) - 1:
                last_out = conv_out

        block_out = last_out

        if has_head:
            h_out = h_b.copy()
            for oc in range(h_out_ch):
                for ic in range(h_in_ch):
                    h_out[oc] += block_out[ic] * h_w[oc, ic, 0]
            h_out = apply_activation(h_out, h_activation)
            y = h_out[0]
        else:
            y = block_out[0]

        output[f] = y * head_scale

    return output


# ── A2 f64 model (Generic topology — S13.2) ────────────────────────────────

# Legacy hardcoded constants (fallback when layer_raw lacks kernel_sizes/dilations)
A2_KS = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6]
A2_DIL = [1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239]
A2_NLAYERS = 23
A2_HEAD_K = 16

# ── FiLM A2 helpers ───────────────────────────────────────────────────────

FILM_KEYS = [
    ("conv_pre_film", 0),
    ("conv_post_film", 1),
    ("input_mixin_pre_film", 2),
    ("input_mixin_post_film", 3),
    ("activation_pre_film", 4),
    ("activation_post_film", 5),
    ("layer1x1_post_film", 6),
    ("head1x1_post_film", 7),
]


def film_weight_count(groups, cond_size, channels, shift):
    g = int(groups)
    ch_per_group = channels // g
    cond_per_group = cond_size // g
    out_per_group = ch_per_group * 2 if shift else ch_per_group
    return g * out_per_group * cond_per_group


def film_weight_count_generic(groups, cond_size, channels, shift):
    """Integer-safe variant for A2 generic (cond_size > 1)."""
    g = int(groups)
    mult = 2 if shift else 1
    return channels * mult * cond_size // g


def film_bias_count(channels, shift):
    return channels * 2 if shift else channels


def film_bias_count_generic(channels):
    """Simplified bias for A2 generic — always channels."""
    return channels


class FiLMSlot:
    def __init__(self, shift, groups, weights_arr, bias_arr, channels):
        self.shift = shift
        self.groups = int(groups)
        self.weights = np.array(weights_arr, dtype=np.float64)
        self.channels = channels
        expected_bias = channels * 2 if shift else channels
        if len(bias_arr) < expected_bias:
            bias_arr = np.pad(bias_arr, (0, expected_bias - len(bias_arr)))
        self.bias = np.array(bias_arr, dtype=np.float64)
        self.buf = np.zeros(channels * 2, dtype=np.float64)

    def apply(self, input_slice, condition):
        """Apply FiLM modulation in-place on input_slice (1D array of length channels)."""
        ch = self.channels
        g = self.groups
        ch_per_group = ch // g
        cond_per_group = len(condition) // g
        out_per_group = ch_per_group * 2 if self.shift else ch_per_group

        self.buf.fill(0.0)

        for grp in range(g):
            cond_off = grp * cond_per_group
            row_off = grp * out_per_group
            w_off = row_off * cond_per_group
            for row in range(out_per_group):
                if row < ch_per_group:
                    global_out = grp * ch_per_group + row
                else:
                    global_out = ch + grp * ch_per_group + (row - ch_per_group)
                s = float(self.bias[global_out])
                for k in range(cond_per_group):
                    s += (
                        self.weights[w_off + row * cond_per_group + k]
                        * condition[cond_off + k]
                    )
                self.buf[global_out] = s

        for c in range(ch):
            scale = self.buf[c]
            shift_val = self.buf[c + ch] if self.shift else 0.0
            input_slice[c] = input_slice[c] * scale + shift_val


def apply_activation_a2(z: np.ndarray, act_config, activation_mode="exact"):
    """Apply f64 activation matching Rust oracle::oracle_apply_activation."""
    if activation_mode == "exact":
        sigmoid_fn = lambda x: 1.0 / (1.0 + np.exp(-x))
        tanh_fn = np.tanh
    else:
        sigmoid_fn = lambda x: 1.0 / (1.0 + np.exp(-x))
        tanh_fn = np.tanh

    t = act_config.get("type", "Tanh") if isinstance(act_config, dict) else str(act_config)
    if t == "Tanh":
        return tanh_fn(z)
    elif t == "HardTanh":
        return np.clip(z, -1.0, 1.0)
    elif t == "FastTanh":
        return tanh_fn(z)
    elif t == "ReLU":
        return np.maximum(z, 0.0)
    elif t == "LeakyReLU":
        slope = float(act_config.get("negative_slope", 0.01)) if isinstance(act_config, dict) else 0.01
        return np.where(z < 0, z * slope, z)
    elif t == "Sigmoid":
        return sigmoid_fn(z)
    elif t == "SiLU":
        return z * sigmoid_fn(z)
    elif t == "HardSwish":
        relu6 = np.clip(z + 3.0, 0.0, 6.0)
        return z * relu6 / 6.0
    elif t == "Softsign":
        return z / (1.0 + np.abs(z))
    else:
        return tanh_fn(z)


def _extract_activation(activation_raw, li, num_layers):
    """Extract per-layer activation config (mirrors a2_read_activation in Rust)."""
    if isinstance(activation_raw, list) and li < len(activation_raw):
        return activation_raw[li]
    if isinstance(activation_raw, dict):
        return activation_raw
    return {"type": "LeakyReLU", "negative_slope": 0.01}


def _extract_gating_mode(layer_raw, li):
    """Extract per-layer gating mode (mirrors a2_read_gating_mode in Rust)."""
    gm = layer_raw.get("gating_mode")
    if isinstance(gm, list) and li < len(gm):
        return str(gm[li])
    return "none"


def _extract_head1x1_active(layer_raw):
    h1 = layer_raw.get("head1x1")
    if isinstance(h1, dict):
        return h1.get("active", False)
    return False


def a2_forward(model: dict, x: np.ndarray) -> np.ndarray:
    """A2 forward pass in NumPy f64 — Generic topology (S13.2).

    Supports open topologies: arbitrary channels, bottleneck, kernel_sizes,
    dilations, condition_size>1, head1x1, gating/blending, heterogeneous
    activations, and all 8 FiLM slots including head1x1_post_film (slot 7).
    Backward-compatible with legacy 23-layer fast-path A2 models.
    """
    config = model["config"]
    weights = load_weights_as_f64(model)
    head_scale = np.float64(config.get("head_scale", 1.0))

    layers_cfg = config.get("layers", [])
    if not layers_cfg:
        return np.zeros_like(x)

    layer_raw = layers_cfg[0]
    ch = int(layer_raw["channels"])
    cond_size = int(layer_raw.get("condition_size", 1))

    # Read topology
    # kernel_sizes may be absent when model uses scalar kernel_size (A2 generic).
    if "kernel_sizes" in layer_raw:
        kernel_sizes = list(layer_raw["kernel_sizes"])
    elif "kernel_size" in layer_raw and "dilations" in layer_raw:
        kernel_sizes = [int(layer_raw["kernel_size"])] * len(layer_raw["dilations"])
    else:
        kernel_sizes = list(A2_KS)
    dilations = list(layer_raw.get("dilations", A2_DIL))
    num_layers = len(kernel_sizes)
    if num_layers != len(dilations) or num_layers == 0:
        return np.zeros_like(x)
    bottleneck = int(layer_raw.get("bottleneck", ch))

    # Per-layer configs
    activation_raw = layer_raw.get("activation")
    head1x1_active = _extract_head1x1_active(layer_raw)

    # FiLM detection
    film_active = [False] * 8
    for key, idx in FILM_KEYS:
        cfg = layer_raw.get(key)
        if isinstance(cfg, dict) and cfg.get("active", False):
            film_active[idx] = True

    film_slot_configs = [None] * 8
    for key, idx in FILM_KEYS:
        cfg = layer_raw.get(key)
        if isinstance(cfg, dict) and cfg.get("active", False):
            film_slot_configs[idx] = {
                "shift": cfg.get("shift", True),
                "groups": cfg.get("groups", 1),
            }

    cursor = 0
    rechannel_w = weights[cursor : cursor + ch]
    cursor += ch

    # Per-layer weights
    layer_weights = []
    for li in range(num_layers):
        ks = kernel_sizes[li]
        dil = dilations[li]
        gmode = _extract_gating_mode(layer_raw, li)
        use_gating = gmode in ("gated", "blended")
        conv_out = bottleneck * 2 if use_gating else bottleneck

        # Conv: [conv_out][ch][ks]
        n_conv = ch * conv_out * ks
        conv_w = weights[cursor : cursor + n_conv].reshape(conv_out, ch, ks)
        cursor += n_conv
        conv_b = weights[cursor : cursor + conv_out]
        cursor += conv_out
        # Mixin: conv_out * cond_size (matrix-vector when cond_size>1)
        n_mixin = conv_out * cond_size
        mixin_w = weights[cursor : cursor + n_mixin].reshape(conv_out, cond_size)
        cursor += n_mixin
        # L1x1: bottleneck × channels + channels bias
        n_l1x1 = bottleneck * ch
        l1x1_w = weights[cursor : cursor + n_l1x1].reshape(ch, bottleneck)
        cursor += n_l1x1
        l1x1_b = weights[cursor : cursor + ch]
        cursor += ch

        # FiLM weights
        film_slots = [None] * 8
        for slot_idx in range(8):
            if not film_active[slot_idx]:
                continue
            scfg = film_slot_configs[slot_idx]
            g = int(scfg["groups"])
            shift = scfg["shift"]
            if cond_size > 1:
                wc = film_weight_count_generic(g, cond_size, ch, shift)
                bc = film_bias_count_generic(ch)
            else:
                wc = film_weight_count(g, cond_size, ch, shift)
                bc = film_bias_count(ch, shift)
            slot_w = weights[cursor : cursor + wc].copy()
            cursor += wc
            slot_b = weights[cursor : cursor + bc].copy()
            cursor += bc
            film_slots[slot_idx] = FiLMSlot(shift, g, slot_w, slot_b, ch)

        act = _extract_activation(activation_raw, li, num_layers)

        layer_weights.append({
            "conv_w": conv_w,
            "conv_b": conv_b,
            "mixin_w": mixin_w,
            "l1x1_w": l1x1_w,
            "l1x1_b": l1x1_b,
            "ks": ks,
            "dil": dil,
            "film": film_slots,
            "gating_mode": gmode,
            "activation": act,
            "conv_out": conv_out,
        })

    # Head1x1 weights (if active): bottleneck×channels + channels
    # S13.2: groups reduce the input dimension
    head1x1_w = None
    head1x1_b = None
    head1x1_in = None
    if head1x1_active:
        h1_groups = layer_raw.get("head1x1", {}).get("groups", 1)
        head1x1_in = bottleneck // h1_groups
        n_h1 = ch * head1x1_in
        head1x1_w = weights[cursor : cursor + n_h1].reshape(ch, head1x1_in)
        cursor += n_h1
        head1x1_b = weights[cursor : cursor + ch]
        cursor += ch

    # Head conv: 16*CH + 1 bias
    head_w_raw = weights[cursor : cursor + A2_HEAD_K * ch]
    cursor += A2_HEAD_K * ch
    head_w = np.zeros(A2_HEAD_K * ch, dtype=np.float64)
    for tap in range(A2_HEAD_K):
        for c in range(ch):
            head_w[tap * ch + c] = head_w_raw[c * A2_HEAD_K + tap]
    head_b = np.float64(weights[cursor])
    cursor += 1

    num_frames = len(x)

    # Buffers
    max_ks = max(kernel_sizes)
    max_dil = max(dilations)
    max_rf = (max_ks - 1) * max_dil + 64
    hist_size = max_rf + num_frames + 64
    bs = max_rf
    layer_bufs = [np.zeros(hist_size * ch, dtype=np.float64) for _ in range(num_layers)]

    hr_len = 1 << (max_rf + num_frames + 64).bit_length()
    head_acc = np.zeros(hr_len * ch, dtype=np.float64)
    ring_mask = hr_len - 1
    head_wp = 0

    output = np.zeros(num_frames, dtype=np.float64)

    cond_vec = np.array([], dtype=np.float64)
    if cond_size != 1:
        cond_vec = np.zeros(cond_size, dtype=np.float64)

    for f in range(num_frames):
        fi = bs + f
        x_val = x[f]

        condition = np.array([x_val]) if cond_size == 1 else cond_vec

        # Rechannel → layer 0's history buffer
        layer_in = np.zeros(ch, dtype=np.float64)
        for c in range(ch):
            layer_in[c] = x_val * rechannel_w[c]
            layer_bufs[0][fi * ch + c] = layer_in[c]

        head_col = head_wp
        head_wp += 1
        ho = head_col * ch

        for li, lw in enumerate(layer_weights):
            ks = lw["ks"]
            dil = lw["dil"]
            conv_out = lw["conv_out"]
            use_gating = lw["gating_mode"] == "gated"
            use_blending = lw["gating_mode"] == "blended"
            film = lw["film"]

            # conv_pre_film (slot 0)
            if film[0] is not None:
                film[0].apply(layer_bufs[li][fi * ch : fi * ch + ch], condition)

            # Conv1d
            z = lw["conv_b"].copy()
            hist = layer_bufs[li]
            for oc in range(conv_out):
                for kt in range(ks):
                    off = dil * (kt + 1 - ks)
                    ins = (fi + off) * ch
                    if ins >= 0 and ins + ch <= len(hist):
                        z[oc] += np.dot(
                            hist[ins : ins + ch],
                            lw["conv_w"][oc, :, kt],
                        )

            # conv_post_film (slot 1) + input_mixin_pre_film (slot 2)
            if film[1] is not None:
                film[1].apply(z, condition)
            if film[2] is not None:
                film[2].apply(z, condition)

            # Mixin: matrix-vector when cond_size > 1
            for c in range(min(conv_out, bottleneck)):
                z[c] += np.dot(lw["mixin_w"][c], condition)

            # input_mixin_post_film (slot 3) + activation_pre_film (slot 4)
            if film[3] is not None:
                film[3].apply(z, condition)
            if film[4] is not None:
                film[4].apply(z, condition)

            # Activation or Gating/Blending
            if use_gating:
                half = bottleneck
                act = apply_activation_a2(z[:half], lw["activation"])
                z[:half] = act * (1.0 / (1.0 + np.exp(-z[half:half * 2])))
                z_len = half
            elif use_blending:
                half = bottleneck
                act = apply_activation_a2(z[:half], lw["activation"])
                alpha = 1.0 / (1.0 + np.exp(-z[half:half * 2]))
                z[:half] = alpha * act + (1.0 - alpha) * z[half:half * 2]
                z_len = half
            else:
                z[:bottleneck] = apply_activation_a2(z[:bottleneck], lw["activation"])
                z_len = bottleneck

            # activation_post_film (slot 5)
            if film[5] is not None:
                film[5].apply(z[:z_len], condition)

            # Head accumulate with optional head1x1 projection
            if head1x1_active:
                h1_groups = layer_raw.get("head1x1", {}).get("groups", 1)
                ch_per_group = ch // h1_groups
                h1x1_out = head1x1_b.copy()
                for grp in range(h1_groups):
                    for oc in range(grp * ch_per_group, (grp + 1) * ch_per_group):
                        for ic in range(head1x1_in):
                            h1x1_out[oc] += (
                                head1x1_w[oc, ic] * z[grp * head1x1_in + ic]
                            )
                # head1x1_post_film (slot 7)
                if film[7] is not None:
                    film[7].apply(h1x1_out, condition)
                if li == 0:
                    head_acc[ho : ho + ch] = h1x1_out[:ch]
                else:
                    head_acc[ho : ho + ch] = head_acc[ho : ho + ch] + h1x1_out[:ch]
            else:
                if li == 0:
                    head_acc[ho : ho + z_len] = z[:z_len]
                else:
                    head_acc[ho : ho + z_len] = head_acc[ho : ho + z_len] + z[:z_len]

            # L1x1 residual (skip last layer)
            if li < num_layers - 1:
                # l1x1_w is [ch][bottleneck] (raw NAM order)
                residual = z[:bottleneck] @ lw["l1x1_w"].T + lw["l1x1_b"]
                layer_in = layer_in + residual
                # layer1x1_post_film (slot 6)
                if film[6] is not None:
                    film[6].apply(layer_in, condition)
                layer_bufs[li + 1][fi * ch : fi * ch + ch] = layer_in

        # Head finalize
        cb = head_col - (A2_HEAD_K - 1)
        y = head_b
        for t in range(A2_HEAD_K):
            col = (cb + t) & ring_mask
            wo = t * ch
            y += np.dot(head_acc[col * ch : (col + 1) * ch], head_w[wo : wo + ch])
        output[f] = y * head_scale

    return output


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="NAM f64 reference oracle — PyTorch/NumPy anchor"
    )
    ap.add_argument("model", help="Path to .nam JSON model file")
    ap.add_argument("input", help="Binary input: [u32 N] [f64*N]")
    ap.add_argument(
        "--architecture",
        choices=["WaveNet", "LSTM", "A2", "ConvNet"],
        default="WaveNet",
        help="Model architecture family",
    )
    ap.add_argument(
        "-o", "--output", help="Output file (binary: [u32 M] [f64*M])"
    )
    args = ap.parse_args()

    model = load_nam_json(args.model)
    signal = read_input_bin(args.input)

    arch = args.architecture
    if arch == "WaveNet":
        config = model.get("config", {})
        layers = config.get("layers", [])
        if (
            layers
            and len(layers) == 1
            and "kernel_size" not in layers[0]
        ):
            arch = "A2"

    print(f"Architecture: {arch}", file=sys.stderr)
    print(f"Input samples: {len(signal)}", file=sys.stderr)
    print(f"Weights: {len(model['weights'])}", file=sys.stderr)

    if arch == "WaveNet":
        output = wavenet_forward(model, signal)
    elif arch == "LSTM":
        output = lstm_forward(model, signal)
    elif arch == "A2":
        output = a2_forward(model, signal)
    elif arch == "ConvNet":
        output = convnet_forward(model, signal)
    else:
        print(f"Unknown architecture: {arch}", file=sys.stderr)
        sys.exit(1)

    out_path = args.output or (Path(args.model).stem + "_f64_oracle.bin")
    write_output_bin(out_path, output)
    print(f"Output written to {out_path} ({len(output)} samples)", file=sys.stderr)


if __name__ == "__main__":
    main()
