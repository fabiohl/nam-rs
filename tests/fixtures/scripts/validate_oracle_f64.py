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


# ── A2 f64 model ───────────────────────────────────────────────────────────

A2_KS = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6]
A2_DIL = [1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239]
A2_NLAYERS = 23
A2_HEAD_K = 16


def a2_forward(model: dict, x: np.ndarray) -> np.ndarray:
    """A2 forward pass in NumPy f64.

    Matches the Rust oracle:
    - [out_ch][in_ch][kernel] conv weight layout
    - Head_w transposed from NAM JSON [channel][tap] to [tap][channel]
    - Linear head accumulator (no ring modulo during writes)
    - K=16 causal convolution for head finalize
    """
    config = model["config"]
    weights = load_weights_as_f64(model)
    head_scale = np.float64(config.get("head_scale", 1.0))

    layers_cfg = config.get("layers", [])
    if not layers_cfg:
        return np.zeros_like(x)

    ch = int(layers_cfg[0]["channels"])
    cursor = 0

    # Rechannel
    rechannel_w = weights[cursor : cursor + ch]
    cursor += ch

    # Per-layer weights
    layer_weights = []
    for li in range(A2_NLAYERS):
        ks = A2_KS[li]
        dil = A2_DIL[li]

        n_conv = ch * ch * ks
        conv_w = weights[cursor : cursor + n_conv].reshape(ch, ch, ks)
        cursor += n_conv
        conv_b = weights[cursor : cursor + ch]
        cursor += ch
        mixin_w = weights[cursor : cursor + ch]
        cursor += ch
        n_l1x1 = ch * ch
        l1x1_w = weights[cursor : cursor + n_l1x1].reshape(ch, ch)
        cursor += n_l1x1
        l1x1_b = weights[cursor : cursor + ch]
        cursor += ch

        layer_weights.append(
            {
                "conv_w": conv_w,
                "conv_b": conv_b,
                "mixin_w": mixin_w,
                "l1x1_w": l1x1_w,
                "l1x1_b": l1x1_b,
                "ks": ks,
                "dil": dil,
            }
        )

    # Head weights — transpose from [channel][tap] to [tap][channel]
    head_w_raw = weights[cursor : cursor + A2_HEAD_K * ch]
    cursor += A2_HEAD_K * ch
    head_w = np.zeros(A2_HEAD_K * ch, dtype=np.float64)
    for tap in range(A2_HEAD_K):
        for c in range(ch):
            head_w[tap * ch + c] = head_w_raw[c * A2_HEAD_K + tap]
    head_b = np.float64(weights[cursor])
    cursor += 1

    num_frames = len(x)
    max_ks = max(A2_KS)
    max_dil = max(A2_DIL)
    max_rf = (max_ks - 1) * max_dil + 64
    hist_size = max_rf + num_frames + 64
    bs = max_rf
    # Per-layer history buffers (matches production and the Rust oracle): each
    # layer reads its own buffer and writes its residual output into the next
    # layer's buffer. A single shared buffer is incorrect (see WaveNet note).
    layer_bufs = [np.zeros(hist_size * ch, dtype=np.float64) for _ in range(A2_NLAYERS)]

    # Head accumulator — flat array (no ring modulo during writes)
    hr_len = 1 << (max_rf + num_frames + 64).bit_length()
    head_acc = np.zeros(hr_len * ch, dtype=np.float64)
    ring_mask = hr_len - 1
    head_wp = 0

    output = np.zeros(num_frames, dtype=np.float64)

    for f in range(num_frames):
        fi = bs + f
        x_val = x[f]

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
            hist = layer_bufs[li]

            # Conv1d + bias (reads this layer's own history buffer)
            z = lw["conv_b"].copy()
            for oc in range(ch):
                for kt in range(ks):
                    off = dil * (kt + 1 - ks)
                    ins = (fi + off) * ch
                    if ins >= 0 and ins + ch <= len(hist):
                        z[oc] += np.dot(
                            hist[ins : ins + ch],
                            lw["conv_w"][oc, :, kt],
                        )

            # Mixin
            z += lw["mixin_w"] * x_val

            # LeakyReLU(0.01)
            z = np.where(z < 0, z * 0.01, z)

            # Head accumulate — linear write (no modulo), matches Rust oracle
            if li == 0:
                head_acc[ho : ho + ch] = z[:ch]
            else:
                head_acc[ho : ho + ch] += z[:ch]

            # L1x1 residual (skip last): next = layer_in + l1x1(z),
            # written into the NEXT layer's history buffer.
            if li < A2_NLAYERS - 1:
                residual = z @ lw["l1x1_w"].T + lw["l1x1_b"]
                layer_in = layer_in + residual
                layer_bufs[li + 1][fi * ch : fi * ch + ch] = layer_in

        # Head finalize — K=16 causal convolution
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
