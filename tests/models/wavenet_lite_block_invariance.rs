// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::common;
use common::{compute_mse, generate_sine_440hz};

use nam_rs::loader::dispatcher::wavenet::select_interleave_width;
use nam_rs::loader::dispatcher::wavenet::transpose_conv1d_interleaved_4wide;
use nam_rs::loader::dispatcher::wavenet::transpose_conv1d_interleaved_8wide;
use nam_rs::loader::dispatcher::wavenet::transpose_conv1d_interleaved_16wide;
use nam_rs::math::common::AlignedVec;
use nam_rs::models::wavenet::{
    Conv1d, DenseLayer, WAVENET_MAX_NUM_FRAMES, WaveNetLayer, WaveNetLayerArray, WaveNetLayerState,
    WaveNetModel,
};

fn test_dense_f32(in_ch: usize, out_ch: usize) -> AlignedVec<f32> {
    AlignedVec::from_vec(vec![0.01f32; out_ch * in_ch])
        .expect("allocation should succeed for test-sized buffers")
}

fn make_conv_weights(in_ch: usize, out_ch: usize, k: usize) -> AlignedVec<f32> {
    let raw_weights = vec![0.01f32; out_ch * k * in_ch];
    let interleave_width = select_interleave_width(out_ch);
    let num_blocks = out_ch.div_ceil(interleave_width);
    let padded_total = num_blocks * interleave_width * in_ch * k;
    let mut weights = AlignedVec::new(padded_total, 0.0)
        .expect("allocation should succeed for test-sized buffers");
    match interleave_width {
        16 => {
            transpose_conv1d_interleaved_16wide(&raw_weights, &mut weights, in_ch, out_ch, k);
        }
        8 => {
            transpose_conv1d_interleaved_8wide(&raw_weights, &mut weights, in_ch, out_ch, k);
        }
        _ => {
            transpose_conv1d_interleaved_4wide(&raw_weights, &mut weights, in_ch, out_ch, k);
        }
    }
    weights
}

fn build_tiny_lite_wavenet() -> WaveNetModel<12, 3, 6> {
    let make_layer_a1 = |dilation: usize| -> WaveNetLayer<1, 12, 3> {
        WaveNetLayer {
            conv1d: Conv1d {
                weights: make_conv_weights(12, 12, 3),
                bias: AlignedVec::from_vec(vec![0.0; 12])
                    .expect("allocation should succeed for test-sized buffers"),
                do_bias: false,
                dilation,
            },
            input_mixin: DenseLayer {
                weights: test_dense_f32(1, 12),
                bias: AlignedVec::from_vec(vec![0.0; 12])
                    .expect("allocation should succeed for test-sized buffers"),
                do_bias: false,
            },
            one_by_one: DenseLayer {
                weights: test_dense_f32(12, 12),
                bias: AlignedVec::from_vec(vec![0.0; 12])
                    .expect("allocation should succeed for test-sized buffers"),
                do_bias: false,
            },
            scratch_mixin: AlignedVec::new(16 * WAVENET_MAX_NUM_FRAMES, 0.0f32)
                .expect("scratch alloc"),
            scratch_conv: AlignedVec::new(16 * WAVENET_MAX_NUM_FRAMES, 0.0f32)
                .expect("scratch alloc"),
        }
    };

    let make_layer_a2 = |dilation: usize| -> WaveNetLayer<1, 6, 3> {
        WaveNetLayer {
            conv1d: Conv1d {
                weights: make_conv_weights(6, 6, 3),
                bias: AlignedVec::from_vec(vec![0.0; 6])
                    .expect("allocation should succeed for test-sized buffers"),
                do_bias: false,
                dilation,
            },
            input_mixin: DenseLayer {
                weights: test_dense_f32(1, 6),
                bias: AlignedVec::from_vec(vec![0.0; 6])
                    .expect("allocation should succeed for test-sized buffers"),
                do_bias: false,
            },
            one_by_one: DenseLayer {
                weights: test_dense_f32(6, 6),
                bias: AlignedVec::from_vec(vec![0.0; 6])
                    .expect("allocation should succeed for test-sized buffers"),
                do_bias: false,
            },
            scratch_mixin: AlignedVec::new(6 * WAVENET_MAX_NUM_FRAMES, 0.0f32)
                .expect("scratch alloc"),
            scratch_conv: AlignedVec::new(6 * WAVENET_MAX_NUM_FRAMES, 0.0f32)
                .expect("scratch alloc"),
        }
    };

    let dilations_1: Vec<usize> = (0..10).map(|i| 1 << i).collect();
    let dilations_2: Vec<usize> = (0..10).map(|i| 1 << i).collect();

    let rf1: usize = dilations_1.iter().map(|&d| (3 - 1) * d).sum();
    let rf2: usize = dilations_2.iter().map(|&d| (3 - 1) * d).sum();

    let layers_1: Vec<WaveNetLayer<1, 12, 3>> =
        dilations_1.iter().map(|&d| make_layer_a1(d)).collect();
    let states_1: Vec<WaveNetLayerState> = (0..layers_1.len())
        .map(|i| {
            WaveNetLayerState::new(16, (3 - 1) * dilations_1[i], i)
                .expect("Failed to create WaveNetLayerState")
        })
        .collect();

    let num_layers_1 = layers_1.len();
    let array1 = WaveNetLayerArray::<1, 1, 12, 3, 6> {
        layers: layers_1,
        states: states_1,
        effective_layers: num_layers_1,
        rechannel: DenseLayer {
            weights: test_dense_f32(1, 12),
            bias: AlignedVec::from_vec(vec![0.0; 12])
                .expect("allocation should succeed for test-sized buffers"),
            do_bias: false,
        },
        head_rechannel: DenseLayer {
            weights: test_dense_f32(12, 6),
            bias: AlignedVec::from_vec(vec![0.0; 6])
                .expect("allocation should succeed for test-sized buffers"),
            do_bias: false,
        },
        array_outputs: AlignedVec::from_vec(vec![0.0; 16 * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        head_accum: AlignedVec::from_vec(vec![0.0; 16 * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        head_outputs: AlignedVec::from_vec(vec![0.0; 6 * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        receptive_field_size: rf1,
        block_size: 16,
        block_buffer: AlignedVec::from_vec(vec![0.0; 16 * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
    };

    let layers_2: Vec<WaveNetLayer<1, 6, 3>> =
        dilations_2.iter().map(|&d| make_layer_a2(d)).collect();
    let states_2: Vec<WaveNetLayerState> = (0..layers_2.len())
        .map(|i| {
            WaveNetLayerState::new(6, (3 - 1) * dilations_2[i], i)
                .expect("Failed to create WaveNetLayerState")
        })
        .collect();

    let num_layers_2 = layers_2.len();
    let array2 = WaveNetLayerArray::<12, 1, 6, 3, 1> {
        layers: layers_2,
        states: states_2,
        effective_layers: num_layers_2,
        rechannel: DenseLayer {
            weights: test_dense_f32(12, 6),
            bias: AlignedVec::from_vec(vec![0.0; 6])
                .expect("allocation should succeed for test-sized buffers"),
            do_bias: false,
        },
        head_rechannel: DenseLayer {
            weights: test_dense_f32(6, 1),
            bias: AlignedVec::from_vec(vec![0.0; 1])
                .expect("allocation should succeed for test-sized buffers"),
            do_bias: true,
        },
        array_outputs: AlignedVec::from_vec(vec![0.0; 6 * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        head_accum: AlignedVec::from_vec(vec![0.0; 6 * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        head_outputs: AlignedVec::from_vec(vec![0.0; WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        receptive_field_size: rf2,
        block_size: 6,
        block_buffer: AlignedVec::from_vec(vec![0.0; 6 * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
    };

    WaveNetModel {
        array1,
        array2,
        head_scale: 0.02,
        receptive_field_size: rf1.max(rf2),
        prewarm_on_reset: true,
    }
}

#[test]
// T1.2 fix: MirroredBuffer::new_aligned guarantees size_elements % channels == 0,
// so the ring-buffer wrap period aligns exactly with the mirror period.
// T1.3: hardened to assert! < 1e-7 (synthetic model gives MSE~1e-20 post-fix).
fn test_wavenet_lite_block_invariance() {
    let num_samples = 16384;
    let input = generate_sine_440hz(num_samples);

    let mut ref_model = build_tiny_lite_wavenet();
    ref_model.prewarm();
    let mut ref_output = vec![0.0f32; num_samples];
    process_in_blocks_lite(&mut ref_model, &input, &mut ref_output, 1);

    let block_sizes = [16, 32, 64];

    for &bs in &block_sizes {
        let mut test_model = build_tiny_lite_wavenet();
        test_model.prewarm();
        let mut test_output = vec![0.0f32; num_samples];
        process_in_blocks_lite(&mut test_model, &input, &mut test_output, bs);

        let test_mse = compute_mse(&ref_output, &test_output);

        eprintln!("P1 block invariance: Lite CH=12 bs=1 vs bs={bs} MSE={test_mse:.6e}");

        assert!(
            test_mse < 1e-7,
            "Block invariance violated (CH=12 ring-buffer desync): bs={bs} MSE={test_mse:.6e}"
        );
    }
}

fn process_in_blocks_lite(
    model: &mut WaveNetModel<12, 3, 6>,
    input: &[f32],
    output: &mut [f32],
    block_size: usize,
) {
    let total = input.len();
    let mut pos = 0;
    while pos < total {
        let end = (pos + block_size).min(total);
        model.process(&input[pos..end], &mut output[pos..end]);
        pos = end;
    }
}

// =============================================================================
// T6.S3.3 — Prop-test: padding lanes (12-15) never become NaN/Inf/subnormal
// =============================================================================

/// Verify that lanes 12-15 of a stride-16 buffer are exactly 0.0 for
/// `num_frames` frames. Returns the first violation as `Err((frame, lane, value))`.
fn check_padding_lanes_zero(
    buf: &[f32],
    stride: usize,
    num_frames: usize,
    _label: &str,
) -> Result<(), (usize, usize, f32)> {
    for f in 0..num_frames {
        let base = f * stride;
        for lane in 12..16 {
            let val = buf[base + lane];
            if val != 0.0f32 {
                return Err((f, lane, val));
            }
        }
    }
    Ok(())
}

/// Deterministic LCG for generating pseudo-random test data without external crates.
#[inline]
fn next_lcg(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    let x = (*state >> 33) as f32 * (1.0f32 / (1u32 << 31) as f32);
    (x * 2.0) - 1.0
}

#[test]
fn test_wavenet_lite_padding_lanes_zero() {
    let mut model = build_tiny_lite_wavenet();
    model.prewarm();

    const STRIDE: usize = 16; // effective_stride(12)
    const NUM_BLOCKS: usize = 100;
    let mut lcg_state: u64 = 0xDEAD_BEEF_CAFE_BABE;

    for block_idx in 0..NUM_BLOCKS {
        let num_frames = ((lcg_state & 0x3F) as usize) + 1; // 1..=64
        lcg_state = lcg_state.wrapping_mul(6364136223846793005).wrapping_add(1);

        let input: Vec<f32> = (0..num_frames).map(|_| next_lcg(&mut lcg_state)).collect();
        let mut output = vec![0.0f32; num_frames];

        let saved_starts: Vec<usize> = model.array1.states.iter().map(|s| s.buffer_start).collect();

        model.process(&input, &mut output);

        for (layer_idx, layer) in model.array1.layers.iter().enumerate() {
            check_padding_lanes_zero(
                layer.scratch_mixin.as_ref(),
                STRIDE,
                num_frames,
                &format!("array1.layer[{layer_idx}].scratch_mixin"),
            )
            .unwrap_or_else(|(f, l, v)| {
                panic!(
                    "array1.layer[{layer_idx}].scratch_mixin frame {f} lane {l} = {v:e}, \
                     expected 0.0 after block {block_idx}"
                )
            });

            check_padding_lanes_zero(
                layer.scratch_conv.as_ref(),
                STRIDE,
                num_frames,
                &format!("array1.layer[{layer_idx}].scratch_conv"),
            )
            .unwrap_or_else(|(f, l, v)| {
                panic!(
                    "array1.layer[{layer_idx}].scratch_conv frame {f} lane {l} = {v:e}, \
                     expected 0.0 after block {block_idx}"
                )
            });

            let lb_start = saved_starts[layer_idx] * STRIDE;
            check_padding_lanes_zero(
                &model.array1.states[layer_idx].layer_buffer
                    [lb_start..lb_start + num_frames * STRIDE],
                STRIDE,
                num_frames,
                &format!("array1.layer[{layer_idx}].layer_buffer"),
            )
            .unwrap_or_else(|(f, l, v)| {
                panic!(
                    "array1.layer[{layer_idx}].layer_buffer frame {f} lane {l} = {v:e}, \
                     expected 0.0 after block {block_idx}"
                )
            });
        }

        check_padding_lanes_zero(
            &model.array1.array_outputs[..num_frames * STRIDE],
            STRIDE,
            num_frames,
            "array1.array_outputs",
        )
        .unwrap_or_else(|(f, l, v)| {
            panic!(
                "array1.array_outputs frame {f} lane {l} = {v:e}, \
                 expected 0.0 after block {block_idx}"
            )
        });

        check_padding_lanes_zero(
            &model.array1.head_accum[..num_frames * STRIDE],
            STRIDE,
            num_frames,
            "array1.head_accum",
        )
        .unwrap_or_else(|(f, l, v)| {
            panic!(
                "array1.head_accum frame {f} lane {l} = {v:e}, \
                 expected 0.0 after block {block_idx}"
            )
        });
    }
}

#[cfg(test)]
mod padding_lanes_proptest {
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig {
            failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::Off)),
            .. ProptestConfig::with_cases(128)
        })]
        #[test]
        #[ignore]
        fn prop_wavenet_lite_padding_lanes_zero(
            num_blocks in 1usize..=64usize,
        ) {
            let mut model = super::build_tiny_lite_wavenet();
            model.prewarm();

            const STRIDE: usize = 16;
            let mut lcg_state: u64 = 0xB16B00B5_13579753;

            for block_idx in 0..num_blocks {
                let num_frames = ((lcg_state & 0x3F) as usize) + 1;
                lcg_state = lcg_state.wrapping_mul(6364136223846793005).wrapping_add(1);

                let input: Vec<f32> = (0..num_frames)
                    .map(|_| super::next_lcg(&mut lcg_state))
                    .collect();
                let mut output = vec![0.0f32; num_frames];

                let saved_starts: Vec<usize> =
                    model.array1.states.iter().map(|s| s.buffer_start).collect();

                model.process(&input, &mut output);

                for (layer_idx, layer) in model.array1.layers.iter().enumerate() {
                    super::check_padding_lanes_zero(
                        layer.scratch_mixin.as_ref(),
                        STRIDE,
                        num_frames,
                        &format!("[proptest] array1.layer[{layer_idx}].scratch_mixin"),
                    )
                    .unwrap_or_else(|(f, l, v)| {
                        panic!(
                            "array1.layer[{layer_idx}].scratch_mixin frame {f} lane {l} = {v:e}, \
                             expected 0.0 after block {block_idx} (num_blocks={num_blocks})"
                        )
                    });

                    super::check_padding_lanes_zero(
                        layer.scratch_conv.as_ref(),
                        STRIDE,
                        num_frames,
                        &format!("[proptest] array1.layer[{layer_idx}].scratch_conv"),
                    )
                    .unwrap_or_else(|(f, l, v)| {
                        panic!(
                            "array1.layer[{layer_idx}].scratch_conv frame {f} lane {l} = {v:e}, \
                             expected 0.0 after block {block_idx} (num_blocks={num_blocks})"
                        )
                    });

                    let lb_start = saved_starts[layer_idx] * STRIDE;
                    super::check_padding_lanes_zero(
                        &model.array1.states[layer_idx].layer_buffer
                            [lb_start..lb_start + num_frames * STRIDE],
                        STRIDE,
                        num_frames,
                        &format!("[proptest] array1.layer[{layer_idx}].layer_buffer"),
                    )
                    .unwrap_or_else(|(f, l, v)| {
                        panic!(
                            "array1.layer[{layer_idx}].layer_buffer frame {f} lane {l} = {v:e}, \
                             expected 0.0 after block {block_idx} (num_blocks={num_blocks})"
                        )
                    });
                }

                super::check_padding_lanes_zero(
                    &model.array1.array_outputs[..num_frames * STRIDE],
                    STRIDE,
                    num_frames,
                    "[proptest] array1.array_outputs",
                )
                .unwrap_or_else(|(f, l, v)| {
                    panic!(
                        "array1.array_outputs frame {f} lane {l} = {v:e}, \
                         expected 0.0 after block {block_idx} (num_blocks={num_blocks})"
                    )
                });

                super::check_padding_lanes_zero(
                    &model.array1.head_accum[..num_frames * STRIDE],
                    STRIDE,
                    num_frames,
                    "[proptest] array1.head_accum",
                )
                .unwrap_or_else(|(f, l, v)| {
                    panic!(
                        "array1.head_accum frame {f} lane {l} = {v:e}, \
                         expected 0.0 after block {block_idx} (num_blocks={num_blocks})"
                    )
                });
            }
        }
    }
}
