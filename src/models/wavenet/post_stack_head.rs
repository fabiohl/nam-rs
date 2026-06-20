// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::math::common::AlignedVec;
use crate::math::common::SimdMath;
use crate::models::a2::activations::ActivationFn;
use crate::models::a2::activations::ActivationType;

use super::common::{WAVENET_MAX_NUM_FRAMES, WaveNetLayerState};
use super::conv1d_dyn::Conv1dDyn;
use crate::loader::nam_json::model::HeadConfig;

/// Post-stack head sub-object for WaveNet / ConvNet architectures.
///
/// Contains a causal Conv1D + activation that processes the signal
/// after the stack of layer arrays, before the final `head_scale` gain.
/// Mirrors the `_Head` structure in NAMCore's `convnet.h:108-118`.
#[repr(align(64))]
pub struct PostStackHead {
    /// Causal 1D convolution (dynamic runtime dimensions).
    pub conv: Conv1dDyn,
    /// Activation function applied after convolution.
    pub activation: ActivationType,
    /// Ring buffer state for causal convolution lookback.
    pub state: WaveNetLayerState,
    /// Scratch buffer for convolution output (out_ch * WAVENET_MAX_NUM_FRAMES).
    scratch: AlignedVec<f32>,
}

impl PostStackHead {
    /// Creates a new `PostStackHead` from the parsed `HeadConfig` and the
    /// input channel count from the last layer array.
    ///
    /// Missing fields in `HeadConfig` fall back to sensible defaults:
    /// - `channels` → `in_channels` (same as the last array's head projection)
    /// - `out_channels` → 1 (mono output)
    /// - `kernel_size` → 3
    /// - `bias` → false
    /// - `activation` → "Tanh"
    ///
    /// Weight and bias arrays are zero-initialized and must be populated
    /// by the dispatcher via `set_weights` and `set_bias`.
    pub fn from_config(config: &HeadConfig, in_channels: usize) -> std::io::Result<Self> {
        let channels = config.channels.unwrap_or(in_channels);
        let out_channels = config.out_channels.unwrap_or(1);
        let kernel = config.kernel_size.unwrap_or(3);
        let do_bias = config.bias.unwrap_or(false);
        let activation = parse_activation(config.activation.as_deref().unwrap_or("Tanh"));

        let num_blocks = out_channels.div_ceil(4);
        let weights_len = num_blocks * kernel * channels * 4;
        let bias_len = out_channels;

        let weights = AlignedVec::new(weights_len, 0.0f32);
        let bias = AlignedVec::new(bias_len, 0.0f32);

        let receptive_field = kernel;
        let state = WaveNetLayerState::new(channels, receptive_field, 0)?;

        let conv = Conv1dDyn {
            weights,
            bias,
            do_bias,
            dilation: 1,
            in_ch: channels,
            out_ch: out_channels,
            num_blocks,
            kernel,
            prefetch_fn: crate::math::common::prefetch_strategy_simple,
        };

        let scratch = AlignedVec::new(out_channels * WAVENET_MAX_NUM_FRAMES, 0.0f32);

        Ok(Self {
            conv,
            activation,
            state,
            scratch,
        })
    }

    /// Returns the receptive field contribution of this head (kernel size).
    /// Must be added to the global model receptive field for prewarm.
    pub fn receptive_field(&self) -> usize {
        self.conv.kernel
    }

    /// Number of output channels produced by this head.
    pub fn out_channels(&self) -> usize {
        self.conv.out_ch
    }

    /// Number of input channels expected by this head.
    pub fn in_channels(&self) -> usize {
        self.conv.in_ch
    }

    /// Loads convolution weights from a flat f32 slice.
    pub fn set_weights(&mut self, weights: &[f32]) {
        let len = self.conv.weights.len().min(weights.len());
        self.conv.weights[..len].copy_from_slice(&weights[..len]);
    }

    /// Loads convolution bias from a flat f32 slice, if present.
    pub fn set_bias(&mut self, bias: &[f32]) {
        let len = self.conv.bias.len().min(bias.len());
        self.conv.bias[..len].copy_from_slice(&bias[..len]);
    }

    /// Public dispatch wrapper that selects the optimal SIMD path.
    ///
    /// # Safety
    /// Input and output slices must have sizes compatible with the head dimensions:
    /// `input.len() == num_frames * in_ch`, `output.len() == num_frames * out_ch`.
    /// The ring buffer state must have been properly initialized (via `prewarm` or
    /// sufficient prior processing) to cover the causal receptive field.
    #[inline(always)]
    pub unsafe fn process_block(&mut self, input: &[f32], output: &mut [f32], num_frames: usize) {
        unsafe {
            crate::math::common::dispatch_simd!(
                self,
                process_block_internal,
                input,
                output,
                num_frames
            )
        };
    }

    /// SIMD-dispatched processing kernel.
    ///
    /// Writes `num_frames` of input into the ring buffer, runs the causal
    /// Conv1D, applies activation, and writes results to output.
    ///
    /// Input layout: frame-interleaved `[f0_c0, f0_c1, ..., f1_c0, ...]`.
    /// Output layout: frame-interleaved `[f0_c0, f0_c1, ..., f1_c0, ...]`.
    ///
    /// # Safety
    /// `input` and `output` must have sizes `num_frames * in_ch` and
    /// `num_frames * out_ch` respectively. The ring buffer must have been
    /// properly initialized.
    #[inline(always)]
    pub unsafe fn process_block_internal<M: SimdMath>(
        &mut self,
        input: &[f32],
        output: &mut [f32],
        num_frames: usize,
    ) {
        let in_ch = self.conv.in_ch;
        let out_ch = self.conv.out_ch;
        let input_len = num_frames * in_ch;

        let buf_start = self.state.buffer_start * in_ch;
        self.state.layer_buffer[buf_start..buf_start + input_len]
            .copy_from_slice(&input[..input_len]);

        let scratch_slice = &mut self.scratch[..num_frames * out_ch];
        unsafe {
            self.conv.process_block(
                &self.state.layer_buffer,
                scratch_slice,
                self.state.buffer_start,
                num_frames,
                None,
            );
        }

        self.activation.apply(scratch_slice);

        output[..num_frames * out_ch].copy_from_slice(scratch_slice);

        self.state.advance_frames(num_frames, in_ch);
    }

    /// Public prewarm wrapper with SIMD dispatch.
    #[cold]
    pub fn prewarm(&mut self) {
        unsafe {
            crate::math::common::dispatch_simd!(self, prewarm_internal);
        }
    }

    /// Fills the conv state buffer with a single frame of silence replicated
    /// backward to cover the entire receptive field.
    ///
    /// # Safety
    /// Must be called via `dispatch_simd!` macro. The state buffer must be
    /// properly allocated and the ring buffer start pointer must be valid.
    #[inline(always)]
    pub unsafe fn prewarm_internal<M: SimdMath>(&mut self) {
        let in_ch = self.conv.in_ch;
        let out_ch = self.conv.out_ch;
        let kernel = self.conv.kernel;

        let buf_start = self.state.buffer_start * in_ch;

        self.state.layer_buffer[buf_start..buf_start + in_ch].fill(0.0);

        let start_idx = self.state.buffer_start * in_ch;
        let src_range = start_idx..start_idx + in_ch;
        for offset in 1..=kernel {
            let dst_idx = (self.state.buffer_start - offset) * in_ch;
            self.state
                .layer_buffer
                .copy_within(src_range.clone(), dst_idx);
        }

        let scratch_slice = &mut self.scratch[..out_ch];
        unsafe {
            self.conv.process_single_frame(
                &self.state.layer_buffer,
                scratch_slice,
                self.state.buffer_start,
                None,
            );
        }
        self.activation.apply(scratch_slice);

        self.state.advance_frames(1, in_ch);
    }
}

/// Maps an activation function name string to an `ActivationType`.
///
/// Supported values match the variant names of `ActivationType`:
/// `"Tanh"`, `"HardTanh"`, `"FastTanh"`, `"ReLU"`, `"Sigmoid"`,
/// `"SiLU"`, `"HardSwish"`, `"Softsign"`.
///
/// Unrecognized strings fall back to `ActivationType::Tanh`.
pub fn parse_activation(name: &str) -> ActivationType {
    match name {
        "Tanh" => ActivationType::Tanh,
        "HardTanh" => ActivationType::HardTanh,
        "FastTanh" => ActivationType::FastTanh,
        "ReLU" => ActivationType::ReLU,
        "Sigmoid" => ActivationType::Sigmoid,
        "SiLU" => ActivationType::SiLU,
        "HardSwish" => ActivationType::HardSwish,
        "Softsign" => ActivationType::Softsign,
        _ => ActivationType::Tanh,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::nam_json::model::HeadConfig;

    #[test]
    fn test_parse_activation_known_names() {
        assert_eq!(parse_activation("Tanh"), ActivationType::Tanh);
        assert_eq!(parse_activation("HardTanh"), ActivationType::HardTanh);
        assert_eq!(parse_activation("FastTanh"), ActivationType::FastTanh);
        assert_eq!(parse_activation("ReLU"), ActivationType::ReLU);
        assert_eq!(parse_activation("Sigmoid"), ActivationType::Sigmoid);
        assert_eq!(parse_activation("SiLU"), ActivationType::SiLU);
        assert_eq!(parse_activation("HardSwish"), ActivationType::HardSwish);
        assert_eq!(parse_activation("Softsign"), ActivationType::Softsign);
    }

    #[test]
    fn test_parse_activation_unknown_falls_back_to_tanh() {
        assert_eq!(parse_activation("UnknownAct"), ActivationType::Tanh);
        assert_eq!(parse_activation(""), ActivationType::Tanh);
    }

    #[test]
    fn test_construction_defaults() {
        let config = HeadConfig {
            channels: None,
            bias: None,
            out_channels: None,
            activation: None,
            kernel_size: None,
        };

        let head = PostStackHead::from_config(&config, 8).expect("should build head");
        assert_eq!(head.in_channels(), 8);
        assert_eq!(head.out_channels(), 1);
        assert_eq!(head.receptive_field(), 3);
        assert!(!head.conv.do_bias);
        assert_eq!(head.activation, ActivationType::Tanh);
    }

    #[test]
    fn test_construction_explicit() {
        let config = HeadConfig {
            channels: Some(16),
            bias: Some(true),
            out_channels: Some(2),
            activation: Some("ReLU".to_string()),
            kernel_size: Some(5),
        };

        let head = PostStackHead::from_config(&config, 8).expect("should build head");
        assert_eq!(head.in_channels(), 16);
        assert_eq!(head.out_channels(), 2);
        assert_eq!(head.receptive_field(), 5);
        assert!(head.conv.do_bias);
        assert_eq!(head.activation, ActivationType::ReLU);
    }

    /// Builds a PostStackHead with identity weights (in_ch=1, out_ch=1, kernel=1).
    /// No bias, Tanh activation.
    fn build_identity_head() -> PostStackHead {
        let in_ch = 1;
        let out_ch: usize = 1;
        let kernel: usize = 1;

        let num_blocks = out_ch.div_ceil(4);
        let weights_len = num_blocks * kernel * in_ch * 4;
        let mut weights = AlignedVec::new(weights_len, 0.0f32);
        weights[0] = 1.0;

        let bias = AlignedVec::new(out_ch, 0.0f32);

        PostStackHead {
            conv: Conv1dDyn {
                weights,
                bias,
                do_bias: false,
                dilation: 1,
                in_ch,
                out_ch,
                num_blocks,
                kernel,
                prefetch_fn: crate::math::common::prefetch_strategy_simple,
            },
            activation: ActivationType::Tanh,
            state: WaveNetLayerState::new(in_ch, kernel, 0).expect("create state"),
            scratch: AlignedVec::new(out_ch * WAVENET_MAX_NUM_FRAMES, 0.0f32),
        }
    }

    #[test]
    fn test_process_single_frame_identity() {
        let mut head = build_identity_head();

        let input = [0.5f32];
        let mut output = [0.0f32];

        unsafe {
            head.process_block(&input, &mut output, 1);
        }

        assert!((output[0] - 0.462117).abs() < 1e-4); // tanh(0.5)
    }

    #[test]
    fn test_process_multi_frame() {
        let mut head = build_identity_head();

        let input = [0.1f32, -0.2, 0.3, -0.4, 0.5];
        let mut output = [0.0f32; 5];

        unsafe {
            head.process_block(&input, &mut output, 5);
        }

        for (i, &inp) in input.iter().enumerate() {
            let expected = inp.tanh();
            assert!(
                (output[i] - expected).abs() < 1e-4,
                "frame {i}: expected {expected}, got {}",
                output[i]
            );
        }
    }

    #[test]
    fn test_prewarm_no_nan() {
        let mut head = build_identity_head();
        head.prewarm();

        let input = [0.0f32];
        let mut output = [0.0f32];
        unsafe {
            head.process_block(&input, &mut output, 1);
        }

        assert!(output[0].is_finite());
    }

    #[test]
    fn test_deterministic() {
        let mut head1 = build_identity_head();
        let mut head2 = build_identity_head();

        head1.prewarm();
        head2.prewarm();

        let input = [0.3f32, -0.1, 0.7, 0.2, -0.9];
        let mut out1 = [0.0f32; 5];
        let mut out2 = [0.0f32; 5];

        unsafe {
            head1.process_block(&input, &mut out1, 5);
            head2.process_block(&input, &mut out2, 5);
        }

        assert_eq!(out1, out2);
    }

    #[test]
    fn test_process_with_weights_and_activation() {
        let in_ch: usize = 1;
        let out_ch: usize = 1;
        let kernel: usize = 3;
        let num_blocks = out_ch.div_ceil(4);
        let weights_len = num_blocks * kernel * in_ch * 4;

        let mut weights = AlignedVec::new(weights_len, 0.0f32);
        weights[0] = 0.5;
        weights[4] = 0.3;
        weights[8] = 0.2;

        let bias = AlignedVec::new(out_ch, 0.0f32);

        let mut head = PostStackHead {
            conv: Conv1dDyn {
                weights,
                bias,
                do_bias: false,
                dilation: 1,
                in_ch,
                out_ch,
                num_blocks,
                kernel,
                prefetch_fn: crate::math::common::prefetch_strategy_simple,
            },
            activation: ActivationType::Tanh,
            state: WaveNetLayerState::new(in_ch, kernel, 0).expect("create state"),
            scratch: AlignedVec::new(out_ch * WAVENET_MAX_NUM_FRAMES, 0.0f32),
        };

        // Frame 0: k=0 reads buf[-2]=0, k=1 reads buf[-1]=0, k=2 reads buf[+0]=1.0
        //   output = 0.5*0 + 0.3*0 + 0.2*1.0 = 0.2 => tanh(0.2)
        // Frame 1: k=0 reads buf[-1]=0, k=1 reads buf[+0]=1.0, k=2 reads buf[+1]=2.0
        //   output = 0.5*0 + 0.3*1.0 + 0.2*2.0 = 0.7 => tanh(0.7)
        // Frame 2: k=0 reads buf[+0]=1.0, k=1 reads buf[+1]=2.0, k=2 reads buf[+2]=3.0
        //   output = 0.5*1.0 + 0.3*2.0 + 0.2*3.0 = 1.7 => tanh(1.7)
        let input = [1.0f32, 2.0, 3.0];
        let mut output = [0.0f32; 3];

        unsafe {
            head.process_block(&input, &mut output, 3);
        }

        let expected = [0.2f32.tanh(), 0.7f32.tanh(), 1.7f32.tanh()];
        for (i, &exp) in expected.iter().enumerate() {
            assert!(
                (output[i] - exp).abs() < 1e-4,
                "frame {i}: expected {exp}, got {}",
                output[i]
            );
        }
    }

    #[test]
    fn test_set_weights_and_bias() {
        let config = HeadConfig {
            channels: Some(1),
            bias: Some(false),
            out_channels: Some(1),
            activation: None,
            kernel_size: Some(1),
        };

        let mut head = PostStackHead::from_config(&config, 1).expect("create head");

        let new_weights = vec![1.0f32, 0.0, 0.0, 0.0];
        head.set_weights(&new_weights);

        let new_bias = vec![0.0f32];
        head.set_bias(&new_bias);

        let input = [0.5f32];
        let mut output = [0.0f32];
        unsafe {
            head.process_block(&input, &mut output, 1);
        }

        assert!((output[0] - 0.462117).abs() < 1e-4); // tanh(0.5)
    }

    #[test]
    fn test_multi_channel_in_out() {
        let in_ch: usize = 2;
        let out_ch: usize = 3;
        let kernel: usize = 1;
        let num_blocks = out_ch.div_ceil(4);
        let weights_len = num_blocks * kernel * in_ch * 4;

        let mut weights = AlignedVec::new(weights_len, 0.0f32);
        // Block 0: out_ch 0,1,2,3 (but out_ch=3 so only 0,1,2)
        // Weights layout: [b][k][in_c][lane]
        // b=0, k=0, in_c=0, lane=0 => weight for out_c=0, in_c=0
        weights[0] = 1.0; // out_c=0, in_c=0
        weights[1] = 0.0; // out_c=0, in_c=1
        // in_c=1: [b=0][k=0][in_c=1][lane=0]
        weights[4] = 0.5; // out_c=0, in_c=1
        // in_c=0: [b=0][k=0][in_c=0][lane=1]
        weights[1] = 0.0; // out_c=1, in_c=0
        weights[5] = 2.0; // out_c=1, in_c=1
        // in_c=0: [b=0][k=0][in_c=0][lane=2]
        weights[2] = 0.3; // out_c=2, in_c=0
        weights[6] = 0.0; // out_c=2, in_c=1

        let bias = AlignedVec::new(out_ch, 0.0f32);

        let mut head = PostStackHead {
            conv: Conv1dDyn {
                weights,
                bias,
                do_bias: false,
                dilation: 1,
                in_ch,
                out_ch,
                num_blocks,
                kernel,
                prefetch_fn: crate::math::common::prefetch_strategy_simple,
            },
            activation: ActivationType::Tanh,
            state: WaveNetLayerState::new(in_ch, kernel, 0).expect("create state"),
            scratch: AlignedVec::new(out_ch * WAVENET_MAX_NUM_FRAMES, 0.0f32),
        };

        // Frame 0: in=[A=1.0, B=0.5]
        //   out[0] = 1.0*1.0 + 0.5*0.5 = 1.0 + 0.25 = 1.25 => tanh(1.25)
        //   out[1] = 0.0*1.0 + 2.0*0.5 = 1.0 => tanh(1.0)
        //   out[2] = 0.3*1.0 + 0.0*0.5 = 0.3 => tanh(0.3)
        let input = [1.0f32, 0.5];
        let mut output = [0.0f32; 3];

        unsafe {
            head.process_block(&input, &mut output, 1);
        }

        assert!((output[0] - 1.25f32.tanh()).abs() < 1e-4);
        assert!((output[1] - 1.0f32.tanh()).abs() < 1e-4);
        assert!((output[2] - 0.3f32.tanh()).abs() < 1e-4);
    }
}
