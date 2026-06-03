// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::math::common::{AlignedVec, PrefetchFn};
use crate::models::wavenet::{Conv1d, DenseLayer};
use crate::models::wavenet::{Conv1dDyn, DenseLayerDyn};

/// Output type for convolution weights, unifying `Conv1d<IN,OUT,K>` and `Conv1dDyn`.
pub(crate) trait ConvWeightsOutput: Sized {
    #[allow(clippy::too_many_arguments)]
    fn from_parts(
        weights: AlignedVec<u16>,
        bias: AlignedVec<f32>,
        do_bias: bool,
        dilation: usize,
        in_ch: usize,
        out_ch: usize,
        k_size: usize,
        prefetch_fn: PrefetchFn,
    ) -> Self;
}

impl<const IN: usize, const OUT: usize, const K: usize> ConvWeightsOutput for Conv1d<IN, OUT, K> {
    #[inline(always)]
    fn from_parts(
        weights: AlignedVec<u16>,
        bias: AlignedVec<f32>,
        do_bias: bool,
        dilation: usize,
        _in_ch: usize,
        _out_ch: usize,
        _k_size: usize,
        prefetch_fn: PrefetchFn,
    ) -> Self {
        Conv1d {
            weights,
            bias,
            do_bias,
            dilation,
            prefetch_fn,
        }
    }
}

impl ConvWeightsOutput for Conv1dDyn {
    #[inline(always)]
    fn from_parts(
        weights: AlignedVec<u16>,
        bias: AlignedVec<f32>,
        do_bias: bool,
        dilation: usize,
        in_ch: usize,
        out_ch: usize,
        k_size: usize,
        prefetch_fn: PrefetchFn,
    ) -> Self {
        Conv1dDyn {
            weights,
            bias,
            do_bias,
            dilation,
            in_ch,
            out_ch,
            num_blocks: out_ch.div_ceil(4),
            kernel: k_size,
            prefetch_fn,
        }
    }
}

/// Output type for dense layer weights, unifying `DenseLayer<IN,OUT>` and `DenseLayerDyn`.
pub(crate) trait DenseWeightsOutput: Sized {
    fn from_parts(
        weights: AlignedVec<u16>,
        bias: AlignedVec<f32>,
        do_bias: bool,
        in_size: usize,
        out_size: usize,
    ) -> Self;

    fn from_parts_head(
        weights: AlignedVec<u16>,
        bias: AlignedVec<f32>,
        do_bias: bool,
        in_size: usize,
        out_size: usize,
        f32_weights: AlignedVec<f32>,
    ) -> Self;
}

impl<const IN: usize, const OUT: usize> DenseWeightsOutput for DenseLayer<IN, OUT> {
    #[inline(always)]
    fn from_parts(
        weights: AlignedVec<u16>,
        bias: AlignedVec<f32>,
        do_bias: bool,
        _in_size: usize,
        _out_size: usize,
    ) -> Self {
        DenseLayer {
            weights,
            bias,
            do_bias,
            f32_weights: None,
        }
    }

    #[inline(always)]
    fn from_parts_head(
        weights: AlignedVec<u16>,
        bias: AlignedVec<f32>,
        do_bias: bool,
        _in_size: usize,
        _out_size: usize,
        f32_weights: AlignedVec<f32>,
    ) -> Self {
        DenseLayer {
            weights,
            bias,
            do_bias,
            f32_weights: Some(f32_weights),
        }
    }
}

impl DenseWeightsOutput for DenseLayerDyn {
    #[inline(always)]
    fn from_parts(
        weights: AlignedVec<u16>,
        bias: AlignedVec<f32>,
        do_bias: bool,
        in_size: usize,
        out_size: usize,
    ) -> Self {
        DenseLayerDyn {
            weights,
            bias,
            do_bias,
            in_size,
            out_size,
        }
    }

    #[inline(always)]
    fn from_parts_head(
        weights: AlignedVec<u16>,
        bias: AlignedVec<f32>,
        do_bias: bool,
        in_size: usize,
        out_size: usize,
        f32_weights: AlignedVec<f32>,
    ) -> Self {
        let _ = f32_weights;
        DenseLayerDyn {
            weights,
            bias,
            do_bias,
            in_size,
            out_size,
        }
    }
}
