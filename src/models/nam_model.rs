// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::slimmable::SlimmableModel;
use super::{NamModel, StaticModel};

impl NamModel for StaticModel {
    #[inline(always)]
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        match self {
            Self::WavenetStandard(m) => m.process(input, output),
            Self::WavenetLite(m) => m.process(input, output),
            Self::WavenetFeather(m) => m.process(input, output),
            Self::WavenetNano(m) => m.process(input, output),
            Self::WavenetA2Full(m) => m.process(input, output),
            Self::WavenetA2Lite(m) => m.process(input, output),
            Self::WavenetA2Dyn(m) => m.process(input, output),
            Self::WavenetA2Cascade(m) => m.process(input, output),
            Self::WavenetDyn(m) => m.process(input, output),
            Self::Container(m) => m.process(input, output),
            Self::Lstm1x3(m) => m.process(input, output),
            Self::Lstm1x8(m) => m.process(input, output),
            Self::Lstm1x12(m) => m.process(input, output),
            Self::Lstm1x16(m) => m.process(input, output),
            Self::Lstm1x24(m) => m.process(input, output),
            Self::Lstm2x8(m) => m.process(input, output),
            Self::Lstm2x12(m) => m.process(input, output),
            Self::Lstm2x16(m) => m.process(input, output),
            Self::Lstm1x40(m) => m.process(input, output),
            Self::Lstm2x24(m) => m.process(input, output),
            Self::LstmDyn(m) => m.process(input, output),
            Self::Linear(m) => unsafe { m.process(input, output) },
            Self::ConvNet(m) => m.process(input, output),
        }
    }

    #[cold]
    fn prewarm(&mut self, num_samples: usize) {
        match self {
            Self::WavenetStandard(m) => m.prewarm(),
            Self::WavenetLite(m) => m.prewarm(),
            Self::WavenetFeather(m) => m.prewarm(),
            Self::WavenetNano(m) => m.prewarm(),
            Self::WavenetA2Full(m) => m.prewarm(),
            Self::WavenetA2Lite(m) => m.prewarm(),
            Self::WavenetA2Dyn(m) => m.prewarm(),
            Self::WavenetA2Cascade(m) => m.prewarm(),
            Self::WavenetDyn(m) => m.prewarm(),
            Self::Container(m) => m.prewarm(num_samples),
            Self::Lstm1x3(m) => m.prewarm(num_samples),
            Self::Lstm1x8(m) => m.prewarm(num_samples),
            Self::Lstm1x12(m) => m.prewarm(num_samples),
            Self::Lstm1x16(m) => m.prewarm(num_samples),
            Self::Lstm1x24(m) => m.prewarm(num_samples),
            Self::Lstm2x8(m) => m.prewarm(num_samples),
            Self::Lstm2x12(m) => m.prewarm(num_samples),
            Self::Lstm2x16(m) => m.prewarm(num_samples),
            Self::Lstm1x40(m) => m.prewarm(num_samples),
            Self::Lstm2x24(m) => m.prewarm(num_samples),
            Self::LstmDyn(m) => m.prewarm(num_samples),
            Self::Linear(m) => m.prewarm(num_samples),
            Self::ConvNet(m) => m.prewarm(),
        }
    }

    fn prewarm_on_reset(&self) -> bool {
        match self {
            Self::WavenetStandard(m) => m.prewarm_on_reset(),
            Self::WavenetLite(m) => m.prewarm_on_reset(),
            Self::WavenetFeather(m) => m.prewarm_on_reset(),
            Self::WavenetNano(m) => m.prewarm_on_reset(),
            Self::WavenetA2Full(m) => m.prewarm_on_reset(),
            Self::WavenetA2Lite(m) => m.prewarm_on_reset(),
            Self::WavenetA2Dyn(m) => m.prewarm_on_reset(),
            Self::WavenetA2Cascade(m) => m.prewarm_on_reset(),
            Self::WavenetDyn(m) => m.prewarm_on_reset(),
            Self::Container(m) => m.prewarm_on_reset(),
            Self::Lstm1x3(m) => m.prewarm_on_reset(),
            Self::Lstm1x8(m) => m.prewarm_on_reset(),
            Self::Lstm1x12(m) => m.prewarm_on_reset(),
            Self::Lstm1x16(m) => m.prewarm_on_reset(),
            Self::Lstm1x24(m) => m.prewarm_on_reset(),
            Self::Lstm2x8(m) => m.prewarm_on_reset(),
            Self::Lstm2x12(m) => m.prewarm_on_reset(),
            Self::Lstm2x16(m) => m.prewarm_on_reset(),
            Self::Lstm1x40(m) => m.prewarm_on_reset(),
            Self::Lstm2x24(m) => m.prewarm_on_reset(),
            Self::LstmDyn(m) => m.prewarm_on_reset(),
            Self::Linear(m) => m.prewarm_on_reset(),
            Self::ConvNet(m) => m.prewarm_on_reset(),
        }
    }

    fn set_prewarm_on_reset(&mut self, val: bool) {
        match self {
            Self::WavenetStandard(m) => m.set_prewarm_on_reset(val),
            Self::WavenetLite(m) => m.set_prewarm_on_reset(val),
            Self::WavenetFeather(m) => m.set_prewarm_on_reset(val),
            Self::WavenetNano(m) => m.set_prewarm_on_reset(val),
            Self::WavenetA2Full(m) => m.set_prewarm_on_reset(val),
            Self::WavenetA2Lite(m) => m.set_prewarm_on_reset(val),
            Self::WavenetA2Dyn(m) => m.set_prewarm_on_reset(val),
            Self::WavenetA2Cascade(m) => m.set_prewarm_on_reset(val),
            Self::WavenetDyn(m) => m.set_prewarm_on_reset(val),
            Self::Container(m) => m.set_prewarm_on_reset(val),
            Self::Lstm1x3(m) => m.set_prewarm_on_reset(val),
            Self::Lstm1x8(m) => m.set_prewarm_on_reset(val),
            Self::Lstm1x12(m) => m.set_prewarm_on_reset(val),
            Self::Lstm1x16(m) => m.set_prewarm_on_reset(val),
            Self::Lstm1x24(m) => m.set_prewarm_on_reset(val),
            Self::Lstm2x8(m) => m.set_prewarm_on_reset(val),
            Self::Lstm2x12(m) => m.set_prewarm_on_reset(val),
            Self::Lstm2x16(m) => m.set_prewarm_on_reset(val),
            Self::Lstm1x40(m) => m.set_prewarm_on_reset(val),
            Self::Lstm2x24(m) => m.set_prewarm_on_reset(val),
            Self::LstmDyn(m) => m.set_prewarm_on_reset(val),
            Self::Linear(m) => m.set_prewarm_on_reset(val),
            Self::ConvNet(m) => m.set_prewarm_on_reset(val),
        }
    }

    fn reset(&mut self, sample_rate: u32, max_buffer_size: usize) -> anyhow::Result<()> {
        match self {
            Self::WavenetStandard(m) => m.reset(sample_rate, max_buffer_size),
            Self::WavenetLite(m) => m.reset(sample_rate, max_buffer_size),
            Self::WavenetFeather(m) => m.reset(sample_rate, max_buffer_size),
            Self::WavenetNano(m) => m.reset(sample_rate, max_buffer_size),
            Self::WavenetA2Full(m) => m.reset(sample_rate, max_buffer_size),
            Self::WavenetA2Lite(m) => m.reset(sample_rate, max_buffer_size),
            Self::WavenetA2Dyn(m) => m.reset(sample_rate, max_buffer_size),
            Self::WavenetA2Cascade(m) => m.reset(sample_rate, max_buffer_size),
            Self::WavenetDyn(m) => m.reset(sample_rate, max_buffer_size),
            Self::Container(m) => m.reset(sample_rate, max_buffer_size),
            Self::Lstm1x3(m) => m.reset(sample_rate, max_buffer_size),
            Self::Lstm1x8(m) => m.reset(sample_rate, max_buffer_size),
            Self::Lstm1x12(m) => m.reset(sample_rate, max_buffer_size),
            Self::Lstm1x16(m) => m.reset(sample_rate, max_buffer_size),
            Self::Lstm1x24(m) => m.reset(sample_rate, max_buffer_size),
            Self::Lstm2x8(m) => m.reset(sample_rate, max_buffer_size),
            Self::Lstm2x12(m) => m.reset(sample_rate, max_buffer_size),
            Self::Lstm2x16(m) => m.reset(sample_rate, max_buffer_size),
            Self::Lstm1x40(m) => m.reset(sample_rate, max_buffer_size),
            Self::Lstm2x24(m) => m.reset(sample_rate, max_buffer_size),
            Self::LstmDyn(m) => m.reset(sample_rate, max_buffer_size),
            Self::Linear(m) => NamModel::reset(m.as_mut(), sample_rate, max_buffer_size),
            Self::ConvNet(m) => NamModel::reset(m.as_mut(), sample_rate, max_buffer_size),
        }
    }

    fn set_max_buffer_size(&mut self, max_buf: usize) -> anyhow::Result<()> {
        match self {
            Self::WavenetStandard(m) => m.set_max_buffer_size(max_buf),
            Self::WavenetLite(m) => m.set_max_buffer_size(max_buf),
            Self::WavenetFeather(m) => m.set_max_buffer_size(max_buf),
            Self::WavenetNano(m) => m.set_max_buffer_size(max_buf),
            Self::WavenetA2Full(m) => m.set_max_buffer_size(max_buf),
            Self::WavenetA2Lite(m) => m.set_max_buffer_size(max_buf),
            Self::WavenetA2Dyn(m) => m.set_max_buffer_size(max_buf),
            Self::WavenetA2Cascade(m) => m.set_max_buffer_size(max_buf),
            Self::WavenetDyn(m) => m.set_max_buffer_size(max_buf),
            Self::Container(m) => m.set_max_buffer_size(max_buf),
            Self::Lstm1x3(m) => m.set_max_buffer_size(max_buf),
            Self::Lstm1x8(m) => m.set_max_buffer_size(max_buf),
            Self::Lstm1x12(m) => m.set_max_buffer_size(max_buf),
            Self::Lstm1x16(m) => m.set_max_buffer_size(max_buf),
            Self::Lstm1x24(m) => m.set_max_buffer_size(max_buf),
            Self::Lstm2x8(m) => m.set_max_buffer_size(max_buf),
            Self::Lstm2x12(m) => m.set_max_buffer_size(max_buf),
            Self::Lstm2x16(m) => m.set_max_buffer_size(max_buf),
            Self::Lstm1x40(m) => m.set_max_buffer_size(max_buf),
            Self::Lstm2x24(m) => m.set_max_buffer_size(max_buf),
            Self::LstmDyn(m) => m.set_max_buffer_size(max_buf),
            Self::Linear(m) => NamModel::set_max_buffer_size(m.as_mut(), max_buf),
            Self::ConvNet(m) => NamModel::set_max_buffer_size(m.as_mut(), max_buf),
        }
    }

    fn prewarm_samples(&self) -> usize {
        match self {
            Self::WavenetStandard(m) => m.prewarm_samples(),
            Self::WavenetLite(m) => m.prewarm_samples(),
            Self::WavenetFeather(m) => m.prewarm_samples(),
            Self::WavenetNano(m) => m.prewarm_samples(),
            Self::WavenetA2Full(m) => m.prewarm_samples(),
            Self::WavenetA2Lite(m) => m.prewarm_samples(),
            Self::WavenetA2Dyn(m) => m.prewarm_samples(),
            Self::WavenetA2Cascade(m) => m.prewarm_samples(),
            Self::WavenetDyn(m) => m.prewarm_samples(),
            Self::Container(m) => m.prewarm_samples(),
            Self::Lstm1x3(m) => m.prewarm_samples(),
            Self::Lstm1x8(m) => m.prewarm_samples(),
            Self::Lstm1x12(m) => m.prewarm_samples(),
            Self::Lstm1x16(m) => m.prewarm_samples(),
            Self::Lstm1x24(m) => m.prewarm_samples(),
            Self::Lstm2x8(m) => m.prewarm_samples(),
            Self::Lstm2x12(m) => m.prewarm_samples(),
            Self::Lstm2x16(m) => m.prewarm_samples(),
            Self::Lstm1x40(m) => m.prewarm_samples(),
            Self::Lstm2x24(m) => m.prewarm_samples(),
            Self::LstmDyn(m) => m.prewarm_samples(),
            Self::Linear(m) => m.prewarm_samples(),
            Self::ConvNet(m) => m.prewarm_samples(),
        }
    }

    fn slimmable_breakpoints(&self) -> Vec<f64> {
        if let Self::Container(c) = self {
            SlimmableModel::slimmable_breakpoints(c.as_ref())
        } else {
            vec![]
        }
    }
}
