// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::loader::nam_json::{NamModelData, NamWavenetTopology};
use crate::models::wavenet::WaveNetModel;

pub(crate) fn build_wavenet_nano(data: &NamModelData) -> anyhow::Result<WaveNetModel<4, 3, 2>> {
    super::standard::build_wavenet_typed::<4, 3, 2>(data, NamWavenetTopology::Nano)
}
