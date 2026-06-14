// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::static_builder::{build_lstm_1layer, build_lstm_2layer};
use crate::loader::nam_json::{NamModelData, get_lstm_topology};
use crate::models::StaticModel;
use anyhow::{Context, bail};

pub(crate) fn build_lstm(data: &NamModelData) -> anyhow::Result<Box<StaticModel>> {
    let (num_layers, hidden_size) = get_lstm_topology(data)
        .context("LSTM geometry not detectable (check num_layers and hidden_size)")?;

    match (num_layers, hidden_size) {
        (1, 3) => {
            let model = build_lstm_1layer::<3, 4, 12>(data, hidden_size)?;
            Ok(Box::new(StaticModel::Lstm1x3(Box::new(model))))
        }
        (1, 8) => {
            let model = build_lstm_1layer::<8, 9, 32>(data, hidden_size)?;
            Ok(Box::new(StaticModel::Lstm1x8(Box::new(model))))
        }
        (1, 12) => {
            let model = build_lstm_1layer::<12, 13, 48>(data, hidden_size)?;
            Ok(Box::new(StaticModel::Lstm1x12(Box::new(model))))
        }
        (1, 16) => {
            let model = build_lstm_1layer::<16, 17, 64>(data, hidden_size)?;
            Ok(Box::new(StaticModel::Lstm1x16(Box::new(model))))
        }
        (1, 24) => {
            let model = build_lstm_1layer::<24, 25, 96>(data, hidden_size)?;
            Ok(Box::new(StaticModel::Lstm1x24(Box::new(model))))
        }
        (2, 8) => {
            let model = build_lstm_2layer::<8, 9, 16, 32>(data, num_layers, hidden_size)?;
            Ok(Box::new(StaticModel::Lstm2x8(Box::new(model))))
        }
        (2, 12) => {
            let model = build_lstm_2layer::<12, 13, 24, 48>(data, num_layers, hidden_size)?;
            Ok(Box::new(StaticModel::Lstm2x12(Box::new(model))))
        }
        (2, 16) => {
            let model = build_lstm_2layer::<16, 17, 32, 64>(data, num_layers, hidden_size)?;
            Ok(Box::new(StaticModel::Lstm2x16(Box::new(model))))
        }
        (1, 40) => {
            let model = build_lstm_1layer::<40, 41, 160>(data, hidden_size)?;
            Ok(Box::new(StaticModel::Lstm1x40(Box::new(model))))
        }
        (2, 24) => {
            let model = build_lstm_2layer::<24, 25, 48, 96>(data, num_layers, hidden_size)?;
            Ok(Box::new(StaticModel::Lstm2x24(Box::new(model))))
        }
        _ => bail!(
            "Unsupported LSTM topology: {} layers × {} hidden units. Static profiles are 1×8..1×40 and 2×8..2×24.",
            num_layers,
            hidden_size
        ),
    }
}
