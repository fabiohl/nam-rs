// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Detection of Linear topologies from model data.

use super::super::data::NamModelData;
use super::super::validation::MAX_RECEPTIVE_FIELD;

/// Checks and returns the Linear geometry (receptive_field, has_bias).
pub fn get_linear_topology(data: &NamModelData) -> Option<(usize, bool)> {
    if data.architecture != "Linear" {
        return None;
    }

    let receptive_field = data.config.receptive_field?;
    if receptive_field > MAX_RECEPTIVE_FIELD {
        log::warn!(
            "Linear receptive_field ({receptive_field}) exceeds maximum \
             {MAX_RECEPTIVE_FIELD} — OOM/DoS protection"
        );
        return None;
    }
    let has_bias = data.config.bias.unwrap_or(false);
    Some((receptive_field, has_bias))
}
