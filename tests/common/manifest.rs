// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use serde::Deserialize;

#[derive(Deserialize, Debug, Clone)]
pub struct ManifestEntry {
    pub filename: String,
    #[allow(dead_code)]
    pub sha256: String,
    pub expected_class: String,
    #[allow(dead_code)]
    pub is_goal_target: bool,
    #[allow(dead_code)]
    pub name: String,
    #[allow(dead_code)]
    pub author: String,
}
