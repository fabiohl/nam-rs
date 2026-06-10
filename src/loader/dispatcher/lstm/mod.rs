// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

pub(crate) mod dispatch;
pub(crate) mod static_builder;
pub(crate) mod weights;

pub(crate) use dispatch::build_lstm;
