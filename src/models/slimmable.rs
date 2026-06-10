// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! `SlimmableModel` trait — models that can dynamically scale quality/complexity
//! at runtime without reallocation.
//!
//! This is the official NAM architecture for runtime quality scaling.

/// Trait for models that can dynamically scale quality/complexity at runtime
/// without reallocation.
///
/// The value `0.0` represents the minimum quality/cheapest model,
/// and `1.0` represents the maximum quality/full model.
///
/// Implementors:
/// - `ContainerModel` (Épico 3): selects between pre-built submodels by threshold.
/// - `SlimmableWavenet` (Épico 6, future): channel-slices a single network.
pub trait SlimmableModel {
    /// Sets the slimmable quality/size level.
    ///
    /// `val` is in `[0.0, 1.0]` where `0.0` = minimum quality and `1.0` = full quality.
    fn set_slimmable_size(&mut self, val: f32);
}
