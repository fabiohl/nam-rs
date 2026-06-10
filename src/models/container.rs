// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! `ContainerModel` — A bundle of pre-trained submodels selected by a quality threshold.
//!
//! Each submodel covers values up to (but not including) its `max_value`.
//! The last submodel is the fallback for values at or above the last threshold.
//!
//! ## Architecture
//!
//! This implements the official NAM `SlimmableContainer` architecture
//! (registered since file version 0.7.0). It is the format used by the
//! mainstream A2 distribution (nano + standard bundle).

use std::cmp::Ordering;

use super::slimmable::SlimmableModel;
use super::{NamModel, StaticModel};

/// A bundle of pre-trained submodels selected by a quality threshold.
///
/// Each submodel covers values up to (but not including) its `max_value`.
/// The last submodel is the fallback for values at or above the last threshold.
///
/// ## Architecture
///
/// This implements the official NAM `SlimmableContainer` architecture
/// (registered since file version 0.7.0). It is the format used by the
/// mainstream A2 distribution (nano + standard bundle).
pub struct ContainerModel {
    submodels: Vec<(f32, Box<StaticModel>)>,
    active_index: usize,
    sample_rate: u32,
    max_buffer_size: usize,
}

impl ContainerModel {
    /// Creates a new `ContainerModel`.
    ///
    /// # Requirements (enforced)
    ///
    /// - `submodels` must be non-empty.
    /// - Sorted by `max_value` ascending.
    /// - Last `max_value` must be `>= 1.0`.
    pub fn new(
        mut submodels: Vec<(f32, Box<StaticModel>)>,
        sample_rate: u32,
    ) -> anyhow::Result<Self> {
        if submodels.is_empty() {
            anyhow::bail!("ContainerModel: no submodels provided");
        }

        submodels.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        for w in submodels.windows(2) {
            if w[1].0 <= w[0].0 {
                anyhow::bail!("ContainerModel: submodels must be sorted by ascending max_value");
            }
        }
        if submodels.last().unwrap().0 < 1.0 {
            anyhow::bail!("ContainerModel: last submodel max_value must be >= 1.0");
        }

        let active_index = submodels.len() - 1;

        let mut container = Self {
            submodels,
            active_index,
            sample_rate,
            max_buffer_size: 0,
        };

        container.prewarm(4096);

        Ok(container)
    }

    /// Returns the list of `(max_value, StaticModel ref)` for diagnostics.
    pub fn submodels(&self) -> &[(f32, Box<StaticModel>)] {
        &self.submodels
    }

    /// Returns the index of the currently active submodel.
    pub fn active_index(&self) -> usize {
        self.active_index
    }

    /// Returns the sample rate used to validate submodels.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub(crate) fn active(&self) -> &StaticModel {
        &self.submodels[self.active_index].1
    }

    pub(crate) fn active_mut(&mut self) -> &mut StaticModel {
        &mut self.submodels[self.active_index].1
    }
}

impl NamModel for ContainerModel {
    #[inline(always)]
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        self.active_mut().process(input, output);
    }

    fn prewarm(&mut self, num_samples: usize) {
        for (_, model) in &mut self.submodels {
            model.prewarm(num_samples);
        }
    }

    fn reset(&mut self, sample_rate: u32, max_buffer_size: usize) {
        self.sample_rate = sample_rate;
        self.max_buffer_size = max_buffer_size;
        for (_, model) in &mut self.submodels {
            model.reset(sample_rate, max_buffer_size);
        }
    }

    fn set_max_buffer_size(&mut self, max_buf: usize) {
        self.max_buffer_size = max_buf;
        for (_, model) in &mut self.submodels {
            model.set_max_buffer_size(max_buf);
        }
    }

    fn prewarm_samples(&self) -> usize {
        self.active().prewarm_samples()
    }
}

impl SlimmableModel for ContainerModel {
    fn set_slimmable_size(&mut self, val: f32) {
        let mut next = self.submodels.len() - 1;

        for (i, (max_value, _)) in self.submodels.iter().enumerate() {
            if val < *max_value {
                next = i;
                break;
            }
        }

        if next == self.active_index {
            return;
        }

        self.submodels[next]
            .1
            .reset(self.sample_rate, self.max_buffer_size);
        self.active_index = next;
    }
}

impl super::sealed::Sealed for ContainerModel {}
