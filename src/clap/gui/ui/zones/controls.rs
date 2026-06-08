// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
//! Zone 2 (center): Controls — Input Gain, Output Gain and Gate Threshold knobs.

use crate::clap::plugin::NamClapShared;
use clack_plugin::host::HostSharedHandle;
use std::sync::atomic::Ordering;

use crate::clap::gui::ui::{
    colors::{COL_AMBER, resolve_color},
    knob::handle_knob,
};

pub(crate) fn draw_zone2_controls(
    ui: &mut egui::Ui,
    shared: &NamClapShared,
    host: &HostSharedHandle,
    current_bypass: bool,
    accent_color: egui::Color32,
) {
    ui.allocate_ui(egui::vec2(240.0, 210.0), |ui| {
        if current_bypass {
            ui.disable();
        }
        ui.vertical(|ui| {
            ui.add_space(12.0);
            let ind_input = shared.cold.param_indication
                [crate::clap::extensions::params::PARAM_INPUT_GAIN as usize]
                .load(Ordering::Relaxed);
            let ind_input_color = resolve_color(
                shared.cold.param_indication_color
                    [crate::clap::extensions::params::PARAM_INPUT_GAIN as usize]
                    .load(Ordering::Relaxed),
                egui::Color32::from_rgb(94, 129, 172),
            );

            let ind_output = shared.cold.param_indication
                [crate::clap::extensions::params::PARAM_OUTPUT_GAIN as usize]
                .load(Ordering::Relaxed);
            let ind_output_color = resolve_color(
                shared.cold.param_indication_color
                    [crate::clap::extensions::params::PARAM_OUTPUT_GAIN as usize]
                    .load(Ordering::Relaxed),
                egui::Color32::from_rgb(94, 129, 172),
            );

            let ind_gate = shared.cold.param_indication
                [crate::clap::extensions::params::PARAM_GATE_THRESH as usize]
                .load(Ordering::Relaxed);
            let ind_gate_color = resolve_color(
                shared.cold.param_indication_color
                    [crate::clap::extensions::params::PARAM_GATE_THRESH as usize]
                    .load(Ordering::Relaxed),
                egui::Color32::from_rgb(94, 129, 172),
            );

            ui.horizontal(|ui| {
                ui.allocate_ui(egui::vec2(78.0, 185.0), |ui| {
                    handle_knob(
                        ui,
                        ui.make_persistent_id("input_gain_knob"),
                        "INPUT",
                        crate::math::constants::GAIN_MIN_DB..=crate::math::constants::GAIN_MAX_DB,
                        0.0,
                        &shared.ui_to_rt.param_input_gain,
                        &shared.ui_to_rt.gesture_flags,
                        &shared.ui_to_rt.gui_param_generation,
                        crate::clap::extensions::params::PARAM_INPUT_GAIN as usize,
                        accent_color,
                        accent_color,
                        host,
                        egui::vec2(70.0, 70.0),
                        ind_input,
                        ind_input_color,
                        " dB",
                    );
                });
                ui.add_space(2.0);
                ui.allocate_ui(egui::vec2(78.0, 185.0), |ui| {
                    handle_knob(
                        ui,
                        ui.make_persistent_id("output_gain_knob"),
                        "OUTPUT",
                        crate::math::constants::GAIN_MIN_DB..=crate::math::constants::GAIN_MAX_DB,
                        0.0,
                        &shared.ui_to_rt.param_output_gain,
                        &shared.ui_to_rt.gesture_flags,
                        &shared.ui_to_rt.gui_param_generation,
                        crate::clap::extensions::params::PARAM_OUTPUT_GAIN as usize,
                        accent_color,
                        accent_color,
                        host,
                        egui::vec2(70.0, 70.0),
                        ind_output,
                        ind_output_color,
                        " dB",
                    );
                });
                ui.add_space(2.0);
                ui.allocate_ui(egui::vec2(70.0, 185.0), |ui| {
                    handle_knob(
                        ui,
                        ui.make_persistent_id("gate_thresh_knob"),
                        "GATE",
                        -90.0..=-40.0,
                        -70.0,
                        &shared.ui_to_rt.param_gate_thresh,
                        &shared.ui_to_rt.gesture_flags,
                        &shared.ui_to_rt.gui_param_generation,
                        crate::clap::extensions::params::PARAM_GATE_THRESH as usize,
                        COL_AMBER,
                        accent_color,
                        host,
                        egui::vec2(42.0, 42.0),
                        ind_gate,
                        ind_gate_color,
                        " dB (Threshold)",
                    );
                });
            });
        });
    });
}
