// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
//! Keyboard focus navigation: Tab/Shift+Tab cycling through UI controls.

pub(crate) fn handle_focus_navigation(ui: &mut egui::Ui, load_btn_id: Option<egui::Id>) {
    let load_btn_id = load_btn_id.unwrap_or_else(|| ui.make_persistent_id("load_model_button"));
    let controls = [
        ui.make_persistent_id("input_gain_knob"),
        ui.make_persistent_id("output_gain_knob"),
        ui.make_persistent_id("gate_thresh_knob"),
        ui.make_persistent_id("bypass_switch"),
        load_btn_id,
    ];

    let mut tab_pressed = false;
    let mut shift_pressed = false;
    ui.input_mut(|i| {
        i.events.retain(|e| {
            if let egui::Event::Key {
                key: egui::Key::Tab,
                pressed: true,
                modifiers,
                ..
            } = e
            {
                tab_pressed = true;
                shift_pressed = modifiers.shift;
                false
            } else {
                true
            }
        });
    });

    if tab_pressed {
        let focused = ui.memory(|mem| mem.focused());
        let next_focus = if let Some(curr) = focused {
            if let Some(idx) = controls.iter().position(|&id| id == curr) {
                if shift_pressed {
                    if idx == 0 {
                        controls[4]
                    } else {
                        controls[idx - 1]
                    }
                } else {
                    if idx == 4 {
                        controls[0]
                    } else {
                        controls[idx + 1]
                    }
                }
            } else if shift_pressed {
                controls[4]
            } else {
                controls[0]
            }
        } else if shift_pressed {
            controls[4]
        } else {
            controls[0]
        };
        ui.memory_mut(|mem| mem.request_focus(next_focus));
    }
}
