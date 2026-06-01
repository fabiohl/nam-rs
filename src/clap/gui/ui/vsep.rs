// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
//! Separador vertical estilizado.

use super::colors::COL_BORDER;

/// Renderiza um separador vertical estilizado (m5) — linha fina #2E3440 ao invés do Separator padrão do egui.
pub fn styled_vsep(ui: &mut egui::Ui) {
    let space = 6.0;
    ui.add_space(space);
    let (rect, _) =
        ui.allocate_exact_size(egui::vec2(1.0, ui.available_height()), egui::Sense::hover());
    ui.painter().line(
        vec![rect.center_top(), rect.center_bottom()],
        egui::Stroke::new(0.5, COL_BORDER),
    );
    ui.add_space(space);
}
