// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
//! Paleta de cores aprovada (T4.0.2) e helpers de resolução de cor.

/// Cor de fundo principal da janela (`#1A1D23`).
pub const COL_BG: egui::Color32 = egui::Color32::from_rgb(26, 29, 35);
/// Cor de fundo de painéis e botões (`#232830`).
pub const COL_PANEL: egui::Color32 = egui::Color32::from_rgb(35, 40, 48);
/// Cor de bordas e separadores (`#2E3440`).
pub const COL_BORDER: egui::Color32 = egui::Color32::from_rgb(46, 52, 64);
/// Cor de texto principal (`#E5E9F0`).
pub const COL_TEXT: egui::Color32 = egui::Color32::from_rgb(229, 233, 240);
/// Cor de texto secundário/atenuado (`#8B95A5`).
pub const COL_MUTED: egui::Color32 = egui::Color32::from_rgb(139, 149, 165);
/// Cor de destaque principal — turquesa padrão (`#00D4AA`).
pub const COL_ACCENT: egui::Color32 = egui::Color32::from_rgb(0, 212, 170);
/// Cor âmbar para avisos e destaque secundário (`#F5A623`).
pub const COL_AMBER: egui::Color32 = egui::Color32::from_rgb(245, 166, 35);
/// Cor verde do medidor VU (`#43E97B`).
pub const COL_VU_GREEN: egui::Color32 = egui::Color32::from_rgb(67, 233, 123);
/// Cor amarela do medidor VU (`#F5CE62`).
pub const COL_VU_YELLOW: egui::Color32 = egui::Color32::from_rgb(245, 206, 98);
/// Cor vermelha do medidor VU e LEDs de clipping (`#F74E4E`).
pub const COL_VU_RED: egui::Color32 = egui::Color32::from_rgb(247, 78, 78);
/// Cor do botão de bypass quando desativado (`#4A4F5A`).
pub const COL_BYPASS_OFF: egui::Color32 = egui::Color32::from_rgb(74, 79, 90);

use crate::clap::plugin::NamClapShared;
use std::sync::atomic::Ordering;

/// Resolve a cor de accent dinâmica do plugin.
///
/// Primeiro tenta usar a cor da track fornecida pelo host DAW (armazenada em
/// `track_accent_color` como ARGB empacotado). Se `alpha == 0` (sentinela de
/// "sem cor"), retorna `COL_ACCENT` (turquesa padrão).
pub fn resolve_accent(shared: &NamClapShared) -> egui::Color32 {
    let packed = shared.track_accent_color.load(Ordering::Relaxed);
    let alpha = (packed >> 24) as u8;
    if alpha == 0 {
        COL_ACCENT
    } else {
        let red = ((packed >> 16) & 0xFF) as u8;
        let green = ((packed >> 8) & 0xFF) as u8;
        let blue = (packed & 0xFF) as u8;
        egui::Color32::from_rgba_unmultiplied(red, green, blue, alpha)
    }
}

/// Resolve uma cor ARGB empacotada em u32 para `egui::Color32`, com fallback.
///
/// Segue a mesma convenção de `resolve_accent`: `alpha == 0` indica ausência
/// de cor e retorna o fallback fornecido.
pub fn resolve_color(packed: u32, fallback: egui::Color32) -> egui::Color32 {
    let alpha = (packed >> 24) as u8;
    if alpha == 0 {
        fallback
    } else {
        let red = ((packed >> 16) & 0xFF) as u8;
        let green = ((packed >> 8) & 0xFF) as u8;
        let blue = (packed & 0xFF) as u8;
        egui::Color32::from_rgba_unmultiplied(red, green, blue, alpha)
    }
}
