// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Implementação da janela de interface gráfica principal.

/// Função de renderização e componentes da UI do egui.
pub mod ui;
/// Janela principal e gerenciador de eventos/desenho com baseview + egui.
pub mod window;

/// Largura padrão da janela do plugin.
pub const GUI_WIDTH: u32 = 600;
/// Altura padrão da janela do plugin.
pub const GUI_HEIGHT: u32 = 275;

/// Estende o lifetime de um `HostSharedHandle` para `'static`.
///
/// # Safety
///
/// O chamador deve garantir que o host CLAP referenciado pelo handle permaneça
/// válido e não seja descarregado enquanto o handle retornado com lifetime `'static`
/// estiver em uso. No contexto do plugin, a janela gráfica (GUI) que usa o handle
/// `'static` é garantidamente destruída/fechada antes da destruição do plugin,
/// assegurando que o tempo de vida real do host englobe todo o uso do handle.
#[inline]
pub(crate) unsafe fn extend_host_lifetime<'a>(
    h: clack_plugin::host::HostSharedHandle<'a>,
) -> clack_plugin::host::HostSharedHandle<'static> {
    // SAFETY: O chamador garante a validade do host ao longo do tempo de vida estendido.
    unsafe { std::mem::transmute(h) }
}
