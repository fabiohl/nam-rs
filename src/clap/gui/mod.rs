// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Implementation of the main graphical user interface window.

/// egui rendering function and UI components.
pub mod ui;
/// Main window and event/draw manager with baseview + egui.
pub mod window;

/// Default width of the plugin window.
pub const GUI_WIDTH: u32 = 600;
/// Default height of the plugin window.
pub const GUI_HEIGHT: u32 = 275;

/// Extends the lifetime of a `HostSharedHandle` to `'static`.
///
/// # Safety
///
/// The caller must ensure that the CLAP host referenced by the handle remains
/// valid and is not unloaded while the returned handle with `'static` lifetime
/// is in use. In the plugin context, the GUI window that uses the `'static`
/// handle is guaranteed to be destroyed/closed before the plugin is destroyed,
/// ensuring that the host's real lifetime encompasses all use of the handle.
#[inline]
pub(crate) unsafe fn extend_host_lifetime<'a>(
    h: clack_plugin::host::HostSharedHandle<'a>,
) -> clack_plugin::host::HostSharedHandle<'static> {
    // SAFETY: The caller guarantees host validity for the extended lifetime.
    unsafe { std::mem::transmute(h) }
}
