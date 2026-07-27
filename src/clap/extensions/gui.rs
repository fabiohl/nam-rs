// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Implementation of the `clap_plugin_gui` extension for NAM-rs.

use crate::clap::gui::{GUI_HEIGHT, GUI_WIDTH};
use crate::clap::gui::lifecycle::{GuiEvent, GuiLifecycle};
use crate::clap::plugin::NamClapMainThread;
use crate::clap::plugin::debug_assert_main_thread;
use clack_extensions::gui::{
    GuiApiType, GuiConfiguration, GuiSize, HostGui, PluginGui, PluginGuiImpl, Window,
};
use clack_plugin::plugin::PluginError;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(feature = "clap-plugin")]
impl<'a> NamClapMainThread<'a> {
    /// Closes all active GUI windows (embedded and floating).
    /// Idempotent — safe to call even when no windows are open.
    fn teardown_gui_resources(&mut self) {
        if let Some(signal) = self.floating_close_signal.take() {
            signal.store(true, Ordering::Release);
        }
        if let Some(handle) = self.floating_thread_handle.take() {
            // R13: watchdog com deadline de 2 s para evitar freeze do host.
            // Se a janela X11/Wayland não responder ao close_signal dentro do prazo,
            // abandonamos o handle (leak controlado de 1 thread — preferível a congelar o DAW).
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
            loop {
                if handle.is_finished() {
                    let _ = handle.join();
                    break;
                }
                if std::time::Instant::now() >= deadline {
                    // Abandono controlado: a thread vive até o processo terminar.
                    // O sistema operacional recolhe todos os recursos (fds, mapeamentos)
                    // no exit do processo. Ver docs/architecture.md §lifecycle-r13.
                    log::warn!(
                        "NAM-rs: floating window thread did not exit within 2 s on destroy \
                         — abandoning handle to avoid host freeze (R13 controlled leak)"
                    );
                    // handle é movido para fora do `if let` e dropado aqui,
                    // sem join — a thread continua rodando detached.
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }
        for sink in [
            &self.shared.cold.dialog_handle_sink,
            &self.shared.cold.ir_dialog_handle_sink,
        ] {
            if let Ok(mut guard) = sink.lock()
                && let Some(h) = guard.take()
            {
                let _ = h.join();
            }
        }
        if let Some(mut window_handle) = self.window_handle.take() {
            window_handle.close();
        }
    }

    /// Returns the static host handle and shared pointer needed by window callbacks.
    ///
    /// # Safety
    ///
    /// The caller must ensure the window is closed before the plugin is destroyed.
    /// See `crate::clap::gui::extend_host_lifetime` for details on the transmute.
    fn host_static_and_shared(
        &self,
    ) -> (
        clack_plugin::host::HostSharedHandle<'static>,
        crate::clap::plugin::NamClapSharedRef,
    ) {
        // SAFETY: `self.shared` is a valid reference to the live plugin shared state.
        // The caller guarantees the window is closed before plugin destruction (see fn doc).
        let shared_ptr = unsafe { crate::clap::plugin::NamClapSharedRef::new(self.shared) };
        let host_shared = self.host.shared();
        let host_static: clack_plugin::host::HostSharedHandle<'static> =
            unsafe { crate::clap::gui::extend_host_lifetime(host_shared) };
        (host_static, shared_ptr)
    }

    /// Builds the common `baseview::WindowOpenOptions` for both embedded and floating windows.
    fn window_options(title: &str) -> baseview::WindowOpenOptions {
        baseview::WindowOpenOptions {
            title: title.to_string(),
            size: baseview::Size::new(GUI_WIDTH as f64, GUI_HEIGHT as f64),
            scale: baseview::WindowScalePolicy::SystemScaleFactor,
            gl_config: Some(baseview::gl::GlConfig::default()),
        }
    }
}

impl<'a> PluginGuiImpl for NamClapMainThread<'a> {
    /// Indicates whether the given graphics API configuration and floating mode is supported.
    ///
    /// Accepts X11 both embedded (preferred) and floating (fallback) modes,
    /// so hosts that only offer floating windows are still usable.
    fn is_api_supported(&mut self, configuration: GuiConfiguration) -> bool {
        configuration.api_type == GuiApiType::X11
    }

    /// Returns the preferred graphics configuration for the plugin (embedded X11).
    /// Falls back to floating only when the host does not offer embedded mode.
    fn get_preferred_api(&mut self) -> Option<GuiConfiguration<'_>> {
        Some(GuiConfiguration {
            api_type: GuiApiType::X11,
            is_floating: false,
        })
    }

    /// Creates and allocates resources for the graphical interface.
    fn create(&mut self, configuration: GuiConfiguration) -> Result<(), PluginError> {
        debug_assert_main_thread(&self.host);
        if !self.is_api_supported(configuration) {
            return Err(PluginError::Message("GUI configuration not supported"));
        }
        let mode = if configuration.is_floating {
            "floating"
        } else {
            "embedded"
        };
        log::info!("GUI mode selected = {mode}");
        #[cfg(feature = "clap-plugin")]
        {
            self.gui_lifecycle = GuiLifecycle::Hidden;
        }
        Ok(())
    }

    /// Frees the resources allocated for the graphical interface.
    fn destroy(&mut self) {
        debug_assert_main_thread(&self.host);
        #[cfg(feature = "clap-plugin")]
        {
            // Notify host that the GUI was destroyed by the plugin
            if let Some(gui_host) = self.host.get_extension::<HostGui>() {
                gui_host.closed(&self.host.shared(), true);
            }
            self.teardown_gui_resources();
            let _ = self.gui_lifecycle.transition(GuiEvent::Destroy);
        }
    }

    /// Sets the absolute scale factor for the GUI.
    fn set_scale(&mut self, scale: f64) -> Result<(), PluginError> {
        use std::sync::atomic::Ordering;
        self.shared
            .cold
            .gui_scale_factor
            .store((scale as f32).to_bits(), Ordering::Relaxed);
        Ok(())
    }

    /// Returns the fixed GUI size (GUI_WIDTH x GUI_HEIGHT pixels).
    fn get_size(&mut self) -> Option<GuiSize> {
        Some(GuiSize {
            width: GUI_WIDTH,
            height: GUI_HEIGHT,
        })
    }

    /// Sets the GUI size. Only the fixed size is accepted.
    fn set_size(&mut self, size: GuiSize) -> Result<(), PluginError> {
        if size.width == GUI_WIDTH && size.height == GUI_HEIGHT {
            Ok(())
        } else {
            Err(PluginError::Message(
                "GUI resizing is not supported in this version",
            ))
        }
    }

    /// Sets the parent window (host) where the GUI should be embedded.
    fn set_parent(&mut self, _window: Window) -> Result<(), PluginError> {
        debug_assert_main_thread(&self.host);
        #[cfg(feature = "clap-plugin")]
        {
            use crate::clap::gui::window::NamPluginWindow;

            if let Some(mut old_handle) = self.window_handle.take() {
                old_handle.close();
            }

            let options = Self::window_options("");
            let (host_static, shared_ptr) = self.host_static_and_shared();

            let close_signal = Arc::new(AtomicBool::new(false));
            let cs = Arc::clone(&close_signal);

            let alive_fence = self.shared.cold.alive_fence.clone();
            let scale_factor = {
                let stored = self.shared.cold.gui_scale_factor.load(Ordering::Relaxed);
                if stored == 0 {
                    1.0f32
                } else {
                    f32::from_bits(stored)
                }
            };

            let window_handle = baseview::Window::open_parented(&_window, options, move |win| {
                NamPluginWindow::new(win, shared_ptr, host_static, cs, alive_fence, scale_factor)
            });

            self.window_handle = Some(window_handle);
        }
        Ok(())
    }

    /// Configures the window to float above the host window (floating fallback mode).
    ///
    /// NOTE: The `_window` parameter provides the host window for a transient-for
    /// stacking relationship (WM_TRANSIENT_FOR). baseview 0.1.1's `open_blocking` API
    /// does not expose transient window support, so the floating window opens as an
    /// independent top-level window. Tracked for future improvement when baseview adds
    /// transient window capabilities.
    fn set_transient(&mut self, _window: Window) -> Result<(), PluginError> {
        debug_assert_main_thread(&self.host);
        #[cfg(feature = "clap-plugin")]
        {
            use crate::clap::gui::window::NamPluginWindow;

            self.teardown_gui_resources();

            let options = Self::window_options("NAM-rs");
            let (host_static, shared_ptr) = self.host_static_and_shared();

            let close_signal = Arc::new(AtomicBool::new(false));
            let cs = Arc::clone(&close_signal);
            let window_ready = Arc::new(AtomicBool::new(false));
            let ready = Arc::clone(&window_ready);

            let alive_fence = self.shared.cold.alive_fence.clone();
            let scale_factor = {
                let stored = self.shared.cold.gui_scale_factor.load(Ordering::Relaxed);
                if stored == 0 {
                    1.0f32
                } else {
                    f32::from_bits(stored)
                }
            };

            let handle = std::thread::spawn(move || {
                baseview::Window::open_blocking(options, move |win| {
                    let window = NamPluginWindow::new(
                        win,
                        shared_ptr,
                        host_static,
                        cs,
                        alive_fence,
                        scale_factor,
                    );
                    ready.store(true, Ordering::Relaxed);
                    window
                });
            });

            // Wait for the window thread to confirm initialization (up to 2 seconds).
            // If NamPluginWindow::new panics or the X11 connection fails,
            // `ready` will never be set and we report the error to the host.
            let start = std::time::Instant::now();
            while !window_ready.load(Ordering::Relaxed) {
                if start.elapsed() > std::time::Duration::from_secs(2) {
                    close_signal.store(true, Ordering::Relaxed);
                    self.floating_thread_handle = Some(handle);
                    self.floating_close_signal = Some(close_signal);
                    return Err(PluginError::Message(
                        "Floating window creation failed: initialization timed out",
                    ));
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }

            self.floating_thread_handle = Some(handle);
            self.floating_close_signal = Some(close_signal);
        }
        Ok(())
    }

    /// Makes the GUI window visible.
    ///
    /// Transitions the lifecycle state from `Hidden` to `ShowRequested`.
    /// The actual window mapping happens on the GUI thread (baseview callback),
    /// which transitions to `Active` once the window is ready.
    fn show(&mut self) -> Result<(), PluginError> {
        debug_assert_main_thread(&self.host);
        #[cfg(feature = "clap-plugin")]
        {
            self.gui_lifecycle.transition(GuiEvent::Show)?;
        }
        Ok(())
    }

    /// Hides the GUI window.
    ///
    /// Transitions the lifecycle state from `Active` to `HideRequested`.
    /// The actual window unmapping happens on the GUI thread.
    /// Resources are preserved so `show()` can re-display the window.
    fn hide(&mut self) -> Result<(), PluginError> {
        debug_assert_main_thread(&self.host);
        #[cfg(feature = "clap-plugin")]
        {
            self.gui_lifecycle.transition(GuiEvent::Hide)?;
        }
        Ok(())
    }

    /// Reports whether the window size can be changed (fixed size).
    fn can_resize(&mut self) -> bool {
        false
    }
}

/// Marker type for extension registration.
pub type NamPluginGui = PluginGui;
