// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Implementation of the `clap_plugin_gui` extension for NAM-rs.

use crate::clap::gui::{GUI_HEIGHT, GUI_WIDTH};
use crate::clap::plugin::NamClapMainThread;
use clack_extensions::gui::{
    GuiApiType, GuiConfiguration, GuiSize, PluginGui, PluginGuiImpl, Window,
};
use clack_plugin::plugin::PluginError;

impl<'a> PluginGuiImpl for NamClapMainThread<'a> {
    /// Indicates whether the given graphics API configuration and floating mode is supported.
    ///
    /// For Linux (X11 via XWayland), we strictly accept the embedded X11 API.
    fn is_api_supported(&mut self, configuration: GuiConfiguration) -> bool {
        configuration.api_type == GuiApiType::X11 && !configuration.is_floating
    }

    /// Returns the preferred graphics configuration for the plugin (embedded X11).
    fn get_preferred_api(&mut self) -> Option<GuiConfiguration<'_>> {
        Some(GuiConfiguration {
            api_type: GuiApiType::X11,
            is_floating: false,
        })
    }

    /// Creates and allocates resources for the graphical interface.
    fn create(&mut self, configuration: GuiConfiguration) -> Result<(), PluginError> {
        if !self.is_api_supported(configuration) {
            return Err(PluginError::Message("GUI configuration not supported"));
        }
        Ok(())
    }

    /// Frees the resources allocated for the graphical interface.
    fn destroy(&mut self) {
        #[cfg(feature = "clap-plugin")]
        if let Some(mut window_handle) = self.window_handle.take() {
            window_handle.close();
        }
    }

    /// Sets the absolute scale factor for the GUI.
    fn set_scale(&mut self, _scale: f64) -> Result<(), PluginError> {
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
        #[cfg(feature = "clap-plugin")]
        {
            use crate::clap::gui::window::NamPluginWindow;

            if let Some(mut old_handle) = self.window_handle.take() {
                old_handle.close();
            }

            let options = baseview::WindowOpenOptions {
                // Empty title: the host (Bitwig) already displays the plugin name in the window frame.
                // Using a title here would cause duplication: "NAM-rs / NAM-rs Neural Amp Modeler".
                title: String::new(),
                size: baseview::Size::new(GUI_WIDTH as f64, GUI_HEIGHT as f64),
                scale: baseview::WindowScalePolicy::SystemScaleFactor,
                gl_config: Some(baseview::gl::GlConfig::default()),
            };

            let shared_ptr = crate::clap::plugin::NamClapSharedRef(self.shared);
            let host_shared = self.host.shared();
            // SAFETY: `host_shared` is a shared handle of the CLAP host whose real lifetime
            // is that of the plugin instance itself, which lives as long as the plugin is loaded.
            // The closure passed to `open_parented` requires `'static` to satisfy the baseview
            // threading API (`Send + 'static`), but the host is guaranteed to be valid for the
            // entire lifetime of the window (the window is closed before the plugin is destroyed via
            // `destroy()`). This transmute is the accepted pattern for integrating CLAP plugins with
            // windowing libraries that require `'static` closures.
            let host_static: clack_plugin::host::HostSharedHandle<'static> =
                unsafe { crate::clap::gui::extend_host_lifetime(host_shared) };

            let window_handle = baseview::Window::open_parented(&_window, options, move |win| {
                NamPluginWindow::new(win, shared_ptr, host_static)
            });

            self.window_handle = Some(window_handle);
        }
        Ok(())
    }

    /// Configures the window to float above the specified window (not supported).
    fn set_transient(&mut self, _window: Window) -> Result<(), PluginError> {
        Err(PluginError::Message("Floating mode is not supported"))
    }

    /// Makes the GUI window visible.
    fn show(&mut self) -> Result<(), PluginError> {
        Ok(())
    }

    /// Hides the GUI window.
    fn hide(&mut self) -> Result<(), PluginError> {
        Ok(())
    }

    /// Reports whether the window size can be changed (fixed size).
    fn can_resize(&mut self) -> bool {
        false
    }
}

/// Marker type for extension registration.
pub type NamPluginGui = PluginGui;
