// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Implementação da extensão `clap_plugin_gui` para o NAM-rs.

use crate::clap::plugin::NamClapMainThread;
use clack_extensions::gui::{
    GuiApiType, GuiConfiguration, GuiSize, PluginGui, PluginGuiImpl, Window,
};
use clack_plugin::plugin::PluginError;

impl<'a> PluginGuiImpl for NamClapMainThread<'a> {
    /// Indica se a configuração de API gráfica e modo de flutuação é suportada.
    ///
    /// Para Linux (X11 via XWayland), aceitamos estritamente a API X11 embutida.
    fn is_api_supported(&mut self, configuration: GuiConfiguration) -> bool {
        configuration.api_type == GuiApiType::X11 && !configuration.is_floating
    }

    /// Retorna a configuração gráfica preferencial para o plugin (X11 embutida).
    fn get_preferred_api(&mut self) -> Option<GuiConfiguration<'_>> {
        Some(GuiConfiguration {
            api_type: GuiApiType::X11,
            is_floating: false,
        })
    }

    /// Cria e aloca recursos para a interface gráfica.
    fn create(&mut self, configuration: GuiConfiguration) -> Result<(), PluginError> {
        if !self.is_api_supported(configuration) {
            return Err(PluginError::Message("Configuração de GUI não suportada"));
        }
        Ok(())
    }

    /// Libera os recursos alocados para a interface gráfica.
    fn destroy(&mut self) {
        self.shared
            .alive_fence
            .store(false, std::sync::atomic::Ordering::Relaxed);
        #[cfg(feature = "clap-plugin")]
        if let Some(mut window_handle) = self.window_handle.take() {
            window_handle.close();
        }
    }

    /// Define o fator de escala absoluto para a GUI.
    fn set_scale(&mut self, _scale: f64) -> Result<(), PluginError> {
        Ok(())
    }

    /// Retorna o tamanho fixo da GUI (600x260 pixels).
    fn get_size(&mut self) -> Option<GuiSize> {
        Some(GuiSize {
            width: 600,
            height: 260,
        })
    }

    /// Define o tamanho da GUI. Aceita apenas o tamanho fixo 600x260.
    fn set_size(&mut self, size: GuiSize) -> Result<(), PluginError> {
        if size.width == 600 && size.height == 260 {
            Ok(())
        } else {
            Err(PluginError::Message(
                "Resizing da GUI não é suportado nesta versão",
            ))
        }
    }

    /// Define a janela pai (host) onde a GUI deve ser embutida.
    fn set_parent(&mut self, _window: Window) -> Result<(), PluginError> {
        #[cfg(feature = "clap-plugin")]
        {
            use crate::clap::gui::window::NamPluginWindow;

            if let Some(mut old_handle) = self.window_handle.take() {
                old_handle.close();
            }

            let options = baseview::WindowOpenOptions {
                // Título vazio: o host (Bitwig) já exibe o nome do plugin no frame da janela.
                // Usar um título aqui causaria duplicação: "NAM-rs / NAM-rs Neural Amp Modeler".
                title: String::new(),
                size: baseview::Size::new(600.0, 260.0),
                scale: baseview::WindowScalePolicy::SystemScaleFactor,
                gl_config: Some(baseview::gl::GlConfig::default()),
            };

            let shared_ptr = crate::clap::plugin::NamClapSharedRef(self.shared);
            let host_shared = self.host.shared();
            let host_static: clack_plugin::host::HostSharedHandle<'static> =
                unsafe { std::mem::transmute(host_shared) };

            let window_handle = baseview::Window::open_parented(&_window, options, move |win| {
                NamPluginWindow::new(win, shared_ptr, host_static)
            });

            self.window_handle = Some(window_handle);
        }
        Ok(())
    }

    /// Configura a janela para flutuar acima da janela especificada (não suportado).
    fn set_transient(&mut self, _window: Window) -> Result<(), PluginError> {
        Err(PluginError::Message(
            "Modo flutuante (floating) não é suportado",
        ))
    }

    /// Torna a janela da GUI visível.
    fn show(&mut self) -> Result<(), PluginError> {
        Ok(())
    }

    /// Oculta a janela da GUI.
    fn hide(&mut self) -> Result<(), PluginError> {
        Ok(())
    }

    /// Informa se o tamanho da janela pode ser alterado (tamanho fixo).
    fn can_resize(&mut self) -> bool {
        false
    }
}

/// Tipo marcador para registro da extensão.
pub type NamPluginGui = PluginGui;
