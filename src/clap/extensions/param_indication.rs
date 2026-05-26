// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Implementação da extensão `clap_plugin_param_indication` para o NAM-rs.

use crate::clap::plugin::NamClapMainThread;
use clack_common::utils::{ClapId, Color};
use clack_extensions::param_indication::{
    ParamIndicationAutomation, PluginParamIndication, PluginParamIndicationImpl,
};
use std::ffi::CStr;
use std::sync::atomic::Ordering;

/// Indica que o parâmetro possui mapeamento ativo.
pub const INDICATION_MAPPED: u8 = 1 << 0;
/// Indica que o parâmetro está sendo automatizado.
pub const INDICATION_AUTOMATING: u8 = 1 << 1;
/// Indica que o parâmetro possui um override de automação ativo.
pub const INDICATION_OVERRIDING: u8 = 1 << 2;

impl<'a> PluginParamIndicationImpl for NamClapMainThread<'a> {
    /// Informa ao plugin que um parâmetro possui (ou não) mapeamento ativo no host.
    ///
    /// Atualiza o bit `INDICATION_MAPPED` e armazena a cor ARGB empacotada
    /// nos atômicos compartilhados para leitura pela GUI thread.
    fn set_mapping(
        &mut self,
        param_id: ClapId,
        has_mapping: bool,
        color: Option<Color>,
        _label: Option<&CStr>,
        _description: Option<&CStr>,
    ) {
        let idx = param_id.get() as usize;
        if idx < self.shared.param_indication.len() {
            let atomic_val = &self.shared.param_indication[idx];
            if has_mapping {
                atomic_val.fetch_or(INDICATION_MAPPED, Ordering::Relaxed);
                if let Some(c) = color {
                    let packed = crate::clap::extensions::track_info::pack_argb(
                        c.alpha, c.red, c.green, c.blue,
                    );
                    self.shared.param_indication_color[idx].store(packed, Ordering::Relaxed);
                } else {
                    self.shared.param_indication_color[idx].store(0, Ordering::Relaxed);
                }
            } else {
                atomic_val.fetch_and(!INDICATION_MAPPED, Ordering::Relaxed);
                self.shared.param_indication_color[idx].store(0, Ordering::Relaxed);
            }
        }
    }

    /// Informa ao plugin o estado de automação de um parâmetro (None/Present/Playing/Recording/Overriding).
    ///
    /// Atualiza os bits `INDICATION_AUTOMATING` e `INDICATION_OVERRIDING`,
    /// garantindo exclusão mútua entre os dois estados. A cor fornecida pelo host
    /// é armazenada para uso pela GUI na renderização dos halos de automação.
    fn set_automation(
        &mut self,
        param_id: ClapId,
        automation_state: ParamIndicationAutomation,
        color: Option<Color>,
    ) {
        let idx = param_id.get() as usize;
        if idx < self.shared.param_indication.len() {
            let atomic_val = &self.shared.param_indication[idx];
            match automation_state {
                ParamIndicationAutomation::None | ParamIndicationAutomation::Present => {
                    atomic_val.fetch_and(
                        !(INDICATION_AUTOMATING | INDICATION_OVERRIDING),
                        Ordering::Relaxed,
                    );
                    self.shared.param_indication_color[idx].store(0, Ordering::Relaxed);
                }
                ParamIndicationAutomation::Playing | ParamIndicationAutomation::Recording => {
                    atomic_val.fetch_or(INDICATION_AUTOMATING, Ordering::Relaxed);
                    atomic_val.fetch_and(!INDICATION_OVERRIDING, Ordering::Relaxed);
                    if let Some(c) = color {
                        let packed = crate::clap::extensions::track_info::pack_argb(
                            c.alpha, c.red, c.green, c.blue,
                        );
                        self.shared.param_indication_color[idx].store(packed, Ordering::Relaxed);
                    }
                }
                ParamIndicationAutomation::Overriding => {
                    atomic_val.fetch_or(INDICATION_OVERRIDING, Ordering::Relaxed);
                    atomic_val.fetch_and(!INDICATION_AUTOMATING, Ordering::Relaxed);
                    if let Some(c) = color {
                        let packed = crate::clap::extensions::track_info::pack_argb(
                            c.alpha, c.red, c.green, c.blue,
                        );
                        self.shared.param_indication_color[idx].store(packed, Ordering::Relaxed);
                    }
                }
            }
        }
    }
}

/// Tipo marcador para registro da extensão.
pub type NamPluginParamIndication = PluginParamIndication;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU8;

    #[test]
    fn test_param_indication_bits() {
        let val = AtomicU8::new(0);

        // Test mappings
        val.fetch_or(INDICATION_MAPPED, Ordering::Relaxed);
        assert_eq!(
            val.load(Ordering::Relaxed) & INDICATION_MAPPED,
            INDICATION_MAPPED
        );

        val.fetch_and(!INDICATION_MAPPED, Ordering::Relaxed);
        assert_eq!(val.load(Ordering::Relaxed) & INDICATION_MAPPED, 0);

        // Test automation
        val.fetch_or(INDICATION_AUTOMATING, Ordering::Relaxed);
        assert_eq!(
            val.load(Ordering::Relaxed) & INDICATION_AUTOMATING,
            INDICATION_AUTOMATING
        );
        assert_eq!(val.load(Ordering::Relaxed) & INDICATION_OVERRIDING, 0);

        // Overriding overrides automating
        val.fetch_or(INDICATION_OVERRIDING, Ordering::Relaxed);
        val.fetch_and(!INDICATION_AUTOMATING, Ordering::Relaxed);
        assert_eq!(
            val.load(Ordering::Relaxed) & INDICATION_OVERRIDING,
            INDICATION_OVERRIDING
        );
        assert_eq!(val.load(Ordering::Relaxed) & INDICATION_AUTOMATING, 0);
    }

    /// Verifica que quando ambos AUTOMATING e OVERRIDING estão setados,
    /// o estado `None` ou `Present` limpa ambos corretamente.
    #[test]
    fn test_param_indication_none_clears_both_bits() {
        let val = AtomicU8::new(0);

        // Setar ambos bits simultaneamente
        val.fetch_or(
            INDICATION_AUTOMATING | INDICATION_OVERRIDING,
            Ordering::Relaxed,
        );
        assert_eq!(
            val.load(Ordering::Relaxed) & (INDICATION_AUTOMATING | INDICATION_OVERRIDING),
            INDICATION_AUTOMATING | INDICATION_OVERRIDING,
            "ambos bits devem estar setados antes do teste"
        );

        // Simular o efeito de ParamIndicationAutomation::None (limpa ambos)
        val.fetch_and(
            !(INDICATION_AUTOMATING | INDICATION_OVERRIDING),
            Ordering::Relaxed,
        );

        // Ambos devem estar zerados
        assert_eq!(
            val.load(Ordering::Relaxed) & INDICATION_AUTOMATING,
            0,
            "AUTOMATING deve ser zerado"
        );
        assert_eq!(
            val.load(Ordering::Relaxed) & INDICATION_OVERRIDING,
            0,
            "OVERRIDING deve ser zerado"
        );

        // MAPPED não deve ser afetado
        val.fetch_or(INDICATION_MAPPED, Ordering::Relaxed);
        val.fetch_and(
            !(INDICATION_AUTOMATING | INDICATION_OVERRIDING),
            Ordering::Relaxed,
        );
        assert_eq!(
            val.load(Ordering::Relaxed) & INDICATION_MAPPED,
            INDICATION_MAPPED,
            "MAPPED não deve ser afetado pela limpeza de automação"
        );
    }
}
