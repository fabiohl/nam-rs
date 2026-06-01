// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Detecção de topologias WaveNet e LSTM a partir dos dados do modelo.

use super::data::NamModelData;

/// As Topologias fechadas e suportadas dentro da modelagem WaveNet nativa.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NamWavenetTopology {
    /// Canais: 16 (Standard)
    Standard,
    /// Canais: 12 (Lite)
    Lite,
    /// Canais: 8 (Feather)
    Feather,
    /// Canais: 4 (Nano)
    Nano,
}

static STD_DILATIONS: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
static LITE_DILATIONS: &[usize] = &[1, 2, 4, 8, 16, 32, 64];
static LITE_DILATIONS_2: &[usize] = &[128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512];

/// Realiza o parsing de uma string de versão no formato SemVer mínimo.
/// Suporta sufixos de pré-lançamento ou metadados e prefixo 'v' ou 'V'.
/// Retorna `Some((major, minor, patch))` ou `None` em caso de falha de parsing.
pub(crate) fn parse_semver(version: &str) -> Option<(u16, u16, u16)> {
    let clean = version.trim().trim_start_matches(['v', 'V']);
    let clean = clean.split('-').next()?.split('+').next()?;
    let mut parts = clean.split('.');
    let major = parts.next()?.trim().parse::<u16>().ok()?;
    let minor = parts.next().unwrap_or("0").trim().parse::<u16>().ok()?;
    let patch = parts.next().unwrap_or("0").trim().parse::<u16>().ok()?;
    Some((major, minor, patch))
}

impl NamModelData {
    /// Detecta se o modelo utiliza a arquitetura WaveNet A2.
    ///
    /// Um modelo é considerado A2 se a arquitetura for "WaveNet" e:
    /// 1. A versão declarada for >= 0.6.0.
    /// 2. Utilizar funções de ativação diferentes de "Tanh".
    pub fn is_wavenet_a2(&self) -> bool {
        if self.architecture != "WaveNet" {
            return false;
        }

        if let Some(ref v) = self.version
            && let Some(ver) = parse_semver(v)
            && ver >= (0, 6, 0)
        {
            return true;
        }

        for layer in &self.config.layers {
            if let Some(ref act) = layer.activation
                && act != "Tanh"
            {
                return true;
            }
        }

        false
    }
}

/// Baseando-se no NeuralModel.cpp (`L:155-218`), verifica a identidade estática da topologia WaveNet.
pub fn get_wavenet_topology(data: &NamModelData) -> Option<NamWavenetTopology> {
    if data.architecture != "WaveNet" || data.config.layers.len() != 2 {
        return None;
    }

    let l0 = &data.config.layers[0];
    let l1 = &data.config.layers[1];

    let l0_gated = l0.gated.unwrap_or(false);
    let l1_gated = l1.gated.unwrap_or(false);
    let l0_head_bias = l0.head_bias.unwrap_or(false);
    let l1_head_bias = l1.head_bias.unwrap_or(false);

    if l0_gated || l1_gated || l0_head_bias || !l1_head_bias {
        return None;
    }

    let channels = l0.channels?;
    let dils_0 = l0.dilations.as_deref()?;
    let dils_1 = l1.dilations.as_deref()?;

    match channels {
        16 if dils_0 == STD_DILATIONS && dils_1 == STD_DILATIONS => {
            Some(NamWavenetTopology::Standard)
        }
        12 if dils_0 == LITE_DILATIONS && dils_1 == LITE_DILATIONS_2 => {
            Some(NamWavenetTopology::Lite)
        }
        8 if dils_0 == LITE_DILATIONS && dils_1 == LITE_DILATIONS_2 => {
            Some(NamWavenetTopology::Feather)
        }
        4 if dils_0 == LITE_DILATIONS && dils_1 == LITE_DILATIONS_2 => {
            Some(NamWavenetTopology::Nano)
        }
        _ => None,
    }
}

/// Verifica e retorna a geometria do LSTM (num_layers, hidden_size).
pub fn get_lstm_topology(data: &NamModelData) -> Option<(usize, usize)> {
    if data.architecture != "LSTM" {
        return None;
    }

    let num_layers = data.config.num_layers?;
    let hidden_size = data.config.hidden_size?;
    Some((num_layers, hidden_size))
}
