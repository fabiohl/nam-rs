// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Parâmetros de configuração para a arquitetura WaveNet A2.
//!
//! Este módulo contém as estruturas que descrevem a topologia de um modelo WaveNet A2,
//! permitindo a construção e validação das camadas de inferência.
//!
//! IMPORTANTE: O suporte à arquitetura A2 está em estágio de "placeholder"
//! aguardando estabilização da implementação de referência.

use crate::models::activations::ActivationType;
use crate::models::film::FiLMConfig;
use crate::models::gating::GatingMode;

/// Parâmetros para configuração do Head 1x1.
///
/// Configura uma convolução 1x1 opcional que envia a saída diretamente para o head
/// (skip connection) em vez de usar a saída da ativação diretamente.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Head1x1Params {
    /// Se a convolução head 1x1 está ativa.
    pub active: bool,
    /// Número de canais de saída para a convolução head 1x1.
    pub out_channels: usize,
    /// Número de grupos para a convolução agrupada.
    pub groups: u32,
}

/// Parâmetros para configuração da Camada 1x1.
///
/// Configura uma convolução 1x1 opcional que processa a saída da ativação
/// para a conexão residual com a próxima camada.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layer1x1Params {
    /// Se a convolução de camada 1x1 está ativa.
    pub active: bool,
    /// Número de grupos para a convolução agrupada.
    pub groups: u32,
}

/// Parâmetros para construção de uma única camada WaveNet A2.
///
/// Contém toda a configuração necessária para instanciar uma camada detalhada.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerParamsA2 {
    /// Tamanho da entrada de condicionamento.
    pub condition_size: usize,
    /// Número de canais de entrada/saída entre as camadas.
    pub channels: usize,
    /// Número de canais internos (bottleneck).
    pub bottleneck: usize,
    /// Tamanho do kernel para a convolução dilatada.
    pub kernel_size: usize,
    /// Fator de dilatação para a convolução.
    pub dilation: usize,
    /// Configuração da função de ativação primária.
    pub activation: ActivationType,
    /// Modo de gating (None, Gated ou Blended).
    pub gating_mode: GatingMode,
    /// Número de grupos para a convolução de entrada.
    pub groups_input: u32,
    /// Número de grupos para a convolução de mixin de entrada.
    pub groups_input_mixin: u32,
    /// Configuração da convolução opcional layer 1x1.
    pub layer1x1: Layer1x1Params,
    /// Configuração da convolução opcional head 1x1.
    pub head1x1: Head1x1Params,
    /// Ativação secundária (utilizada para gating/blending).
    pub secondary_activation: ActivationType,
    /// Parâmetros FiLM antes da convolução de entrada.
    pub conv_pre_film: FiLMConfig,
    /// Parâmetros FiLM após a convolução de entrada.
    pub conv_post_film: FiLMConfig,
    /// Parâmetros FiLM antes do mixin de entrada.
    pub input_mixin_pre_film: FiLMConfig,
    /// Parâmetros FiLM após o mixin de entrada.
    pub input_mixin_post_film: FiLMConfig,
    /// Parâmetros FiLM antes da ativação.
    pub activation_pre_film: FiLMConfig,
    /// Parâmetros FiLM após a ativação.
    pub activation_post_film: FiLMConfig,
    /// Parâmetros FiLM após a convolução layer 1x1.
    pub layer1x1_post_film: FiLMConfig,
    /// Parâmetros FiLM após a convolução head 1x1.
    pub head1x1_post_film: FiLMConfig,
}

/// Parâmetros para construção de um Array de Camadas (LayerArray) WaveNet A2.
///
/// Configura múltiplas camadas que compartilham a mesma contagem de canais
/// e tamanho de kernel, mas podem ter dilatações e ativações distintas.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerArrayParamsA2 {
    /// Tamanho da entrada (número de canais).
    pub input_size: usize,
    /// Tamanho da entrada de condicionamento.
    pub condition_size: usize,
    /// Tamanho da saída do head (após o rechannel).
    pub head_size: usize,
    /// Tamanho do kernel da convolução de rechannel do head (>= 1).
    pub head_kernel_size: usize,
    /// Número de canais em cada camada.
    pub channels: usize,
    /// Tamanho do bottleneck (contagem de canais internos).
    pub bottleneck: usize,
    /// Tamanhos de kernel por camada.
    pub kernel_sizes: Vec<usize>,
    /// Vetor de fatores de dilatação, um por camada.
    pub dilations: Vec<usize>,
    /// Vetor de configurações de ativação primária, uma por camada.
    pub activations: Vec<ActivationType>,
    /// Vetores de modos de gating, um por camada.
    pub gating_modes: Vec<GatingMode>,
    /// Se deve utilizar bias no rechannel do head.
    pub head_bias: bool,
    /// Número de grupos para as convoluções de entrada.
    pub groups_input: u32,
    /// Número de grupos para as convoluções de mixin de entrada.
    pub groups_input_mixin: u32,
    /// Parâmetros para as convoluções opcionais layer 1x1.
    pub layer1x1: Layer1x1Params,
    /// Parâmetros para as convoluções opcionais head 1x1.
    pub head1x1: Head1x1Params,
    /// Vetor de configurações de ativação secundária para gating/blending.
    pub secondary_activations: Vec<ActivationType>,
    /// Parâmetros FiLM antes das convoluções de entrada.
    pub conv_pre_film: FiLMConfig,
    /// Parâmetros FiLM após as convoluções de entrada.
    pub conv_post_film: FiLMConfig,
    /// Parâmetros FiLM antes dos mixins de entrada.
    pub input_mixin_pre_film: FiLMConfig,
    /// Parâmetros FiLM após os mixins de entrada.
    pub input_mixin_post_film: FiLMConfig,
    /// Parâmetros FiLM antes da ativação.
    pub activation_pre_film: FiLMConfig,
    /// Parâmetros FiLM após a ativação.
    pub activation_post_film: FiLMConfig,
    /// Parâmetros FiLM após as convoluções layer 1x1.
    pub layer1x1_post_film: FiLMConfig,
    /// Parâmetros FiLM após as convoluções head 1x1.
    pub head1x1_post_film: FiLMConfig,
}

/// Parâmetros para o Head opcional pós-stack.
///
/// Corresponde ao componente `Head` do WaveNet no NAM.
#[derive(Debug, Clone, PartialEq)]
pub struct HeadParams {
    /// Canais de entrada (geralmente herdados da última camada).
    pub in_channels: usize,
    /// Canais internos do head.
    pub channels: usize,
    /// Canais de saída final.
    pub out_channels: usize,
    /// Tamanhos de kernel das convoluções do head.
    pub kernel_sizes: Vec<usize>,
    /// Configuração da ativação do head.
    pub activation: ActivationType,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_params_a2_construction() {
        let _params = LayerParamsA2 {
            condition_size: 1,
            channels: 16,
            bottleneck: 16,
            kernel_size: 3,
            dilation: 1,
            activation: ActivationType::Tanh,
            gating_mode: GatingMode::None,
            groups_input: 1,
            groups_input_mixin: 1,
            layer1x1: Layer1x1Params {
                active: false,
                groups: 1,
            },
            head1x1: Head1x1Params {
                active: false,
                out_channels: 16,
                groups: 1,
            },
            secondary_activation: ActivationType::Sigmoid,
            conv_pre_film: FiLMConfig::default(),
            conv_post_film: FiLMConfig::default(),
            input_mixin_pre_film: FiLMConfig::default(),
            input_mixin_post_film: FiLMConfig::default(),
            activation_pre_film: FiLMConfig::default(),
            activation_post_film: FiLMConfig::default(),
            layer1x1_post_film: FiLMConfig::default(),
            head1x1_post_film: FiLMConfig::default(),
        };
    }

    #[test]
    fn test_layer_array_params_a2_construction() {
        let _params = LayerArrayParamsA2 {
            input_size: 1,
            condition_size: 1,
            head_size: 1,
            head_kernel_size: 1,
            channels: 16,
            bottleneck: 16,
            kernel_sizes: vec![3, 3],
            dilations: vec![1, 2],
            activations: vec![ActivationType::Tanh, ActivationType::Tanh],
            gating_modes: vec![GatingMode::None, GatingMode::None],
            head_bias: true,
            groups_input: 1,
            groups_input_mixin: 1,
            layer1x1: Layer1x1Params {
                active: false,
                groups: 1,
            },
            head1x1: Head1x1Params {
                active: false,
                out_channels: 16,
                groups: 1,
            },
            secondary_activations: vec![ActivationType::Sigmoid, ActivationType::Sigmoid],
            conv_pre_film: FiLMConfig::default(),
            conv_post_film: FiLMConfig::default(),
            input_mixin_pre_film: FiLMConfig::default(),
            input_mixin_post_film: FiLMConfig::default(),
            activation_pre_film: FiLMConfig::default(),
            activation_post_film: FiLMConfig::default(),
            layer1x1_post_film: FiLMConfig::default(),
            head1x1_post_film: FiLMConfig::default(),
        };
    }

    #[test]
    fn test_head_params_construction() {
        let _params = HeadParams {
            in_channels: 16,
            channels: 16,
            out_channels: 1,
            kernel_sizes: vec![1, 1],
            activation: ActivationType::Tanh,
        };
    }
}
