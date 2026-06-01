// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Pipeline DSP de reprodução e configurações do host PipeWire (standalone).

use super::bridge::DspBridgeReader;

#[cfg(feature = "standalone")]
use crate::common::diagnostics::SystemSnapshot;
#[cfg(feature = "standalone")]
use minstant::Anchor;
#[cfg(feature = "standalone")]
use pipewire as pw;

/// das instâncias essenciais do PipeWire (`StreamBox` e `Listener`).
#[cfg(feature = "standalone")]
pub(crate) struct AppState<S1, L1, S2, L2> {
    #[allow(dead_code)]
    pub capture_stream: S1,
    #[allow(dead_code)]
    pub capture_listener: L1,
    #[allow(dead_code)]
    pub playback_stream: S2,
    #[allow(dead_code)]
    pub playback_listener: L2,
}

/// Configurações para inicialização do host PipeWire.
#[cfg(feature = "standalone")]
pub struct PipewireHostConfig {
    /// Tamanho do buffer de áudio solicitado.
    pub buffer_size: u32,
    /// Âncora de tempo para telemetria RDTSC.
    pub tsc_anchor: Anchor,
    /// Snapshot do sistema para diagnósticos.
    pub sys: SystemSnapshot,
}

/// Pipeline DSP de Reprodução (Bridge → Hardware).
#[cfg(feature = "standalone")]
#[inline(always)]
pub(crate) fn playback_dsp_cycle(
    stream: &pw::stream::Stream,
    bridge: DspBridgeReader,
    last_bridge_gen: &mut u64,
) {
    bridge.read_block(last_bridge_gen, |buf_l, buf_r| {
        let n_samples = buf_l.len();
        if n_samples == 0 {
            return;
        }

        // Pede ao sistema de som (PipeWire) um espaço vazio para colocar o áudio.
        let mut buf = match stream.dequeue_buffer() {
            Some(b) => b,
            None => return,
        };

        let datas = buf.datas_mut();
        if datas.len() < 2 {
            return;
        }

        // Separa os canais Esquerdo e Direito para a entrega final.
        let (datas_left, datas_right) = datas.split_at_mut(1);
        let data_l = &mut datas_left[0];
        let data_r = &mut datas_right[0];

        let max_l = data_l.as_raw().maxsize as usize / std::mem::size_of::<f32>();
        let max_r = data_r.as_raw().maxsize as usize / std::mem::size_of::<f32>();
        let n_out = n_samples.min(max_l).min(max_r);
        if n_out == 0 {
            return;
        }

        // Copia o som processado diretamente para as saídas da sua placa de som.
        if let Some(raw_l) = data_l.data() {
            let out_l =
                unsafe { std::slice::from_raw_parts_mut(raw_l.as_mut_ptr().cast::<f32>(), n_out) };
            out_l.copy_from_slice(&buf_l[..n_out]);
        }
        if let Some(raw_r) = data_r.data() {
            let out_r =
                unsafe { std::slice::from_raw_parts_mut(raw_r.as_mut_ptr().cast::<f32>(), n_out) };
            out_r.copy_from_slice(&buf_r[..n_out]);
        }

        // Informa ao hardware exatamente quanto de som foi entregue agora.
        {
            let chunk = data_l.chunk_mut();
            *chunk.size_mut() = (n_out * std::mem::size_of::<f32>()) as u32;
            *chunk.offset_mut() = 0;
            *chunk.stride_mut() = std::mem::size_of::<f32>() as i32;
        }
        {
            let chunk = data_r.chunk_mut();
            *chunk.size_mut() = (n_out * std::mem::size_of::<f32>()) as u32;
            *chunk.offset_mut() = 0;
            *chunk.stride_mut() = std::mem::size_of::<f32>() as i32;
        }
    });
}

/// Constrói um SPA Pod de formato de áudio F32P stereo para negociação PipeWire.
///
/// # Safety
/// O pod binário retornado aponta diretamente para o `format_buf` fornecido.
#[cfg(feature = "standalone")]
pub(crate) unsafe fn build_spa_format_pod<'a>(
    audio_info: &pw::spa::param::audio::AudioInfoRaw,
    format_buf: &'a mut [u8; 1024],
) -> anyhow::Result<&'a pw::spa::pod::Pod> {
    unsafe {
        let mut builder: pw::spa::sys::spa_pod_builder = std::mem::zeroed();
        // Prepara um "construtor" para criar o contrato de formato de áudio.
        pw::spa::sys::spa_pod_builder_init(
            &mut builder,
            format_buf.as_mut_ptr().cast(),
            format_buf.len() as u32,
        );

        // Constrói o documento binário (SPA Pod) que descreve o áudio (Ex: 48kHz, Estéreo).
        // Esse documento é o que o PipeWire usa para entender como enviar som para nós.
        let pod_ptr = pw::spa::sys::spa_format_audio_raw_build(
            &mut builder,
            pw::spa::param::ParamType::EnumFormat.as_raw(),
            &audio_info.as_raw(),
        );

        if pod_ptr.is_null() {
            // Se falhar, o sistema não saberá como negociar o som com a sua placa.
            return Err(anyhow::anyhow!(
                "Failed to build the audio negotiation document (SPA Pod)"
            ));
        }

        // Retorna o contrato pronto para ser assinado e usado pelo sistema.
        Ok(&*(pod_ptr as *const pw::spa::pod::Pod))
    }
}
