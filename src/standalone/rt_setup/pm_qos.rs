// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! PM QoS e detecção de hardware de áudio.
//!
//! Funções para bloquear C-States profundos da CPU e detectar
//! o sink de hardware padrão do sistema via PipeWire.

/// Detecta dinamicamente o sink de hardware padrão do sistema via `pw-metadata`.
///
/// Esta função tenta identificar para qual dispositivo físico o áudio deve ser enviado
/// por padrão. Ela faz o parsing da saída do utilitário `pw-metadata` do PipeWire.
///
/// Retorna `Some(name)` se encontrar um sink válido que não seja o próprio NAM-rs,
/// ou `None` caso contrário (permitindo que o roteamento seja decidido pelo WirePlumber).
pub fn detect_hardware_sink() -> Option<String> {
    // Executa o comando externo para ler metadados do servidor PipeWire
    let out = std::process::Command::new("pw-metadata")
        .args(["-n", "default", "0", "default.audio.sink"])
        .output()
        .ok()?;

    // Converte a saída bruta para string (UTF-8 com perdas)
    let s = String::from_utf8_lossy(&out.stdout);

    // Parsing manual: localiza a chave "name" na saída JSON-like
    let start = s.find("\"name\":\"")?;
    let rest = &s[start + 8..];
    let end = rest.find('"')?;
    let name = &rest[..end];

    // Evitamos o "loop infinito" de roteamento se o default detectado for o próprio input do NAM-rs.
    if name == "NAM-rs-input" || name == "NAM-rs-standalone" {
        None
    } else {
        // Retorna o nome do hardware real (ex: 'alsa_output.pci-0000_00_1f.3.analog-stereo')
        Some(name.to_string())
    }
}

/// Impede que o processador entre em C-States de economia de energia,
/// garantindo latência de despertar de 0ms para processamento de áudio RT.
///
/// **Aviso:** Esta proteção é **sistêmica (global)** e afeta todos os cores da CPU,
/// não apenas a thread que executa esta função.
///
/// Utiliza a interface PM QoS do kernel Linux para solicitar latência zero.
///
/// RETORNO: O arquivo `File`. Ele DEVE ser mantido vivo no escopo principal.
/// Se o descritor de arquivo for fechado (drop), o kernel anula a proteção.
pub fn lock_cpu_c_states() -> Option<std::fs::File> {
    match std::fs::OpenOptions::new()
        .write(true)
        .open("/dev/cpu_dma_latency")
    {
        Ok(mut file) => {
            // Valor 0 indica tolerância zero a latência de transição de energia.
            let zero: i32 = 0;
            if std::io::Write::write_all(&mut file, &zero.to_ne_bytes()).is_ok() {
                log::info!("⚡ PM QoS Lock: Deep CPU C-States disabled (Zero DMA Latency).");
                return Some(file);
            }
            log::warn!("PM QoS: Failed to write to /dev/cpu_dma_latency.");
            None
        }
        Err(e) => {
            // Frequentemente falha se não houver permissão de escrita ou se o arquivo não existir.
            log::warn!(
                "PM QoS: Access denied to /dev/cpu_dma_latency ({}). \
                 Consider creating a udev rule for the 'audio' group.",
                e
            );
            None
        }
    }
}
