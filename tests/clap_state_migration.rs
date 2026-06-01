// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![cfg(feature = "clap-plugin")]

use clack_extensions::state::PluginState;
use clack_host::prelude::*;
use nam_rs::clap::plugin::NamClapPlugin;

struct TestHostShared;
impl<'a> SharedHandler<'a> for TestHostShared {
    fn request_restart(&self) {}
    fn request_process(&self) {}
    fn request_callback(&self) {}
}

struct TestHost;
impl HostHandlers for TestHost {
    type Shared<'a> = TestHostShared;
    type MainThread<'a> = ();
    type AudioProcessor<'a> = ();
}

fn create_plugin_instance() -> PluginInstance<TestHost> {
    let entry =
        PluginEntry::load_from_clack::<clack_plugin::entry::SinglePluginEntry<NamClapPlugin>>(
            c"/test",
        )
        .expect("Falha ao carregar PluginEntry");

    let host_info = HostInfo::new(
        "NAM-rs-Test",
        "NAM",
        "https://github.com/fabiohl/nam-rs",
        "0.1.0",
    )
    .expect("Falha ao criar HostInfo");

    PluginInstance::<TestHost>::new(
        |_| TestHostShared,
        |_| (),
        &entry,
        c"br.eti.fabiolima.nam-rs",
        &host_info,
    )
    .expect("Falha ao instanciar plugin")
}

#[test]
fn test_integration_v0_legacy_load() {
    let mut instance = create_plugin_instance();
    let state_ext = instance
        .plugin_handle()
        .get_extension::<PluginState>()
        .expect("PluginState extension not found");

    // Payload v0 sem envelope (JSON puro de NamPluginParams)
    let v0_json = r#"{"input_gain_db": 3.0,"output_gain_db": -6.0,"gate_threshold_db": -50.0,"model_path": null,"bypass": true}"#;

    {
        let mut handle = instance.plugin_handle();
        state_ext
            .load(&mut handle, &mut v0_json.as_bytes())
            .expect("Falha ao carregar estado v0");
    }

    // Verifica que o estado interno na main thread foi devidamente atualizado e migrado
    let raw_plugin_ptr = instance.plugin_handle().as_raw_ptr();
    let shared_ptr = unsafe {
        clack_plugin::extensions::wrapper::PluginWrapper::<NamClapPlugin>::handle(
            raw_plugin_ptr,
            |wrapper| Ok(wrapper.shared() as *const nam_rs::clap::plugin::NamClapShared),
        )
        .expect("Falha ao obter wrapper do plugin")
    };
    let shared = unsafe { &*shared_ptr };

    assert_eq!(
        f32::from_bits(
            shared
                .param_input_gain
                .load(std::sync::atomic::Ordering::Relaxed)
        ),
        3.0
    );
    assert_eq!(
        f32::from_bits(
            shared
                .param_output_gain
                .load(std::sync::atomic::Ordering::Relaxed)
        ),
        -6.0
    );
    assert_eq!(
        f32::from_bits(
            shared
                .param_gate_thresh
                .load(std::sync::atomic::Ordering::Relaxed)
        ),
        -50.0
    );
    assert_eq!(
        shared
            .param_bypass
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );
}

#[test]
fn test_integration_v1_round_trip() {
    let mut instance = create_plugin_instance();
    let state_ext = instance
        .plugin_handle()
        .get_extension::<PluginState>()
        .expect("PluginState extension not found");

    // 1. Modificar valores atômicos do plugin simulando ação do usuário
    let raw_plugin_ptr = instance.plugin_handle().as_raw_ptr();
    let shared_ptr = unsafe {
        clack_plugin::extensions::wrapper::PluginWrapper::<NamClapPlugin>::handle(
            raw_plugin_ptr,
            |wrapper| Ok(wrapper.shared() as *const nam_rs::clap::plugin::NamClapShared),
        )
        .expect("Falha ao obter wrapper do plugin")
    };
    let shared = unsafe { &*shared_ptr };

    shared
        .param_input_gain
        .store(1.5f32.to_bits(), std::sync::atomic::Ordering::Relaxed);
    shared
        .param_output_gain
        .store((-2.0f32).to_bits(), std::sync::atomic::Ordering::Relaxed);
    shared
        .param_gate_thresh
        .store((-65.0f32).to_bits(), std::sync::atomic::Ordering::Relaxed);
    shared
        .param_bypass
        .store(0, std::sync::atomic::Ordering::Relaxed);

    // 2. Salvar estado
    let mut output_buf = Vec::new();
    {
        let mut handle = instance.plugin_handle();
        state_ext
            .save(&mut handle, &mut output_buf)
            .expect("Falha ao salvar estado");
    }

    // 3. Resetar valores atômicos no plugin
    shared
        .param_input_gain
        .store(0.0f32.to_bits(), std::sync::atomic::Ordering::Relaxed);
    shared
        .param_output_gain
        .store(0.0f32.to_bits(), std::sync::atomic::Ordering::Relaxed);
    shared
        .param_gate_thresh
        .store(0.0f32.to_bits(), std::sync::atomic::Ordering::Relaxed);
    shared
        .param_bypass
        .store(1, std::sync::atomic::Ordering::Relaxed);

    // 4. Carregar estado salvo
    {
        let mut handle = instance.plugin_handle();
        state_ext
            .load(&mut handle, &mut output_buf.as_slice())
            .expect("Falha ao carregar estado salvo");
    }

    // 5. Verificar que valores foram devidamente restaurados
    assert_eq!(
        f32::from_bits(
            shared
                .param_input_gain
                .load(std::sync::atomic::Ordering::Relaxed)
        ),
        1.5
    );
    assert_eq!(
        f32::from_bits(
            shared
                .param_output_gain
                .load(std::sync::atomic::Ordering::Relaxed)
        ),
        -2.0
    );
    assert_eq!(
        f32::from_bits(
            shared
                .param_gate_thresh
                .load(std::sync::atomic::Ordering::Relaxed)
        ),
        -65.0
    );
    assert_eq!(
        shared
            .param_bypass
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
}

#[test]
fn test_integration_forward_v1_to_v2() {
    let mut instance = create_plugin_instance();
    let state_ext = instance
        .plugin_handle()
        .get_extension::<PluginState>()
        .expect("PluginState extension not found");

    // Payload com versão futura (version: 2) e parâmetros adicionais hipotéticos
    let v2_json = r#"{
        "version": 2,
        "params": {
            "input_gain_db": 4.5,
            "output_gain_db": -5.0,
            "gate_threshold_db": -60.0,
            "model_path": null,
            "model_basename": null,
            "model_search_paths": [],
            "bypass": false,
            "new_future_gain": 12.0
        }
    }"#;

    // Deve carregar com sucesso aplicando a migração (ou ignorando campos desconhecidos) sem panics
    {
        let mut handle = instance.plugin_handle();
        state_ext
            .load(&mut handle, &mut v2_json.as_bytes())
            .expect("Falha ao processar envelope futuro v2");
    }

    let raw_plugin_ptr = instance.plugin_handle().as_raw_ptr();
    let shared_ptr = unsafe {
        clack_plugin::extensions::wrapper::PluginWrapper::<NamClapPlugin>::handle(
            raw_plugin_ptr,
            |wrapper| Ok(wrapper.shared() as *const nam_rs::clap::plugin::NamClapShared),
        )
        .expect("Falha ao obter wrapper do plugin")
    };
    let shared = unsafe { &*shared_ptr };

    assert_eq!(
        f32::from_bits(
            shared
                .param_input_gain
                .load(std::sync::atomic::Ordering::Relaxed)
        ),
        4.5
    );
    assert_eq!(
        f32::from_bits(
            shared
                .param_output_gain
                .load(std::sync::atomic::Ordering::Relaxed)
        ),
        -5.0
    );
    assert_eq!(
        f32::from_bits(
            shared
                .param_gate_thresh
                .load(std::sync::atomic::Ordering::Relaxed)
        ),
        -60.0
    );
    assert_eq!(
        shared
            .param_bypass
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
}
