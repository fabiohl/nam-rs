// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! NAM-rs plugin definition and its CLAP lifecycle components.

mod shared;
pub use shared::{
    ClapParamPayload, ColdShared, NamClapShared, NamClapSharedRef, NamModelMetadata, PendingModel,
    RENDER_MODE_OFFLINE, RENDER_MODE_REALTIME, RtToUi, UiToRt,
};

mod main_thread;
pub use main_thread::{NamClapMainThread, debug_assert_main_thread};

use crate::clap::descriptor::nam_descriptor;
use crate::clap::processor::NamClapProcessor;
use crate::common::diagnostics::SystemSnapshot;
use crate::common::params::NamPluginParams;
use crate::common::spsc::{GcOverflowBuffer, RtStatusFlags};
use clack_plugin::prelude::*;
use rtrb::RingBuffer;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

/// NAM-rs plugin: main entry point for the CLAP lifecycle.
pub struct NamClapPlugin;

impl Plugin for NamClapPlugin {
    type AudioProcessor<'a> = NamClapProcessor<'a>;
    type Shared<'a> = NamClapShared;
    type MainThread<'a> = NamClapMainThread<'a>;

    fn declare_extensions(
        builder: &mut PluginExtensions<Self>,
        _shared: Option<&Self::Shared<'_>>,
    ) {
        builder.register::<clack_extensions::audio_ports::PluginAudioPorts>();
        builder.register::<clack_extensions::params::PluginParams>();
        builder.register::<clack_extensions::state::PluginState>();
        builder.register::<crate::clap::extensions::latency::NamPluginLatency>();
        builder.register::<crate::clap::extensions::track_info::NamPluginTrackInfo>();
        builder.register::<crate::clap::extensions::remote_controls::NamPluginRemoteControls>();
        builder.register::<crate::clap::extensions::param_indication::NamPluginParamIndication>();
        builder.register::<clack_extensions::preset_discovery::PluginPresetLoad>();
        builder.register::<crate::clap::extensions::render::NamPluginRender>();
        builder.register::<crate::clap::extensions::state_context::NamPluginStateContext>();

        #[cfg(feature = "clap-plugin")]
        builder.register::<crate::clap::extensions::gui::NamPluginGui>();
    }
}

impl DefaultPluginFactory for NamClapPlugin {
    fn get_descriptor() -> PluginDescriptor {
        nam_descriptor()
    }

    fn new_shared(_host: HostSharedHandle<'_>) -> Result<Self::Shared<'_>, PluginError> {
        static INIT_PANIC_HOOK: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        INIT_PANIC_HOOK.get_or_init(|| {
            crate::common::panic_hook::install_panic_hook("clap");
        });

        let (param_tx, param_rx) = RingBuffer::new(8);
        let (gc_tx, gc_rx) = RingBuffer::new(32); // Increased capacity for the plugin
        let (slimmable_tx, slimmable_rx) = RingBuffer::new(4);

        Ok(NamClapShared {
            rt_to_ui: RtToUi {
                ui_peak_l: AtomicU32::new(0.0f32.to_bits()),
                ui_peak_r: AtomicU32::new(0.0f32.to_bits()),
                ui_clipped: std::sync::atomic::AtomicBool::new(false),
                current_latency: AtomicU32::new(0),
                active_channel_count: AtomicU32::new(1),
            },
            ui_to_rt: UiToRt {
                param_input_gain: AtomicU32::new(0.0f32.to_bits()),
                param_output_gain: AtomicU32::new(0.0f32.to_bits()),
                param_gate_thresh: AtomicU32::new((-70.0f32).to_bits()),
                param_bypass: AtomicU32::new(0),
                param_adaptive_compute: AtomicU32::new(1), // Conservative by default in CLAP plugin
                param_slim_override: AtomicU32::new(0),    // Auto by default
                param_oversample: AtomicU32::new(0),       // Off by default
                param_activation: AtomicU32::new(1),       // Standard (exact-grade) by default
                gesture_flags: AtomicU32::new(0),
                gui_param_generation: AtomicU32::new(0),
            },
            cold: ColdShared {
                param_tx: Mutex::new(Some(param_tx)),
                param_rx: Mutex::new(Some(param_rx)),
                gc_tx: Mutex::new(Some(gc_tx)),
                gc_rx: Mutex::new(Some(gc_rx)),
                gc_overflow: Arc::new(GcOverflowBuffer::new(crate::common::spsc::SPSC_CAPACITY)),
                rt_status: Arc::new(RtStatusFlags::new()),
                model_sample_rate: AtomicU32::new(48000),
                sample_rate: AtomicU32::new(0),
                buffer_size: AtomicU32::new(0),
                track_accent_color: AtomicU32::new(0),
                param_indication: [
                    std::sync::atomic::AtomicU8::new(0),
                    std::sync::atomic::AtomicU8::new(0),
                    std::sync::atomic::AtomicU8::new(0),
                    std::sync::atomic::AtomicU8::new(0),
                    std::sync::atomic::AtomicU8::new(0),
                    std::sync::atomic::AtomicU8::new(0),
                    std::sync::atomic::AtomicU8::new(0),
                    std::sync::atomic::AtomicU8::new(0),
                    std::sync::atomic::AtomicU8::new(0),
                ],
                param_indication_color: [
                    std::sync::atomic::AtomicU32::new(0),
                    std::sync::atomic::AtomicU32::new(0),
                    std::sync::atomic::AtomicU32::new(0),
                    std::sync::atomic::AtomicU32::new(0),
                    std::sync::atomic::AtomicU32::new(0),
                    std::sync::atomic::AtomicU32::new(0),
                    std::sync::atomic::AtomicU32::new(0),
                    std::sync::atomic::AtomicU32::new(0),
                    std::sync::atomic::AtomicU32::new(0),
                ],
                model_load_counter: AtomicU32::new(0),
                ui_model_name: Mutex::new(String::new()),
                ui_model_metadata: Mutex::new(None),
                ui_pending_model: Mutex::new(None),
                ui_loading: std::sync::atomic::AtomicBool::new(false),
                ui_load_error: std::sync::atomic::AtomicBool::new(false),
                ui_load_error_msg: Mutex::new(String::new()),
                ui_model_info: Mutex::new(None),
                alive_fence: Arc::new(std::sync::atomic::AtomicBool::new(true)),
                render_mode: AtomicU32::new(0),
                gui_scale_factor: AtomicU32::new(0),
                ir_path: Mutex::new(None),
                ui_pending_ir: Mutex::new(None),
                ui_ir_loading: std::sync::atomic::AtomicBool::new(false),
                ui_ir_load_error: std::sync::atomic::AtomicBool::new(false),
                ui_ir_load_error_msg: Mutex::new(String::new()),
                ui_clear_ir: std::sync::atomic::AtomicBool::new(false),
                ir_raw_samples: Mutex::new(None),
                slimmable_tx: Mutex::new(Some(slimmable_tx)),
                slimmable_rx: Mutex::new(Some(slimmable_rx)),
                full_wavenet_model: Mutex::new(None),
                pending_model: Mutex::new(None),
                dialog_state: None,
                ir_dialog_state: None,
                dialog_handle_sink: Mutex::new(None),
                ir_dialog_handle_sink: Mutex::new(None),
            },
        })
    }

    fn new_main_thread<'a>(
        mut host: HostMainThreadHandle<'a>,
        shared: &'a Self::Shared<'a>,
    ) -> Result<Self::MainThread<'a>, PluginError> {
        // Initial track color query from the host
        if let Some(track_info_ext) =
            host.get_extension::<clack_extensions::track_info::HostTrackInfo>()
        {
            let mut buffer = clack_extensions::track_info::TrackInfoBuffer::new();
            if let Some(color) = track_info_ext
                .get(&mut host, &mut buffer)
                .and_then(|info| info.color())
            {
                let packed = crate::clap::extensions::track_info::pack_argb(
                    color.alpha,
                    color.red,
                    color.green,
                    color.blue,
                );
                shared
                    .cold
                    .track_accent_color
                    .store(packed, Ordering::Relaxed);
            }
        }

        // Extracts the Main Thread's exclusive channels from shared state
        let param_tx = shared
            .cold
            .param_tx
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .ok_or(PluginError::Message("param_tx producer already taken"))?;

        let gc_rx = shared
            .cold
            .gc_rx
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .ok_or(PluginError::Message("gc_rx consumer already taken"))?;

        let slimmable_tx = shared
            .cold
            .slimmable_tx
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .ok_or(PluginError::Message("slimmable_tx producer already taken"))?;

        #[cfg_attr(test, allow(unused_mut, clippy::allow_attributes))]
        let main_thread = NamClapMainThread {
            shared,
            params: NamPluginParams::default(),
            host,
            sys: SystemSnapshot::capture(),
            param_tx,
            gc_rx,
            slimmable_tx,
            last_reported_latency: 0,
            #[cfg(feature = "clap-plugin")]
            window_handle: None,
            #[cfg(feature = "clap-plugin")]
            floating_thread_handle: None,
            #[cfg(feature = "clap-plugin")]
            floating_close_signal: None,
            #[cfg(feature = "clap-plugin")]
            dialog_handle: None,
            #[cfg(feature = "clap-plugin")]
            dialog_state: None,
            #[cfg(feature = "clap-plugin")]
            ir_dialog_handle: None,
            #[cfg(feature = "clap-plugin")]
            ir_dialog_state: None,
        };

        Ok(main_thread)
    }
}

#[cfg(test)]
pub(crate) use shared::make_test_shared;

#[cfg(test)]
#[path = "../plugin_test.rs"]
mod plugin_test;
