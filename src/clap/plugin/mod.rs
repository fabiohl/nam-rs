// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! NAM-rs plugin definition and its CLAP lifecycle components.

mod shared;
pub use shared::{ClapParamPayload, NamClapShared, NamClapSharedRef, NamModelMetadata};

mod main_thread;
pub use main_thread::NamClapMainThread;

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

        #[cfg(feature = "clap-plugin")]
        builder.register::<crate::clap::extensions::gui::NamPluginGui>();
    }
}

impl DefaultPluginFactory for NamClapPlugin {
    fn get_descriptor() -> PluginDescriptor {
        nam_descriptor()
    }

    fn new_shared(_host: HostSharedHandle<'_>) -> Result<Self::Shared<'_>, PluginError> {
        let (param_tx, param_rx) = RingBuffer::new(8);
        let (gc_tx, gc_rx) = RingBuffer::new(32); // Increased capacity for the plugin

        Ok(NamClapShared {
            param_tx: Mutex::new(Some(param_tx)),
            param_rx: Mutex::new(Some(param_rx)),
            gc_tx: Mutex::new(Some(gc_tx)),
            gc_rx: Mutex::new(Some(gc_rx)),
            gc_overflow: Arc::new(GcOverflowBuffer::new(64)),
            rt_status: Arc::new(RtStatusFlags::new()),
            current_latency: AtomicU32::new(0),
            model_sample_rate: AtomicU32::new(48000),
            param_input_gain: AtomicU32::new(0.0f32.to_bits()),
            param_output_gain: AtomicU32::new(0.0f32.to_bits()),
            param_gate_thresh: AtomicU32::new((-70.0f32).to_bits()),
            param_bypass: AtomicU32::new(0),
            ui_peak_l: AtomicU32::new(0.0f32.to_bits()),
            ui_peak_r: AtomicU32::new(0.0f32.to_bits()),
            ui_clipped: std::sync::atomic::AtomicBool::new(false),
            ui_model_name: Mutex::new(String::new()),
            ui_model_metadata: Mutex::new(None),
            ui_pending_model: Mutex::new(None),
            ui_loading: std::sync::atomic::AtomicBool::new(false),
            ui_load_error: std::sync::atomic::AtomicBool::new(false),
            ui_load_error_msg: Mutex::new(String::new()),
            sample_rate: AtomicU32::new(0),
            active_channel_count: AtomicU32::new(1),
            track_accent_color: AtomicU32::new(0),
            param_indication: [
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
            ],
            model_load_counter: AtomicU32::new(0),
            alive_fence: Arc::new(std::sync::atomic::AtomicBool::new(true)),
            gesture_flags: AtomicU32::new(0),
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
                shared.track_accent_color.store(packed, Ordering::Relaxed);
            }
        }

        // Extracts the Main Thread's exclusive channels from shared state
        let param_tx = shared
            .param_tx
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .ok_or(PluginError::Message("param_tx producer already taken"))?;

        let gc_rx = shared
            .gc_rx
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .ok_or(PluginError::Message("gc_rx consumer already taken"))?;

        #[cfg_attr(test, allow(unused_mut))]
        let main_thread = NamClapMainThread {
            shared,
            params: NamPluginParams::default(),
            host,
            sys: SystemSnapshot::capture(),
            param_tx,
            gc_rx,
            last_reported_latency: 0,
            #[cfg(feature = "clap-plugin")]
            window_handle: None,
        };

        Ok(main_thread)
    }
}

#[cfg(test)]
#[path = "../plugin_test.rs"]
mod plugin_test;
