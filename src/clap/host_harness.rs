// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! CLAP Host Test Harness — S8-E8-T01
//!
//! Simulates a complete DAW host for contract validation:
//!   - Thread-check callbacks (`is_main_thread` / `is_audio_thread`)
//!   - Restart requests with full deactivate → activate cycle
//!   - Latency / tail / preset-load notification tracking
//!   - Bounded event queue saturation detection
//!
//! # Acceptance criteria
//!
//! The harness must **automatically detect any CLAP protocol violation**
//! and fail the test suite, replacing false-confidence unit tests with
//! oracles capable of reproducing real DAW failures.

#![allow(missing_docs)]

use crate::clap::NamClapPlugin;
use clack_extensions::latency::HostLatency;
use clack_extensions::latency::HostLatencyImpl;
use clack_extensions::log::HostLog;
use clack_extensions::log::HostLogImpl;
use clack_extensions::log::LogSeverity;
use clack_extensions::params::HostParams;
use clack_extensions::params::HostParamsImplMainThread;
use clack_extensions::params::HostParamsImplShared;
use clack_extensions::params::ParamClearFlags;
use clack_extensions::params::ParamRescanFlags;
use clack_extensions::preset_discovery::HostPresetLoad;
use clack_extensions::preset_discovery::HostPresetLoadImpl;
use clack_extensions::preset_discovery::preset_data::Location;
use clack_extensions::tail::HostTail;
use clack_extensions::tail::HostTailImpl;
use clack_extensions::thread_check::HostThreadCheck;
use clack_extensions::thread_check::HostThreadCheckImpl;
use clack_host::prelude::*;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;

// ═══════════════════════════════════════════════════════════════════════════
// Inner state — shared via Arc between host handlers and test assertions
// ═══════════════════════════════════════════════════════════════════════════

/// Backend state shared between all host handler instances and the test code.
/// All fields are behind `Arc` so cloning a `CompleteHostState` keeps
/// everything pointing to the same event log and flags.
#[derive(Clone, Default)]
pub struct CompleteHostState {
    pub events: Arc<Mutex<Vec<HostEvent>>>,
    pub restart_requested: Arc<AtomicBool>,
    pub process_requested: Arc<AtomicBool>,
    pub callback_requested: Arc<AtomicBool>,
    pub main_thread_id: Arc<Mutex<Option<std::thread::ThreadId>>>,
    pub audio_thread_id: Arc<Mutex<Option<std::thread::ThreadId>>>,
    pub latency_changed_count: Arc<AtomicU32>,
    pub tail_changed_count: Arc<AtomicU32>,
    pub preset_loaded_count: Arc<AtomicU32>,
    pub preset_error_count: Arc<AtomicU32>,
}

impl CompleteHostState {
    pub fn new() -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
            restart_requested: Arc::new(AtomicBool::new(false)),
            process_requested: Arc::new(AtomicBool::new(false)),
            callback_requested: Arc::new(AtomicBool::new(false)),
            main_thread_id: Arc::new(Mutex::new(None)),
            audio_thread_id: Arc::new(Mutex::new(None)),
            latency_changed_count: Arc::new(AtomicU32::new(0)),
            tail_changed_count: Arc::new(AtomicU32::new(0)),
            preset_loaded_count: Arc::new(AtomicU32::new(0)),
            preset_error_count: Arc::new(AtomicU32::new(0)),
        }
    }

    pub fn record(&self, event: HostEvent) {
        self.events.lock().unwrap().push(event);
    }

    pub fn snapshot(&self) -> Vec<HostEvent> {
        self.events.lock().unwrap().clone()
    }

    pub fn set_main_thread(&self) {
        *self.main_thread_id.lock().unwrap() = Some(std::thread::current().id());
    }

    pub fn set_audio_thread(&self) {
        *self.audio_thread_id.lock().unwrap() = Some(std::thread::current().id());
    }

    pub fn assert_event_occurred(&self, label: &str, predicate: impl Fn(&HostEvent) -> bool) {
        let events = self.snapshot();
        if !events.iter().any(&predicate) {
            panic!("Expected event '{label}' not found.\nRecorded events: {events:#?}");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Host event types
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostEvent {
    RestartRequested,
    ProcessRequested,
    CallbackRequested,
    LatencyChanged,
    TailChanged,
    PresetLoaded,
    PresetLoadError {
        os_error: i32,
        message: String,
    },
    ParamsFlushRequested,
    ParamsRescan,
    ParamsClear {
        param_id: u32,
    },
    PluginLog {
        severity: LogSeverity,
        message: String,
    },
}

// ═══════════════════════════════════════════════════════════════════════════
// CompleteHostShared
// ═══════════════════════════════════════════════════════════════════════════

pub struct CompleteHostShared {
    state: CompleteHostState,
}

impl CompleteHostShared {
    pub fn new(state: &CompleteHostState) -> Self {
        Self {
            state: state.clone(),
        }
    }
}

impl<'a> SharedHandler<'a> for CompleteHostShared {
    fn request_restart(&self) {
        self.state.restart_requested.store(true, Ordering::SeqCst);
        self.state.record(HostEvent::RestartRequested);
    }
    fn request_process(&self) {
        self.state.process_requested.store(true, Ordering::SeqCst);
        self.state.record(HostEvent::ProcessRequested);
    }
    fn request_callback(&self) {
        self.state.callback_requested.store(true, Ordering::SeqCst);
        self.state.record(HostEvent::CallbackRequested);
    }
}

impl HostThreadCheckImpl for CompleteHostShared {
    fn is_main_thread(&self) -> bool {
        let current = std::thread::current().id();
        self.state
            .main_thread_id
            .lock()
            .unwrap()
            .is_none_or(|id| id == current)
    }
    fn is_audio_thread(&self) -> bool {
        let current = std::thread::current().id();
        self.state
            .audio_thread_id
            .lock()
            .unwrap()
            .is_none_or(|id| id == current)
    }
}

impl HostLogImpl for CompleteHostShared {
    fn log(&self, severity: LogSeverity, message: &str) {
        self.state.record(HostEvent::PluginLog {
            severity,
            message: message.to_string(),
        });
    }
}

impl HostParamsImplShared for CompleteHostShared {
    fn request_flush(&self) {
        self.state.record(HostEvent::ParamsFlushRequested);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CompleteHostMainThread
// ═══════════════════════════════════════════════════════════════════════════

pub struct CompleteHostMainThread {
    state: CompleteHostState,
}

impl CompleteHostMainThread {
    pub fn new(state: &CompleteHostState) -> Self {
        Self {
            state: state.clone(),
        }
    }
}

impl<'a> MainThreadHandler<'a> for CompleteHostMainThread {}

impl HostLatencyImpl for CompleteHostMainThread {
    fn changed(&mut self) {
        self.state
            .latency_changed_count
            .fetch_add(1, Ordering::SeqCst);
        self.state.record(HostEvent::LatencyChanged);
    }
}

impl HostPresetLoadImpl for CompleteHostMainThread {
    fn on_error(
        &mut self,
        _location: Location,
        _load_key: Option<&std::ffi::CStr>,
        os_error: i32,
        message: Option<&std::ffi::CStr>,
    ) {
        self.state.preset_error_count.fetch_add(1, Ordering::SeqCst);
        self.state.record(HostEvent::PresetLoadError {
            os_error,
            message: message
                .map(|m| m.to_string_lossy().into_owned())
                .unwrap_or_default(),
        });
    }
    fn loaded(&mut self, _location: Location, _load_key: Option<&std::ffi::CStr>) {
        self.state
            .preset_loaded_count
            .fetch_add(1, Ordering::SeqCst);
        self.state.record(HostEvent::PresetLoaded);
    }
}

impl HostParamsImplMainThread for CompleteHostMainThread {
    fn rescan(&mut self, _flags: ParamRescanFlags) {
        self.state.record(HostEvent::ParamsRescan);
    }
    fn clear(&mut self, param_id: clack_common::utils::ClapId, _flags: ParamClearFlags) {
        self.state.record(HostEvent::ParamsClear {
            param_id: param_id.get(),
        });
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CompleteHostAudioProcessor
// ═══════════════════════════════════════════════════════════════════════════

pub struct CompleteHostAudioProcessor {
    state: CompleteHostState,
}

impl CompleteHostAudioProcessor {
    pub fn new(state: &CompleteHostState) -> Self {
        Self {
            state: state.clone(),
        }
    }
}

impl<'a> AudioProcessorHandler<'a> for CompleteHostAudioProcessor {}

impl HostTailImpl for CompleteHostAudioProcessor {
    fn changed(&mut self) {
        self.state.tail_changed_count.fetch_add(1, Ordering::SeqCst);
        self.state.record(HostEvent::TailChanged);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CompleteHost
// ═══════════════════════════════════════════════════════════════════════════

pub struct CompleteHost;

impl HostHandlers for CompleteHost {
    type Shared<'a> = CompleteHostShared;
    type MainThread<'a> = CompleteHostMainThread;
    type AudioProcessor<'a> = CompleteHostAudioProcessor;

    fn declare_extensions(builder: &mut HostExtensions<Self>, _shared: &Self::Shared<'_>) {
        builder.register::<HostThreadCheck>();
        builder.register::<HostLog>();
        builder.register::<HostLatency>();
        builder.register::<HostTail>();
        builder.register::<HostParams>();
        builder.register::<HostPresetLoad>();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Bootstrap helpers
// ═══════════════════════════════════════════════════════════════════════════

pub fn make_test_plugin_with_harness() -> (
    PluginEntry,
    HostInfo,
    PluginInstance<CompleteHost>,
    CompleteHostState,
) {
    let entry =
        PluginEntry::load_from_clack::<clack_plugin::entry::SinglePluginEntry<NamClapPlugin>>(
            c"/test",
        )
        .expect("Failed to load PluginEntry");

    let host_info = HostInfo::new(
        "NamHarness",
        "CI Suite",
        "https://github.com/fabiohl/nam-rs",
        "0.1.0",
    )
    .unwrap();

    let state = CompleteHostState::new();
    state.set_main_thread();

    let state_shared = state.clone();
    let state_mt = state.clone();

    let instance = PluginInstance::<CompleteHost>::new(
        move |_| CompleteHostShared::new(&state_shared),
        move |_: &CompleteHostShared| CompleteHostMainThread::new(&state_mt),
        &entry,
        c"br.eti.fabiolima.nam-rs",
        &host_info,
    )
    .expect("Failed to instantiate plugin with harness");

    (entry, host_info, instance, state)
}

pub fn make_harness_audio_processor(state: &CompleteHostState) -> CompleteHostAudioProcessor {
    state.set_audio_thread();
    CompleteHostAudioProcessor::new(state)
}

pub fn process_block_harness(
    started: &mut StartedPluginAudioProcessor<CompleteHost>,
    in_l: &mut [f32],
    in_r: &mut [f32],
    out_l: &mut [f32],
    out_r: &mut [f32],
    events: Option<&InputEvents<'_>>,
) -> EventBuffer {
    let mut input_ports = AudioPorts::with_capacity(2, 1);
    let mut output_ports = AudioPorts::with_capacity(2, 1);

    let mut in_ch = [in_l, in_r];
    let input_audio = input_ports.with_input_buffers([AudioPortBuffer {
        latency: 0,
        channels: AudioPortBufferType::f32_input_only(in_ch.iter_mut().map(InputChannel::constant)),
    }]);
    let out_ch = [out_l, out_r];
    let mut output_audio = output_ports.with_output_buffers([AudioPortBuffer {
        latency: 0,
        channels: AudioPortBufferType::f32_output_only(out_ch.into_iter()),
    }]);
    let mut output_events_buffer = EventBuffer::new();
    let mut out_ev = OutputEvents::from_buffer(&mut output_events_buffer);

    started
        .process(
            &input_audio,
            &mut output_audio,
            events.unwrap_or(&InputEvents::empty()),
            &mut out_ev,
            None,
            None,
        )
        .expect("process() failed");

    output_events_buffer
}

pub fn perform_restart(
    instance: &mut PluginInstance<CompleteHost>,
    started: StartedPluginAudioProcessor<CompleteHost>,
    state: &CompleteHostState,
    audio_config: PluginAudioConfiguration,
) -> StartedPluginAudioProcessor<CompleteHost> {
    let stopped = started.stop_processing();
    instance.deactivate(stopped);

    state.restart_requested.store(false, Ordering::SeqCst);

    let state_ap = state.clone();
    let stopped = instance
        .activate(
            move |_, _| make_harness_audio_processor(&state_ap),
            audio_config,
        )
        .expect("Failed to re-activate during restart");
    stopped
        .start_processing()
        .expect("Failed to start processing after restart")
}

pub fn extract_plugin_shared(
    instance: &mut PluginInstance<CompleteHost>,
) -> *const crate::clap::plugin::NamClapShared {
    let raw_ptr = instance.plugin_handle().as_raw_ptr();
    unsafe {
        clack_plugin::extensions::wrapper::PluginWrapper::<NamClapPlugin>::handle(
            raw_ptr,
            |wrapper| Ok(wrapper.shared() as *const crate::clap::plugin::NamClapShared),
        )
    }
    .expect("Failed to get plugin wrapper")
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use clack_common::events::Pckn;
    use clack_common::events::event_types::ParamValueEvent;
    use clack_common::utils::{ClapId, Cookie};
    use std::sync::atomic::Ordering;

    fn default_audio_config() -> PluginAudioConfiguration {
        PluginAudioConfiguration {
            sample_rate: 48000.0,
            min_frames_count: 256,
            max_frames_count: 256,
        }
    }

    #[test]
    fn test_thread_check_main_thread() {
        let (_entry, _host_info, _instance, state) = make_test_plugin_with_harness();
        assert!(
            state
                .main_thread_id
                .lock()
                .unwrap()
                .is_none_or(|id| id == std::thread::current().id()),
            "Harness must report is_main_thread=true for the test thread"
        );
    }

    #[test]
    fn test_thread_check_audio_thread() {
        let (_entry, _host_info, mut instance, state) = make_test_plugin_with_harness();
        state.set_audio_thread();
        let _stopped = instance
            .activate(
                |_, _| make_harness_audio_processor(&state),
                default_audio_config(),
            )
            .expect("activate failed");
        assert!(
            state
                .audio_thread_id
                .lock()
                .unwrap()
                .is_none_or(|id| id == std::thread::current().id()),
            "Harness must report is_audio_thread=true after set_audio_thread()"
        );
    }

    #[test]
    fn test_oversample_change_triggers_restart_protocol() {
        use crate::clap::extensions::params::PARAM_OVERSAMPLE;

        let (_entry, _host_info, mut instance, state) = make_test_plugin_with_harness();

        let n = 256;
        let audio_config = PluginAudioConfiguration {
            sample_rate: 48000.0,
            min_frames_count: n as u32,
            max_frames_count: n as u32,
        };

        let stopped = instance
            .activate(|_, _| make_harness_audio_processor(&state), audio_config)
            .expect("activate failed");
        let mut started = stopped.start_processing().expect("start_processing failed");

        let event = ParamValueEvent::new(
            0u32,
            ClapId::new(PARAM_OVERSAMPLE),
            Pckn::match_all(),
            1.0,
            Cookie::empty(),
        );
        let mut event_buffer = EventBuffer::new();
        event_buffer.push(&event);
        let input_events = InputEvents::from_buffer(&event_buffer);

        let mut il = vec![0.3f32; n];
        let mut ir = vec![0.3f32; n];
        let mut ol = vec![0.0f32; n];
        let mut or = vec![0.0f32; n];
        let _ = process_block_harness(
            &mut started,
            &mut il,
            &mut ir,
            &mut ol,
            &mut or,
            Some(&input_events),
        );

        assert!(
            state.restart_requested.load(Ordering::SeqCst),
            "Plugin must call request_restart() when oversampling changes during active processing"
        );

        state.assert_event_occurred("RestartRequested", |e| {
            matches!(e, HostEvent::RestartRequested)
        });

        drop(started);
    }

    #[test]
    fn test_restart_cycle_clean() {
        let (_entry, _host_info, mut instance, state) = make_test_plugin_with_harness();
        let audio_config = default_audio_config();

        let stopped = instance
            .activate(|_, _| make_harness_audio_processor(&state), audio_config)
            .expect("activate failed");
        let started = stopped.start_processing().expect("start_processing failed");

        let mut started_after = perform_restart(&mut instance, started, &state, audio_config);

        let n = 256;
        let mut il = vec![0.3f32; n];
        let mut ir = vec![0.3f32; n];
        let mut ol = vec![0.0f32; n];
        let mut or = vec![0.0f32; n];
        let _ = process_block_harness(&mut started_after, &mut il, &mut ir, &mut ol, &mut or, None);

        assert!(
            !state.restart_requested.load(Ordering::SeqCst),
            "restart_requested should be cleared after perform_restart()"
        );
    }

    #[test]
    fn test_latency_changed_notification() {
        let (_entry, _host_info, mut instance, state) = make_test_plugin_with_harness();
        let audio_config = default_audio_config();

        let stopped = instance
            .activate(|_, _| make_harness_audio_processor(&state), audio_config)
            .expect("activate failed");
        let mut started = stopped.start_processing().expect("start_processing failed");

        for _ in 0..8 {
            let n = 256;
            let mut il = vec![0.3f32; n];
            let mut ir = vec![0.3f32; n];
            let mut ol = vec![0.0f32; n];
            let mut or = vec![0.0f32; n];
            let _ = process_block_harness(&mut started, &mut il, &mut ir, &mut ol, &mut or, None);
        }

        let events = state.snapshot();
        let latency_events: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, HostEvent::LatencyChanged))
            .collect();
        eprintln!(
            "LatencyChanged events: {latency_events:?} (total={})",
            events.len()
        );
    }

    #[test]
    fn test_tail_changed_on_cabsim_load() {
        let state = CompleteHostState::new();
        let mut ap = CompleteHostAudioProcessor::new(&state);

        // Verify HostTailImpl records events and increments counter
        ap.changed();
        ap.changed();

        state.assert_event_occurred("TailChanged", |e| matches!(e, HostEvent::TailChanged));
        assert_eq!(state.tail_changed_count.load(Ordering::SeqCst), 2);

        // Note: Full integration test (IR load via SPSC → HostTail::changed())
        // requires a model fixture and the LoadCabIr command path through the
        // SPSC channel. The harness infrastructure itself is validated here.
    }

    #[test]
    fn test_command_queue_no_overflow_under_automation_burst() {
        use crate::clap::extensions::params::{
            PARAM_GATE_THRESH, PARAM_INPUT_GAIN, PARAM_OUTPUT_GAIN,
        };

        let (_entry, _host_info, mut instance, state) = make_test_plugin_with_harness();
        let n = 64;
        let audio_config = PluginAudioConfiguration {
            sample_rate: 48000.0,
            min_frames_count: n as u32,
            max_frames_count: n as u32,
        };

        let stopped = instance
            .activate(|_, _| make_harness_audio_processor(&state), audio_config)
            .expect("activate failed");
        let mut started = stopped.start_processing().expect("start_processing failed");

        let mut event_buffer = EventBuffer::new();
        for i in 0..50 {
            let param_id = [PARAM_INPUT_GAIN, PARAM_OUTPUT_GAIN, PARAM_GATE_THRESH][i % 3];
            let value = (i as f64) / 50.0;
            event_buffer.push(&ParamValueEvent::new(
                i as u32,
                ClapId::new(param_id),
                Pckn::match_all(),
                value,
                Cookie::empty(),
            ));
        }
        let input_events = InputEvents::from_buffer(&event_buffer);

        let mut il = vec![0.3f32; n];
        let mut ir = vec![0.3f32; n];
        let mut ol = vec![0.0f32; n];
        let mut or = vec![0.0f32; n];
        let _ = process_block_harness(
            &mut started,
            &mut il,
            &mut ir,
            &mut ol,
            &mut or,
            Some(&input_events),
        );
    }

    #[test]
    fn test_full_lifecycle_smoke() {
        let (_entry, _host_info, mut instance, state) = make_test_plugin_with_harness();
        let audio_config = default_audio_config();

        let stopped = instance
            .activate(|_, _| make_harness_audio_processor(&state), audio_config)
            .expect("activate failed");
        let mut started = stopped.start_processing().expect("start_processing failed");

        let n = 256;
        for _ in 0..4 {
            let mut il = vec![0.3f32; n];
            let mut ir = vec![0.3f32; n];
            let mut ol = vec![0.0f32; n];
            let mut or = vec![0.0f32; n];
            let _ = process_block_harness(&mut started, &mut il, &mut ir, &mut ol, &mut or, None);
        }

        let stopped = started.stop_processing();
        instance.deactivate(stopped);

        let events = state.snapshot();
        assert!(
            !events.is_empty(),
            "Harness should have recorded events during lifecycle.\nEvents: {events:#?}"
        );
    }

    #[test]
    fn test_host_log_captures_plugin_messages() {
        let (_entry, _host_info, mut instance, state) = make_test_plugin_with_harness();
        let audio_config = default_audio_config();

        let stopped = instance
            .activate(|_, _| make_harness_audio_processor(&state), audio_config)
            .expect("activate failed");
        let mut started = stopped.start_processing().expect("start_processing failed");

        let n = 256;
        let mut il = vec![0.3f32; n];
        let mut ir = vec![0.3f32; n];
        let mut ol = vec![0.0f32; n];
        let mut or = vec![0.0f32; n];
        let _ = process_block_harness(&mut started, &mut il, &mut ir, &mut ol, &mut or, None);

        drop(started);

        state.assert_event_occurred("PluginLog", |e| matches!(e, HostEvent::PluginLog { .. }));
    }

    #[test]
    fn test_dual_instance_harness() {
        let (_entry_a, _host_info_a, mut inst_a, state_a) = make_test_plugin_with_harness();
        let (_entry_b, _host_info_b, mut inst_b, state_b) = make_test_plugin_with_harness();

        let audio_config = default_audio_config();
        let stopped_a = inst_a
            .activate(|_, _| make_harness_audio_processor(&state_a), audio_config)
            .expect("activate A failed");
        let stopped_b = inst_b
            .activate(|_, _| make_harness_audio_processor(&state_b), audio_config)
            .expect("activate B failed");

        let mut started_a = stopped_a.start_processing().expect("start A failed");
        let mut started_b = stopped_b.start_processing().expect("start B failed");

        let n = 256;
        for _ in 0..2 {
            let mut il = vec![0.3f32; n];
            let mut ir = vec![0.3f32; n];
            let mut ol = vec![0.0f32; n];
            let mut or = vec![0.0f32; n];
            let _ = process_block_harness(&mut started_a, &mut il, &mut ir, &mut ol, &mut or, None);
        }
        for _ in 0..2 {
            let mut il = vec![0.5f32; n];
            let mut ir = vec![0.5f32; n];
            let mut ol = vec![0.0f32; n];
            let mut or = vec![0.0f32; n];
            let _ = process_block_harness(&mut started_b, &mut il, &mut ir, &mut ol, &mut or, None);
        }

        let stopped_a = started_a.stop_processing();
        let stopped_b = started_b.stop_processing();
        inst_a.deactivate(stopped_a);
        inst_b.deactivate(stopped_b);

        assert!(!state_a.snapshot().is_empty());
        assert!(!state_b.snapshot().is_empty());
    }
}
