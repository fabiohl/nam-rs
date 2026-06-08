// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::common::spsc::RtStatusFlags;
use crate::dsp::gate::GateState;

#[inline(always)]
pub(super) fn report_gate_flags(rt_status: &RtStatusFlags, gate_state: GateState) {
    match gate_state {
        GateState::Closed => {
            rt_status.set_flag(crate::common::spsc::RT_STATUS_IS_SILENT);
            rt_status.clear_flag(crate::common::spsc::RT_STATUS_IS_FADING);
        }
        GateState::FadingIn | GateState::FadingOut => {
            rt_status.clear_flag(crate::common::spsc::RT_STATUS_IS_SILENT);
            rt_status.set_flag(crate::common::spsc::RT_STATUS_IS_FADING);
        }
        GateState::Open => {
            rt_status.clear_flag(crate::common::spsc::RT_STATUS_IS_SILENT);
            rt_status.clear_flag(crate::common::spsc::RT_STATUS_IS_FADING);
        }
    }
}
