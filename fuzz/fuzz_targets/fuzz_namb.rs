// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

#![no_main]
use libfuzzer_sys::fuzz_target;
use nam_rs::loader::namb::parse_namb;
use nam_rs::loader::dispatcher::build_model;

fuzz_target!(|data: &[u8]| {
    // Tenta carregar o modelo binário (.namb).
    if let Ok(model_data) = parse_namb(data) {
        // Se o parse passou, tenta construir o modelo real para validar semântica.
        let _ = build_model(&model_data);
    }
});
