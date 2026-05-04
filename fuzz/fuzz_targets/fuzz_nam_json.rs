// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

#![no_main]
use libfuzzer_sys::fuzz_target;
use nam_rs::loader::nam_json::parse_nam_json;
use nam_rs::loader::dispatcher::build_model;

fuzz_target!(|data: &[u8]| {
    // Converte os bytes do fuzzer para string UTF-8.
    if let Ok(s) = std::str::from_utf8(data) {
        if let Ok(model_data) = parse_nam_json(s) {
            // Tenta construir o modelo para validar consistência semântica.
            let _ = build_model(&model_data);
        }
    }
});
