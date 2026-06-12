// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! CLI binary to generate deterministic synthetic LSTM `.nam` fixture files.
//!
//! Produces one `.nam` file per LSTM variant for which golden vectors
//! have not yet been generated from real trained models. The weight
//! stream is deterministic (sawtooth-modulo, seeded) and matches the
//! exact byte layout expected by the dispatcher (`build_lstm_1layer` /
//! `build_lstm_2layer`).
//!
//! ## Usage
//! ```bash
//! cargo run --release --bin gen_lstm_fixtures -- tests/fixtures/models
//! ```

use std::fs;
use std::io::Write;
use std::path::PathBuf;

struct LstmVariant {
    num_layers: usize,
    hidden_size: usize,
    filename: &'static str,
    label: &'static str,
}

const VARIANTS: &[LstmVariant] = &[
    LstmVariant {
        num_layers: 1,
        hidden_size: 8,
        filename: "BossLSTM-1x8.nam",
        label: "LSTM 1×8",
    },
    LstmVariant {
        num_layers: 1,
        hidden_size: 12,
        filename: "BossLSTM-1x12.nam",
        label: "LSTM 1×12",
    },
    LstmVariant {
        num_layers: 1,
        hidden_size: 24,
        filename: "BossLSTM-1x24.nam",
        label: "LSTM 1×24",
    },
    LstmVariant {
        num_layers: 1,
        hidden_size: 40,
        filename: "BossLSTM-1x40.nam",
        label: "LSTM 1×40",
    },
    LstmVariant {
        num_layers: 2,
        hidden_size: 12,
        filename: "BossLSTM-2x12.nam",
        label: "LSTM 2×12",
    },
    LstmVariant {
        num_layers: 2,
        hidden_size: 16,
        filename: "BossLSTM-2x16.nam",
        label: "LSTM 2×16",
    },
    LstmVariant {
        num_layers: 2,
        hidden_size: 24,
        filename: "BossLSTM-2x24.nam",
        label: "LSTM 2×24",
    },
];

fn total_weights(num_layers: usize, hidden_size: usize) -> usize {
    let h = hidden_size;
    if num_layers == 1 {
        4 * h * (1 + h) + 7 * h + 1
    } else {
        12 * h * h + 17 * h + 1
    }
}

fn generate_weights(num_layers: usize, hidden_size: usize) -> Vec<f32> {
    let n = total_weights(num_layers, hidden_size);
    let mut weights = Vec::with_capacity(n);
    let mut val = 0.05f32;
    let mut current_input_size = 1usize;

    for _ in 0..num_layers {
        let ih = current_input_size + hidden_size;
        let w_size = 4 * hidden_size * ih;
        for _ in 0..w_size {
            weights.push(val);
            val = (val + 0.007) % 0.3;
        }
        let b_size = 4 * hidden_size;
        for _ in 0..b_size {
            weights.push(val);
            val = (val + 0.007) % 0.3;
        }
        for _ in 0..hidden_size {
            weights.push(val);
            val = (val + 0.007) % 0.3;
        }
        for _ in 0..hidden_size {
            weights.push(val);
            val = (val + 0.007) % 0.3;
        }
        current_input_size = hidden_size;
    }

    for _ in 0..hidden_size {
        weights.push(val);
        val = (val + 0.007) % 0.3;
    }
    weights.push(val);

    assert_eq!(
        weights.len(),
        n,
        "weight count mismatch for {num_layers}x{hidden_size}"
    );
    weights
}

fn build_json(num_layers: usize, hidden_size: usize, weights: &[f32], label: &str) -> String {
    let weights_str: Vec<String> = weights.iter().map(|w| format!("{w:.12}")).collect();
    let weights_json = weights_str.join(", ");

    format!(
        r#"{{"version":"0.5.0","architecture":"LSTM","config":{{"input_size":1,"num_layers":{num_layers},"hidden_size":{hidden_size}}},"metadata":{{"name":"{label} Fixture (synthetic)","modeled_by":"src/bin/gen_lstm_fixtures.rs"}},"weights":[{weights_json}]}}"#
    )
}

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("tests/fixtures/models"));

    fs::create_dir_all(&out_dir).expect("Failed to create output directory");

    for var in VARIANTS {
        let weights = generate_weights(var.num_layers, var.hidden_size);
        let json = build_json(var.num_layers, var.hidden_size, &weights, var.label);
        let path = out_dir.join(var.filename);

        let mut file = fs::File::create(&path).expect("Failed to create .nam file");
        file.write_all(json.as_bytes())
            .expect("Failed to write .nam file");

        println!(
            "Written {}  ({} layers × H={}, {} weights)",
            path.display(),
            var.num_layers,
            var.hidden_size,
            weights.len()
        );
    }

    println!("Done — {} LSTM fixture files generated.", VARIANTS.len());
}
