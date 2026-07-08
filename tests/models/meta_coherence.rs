// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  Meta-tests that enforce catalog↔test coherence.
// 
//  Ensures every `.nam` model referenced by an `#[ignore]` test
//  (golden-missing reason) is registered in the canonical CATALOG
//  array in `golden_gen_build.sh`. Prevents silent drift where a
//  model is added to tests but forgotten in the golden generator.
// 
//  Part of Tarefa C.1.4 (Épico C — Sprint C.1).

use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

/// Models referenced by `#[ignore]` tests that are intentionally
/// absent from the CATALOG. Each entry MUST document the reason and
/// the sprint/task that will resolve it. Entries MUST be removed
/// when the corresponding gap is closed.
#[allow(dead_code)]
#[derive(Debug)]
struct CatalogGap {
    nam_file: &'static str,
    source: &'static str,
    reason: &'static str,
}

const CATALOG_EXCEPTIONS: &[CatalogGap] = &[
    CatalogGap {
        nam_file: "linear_fft_rf2048.nam",
        source: "tests/linear_fft_test.rs",
        reason: "golden pipeline not yet implemented (F3b item 2 — Sprint C.2, Task C.2.2)",
    },
    CatalogGap {
        nam_file: "linear_fft_rf4096.nam",
        source: "tests/linear_fft_test.rs",
        reason: "golden pipeline not yet implemented (F3b item 2 — Sprint C.2, Task C.2.2)",
    },
    CatalogGap {
        nam_file: "linear_fft_rf8192.nam",
        source: "tests/linear_fft_test.rs",
        reason: "golden pipeline not yet implemented (F3b item 2 — Sprint C.2, Task C.2.2)",
    },
];

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn golden_gen_build_path() -> PathBuf {
    fixture_dir().join("golden_gen_build.sh")
}

/// Parses the CATALOG array from golden_gen_build.sh.
///
/// Returns the set of `.nam` filenames registered in the canonical
/// catalog. The CATALOG format is:
///   "nam_file:golden_name:label:v2_scope[:skip_srs]"
fn parse_catalog() -> HashSet<String> {
    let script = golden_gen_build_path();
    let content = fs::read_to_string(&script).expect("Failed to read golden_gen_build.sh");

    let mut in_catalog = false;
    let mut nam_files = HashSet::new();

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed == "CATALOG=(" {
            in_catalog = true;
            continue;
        }
        if in_catalog {
            if trimmed == ")" {
                break;
            }
            if let Some(inner) = trimmed.strip_prefix('"').and_then(|s| s.strip_suffix('"'))
                && let Some(nam_file) = inner.split(':').next()
                && nam_file.ends_with(".nam")
            {
                nam_files.insert(nam_file.to_string());
            }
        }
    }

    assert!(
        !nam_files.is_empty(),
        "CATALOG array in golden_gen_build.sh is empty or could not be parsed. \
         Check the format and ensure the array contains valid entries."
    );
    assert!(
        nam_files.len() >= 15,
        "Expected ≥ 15 models in CATALOG, found {}. \
         Regression — fewer models than documented 20 entries.",
        nam_files.len(),
    );
    nam_files
}

/// Extracts `.nam` string literals from a block of text.
///
/// Only returns strings that pass `is_valid_model_filename` — i.e.,
/// actual model references, not error/log strings that happen to
/// end in `.nam`.
fn extract_nam_strings(text: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut in_string = false;
    let mut current = String::new();

    for ch in text.chars() {
        match ch {
            '"' => {
                if in_string {
                    if current.ends_with(".nam") && current.len() > 4 {
                        let candidate = std::mem::take(&mut current);
                        if is_valid_model_filename(&candidate) {
                            result.push(candidate);
                        }
                    }
                    current.clear();
                    in_string = false;
                } else {
                    in_string = true;
                }
            }
            _ if in_string => current.push(ch),
            _ => {}
        }
    }
    result
}

fn is_valid_model_filename(s: &str) -> bool {
    // Reject strings that are clearly error/log messages
    let lower = s.to_lowercase();
    if lower.contains("failed") || lower.contains("not found") || s.len() > 120 {
        return false;
    }
    // Model filenames use a limited character set (allow spaces)
    s.chars().all(|c| c.is_ascii_graphic() || c == ' ')
        && s.matches('.').count() <= 2
        && s.ends_with(".nam")
}

/// Scans a test source file for `#[ignore]` blocks and extracts
/// `.nam` model references found within each ignored test function.
///
/// Returns `(nam_file, source_file_name, test_fn_signature)` tuples.
fn scan_ignored_test_models(
    file_path: &PathBuf,
    source_name: &str,
) -> Vec<(String, String, String)> {
    let content = match fs::read_to_string(file_path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let mut result = Vec::new();

    let blocks: Vec<&str> = content.split("#[ignore").collect();
    for block in &blocks[1..] {
        let fn_sig = block
            .lines()
            .find(|l| l.trim().starts_with("fn test_"))
            .map(|l| l.trim().to_string())
            .unwrap_or_else(|| String::from("(unknown test fn)"));

        let window: String = block.lines().take(50).collect::<Vec<_>>().join("\n");

        for nam_file in extract_nam_strings(&window) {
            result.push((nam_file, source_name.to_string(), fn_sig.clone()));
        }
    }

    result
}

/// Every `.nam` model referenced in an `#[ignore]` test (in golden-relevant
/// test files) MUST be registered in the canonical CATALOG array of
/// `golden_gen_build.sh`.
///
/// This prevents drift where a model is added to test suites but
/// forgotten in the golden generator — a structural gap that silently
/// keeps the corresponding golden tests `#[ignore]`d forever.
///
/// ## Known exceptions
///
/// Models in `CATALOG_EXCEPTIONS` are intentionally absent from the
/// CATALOG and are tracked for resolution in future sprints. Each
/// exception has a documented reason and must be removed when the gap
/// is closed.
///
/// ## Scope
///
/// Currently scans `tests/golden_vectors.rs` and `tests/linear_fft_test.rs`.
/// `tests/cpp_parity.rs` is excluded because its `#[ignore]` is due to
/// C++ toolchain dependency, not missing goldens.
#[test]
fn test_ignored_models_are_in_catalog() {
    let catalog = parse_catalog();

    let exceptions_by_nam: HashSet<&str> = CATALOG_EXCEPTIONS.iter().map(|e| e.nam_file).collect();

    let test_files: &[(&str, &str)] = &[
        ("tests/golden_vectors.rs", "golden_vectors.rs"),
        ("tests/linear_fft_test.rs", "linear_fft_test.rs"),
    ];

    let mut checked = 0usize;

    for (rel_path, source_name) in test_files {
        let full_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel_path);
        let models = scan_ignored_test_models(&full_path, source_name);

        for (nam_file, _, fn_sig) in &models {
            // Skip known exceptions — gaps already documented and tracked
            if exceptions_by_nam.contains(nam_file.as_str()) {
                let gap = CATALOG_EXCEPTIONS
                    .iter()
                    .find(|e| e.nam_file == nam_file.as_str())
                    .unwrap();
                eprintln!(
                    "  INFO: {nam_file} (test: {fn_sig}) is a known gap — {reason}",
                    reason = gap.reason,
                );
                continue;
            }

            assert!(
                catalog.contains(nam_file.as_str()),
                "Model '{nam_file}' is referenced by an #[ignore] test in {source_name} \
                 (fn: {fn_sig}) but is NOT in the CATALOG array of golden_gen_build.sh.\n\
                 \n\
                 Add '{nam_file}' to CATALOG in tests/fixtures/golden_gen_build.sh, \
                 or document it as a known gap in CATALOG_EXCEPTIONS \
                 (tests/meta_coherence.rs) with a reason and tracking sprint/task.\n\
                 \n\
                 This is the guard against F3-class bugs: models that silently enter \
                 the test suite but never have their goldens generated because the \
                 generator script was never updated.",
            );
            checked += 1;
        }
    }

    assert!(
        checked >= 10,
        "Only {checked} models were checked for catalog coherence. \
         Expected ≥ 10 (the v2 golden_vectors.rs subset plus golden-missing exceptions). \
         Possible regression: scanner is missing valid .nam references or the test \
         structure has changed.",
    );
}
