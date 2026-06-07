// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Preset Discovery Factory: exposes `.nam`/`.namb` model files as CLAP presets.
//!
//! The host's preset browser can index model directories and show metadata
//! extracted from the model files (name, author, gear).

use crate::loader::nam_json::NamMetadata;
use clack_extensions::preset_discovery::prelude::*;
use clack_plugin::prelude::PluginError;
use std::ffi::{CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The unique provider ID for NAM model presets.
const NAM_PROVIDER_ID: &str = "br.eti.fabiolima.nam-rs.nam-models";

/// File extensions for NAM models.
const NAM_FILE_EXT_NAM: &CStr = c"nam";
const NAM_FILE_EXT_NAMB: &CStr = c"namb";

/// Default search directory relative to user home.
const DEFAULT_MODEL_DIR: &str = ".nam/models";

/// Returns the provider descriptor (singleton).
fn provider_descriptor() -> &'static ProviderDescriptor {
    static DESCRIPTOR: OnceLock<ProviderDescriptor> = OnceLock::new();
    DESCRIPTOR.get_or_init(|| {
        ProviderDescriptor::new(NAM_PROVIDER_ID, "NAM Models").with_vendor("Fabio Lima")
    })
}

/// Preset discovery factory for NAM model files (`.nam`/`.namb`).
///
/// Allows hosts to discover and index NAM model files as presets in the host's preset browser.
pub struct NamPresetDiscoveryFactory;

impl PresetDiscoveryFactoryImpl for NamPresetDiscoveryFactory {
    fn provider_count(&self) -> u32 {
        1
    }

    fn provider_descriptor(&self, index: u32) -> Option<&ProviderDescriptor> {
        match index {
            0 => Some(provider_descriptor()),
            _ => None,
        }
    }

    fn create_provider<'a>(
        &'a self,
        host_info: IndexerInfo<'a>,
        provider_id: &CStr,
    ) -> Option<ProviderInstance<'a>> {
        if provider_id.to_bytes() != NAM_PROVIDER_ID.as_bytes() {
            return None;
        }

        Some(ProviderInstance::new(
            host_info,
            provider_descriptor(),
            |mut indexer| {
                // Declare file types supported by NAM.
                let _ = indexer.declare_filetype(FileType {
                    name: c"NAM Model (.nam)",
                    description: Some(c"Neural Amp Modeler JSON format"),
                    file_extension: Some(NAM_FILE_EXT_NAM),
                });
                let _ = indexer.declare_filetype(FileType {
                    name: c"NAM Binary (.namb)",
                    description: Some(c"Neural Amp Modeler binary format"),
                    file_extension: Some(NAM_FILE_EXT_NAMB),
                });

                // Declare the default model directory (~/.nam/models) if it exists.
                if let Some(home) = resolve_home_model_dir() {
                    let home_cstr = CString::new(home.to_string_lossy().as_bytes().to_vec())
                        .unwrap_or_default();
                    let location = LocationInfo {
                        name: c"NAM User Models",
                        flags: Flags::IS_FACTORY_CONTENT,
                        location: Location::File { path: &home_cstr },
                    };
                    let _ = indexer.declare_location(location);
                }

                Ok(NamPresetProvider)
            },
        ))
    }
}

/// Resolves `~/.nam/models` to an absolute path if the directory exists.
fn resolve_home_model_dir() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let dir = Path::new(&home).join(DEFAULT_MODEL_DIR);
    if dir.is_dir() { Some(dir) } else { None }
}

struct NamPresetProvider;

impl ProviderImpl<'_> for NamPresetProvider {
    fn get_metadata(
        &mut self,
        location: Location,
        receiver: &mut MetadataReceiver,
    ) -> Result<(), PluginError> {
        let Location::File { path } = location else {
            return Ok(());
        };

        let file_path = Path::new(
            std::str::from_utf8(path.to_bytes())
                .map_err(|_| PluginError::Message("Invalid UTF-8 in preset file path"))?,
        );

        let name = file_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Unknown Model");
        let name_cstr =
            CString::new(name).map_err(|_| PluginError::Message("Invalid preset name"))?;

        let load_key_cstr =
            CString::new(file_path.to_string_lossy().as_bytes().to_vec()).unwrap_or_default();
        let load_key_cstr_ref: &CStr = &load_key_cstr;

        receiver
            .begin_preset(Some(&name_cstr), Some(load_key_cstr_ref))
            .map_err(|_| PluginError::Message("Failed to begin preset"))?;

        // Try to extract metadata from the model file.
        if let Some(meta) = extract_model_metadata(file_path) {
            if let Some(modeled_by) = meta.modeled_by.as_deref()
                && let Ok(creator) = CString::new(modeled_by)
            {
                receiver.add_creator(&creator);
            }

            if let Some(gear_model) = meta.gear_model.as_deref()
                && let Ok(gm) = CString::new(gear_model)
            {
                receiver.set_description(&gm);
            }

            if let Some(gear_make) = meta.gear_make.as_deref()
                && let Ok(gmake) = CString::new(gear_make)
            {
                receiver.add_extra_info(c"gear_make", &gmake);
            }

            if let Some(gear_type) = meta.gear_type.as_deref()
                && let Ok(gtype) = CString::new(gear_type)
            {
                receiver.add_extra_info(c"gear_type", &gtype);
            }

            if let Some(tone_type) = &meta.tone_type {
                let feature = format!("tone:{}", tone_type);
                if let Ok(fc) = CString::new(feature) {
                    receiver.add_feature(&fc);
                }
            }
        }

        Ok(())
    }
}

/// Lightweight metadata extraction from `.nam` (JSON) files.
///
/// Only parses the `metadata` key to avoid loading weights tensors.
/// Returns `None` for `.namb` files or unparseable files.
fn extract_model_metadata(path: &Path) -> Option<NamMetadata> {
    let ext = path.extension()?.to_str()?;
    match ext.to_lowercase().as_str() {
        "nam" => extract_nam_json_metadata(path),
        _ => None,
    }
}

/// Parses just the `metadata` field from a `.nam` JSON file.
fn extract_nam_json_metadata(path: &Path) -> Option<NamMetadata> {
    let bytes = std::fs::read(path).ok()?;
    let text = std::str::from_utf8(&bytes).ok()?;

    // Locate the "metadata" key in the JSON.
    let marker = "\"metadata\"";
    let pos = text.find(marker)?;

    // Find the opening brace after the marker.
    let after_key = &text[pos + marker.len()..];
    let brace_start = after_key.find('{')?;
    let metadata_slice = &after_key[brace_start..];

    // Extract the matching JSON object by counting braces.
    let json_str = extract_balanced_json(metadata_slice)?;

    // Parse just the metadata object.
    let meta: NamMetadata = serde_json::from_str(json_str).ok()?;

    // Return None if metadata is effectively empty (no useful fields).
    let has_content = meta.name.is_some()
        || meta.modeled_by.is_some()
        || meta.gear_make.is_some()
        || meta.gear_model.is_some();
    has_content.then_some(meta)
}

/// Extracts a balanced JSON object/array from the start of the string.
fn extract_balanced_json(s: &str) -> Option<&str> {
    let mut chars = s.chars();
    let first = chars.next()?;
    let (open, close) = match first {
        '{' => ('{', '}'),
        '[' => ('[', ']'),
        _ => return None,
    };

    let mut depth = 1u32;
    let mut in_string = false;
    let mut prev_was_backslash = false;

    for (end_idx, ch) in (1usize..).zip(chars) {
        let end_idx = end_idx + 1;
        if in_string {
            if prev_was_backslash {
                prev_was_backslash = false;
            } else if ch == '\\' {
                prev_was_backslash = true;
            } else if ch == '"' {
                in_string = false;
            }
        } else {
            match ch {
                '"' => in_string = true,
                c if c == open => depth += 1,
                c if c == close => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(&s[..end_idx]);
                    }
                }
                _ => {}
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_balanced_json_object() {
        let s = r#"{"name": "test", "nested": {"a": 1}}"#;
        let result = extract_balanced_json(s).unwrap();
        assert_eq!(result, s);
    }

    #[test]
    fn test_extract_balanced_json_nested_braces_in_strings() {
        let s = r#"{"data": "{\"key\": \"val\"}", "x": 1}extra"#;
        let result = extract_balanced_json(s).unwrap();
        assert_eq!(result, r#"{"data": "{\"key\": \"val\"}", "x": 1}"#);
    }

    #[test]
    fn test_extract_balanced_json_array() {
        let s = r#"[1, 2, {"a": 3}, [4, 5]]trailing"#;
        let result = extract_balanced_json(s).unwrap();
        assert_eq!(result, r#"[1, 2, {"a": 3}, [4, 5]]"#);
    }

    #[test]
    fn test_metadata_extraction_from_nam_file() {
        let test_dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/models");
        let lstm_path = test_dir.join("lstm.nam");

        if lstm_path.exists() {
            let meta = extract_model_metadata(&lstm_path);
            assert!(meta.is_some(), "Should extract metadata from lstm.nam");
            let m = meta.unwrap();
            assert!(
                m.name.is_some() || m.modeled_by.is_some(),
                "Metadata should contain at least name or author"
            );
        }
    }

    #[test]
    fn test_factory_returns_one_provider() {
        let factory = NamPresetDiscoveryFactory;
        assert_eq!(factory.provider_count(), 1);
    }

    #[test]
    fn test_factory_descriptor_is_valid() {
        let factory = NamPresetDiscoveryFactory;
        let desc = factory.provider_descriptor(0).unwrap();
        assert!(desc.id().is_some());
        assert!(desc.name().is_some());
        assert_eq!(desc.id().unwrap().to_bytes(), NAM_PROVIDER_ID.as_bytes());
    }
}
