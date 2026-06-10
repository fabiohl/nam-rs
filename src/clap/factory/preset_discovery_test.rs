// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#[cfg(test)]
mod tests {
    use crate::clap::factory::preset_discovery::*;

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
        let model_path = test_dir.join("BossWN-standard.nam");

        if model_path.exists() {
            let meta = extract_model_metadata(&model_path);
            assert!(
                meta.is_some(),
                "Should extract metadata from BossWN-standard.nam"
            );
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
