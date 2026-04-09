#![no_main]

use libfuzzer_sys::fuzz_target;
use nam_rs::loader::nam_json::parse_nam_json;

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        let _ = parse_nam_json(s);
    }
});
