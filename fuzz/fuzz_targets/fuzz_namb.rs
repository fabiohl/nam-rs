#![no_main]

use libfuzzer_sys::fuzz_target;
use nam_rs::loader::namb::parse_namb;

fuzz_target!(|data: &[u8]| {
    let _ = parse_namb(data);
});
