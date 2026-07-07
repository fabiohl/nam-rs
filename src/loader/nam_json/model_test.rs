// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

#[test]
fn parse_linear_implementation_case_insensitive() {
    assert_eq!("auto".parse(), Ok(LinearImplementation::Auto));
    assert_eq!("Auto".parse(), Ok(LinearImplementation::Auto));
    assert_eq!("AUTO".parse(), Ok(LinearImplementation::Auto));
    assert_eq!("direct".parse(), Ok(LinearImplementation::Direct));
    assert_eq!("Direct".parse(), Ok(LinearImplementation::Direct));
    assert_eq!("DIRECT".parse(), Ok(LinearImplementation::Direct));
    assert_eq!("fft".parse(), Ok(LinearImplementation::Fft));
    assert_eq!("Fft".parse(), Ok(LinearImplementation::Fft));
    assert_eq!("FFT".parse(), Ok(LinearImplementation::Fft));
}

#[test]
fn parse_linear_implementation_invalid() {
    assert_eq!("legacy".parse::<LinearImplementation>(), Err(()));
    assert_eq!("unknown".parse::<LinearImplementation>(), Err(()));
    assert_eq!("".parse::<LinearImplementation>(), Err(()));
}
