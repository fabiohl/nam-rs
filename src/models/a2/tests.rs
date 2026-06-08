// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#[cfg(test)]
mod tests_placeholder {
    use crate::models::NamModel;
    use crate::models::a2::WavenetA2Placeholder;

    #[test]
    fn test_wavenet_a2_placeholder_silence() {
        let mut model = WavenetA2Placeholder::new(8);
        let input = [1.0f32; 10];
        let mut output = [1.0f32; 10];
        model.process(&input, &mut output);
        for val in output.iter() {
            assert_eq!(*val, 0.0);
        }
    }
}
