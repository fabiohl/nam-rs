// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#[cfg(test)]
mod tests_a2_real {
    use crate::models::a2::WaveNetA2;

    #[test]
    fn test_wavenet_a2_lite_process() {
        let mut model = WaveNetA2::<3>::new();
        model.prewarm();
        let input = vec![0.01f32; 64];
        let mut output = vec![0.0f32; 64];
        model.process(&input, &mut output);
        for &s in output.iter() {
            assert!(s.is_finite(), "A2-Lite output must be finite");
        }
    }

    #[test]
    fn test_wavenet_a2_full_process() {
        let mut model = WaveNetA2::<8>::new();
        model.prewarm();
        let input = vec![0.01f32; 64];
        let mut output = vec![0.0f32; 64];
        model.process(&input, &mut output);
        for &s in output.iter() {
            assert!(s.is_finite(), "A2-Full output must be finite");
        }
    }
}
