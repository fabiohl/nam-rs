// Minimal test: CH=12 dense layer with known weights, compare SIMD vs scalar
#[test]
fn test_dense_ch12_scalar_vs_simd() {
    use crate::math::common::AlignedVec;
    
    // Create a 12x12 DenseLayer with known weights
    let mut raw = vec![0.0f32; 12 * 12];
    
    // Fill with deterministic values
    for i in 0..12*12 {
        raw[i] = (i as f32 - 72.0) * 0.01; // range ~[-0.72, 0.72]
    }
    
    let bias = AlignedVec::from_vec(vec![0.1f32; 12]);
    
    let dense = crate::models::wavenet::DenseLayer::<12, 12> {
        bias: bias.clone(),
        do_bias: true,
        weights: AlignedVec::from_vec(raw.clone()),
    };
    
    let input: Vec<f32> = (0..12).map(|i| (i as f32 - 6.0) * 0.2).collect();
    let mut simd_out = vec![0.0f32; 12];
    let mut scalar_out = vec![0.0f32; 12];
    
    // SIMD path
    unsafe { dense.process_block::<crate::math::common::Avx2Math>(&input, &mut simd_out, 1); }
    
    // Scalar reference (column-major f32 weights: f32_weights[ic * out + oc])
    for oc in 0..12 {
        let mut sum = bias[oc];
        for ic in 0..12 {
            let w = dense.weights[ic * 12 + oc];
            sum += input[ic] * w;
        }
        scalar_out[oc] = sum;
    }
    
    for i in 0..12 {
        let diff = (simd_out[i] - scalar_out[i]).abs();
        assert!(diff < 1e-5, "Dense CH=12 SIMD mismatch at channel {}: {} vs {}, diff={}", i, simd_out[i], scalar_out[i], diff);
    }
}

#[test]
fn test_conv1d_ch12_scalar_vs_simd() {
    use crate::math::common::AlignedVec;
    use crate::models::wavenet::Conv1d;
    
    const CH: usize = 12;
    const K: usize = 3;
    let dil: usize = 1;
    
    // Create raw conv weights (row-major for scalar reference)
    let mut raw = vec![0.0f32; CH * K * CH];
    for i in 0..CH*K*CH {
        raw[i] = (i as f32 - (CH*K*CH/2) as f32) * 0.01;
    }
    
    // Transpose to interleaved-4-wide (f32)
    let mut weights_f32 = AlignedVec::new(CH * K * CH, 0.0f32);
    crate::loader::dispatcher::wavenet::transpose_conv1d_interleaved_4wide(
        &raw, &mut weights_f32, CH, CH, K);
    
    let bias = AlignedVec::from_vec(vec![0.01f32; CH]);
    
    let conv = Conv1d::<CH, CH, K> {
        weights: weights_f32,
        bias: bias.clone(),
        do_bias: true,
        dilation: dil,
        prefetch_fn: crate::math::common::prefetch_strategy_simple,
    };
    
    // Create state buffer (CH * 10 elements)
    let mut state = vec![0.0f32; CH * 10];
    for i in 0..CH*10 {
        state[i] = (i as f32 - (CH*5) as f32) * 0.1;
    }
    
    let frame_idx = 5;
    let mut simd_out = vec![0.0f32; CH];
    let mut scalar_out = vec![0.0f32; CH];
    
    // SIMD path
    unsafe { conv.process_single_frame(&state, &mut simd_out, frame_idx); }
    
    // Scalar reference with interleaved weights
    for oc in 0..CH {
        let mut sum = bias[oc];
        for k in 0..K {
            let offset = (dil as isize) * ((k as isize) + 1 - (K as isize));
            let in_idx = ((frame_idx as isize) + offset) as usize * CH;
            for ic in 0..CH {
                let b = oc / 4;
                let lane = oc % 4;
                let w_idx = b * (K * CH * 4) + k * (CH * 4) + ic * 4 + lane;
                let w = conv.weights[w_idx];
                sum += state[in_idx + ic] * w;
            }
        }
        scalar_out[oc] = sum;
    }
    
    let mut max_diff = 0.0f32;
    for i in 0..CH {
        let diff = (simd_out[i] - scalar_out[i]).abs();
        if diff > max_diff { max_diff = diff; }
    }
    assert!(max_diff < 1e-5, "Conv1D CH=12 SIMD mismatch: max_diff={}", max_diff);
}

