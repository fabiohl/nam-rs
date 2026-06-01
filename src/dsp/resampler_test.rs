// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

#[test]
fn test_bypass_48k() {
    // Quando a taxa de entrada é igual à de saída (ex: 48kHz para 48kHz),
    // o sistema entra em modo "Bypass" (atalho), apenas copiando o som sem processar.
    let mut rs = NamResampler::new(48_000, 48_000, 256).expect("new falhou");
    assert!(rs.is_bypass(), "48k deve ser bypass");

    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let input_r = [5.0f32, 4.0, 3.0, 2.0, 1.0];
    let mut output = [0.0f32; 5];
    let mut output_r = [0.0f32; 5];

    let n = rs.process_input(&input, &input_r, &mut output, &mut output_r);
    assert_eq!(n, 5);
    assert_eq!(output, input, "bypass deve copiar exatamente L");
    assert_eq!(output_r, input_r, "bypass deve copiar exatamente R");

    let n2 = rs.process_output(&input, &input_r, &mut output, &mut output_r);
    assert_eq!(n2, 5);
    assert_eq!(output, input);
    assert_eq!(output_r, input_r);
}

#[test]
fn test_downsample_96k_to_48k() {
    // Teste de Redução (Downsampling): Passando de alta resolução (96kHz) para o padrão (48kHz).
    // Esperamos que o número de amostras na saída seja aproximadamente metade do número na entrada.
    let chunk = 512usize;
    let mut rs = NamResampler::new(96_000, 48_000, chunk).expect("new falhou");
    assert!(!rs.is_bypass());

    let input = vec![0.5f32; chunk];
    let input_r = vec![0.5f32; chunk];
    let mut output = vec![0.0f32; chunk * 2];
    let mut output_r = vec![0.0f32; chunk * 2];
    let n = rs.process_input(&input, &input_r, &mut output, &mut output_r);

    let expected_approx = chunk / 2;
    assert!(
        n >= expected_approx.saturating_sub(64) && n <= expected_approx + 64,
        "96k→48k: esperado ~{expected_approx} amostras, obteve {n}"
    );
}

#[test]
fn test_upsample_44k_to_48k() {
    // Teste de Aumento (Upsampling): Passando de qualidade de CD (44.1kHz) para o padrão de estúdio (48kHz).
    // Aqui o número de amostras aumenta ligeiramente (proporção de aproximadamente 10% a mais).
    let chunk = 441usize;
    let mut rs = NamResampler::new(44_100, 48_000, chunk).expect("new falhou");
    assert!(!rs.is_bypass());

    let input = vec![0.3f32; chunk];
    let input_r = vec![0.3f32; chunk];
    let mut output = vec![0.0f32; chunk * 2];
    let mut output_r = vec![0.0f32; chunk * 2];
    let n = rs.process_input(&input, &input_r, &mut output, &mut output_r);

    let expected_approx = (chunk as f64 * 48_000.0 / 44_100.0) as usize;
    assert!(
        n >= expected_approx.saturating_sub(64) && n <= expected_approx + 64,
        "44.1k→48k: esperado ~{expected_approx} amostras, obteve {n}"
    );
}

#[test]
fn test_output_upsample_48k_to_96k() {
    // Teste de Aumento na Saída: Quando o som interno do modelo (48kHz)
    // precisa ser enviado para uma placa de som configurada em 96kHz.
    let chunk = 256usize;
    let mut rs = NamResampler::new(96_000, 48_000, chunk).expect("new falhou");

    let inner_out_size = chunk / 2;
    let input = vec![0.4f32; inner_out_size];
    let input_r = vec![0.4f32; inner_out_size];
    let mut output = vec![0.0f32; chunk * 2];
    let mut output_r = vec![0.0f32; chunk * 2];
    let n = rs.process_output(&input, &input_r, &mut output, &mut output_r);

    let expected_approx = inner_out_size * 2;
    assert!(
        n >= expected_approx.saturating_sub(64) && n <= expected_approx + 64,
        "48k→96k (output): esperado ~{expected_approx} amostras, obteve {n}"
    );
}

#[test]
fn test_roundtrip_96k() {
    // Teste de "Ida e Volta" (Roundtrip): O teste mais rigoroso.
    // Convertemos de 96kHz para 48kHz (ida) e depois voltamos para 96kHz (volta).
    // Ao final desse "telefone sem fio", o som deve ser preservado e ter energia (não pode ficar mudo).
    let chunk = 1024usize;
    let mut rs = NamResampler::new(96_000, 48_000, chunk).expect("new falhou");

    let n_total = chunk * 4;
    let input: Vec<f32> = (0..n_total)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 96_000.0).sin())
        .collect();
    let input_r = input.clone();

    let mut total_mid = 0usize;
    let mut total_out = 0usize;
    let mut mid_energy_sum = 0.0f32;
    let mut out_energy_sum = 0.0f32;

    for start in (0..n_total).step_by(chunk) {
        let end = (start + chunk).min(n_total);
        let blk_l = &input[start..end];
        let blk_r = &input_r[start..end];

        let mut mid = vec![0.0f32; chunk];
        let mut mid_r = vec![0.0f32; chunk];
        let n_mid = rs.process_input(blk_l, blk_r, &mut mid, &mut mid_r);
        total_mid += n_mid;
        mid_energy_sum += mid[..n_mid].iter().map(|x| x * x).sum::<f32>();

        if n_mid > 0 {
            let mut out = vec![0.0f32; chunk * 2];
            let mut out_r = vec![0.0f32; chunk * 2];
            let n_out = rs.process_output(&mid[..n_mid], &mid_r[..n_mid], &mut out, &mut out_r);
            total_out += n_out;
            out_energy_sum += out[..n_out].iter().map(|x| x * x).sum::<f32>();
        }
    }

    assert!(
        total_mid > 0,
        "process_input não produziu nenhuma amostra em {n_total} frames"
    );
    assert!(
        total_out > 0,
        "process_output não produziu nenhuma amostra (mid_total={total_mid})"
    );
    assert!(
        mid_energy_sum > 0.0,
        "Energia intermediária (96→48) é zero (mid_total={total_mid})"
    );

    let energy_in = input.iter().map(|x| x * x).sum::<f32>() / n_total as f32;
    let energy_out = out_energy_sum / total_out.max(1) as f32;

    assert!(
        energy_out > energy_in * 0.05,
        "Energia no roundtrip colapsou: in={energy_in:.4}, out={energy_out:.4}, \
         mid_samples={total_mid}, out_samples={total_out}, mid_energy={mid_energy_sum:.4}"
    );
}

#[test]
fn test_impulse_response_input() {
    // Teste de "Clique" (Impulso): Enviamos um único estalo digital (um valor 1.0)
    // seguido de silêncio para ver como o filtro reage. É como bater em um sino
    // e medir a vibração resultante: o som deve aparecer e sumir naturalmente.
    let chunk = 512usize;
    let mut rs = NamResampler::new(96_000, 48_000, chunk).expect("new falhou");

    let mut input = vec![0.0f32; chunk];
    input[0] = 1.0;
    let mut input_r = vec![0.0f32; chunk];
    input_r[0] = 1.0;

    let mut output = vec![0.0f32; chunk];
    let mut output_r = vec![0.0f32; chunk];
    let n = rs.process_input(&input, &input_r, &mut output, &mut output_r);
    assert!(n > 0);

    let energy: f32 = output[..n].iter().map(|x| x * x).sum();
    assert!(
        energy > 0.0 && energy.is_finite(),
        "Resposta ao impulso inválida: energy={energy}"
    );
    let peak = output[..n].iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(peak <= 1.5, "Pico excessivo: {peak:.4}");
}

#[test]
fn test_impulse_response_output() {
    // Mesma lógica do teste anterior (teste do clique/sino), mas aplicado à saída.
    // Verificamos se o sinal de saída mantém energia e não causa distorções bizarras.
    let chunk = 256usize;
    let mut rs = NamResampler::new(96_000, 48_000, chunk).expect("new falhou");

    let inner_out_approx = chunk / 2;
    let mut input = vec![0.0f32; inner_out_approx];
    input[0] = 1.0;
    let mut input_r = vec![0.0f32; inner_out_approx];
    input_r[0] = 1.0;

    let mut output = vec![0.0f32; chunk];
    let mut output_r = vec![0.0f32; chunk];
    let n = rs.process_output(&input, &input_r, &mut output, &mut output_r);
    assert!(n > 0);

    let energy: f32 = output[..n].iter().map(|x| x * x).sum();
    assert!(
        energy > 0.0 && energy.is_finite(),
        "Resposta ao impulso (output) inválida: energy={energy}"
    );
    let peak = output[..n].iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(peak <= 4.0, "Pico excessivo (output): {peak:.4}");
}

#[test]
fn test_phase_accum_underflow_guard() {
    // Usamos ResamplerCore diretamente para testar a lógica interna
    let mut core = ResamplerCore::new(44100, 48000);

    // Simula um drift negativo (underflow)
    core.phase_accum = -1e-15;

    let in_l = [0.0f32; 64];
    let in_r = [0.0f32; 64];
    let mut out_l = [0.0f32; 64];
    let mut out_r = [0.0f32; 64];

    // O processamento deve ocorrer sem pânico (o 'as usize' do clamp 0.0 é seguro)
    let n = dispatch_simd!(core, process_internal, &in_l, &in_r, &mut out_l, &mut out_r);
    assert!(n > 0);
    assert!(
        core.phase_accum >= 0.0,
        "Accumulator deve ter sido clampado para >= 0"
    );
}

#[test]
fn test_resampler_micro_soak() {
    // Teste de "micro-estabilidade" para CI.
    // Processa 1M de amostras para diversas taxas e verifica invariantes.
    let rate_pairs = [
        (44100, 48000),
        (48000, 44100),
        (96000, 48000),
        (22050, 48000),
        (88200, 48000),
    ];

    let chunk_size = 512;
    let n_iterations = 2000; // Total ~1M samples per pair

    let in_l = vec![0.1f32; chunk_size];
    let in_r = vec![0.1f32; chunk_size];
    let mut out_l = vec![0.0f32; chunk_size * 4];
    let mut out_r = vec![0.0f32; chunk_size * 4];

    for (from, to) in rate_pairs {
        let mut rs = NamResampler::new(from, to, chunk_size).unwrap();

        for _ in 0..n_iterations {
            let n = rs.process_input(&in_l, &in_r, &mut out_l, &mut out_r);

            // Invariante básico: saída deve ser finita
            for i in 0..n {
                assert!(out_l[i].is_finite());
                assert!(out_r[i].is_finite());
            }

            // Acessa o core interno para verificar o acumulador (via ResamplerCore)
            if let Some(ref core) = rs.inner {
                assert!(
                    core.phase_accum >= 0.0,
                    "Underflow detectado em {}->{}",
                    from,
                    to
                );
                // O acumulador pode ser >= NUM_PHASES se o bloco de entrada terminou
                // antes de consumir o necessário para a próxima amostra de saída.
                assert!(
                    core.phase_accum < NUM_PHASES as f64 + core.phase_step * 2.0,
                    "Overflow detectado em {}->{}",
                    from,
                    to
                );
            }
        }
    }
}

#[test]
fn test_latency_calculation() {
    // 1. Bypass: latência deve ser 0
    let rs_48 = NamResampler::new(48_000, 48_000, 256).unwrap();
    assert_eq!(rs_48.latency_samples(48_000), 0);

    // 2. 44.1k -> 48k (Upsampling)
    // latency_in  = 16 * (44100 / 48000) = 14.7
    // latency_out = 16
    // total = 30.7 -> round -> 31 samples
    let rs_44 = NamResampler::new(44_100, 48_000, 256).unwrap();
    assert_eq!(rs_44.latency_samples(44_100), 31);

    // 3. 96k -> 48k (Downsampling)
    // latency_in  = 16 * (96000 / 48000) = 32
    // latency_out = 16
    // total = 48 samples
    let rs_96 = NamResampler::new(96_000, 48_000, 256).unwrap();
    assert_eq!(rs_96.latency_samples(96_000), 48);

    // 4. Zero rate: deve retornar 0 (guardrail)
    assert_eq!(rs_44.latency_samples(0), 0);
}

#[test]
fn test_resampler_mono_equivalence() {
    // 1. Teste de Equivalência em modo Bypass (48kHz -> 48kHz)
    {
        let mut rs = NamResampler::new(48_000, 48_000, 256).unwrap();
        assert!(rs.is_bypass());

        let in_l = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut out_l_stereo = [0.0f32; 5];
        let mut out_r_stereo = [0.0f32; 5];
        let mut out_l_mono = [0.0f32; 5];
        let mut out_r_mono = [0.0f32; 5];

        let n_stereo = rs.process_input(&in_l, &in_l, &mut out_l_stereo, &mut out_r_stereo);
        let n_mono = rs.process_input_mono(&in_l, &mut out_l_mono, &mut out_r_mono);

        assert_eq!(n_stereo, n_mono);
        assert_eq!(out_l_stereo, out_l_mono);
        assert_eq!(out_r_stereo, out_r_mono);
        assert_eq!(out_l_mono, out_r_mono);

        // Mesma coisa para process_output
        let n_out_stereo = rs.process_output(&in_l, &in_l, &mut out_l_stereo, &mut out_r_stereo);
        let n_out_mono = rs.process_output_mono(&in_l, &mut out_l_mono, &mut out_r_mono);

        assert_eq!(n_out_stereo, n_out_mono);
        assert_eq!(out_l_stereo, out_l_mono);
        assert_eq!(out_r_stereo, out_r_mono);
    }

    // 2. Teste de Equivalência em modo ativo (44.1kHz -> 48kHz)
    {
        let chunk = 256;
        let mut rs_stereo = NamResampler::new(44_100, 48_000, chunk).unwrap();
        let mut rs_mono = NamResampler::new(44_100, 48_000, chunk).unwrap();

        let in_l: Vec<f32> = (0..chunk).map(|i| (i as f32 * 0.05).sin()).collect();
        let mut out_l_stereo = vec![0.0f32; chunk * 2];
        let mut out_r_stereo = vec![0.0f32; chunk * 2];
        let mut out_l_mono = vec![0.0f32; chunk * 2];
        let mut out_r_mono = vec![0.0f32; chunk * 2];

        let n_stereo = rs_stereo.process_input(&in_l, &in_l, &mut out_l_stereo, &mut out_r_stereo);
        let n_mono = rs_mono.process_input_mono(&in_l, &mut out_l_mono, &mut out_r_mono);

        assert_eq!(n_stereo, n_mono);
        for i in 0..n_stereo {
            assert!((out_l_stereo[i] - out_l_mono[i]).abs() < 1e-4, "Mismatch L at index {}: stereo={}, mono={}", i, out_l_stereo[i], out_l_mono[i]);
            assert!((out_r_stereo[i] - out_r_mono[i]).abs() < 1e-4, "Mismatch R at index {}: stereo={}, mono={}", i, out_r_stereo[i], out_r_mono[i]);
            assert_eq!(out_l_mono[i], out_r_mono[i]);
        }

        // Output stage (48kHz -> 44.1kHz)
        let mut rs_out_stereo = NamResampler::new(48_000, 44_100, chunk).unwrap();
        let mut rs_out_mono = NamResampler::new(48_000, 44_100, chunk).unwrap();

        let in_mid = &out_l_stereo[..n_stereo];
        let mut out_final_l_stereo = vec![0.0f32; chunk * 2];
        let mut out_final_r_stereo = vec![0.0f32; chunk * 2];
        let mut out_final_l_mono = vec![0.0f32; chunk * 2];
        let mut out_final_r_mono = vec![0.0f32; chunk * 2];

        let n_final_stereo = rs_out_stereo.process_output(in_mid, in_mid, &mut out_final_l_stereo, &mut out_final_r_stereo);
        let n_final_mono = rs_out_mono.process_output_mono(in_mid, &mut out_final_l_mono, &mut out_final_r_mono);

        assert_eq!(n_final_stereo, n_final_mono);
        for i in 0..n_final_stereo {
            assert!((out_final_l_stereo[i] - out_final_l_mono[i]).abs() < 1e-4, "Mismatch final L at index {}: stereo={}, mono={}", i, out_final_l_stereo[i], out_final_l_mono[i]);
            assert!((out_final_r_stereo[i] - out_final_r_mono[i]).abs() < 1e-4, "Mismatch final R at index {}: stereo={}, mono={}", i, out_final_r_stereo[i], out_final_r_mono[i]);
            assert_eq!(out_final_l_mono[i], out_final_r_mono[i]);
        }
    }
}

