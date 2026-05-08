// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

#[cfg(test)]
mod tests {
    use crate::dsp::gate::*;

    #[test]
    fn test_gate_params_default() {
        let params = GateParams::default();
        assert_eq!(params.threshold_open_db, -70.0);
        assert_eq!(params.threshold_close_db, -80.0);
        assert_eq!(params.hold_frames, 2048);
        assert_eq!(params.fade_frames, 256);
    }

    #[test]
    fn test_hysteresis_basic_transitions() {
        let mut dh = DynamicHysteresis::new();
        let params = GateParams {
            threshold_open_db: -10.0,
            threshold_close_db: -20.0,
            hold_frames: 10,
            fade_frames: 10,
            mono_epsilon: 1e-4,
        };
        // Usamos valores lineares diretos para o teste (como se fossem squared energy)
        let th_open = 1.0;
        let th_close = 0.5;

        assert_eq!(dh.state(), GateState::Open);
        assert_eq!(dh.multiplier(), 1.0);

        // 1. Sinal cai abaixo do threshold de fechamento
        dh.update(0.1, th_open, th_close, &params, 5);
        assert_eq!(
            dh.state(),
            GateState::Open,
            "Deve permanecer Open durante o hold"
        );

        // 2. Passa o tempo de hold
        dh.update(0.1, th_open, th_close, &params, 5);
        assert_eq!(
            dh.state(),
            GateState::FadingOut,
            "Deve entrar em FadingOut após hold_frames"
        );

        // 3. Meio do fade out
        dh.update(0.1, th_open, th_close, &params, 5);
        assert_eq!(dh.state(), GateState::FadingOut);
        assert_eq!(dh.multiplier(), 0.5); // 5/10

        // 4. Conclui fade out
        dh.update(0.1, th_open, th_close, &params, 5);
        assert_eq!(dh.state(), GateState::Closed);
        assert_eq!(dh.multiplier(), 0.0);

        // 5. Sinal volta (acima do open)
        dh.update(2.0, th_open, th_close, &params, 1);
        assert_eq!(dh.state(), GateState::FadingIn);
        assert_eq!(
            dh.multiplier(),
            0.1,
            "Transição inicial de Closed para FadingIn inicia o fade imediatamente"
        );

        // 6. Progresso do fade in
        dh.update(2.0, th_open, th_close, &params, 4);
        assert_eq!(dh.state(), GateState::FadingIn);
        assert_eq!(dh.multiplier(), 0.5); // 5/10

        // 7. Conclui fade in
        dh.update(2.0, th_open, th_close, &params, 5);
        assert_eq!(dh.state(), GateState::Open);
        assert_eq!(dh.multiplier(), 1.0);
    }

    #[test]
    fn test_hysteresis_interrupted_fade() {
        let mut dh = DynamicHysteresis::new();
        let params = GateParams {
            hold_frames: 10,
            fade_frames: 10,
            ..Default::default()
        };
        let th_open = 1.0;
        let th_close = 0.5;

        // Força para FadingOut
        dh.update(0.1, th_open, th_close, &params, 11);
        assert_eq!(dh.state(), GateState::FadingOut);

        // Avança fade out até a metade
        dh.update(0.1, th_open, th_close, &params, 5);
        assert_eq!(dh.multiplier(), 0.5);

        // Interrompe com sinal alto -> Deve entrar em FadingIn IMEDIATAMENTE a partir de onde parou
        dh.update(2.0, th_open, th_close, &params, 1);
        assert_eq!(dh.state(), GateState::FadingIn);
        assert_eq!(dh.multiplier(), 0.6);

        // Avança um pouco o fade in
        dh.update(2.0, th_open, th_close, &params, 2);
        assert_eq!(dh.multiplier(), 0.8);

        // Interrompe novamente com silêncio
        dh.update(0.1, th_open, th_close, &params, 1);
        assert_eq!(dh.state(), GateState::FadingOut);
        assert_eq!(dh.multiplier(), 0.1);
    }

    #[test]
    fn test_hysteresis_apply_gain_ramp() {
        let params = GateParams {
            hold_frames: 2048,
            fade_frames: 100,
            ..Default::default()
        };
        let mut buffer = [1.0f32; 10];

        let mut dh = DynamicHysteresis::new();
        dh.update(0.0, 1.0, 0.5, &params, 2047); // Quase no limite do hold
        assert_eq!(dh.state(), GateState::Open);

        dh.update(0.0, 1.0, 0.5, &params, 10); // Passa o hold e inicia FadingOut
        assert_eq!(dh.state(), GateState::FadingOut);
        assert_eq!(
            dh.multiplier(),
            1.0,
            "No bloco de transição o multiplier ainda é 1.0"
        );

        dh.update(0.0, 1.0, 0.5, &params, 10); // Primeiro bloco real de fade
        assert_eq!(dh.multiplier(), 0.9); // 90/100

        buffer.fill(1.0);
        dh.apply_gain_rt(&mut buffer, 10);
        // Deve ser rampa de 1.0 a 0.9. O primeiro sample (index 0) recebe o ganho 'start' (1.0).
        assert!((buffer[0] - 1.0).abs() < 1e-3);
        assert!((buffer[9] - 0.91).abs() < 1e-3);

        // Caso FadingIn: deve aplicar rampa ascendente
        let mut dh = DynamicHysteresis::new();
        dh.update(0.0, 1.0, 0.5, &params, 2048); // Passa hold -> FadingOut
        dh.update(0.0, 1.0, 0.5, &params, 101); // Passa fade -> Closed
        assert_eq!(dh.state(), GateState::Closed);

        dh.update(2.0, 1.0, 0.5, &params, 1); // Transição para FadingIn, counter=0 -> 1
        assert_eq!(dh.multiplier(), 0.01);

        dh.update(2.0, 1.0, 0.5, &params, 10); // Avança fade in, counter=11, mult=0.11
        assert_eq!(dh.multiplier(), 0.11);

        buffer.fill(1.0);
        dh.apply_gain_rt(&mut buffer, 10);
        // Deve ser rampa de 0.01 a 0.11. O primeiro sample recebe 0.01.
        assert!((buffer[0] - 0.01).abs() < 1e-3);
        assert!((buffer[9] - 0.10).abs() < 1e-3);
    }

    #[test]
    fn test_sub_block_granularity() {
        // Teste da Tarefa 4.1 / 4.2
        let mut dh = DynamicHysteresis::new();
        let params = GateParams {
            hold_frames: 2048,
            fade_frames: 256,
            ..Default::default()
        };
        let th_open = 1.0;
        let th_close = 0.5;

        // Força transição para FadingOut
        dh.update(0.0, th_open, th_close, &params, 2048);
        assert_eq!(dh.state(), GateState::FadingOut);

        // Passa um bloco enorme de 4096 amostras, mas o fade_frames é só 256!
        // A rampa deve acontecer estritamente nos primeiros 256 samples,
        // e o resto do bloco (3840 samples) deve ser zerado.
        dh.update(0.0, th_open, th_close, &params, 4096);
        assert_eq!(dh.state(), GateState::Closed);
        assert_eq!(dh.multiplier(), 0.0);

        let mut buffer = vec![1.0f32; 4096];
        dh.apply_gain_rt(&mut buffer, 4096);

        // Verifica a interpolação estrita nos primeiros 256 samples
        assert!(
            (buffer[0] - 1.0).abs() < 1e-3,
            "Início do fade-out deve ser 1.0"
        );
        assert!(
            (buffer[128] - 0.5).abs() < 1e-2,
            "Meio do fade-out deve ser 0.5"
        );

        // A rampa termina no sample 255. O sample 255 deve ter (256 - 255) / 256 ~= 0.0039.
        // O sample 256 deve ser estritamente 0.0.
        assert_eq!(buffer[256], 0.0, "Fim da rampa não zerou estritamente");
        assert_eq!(
            buffer[4095], 0.0,
            "Restante do buffer não preenchido com zeros"
        );

        // Transição imediata de Closed para Open (FadingIn) com bloco de 4096
        dh.update(2.0, th_open, th_close, &params, 4096);
        assert_eq!(dh.state(), GateState::Open);
        assert_eq!(dh.multiplier(), 1.0);

        let mut buffer2 = vec![1.0f32; 4096];
        dh.apply_gain_rt(&mut buffer2, 4096);

        // Verifica a interpolação estrita nos primeiros 256 samples (FadingIn)
        assert_eq!(buffer2[0], 0.0, "Início do fade-in deve ser 0.0");
        assert!(
            (buffer2[128] - 0.5).abs() < 1e-2,
            "Meio do fade-in deve ser 0.5"
        );

        // A partir do sample 256, o sinal deve estar 100% aberto
        assert_eq!(
            buffer2[256], 1.0,
            "Fim da rampa não abriu estritamente para 1.0"
        );
        assert_eq!(
            buffer2[4095], 1.0,
            "Restante do buffer não preservado em 1.0"
        );
    }
}
