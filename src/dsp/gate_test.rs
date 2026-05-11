// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

#[cfg(test)]
/// Módulo de testes para o Noise Gate (redutor de ruído).
/// O Noise Gate silencia o áudio quando o volume cai abaixo de um certo nível,
/// eliminando ruídos indesejados (como o chiado de uma guitarra parada) ou
/// quando queremos silenciar partes específicas do áudio.
mod tests {
    use crate::dsp::gate::*;

    /// Verifica se as configurações padrão do redutor de ruído estão corretas.
    #[test]
    fn test_gate_params_default() {
        let params = GateParams::default();
        // O padrão é começar a fechar em -80dB e abrir em -70dB.
        assert_eq!(params.threshold_open_db, -70.0);
        assert_eq!(params.threshold_close_db, -80.0);
        // Tempo que ele espera antes de começar a fechar (hold) e tempo de suavização (fade).
        assert_eq!(params.hold_frames, 2048);
        assert_eq!(params.fade_frames, 256);
    }

    /// Testa o ciclo de vida básico do portão de ruído:
    /// Aberto -> Esperando (Hold) -> Fechando (FadeOut) -> Fechado -> Abrindo (FadeIn) -> Aberto.
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
        // Usamos valores simples para o teste: 1.0 é "alto", 0.1 é "silêncio".
        let th_open = 1.0;
        let th_close = 0.5;

        // Começa totalmente aberto.
        assert_eq!(dh.state(), GateState::Open);
        assert_eq!(dh.multiplier(), 1.0);

        // 1. O volume cai abaixo do nível de fechamento.
        // Ele deve continuar aberto por um tempo (período de 'hold').
        dh.update(0.1, th_open, th_close, &params, 5);
        assert_eq!(
            dh.state(),
            GateState::Open,
            "Deve permanecer Open durante o hold"
        );

        // 2. O tempo de 'hold' acaba. Agora ele começa a fechar suavemente.
        dh.update(0.1, th_open, th_close, &params, 5);
        assert_eq!(
            dh.state(),
            GateState::FadingOut,
            "Deve entrar em FadingOut após hold_frames"
        );

        // 3. No meio do fechamento (fade out), o volume deve estar pela metade.
        dh.update(0.1, th_open, th_close, &params, 5);
        assert_eq!(dh.state(), GateState::FadingOut);
        assert_eq!(dh.multiplier(), 0.5); // 5 de 10 passos concluídos.

        // 4. Conclui o fechamento. Volume agora é zero (totalmente silenciado).
        dh.update(0.1, th_open, th_close, &params, 5);
        assert_eq!(dh.state(), GateState::Closed);
        assert_eq!(dh.multiplier(), 0.0);

        // 5. O som volta a ficar alto (acima do nível de abertura).
        // Ele deve começar a abrir suavemente.
        dh.update(2.0, th_open, th_close, &params, 1);
        assert_eq!(dh.state(), GateState::FadingIn);
        assert_eq!(
            dh.multiplier(),
            0.1,
            "Começa a abrir imediatamente quando o som volta"
        );

        // 6. Progresso da abertura (fade in).
        dh.update(2.0, th_open, th_close, &params, 4);
        assert_eq!(dh.state(), GateState::FadingIn);
        assert_eq!(dh.multiplier(), 0.5);

        // 7. Totalmente aberto novamente.
        dh.update(2.0, th_open, th_close, &params, 5);
        assert_eq!(dh.state(), GateState::Open);
        assert_eq!(dh.multiplier(), 1.0);
    }

    /// Testa o que acontece se o som voltar enquanto o portão ainda estava fechando.
    /// Ele deve parar de fechar e começar a abrir imediatamente de onde parou.
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

        // Força o início do fechamento (fade out).
        dh.update(0.1, th_open, th_close, &params, 11);
        assert_eq!(dh.state(), GateState::FadingOut);

        // Avança o fechamento até a metade (multiplier = 0.5).
        dh.update(0.1, th_open, th_close, &params, 5);
        assert_eq!(dh.multiplier(), 0.5);

        // O som volta forte no meio do fechamento!
        // O portão deve decidir abrir a partir de onde estava.
        dh.update(2.0, th_open, th_close, &params, 1);
        assert_eq!(dh.state(), GateState::FadingIn);
        assert_eq!(dh.multiplier(), 0.6); // Estava em 0.5 e subiu para 0.6.

        // Avança mais um pouco a abertura.
        dh.update(2.0, th_open, th_close, &params, 2);
        assert_eq!(dh.multiplier(), 0.8);

        // O som some de novo. Ele volta a fechar imediatamente.
        dh.update(0.1, th_open, th_close, &params, 1);
        assert_eq!(dh.state(), GateState::FadingOut);
        assert_eq!(dh.multiplier(), 0.1);
    }

    /// Verifica se a suavização do volume (rampa de ganho) está sendo aplicada corretamente no áudio.
    #[test]
    fn test_hysteresis_apply_gain_ramp() {
        let params = GateParams {
            hold_frames: 2048,
            fade_frames: 100,
            ..Default::default()
        };
        let mut buffer = [1.0f32; 10];

        let mut dh = DynamicHysteresis::new();
        // Simula silêncio quase completando o tempo de espera (hold).
        dh.update(0.0, 1.0, 0.5, &params, 2047);
        assert_eq!(dh.state(), GateState::Open);

        // Passa o hold e inicia o processo de fechamento suave (fade out).
        dh.update(0.0, 1.0, 0.5, &params, 10);
        assert_eq!(dh.state(), GateState::FadingOut);
        assert_eq!(
            dh.multiplier(),
            1.0,
            "No bloco de transição o multiplier ainda é 1.0"
        );

        // Primeiro bloco real de fade.
        dh.update(0.0, 1.0, 0.5, &params, 10);
        assert_eq!(dh.multiplier(), 0.9); // 90 de 100 passos concluídos.

        buffer.fill(1.0);
        dh.apply_gain_rt(&mut buffer, 10);
        // Deve ser uma rampa suave: o primeiro sample mantém o volume e o último já está reduzido.
        assert!((buffer[0] - 1.0).abs() < 1e-3);
        assert!((buffer[9] - 0.91).abs() < 1e-3);

        // Testa agora a abertura suave (FadingIn).
        let mut dh = DynamicHysteresis::new();
        dh.update(0.0, 1.0, 0.5, &params, 2048); // Força fechar.
        dh.update(0.0, 1.0, 0.5, &params, 101); // Garante que fechou totalmente.
        assert_eq!(dh.state(), GateState::Closed);

        // Som volta, inicia a abertura.
        dh.update(2.0, 1.0, 0.5, &params, 1);
        assert_eq!(dh.multiplier(), 0.01);

        dh.update(2.0, 1.0, 0.5, &params, 10);
        assert_eq!(dh.multiplier(), 0.11);

        buffer.fill(1.0);
        dh.apply_gain_rt(&mut buffer, 10);
        // O volume deve subir gradualmente de quase zero (0.01) para 0.11.
        assert!((buffer[0] - 0.01).abs() < 1e-3);
        assert!((buffer[9] - 0.10).abs() < 1e-3);
    }

    /// Testa como o sistema lida com blocos de áudio muito grandes de uma só vez.
    /// A suavização deve acontecer apenas no tempo correto e o resto deve ser processado.
    #[test]
    fn test_sub_block_granularity() {
        let mut dh = DynamicHysteresis::new();
        let params = GateParams {
            hold_frames: 2048,
            fade_frames: 256,
            ..Default::default()
        };
        let th_open = 1.0;
        let th_close = 0.5;

        // Prepara para fechar (fade out).
        dh.update(0.0, th_open, th_close, &params, 2048);
        assert_eq!(dh.state(), GateState::FadingOut);

        // Processa um bloco gigante de 4096 amostras, mas o tempo de suavização é só de 256!
        // O sistema deve fechar nos primeiros 256 samples e silenciar o restante do bloco.
        dh.update(0.0, th_open, th_close, &params, 4096);
        assert_eq!(dh.state(), GateState::Closed);
        assert_eq!(dh.multiplier(), 0.0);

        let mut buffer = vec![1.0f32; 4096];
        dh.apply_gain_rt(&mut buffer, 4096);

        // Verifica a suavização exata nos primeiros 256 samples.
        assert!(
            (buffer[0] - 1.0).abs() < 1e-3,
            "Início do fade-out deve ser volume máximo"
        );
        assert!(
            (buffer[128] - 0.5).abs() < 1e-2,
            "Meio do fade-out deve ser metade do volume"
        );

        // A partir do sample 256, tudo deve ser silêncio absoluto.
        assert_eq!(buffer[256], 0.0, "Fim da rampa não silenciou estritamente");
        assert_eq!(
            buffer[4095], 0.0,
            "Restante do buffer não preenchido com zeros"
        );

        // Testa o mesmo para a abertura (FadingIn) com bloco gigante.
        dh.update(2.0, th_open, th_close, &params, 4096);
        assert_eq!(dh.state(), GateState::Open);
        assert_eq!(dh.multiplier(), 1.0);

        let mut buffer2 = vec![1.0f32; 4096];
        dh.apply_gain_rt(&mut buffer2, 4096);

        // Começa em silêncio.
        assert_eq!(buffer2[0], 0.0, "Início do fade-in deve ser silêncio");
        assert!(
            (buffer2[128] - 0.5).abs() < 1e-2,
            "Meio do fade-in deve ser metade do volume"
        );

        // A partir do sample 256, o volume deve estar 100% aberto.
        assert_eq!(
            buffer2[256], 1.0,
            "Fim da rampa não abriu totalmente o volume"
        );
        assert_eq!(
            buffer2[4095], 1.0,
            "Restante do buffer não preservado em volume máximo"
        );
    }
}
