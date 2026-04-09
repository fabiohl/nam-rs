# tests/fixtures — Golden Vectors para Validação Cross-Reference C++ ↔ Rust

## Arquivos neste diretório

   golden_wavenet_standard.bin  — Gerado pelo C++ NeuralAudio (BossWN-standard.nam)
   golden_lstm_1x16.bin         — Gerado pelo C++ NeuralAudio (BossLSTM-1x16.nam)

## Formato binário (.golden.bin)

   [u32 num_samples LE]
   [f32×N input samples LE]       — senoidal 440Hz a 48kHz (512 amostras)
   [f32×N expected output LE]     — output do C++ NeuralAudio Internal mode

## Para regenerar

   ./utils/golden_gen_build.sh

Estes arquivos são commitados no repositório para que os testes Rust
(test_golden_vectors_wavenet, test_golden_vectors_lstm) executem sem
precisar recompilar o NeuralAudio C++.
Se os golden vectors não existirem, os testes fazem skip gracioso.
