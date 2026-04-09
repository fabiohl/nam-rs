// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

/// Golden Vector Generator — Gera arquivos `.golden.bin` de referência C++.
///
/// Uso: ./golden_gen <model.nam> <output.golden.bin>
///
/// Gera 512 amostras de sinal senoidal 440 Hz a 48 kHz, processa pelo motor
/// NeuralAudio Internal (C++), e grava o par (input, output) em formato binário
/// little-endian para validação cross-reference no motor Rust.
///
/// Formato de saída:
///   [u32 num_samples LE]
///   [f32×N input samples LE]
///   [f32×N expected output LE]

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include <NeuralAudio/NeuralModel.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static constexpr int NUM_SAMPLES = 512;
static constexpr int BLOCK_SIZE = 64;
static constexpr float SAMPLE_RATE = 48000.0f;
static constexpr float FREQ_HZ = 440.0f;

int main(int argc, char* argv[])
{
    if (argc != 3) {
        std::cerr << "Uso: " << argv[0] << " <model.nam> <output.golden.bin>" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Gera golden vectors (referência C++) para validação cross-reference Rust." << std::endl;
        std::cerr << "  model.nam       — arquivo .nam a processar" << std::endl;
        std::cerr << "  output.golden.bin — arquivo de saída binário" << std::endl;
        return 1;
    }

    const char* modelPath = argv[1];
    const char* outputPath = argv[2];

    // Configurar tamanho máximo de buffer
    NeuralAudio::NeuralModel::SetDefaultMaxAudioBufferSize(BLOCK_SIZE);

    // Forçar modo Internal (implementação estática otimizada)
    NeuralAudio::NeuralModel::SetWaveNetLoadMode(NeuralAudio::EModelLoadMode::Internal);
    NeuralAudio::NeuralModel::SetLSTMLoadMode(NeuralAudio::EModelLoadMode::Internal);

    // Carregar modelo
    NeuralAudio::NeuralModel* model = nullptr;
    try {
        model = NeuralAudio::NeuralModel::CreateFromFile(modelPath);
    } catch (const std::exception& e) {
        std::cerr << "Erro ao carregar modelo: " << e.what() << std::endl;
        return 1;
    }

    if (model == nullptr) {
        std::cerr << "Falha ao carregar modelo: " << modelPath << std::endl;
        return 1;
    }

    std::cout << "Modelo carregado: " << modelPath << std::endl;
    std::cout << "  Estático: " << (model->IsStatic() ? "sim" : "não") << std::endl;

    // Prewarm
    model->Prewarm();
    std::cout << "  Prewarm concluído." << std::endl;

    // Gerar sinal senoidal 440 Hz a 48 kHz (idêntico ao Rust)
    std::vector<float> input(NUM_SAMPLES);
    for (int i = 0; i < NUM_SAMPLES; i++) {
        input[i] = static_cast<float>(
            std::sin(2.0 * M_PI * FREQ_HZ * static_cast<double>(i) / static_cast<double>(SAMPLE_RATE))
        );
    }

    // Processar em blocos de BLOCK_SIZE
    std::vector<float> output(NUM_SAMPLES, 0.0f);
    for (int pos = 0; pos < NUM_SAMPLES; pos += BLOCK_SIZE) {
        int blockLen = std::min(BLOCK_SIZE, NUM_SAMPLES - pos);
        model->Process(input.data() + pos, output.data() + pos, blockLen);
    }

    // Verificar sanidade da saída
    bool hasNaN = false;
    for (int i = 0; i < NUM_SAMPLES; i++) {
        if (!std::isfinite(output[i])) {
            hasNaN = true;
            std::cerr << "WARN: saída[" << i << "] = " << output[i] << " (não finito)" << std::endl;
        }
    }
    if (hasNaN) {
        std::cerr << "ERRO: saída contém valores não finitos. Abortando." << std::endl;
        delete model;
        return 1;
    }

    // Gravar arquivo golden.bin
    std::ofstream ofs(outputPath, std::ios::binary);
    if (!ofs) {
        std::cerr << "Erro ao abrir arquivo de saída: " << outputPath << std::endl;
        delete model;
        return 1;
    }

    // u32 num_samples LE
    uint32_t numSamples = static_cast<uint32_t>(NUM_SAMPLES);
    ofs.write(reinterpret_cast<const char*>(&numSamples), sizeof(uint32_t));

    // f32×N input samples LE
    ofs.write(reinterpret_cast<const char*>(input.data()), NUM_SAMPLES * sizeof(float));

    // f32×N expected output LE
    ofs.write(reinterpret_cast<const char*>(output.data()), NUM_SAMPLES * sizeof(float));

    ofs.close();

    std::cout << "Golden vectors gravados em: " << outputPath << std::endl;
    std::cout << "  Amostras: " << NUM_SAMPLES << std::endl;
    std::cout << "  Tamanho: " << (4 + NUM_SAMPLES * 4 * 2) << " bytes" << std::endl;

    // Calcular RMS para referência
    double rms = 0.0;
    for (int i = 0; i < NUM_SAMPLES; i++) {
        rms += static_cast<double>(output[i]) * static_cast<double>(output[i]);
    }
    rms = std::sqrt(rms / static_cast<double>(NUM_SAMPLES));
    std::cout << "  RMS saída: " << rms << std::endl;

    delete model;
    return 0;
}
