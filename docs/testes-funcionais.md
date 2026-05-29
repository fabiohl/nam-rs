<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved. -->

# Roteiro-Mestre de Testes Funcionais (por humanos) — NAM-rs (Plugin CLAP)

**Público:** Especialistas UI/UX e usuários finais.
**DAWs focais:** Bitwig Studio 6+ e Fender Studio Pro 8+ (Linux, Flatpak).
**Preparação:** `~/.clap/nam-rs.clap` instalado (build Release), ≥2 modelos `.nam` disponíveis (sendo 1 deles um arquivo bogus/inválido), DI de guitarra ou gerador de sinal na trilha, `pw-top` aberto para vigiar XRUNs. (Recomendado: Buffer inicial de 128 samples @ 48 kHz).

---

## Bloco 1 — Primeira Sessão (Quick Wins, ~10 min)

Objetivo: primeira impressão — layout, carregamento, som, controles básicos.

### 1.1 Layout e identidade

- [ ] Abrir GUI do NAM-rs → janela fixa **600×275 px** (sem decoração de host).
- [ ] Zona 1 (esquerda): logo `"NAM-rs⚡"` turquesa, subtítulo `"Neural Amp Modeler"`, versão + badge SIMD, botão `[📂 Load Model]`, caixa de modelo com fundo escuro.
- [ ] Zona 2 (centro): 3 knobs — **INPUT** (70px, turquesa), **OUTPUT** (70px, turquesa), **GATE** (42px, âmbar).
- [ ] Zona 3 (direita): medidor VU **adaptativo** — 1 barra centralizada (sem label) de 16px (mono).
- [ ] Zona 4 (extrema direita): toggle **BYPASS** com LED e label `"ACTIVE"`/`"BYPASSED"`.
- [ ] Zona 5 (footer): status bar com telemetria RT (sample rate, latência, DSP load, CPU cycles, last N samples, RT priority, overruns/overloads, flags) e linha inferior com metadados do modelo (se carregado).
- [ ] 3 separadores verticais finos visíveis entre as zonas 1–4.

### 1.2 Primeiro carregamento e som

- [ ] `[📂 Load Model]` → picker do sistema abre sem travar DAW.
- [ ] Selecionar modelo `.nam` → animação `"Loading"` → `"Loading."` → `"Loading.."` → `"Loading..."` → nome do modelo. Áudio processado audível imediatamente.
- [ ] Cancelar picker → volta ao estado anterior, botão segue clicável.
- [ ] Arrastar knob INPUT → volume muda sem estalos (zipper noise). Arco turquesa acompanha fluido.
- [ ] Bypass ON → LED cinza, label `"BYPASSED"`, áudio = sinal limpo (dry). Bypass OFF → volta processamento sem estalo.

✅ **PASS rápido:** GUI bonita, carrega modelo, faz som, knobs e bypass funcionam.

---

## Bloco 2 — Validação por Funcionalidade

Cada seção testável após mexer na feature correspondente. Autocontida, ~5–15 min cada.

---

### 2A — File Picker & Thread Safety

- [ ] Picker aberto → DAW responsiva (arraste janela, mova faders de outra trilha). Playback não para.
- [ ] Carregar modelo diferente por cima → nome atualiza, áudio muda sem parar playback.
- [ ] Arquivo inválido (`invalid.nam`) → `"⚠ Load failed"` vermelho por ~3s, depois volta ao estado anterior. Sem crash.
- [ ] Arquivo `.nam` de 0 bytes → mesmo tratamento de erro.
- [ ] Modelo inválido com modelo válido já carregado → modelo anterior preservado, áudio não interrompe.
- [ ] (Fender Studio Pro) Confirmar que não trava apesar de GUI limitada — parâmetros genéricos do host funcionam.

---

### 2B — Knobs: Range, Fine-Tune, Reset, Glow

| Knob       | Range                   | Default    |
| ---------- | ----------------------- | ---------- |
| **INPUT**  | −96.0 a +30.0 dB        | 0.0 dB     |
| **OUTPUT** | −96.0 a +30.0 dB        | 0.0 dB     |
| **GATE**   | −90.0 a −40.0 dB        | −70.0 dB   |

- [ ] Arrastar cada knob até extremos → tooltip (hover) mostra valor correto, limites respeitados.
- [ ] Hover sobre knob → tooltip com 2 casas decimais (ex: `"3.50 dB"`). INPUT/OUTPUT: `"X.XX dB"`. GATE: `"X.XX dB (Threshold)"`.
- [ ] **Ctrl+Drag (fine-tune):** mesma distância de arraste = ~10× menos variação. Ctrl+scroll também 10× mais lento.
- [ ] **Double-click** no knob → reseta ao default imediatamente (INPUT/OUTPUT → 0.0, GATE → −70.0).
- [ ] Durante arraste → glow (halo semitransparente) visível no arco. Desaparece ao soltar.
- [ ] Bypass toggle: LED + label alternam instantaneamente. Áudio dry/processed sem estalo.

---

### 2C — Medição VU, Peak Hold & Clipping (Mono)

> **Comportamento mono do plugin:** O plugin CLAP opera estritamente em mono (1 canal). Consequentemente, a Zona 3 exibe sempre um único medidor centralizado sem label, independentemente de ser inserido em trilha mono ou estéreo da DAW (onde o roteamento/processamento estéreo é gerenciado pelo host).

- [ ] Inserir NAM-rs na DAW → Zona 3 exibe **1 medidor centralizado sem label** (16px de largura) em uma zona de ~76px.
- [ ] Alimentar com sinal dinâmico → barra VU única com gradiente tricolor: verde (−60 a −12 dB) → amarelo (−12 a −3 dB) → vermelho (−3 a +6 dB).
- [ ] Transientes rápidos (pick attack) → barra responde sem atraso visual (~33 fps).
- [ ] Provocar pico e parar sinal → marca de peak hold permanece ~2s, depois decai suavemente.
- [ ] Saturar saída (>0 dBFS) → LED vermelho no topo do medidor único **persiste**. Clique no LED ou na barra → reseta.

---

### 2D — Automação & Remote Controls

> **Host primário:** Bitwig Studio.

- [ ] Trilha em modo Write/Latch → arrastar knob INPUT na GUI por ~3s → parar. Grid de automação mostra curva suave, sem saltos, com pontos de ancoragem no início/fim.
- [ ] Repetir para OUTPUT, GATE e BYPASS.
- [ ] Desenhar rampa de automação manual para `output_gain_db` → playback: knob OUTPUT na GUI move suave, áudio acompanha sem zipper noise.
- [ ] Device Panel do Bitwig → 2 páginas: **"Main"** (INPUT, OUTPUT, BYPASS) e **"Gate"** (GATE). Sincronia bidirecional: GUI ↔ Device Panel.
- [ ] (Fender Studio Pro) Mover parâmetros no mixer do host → GUI reflete. Mover na GUI → host reflete.

---

### 2E — Accent Color Dinâmico

> **Host:** Bitwig Studio (requer `track_info`).

- [ ] Cor da trilha alterada para vermelho → knobs INPUT/OUTPUT + LED bypass mudam para vermelho em <100ms. Medidores VU **não** mudam.
- [ ] Mudar para azul, verde → acompanha.
- [ ] Remover cor da trilha → volta ao turquesa padrão (`#00D4AA`).
- [ ] (Fender Studio Pro) Sem `track_info` → mantém turquesa, sem erros.

---

### 2F — Persistência (Save/Reload)

- [ ] Ajustar INPUT=+3.5, OUTPUT=−6.0, GATE=−55.0, modelo carregado.
- [ ] Salvar projeto → fechar DAW completamente → reabrir → carregar projeto.
- [ ] Todos os parâmetros preservados nos valores exatos. Modelo recarregado (nome visível). Áudio idêntico.
- [ ] Repetir no Fender Studio Pro.
- [ ] Mover arquivo do modelo de lugar → reabrir projeto → `"No model loaded"` sem crash.

---

### 2G — Drag & Drop + DSP Load Meter

- [ ] Arrastar `.nam` do file manager sobre o plugin → overlay `"Drop NAM Model Here ⬇️"` aparece. Soltar → carrega modelo.
- [ ] Arrastar `.wav` → overlay aparece mas ignora ao soltar.
- [ ] Status bar: indicador `"DSP: XX.X%"` presente na telemetria (verde <50%, âmbar 50-80%, vermelho >80%).
- [ ] Hover no DSP Load → tooltip descreve a porcentagem de uso do tempo real.

---

### 2H — Parameter Indication

> **Host:** Bitwig Studio.

- [ ] MIDI Learn no knob INPUT → halo pontilhado com 6 dots azuis (`#5e81ac`) ao redor do knob.
- [ ] Playback de automação em `output_gain_db` → arco do OUTPUT pulsa suavemente (alpha 0.3→1.0, ciclo ~1s).
- [ ] Override manual (mover knob na GUI) durante automação ativa → arco fica âmbar (`#F5A623`) temporariamente. Volta ao normal ao soltar.

---

### 2I — Acessibilidade (Teclado)

- [ ] Tab → foco cicla: INPUT → OUTPUT → GATE → BYPASS → Load Model → INPUT. Focus ring visível.
- [ ] Shift+Tab → ordem inversa.
- [ ] Knob focado: ↑/→ = +1.0 dB, ↓/← = −1.0 dB. Ctrl+↑ = +0.1 dB, Ctrl+↓ = −0.1 dB. Limites respeitados.
- [ ] Load Model focado → Space/Enter abre picker.
- [ ] BYPASS focado → Space/Enter alterna bypass.
- [ ] Contraste de texto OK: `COL_MUTED` legível sobre `COL_PANEL`, `COL_TEXT` legível sobre `COL_BG`.

---

### 2J — Modulação LFO em Tempo Real

> **Host:** Bitwig Studio.

- [ ] LFO em `input_gain_db` a 1–5 Hz, ±6 dB → arco oscila suave, áudio sem zipper noise.
- [ ] LFO a 20 Hz → áudio modula como tremolo rápido sem artefatos ou picos de CPU.

---

### 2K — Compensação Dinâmica de Latência

- [ ] Mudar sample rate do projeto (44.1→96 kHz) → status bar atualiza (`"96kHz"`), latência atualiza. Bitwig recalcula PDC sem dessincronia.
- [ ] Toggle bypass ou trocar modelo com resampling diferente → latência reportada muda, Bitwig atualiza PDC imediatamente.

---

## Bloco 3 — Estresse & Pedantaria

Executar **após** Blocos 1 e 2 passarem. Buffer de 128 samples @ 48 kHz.

---

### 3.1 Spam de Interface

- [ ] Alternar bypass 20+ vezes consecutivas via GUI. Sem crash, sem artefato.
- [ ] Abrir/fechar GUI 20+ vezes em <30s com playback ativo.
- [ ] (Bitwig) Alternar modos de hosting (*Together*, *Individually*, *Individually strict*) e repetir spam.

---

### 3.2 Carga Rápida Concorrente

- [ ] Carregar 10 modelos diferentes em <1 minuto, com áudio rodando.
- [ ] Memória RSS estável (crescimento <2 MB após 10 reloads). Sem leak visível.

---

### 3.3 Modulação Extrema

- [ ] (Bitwig) LFO a 20–100 Hz modulando `input_gain_db` por ≥5 min com buffer 128 samples. Zero zipper noise, zero XRUNs.
- [ ] (Fender) Envelopes/LFOs de canal modulando parâmetros por ≥5 min.

---

### 3.4 Gate FSM em Silêncio

- [ ] Parar playback por 10s → saída = silêncio limpo (sem ruído residual, sem denormals).
- [ ] Retomar playback → áudio volta sem estalo, sem perda de transiente.

---

### 3.5 Multi-Instância

- [ ] Com 2 instâncias processando, adicionar 3ª durante playback → as 2 primeiras não sofrem interrupção.
- [ ] Deletar 3ª instância durante playback → as 2 restantes continuam.
- [ ] Abrir File Picker em 2 instâncias simultaneamente → ambos funcionam independentemente.
- [ ] Fechar GUI de uma instância, manter outra aberta → áudio de ambas segue normal. Reabrir GUI → estado preservado.
- [ ] 3 instâncias: bypass na 1ª, ativa na 2ª, carregar modelo na 3ª → cada uma independente.

---

### 3.6 Endurance 1 Hora

- [ ] Projeto com 4 instâncias (2× WaveNet + 2× LSTM), 2 LFOs cada, playback contínuo por **60 min**.
- [ ] Monitorar a cada 30s: RSS, file descriptors, threads, XRUNs.
- [ ] **Aceite:** zero crashes, RSS estabiliza (variação <5 MB após warmup), zero vazamento de FDs/threads, zero XRUNs.

---

### 3.7 Null Test de Bypass

- [ ] Trilha extra: NAM-rs em bypass + sinal idêntico em paralelo com fase invertida + ADC ativo.
- [ ] Resultado = silêncio absoluto (<−120 dBFS). Bypass é bit-transparente.

---

### 3.8 Determinismo de Bounce Offline

- [ ] Com processamento ativo, bounce offline 2 vezes consecutivas.
- [ ] Arquivos WAV idênticos bit-a-bit (`cmp`). **Usar bounce offline, não tempo-real.**

---

## Critérios de Release

- [ ] Zero crashes, panics ou congelamentos em qualquer operação.
- [ ] Zero XRUNs registrados no `pw-top` durante toda a sessão.
- [ ] Zero zipper noise audível em knobs, automação ou modulação.
- [ ] Renderização visual estável a ~33 fps, sem flicker ou artefatos.
- [ ] Fluxo completo: instanciar → carregar → ajustar → salvar → fechar → reabrir → estado preservado.

---

## Template de Bug Report

```text
**Teste:** <ID, ex: 2C.4>
**OS/Kernel:** <ex: Ubuntu 24.04, Linux 6.8-lowlatency>
**DAW:** <nome e versão, ex: Bitwig Studio 6.0.6 Flatpak>
**Modelo:** <arquivo .nam, ex: jcm800.nam>
**Buffer/Sample Rate:** <ex: 128 samples @ 48 kHz>
**Esperado:** <comportamento descrito no roteiro>
**Observado:** <o que realmente aconteceu>
**Anexos:** screenshot/vídeo da GUI, log da DAW, XRUNs do pw-top, telemetria RT da status bar.
```
