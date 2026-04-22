# Ideias de pesquisa para as próximas versões 1.x e/ou 2.x

---

## Idéia 1. Filtros de Fase Mínima (Minimum Phase) no Resampler (96 kHz)

Esforço: 🔴 Alto · Impacto: 🟡 Médio (~1 ms latência) · Risco: 🔴 Alto

### 1.1 Diagnóstico

O `NamResampler` em `resampler.rs` usa `rubato::SincFixedIn` com filtro FIR de **Fase Linear** (Kaiser-BlackmanHarris2, `sinc_len=256`, interpolação Hermite cúbica).
Filtros de fase linear possuem group delay simétrico de `sinc_len/2 = 128` taps. Para o roundtrip 96k → 48k → 96k, isso adiciona **~1.5 ms** de latência pura na matemática do resampler, além de *pre-ringing* audível.
Amplificadores valvulados reais são sistemas de fase mínima por natureza — o pre-ringing dos FIR lineares é um artefato estranho à experiência de tocar guitarra.

### 1.2 Solução Proposta

Migrar para coeficientes FIR de **Fase Mínima** (Minimum Phase Sinc). Duas abordagens:

#### **Abordagem A — Tabela estática de coeficientes:**

1. Extrair os coeficientes Kaiser-Blackman do `rubato` (ou gerá-los independentemente).
2. Aplicar a **Transformada de Hilbert** para converter para fase mínima.
3. Injetar como tabela `const` no código, usada por um resampler FIR customizado.

Considerar trabalhar junto ao upstream do Rubato 2.0?

#### **Abordagem B — Resampler FIR customizado:**

1. Implementar resampler polifásico FIR sem dependência de `rubato`.
2. Usar coeficientes MPS (Minimum Phase Sinc) pré-computados.
3. Controle total da API e dos buffers internos.

### 1.3 Riscos Críticos

- **Resposta audível muda:** A fase mínima redistribui a energia temporal do filtro. Isso altera a assinatura sonora do resampler e requer validação perceptual com músicos.
- **Invalidação de golden vectors:** Todos os testes de roundtrip do resampler e golden vectors que passam pelo resampling path precisariam ser re-gerados.
- **API do `rubato`:** Não expõe coeficientes de fase mínima. Qualquer solução requer implementação custom ou fork.
- **SNR:** Filtros de fase mínima podem ter rejeição de aliasing ligeiramente inferior em stopband. Requer validação com impulso de Dirac e FFT.
