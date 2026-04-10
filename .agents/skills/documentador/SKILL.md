---
name: documentador
description: Especialista em documentação técnica e arquitetural. Garante que o conhecimento do projeto (arquitetura e requisitos) esteja sempre sincronizado com a implementação.
---

# Skill: Documentador

## When to use this skill

Use esta skill ao final de cada ciclo de desenvolvimento ou quando houver necessidade de manter a "fonte da verdade" do projeto atualizada. Deve ser ativada ao receber solicitações expressas para documentar o sistema, mormente em mudanças de DSP, SIMD Vetorial (WaveNet, LSTM) ou algoritmos FastMath, mitigando perda de sabedoria endêmica da arquitetura.

## Instructions

### 1. Análise de Impacto

Antes de consolidar a documentação final ou propor mudanças, identifique quais documentos em `docs/` e diretrizes de IAs (`.agents/`) são afetados. Assegure total rastreabilidade da documentação nativa do repositório.

### 2. Hierarquia de Documentos (ordem de prioridade)

1. **`docs/architecture.md`** — é a **bíblia atual do projeto** e a única fonte de verdade sobre arquitetura. Toda mudança arquitetural deve ser refletida aqui **primeiro**.
2. **`README.md`** — visão geral para novos colaboradores (instalação, uso básico, dependências de sistema).

4. **`.agents/`** — definições de IA. Atualizar quando a arquitetura evoluir de forma que as skills descrevam padrões obsoletos.

### 3. Princípio "Use the Force, Read the Source"

A documentação do NAM-rs é guia, não substituto do código. Siga estes princípios:

#### Documentação → Princípios, Justificativas e Orientações

A documentação contém:
- **Princípios inegociáveis** — regras que nunca devem ser quebradas (ex: "ZERO alocação no `process()`", "`#[repr(align(128))]` em `ParamPayload`").
- **Justificativas técnicas** — o *porquê* das decisões (ex: por que `rubato 0.16` e não `2.0`? Por que `SCHED_FIFO`?). Isso previne regressões por quem não conhece o histórico.
- **Orientações para desenvolvedores** — como usar os patterns estabelecidos (diagnósticos, SPSC, gain staging, etc.).

#### Código-fonte → Implementação e Detalhes

A documentação **deve sempre** apontar para onde o leitor pode aprofundar no código-fonte:

- Ao descrever um módulo, **linke para o arquivo** (ex: "Veja implementação completa em `src/diagnostics.rs`").
- Ao explicar um pattern, **indique a função** (ex: "O builder fluente está em `NamDiagnostic::new()` — `src/diagnostics.rs`").
- Ao documentar uma decisão, **cite o commit ou a linha** relevante quando possível.
- **Nunca duplique código verbatim** na documentação — ele envelhece. Prefira referências ao arquivo e uma explicação conceitual.

#### Na prática

```markdown
## Exemplo de boa documentação

O NAM-rs usa filtro Sinc Kaiser com `sinc_len=256` para resampling bidirecional.
A escolha de BlackmanHarris2 garante atenuação >−100 dB na stop-band.

> **Código-fonte:** `src/dsp/resampler.rs` — função `sinc_params()` (parâmetros do filtro)
> e `NamResampler::new()` (construção dos resamplers input+output).
```

### 4. Boas Práticas

- Nunca apague arquivos contendo a matriz mental do sistema de tempo real sem realocação sistemática.
- Edição do arquivo `docs/NAM-rs-referência.md` só é permitida se for cirúrgicas, ou na forma de notas e observações.
- Aproveite a documentação para justificar decisões de design (por que SCHED_FIFO? por que `#[repr(align(128))]`? por que `rubato 0.16` e não 2.0?). Isso evita regressões arquiteturais futuras.
- Ao documentar o sistema de diagnósticos (`src/diagnostics.rs`), mantenha o catálogo de códigos `Exxxx` sincronizado entre `docs/architecture.md` e o enum `NamErrorCode` no código-fonte — mas o **enum é a fonte da verdade**.
