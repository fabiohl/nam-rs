<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Achado Focado: `wavenet_a2_max.nam` Ativamente Quebrado

Agente de IA: Ainda não tente resolve-lo. É lícito tomar conhecimento dele e contribuir com novos aprendizados.
Mas ele só será atacado em momento oportuno por decisão do Product Owner do NAM-rs.

---

## 0. Proveniência e status upstream do fixture `wavenet_a2_max.nam`

Antes de qualquer diagnóstico de código, é essencial entender **o que este arquivo realmente é**
dentro do próprio projeto upstream NAMCore (C++), pois isso baliza tanto a interpretação da
divergência quanto a prioridade real da correção.

### 0.1 É um exemplo oficial do NAMCore, não um fixture sintético deste repositório

```shell
$ cmp tests/fixtures/NeuralAmpModelerCore/example_models/wavenet_a2_max.nam tests/fixtures/models/wavenet_a2_max.nam
# (sem saída — arquivos byte-a-byte idênticos, mesmo sha256)
```

`tests/fixtures/models/wavenet_a2_max.nam` é uma cópia **byte-idêntica** de
`tests/fixtures/NeuralAmpModelerCore/example_models/wavenet_a2_max.nam` — um exemplo shipado pelo
próprio projeto upstream `NeuralAmpModelerCore` (vendorizado neste repo em
`tests/fixtures/NeuralAmpModelerCore/`, versão `v0.5.3` conforme `NAM/version.h`). **Não foi criado
por este repositório** (ao contrário de `wavenet_a2_film_lite/full`, `a2_dynamic_gated_ch8`, etc.,
que são gerados por `tests/fixtures/generate_a2_fixtures.py`, próprio deste projeto).

### 0.2 O próprio arquivo se autodescreve como um caso de teste experimental

```json
"notes": [
    "This model is meant as a 'test case' to contain all of the new features that are being considered for A2.",
    "It doesn't have slimmability."
]
```

Esta é uma autodescrição explícita de "kitchen sink" — um modelo deliberadamente construído para
**combinar simultaneamente** todas as features exóticas do A2 (FiLM em 8/8 pontos, `head1x1`
agrupado, `layer1x1` agrupado, `condition_dsp` aninhado, `condition_size>1`), não um modelo
treinado real nem uma referência de correção validada.

### 0.3 O modelo quebra deliberadamente TODAS as invariantes do fast-path A2 do próprio C++

Comparando com `example_models/A2.nam` (a forma **canônica/de referência** do A2 no NAMCore — um
`SlimmableContainer` com dois submodelos, "A2 nano" `CH=3` e "A2 standard" `CH=8`, ambos sem FiLM,
sem `head1x1`, sem `condition_dsp`, todos os `groups=1`, 23 camadas, `head={out_channels:1,
kernel_size:16, bias:true}`), `wavenet_a2_max.nam` viola **cada uma** das invariantes documentadas
em `a2_fast.cpp:40-52`:

```cpp
// tests/fixtures/NeuralAmpModelerCore/NAM/wavenet/a2_fast.cpp:40-52
// Architectural invariants (checked once by is_a2_shape before we get here):
//   - single layer array with 23 layers
//   - Bottleneck == Channels
//   - condition_size == input_size == out_channels == 1
//   - LeakyReLU(0.01) on every layer, no gating, no FiLM, no head1x1
//   - layer1x1 active (groups=1), head rechannel conv k=16 bias=true
//   - no post-stack head
```

`wavenet_a2_max.nam` tem 2 camadas (não 23), `condition_size=8`, `head1x1` ativo com `groups=2`,
FiLM ativo em todos os 8 pontos, `layer1x1.groups=2`, e cabeçalho legado (não o
`{kernel_size:16}` canônico). **O próprio C++ upstream classificaria este modelo como fora do
fast-path A2**, roteando-o para o motor genérico de WaveNet — exatamente o comportamento que o
dispatcher Rust também adota (`WaveNetA2Dyn`, via a classificação `A2TopologyResult::Dynamic`).

### 0.4 O NAMCore upstream nunca valida a saída numérica deste modelo — apenas "não é NaN/Inf"

Buscando em todo o repositório vendorizado (`tools/test/*.cpp`) por referências a
`wavenet_a2_max.nam`:

* **`tools/test/test_get_dsp.cpp:185-207`** — `test_load_and_process_nam_files()` carrega e
  processa `wavenet_a2_max.nam` (junto com `wavenet.nam`, `lstm.nam`,
  `wavenet_condition_dsp.nam`) e verifica **apenas** `isfinite(output[i])` para cada amostra —
  nenhuma comparação com valores esperados.
* **`tools/test/test_container.cpp`** (linhas 144, 152, 160, 202, 384) — usa
  `wavenet_a2_max.nam` como o submodelo "grande" em testes de `SlimmableContainer`, novamente
  verificando apenas `isfinite(output)` e que submodelos diferentes produzem saídas diferentes.
* **`tools/test/test_a2_fast.cpp`** — testa o fast-path A2 com configs sintéticas construídas
  programaticamente (`build_a2_config(channels)`), **nunca carrega `wavenet_a2_max.nam` nem
  nenhum `.nam` do disco**.
* **CI (`.github/workflows/build.yml:46-56`)** — só renderiza/benchmarka `wavenet.nam` e
  `lstm.nam`; `wavenet_a2_max.nam` **nunca é renderizado em CI**.
* Existe um script Python **dedicado e vendorizado no próprio NAMCore**
  (`tests/fixtures/NeuralAmpModelerCore/generate_weights_a2.py` — não confundir com
  `tests/fixtures/generate_a2_fixtures.py`, deste repositório) cuja única finalidade é gerar
  pesos aleatórios para `wavenet_a2_max.nam` — confirmando, mais uma vez, que os pesos deste
  arquivo são **aleatórios/sintéticos**, não provenientes de um modelo treinado real.

### 0.5 Conclusão sobre proveniência — o que isso muda (e o que não muda) no diagnóstico

* **Não muda:** perseguir paridade Rust vs. C++ para este modelo continua sendo um objetivo bem
  posto — o `golden_wavenet_a2_max.bin` foi renderizado executando o binário `render` real do
  NAMCore vendorizado contra este `.nam` exato, então ele captura fielmente "o que esta build
  específica do C++ calcula para esta entrada exata". Não é diferente, em espírito, dos demais
  goldens A2 dinâmicos deste projeto (`a2_dynamic_gated_ch8`, `wavenet_a2_film_*`), que também são
  fixtures sintéticos com pesos aleatórios, não modelos treinados reais.
* **Muda a priorização e a moldura do problema:** as features individuais exercitadas por este
  modelo (`FiLM`, `condition_dsp`, convoluções agrupadas, `head1x1`) **são features de produção
  documentadas e suportadas desde a v0.4.0 do NAMCore** (`docs/wavenet_walkthrough.rst:30-34`), e
  já usadas por outros fixtures deste repositório de forma **isolada** (uma feature exótica por
  vez). O que é exclusivo de `wavenet_a2_max.nam` é a **combinação simultânea máxima** de todas
  elas — e, como a auditoria abaixo mostra, é justamente essa combinação que expõe bugs em
  primitivas que, individualmente, nunca haviam sido exercitadas com `groups>1` ou em cadência
  "por camada" por nenhum outro fixture. **Corrigir essas primitivas tem valor que transcende este
  modelo específico**: qualquer `.nam` real futuro (treinado) que venha a usar `head1x1` com mais
  de uma camada dilatada, ou `layer1x1.groups>1`, ou `groups_input_mixin>1`, atingiria os mesmos
  bugs de produção hoje ocultos por falta de cobertura de teste.

---

## Achado Único: `wavenet_a2_max.nam` — Divergência Catastrófica de Produção (ESR ≈ 36–107)

**Status:** 🔴 Aberto — bloqueado em produção. Diagnóstico anterior (`condition_dsp`/oráculo f64)
**refutado como explicação suficiente**. **Três bugs de produção distintos e independentes foram
identificados e confirmados com certeza absoluta** — via leitura pareada de código C++/Rust **e**
reconciliação exata (sem nenhum resíduo) do orçamento de pesos contra o script autoritativo de
geração de pesos do próprio NAMCore upstream (`generate_weights_a2.py`):

* **Bug A (estrutural, o mais severo):** `head1x1` é um submódulo **por camada/dilatação** no C++
  (cada uma das `N` camadas dilatadas tem seu próprio `head1x1` com pesos independentes), mas o
  motor dinâmico A2 do Rust (`WaveNetA2Dyn`) o modela como um único componente **por array**,
  compartilhado por todas as camadas.
* **Bug B:** `layer1x1.groups` e `groups_input_mixin` são parâmetros de convolução agrupada
  declarados no JSON, mas silenciosamente ignorados pelo motor dinâmico A2 (assume `groups=1`
  sempre).
* **Bug C:** o kernel do cabeçalho final (`head_rechannel`) está hardcoded em `16`
  (`A2_HEAD_KERNEL_SIZE`) no Rust, mas o formato de cabeçalho legado deste modelo
  (`head_size`/`head_bias` planos, sem objeto `"head"` aninhado) exige `kernel_size=1` — uma
  simples projeção `Conv1x1`, não uma convolução temporal de 16 taps.

### 1. Sintoma e histórico do bloqueio

O modelo `wavenet_a2_max.nam` (`tests/fixtures/models/wavenet_a2_max.nam`) é o único fixture A2
"flagship" do repositório inteiro que combina, simultaneamente:

* `condition_size = 8` (o único fixture com `condition_size > 1` em toda a suíte de testes);
* `head1x1.groups = 2` e `layer1x1.groups = 2` (os únicos com `groups > 1` nesses dois campos);
* `groups_input_mixin = 4` (o único com esse campo `> 1`);
* Todos os 8 slots de FiLM ativos simultaneamente;
* Um sub-modelo `condition_dsp` aninhado (WaveNet de 2 arrays, ele mesmo despachado
  recursivamente pelo mesmo dispatcher — e que **também** ativa `head1x1` em ambos os arrays,
  logo **também** sofre do Bug A e do Bug C internamente, ver §3.7);
* Formato de cabeçalho **legado** (`head_size`/`head_bias` planos, **sem** o objeto aninhado
  `"head": {"kernel_size": ...}` usado por todos os outros fixtures A2 e pelo `A2.nam` canônico).

Ao carregar e processar este modelo através do motor de produção real (`WaveNetA2Dyn`), o áudio de
saída diverge catastroficamente do golden C++ (`golden_wavenet_a2_max.bin`):

| Medição | Valor                                                                |
| ------- | -------------------------------------------------------------------- |
| MSE     | 2.46e3 (medição original) → 7.30e3 (após correções de FiLM B1/B2/B3) |
| SNR     | −15.6 dB → −20.3 dB                                                  |
| ESR     | 3.61e1 (36.1) → 1.07e2 (107)                                         |

Por cautela, o modelo foi bloqueado em produção via um guard hardcoded:

```rust
// src/loader/dispatcher/wavenet/mod.rs:85-91
#[cold]
#[inline(never)]
fn is_disabled_broken_a2_flagship(
    num_arrays: usize,
    condition_size: usize,
    has_condition_dsp: bool,
) -> bool {
    num_arrays == 1 && has_condition_dsp && condition_size == 8
}
```

```rust
// src/loader/dispatcher/wavenet/mod.rs:181-187 (chamada, antes de qualquer leitura de pesos)
if is_disabled_broken_a2_flagship(num_arrays, condition_size, has_condition_dsp) {
    bail!(
        "WaveNet A2 flagship (single-array, condition_dsp, condition_size=8) is disabled: \
         confirmed wrong audio output vs NAMcore golden — see docs/cpp_parity_map.md §7.1. \
         Model is not removed; re-enable requires closing the condition_dsp parity gap (§4.4)."
    );
}
```

Os testes associados permanecem todos `#[ignore]`d, aguardando a correção:

| Teste                                     | Arquivo:linha                              | Motivo do `#[ignore]`                                                             |
| ----------------------------------------- | ------------------------------------------ | --------------------------------------------------------------------------------- |
| `test_golden_vectors_wavenet_a2_max`      | `tests/models/golden_vectors.rs:2037`      | "model disabled — confirmed broken; inference path blocked at dispatch"           |
| `test_oracle_vs_python_anchor_a2_generic` | `tests/parity/reference_oracle_f64.rs:866` | idem                                                                              |
| `test_oracle_a2_generic`                  | `tests/parity/reference_oracle_f64.rs:889` | "condition_dsp output divergence between production and oracle — ESR=1e5 (50 dB)" |
| `test_decomposition_a2_generic`           | `tests/parity/reference_oracle_f64.rs:901` | idem                                                                              |
| `test_combined_simulation_a2_generic`     | `tests/parity/reference_oracle_f64.rs:928` | idem                                                                              |
| `live_cross_validation_wavenet_a2_max`    | `tests/parity/cpp_parity.rs:1124`          | corpo inteiro comentado — nunca esteve ativo                                      |

### 2. Por que o diagnóstico anterior ("bug no oráculo f64 `condition_dsp`") é **insuficiente** como explicação da divergência de produção

O `TODO-parity.md` legado atribuía a causa raiz a três bugs no **oráculo f64 de teste**
(`src/testing/reference_oracle/a2.rs`): (1) leitura de pesos do `head1x1` usando `ch` em vez de
`head_accum_size`; (2) `ch_per_group` calculado com `ch` em vez de `head_accum_size`; (3)
finalização do cabeçalho para `head_size > 1` com dimensionamento incorreto do buffer de saída.

**Esses três bugs são reais e permanecem no oráculo** (confirmados nesta auditoria, código
inalterado desde o diagnóstico original):

```rust
// src/testing/reference_oracle/a2.rs:306-322 — campo head_b é escalar, não Vec<f64>
struct ArrayState {
    ch: usize,
    head_accum_size: usize,
    // ...
    head1x1_w: Vec<f64>,
    head1x1_b: Vec<f64>,
    head_w: Vec<f64>,
    head_b: f64,           // <- deveria ser Vec<f64> para head_size > 1
    // ...
}

// src/testing/reference_oracle/a2.rs:490-499 — usa `ch`, não `head_accum_size`
let head1x1_w: Vec<f64> = if head1x1_active {
    cursor.read_f64(ch * h1_in_size)   // BUG: deveria ser head_accum_size * h1_in_size
} else { vec![] };
let head1x1_b: Vec<f64> = if head1x1_active {
    cursor.read_f64(ch)                // BUG: deveria ser head_accum_size
} else { vec![] };

// src/testing/reference_oracle/a2.rs:754 — idem
let ch_per_group = ch / h1_groups;      // BUG: deveria ser head_accum_size / h1_groups
```

**Porém, esses bugs residem exclusivamente em `src/testing/reference_oracle/a2.rs` — um módulo de
teste que nunca é executado pelo motor de produção.** O guard `is_disabled_broken_a2_flagship`
bloqueia o **dispatcher de produção** (`WaveNetA2Dyn`, via `build_wavenet`), e a métrica ESR≈36–107
citada acima foi medida **entre a saída de produção e o golden C++** (`test_golden_vectors_...`),
**não** entre o oráculo f64 e um anchor Python. A cadeia lógica do plano de correção anterior —
"corrigir o oráculo → remover o guard → o modelo funciona" — contém uma **lacuna lógica**: corrigir
`src/testing/reference_oracle/a2.rs` restaura a autoconsistência dos testes de oráculo
(`test_oracle_a2_generic` e companhia), mas **não tem nenhum efeito sobre o código de produção**
que o guard bloqueia, porque produção nunca invoca o oráculo. Prosseguir apenas com essa correção
teria consumido uma sprint inteira sem mover a métrica que realmente importa (ESR de produção vs.
golden C++). Adicionalmente, o oráculo nem sequer trata `head1x1` como um submódulo por camada
(Bug A abaixo) — teria a mesma limitação estrutural do código de produção, mesmo depois de
corrigidos os 3 bugs pontuais já documentados.

*(A correção do oráculo continua válida e recomendada — ver Epic 6 — mas deve ser tratada como
**ortogonal e não bloqueante** para a resolução deste achado, e nunca deve ser confundida com uma
correção do problema de produção.)*

### 3. Três bugs de produção confirmados e reconciliados exatamente contra o script autoritativo de pesos do NAMCore

#### 3.1 A ferramenta decisiva: `generate_weights_a2.py` (script upstream, vendorizado, autoritativo)

O NAMCore vendorizado inclui, na raiz de `tests/fixtures/NeuralAmpModelerCore/`, o script
**`generate_weights_a2.py`** — a própria ferramenta usada pelo projeto upstream para gerar os
pesos aleatórios de `wavenet_a2_max.nam` (docstring: *"Generate weights for wavenet_a2_max.nam
file. This script handles the full A2 architecture including: FiLM ... head1x1 ... condition_dsp
... Advanced gating modes ..."*). Este script contém, em Python, uma reimplementação **exata e
autoritativa** da fórmula de contagem de pesos que o C++ espera para cada componente — e foi
literalmente usado para produzir os 818 + 1052 valores de `f32` presentes no arquivo `.nam` real.
Executá-lo contra a config real do modelo é portanto o **método de verificação mais autoritativo
disponível**, superior a qualquer reconstrução manual de fórmulas:

```shell
$ python3 -c "
import json, sys
sys.path.insert(0, 'tests/fixtures/NeuralAmpModelerCore')
from generate_weights_a2 import count_wavenet_weights
d = json.load(open('tests/fixtures/models/wavenet_a2_max.nam'))
print(count_wavenet_weights(d['config']))                       # main model
print(count_wavenet_weights(d['config']['condition_dsp']['config']))  # condition_dsp
"
818
1052
```

**Ambos os totais reconciliam exatamente e sem nenhum resíduo** com os tamanhos reais dos arrays
`weights` no arquivo (`len(d['weights']) == 818`, `len(d['condition_dsp']['weights']) == 1052`).
Isso prova, de forma definitiva, que a fórmula do script **é** a verdade arquitetural pretendida
para este arquivo — permitindo diagnosticar com certeza absoluta (não por hipótese) onde o Rust
diverge.

#### 3.2 Bug A (🔴 o mais severo, estrutural): `head1x1` é por-camada no C++, mas por-array no Rust

**Evidência C++ — `_head1x1` é membro da classe `Layer` (uma camada dilatada), não da
`LayerArray`:**

```cpp
// tests/fixtures/NeuralAmpModelerCore/NAM/wavenet/detail.h:37-77 (classe Layer, uma por dilatação)
class Layer
{
  Layer(const LayerParams& params)
  : _conv(...), _input_mixin(...), ...
  {
    if (params.layer1x1_params.active) { _layer1x1 = std::make_unique<Conv1x1>(...); }
    if (params.head1x1_params.active)
    {
      _head1x1 = std::make_unique<Conv1x1>(
          params.bottleneck, params.head1x1_params.out_channels, true, params.head1x1_params.groups);
    }
    // ...
    if (params.head1x1_post_film_params.active && params.head1x1_params.active)
    {
      _head1x1_post_film = std::make_unique<FiLM>(...);   // FiLM do head1x1 TAMBÉM é por camada
    }
  }
  // ...
};
```

Cada uma das `N` camadas dilatadas de um array (`dilations.len()`) instancia seu **próprio**
`_head1x1` (e seu próprio `_head1x1_post_film`), com pesos independentes.

**Evidência de ordem de leitura de pesos — `Layer::set_weights_` lê `head1x1` dentro do bloco de
CADA camada, antes das FiLM, e é chamado uma vez por camada pelo laço da `LayerArray`:**

```cpp
// tests/fixtures/NeuralAmpModelerCore/NAM/wavenet/model.cpp:135-164
void nam::wavenet::detail::Layer::set_weights_(std::vector<float>::iterator& weights)
{
  this->_conv.set_weights_(weights);
  this->_input_mixin.set_weights_(weights);
  if (this->_layer1x1) { this->_layer1x1->set_weights_(weights); }
  if (this->_head1x1) { this->_head1x1->set_weights_(weights); }   // <- POR CAMADA, antes do FiLM
  if (this->_conv_pre_film) this->_conv_pre_film->set_weights_(weights);
  // ... (demais 6 slots de FiLM, incluindo _head1x1_post_film por último)
}
```

**Evidência de execução em runtime — `Layer::Process` aplica seu próprio `_head1x1` (e seu próprio
`_head1x1_post_film`) a cada chamada, uma por camada:**

```cpp
// tests/fixtures/NeuralAmpModelerCore/NAM/wavenet/model.cpp:273-297
if (this->_head1x1)
{
    this->_head1x1->process_(this->_z.leftCols(num_frames), num_frames);   // pesos DESTA camada
    if (this->_head1x1_post_film)
    {
        this->_head1x1_post_film->Process_(this->_head1x1->GetOutput(), condition, num_frames);
    }
    this->_output_head... = this->_head1x1->GetOutput()...;   // contribuição desta camada ao head
}
```

**Evidência Rust — `head1x1_w`/`head1x1_b` vivem em `WaveNetA2Dyn` (nível de array), não em
`A2Layer` (nível de camada), e são carregados uma única vez, DEPOIS do laço de camadas:**

```rust
// src/models/a2/model/dynamic/build.rs:44-65 (load_weights_inner)
self.load_rechannel_weights(weights, pos, total)?;
let mut layers = Vec::with_capacity(self.num_layers);
for i in 0..self.num_layers {
    let layer = self.load_per_layer_weights(weights, pos, total, i)?;   // head1x1 NÃO está aqui
    layers.push(layer);
}
self.load_head1x1_weights(weights, pos, total)?;    // <- ÚNICA leitura, para TODAS as camadas
self.load_head_conv_and_scale(weights, pos, total)?;
```

`src/models/a2/layer.rs` (struct `A2Layer`, o equivalente Rust de `Layer`) **não tem nenhum campo
`head1x1_w`/`head1x1_b`** — apenas `conv`, `mixin_w`, `l1x1_w`, `l1x1_b` e os 8 slots de FiLM. O
`head1x1_post_film` (linha 66 de `layer.rs`) **está** presente por camada, mas **usa os pesos
compartilhados do array** (`self.head1x1_w`/`b` em `WaveNetA2Dyn`) durante o acumulador de head em
`process.rs:451-473`, em vez de pesos próprios de cada camada.

**Consequência quantificada para `wavenet_a2_max.nam` (2 camadas por dilatação):** o C++ precisa
consumir `head1x1` (peso+bias) **duas vezes** — uma por camada, com valores distintos — mais o
respectivo `head1x1_post_film` também duas vezes. O Rust consome **uma única vez**. Isso desalinha
o stream de pesos a partir do fim da primeira camada, corrompendo a leitura de **toda a segunda
camada em diante** — inclusive o cabeçalho final. Este é, isoladamente, suficiente para explicar a
divergência catastrófica observada, e é **o primeiro bug a corrigir**, pois qualquer correção dos
Bugs B/C antes deste continuaria operando sobre um layout de pesos fundamentalmente errado.

**Nota de escopo geral:** este bug afeta **qualquer** modelo A2 dinâmico com `head1x1.active=true`
e mais de uma camada dilatada por array — não é exclusivo de `wavenet_a2_max.nam`. **Confirmado
por verificação direta do JSON (§3.6): nenhum outro fixture atual tem `head1x1.active=true`** —
`a2_dynamic_gated_ch8`/`a2_dynamic_blended_ch3` (23 camadas canônicas) têm `head1x1.active=false`,
portanto o bug está dormente neles, sem nenhuma regressão silenciosa em produção hoje.

#### 3.3 Bug B: `layer1x1.groups` e `groups_input_mixin` ignorados no motor dinâmico A2

**Evidência C++ — `layer1x1` e `input_mixin` são convoluções agrupadas:**

```cpp
// tests/fixtures/NeuralAmpModelerCore/NAM/wavenet/detail.h:44-56
Layer(const LayerParams& params)
: _conv(params.channels, ..., params.groups_input)
, _input_mixin(params.condition_size, ..., false, params.groups_input_mixin)   // GROUPED
{
  if (params.layer1x1_params.active)
  {
    _layer1x1 = std::make_unique<Conv1x1>(
        params.bottleneck, params.channels, true, params.layer1x1_params.groups);  // GROUPED
  }
}
```

O construtor de `Conv1x1` valida a divisibilidade por grupos e, na leitura de pesos, consome
**apenas** `out_channels × (in_channels / groups)` valores do stream — não `out_channels ×
in_channels` — escrevendo-os apenas nos blocos diagonais de uma matriz densa previamente
zero-inicializada:

```cpp
// tests/fixtures/NeuralAmpModelerCore/NAM/dsp.cpp — Conv1x1::set_weights_ (linhas ~363-390)
const long out_per_group = out_channels / numGroups;
const long in_per_group = in_channels / numGroups;
for (int g = 0; g < numGroups; g++)
  for (auto i = 0; i < out_per_group; i++)
    for (auto j = 0; j < in_per_group; j++)
      this->_weight(g * out_per_group + i, g * in_per_group + j) = *(weights++);
```

O forward pass subsequente (GEMM denso `_weight * input`) funciona corretamente porque as entradas
fora do bloco diagonal permanecem exatamente `0.0` (herdadas do `setZero()` em `set_size_`) — a
multiplicação por zero simplesmente não contribui à soma. **O agrupamento em C++ é, portanto,
implementado inteiramente na fase de carregamento de pesos (contagem reduzida + scatter em blocos
diagonais); o forward pass permanece denso e inalterado.**

**Evidência Rust — `layer1x1` e `input_mixin` sempre assumem `groups = 1`:**

```rust
// src/models/a2/model/dynamic/build.rs:138 (mixin) e :150 (l1x1)
let mixin_count = conv_out * self.condition_size;   // ignora groups_input_mixin
let l1x1_w_count = bottleneck * channels;           // ignora layer1x1.groups
```

```rust
// src/models/a2/layer.rs:168-180 (A2Layer::new_dyn) — asserts documentam a suposição densa
debug_assert_eq!(mixin_w.len(), ch * condition_size);        // denso, sem /groups
debug_assert_eq!(l1x1_w.len(), bottleneck * l1x1_out_ch);    // denso, sem /groups
```

```rust
// src/models/a2/model/dynamic/process.rs:393-400 (mixin) e :502-513 (l1x1) — laços densos
for c in 0..z_out_ch {
    let base = c * cond_size;
    let mut sum = 0.0;
    for k in 0..cond_size { sum += layer.mixin_w[base + k] * cond_for_mixin[k]; }  // 0..cond_size INTEIRO
    mixin_scratch[c] = sum;
}
for oc in 0..channels {
    let mut sum = l1x1_b[oc];
    for ic in 0..bottleneck { sum += l1x1_w[ic * channels + oc] * z_scratch[ic]; }  // 0..bottleneck INTEIRO
    l1x1_scratch[oc] = sum;
}
```

Em nenhum dos três pontos (`build.rs`, `layer.rs`, `process.rs`) existe qualquer leitura de
`layer1x1.groups` ou `groups_input_mixin` do JSON.

**Prova por contraste — o padrão de correção já existe no mesmo motor, aplicado ao `head1x1`:**

```rust
// src/models/a2/model/dynamic/process.rs:451-473
// S14.2 (PM-15): Correct grouped head1x1 accumulation.
let h1_in = head1x1_w.len() / head_accum_size;
let h1_groups = bottleneck.checked_div(h1_in).unwrap_or(1);
let ch_per_group = head_accum_size / h1_groups;
for grp in 0..h1_groups {
    for oc in grp * ch_per_group..(grp + 1) * ch_per_group {
        let mut sum = head1x1_b[oc];
        for ic in 0..h1_in { sum += head1x1_w[oc * h1_in + ic] * z_scratch[grp * h1_in + ic]; }
        head1x1_scratch[oc] = sum;
    }
}
```

O comentário `S14.2 (PM-15): Correct grouped head1x1 accumulation` prova que esta é uma correção
deliberada anterior — a mesma disciplina nunca foi estendida a `layer1x1`/`input_mixin`. **Importante:**
"o padrão já existe implementado" refere-se apenas à *estrutura do código*; a *correção* desse
padrão específico para `head1x1.groups > 1` contra o C++ real permanece empiricamente não validada
(nenhum fixture com `head1x1.groups > 1` jamais passou por um golden — ver §3.5) — não deve ser
copiado como referência de correção comprovada, apenas como referência de estilo de implementação.
A estratégia Rust (buffer reduzido + laço restrito ao grupo, em vez do buffer denso
zero-preenchido do C++) é preferível por evitar multiplicações por zero no hot-path, alinhada à
filosofia de eficiência de RT-safety do projeto (`.agents/rules/rust.md` §"DSP & RT-Safe Math").

#### 3.4 Bug C (agora **confirmado**, não mais hipótese): cabeçalho final deveria ser `Conv1x1` (K=1), não `Conv1D` de 16 taps

```cpp
// tests/fixtures/NeuralAmpModelerCore/NAM/wavenet/model.cpp:894-898 — formato legado força K=1
else if (layer_config.find("head_size") != layer_config.end())
{
    head_size = layer_config["head_size"].get<int>();
    head_kernel_size = 1;                                    // forçado a 1 no formato legado
    head_bias = layer_config.at("head_bias").get<bool>();
}
```

O próprio `generate_weights_a2.py` confirma isso na definição de `count_layer_array_weights`
(docstring: *"Head rechannel Conv1x1: (head_output_size, head_size, bias=head_bias)"*) — o
cabeçalho final é modelado como um `Conv1x1` (kernel implícito = 1), não uma convolução temporal:

```python
# tests/fixtures/NeuralAmpModelerCore/generate_weights_a2.py:221-225
weight_count += count_conv1x1_weights(
    head_output_size, head_size,
    has_bias=head_bias, groups=1
)
```

Para `wavenet_a2_max.nam` (`head_output_size=4` pois `head1x1` está ativo com `out_channels=4`,
`head_size=1`, `head_bias=true`): `count_conv1x1_weights(4, 1, bias=True) = 4*1 + 1 = 5` pesos.

Já o Rust hardcoda `A2_HEAD_KERNEL_SIZE=16` incondicionalmente:

```rust
// src/models/a2/model/dynamic/build.rs:246-260 (load_head_conv_and_scale)
let head_k = crate::models::a2::params::A2_HEAD_KERNEL_SIZE;   // = 16, sempre
// head_size == 1:
let head_w_f32 = read_slice(weights, pos, head_k * channels, total, "head_w")?;  // 16*4 = 64, não 4
```

Rust lê **64 pesos** (mais bias e scale = 66) onde o C++ espera **4 pesos** (mais bias = 5).
**Esta divergência de 61 valores é, por si só, catastrófica** — mas ocorre no **fim** do stream de
pesos do array externo, então **não** é a causa da corrupção observada em `test_oracle_a2_generic`
sobre o `condition_dsp` (que ocorre por um caminho de dispatch totalmente separado, ver §3.7); é,
isoladamente, a causa da leitura incorreta do cabeçalho final do array externo.

*(Nota: o kernel `16` está correto para os fixtures A2 canônicos, que declaram um objeto `"head"`
aninhado explícito com `kernel_size:16` no JSON — ver §0.3. O bug é usar esse valor
incondicionalmente, sem checar a presença do objeto `"head"` aninhado, aplicando-o também ao
formato legado onde o C++ define `kernel_size=1`.)*

#### 3.5 Reconciliação exata do orçamento de pesos (818 = 4 + 2×404 + 5 + 1 — sem nenhum resíduo)

Usando o script autoritativo (§3.1) para decompor exatamente o array externo do modelo:

| Componente                                                 | Fórmula (script)                                       | Valor     |
| ---------------------------------------------------------- | ------------------------------------------------------ | --------- |
| Rechannel (`Conv1x1(input_size=1 → channels=4)`, sem bias) | `count_conv1x1_weights(1, 4, False, 1)`                | **4**     |
| Camada 0 (dilação 1) — completa, incl. `head1x1` e 8 FiLM  | `count_layer_weights(layer, cond_size=8, layer_idx=0)` | **404**   |
| Camada 1 (dilação 2) — completa, incl. `head1x1` e 8 FiLM  | `count_layer_weights(layer, cond_size=8, layer_idx=1)` | **404**   |
| Head rechannel final (`Conv1x1(4 → 1)`, com bias, **K=1**) | `count_conv1x1_weights(4, 1, True, 1)`                 | **5**     |
| `head_scale` (escalar global)                              | `+1`                                                   | **1**     |
| **Total**                                                  |                                                        | **818**✅ |

Decompondo uma camada (404 pesos), com os valores reais do modelo (`channels=4, bottleneck=4,
condition_size=8, groups_input=1, groups_input_mixin=4, layer1x1.groups=2, head1x1.groups=2,
head1x1.out_channels=4, kernel_size=4`):

| Sub-componente                                            | Fórmula                      | Valor   |
| --------------------------------------------------------- | ---------------------------- | ------- |
| `conv` (Conv1D, K=4, groups_input=1)                      | `4 * (4*4//1) + 4` (bias)    | 68      |
| `input_mixin` (Conv1x1, groups=4, sem bias)               | `(4*8//4)`                   | 8       |
| `layer1x1` (Conv1x1, groups=2, com bias)                  | `(4*4//2) + 4`               | 12      |
| `head1x1` (Conv1x1, groups=2, com bias) — **por camada!** | `(4*4//2) + 4`               | 12      |
| 8 slots de FiLM (várias combinações de `groups`)          | soma individual (ver script) | 304     |
| **Total por camada**                                      |                              | **404** |

**Esta reconciliação fecha exatamente, sem nenhum resíduo, e confirma com certeza absoluta —
não por hipótese — os três bugs descritos em §3.2-§3.4**: (1) `head1x1` deve ser lido **duas
vezes** (uma por camada), não uma vez para o array inteiro; (2) `input_mixin`/`layer1x1` devem
usar a contagem **agrupada** (8 e 12, não 32 e 16 como o Rust assume hoje); (3) o cabeçalho final
deve ter **5** pesos (`Conv1x1`, K=1), não 66 (`Conv1D`, K=16, como o Rust assume hoje).

*(Nota metodológica: uma tentativa anterior de reconciliar este orçamento manualmente, sem o
script autoritativo, não fechou sob nenhuma hipótese simples testada — porque a hipótese de
`head1x1` "uma vez por array" nunca foi questionada. Isso é, em si, uma lição de processo: a
reconciliação aritmética só é conclusiva quando ancorada em uma fonte autoritativa (o script
upstream), não em fórmulas reconstruídas por leitura de código isolada, que podem compartilhar a
mesma suposição estrutural errada em ambos os lados da comparação.)*

#### 3.6 Prova por ausência total de cobertura de teste

Varredura de **todos** os fixtures WaveNet do repositório (`tests/fixtures/models/*.nam`) por
`layer1x1.groups`, `groups_input_mixin`, `groups_input`, `head1x1.groups` e `condition_size`
diferentes de `1`, e pelo número de camadas dilatadas quando `head1x1` está ativo:

| Fixture                                            | `layer1x1.groups` | `groups_input_mixin` | `head1x1.groups` | `condition_size` | `head1x1.active`       | nº camadas    |
| -------------------------------------------------- | ----------------- | -------------------- | ---------------- | ---------------- | ---------------------- | ------------- |
| `wavenet_a2_max.nam`                               | **2**             | **4**                | **2**            | **8**            | **true**               | **2**         |
| `wavenet_condition_dsp.nam` (standalone, 2 arrays) | 1                 | 1                    | 1                | 3                | **false** (confirmado) | 2 e 1         |
| `a2_dynamic_gated_ch8.nam`                         | 1                 | 1                    | 1                | 1                | **false** (confirmado) | 23 (canônico) |
| `a2_dynamic_blended_ch3.nam`                       | 1                 | 1                    | 1                | 1                | **false** (confirmado) | 23 (canônico) |
| `wavenet_a2_film_full.nam`/`_lite.nam`             | 1                 | 1                    | 1                | 1                | **false** (confirmado) | 23 (canônico) |
| todos os demais A1/A2                              | 1                 | 1                    | 1                | 1                | inactive               | 23 (canônico) |

**`wavenet_a2_max.nam` é o único fixture em toda a suíte que exercita `layer1x1.groups > 1`,
`groups_input_mixin > 1`, ou `head1x1.active=true` com menos de 23 camadas.** Isso confirma: nem o
Bug A, nem o Bug B, nem o Bug C jamais foram expostos por nenhum teste até agora.

**Confirmado nesta sessão** (verificação direta do JSON de cada fixture, `head1x1.active`):
`a2_dynamic_gated_ch8`, `a2_dynamic_blended_ch3`, `wavenet_a2_film_full`, `wavenet_a2_film_lite` e
o próprio `wavenet_condition_dsp.nam` (fixture standalone, distinto do `condition_dsp` embutido em
`wavenet_a2_max.nam` — ver §3.7) têm **todos** `head1x1.active=false`. Isso resolve a aparente
contradição: **o Bug A está dormente (nunca exercitado) em todos os fixtures que hoje passam** —
quando `head1x1` está inativo, nenhum peso de `head1x1` é consumido em nenhum dos dois lados
(C++/Rust), então a questão "por camada vs. por array" nunca se manifesta. Não há, portanto,
nenhuma regressão silenciosa em produção hoje — o bug é real, mas **atualmente inatingível por
qualquer modelo em uso**, exceto `wavenet_a2_max.nam` (bloqueado) e o `condition_dsp` nele
embutido (também bloqueado, transitivamente, pelo mesmo guard).

#### 3.7 Impacto sobre `condition_dsp` — o mesmo motor, os mesmos três bugs, duas vezes

O sub-modelo `condition_dsp` embutido em `wavenet_a2_max.nam` (2 arrays, `head1x1.active=true` em
**ambos**, sem objeto `"head"` aninhado em nenhum dos dois, dilatações `[1,2]` e `[1,3,5]`
respectivamente) é despachado **recursivamente pelo mesmo dispatcher** (`build_model` →
`build_wavenet` → `is_a2_shape` → `WaveNetA2Dyn`/`WavenetA2Cascade`). Isso significa que os Bugs
A e C (per-camada `head1x1` e cabeçalho `K=1`) **também afetam a construção do `condition_dsp`
internamente** — confirmado pela mesma reconciliação exata via `generate_weights_a2.py`:

```shell
$ python3 -c "... count_wavenet_weights(condition_dsp_config) ..."
1052   # idêntico a len(condition_dsp['weights']) — mesma reconciliação exata
```

Isso torna o diagnóstico anterior do "bug no `condition_dsp`" (Achado 2 legado, atribuído
inteiramente ao oráculo f64) **ainda mais claramente insuficiente**: mesmo que o oráculo fosse
perfeito, o **próprio `condition_dsp` de produção** (construído pelo mesmo `WaveNetA2Dyn`/
`WavenetA2Cascade` buscado pelos Bugs A/B/C) já estaria corrompido internamente, de forma
totalmente independente de qualquer bug do oráculo de teste.

### 4. Riscos e escopo deste achado

* **Bug A (estrutural) — risco de implementação: alto.** Exige mover `head1x1_w`/`head1x1_b` (e a
  respectiva instância de `head1x1_post_film`) de `WaveNetA2Dyn` (nível de array) para `A2Layer`
  (nível de camada) — uma mudança de layout de struct, não apenas de fórmula. Toca
  `src/models/a2/layer.rs`, `src/models/a2/model/dynamic/build.rs`,
  `src/models/a2/model/dynamic/mod.rs` e `src/models/a2/model/dynamic/process.rs`. Requer
  reverificação cuidadosa de todos os modelos que ativam `head1x1` hoje.
* **Bugs B/C — risco de implementação: médio.** Mudanças de fórmula localizadas, sem alteração de
  layout de struct (Bug B pode reaproveitar o padrão já existente de `head1x1`; Bug C é uma
  correção isolada em `load_head_conv_and_scale`).
* **Risco de regressão: médio-alto, compartilhado entre os três bugs.** Todo o código tocado é
  compartilhado por **todos** os modelos que passam por `WaveNetA2Dyn`/`WavenetA2Cascade`
  (`a2_dynamic_gated_ch8`, `a2_dynamic_blended_ch3`, os fixtures FiLM, e o próprio
  `wavenet_condition_dsp.nam`). Qualquer alteração exige reverificação total desses modelos — a
  condição de aceite obrigatória para cada bug é: **quando a condição degenerada se aplica
  (`groups==1`, `head1x1` com uma única camada, cabeçalho com objeto `"head"` explícito), o
  comportamento deve permanecer bit-idêntico ao atual.**
* **Não bloqueante para este achado, mas deve ser corrigido em paralelo (Epic 6):** os 3 bugs do
  oráculo f64 documentados na seção 2 permanecem reais e devem ser corrigidos para que
  `test_oracle_a2_generic` e companhia voltem a ser úteis como ferramenta de decomposição de erro
  — mas **não desbloqueiam, por si só, o guard de produção**.
* **Achado adjacente, não confirmado, fora do escopo imediato:** as fórmulas
  `film_weight_count_generic`/`film_bias_count_generic` (`src/models/a2/weights_layout.rs:46-64`),
  usadas exclusivamente quando `condition_size > 1`, também têm **cobertura de teste zero** fora de
  `wavenet_a2_max.nam`. A reconciliação exata da §3.5 já as valida indiretamente para os valores de
  `groups`/`condition_size` presentes neste modelo específico (o total de 304 por camada bateu
  exatamente), o que é uma evidência forte — mas não uma prova formal e exaustiva para todas as
  combinações possíveis de `groups`/`shift`/`condition_size`. Recomenda-se auditoria dedicada (ver
  Epic 5).

---

## Proposta de Solução — Caminho de Investigação e Correção

Diferente da versão anterior deste documento, a causa raiz **não precisa mais ser isolada por
instrumentação** — os três bugs já estão confirmados com certeza matemática (§3.5). O trabalho
remanescente é de **implementação e verificação**, na ordem correta de dependência.

### Fase A — Validação cruzada com o script autoritativo (leve, não bloqueante, alta confiança)

1. Portar (ou invocar via `pyo3`/subprocesso em um teste de desenvolvimento, não em produção) as
   fórmulas de `generate_weights_a2.py` para um utilitário Rust de checagem de orçamento de pesos
   — reutilizável para qualquer fixture A2 futuro, não descartável após este achado. Sugestão de
   local: `tests/parity/` ou `utils/`.
2. Usar esse utilitário como **gate automatizado**: antes de cada fixture A2 genérico ser
   aceito/commitado, validar que `count_wavenet_weights(config) == len(weights)` — capturando
   erros de contagem de pesos (como os Bugs A/B/C) **na fonte**, antes de chegarem a testes de
   ESR/SNR de alto nível.
3. Esta fase é **auxiliar e de baixo custo** — não bloqueia o início da Fase B, mas deve ser
   concluída antes da Fase F (regeneração/remedição final), como camada extra de confiança.

### Fase B — Corrigir Bug A: `head1x1` por camada, não por array (produção — fazer PRIMEIRO)

0. **Pré-requisito de verificação — já resolvido nesta sessão (ver §3.6):** confirmado que
   `a2_dynamic_gated_ch8.nam` e `a2_dynamic_blended_ch3.nam` têm `head1x1.active=false`. O Bug A
   está dormente nesses fixtures (e em todos os demais fixtures atualmente passando) — nenhuma
   regressão silenciosa em produção hoje. **Ainda assim, a Fase B.4 (condição de aceite) deve
   reexecutar esses dois fixtures após a correção**, pois a refatoração estrutural (mover
   `head1x1_w`/`head1x1_b` de `WaveNetA2Dyn` para `A2Layer`) toca as mesmas structs que eles usam,
   mesmo com `head1x1` inativo.
1. **Estrutura (`src/models/a2/layer.rs`):** mover os campos `head1x1_w: AlignedVec<f32>` e
   `head1x1_b: AlignedVec<f32>` de `WaveNetA2Dyn` (`src/models/a2/model/dynamic/mod.rs`) para
   `A2Layer`, ao lado de `l1x1_w`/`l1x1_b`. Adicionar também `h1_in: usize`/`h1_groups: usize` (ou
   recalculá-los a partir do tamanho do buffer, como já é feito hoje em `process.rs:451-473`) por
   camada, já que `head1x1.out_channels`/`groups` são, em princípio, configuráveis por camada no
   C++ (`LayerParams` é por-camada), mesmo que a JSON de `wavenet_a2_max.nam` use um único objeto
   `head1x1` broadcast igualmente para as 2 camadas.
2. **Loader (`src/models/a2/model/dynamic/build.rs`):** mover a leitura de `head1x1` de
   `load_head1x1_weights` (hoje chamada uma vez, depois do laço de camadas) para dentro de
   `load_per_layer_weights`, posicionada **depois de `l1x1_b`, antes das FiLM** — replicando
   exatamente a ordem confirmada em `Layer::set_weights_` (`model.cpp:135-164`: `conv → mixin →
   layer1x1 → head1x1 → FiLM×8`). Chamada uma vez por `i in 0..num_layers`.
3. **Runtime (`src/models/a2/model/dynamic/process.rs`):** mover a lógica de acumulação de
   `head1x1` (hoje em `process.rs:451-473`, usando os campos array-level `self.head1x1_w`/`b`) para
   dentro do processamento por camada, usando os novos campos por-camada de `A2Layer`. O
   `head1x1_post_film` (que já é por-camada em `A2Layer::head1x1_post_film`) deve continuar
   operando sobre a saída do `head1x1` **desta mesma camada**.
4. **Condição de aceite obrigatória:** para modelos com `num_layers==1` (nenhum fixture atual tem
   isso com `head1x1` ativo, mas é o caso trivial), ou quando `head1x1.active==false`, o resultado
   deve ser bit-idêntico ao comportamento atual.
5. Espelhar a mesma correção nos caminhos estáticos (`conv1d_ch3/simd.rs`, `conv1d_ch8/simd.rs`) —
   mesma nota de "dead code hoje, mas dívida técnica" já registrada para bugs anteriores (B1/B2/B3
   do Achado F2, já corrigido): o dispatcher só roteia modelos com `head1x1` ativo para
   `WaveNetA2Dyn`, nunca para o caminho estático, mas os dois devem permanecer espelhados por
   higiene e para não reintroduzir divergência silenciosa caso a política de roteamento mude.

### Fase C — Corrigir Bug B: `groups` ignorado em `layer1x1`/`input_mixin`

1. **Loader (`build.rs`):** ler `layer1x1.groups` e `groups_input_mixin` do JSON
   (`layer_cfg.layer_raw`), no mesmo ponto onde `head1x1.groups`/`h1_in_size` já são lidos hoje em
   `src/loader/dispatcher/wavenet/mod.rs:260-271`. Recalcular `mixin_count` (`build.rs:138`) como
   `conv_out * (condition_size / groups_input_mixin)` e `l1x1_w_count` (`build.rs:150`) como
   `channels * (bottleneck / layer1x1_groups)` — mirroring `h1_w_count = channels * h1_in`.
2. **Runtime (`process.rs`):** restringir os laços de `mixin`/`l1x1` ao grupo correspondente,
   replicando a estrutura de código já usada para `head1x1` (agora movida para `A2Layer` pela
   Fase B).
3. **Condição de aceite:** para `groups==1` (todos os fixtures atuais), resultado bit-idêntico ao
   atual.
4. Testes unitários dedicados (isolados, sem depender de golden/fixture), mesmo estilo de
   `test_wavenet_a2_dyn_bug_b1_mixin_post_film` (`src/models/a2/model/dynamic_test.rs`).

### Fase D — Corrigir Bug C: cabeçalho final deve respeitar `head_kernel_size=1` no formato legado

1. Em `load_head_conv_and_scale` (`build.rs:240-349`), ler `kernel_size` do objeto `"head"`
   aninhado quando presente (formato canônico, atualmente sempre `16` para os fixtures deste
   repo); quando ausente (formato legado `head_size`/`head_bias` plano), usar `kernel_size=1`
   (`Conv1x1`), replicando exatamente `model.cpp:894-898`.
2. **Condição de aceite:** para modelos com objeto `"head"` aninhado presente (todos os fixtures
   atuais exceto `wavenet_a2_max.nam` e o `condition_dsp` nele embutido), resultado bit-idêntico
   ao atual.

### Fase E — Auditoria de cobertura remanescente

1. Auditar `film_weight_count_generic`/`film_bias_count_generic`
   (`src/models/a2/weights_layout.rs:46-64`) formalmente para outras combinações de
   `groups`/`shift`/`condition_size` não cobertas por `wavenet_a2_max.nam` — a reconciliação exata
   da §3.5 já dá evidência forte de que a fórmula está correta *para os valores presentes neste
   modelo*, mas não é uma prova exaustiva.
2. Auditar `groups_input` do `_conv` dilatado (`groups_input=1` neste fixture — não contribui à
   divergência atual, mas é outro parâmetro sem nenhuma cobertura de teste com valor `> 1`).

### Fase F — Regeneração de golden, remedição e desbloqueio do guard

1. Após B+C+D implementadas, rodar o utilitário da Fase A para confirmar reconciliação exata de
   pesos, depois `test_golden_vectors_wavenet_a2_max` (removendo o `#[ignore]` temporariamente) e
   medir ESR/SNR/MSE reais contra o golden C++ existente (`golden_wavenet_a2_max.bin` — não deveria
   ser necessário regenerar o golden, pois ele já foi renderizado pelo NAMCore real a partir do
   `.nam` original inalterado).
2. Critério de desbloqueio: SNR > 90 dB (consistente com o piso de precisão f32 observado em todos
   os demais modelos A2 dinâmicos já validados — `a2_dynamic_gated_ch8`: 103 dB,
   `a2_dynamic_blended_ch3`: 133 dB, FiLM 4-slots: 124–139 dB). Lembrar (§0.5) que este golden
   captura "o que esta build do C++ calcula", não uma verdade absoluta externa — mas é o padrão de
   comparação já usado por todos os demais fixtures A2 dinâmicos deste projeto.
3. Só então remover/ajustar `is_disabled_broken_a2_flagship`
   (`src/loader/dispatcher/wavenet/mod.rs:85-91`), reabilitar os testes listados na seção 1, e
   atualizar `docs/cpp_parity_map.md` §4.4/§7.1 com o novo diagnóstico e as medições finais.
4. Reexecutar explicitamente `a2_dynamic_gated_ch8`, `a2_dynamic_blended_ch3`, `A2 Full/Lite` e os
   3 fixtures FiLM (todos com `groups == 1` e `head1x1.active == false`, confirmado em §3.6) para
   confirmar **zero regressão**.

---

## Epics (agrupamento para planejamento futuro — `TODO-sprints.md` a ser gerado quando solicitado)

### Epic 1 — Validação cruzada com `generate_weights_a2.py` (leve, alta confiança)

1. Portar/invocar as fórmulas do script autoritativo como utilitário Rust reutilizável (Fase A.1).
2. Integrar como gate automatizado de orçamento de pesos para fixtures A2 futuros (Fase A.2).

### Epic 2 — Correção de produção: Bug A — `head1x1` por camada, não por array (fazer PRIMEIRO)

1. ✅ Já verificado (§3.6, Fase B.0): `head1x1.active=false` em `a2_dynamic_gated_ch8`/
   `_blended_ch3` — bug dormente, sem regressão silenciosa hoje.
2. Mover `head1x1_w`/`head1x1_b` de `WaveNetA2Dyn` para `A2Layer` (Fase B.1).
3. Mover a leitura de pesos para dentro do laço por camada, na posição correta (Fase B.2).
4. Mover a lógica de acumulação em runtime para operar por camada (Fase B.3).
5. Espelhar nos caminhos estáticos CH3/CH8 (Fase B.5).
6. Testes unitários dedicados provando `num_layers==1 → idêntico ao atual` e
   `num_layers>1 com head1x1 → pesos e saída independentes por camada`.

### Epic 3 — Correção de produção: Bug B — `groups` em `layer1x1`/`input_mixin`

1. Loader: propagar e aplicar `layer1x1.groups`/`groups_input_mixin` em `build.rs` (Fase C.1).
2. Runtime: restringir os laços ao grupo em `process.rs`, reaproveitando o padrão da Fase B
   (Fase C.2).
3. Testes unitários dedicados (`groups==1` idêntico; `groups>1` calculado manualmente) (Fase C.4).

### Epic 4 — Correção de produção: Bug C — `head_kernel_size` para formato legado

1. Ler `kernel_size` do objeto `"head"` aninhado quando presente; usar `1` no formato legado
   (Fase D.1).

### Epic 5 — Auditoria de cobertura remanescente

1. Auditar `film_weight_count_generic`/`film_bias_count_generic` para combinações não cobertas por
   `wavenet_a2_max.nam` (Fase E.1).
2. Auditar `groups_input` do `_conv` dilatado (Fase E.2).

### Epic 6 — Correção do oráculo f64 (ortogonal, não bloqueante — Achado 2 legado)

1. Corrigir `ArrayState::head_b` para `Vec<f64>` (`src/testing/reference_oracle/a2.rs:306-322`).
2. Corrigir a leitura de `head1x1_w`/`head1x1_b` para usar `head_accum_size` em vez de `ch`
   (`a2.rs:490-499`).
3. Corrigir `ch_per_group` para usar `head_accum_size` em vez de `ch` (`a2.rs:754`).
4. Corrigir a finalização do cabeçalho do oráculo para `head_size > 1` (buffer de saída
   `num_frames * head_size`, loop interleaved por canal).
5. **Adicionalmente** (novo, decorrente do Achado A desta sessão): avaliar se o oráculo também
   precisa tratar `head1x1` como por-camada — hoje ele compartilha a mesma limitação estrutural do
   código de produção (verificar `src/testing/reference_oracle/a2.rs` quanto a este ponto
   especificamente antes de declarar o oráculo corrigido).
6. Reabilitar `test_oracle_vs_python_anchor_a2_generic`, `test_oracle_a2_generic`,
   `test_decomposition_a2_generic`, `test_combined_simulation_a2_generic`.
7. **Explicitamente não depende de, nem bloqueia, os Epics 1-5** — pode ser executado em paralelo
   por outro engenheiro/sessão.

### Epic 7 — Regeneração, remedição, desbloqueio e documentação

1. Rodar o utilitário da Epic 1 para confirmar reconciliação exata, depois
   `test_golden_vectors_wavenet_a2_max` sem `#[ignore]` e medir ESR/SNR/MSE reais (Fase F.1-F.2).
2. Remover/ajustar `is_disabled_broken_a2_flagship` e reabilitar todos os testes da seção 1
   (Fase F.3).
3. Reverificar zero regressão nos modelos de risco (Fase F.4).
4. Atualizar `docs/cpp_parity_map.md` §4.4/§7.1 com o diagnóstico final e as medições, incluindo a
   proveniência upstream do fixture (§0) e os três bugs A/B/C.
5. Homologação final: `utils/lints.sh` + `utils/tests-quick.sh` + `utils/quality-dashboard.sh`
   (modo completo, ao menos uma vez antes do merge).
