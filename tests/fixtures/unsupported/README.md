# Modelos Não Suportados (Legacy)

Este diretório contém artefatos de arquivos de modelo que não são suportados nativamente pelo pipeline unificado Core / `nam-rs`.

## `tw40_blues_deluxe_deerinkstudios.json`

O arquivo `tw40_blues_deluxe_deerinkstudios.json` foi identificado e classificado como um "Legacy Keras format" (formato legado / exportação original neural).
Diferentemente dos modelos .nam atuais ou padrões `NAMB` binários otimizados que contém metadata e cabeçalhos universais ("architecture", "version", "config", "weights"), este arquétipo expõe pesos estruturais enraizados (in_shape, layers).

Como o `nam-rs` foi arquitetado rigorosamente sob o novo formato determinístico suportado nativamente pelo Neural Amp Modeler Core, parsables como este arquivo não são digeridos ou englobados. Ele foi alocado neste diretório para documentação e reprocessamento caso ferramentas de migração/formato futuramente se tornem um alvo para o motor (fora do escopo corrente e fase Beta).
