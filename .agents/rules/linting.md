---
trigger: glob
description: Diretrizes mandatórias de garantia de qualidade (Linting) para o encerramento das submissões da IA.
globs: **/*
---

# Qualidade e Linting ao Fim das Atividades

1. **Regra de Conclusão**: Ao término de *qualquer atividade* executada por você (a IA) que crie ou altere arquivos Rust, shell scripts ou qualquer artefato de código-fonte no projeto, você jamais deve considerar o passo encerrado ou reportar término definitivo ao usuário sem antes realizar a validação final.
2. **Documentação Atualizada:** Verifique se houve alguma alteração arquitetural relevante. Acione a skill `documentador` se for o caso.
3. **Compilação sem Erros**: Antes de rodar `lints.sh`, garanta que `cargo build` compila sem erros — o clippy só chega a checar warnings se a compilação básica já passar. Em caso de erros de compilação, corrija-os primeiro. Faça o mesmo com o `cargo check`.
4. **Validação Obrigatória**: Conclua sua atividade sempre executando, ao final, o script de checagem local: `utils/lints.sh`.
5. **Correção Exaustiva**: Se a execução de qualquer destes passos retornar quaisquer quebras, erros, falhas de formatação ou *warnings* no console, a sua tarefa ainda não terminou.
6. **Ciclo de Ajustes**: Identifique a fonte dos problemas apresentados (skill `debugger`), realize as correções necessárias no código e reexecute o passo que apontou o erro. Somente conclua sua atividade de fato e informe ao usuário o encerramento quando todos os problemas forem sanados e a checagem passar com status de sucesso absoluto.
7. **Higiene do Repositório**: Ao final de toda atividade, verifique se não há arquivos temporários, de log, ou artefatos de debug não listados no `.gitignore` que possam ter sido gerados durante a execução (ex: `console.log`, dumps, `.tmp`). Não polua o histórico git com artefatos de trabalho.
