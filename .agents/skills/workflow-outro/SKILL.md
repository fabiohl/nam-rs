---
name: workflow-outro
description: Trigger workflows defined in .agents/workflows/
---

# Skill: Trigger workflows

## When to use this skill

When the user asks to run a workflow by its name, as example `WORKFLOW-NAME`.

## Instructions

1. **Workflow Discovery**: If the requested workflow name is not fully specified or there is ambiguity, list the `.agents/workflows/` directory to discover available workflows.
2. **Execution**: Execute the workflow following the exact steps in `.agents/workflows/WORKFLOW-NAME.md`.
3. **Skill Delegation**: Pay attention to any specific skills mentioned in the workflow (e.g. `documentador`, `implementador`, `pesquisador-inovador`) and load their respective instructions to guide your execution.
4. **Fallback**: If the requested workflow does not exist, list the available workflows to the user and ask for clarification.
