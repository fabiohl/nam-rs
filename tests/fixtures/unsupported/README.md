# Unsupported Models (Legacy)

This directory contains model file artifacts that are not natively supported by the unified Core / `nam-rs` pipeline.

## `tw40_blues_deluxe_deerinkstudios.json`

The file `tw40_blues_deluxe_deerinkstudios.json` was identified and classified as a "Legacy Keras format" (legacy format / original neural export).
Unlike current .nam models or optimized binary `NAMB` standards that contain metadata and universal headers ("architecture", "version", "config", "weights"), this archetype exposes rooted structural weights (in_shape, layers).

Since `nam-rs` is strictly architected under the new deterministic format natively supported by the Neural Amp Modeler Core, parsables such as this file are not digested or encompassed. It was placed in this directory for documentation and reprocessing in case migration/format tools eventually become a target for the engine.
