#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva.

set -xeuo pipefail
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
#cargo clippy --all-targets --all-features -- -D warnings -D clippy::pedantic -D clippy::nursery -D clippy::cargo