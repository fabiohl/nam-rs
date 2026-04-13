#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva.

set -xeuo pipefail

echo "Updates no cargo do nam-rs"
cargo upgrade --verbose
cargo update --verbose