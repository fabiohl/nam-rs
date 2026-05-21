#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

set -xeuo pipefail
cargo test              # Costuma demorar quase 1,5 minutos
utils/build-clap.sh     # Costuma demorar 0,5 minuto (quando do zero)
clap-validator validate ~/.clap/nam-rs.clap # Costuma demorar poucos segundos
cargo bench             # Costuma demorar 5,5 minutos
