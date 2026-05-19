#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

set -xeuo pipefail
cargo test              # Costuma demorar 1,5 minuto
cargo bench             # Costuma demorar 7+ minutos
utils/build-clap.sh     # Costuma demorar 0,5 minuto
clap-validator validate ~/.clap/nam-rs.clap # Costuma demorar poucos segundos
