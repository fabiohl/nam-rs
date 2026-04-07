#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva.

set -xeuo pipefail

echo "Updates no cargo do nam-rs"
cargo upgrade --verbose
cargo update --verbose

echo "Update dos githubs de inspiração"
cd github.com/mikeoliphant/NeuralAudio ; git pull ; git fsck ; git gc ; cd -
cd github.com/mikeoliphant/NeuralAudio/Utils/deps/argparse ; git pull ; git fsck ; git gc ; cd -
cd github.com/mikeoliphant/NeuralAudio/deps/math_approx ; git pull ; git fsck ; git gc ; cd -
cd github.com/mikeoliphant/NeuralAudio/deps/NeuralAmpModelerCore ; git pull ; git fsck ; git gc ; cd -
cd github.com/mikeoliphant/NeuralAudio/deps/RTNeural ; git pull ; git fsck ; git gc ; cd -