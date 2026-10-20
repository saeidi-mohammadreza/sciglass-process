#!/usr/bin/env bash
# Download the open SciGlass release (plain-text conversion, ~190 MB) next to this repo.
set -euo pipefail
dest="${1:-../SciGlass}"
if [ -d "$dest/sciglass" ]; then echo "already present: $dest"; exit 0; fi
git clone --depth 1 https://github.com/drcassar/SciGlass "$dest"
echo "done: $dest"
