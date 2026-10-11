#!/bin/zsh
set -eu
dashboard_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$dashboard_dir/dist"
(sleep 1; open 'http://127.0.0.1:8765') &
exec python3 -m http.server 8765 --bind 127.0.0.1
