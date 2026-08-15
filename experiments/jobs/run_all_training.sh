#!/bin/bash
# Thin wrapper — delegates to run_all.sh
exec "$(dirname "${BASH_SOURCE[0]}")/run_all.sh" --train