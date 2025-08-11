#!/bin/sh
# ==============================================================================
# Project : AG News Text Classification
# Team    : Aimer PAM
# Author  : Vo Hai Dung
# License : MIT
# ==============================================================================
# One-step launcher for a macOS or Linux machine with no developer tooling.
#
# The uv installer places the binary in a per-user directory and needs no sudo.
# PATH is extended for this process only, so the usual "open a new terminal"
# step after installation is not required. Both the current install location
# and the legacy Cargo one are probed, and curl falls back to wget on the
# minimal images that ship only one of them.
#
# Usage:  sh bootstrap.sh  [any run_demo.py flags]

set -eu

cd "$(dirname "$0")"

if [ ! -f run_demo.py ]; then
    echo "run_demo.py was not found next to this file." >&2
    echo "Run bootstrap.sh from inside the project folder, not from a copy of" >&2
    echo "the script on its own." >&2
    exit 1
fi

UV_LOCAL="${XDG_BIN_HOME:-$HOME/.local/bin}"
UV_CARGO="$HOME/.cargo/bin"

find_uv() {
    command -v uv >/dev/null 2>&1 && return 0
    for dir in "$UV_LOCAL" "$UV_CARGO"; do
        if [ -x "$dir/uv" ]; then
            PATH="$dir:$PATH"
            export PATH
            return 0
        fi
    done
    return 1
}

if ! find_uv; then
    echo "Installing uv into $UV_LOCAL ..."
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        echo "Neither curl nor wget is available; install one of them first." >&2
        exit 1
    fi

    if ! find_uv; then
        echo >&2
        echo "uv could not be installed automatically." >&2
        echo "Install it by hand from https://docs.astral.sh/uv/getting-started/installation/" >&2
        echo "then run this file again." >&2
        exit 1
    fi
fi

exec uv run run_demo.py "$@"
