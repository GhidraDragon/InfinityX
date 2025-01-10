#!/usr/bin/env bash
#
# select_architecture.sh
#
# A script for macOS that lets you run a command under a chosen CPU architecture.
# On Apple Silicon (M1/M2), you can run commands under Rosetta (Intel) or natively.
# On Intel-only Macs, specifying x86_64 will simply run commands normally,
# and attempting to specify arm64 will fail if the system doesn't support it.
#
# Usage:
#   ./select_architecture.sh [--intel | --apple-silicon] <command> [args...]
#
# Examples:
#   ./select_architecture.sh --intel uname -m
#   ./select_architecture.sh --apple-silicon uname -m

# Print usage if not enough arguments are provided.
function usage() {
  echo "Usage: $0 [--intel | --apple-silicon] <command> [args...]"
  echo ""
  echo "  --intel          Run the specified command under the x86_64 (Intel) architecture."
  echo "  --apple-silicon  Run the specified command under the arm64 (Apple Silicon) architecture."
  echo ""
  echo "Examples:"
  echo "  $0 --intel uname -m"
  echo "  $0 --apple-silicon uname -m"
  exit 1
}

# Check for at least two arguments: the architecture flag and a command.
if [ $# -lt 2 ]; then
  usage
fi

# Determine the requested architecture.
ARCH_FLAG="$1"
shift

case "$ARCH_FLAG" in
  --intel)
    ARCH="x86_64"
    ;;
  --apple-silicon)
    ARCH="arm64"
    ;;
  *)
    echo "Error: Unknown architecture option '$ARCH_FLAG'"
    usage
    ;;
esac

# On Apple Silicon, 'arch -x86_64' runs commands under Rosetta (Intel),
# 'arch -arm64' runs them natively.
# On Intel-only Macs, 'arch -x86_64' is typically valid, and 'arch -arm64' will fail if Rosetta isn't supported.
# We run the user's command with the chosen architecture.
arch -${ARCH} "$@"
