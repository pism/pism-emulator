#!/usr/bin/env bash
# Print the version after bumping CURRENT (vX.Y.Z) by PART (major, minor or patch).
#   next_version.sh v0.41.2 minor  ->  v0.42.0
set -euo pipefail

current="${1#v}"
part="$2"

if ! [[ "$current" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
  echo "error: '$1' is not a vX.Y.Z version" >&2
  exit 1
fi
major="${BASH_REMATCH[1]}"
minor="${BASH_REMATCH[2]}"
patch="${BASH_REMATCH[3]}"

case "$part" in
  major) echo "v$((major + 1)).0.0" ;;
  minor) echo "v${major}.$((minor + 1)).0" ;;
  patch) echo "v${major}.${minor}.$((patch + 1))" ;;
  *)
    echo "error: unknown bump part '$part'" >&2
    exit 1
    ;;
esac
