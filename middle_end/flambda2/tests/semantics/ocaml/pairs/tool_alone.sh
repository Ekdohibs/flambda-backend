#!/bin/sh

# Runs the test tool on a unit of a pair with a load path that has the .cmx
# of one unit and no other (see README.md, "The load path"):
#
#   tool_alone.sh <flsemantics.exe> <F.cmx> [<flags>] <X.fl>
#
# The rules of this directory give the tool the directory of the build as
# its load path, which has the .cmx of every unit here. A unit whose source
# has the line
#
#   (* load path: the first unit alone *)
#
# is run by this script instead: the tool's Simplify then finds the .cmx of
# the first unit of the pair, in a directory of its own, and the .cmx of no
# other unit, so a unit that the first unit uses is one that was not read.

set -eu

tool=$1
cmx=$2
shift 2

case "$tool" in
/*) ;;
*) tool="$PWD/$tool" ;;
esac

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cp "$cmx" "$work/"
"$tool" -I "$work" "$@"
