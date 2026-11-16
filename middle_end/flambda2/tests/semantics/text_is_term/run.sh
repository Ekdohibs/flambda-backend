#!/bin/sh

# Runs the driver that checks that the text of a term, read back, is the term
# (../../tools/textisterm/fltextisterm.ml) on one OCaml source:
#
#   run.sh <fltextisterm.exe> <X.ml or X.mli>
#
# The unit is compiled with no standard library and no initially opened
# module, with X.mli if that is beside X.ml, in a directory of its own. What
# the driver prints on standard output is printed, then its exit status; what
# the compiler writes on standard error is not shown.

set -u

driver=$1
source=$2

case "$driver" in
/*) ;;
*) driver="$PWD/$driver" ;;
esac

dir=$(dirname "$source")
file=$(basename "$source")
name=${file%.*}

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cp "$source" "$work/$file"
files=$file
if [ "$file" = "$name.ml" ] && [ -f "$dir/$name.mli" ]; then
  cp "$dir/$name.mli" "$work/$name.mli"
  files="$name.mli $name.ml"
fi
cd "$work"
"$driver" -nopervasives -nostdlib -stop-after middle_end -c $files 2>/dev/null
echo "exit status $?"
