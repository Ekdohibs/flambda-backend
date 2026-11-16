#!/bin/sh

# Runs the driver that checks that the text of a term, read back, is the term
# (../../tools/textisterm/fltextisterm.ml):
#
#   run.sh <fltextisterm.exe> <X.ml, X.mli or an option of the compiler> ...
#
# The sources are copied to a directory of its own, X.mli with X.ml if it is
# beside it, and compiled there with no standard library and no initially
# opened module, with the options given, which come first on the command line
# of the driver (the compiler asks for -depend there). The last line that the driver prints
# on standard output is printed, then its exit status; what the compiler
# writes on standard error is not shown.

set -u

driver=$1
shift

case "$driver" in
/*) ;;
*) driver="$PWD/$driver" ;;
esac

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
args=
for arg in "$@"; do
  case "$arg" in
  *.ml | *.mli)
    dir=$(dirname "$arg")
    file=$(basename "$arg")
    name=${file%.*}
    if [ "$file" = "$name.ml" ] && [ -f "$dir/$name.mli" ]; then
      cp "$dir/$name.mli" "$work/$name.mli"
      args="$args $name.mli"
    fi
    cp "$arg" "$work/$file"
    args="$args $file"
    ;;
  *) args="$args $arg" ;;
  esac
done
cd "$work"
"$driver" $args -nopervasives -nostdlib -stop-after middle_end -c \
  >stdout 2>/dev/null
status=$?
tail -n 1 stdout
echo "exit status $status"
