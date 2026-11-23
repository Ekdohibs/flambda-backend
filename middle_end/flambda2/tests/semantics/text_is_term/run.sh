#!/bin/sh

# Runs the driver that checks that the text of a term, read back, is the term
# (../../tools/textisterm/fltextisterm.ml):
#
#   run.sh <fltextisterm.exe> <argument> ...
#
# The driver is run in a directory of its own, with no standard library and
# no initially opened module, stopped after the middle end. An argument that
# is X.ml or X.mli is copied there (X.mli with X.ml, and given before it, if
# it is beside it) and given to the driver under its name; copy:FILE:NAME
# copies FILE there as NAME and gives the driver nothing; any other argument
# is given as it is. The arguments come first on the command line of the
# driver (the compiler asks for -depend there). The last line that the
# driver prints on standard output is printed, then its exit status; what
# the compiler writes on standard error is not shown.
#
# first:X.ml copies X.ml there and has the driver compile it to its end
# before the run that is shown, with no standard library, no initially opened
# module and no other option, so that the unit of that run can name the unit
# X. If the driver does not exit with 0 on it, which it does only if the text
# of X is its term, "X.ml is not compiled" is printed and nothing is run.

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
firsts=
for arg in "$@"; do
  case "$arg" in
  first:*)
    file=${arg#first:}
    cp "$file" "$work/$(basename "$file")"
    firsts="$firsts $(basename "$file")"
    ;;
  copy:*)
    rest=${arg#copy:}
    cp "${rest%:*}" "$work/${rest##*:}"
    ;;
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
for file in $firsts; do
  if ! "$driver" -nopervasives -nostdlib -c "$file" >/dev/null 2>&1; then
    echo "$file is not compiled"
    exit 0
  fi
done
"$driver" $args -nopervasives -nostdlib -stop-after middle_end -c \
  >stdout 2>/dev/null
status=$?
tail -n 1 stdout
echo "exit status $status"
