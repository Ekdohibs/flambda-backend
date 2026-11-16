#!/bin/sh

# Checks that the driver of the compiler's own run
# (../../tools/ownrun/flsemantics_ownrun.ml) compiles a unit as the compiler
# does:
#
#   same_object.sh <ocamlopt> <flsemantics_ownrun.exe> <X.ml> <flag> ...
#
# X.ml (with X.mli, given first, if it is beside X.ml) is compiled to the end
# of the compilation of a unit, by the compiler and then by the driver, each
# with "<flag> ... -c [X.mli] X.ml" and in the same directory, which has only
# the sources when either starts: the path of the directory may get into what
# is written. The object file, the .cmx file and the .cmi file that the driver
# writes must be the same bytes as those that the compiler writes. Where they
# are, nothing is printed; where they are not, it says which, and exits with
# 1.
#
# What the driver prints of the unit is not shown. Its status may be 1 where
# the compiler's is 0: that is of the check, and the unit is compiled whatever
# the check says.

set -eu

if [ $# -lt 3 ]; then
  echo "usage: $0 <ocamlopt> <flsemantics_ownrun.exe> X.ml <flag> ..." >&2
  exit 2
fi

ocamlopt=$1
driver=$2
source=$3
shift 3

case "$ocamlopt" in
/*) ;;
*) ocamlopt="$PWD/$ocamlopt" ;;
esac
case "$driver" in
/*) ;;
*) driver="$PWD/$driver" ;;
esac

dir=$(dirname "$source")
name=$(basename "$source" .ml)

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir "$work/compiler"

# The directory in which both compile, with the sources and nothing else.
files=
sources() {
  rm -rf "$work/unit"
  mkdir "$work/unit"
  cp "$source" "$work/unit/$name.ml"
  files="$name.ml"
  if [ -f "$dir/$name.mli" ]; then
    cp "$dir/$name.mli" "$work/unit/$name.mli"
    files="$name.mli $name.ml"
  fi
}

sources
if ! (cd "$work/unit" && "$ocamlopt" "$@" -c $files) \
  >"$work/compiler.log" 2>&1; then
  cat "$work/compiler.log"
  echo "$name.ml, compiled with [$*]: the compiler failed"
  exit 1
fi
for kind in o cmx cmi; do
  mv "$work/unit/$name.$kind" "$work/compiler/$name.$kind"
done

sources
status=0
(cd "$work/unit" && "$driver" "$@" -c $files) >"$work/driver.log" 2>&1 ||
  status=$?
if [ "$status" -gt 1 ]; then
  cat "$work/driver.log"
  echo "$name.ml, compiled with [$*]: the driver exits with $status"
  exit 1
fi

differ=
for kind in o cmx cmi; do
  if ! cmp -s "$work/compiler/$name.$kind" "$work/unit/$name.$kind"; then
    differ="$differ $name.$kind"
  fi
done
if [ -n "$differ" ]; then
  echo "$name.ml, compiled with [$*]: the driver does not write what the"
  echo "compiler writes:$differ"
  exit 1
fi
