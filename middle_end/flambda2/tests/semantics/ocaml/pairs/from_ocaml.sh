#!/bin/sh

# Compiles a unit of a pair as far as the middle end, and keeps what a second
# unit and the test tool need of it (see README.md):
#
#   from_ocaml.sh <ocamlopt> <X.ml> <X> [<F>]
#
# Made in the current directory: X.fl.made, the term that the compiler gives
# to Simplify, as -drawfexpr prints it; and X.cmi and X.cmx, as the compiler
# writes them where it prints nothing (see below). With <F>, the unit is
# compiled against F.cmi and F.cmx, which are in the current directory.
#
# As in the directory above, the unit is compiled with no standard library
# and no initially opened module, and the compilation stops after the middle
# end. A source may give more flags for its own compilation, on a line of its
# own:
#
#   (* flags: -g *)
#
# As in the directory above, no unit is refused for what its term has.

set -eu

ocamlopt=$1
source=$2
name=$3
first=${4:-}

case "$ocamlopt" in
/*) ;;
*) ocamlopt="$PWD/$ocamlopt" ;;
esac

flags=$(sed -n 's/^(\* flags: \(.*\) \*)$/\1/p' "$source" | head -n 1)

# In a directory of its own, and under the name X.ml, as the names of
# anonymous functions have the name of the file as it is given. The compiler
# looks for the .cmi and the .cmx of the first unit in the directory it is
# run in.
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cp "$source" "$work/$name.ml"
if [ -n "$first" ]; then
  cp "$first.cmi" "$first.cmx" "$work/"
fi
(cd "$work" &&
  "$ocamlopt" -nopervasives -nostdlib $flags -stop-after middle_end -c \
    "$name.ml" -drawfexpr-to "$name.fl")

# The unit is compiled once more, with no option that prints, and X.cmi and
# X.cmx are those of this second compilation. The compilation that prints the
# text gives other numbers to the names that the compiler makes after it has
# printed: a symbol that Simplify lifts then has another name in the .cmx
# than the one that the compiler gives it where it prints nothing, and that
# name is in what the tool prints of a unit that uses this one.
(cd "$work" &&
  "$ocamlopt" -nopervasives -nostdlib $flags -stop-after middle_end -c \
    "$name.ml")

cp "$work/$name.fl" "$name.fl.made"
cp "$work/$name.cmi" "$work/$name.cmx" .
