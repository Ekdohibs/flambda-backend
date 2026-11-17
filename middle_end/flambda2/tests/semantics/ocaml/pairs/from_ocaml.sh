#!/bin/sh

# Compiles a unit of a pair as far as the middle end, and keeps what a second
# unit and the test tool need of it (see README.md):
#
#   from_ocaml.sh <ocamlopt> <X.ml> <X> [<F>]
#
# Made in the current directory: X.fl.made, the term that the compiler gives
# to Simplify, as -drawfexpr prints it; and X.cmi and X.cmx, which the
# compiler writes. With <F>, the unit is compiled against F.cmi and F.cmx,
# which are in the current directory.
#
# As in the directory above, the unit is compiled with no standard library
# and no initially opened module, and the compilation stops after the middle
# end. A source may give more flags for its own compilation, on a line of its
# own:
#
#   (* flags: -g *)
#
# As in the directory above, a unit is refused for the test of being an
# integer that is for any value, or for a call that is a probe (see below);
# the text of a term lost both when the refusals were written, and writes
# both now. The refusal for a probe is of a unit that the tool is run on, one
# that has X.expected beside its source: a unit that has none is compiled for
# the unit that uses it, and no verdict is given of its text.

set -eu

ocamlopt=$1
source=$2
name=$3
first=${4:-}

case "$ocamlopt" in
/*) ;;
*) ocamlopt="$PWD/$ocamlopt" ;;
esac

dir=$(dirname "$source")

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

# The text wrote the two tests of being an integer alike, and its reader gave
# the one that is for variants (label C2-26); the text now writes the one that
# is for any value apart, and the reader takes it. The unit is still compiled
# once more, for what -drawflambda prints of the term before Simplify, which
# is kept in a file; and a unit whose term has the test that is for any value,
# printed as Is_int_generic, is still refused unless generic_is_int.txt,
# beside the source, names it. Nothing is given back of a unit that is
# refused, or whose second compilation fails or prints no term.
if ! (cd "$work" &&
  "$ocamlopt" -nopervasives -nostdlib $flags -stop-after middle_end -c \
    "$name.ml" -color never -drawflambda >"$name.raw" 2>&1); then
  cat "$work/$name.raw" >&2
  echo "$name.ml: the compilation that prints the term failed" >&2
  exit 1
fi
if ! grep -q 'module_symbol' "$work/$name.raw"; then
  echo "$name.ml: -drawflambda printed no term" >&2
  exit 1
fi
if grep -q 'Is_int_generic' "$work/$name.raw"; then
  if ! grep -qx "$name" "$dir/generic_is_int.txt"; then
    echo "$name.ml: the term that the compiler gives to Simplify has" >&2
    echo "Is_int_generic, the test of being an integer that is for any" >&2
    echo "value. The text wrote it as the test that is for variants, and" >&2
    echo "writes it apart now; this script still refuses a unit that has" >&2
    echo "it. generic_is_int.txt says how such a unit may be kept." >&2
    exit 1
  fi
fi

# What the compiler printed, as one line with single spaces: it cuts what it
# prints into lines by their width.
tr '\n' ' ' <"$work/$name.raw" | tr -s ' ' >"$work/$name.line"

# The program runs the handler of a probe only while the probe is enabled,
# and the model has nothing of that: the tool does not check a unit that has
# a call which is a probe. (The text did not say that a call is one, and the
# call was read as a plain one, label C2-17; it says so now, and the reader
# takes it.) The compiler prints "(probe NAME)" in a call that is one and
# "(probe ())" in any other. A unit whose term has one is refused if the tool
# is run on it.
if [ -f "$dir/$name.expected" ] &&
  grep -q '(probe [^(]' "$work/$name.line"; then
  echo "$name.ml: the term that the compiler gives to Simplify has a call" >&2
  echo "that is a probe. The program runs the handler of a probe only" >&2
  echo "while the probe is enabled, and the model has nothing of that:" >&2
  echo "the tool does not check a unit that has one." >&2
  exit 1
fi

cp "$work/$name.fl" "$name.fl.made"
cp "$work/$name.cmi" "$work/$name.cmx" .
