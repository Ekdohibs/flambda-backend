#!/bin/sh

# Makes of an OCaml source the term that the compiler gives to Simplify, in
# the syntax that the test tool reads (see README.md):
#
#   from_ocaml.sh <ocamlopt> <X.ml> <target>
#
# The unit is compiled with no standard library and no initially opened
# module, so that it needs nothing but itself, and with X.mli if that is
# beside it. The compilation stops after the middle end: what is printed is
# what -drawfexpr prints, the unit as it is after the conversion from Lambda
# and before Simplify.
#
#   from_ocaml.sh <ocamlopt> <X.ml> <target> <directory>
#
# The same, for a source written against the standard library (library/):
# the unit is compiled with the standard library that is in the directory
# given, and with Stdlib opened as the compiler opens it. The unit is refused
# if its term names a symbol of another unit or a function of the runtime
# (see the end of this file), so the term that is kept needs nothing but
# itself here too: what the source takes from the library is then what the
# type checker and the conversion from Lambda put in the term, primitives
# above all.
#
# A unit is refused for two things in its term (see below): the test of being
# an integer that is for any value, unless generic_is_int.txt names the unit;
# a call that is a probe. The text of a term lost both when the refusals were
# written, and writes both now; the refusals are kept.

set -eu

ocamlopt=$1
source=$2
target=$3
library=${4-}

case "$ocamlopt" in
/*) ;;
*) ocamlopt="$PWD/$ocamlopt" ;;
esac

# With no directory, no standard library and no module opened. With one, that
# directory in the place of the compiler's own standard library, so that the
# library is the one named and no other. The options are kept as the
# arguments of this script from here on, the directory as one argument
# whatever its name has in it.
case "$library" in
"") set -- -nopervasives -nostdlib ;;
/*) set -- -nostdlib -I "$library" ;;
*) set -- -nostdlib -I "$PWD/$library" ;;
esac

dir=$(dirname "$source")
name=$(basename "$source" .ml)

# In a directory of its own, as the compiler writes X.cmi and X.cmx too, and
# under the name X.ml, as the names of anonymous functions have the name of
# the file as it is given.
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cp "$source" "$work/$name.ml"
interface=
if [ -f "$dir/$name.mli" ]; then
  cp "$dir/$name.mli" "$work/$name.mli"
  interface="$name.mli"
fi
(cd "$work" &&
  "$ocamlopt" "$@" -stop-after middle_end -c \
    $interface "$name.ml" -drawfexpr-to "$name.fl")

# The text wrote the two tests of being an integer alike, and its reader gave
# the one that is for variants (label C2-26); the text now writes the one that
# is for any value apart, and the reader takes it. The unit is still compiled
# once more, for what -drawflambda prints of the term before Simplify, which
# is kept in a file; and a unit whose term has the test that is for any value,
# printed as Is_int_generic, is still refused unless generic_is_int.txt,
# beside the source, names it. Nothing is given back of a unit that is
# refused, or whose second compilation fails or prints no term.
if ! (cd "$work" &&
  "$ocamlopt" "$@" -stop-after middle_end -c \
    $interface "$name.ml" -color never -drawflambda >"$name.raw" 2>&1); then
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
# "(probe ())" in any other. A unit whose term has one is refused.
if grep -q '(probe [^(]' "$work/$name.line"; then
  echo "$name.ml: the term that the compiler gives to Simplify has a call" >&2
  echo "that is a probe. The program runs the handler of a probe only" >&2
  echo "while the probe is enabled, and the model has nothing of that:" >&2
  echo "the tool does not check a unit that has one." >&2
  exit 1
fi

# A unit compiled with the standard library may name a symbol of a unit of
# the library: a function or a value that the library defines. Its term then
# needs that unit, and the tool would have to be given the .cmx file of the
# library; such a unit is refused here. The text writes a symbol of another
# unit with the name of that unit and a dot before it (and the linkage name
# of the unit after a slash, where it has one), and a symbol of the unit
# itself with neither. Each of those names is written as it is if it is made
# of letters, digits, underscores and quotes, and between backquotes if not
# (print_fexpr.ml, symbol0 and symbol_part). A predefined exception is
# written with the name `*predef*`, is not of the library, and is let
# through. A function of the runtime is written with the name `*extern*`,
# and is refused with the units: the tool reads no term that calls one.
#
# This is a test on the text of the term and no parser: it looks for a
# dollar followed by such a name and a dot, wherever that is. So it may
# refuse a unit for a string constant that reads so; it lets through no unit
# whose term has such a symbol.
if [ -n "$library" ]; then
  part='\(`[^`]*`\|[A-Za-z0-9_'"'"']\{1,\}\)'
  others=$(grep -o '\$'"$part"'\(/'"$part"'\)\{0,1\}\.' "$work/$name.fl" |
    grep -v -x -F '$`*predef*`.' | sort -u || true)
  if [ -n "$others" ]; then
    echo "$name.ml: the term that the compiler gives to Simplify names a" >&2
    echo "symbol of another unit, or of the runtime:" >&2
    echo "$others" | sed 's/\.$//; s/^/  /' >&2
    echo "The units kept here need nothing but themselves: a unit that" >&2
    echo "uses a function or a value that the standard library defines, or" >&2
    echo "a function of the runtime, is not one of them." >&2
    exit 1
  fi
fi

cp "$work/$name.fl" "$target"
