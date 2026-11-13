#!/bin/sh

# Writes the dune rules that take each example through fexpr and back, for
# run/dune to include (see dune). The arguments are the .fl files of the
# directory above, of ../ocaml and of ../ocaml/library, and the .expected
# files that are here. The rules are for run/, so they name the files with
# one more ../ .
#
# An example ../X.fl has the name X, ../ocaml/X.fl the name ocaml_X and
# ../ocaml/library/X.fl the name ocaml_library_X. If
# NAME.expected is here, what the tool prints is compared with it. If not,
# the tool must print that the round trip is the same, or that the example is
# not converted (check.sh).

set -eu

names=" "
expected=" "
for file in "$@"; do
  case "$file" in
  *.expected) expected="$expected$(basename "$file" .expected) " ;;
  esac
done

for file in "$@"; do
  # A star of a pattern takes a slash too, so the longer directory is first.
  case "$file" in
  ../ocaml/library/*.fl) name="ocaml_library_$(basename "$file" .fl)" ;;
  ../ocaml/*.fl) name="ocaml_$(basename "$file" .fl)" ;;
  ../*.fl) name="$(basename "$file" .fl)" ;;
  *) continue ;;
  esac
  names="$names$name "
  cat <<RULES
(rule
 (with-stdout-to
  $name.output
  (run ../../../tools/flsemantics.exe -round-trip %{dep:../$file})))

RULES
  case "$expected" in
  *" $name "*)
    cat <<RULES
(rule
 (alias runtest)
 (action
  (diff ../$name.expected $name.output)))

RULES
    ;;
  *)
    cat <<RULES
(rule
 (alias runtest)
 (action
  (run sh %{dep:../check.sh} %{dep:$name.output})))

RULES
    ;;
  esac
done

# An expected file whose example is gone would never be looked at.
for name in $expected; do
  case "$names" in
  *" $name "*) ;;
  *)
    cat <<RULES
(rule
 (alias runtest)
 (action
  (progn
   (echo "roundtrip/$name.expected has no example\n")
   (run false))))

RULES
    ;;
  esac
done
