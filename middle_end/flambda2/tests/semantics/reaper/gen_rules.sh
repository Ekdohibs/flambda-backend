#!/bin/sh

# Writes the dune rules for the examples of this directory, for run/dune to
# include (see dune). The arguments are the .fl and the .expected files that
# are here. The rules are for run/, so they name the files here with ../ .

set -eu

examples=" "
for file in "$@"; do
  case "$file" in
  *.fl) examples="$examples$(basename "$file" .fl) " ;;
  esac
done

for name in $examples; do
  cat <<RULES
(rule
 (with-stdout-to
  $name.output
  (run ../../../tools/flreaper.exe %{dep:../$name.fl})))

(rule
 (alias runtest)
 (action
  (diff ../$name.expected $name.output)))

RULES
done

# An expected file whose example is gone would never be looked at.
for file in "$@"; do
  case "$file" in
  *.expected)
    name=$(basename "$file" .expected)
    case "$examples" in
    *" $name "*) ;;
    *)
      cat <<RULES
(rule
 (alias runtest)
 (action
  (progn
   (echo "$name.expected has no $name.fl beside it\n")
   (run false))))

RULES
      ;;
    esac
    ;;
  esac
done
