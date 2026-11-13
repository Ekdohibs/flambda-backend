#!/bin/sh

# Writes the dune rules for the examples of this directory, for run/dune to
# include (see dune). The arguments are the .fl and the .expected files that
# are here. The rules are for run/, so they name the files here with ../ .
# An example whose first line is "(* flreaper: FLAGS *)" is run with FLAGS,
# options of the compiler, before its name.

set -eu

examples=" "
for file in "$@"; do
  case "$file" in
  *.fl) examples="$examples$(basename "$file" .fl) " ;;
  esac
done

flags_of () {
  for file in "$@"; do
    case "$file" in
    */"$name.fl" | "$name.fl")
      sed -n '1s/^(\* flreaper: \(.*\) \*)$/\1 /p' "$file" ;;
    esac
  done
}

for name in $examples; do
  flags=$(flags_of "$@")
  cat <<RULES
(rule
 (with-stdout-to
  $name.output
  (run ../../../tools/flreaper.exe $flags%{dep:../$name.fl})))

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
