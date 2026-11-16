#!/bin/sh

# Writes the dune rules for the pairs of units of this directory, for
# run/dune to include (see dune). The arguments are the .ml, .fl and
# .expected files that are here. The rules are for run/, so they name the
# files here with ../ .

set -eu

units=" "
checked=" "
here=.
for file in "$@"; do
  case "$file" in
  *.ml)
    units="$units$(basename "$file" .ml) "
    here=$(dirname "$file")
    ;;
  *.expected) checked="$checked$(basename "$file" .expected) " ;;
  esac
done

# The first unit of a unit: the longest name of another unit that, followed
# by an underscore, begins its name. Nothing for a unit that has none.
first_of() {
  found=
  for candidate in $units; do
    case "$1" in
    "$candidate"_*)
      if [ ${#candidate} -gt ${#found} ]; then found=$candidate; fi
      ;;
    esac
  done
  echo "$found"
}

# The flags that a source gives for itself, on a line "(* flags: ... *)":
# from_ocaml.sh compiles the unit with them, and the tool is given them too.
flags_of() {
  sed -n 's/^(\* flags: \(.*\) \*)$/\1/p' "$here/$1.ml" | head -n 1
}

# Whether a source asks, on a line "(* load path: the first unit alone *)",
# that the tool be given the .cmx of its first unit and of no other unit:
# the tool is then run by tool_alone.sh.
alone_of() {
  grep -c '^(\* load path: the first unit alone \*)$' "$here/$1.ml" || true
}

# The rule that makes a term gives the compiler to the script as an
# argument, and that alone does not make dune run the rule again when
# the compiler changes: so the rule names the compiler among what it
# depends on, and a term is made again, and compared, after a change of
# the compiler that touches no file here.
for name in $units; do
  first=$(first_of "$name")
  flags=$(flags_of "$name")
  if [ -n "$flags" ]; then flags="$flags "; fi
  needs=
  against=
  # from_ocaml.sh reads the list of the units that may have the test of
  # being an integer that is for any value, and looks whether the unit has
  # an expected file.
  made_needs="../generic_is_int.txt"
  case "$checked" in
  *" $name "*) made_needs="$made_needs ../$name.expected" ;;
  esac
  if [ -n "$first" ]; then
    # The tool is given the directory of the build as its load path, where
    # it finds the .cmx of the first unit, and of the first unit of that one
    # in turn: all of them are needed.
    needs="
 (deps $first.cmi $first.cmx"
    before=$(first_of "$first")
    while [ -n "$before" ]; do
      needs="$needs $before.cmi $before.cmx"
      before=$(first_of "$before")
    done
    needs="$needs)"
    against=" $first"
    made_needs="$made_needs $first.cmi $first.cmx"
  fi
  tool="(run ../../../../tools/flsemantics.exe $flags-I . %{dep:../$name.fl})"
  if [ -n "$first" ] && [ "$(alone_of "$name")" != 0 ]; then
    needs="
 (deps $first.cmi $first.cmx)"
    tool="(run sh %{dep:../tool_alone.sh}
    %{dep:../../../../tools/flsemantics.exe} $first.cmx
    $flags%{dep:../$name.fl})"
  fi
  cat <<RULES
(rule
 (targets $name.fl.made $name.cmi $name.cmx)
 (deps %{ocamlopt} $made_needs)
 (action
  (run sh %{dep:../from_ocaml.sh} %{ocamlopt} %{dep:../$name.ml}
   $name$against)))

(rule
 (alias runtest)
 (action
  (diff ../$name.fl $name.fl.made)))

RULES
  # A unit with no expected file is compiled, for the unit that uses it,
  # and its term is compared; the tool is not run on it.
  case "$checked" in
  *" $name "*)
    cat <<RULES
(rule$needs
 (action
  (with-stdout-to
   $name.output
   $tool)))

(rule
 (alias runtest)
 (action
  (diff ../$name.expected $name.output)))

RULES
    ;;
  esac
done

# A term or an expected file whose source is gone would never be looked at.
for file in "$@"; do
  case "$file" in
  *.fl | *.expected)
    name=$(basename "$file")
    name=${name%.*}
    case "$units" in
    *" $name "*) ;;
    *)
      cat <<RULES
(rule
 (alias runtest)
 (action
  (progn
   (echo "$(basename "$file") has no $name.ml beside it\n")
   (run false))))

RULES
      ;;
    esac
    ;;
  esac
done
