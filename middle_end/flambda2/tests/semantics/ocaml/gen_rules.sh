#!/bin/sh

# Writes the dune rules for the OCaml examples of this directory, for
# run/dune to include (see dune). The arguments are the .ml, .mli, .fl,
# .expected and .own_expected files that are here, tool_fails.txt and
# own_run_numbers.txt. The rules are for run/, so they name the files here
# with ../ .

set -eu

examples=" "
interfaces=" "
own_expected=" "
failing=" "
numbers=" "
for file in "$@"; do
  case "$file" in
  *.mli) interfaces="$interfaces$(basename "$file" .mli) " ;;
  *.ml) examples="$examples$(basename "$file" .ml) " ;;
  *.own_expected)
    own_expected="$own_expected$(basename "$file" .own_expected) "
    ;;
  tool_fails.txt | */tool_fails.txt)
    # The units on which the tool is expected to fail: the lines of the list
    # that are not comments. (dune gives the name with ./ before it.)
    failing="$failing$(grep -v '^#' "$file" | tr '\n' ' ') "
    ;;
  own_run_numbers.txt | */own_run_numbers.txt)
    # The units whose own run is compared with X.expected as own_same.sh
    # compares when it is given "numbers": the lines of the list that are
    # not comments.
    numbers="$numbers$(grep -v '^#' "$file" | tr '\n' ' ') "
    ;;
  esac
done

# The rule that makes a term gives the compiler to the script as an
# argument, and that alone does not make dune run the rule again when
# the compiler changes: so the rule names the compiler among what it
# depends on, and a term is made again, and compared, after a change of
# the compiler that touches no file here.
for name in $examples; do
  interface=
  case "$interfaces" in
  *" $name "*) interface=" %{dep:../$name.mli}" ;;
  esac
  # The tool exits with the status 1 where it fails on a unit: the rule takes
  # that status, and no other, of a unit that tool_fails.txt names.
  run="(run ../../../tools/flsemantics.exe %{dep:../$name.fl})"
  case "$failing" in
  *" $name "*)
    run="(with-accepted-exit-codes
   1
   $run)"
    ;;
  esac
  cat <<RULES
(rule
 (targets $name.fl.made)
 (deps %{ocamlopt} %{dep:../generic_is_int.txt}$interface)
 (action
  (run sh %{dep:../from_ocaml.sh} %{ocamlopt} %{dep:../$name.ml}
   $name.fl.made)))

(rule
 (alias runtest)
 (action
  (diff ../$name.fl $name.fl.made)))

(rule
 (with-stdout-to
  $name.output
  $run))

(rule
 (alias runtest)
 (action
  (diff ../$name.expected $name.output)))

RULES
  # The compiler's own run of the unit (README.md, "The compiler's own
  # run"): the driver compiles X.ml, and what it prints must be what
  # X.expected has, the file that the tool on the text is compared with. The
  # driver is built in the main context only, so the rules are for that
  # context. It exits with the status 1 where the tool does: the rule takes
  # that status, and no other, of a unit that tool_fails.txt names.
  own="(run sh %{dep:../own_run.sh}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe} %{dep:../$name.ml})"
  case "$failing" in
  *" $name "*)
    own="(with-accepted-exit-codes
    1
    $own)"
    ;;
  esac
  # With what the own run is compared, and how. A unit that has
  # X.own_expected: with that file. A unit that own_run_numbers.txt names:
  # with X.expected, as own_same.sh compares when it is given "numbers"
  # (one line of each, in which the words that it takes for variables are
  # compared without their numbers). Any other unit: with X.expected, byte
  # for byte.
  case "$own_expected" in
  *" $name "*) same="(diff ../$name.own_expected $name.own_output)" ;;
  *)
    how=
    case "$numbers" in
    *" $name "*) how=" numbers" ;;
    esac
    same="(run sh %{dep:../own_same.sh}$how %{dep:../$name.expected}
   %{dep:$name.own_output})"
    ;;
  esac
  cat <<RULES
(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps$interface)
 (action
  (with-stdout-to
   $name.own_output
   $own)))

(rule
 (alias runtest)
 (enabled_if
  (= %{context_name} "main"))
 (action
  $same))

RULES
done

# A term or an expected file whose source is gone would never be looked at.
for file in "$@"; do
  case "$file" in
  *.fl | *.expected | *.own_expected | *.mli)
    name=$(basename "$file")
    name=${name%.*}
    case "$examples" in
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

# A name of tool_fails.txt that is of no unit here would never be looked at.
for name in $failing; do
  case "$examples" in
  *" $name "*) ;;
  *)
    cat <<RULES
(rule
 (alias runtest)
 (action
  (progn
   (echo "tool_fails.txt names $name, and no $name.ml is beside it\n")
   (run false))))

RULES
    ;;
  esac
done

# Nor would a name of own_run_numbers.txt that is of no unit here; and the
# list would say nothing of a unit that has X.own_expected, whose own run is
# compared with that file.
for name in $numbers; do
  what=
  case "$examples" in
  *" $name "*) ;;
  *) what="and no $name.ml is beside it" ;;
  esac
  case "$own_expected" in
  *" $name "*)
    what="and its own run is compared with $name.own_expected"
    ;;
  esac
  if [ -n "$what" ]; then
    cat <<RULES
(rule
 (alias runtest)
 (action
  (progn
   (echo "own_run_numbers.txt names $name, $what\n")
   (run false))))

RULES
  fi
done
