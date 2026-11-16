#!/bin/sh

# Writes the dune rules for the OCaml examples of this directory, for
# run/dune to include (see dune). The arguments are the .ml, .mli, .fl,
# .expected and .own_expected files that are here. The rules are for run/,
# so they name the files here with ../ .

set -eu

examples=" "
interfaces=" "
own_runs=" "
for file in "$@"; do
  case "$file" in
  *.mli) interfaces="$interfaces$(basename "$file" .mli) " ;;
  *.ml) examples="$examples$(basename "$file" .ml) " ;;
  *.own_expected) own_runs="$own_runs$(basename "$file" .own_expected) " ;;
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
  (run ../../../tools/flsemantics.exe %{dep:../$name.fl})))

(rule
 (alias runtest)
 (action
  (diff ../$name.expected $name.output)))

RULES
  # The compiler's own run of the unit, for a unit that has X.own_expected
  # (README.md, "The compiler's own run"). The driver is built in the main
  # context only, so the rules are for that context.
  case "$own_runs" in
  *" $name "*)
    cat <<RULES
(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps$interface)
 (action
  (with-stdout-to
   $name.own_output
   (run sh %{dep:../own_run.sh}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe} %{dep:../$name.ml}))))

(rule
 (alias runtest)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../$name.own_expected $name.own_output)))

RULES
    ;;
  esac
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
