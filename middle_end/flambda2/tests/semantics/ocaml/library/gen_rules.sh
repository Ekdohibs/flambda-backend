#!/bin/sh

# Writes the dune rules for the OCaml examples of this directory, for
# run/dune to include (see dune). The arguments are the .ml, .mli, .fl and
# .expected files that are here, and tool_fails.txt. The rules are for run/,
# so they name the files here with ../ .
#
# The sources are compiled with the standard library that the compiler of
# the context finds (%{ocaml_where}), and so, in the context in which the
# tests are run, with the one that the build of this tree made with its own
# compiler; in another context it would be that of another compiler. So the
# two rules that make a term again and compare it with the one kept are for
# the context main only. A term depends on the interfaces of the library,
# and is made again when one of them changes. The two rules that run the
# tool on the term kept and compare what it prints are for every context, as
# those of the directory above are: they need no library.

set -eu

examples=" "
interfaces=" "
failing=" "
for file in "$@"; do
  case "$file" in
  *.mli) interfaces="$interfaces$(basename "$file" .mli) " ;;
  *.ml) examples="$examples$(basename "$file" .ml) " ;;
  tool_fails.txt | */tool_fails.txt)
    # The units on which the tool is expected to fail: the lines of the list
    # that are not comments. (dune gives the name with ./ before it.)
    failing="$failing$(grep -v '^#' "$file" | tr '\n' ' ') "
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
  run="(run ../../../../tools/flsemantics.exe %{dep:../$name.fl})"
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
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)$interface)
 (action
  (run sh %{dep:../../from_ocaml.sh} %{ocamlopt} %{dep:../$name.ml}
   $name.fl.made %{ocaml_where})))

(rule
 (alias runtest)
 (enabled_if
  (= %{context_name} "main"))
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
done

# A term or an expected file whose source is gone would never be looked at.
for file in "$@"; do
  case "$file" in
  *.fl | *.expected | *.mli)
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
