#!/bin/sh
# One file of the standard library, compiled as the build of the tree
# compiles it, by the compiler that built the library and then by the driver
# that checks the compiler's own run (../../tools/ownrun/flsemantics_ownrun.ml);
# and what the two write and print set beside each other and beside the
# build's own files. See README.md.
#
#   stdlib_run.sh <ocamlopt> <driver> <objinfo> <where> <source> <form>
#     <unit> <status> <awk> <file>... -- <word>...
#
# <ocamlopt> is the compiler that built the library, and <where> the directory
# in which it built it. <source> is the real file stdlib/X.ml, and <form> says
# how the build gives it to the compiler: "numbered", after a line that gives
# its name and the number 2, or "copied", as it is. <unit> is the name that the
# build gives the unit (stdlib__List). <status> is the status with which the
# driver must exit. <awk> is the script that the build's command gives to -pp,
# or "-" if it gives none. Each <file> is a file of <where> that the build's
# rule for the unit depends on. The <word>s are those of the build's command,
# from its first flag to the name of the source.
#
# Where every comparison holds, what the driver says of the unit is printed
# on the standard output, and nothing else is: that is what the rule
# compares with the stored file. Where one fails, the script says which on
# the standard error and exits with 1; it prints nothing on the standard
# output then, but where the driver exits with another status than the one
# expected: there it prints what the driver said, for whoever looks for the
# cause, before it says so.

set -eu

fail() {
  echo "stdlib_run.sh: $unit: $1" >&2
  exit 1
}

unit="(no unit)"
if [ $# -lt 11 ]; then
  fail "usage: $0 <ocamlopt> <driver> <objinfo> <where> <source> <form> \
<unit> <status> <awk> <file>... -- <word>..."
fi

absolute() {
  case "$1" in
  /*) echo "$1" ;;
  *) echo "$PWD/$1" ;;
  esac
}

ocamlopt=$(absolute "$1")
driver=$(absolute "$2")
objinfo=$(absolute "$3")
where=$(absolute "$4")
source=$5
form=$6
unit=$7
status=$8
awk=$9
shift 9

name=$(basename "$source")

# A directory of its own, in which stdlib/build is where everything is
# compiled: the build compiles in a directory of that name under the root of
# its workspace, and with -g -absname the terms have the path of the source.
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
build="$work/stdlib/build"
mkdir -p "$build"

# The source, as the build gives it to the compiler. It must be, byte for
# byte, the copy of it in the directory where the library was built.
case "$form" in
numbered)
  {
    printf '# 2 "%s"\n' "$name"
    cat "$source"
  } >"$build/$name"
  ;;
copied) cp "$source" "$build/$name" ;;
*) fail "the form of the source is neither numbered nor copied: $form" ;;
esac
if ! cmp -s "$build/$name" "$where/$name"; then
  fail "$source, as the build gives it to the compiler, is not the copy of \
it in the directory where the library was built: the library was not built \
from it as it is now (build the compiler again)"
fi

# The files of the other units that the build's rule names, and no other.
while [ "$1" != "--" ]; do
  cp "$where/$1" "$build/$1"
  shift
  if [ $# -eq 0 ]; then
    fail "no -- between the files and the words of the command"
  fi
done
shift
if [ "$awk" != "-" ]; then
  cp "$awk" "$work/stdlib/$(basename "$awk")"
fi

# The environment of the build's action, where it shows in what is written:
# the prefix of the paths, which dune sets to the root of its workspace.
# What is left of "$@" is the build's command.
unset OCAMLPARAM OCAMLRUNPARAM
BUILD_PATH_PREFIX_MAP="/workspace_root=$work"
export BUILD_PATH_PREFIX_MAP

# The compiler that built the library, given the build's command and two
# flags that print the terms before and after Simplify. It must write the
# build's own .cmx and .o, byte for byte: the check is that this compilation,
# in this directory and with the two flags more, wrote what the build wrote.
if ! (cd "$build" &&
  "$ocamlopt" -drawflambda -dflambda "$@" \
    >"$work/boot.out" 2>"$work/boot.err"); then
  cat "$work/boot.err" >&2
  fail "the compiler that built the library fails on the build's command"
fi
if [ -s "$work/boot.out" ]; then
  fail "the compiler that built the library printed on its standard output"
fi
for suffix in cmx o; do
  if ! cmp -s "$build/$unit.$suffix" "$where/$unit.$suffix"; then
    fail "the compiler that built the library, given the command of this \
rule, does not write the bytes of the build's $unit.$suffix"
  fi
  rm "$build/$unit.$suffix"
done

# The driver, given the same command: the compilation goes to its end, and
# the check is made of its two terms.
got=0
(cd "$build" &&
  "$driver" -drawflambda -dflambda "$@" \
    >"$work/own.out" 2>"$work/own.err") || got=$?
if [ "$got" != "$status" ]; then
  cat "$work/own.out"
  fail "the driver exits with $got, and $status was expected \
(driver_status.txt)"
fi

# The two terms that the driver printed must be those that the compiler
# that built the library printed.
if ! cmp -s "$work/boot.err" "$work/own.err"; then
  fail "what the driver prints of the terms before and after Simplify is \
not what the compiler that built the library prints"
fi

# ocamlobjinfo must print the same of the driver's .cmx and of the build's
# own, each under the same name, but for the line of the CRC of the
# implementation, which is of the bytes. (The two files are not the same
# bytes for most units, and the print has not everything that a .cmx holds:
# README.md.)
for which in own built; do
  mkdir "$work/$which"
done
cp "$build/$unit.cmx" "$work/own/$unit.cmx"
cp "$where/$unit.cmx" "$work/built/$unit.cmx"
for which in own built; do
  if ! (cd "$work/$which" && "$objinfo" "$unit.cmx") \
    >"$work/$which.info" 2>&1; then
    fail "ocamlobjinfo fails on the $which $unit.cmx"
  fi
  if [ "$(grep -c '^CRC of implementation:' "$work/$which.info")" != 1 ]; then
    fail "what ocamlobjinfo prints of the $which $unit.cmx has not one \
line of the CRC of the implementation"
  fi
  if ! grep -v '^CRC of implementation:' "$work/$which.info" \
    >"$work/$which.rest"; then
    fail "ocamlobjinfo prints nothing of the $which $unit.cmx but the line \
of the CRC of the implementation"
  fi
done
if ! cmp -s "$work/own.rest" "$work/built.rest"; then
  fail "ocamlobjinfo does not print the same of the .cmx that the driver \
writes and of the build's own, the line of the CRC of the implementation \
left out: the driver's .cmx does not hold what the build's own holds"
fi

cat "$work/own.out"
