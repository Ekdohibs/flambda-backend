#!/usr/bin/env bash

autoconf
./configure --enable-middle-end=flambda2
make runtime-stdlib
make minimizer

OCAMLLIB=_build/install/runtime_stdlib/lib/ocaml_runtime_stdlib/
export OCAMLLIB


COMPILE="_build/_bootinstall/bin/ocamlopt.opt -c -Oclassic"
INPUT=ocaml/testsuite/tests/backtrace/inline_traversal_test.ml
OUTPUT=minimized.ml

# Fails
$COMPILE $INPUT 2>/dev/null
echo $?

# Minimize exemple
_build/default/chamelon/chamelon.exe -c "$COMPILE" $INPUT -o $OUTPUT

# Still fails
$COMPILE $OUTPUT 2>/dev/null
echo $?

