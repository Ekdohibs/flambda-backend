#!/usr/bin/env bash

eval $(opam env)
if [ ! -f Makefile.config ]; then
  autoconf
  ./configure --enable-middle-end=flambda2 --enable-runtime=runtime4
fi
make
make minimizer

OCAMLLIB=_build/install/runtime_stdlib/lib/ocaml_runtime_stdlib/
export OCAMLLIB


FLAGS=( "-Oclassic" "-O3" "" "" "-principal -I _build/main/ocaml/.ocamlcommon.objs/byte -I _build/main/ocaml/.ocamlcommon.objs/native -dflambda" "-principal -I _build/main/ocaml/.ocamlcommon.objs/byte -I _build/main/ocaml/.ocamlcommon.objs/native -dflambda" )
CHAMELON_FLAGS=( "" "" "" "" "-e Defining_expr_of_let" "-e Defining_expr_of_let")
INPUTS=( ocaml/testsuite/tests/backtrace/inline_traversal_test.ml binary_packing.ml letrec.ml seqtest.ml ocaml/typing/env.ml env.ml )
OUTPUTS=( inline_traversal_test_min.ml binary_packing_min.ml letrec_min.ml seqtest_min.ml env.ml env_min.ml )
EXTRACOMMAND=( "" "" "" "" "cp ocaml/typing/env.mli env.mli" "" )

ITERATE=${1:-${!FLAGS[@]}}

for idx in $ITERATE; do
  COMPILE="_build/_bootinstall/bin/ocamlopt.opt -c ${FLAGS[$idx]}"
  INPUT=${INPUTS[$idx]}
  OUTPUT=${OUTPUTS[$idx]}

  ${EXTRACOMMAND[$idx]}

  # Fails
  $COMPILE $INPUT 2>/dev/null
  echo $?

  # Minimize exemple
  _build/default/chamelon/chamelon.exe -c "$COMPILE" $INPUT -o $OUTPUT ${CHAMELON_FLAGS[$idx]}

  # Still fails
  $COMPILE $OUTPUT 2>/dev/null
  echo $?

done


for idx in $ITERATE; do
  INPUT=${INPUTS[$idx]}
  OUTPUT=${OUTPUTS[$idx]}

  echo "Input: $INPUT, $(wc -l < $INPUT) lines"
  echo "Output: $OUTPUT, $(wc -l < $OUTPUT) lines"
  echo ""
done
