;; The rules of ../ for each unit of the standard library that the build of
;; the tree compiles to native code (../README.md). ../gen_rules.exe wrote
;; this file from the rules of the build. Each of the two aliases of the
;; directory compares it with what that program writes now, and `dune
;; promote` then takes the new rules: it is not to be changed by hand.

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/arg.mli
  ../../../../../../stdlib/array.ml
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/buffer.ml
  ../../../../../../stdlib/buffer.mli
  ../../../../../../stdlib/bytes.ml
  ../../../../../../stdlib/bytes.mli
  ../../../../../../stdlib/camlinternalFormat.ml
  ../../../../../../stdlib/camlinternalFormat.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/char.ml
  ../../../../../../stdlib/char.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/int.ml
  ../../../../../../stdlib/int.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.ml
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/printf.ml
  ../../../../../../stdlib/printf.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/string.ml
  ../../../../../../stdlib/string.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli
  ../../../../../../stdlib/uchar.ml
  ../../../../../../stdlib/uchar.mli)
 (action
  (with-stdout-to
   arg.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/arg.ml}
    numbered stdlib__Arg 0
    -
    camlinternalFormat.cmi camlinternalFormat.cmx
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Arg.cmi stdlib__Array.cmi stdlib__Array.cmx stdlib__Atomic.cmi
    stdlib__Atomic.cmx stdlib__Buffer.cmi stdlib__Buffer.cmx
    stdlib__Bytes.cmi stdlib__Bytes.cmx stdlib__Char.cmi stdlib__Char.cmx
    stdlib__Either.cmi stdlib__Either.cmx stdlib__Int.cmi stdlib__Int.cmx
    stdlib__Int32.cmi stdlib__Int32.cmx stdlib__Int64.cmi
    stdlib__Int64.cmx stdlib__Lazy.cmi stdlib__Lazy.cmx stdlib__List.cmi
    stdlib__List.cmx stdlib__Obj.cmi stdlib__Obj.cmx stdlib__Printf.cmi
    stdlib__Printf.cmx stdlib__Seq.cmi stdlib__Seq.cmx stdlib__String.cmi
    stdlib__String.cmx stdlib__Sys.cmi stdlib__Sys.cmx stdlib__Uchar.cmi
    stdlib__Uchar.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Arg.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Arg.cmx -c ./arg.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../arg.own_expected
   arg.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli)
 (action
  (with-stdout-to
   array.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/array.ml}
    numbered stdlib__Array 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Array.cmi stdlib__Atomic.cmi stdlib__Atomic.cmx
    stdlib__Either.cmi stdlib__Either.cmx stdlib__Int32.cmi
    stdlib__Int32.cmx stdlib__Int64.cmi stdlib__Int64.cmx stdlib__Lazy.cmi
    stdlib__Lazy.cmx stdlib__Obj.cmi stdlib__Obj.cmx stdlib__Seq.cmi
    stdlib__Seq.cmx stdlib__Sys.cmi stdlib__Sys.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Array.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Array.cmx -c ./array.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../array.own_expected
   array.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/array.ml
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/arrayLabels.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli)
 (action
  (with-stdout-to
   arrayLabels.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/arrayLabels.ml}
    numbered stdlib__ArrayLabels 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Array.cmi stdlib__Array.cmx stdlib__ArrayLabels.cmi
    stdlib__Atomic.cmi stdlib__Atomic.cmx stdlib__Either.cmi
    stdlib__Either.cmx stdlib__Int32.cmi stdlib__Int32.cmx
    stdlib__Int64.cmi stdlib__Int64.cmx stdlib__Lazy.cmi stdlib__Lazy.cmx
    stdlib__Obj.cmi stdlib__Obj.cmx stdlib__Seq.cmi stdlib__Seq.cmx
    stdlib__Sys.cmi stdlib__Sys.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__ArrayLabels.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__ArrayLabels.cmx -c ./arrayLabels.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../arrayLabels.own_expected
   arrayLabels.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/atomic.mli)
 (action
  (with-stdout-to
   atomic.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/atomic.ml}
    numbered stdlib__Atomic 0
    -
    stdlib__Atomic.cmi
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Atomic.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Atomic.cmx -c ./atomic.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../atomic.own_expected
   atomic.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/array.ml
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/backoff.mli
  ../../../../../../stdlib/bigarray.ml
  ../../../../../../stdlib/bigarray.mli
  ../../../../../../stdlib/bool.ml
  ../../../../../../stdlib/bool.mli
  ../../../../../../stdlib/buffer.ml
  ../../../../../../stdlib/buffer.mli
  ../../../../../../stdlib/bytes.ml
  ../../../../../../stdlib/bytes.mli
  ../../../../../../stdlib/callback.ml
  ../../../../../../stdlib/callback.mli
  ../../../../../../stdlib/camlinternalFormat.ml
  ../../../../../../stdlib/camlinternalFormat.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/char.ml
  ../../../../../../stdlib/char.mli
  ../../../../../../stdlib/complex.ml
  ../../../../../../stdlib/complex.mli
  ../../../../../../stdlib/condition.ml
  ../../../../../../stdlib/condition.mli
  ../../../../../../stdlib/digest.ml
  ../../../../../../stdlib/digest.mli
  ../../../../../../stdlib/domain.ml
  ../../../../../../stdlib/domain.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/float.ml
  ../../../../../../stdlib/float.mli
  ../../../../../../stdlib/fun.ml
  ../../../../../../stdlib/fun.mli
  ../../../../../../stdlib/in_channel.ml
  ../../../../../../stdlib/in_channel.mli
  ../../../../../../stdlib/int.ml
  ../../../../../../stdlib/int.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.ml
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/map.ml
  ../../../../../../stdlib/map.mli
  ../../../../../../stdlib/modes.ml
  ../../../../../../stdlib/modes.mli
  ../../../../../../stdlib/mutex.ml
  ../../../../../../stdlib/mutex.mli
  ../../../../../../stdlib/nativeint.ml
  ../../../../../../stdlib/nativeint.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/printexc.ml
  ../../../../../../stdlib/printexc.mli
  ../../../../../../stdlib/printf.ml
  ../../../../../../stdlib/printf.mli
  ../../../../../../stdlib/random.ml
  ../../../../../../stdlib/random.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/string.ml
  ../../../../../../stdlib/string.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli
  ../../../../../../stdlib/uchar.ml
  ../../../../../../stdlib/uchar.mli)
 (action
  (with-stdout-to
   backoff.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/backoff.ml}
    numbered stdlib__Backoff 0
    -
    camlinternalFormat.cmi camlinternalFormat.cmx
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Array.cmi stdlib__Array.cmx stdlib__Atomic.cmi
    stdlib__Atomic.cmx stdlib__Backoff.cmi stdlib__Bigarray.cmi
    stdlib__Bigarray.cmx stdlib__Bool.cmi stdlib__Bool.cmx
    stdlib__Buffer.cmi stdlib__Buffer.cmx stdlib__Bytes.cmi
    stdlib__Bytes.cmx stdlib__Callback.cmi stdlib__Callback.cmx
    stdlib__Char.cmi stdlib__Char.cmx stdlib__Complex.cmi
    stdlib__Complex.cmx stdlib__Condition.cmi stdlib__Condition.cmx
    stdlib__Digest.cmi stdlib__Digest.cmx stdlib__Domain.cmi
    stdlib__Domain.cmx stdlib__Either.cmi stdlib__Either.cmx
    stdlib__Float.cmi stdlib__Float.cmx stdlib__Fun.cmi stdlib__Fun.cmx
    stdlib__In_channel.cmi stdlib__In_channel.cmx stdlib__Int.cmi
    stdlib__Int.cmx stdlib__Int32.cmi stdlib__Int32.cmx stdlib__Int64.cmi
    stdlib__Int64.cmx stdlib__Lazy.cmi stdlib__Lazy.cmx stdlib__List.cmi
    stdlib__List.cmx stdlib__Map.cmi stdlib__Map.cmx stdlib__Modes.cmi
    stdlib__Modes.cmx stdlib__Mutex.cmi stdlib__Mutex.cmx
    stdlib__Nativeint.cmi stdlib__Nativeint.cmx stdlib__Obj.cmi
    stdlib__Obj.cmx stdlib__Printexc.cmi stdlib__Printexc.cmx
    stdlib__Printf.cmi stdlib__Printf.cmx stdlib__Random.cmi
    stdlib__Random.cmx stdlib__Seq.cmi stdlib__Seq.cmx stdlib__String.cmi
    stdlib__String.cmx stdlib__Sys.cmi stdlib__Sys.cmx stdlib__Uchar.cmi
    stdlib__Uchar.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Backoff.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Backoff.cmx -c ./backoff.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../backoff.own_expected
   backoff.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/array.ml
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/bigarray.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/complex.ml
  ../../../../../../stdlib/complex.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/float.ml
  ../../../../../../stdlib/float.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.ml
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli)
 (action
  (with-stdout-to
   bigarray.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/bigarray.ml}
    numbered stdlib__Bigarray 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Array.cmi stdlib__Array.cmx stdlib__Atomic.cmi
    stdlib__Atomic.cmx stdlib__Bigarray.cmi stdlib__Complex.cmi
    stdlib__Complex.cmx stdlib__Either.cmi stdlib__Either.cmx
    stdlib__Float.cmi stdlib__Float.cmx stdlib__Int32.cmi
    stdlib__Int32.cmx stdlib__Int64.cmi stdlib__Int64.cmx stdlib__Lazy.cmi
    stdlib__Lazy.cmx stdlib__List.cmi stdlib__List.cmx stdlib__Obj.cmi
    stdlib__Obj.cmx stdlib__Seq.cmi stdlib__Seq.cmx stdlib__Sys.cmi
    stdlib__Sys.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Bigarray.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Bigarray.cmx -c ./bigarray.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../bigarray.own_expected
   bigarray.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/bool.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli)
 (action
  (with-stdout-to
   bool.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/bool.ml}
    numbered stdlib__Bool 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx stdlib.cmi
    stdlib.cmx stdlib__Bool.cmi
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Bool.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Bool.cmx -c ./bool.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../bool.own_expected
   bool.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/array.ml
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/buffer.mli
  ../../../../../../stdlib/bytes.ml
  ../../../../../../stdlib/bytes.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/char.ml
  ../../../../../../stdlib/char.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/int.ml
  ../../../../../../stdlib/int.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.ml
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/string.ml
  ../../../../../../stdlib/string.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli
  ../../../../../../stdlib/uchar.ml
  ../../../../../../stdlib/uchar.mli)
 (action
  (with-stdout-to
   buffer.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/buffer.ml}
    numbered stdlib__Buffer 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Array.cmi stdlib__Array.cmx stdlib__Atomic.cmi
    stdlib__Atomic.cmx stdlib__Buffer.cmi stdlib__Bytes.cmi
    stdlib__Bytes.cmx stdlib__Char.cmi stdlib__Char.cmx stdlib__Either.cmi
    stdlib__Either.cmx stdlib__Int.cmi stdlib__Int.cmx stdlib__Int32.cmi
    stdlib__Int32.cmx stdlib__Int64.cmi stdlib__Int64.cmx stdlib__Lazy.cmi
    stdlib__Lazy.cmx stdlib__List.cmi stdlib__List.cmx stdlib__Obj.cmi
    stdlib__Obj.cmx stdlib__Seq.cmi stdlib__Seq.cmx stdlib__String.cmi
    stdlib__String.cmx stdlib__Sys.cmi stdlib__Sys.cmx stdlib__Uchar.cmi
    stdlib__Uchar.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Buffer.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Buffer.cmx -c ./buffer.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../buffer.own_expected
   buffer.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/bytes.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/char.ml
  ../../../../../../stdlib/char.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/int.ml
  ../../../../../../stdlib/int.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli
  ../../../../../../stdlib/uchar.ml
  ../../../../../../stdlib/uchar.mli)
 (action
  (with-stdout-to
   bytes.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/bytes.ml}
    numbered stdlib__Bytes 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Atomic.cmi stdlib__Atomic.cmx stdlib__Bytes.cmi
    stdlib__Char.cmi stdlib__Char.cmx stdlib__Either.cmi
    stdlib__Either.cmx stdlib__Int.cmi stdlib__Int.cmx stdlib__Int32.cmi
    stdlib__Int32.cmx stdlib__Int64.cmi stdlib__Int64.cmx stdlib__Lazy.cmi
    stdlib__Lazy.cmx stdlib__Obj.cmi stdlib__Obj.cmx stdlib__Seq.cmi
    stdlib__Seq.cmx stdlib__Sys.cmi stdlib__Sys.cmx stdlib__Uchar.cmi
    stdlib__Uchar.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Bytes.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Bytes.cmx -c ./bytes.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../bytes.own_expected
   bytes.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/bytes.ml
  ../../../../../../stdlib/bytes.mli
  ../../../../../../stdlib/bytesLabels.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/char.ml
  ../../../../../../stdlib/char.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/int.ml
  ../../../../../../stdlib/int.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli
  ../../../../../../stdlib/uchar.ml
  ../../../../../../stdlib/uchar.mli)
 (action
  (with-stdout-to
   bytesLabels.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/bytesLabels.ml}
    numbered stdlib__BytesLabels 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Atomic.cmi stdlib__Atomic.cmx stdlib__Bytes.cmi
    stdlib__Bytes.cmx stdlib__BytesLabels.cmi stdlib__Char.cmi
    stdlib__Char.cmx stdlib__Either.cmi stdlib__Either.cmx stdlib__Int.cmi
    stdlib__Int.cmx stdlib__Int32.cmi stdlib__Int32.cmx stdlib__Int64.cmi
    stdlib__Int64.cmx stdlib__Lazy.cmi stdlib__Lazy.cmx stdlib__Obj.cmi
    stdlib__Obj.cmx stdlib__Seq.cmi stdlib__Seq.cmx stdlib__Sys.cmi
    stdlib__Sys.cmx stdlib__Uchar.cmi stdlib__Uchar.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__BytesLabels.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__BytesLabels.cmx -c ./bytesLabels.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../bytesLabels.own_expected
   bytesLabels.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/callback.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli)
 (action
  (with-stdout-to
   callback.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/callback.ml}
    numbered stdlib__Callback 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx stdlib.cmi
    stdlib.cmx stdlib__Callback.cmi stdlib__Int32.cmi stdlib__Int32.cmx
    stdlib__Int64.cmi stdlib__Int64.cmx stdlib__Obj.cmi stdlib__Obj.cmx
    stdlib__Sys.cmi stdlib__Sys.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Callback.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Callback.cmx -c ./callback.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../callback.own_expected
   callback.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/camlinternalAtomic.mli)
 (action
  (with-stdout-to
   camlinternalAtomic.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/camlinternalAtomic.ml}
    numbered camlinternalAtomic 0
    -
    camlinternalAtomic.cmi
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file camlinternalAtomic.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    camlinternalAtomic.cmx -c ./camlinternalAtomic.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../camlinternalAtomic.own_expected
   camlinternalAtomic.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/camlinternalComprehension.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli)
 (action
  (with-stdout-to
   camlinternalComprehension.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/camlinternalComprehension.ml}
    numbered camlinternalComprehension 0
    -
    camlinternalComprehension.cmi camlinternalFormatBasics.cmi
    camlinternalFormatBasics.cmx stdlib.cmi stdlib.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file camlinternalComprehension.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    camlinternalComprehension.cmx -c ./camlinternalComprehension.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../camlinternalComprehension.own_expected
   camlinternalComprehension.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/array.ml
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/buffer.ml
  ../../../../../../stdlib/buffer.mli
  ../../../../../../stdlib/bytes.ml
  ../../../../../../stdlib/bytes.mli
  ../../../../../../stdlib/camlinternalFormat.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/char.ml
  ../../../../../../stdlib/char.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/int.ml
  ../../../../../../stdlib/int.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.ml
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/string.ml
  ../../../../../../stdlib/string.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli
  ../../../../../../stdlib/uchar.ml
  ../../../../../../stdlib/uchar.mli)
 (action
  (with-stdout-to
   camlinternalFormat.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/camlinternalFormat.ml}
    numbered camlinternalFormat 0
    -
    camlinternalFormat.cmi camlinternalFormatBasics.cmi
    camlinternalFormatBasics.cmx camlinternalLazy.cmi camlinternalLazy.cmx
    stdlib.cmi stdlib.cmx stdlib__Array.cmi stdlib__Array.cmx
    stdlib__Atomic.cmi stdlib__Atomic.cmx stdlib__Buffer.cmi
    stdlib__Buffer.cmx stdlib__Bytes.cmi stdlib__Bytes.cmx
    stdlib__Char.cmi stdlib__Char.cmx stdlib__Either.cmi
    stdlib__Either.cmx stdlib__Int.cmi stdlib__Int.cmx stdlib__Int32.cmi
    stdlib__Int32.cmx stdlib__Int64.cmi stdlib__Int64.cmx stdlib__Lazy.cmi
    stdlib__Lazy.cmx stdlib__List.cmi stdlib__List.cmx stdlib__Obj.cmi
    stdlib__Obj.cmx stdlib__Seq.cmi stdlib__Seq.cmx stdlib__String.cmi
    stdlib__String.cmx stdlib__Sys.cmi stdlib__Sys.cmx stdlib__Uchar.cmi
    stdlib__Uchar.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file camlinternalFormat.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    camlinternalFormat.cmx -c ./camlinternalFormat.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../camlinternalFormat.own_expected
   camlinternalFormat.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/camlinternalFormatBasics.mli)
 (action
  (with-stdout-to
   camlinternalFormatBasics.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/camlinternalFormatBasics.ml}
    numbered camlinternalFormatBasics 0
    -
    camlinternalFormatBasics.cmi
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file camlinternalFormatBasics.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    camlinternalFormatBasics.cmx -c ./camlinternalFormatBasics.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../camlinternalFormatBasics.own_expected
   camlinternalFormatBasics.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli)
 (action
  (with-stdout-to
   camlinternalLazy.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/camlinternalLazy.ml}
    numbered camlinternalLazy 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi stdlib.cmi stdlib.cmx stdlib__Int32.cmi
    stdlib__Int32.cmx stdlib__Int64.cmi stdlib__Int64.cmx stdlib__Obj.cmi
    stdlib__Obj.cmx stdlib__Sys.cmi stdlib__Sys.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file camlinternalLazy.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    camlinternalLazy.cmx -c ./camlinternalLazy.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../camlinternalLazy.own_expected
   camlinternalLazy.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/array.ml
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/bytes.ml
  ../../../../../../stdlib/bytes.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/camlinternalMod.mli
  ../../../../../../stdlib/camlinternalOO.ml
  ../../../../../../stdlib/camlinternalOO.mli
  ../../../../../../stdlib/char.ml
  ../../../../../../stdlib/char.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/int.ml
  ../../../../../../stdlib/int.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.ml
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/map.ml
  ../../../../../../stdlib/map.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/string.ml
  ../../../../../../stdlib/string.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli
  ../../../../../../stdlib/uchar.ml
  ../../../../../../stdlib/uchar.mli)
 (action
  (with-stdout-to
   camlinternalMod.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/camlinternalMod.ml}
    numbered camlinternalMod 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx camlinternalMod.cmi
    camlinternalOO.cmi camlinternalOO.cmx stdlib.cmi stdlib.cmx
    stdlib__Array.cmi stdlib__Array.cmx stdlib__Atomic.cmi
    stdlib__Atomic.cmx stdlib__Bytes.cmi stdlib__Bytes.cmx
    stdlib__Char.cmi stdlib__Char.cmx stdlib__Either.cmi
    stdlib__Either.cmx stdlib__Int.cmi stdlib__Int.cmx stdlib__Int32.cmi
    stdlib__Int32.cmx stdlib__Int64.cmi stdlib__Int64.cmx stdlib__Lazy.cmi
    stdlib__Lazy.cmx stdlib__List.cmi stdlib__List.cmx stdlib__Map.cmi
    stdlib__Map.cmx stdlib__Obj.cmi stdlib__Obj.cmx stdlib__Seq.cmi
    stdlib__Seq.cmx stdlib__String.cmi stdlib__String.cmx stdlib__Sys.cmi
    stdlib__Sys.cmx stdlib__Uchar.cmi stdlib__Uchar.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file camlinternalMod.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    camlinternalMod.cmx -c ./camlinternalMod.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../camlinternalMod.own_expected
   camlinternalMod.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/array.ml
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/bytes.ml
  ../../../../../../stdlib/bytes.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/camlinternalOO.mli
  ../../../../../../stdlib/char.ml
  ../../../../../../stdlib/char.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/int.ml
  ../../../../../../stdlib/int.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.ml
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/map.ml
  ../../../../../../stdlib/map.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/string.ml
  ../../../../../../stdlib/string.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli
  ../../../../../../stdlib/uchar.ml
  ../../../../../../stdlib/uchar.mli)
 (action
  (with-stdout-to
   camlinternalOO.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/camlinternalOO.ml}
    numbered camlinternalOO 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx camlinternalOO.cmi
    stdlib.cmi stdlib.cmx stdlib__Array.cmi stdlib__Array.cmx
    stdlib__Atomic.cmi stdlib__Atomic.cmx stdlib__Bytes.cmi
    stdlib__Bytes.cmx stdlib__Char.cmi stdlib__Char.cmx stdlib__Either.cmi
    stdlib__Either.cmx stdlib__Int.cmi stdlib__Int.cmx stdlib__Int32.cmi
    stdlib__Int32.cmx stdlib__Int64.cmi stdlib__Int64.cmx stdlib__Lazy.cmi
    stdlib__Lazy.cmx stdlib__List.cmi stdlib__List.cmx stdlib__Map.cmi
    stdlib__Map.cmx stdlib__Obj.cmi stdlib__Obj.cmx stdlib__Seq.cmi
    stdlib__Seq.cmx stdlib__String.cmi stdlib__String.cmx stdlib__Sys.cmi
    stdlib__Sys.cmx stdlib__Uchar.cmi stdlib__Uchar.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file camlinternalOO.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    camlinternalOO.cmx -c ./camlinternalOO.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../camlinternalOO.own_expected
   camlinternalOO.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/array.ml
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/bigarray.ml
  ../../../../../../stdlib/bigarray.mli
  ../../../../../../stdlib/buffer.ml
  ../../../../../../stdlib/buffer.mli
  ../../../../../../stdlib/bytes.ml
  ../../../../../../stdlib/bytes.mli
  ../../../../../../stdlib/callback.ml
  ../../../../../../stdlib/callback.mli
  ../../../../../../stdlib/camlinternalFormat.ml
  ../../../../../../stdlib/camlinternalFormat.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/camlinternalQuote.mli
  ../../../../../../stdlib/char.ml
  ../../../../../../stdlib/char.mli
  ../../../../../../stdlib/complex.ml
  ../../../../../../stdlib/complex.mli
  ../../../../../../stdlib/condition.ml
  ../../../../../../stdlib/condition.mli
  ../../../../../../stdlib/digest.ml
  ../../../../../../stdlib/digest.mli
  ../../../../../../stdlib/domain.ml
  ../../../../../../stdlib/domain.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/float.ml
  ../../../../../../stdlib/float.mli
  ../../../../../../stdlib/format.ml
  ../../../../../../stdlib/format.mli
  ../../../../../../stdlib/fun.ml
  ../../../../../../stdlib/fun.mli
  ../../../../../../stdlib/hashtbl.ml
  ../../../../../../stdlib/hashtbl.mli
  ../../../../../../stdlib/in_channel.ml
  ../../../../../../stdlib/in_channel.mli
  ../../../../../../stdlib/int.ml
  ../../../../../../stdlib/int.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.ml
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/map.ml
  ../../../../../../stdlib/map.mli
  ../../../../../../stdlib/modes.ml
  ../../../../../../stdlib/modes.mli
  ../../../../../../stdlib/mutex.ml
  ../../../../../../stdlib/mutex.mli
  ../../../../../../stdlib/nativeint.ml
  ../../../../../../stdlib/nativeint.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/option.ml
  ../../../../../../stdlib/option.mli
  ../../../../../../stdlib/printexc.ml
  ../../../../../../stdlib/printexc.mli
  ../../../../../../stdlib/printf.ml
  ../../../../../../stdlib/printf.mli
  ../../../../../../stdlib/queue.ml
  ../../../../../../stdlib/queue.mli
  ../../../../../../stdlib/random.ml
  ../../../../../../stdlib/random.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stack.ml
  ../../../../../../stdlib/stack.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/string.ml
  ../../../../../../stdlib/string.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli
  ../../../../../../stdlib/uchar.ml
  ../../../../../../stdlib/uchar.mli)
 (action
  (with-stdout-to
   camlinternalQuote.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/camlinternalQuote.ml}
    numbered camlinternalQuote 0
    -
    camlinternalFormat.cmi camlinternalFormat.cmx
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx camlinternalQuote.cmi
    stdlib.cmi stdlib.cmx stdlib__Array.cmi stdlib__Array.cmx
    stdlib__Atomic.cmi stdlib__Atomic.cmx stdlib__Bigarray.cmi
    stdlib__Bigarray.cmx stdlib__Buffer.cmi stdlib__Buffer.cmx
    stdlib__Bytes.cmi stdlib__Bytes.cmx stdlib__Callback.cmi
    stdlib__Callback.cmx stdlib__Char.cmi stdlib__Char.cmx
    stdlib__Complex.cmi stdlib__Complex.cmx stdlib__Condition.cmi
    stdlib__Condition.cmx stdlib__Digest.cmi stdlib__Digest.cmx
    stdlib__Domain.cmi stdlib__Domain.cmx stdlib__Either.cmi
    stdlib__Either.cmx stdlib__Float.cmi stdlib__Float.cmx
    stdlib__Format.cmi stdlib__Format.cmx stdlib__Fun.cmi stdlib__Fun.cmx
    stdlib__Hashtbl.cmi stdlib__Hashtbl.cmx stdlib__In_channel.cmi
    stdlib__In_channel.cmx stdlib__Int.cmi stdlib__Int.cmx
    stdlib__Int32.cmi stdlib__Int32.cmx stdlib__Int64.cmi
    stdlib__Int64.cmx stdlib__Lazy.cmi stdlib__Lazy.cmx stdlib__List.cmi
    stdlib__List.cmx stdlib__Map.cmi stdlib__Map.cmx stdlib__Modes.cmi
    stdlib__Modes.cmx stdlib__Mutex.cmi stdlib__Mutex.cmx
    stdlib__Nativeint.cmi stdlib__Nativeint.cmx stdlib__Obj.cmi
    stdlib__Obj.cmx stdlib__Option.cmi stdlib__Option.cmx
    stdlib__Printexc.cmi stdlib__Printexc.cmx stdlib__Printf.cmi
    stdlib__Printf.cmx stdlib__Queue.cmi stdlib__Queue.cmx
    stdlib__Random.cmi stdlib__Random.cmx stdlib__Seq.cmi stdlib__Seq.cmx
    stdlib__Stack.cmi stdlib__Stack.cmx stdlib__String.cmi
    stdlib__String.cmx stdlib__Sys.cmi stdlib__Sys.cmx stdlib__Uchar.cmi
    stdlib__Uchar.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file camlinternalQuote.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    camlinternalQuote.cmx -c ./camlinternalQuote.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../camlinternalQuote.own_expected
   camlinternalQuote.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/char.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli)
 (action
  (with-stdout-to
   char.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/char.ml}
    numbered stdlib__Char 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx stdlib.cmi
    stdlib.cmx stdlib__Char.cmi
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Char.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Char.cmx -c ./char.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../char.own_expected
   char.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/array.ml
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/complex.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/float.ml
  ../../../../../../stdlib/float.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.ml
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli)
 (action
  (with-stdout-to
   complex.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/complex.ml}
    numbered stdlib__Complex 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Array.cmi stdlib__Array.cmx stdlib__Atomic.cmi
    stdlib__Atomic.cmx stdlib__Complex.cmi stdlib__Either.cmi
    stdlib__Either.cmx stdlib__Float.cmi stdlib__Float.cmx
    stdlib__Int32.cmi stdlib__Int32.cmx stdlib__Int64.cmi
    stdlib__Int64.cmx stdlib__Lazy.cmi stdlib__Lazy.cmx stdlib__List.cmi
    stdlib__List.cmx stdlib__Obj.cmi stdlib__Obj.cmx stdlib__Seq.cmi
    stdlib__Seq.cmx stdlib__Sys.cmi stdlib__Sys.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Complex.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Complex.cmx -c ./complex.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../complex.own_expected
   complex.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/condition.mli
  ../../../../../../stdlib/mutex.ml
  ../../../../../../stdlib/mutex.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli)
 (action
  (with-stdout-to
   condition.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/condition.ml}
    numbered stdlib__Condition 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx stdlib.cmi
    stdlib.cmx stdlib__Condition.cmi stdlib__Mutex.cmi stdlib__Mutex.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Condition.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Condition.cmx -c ./condition.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../condition.own_expected
   condition.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/array.ml
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/bigarray.ml
  ../../../../../../stdlib/bigarray.mli
  ../../../../../../stdlib/buffer.ml
  ../../../../../../stdlib/buffer.mli
  ../../../../../../stdlib/bytes.ml
  ../../../../../../stdlib/bytes.mli
  ../../../../../../stdlib/camlinternalFormat.ml
  ../../../../../../stdlib/camlinternalFormat.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/char.ml
  ../../../../../../stdlib/char.mli
  ../../../../../../stdlib/complex.ml
  ../../../../../../stdlib/complex.mli
  ../../../../../../stdlib/digest.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/float.ml
  ../../../../../../stdlib/float.mli
  ../../../../../../stdlib/fun.ml
  ../../../../../../stdlib/fun.mli
  ../../../../../../stdlib/in_channel.ml
  ../../../../../../stdlib/in_channel.mli
  ../../../../../../stdlib/int.ml
  ../../../../../../stdlib/int.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.ml
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/modes.ml
  ../../../../../../stdlib/modes.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/printexc.ml
  ../../../../../../stdlib/printexc.mli
  ../../../../../../stdlib/printf.ml
  ../../../../../../stdlib/printf.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/string.ml
  ../../../../../../stdlib/string.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli
  ../../../../../../stdlib/uchar.ml
  ../../../../../../stdlib/uchar.mli)
 (action
  (with-stdout-to
   digest.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/digest.ml}
    numbered stdlib__Digest 0
    -
    camlinternalFormat.cmi camlinternalFormat.cmx
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Array.cmi stdlib__Array.cmx stdlib__Atomic.cmi
    stdlib__Atomic.cmx stdlib__Bigarray.cmi stdlib__Bigarray.cmx
    stdlib__Buffer.cmi stdlib__Buffer.cmx stdlib__Bytes.cmi
    stdlib__Bytes.cmx stdlib__Char.cmi stdlib__Char.cmx
    stdlib__Complex.cmi stdlib__Complex.cmx stdlib__Digest.cmi
    stdlib__Either.cmi stdlib__Either.cmx stdlib__Float.cmi
    stdlib__Float.cmx stdlib__Fun.cmi stdlib__Fun.cmx
    stdlib__In_channel.cmi stdlib__In_channel.cmx stdlib__Int.cmi
    stdlib__Int.cmx stdlib__Int32.cmi stdlib__Int32.cmx stdlib__Int64.cmi
    stdlib__Int64.cmx stdlib__Lazy.cmi stdlib__Lazy.cmx stdlib__List.cmi
    stdlib__List.cmx stdlib__Modes.cmi stdlib__Modes.cmx stdlib__Obj.cmi
    stdlib__Obj.cmx stdlib__Printexc.cmi stdlib__Printexc.cmx
    stdlib__Printf.cmi stdlib__Printf.cmx stdlib__Seq.cmi stdlib__Seq.cmx
    stdlib__String.cmi stdlib__String.cmx stdlib__Sys.cmi stdlib__Sys.cmx
    stdlib__Uchar.cmi stdlib__Uchar.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Digest.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Digest.cmx -c ./digest.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../digest.own_expected
   digest.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/array.ml
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/buffer.ml
  ../../../../../../stdlib/buffer.mli
  ../../../../../../stdlib/bytes.ml
  ../../../../../../stdlib/bytes.mli
  ../../../../../../stdlib/callback.ml
  ../../../../../../stdlib/callback.mli
  ../../../../../../stdlib/camlinternalFormat.ml
  ../../../../../../stdlib/camlinternalFormat.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/char.ml
  ../../../../../../stdlib/char.mli
  ../../../../../../stdlib/condition.ml
  ../../../../../../stdlib/condition.mli
  ../../../../../../stdlib/domain.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/int.ml
  ../../../../../../stdlib/int.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.ml
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/map.ml
  ../../../../../../stdlib/map.mli
  ../../../../../../stdlib/modes.ml
  ../../../../../../stdlib/modes.mli
  ../../../../../../stdlib/mutex.ml
  ../../../../../../stdlib/mutex.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/printexc.ml
  ../../../../../../stdlib/printexc.mli
  ../../../../../../stdlib/printf.ml
  ../../../../../../stdlib/printf.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/string.ml
  ../../../../../../stdlib/string.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli
  ../../../../../../stdlib/uchar.ml
  ../../../../../../stdlib/uchar.mli)
 (action
  (with-stdout-to
   domain.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/domain.ml}
    numbered stdlib__Domain 0
    -
    camlinternalFormat.cmi camlinternalFormat.cmx
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Array.cmi stdlib__Array.cmx stdlib__Atomic.cmi
    stdlib__Atomic.cmx stdlib__Buffer.cmi stdlib__Buffer.cmx
    stdlib__Bytes.cmi stdlib__Bytes.cmx stdlib__Callback.cmi
    stdlib__Callback.cmx stdlib__Char.cmi stdlib__Char.cmx
    stdlib__Condition.cmi stdlib__Condition.cmx stdlib__Domain.cmi
    stdlib__Either.cmi stdlib__Either.cmx stdlib__Int.cmi stdlib__Int.cmx
    stdlib__Int32.cmi stdlib__Int32.cmx stdlib__Int64.cmi
    stdlib__Int64.cmx stdlib__Lazy.cmi stdlib__Lazy.cmx stdlib__List.cmi
    stdlib__List.cmx stdlib__Map.cmi stdlib__Map.cmx stdlib__Modes.cmi
    stdlib__Modes.cmx stdlib__Mutex.cmi stdlib__Mutex.cmx stdlib__Obj.cmi
    stdlib__Obj.cmx stdlib__Printexc.cmi stdlib__Printexc.cmx
    stdlib__Printf.cmi stdlib__Printf.cmx stdlib__Seq.cmi stdlib__Seq.cmx
    stdlib__String.cmi stdlib__String.cmx stdlib__Sys.cmi stdlib__Sys.cmx
    stdlib__Uchar.cmi stdlib__Uchar.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Domain.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Domain.cmx -c ./domain.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../domain.own_expected
   domain.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/array.ml
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/buffer.ml
  ../../../../../../stdlib/buffer.mli
  ../../../../../../stdlib/bytes.ml
  ../../../../../../stdlib/bytes.mli
  ../../../../../../stdlib/camlinternalFormat.ml
  ../../../../../../stdlib/camlinternalFormat.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/char.ml
  ../../../../../../stdlib/char.mli
  ../../../../../../stdlib/dynamic.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/int.ml
  ../../../../../../stdlib/int.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.ml
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/modes.ml
  ../../../../../../stdlib/modes.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/printexc.ml
  ../../../../../../stdlib/printexc.mli
  ../../../../../../stdlib/printf.ml
  ../../../../../../stdlib/printf.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/string.ml
  ../../../../../../stdlib/string.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli
  ../../../../../../stdlib/uchar.ml
  ../../../../../../stdlib/uchar.mli)
 (action
  (with-stdout-to
   dynamic.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/dynamic.ml}
    numbered stdlib__Dynamic 0
    -
    camlinternalFormat.cmi camlinternalFormat.cmx
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Array.cmi stdlib__Array.cmx stdlib__Atomic.cmi
    stdlib__Atomic.cmx stdlib__Buffer.cmi stdlib__Buffer.cmx
    stdlib__Bytes.cmi stdlib__Bytes.cmx stdlib__Char.cmi stdlib__Char.cmx
    stdlib__Dynamic.cmi stdlib__Either.cmi stdlib__Either.cmx
    stdlib__Int.cmi stdlib__Int.cmx stdlib__Int32.cmi stdlib__Int32.cmx
    stdlib__Int64.cmi stdlib__Int64.cmx stdlib__Lazy.cmi stdlib__Lazy.cmx
    stdlib__List.cmi stdlib__List.cmx stdlib__Modes.cmi stdlib__Modes.cmx
    stdlib__Obj.cmi stdlib__Obj.cmx stdlib__Printexc.cmi
    stdlib__Printexc.cmx stdlib__Printf.cmi stdlib__Printf.cmx
    stdlib__Seq.cmi stdlib__Seq.cmx stdlib__String.cmi stdlib__String.cmx
    stdlib__Sys.cmi stdlib__Sys.cmx stdlib__Uchar.cmi stdlib__Uchar.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Dynamic.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Dynamic.cmx -c ./dynamic.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../dynamic.own_expected
   dynamic.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/array.ml
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/buffer.ml
  ../../../../../../stdlib/buffer.mli
  ../../../../../../stdlib/bytes.ml
  ../../../../../../stdlib/bytes.mli
  ../../../../../../stdlib/camlinternalFormat.ml
  ../../../../../../stdlib/camlinternalFormat.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/camlinternalOO.ml
  ../../../../../../stdlib/camlinternalOO.mli
  ../../../../../../stdlib/char.ml
  ../../../../../../stdlib/char.mli
  ../../../../../../stdlib/dynarray.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/int.ml
  ../../../../../../stdlib/int.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.ml
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/map.ml
  ../../../../../../stdlib/map.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/printf.ml
  ../../../../../../stdlib/printf.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/string.ml
  ../../../../../../stdlib/string.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli
  ../../../../../../stdlib/uchar.ml
  ../../../../../../stdlib/uchar.mli)
 (action
  (with-stdout-to
   dynarray.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/dynarray.ml}
    numbered stdlib__Dynarray 0
    -
    camlinternalFormat.cmi camlinternalFormat.cmx
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx camlinternalOO.cmi
    camlinternalOO.cmx stdlib.cmi stdlib.cmx stdlib__Array.cmi
    stdlib__Array.cmx stdlib__Atomic.cmi stdlib__Atomic.cmx
    stdlib__Buffer.cmi stdlib__Buffer.cmx stdlib__Bytes.cmi
    stdlib__Bytes.cmx stdlib__Char.cmi stdlib__Char.cmx
    stdlib__Dynarray.cmi stdlib__Either.cmi stdlib__Either.cmx
    stdlib__Int.cmi stdlib__Int.cmx stdlib__Int32.cmi stdlib__Int32.cmx
    stdlib__Int64.cmi stdlib__Int64.cmx stdlib__Lazy.cmi stdlib__Lazy.cmx
    stdlib__List.cmi stdlib__List.cmx stdlib__Map.cmi stdlib__Map.cmx
    stdlib__Obj.cmi stdlib__Obj.cmx stdlib__Printf.cmi stdlib__Printf.cmx
    stdlib__Seq.cmi stdlib__Seq.cmx stdlib__String.cmi stdlib__String.cmx
    stdlib__Sys.cmi stdlib__Sys.cmx stdlib__Uchar.cmi stdlib__Uchar.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Dynarray.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Dynarray.cmx -c ./dynarray.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../dynarray.own_expected
   dynarray.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/array.ml
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/buffer.ml
  ../../../../../../stdlib/buffer.mli
  ../../../../../../stdlib/bytes.ml
  ../../../../../../stdlib/bytes.mli
  ../../../../../../stdlib/callback.ml
  ../../../../../../stdlib/callback.mli
  ../../../../../../stdlib/camlinternalFormat.ml
  ../../../../../../stdlib/camlinternalFormat.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/char.ml
  ../../../../../../stdlib/char.mli
  ../../../../../../stdlib/condition.ml
  ../../../../../../stdlib/condition.mli
  ../../../../../../stdlib/domain.ml
  ../../../../../../stdlib/domain.mli
  ../../../../../../stdlib/effect.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/fun.ml
  ../../../../../../stdlib/fun.mli
  ../../../../../../stdlib/int.ml
  ../../../../../../stdlib/int.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.ml
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/map.ml
  ../../../../../../stdlib/map.mli
  ../../../../../../stdlib/modes.ml
  ../../../../../../stdlib/modes.mli
  ../../../../../../stdlib/mutex.ml
  ../../../../../../stdlib/mutex.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/printexc.ml
  ../../../../../../stdlib/printexc.mli
  ../../../../../../stdlib/printf.ml
  ../../../../../../stdlib/printf.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/string.ml
  ../../../../../../stdlib/string.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli
  ../../../../../../stdlib/uchar.ml
  ../../../../../../stdlib/uchar.mli)
 (action
  (with-stdout-to
   effect.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/effect.ml}
    numbered stdlib__Effect 0
    -
    camlinternalFormat.cmi camlinternalFormat.cmx
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Array.cmi stdlib__Array.cmx stdlib__Atomic.cmi
    stdlib__Atomic.cmx stdlib__Buffer.cmi stdlib__Buffer.cmx
    stdlib__Bytes.cmi stdlib__Bytes.cmx stdlib__Callback.cmi
    stdlib__Callback.cmx stdlib__Char.cmi stdlib__Char.cmx
    stdlib__Condition.cmi stdlib__Condition.cmx stdlib__Domain.cmi
    stdlib__Domain.cmx stdlib__Effect.cmi stdlib__Either.cmi
    stdlib__Either.cmx stdlib__Fun.cmi stdlib__Fun.cmx stdlib__Int.cmi
    stdlib__Int.cmx stdlib__Int32.cmi stdlib__Int32.cmx stdlib__Int64.cmi
    stdlib__Int64.cmx stdlib__Lazy.cmi stdlib__Lazy.cmx stdlib__List.cmi
    stdlib__List.cmx stdlib__Map.cmi stdlib__Map.cmx stdlib__Modes.cmi
    stdlib__Modes.cmx stdlib__Mutex.cmi stdlib__Mutex.cmx stdlib__Obj.cmi
    stdlib__Obj.cmx stdlib__Printexc.cmi stdlib__Printexc.cmx
    stdlib__Printf.cmi stdlib__Printf.cmx stdlib__Seq.cmi stdlib__Seq.cmx
    stdlib__String.cmi stdlib__String.cmx stdlib__Sys.cmi stdlib__Sys.cmx
    stdlib__Uchar.cmi stdlib__Uchar.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Effect.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Effect.cmx -c ./effect.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../effect.own_expected
   effect.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli)
 (action
  (with-stdout-to
   either.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/either.ml}
    numbered stdlib__Either 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx stdlib.cmi
    stdlib.cmx stdlib__Either.cmi
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Either.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Either.cmx -c ./either.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../either.own_expected
   either.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/array.ml
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/bigarray.ml
  ../../../../../../stdlib/bigarray.mli
  ../../../../../../stdlib/buffer.ml
  ../../../../../../stdlib/buffer.mli
  ../../../../../../stdlib/bytes.ml
  ../../../../../../stdlib/bytes.mli
  ../../../../../../stdlib/callback.ml
  ../../../../../../stdlib/callback.mli
  ../../../../../../stdlib/camlinternalFormat.ml
  ../../../../../../stdlib/camlinternalFormat.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/char.ml
  ../../../../../../stdlib/char.mli
  ../../../../../../stdlib/complex.ml
  ../../../../../../stdlib/complex.mli
  ../../../../../../stdlib/condition.ml
  ../../../../../../stdlib/condition.mli
  ../../../../../../stdlib/digest.ml
  ../../../../../../stdlib/digest.mli
  ../../../../../../stdlib/domain.ml
  ../../../../../../stdlib/domain.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/ephemeron.mli
  ../../../../../../stdlib/float.ml
  ../../../../../../stdlib/float.mli
  ../../../../../../stdlib/fun.ml
  ../../../../../../stdlib/fun.mli
  ../../../../../../stdlib/hashtbl.ml
  ../../../../../../stdlib/hashtbl.mli
  ../../../../../../stdlib/in_channel.ml
  ../../../../../../stdlib/in_channel.mli
  ../../../../../../stdlib/int.ml
  ../../../../../../stdlib/int.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.ml
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/map.ml
  ../../../../../../stdlib/map.mli
  ../../../../../../stdlib/modes.ml
  ../../../../../../stdlib/modes.mli
  ../../../../../../stdlib/mutex.ml
  ../../../../../../stdlib/mutex.mli
  ../../../../../../stdlib/nativeint.ml
  ../../../../../../stdlib/nativeint.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/printexc.ml
  ../../../../../../stdlib/printexc.mli
  ../../../../../../stdlib/printf.ml
  ../../../../../../stdlib/printf.mli
  ../../../../../../stdlib/random.ml
  ../../../../../../stdlib/random.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/string.ml
  ../../../../../../stdlib/string.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli
  ../../../../../../stdlib/uchar.ml
  ../../../../../../stdlib/uchar.mli)
 (action
  (with-stdout-to
   ephemeron.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/ephemeron.ml}
    numbered stdlib__Ephemeron 0
    -
    camlinternalFormat.cmi camlinternalFormat.cmx
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Array.cmi stdlib__Array.cmx stdlib__Atomic.cmi
    stdlib__Atomic.cmx stdlib__Bigarray.cmi stdlib__Bigarray.cmx
    stdlib__Buffer.cmi stdlib__Buffer.cmx stdlib__Bytes.cmi
    stdlib__Bytes.cmx stdlib__Callback.cmi stdlib__Callback.cmx
    stdlib__Char.cmi stdlib__Char.cmx stdlib__Complex.cmi
    stdlib__Complex.cmx stdlib__Condition.cmi stdlib__Condition.cmx
    stdlib__Digest.cmi stdlib__Digest.cmx stdlib__Domain.cmi
    stdlib__Domain.cmx stdlib__Either.cmi stdlib__Either.cmx
    stdlib__Ephemeron.cmi stdlib__Float.cmi stdlib__Float.cmx
    stdlib__Fun.cmi stdlib__Fun.cmx stdlib__Hashtbl.cmi
    stdlib__Hashtbl.cmx stdlib__In_channel.cmi stdlib__In_channel.cmx
    stdlib__Int.cmi stdlib__Int.cmx stdlib__Int32.cmi stdlib__Int32.cmx
    stdlib__Int64.cmi stdlib__Int64.cmx stdlib__Lazy.cmi stdlib__Lazy.cmx
    stdlib__List.cmi stdlib__List.cmx stdlib__Map.cmi stdlib__Map.cmx
    stdlib__Modes.cmi stdlib__Modes.cmx stdlib__Mutex.cmi
    stdlib__Mutex.cmx stdlib__Nativeint.cmi stdlib__Nativeint.cmx
    stdlib__Obj.cmi stdlib__Obj.cmx stdlib__Printexc.cmi
    stdlib__Printexc.cmx stdlib__Printf.cmi stdlib__Printf.cmx
    stdlib__Random.cmi stdlib__Random.cmx stdlib__Seq.cmi stdlib__Seq.cmx
    stdlib__String.cmi stdlib__String.cmx stdlib__Sys.cmi stdlib__Sys.cmx
    stdlib__Uchar.cmi stdlib__Uchar.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Ephemeron.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Ephemeron.cmx -c ./ephemeron.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../ephemeron.own_expected
   ephemeron.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/array.ml
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/bigarray.ml
  ../../../../../../stdlib/bigarray.mli
  ../../../../../../stdlib/buffer.ml
  ../../../../../../stdlib/buffer.mli
  ../../../../../../stdlib/bytes.ml
  ../../../../../../stdlib/bytes.mli
  ../../../../../../stdlib/callback.ml
  ../../../../../../stdlib/callback.mli
  ../../../../../../stdlib/camlinternalFormat.ml
  ../../../../../../stdlib/camlinternalFormat.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/char.ml
  ../../../../../../stdlib/char.mli
  ../../../../../../stdlib/complex.ml
  ../../../../../../stdlib/complex.mli
  ../../../../../../stdlib/condition.ml
  ../../../../../../stdlib/condition.mli
  ../../../../../../stdlib/digest.ml
  ../../../../../../stdlib/digest.mli
  ../../../../../../stdlib/domain.ml
  ../../../../../../stdlib/domain.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/filename.mli
  ../../../../../../stdlib/float.ml
  ../../../../../../stdlib/float.mli
  ../../../../../../stdlib/fun.ml
  ../../../../../../stdlib/fun.mli
  ../../../../../../stdlib/in_channel.ml
  ../../../../../../stdlib/in_channel.mli
  ../../../../../../stdlib/int.ml
  ../../../../../../stdlib/int.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.ml
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/map.ml
  ../../../../../../stdlib/map.mli
  ../../../../../../stdlib/modes.ml
  ../../../../../../stdlib/modes.mli
  ../../../../../../stdlib/mutex.ml
  ../../../../../../stdlib/mutex.mli
  ../../../../../../stdlib/nativeint.ml
  ../../../../../../stdlib/nativeint.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/printexc.ml
  ../../../../../../stdlib/printexc.mli
  ../../../../../../stdlib/printf.ml
  ../../../../../../stdlib/printf.mli
  ../../../../../../stdlib/random.ml
  ../../../../../../stdlib/random.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/string.ml
  ../../../../../../stdlib/string.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli
  ../../../../../../stdlib/uchar.ml
  ../../../../../../stdlib/uchar.mli)
 (action
  (with-stdout-to
   filename.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/filename.ml}
    numbered stdlib__Filename 0
    -
    camlinternalFormat.cmi camlinternalFormat.cmx
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Array.cmi stdlib__Array.cmx stdlib__Atomic.cmi
    stdlib__Atomic.cmx stdlib__Bigarray.cmi stdlib__Bigarray.cmx
    stdlib__Buffer.cmi stdlib__Buffer.cmx stdlib__Bytes.cmi
    stdlib__Bytes.cmx stdlib__Callback.cmi stdlib__Callback.cmx
    stdlib__Char.cmi stdlib__Char.cmx stdlib__Complex.cmi
    stdlib__Complex.cmx stdlib__Condition.cmi stdlib__Condition.cmx
    stdlib__Digest.cmi stdlib__Digest.cmx stdlib__Domain.cmi
    stdlib__Domain.cmx stdlib__Either.cmi stdlib__Either.cmx
    stdlib__Filename.cmi stdlib__Float.cmi stdlib__Float.cmx
    stdlib__Fun.cmi stdlib__Fun.cmx stdlib__In_channel.cmi
    stdlib__In_channel.cmx stdlib__Int.cmi stdlib__Int.cmx
    stdlib__Int32.cmi stdlib__Int32.cmx stdlib__Int64.cmi
    stdlib__Int64.cmx stdlib__Lazy.cmi stdlib__Lazy.cmx stdlib__List.cmi
    stdlib__List.cmx stdlib__Map.cmi stdlib__Map.cmx stdlib__Modes.cmi
    stdlib__Modes.cmx stdlib__Mutex.cmi stdlib__Mutex.cmx
    stdlib__Nativeint.cmi stdlib__Nativeint.cmx stdlib__Obj.cmi
    stdlib__Obj.cmx stdlib__Printexc.cmi stdlib__Printexc.cmx
    stdlib__Printf.cmi stdlib__Printf.cmx stdlib__Random.cmi
    stdlib__Random.cmx stdlib__Seq.cmi stdlib__Seq.cmx stdlib__String.cmi
    stdlib__String.cmx stdlib__Sys.cmi stdlib__Sys.cmx stdlib__Uchar.cmi
    stdlib__Uchar.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Filename.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Filename.cmx -c ./filename.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../filename.own_expected
   filename.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/array.ml
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/float.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.ml
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli)
 (action
  (with-stdout-to
   float.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/float.ml}
    numbered stdlib__Float 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Array.cmi stdlib__Array.cmx stdlib__Atomic.cmi
    stdlib__Atomic.cmx stdlib__Either.cmi stdlib__Either.cmx
    stdlib__Float.cmi stdlib__Int32.cmi stdlib__Int32.cmx
    stdlib__Int64.cmi stdlib__Int64.cmx stdlib__Lazy.cmi stdlib__Lazy.cmx
    stdlib__List.cmi stdlib__List.cmx stdlib__Obj.cmi stdlib__Obj.cmx
    stdlib__Seq.cmi stdlib__Seq.cmx stdlib__Sys.cmi stdlib__Sys.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Float.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Float.cmx -c ./float.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../float.own_expected
   float.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/array.ml
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/buffer.ml
  ../../../../../../stdlib/buffer.mli
  ../../../../../../stdlib/bytes.ml
  ../../../../../../stdlib/bytes.mli
  ../../../../../../stdlib/callback.ml
  ../../../../../../stdlib/callback.mli
  ../../../../../../stdlib/camlinternalFormat.ml
  ../../../../../../stdlib/camlinternalFormat.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/char.ml
  ../../../../../../stdlib/char.mli
  ../../../../../../stdlib/condition.ml
  ../../../../../../stdlib/condition.mli
  ../../../../../../stdlib/domain.ml
  ../../../../../../stdlib/domain.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/format.mli
  ../../../../../../stdlib/int.ml
  ../../../../../../stdlib/int.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.ml
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/map.ml
  ../../../../../../stdlib/map.mli
  ../../../../../../stdlib/modes.ml
  ../../../../../../stdlib/modes.mli
  ../../../../../../stdlib/mutex.ml
  ../../../../../../stdlib/mutex.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/printexc.ml
  ../../../../../../stdlib/printexc.mli
  ../../../../../../stdlib/printf.ml
  ../../../../../../stdlib/printf.mli
  ../../../../../../stdlib/queue.ml
  ../../../../../../stdlib/queue.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stack.ml
  ../../../../../../stdlib/stack.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/string.ml
  ../../../../../../stdlib/string.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli
  ../../../../../../stdlib/uchar.ml
  ../../../../../../stdlib/uchar.mli)
 (action
  (with-stdout-to
   format.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/format.ml}
    numbered stdlib__Format 0
    -
    camlinternalFormat.cmi camlinternalFormat.cmx
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Array.cmi stdlib__Array.cmx stdlib__Atomic.cmi
    stdlib__Atomic.cmx stdlib__Buffer.cmi stdlib__Buffer.cmx
    stdlib__Bytes.cmi stdlib__Bytes.cmx stdlib__Callback.cmi
    stdlib__Callback.cmx stdlib__Char.cmi stdlib__Char.cmx
    stdlib__Condition.cmi stdlib__Condition.cmx stdlib__Domain.cmi
    stdlib__Domain.cmx stdlib__Either.cmi stdlib__Either.cmx
    stdlib__Format.cmi stdlib__Int.cmi stdlib__Int.cmx stdlib__Int32.cmi
    stdlib__Int32.cmx stdlib__Int64.cmi stdlib__Int64.cmx stdlib__Lazy.cmi
    stdlib__Lazy.cmx stdlib__List.cmi stdlib__List.cmx stdlib__Map.cmi
    stdlib__Map.cmx stdlib__Modes.cmi stdlib__Modes.cmx stdlib__Mutex.cmi
    stdlib__Mutex.cmx stdlib__Obj.cmi stdlib__Obj.cmx stdlib__Printexc.cmi
    stdlib__Printexc.cmx stdlib__Printf.cmi stdlib__Printf.cmx
    stdlib__Queue.cmi stdlib__Queue.cmx stdlib__Seq.cmi stdlib__Seq.cmx
    stdlib__Stack.cmi stdlib__Stack.cmx stdlib__String.cmi
    stdlib__String.cmx stdlib__Sys.cmi stdlib__Sys.cmx stdlib__Uchar.cmi
    stdlib__Uchar.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Format.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Format.cmx -c ./format.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../format.own_expected
   format.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/array.ml
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/buffer.ml
  ../../../../../../stdlib/buffer.mli
  ../../../../../../stdlib/bytes.ml
  ../../../../../../stdlib/bytes.mli
  ../../../../../../stdlib/camlinternalFormat.ml
  ../../../../../../stdlib/camlinternalFormat.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/char.ml
  ../../../../../../stdlib/char.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/fun.mli
  ../../../../../../stdlib/int.ml
  ../../../../../../stdlib/int.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.ml
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/modes.ml
  ../../../../../../stdlib/modes.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/printexc.ml
  ../../../../../../stdlib/printexc.mli
  ../../../../../../stdlib/printf.ml
  ../../../../../../stdlib/printf.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/string.ml
  ../../../../../../stdlib/string.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli
  ../../../../../../stdlib/uchar.ml
  ../../../../../../stdlib/uchar.mli)
 (action
  (with-stdout-to
   fun.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/fun.ml}
    numbered stdlib__Fun 0
    -
    camlinternalFormat.cmi camlinternalFormat.cmx
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Array.cmi stdlib__Array.cmx stdlib__Atomic.cmi
    stdlib__Atomic.cmx stdlib__Buffer.cmi stdlib__Buffer.cmx
    stdlib__Bytes.cmi stdlib__Bytes.cmx stdlib__Char.cmi stdlib__Char.cmx
    stdlib__Either.cmi stdlib__Either.cmx stdlib__Fun.cmi stdlib__Int.cmi
    stdlib__Int.cmx stdlib__Int32.cmi stdlib__Int32.cmx stdlib__Int64.cmi
    stdlib__Int64.cmx stdlib__Lazy.cmi stdlib__Lazy.cmx stdlib__List.cmi
    stdlib__List.cmx stdlib__Modes.cmi stdlib__Modes.cmx stdlib__Obj.cmi
    stdlib__Obj.cmx stdlib__Printexc.cmi stdlib__Printexc.cmx
    stdlib__Printf.cmi stdlib__Printf.cmx stdlib__Seq.cmi stdlib__Seq.cmx
    stdlib__String.cmi stdlib__String.cmx stdlib__Sys.cmi stdlib__Sys.cmx
    stdlib__Uchar.cmi stdlib__Uchar.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Fun.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Fun.cmx -c ./fun.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../fun.own_expected
   fun.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/array.ml
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/buffer.ml
  ../../../../../../stdlib/buffer.mli
  ../../../../../../stdlib/bytes.ml
  ../../../../../../stdlib/bytes.mli
  ../../../../../../stdlib/callback.ml
  ../../../../../../stdlib/callback.mli
  ../../../../../../stdlib/camlinternalFormat.ml
  ../../../../../../stdlib/camlinternalFormat.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/char.ml
  ../../../../../../stdlib/char.mli
  ../../../../../../stdlib/condition.ml
  ../../../../../../stdlib/condition.mli
  ../../../../../../stdlib/domain.ml
  ../../../../../../stdlib/domain.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/fun.ml
  ../../../../../../stdlib/fun.mli
  ../../../../../../stdlib/gc.mli
  ../../../../../../stdlib/int.ml
  ../../../../../../stdlib/int.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.ml
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/map.ml
  ../../../../../../stdlib/map.mli
  ../../../../../../stdlib/modes.ml
  ../../../../../../stdlib/modes.mli
  ../../../../../../stdlib/mutex.ml
  ../../../../../../stdlib/mutex.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/printexc.ml
  ../../../../../../stdlib/printexc.mli
  ../../../../../../stdlib/printf.ml
  ../../../../../../stdlib/printf.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/string.ml
  ../../../../../../stdlib/string.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli
  ../../../../../../stdlib/uchar.ml
  ../../../../../../stdlib/uchar.mli)
 (action
  (with-stdout-to
   gc.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/gc.ml}
    numbered stdlib__Gc 0
    -
    camlinternalFormat.cmi camlinternalFormat.cmx
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Array.cmi stdlib__Array.cmx stdlib__Atomic.cmi
    stdlib__Atomic.cmx stdlib__Buffer.cmi stdlib__Buffer.cmx
    stdlib__Bytes.cmi stdlib__Bytes.cmx stdlib__Callback.cmi
    stdlib__Callback.cmx stdlib__Char.cmi stdlib__Char.cmx
    stdlib__Condition.cmi stdlib__Condition.cmx stdlib__Domain.cmi
    stdlib__Domain.cmx stdlib__Either.cmi stdlib__Either.cmx
    stdlib__Fun.cmi stdlib__Fun.cmx stdlib__Gc.cmi stdlib__Int.cmi
    stdlib__Int.cmx stdlib__Int32.cmi stdlib__Int32.cmx stdlib__Int64.cmi
    stdlib__Int64.cmx stdlib__Lazy.cmi stdlib__Lazy.cmx stdlib__List.cmi
    stdlib__List.cmx stdlib__Map.cmi stdlib__Map.cmx stdlib__Modes.cmi
    stdlib__Modes.cmx stdlib__Mutex.cmi stdlib__Mutex.cmx stdlib__Obj.cmi
    stdlib__Obj.cmx stdlib__Printexc.cmi stdlib__Printexc.cmx
    stdlib__Printf.cmi stdlib__Printf.cmx stdlib__Seq.cmi stdlib__Seq.cmx
    stdlib__String.cmi stdlib__String.cmx stdlib__Sys.cmi stdlib__Sys.cmx
    stdlib__Uchar.cmi stdlib__Uchar.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Gc.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Gc.cmx -c ./gc.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../gc.own_expected
   gc.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/array.ml
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/bigarray.ml
  ../../../../../../stdlib/bigarray.mli
  ../../../../../../stdlib/buffer.ml
  ../../../../../../stdlib/buffer.mli
  ../../../../../../stdlib/bytes.ml
  ../../../../../../stdlib/bytes.mli
  ../../../../../../stdlib/callback.ml
  ../../../../../../stdlib/callback.mli
  ../../../../../../stdlib/camlinternalFormat.ml
  ../../../../../../stdlib/camlinternalFormat.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/char.ml
  ../../../../../../stdlib/char.mli
  ../../../../../../stdlib/complex.ml
  ../../../../../../stdlib/complex.mli
  ../../../../../../stdlib/condition.ml
  ../../../../../../stdlib/condition.mli
  ../../../../../../stdlib/digest.ml
  ../../../../../../stdlib/digest.mli
  ../../../../../../stdlib/domain.ml
  ../../../../../../stdlib/domain.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/float.ml
  ../../../../../../stdlib/float.mli
  ../../../../../../stdlib/fun.ml
  ../../../../../../stdlib/fun.mli
  ../../../../../../stdlib/hashtbl.mli
  ../../../../../../stdlib/in_channel.ml
  ../../../../../../stdlib/in_channel.mli
  ../../../../../../stdlib/int.ml
  ../../../../../../stdlib/int.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.ml
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/map.ml
  ../../../../../../stdlib/map.mli
  ../../../../../../stdlib/modes.ml
  ../../../../../../stdlib/modes.mli
  ../../../../../../stdlib/mutex.ml
  ../../../../../../stdlib/mutex.mli
  ../../../../../../stdlib/nativeint.ml
  ../../../../../../stdlib/nativeint.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/printexc.ml
  ../../../../../../stdlib/printexc.mli
  ../../../../../../stdlib/printf.ml
  ../../../../../../stdlib/printf.mli
  ../../../../../../stdlib/random.ml
  ../../../../../../stdlib/random.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/string.ml
  ../../../../../../stdlib/string.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli
  ../../../../../../stdlib/uchar.ml
  ../../../../../../stdlib/uchar.mli)
 (action
  (with-stdout-to
   hashtbl.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/hashtbl.ml}
    numbered stdlib__Hashtbl 0
    -
    camlinternalFormat.cmi camlinternalFormat.cmx
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Array.cmi stdlib__Array.cmx stdlib__Atomic.cmi
    stdlib__Atomic.cmx stdlib__Bigarray.cmi stdlib__Bigarray.cmx
    stdlib__Buffer.cmi stdlib__Buffer.cmx stdlib__Bytes.cmi
    stdlib__Bytes.cmx stdlib__Callback.cmi stdlib__Callback.cmx
    stdlib__Char.cmi stdlib__Char.cmx stdlib__Complex.cmi
    stdlib__Complex.cmx stdlib__Condition.cmi stdlib__Condition.cmx
    stdlib__Digest.cmi stdlib__Digest.cmx stdlib__Domain.cmi
    stdlib__Domain.cmx stdlib__Either.cmi stdlib__Either.cmx
    stdlib__Float.cmi stdlib__Float.cmx stdlib__Fun.cmi stdlib__Fun.cmx
    stdlib__Hashtbl.cmi stdlib__In_channel.cmi stdlib__In_channel.cmx
    stdlib__Int.cmi stdlib__Int.cmx stdlib__Int32.cmi stdlib__Int32.cmx
    stdlib__Int64.cmi stdlib__Int64.cmx stdlib__Lazy.cmi stdlib__Lazy.cmx
    stdlib__List.cmi stdlib__List.cmx stdlib__Map.cmi stdlib__Map.cmx
    stdlib__Modes.cmi stdlib__Modes.cmx stdlib__Mutex.cmi
    stdlib__Mutex.cmx stdlib__Nativeint.cmi stdlib__Nativeint.cmx
    stdlib__Obj.cmi stdlib__Obj.cmx stdlib__Printexc.cmi
    stdlib__Printexc.cmx stdlib__Printf.cmi stdlib__Printf.cmx
    stdlib__Random.cmi stdlib__Random.cmx stdlib__Seq.cmi stdlib__Seq.cmx
    stdlib__String.cmi stdlib__String.cmx stdlib__Sys.cmi stdlib__Sys.cmx
    stdlib__Uchar.cmi stdlib__Uchar.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Hashtbl.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Hashtbl.cmx -c ./hashtbl.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../hashtbl.own_expected
   hashtbl.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/array.ml
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/iarray.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli)
 (action
  (with-stdout-to
   iarray.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/iarray.ml}
    numbered stdlib__Iarray 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Array.cmi stdlib__Array.cmx stdlib__Atomic.cmi
    stdlib__Atomic.cmx stdlib__Either.cmi stdlib__Either.cmx
    stdlib__Iarray.cmi stdlib__Int32.cmi stdlib__Int32.cmx
    stdlib__Int64.cmi stdlib__Int64.cmx stdlib__Lazy.cmi stdlib__Lazy.cmx
    stdlib__Obj.cmi stdlib__Obj.cmx stdlib__Seq.cmi stdlib__Seq.cmx
    stdlib__Sys.cmi stdlib__Sys.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Iarray.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Iarray.cmx -c ./iarray.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../iarray.own_expected
   iarray.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/array.ml
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/bigarray.ml
  ../../../../../../stdlib/bigarray.mli
  ../../../../../../stdlib/buffer.ml
  ../../../../../../stdlib/buffer.mli
  ../../../../../../stdlib/bytes.ml
  ../../../../../../stdlib/bytes.mli
  ../../../../../../stdlib/camlinternalFormat.ml
  ../../../../../../stdlib/camlinternalFormat.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/char.ml
  ../../../../../../stdlib/char.mli
  ../../../../../../stdlib/complex.ml
  ../../../../../../stdlib/complex.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/float.ml
  ../../../../../../stdlib/float.mli
  ../../../../../../stdlib/fun.ml
  ../../../../../../stdlib/fun.mli
  ../../../../../../stdlib/in_channel.mli
  ../../../../../../stdlib/int.ml
  ../../../../../../stdlib/int.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.ml
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/modes.ml
  ../../../../../../stdlib/modes.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/printexc.ml
  ../../../../../../stdlib/printexc.mli
  ../../../../../../stdlib/printf.ml
  ../../../../../../stdlib/printf.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/string.ml
  ../../../../../../stdlib/string.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli
  ../../../../../../stdlib/uchar.ml
  ../../../../../../stdlib/uchar.mli)
 (action
  (with-stdout-to
   in_channel.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/in_channel.ml}
    numbered stdlib__In_channel 0
    -
    camlinternalFormat.cmi camlinternalFormat.cmx
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Array.cmi stdlib__Array.cmx stdlib__Atomic.cmi
    stdlib__Atomic.cmx stdlib__Bigarray.cmi stdlib__Bigarray.cmx
    stdlib__Buffer.cmi stdlib__Buffer.cmx stdlib__Bytes.cmi
    stdlib__Bytes.cmx stdlib__Char.cmi stdlib__Char.cmx
    stdlib__Complex.cmi stdlib__Complex.cmx stdlib__Either.cmi
    stdlib__Either.cmx stdlib__Float.cmi stdlib__Float.cmx stdlib__Fun.cmi
    stdlib__Fun.cmx stdlib__In_channel.cmi stdlib__Int.cmi stdlib__Int.cmx
    stdlib__Int32.cmi stdlib__Int32.cmx stdlib__Int64.cmi
    stdlib__Int64.cmx stdlib__Lazy.cmi stdlib__Lazy.cmx stdlib__List.cmi
    stdlib__List.cmx stdlib__Modes.cmi stdlib__Modes.cmx stdlib__Obj.cmi
    stdlib__Obj.cmx stdlib__Printexc.cmi stdlib__Printexc.cmx
    stdlib__Printf.cmi stdlib__Printf.cmx stdlib__Seq.cmi stdlib__Seq.cmx
    stdlib__String.cmi stdlib__String.cmx stdlib__Sys.cmi stdlib__Sys.cmx
    stdlib__Uchar.cmi stdlib__Uchar.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__In_channel.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__In_channel.cmx -c ./in_channel.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../in_channel.own_expected
   in_channel.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/int.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli)
 (action
  (with-stdout-to
   int.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/int.ml}
    numbered stdlib__Int 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx stdlib.cmi
    stdlib.cmx stdlib__Int.cmi
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Int.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Int.cmx -c ./int.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../int.own_expected
   int.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli)
 (action
  (with-stdout-to
   int32.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/int32.ml}
    numbered stdlib__Int32 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx stdlib.cmi
    stdlib.cmx stdlib__Int32.cmi stdlib__Sys.cmi stdlib__Sys.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Int32.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Int32.cmx -c ./int32.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../int32.own_expected
   int32.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli)
 (action
  (with-stdout-to
   int64.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/int64.ml}
    numbered stdlib__Int64 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx stdlib.cmi
    stdlib.cmx stdlib__Int64.cmi
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Int64.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Int64.cmx -c ./int64.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../int64.own_expected
   int64.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli)
 (action
  (with-stdout-to
   lazy.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/lazy.ml}
    numbered stdlib__Lazy 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Int32.cmi stdlib__Int32.cmx stdlib__Int64.cmi
    stdlib__Int64.cmx stdlib__Lazy.cmi stdlib__Obj.cmi stdlib__Obj.cmx
    stdlib__Sys.cmi stdlib__Sys.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Lazy.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Lazy.cmx -c ./lazy.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../lazy.own_expected
   lazy.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/array.ml
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/bytes.ml
  ../../../../../../stdlib/bytes.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/char.ml
  ../../../../../../stdlib/char.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/int.ml
  ../../../../../../stdlib/int.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/lexing.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli
  ../../../../../../stdlib/uchar.ml
  ../../../../../../stdlib/uchar.mli)
 (action
  (with-stdout-to
   lexing.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/lexing.ml}
    numbered stdlib__Lexing 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Array.cmi stdlib__Array.cmx stdlib__Atomic.cmi
    stdlib__Atomic.cmx stdlib__Bytes.cmi stdlib__Bytes.cmx
    stdlib__Char.cmi stdlib__Char.cmx stdlib__Either.cmi
    stdlib__Either.cmx stdlib__Int.cmi stdlib__Int.cmx stdlib__Int32.cmi
    stdlib__Int32.cmx stdlib__Int64.cmi stdlib__Int64.cmx stdlib__Lazy.cmi
    stdlib__Lazy.cmx stdlib__Lexing.cmi stdlib__Obj.cmi stdlib__Obj.cmx
    stdlib__Seq.cmi stdlib__Seq.cmx stdlib__Sys.cmi stdlib__Sys.cmx
    stdlib__Uchar.cmi stdlib__Uchar.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Lexing.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Lexing.cmx -c ./lexing.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../lexing.own_expected
   lexing.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli)
 (action
  (with-stdout-to
   list.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/list.ml}
    numbered stdlib__List 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Atomic.cmi stdlib__Atomic.cmx stdlib__Either.cmi
    stdlib__Either.cmx stdlib__Int32.cmi stdlib__Int32.cmx
    stdlib__Int64.cmi stdlib__Int64.cmx stdlib__Lazy.cmi stdlib__Lazy.cmx
    stdlib__List.cmi stdlib__Obj.cmi stdlib__Obj.cmx stdlib__Seq.cmi
    stdlib__Seq.cmx stdlib__Sys.cmi stdlib__Sys.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__List.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__List.cmx -c ./list.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../list.own_expected
   list.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.ml
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/listLabels.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli)
 (action
  (with-stdout-to
   listLabels.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/listLabels.ml}
    numbered stdlib__ListLabels 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Atomic.cmi stdlib__Atomic.cmx stdlib__Either.cmi
    stdlib__Either.cmx stdlib__Int32.cmi stdlib__Int32.cmx
    stdlib__Int64.cmi stdlib__Int64.cmx stdlib__Lazy.cmi stdlib__Lazy.cmx
    stdlib__List.cmi stdlib__List.cmx stdlib__ListLabels.cmi
    stdlib__Obj.cmi stdlib__Obj.cmx stdlib__Seq.cmi stdlib__Seq.cmx
    stdlib__Sys.cmi stdlib__Sys.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__ListLabels.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__ListLabels.cmx -c ./listLabels.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../listLabels.own_expected
   listLabels.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.ml
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/map.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli)
 (action
  (with-stdout-to
   map.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/map.ml}
    numbered stdlib__Map 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Atomic.cmi stdlib__Atomic.cmx stdlib__Either.cmi
    stdlib__Either.cmx stdlib__Int32.cmi stdlib__Int32.cmx
    stdlib__Int64.cmi stdlib__Int64.cmx stdlib__Lazy.cmi stdlib__Lazy.cmx
    stdlib__List.cmi stdlib__List.cmx stdlib__Map.cmi stdlib__Obj.cmi
    stdlib__Obj.cmx stdlib__Seq.cmi stdlib__Seq.cmx stdlib__Sys.cmi
    stdlib__Sys.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Map.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Map.cmx -c ./map.ml))))

(rule
 (alias runtest-stdlib-long)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../map.own_expected
   map.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/bytes.ml
  ../../../../../../stdlib/bytes.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/char.ml
  ../../../../../../stdlib/char.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/int.ml
  ../../../../../../stdlib/int.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/marshal.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli
  ../../../../../../stdlib/uchar.ml
  ../../../../../../stdlib/uchar.mli)
 (action
  (with-stdout-to
   marshal.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/marshal.ml}
    numbered stdlib__Marshal 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Atomic.cmi stdlib__Atomic.cmx stdlib__Bytes.cmi
    stdlib__Bytes.cmx stdlib__Char.cmi stdlib__Char.cmx stdlib__Either.cmi
    stdlib__Either.cmx stdlib__Int.cmi stdlib__Int.cmx stdlib__Int32.cmi
    stdlib__Int32.cmx stdlib__Int64.cmi stdlib__Int64.cmx stdlib__Lazy.cmi
    stdlib__Lazy.cmx stdlib__Marshal.cmi stdlib__Obj.cmi stdlib__Obj.cmx
    stdlib__Seq.cmi stdlib__Seq.cmx stdlib__Sys.cmi stdlib__Sys.cmx
    stdlib__Uchar.cmi stdlib__Uchar.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Marshal.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Marshal.cmx -c ./marshal.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../marshal.own_expected
   marshal.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/modes.mli)
 (action
  (with-stdout-to
   modes.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/modes.ml}
    numbered stdlib__Modes 0
    -
    stdlib__Modes.cmi
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Modes.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Modes.cmx -c ./modes.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../modes.own_expected
   modes.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/array.ml
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/bigarray.ml
  ../../../../../../stdlib/bigarray.mli
  ../../../../../../stdlib/buffer.ml
  ../../../../../../stdlib/buffer.mli
  ../../../../../../stdlib/bytes.ml
  ../../../../../../stdlib/bytes.mli
  ../../../../../../stdlib/callback.ml
  ../../../../../../stdlib/callback.mli
  ../../../../../../stdlib/camlinternalFormat.ml
  ../../../../../../stdlib/camlinternalFormat.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/char.ml
  ../../../../../../stdlib/char.mli
  ../../../../../../stdlib/complex.ml
  ../../../../../../stdlib/complex.mli
  ../../../../../../stdlib/condition.ml
  ../../../../../../stdlib/condition.mli
  ../../../../../../stdlib/digest.ml
  ../../../../../../stdlib/digest.mli
  ../../../../../../stdlib/domain.ml
  ../../../../../../stdlib/domain.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/float.ml
  ../../../../../../stdlib/float.mli
  ../../../../../../stdlib/fun.ml
  ../../../../../../stdlib/fun.mli
  ../../../../../../stdlib/hashtbl.ml
  ../../../../../../stdlib/hashtbl.mli
  ../../../../../../stdlib/in_channel.ml
  ../../../../../../stdlib/in_channel.mli
  ../../../../../../stdlib/int.ml
  ../../../../../../stdlib/int.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.ml
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/map.ml
  ../../../../../../stdlib/map.mli
  ../../../../../../stdlib/modes.ml
  ../../../../../../stdlib/modes.mli
  ../../../../../../stdlib/moreLabels.mli
  ../../../../../../stdlib/mutex.ml
  ../../../../../../stdlib/mutex.mli
  ../../../../../../stdlib/nativeint.ml
  ../../../../../../stdlib/nativeint.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/printexc.ml
  ../../../../../../stdlib/printexc.mli
  ../../../../../../stdlib/printf.ml
  ../../../../../../stdlib/printf.mli
  ../../../../../../stdlib/random.ml
  ../../../../../../stdlib/random.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/set.ml
  ../../../../../../stdlib/set.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/string.ml
  ../../../../../../stdlib/string.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli
  ../../../../../../stdlib/uchar.ml
  ../../../../../../stdlib/uchar.mli)
 (action
  (with-stdout-to
   moreLabels.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/moreLabels.ml}
    numbered stdlib__MoreLabels 0
    -
    camlinternalFormat.cmi camlinternalFormat.cmx
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Array.cmi stdlib__Array.cmx stdlib__Atomic.cmi
    stdlib__Atomic.cmx stdlib__Bigarray.cmi stdlib__Bigarray.cmx
    stdlib__Buffer.cmi stdlib__Buffer.cmx stdlib__Bytes.cmi
    stdlib__Bytes.cmx stdlib__Callback.cmi stdlib__Callback.cmx
    stdlib__Char.cmi stdlib__Char.cmx stdlib__Complex.cmi
    stdlib__Complex.cmx stdlib__Condition.cmi stdlib__Condition.cmx
    stdlib__Digest.cmi stdlib__Digest.cmx stdlib__Domain.cmi
    stdlib__Domain.cmx stdlib__Either.cmi stdlib__Either.cmx
    stdlib__Float.cmi stdlib__Float.cmx stdlib__Fun.cmi stdlib__Fun.cmx
    stdlib__Hashtbl.cmi stdlib__Hashtbl.cmx stdlib__In_channel.cmi
    stdlib__In_channel.cmx stdlib__Int.cmi stdlib__Int.cmx
    stdlib__Int32.cmi stdlib__Int32.cmx stdlib__Int64.cmi
    stdlib__Int64.cmx stdlib__Lazy.cmi stdlib__Lazy.cmx stdlib__List.cmi
    stdlib__List.cmx stdlib__Map.cmi stdlib__Map.cmx stdlib__Modes.cmi
    stdlib__Modes.cmx stdlib__MoreLabels.cmi stdlib__Mutex.cmi
    stdlib__Mutex.cmx stdlib__Nativeint.cmi stdlib__Nativeint.cmx
    stdlib__Obj.cmi stdlib__Obj.cmx stdlib__Printexc.cmi
    stdlib__Printexc.cmx stdlib__Printf.cmi stdlib__Printf.cmx
    stdlib__Random.cmi stdlib__Random.cmx stdlib__Seq.cmi stdlib__Seq.cmx
    stdlib__Set.cmi stdlib__Set.cmx stdlib__String.cmi stdlib__String.cmx
    stdlib__Sys.cmi stdlib__Sys.cmx stdlib__Uchar.cmi stdlib__Uchar.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__MoreLabels.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__MoreLabels.cmx -c ./moreLabels.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../moreLabels.own_expected
   moreLabels.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/mutex.mli)
 (action
  (with-stdout-to
   mutex.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/mutex.ml}
    numbered stdlib__Mutex 0
    -
    stdlib__Mutex.cmi
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Mutex.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Mutex.cmx -c ./mutex.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../mutex.own_expected
   mutex.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/nativeint.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli)
 (action
  (with-stdout-to
   nativeint.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/nativeint.ml}
    numbered stdlib__Nativeint 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx stdlib.cmi
    stdlib.cmx stdlib__Nativeint.cmi stdlib__Sys.cmi stdlib__Sys.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Nativeint.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Nativeint.cmx -c ./nativeint.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../nativeint.own_expected
   nativeint.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli)
 (action
  (with-stdout-to
   obj.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/obj.ml}
    numbered stdlib__Obj 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx stdlib.cmi
    stdlib.cmx stdlib__Int32.cmi stdlib__Int32.cmx stdlib__Int64.cmi
    stdlib__Int64.cmx stdlib__Obj.cmi stdlib__Sys.cmi stdlib__Sys.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Obj.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Obj.cmx -c ./obj.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../obj.own_expected
   obj.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/array.ml
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/bytes.ml
  ../../../../../../stdlib/bytes.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/camlinternalOO.ml
  ../../../../../../stdlib/camlinternalOO.mli
  ../../../../../../stdlib/char.ml
  ../../../../../../stdlib/char.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/int.ml
  ../../../../../../stdlib/int.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.ml
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/map.ml
  ../../../../../../stdlib/map.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/oo.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/string.ml
  ../../../../../../stdlib/string.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli
  ../../../../../../stdlib/uchar.ml
  ../../../../../../stdlib/uchar.mli)
 (action
  (with-stdout-to
   oo.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/oo.ml}
    numbered stdlib__Oo 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx camlinternalOO.cmi
    camlinternalOO.cmx stdlib.cmi stdlib.cmx stdlib__Array.cmi
    stdlib__Array.cmx stdlib__Atomic.cmi stdlib__Atomic.cmx
    stdlib__Bytes.cmi stdlib__Bytes.cmx stdlib__Char.cmi stdlib__Char.cmx
    stdlib__Either.cmi stdlib__Either.cmx stdlib__Int.cmi stdlib__Int.cmx
    stdlib__Int32.cmi stdlib__Int32.cmx stdlib__Int64.cmi
    stdlib__Int64.cmx stdlib__Lazy.cmi stdlib__Lazy.cmx stdlib__List.cmi
    stdlib__List.cmx stdlib__Map.cmi stdlib__Map.cmx stdlib__Obj.cmi
    stdlib__Obj.cmx stdlib__Oo.cmi stdlib__Seq.cmi stdlib__Seq.cmx
    stdlib__String.cmi stdlib__String.cmx stdlib__Sys.cmi stdlib__Sys.cmx
    stdlib__Uchar.cmi stdlib__Uchar.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Oo.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Oo.cmx -c ./oo.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../oo.own_expected
   oo.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/option.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli)
 (action
  (with-stdout-to
   option.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/option.ml}
    numbered stdlib__Option 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Atomic.cmi stdlib__Atomic.cmx stdlib__Either.cmi
    stdlib__Either.cmx stdlib__Int32.cmi stdlib__Int32.cmx
    stdlib__Int64.cmi stdlib__Int64.cmx stdlib__Lazy.cmi stdlib__Lazy.cmx
    stdlib__Obj.cmi stdlib__Obj.cmx stdlib__Option.cmi stdlib__Seq.cmi
    stdlib__Seq.cmx stdlib__Sys.cmi stdlib__Sys.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Option.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Option.cmx -c ./option.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../option.own_expected
   option.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/array.ml
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/bigarray.ml
  ../../../../../../stdlib/bigarray.mli
  ../../../../../../stdlib/buffer.ml
  ../../../../../../stdlib/buffer.mli
  ../../../../../../stdlib/bytes.ml
  ../../../../../../stdlib/bytes.mli
  ../../../../../../stdlib/camlinternalFormat.ml
  ../../../../../../stdlib/camlinternalFormat.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/char.ml
  ../../../../../../stdlib/char.mli
  ../../../../../../stdlib/complex.ml
  ../../../../../../stdlib/complex.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/float.ml
  ../../../../../../stdlib/float.mli
  ../../../../../../stdlib/fun.ml
  ../../../../../../stdlib/fun.mli
  ../../../../../../stdlib/int.ml
  ../../../../../../stdlib/int.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.ml
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/modes.ml
  ../../../../../../stdlib/modes.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/out_channel.mli
  ../../../../../../stdlib/printexc.ml
  ../../../../../../stdlib/printexc.mli
  ../../../../../../stdlib/printf.ml
  ../../../../../../stdlib/printf.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/string.ml
  ../../../../../../stdlib/string.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli
  ../../../../../../stdlib/uchar.ml
  ../../../../../../stdlib/uchar.mli)
 (action
  (with-stdout-to
   out_channel.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/out_channel.ml}
    numbered stdlib__Out_channel 0
    -
    camlinternalFormat.cmi camlinternalFormat.cmx
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Array.cmi stdlib__Array.cmx stdlib__Atomic.cmi
    stdlib__Atomic.cmx stdlib__Bigarray.cmi stdlib__Bigarray.cmx
    stdlib__Buffer.cmi stdlib__Buffer.cmx stdlib__Bytes.cmi
    stdlib__Bytes.cmx stdlib__Char.cmi stdlib__Char.cmx
    stdlib__Complex.cmi stdlib__Complex.cmx stdlib__Either.cmi
    stdlib__Either.cmx stdlib__Float.cmi stdlib__Float.cmx stdlib__Fun.cmi
    stdlib__Fun.cmx stdlib__Int.cmi stdlib__Int.cmx stdlib__Int32.cmi
    stdlib__Int32.cmx stdlib__Int64.cmi stdlib__Int64.cmx stdlib__Lazy.cmi
    stdlib__Lazy.cmx stdlib__List.cmi stdlib__List.cmx stdlib__Modes.cmi
    stdlib__Modes.cmx stdlib__Obj.cmi stdlib__Obj.cmx
    stdlib__Out_channel.cmi stdlib__Printexc.cmi stdlib__Printexc.cmx
    stdlib__Printf.cmi stdlib__Printf.cmx stdlib__Seq.cmi stdlib__Seq.cmx
    stdlib__String.cmi stdlib__String.cmx stdlib__Sys.cmi stdlib__Sys.cmx
    stdlib__Uchar.cmi stdlib__Uchar.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Out_channel.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Out_channel.cmx -c ./out_channel.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../out_channel.own_expected
   out_channel.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/pair.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli)
 (action
  (with-stdout-to
   pair.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/pair.ml}
    numbered stdlib__Pair 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx stdlib.cmi
    stdlib.cmx stdlib__Pair.cmi
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Pair.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Pair.cmx -c ./pair.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../pair.own_expected
   pair.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/array.ml
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/bytes.ml
  ../../../../../../stdlib/bytes.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/char.ml
  ../../../../../../stdlib/char.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/int.ml
  ../../../../../../stdlib/int.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/lexing.ml
  ../../../../../../stdlib/lexing.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/parsing.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli
  ../../../../../../stdlib/uchar.ml
  ../../../../../../stdlib/uchar.mli)
 (action
  (with-stdout-to
   parsing.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/parsing.ml}
    numbered stdlib__Parsing 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Array.cmi stdlib__Array.cmx stdlib__Atomic.cmi
    stdlib__Atomic.cmx stdlib__Bytes.cmi stdlib__Bytes.cmx
    stdlib__Char.cmi stdlib__Char.cmx stdlib__Either.cmi
    stdlib__Either.cmx stdlib__Int.cmi stdlib__Int.cmx stdlib__Int32.cmi
    stdlib__Int32.cmx stdlib__Int64.cmi stdlib__Int64.cmx stdlib__Lazy.cmi
    stdlib__Lazy.cmx stdlib__Lexing.cmi stdlib__Lexing.cmx stdlib__Obj.cmi
    stdlib__Obj.cmx stdlib__Parsing.cmi stdlib__Seq.cmi stdlib__Seq.cmx
    stdlib__Sys.cmi stdlib__Sys.cmx stdlib__Uchar.cmi stdlib__Uchar.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Parsing.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Parsing.cmx -c ./parsing.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../parsing.own_expected
   parsing.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/array.ml
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/buffer.ml
  ../../../../../../stdlib/buffer.mli
  ../../../../../../stdlib/bytes.ml
  ../../../../../../stdlib/bytes.mli
  ../../../../../../stdlib/camlinternalFormat.ml
  ../../../../../../stdlib/camlinternalFormat.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/camlinternalOO.ml
  ../../../../../../stdlib/camlinternalOO.mli
  ../../../../../../stdlib/char.ml
  ../../../../../../stdlib/char.mli
  ../../../../../../stdlib/dynarray.ml
  ../../../../../../stdlib/dynarray.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/int.ml
  ../../../../../../stdlib/int.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.ml
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/map.ml
  ../../../../../../stdlib/map.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/pqueue.mli
  ../../../../../../stdlib/printf.ml
  ../../../../../../stdlib/printf.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/string.ml
  ../../../../../../stdlib/string.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli
  ../../../../../../stdlib/uchar.ml
  ../../../../../../stdlib/uchar.mli)
 (action
  (with-stdout-to
   pqueue.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/pqueue.ml}
    numbered stdlib__Pqueue 0
    -
    camlinternalFormat.cmi camlinternalFormat.cmx
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx camlinternalOO.cmi
    camlinternalOO.cmx stdlib.cmi stdlib.cmx stdlib__Array.cmi
    stdlib__Array.cmx stdlib__Atomic.cmi stdlib__Atomic.cmx
    stdlib__Buffer.cmi stdlib__Buffer.cmx stdlib__Bytes.cmi
    stdlib__Bytes.cmx stdlib__Char.cmi stdlib__Char.cmx
    stdlib__Dynarray.cmi stdlib__Dynarray.cmx stdlib__Either.cmi
    stdlib__Either.cmx stdlib__Int.cmi stdlib__Int.cmx stdlib__Int32.cmi
    stdlib__Int32.cmx stdlib__Int64.cmi stdlib__Int64.cmx stdlib__Lazy.cmi
    stdlib__Lazy.cmx stdlib__List.cmi stdlib__List.cmx stdlib__Map.cmi
    stdlib__Map.cmx stdlib__Obj.cmi stdlib__Obj.cmx stdlib__Pqueue.cmi
    stdlib__Printf.cmi stdlib__Printf.cmx stdlib__Seq.cmi stdlib__Seq.cmx
    stdlib__String.cmi stdlib__String.cmx stdlib__Sys.cmi stdlib__Sys.cmx
    stdlib__Uchar.cmi stdlib__Uchar.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Pqueue.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Pqueue.cmx -c ./pqueue.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../pqueue.own_expected
   pqueue.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/array.ml
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/buffer.ml
  ../../../../../../stdlib/buffer.mli
  ../../../../../../stdlib/bytes.ml
  ../../../../../../stdlib/bytes.mli
  ../../../../../../stdlib/camlinternalFormat.ml
  ../../../../../../stdlib/camlinternalFormat.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/char.ml
  ../../../../../../stdlib/char.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/int.ml
  ../../../../../../stdlib/int.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.ml
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/modes.ml
  ../../../../../../stdlib/modes.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/printexc.mli
  ../../../../../../stdlib/printf.ml
  ../../../../../../stdlib/printf.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/string.ml
  ../../../../../../stdlib/string.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli
  ../../../../../../stdlib/uchar.ml
  ../../../../../../stdlib/uchar.mli)
 (action
  (with-stdout-to
   printexc.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/printexc.ml}
    numbered stdlib__Printexc 0
    -
    camlinternalFormat.cmi camlinternalFormat.cmx
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Array.cmi stdlib__Array.cmx stdlib__Atomic.cmi
    stdlib__Atomic.cmx stdlib__Buffer.cmi stdlib__Buffer.cmx
    stdlib__Bytes.cmi stdlib__Bytes.cmx stdlib__Char.cmi stdlib__Char.cmx
    stdlib__Either.cmi stdlib__Either.cmx stdlib__Int.cmi stdlib__Int.cmx
    stdlib__Int32.cmi stdlib__Int32.cmx stdlib__Int64.cmi
    stdlib__Int64.cmx stdlib__Lazy.cmi stdlib__Lazy.cmx stdlib__List.cmi
    stdlib__List.cmx stdlib__Modes.cmi stdlib__Modes.cmx stdlib__Obj.cmi
    stdlib__Obj.cmx stdlib__Printexc.cmi stdlib__Printf.cmi
    stdlib__Printf.cmx stdlib__Seq.cmi stdlib__Seq.cmx stdlib__String.cmi
    stdlib__String.cmx stdlib__Sys.cmi stdlib__Sys.cmx stdlib__Uchar.cmi
    stdlib__Uchar.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Printexc.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Printexc.cmx -c ./printexc.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../printexc.own_expected
   printexc.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/array.ml
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/buffer.ml
  ../../../../../../stdlib/buffer.mli
  ../../../../../../stdlib/bytes.ml
  ../../../../../../stdlib/bytes.mli
  ../../../../../../stdlib/camlinternalFormat.ml
  ../../../../../../stdlib/camlinternalFormat.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/char.ml
  ../../../../../../stdlib/char.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/int.ml
  ../../../../../../stdlib/int.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.ml
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/printf.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/string.ml
  ../../../../../../stdlib/string.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli
  ../../../../../../stdlib/uchar.ml
  ../../../../../../stdlib/uchar.mli)
 (action
  (with-stdout-to
   printf.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/printf.ml}
    numbered stdlib__Printf 0
    -
    camlinternalFormat.cmi camlinternalFormat.cmx
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Array.cmi stdlib__Array.cmx stdlib__Atomic.cmi
    stdlib__Atomic.cmx stdlib__Buffer.cmi stdlib__Buffer.cmx
    stdlib__Bytes.cmi stdlib__Bytes.cmx stdlib__Char.cmi stdlib__Char.cmx
    stdlib__Either.cmi stdlib__Either.cmx stdlib__Int.cmi stdlib__Int.cmx
    stdlib__Int32.cmi stdlib__Int32.cmx stdlib__Int64.cmi
    stdlib__Int64.cmx stdlib__Lazy.cmi stdlib__Lazy.cmx stdlib__List.cmi
    stdlib__List.cmx stdlib__Obj.cmi stdlib__Obj.cmx stdlib__Printf.cmi
    stdlib__Seq.cmi stdlib__Seq.cmx stdlib__String.cmi stdlib__String.cmx
    stdlib__Sys.cmi stdlib__Sys.cmx stdlib__Uchar.cmi stdlib__Uchar.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Printf.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Printf.cmx -c ./printf.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../printf.own_expected
   printf.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/queue.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli)
 (action
  (with-stdout-to
   queue.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/queue.ml}
    numbered stdlib__Queue 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Atomic.cmi stdlib__Atomic.cmx stdlib__Either.cmi
    stdlib__Either.cmx stdlib__Int32.cmi stdlib__Int32.cmx
    stdlib__Int64.cmi stdlib__Int64.cmx stdlib__Lazy.cmi stdlib__Lazy.cmx
    stdlib__Obj.cmi stdlib__Obj.cmx stdlib__Queue.cmi stdlib__Seq.cmi
    stdlib__Seq.cmx stdlib__Sys.cmi stdlib__Sys.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Queue.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Queue.cmx -c ./queue.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../queue.own_expected
   queue.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/array.ml
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/bigarray.ml
  ../../../../../../stdlib/bigarray.mli
  ../../../../../../stdlib/buffer.ml
  ../../../../../../stdlib/buffer.mli
  ../../../../../../stdlib/bytes.ml
  ../../../../../../stdlib/bytes.mli
  ../../../../../../stdlib/callback.ml
  ../../../../../../stdlib/callback.mli
  ../../../../../../stdlib/camlinternalFormat.ml
  ../../../../../../stdlib/camlinternalFormat.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/camlinternalQuote.ml
  ../../../../../../stdlib/camlinternalQuote.mli
  ../../../../../../stdlib/char.ml
  ../../../../../../stdlib/char.mli
  ../../../../../../stdlib/complex.ml
  ../../../../../../stdlib/complex.mli
  ../../../../../../stdlib/condition.ml
  ../../../../../../stdlib/condition.mli
  ../../../../../../stdlib/digest.ml
  ../../../../../../stdlib/digest.mli
  ../../../../../../stdlib/domain.ml
  ../../../../../../stdlib/domain.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/float.ml
  ../../../../../../stdlib/float.mli
  ../../../../../../stdlib/format.ml
  ../../../../../../stdlib/format.mli
  ../../../../../../stdlib/fun.ml
  ../../../../../../stdlib/fun.mli
  ../../../../../../stdlib/hashtbl.ml
  ../../../../../../stdlib/hashtbl.mli
  ../../../../../../stdlib/in_channel.ml
  ../../../../../../stdlib/in_channel.mli
  ../../../../../../stdlib/int.ml
  ../../../../../../stdlib/int.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.ml
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/map.ml
  ../../../../../../stdlib/map.mli
  ../../../../../../stdlib/modes.ml
  ../../../../../../stdlib/modes.mli
  ../../../../../../stdlib/mutex.ml
  ../../../../../../stdlib/mutex.mli
  ../../../../../../stdlib/nativeint.ml
  ../../../../../../stdlib/nativeint.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/option.ml
  ../../../../../../stdlib/option.mli
  ../../../../../../stdlib/printexc.ml
  ../../../../../../stdlib/printexc.mli
  ../../../../../../stdlib/printf.ml
  ../../../../../../stdlib/printf.mli
  ../../../../../../stdlib/queue.ml
  ../../../../../../stdlib/queue.mli
  ../../../../../../stdlib/quote.mli
  ../../../../../../stdlib/random.ml
  ../../../../../../stdlib/random.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stack.ml
  ../../../../../../stdlib/stack.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/string.ml
  ../../../../../../stdlib/string.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli
  ../../../../../../stdlib/uchar.ml
  ../../../../../../stdlib/uchar.mli)
 (action
  (with-stdout-to
   quote.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/quote.ml}
    numbered stdlib__Quote 0
    -
    camlinternalFormat.cmi camlinternalFormat.cmx
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx camlinternalQuote.cmi
    camlinternalQuote.cmx stdlib.cmi stdlib.cmx stdlib__Array.cmi
    stdlib__Array.cmx stdlib__Atomic.cmi stdlib__Atomic.cmx
    stdlib__Bigarray.cmi stdlib__Bigarray.cmx stdlib__Buffer.cmi
    stdlib__Buffer.cmx stdlib__Bytes.cmi stdlib__Bytes.cmx
    stdlib__Callback.cmi stdlib__Callback.cmx stdlib__Char.cmi
    stdlib__Char.cmx stdlib__Complex.cmi stdlib__Complex.cmx
    stdlib__Condition.cmi stdlib__Condition.cmx stdlib__Digest.cmi
    stdlib__Digest.cmx stdlib__Domain.cmi stdlib__Domain.cmx
    stdlib__Either.cmi stdlib__Either.cmx stdlib__Float.cmi
    stdlib__Float.cmx stdlib__Format.cmi stdlib__Format.cmx
    stdlib__Fun.cmi stdlib__Fun.cmx stdlib__Hashtbl.cmi
    stdlib__Hashtbl.cmx stdlib__In_channel.cmi stdlib__In_channel.cmx
    stdlib__Int.cmi stdlib__Int.cmx stdlib__Int32.cmi stdlib__Int32.cmx
    stdlib__Int64.cmi stdlib__Int64.cmx stdlib__Lazy.cmi stdlib__Lazy.cmx
    stdlib__List.cmi stdlib__List.cmx stdlib__Map.cmi stdlib__Map.cmx
    stdlib__Modes.cmi stdlib__Modes.cmx stdlib__Mutex.cmi
    stdlib__Mutex.cmx stdlib__Nativeint.cmi stdlib__Nativeint.cmx
    stdlib__Obj.cmi stdlib__Obj.cmx stdlib__Option.cmi stdlib__Option.cmx
    stdlib__Printexc.cmi stdlib__Printexc.cmx stdlib__Printf.cmi
    stdlib__Printf.cmx stdlib__Queue.cmi stdlib__Queue.cmx
    stdlib__Quote.cmi stdlib__Random.cmi stdlib__Random.cmx
    stdlib__Seq.cmi stdlib__Seq.cmx stdlib__Stack.cmi stdlib__Stack.cmx
    stdlib__String.cmi stdlib__String.cmx stdlib__Sys.cmi stdlib__Sys.cmx
    stdlib__Uchar.cmi stdlib__Uchar.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Quote.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Quote.cmx -c ./quote.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../quote.own_expected
   quote.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/array.ml
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/bigarray.ml
  ../../../../../../stdlib/bigarray.mli
  ../../../../../../stdlib/buffer.ml
  ../../../../../../stdlib/buffer.mli
  ../../../../../../stdlib/bytes.ml
  ../../../../../../stdlib/bytes.mli
  ../../../../../../stdlib/callback.ml
  ../../../../../../stdlib/callback.mli
  ../../../../../../stdlib/camlinternalFormat.ml
  ../../../../../../stdlib/camlinternalFormat.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/char.ml
  ../../../../../../stdlib/char.mli
  ../../../../../../stdlib/complex.ml
  ../../../../../../stdlib/complex.mli
  ../../../../../../stdlib/condition.ml
  ../../../../../../stdlib/condition.mli
  ../../../../../../stdlib/digest.ml
  ../../../../../../stdlib/digest.mli
  ../../../../../../stdlib/domain.ml
  ../../../../../../stdlib/domain.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/float.ml
  ../../../../../../stdlib/float.mli
  ../../../../../../stdlib/fun.ml
  ../../../../../../stdlib/fun.mli
  ../../../../../../stdlib/in_channel.ml
  ../../../../../../stdlib/in_channel.mli
  ../../../../../../stdlib/int.ml
  ../../../../../../stdlib/int.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.ml
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/map.ml
  ../../../../../../stdlib/map.mli
  ../../../../../../stdlib/modes.ml
  ../../../../../../stdlib/modes.mli
  ../../../../../../stdlib/mutex.ml
  ../../../../../../stdlib/mutex.mli
  ../../../../../../stdlib/nativeint.ml
  ../../../../../../stdlib/nativeint.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/printexc.ml
  ../../../../../../stdlib/printexc.mli
  ../../../../../../stdlib/printf.ml
  ../../../../../../stdlib/printf.mli
  ../../../../../../stdlib/random.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/string.ml
  ../../../../../../stdlib/string.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli
  ../../../../../../stdlib/uchar.ml
  ../../../../../../stdlib/uchar.mli)
 (action
  (with-stdout-to
   random.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/random.ml}
    numbered stdlib__Random 0
    -
    camlinternalFormat.cmi camlinternalFormat.cmx
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Array.cmi stdlib__Array.cmx stdlib__Atomic.cmi
    stdlib__Atomic.cmx stdlib__Bigarray.cmi stdlib__Bigarray.cmx
    stdlib__Buffer.cmi stdlib__Buffer.cmx stdlib__Bytes.cmi
    stdlib__Bytes.cmx stdlib__Callback.cmi stdlib__Callback.cmx
    stdlib__Char.cmi stdlib__Char.cmx stdlib__Complex.cmi
    stdlib__Complex.cmx stdlib__Condition.cmi stdlib__Condition.cmx
    stdlib__Digest.cmi stdlib__Digest.cmx stdlib__Domain.cmi
    stdlib__Domain.cmx stdlib__Either.cmi stdlib__Either.cmx
    stdlib__Float.cmi stdlib__Float.cmx stdlib__Fun.cmi stdlib__Fun.cmx
    stdlib__In_channel.cmi stdlib__In_channel.cmx stdlib__Int.cmi
    stdlib__Int.cmx stdlib__Int32.cmi stdlib__Int32.cmx stdlib__Int64.cmi
    stdlib__Int64.cmx stdlib__Lazy.cmi stdlib__Lazy.cmx stdlib__List.cmi
    stdlib__List.cmx stdlib__Map.cmi stdlib__Map.cmx stdlib__Modes.cmi
    stdlib__Modes.cmx stdlib__Mutex.cmi stdlib__Mutex.cmx
    stdlib__Nativeint.cmi stdlib__Nativeint.cmx stdlib__Obj.cmi
    stdlib__Obj.cmx stdlib__Printexc.cmi stdlib__Printexc.cmx
    stdlib__Printf.cmi stdlib__Printf.cmx stdlib__Random.cmi
    stdlib__Seq.cmi stdlib__Seq.cmx stdlib__String.cmi stdlib__String.cmx
    stdlib__Sys.cmi stdlib__Sys.cmx stdlib__Uchar.cmi stdlib__Uchar.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Random.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Random.cmx -c ./random.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../random.own_expected
   random.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/repr.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli)
 (action
  (with-stdout-to
   repr.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/repr.ml}
    numbered stdlib__Repr 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx stdlib.cmi
    stdlib.cmx stdlib__Repr.cmi
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Repr.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Repr.cmx -c ./repr.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../repr.own_expected
   repr.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/result.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli)
 (action
  (with-stdout-to
   result.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/result.ml}
    numbered stdlib__Result 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Atomic.cmi stdlib__Atomic.cmx stdlib__Either.cmi
    stdlib__Either.cmx stdlib__Int32.cmi stdlib__Int32.cmx
    stdlib__Int64.cmi stdlib__Int64.cmx stdlib__Lazy.cmi stdlib__Lazy.cmx
    stdlib__Obj.cmi stdlib__Obj.cmx stdlib__Result.cmi stdlib__Seq.cmi
    stdlib__Seq.cmx stdlib__Sys.cmi stdlib__Sys.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Result.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Result.cmx -c ./result.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../result.own_expected
   result.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/array.ml
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/buffer.ml
  ../../../../../../stdlib/buffer.mli
  ../../../../../../stdlib/bytes.ml
  ../../../../../../stdlib/bytes.mli
  ../../../../../../stdlib/camlinternalFormat.ml
  ../../../../../../stdlib/camlinternalFormat.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/char.ml
  ../../../../../../stdlib/char.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/int.ml
  ../../../../../../stdlib/int.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.ml
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/printf.ml
  ../../../../../../stdlib/printf.mli
  ../../../../../../stdlib/scanf.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/string.ml
  ../../../../../../stdlib/string.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli
  ../../../../../../stdlib/uchar.ml
  ../../../../../../stdlib/uchar.mli)
 (action
  (with-stdout-to
   scanf.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/scanf.ml}
    numbered stdlib__Scanf 0
    -
    camlinternalFormat.cmi camlinternalFormat.cmx
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Array.cmi stdlib__Array.cmx stdlib__Atomic.cmi
    stdlib__Atomic.cmx stdlib__Buffer.cmi stdlib__Buffer.cmx
    stdlib__Bytes.cmi stdlib__Bytes.cmx stdlib__Char.cmi stdlib__Char.cmx
    stdlib__Either.cmi stdlib__Either.cmx stdlib__Int.cmi stdlib__Int.cmx
    stdlib__Int32.cmi stdlib__Int32.cmx stdlib__Int64.cmi
    stdlib__Int64.cmx stdlib__Lazy.cmi stdlib__Lazy.cmx stdlib__List.cmi
    stdlib__List.cmx stdlib__Obj.cmi stdlib__Obj.cmx stdlib__Printf.cmi
    stdlib__Printf.cmx stdlib__Scanf.cmi stdlib__Seq.cmi stdlib__Seq.cmx
    stdlib__String.cmi stdlib__String.cmx stdlib__Sys.cmi stdlib__Sys.cmx
    stdlib__Uchar.cmi stdlib__Uchar.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Scanf.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Scanf.cmx -c ./scanf.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../scanf.own_expected
   scanf.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/condition.ml
  ../../../../../../stdlib/condition.mli
  ../../../../../../stdlib/mutex.ml
  ../../../../../../stdlib/mutex.mli
  ../../../../../../stdlib/semaphore.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli)
 (action
  (with-stdout-to
   semaphore.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/semaphore.ml}
    numbered stdlib__Semaphore 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx stdlib.cmi
    stdlib.cmx stdlib__Condition.cmi stdlib__Condition.cmx
    stdlib__Mutex.cmi stdlib__Mutex.cmx stdlib__Semaphore.cmi
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Semaphore.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Semaphore.cmx -c ./semaphore.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../semaphore.own_expected
   semaphore.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli)
 (action
  (with-stdout-to
   seq.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/seq.ml}
    numbered stdlib__Seq 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Atomic.cmi stdlib__Atomic.cmx stdlib__Either.cmi
    stdlib__Either.cmx stdlib__Int32.cmi stdlib__Int32.cmx
    stdlib__Int64.cmi stdlib__Int64.cmx stdlib__Lazy.cmi stdlib__Lazy.cmx
    stdlib__Obj.cmi stdlib__Obj.cmx stdlib__Seq.cmi stdlib__Sys.cmi
    stdlib__Sys.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Seq.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Seq.cmx -c ./seq.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../seq.own_expected
   seq.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.ml
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/set.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli)
 (action
  (with-stdout-to
   set.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/set.ml}
    numbered stdlib__Set 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Atomic.cmi stdlib__Atomic.cmx stdlib__Either.cmi
    stdlib__Either.cmx stdlib__Int32.cmi stdlib__Int32.cmx
    stdlib__Int64.cmi stdlib__Int64.cmx stdlib__Lazy.cmi stdlib__Lazy.cmx
    stdlib__List.cmi stdlib__List.cmx stdlib__Obj.cmi stdlib__Obj.cmx
    stdlib__Seq.cmi stdlib__Seq.cmx stdlib__Set.cmi stdlib__Sys.cmi
    stdlib__Sys.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Set.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Set.cmx -c ./set.ml))))

(rule
 (alias runtest-stdlib-long)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../set.own_expected
   set.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.ml
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stack.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli)
 (action
  (with-stdout-to
   stack.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/stack.ml}
    numbered stdlib__Stack 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Atomic.cmi stdlib__Atomic.cmx stdlib__Either.cmi
    stdlib__Either.cmx stdlib__Int32.cmi stdlib__Int32.cmx
    stdlib__Int64.cmi stdlib__Int64.cmx stdlib__Lazy.cmi stdlib__Lazy.cmx
    stdlib__List.cmi stdlib__List.cmx stdlib__Obj.cmi stdlib__Obj.cmx
    stdlib__Seq.cmi stdlib__Seq.cmx stdlib__Stack.cmi stdlib__Sys.cmi
    stdlib__Sys.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Stack.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Stack.cmx -c ./stack.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../stack.own_expected
   stack.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/stdLabels.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli)
 (action
  (with-stdout-to
   stdLabels.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/stdLabels.ml}
    numbered stdlib__StdLabels 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx stdlib.cmi
    stdlib.cmx stdlib__StdLabels.cmi
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__StdLabels.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__StdLabels.cmx -c ./stdLabels.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../stdLabels.own_expected
   stdLabels.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/std_exit.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli)
 (action
  (with-stdout-to
   std_exit.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/std_exit.ml}
    numbered std_exit 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx std_exit.cmi
    stdlib.cmi stdlib.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file std_exit.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    std_exit.cmx -c ./std_exit.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../std_exit.own_expected
   std_exit.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/stdlib.mli)
 (action
  (with-stdout-to
   stdlib.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/stdlib.ml}
    numbered stdlib 0
    %{dep:../../../../../../stdlib/expand_module_aliases.awk}
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx stdlib.cmi
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -pp "awk -f ../expand_module_aliases.awk" -cmi-file
    stdlib.cmi %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt}
    -o stdlib.cmx -c ./stdlib.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../stdlib.own_expected
   stdlib.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/array.ml
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/bytes.ml
  ../../../../../../stdlib/bytes.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/char.ml
  ../../../../../../stdlib/char.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/int.ml
  ../../../../../../stdlib/int.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.ml
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/string.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli
  ../../../../../../stdlib/uchar.ml
  ../../../../../../stdlib/uchar.mli)
 (action
  (with-stdout-to
   string.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/string.ml}
    numbered stdlib__String 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Array.cmi stdlib__Array.cmx stdlib__Atomic.cmi
    stdlib__Atomic.cmx stdlib__Bytes.cmi stdlib__Bytes.cmx
    stdlib__Char.cmi stdlib__Char.cmx stdlib__Either.cmi
    stdlib__Either.cmx stdlib__Int.cmi stdlib__Int.cmx stdlib__Int32.cmi
    stdlib__Int32.cmx stdlib__Int64.cmi stdlib__Int64.cmx stdlib__Lazy.cmi
    stdlib__Lazy.cmx stdlib__List.cmi stdlib__List.cmx stdlib__Obj.cmi
    stdlib__Obj.cmx stdlib__Seq.cmi stdlib__Seq.cmx stdlib__String.cmi
    stdlib__Sys.cmi stdlib__Sys.cmx stdlib__Uchar.cmi stdlib__Uchar.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__String.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__String.cmx -c ./string.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../string.own_expected
   string.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/array.ml
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/bytes.ml
  ../../../../../../stdlib/bytes.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/char.ml
  ../../../../../../stdlib/char.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/int.ml
  ../../../../../../stdlib/int.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.ml
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/string.ml
  ../../../../../../stdlib/string.mli
  ../../../../../../stdlib/stringLabels.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli
  ../../../../../../stdlib/uchar.ml
  ../../../../../../stdlib/uchar.mli)
 (action
  (with-stdout-to
   stringLabels.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/stringLabels.ml}
    numbered stdlib__StringLabels 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Array.cmi stdlib__Array.cmx stdlib__Atomic.cmi
    stdlib__Atomic.cmx stdlib__Bytes.cmi stdlib__Bytes.cmx
    stdlib__Char.cmi stdlib__Char.cmx stdlib__Either.cmi
    stdlib__Either.cmx stdlib__Int.cmi stdlib__Int.cmx stdlib__Int32.cmi
    stdlib__Int32.cmx stdlib__Int64.cmi stdlib__Int64.cmx stdlib__Lazy.cmi
    stdlib__Lazy.cmx stdlib__List.cmi stdlib__List.cmx stdlib__Obj.cmi
    stdlib__Obj.cmx stdlib__Seq.cmi stdlib__Seq.cmx stdlib__String.cmi
    stdlib__String.cmx stdlib__StringLabels.cmi stdlib__Sys.cmi
    stdlib__Sys.cmx stdlib__Uchar.cmi stdlib__Uchar.cmx
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__StringLabels.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__StringLabels.cmx -c ./stringLabels.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../stringLabels.own_expected
   stringLabels.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/sys.mli)
 (action
  (with-stdout-to
   sys.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/sys.ml}
    copied stdlib__Sys 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx stdlib.cmi
    stdlib.cmx stdlib__Sys.cmi
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Sys.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Sys.cmx -c ./sys.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../sys.own_expected
   sys.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli
  ../../../../../../stdlib/type.mli)
 (action
  (with-stdout-to
   type.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/type.ml}
    numbered stdlib__Type 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx stdlib.cmi
    stdlib.cmx stdlib__Int32.cmi stdlib__Int32.cmx stdlib__Int64.cmi
    stdlib__Int64.cmx stdlib__Obj.cmi stdlib__Obj.cmx stdlib__Sys.cmi
    stdlib__Sys.cmx stdlib__Type.cmi
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Type.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Type.cmx -c ./type.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../type.own_expected
   type.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/char.ml
  ../../../../../../stdlib/char.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/uchar.mli)
 (action
  (with-stdout-to
   uchar.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/uchar.ml}
    numbered stdlib__Uchar 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx stdlib.cmi
    stdlib.cmx stdlib__Char.cmi stdlib__Char.cmx stdlib__Uchar.cmi
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Uchar.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Uchar.cmx -c ./uchar.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../uchar.own_expected
   uchar.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/unit.mli)
 (action
  (with-stdout-to
   unit.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/unit.ml}
    numbered stdlib__Unit 0
    -
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx stdlib.cmi
    stdlib.cmx stdlib__Unit.cmi
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Unit.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Unit.cmx -c ./unit.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../unit.own_expected
   unit.own_output)))

(rule
 (enabled_if
  (= %{context_name} "main"))
 (deps
  %{ocamlopt}
  (glob_files %{ocaml_where}/*.cmi)
  (glob_files %{ocaml_where}/*.cmx)
  (glob_files %{ocaml_where}/*.o)
  (glob_files %{ocaml_where}/*.ml)
  ../../../../../../stdlib/array.ml
  ../../../../../../stdlib/array.mli
  ../../../../../../stdlib/atomic.ml
  ../../../../../../stdlib/atomic.mli
  ../../../../../../stdlib/bigarray.ml
  ../../../../../../stdlib/bigarray.mli
  ../../../../../../stdlib/buffer.ml
  ../../../../../../stdlib/buffer.mli
  ../../../../../../stdlib/bytes.ml
  ../../../../../../stdlib/bytes.mli
  ../../../../../../stdlib/callback.ml
  ../../../../../../stdlib/callback.mli
  ../../../../../../stdlib/camlinternalFormat.ml
  ../../../../../../stdlib/camlinternalFormat.mli
  ../../../../../../stdlib/camlinternalFormatBasics.ml
  ../../../../../../stdlib/camlinternalFormatBasics.mli
  ../../../../../../stdlib/camlinternalLazy.ml
  ../../../../../../stdlib/camlinternalLazy.mli
  ../../../../../../stdlib/char.ml
  ../../../../../../stdlib/char.mli
  ../../../../../../stdlib/complex.ml
  ../../../../../../stdlib/complex.mli
  ../../../../../../stdlib/condition.ml
  ../../../../../../stdlib/condition.mli
  ../../../../../../stdlib/digest.ml
  ../../../../../../stdlib/digest.mli
  ../../../../../../stdlib/domain.ml
  ../../../../../../stdlib/domain.mli
  ../../../../../../stdlib/either.ml
  ../../../../../../stdlib/either.mli
  ../../../../../../stdlib/float.ml
  ../../../../../../stdlib/float.mli
  ../../../../../../stdlib/fun.ml
  ../../../../../../stdlib/fun.mli
  ../../../../../../stdlib/hashtbl.ml
  ../../../../../../stdlib/hashtbl.mli
  ../../../../../../stdlib/in_channel.ml
  ../../../../../../stdlib/in_channel.mli
  ../../../../../../stdlib/int.ml
  ../../../../../../stdlib/int.mli
  ../../../../../../stdlib/int32.ml
  ../../../../../../stdlib/int32.mli
  ../../../../../../stdlib/int64.ml
  ../../../../../../stdlib/int64.mli
  ../../../../../../stdlib/lazy.ml
  ../../../../../../stdlib/lazy.mli
  ../../../../../../stdlib/list.ml
  ../../../../../../stdlib/list.mli
  ../../../../../../stdlib/map.ml
  ../../../../../../stdlib/map.mli
  ../../../../../../stdlib/modes.ml
  ../../../../../../stdlib/modes.mli
  ../../../../../../stdlib/mutex.ml
  ../../../../../../stdlib/mutex.mli
  ../../../../../../stdlib/nativeint.ml
  ../../../../../../stdlib/nativeint.mli
  ../../../../../../stdlib/obj.ml
  ../../../../../../stdlib/obj.mli
  ../../../../../../stdlib/option.ml
  ../../../../../../stdlib/option.mli
  ../../../../../../stdlib/printexc.ml
  ../../../../../../stdlib/printexc.mli
  ../../../../../../stdlib/printf.ml
  ../../../../../../stdlib/printf.mli
  ../../../../../../stdlib/random.ml
  ../../../../../../stdlib/random.mli
  ../../../../../../stdlib/seq.ml
  ../../../../../../stdlib/seq.mli
  ../../../../../../stdlib/stdlib.ml
  ../../../../../../stdlib/stdlib.mli
  ../../../../../../stdlib/string.ml
  ../../../../../../stdlib/string.mli
  ../../../../../../stdlib/sys.ml
  ../../../../../../stdlib/sys.mli
  ../../../../../../stdlib/uchar.ml
  ../../../../../../stdlib/uchar.mli
  ../../../../../../stdlib/weak.mli)
 (action
  (with-stdout-to
   weak.own_output
   (run sh %{dep:../stdlib_run.sh} %{ocamlopt}
    %{dep:../../../tools/ownrun/flsemantics_ownrun.exe}
    %{dep:../../../../../../tools/objinfo.exe} %{ocaml_where}
    %{dep:../../../../../../stdlib/weak.ml}
    numbered stdlib__Weak 0
    -
    camlinternalFormat.cmi camlinternalFormat.cmx
    camlinternalFormatBasics.cmi camlinternalFormatBasics.cmx
    camlinternalLazy.cmi camlinternalLazy.cmx stdlib.cmi stdlib.cmx
    stdlib__Array.cmi stdlib__Array.cmx stdlib__Atomic.cmi
    stdlib__Atomic.cmx stdlib__Bigarray.cmi stdlib__Bigarray.cmx
    stdlib__Buffer.cmi stdlib__Buffer.cmx stdlib__Bytes.cmi
    stdlib__Bytes.cmx stdlib__Callback.cmi stdlib__Callback.cmx
    stdlib__Char.cmi stdlib__Char.cmx stdlib__Complex.cmi
    stdlib__Complex.cmx stdlib__Condition.cmi stdlib__Condition.cmx
    stdlib__Digest.cmi stdlib__Digest.cmx stdlib__Domain.cmi
    stdlib__Domain.cmx stdlib__Either.cmi stdlib__Either.cmx
    stdlib__Float.cmi stdlib__Float.cmx stdlib__Fun.cmi stdlib__Fun.cmx
    stdlib__Hashtbl.cmi stdlib__Hashtbl.cmx stdlib__In_channel.cmi
    stdlib__In_channel.cmx stdlib__Int.cmi stdlib__Int.cmx
    stdlib__Int32.cmi stdlib__Int32.cmx stdlib__Int64.cmi
    stdlib__Int64.cmx stdlib__Lazy.cmi stdlib__Lazy.cmx stdlib__List.cmi
    stdlib__List.cmx stdlib__Map.cmi stdlib__Map.cmx stdlib__Modes.cmi
    stdlib__Modes.cmx stdlib__Mutex.cmi stdlib__Mutex.cmx
    stdlib__Nativeint.cmi stdlib__Nativeint.cmx stdlib__Obj.cmi
    stdlib__Obj.cmx stdlib__Option.cmi stdlib__Option.cmx
    stdlib__Printexc.cmi stdlib__Printexc.cmx stdlib__Printf.cmi
    stdlib__Printf.cmx stdlib__Random.cmi stdlib__Random.cmx
    stdlib__Seq.cmi stdlib__Seq.cmx stdlib__String.cmi stdlib__String.cmx
    stdlib__Sys.cmi stdlib__Sys.cmx stdlib__Uchar.cmi stdlib__Uchar.cmx
    stdlib__Weak.cmi
    --
    -nopervasives -directory stdlib -strict-sequence -g -absname
    -extension runtime_metaprogramming -nostdlib -safe-string
    -strict-formats -no-alias-deps -w +a-4-9-40-41-42-44-45-48-66-67-70 -w
    -221 -principal -cmi-file stdlib__Weak.cmi
    %{read-lines:../../../../../../ocamlopt_stdlib_flags.txt} -o
    stdlib__Weak.cmx -c ./weak.ml))))

(rule
 (alias runtest-stdlib)
 (enabled_if
  (= %{context_name} "main"))
 (action
  (diff ../weak.own_expected
   weak.own_output)))
