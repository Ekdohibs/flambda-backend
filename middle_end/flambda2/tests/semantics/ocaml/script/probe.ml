(* A unit with a probe. from_ocaml.sh makes its text (probe.says); it refused
   the unit while the text did not say that the call of the handler is one. *)

external ( + ) : int -> int -> int = "%addint"

external opaque : 'a -> 'a = "%opaque"

let[@inline never] note (_ : int) = ()

let bump x =
  [%probe "bump" (note x)];
  x + 1

let result = bump (opaque 1)
