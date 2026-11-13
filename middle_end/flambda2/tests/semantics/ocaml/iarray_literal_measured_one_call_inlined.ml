(* As iarray_measured_one_call_inlined.ml, with an array that the unit
   makes where it is measured: its second element is not a constant, so
   it is made as the unit runs, and Simplify binds it to a symbol. The
   function that measures it is not inlined, and [g] is inlined at one
   call and kept at another. *)

external length : 'a iarray -> int = "%array_length"

external ( + ) : int -> int -> int = "%addint"

external opaque : 'a -> 'a = "%opaque"

let[@inline never] [@local never] len (a : int iarray) : int = length a

let g (x : int) : int = x + 1

let a : int = g 3

let b : int = (g [@inlined never]) (opaque 4)

let n : int = len [: 5; opaque 6 :]
