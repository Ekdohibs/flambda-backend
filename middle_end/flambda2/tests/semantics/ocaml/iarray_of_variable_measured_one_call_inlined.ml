(* As iarray_measured_one_call_inlined.ml, with elements that are not
   constants: the function that makes the array is given a value that
   Simplify knows nothing of. *)

external length : 'a iarray -> int = "%array_length"

external ( + ) : int -> int -> int = "%addint"

external opaque : 'a -> 'a = "%opaque"

let[@inline never] [@local never] make (x : int) : int iarray = [: x; x :]

let[@inline never] [@local never] len (a : int iarray) : int = length a

let g (x : int) : int = x + 1

let a : int = g 3

let b : int = (g [@inlined never]) (opaque 4)

let n : int = len (make (opaque 5))
