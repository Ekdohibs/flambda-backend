(* As iarray_read_one_call_inlined.ml, with elements that are not
   constants: the function that makes the array is given a value that
   Simplify knows nothing of. *)

external get : 'a iarray -> int -> 'a = "%array_unsafe_get"

external ( + ) : int -> int -> int = "%addint"

external opaque : 'a -> 'a = "%opaque"

let[@inline never] [@local never] make (x : int) : int iarray = [: x; x :]

let[@inline never] [@local never] first (a : int iarray) : int = get a 0

let g (x : int) : int = x + 1

let a : int = g 3

let b : int = (g [@inlined never]) (opaque 4)

let n : int = first (make (opaque 5))
