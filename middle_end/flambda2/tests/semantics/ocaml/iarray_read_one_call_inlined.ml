(* As iarray_measured_one_call_inlined.ml, with a load of the first
   element in the place of the length: an immutable array made by a
   function that is not inlined and read by another at a constant
   index, in a unit that has a function, [g], which Simplify inlines at
   one call and keeps at another. *)

external get : 'a iarray -> int -> 'a = "%array_unsafe_get"

external ( + ) : int -> int -> int = "%addint"

external opaque : 'a -> 'a = "%opaque"

let[@inline never] [@local never] make (x : int) : int iarray = [: x; x :]

let[@inline never] [@local never] first (a : int iarray) : int = get a 0

let g (x : int) : int = x + 1

let a : int = g 3

let b : int = (g [@inlined never]) (opaque 4)

let n : int = first (make 5)
