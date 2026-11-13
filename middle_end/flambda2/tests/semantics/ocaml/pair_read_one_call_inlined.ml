(* As iarray_read_one_call_inlined.ml, with a pair in the place of the immutable
   array: a pair made by a function that is not inlined and its first component
   read by another, in a unit that has a function, [g], which Simplify inlines
   at one call and keeps at another. *)

external ( + ) : int -> int -> int = "%addint"

external opaque : 'a -> 'a = "%opaque"

let[@inline never] [@local never] make (x : int) : int * int = x, x

let[@inline never] [@local never] first (p : int * int) : int =
  match p with a, _ -> a

let g (x : int) : int = x + 1

let a : int = g 3

let b : int = (g [@inlined never]) (opaque 4)

let n : int = first (make 5)
