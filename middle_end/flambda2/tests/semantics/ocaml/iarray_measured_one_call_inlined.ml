(* An immutable array made by a function that is not inlined and
   measured by another, in a unit that has a function, [g], which
   Simplify inlines at one call and keeps at another. Simplify leaves
   the array and its length where they are. With a call of [g] inlined
   in one term and not in the other, the validation makes the two terms
   the same by inlining every call of known code on both sides, and the
   length is then a primitive of an array that the same code makes, in
   the term after as in the term before. *)

external length : 'a iarray -> int = "%array_length"

external ( + ) : int -> int -> int = "%addint"

external opaque : 'a -> 'a = "%opaque"

let[@inline never] [@local never] make (x : int) : int iarray = [: x; x :]

let[@inline never] [@local never] len (a : int iarray) : int = length a

let g (x : int) : int = x + 1

let a : int = g 3

let b : int = (g [@inlined never]) (opaque 4)

let n : int = len (make 5)
