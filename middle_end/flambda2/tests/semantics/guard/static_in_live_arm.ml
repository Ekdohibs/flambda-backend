(* A function that adds at the type its witness gives is inlined at toplevel,
   with the witness hidden and the two numbers known to be boxed int64s. The arm
   of [I], which adds them as integers, cannot be taken with such numbers. Of
   the term that Simplify makes of this unit, a static binding stands in a
   handler that Simplify's own rule for the term it is given does not count as
   toplevel (label C5-11). *)

external ( + ) : int -> int -> int = "%addint"

external add64 : int64 -> int64 -> int64 = "%int64_add"

external to_int : int64 -> int = "%int64_to_int"

external opaque : 'a -> 'a = "%opaque"

type _ num =
  | I : int num
  | L : int64 num

let[@inline] add : type a. a num -> a -> a -> a =
 fun w x y -> match w with I -> x + y | L -> add64 x y

let l = to_int (add (opaque L) 40L 2L)
