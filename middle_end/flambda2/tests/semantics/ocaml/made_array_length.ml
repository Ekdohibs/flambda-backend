(* A mutable array is made of three values that Simplify does not know, one of
   them is stored again, and the length of the array is taken. Simplify knows
   the length from the allocation, so the term after has the number 3 where the
   term before takes the length. The elements are options, so the store is of
   values on both sides. *)

external opaque : 'a -> 'a = "%opaque"

external length : 'a array -> int = "%array_length"

external unsafe_set : 'a array -> int -> 'a -> unit = "%array_unsafe_set"

let result =
  let a = [| opaque (Some 1); opaque None; opaque None |] in
  unsafe_set a 0 (opaque None);
  length a
