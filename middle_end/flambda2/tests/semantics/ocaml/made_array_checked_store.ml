(* A mutable array of integers is made and an element is stored with the check
   of the index, which takes the length of the array. Simplify knows the length
   from the allocation and removes the check, and it writes the store as one
   into an array of immediates. *)

external opaque : 'a -> 'a = "%opaque"

external unsafe_get : 'a array -> int -> 'a = "%array_unsafe_get"

external set : 'a array -> int -> 'a -> unit = "%array_safe_set"

let result =
  let a = [| opaque 1; opaque 2 |] in
  set a 1 (opaque 7);
  unsafe_get a 1
