(* A mutable array is made before a loop, and at each turn of the loop an
   element is stored with the check of the index, which takes the length of the
   array. Simplify knows the length from the allocation: in the term after the
   check compares the index with the number 3, where the term before takes the
   length of the array at each turn. The sharing of primitives of the
   validation, which knows the length of an array made by a [Let], carries what
   it knows of the allocation into the handler of a recursive continuation,
   which the loop is: the length in the loop of the term before becomes that
   number too, and this unit is validated with its imports. *)

external opaque : 'a -> 'a = "%opaque"

external set : 'a array -> int -> 'a -> unit = "%array_safe_set"

external unsafe_get : 'a array -> int -> 'a = "%array_unsafe_get"

let result =
  let a = [| opaque None; opaque None; opaque (Some 2) |] in
  for i = 0 to 2 do
    set a i None
  done;
  match unsafe_get a 0 with None -> 0 | Some n -> n
