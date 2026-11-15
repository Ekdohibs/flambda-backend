(* A mutable array is made before a loop, and at each turn of the loop an
   element is stored with the check of the index, which takes the length of the
   array. Simplify knows the length from the allocation: in the term after the
   check compares the index with the number 3, where the term before takes the
   length of the array at each turn. The two passes of the validation that know
   the length of an array made by a [Let], the sharing of primitives and the
   pass of the lengths alone, do not carry what they know of the allocation into
   the handler of a recursive continuation, which the loop is. The term before
   keeps the length in its loop, and this unit is not validated. *)

external opaque : 'a -> 'a = "%opaque"

external set : 'a array -> int -> 'a -> unit = "%array_safe_set"

external unsafe_get : 'a array -> int -> 'a = "%array_unsafe_get"

let result =
  let a = [| opaque None; opaque None; opaque (Some 2) |] in
  for i = 0 to 2 do
    set a i None
  done;
  match unsafe_get a 0 with None -> 0 | Some n -> n
