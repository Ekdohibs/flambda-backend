(* A toplevel that takes the field of another unit of which the .cmx says
   nothing down one arm of a test and a number down the other, and adds to what
   it has. *)

external opaque : 'a -> 'a = "%opaque"

external ( + ) : int -> int -> int = "%addint"

let chosen = if opaque true then Hidden.hidden else 3

let more = chosen + 1
