(* Three functions defined together with [let rec .. and ..], none of which is
   recursive: the first calls the second, the third has the type of the second
   and another body, and the unit calls the second and the third. *)

external ( + ) : int -> int -> int = "%addint"

external ( * ) : int -> int -> int = "%mulint"

external opaque : 'a -> 'a = "%opaque"

let rec first x = second (x + 1)

and second x = x * 3

and third x = x * 5

let result = second 1 + third (opaque 4)
