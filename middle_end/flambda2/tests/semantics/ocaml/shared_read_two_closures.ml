(* A value that is computed when the unit runs is held by two closures, as the
   value of a slot of each. A third function calls both, and both calls are
   inlined: once the value is read from one closure, the read from the other is
   that value. *)

external ( + ) : int -> int -> int = "%addint"

external ( * ) : int -> int -> int = "%mulint"

external opaque : 'a -> 'a = "%opaque"

let scale = opaque 3

let shift x = x + scale

let times x = x * scale

let both x = shift (times x)
