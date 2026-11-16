(* A function of two integers, called directly, in a unit that also has a call
   of unknown arity, of another function. *)

external ( + ) : int -> int -> int = "%addint"

external opaque : 'a -> 'a = "%opaque"

let[@inline never] sum a b = a + b

let[@inline never] g (x : int) = x + 1

let result = sum (opaque 3) 4, (opaque g) 1
