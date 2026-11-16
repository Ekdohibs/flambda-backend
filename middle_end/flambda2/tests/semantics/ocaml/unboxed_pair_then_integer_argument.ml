(* A function of an unboxed pair and an integer, called directly: its code
   has three parameters, of which the first two are the components of the
   pair. *)

external ( + ) : int -> int -> int = "%addint"

external opaque : 'a -> 'a = "%opaque"

let[@inline never] sum #(a, b) c = a + b + c

let result = sum #(opaque 3, 4) 5
