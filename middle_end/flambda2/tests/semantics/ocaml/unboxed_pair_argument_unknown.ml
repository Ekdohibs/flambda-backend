(* A function of an unboxed pair and an integer, called through a value of
   which nothing is known. The call is of unknown arity, and in the arity of
   its arguments the pair is one component. *)

external ( + ) : int -> int -> int = "%addint"

external opaque : 'a -> 'a = "%opaque"

let[@inline never] sum #(a, b) c = a + b + c

let result = (opaque sum) #(3, 4) 5
