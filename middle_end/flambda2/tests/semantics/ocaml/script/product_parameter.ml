(* A call of unknown arity, in a unit that has a function of an unboxed pair.
   The text writes the two parameters of that function as one group:
   from_ocaml.sh makes the text of the unit (product_parameter.says). It
   refused the unit while the text wrote the parameters as two, and nothing
   in it said that the call of unknown arity does not reach the function. *)

external ( + ) : int -> int -> int = "%addint"

external opaque : 'a -> 'a = "%opaque"

let[@inline never] sum #(a, b) = a + b

let[@inline never] g (x : int) = x + 1

let result = sum #(opaque 3, 4), (opaque g) 1
