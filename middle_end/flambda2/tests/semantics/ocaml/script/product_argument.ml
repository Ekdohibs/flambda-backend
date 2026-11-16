(* A call of unknown arity that gives an unboxed pair. The text writes the
   arity of the arguments of the call, in which the pair is one component:
   from_ocaml.sh makes the text of the unit (product_argument.says). It
   refused the unit while the text did not say that the two arguments of that
   call are one. *)

external ( + ) : int -> int -> int = "%addint"

external opaque : 'a -> 'a = "%opaque"

let[@inline never] sum #(a, b) c = a + b + c

let result = (opaque sum) #(3, 4) 5
