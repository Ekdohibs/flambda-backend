(* A call of unknown arity that gives an argument of no kind, the unboxed
   product of nothing. The compiler's call has two arguments and its code one
   parameter, so it is an over-application. The text writes the arity of the
   arguments of the call, in which the product of nothing is one component:
   from_ocaml.sh makes the text of the unit (void_argument.says). It refused
   the unit while the text wrote nothing of the second argument, and the call
   was read as a full application. *)

external opaque : 'a -> 'a = "%opaque"

let[@inline never] h (c : int) = fun #() -> c

let result = (opaque h) 5 #()
