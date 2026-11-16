(* A partial application, by a call of unknown arity, of a function whose
   second parameter has no kind. The text writes that parameter, as "#()":
   from_ocaml.sh makes the text of the unit (void_parameter.says). It refused
   the unit while the text wrote one parameter for the code: the call, which
   has one argument, was read as a full application, and the unit read had 5
   where the program has a closure. *)

external opaque : 'a -> 'a = "%opaque"

let[@inline never] h (c : int) #() = c

let p = (opaque h) 5
