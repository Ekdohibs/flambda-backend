(* An over-application whose second argument is an unboxed float: [first] is
   given its one argument and what it returns is given the float. The call is
   of unknown arity, and the text of the term writes the kinds of its two
   arguments. *)

external box : float# -> float = "%box_float"

external opaque : 'a -> 'a = "%opaque"

let[@inline never] second (x : float#) = box x

let[@inline never] first (_ : int) = second

let result = (opaque first) 1 #1.5
