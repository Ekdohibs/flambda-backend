(* A function of an unboxed float, called through a value of which nothing is
   known. The call is of unknown arity, and the text of the term writes the
   arity of its arguments, which is one component of the kind of an unboxed
   float, and not a value. *)

external box : float# -> float = "%box_float"

external opaque : 'a -> 'a = "%opaque"

let[@inline never] f (x : float#) = box x

let result = (opaque f) #1.5
