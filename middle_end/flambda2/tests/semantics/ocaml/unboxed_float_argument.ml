(* A function of an unboxed float, called directly. The text of the term
   writes the arity of the arguments of the call, which is one component of
   the kind of an unboxed float, and not a value. *)

external box : float# -> float = "%box_float"

external unbox : float -> float# = "%unbox_float"

let[@inline never] f (x : float#) = box x

let result = f (unbox 2.5), f #1.5
