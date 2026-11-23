(* The comparison of two floats that gives -1, 0 or 1. The text writes it
   [%float_comp.qmark], as it writes the like comparison of two integers; it
   wrote [%float_comp.lt], which is read as "less than". So the text is the
   term. *)

external compare : float -> float -> int = "%compare"

let f a b = compare a b
