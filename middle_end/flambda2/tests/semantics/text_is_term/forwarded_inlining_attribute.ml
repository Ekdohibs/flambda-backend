(* A call marked [@inlined forward]. The text writes the attribute
   ("inlined(forward)"), and the reader takes it; the grammar had no rule for
   it, and the text was not read back. So the text is the term. *)

let apply f x = (f [@inlined forward]) x
