(* A unit with a call of a C function. The text of such a call has the region
   that the grammar asks for after [ccall], so the text is read back, and it is
   the term. *)

external f : int -> int = "caml_f"

let g x = f x
