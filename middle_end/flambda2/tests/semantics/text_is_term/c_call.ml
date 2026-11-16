(* A unit with a call of a C function. The text of its term is not read back:
   the reader has no syntax for what the printer writes of such a call. *)

external f : int -> int = "caml_f"

let g x = f x
