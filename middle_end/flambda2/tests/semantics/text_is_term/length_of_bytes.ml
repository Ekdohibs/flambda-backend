(* The length of a byte sequence. The text writes the primitive as
   [%bytes_length], and it is read as the length of a byte sequence, not of a
   string; so the text is the term. *)

external length : bytes -> int = "%bytes_length"

let f b = length b
