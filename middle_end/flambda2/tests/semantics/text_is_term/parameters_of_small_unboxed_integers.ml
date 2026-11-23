(* A parameter that is an unboxed 8-bit integer and one that is an unboxed
   16-bit integer. The text writes their kinds [int8] and [int16], and the
   reader takes them; it had no rule for them, and the text was not read back.
   So the text is the term. *)

let f (x : int8#) = x

let g (x : int16#) = x
