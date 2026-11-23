(* A parameter that is a boxed 128-bit vector. The text writes its kind [vec128
   boxed], and the reader takes it; it had no rule for the kind of a boxed
   vector, and the text was not read back. So the text is the term. *)

let f (x : int64x2) = x
