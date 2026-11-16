(* A function that takes an unboxed pair. Its code has one parameter that is a
   product of two components, and the text writes two parameters; so the text
   is not the term, and the driver names that difference. *)

let first (p : #(int * int)) =
  let #(a, _) = p in
  a
