(* A function that takes an unboxed pair. The arity of the parameters of its
   code has one component, a product of two; the code binds two parameters, and
   the text writes the two as a group. So the text is the term. *)

let first (p : #(int * int)) =
  let #(a, _) = p in
  a
