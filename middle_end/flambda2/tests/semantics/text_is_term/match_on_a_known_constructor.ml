(* A match on a value whose constructor is known. In the arm that is not taken
   the compiler puts an invalid expression, where the argument of the other
   constructor would be read. The text writes the kind of that expression, so
   the text is the term. *)

type t =
  | A
  | B of int

let f () = match A with A -> 0 | B n -> n
