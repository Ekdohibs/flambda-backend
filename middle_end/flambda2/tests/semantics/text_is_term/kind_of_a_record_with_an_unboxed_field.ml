(* The parameter and the result of [f] are records with an unboxed field, and
   their kind says so: a block with the tag 0 whose fields are a tagged integer
   and an unboxed float, which has the shape of a mixed block. The text writes
   the kinds of the fields ([[ 0 of imm tagged * float ]]) and not the shape,
   and the reader gives the shape that the kinds give; it gave the shape of a
   block of values. So the text is the term. *)

type t =
  { a : int;
    b : float#
  }

let f (x : t) : t = x
