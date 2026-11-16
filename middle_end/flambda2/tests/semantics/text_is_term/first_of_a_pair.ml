(* A function of a pair. The kind of its parameter says that the pair is a block
   of two fields, each of them any value, which may be null. The text writes
   that such a kind may be null, so the text is the term. *)

let first p =
  let a, _ = p in
  a
