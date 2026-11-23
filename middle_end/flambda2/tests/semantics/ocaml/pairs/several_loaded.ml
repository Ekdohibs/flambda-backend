(* A toplevel that loads a field of a block of another unit, where that block is
   itself a field of which the .cmx says nothing. *)

let one = match Several.pair with x, _ -> x
