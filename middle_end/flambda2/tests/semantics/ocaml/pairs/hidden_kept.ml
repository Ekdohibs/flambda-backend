(* A toplevel that loads the field of another unit of which the .cmx says
   nothing, beside one that it knows, and puts the two in a block that it makes.
   It does not inspect the first. *)

let kept = Hidden.hidden, Hidden.known
