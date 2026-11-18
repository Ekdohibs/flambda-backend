(* A toplevel that tests the field of another unit of which the .cmx says
   nothing. *)

let which = match Hidden.hidden with 0 -> 1 | _ -> 2
