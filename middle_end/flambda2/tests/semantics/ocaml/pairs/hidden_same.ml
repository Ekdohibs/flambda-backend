(* A toplevel that compares the field of another unit of which the .cmx says
   nothing with itself. *)

external ( == ) : 'a -> 'a -> bool = "%eq"

let same = Hidden.hidden == Hidden.hidden
