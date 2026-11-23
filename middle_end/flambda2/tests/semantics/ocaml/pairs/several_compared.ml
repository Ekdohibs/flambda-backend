(* A toplevel that compares two fields of another unit of which the .cmx says
   nothing. *)

external ( == ) : 'a -> 'a -> bool = "%eq"

let same = Several.first == Several.second
