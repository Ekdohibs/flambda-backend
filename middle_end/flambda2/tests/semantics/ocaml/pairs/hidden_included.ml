(* A unit that includes another unit whose module block has a field of which the
   .cmx says nothing. Its toplevel loads that field and makes it a field of its
   own module block, and does nothing else with it. *)

include Hidden
