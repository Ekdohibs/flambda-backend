(* A unit that exports a function. Its term before Simplify is given to the tool
   as the term after too (the option -after): see dune. *)

external ( + ) : int -> int -> int = "%addint"

let f x = x + 1
