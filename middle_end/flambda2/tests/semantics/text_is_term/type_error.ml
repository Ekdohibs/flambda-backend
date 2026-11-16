(* A unit that the compiler refuses. *)

external ( + ) : int -> int -> int = "%addint"

let x = 1 + ()
