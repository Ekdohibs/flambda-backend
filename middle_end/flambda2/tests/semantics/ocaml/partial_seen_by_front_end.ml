(* The control of partial_seen_once_inlined.ml: the partial application is
   written where the front end sees it, and the front end makes the wrapper;
   both terms have it. *)

external ( + ) : int -> int -> int = "%addint"

let[@inline never] add (a : int) (b : int) : int = a + b

let add_to (v : int) : int -> int = add v
