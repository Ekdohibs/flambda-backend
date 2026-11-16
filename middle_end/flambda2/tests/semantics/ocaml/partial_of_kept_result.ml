(* The control of partial_of_inlined_result.ml: the first call is never inlined.
   It stays, the call of its result stays a call of unknown arity, and no
   wrapper is made. *)

external ( + ) : int -> int -> int = "%addint"

let[@inline never] add (a : int) (b : int) : int = a + b

let[@inline never] chosen () = add

let add_to (v : int) : int -> int = (chosen ()) v
