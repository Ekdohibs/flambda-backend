(* A function of two parameters that is the result of a call, given one
   argument. The front end sees a call, with one argument, of whatever the first
   call returns. Simplify inlines the first call, then sees a partial
   application of [add], and makes a wrapper of it as in
   partial_seen_once_inlined.ml. *)

external ( + ) : int -> int -> int = "%addint"

let[@inline never] add (a : int) (b : int) : int = a + b

let[@inline] chosen () = add

let add_to (v : int) : int -> int = (chosen ()) v
