(* The first unit of three: a function, to be inlined, that makes a closure of
   what it is given. *)

external ( + ) : int -> int -> int = "%addint"

let[@inline] make y = fun[@inline never] x -> x + y
