(* A boxed number that a closure holds in a value slot. The unit makes the
   number once, of a constant, and the function adds it to its argument.
   Simplify puts the constant in the code of the function, whose closure then
   has no value slot, and binds no symbol to the boxed number, which nothing
   reads any more. *)

external of_int : int -> int64 = "%int64_of_int"

external to_int : int64 -> int = "%int64_to_int"

external add : int64 -> int64 -> int64 = "%int64_add"

let add_limit =
  let limit = of_int 1000 in
  fun n -> to_int (add n limit)
