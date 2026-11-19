(* A control of boxed_number_in_slot.ml: the boxed number is a field of the
   module too, so Simplify binds a symbol to it, and that symbol is the one that
   stands for it. *)

external of_int : int -> int64 = "%int64_of_int"

external to_int : int64 -> int = "%int64_to_int"

external add : int64 -> int64 -> int64 = "%int64_add"

let limit = of_int 1000

let add_limit n = to_int (add n limit)
