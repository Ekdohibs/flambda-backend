(* As boxed_number_in_slot.ml, with two functions that read the one boxed
   number, each from its own closure. The number is not a field of the module.
   Simplify puts the constant in the code of both and binds no symbol to the
   boxed number. *)

external of_int : int -> int64 = "%int64_of_int"

external to_int : int64 -> int = "%int64_to_int"

external add : int64 -> int64 -> int64 = "%int64_add"

external sub : int64 -> int64 -> int64 = "%int64_sub"

let limit = of_int 1000

let add_limit n = to_int (add n limit)

let sub_limit n = to_int (sub n limit)
