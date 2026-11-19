(* A control of boxed_number_in_slot.ml: the number is not known where the unit
   is compiled, so the constant cannot go into the code of the function. *)

external of_int : int -> int64 = "%int64_of_int"

external to_int : int64 -> int = "%int64_to_int"

external add : int64 -> int64 -> int64 = "%int64_add"

external opaque : 'a -> 'a = "%opaque"

let add_limit =
  let limit = of_int (opaque 1000) in
  fun n -> to_int (add n limit)
