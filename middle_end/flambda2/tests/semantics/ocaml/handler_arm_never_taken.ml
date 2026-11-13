(* A handler with an arm for an exception that the body cannot raise, beside the
   arm for the one it does raise. *)
external raise : exn -> 'a = "%raise"

external ( < ) : int -> int -> bool = "%lessthan"

external opaque : 'a -> 'a = "%opaque"

let result =
  try if opaque 3 < 5 then raise Not_found else 0 with
  | Division_by_zero -> 1
  | Not_found -> 2
