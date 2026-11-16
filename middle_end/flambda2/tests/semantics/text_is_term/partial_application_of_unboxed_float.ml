(* A function whose first parameter is an unboxed float, given that argument
   alone. The closure of the partial application keeps the number in a value
   slot that is not of the kind of values, and the reader stops on the text
   of it: the text is not read back. *)

external unbox : float -> float# = "%unbox_float"

external opaque : 'a -> 'a = "%opaque"

let[@inline never] first (a : float#) (b : int) : float# =
  let _ = b in
  a

let partial = first (unbox (opaque 1.5))
