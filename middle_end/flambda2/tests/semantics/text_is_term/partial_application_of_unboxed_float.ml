(* A function whose first parameter is an unboxed float, given that argument
   alone. The closure of the partial application keeps the number in a value
   slot that is not of the kind of values. The code of the closure, which the
   text has first, reads the number from the slot, and the text says the kind
   of the slot there ([%project_value_slot] with [float] after the name of the
   slot). It said none, the reader took the slot for one of a value, and it
   stopped where the closure is made with a float in the slot. So the text is
   the term. *)

external unbox : float -> float# = "%unbox_float"

external opaque : 'a -> 'a = "%opaque"

let[@inline never] first (a : float#) (b : int) : float# =
  let _ = b in
  a

let partial = first (unbox (opaque 1.5))
