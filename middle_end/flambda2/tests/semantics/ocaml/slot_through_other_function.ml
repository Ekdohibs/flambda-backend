(* A control: this unit is not validated. [w] and [v] are one recursive set. [w]
   calls [t] and does not mention [u]; [v] calls [u]. So the set has the value
   slot of [u], and the code of [w] has no use of it. Where Simplify inlines [t]
   into the code of [w], the read of [u] in the copy is a projection from the
   closure that the set holds in the value slot of [t]. Simplify makes it the
   read of the value slot of [w] that holds [u], and has that read at the head
   of the body, where the conversion from Lambda put a projection of every value
   slot of the set. The rounds of the validation make the same read
   (FlambdaSlotsThroughSpec.read_slots_through), and put it where the read
   through [t] was, in a branch: the projection at the head, which nothing used,
   is gone by then. The rounds do not move that projection out of the branch to
   the head of the code, so the two terms differ by its place.

   This unit is a variation of slot_through_plain.ml of this directory, which V3
   wrote with it, as a way to have the value slot in the set and no read of it
   in the code of [w]: [t] is the function of that unit, and [w] and [v] are a
   recursive set in the place of its [w]. The comment of that unit says how it
   was made and what it shares with functions of stdlib/set.ml of this
   repository. No further library function is identified here; [v] is a name in
   the functions that that comment names, of an element and of no function. C3
   added the declaration of [opaque], [next] and [result] for this unit. The
   operators [+], [-] and [=] and [opaque] are declared as many units of the
   tests declare them; this comment names no file of the library for them. *)

external ( + ) : int -> int -> int = "%addint"

external ( - ) : int -> int -> int = "%subint"

external ( = ) : int -> int -> bool = "%equal"

external opaque : 'a -> 'a = "%opaque"

let make (u : int -> int) =
  let[@inline] t x = u x + 1 in
  let rec w n x = if n = 0 then t x else v (n - 1) x
  and v n x = if n = 0 then u x else w (n - 1) x in
  w

let next x = x + 1

let result = (opaque make) next 2 3
