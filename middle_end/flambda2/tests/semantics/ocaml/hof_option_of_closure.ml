(* A closure under Some, matched and called; None in the other arm. Simplify
   inlines the three calls of [call_or]. It unboxes [o]: where the arms of the
   [if] meet, a continuation takes whether [o] is [None] and, at the default
   flags, the closure. At the default flags the call of that closure is left, as
   a call of an unknown function. The closure of the second call is known at
   both settings and its call is inlined. With -O3 the closure of [o] is known
   where the arms meet: its call is inlined too, and the closure is not made.

   The test of this unit in the tree is made at the default flags and does not
   exercise the change that the unit entered with (the validation lets a call
   return to a continuation that it moves out of a handler,
   FlambdaLiftContsSpec.only_applied): the expected file is the same with and
   without it. What that change does for the unit is with -O3 only, where the
   unit is validated with it and not without; no test of the tree shows it. *)

external ( + ) : int -> int -> int = "%addint"

external ( * ) : int -> int -> int = "%mulint"

external ( = ) : int -> int -> bool = "%equal"

external opaque : 'a -> 'a = "%opaque"

let[@inline] call_or o d x = match o with None -> d | Some f -> f x

let[@inline never] body c x =
  let o = if c then Some (fun y -> y + x) else None in
  call_or o 100 5 + call_or (Some (fun y -> y * 2)) 0 x + call_or None x 0

let result = body (opaque true) 3 + body false (opaque 4)
