(* A handler at the top of the unit, around a call that is given a function
   written at the call. Of the term that Simplify makes of this unit, a static
   binding stands in a handler that Simplify's own rule for the term it is given
   does not count as toplevel (label T5b-4). *)

external ( + ) : int -> int -> int = "%addint"

external opaque : 'a -> 'a = "%opaque"

let[@inline never] apply f x = f x

let result =
  (try apply (fun v -> v + 1) (opaque 1) with Not_found -> 0)
  + apply (fun v -> v) 2
