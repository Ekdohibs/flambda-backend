(* A [try] at the toplevel of a unit whose body calls a function and then always
   raises. Of the term that Simplify makes of this unit, a static binding stands
   in a handler that Simplify's own rule for the term it is given does not count
   as toplevel (label C5-11). *)

external raise : exn -> 'a = "%raise"

external opaque : 'a -> 'a = "%opaque"

let[@inline never] f x = if x then raise Not_found else 1

let result =
  try
    let _ = f (opaque true) in
    let _ = f (opaque false) in
    raise Not_found
  with _ -> 2
