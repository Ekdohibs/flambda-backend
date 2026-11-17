(* A test at toplevel whose second arm raises an exception built in the arm. Of
   the term that Simplify makes of this unit, a static binding stands in a
   handler that Simplify's own rule for the term it is given does not count as
   toplevel. *)

external raise : exn -> 'a = "%raise"

external opaque : 'a -> 'a = "%opaque"

external ( + ) : int -> int -> int = "%addint"

external ( > ) : int -> int -> bool = "%greaterthan"

external ( < ) : int -> int -> bool = "%lessthan"

let n : int = opaque 3

let x : int =
  if n > 0 then n + 1 else raise (Failure (if n < -5 then "low" else "neg"))

let y = x, x
