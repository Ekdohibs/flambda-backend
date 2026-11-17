(* A [try] whose body calls a function that can raise, builds a block, and calls
   the function again; the handler builds the same block. Of the term that
   Simplify makes of this unit, a static binding stands in a handler that
   Simplify's own rule for the term it is given does not count as toplevel
   (label T5b-4). *)

external raise : exn -> 'a = "%raise"

external opaque : 'a -> 'a = "%opaque"

external ( + ) : int -> int -> int = "%addint"

external ( > ) : int -> int -> bool = "%greaterthan"

let[@inline never] f (x : int) : int = if x > 5 then raise Not_found else x

let[@inline never] g (x : int) : int = x + 1

let r =
  try
    let a = f (opaque 1) in
    let p = g, 1 in
    let b = f (opaque 2) in
    p, a + b
  with Not_found -> (g, 1), 0

let pair = r, r
