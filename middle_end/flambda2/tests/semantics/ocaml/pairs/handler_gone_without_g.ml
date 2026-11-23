(* A control of handler_kept_with_g.ml: the same source without the first line.
   Simplify makes the raise a jump to the handler, and the term after has no
   trap. *)

external raise : exn -> 'a = "%raise"

external ( > ) : int -> int -> bool = "%greaterthan"

let find (x : int) : int =
  try if x > 0 then raise Not_found else 1 with Not_found -> 0
