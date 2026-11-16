(* flags: -g *)

(* A unit that is here for its first line and not for a pair. A [try] inside a
   [try]: the body raises by [raise_notrace], the inner handler raises again
   what it caught, and the outer handler catches that by name. With [-g]
   Simplify makes the first raise a jump to the inner handler, as it does
   without [-g]. The second is a raise of what a handler caught, and with [-g]
   Simplify does not make it a jump to the outer handler: the term after keeps
   the outer handler, the push and the pop of its trap, and that raise. *)

external raise : exn -> 'a = "%raise"

external raise_notrace : exn -> 'a = "%raise_notrace"

external ( > ) : int -> int -> bool = "%greaterthan"

let find (x : int) : int =
  try try if x > 0 then raise_notrace Not_found else 1 with e -> raise e
  with Not_found -> 0
