(* flags: -g *)

(* A unit that is here for its first line and not for a pair. A function that
   raises an exception inside the [try] that catches it. With [-g] a raise
   records where it was raised, and Simplify does not make it a jump to the
   handler: the term after keeps the handler, the push and the pop of the trap,
   and the raise. *)

external raise : exn -> 'a = "%raise"

external ( > ) : int -> int -> bool = "%greaterthan"

let find (x : int) : int =
  try if x > 0 then raise Not_found else 1 with Not_found -> 0
