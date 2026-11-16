(* The control of handler_of_reraise_kept_with_g.ml: the same source without the
   first line. Simplify makes both raises jumps, and the term after has no
   trap. *)

external raise : exn -> 'a = "%raise"

external raise_notrace : exn -> 'a = "%raise_notrace"

external ( > ) : int -> int -> bool = "%greaterthan"

let find (x : int) : int =
  try try if x > 0 then raise_notrace Not_found else 1 with e -> raise e
  with Not_found -> 0
