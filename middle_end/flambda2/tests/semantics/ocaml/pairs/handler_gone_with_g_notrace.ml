(* flags: -g *)

(* A control of handler_kept_with_g.ml, here for its first line too: the
   exception is raised by [raise_notrace], which records nothing, and with [-g]
   Simplify makes that raise a jump to the handler as it does without [-g]. *)

external raise_notrace : exn -> 'a = "%raise_notrace"

external ( > ) : int -> int -> bool = "%greaterthan"

let find (x : int) : int =
  try if x > 0 then raise_notrace Not_found else 1 with Not_found -> 0
