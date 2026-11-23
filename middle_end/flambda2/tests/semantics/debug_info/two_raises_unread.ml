(* A [try] at the toplevel of a unit whose handler does not read the exception,
   and whose body raises one of two predefined exceptions. Given [-g], Simplify
   keeps the handler; the handler demoted, it still has the exception for a
   parameter, which is given two symbols, where the validation has removed that
   parameter from the term before: the unit is not validated. *)

external raise : exn -> 'a = "%raise"

external opaque : 'a -> 'a = "%opaque"

let result =
  try if opaque true then raise Not_found else raise End_of_file with _ -> 2
