(* The handler is for another exception than the one raised: the handler raises
   it again and the initialisation of the unit ends with it. *)
external raise : exn -> 'a = "%raise"

external opaque : 'a -> 'a = "%opaque"

let result =
  try if opaque true then raise Not_found else 1 with End_of_file -> 0
