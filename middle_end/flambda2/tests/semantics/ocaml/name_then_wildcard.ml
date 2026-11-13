(* A handler with an arm for one exception and an arm for any other; the
   exception raised is the one of a division by zero. *)
external ( / ) : int -> int -> int = "%divint"

external opaque : 'a -> 'a = "%opaque"

let result = try 10 / opaque 0 with Not_found -> 1 | _ -> 2
