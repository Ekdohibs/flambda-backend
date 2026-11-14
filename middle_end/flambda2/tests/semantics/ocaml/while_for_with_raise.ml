(* A for loop left by a raise. *)

external ( + ) : int -> int -> int = "%addint"

external ( = ) : int -> int -> bool = "%equal"

external opaque : 'a -> 'a = "%opaque"

external raise : exn -> 'a = "%raise"

type 'a ref = { mutable contents : 'a }

external ref : 'a -> 'a ref = "%makemutable"

external ( ! ) : 'a ref -> 'a = "%field0"

external ( := ) : 'a ref -> 'a -> unit = "%setfield0"

let f n =
  let seen = ref 0 in
  try
    for i = 0 to n do
      if i = 3 then raise Not_found;
      seen := !seen + 1
    done;
    !seen
  with Not_found -> !seen + 100

let result = f (opaque 10) + f (opaque 2)
