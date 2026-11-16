(* flags: -O3 *)

(* As known_arity_lists_older_code.ml, and then the unit calls a number, which
   is undefined in the model whatever the lists are. The list of the call in
   [run] has not the newer versions of the code that it names: they are bound
   later, for the copy of [run] at toplevel, out of the scope of that call. A
   list that may be short and a run that is undefined by something else: the
   tool reports the runs as it does of any unit. *)

external ( + ) : int -> int -> int = "%addint"

external ( - ) : int -> int -> int = "%subint"

external ( = ) : int -> int -> bool = "%equal"

external opaque : 'a -> 'a = "%opaque"

external magic : 'a -> 'b = "%identity"

let rec even n = if n = 0 then 1 else odd (n - 1)

and odd n = if n = 0 then 0 else even (n - 1)

let[@inline] apply f x = f x

let run c n =
  let add_n x = x + n in
  let sub_n x = x - n in
  let f = if c then add_n else sub_n in
  apply f 10 + even n

let result = run (opaque true) 3

let wrong = (magic (opaque 0) : int -> int) 1
