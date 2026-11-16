(* flags: -O3 *)

(* One of two local closures, chosen by a test, is called in a function that is
   then inlined where it is called. The call in the copy lists the code of the
   two closures as the function has them; the closures that reach it are those
   of the copy, of newer versions of that code. *)

external ( + ) : int -> int -> int = "%addint"

external ( - ) : int -> int -> int = "%subint"

external ( = ) : int -> int -> bool = "%equal"

external opaque : 'a -> 'a = "%opaque"

let rec even n = if n = 0 then 1 else odd (n - 1)

and odd n = if n = 0 then 0 else even (n - 1)

let[@inline] apply f x = f x

let run c n =
  let add_n x = x + n in
  let sub_n x = x - n in
  let f = if c then add_n else sub_n in
  apply f 10 + even n

let result = run (opaque true) 3
