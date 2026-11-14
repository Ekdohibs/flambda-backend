(* As [rec_helper_called_outside], the two functions local to a function.
   Simplify inlines the call of [second] in [run], and leaves the call of
   [second] in [first] and the call of [first] in [run]. *)

external ( + ) : int -> int -> int = "%addint"

external ( * ) : int -> int -> int = "%mulint"

external opaque : 'a -> 'a = "%opaque"

let run n =
  let rec first x = second (x + 1) and second x = x * n in
  first 2 + second 1

let result = run (opaque 3)
