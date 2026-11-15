(* A function that is always inlined makes a recursive function local to it,
   which uses its parameter, and calls it; another function calls the first
   twice. Simplify inlines the two calls of [count_from] in [twice]. Each copy
   of the body makes its closure of [go] where it stands, as the closure holds a
   variable, and Simplify gives each copy new code: the term after has the code
   of [go] three times, once for the closure that the code of [count_from] makes
   and once for each of the two that the code of [twice] makes. With -O3 it
   inlines the call of [twice] too, and the term after has the code of [go] five
   times. *)

external ( + ) : int -> int -> int = "%addint"

external ( - ) : int -> int -> int = "%subint"

external ( <= ) : int -> int -> bool = "%lessequal"

external opaque : 'a -> 'a = "%opaque"

let[@inline] count_from (q : int) =
  let rec go (n : int) = if n <= 0 then q else go (n - 1) + 1 in
  go 3

let twice (a : int) = count_from a + count_from (a + 1)

let result = twice (opaque 5)
