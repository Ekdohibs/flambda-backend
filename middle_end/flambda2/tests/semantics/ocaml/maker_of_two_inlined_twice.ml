(* A function that makes two closures, marked for inlining and called twice.
   [down] is recursive and stays: Simplify gives each of its two closures a
   symbol and code of its own. [up] is a loop marked for inlining: Simplify
   inlines it at its two calls, and the term after has no closure of it. In the
   term before, once [make] is inlined, each [up] is a closure that a [Let]
   makes and that is read from a pair, and its call is indirect. The lifting
   gives the closures of [down] their symbols and the copies of their code; it
   leaves the closures of [up] to their [Let]s, the term after having no symbol
   for them. The rounds that follow the lifting make the calls of those closures
   direct (FlambdaClosureCallsSpec.call_made_closures), and the next round
   inlines them, as Simplify did. *)

external ( + ) : int -> int -> int = "%addint"

external ( - ) : int -> int -> int = "%subint"

external ( = ) : int -> int -> bool = "%equal"

external opaque : 'a -> 'a = "%opaque"

let[@inline] make limit =
  let rec down n = if n = limit then 0 else 1 + down (n - 1) in
  let[@inline] rec up n steps =
    if n = limit then steps else up (n + 1) (steps + 1)
  in
  down, up

let result =
  let down_three, up_three = make 3 in
  let down_five, up_five = make 5 in
  down_three (opaque 9)
  + up_three (opaque 0) 0
  + down_five (opaque 9)
  + up_five (opaque 0) 0
