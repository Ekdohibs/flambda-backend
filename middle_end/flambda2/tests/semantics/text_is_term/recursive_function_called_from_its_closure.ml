(* A recursive function that returns a closure that calls it. Simplify makes the
   code of that closure call the recursive function by its symbol, under a
   coercion: a change of depth from the depth "do not inline" to the same depth,
   which is not the identity coercion (the function of the compiler that makes a
   change of depth gives the identity only if its two depths are one value in
   memory). The unit is checked with -after-simplify: on the term that Simplify
   returns. The text writes the coercion ([~ depth do_not_inline ->
   do_not_inline]); the reader makes both depths from one constant, so it reads
   the identity, and the text is not the term. *)

external ( + ) : int -> int -> int = "%addint"

external ( - ) : int -> int -> int = "%subint"

external ( = ) : int -> int -> bool = "%equal"

let rec count n = if n = 0 then fun b -> b else fun b -> count (n - 1) (b + 1)
