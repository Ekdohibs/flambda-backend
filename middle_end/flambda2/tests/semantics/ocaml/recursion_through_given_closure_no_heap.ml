(* [recursion_through_given_closure_two_calls.ml] without the mutable field:
   [apply] only calls the closure that [fib] gives it. Simplify inlines [apply]
   in the code of [fib] and makes the call of the closure there a direct call of
   its code, which is too large to be inlined with the default flags: the term
   after has a cycle of direct calls, from the code of [fib] to the code of the
   closure and back, and the term before has none. *)

external ( + ) : int -> int -> int = "%addint"

external ( - ) : int -> int -> int = "%subint"

external ( < ) : int -> int -> bool = "%lessthan"

let apply key compute = compute key

let rec fib n =
  if n < 2 then n else apply n (fun n -> fib (n - 1) + fib (n - 2))

let result = fib 10
