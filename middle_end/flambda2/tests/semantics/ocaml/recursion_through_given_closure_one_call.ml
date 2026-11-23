(* [recursion_through_given_closure_two_calls.ml] with one call of [fib] in the
   closure that [fib] gives to [memo], where that unit has two. Simplify inlines
   nothing here either: the closure is small, and its only call is the indirect
   call in [memo]. Before it took the compiler's mark of [fib] the tool printed
   its verdict of this unit, not validated, and then did not end in a minute of
   processor time. *)

external ( + ) : int -> int -> int = "%addint"

external ( - ) : int -> int -> int = "%subint"

external ( < ) : int -> int -> bool = "%lessthan"

type table = { mutable misses : int }

let memo table key compute =
  table.misses <- table.misses + 1;
  compute key

let rec fib table n =
  if n < 2 then n else memo table n (fun n -> fib table (n - 1))

let result =
  let t = { misses = 0 } in
  fib t 10
