(* A recursive function that is on no cycle of direct calls, beside a write to a
   mutable field. [fib] gives [memo], another function of the unit, a closure
   that calls [fib] twice; [memo] counts the call in a mutable field and calls
   the closure. [fib] calls [memo] directly, [memo] calls the closure
   indirectly, and [fib] only makes the closure. Simplify inlines none of the
   three: the compiler marks [fib] recursive, and [memo] and the closure are too
   large to be inlined with the default flags. It makes the closures of [memo]
   and of [fib] symbols. The tool leaves the calls of [fib] in the rounds of the
   validation by the compiler's mark: by direct calls alone [fib] is not
   recursive. Before it took the mark the tool ended with no stack left on this
   unit. Read from the definitions of the rounds, and not seen on this unit:
   without the mark they inline [memo] in the code of [fib], make the call of
   the closure direct and inline the code of the closure, which by then has the
   body of [fib] in it. *)

external ( + ) : int -> int -> int = "%addint"

external ( - ) : int -> int -> int = "%subint"

external ( < ) : int -> int -> bool = "%lessthan"

type table = { mutable misses : int }

let memo table key compute =
  table.misses <- table.misses + 1;
  compute key

let rec fib table n =
  if n < 2
  then n
  else memo table n (fun n -> fib table (n - 1) + fib table (n - 2))

let result =
  let t = { misses = 0 } in
  fib t 10
