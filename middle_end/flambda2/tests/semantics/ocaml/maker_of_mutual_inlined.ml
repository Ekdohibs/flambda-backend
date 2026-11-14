(* A function that makes two local functions that call each other and gives one
   of them back, marked for inlining and exported. Simplify inlines [make] at
   its call. The two functions of that copy capture a value that the unit
   computes once, so Simplify gives them symbols and code of their own; [make]
   keeps the code that there was. The lifting gives the two closures of the copy
   their symbols and copies of the code (FlambdaCopyCodeSpec.copy_static_code).
   That makes indirect the two calls in the code that [make] keeps: each named
   code that is copied, and its callee is a variable, the other function
   projected from [my_closure]. The rounds that follow the lifting make those
   calls direct again, as they are in the term after
   (FlambdaClosureOfCodeSpec.know_own_closures): in the body of that code
   [my_closure] is a closure of it, and the one set of the unit that has it at
   its function slot has the code of the other function at the other slot. The
   tool asks for those rounds together with the rounds that make direct the
   calls of the closures that a [Let] makes, as the term before has such a
   [Let]; the verdict has a line for each. *)

external ( + ) : int -> int -> int = "%addint"

external ( - ) : int -> int -> int = "%subint"

external ( = ) : int -> int -> bool = "%equal"

external opaque : 'a -> 'a = "%opaque"

let[@inline] make n =
  let rec down k = if k = 0 then n else up (k - 1)
  and up k = if k = 0 then n + 1 else down (k - 1) in
  down

let result = (make (opaque 4)) 3
