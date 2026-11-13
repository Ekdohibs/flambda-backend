(* A function that makes a closure, exported and also called in the unit:
   [adder] is marked for inlining, and [add_three] is the closure that the
   inlined call makes, which captures the constant 3. Simplify gives the closure
   of the inlined copy new code, which adds 3 and reads no value slot, and keeps
   the code that reads the slot for the closures that [adder] makes when another
   unit calls it. The term before has one code for both. The validation gives it
   a copy of that code under the new code id, for the symbol that the closure is
   lifted to (FlambdaCopyCodeSpec.copy_static_code). *)

external ( + ) : int -> int -> int = "%addint"

external opaque : 'a -> 'a = "%opaque"

let[@inline] adder k =
  let add x = x + k in
  add

let add_three = adder 3

let result = add_three (opaque 4)
