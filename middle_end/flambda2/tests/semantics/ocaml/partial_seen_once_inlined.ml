(* A function of two parameters given one argument through a function that
   applies whatever it is given. The front end sees, in that function, a call of
   unknown arity. Simplify inlines it in [add_to] and then sees a partial
   application of [add], of which it makes a wrapper: new code, a stub that
   reads the first argument from its own closure and calls [add] with both, and
   the closure of that code, which holds the argument in a value slot. *)

external ( + ) : int -> int -> int = "%addint"

let[@inline never] add (a : int) (b : int) : int = a + b

let[@inline] apply f x = f x

let add_to (v : int) : int -> int = apply add v
