(* A function that is always inlined counts its calls in a reference and returns
   the pair of its argument and of twice its argument. It is called once, with a
   constant; the unit then makes a pair of two values that are not known.
   Simplify inlines the call: the pair [(1, 2)] is a symbol, and so is the pair
   of the two values.

   Both terms keep the reference and the store into it, so each has a primitive
   of the heap, and a lifting of the validation lifts nothing in a term given to
   it that has one. The unit is not validated, and the second check is asked. In
   the term before as it is given the call is not inlined, and nothing makes the
   pair [(1, 2)]. So the first guess of the liftings gives the symbol of that
   pair to the pair of the two values, whose fields agree with anything. The
   tool searches once more after the refusal, and its guess then prefers, for a
   block whose fields are all variables of which nothing is known, a symbol
   whose block has a variable among its fields. With that guess the second check
   relates the two terms, which is no validation of the unit. *)

type 'a ref = { mutable contents : 'a }

external ref : 'a -> 'a ref = "%makemutable"

external ( ! ) : 'a ref -> 'a = "%field0"

external ( := ) : 'a ref -> 'a -> unit = "%setfield0"

external ( + ) : int -> int -> int = "%addint"

external ( * ) : int -> int -> int = "%mulint"

external opaque : 'a -> 'a = "%opaque"

let calls = ref 0

let[@inline] entry (x : int) =
  calls := !calls + 1;
  x, 2 * x

let first = entry 1

let counts = opaque 3, opaque 4

let result = first, counts
