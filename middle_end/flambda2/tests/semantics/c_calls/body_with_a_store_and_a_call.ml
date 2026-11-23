(* A control of what the reading can cost a unit that calls no C function. This
   unit names the function of [Body_with_a_store] and does not call it, defines
   a function of its own, of which Simplify makes a symbol, and calls that
   function once at top level through the primitive [%opaque], so that the call
   is in both terms and is not a call of known code. The validation needs a
   lifting to symbols, and a lifting lifts nothing in a term that has a call
   outside the bodies of its code and a primitive of the heap in the code bound
   in front of it: the body of [Body_with_a_store.store], now that it is bound.
   The unit is NOT validated, and the tool prints the line of the guard. With
   that code declared deleted in front of the unit it is validated with its
   imports. Without the call it is [Body_with_a_store_not_called], which
   decision D65 concerns. *)

external opaque : 'a -> 'a = "%opaque"

let store = Body_with_a_store.store

let same x = x

let one = (opaque same) 1
