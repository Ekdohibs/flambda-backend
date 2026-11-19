(* A control of what the reading can cost a unit that calls no C function. This
   unit names the function of [Body_with_a_store] and does not call it, and
   defines a function of its own, of which Simplify makes a symbol. The
   validation needs a lifting to symbols for that, and a lifting lifts nothing
   in a term that has a primitive of the heap in the code bound in front of it:
   the body of [Body_with_a_store.store], now that it is bound. The unit is NOT
   validated, and the tool says why. With that code declared deleted in front of
   the unit it is validated with its imports. *)

let store = Body_with_a_store.store

let same x = x
