(* A control of decision D65. This unit names the function of
   [Body_with_a_store] and does not call it, and defines a function of its own,
   of which Simplify makes a symbol. The validation needs a lifting to symbols
   for that. The code bound in front of the unit, the body of
   [Body_with_a_store.store], has a primitive of the heap. The unit has no call
   outside the bodies of its code, so the two tests of the liftings look outside
   those bodies only: it is validated with its imports, and its output carries
   the mark of that decision. Before decision D65 the unit was NOT validated. *)

let store = Body_with_a_store.store

let same x = x
