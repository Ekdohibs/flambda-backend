(* A control of the mark of decision D65 under the line of an acceptance of a
   pair whose run before is undefined. [throw] raises its parameter and is not
   called here: the liftings' test of the exits as it was, which reads the
   bodies of the code of the unit, refuses the unit, and the finer test, which
   reads outside them where the unit calls nothing there, passes it. The
   comparison of an immutable array with itself by [==] has no value in the
   model, so the unit is undefined before Simplify and after. The validation
   accepts the pair at the step that lifts, which says nothing of a unit whose
   run before is undefined, and the line of the mark is the last line in
   brackets under the line of that acceptance. iarray_phys_equal.ml is this
   unit without the function: its expected file has the same line of
   acceptance and no line of the mark. *)

external opaque : 'a -> 'a = "%opaque"

external ( == ) : 'a -> 'a -> bool = "%eq"

external raise : exn -> 'a = "%raise"

let[@inline never] throw (e : exn) : int = raise e

let table = [: 1; 2; 3 :]

let result = if opaque table == opaque table then 1 else 0
