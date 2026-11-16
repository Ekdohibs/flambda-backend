(* A unit with a call of a function that is not known. The text of its term does
   not write the subkinds of the arity of the arguments of such a call (here,
   that the argument is an immediate), so the text is not the term; the driver
   names that difference. *)

external opaque : 'a -> 'a = "%opaque"

let f x = x

let y = (opaque f) 1
