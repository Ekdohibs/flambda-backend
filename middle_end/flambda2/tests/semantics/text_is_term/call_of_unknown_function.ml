(* A unit with a call of a function that is not known. The text of its term
   writes the arity of the arguments of such a call where it is not one plain
   value for each argument (here, the argument is an immediate), so the text is
   the term. *)

external opaque : 'a -> 'a = "%opaque"

let f x = x

let y = (opaque f) 1
