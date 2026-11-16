(* A unit with a functor. The text says of code that it is that of a functor, so
   the text is the term. *)

module F (X : sig
  val x : int
end) =
struct
  let y = X.x
end
