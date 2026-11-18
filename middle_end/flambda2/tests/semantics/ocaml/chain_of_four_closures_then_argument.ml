(* The control of chain_of_five_closures_then_argument.ml: four functions, each
   after the first calling the one before, and a fifth that calls the fourth and
   uses the argument of the functor. Simplify makes the four closures symbols.
   (By T4.) *)

external ( + ) : int -> int -> int = "%addint"

module Make (X : sig
  val v : int
end) =
struct
  let a x = x + 1

  let b x = a x

  let c x = b x

  let d x = c x

  let e x = d x + X.v
end
