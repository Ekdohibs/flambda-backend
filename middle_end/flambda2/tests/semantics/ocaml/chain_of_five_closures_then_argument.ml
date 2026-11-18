(* As chain_of_five_in_functor.ml, with a chain that the block does not end:
   five functions, each after the first calling the one before, and a sixth that
   calls the fifth and uses the functor's argument. Before Simplify each closure
   holds the one before in a value slot, in the code of the functor. Simplify
   makes the five closures symbols; the closure of the sixth, which holds the
   argument, and the block of the functor's result stay in the code of the
   functor. So the five values that are lifted in turn are all closures, where
   in chain_of_five_in_functor.ml the fifth is the block. (By T4.) *)

external ( + ) : int -> int -> int = "%addint"

module Make (X : sig
  val v : int
end) =
struct
  let a x = x + 1

  let b x = a x

  let c x = b x

  let d x = c x

  let e x = d x

  let f x = e x + X.v
end
