(* The control of chain_of_five_in_functor.ml: three functions and the block of
   the functor's result, four values, each made of the one before. Simplify
   makes all four symbols. (By T4.) *)

external ( + ) : int -> int -> int = "%addint"

module Make (X : sig end) = struct
  let a x = x + 1

  let b x = a x

  let c x = b x
end
