(* Four functions, each calling the one before, in a functor that does not use
   its argument. Before Simplify each closure holds the one before in a value
   slot, in the code of the functor, and the block of the functor's result holds
   the four: five values, each made of the one before. Simplify makes all five
   symbols, and the code of the functor returns the symbol of the block. A
   search that lifts a value only once its parts are symbols has to lift five
   times in turn here. The functor [Make] of the standard library's [Map] has
   such a chain, a longer one. (Found and reduced by T4.) *)

external ( + ) : int -> int -> int = "%addint"

module Make (X : sig end) = struct
  let a x = x + 1

  let b x = a x

  let c x = b x

  let d x = c x
end
