(* As chain_of_five_in_functor.ml, with no functor: a function that makes four
   closures, each calling the one before, and returns them in a block. Simplify
   makes the four closures and the block symbols, and the code of the function
   returns the symbol of the block. (By T4.) *)

external ( + ) : int -> int -> int = "%addint"

let make () =
  let a x = x + 1 in
  let b x = a x in
  let c x = b x in
  let d x = c x in
  a, b, c, d
