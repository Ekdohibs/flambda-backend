(* The control of chain_of_five_in_function.ml: three closures and the block,
   four values, each made of the one before. (By T4.) *)

external ( + ) : int -> int -> int = "%addint"

let make () =
  let a x = x + 1 in
  let b x = a x in
  let c x = b x in
  a, b, c
