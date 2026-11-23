type tree =
  | Leaf
  | Node of tree * int * tree

val with_node : unit -> (tree -> int -> tree -> tree) * int

val either : bool -> (tree -> int -> tree -> tree) * int
