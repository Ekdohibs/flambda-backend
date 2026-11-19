type tree =
  | Leaf
  | Node of tree * int * tree

val of_count : int -> tree

val of_any : int -> tree
