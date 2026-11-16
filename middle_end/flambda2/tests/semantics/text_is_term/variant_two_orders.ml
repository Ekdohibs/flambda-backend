(* Two functions with a parameter that is a [t2].

   [f] matches on a GADT, so the front end takes the kinds of its two parameters
   (it is a function of a pair, which the front end makes a function of two
   parameters) to be the join of the kinds of the patterns of its cases
   ([Translcore.join_layout_of_cases], [Lambda.join_value_kind]): in the first
   case the second component is a [t1], in the second a [t2], and the join of
   the two kinds has the constructors of [t2], which [Lambda] sorts in
   increasing order.

   [h] has one type for its parameter, and the kind is the one that [Typeopt]
   gives [t2], with the constructors in the order in which it meets them, the
   last first. *)
type t1 =
  | A1
  | C1 of int

type t2 =
  | A2
  | B2
  | C2 of int

type _ w =
  | W1 : t1 w
  | W2 : t2 w

let f : type a. a w * a -> int = function W1, _ -> 1 | W2, _ -> 2

let h (x : t2) = match x with A2 -> 0 | _ -> 1
