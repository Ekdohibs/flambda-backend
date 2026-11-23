open Datatypes

module Nat = struct
  (** val eqb : nat -> nat -> bool **)

  let rec eqb n m =
    match n with
    | O -> ( match m with O -> true | S _ -> false)
    | S n' -> ( match m with O -> false | S m' -> eqb n' m')

  (** val leb : nat -> nat -> bool **)

  let rec leb n m =
    match n with
    | O -> true
    | S n' -> ( match m with O -> false | S m' -> leb n' m')

  (** val ltb : nat -> nat -> bool **)

  let ltb n m = leb (S n) m

  (** val max : nat -> nat -> nat **)

  let rec max n m =
    match n with
    | O -> m
    | S n' -> ( match m with O -> n | S m' -> S (max n' m'))

  (** val min : nat -> nat -> nat **)

  let rec min n m =
    match n with
    | O -> O
    | S n' -> ( match m with O -> O | S m' -> S (min n' m'))

  (** val iter : nat -> ('a1 -> 'a1) -> 'a1 -> 'a1 **)

  let rec iter n f x = match n with O -> x | S n0 -> f (iter n0 f x)
end
