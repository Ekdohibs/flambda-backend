open Datatypes

module Nat : sig
  val eqb : nat -> nat -> bool

  val leb : nat -> nat -> bool

  val ltb : nat -> nat -> bool

  val max : nat -> nat -> nat

  val min : nat -> nat -> nat

  val iter : nat -> ('a1 -> 'a1) -> 'a1 -> 'a1
end
