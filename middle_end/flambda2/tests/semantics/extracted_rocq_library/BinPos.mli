open BinNums
open Datatypes

module Pos : sig
  val succ : positive -> positive

  val compare_cont : comparison -> positive -> positive -> comparison

  val compare : positive -> positive -> comparison

  val eqb : positive -> positive -> bool

  val leb : positive -> positive -> bool

  val max : positive -> positive -> positive

  val ltb : positive -> positive -> bool

  val eq_dec : positive -> positive -> bool
end
