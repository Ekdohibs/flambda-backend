module Comparison : sig
  (* Given two things x and y, a judgment of whether they are alpha-equivalent.
   * In the case that they are not, this includes an "approximator," a term
   * that's alpha-equivalent to x and as close as possible to y.  The
   * approximator can be used to generate a diff that (hopefully) eliminates
   * spurious differences. *)
  type 'a t =
    | Equivalent (* x and y are alpha-equivalent *)
    | Different of { approximant : 'a }
  (* x and y are not alpha-equivalent, and furthermore [approximant] is
   * alpha-equivalent to x and textually the same as y in places where x and
   * y agree *)

  val print :
    (Format.formatter -> 'a -> unit) -> Format.formatter -> 'a t -> unit
end

val flambda_units :
  Flambda_unit.t -> Flambda_unit.t -> Flambda_unit.t Comparison.t

(** Like [flambda_units], with names for what the two units were found to differ
    in, in the order in which they were found; the first is the first difference
    in the order in which the units are walked. The names are fixed strings
    (with, for a primitive, its name), so that units can be counted by them. The
    list is empty if the units are equivalent. It need not have every
    difference: where two things differ, what stands beside them is in some
    places not compared any further. *)
val flambda_units_and_differences :
  Flambda_unit.t -> Flambda_unit.t -> Flambda_unit.t Comparison.t * string list
