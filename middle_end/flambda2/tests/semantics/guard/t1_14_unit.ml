(* A unit whose value [r] is [Some (S "ccc")] (label T1-14). With
   -flambda2-match-in-match, at -O2 and at -O3, the term that Simplify makes of
   it binds the block of [Some] in the handler of one arm of a test and mentions
   its symbol in the handler of the other arm: the field of the block that holds
   it is never written, and [r] is read as [Some] of the immediate 0. The file
   must keep its name: of the same text under another name Simplify makes
   another term, which binds the block before the test. *)
type v =
  | A
  | C of int
  | S of string

type cell = { mutable m : v }

let[@inline always] f c0 =
  match c0 with
  | { m = C _ } -> None
  | _
    when c0.m <- A;
         false ->
    None
  | { m = A } -> None
  | { m = S _ as z } -> Some z

let r = f { m = S "ccc" }
