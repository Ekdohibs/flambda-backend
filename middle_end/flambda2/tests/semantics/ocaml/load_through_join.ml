(* A value that one branch of a test builds with the first constructor and the
   other with the second, matched where the two branches meet. Simplify gives
   the continuation where they meet the tag and the fields, and leaves the match
   there as a Switch on the tag, to which each branch gives a number. The arm of
   the third constructor cannot run: no branch gives its number, and nothing
   gives the fields that it adds up. *)

external ( + ) : int -> int -> int = "%addint"

external opaque : 'a -> 'a = "%opaque"

type t =
  | A of int
  | B of int * int
  | C of int * int * int

let[@inline] sum x =
  match x with A a -> a | B (a, b) -> a + b | C (a, b, c) -> a + b + c

let f c n =
  let v = if c then A n else B (n, 1) in
  sum v

let result = f (opaque true) (opaque 4) + f (opaque false) (opaque 4)
