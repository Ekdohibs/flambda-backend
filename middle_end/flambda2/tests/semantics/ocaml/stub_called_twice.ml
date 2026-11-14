(* A partial application of a partial application. The wrapper of [combine 7] is
   the code of a stub. Simplify leaves the call of it in the wrapper of
   [from_seven 8], which is the code of a stub too, and inlines it at
   [from_seven 1 2]. *)

external ( + ) : int -> int -> int = "%addint"

external ( * ) : int -> int -> int = "%mulint"

let combine a b c = (a * 100) + (b * 10) + c

let from_seven = combine 7

let from_seven_eight = from_seven 8

let seven_one_two = from_seven 1 2
