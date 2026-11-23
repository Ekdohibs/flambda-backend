(* A unit for the two controls that have code of another unit. It is compiled
   first, with the default options of the optimiser and not with -Oclassic, so
   that what it keeps for other units has the code of its three functions. The
   second calls the first, and the third applies the first to one argument of
   its two: the unit that calls the second or the third with -Oclassic has that
   call replaced by the body, and so has code of this unit in its term, [add] in
   the one and the code of the partial application in the other. *)

external ( + ) : int -> int -> int = "%addint"

let[@inline never] add x y = x + y

let add_one x = add x 1

let[@inline] adder x = add x
