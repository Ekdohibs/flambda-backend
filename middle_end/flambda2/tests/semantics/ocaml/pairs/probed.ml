(* The first unit of a pair: a function that is inlined wherever it may be, and
   whose body has a probe. The text of its term says that the call is a probe,
   and the tool does not check the unit: the model has nothing of a probe, and
   the conversion refuses a call that is one. *)

external ( + ) : int -> int -> int = "%addint"

let[@inline never] note (_ : int) = ()

let[@inline] bump x =
  [%probe "bump" (note x)];
  x + 1
