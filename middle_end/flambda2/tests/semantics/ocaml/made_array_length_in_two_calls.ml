(* A function takes the length of its argument, tests it and uses it. It is
   called twice, each time on an array that is made at the call, and with the
   default flags Simplify inlines neither call. The validation inlines both
   calls in the two terms; the length is then that of an array made by a [Let]
   in scope, in the term after as in the term before, and the rounds of each
   make it its constant. The unit is validated without the rule too; it is here
   so that the rule stays in the rounds of both terms. *)

external opaque : 'a -> 'a = "%opaque"

external ( + ) : int -> int -> int = "%addint"

external ( = ) : int -> int -> bool = "%equal"

external length : 'a array -> int = "%array_length"

external unsafe_get : 'a array -> int -> 'a = "%array_unsafe_get"

let pick (a : int option array) default =
  if length a = 0
  then default
  else match unsafe_get a 0 with None -> default | Some n -> n + length a

let result =
  pick [| opaque (Some 4); opaque None |] 9 + pick [| opaque None |] 1
