(* A function that returns a function of the unboxed product of nothing,
   given its argument and that product by one call of unknown arity: an
   over-application. Its second argument is a component of the arity, [#()],
   which gives no kind once the arity is flattened. *)

external opaque : 'a -> 'a = "%opaque"

let[@inline never] h (c : int) = fun #() -> c

let result = (opaque h) 5 #()
