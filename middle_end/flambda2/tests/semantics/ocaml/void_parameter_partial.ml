(* A function whose second parameter is the unboxed product of nothing, given
   its first argument alone by a call of unknown arity: a partial application,
   and the unit keeps the closure that it makes. That product is a component
   of the arity, [#()], which gives no kind once the arity is flattened. *)

external opaque : 'a -> 'a = "%opaque"

let[@inline never] h (c : int) #() = c

let p = (opaque h) 5
