(* A function whose second parameter is the unboxed product of nothing, called
   directly. That product is a component of the arity, [#()], which gives no
   kind once the arity is flattened. *)

external opaque : 'a -> 'a = "%opaque"

let[@inline never] h (c : int) #() = c

let result = h (opaque 5) #()
