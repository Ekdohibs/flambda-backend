(* A functor applied twice, the first application marked to be inlined and the
   second marked never to be. Simplify inlines the functor at the first and
   leaves the call of it at the second. *)

external ( + ) : int -> int -> int = "%addint"

external opaque : 'a -> 'a = "%opaque"

module type Shift = sig
  val by : int
end

module Add (S : Shift) = struct
  let apply x = x + S.by

  let twice x = apply (apply x)
end

module Add_three = Add [@inlined] (struct
  let by = 3
end)

module Add_five = Add [@inlined never] (struct
  let by = 5
end)

let result = Add_three.twice (opaque 4) + Add_five.twice (opaque 1)
