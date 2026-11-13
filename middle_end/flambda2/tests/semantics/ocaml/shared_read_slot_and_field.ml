(* A value that is computed when the unit runs is held by a closure, as the
   value of one of its slots, and by the block of the module, as a field. A
   function reads it from both: once it is read from the closure, the read of
   the field is that value. *)

external ( + ) : int -> int -> int = "%addint"

external ( * ) : int -> int -> int = "%mulint"

external opaque : 'a -> 'a = "%opaque"

module Prices = struct
  let each = opaque 5

  let of_items n = n * each
end

let cost boxes = Prices.of_items boxes + Prices.each
