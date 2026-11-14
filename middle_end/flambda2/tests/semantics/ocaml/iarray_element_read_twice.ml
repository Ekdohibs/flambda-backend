(* A function that Simplify does not inline reads one element of an immutable
   array twice. Simplify shares the two loads. The validation shares them too:
   the load of an element of an immutable array is among the primitives that
   it does not compute again (FlambdaCseSpec.shareable). *)

external ( + ) : int -> int -> int = "%addint"

external opaque : 'a -> 'a = "%opaque"

external get : 'a iarray -> int -> 'a = "%array_unsafe_get"

let[@inline never] twice (a : int iarray) i = get a i + get a i

let result = twice [: 4; opaque 6 :] 1 + twice [: 4; 6 :] (opaque 0)
