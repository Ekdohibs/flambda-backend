(* The second unit of three. It keeps a closure that the function of the first
   unit makes, of a value that is not known. The call is inlined, and Simplify
   makes no new code for the closure, as it finds nothing to improve in it: the
   closure is a symbol of this unit, and its code is still that of the first
   unit. This unit also gives the function of the first unit again. *)

external opaque : int -> int = "%opaque"

let add = Maker.make (opaque 1)

let make = Maker.make

let one = 1
