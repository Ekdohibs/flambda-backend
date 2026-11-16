(* A unit with a division, which is checked: the continuation that raises
   [Division_by_zero] is cold. The text says of a continuation that it is cold,
   so the text is the term. *)

external ( / ) : int -> int -> int = "%divint"

external opaque : 'a -> 'a = "%opaque"

let q = 7 / opaque 2
