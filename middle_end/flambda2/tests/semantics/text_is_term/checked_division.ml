(* A unit with a division, which is checked: the continuation that raises
   [Division_by_zero] is cold. The text does not say of a continuation that it
   is cold, so the text is not the term; the driver names that difference. *)

external ( / ) : int -> int -> int = "%divint"

external opaque : 'a -> 'a = "%opaque"

let q = 7 / opaque 2
