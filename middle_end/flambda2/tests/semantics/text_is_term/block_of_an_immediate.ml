(* A unit that makes a block with a field that is an immediate. The text of its
   term does not write the kinds of the fields of a block that is made, so the
   text is not the term; the driver names that difference. *)

external opaque : 'a -> 'a = "%opaque"

let p = opaque 1, 2
