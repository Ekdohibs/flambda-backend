(* A unit that makes a block with a field that is an immediate. The text of its
   term writes the kinds of the fields of a block that is made, on the
   arguments, where they are not all that of any value; so the text is the
   term. *)

external opaque : 'a -> 'a = "%opaque"

let p = opaque 1, 2
