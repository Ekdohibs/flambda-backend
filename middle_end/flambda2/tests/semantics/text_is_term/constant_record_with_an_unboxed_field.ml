(* A constant record with an unboxed field is a static block that has the shape
   of a mixed block. The text writes the kinds of the fields of such a block
   after its tag ([Block 0 of val * float (1, 0x1p+1)]), and the reader makes
   the shape from them; it wrote none, and the block was read as a block of
   values. So the text is the term. *)

type t =
  { a : int;
    b : float#
  }

let r = { a = 1; b = #2.0 }
