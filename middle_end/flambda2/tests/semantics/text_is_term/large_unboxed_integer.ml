(* Unboxed 64-bit and native integers that an OCaml integer of 63 bits does not
   hold. The reader reads the digits of a constant as its suffix says; it read
   them all as an OCaml integer first, and stopped on these. So the text is the
   term. *)

let f () = #4611686018427387904L

let g () = -#9223372036854775808L

let h () = #4611686018427387904n
