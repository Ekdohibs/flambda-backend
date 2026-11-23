(* A read of two elements of an array of integers as one 128-bit vector. The
   read has the kind of an array of immediates and gives a vector, and the text
   says both ([%array_load] with [imm] and [vec128]); it wrote the first only,
   and the read was read as one that gives an immediate. So the text is the
   term. *)

external get : int array -> int -> int64x2# = "%caml_int_array_get128u#"

let first_two (a : int array) = get a 0
