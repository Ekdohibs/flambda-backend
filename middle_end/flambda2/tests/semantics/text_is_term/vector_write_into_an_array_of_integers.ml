(* A write of one 128-bit vector over two elements of an array of integers. The
   write has the kind of an array of immediates and takes a vector, and the
   text says both ([%array_set] with [imm] and [vec128]); it wrote the first
   only, and the write was read as one that takes an immediate. So the text is
   the term. *)

external set : int array -> int -> int64x2# -> unit
  = "%caml_int_array_set128u#"

let first_two (a : int array) (v : int64x2#) = set a 0 v
