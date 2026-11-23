(* The first unit of two: a function of a pair, kept in the .cmx by its mark,
   whose body calls a C function. The compiler makes tupled code of it, and the
   conversion binds no tupled code of another unit: it converts the body, meets
   the C function there, and declares the code deleted. The run of this unit
   calls nothing. Its second unit is checked with the .cmx of this one (see
   dune). *)

external work : int -> int = "c_work"

let[@inline] apply_to_first (x, _) = work x
