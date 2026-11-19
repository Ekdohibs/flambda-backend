(* The first unit of two: a function, kept in the .cmx by its mark, whose body
   calls a C function and stores what it returns in a cell that it is given. The
   store is a primitive of the heap. The run of this unit calls nothing. Its
   second unit is checked with the .cmx of this one (see dune). *)

type cell = { mutable last : int }

external work : int -> int = "c_work"

let[@inline] store cell x = cell.last <- work x
