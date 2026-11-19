(* The first unit of three, of the shape of a file of the library that another
   file uses: a function whose body calls a C function in one arm, and returns
   without calling one in the other. The run of this unit calls nothing. Its two
   second units are checked with the .cmx of this one: the body is then bound in
   front of them, and their runs enter it (see dune). The function is marked
   [@inline] so that the .cmx keeps its body. Without the mark the file keeps
   the metadata of the code alone; the code is then declared deleted in front of
   a second unit, as it was before this reading, and a run that calls it is
   undefined at that call. *)

external work : int -> int = "c_work"

let[@inline] apply_if flag x = if flag then work x else x
