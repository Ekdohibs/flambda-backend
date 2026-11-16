(* A unit of which the check reads a section of the .cmx file of another unit
   that the compilation has open: the code of [Format.asprintf], in a file of
   the standard library that is longer than the buffer of a channel. *)

let shown x = Format.asprintf "%d" x
