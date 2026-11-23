(* Compiled with -Oclassic, after other_unit.ml. The call of
   [Other_unit.add_one] is replaced by the body of that function before
   Simplify, and the body calls [Other_unit.add]: the term has a direct call of
   code of another unit. The text writes the code ID of that code by its name
   alone, as it writes one of code that the unit binds, and the reader, which
   finds no code of that name, makes a new code ID of the unit that it reads. So
   the text is not the term: it is read as a call of other code. The reader
   cannot do otherwise: nothing makes the code ID of another unit from what a
   text can say (discrepancy X2-9). *)

let f x = Other_unit.add_one x
