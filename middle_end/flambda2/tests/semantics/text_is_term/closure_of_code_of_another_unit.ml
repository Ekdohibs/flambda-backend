(* Compiled with -Oclassic, after other_unit.ml. The call of [Other_unit.adder]
   is replaced by the body of that function before Simplify, and the body makes
   the closure of a partial application, whose code is code of [Other_unit]. The
   text writes the code ID of that code by its name alone, and the reader stops
   on a closure of code that the text does not bind: the text is not read back
   (discrepancy X2-9). *)

let g x = Other_unit.adder x
