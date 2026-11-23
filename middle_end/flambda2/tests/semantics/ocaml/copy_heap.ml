(* A pair built from the two fields of a pair that is a parameter. The parameter
   has the mode heap in the compiler's term, and the compiler's Simplify puts it
   in the place of the new pair. The text of the term has no modes of
   parameters, and the tool's Simplify keeps the two loads and the pair. So the
   two ways do not print the same of this unit: it has copy_heap.own_expected
   (README.md, "The rule"). T3's source. *)

external fst : 'a * 'b -> 'a = "%field0_immut"

external snd : 'a * 'b -> 'b = "%field1_immut"

let copy (pair : int * int) = fst pair, snd pair
