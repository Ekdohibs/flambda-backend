(* The same as copy_heap.ml, with a parameter that may be on the stack of
   the caller: its mode is not heap in the compiler's term. The two ways
   print the same of this unit. T3's source. *)

external fst : ('a * 'b[@local_opt]) -> 'a = "%field0_immut"

external snd : ('a * 'b[@local_opt]) -> 'b = "%field1_immut"

let copy (local_ (pair : int * int)) = fst pair, snd pair
