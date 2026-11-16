(* The test for an immediate that works on any value ([Obj.is_int]). The text
   writes it [%is_int.generic]; it wrote [%is_int], which is read as the test
   for a value of a variant type. So the text is the term. *)

external is_int : 'a -> bool = "%obj_is_int"

let f x = is_int x
