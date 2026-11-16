(* flags: -w -58 *)
(* load path: the first unit alone *)

(* A control: the unit of never_called_again.ml, with the tool given the .cmx of
   the second unit and no other, so that the unit of the third is not read. The
   prefix binds the code of the second unit, which names the code of the third,
   and declares nothing of a unit that was not read: the check of static names
   fails of both terms with that prefix in front of them, and the guard of the
   tool stops the unit ("not checked"). *)

let three = Never_called.add_two 1
