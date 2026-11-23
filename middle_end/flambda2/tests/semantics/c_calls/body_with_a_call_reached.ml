(* A control: the run of this unit calls the function of [Body_with_a_call] in
   the arm that calls the C function. The body is bound in front of the unit;
   the run enters it and is undefined at the call of the C function, which the
   tool names. Whatever the validation says, it says nothing of this unit. *)

let result = Body_with_a_call.apply_if true 3
