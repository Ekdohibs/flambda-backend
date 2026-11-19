(* A control: the run of this unit calls the function of [Body_with_a_call] in
   the arm that calls no C function. The body is bound in front of the unit; the
   run enters it and returns, and the verdict of the unit is of that run. With
   the code of the body declared deleted in front of the unit, the run is
   undefined at the call of the body. *)

let result = Body_with_a_call.apply_if false 3
