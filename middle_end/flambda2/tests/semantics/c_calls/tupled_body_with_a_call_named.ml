(* A control of what is declared in front of a unit. This unit names the
   function of [Tupled_body_with_a_call] and does not call it. The body of that
   function is converted and then not bound, so nothing that is in front of this
   unit calls the C function [c_work]: it is not declared, and the tool prints
   no line of a C function. *)

let apply_to_first = Tupled_body_with_a_call.apply_to_first
