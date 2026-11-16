(* An identity that only the middle end does not see through ([Obj.magic]). The
   text writes it [%opaque.middle_end_only]; it wrote [%opaque], which is read
   as the identity that nothing sees through ([Sys.opaque_identity]). So the
   text is the term. *)

external magic : 'a -> 'b = "%obj_magic"

let f (x : int) : string = magic x
