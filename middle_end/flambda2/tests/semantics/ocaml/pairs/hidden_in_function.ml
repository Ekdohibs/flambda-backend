(* A function of this unit loads the field of another unit of which the .cmx
   says nothing, from the module block of that unit, and returns it. Nothing
   calls the function. *)

let[@inline never] get () = Hidden.hidden
