(* A control of raise_made_block_inlined.ml: the function that is inlined raises
   an exception that has no argument, so that no block is made. Simplify inlines
   the call, and the code of [get] raises the symbol of the exception, which the
   unit imports. *)

external raise : exn -> 'a = "%raise"

let[@inline] fail () : int = raise Not_found

let get (o : int option) : int = match o with Some x -> x | None -> fail ()
