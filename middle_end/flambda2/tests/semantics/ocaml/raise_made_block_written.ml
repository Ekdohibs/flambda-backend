(* A control of raise_made_block_inlined.ml: the same exception, made of the
   same constant string, where it is raised, with no function to inline. The
   front end makes the block of the exception a symbol, and the code of [get]
   raises that symbol before Simplify and after. *)

external raise : exn -> 'a = "%raise"

let get (o : int option) : int =
  match o with Some x -> x | None -> raise (Invalid_argument "get")
