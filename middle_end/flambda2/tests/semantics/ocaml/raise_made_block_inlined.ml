(* A function that raises an exception made of its argument, inlined where the
   argument is a constant string. Simplify inlines the call, makes the block of
   the exception a symbol, which the code of [get] raises, and deletes the code
   of [fail], which nothing uses any more: the interface keeps [fail] out of the
   module block. The term before has that code, which raises a variable that
   holds a block made of its parameter. *)

external raise : exn -> 'a = "%raise"

let[@inline] fail (reason : string) : int = raise (Invalid_argument reason)

let get (o : int option) : int = match o with Some x -> x | None -> fail "get"
