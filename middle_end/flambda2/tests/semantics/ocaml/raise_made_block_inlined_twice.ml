(* As raise_made_block_inlined.ml, with two functions that each give the
   function that raises a constant string of their own: Simplify inlines both
   calls and makes two blocks of exceptions symbols, one raised by the code of
   each function. *)

external raise : exn -> 'a = "%raise"

type t =
  | A of int
  | B of int

let[@inline] fail (reason : string) : int = raise (Invalid_argument reason)

let get_a (v : t) : int = match v with A x -> x | B _ -> fail "get_a"

let get_b (v : t) : int = match v with B x -> x | A _ -> fail "get_b"
