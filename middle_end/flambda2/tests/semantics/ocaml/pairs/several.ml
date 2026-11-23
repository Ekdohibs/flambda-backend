(* The first unit of a pair: a module block with four fields of which the .cmx
   says nothing. *)

external opaque : 'a -> 'a = "%opaque"

let first = opaque 7

let second = opaque 8

let pair = opaque (1, 2)

let apply = opaque (fun (x : int) -> x)
