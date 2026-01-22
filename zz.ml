external (+) : int -> int -> int = "%addint"
external opaque : 'a -> 'a = "%opaque"
type t = A of int | B of int | C of int

let[@inline] f y = match y with A x -> B x | B x -> C x | C x -> A (x + 1)

let g h y =
  let y = match y with A x -> B x | B x -> C x | C x -> A (x + 1) in
  h y 0

let g y =
 g (fun[@inline] x p -> let h = opaque p in
   let h = h + 1 in
   let h = h + 1 in
   let h = h + 1 in
   let h = h + 1 in
   let h = h + 1 in
   let h = h + 1 in
   (* let h = h + 1 in *)
 h, f x) y