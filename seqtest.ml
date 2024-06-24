external (+) : int -> int -> int = "%addint"
external ( * ) : int -> int -> int = "%mulint"
external (>) : 'a -> 'a -> bool = "%greaterthan"

module Option = struct
  type 'a t = { f : 'c. 'c -> ('a -> 'c) -> 'c } [@@unboxed]
  let none = { f = fun[@inline] x _ -> x }
  let[@inline] some x = { f = fun[@inline] _ g -> (g[@inlined hint]) x }
  let[@inline] match_ { f } none some =
    (f[@inlined hint]) none some
end

module Seq = struct

  type _ t = State : ('s * ('s -> ('a * 's) Option.t)) -> 'a t

  let[@inline] fold_left fold_left_next f acc (State (s, next)) =
    Option.match_ ((next[@inlined hint]) s)
      acc (fun[@loop always] (x, s') -> fold_left_next f (f acc x) (State (s', next)))

  let[@inline] fold_left f acc (State (s, next)) =
    let[@inline][@loop always] rec aux acc s =
      Option.match_ ((next[@inlined hint]) s)
        acc (fun[@loop always] (x, s') -> (aux[@inlined]) (f acc x) s')
    in
    aux acc s

  let[@inline] map f (State (s, next)) =
    State (s, fun s ->
      Option.match_ ((next[@inlined hint]) s)
        Option.none
        (fun (x, s') -> Option.some (f x, s')))

  let[@inline] unfold f acc =
    State (acc, f)

end

let[@inline] square x = x * x

let[@inline] ints lo hi =
  Seq.unfold (fun i -> if i > hi then Option.none else Option.some (i, i + 1)) lo

let[@inline] sum s =
  Seq.fold_left (+) 0 s

let foo () =
  sum (Seq.map square (ints 0 11))