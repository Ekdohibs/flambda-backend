external (+) : int -> int -> int = "%addint"

let test x =
  let[@inline never][@local never] f y = let () = () in fun z -> x + y + z in
  let[@inline never][@local never] g f = f x x in
  g f