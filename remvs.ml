type 'a box = BOX of 'a
type ('a, 'b) t = A of 'a | B of 'b

let[@inline] tt x =
  let () = () in (fun () -> x)

let g b c h y z =
  let uu = if c then BOX (BOX (A (tt (if b then y else z)))) else BOX (BOX (B z)) in
  match uu with
  | BOX (BOX (A f)) -> (f[@inlined]) ()
  | BOX (BOX (B x)) -> x

let[@inline never] ww c =
  (g[@inlined]) true c () (BOX 12) (BOX 12)

let BOX x = ww true