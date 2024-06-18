let rec f =
  let x = if Sys.opaque_identity false then [f] else [] in
  fun y ->
    let rec aux s l =
      match l with
      | [] -> s
      | g :: l -> aux (s + g y) l
    in
    aux 0 x
