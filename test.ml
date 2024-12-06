(* TEST
 ocamlopt_flags += " -O3 ";
*)

let f () = assert false

let l = (Lazy.from_fun[@inlined never]) f

let _ =
  try Lazy.force l
  with Lazy.Undefined -> print_endline "Undefined"
