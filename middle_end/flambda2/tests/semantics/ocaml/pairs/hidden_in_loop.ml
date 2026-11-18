(* A loop at toplevel loads, each time round, the field of another unit of which
   the .cmx says nothing, and gives it to the identity that Simplify does not
   look through. *)

external opaque : 'a -> 'a = "%opaque"

let () =
  for _i = 1 to 2 do
    let _ = opaque Hidden.hidden in
    ()
  done
