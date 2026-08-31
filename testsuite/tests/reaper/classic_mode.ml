(* TEST
   flambda2;
   flags += "-Oclassic -flambda2-reaper";
   { native; check-program-output; }
 *)

(* Check that the reaper runs correctly in classic mode. *)

external opaque : 'a -> 'a = "%opaque"

let[@inline never] [@local never] make_pair y = 123456, y

let[@inline never] [@local never] use () =
  let p = make_pair (opaque 42) in
  snd p

let[@inline never] [@local never] apply_closure () =
  let x = opaque 10 in
  let f y = x + y in
  (opaque f) 5

let () =
  print_int (use ());
  print_newline ();
  print_int (apply_closure ());
  print_newline ()
