(* A function made tail recursive by [@tail_mod_cons] makes each cell of its
   result with a field to be filled later, and fills it with a write that is the
   first write of the field, not an assignment. The text says so of the write
   ([%array_set] with [values] and [value.init]); it wrote neither, and the
   write was read as an assignment. So the text is the term. *)

let[@tail_mod_cons] rec map (f : int -> int) (l : int list) =
  match l with [] -> [] | x :: rest -> f x :: map f rest
