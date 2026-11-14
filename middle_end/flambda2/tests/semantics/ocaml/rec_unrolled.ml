(* A recursive function unrolled three times at one call. Simplify inlines
   [length] at the call in [three] and at the call that each inlining brings,
   three times in all, and leaves the fourth; the call of [length] to itself in
   its own code is as it was. *)

external ( + ) : int -> int -> int = "%addint"

let rec length (l : int list) = match l with [] -> 0 | _ :: t -> 1 + length t

let three () = (length [@unrolled 3]) [1; 2; 3]

let result = three ()
