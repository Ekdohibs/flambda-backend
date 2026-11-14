(* A function that returns an option, inlined where its result is matched and
   [None] fails an assertion. Simplify unboxes the option: the three ends of the
   function come to one continuation with two parameters, whether the result is
   [None] and the field of [Some], and its handler tests the first. One arm
   returns the field as it is; the other raises [Assert_failure], which is all
   that the two ends that were [None] do. *)

external ( = ) : int -> int -> bool = "%equal"

let get (t : int list) i =
  match t with [] -> None | x :: _ -> if i = 0 then Some x else None

let nth t i =
  match (get [@inlined]) t i with Some x -> x | None -> assert false
