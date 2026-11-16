(* With -Oclassic a recursive call that the source marks [@unrolled 3] keeps the
   attribute and is not inlined, and the term says that the attribute is not
   used and why. The text says so too ([unroll(3,
   unused_because_function_unknown)]); it wrote [unroll(3)], which is read as an
   attribute that is expected to be used. So the text is the term. *)

external ( - ) : int -> int -> int = "%subint"

external ( <= ) : int -> int -> bool = "%lessequal"

let rec f (x : int) = if x <= 0 then 0 else (f [@unrolled 3]) (x - 1)
