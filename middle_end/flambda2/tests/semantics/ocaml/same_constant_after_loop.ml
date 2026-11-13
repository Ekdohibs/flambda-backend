(* A condition, or else a comparison with what a loop leaves in a reference that
   it never changes. Simplify finds on its way back up the term that the
   reference still holds what it held before the loop, and leaves the comparison
   where it was: the continuation that the two paths come to keeps its
   parameter. Once the comparison is folded, both paths give that parameter the
   same constant. *)

external ( < ) : int -> int -> bool = "%lessthan"

external ( <> ) : int -> int -> bool = "%notequal"

external ( + ) : int -> int -> int = "%addint"

external ( || ) : bool -> bool -> bool = "%sequor"

external opaque : 'a -> 'a = "%opaque"

type 'a ref = { mutable contents : 'a }

external ref : 'a -> 'a ref = "%makemutable"

external ( ! ) : 'a ref -> 'a = "%field0"

external ( := ) : 'a ref -> 'a -> unit = "%setfield0"

let unchanged () =
  let i = ref 0 in
  let acc = ref 7 in
  while !i < 4 do
    acc := !acc;
    i := !i + 1
  done;
  !acc

let result : bool = opaque 1 < 2 || 9 <> unchanged ()
