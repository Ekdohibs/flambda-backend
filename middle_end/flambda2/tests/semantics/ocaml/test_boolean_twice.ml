(* A boolean tested, then tested again and negated. Simplify knows the boolean
   in each arm of the first test: the second tests are gone, and each arm of the
   first returns its number. *)

external ( + ) : int -> int -> int = "%addint"

external ( * ) : int -> int -> int = "%mulint"

external ( = ) : int -> int -> bool = "%equal"

external not : bool -> bool = "%boolnot"

external opaque : 'a -> 'a = "%opaque"

let f b = if b then if b then 1 else 2 else if not b then 3 else 4

let result = (f (opaque true) * 10) + f (opaque false)
