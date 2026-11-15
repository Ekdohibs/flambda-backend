(* Two recursive functions that capture nothing, local to the value of the unit.
   The first is made below a value that is not known and is not a loop: it adds
   one to what its own call returns, so its code calls it by name. The second is
   made in an arm of a test. Simplify makes both closures symbols, and the code
   of the first calls that symbol.

   For the validation the two closures of the term before are lifted to those
   symbols. Lifted where its [let] stands, the first is bound below the value
   that is not known, and its code, which is bound first, names it; the second,
   which is in an arm, is lifted to the head of the unit. The two terms then
   differ by the place of one binding. The tool searches once more after the
   refusal, and its guess then gives the first closure to the lifting to the
   head too. *)

external ( + ) : int -> int -> int = "%addint"

external ( - ) : int -> int -> int = "%subint"

external ( <= ) : int -> int -> bool = "%lessequal"

external ( || ) : bool -> bool -> bool = "%sequor"

external opaque : 'a -> 'a = "%opaque"

let result =
  let x = opaque 9 in
  let rec depth (n : int) = if n <= 0 then 5 else depth (n - 1) + 1 in
  depth (opaque 2) <= x
  || (let rec count (n : int) = if n <= 0 then 7 else count (n - 1) in
      count (opaque 3))
     <= 9
