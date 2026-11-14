(* An exhaustive match from four constant constructors to four integers that
   follow no rule, of which Simplify makes a table, and an immutable array of
   the source that has the same four integers, read by a second function. The
   term after has two static arrays with the same elements. The tool first
   gives the array of the source the symbol of the table, and the two terms
   are then the same but for those two symbols; it guesses again with the
   other symbol, and the unit is validated. *)

external ( + ) : int -> int -> int = "%addint"

external opaque : 'a -> 'a = "%opaque"

external get : 'a iarray -> int -> 'a = "%array_unsafe_get"

type dir =
  | North
  | East
  | South
  | West

let[@inline never] [@local never] by_match d =
  match d with North -> 3 | East -> 1 | South -> 4 | West -> 15

let table : int iarray = [: 3; 1; 4; 15 :]

let[@inline never] [@local never] by_table i = get table i

let result = by_match (opaque West) + by_table (opaque 3)
