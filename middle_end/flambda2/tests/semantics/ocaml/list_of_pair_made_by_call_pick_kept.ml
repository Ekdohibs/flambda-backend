(* list_of_pair_made_by_call.ml with [pick] not marked to be inlined: its call
   stays in [run]. The first guess of what Simplify lifted gives the cell of the
   list the symbol of the pair here too, and the test tool guesses again as it
   does for that unit. Before it did, this unit was validated by a later
   comparison, the one that uses what the symbols are bound to. *)

external ( + ) : int -> int -> int = "%addint"

external ( - ) : int -> int -> int = "%subint"

let pick c = match c with 0 -> fun a -> a + 1 | _ -> fun a -> a - 1

let rec run acc program =
  match program with [] -> acc | (_, c) :: rest -> run (pick c acc) rest

let op c = true, c

let result = run 0 [op 0]
