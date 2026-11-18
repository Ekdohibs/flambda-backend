(* A list of one pair, the pair made by a call: [op 0] is [(true, 0)], a pair
   that ends in 0. Simplify inlines [op] and makes the pair and the list
   symbols. Where the test tool first guesses what Simplify lifted, the pair is
   still the result of the call, and the cell of the list, a block of a variable
   and 0, agrees with the pair, a block of 1 and 0: the cell takes the symbol of
   the pair. Once the validation has inlined [op], the pair that it brings has
   no symbol left to take. The tool sees that the symbol is bound to a block
   that cannot be the one that the term after binds it to (its first field is
   the pair, a block, where the term after has a constant), and guesses again
   with the cell banned for that symbol: the cell then takes the symbol of the
   list and the pair its own. [pick], which returns one of two closures, is
   marked to be inlined, and Simplify inlines it into [run]: without that the
   unit was validated all the same, by the comparison that uses what the symbols
   are bound to (list_of_pair_made_by_call_pick_kept.ml). *)

external ( + ) : int -> int -> int = "%addint"

external ( - ) : int -> int -> int = "%subint"

let[@inline] pick c = match c with 0 -> fun a -> a + 1 | _ -> fun a -> a - 1

let rec run acc program =
  match program with [] -> acc | (_, c) :: rest -> run (pick c acc) rest

let op c = true, c

let result = run 0 [op 0]
