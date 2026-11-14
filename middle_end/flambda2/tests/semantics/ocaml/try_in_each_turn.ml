(* A recursive function over a list with a handler at each element: a quotient
   that cannot be computed counts for nothing. *)
external ( / ) : int -> int -> int = "%divint"

external ( + ) : int -> int -> int = "%addint"

external opaque : 'a -> 'a = "%opaque"

let rec sum_of_quotients n divisors =
  match divisors with
  | [] -> 0
  | d :: rest ->
    (try n / d with Division_by_zero -> 0) + sum_of_quotients n rest

let result = sum_of_quotients (opaque 60) [1; 0; 2; 0; 3]
