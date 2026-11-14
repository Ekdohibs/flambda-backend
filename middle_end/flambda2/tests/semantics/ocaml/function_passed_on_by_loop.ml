(* A recursive function that passes on to itself, as it is, the function that it
   is given, inlined where it is called with a closure. Its self tail call is a
   jump, so what is inlined is a loop, and the loop has the two parameters of
   the function. The first holds the closure at every turn; Simplify inlines the
   closure in the loop, and the term that it gives calls nothing outside the
   code of the function. *)

external ( > ) : int -> int -> bool = "%greaterthan"

external opaque : 'a -> 'a = "%opaque"

let rec exists : (int -> bool) -> int list -> bool =
 fun p l ->
  match l with [] -> false | x :: t -> if p x then true else exists p t

let result = (exists [@inlined]) (fun x -> x > 2) (opaque [1; 2; 3])
