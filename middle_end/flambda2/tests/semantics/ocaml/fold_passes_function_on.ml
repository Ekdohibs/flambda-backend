(* A fold over a list, inlined where it is called with a closure. Its self tail
   call is a jump, so what is inlined is a loop with the three parameters of the
   function: the function to apply, which every turn passes on as it is, the
   result so far, which changes, and the list. Simplify inlines the closure in
   the loop, and the term that it gives calls nothing outside the code of the
   function. *)

external ( + ) : int -> int -> int = "%addint"

external opaque : 'a -> 'a = "%opaque"

let[@inline] rec fold : (int -> int -> int) -> int -> int list -> int =
 fun f acc l -> match l with [] -> acc | x :: t -> fold f (f acc x) t

let result = fold (fun a x -> a + x) 0 (opaque [1; 2; 3])
