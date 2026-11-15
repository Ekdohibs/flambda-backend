(* A function makes a closure that holds its parameter, and then a recursive
   function that holds that closure and gives it to a function that is always
   inlined, which calls it twice; it returns the recursive function. In the code
   of [sum], Simplify knows the code of the closure that the value slot holds,
   from where the set of [sum] is made: it makes the two calls of [f] direct
   once [twice] is inlined, and inlines them. The closure of [scale] may be a
   different value at different calls of [sum_scaled]; what is known of the
   value slot is its code. *)

external ( + ) : int -> int -> int = "%addint"

external ( - ) : int -> int -> int = "%subint"

external ( * ) : int -> int -> int = "%mulint"

external ( <= ) : int -> int -> bool = "%lessequal"

external opaque : 'a -> 'a = "%opaque"

let[@inline] twice (f : int -> int) (n : int) = f (f n)

let sum_scaled (factor : int) =
  let scale (n : int) = n * factor in
  let rec sum (n : int) (acc : int) =
    if n <= 0 then acc else sum (n - 1) (acc + twice scale n)
  in
  sum

let result = (sum_scaled (opaque 2)) 3 0
