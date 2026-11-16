(* A function whose name is an operator with a slash in it, and a function that
   takes such an operator as a parameter. The text writes a variable as its
   name, a slash and its stamp, and the reader takes the stamp to be what
   follows the last slash; it stopped on a name with a slash of its own. So the
   text is the term. *)

external ( - ) : int -> int -> int = "%subint"

let ( / ) a b = a - b

let f ( /. ) x = x /. 2
