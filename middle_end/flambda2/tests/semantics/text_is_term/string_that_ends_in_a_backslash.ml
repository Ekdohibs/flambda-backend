(* A string that ends in a backslash, and a later string. The text writes the
   backslash as two, and the reader takes a backslash with the character after
   it; it took the second backslash with the quote that ends the string, read on
   to the quote that begins the next one, and stopped. So the text is the
   term. *)

let a = "x\\"

let b = "y"
