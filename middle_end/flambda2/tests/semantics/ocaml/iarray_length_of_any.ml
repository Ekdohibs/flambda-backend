(* The length of an immutable array of any elements: the primitive that
   the front end makes of it carries no kind of array
   ([Array_length Float_array_opt_dynamic]). The function is exported and
   Simplify inlines its one call. *)

external length : 'a iarray -> int = "%array_length"

let len (a : 'a iarray) : int = length a

let n : int = len [: 1; 2; 3 :]
