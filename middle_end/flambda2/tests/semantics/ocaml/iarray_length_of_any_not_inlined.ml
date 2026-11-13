(* As iarray_length_of_any.ml, with the function not inlined, and a
   second array whose elements are not constants. *)

external length : 'a iarray -> int = "%array_length"

let[@inline never] [@local never] len (a : 'a iarray) : int = length a

let n : int = len [: 1; 2; 3 :]

let m : int = len [: n; n :]
