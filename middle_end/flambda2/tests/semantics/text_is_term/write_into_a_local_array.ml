(* A write into an array of values that may be on the stack: the array is a
   parameter that the source marks [local_]. The write is an assignment that is
   marked local, and the text says so of it ([%array_set] with [values] and
   [value.lassign]); it wrote neither, and the write was read as an assignment
   into an array of the heap. So the text is the term. *)

external store : ('a array[@local_opt]) -> int -> 'a -> unit
  = "%array_unsafe_set"

let first (local_ a : string array) (x : string) = store a 0 x
