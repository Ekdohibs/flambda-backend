(* A unit of the shape of the library's Int and Bool: two functions that the
   unit exports, each of which calls a C function. One of the two C functions is
   said not to allocate ([@@noalloc]), which the call says and the conversion
   keeps. The run of the unit calls neither. *)

external format_int : string -> int -> string = "c_format_int"

external hash_param : int -> int -> int -> 'a -> int = "c_hash" [@@noalloc]

let to_string x = format_int "%d" x

let hash (x : int) = hash_param 10 100 0 x
