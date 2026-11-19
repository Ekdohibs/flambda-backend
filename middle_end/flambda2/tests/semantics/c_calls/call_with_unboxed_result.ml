(* A unit of the shape of the library's Int32: a C function whose result is not
   a value (an unboxed 32-bit integer), called under a handler, and a C function
   that is given a boxed one. The conversion takes a call of a C function of
   every arity. The run of the unit calls neither. *)

external of_string : string -> (int32[@unboxed])
  = "c_int32_of_string" "c_int32_of_string_unboxed"

external format : string -> int32 -> string = "c_int32_format"

let to_string n = format "%d" n

let of_string_opt s = try Some (of_string s) with Failure _ -> None
