(* A unit of the shape of the library's Mutex: a function that calls a C
   function, then a function that it is given, and then a second C function
   whether the function returned or raised. The C functions are called under,
   and in, an exception handler. The run of the unit calls nothing. *)

type t

external lock : t -> unit = "c_lock"

external unlock : t -> unit = "c_unlock"

external reraise : exn -> 'a = "%reraise"

let protect m f =
  lock m;
  match f () with
  | x ->
    unlock m;
    x
  | exception e ->
    unlock m;
    reraise e
