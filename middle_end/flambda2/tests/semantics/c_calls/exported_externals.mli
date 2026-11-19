(* The interface of exported_externals.ml. It says [val] of the three C
   functions, as condition.mli and dynamic.mli say of theirs: so the compiler
   makes a function for each, whose body is the call of the C function. *)

type t

val create : unit -> t

val wait : t -> int -> unit

val signal : t -> unit

val wait_and_signal : t -> int -> unit
