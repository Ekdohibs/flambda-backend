(* A unit of the shape of the library's Condition and Dynamic: C functions that
   the unit declares, and exports through an interface that says [val] of them
   (exported_externals.mli), so that the compiler makes a function for each
   whose body is the call; and one function that calls two of them one after the
   other. The run of the unit calls none of them: it returns, and the verdict is
   of that run. The calls are compared as terms (see dune, and "Calls of C
   functions" in ../../tools/flsemantics.ml). *)

type t

external create : unit -> t = "c_create"

external wait : t -> int -> unit = "c_wait"

external signal : t -> unit = "c_signal"

let wait_and_signal c n =
  wait c n;
  signal c
