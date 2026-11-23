(* A control: a unit that calls a C function as it is initialised, as a unit of
   the library does that defines an exception or registers a value. Its run
   reaches the call and is undefined there, in the term before and in the term
   after. The tool says so and names the call; whatever the validation says, it
   says nothing of this unit. *)

external fresh_id : unit -> int = "c_fresh_id"

let id = fresh_id ()
