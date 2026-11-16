(* The seven operations of effects. Each is a call of a kind of its own, with no
   callee and no arguments: what the operation is given is in the kind of the
   call. The text writes it ([effect(perform x)] and the like); the printer
   stopped on such a call. So the text is the term. The types of the externals
   are not those of the library: the term has the operations all the same. *)

type _ eff += E : int eff

external perform : 'a eff -> 'a = "%perform"

external reperform : 'a -> 'b -> 'c -> 'd = "%reperform"

external with_stack : 'a -> 'b -> 'c -> 'd -> 'e -> 'f = "%with_stack"

external with_stack_preemptible : 'a -> 'b -> 'c -> 'd -> 'e -> 'f -> 'g
  = "%with_stack_preemptible"

external continue : 'a -> 'b -> 'c = "%continue"

external discontinue : 'a -> 'b -> 'c = "%discontinue"

external discontinue_with_backtrace : 'a -> 'b -> 'c -> 'd
  = "%discontinue_with_backtrace"

let f1 () = perform E

let f2 a b c = reperform a b c

let f3 a b c d e = with_stack a b c d e

let f4 a b c d e f = with_stack_preemptible a b c d e f

let f5 a b = continue a b

let f6 a b = discontinue a b

let f7 a b c = discontinue_with_backtrace a b c
