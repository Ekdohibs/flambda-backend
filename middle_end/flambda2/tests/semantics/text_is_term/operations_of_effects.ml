(**************************************************************************)
(*                                                                        *)
(*                                 OCaml                                  *)
(*                                                                        *)
(*      KC Sivaramakrishnan, Indian Institute of Technology, Madras       *)
(*                                                                        *)
(*   Copyright 2021 Indian Institute of Technology, Madras                *)
(*                                                                        *)
(*   All rights reserved.  This file is distributed under the terms of    *)
(*   the GNU Lesser General Public License version 2.1, with the          *)
(*   special exception on linking described in the file LICENSE.          *)
(*                                                                        *)
(**************************************************************************)

(* The seven operations of effects. Each is a call of a kind of its own, with no
   callee and no arguments: what the operation is given is in the kind of the
   call. The text writes it ([effect(perform x)] and the like); the printer
   stopped on such a call. So the text is the term. The types of the externals
   are not those of the library: the term has the operations all the same.

   The seven externals are those of stdlib/effect.ml of this repository by their
   names and their primitives, and none is a declaration of that file as it
   stands. [perform] matches its line 23 with ['a eff] for ['a t]. The six
   others have a type variable for each argument and for the result, where the
   file has its own types: [reperform] (its lines 88 and 89), [with_stack]
   (lines 91 to 97), [with_stack_preemptible] (lines 99 to 106), [continue]
   (line 82), [discontinue] (line 83) and [discontinue_with_backtrace] (lines 84
   to 86). Their order here is that of the constructors of [Call_kind.Effect.t],
   not that of the file. The constructor [E] and the functions [f1] to [f7] are
   not of the library. The line numbers are those of the file when this comment
   was written.

   The notice at the head of this file is lines 1 to 13 of stdlib/effect.ml, as
   that file has them. *)

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
