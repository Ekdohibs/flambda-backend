(**************************************************************************)
(*                                                                        *)
(*                                 OCaml                                  *)
(*                                                                        *)
(*             Xavier Leroy, projet Cristal, INRIA Rocquencourt           *)
(*                                                                        *)
(*   Copyright 1996 Institut National de Recherche en Informatique et     *)
(*     en Automatique.                                                    *)
(*                                                                        *)
(*   All rights reserved.  This file is distributed under the terms of    *)
(*   the GNU Lesser General Public License version 2.1, with the          *)
(*   special exception on linking described in the file LICENSE.          *)
(*                                                                        *)
(**************************************************************************)

(* The same as copy_heap.ml, with a parameter that may be on the stack of
   the caller: its mode is not heap in the compiler's term. The two ways
   print the same of this unit. T3's source. *)

(* The notice at the head of this file is that of stdlib/stdlib.ml (its lines
   1 to 14). The two declarations below follow those of [fst] and [snd] in
   that file (lines 245 and 246): they keep the name, the type of the
   argument with its [@local_opt], and the primitive; they leave out the
   [@local_opt] of the result and @@ portable. Neither declaration of the
   library is here unchanged. The function [copy] is not the library's. *)

external fst : ('a * 'b[@local_opt]) -> 'a = "%field0_immut"

external snd : ('a * 'b[@local_opt]) -> 'b = "%field1_immut"

let copy (local_ (pair : int * int)) = fst pair, snd pair
