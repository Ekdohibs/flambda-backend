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

(* [fst] and [snd], declared with the primitives that the standard library
   declares them with. Simplify inlines [sum] and reads the two fields of the
   pair, which is then not built.

   The declarations of [fst] and [snd] match lines 245 and 246 of
   stdlib/stdlib.ml of this repository in their names and their primitives, and
   are not those lines: the types here are ['a * 'b -> 'a] and ['a * 'b -> 'b],
   where each of the two lines has the mark [[@local_opt]] on the argument and
   on the result, and [@@ portable] after the type. [( + )] is declared as many
   units of this directory declare it; this comment names no file of the library
   for it. [opaque] declares the primitive ["%opaque"] under a name and a type
   of this unit. [sum] and [result] are not of the library. The line numbers are
   those of the file when this comment was written.

   The notice at the head of this file is lines 1 to 14 of stdlib/stdlib.ml, as
   that file has them. *)

external fst : 'a * 'b -> 'a = "%field0_immut"

external snd : 'a * 'b -> 'b = "%field1_immut"

external ( + ) : int -> int -> int = "%addint"

external opaque : 'a -> 'a = "%opaque"

let[@inline] sum pair = fst pair + snd pair

let result = sum (opaque 1, 2)
