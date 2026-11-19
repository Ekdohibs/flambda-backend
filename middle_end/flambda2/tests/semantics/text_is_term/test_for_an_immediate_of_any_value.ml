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

(* The test for an immediate that works on any value ([Obj.is_int]). The text
   writes it [%is_int.generic]; it wrote [%is_int], which is read as the test
   for a value of a variant type. So the text is the term.

   The declaration of [is_int] matches line 38 of stdlib/obj.mli of this
   repository in its name and its primitive, and is not that line: the argument
   has the type ['a] here, where the line has [t @ contended]. The function [f]
   is not of the library. The line number is that of the file when this comment
   was written.

   The notice at the head of this file is lines 1 to 14 of stdlib/obj.mli, as
   that file has them. *)

external is_int : 'a -> bool = "%obj_is_int"

let f x = is_int x
