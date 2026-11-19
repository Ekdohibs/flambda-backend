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

(* An identity that only the middle end does not see through ([Obj.magic]). The
   text writes it [%opaque.middle_end_only]; it wrote [%opaque], which is read
   as the identity that nothing sees through ([Sys.opaque_identity]). So the
   text is the term.

   The declaration of [magic] matches line 31 of stdlib/obj.mli of this
   repository, byte for byte. The function [f] is not of the library. The line
   number is that of the file when this comment was written.

   The notice at the head of this file is lines 1 to 14 of stdlib/obj.mli, as
   that file has them. *)

external magic : 'a -> 'b = "%obj_magic"

let f (x : int) : string = magic x
