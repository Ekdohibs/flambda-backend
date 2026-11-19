(**************************************************************************)
(*                                                                        *)
(*                                 OCaml                                  *)
(*                                                                        *)
(*           Damien Doligez, projet Gallium, INRIA Rocquencourt           *)
(*                                                                        *)
(*   Copyright 2014 Institut National de Recherche en Informatique et     *)
(*     en Automatique.                                                    *)
(*                                                                        *)
(*   All rights reserved.  This file is distributed under the terms of    *)
(*   the GNU Lesser General Public License version 2.1, with the          *)
(*   special exception on linking described in the file LICENSE.          *)
(*                                                                        *)
(**************************************************************************)

(* The same load with the test of the index that [String.get_uint16_le] makes.

   The declaration of [get16] matches line 278 of stdlib/string.ml of this
   repository in its type and its primitive, and is not that line: the name
   there is [get_uint16_ne], and the line has [@@ portable] after the type.
   [opaque] declares the primitive ["%opaque"] under a name and a type of this
   unit. [result] is not of the library. The line number is that of the file
   when this comment was written.

   The notice at the head of this file is lines 1 to 14 of stdlib/string.ml, as
   that file has them. *)

external get16 : string -> int -> int = "%caml_string_get16"

external opaque : 'a -> 'a = "%opaque"

let result = get16 (opaque "hello") 3
