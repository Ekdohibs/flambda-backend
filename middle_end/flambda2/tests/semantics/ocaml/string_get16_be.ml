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

(* Two bytes of a string read as one integer, the first being the more
   significant: [String.get_uint16_be] on a little-endian machine, which swaps
   the two bytes that it loads.

   What matches the library is in stdlib/bytes.ml of this repository: the
   function of [String] calls the one of [Bytes] (stdlib/string.ml, line 284).
   The declaration of [bswap16] matches line 482 of stdlib/bytes.ml in its type
   and its primitive, and is not that line: the name there is [swap16], and the
   line has [@@ portable] after the type. The body of [get_uint16_be] matches,
   of the function of that name at lines 503 to 505 of that file, the arm that a
   little-endian machine takes, [swap16 (get_uint16_ne b i)], with the two names
   of this unit, and with a load that makes no test of the index where the
   library's makes one: [get16] declares the primitive ["%caml_string_get16u"],
   which no file of stdlib/ declared when this comment was written. [opaque]
   declares the primitive ["%opaque"] under a name and a type of this unit.
   [result] is not of the library. The line numbers are those of the two files
   when this comment was written.

   The notice at the head of this file is lines 1 to 14 of stdlib/bytes.ml, as
   that file has them. *)

external get16 : string -> int -> int = "%caml_string_get16u"

external bswap16 : int -> int = "%bswap16"

external opaque : 'a -> 'a = "%opaque"

let[@inline] get_uint16_be s i = bswap16 (get16 s i)

let result = get_uint16_be (opaque "hello") 1
