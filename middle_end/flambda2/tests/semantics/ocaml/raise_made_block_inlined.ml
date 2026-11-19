(**************************************************************************)
(*                                                                        *)
(*                                 OCaml                                  *)
(*                                                                        *)
(*                         The OCaml programmers                          *)
(*                                                                        *)
(*   Copyright 2018 Institut National de Recherche en Informatique et     *)
(*     en Automatique.                                                    *)
(*                                                                        *)
(*   All rights reserved.  This file is distributed under the terms of    *)
(*   the GNU Lesser General Public License version 2.1, with the          *)
(*   special exception on linking described in the file LICENSE.          *)
(*                                                                        *)
(**************************************************************************)

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

(* A function that raises an exception made of its argument, inlined where the
   argument is a constant string. Simplify inlines the call, makes the block of
   the exception a symbol, which the code of [get] raises, and deletes the code
   of [fail], which nothing uses any more: the interface keeps [fail] out of the
   module block. The term before has that code, which raises a variable that
   holds a block made of its parameter.

   T5b wrote this source on 16 November 2026, to show with no use of the library
   a shape that an account of V3's, which is not in this tree, had found in
   functions of the standard library of this repository. That account names
   them: [Option.get] (stdlib/option.ml, line 25) and [invalid_arg]
   (stdlib/stdlib.ml, line 39), which [Option.get] calls. The shape is the
   library's: [fail] is a function of a string that raises [Invalid_argument] of
   it, as [invalid_arg] is, and [get] gives the argument of [Some] and, on
   [None], calls that function with a constant string, as [Option.get] does. The
   name [get] is the library's. What differs from those two definitions: the
   function of the string is named [fail], and has its parameter [reason] for
   [s], the types of its parameter and of its result written, and the attribute
   [[@inline]]; [get] has [match] on a parameter [o] whose type is written for
   [function], [x] for [v], the type of its result written, and the string
   ["get"] for the library's; and [raise] is declared here, as a primitive.
   T5b's account of 16 November 2026 says that it wrote the source from the
   shapes that V3's account describes in words, that it copied no source text
   from that account, and that it did not open the source of [invalid_arg] for
   the commit. T5b said on 19 November 2026 that it has no record of whether it
   opened the sources that V3 had cut down from the library for its account. The
   line numbers are those of the two files when this comment was written.

   The notices at the head of this file are lines 1 to 14 of stdlib/option.ml
   and lines 1 to 14 of stdlib/stdlib.ml, as those files have them. *)

external raise : exn -> 'a = "%raise"

let[@inline] fail (reason : string) : int = raise (Invalid_argument reason)

let get (o : int option) : int = match o with Some x -> x | None -> fail "get"
