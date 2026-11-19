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

(* A control of raise_made_block_inlined.ml: the same exception, made of the
   same constant string, where it is raised, with no function to inline. The
   front end makes the block of the exception a symbol, and the code of [get]
   raises that symbol before Simplify and after.

   T5b wrote this source on 16 November 2026 as a control of
   raise_made_block_inlined.ml, which it wrote that day after an account of
   V3's, not in this tree, of functions of the standard library of this
   repository: [Option.get] (stdlib/option.ml, line 25) and [invalid_arg]
   (stdlib/stdlib.ml, line 39), which [Option.get] calls. [get] here is the
   [get] of that unit with the body of its [fail] in the place of the call: it
   gives the argument of [Some] and, on [None], raises [Invalid_argument] of a
   constant string, which is what [Option.get] does once the body of
   [invalid_arg] is put in the place of its call. The name [get] is the
   library's. What differs from those two definitions: [match] on a parameter
   [o] whose type is written for [function], [x] for [v], the type of the result
   written, the string ["get"] for the library's, the raise where the library
   has a call, and [raise] declared here, as a primitive. What the comment of
   raise_made_block_inlined.ml says of T5b's account of 16 November 2026, and of
   what T5b said on 19 November 2026, holds of this unit too. The line numbers
   are those of the two files when this comment was written.

   The notices at the head of this file are lines 1 to 14 of stdlib/option.ml
   and lines 1 to 14 of stdlib/stdlib.ml, as those files have them. *)

external raise : exn -> 'a = "%raise"

let get (o : int option) : int =
  match o with Some x -> x | None -> raise (Invalid_argument "get")
