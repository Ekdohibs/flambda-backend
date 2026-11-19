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

(* A control of raise_made_block_inlined.ml: the function that is inlined raises
   an exception that has no argument, so that no block is made. Simplify inlines
   the call, and the code of [get] raises the symbol of the exception, which the
   unit imports.

   T5b wrote this source on 16 November 2026 as a control of
   raise_made_block_inlined.ml, which it wrote that day after an account of
   V3's, not in this tree, of functions of the standard library of this
   repository; [Option.get] (stdlib/option.ml, line 25) is one of them. [get]
   here is the [get] of that unit with another call on [None]: it gives the
   argument of [Some], as [Option.get] does, and the name [get] is the
   library's. What differs from the definition of [Option.get]: [match] on a
   parameter [o] whose type is written for [function], [x] for [v], the type of
   the result written, and, on [None], a call of [fail] with [()] where the
   library calls [invalid_arg] with a string. [fail] is not [invalid_arg]: it
   takes no message, makes no exception of an argument, and raises [Not_found].
   [raise] is declared here, as a primitive. What the comment of
   raise_made_block_inlined.ml says of T5b's account of 16 November 2026, and of
   what T5b said on 19 November 2026, holds of this unit too. The line number is
   that of the file when this comment was written.

   The notice at the head of this file is lines 1 to 14 of stdlib/option.ml, as
   that file has them. *)

external raise : exn -> 'a = "%raise"

let[@inline] fail () : int = raise Not_found

let get (o : int option) : int = match o with Some x -> x | None -> fail ()
