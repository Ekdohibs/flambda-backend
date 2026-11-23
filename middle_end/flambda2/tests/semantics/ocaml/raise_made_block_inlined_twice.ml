(**************************************************************************)
(*                                                                        *)
(*                                 OCaml                                  *)
(*                                                                        *)
(*         Gabriel Scherer, projet Parsifal, INRIA Saclay                 *)
(*                                                                        *)
(*   Copyright 2019 Institut National de Recherche en Informatique et     *)
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

(* As raise_made_block_inlined.ml, with two functions that each give the
   function that raises a constant string of their own: Simplify inlines both
   calls and makes two blocks of exceptions symbols, one raised by the code of
   each function.

   T5b wrote this source on 16 November 2026 with raise_made_block_inlined.ml,
   for the same end and after the same account of V3's, which names [get_left]
   and [get_right] of [Either] (stdlib/either.ml, lines 37 to 43, on the type of
   line 19) and [invalid_arg] (stdlib/stdlib.ml, line 39), which they call. The
   shape is the library's: a type of two constructors, and two functions, each
   of which gives the argument of one constructor and, on the other, calls with
   a constant string the function that raises [Invalid_argument] of a string.
   The name [t] of the type is the library's. What differs from those
   definitions: the type has no parameter, and its constructors [A] and [B]
   carry an integer each where [Left] and [Right] carry the two parameters of
   the library's type; the two functions are named [get_a] and [get_b], and each
   has [match] on a parameter [v] whose type is written for [function], [x] for
   the library's [v], the other constructor by its name for [_], the type of its
   result written, and a string of its own for the library's; [fail] is that of
   raise_made_block_inlined.ml, whose comment says how it differs from
   [invalid_arg]; and [raise] is declared here, as a primitive. What the comment
   of raise_made_block_inlined.ml says of T5b's account of 16 November 2026, and
   of what T5b said on 19 November 2026, holds of this unit too. The line
   numbers are those of the two files when this comment was written.

   The notices at the head of this file are lines 1 to 14 of stdlib/either.ml
   and lines 1 to 14 of stdlib/stdlib.ml, as those files have them. *)

external raise : exn -> 'a = "%raise"

type t =
  | A of int
  | B of int

let[@inline] fail (reason : string) : int = raise (Invalid_argument reason)

let get_a (v : t) : int = match v with A x -> x | B _ -> fail "get_a"

let get_b (v : t) : int = match v with B x -> x | A _ -> fail "get_b"
