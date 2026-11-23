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

(* The control of bool_kept_across_call.ml: the boolean that the call returns is
   tested where it is received, and nothing is kept across the second call.

   This unit was written as a variation of bool_kept_across_call.ml of this
   directory, whose function [count] is a reduction of the functions [filter]
   and [partition] of stdlib/map.ml (lines 432 to 440 and 454 to 463) and of
   stdlib/set.ml (lines 398 to 407 and 409 to 418) of this repository. The
   comment of that unit says how it was made and what is not of the library.
   Here the form of those functions is not kept: the result of the call of the
   unknown function is tested where the call returns, and no name is bound to
   it. What stays of it is a recursive function on a list that calls an unknown
   function on each element and counts. The operator [+] is declared as many
   units of this directory declare it; this comment names no file of the library
   for it. The line numbers are those of the two files when this comment was
   written.

   The notice at the head of this file is lines 1 to 14 of stdlib/map.ml, as
   that file has them; lines 1 to 14 of stdlib/set.ml are the same bytes. *)

external ( + ) : int -> int -> int = "%addint"

let rec count p l =
  match l with [] -> 0 | x :: r -> if p x then count p r + 1 else count p r
