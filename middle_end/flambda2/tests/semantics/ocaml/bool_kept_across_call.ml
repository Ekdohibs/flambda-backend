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

(* A boolean that a call of an unknown function returns is kept across another
   call and tested after it. Simplify untags the boolean where the first call
   returns (it unboxes the parameter of the continuation that the call returns
   to), and the test after the second call reads the untagged number; the term
   before untags at the test.

   The function [count] is a reduction of the functions [filter] and [partition]
   of stdlib/map.ml (lines 432 to 440 and 454 to 463) and of stdlib/set.ml
   (lines 398 to 407 and 409 to 418) of this repository, each of which has them
   a second time further down, and keeps their form: the result of a call of an
   unknown function is bound to a name, a recursive call follows, and the name
   is tested after it. It was made when the validation of those functions was
   looked into, with their text at hand. Those functions go down a tree and
   build one; [count] goes down a list and counts: its name, its arms and its
   result are not the library's, and no line of those functions is here. The
   operator [+] is declared as many units of this directory declare it; this
   comment names no file of the library for it. The line numbers are those of
   the two files when this comment was written.

   The notice at the head of this file is lines 1 to 14 of stdlib/map.ml, as
   that file has them; lines 1 to 14 of stdlib/set.ml are the same bytes. *)

external ( + ) : int -> int -> int = "%addint"

let rec count p l =
  match l with
  | [] -> 0
  | x :: r ->
    let b = p x in
    let n = count p r in
    if b then n + 1 else n
