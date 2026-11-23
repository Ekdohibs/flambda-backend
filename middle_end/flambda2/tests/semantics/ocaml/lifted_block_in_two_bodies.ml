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

(* [with_node] makes a block of [node], a function of the unit, and a constant.
   Simplify lifts that block to a symbol where it is made, and then inlines
   [with_node] into [either]: its term names the one symbol in the bodies of the
   two functions. The validation lifts after its own inlining, when the [Let]
   that makes the block is in both bodies, and gives the two [Let]s the one
   symbol (FlambdaLiftFromCodeSpec.put_once_each).
   lifted_closure_in_two_bodies.ml is the same of a closure.

   This unit was written as a variation of lifted_closure_in_two_bodies.ml of
   this directory, which is a reduction of the functions [of_sorted_list] and
   [of_list] of stdlib/set.ml of this repository (lines 561 to 580 and 584 to
   592). The comment of that unit says how it was made and what is and what is
   not of the library; the type [tree] and the function [node] are those of that
   unit. Here the shape of those functions is not kept: there is no local
   function, and what is lifted is a block that [with_node] makes of [node] and
   a constant. What stays of it is a function that makes a constant and a second
   function that calls it in one of its arms. The line numbers are those of the
   file when this comment was written.

   The notice at the head of this file is lines 1 to 14 of stdlib/set.ml, as
   that file has them. *)

type tree =
  | Leaf
  | Node of tree * int * tree

let[@inline never] node l v r = Node (l, v, r)

let[@inline] with_node () = node, 0

let either b = if b then with_node () else node, 1
