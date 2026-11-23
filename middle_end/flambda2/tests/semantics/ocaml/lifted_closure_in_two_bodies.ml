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

(* [of_count] makes the closure of its local function [down], which captures
   nothing but [node], a function of the unit. Simplify lifts that closure to a
   symbol where it is made, and then inlines [of_count] into [of_any]: its term
   names the one symbol in the bodies of the two functions. The validation lifts
   after its own inlining, when the [Let] that makes the closure is in both
   bodies, and gives the two [Let]s the one symbol
   (FlambdaLiftFromCodeSpec.put_once_each).

   [of_count] and [of_any] are a reduction of the functions [of_sorted_list] and
   [of_list] of stdlib/set.ml of this repository (lines 561 to 580 and 584 to
   592), which the file has a second time further down, and keep their shape: a
   function with a local recursive function that captures nothing but a function
   of the unit ([sub] there, which calls [create]; [down] here, which calls
   [node]), and a second function that calls the first in one of its arms. The
   unit was made when the validation of [of_list] was looked into, with the text
   of the two functions at hand. The constructor [Node] has the name of the
   library's, and [node] takes the parameters [l], [v] and [r] of [create] (line
   94). The type here has three plain fields where the library's has a record of
   four; the names [of_count], [down], [of_any], [node], [tree] and [Leaf], the
   bodies and the attributes are not the library's, and no line of those
   functions is here. The operators [-], [=] and [<] are declared as many units
   of this directory declare them; this comment names no file of the library for
   them. The line numbers are those of the file when this comment was written.

   The notice at the head of this file is lines 1 to 14 of stdlib/set.ml, as
   that file has them. *)

external ( - ) : int -> int -> int = "%subint"

external ( = ) : int -> int -> bool = "%equal"

external ( < ) : int -> int -> bool = "%lessthan"

type tree =
  | Leaf
  | Node of tree * int * tree

let[@inline never] node l v r = Node (l, v, r)

let[@inline] of_count n =
  let rec down n = if n = 0 then Leaf else node (down (n - 1)) n Leaf in
  down n

let of_any n = if n < 0 then Leaf else of_count n
