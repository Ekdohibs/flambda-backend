(**************************************************************************)
(*                                                                        *)
(*                                 OCaml                                  *)
(*                                                                        *)
(*   Xavier Leroy and Pascal Cuoq, projet Cristal, INRIA Rocquencourt     *)
(*                                                                        *)
(*   Copyright 1995 Institut National de Recherche en Informatique et     *)
(*     en Automatique.                                                    *)
(*                                                                        *)
(*   All rights reserved.  This file is distributed under the terms of    *)
(*   the GNU Lesser General Public License version 2.1, with the          *)
(*   special exception on linking described in the file LICENSE.          *)
(*                                                                        *)
(**************************************************************************)

(* A unit of the shape of the library's Condition and Dynamic: C functions that
   the unit declares, and exports through an interface that says [val] of them
   (exported_externals.mli), so that the compiler makes a function for each
   whose body is the call; and one function that calls two of them one after the
   other. The run of the unit calls none of them: it returns, and the verdict is
   of that run. The calls are compared as terms (see dune, and "Calls of C
   functions" in ../../tools/flsemantics.ml).

   Where the items are from, in stdlib/ of this repository as it was when the
   unit was written: [t] and the three externals are adapted from
   stdlib/condition.ml, and none is a line of it as it stands: [t] from its line
   18, [create] from line 19, [wait] from lines 20 and 21 and [signal] from line
   22, with the kind and the mode annotations taken out, the names of the C
   functions changed to names that begin with c_, and [int] where [wait] has a
   mutex. [wait_and_signal] is in no file of the library. Nothing of
   stdlib/dynamic.ml is here: it is named for the shape alone.

   The notice at the head of this file is that of stdlib/condition.ml, as that
   file has it. *)

type t

external create : unit -> t = "c_create"

external wait : t -> int -> unit = "c_wait"

external signal : t -> unit = "c_signal"

let wait_and_signal c n =
  wait c n;
  signal c
