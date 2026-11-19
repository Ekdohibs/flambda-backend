(**************************************************************************)
(*                                                                        *)
(*                                 OCaml                                  *)
(*                                                                        *)
(*          Xavier Leroy and Damien Doligez, INRIA Rocquencourt           *)
(*                                                                        *)
(*   Copyright 1996 Institut National de Recherche en Informatique et     *)
(*     en Automatique.                                                    *)
(*                                                                        *)
(*   All rights reserved.  This file is distributed under the terms of    *)
(*   the GNU Lesser General Public License version 2.1, with the          *)
(*   special exception on linking described in the file LICENSE.          *)
(*                                                                        *)
(**************************************************************************)

(* The interface of exported_externals.ml. It says [val] of the three C
   functions, as condition.mli and dynamic.mli say of theirs: so the compiler
   makes a function for each, whose body is the call of the C function.

   Where the items are from, in stdlib/ of this repository as it was when the
   unit was written: [val create] is line 149 of stdlib/condition.mli as it
   stands. [t], [val wait] and [val signal] are adapted from its lines 146, 155
   and 167, with the kind and the mode annotations taken out and [int] where
   [wait] has a mutex. [val wait_and_signal] is in no file of the library.
   Nothing of stdlib/dynamic.mli is here: it is named for the shape alone.

   The notice at the head of this file is that of stdlib/condition.mli, as that
   file has it. *)

type t

val create : unit -> t

val wait : t -> int -> unit

val signal : t -> unit

val wait_and_signal : t -> int -> unit
