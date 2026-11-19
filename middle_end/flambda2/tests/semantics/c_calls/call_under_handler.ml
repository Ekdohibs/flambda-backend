(**************************************************************************)
(*                                                                        *)
(*                                 OCaml                                  *)
(*                                                                        *)
(*           Xavier Leroy and Pascal Cuoq, INRIA Rocquencourt             *)
(*                                                                        *)
(*   Copyright 1995 Institut National de Recherche en Informatique et     *)
(*     en Automatique.                                                    *)
(*                                                                        *)
(*   All rights reserved.  This file is distributed under the terms of    *)
(*   the GNU Lesser General Public License version 2.1, with the          *)
(*   special exception on linking described in the file LICENSE.          *)
(*                                                                        *)
(**************************************************************************)

(* A unit of the shape of the library's Mutex: a function that calls a C
   function, then a function that it is given, and then a second C function
   whether the function returned or raised. The C functions are called under,
   and in, an exception handler. The run of the unit calls nothing.

   Where the items are from, in stdlib/ of this repository as it was when the
   unit was written: all five are adapted from stdlib/mutex.ml, and none is a
   whole declaration or function of it as it stands: [t] from its line 16,
   [lock] and [unlock] from lines 18 and 20, [reraise] from line 23 and
   [protect] from lines 28 to 37, with the kind and the mode annotations, the
   attribute [@inline never] and a comment taken out, the names of the C
   functions changed to names that begin with c_, and [reraise e] where the file
   has a match on it whose one case is a refutation. Five lines of the body of
   [protect] are lines of the file as they stand: its lines 29, 30, 31, 33 and
   35 ([lock m;], [match f () with], [| x ->], [| exception e ->] and [unlock
   m;], which is here twice).

   The notice at the head of this file is that of stdlib/mutex.ml, as that file
   has it. *)

type t

external lock : t -> unit = "c_lock"

external unlock : t -> unit = "c_unlock"

external reraise : exn -> 'a = "%reraise"

let protect m f =
  lock m;
  match f () with
  | x ->
    unlock m;
    x
  | exception e ->
    unlock m;
    reraise e
