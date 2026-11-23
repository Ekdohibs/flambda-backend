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

(* A unit of the shape of the library's Int and Bool: two functions that the
   unit exports, each of which calls a C function. One of the two C functions is
   said not to allocate ([@@noalloc]), which the call says and the conversion
   keeps. The run of the unit calls neither.

   Where the items are from, in stdlib/ of this repository as it was when the
   unit was written: [to_string] is line 56 of stdlib/int.ml as it stands. The
   other three are adapted from that file: [format_int] from its line 55,
   [hash_param] from [seeded_hash_param] (lines 61 and 62; stdlib/bool.ml has
   the same for [bool], lines 45 and 46) and [hash] from line 64 (line 49 of
   stdlib/bool.ml), with the mode annotations taken out, the names of the C
   functions changed to names that begin with c_, the name of
   [seeded_hash_param] shortened, ['a] for the type of what is hashed, and the
   type of its argument written in [hash] ([(x : int)], where the file has [x]).

   The notice at the head of this file is that of stdlib/int.ml, as that file
   has it; stdlib/bool.ml has the same fourteen lines. *)

external format_int : string -> int -> string = "c_format_int"

external hash_param : int -> int -> int -> 'a -> int = "c_hash" [@@noalloc]

let to_string x = format_int "%d" x

let hash (x : int) = hash_param 10 100 0 x
