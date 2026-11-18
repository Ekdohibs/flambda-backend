# Generated files of modules of Rocq's library

The `.ml` and `.mli` files of this directory are generated. They are what
Rocq's extraction writes for the modules of Rocq's own library that the
machine of the Flambda 2 semantics uses. The generated files of the machine
itself are in `../extracted/`. Like those, they are for tests only: no part
of the compiler is built from them.

None of them is edited by hand, and none has a header: the extraction
writes none. `LICENSE` is a copy (see below). This file and `dune` are
written by hand.

## What the files are generated from

Twelve modules of Rocq's library, as the opam packages `rocq-core` 9.3.0
and `rocq-stdlib` 9.2.0 install them. The two files of a module have its
name. The source file is given under the directory `lib/coq` of the opam
switch.

| Module | Rocq's name for it | Source file | Package |
|---|---|---|---|
| `BinInt` | `Stdlib.ZArith.BinInt` | `user-contrib/Stdlib/ZArith/BinInt.v` | `rocq-stdlib` |
| `BinNums` | `Corelib.Numbers.BinNums` | `theories/Numbers/BinNums.v` | `rocq-core` |
| `BinPos` | `Stdlib.PArith.BinPos` | `user-contrib/Stdlib/PArith/BinPos.v` | `rocq-stdlib` |
| `Bool` | `Stdlib.Bool.Bool` | `user-contrib/Stdlib/Bool/Bool.v` | `rocq-stdlib` |
| `Datatypes` | `Corelib.Init.Datatypes` | `theories/Init/Datatypes.v` | `rocq-core` |
| `List` | `Stdlib.Lists.List` | `user-contrib/Stdlib/Lists/List.v` | `rocq-stdlib` |
| `ListDef` | `Corelib.Lists.ListDef` | `theories/Lists/ListDef.v` | `rocq-core` |
| `Nat` | `Corelib.Init.Nat` | `theories/Init/Nat.v` | `rocq-core` |
| `NatDef` | `Corelib.BinNums.NatDef` | `theories/BinNums/NatDef.v` | `rocq-core` |
| `PeanoNat` | `Stdlib.Arith.PeanoNat` | `user-contrib/Stdlib/Arith/PeanoNat.v` | `rocq-stdlib` |
| `PosDef` | `Corelib.BinNums.PosDef` | `theories/BinNums/PosDef.v` | `rocq-core` |
| `Specif` | `Corelib.Init.Specif` | `theories/Init/Specif.v` | `rocq-core` |

The names are those that Rocq prints for `Locate Module`. For `BinNums`,
`ListDef`, `Nat`, `NatDef` and `PosDef` Rocq has a module of that name
under `Stdlib` as well; for one definition of each, `Locate` gives the
module under `Corelib`, which is the one in the table. The files of
`Specif` are empty.

Nothing in this tree names these modules for the extraction. The one
command `Separate Extraction` of
`middle_end/flambda2/rocq/validation/checks/FlambdaExtract.v` names
definitions of this tree's own `.v` files, and Rocq writes a file for each
module that those definitions use. So a change to the model can bring a
module here, take one away, or change what a file has. The table is then
changed in the same commit, with the entry of
`middle_end/flambda2/rocq/EXTERNAL-SOURCES.md` for these files.

## Licence and copyright

Rocq's library is distributed under the GNU Lesser General Public License,
version 2.1; the opam files of both packages give `LGPL-2.1-only`.

`LICENSE` in this directory is a copy, byte for byte, of the file that
`rocq-core` 9.3.0 installs as `doc/rocq-core/LICENSE` in the opam switch
(24381 bytes, sha256
`d4594b82f4d50840df6a7e9d14132a8c0a3cc05d0ac46d15310a264a1f75447e`).
The file `LICENSE` of the source of `rocq-stdlib` 9.2.0 has the same
bytes.

Each of the twelve source files has this header, which is Rocq's statement
of the copyright and of the licence, copied here from
`theories/Init/Datatypes.v`:

    (************************************************************************)
    (*         *      The Rocq Prover / The Rocq Development Team           *)
    (*  v      *         Copyright INRIA, CNRS and contributors             *)
    (* <O___,, * (see version control and CREDITS file for authors & dates) *)
    (*   \VV/  **************************************************************)
    (*    //   *    This file is distributed under the terms of the         *)
    (*         *     GNU Lesser General Public License Version 2.1          *)
    (*         *     (see LICENSE file for the text of the license)         *)
    (************************************************************************)

In `BinInt.v`, `BinPos.v` and `PosDef.v` it comes after one line,
`(* -*- coding: utf-8 -*- *)`. In `PeanoNat.v` it is followed by one more
line in the same frame, `Evgeny Makarov, INRIA, 2007`. `BinInt.v` has,
further down, the line
`(** Initial author: Pierre Crégut, CNET, Lannion, France *)`. Lines that
name an author or a year were looked for in the first 25 lines of each
file. Beyond those lines one search was made, of the whole of `BinInt.v`
for the words `Initial author`, and no other.

For authors and dates the header points to Rocq's version control and to
its file `CREDITS`. That file was found in the source of `rocq-core`, and
not by a search for its name under the installed `doc/`, `lib/rocq-core/`,
`lib/rocq-stdlib/` and `lib/coq/` (to depth 2); it is not copied here.

## How the files are made again

`make -C rocq all` compiles `FlambdaExtract.v`, which makes Rocq write the
`.ml` and `.mli` files into `../extracted/`. The same rule of
`rocq/Makefile` runs `ocamlformat` on them and then moves here every file
whose module is not a `.v` file of this tree. The rule has no list of
names.

To check that the generated files are what the rule writes, delete the
`.ml` and `.mli` files of this directory and of `../extracted/`, delete
`middle_end/flambda2/rocq/validation/checks/FlambdaExtract.vo`, and run
`make -C rocq all`. After the build succeeds, check that `git status`
shows no change in either directory.
`rocq/HACKING.md` says more under "An extracted file that is no longer
written".
