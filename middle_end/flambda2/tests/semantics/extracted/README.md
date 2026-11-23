# Generated files of the machine of the Flambda 2 semantics

The `.ml` and `.mli` files of this directory are generated. They are what
Rocq's extraction writes for `.v` files of this tree: the machine of the
Flambda 2 semantics, the checks and the rewrites that the tests run with
it, and what those use. The one command `Separate Extraction` of
`middle_end/flambda2/rocq/validation/checks/FlambdaExtract.v` names what
is extracted, and `make -C rocq all` writes the files again. They are for
tests only: no part of the compiler is built from them. The generated
files of the modules of Rocq's own library that they use are in
`../extracted_rocq_library/`, with a `README.md` of their own.

None of the generated files is edited by hand, and none has a header: the
extraction writes none. This file and `dune` are written by hand.

## What a file is generated from

The two files of a module, `M.ml` and `M.mli`, are generated from the
`.v` file of the same name, `M.v`. For every module but two, that file is
in a folder of `middle_end/flambda2/rocq/`:

    find middle_end/flambda2/rocq -name M.v

`OxCamlList` and `OxCamlOption` are generated from `rocq/OxCamlList.v`
and `rocq/OxCamlOption.v`.

## Licence and copyright

A generated file has no notice of its own. It stands under the notice at
the head of the `.v` file that it is generated from:

- For most modules that is the licence header of the `.v` files of this
  tree, the MIT licence (the first 26 lines of `rocq/OxCamlLogic.v`).
- Some `.v` files begin instead with the header of a file of the
  compiler. The comment under that header says which file of the compiler
  it is and why the `.v` file has its header: what the `.v` file takes
  from that file, or how one part of it stands to it. Where it names
  other files of the compiler, it gives the lines of authors and of
  copyright of each whose header differs. The files of the compiler are
  under the GNU Lesser General Public License version 2.1, with the
  special exception on linking described in the file `LICENSE` at the
  root of the tree. A module of this directory whose `.v` file is one of
  those stands under the notice at the head of that `.v` file, as the
  `.v` file does. The list of those `.v` files is the variable
  `FLAMBDA2_TRANSCRIBED` of `rocq/Makefile`. Its recipe `layout` checks
  that a `.v` file of the list does not begin with the licence header of
  the others, and that every other `.v` file does; it does not compare a
  header with the file of the compiler.

So, to know the notice of `M.ml` or `M.mli`, read the head of `M.v`.
