# The Rocq development of Flambda 2

The models of Flambda 2 and their proofs: the terms and their semantics,
the types library and what Simplify does with it, the rewrites that the
validation applies, and the validation itself. `rocq/HACKING.md` ("The
semantics of Flambda 2") says how the work is done and how to build it,
`PARALLEL-PLAN.md` here how it is shared out, and `OWNERS.md` who owns
which file.

## The tree

The `.v` files are in folders by subject. The order is an order to read
in.

```
terms/            the terms as data: expressions, kinds, primitives; maps
                  keyed by names; binders
semantics/        the machine, values, primitives, calls, units and exits;
                  refinement and how to prove it
wellformed/       what a unit must be: well-formed, well-kinded, in scope,
                  in range; traps
heap/             the heap: its primitives, the prototype machine, array
                  values
regions/          regions and local allocation
  machine/        the machine of regions, its invariant, calls
  erasure/        relations between runs with regions and runs after erasure
  check/          the check of the regions of a term, and that it is sound
  ends/           the second verdict of the check, about the ends of regions
nominal/          names: renaming, alpha-equivalence, freshness, free names;
                  the comparison of terms and the renaming apart that the
                  validation runs
types/            the model of the types library
  grammar/        the types: grammar, meaning, well-formedness, equality,
                  renaming
  env/            typing environments: meaning, invariant, extensions,
                  binding times, renaming
  aliases/        the aliases of a typing environment; following them
                  through a term, knowing what symbols are bound to
  meet/           the meet of two types (MeetEnv and ExpandHead are here)
  join/           the n-way join of environments and of types (JoinEnv is
                  here)
  provers/        the provers and reification
  naive/          the first, naive model of types; following aliases through
                  a term
simplify/         the model of Simplify: a term rewritten from the types
rewrites/         the rewrites that the validation applies, a folder a
                  family
  common/         the first rewrites (inlining calls, folding constants);
                  what the others share
  calls/          inlining and calls
  closures/       closures, code and value slots
  statics/        static bindings and symbols
  lifting/        a Let made a static binding of a symbol
  continuations/  continuations and jumps (LiftConts is here); exception
                  handlers
  switches/       switches: merged arms, the number of an arm, the scrutinee
  params/         parameters of continuations: aliases, unused, dead,
                  constant; returns
  loops/          Loopify, and the rewrites inside recursive continuations
  unboxing/       unboxing and untagging: numbers, parameters, variants
  lets/           Lets: a primitive not computed twice, identities,
                  comparisons, known reads
validation/       the validation of what Simplify did to a unit
  checks/         the checks, that they are sound, the extraction
  names/          the validation commutes with renaming and keeps scope
```

Beside the folders: `status/` (the status of each stream of the work),
`requests/` (the notes of changes to the core), and at the top
`NAMES.md` (how the model treats bound names), `DECISIONS.md` (the index
of the decisions that the files cite), `SEMANTICS-DECISIONS.md` (what
the model says a program means, wherever that was a choice, in short),
`OWNERS.md`, `PARALLEL-PLAN.md`, `EXTERNAL-SOURCES.md` (the sources
outside this repository that were read for the work, kept current:
whoever reads one writes its entry) and `_CoqProject` (for an editor:
see below).

- **A model stays together.** `FlambdaFooSpec.v` and `FlambdaFooProofs.v`
  are side by side, with the Proofs files of lemmas that serve them, and
  examples are beside what they are examples of. A reviewer reads the Spec
  files: `make -C rocq review` typesets them and leaves the Proofs files
  out.
- **`nominal/` is the theory of names**, with the two things written in
  it that the validation runs: the comparison of terms and the renaming
  apart. What says that one rewrite commutes with renaming or keeps scope
  (`...NamesProofs.v`, `...Scope...`, `...Rename...`) is beside that
  rewrite. The files that collect those laws for the validation are
  `validation/names/`.
- **Where the compiler has a directory for the subject, the name is the
  compiler's**: `nominal/`, `terms/`, `simplify/`, `types/` with its
  `env/` and `grammar/`. One difference: the compiler has `meet_env` and
  `join_env` in `types/env/`; here `FlambdaMeetEnv*` and `FlambdaJoinEnv*`
  are with the meet and the join.
- **Two names that invite a mistake.** `FlambdaLiftConts*` moves
  continuations out of a handler and is in `rewrites/continuations/`;
  `rewrites/lifting/` is the lifting of a `Let` to a symbol.
  `types/naive/` is the first, simple model of types, with the following
  of aliases that the validation runs in its rounds (`FlambdaAliases*`).
  The following of aliases that knows what symbols are bound to
  (`FlambdaEnvAliasesFollow*`), which the validation runs when it is asked
  again (`validation/checks/FlambdaValidateKnowing*`), is in
  `types/aliases/`.
- **The folders are not layers of the build.** A file may require a file
  of any folder: the machine of `semantics/` requires
  `heap/FlambdaHeapSpec.v`.

## The folders are not libraries

The name of a file is its name to Rocq: `FlambdaCseSpec.v` is
`OxCamlFlambda2.FlambdaCseSpec` in any folder of that library.
`rocq/Makefile` gives each folder to Rocq, with `-Q`, under the library
of its files: `OxCamlFlambda2Types` for the folders of `types/` and for
`simplify/`, `OxCamlFlambda2` for the others. So a file can move between
the folders of its library with no change to a `Require` sentence, to
the name of a theorem, or to what is extracted. Three rules follow, and
`make -C rocq all` stops, naming the file or the folder, where one is
broken:

- a folder is of one library;
- the folders given to Rocq are not nested, and each `.v` file is
  directly in one of them, so none is at the top (Rocq binds the folders
  under a folder that it is given, so a file of an inner folder would
  have two names);
- the names of the files are unique across the folders.

## Adding or moving a file

- **A new file** goes in the folder of its subject, beside the other file
  of its model. Its line in `FLAMBDA2` or `FLAMBDA2_TYPES` of
  `rocq/Makefile` gives it with its folder (`rewrites/lets/FlambdaFoo.v`),
  at the place of the name of the file in alphabetical order ("Adding a
  module" in `rocq/HACKING.md` says who commits that line). A `.v` file
  under this directory that is in neither list stops `make -C rocq all`,
  whether Git tracks it or not.
- **A file moved to another folder of its library**: `git mv`, and the
  folder in its line of the Makefile. Its compiled files at the old place
  must go, or Rocq stops at "Required library ... matches several files
  in path": `make -C rocq clean` removes the compiled files of every
  folder.
- **A new folder** is added to `FLAMBDA2_FOLDERS` or
  `FLAMBDA2_TYPES_FOLDERS` in `rocq/Makefile`, to `_CoqProject` here, and
  to the tree above.

## Stepping through a file in an editor

`_CoqProject` here gives an editor the flags that `rocq/Makefile` gives
to Rocq (`make -C rocq all` stops if the two differ). Build first
(`make -C rocq all` at the root of the repository): an editor needs the
compiled files of what a file requires.

## Before the folders

The files of `types/` and `simplify/` were in
`middle_end/flambda2/types/rocq/`, and the others were side by side at
the top of this directory. A status or a request written then may still
say "in `middle_end/flambda2/types/rocq/`"; no file has changed its name,
so a file named there is found by its name.
