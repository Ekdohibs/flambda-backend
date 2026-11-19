# Calls of C functions: the generated texts and their notices

The comment at the head of `dune` says what each test of this directory
is and which rule runs it. `../ocaml/README.md` says how a source is
made a text and what a verdict means. This file says under which notice
the texts stand that the compiler generates here.

## Which texts are generated

A file `X.fl` of this directory that has a source `X.ml` beside it is
generated. A rule of `dune` makes it from `X.ml` with the compiler of
the tree, through `../ocaml/from_ocaml.sh` or, for a first unit and the
units that are compiled against it, `../ocaml/pairs/from_ocaml.sh`; a
rule of the alias `runtest` compares the stored file with what that
rule makes. So the stored file is what the compiler prints for the
source. It is not edited by hand, and it has no header: the compiler
prints none.

A file `X.fl` with no source beside it is written by hand, and begins
with a comment that says what it is.

## The texts whose source has a notice of the standard library

Four sources begin with the notice of a file of `stdlib/`, byte for
byte, because items of theirs are taken or adapted from that file; the
source's own comment, under the notice, says which. The text generated
from each of them stands under the notice and the credits that are at
the head of its source:

- `call_in_function.fl` is generated from `call_in_function.ml`, whose
  notice is that of `stdlib/int.ml` and of `stdlib/bool.ml`, which have
  the same;
- `call_under_handler.fl` is generated from `call_under_handler.ml`,
  whose notice is that of `stdlib/mutex.ml`;
- `call_with_unboxed_result.fl` is generated from
  `call_with_unboxed_result.ml`, whose notice is that of
  `stdlib/int32.ml`;
- `exported_externals.fl` is generated from `exported_externals.ml`,
  whose notice is that of `stdlib/condition.ml`.

The compiler reads the interface `exported_externals.mli` when it
compiles `exported_externals.ml`. The interface begins with the notice
of `stdlib/condition.mli`, from which four of its five declarations
are taken or adapted; its own comment, under the notice, says which.

No other source of this directory begins with the notice of another
file. As `../ocaml/README.md` says of the sources of its directory, that
a source has no such notice is not a finding that it has no text of the
library.
