# C3: finer tests for the liftings. A core change request

From C3, for the human (section 6: decision D65, which is his, and
which this change takes one side of), C0 (who applies it), R2 (who
reviews what the two finer tests mean), R1 (the text), and the owners
of the files that it touches (section 5).

**The human ruled on decision D65 on 18 November 2026: yes,
provisionally; to be come back to.** The change lets more pairs be
validated, by proved statements, and what it gives away is not in any
statement: section 3 says it.

## Why

Each of the four liftings of a unit
(`FlambdaLiftSpec.lift_unit`,
`FlambdaLiftFromArmsSpec.lift_to_head_unit`,
`FlambdaLiftFromCodeSpec.lift_from_code_unit`,
`FlambdaLiftAroundSpec.lift_around_unit`), and the copy of code for a
`Let` (`FlambdaCopyLetsSpec.copy_let_code`), does nothing in a unit
unless the unit passes two tests: that it exits with symbols and
constants only, and that it has no primitive of the heap (decision
D40, and decision D43 for what the first refuses). Both tests looked
into the bodies of the unit's code. So:

- in a unit one of whose functions raises a block that it makes
  (`let get = function Some v -> v | None -> invalid_arg "..."`) no
  lifting is made, and a pair that needs one is not validated;
- in a unit one of whose functions stores in a cell (`r.v <- v`) no
  lifting is made either.

A module of a library is often a unit of that sort, and its own run
often calls nothing: it makes closures, puts them in a block and
returns the symbol of the block (section 4 has a count).

## 1. The two tests, as they were and as they are

They were `FlambdaLiftSpec.exits_plain_knowing [] [] e` and
`FlambdaHeapFreeSpec.expr_free_of_heap e`. Both stay in the tree,
with every statement about them, as they are.

The five functions now ask `FlambdaLiftSpec.exits_plain_fine e` and
`FlambdaHeapFreeSpec.free_of_heap_fine e`:

    free_of_heap_fine e :=
      expr_free_of_heap e
      || (calls_nothing e && expr_free_of_heap_outside_code e)

    exits_plain_fine e :=
      exits_plain_knowing [] [] e
      || (calls_nothing e && exits_plain_knowing [] [] (strip e))

- `FlambdaHeapFreeSpec.calls_nothing e`: the expression has no
  `Apply`. Handlers are looked into, those of a recursive group too;
  the code that a `Let_static` binds is not.
- `FlambdaHeapFreeSpec.expr_free_of_heap_outside_code e`: no `Let` of
  a primitive of the heap, the code that a `Let_static` binds apart.
- `FlambdaLiftSpec.strip e`: the expression with `Invalid` for the
  body of each piece of code that it binds, and nothing else changed.
  Code whose body is `Invalid` passes the test of the exits, so the
  test of `strip e` is the test of `e` with every binding of code
  counted as passing.

Each accepts whatever the test before it accepted, and more.

## 2. Why the statements stay true

From the first state of a unit, with no continuation given, the
first entry into code needs an `Apply`. (An `Apply_cont` enters code
where its continuation is what is left of an over-application,
`FlambdaSemanticsSpec.apply_cont_value`; a call makes such a
continuation, so a run that has made no call has none.) So the run
of a unit that has no `Apply` outside the bodies of its code enters
none. Then no exit of code is an exit of that run, and no primitive
of the heap that is in code is executed.

The condition is of the text (`calls_nothing e = true`), and it is
more than that one run enters no code: an `Apply` in a handler that
no run reaches makes it fail.

Proved, with nothing admitted and no axiom:

- `FlambdaHeapFreeSpec.calls_nothing_exits_empty_statement`: a unit
  that calls nothing and has no primitive of the heap outside its
  code, run from the empty heap and from any static environment,
  exits with the empty heap. The proof is that of
  `free_of_heap_exits_empty_statement` with the cases of a call taken
  out.
- `FlambdaHeapFreeSpec.free_of_heap_fine_exits_empty_statement`: the
  statement of the guard, with the finer guard.
- `FlambdaLiftSpec.calls_nothing_exits_plain_statement`: a unit that
  calls nothing and passes the test with its code taken out, run from
  any static environment, exits with symbols and constants only. The
  proof does not open that of the test before: a unit that calls
  nothing and the unit stripped run step for step alike, to the same
  exit, and the statement of the test before is used of the unit
  stripped.
- `FlambdaLiftSpec.exits_plain_fine_sound_statement` and
  `exits_plain_fine_sound_from_statement`: the two statements of the
  test before, with the finer test.
- That the tests answer of a term renamed what they answer of the
  term: `calls_nothing_rename_statement`,
  `expr_free_of_heap_outside_code_rename_statement`,
  `free_of_heap_fine_rename_statement`, `strip_equivariant_statement`,
  `exits_plain_fine_equivariant_statement`.

The two statements of each of the five functions
(`..._program_refines_statement`, `..._refines_imports_statement`)
keep their text, and so does every other statement of the `Spec`
files. Their proofs change in one place each: where they carried the
invariant of the test before beside their simulation, they carry
what the test is for, that every exit of the run from here has
symbols and constants only (`FlambdaLiftProofs.exits_are_plain`). No
simulation is changed.

## 3. What is given away

It is not in a statement, so it is said here, and in the two `CR`s
(at `exits_plain_fine` and at `free_of_heap_fine`).

The statement of a verdict that rests on a lifting is of the unit's
own run (`FlambdaUnusedStaticsSpec.program_refines`,
`FlambdaImportsSpec.unit_refines`). For a unit whose own run enters
no code it says nothing of the bodies of its functions, with these
tests or with those before: no state of that run is in a body. What
the validation has of the bodies of such a unit is that the two
terms, bodies included, are the same once its rewrites are made,
each of which has its own statement. That is evidence about the
text, and no theorem about a call of one of the unit's functions:
the statement of a lifting does not speak of one.

The tests of before are of the text of all the code of a unit,
entered or not. With them a lifting was made, and a verdict could
rest on it, only in a unit whose code passes the test of the exits
and has no primitive of the heap. The finer tests ask that only of a
unit that has an `Apply` outside the bodies of its code. So a unit
that has none, and passes the two tests outside those bodies, passes
them though its functions raise what they make or store; where the
other checks of a lifting pass (the result is a well-formed unit,
and what each lifting asks besides) it may be lifted, and a verdict
may rest on that lifting, on a run that calls none of those
functions. "Validated" then says of it what the verdict already said
of a module whose functions do neither, and no more.

Neither way is it a rule about every unit. A pair may be validated
with no lifting, whatever its functions do: the first comparisons of
`FlambdaValidateSpec.validate_unit` come before any. And a unit that
passes the finer tests may fail another check of a lifting, or the
comparison that follows.

The other answers are in section 6.

## 4. What it turns

Measured on 17 November 2026, with the test tool that the tree then
had, built with the extracted files that `make -C rocq all` made from
the definitions of this change. Those files are not in the commit of
the definitions (section 5), so the tool of that commit does not give
these figures.
All are of the compiler's own pair of terms, with the units compiled
as the build of the standard library compiles them. The same figures
were found before the definitions were written, with the two tests
written by hand in a copy of the tool.

- Of six units of the standard library that are compared and not
  validated, and were looked into for every cause
  (`camlinternalAtomic`, `stdlib__Either`, `stdlib__Option`,
  `stdlib__Result`, `stdlib__Pqueue`, `stdlib__StringLabels`), the
  finer tests alone turn one, `camlinternalAtomic`, by the guard of
  the heap. `Either`, `Option`, `Result` and `Pqueue` need the finer
  test of the exits and something else each (below), and stay not
  validated; `StringLabels` needs neither. Of `stdlib__Map` and
  `stdlib__Set`, looked into by V3 with the finer test of the exits
  written by hand and one change more, V3's, that is not in this
  tree: with the two together `Map` is validated with its imports and
  `Set` is not. Neither unit was measured with the finer tests alone.
- Of 55 small units cut from `either.ml`, `option.ml` and
  `result.ml`: 43 are validated before, 50 with the finer tests. Most
  of the cuts are one function each; one is a constant (`let none =
  None`) and one is the `Syntax` part of `result.ml` with `bind`,
  `map` and `product`. Two cuts more, of `Either.iter` and
  `Either.for_all`, do not compile alone and are not counted.
  The seven that turn are the functions that raise (`Either.get_left`,
  `get_right`; `Option.get`; `Result.get_ok`, `get_ok'`, `get_error`,
  `error_to_failure`). The five that do not: Simplify makes a closure
  of a wrapper of a partial application of a function that it knows
  from another unit (`Option.to_seq`, `Result.to_seq`), or gives back
  the block that it matched for a block made of its fields
  (`Either.find_left`, `Option.to_result`, `Result.to_option`).
- Of 78 units of the standard library, 52 have no call outside the
  bodies of their code in the term before Simplify, by a count on
  the text of the terms; 11 more have only calls of C functions
  there; 15 call an OCaml function.

## 5. What changes in the tree

Definitions of the `Spec` files. None is removed and none is renamed.

| Definition | |
|---|---|
| `FlambdaLiftSpec.exits_plain`, `exits_plain_knowing`, `FlambdaHeapFreeSpec.expr_free_of_heap`, `statics_free_of_heap` | stay, with their statements |
| `FlambdaHeapFreeSpec.calls_nothing`, `expr_free_of_heap_outside_code`, `free_of_heap_fine` | new |
| `FlambdaLiftSpec.strip_code`, `strip_binding`, `strip`, `exits_plain_fine` | new |
| `FlambdaLiftSpec.lift_unit`, `FlambdaLiftFromArmsSpec.lift_to_head_unit`, `FlambdaLiftFromCodeSpec.lift_from_code_unit`, `FlambdaLiftAroundSpec.lift_around_unit`, `FlambdaCopyLetsSpec.copy_let_code` | another body: the two tests asked |

Statements, in the files that `rocq/check-changed-defs.py` reads (the
`Spec` files and the three files of the model). Ten are new (section
2). None changes its text and none is removed. The script counts the
statements in which a changed definition occurs, directly or through
other definitions: 79 for each of the four liftings, 22 for
`copy_let_code`. That is a count of names, and no claim about each
of those statements.

The `Proofs` files, which the script does not read. Lemmas are added.
The relation that each refinement of the five carries has another
second part (`unit_lifted_plain` and its like,
`FlambdaCopyLetsProofs.unit_related`). One lemma goes,
`FlambdaCopyLetsProofs.exits_plain_renamed`: the laws of the finer
tests are used in its place.

Files of other streams, each with its owner's word:

- proofs only: `validation/names/FlambdaValidatorEquivarianceProofs.v`
  (four lemmas: the liftings of a renamed unit),
  `validation/names/FlambdaStaticsScopeProofs.v` (one lemma),
  `validation/checks/FlambdaValidateCopyingProofs.v` (one line);
- words only, where a header names the two tests:
  `validation/checks/FlambdaValidateRelatedSpec.v`,
  `rewrites/lifting/FlambdaLiftObservedSpec.v`,
  `validation/checks/FlambdaValidateCopyingSpec.v`, `NAMES.md`.

The functions of `FlambdaLiftObservedSpec.v`, which make the liftings
without the two tests, and the second check of a unit
(`FlambdaValidateRelatedSpec.v`) do not change.

Not done in this commit:

- The extracted files of the test tool
  (`tests/semantics/extracted/`) are not regenerated, so the test
  tool as committed still runs the tests as they were. The tool
  compiles with them regenerated, and the measure of section 4 was
  made so.
- The test tool (`tests/tools/flsemantics.ml`) names the two tests in
  its comments and in what it prints of a unit that is not lifted;
  those places are not changed.
- The expected files of `tests/semantics/` are not regenerated, and
  its aliases were not run. `tests/semantics/ocaml/README.md`, which
  describes the two tests as they were, is not changed.

Those go together, in a commit of the test tool.

## 6. The decision: D65

Whether a verdict of the validation may rest on a lifting that is
made in a unit which has no `Apply` outside the bodies of its code,
so that its own run enters none of them, where the tests of the
lifting do not look into those bodies.

- **Yes** (this change). The statements are true as they stand, and
  a pair that the two tests alone kept from being validated may be
  validated. What "validated" says of such a unit is of its own run,
  which makes its closures and its module block and calls nothing.
- **No**: keep the tests as they were (`exits_plain_knowing`,
  `expr_free_of_heap`). No such lifting is made, and a pair that
  needs it stays not validated unless another comparison succeeds,
  with the line of the second check under it for what that accepts
  (decision D43).
- **Not yet**: wait for a statement that speaks of the functions of a
  unit under every caller, and decide then what the two tests are to
  be. The tree has no such statement.

The human's answer, on 18 November 2026: yes, provisionally; to be
come back to. `DECISIONS.md` has the entry.
