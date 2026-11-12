# Restructuring the Flambda 2 Rocq development for review

Suggestions for `middle_end/flambda2/rocq/` and
`middle_end/flambda2/types/rocq/`. The figures come from
`rocq/proof-map.py`, which reads the sources as text (it finds about 99%
of the references that Rocq records, and adds a few percent that are not
there), so take them as sizes, not as exact counts. Lines are lines of
Rocq without comments and blank lines.

## 1. Where things stand

The rule of `rocq/HACKING.md` ("The layout of a model") is that a Spec
file holds "everything a reviewer has to read about the model and nothing
else", and that it "mentions an implementation function by name, never
its body". The dependency discipline that goes with it holds everywhere:
no Spec, Model or implementation file requires a Proofs file, and no
statement mentions anything defined in one.

What a Spec file *contains* has drifted from the rule:

| In the 191 Spec files                                   | lines  |
|---------------------------------------------------------|-------:|
| propositions (`: Prop` definitions and inductives)      |  4,668 |
| types                                                   |    359 |
| functions that some proposition or type is defined with |  5,817 |
| functions that no proposition or type reaches           | 10,377 |
| statements of results (claims no other claim uses)      |  9,028 |
| statements of lemmas (claims some other claim uses)     |  9,556 |

So of about 40,000 lines under the name Spec, a quarter is algorithm and
another quarter is statements that nobody has to read. In the other
direction, about 8,000 lines of functions in implementation files of the
types library are part of what some result means.

A reviewer told to read `*Spec.v` and `*Model.v` reads 42,000 lines. The
meaning of `validate_unit_sound` is under 1,600.

## 2. Move algorithms out of Spec files

1,208 functions in 127 Spec files are reached from no proposition and no
type: they are only ever the thing a statement is *about*. 44 Spec files
are more than half such code, and 72 Spec files define no proposition
and no function of the semantics at all: they state facts about their
own functions in vocabulary defined elsewhere.

`FlambdaLoopifySpec.v` is typical: `is_self_tail_call`, `loop_calls`,
`loop_safe`, `loop_handler`, `looped_code`, `loopable`, `loopify_code`,
`loopify_unit`, then one statement, `program_refines e (loopify_unit G
e)`. A reader of that statement needs `program_refines`. The 128 lines of
the rewrite are what the theorem is about, and it is true whatever they
do.

The largest:

| Spec file                          | movable | kept (propositions, semantic functions) |
|------------------------------------|--------:|----:|
| `FlambdaSimplifyPassSpec`          |     711 |  10 |
| `FlambdaUnboxVariantsExamplesSpec` |     466 |   2 |
| `FlambdaWellFormedSpec` (see 2.1)  |     414 | 196 |
| `FlambdaSimplifyPruningSpec`       |     373 |  19 |
| `FlambdaNormaliseSpec`             |     365 | 325 |
| `FlambdaTypeMeaningBoolSpec`       |     354 |   8 |
| `FlambdaWellKindedSpec` (see 2.1)  |     347 |   0 |
| `FlambdaEnvAliasesFollowSpec`      |     321 |  46 |
| `FlambdaSimplifySwitchMergeSpec`   |     293 |   5 |
| `FlambdaStaticsInScopeSpec`        |     275 |  54 |
| `FlambdaRenameApartSpec`           |     250 |   0 |
| `FlambdaKnownSymbolsSpec`          |     221 |   9 |
| `FlambdaValidateSpec`              |     174 |  23 |

**Suggestion.** For each such model `X`, move the functions to `X.v` (the
implementation file that the layout already provides for) and leave in
`XSpec.v` the propositions, and the statements, which name the functions.
`XSpec.v` then requires `X.v`, as the Spec files of the types library
already require `FlambdaMeet.v`.

Cost: `FlambdaExtract.v` names what it extracts by module
(`FlambdaValidateSpec.validate_unit`), and the test tool under
`tests/tools/` uses the extracted module names, so each move renames an
OCaml module there. It is mechanical, and best done one model at a time by
the model's owner, with the integrator taking the extraction list.

### 2.1 Three kinds of function, only one of which moves

The tool cannot tell these apart, and the split should be made by hand:

1. **Rewrites and checkers that are the subject of a theorem**
   (`loopify_unit`, `validate_unit`, `normalise_before`,
   `prune_by_types`). Move to `X.v`.
2. **Functions of the semantics** (`step`, `eval_prim`, `run`): what the
   propositions are defined with. They stay in Spec. About 5,800 lines,
   concentrated in `FlambdaPrimitiveSpec`, `FlambdaSemanticsSpec`,
   `FlambdaHeapSpec`, `FlambdaMapSpec`.
3. **Boolean conditions used as premises** (`well_formed_unit`,
   the checks of `FlambdaWellKindedSpec`, `names_in_scope`). No
   proposition is defined with them, so they show as movable above, but
   they are hypotheses of results (`well_formed_unit u e = true -> ...`)
   and a reader has to know what they demand. They stay in Spec. It would
   help to say which they are: a results file (section 5) that names them,
   or a `Definition well_formed (u) (e) : Prop := well_formed_unit u e =
   true` used in the statements, which makes them propositions to the
   tool and to a reader alike.

`FlambdaNormaliseSpec` shows a fourth case worth looking at: 325 lines of
its functions are reached from a proposition somewhere, so some
specification is written with a normalisation. Either that proposition is
really about the rewrite (and could take it as an argument), or the
rewrite has become part of what is trusted.

## 3. A file for types

Types are spread thinly: 98 type declarations in 46 Spec files, 82 in
implementation files, 48 in the three Model files. They are little text
(about 1,100 lines in all) but every reader needs them first, and they are
what a statement quantifies over.

For the terms library the Model files already do this job
(`FlambdaModel`, `FlambdaKindModel`, `FlambdaPrimitiveModel`: types and
the few functions that read them). The types library has no equivalent:
`FlambdaTypeGrammar.v` is 114 lines of types and over 900 lines of
functions, `FlambdaJoinEnv.v` has 77 lines of types in 1,400.

**Suggestion.** Yes to a types file, with a rule for what goes in it
rather than "every `Inductive`":

- `XTypes.v` holds the data a model's propositions or statements mention:
  inductives, records, type abbreviations, and the *observers* that only
  take them apart (projections, `kind`, equality tests). No algorithm.
  It requires only other Types and Model files.
- A Types file that copies a type of the compiler is what is now called
  Model. Keeping the name `*Model.v` for exactly those (with the header
  saying which `.ml` to compare with) and `*Types.v` for the rest keeps
  the distinction between "check against the compiler" and "just read".
- A type that only an algorithm uses (an accumulator, the state of a
  traversal) goes with the algorithm in `X.v`. One that a statement
  mentions, even only to quantify over it, goes in the Types file.
- Inductive *propositions* are not types in this sense. They stay in Spec.

First candidates: split `FlambdaTypeGrammar.v` into
`FlambdaTypeGrammarModel.v` (the grammar, after `type_grammar.ml`) and the
functions; the same for `FlambdaTypingEnv.v`, `FlambdaEnvAliases.v`,
`FlambdaJoinEnv.v`. Then the machine's own data
(`FlambdaValueSpec`, the `state` of `FlambdaSemanticsSpec`, the heap of
`FlambdaHeapSpec`) into Types files, so that the semantics files are
functions and propositions only.

This also fixes most of section 4.

## 4. Specifications written with the implementation (types library)

About 8,000 lines of functions in implementation files are meaning of
some result: `FlambdaMeet` (1,500), `FlambdaJoinEnv` (1,300),
`FlambdaJoinTypes` (1,200), `FlambdaProvers` (1,000), `FlambdaTypeGrammar`
(900), `FlambdaTypingEnv` and `FlambdaEnvAliases` (450 each),
`FlambdaMeetEnv` (400). The Spec files that pull them in most are
`FlambdaJoinSpec`, `FlambdaJoinStateSpec`, `FlambdaJoinEnvSpec`,
`FlambdaMeetSpec`, `FlambdaTypingEnvKindedSpec`, `FlambdaJoinMeetSpec`.

This is the pattern that can make a theorem say less than it seems to: a
proposition about the join defined by calling the join. Much of it will
be innocent, in two ways that a restructuring would make visible:

- A proposition reads a data structure through its accessors
  (`Typing_env.find`, the alias lookups). Once the observers are in a
  Types file (section 3), these stop counting.
- A statement is an equation between functions with no proposition
  (717 claims are; see section 6).

What is left after those two is worth a reviewer's time: each
proposition in a Spec file whose definition calls `meet`, `join`, or a
prover. For each, either pass the function in as an argument of the
proposition (so that the proposition is about any function and the
theorem instantiates it), or say in the header that the function is
specification and is to be read.

## 5. Say which theorems are the results

Of 2,291 claims, 1,185 are used in the proof of another claim. Their
statements are 9,556 lines of Spec that nobody has to read: Rocq has
checked them, and what they are for is the claim above. Of the 1,106
that are left, 519 rest on no other claim (small facts), and about 220
are examples.

HACKING.md already has the test: "start from the theorems
ROCQ-REVIEW-GUIDE.md names and follow their definitions; what is reached
is Spec, and what is not, however meaningful, is a Proofs definition."
For the loop passes the guide names three statements. For Flambda 2
nothing names them, so everything is a result.

**Suggestions.**

- One file, `FlambdaResults.v` (or a section of the review guide), that
  lists the claims a reviewer is asked to believe, each with a sentence.
  `proof-map.py --result ...` then gives the reading list, and can be made
  to fail when a Spec file holds something outside the cone of that list.
- A claim that only proofs use becomes a plain `Lemma` in the Proofs
  file, as the layout says. Where a lemma statement is kept in a Spec
  because it helps to interpret a result (`step_deterministic` is the
  example in HACKING.md), keep it, and say so next to it.
- If moving 1,185 statements is too much churn, a cheaper version: two
  parts in each Spec file under fixed banners, "Results" and "Lemmas (not
  to be reviewed)", which the review document and the tool can go by.

## 6. Statements of equations only

717 claims name no proposition: equivariance (`f (rename r e) = rename r
(f e)`), commutation, the examples. With no proposition to carry the
meaning, every function named has to be read, which is why the
equivariance of the validation costs 11,000 lines to review and its
soundness 1,600. Most are in `FlambdaEnvRenameSpec` (63),
`FlambdaValidatorEquivarianceSpec` (50), `FlambdaTypingEnvExtensionSpec`
(28), `FlambdaTypeWellFormedSpec` (25).

**Suggestion.** State them through one small proposition, as
`FlambdaValidatorEquivarianceSpec` already does for 17 of its claims
(`rewrite_equivariant f`). The reader reads `rewrite_equivariant` once,
and every instance then has a subject. Also decide whether these are
results at all: most exist to be used by a soundness proof, and belong
under section 5's "lemmas".

## 7. Examples

22 files (4,000 lines, about 220 claims) are named `*ExamplesSpec.v`. They
matter for a reason of their own: they are the evidence that a hypothesis
can be met and that a checker answers yes on something, which no
soundness theorem shows. But they are read differently (run, skimmed),
and they hold a lot of algorithm-like setup
(`FlambdaUnboxVariantsExamplesSpec`: 466 lines).

**Suggestion.** Name them `XExamples.v` and `XExamplesProofs.v`, outside
the Spec count, and have the header of each say which hypothesis or
checker it shows to be non-empty. A table of "checker, example that
passes it" would be a good page of the review guide.

## 8. Small things a check could enforce

- Nine `_statement` definitions are in Proofs files
  (`FlambdaAlphaProofs`, `FlambdaFreshBindersProofs`,
  `FlambdaNominalProofs`). Either they are lemmas (drop the suffix) or
  they belong in a Spec.
- 17 claims of `FlambdaValidatorEquivarianceSpec` are proved by a `Lemma`
  whose type is not written as the statement (`Lemma f_equivariant :
  rewrite_equivariant f`), and one by a `Fixpoint`. `make claims` accepts
  them by conversion. Writing `Theorem X : X_statement` everywhere makes
  the pairing visible to a reader and to simple tools.
- The headers of three Proofs files mention review
  (`FlambdaJoinDefineProofs`, `FlambdaJoinEnvProofs`,
  `FlambdaJoinStateProofs`), and ten Spec files do not say what in them
  is to be reviewed. Worth a look.
- 56 names are defined at top level in more than one non-Proofs file
  (`naked_immediate` in four, `simple_eqb`, `outcome`,
  `canonical_named` in three). Which one a statement means then depends
  on the order of the `Require`s. Share or rename.
- `FlambdaMeetValidity.v` is an implementation file with propositions and
  18 proved lemmas in it: it wants the three-way split.
- The largest Spec files (`FlambdaSimplifyPassSpec` 1,200 lines,
  `FlambdaHeapSpec` 1,050) shrink a great deal under sections 2 and 5.
  What is left of `FlambdaHeapSpec` may still want cutting in two.

## 9. The layout this arrives at

| File                | Holds                                               | Review |
|---------------------|-----------------------------------------------------|--------|
| `XModel.v`          | types copied from the compiler, and their observers | against the compiler |
| `XTypes.v`          | other data that statements mention, and observers   | read |
| `XSpec.v`           | propositions, functions of the semantics, boolean premises, statements of results | read |
| `X.v`               | algorithms: what the theorems are about             | none (tests run it) |
| `XProofs.v`         | lemmas and proofs                                   | none (Rocq) |
| `XExamples.v`       | instances that show hypotheses and checkers non-empty | skim, run |
| `FlambdaResults.v`  | the list of claims a reviewer is asked to believe   | read first |

Allowed requires, each of which a script can check: Model and Types
require only Model and Types; `X.v` requires those and other
implementation files; Spec requires all of the above but never a Proofs
file; Proofs anything.

## 10. Order of work

1. Write the list of results (section 5). It costs nothing, changes no
   file, and sizes everything else: run `proof-map.py --result ...` on it.
2. The small things of section 8.
3. Split the grammar and environment files of the types library into
   Model/Types and functions (section 3), then look at what section 4
   still shows.
4. Move the algorithms out of the Spec files of the terms library, model
   by model, largest first (section 2). This is the bulk of the gain in
   the reading list of anyone who reads by file name.
5. Demote or mark the lemma statements (section 5), and wrap the
   equational claims (section 6).
6. Rename the examples (section 7).

After each step the first page of `rocq/proof-map.pdf` should show "By
file name" closer to "Their meaning only"; when the two agree, the file
names say what to read.
