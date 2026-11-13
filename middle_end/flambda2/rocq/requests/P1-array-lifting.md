# P1: an immutable array lifted to a symbol, and read where it is known

For the source `[: x; y :]` once it is a value of the model: the
term after Simplify has a symbol where the term before has the
`Let` of the array, and a constant where the term before has the
length or an element of an array that it has just made.

For the human, V3 (the lifting), V2 (the rounds of the validation,
the tool and the corpus), and the reviewers R3 (the lifting), R1
(the record), R5 (the primitives) and R6 (the rewrites of the
validation).

Two changes are asked, each a series of its own, the second after
the first. When this was written nothing of either was in the tree.
The definitions of sections 2 and 3 were written and compiled, with
no proof, on a5e4605469, whose `.v` files and tools are those of
3270ad86ed; the test tool was built with them, and the measures of
section 4 were made with that tool. No statement was proved then,
and nothing here is a verdict on any unit.

Since it was written the orchestrator has permitted both, the first
as an explicit change of the relation `value_lifted`, on conditions
that sections 2.2 to 2.7 now meet or name; and the statements of
2.3 were compiled against the two definitions, the last of them
with `lift_simulates_statement` taken as a hypothesis. The first
series is written by the owner of the lifting, in its files, with
this stream's two files; the second by this stream.

The first series is in the tree now. Four things in this file are
of the series as it went in and not as it was asked: the end of
section 2.3 (what is proved of a unit that makes no array); in
section 2.4 the classes of the statements that have the relation
and the counts made on the tree that has the series, which has
gained statements since the request was written; and at the end of
the first paragraph of section 2.6 what is said of an array of the
source that only a later lifting gives a symbol. `status/V3.md`
says what becomes of each statement.

## 1. Why

The immutable array as a value is in the tree
(`requests/P1-immutable-array-values.md`, `status/P1.md`): the term
before is converted and run. It is seldom validated. Section 4.2
of that request has the case of `lift_named` and the rule from an
array value to a symbol, which its series left out by agreement
with the owner of the lifting; this is their request, with the
measure that the earlier one could not make.

On a set of 135 units written to try immutable arrays as values
(not in this tree), the tool of this tree does not validate 58 at
the default flags and 63 at `-O3`. Read from the two terms that
the tool prints, 43 and 44 of them have an array that is a `Let`
in the term before and a symbol in the term after, or is gone from
the term after with its reads.

## 2. The lifting (the first series)

### 2.1 What Simplify does

`FlambdaLiftSpec.v` says it already, for the three things that
Simplify lifts by the type of the result of a primitive that only
allocates: an immutable block of values, a boxed number, an
immutable array. For the array, read again for this request:

- `simplify/simplify_variadic_primitive.ml`, `simplify_make_array`:
  a `Make_array` at `Immutable` has the type of an immutable array
  of the types of its arguments (`T.immutable_array ~fields`), and
  asks to be reified (`~try_reify:true`).
- `types/reify.ml`, `reify`: such a type whose element kind is
  `Value` is rebuilt as `Immutable_value_array` of its fields where
  each field is a constant, a symbol, or a variable that is defined
  at the toplevel of the unit or is a projection from a symbol
  (`try_to_reify_fields`, as for a block); one of no field as
  `Empty_array`.
- `simplify/lifting/reification.ml`, `create_static_const` and
  `lift`: the symbol is named after the variable, unless an equal
  constant has been lifted already, when the symbol of that one is
  used (`DA.find_shareable_constant`).

So an array made at the toplevel of the unit is lifted with a
variable among its elements too, and an array of symbols and
constants is lifted out of a function or a loop, as a block is.

### 2.2 The definitions

In `FlambdaLiftSpec.v`, as section 4.2 of the earlier request gave
them:

    lift_named, one case more:
      | Lift_to_symbol s,
        Prim (Variadic (Make_immutable_array _)) elements =>
        Some
          ([Bind_symbol s
              (Static_const_immutable_value_array elements)], s)

    value_lifted, one rule more:
      Lifted_to_array:
        lookup s (static_values statics')
        = Some (Static_immutable_array elements') ->
        Forall2 (value_lifted statics') elements elements' ->
        value_lifted statics' (Value_immutable_array elements)
          (Value_symbol s)

Nothing else is defined. The lifting out of code, out of the arms
of a test and to a group around go through `lift_named`
(`FlambdaLiftFromCodeSpec.lifted_out`,
`FlambdaLiftAroundSpec.alias_of`), so they take the array with no
change of their text.

What the case does that the `Let` did not:

- The array kind is dropped: the constant has none. The `Let` has
  no value where an element is not one that its kind holds
  (`FlambdaPrimitiveSpec.array_made`) or is a symbol bound to
  nothing; the binding has a value wherever its elements have. So
  the term lifted is defined where the term was not. That is a
  refinement and no equivalence, and a refinement is what every
  statement of the lifting concludes.
- Two `Let`s of one array may be given one symbol where the
  lifting shares (out of one body of code, or as aliases of a
  symbol bound around them). No array value has an identity in the
  model: `Phys_equal` of two has no value
  (`FlambdaPrimitiveSpec.eval_phys_equal`; decision D19), so a unit
  that could tell is undefined before it is lifted. The compiler's
  own choice between one symbol and two is by the debugging
  information of the elements and by how the copies were made
  (T5-11); the model has no part in it.
- Nothing of a mutable array, and nothing of an array whose kind is
  not one of values: the primitive lifted is
  `Make_immutable_array`, which the conversion makes of a
  `Make_array` marked `Immutable` at one of the three array kinds
  of values and of nothing else.
- The empty array: Simplify binds `Empty_array`, which the
  conversion reads as the static immutable array of no element.
  A `Make_immutable_array` of no argument is lifted to it like
  another.

### 2.3 The statement whose text changes

One, in `FlambdaNoArrayValueSpec.v`, a file of this stream. Today:

    value_lifted_array_statement:
      value_lifted statics' (Value_immutable_array elements) v' ->
      exists elements',
        v' = Value_immutable_array elements'
        /\ Forall2 (value_lifted statics') elements elements'

It is false with the rule. It becomes, with the comment before it
("to an array value of its elements lifted, or to a symbol bound
to an immutable array of its elements lifted, and to nothing
else"):

      value_lifted statics' (Value_immutable_array elements) v' ->
      exists elements',
        Forall2 (value_lifted statics') elements elements'
        /\ (v' = Value_immutable_array elements'
            \/ exists s,
                 v' = Value_symbol s
                 /\ lookup s (static_values statics')
                    = Some (Static_immutable_array elements'))

Its conclusion is weaker: this is for the orchestrator, and for R3,
who signed the text that is there. The header of that file says of
it "a rule that did would be a case more of it": that sentence
goes, and the paragraph says what the statement then says.

Three statements are added after it in the same file, so that what
`lift_simulates_statement` said before the rule it says, as a
proved statement, of a unit that makes no array. No copy of the
relation as it was is kept. The rule is the only one that reads a
symbol bound to an immutable array, so the relation without it is
the relation under the static environment with those bindings left
out:

    without_static_arrays statics :=
      the static environment with the symbols that are bound to an
      immutable array left out

    no_array_lifted_to_symbol_statement:
      ~ value_lifted (without_static_arrays statics')
          (Value_immutable_array elements) (Value_symbol s)

    value_lifted_reads_no_static_array_statement:
      value_lifted statics' v v' ->
      no_array_value v = true ->
      value_lifted (without_static_arrays statics') v v'

    lift_simulates_free_of_arrays_statement:
      (the premises of lift_simulates_statement) ->
      makes_array e = false ->
      behaves (the unit, run on its own) (Exits k vs hp) ->
      (forall v, v in vs -> no_array_value v = true)
      /\ exists n last vs',
           run n (the unit lifted, run on its own) = Step last
           /\ step last = Exit k vs' hp
           /\ Forall2
                (value_lifted
                   (without_static_arrays (state_statics last)))
                vs vs'

The third follows from the theorem of `lift_simulates_statement`,
from `run_free_of_arrays_statement` and from the second.

As it went in. The sentence above that the relation without the
rule is the relation under the static environment with those
bindings left out is not true as it stands, and the three
statements do not say it: a static environment may bind a symbol
twice, and leaving the binding to an array out uncovers the other.
What the second proves is one direction, from the relation to the
relation under that environment; the converse is not claimed. So
the series has, beside the three, the rules named and two
statements under the static environment as it is, which are what
keeps the old reading:

    value_lifted_no_array statics' :=
      the rules of value_lifted but Lifted_array and
      Lifted_to_array, each as it is, under statics'

    value_lifted_no_array_exact_statement:
      value_lifted_no_array statics' v v'
      <-> value_lifted statics' v v' /\ no_array_value v = true

    lift_simulates_no_array_statement:
      (the premises of lift_simulates_statement) ->
      makes_array e = false ->
      behaves (the unit, run on its own) (Exits k vs hp) ->
      exists n last vs',
        run n (the unit lifted, run on its own) = Step last
        /\ step last = Exit k vs' hp
        /\ Forall2 (value_lifted_no_array (state_statics last))
             vs vs'

The rules named are not said to be the relation as it once was,
which had `Lifted_array` before it had `Lifted_to_array`: they are
what the relation is at the values that hold no array value.

### 2.4 What each changed definition reaches

Three conclusions become weaker, and no other. That of
`lift_simulates_statement`, whose text is the same: the relation in
it relates one more kind of pair. That of
`value_lifted_array_statement`, whose text changes (2.3). And that
of `value_lifted_to_array_statement`, whose text is the same, in
the relation of the elements only: what is lifted to an array value
is still an array value, and its elements are related to those of
the other by the relation, which now relates an array of one empty
array to an array of one symbol bound to an empty static array. The
request as it was written named the first two and not the third.

Counted by `rocq/check-changed-defs.py` between a5e4605469 and the
definitions of 2.2 alone: no statement added or removed, no
definition added, two definitions whose body changes. (The text
change of 2.3 is not in that count: the prototype kept the
statement as it is.)

`lift_named` reaches 82 statements, all through other definitions
(`lift_expr`, `lift_unit`, `lifted_out`, `lift_from_code`,
`lift_to_head`, `alias_of`, `lift_around`, `with_lifting`,
`lifted_in_turn`, `validate_unit` and what is built on them). The
function gives the bindings of one more form of `Let` where it
gave none. No premise is added to any of them and no conclusion is
lost, apart from the weaker conclusion of
`lift_simulates_statement`, which is at the head of this section;
each is then a statement about the function as extended, so the
case of the array is new proof in each proof that goes by the
cases of `lift_named` or by the rules of `value_lifted`. A guess
that names no `Let` of an array gives what it gave. Of a guess that
names such a `Let` no inclusion is claimed in either direction:
`lift_expr` left the `Let` as it was and went on, and now lifts it,
so a lifting may give another term, or none where it gave one, and
the lifted term may fail a check that the term with the `Let` left
passed.

`value_lifted` is in the text of four statements and of no other:
searched as a word in every `.v` file of the two libraries, it is
outside comments in `FlambdaLiftSpec.v`, in
`FlambdaNoArrayValueSpec.v` and in the files of proofs of the
lifting and of that file; no definition of a `Spec` file is made
with it but the relation itself. The four, each directly:

- `lift_simulates_statement` has it in its conclusion: the values
  at the exit of the term lifted are those of the term up to it.
  The relation relates one more kind of pair, an array value and a
  symbol bound to an array of its elements lifted. That is what
  the statement is for (it says the same of a block, a boxed
  number and a closure), and it is a weaker conclusion: for the
  orchestrator. What it said of a unit that makes no array it
  still says: the run of such a unit holds no array value
  (`FlambdaNoArrayValueSpec.run_free_of_arrays_statement`), the
  new rule has an array value for its first value, as
  `Lifted_array` has, and a derivation goes by the head of the
  first value. `lift_simulates_free_of_arrays_statement` (2.3) is
  that, as a statement to be proved in the series.
- `value_lifted_array_statement`: 2.3.
- `value_lifted_to_array_statement` (nothing but an array value is
  lifted to an array value) and
  `value_lifted_no_array_value_statement` (a value that holds no
  array value is lifted to values that hold none): their texts
  stay and they are true of the relation with the rule; each is
  proved again, with the case.

The statements that a unit lifted refines the unit keep their
checks (`exits_plain`, and that the lifted form is a well-formed
unit): an array value is no plain value, so a unit that could
leave with one is refused as before. The six statements of
`FlambdaLiftObservedSpec.v` are among the 82: no primitive is
added, so what may follow a unit is what it was, and they are
reproved as conclusions.

The statements of the verdict that lifts are among the 82, and
none has `value_lifted` in its text. What "validated once lifted"
allows from now on: an immutable array of values that the unit
makes as it runs may be, in the term that is compared with the term
after, a symbol bound to a static immutable array of the same
elements, lifted. That is of the statements whose premise the
verdict that lifts satisfies. Of the fourteen of the validation
below those are four: `validate_unit_sound_statement`,
`validate_unit_knowing_sound_statement` and the two whose names end
`_sound_verdict_imports_statement`. The ten others (the two whose
names end `_sound_everywhere_statement`, and the eight others of
`FlambdaValidateImportsSpec.v`) accept only verdicts that are
decided before anything is lifted: they are in the list because
the search reaches them through the name of `validate_unit`, and
nothing in them changes. They are:

- of a unit run on its own: `validate_unit_sound_statement`,
  `validate_unit_sound_everywhere_statement`,
  `validate_unit_knowing_sound_statement`,
  `validate_unit_knowing_sound_everywhere_statement`;
- of a unit run from its imports, the ten of
  `FlambdaValidateImportsSpec.v`:
  `validate_unit_sound_imports_statement`,
  `validate_unit_sound_verdict_imports_statement`,
  `validate_unit_sound_from_statement`,
  `validate_unit_sound_head_from_statement`,
  `validate_unit_sound_head_imports_statement`, and the five of the
  second asking, whose names have `knowing` after `validate_unit`;
- of the four liftings, each on its own and from the imports:
  `lift_unit_program_refines_statement`,
  `lift_unit_refines_imports_statement`,
  `lift_from_code_unit_program_refines_statement`,
  `lift_from_code_unit_refines_imports_statement`,
  `lift_to_head_unit_program_refines_statement`,
  `lift_to_head_unit_refines_imports_statement`,
  `lift_around_unit_program_refines_statement`,
  `lift_around_unit_refines_imports_statement`;
- up to what may follow a unit, the six of
  `FlambdaLiftObservedSpec.v`:
  `lift_unit_observed_refines_statement`,
  `lift_from_code_unit_observed_refines_statement`,
  `lift_to_head_unit_observed_refines_statement`,
  `lift_around_unit_observed_refines_statement`,
  `liftings_observed_refines_statement`,
  `with_lifting_observed_refines_statement`.

With `lift_simulates_statement` those are 29. Of the 82, 33 are
reached only through the name `alias_of`, which
`FlambdaForwardSpec.v` defines as well as `FlambdaLiftAroundSpec.v`
(the search counts a use of a name for every file that defines
it): the statements of `forward_applications`, those of the rounds
of the first phase, and thirteen that an example is validated;
`lift_named` is in none of them. The 20 that are left are of
renaming (thirteen: the four liftings and what they are made of,
`with_lifting`, `lifted_in_turn`, `validate_unit`,
`validate_unit_knowing`) and of scope (seven: the six of
`FlambdaStaticsScopeSpec.v` and
`lifted_in_turn_keeps_scope_statement`).

The passage of `rocq/TRUST-LEDGER.md` on what the tool reports says
of the fifth verdict that something is lifted; what may be lifted,
with the array, is added to it in the series.

The law of the primitives that the lifting uses
(`FlambdaPrimitiveSpec.eval_prim_related_statement`) is not
changed: its premise `relates_observations` already asks, of two
related values the first of which is read as an array by
`array_of`, that the second is and that the elements are related,
which the rule gives.

As counted on the tree that has the series, by the same search over
its two commits of definitions and proofs: one statement changes
text (`value_lifted_array_statement`); five are added (the three of
2.3 and the two of its end); three definitions are added
(`is_static_array`, `without_static_arrays`,
`value_lifted_no_array`); two change (`lift_named`,
`value_lifted`).

`value_lifted` is directly in eight statements and in no other: the
four above and four of the five added (the fifth has the rules
named in its place). Of the four that were there: three have a
weaker conclusion (`lift_simulates_statement`;
`value_lifted_array_statement` and
`value_lifted_to_array_statement`, whose premise is weaker too, the
relation being their premise; the last in the relation of its
elements only, as said at the head of this section) and one a
weaker premise only, so that it says of more pairs what it said
(`value_lifted_no_array_value_statement`).

`lift_named` reaches 87 statements, none directly, where it reached
82. Of the 87, 34 are reached only through the name `alias_of`. Of
the 53 that are left, 49 are those above, and two are added:
`lift_simulates_free_of_arrays_statement` and
`lift_simulates_no_array_statement`. So the statements listed
above for the verdict that lifts are 29 with
`lift_simulates_statement` (that one, the 8 of the four liftings,
the 6 up to what may follow a unit, and the 14 of the validation,
of which four have a premise that the verdict that lifts
satisfies), and 31 with the two added.

The two others are apart: `validate_unit_related_sound_statement`
and `validate_unit_related_under_observers_statement`, of
`FlambdaValidateRelatedSpec.v`, the statements of a second check of
a unit that the tree has gained. They reach `lift_named` through
`with_lifting_observed`. They are not of the verdict that lifts:
the answer of that check, `Related_once_lifted`, is of a type of
its own and no verdict of the validation. When the counts of this
section were made the tool did not call the check. The test tool
now asks the check only of a unit that it reports as not validated,
and prints what the check accepts on a line of its own, which has
not the word "validated". Neither has `value_lifted` in its text.
The check makes its liftings without the test of the exits, so the
exits of a pair that it accepts may now differ by an array value
against a symbol bound to a static immutable array of the same
elements lifted.

### 2.5 The comments

`FlambdaLiftSpec.v`: the item "immutable arrays" of "What is not
here" goes; "What is here" gains the third form; the paragraph
before `value_lifted` says "and none is made a symbol either" of
an array, which becomes what it says of a block; "Identity" names
the array beside the block. `FlambdaLiftFromCodeSpec.v` and
`FlambdaLiftAroundSpec.v` each list what `lift_named` lifts, and
say why what is shared has no identity: the array is added to
both lists. These three files are the lifting's: the texts are
its owner's to give or to approve.

### 2.6 The test tool

`tests/tools/flsemantics.ml`, in the guess of the lifting, where
there is a case for `Make_block` there is one for
`Make_immutable_array`, three in all: a `Let` at toplevel (elements
that are variables agree with anything, as fields do); a `Let` in
code or in an arm, of symbols and constants only; and the `Let`s
to make aliases of a symbol bound around them. And one filter: the
tables to bind at the head of the term before
(`tables_guess`) are symbols that the term after binds to an
immutable array and the term before does not bind; an array of the
source that the same lifting gives to a `Let` is one of those and
is no table, and with it among the tables the check of
`FlambdaUnusedHeadStaticsSpec.add_head_tables` fails for all of
them. The symbols that the first lifting gives to `Let`s are left
out of its tables. An array of the source that only a later
lifting gives a symbol (one whose elements are closures that are
lifted first, or one made an alias of a symbol bound around it) is
still taken for a table at the first lifting: the filter does not
cover it, and such a unit may stay not validated. It is not said
that it does: where the table is bound at the head of the term
before, a later lifting may make the `Let` an alias of it.

All of it is a guess: `lift_unit` and its fellows check it, and a
wrong guess is a unit that is not validated. Nothing of the
conversion changes, and nothing of the trust ledger.

### 2.7 Proofs, and who writes them

The owner of the lifting, in its files; this stream writes the
patch of `FlambdaNoArrayValueSpec.v` and of its proofs (2.3), and
each has the other's word on its files. The case of the array
follows that of the block without its tag: in this tree
`Lifted_to_block` is named in `FlambdaLiftProofs.v` (twice: the
relation under a larger static environment, and the step of a
lifted `Let`), in `FlambdaLiftFromCodeProofs.v` and in
`FlambdaLiftAroundProofs.v` (once each); the other uses are
inversions and constructors found by search. `lift_named` is used
in `FlambdaLiftProofs.v`, `FlambdaLiftFromCodeProofs.v`,
`FlambdaLiftAroundProofs.v` and `FlambdaStaticsScopeProofs.v`.

## 3. The length and the elements of a known array (the second series)

### 3.1 What Simplify does

- `simplify/simplify_unary_primitive.ml`, `simplify_array_length`:
  the length is a projection of the type of the array; where that
  type has the elements, the result is their number, a constant.
- `simplify/simplify_binary_primitive.ml`, `simplify_array_load`:
  where the array is proved an immutable array of known fields
  (`T.prove_is_immutable_array`) and the index is one known tagged
  immediate, the result has the type of that field, reified: the
  element itself where it is a constant, a symbol or a variable;
  `Invalid` where the index is out of the bounds. The mutability
  that the load carries is not looked at.

The type comes from `simplify_make_array` (2.1) and, for a symbol,
from its static constant. So in the term after, the length of an
array that the same code has just made is a constant, the test of
an index against it is decided or is a test against a constant,
and a load at a constant index is the element; an array that is
then unused is gone.

### 3.2 The definitions

New files of this stream, `FlambdaArrayLoad*`, after
`FlambdaBlockLoadSpec.v`, which does the same for the fields of a
block that is known from the term alone. `load_known_arrays` goes
through a term keeping, for a variable bound by a `Let` to
`Make_immutable_array` and for a symbol bound by a `Let_static` to
`Static_const_immutable_value_array`, the elements
(`known_arrays`; an element is forgotten where a variable that it
names is bound again, and the whole entry where the variable
itself is: `FlambdaBlockLoadSpec.forget_field`). Then, of a `Let`
whose defining expression is a primitive on such a variable or
symbol:

- `Array_length _` becomes the number of elements, a tagged
  immediate, if it is in range;
- `Array_load _ _ _` at an index that is a constant tagged
  immediate, at least 0 and below the number of elements, becomes
  the element, if what it holds is still known;
- anything else stays: a load at an index that is a variable, a
  load out of the bounds.

A handler of a non-recursive continuation is gone into, with what
is known where it is defined less its parameters; the handlers of
recursive continuations and the bodies of code are left
(`FlambdaNormaliseSpec.rewrite_code` and the rounds take the pass
into them, with nothing known at the start), as for blocks.

The statements, as for `load_known_blocks`: the result refines the
term under every environment
(`load_known_arrays_refines_statement`: `expr_refines e
(load_known_arrays e)`); it commutes with renaming; it keeps
scope. It is a refinement and no equivalence: the load has no
value where the element is not one that its load kind holds
(`FlambdaPrimitiveSpec.array_loaded`), and the element has. So it
is for the term before only.

### 3.3 Where it goes

In `FlambdaValidateSpec.v`, beside `load_known_blocks` in the two
places that apply it, `normalise_expr_before` and
`normalise_expr_params_knowing`:

    unbox_known_numbers (load_known_arrays (load_known_blocks e))

It is none of `merge_tables`, `unbox_loops_knowing`, `untag_params`
and `give_scrutinees`, and moves none of them: it comes before all
of them in a round where they are, at the place of the load of
blocks. That file is V2's: the two lines are asked of it.

Since then the pass has been taken out of the two rounds, where it
made the validation refuse units that it accepts without it, and
is applied by a function beside the validation
(`FlambdaValidateReadingSpec.v`; `status/P1.md`, "The known arrays
read beside the validation"). 3.4 is of the rounds with the pass.

### 3.4 What it reaches

Counted between the definitions of 2.2 and those with 3.2 and 3.3:
one statement added (the refinement; the two others of 3.2 were
not written), 18 definitions added, two changed.
`normalise_expr_before` reaches 39 statements, 10 directly;
`normalise_expr_params_knowing` reaches 43, none directly. The
round of the term before applies one rewrite more; no premise is
added and no conclusion lost, and the proofs that the round
refines, commutes with renaming and keeps scope each gain one
step. Nine of the ten that have `normalise_expr_before` directly
are statements that an example is not validated today
(`..._not_validated_today_statement`, in
`FlambdaUnboxParamsExamplesSpec.v` and
`FlambdaUnboxVariantsExamplesSpec.v`): their terms have no array,
so the rewrite gives them back; they are computed again by the
build. The tenth is `normalise_expr_before_keeps_scope_statement`.

### 3.5 What it does not do

- The length known where two arms meet: each arm gives one
  continuation an array of the same length, and the term after has
  no test of the index after them. That is known by a join of two
  types, or by a pass of the kind that gives a join what is read
  in scope; it is not this.
- The length of an array that the code does not make: one held in
  a value slot of a closure whose every closure is made with an
  array of that length.
- The tests of an index under the arm of a `Switch` on the length.
- A load taken twice: the second is not shared
  (`FlambdaCseSpec.shareable` says no of `Array_load`).
- A load at a constant index out of the bounds of a known array:
  Simplify makes it `Invalid`; here the `Let` stays, with no
  value.

## 4. The measure

Three tools built from one tree, a5e4605469: its own; A, with 2.2
and 2.6; B, with A and 3.2, 3.3. Each row is the number of units
not validated, at the default flags and at `-O3` where there are
two settings.

- The 135 units of section 1: this tree 58 / 63; A 27 / 34; B
  14 / 19.
- 68 units written to try arrays of constants (not in this tree):
  this tree 46 / 46; A 10 / 11; B 2 / 3.

On both sets every unit whose verdict changes goes from not
validated to validated, or to validated with its imports; no unit
goes the other way, and every other row of the counts is as it
was. The lines of the two runs are the same for every unit. One
unit of the first set, undefined in both runs, goes from "nothing
to validate" to a validation that says nothing of it, and stays
counted as undefined.

The examples of the tree (145 terms and 482 OCaml units), by the
test of the tree run with each tool: with A two expected files
move, `ocaml/iarray_literal` (not validated; validated once
lifted) and `ocaml/iarray_phys_equal` (undefined before Simplify:
"nothing to validate" becomes a validation followed by "the term
before is undefined in the model, so the validation says nothing
of this unit"). With B two more: `ocaml/iarray_exported` (not
validated; validated once lifted) and `static_array.fl`
(validated once lifted; validated once normalised). None goes
from validated to anything else.

The census of generated units: with A and with B, none of the 600
outputs and none of the 240 differs from those of the tool of the
tree.

What is left not validated of the 135 with B, by what the two
terms show, default / `-O3`: a table that Simplify makes of a
`Switch` and another static array with the same elements, which
the terms cannot tell apart, 4 / 4; a test made twice, the second
under an arm of the first, 4 / 4 (two of them with a load taken
twice); the length known in code that does not make the array,
2 / 2; the tests of an index under an arm of a `Switch` on the
length, 1 / 1; a length taken twice, 1 / 1; the length known where
two arms meet, 0 / 5; a recursive function made a loop and inlined
at its call, 0 / 1; a `Let` of a constant whose symbol is bound in
the body of its `Let_cont`, and so not around it, 1 / 1; and a
group of constants that Simplify binds under two `Let`s, in a unit
that mentions an import, so that the comparison with the static
bindings first is not made (`FlambdaHoistStaticsSpec.v` says so
of the pass), 1 / 0. The last two were reproduced with blocks and
no array.

## 5. What is decided here, and what is asked

Decided, as the earlier request decided it: the case and the rule
of 2.2 are those of a block without its tag; the array kind is
dropped.

Asked:

1. Of the orchestrator: the weaker conclusions of
   `value_lifted_array_statement` (its text) and of
   `lift_simulates_statement` (by the rule), with 2.4 as their
   record; and whether section 3 is a series of this stream.
   Ruled: both are permitted. The first on conditions: the record
   of 2.4 with every statement that has the relation; both texts
   of the restated statement (2.3); the old reading kept as a
   proved statement for a unit that makes no array (2.3);
   immutable arrays of values only (2.2); the words of item 2 to
   4, and those of the owners of the form from the imports and of
   the observed forms; and the measure of section 4 made again
   with the tool of the series, with the processor time on the
   census. The second under the ordinary gate of a changed
   definition, written by this stream.
2. Of the owner of the lifting: its word on 2.2, on the comments
   of 2.5, and on the patch to its proofs when it is written.
3. Of the owner of the validation: the two lines of 3.3, and the
   filter of 2.6 in the tool.
4. Of the reviewers: R3 the lifting and the statement of 2.3; R1
   the record of 2.4 and 3.4; R5 that the folds of 3.2 are what
   the primitives compute; R6 the pass of section 3 and its place
   in the round.
