# Names in the model of Flambda 2

How the model treats bound names, what is proved about them, and what a
change to the model owes. Owner: N1. The compiler's side is
`middle_end/flambda2/nominal/`; section 10 says what it would take
for it to obey the same laws.

## Where it stands

As of the tree in which the heap is a component of the state of the
machine, with the passes for loops and the meet of types under
renaming. The sections below say each thing in full.

**One account of the free names.** The free names of a term are one
function for each sort (`FlambdaBindersSpec.v`), which goes into code.
Written with it: the three conditions on binders of well-formedness;
the statements that the names a binder binds may be renamed
(`FlambdaCanonicalFreeNamesSpec.v`); the rules of `alpha_equivalent`;
what a rewrite that makes names owes. An older account, relations
that do not go into code, is still defined. The statements that a
binding not used may be removed are written with it on purpose
(`FlambdaNominalSpec.v`): they ask nothing of the code, a name that
occurs only in code does not stop a binding from being removed, and
the rewrites of the validator that remove bindings are proved from
them. The other places where it is still used are in the last table
here.

**What commutes with renaming.**

| What | Where | By |
|------|-------|----|
| The machine: a step, a run, a behaviour; equivalence and refinement, of states and of expressions | `FlambdaEquivarianceSpec.v` | N1 |
| What a run leaves at an exit: that a state leaves a world, and refinement with the worlds of the two exits related | `FlambdaExitWorldActionsSpec.v` | N1 |
| What may follow a unit: that a continuation stands for an observer, that no observer tells two worlds apart, and the refinement of two terms up to what may follow; the last two of every class of what a continuation may stand for that a renaming keeps | `FlambdaObserversActionsSpec.v`; of the observers that make no array, `FlambdaNoArrayValueObserversSpec.v` | N1; the owner of that file |
| The machine with a stack of exception handlers and of calls | `FlambdaTrapSpec.v` | the owner of exceptions |
| The machine of `FlambdaHeapMachineSpec.v`, which runs the primitives of the heap, and its refinements; the heap is left as it is | `FlambdaHeapEquivarianceSpec.v` | N1 |
| That a term makes no immutable array and that a state is free of arrays (it holds no array value and may come to run nothing that makes one); the refinement of expressions of the machine of `FlambdaHeapMachineSpec.v` restricted to the closed whole source states that are free of arrays | `FlambdaNoArrayValueSpec.v` | the owner of that file |
| The free names; the three conditions on binders; `alpha_equivalent` | `FlambdaAlphaSpec.v` | N1 |
| The comparison `expr_eqb` | `FlambdaCompareAlphaSpec.v` | N1 |
| The rewrites of the normalisation, the passes for loops, the sharing of a handler that is the same as one bound around it, the parameters bound again at the head of their handlers and their twins, the parameters of a loop that every turn passes on taken away, the parameters given one symbol or one constant bound at the head of their handlers, the applications made to go to the arm of the number that they give, the checks of well-formedness (exception handlers and trap actions among them), the check of kinds, the normalisation of the term after and the first phase of that of the term before | `FlambdaValidatorEquivarianceSpec.v` | N3 |
| The normalisation of the term before and that of a term in which something was lifted: up to the names that a round makes. `validate_unit`: where the code of the term before is closed | `FlambdaValidatorEquivarianceSpec.v` | N3 |
| Identities of primitives, lifting, lifting out of code, the binding of `my_closure`, ranges, sharing of primitives and of projections, the recovery of comparisons, folding at known symbols, unused and known value slots, the order of static bindings, the static closures in sight, code moved up, code and static bindings bound first, lifting out of the arms of a test, inlining within a group of code, the unboxing of parameters, of the reads of a loop and of the boxed numbers that continuations pass to one another, the merging of the arms of a `Switch` and their merging into a load from a table, the continuations that a handler binds on its way to its `Switch` bound outside it, the demotion of exception handlers, the rewrite of the calls of code that never returns, the fold of a `Phys_equal` of a simple against itself or of two different symbols, the calls of the unit's other code made indirect in the code of a stub, the lifting to a group around, a `Let` of a variable to the variable just bound made one `Let` with it, the reading of a known immutable array, the sharing of the reads of two places that hold one variable, the inlining of the calls of known siblings in the term after, the copy of code for static sets of closures, the calls of the closures that a `Let` in sight makes made direct, what `my_closure` is known to be from its code used in the bodies of code, the read of a known block where it is sure to have a value and the removal of the unused allocations of constants, in the term after | the `Spec` file of each pass | the owner of each |
| The second asking of the validation, `validate_unit_knowing`: where the code of the term before is closed | `FlambdaValidateKnowingSpec.v` | the owner of that file, N3 |
| The tests that a verdict holds of a unit run from its imports: those that read static names alone, with no premise; those of a verdict, where the code of the term before is closed | `FlambdaValidateImportsRenameSpec.v` | N3 |
| The second check of a unit, `validate_unit_related`: with no premise | `FlambdaValidateRelatedRenameSpec.v` | N3 |
| The validation asked of the term before with its known arrays read, `validate_unit_reading` and `validate_unit_knowing_reading`: where the code of the term before is closed | `FlambdaValidateReadingRenameSpec.v` | N3 |
| Typing environments and their levels, extensions, aliases, types, provers | the types library (`FlambdaEnvRenameSpec.v`, `FlambdaTypeRenameSpec.v`, `FlambdaEnvAliasesRenamingSpec.v`, `FlambdaTypingEnvExtensionSpec.v`, `FlambdaProversRenameSpec.v`) | N3 and the owners |
| The model of Simplify: the function that rewrites a term from the types, the same with pruning, the rewrites of a `Switch`, following aliases with the aliases of the library | the types library (section 7 has the files) | the owners of those files |
| The meet of types and of expanded types, and the functions that add to a typing environment, given a join that commutes with renaming; with the join that forgets, with no premise | the types library (`FlambdaMeetRenameSpec.v`) | the owner of the meet, N3 |
| The join of typing environments and the join that Simplify calls, with the source of new names renamed; with them, the meet of the model with its own join | the types library (`FlambdaEnvRenameJoinSpec.v`, `FlambdaJoinMeetRenameSpec.v`; the traversal of types, as lemmas of `FlambdaEnvRenameJoinTypesProofs.v`) | N3; N1 for the traversal |

The rows down to that of the rewrites of the normalisation are
equations, or equivalences, that hold of every state and of every
term. The row after it is not, nor the second asking of the
validation, nor the tests of a verdict from the imports, nor the
validation with the known arrays read: a round of the term before
makes names. The second check has no premise: it tests itself that the
code of the two units is closed.

**What has a finite support.** Every expression and every state
(`FlambdaStateSupportSpec.v`); a state with a stack (the same file); a
state of the machine of `FlambdaHeapMachineSpec.v`
(`FlambdaHeapEquivarianceSpec.v`). So what each of the three machines
runs is a nominal set, and its execution is an equivariant function.
None of the three is an instance of the interface; section 6 says
why.

**The locations of the heap** are a sort of names of their own. Their
permutations act on values, cells, heaps, environments, continuations,
the static environment and states, and each of these with its
locations obeys the laws of the interface (`FlambdaLocationsSpec.v`);
the action commutes with that of the renamings. The primitives go
through it by an equation, but for the two that bind a location,
which do up to the location chosen
(`FlambdaLocationsPrimitiveSpec.v`); a step, a run and a behaviour of
the machine do in that form (`FlambdaLocationsMachineSpec.v`). Section
4 has the premise.

**A rewrite that makes names** does not commute with renaming. It
owes two things, defined in `FlambdaValidatorEquivarianceSpec.v`:
that it commutes up to the names that it makes (`equivariant_up_to`),
and that it keeps scope (`keeps_scope`). Rewrites that have both
compose, and the comparison answers the same on their results. Two
are in the tree: inlining at fresh names (`FlambdaInlineFreshSpec.v`:
two choices of names give alpha-equivalent terms) and giving every
binder a name of its own (`FlambdaRenameApartSpec.v`: the result on a
renamed term is the result under a renaming that agrees with the
first on the names of the term). The validation applies the second
in each round of the term before. Every rewrite of a round keeps
scope (`FlambdaKeepsScopeSpec.v`, `FlambdaValidateScopeSpec.v`,
`FlambdaLoopsScopeSpec.v`, `FlambdaStaticsScopeSpec.v`; for some
passes a file of the pass, which section 7 names), so the
normalisation of the term before
commutes with renaming up to the names that it makes, and
`validate_unit` answers the same on a pair of terms and on the pair
renamed where the code of the term before is closed. A lifting holds
the two continuations of its unit, which the term may no longer
mention when the lifting is made. So the rounds after a lifting are
given those continuations, make no name that is one of them
(`rename_apart_avoiding`), and are stated up to the names that they
make with one clause more: the second renaming does to the
continuations given what the first does
(`equivariant_up_to_avoiding`).

**Still written in the older style.** `requests/N1-one-free-names.md`,
section 7, has the reason for each and what it would take.

| What | Whose |
|------|-------|
| The three statements that a binding not used may be removed, and three that say that their premise is kept by renaming and by alpha-equivalence: on purpose | N1 |
| The relations `var_free_in_expr` and `cont_free_in_expr`, as definitions: for those six statements, the rows below and the proofs | N1 |
| `FlambdaApplySpec.inlinable` | the integrator |
| The checks `FlambdaNormaliseSpec.expr_in_scope` and `FlambdaWellFormedSpec.free_continuations`, which run | the integrator |
| The tests of freedom `var_free_b` and `cont_free_b` of the rewrites that remove bindings | the integrator |
| The older forms of the statements that rename, as lemmas of `FlambdaNominalProofs.v` and `FlambdaAlphaProofs.v`, for the proofs that use them | N1 |

**Owed.** That the join of typing environments, with one source of
new names, answers the renamed answer up to the variables that it
makes (section 7); the functions on types that have no statement
yet; the rows of the table above; and, when a step of the machine
comes to allocate, the proofs about a step under a permutation of
locations again (section 4).

## 1. The names

Two sorts of name are bound by terms and moved by renamings: variables
and continuations. Symbols, code ids, value slots and function slots
are global: no renaming moves one. A run-time value holds no variable
and no continuation (a closure holds code ids and values), so nothing
is renamed in a value. If a new place comes to hold names, it is a
sort of its own. The locations of the heap (`FlambdaHeapSpec.v`) are
one: no renaming of variables or of continuations moves a location,
and the permutations of locations (`FlambdaLocationsSpec.v`) move
nothing in a term, so the two actions commute
(`rename_permute_state_statement`).

Names are compared by equality only. The one function that looks at
the number of a name is `fresh_name` (`FlambdaNameAbstractionSpec.v`),
which gives a name outside a list of names. `fresh_location`
(`FlambdaHeapSpec.v`) is that function on the domain of a heap, written
out there because the primitives of the heap come before the semantics
(`FlambdaHeapMachineSpec.fresh_location_fresh_name_statement`). The
machine of regions is apart from this: it draws its symbols from a
counter and tells a symbol that it drew from any other by its number
(section 9).

## 2. Where things are

| File | What it has |
|------|-------------|
| `FlambdaBindersSpec.v` | What each binder binds and where; the free variables and free continuations of a term, as one function for each (they go into code); the three conditions on binders: `names_in_scope`, `binders_distinct`, `code_closed` |
| `FlambdaScopeSpec.v` | `state_in_scope`: what a run keeps of scope; a state in scope reads no variable without a value and leaves only by a continuation it was given |
| `FlambdaNominalSpec.v` | Permutations and renamings; the action on terms, `rename_expr`, code included; that the binders of code may be renamed; the older relations of free names, and with them that a binding which is not used may be removed |
| `FlambdaAlphaSpec.v` | The action is that of a group; the free names and the three conditions are equivariant; `alpha_equivalent`, by its generators, is equivariant, keeps the free names and the three conditions, and is sound for the semantics |
| `FlambdaCompareAlphaSpec.v` | The comparison `expr_eqb` is an equivalence relation, contains `alpha_equivalent` and does not depend on the names; two pairs of terms that it accepts outside well-formed terms |
| `FlambdaEquivarianceSpec.v` | The action on states; a step, a run and a behaviour commute with it; two rewrites commute with it |
| `FlambdaNameAbstractionSpec.v` | The interface of `nominal/`: things that contain names, bindable names, name abstractions, `pattern_match`, with their laws; `fresh_name` |
| `FlambdaNominalInstancesSpec.v` | The terms as instances of the interface; each binder as a name abstraction |
| `FlambdaFreeNamesSpec.v` | The older accounts of free names agree with the one function where code is closed, and what is free by them is among the free names on every term |
| `FlambdaCanonicalFreeNamesSpec.v` | The names that a `Let`, a `Let_cont` or a handler binds may be renamed; corollaries, with the one function, of the statements about a binding that is not used |
| `FlambdaStateSupportSpec.v` | Every expression and every state has a finite support: with the action, the states are a nominal set |
| `FlambdaHeapEquivarianceSpec.v` | The action on the states of the machine with a heap, which leaves the heap alone; a step, a run and a behaviour of that machine commute with it; its refinements are preserved by renaming; its states have a finite support |
| `FlambdaLocationsSpec.v` | The action of the permutations of locations on what the machine holds; each thing that holds locations obeys the laws of the interface, with its locations for its names; the action commutes with that of the renamings; lookups go through it |
| `FlambdaLocationsPrimitiveSpec.v` | A primitive under a permutation of locations: an equation where it binds no location; every primitive up to the location chosen, where the heap binds the locations of all its entries; that premise is kept; a heap with a hidden entry where it fails |
| `FlambdaLocationsMachineSpec.v` | A step, a run and a behaviour under a permutation of locations, up to the locations allocated, for a state whose heap binds every location of the state; a step, a run and the action keep that; the same of what a `Let` is to run when a step allocates |
| `FlambdaExitWorldActionsSpec.v` | The two actions on what a run leaves at an exit: a state leaves a world exactly when the state renamed leaves the world renamed; refinement with the worlds related, the relation renamed; under a permutation of locations, the world that a state leaves, up to the locations allocated |
| `FlambdaObserversActionsSpec.v` | The two actions on what may follow a unit: under a renaming, a continuation stands for an observer, no observer tells two worlds apart, and two terms are refined up to what may follow, exactly when the same holds of them renamed; under a renaming also, the two relations of a class of what a continuation may stand for, for every class that a renaming keeps; under one permutation of locations applied to both sides, what the way out compares of two worlds, and nothing of the relations of a class |
| `FlambdaFreshQuantifierSpec.v`, `FlambdaFreshAbstractionSpec.v` | The quantifier "for all but finitely many names"; `fresh_names`; equal abstractions of one name and of a list of distinct names, said with the quantifier |
| `FlambdaFreshBindersSpec.v`, `FlambdaAlphaFreshSpec.v`, `FlambdaAlphaGroupSpec.v` | For each binder, the rule that compares two binders at fresh names, and that it can be read both ways on terms with distinct binders and closed code; the recursive group as a name abstraction |
| `FlambdaCompareConverseSpec.v` | What the comparison accepts is alpha-equivalent, on terms with distinct binders and closed code |
| `FlambdaInlineFreshSpec.v` | Inlining a call with the binders of the code exchanged for fresh names; two choices give alpha-equivalent terms |
| `FlambdaValidatorEquivarianceSpec.v` | Each rewrite and each check of the validator commutes with renaming: the rewrites of the normalisation, the checks of well-formedness and of kinds; what a rewrite that makes names owes instead, and with it the normalisation of the term before and `validate_unit` |

The four rows before the last are the second prover's files, and the
last is the third prover's.

In the types library, `FlambdaEnvAliasesRenamingSpec.v` has the action
on the aliases, and `FlambdaTypeRenameSpec.v` and
`FlambdaTypeNominalSpec.v` the action on types and on environment
extensions, each with the same laws, as instances of the interface.
`FlambdaEnvRenameSpec.v` has the action on typing environments and
their levels, with the same laws, and says that the functions of a
typing environment commute with it. `FlambdaMeetRenameSpec.v` has the
action on the environment of the meet, and says that the meet
commutes with it when the join that it is given does.

## 3. The binders

| Binder | Binds | In scope in |
|--------|-------|-------------|
| `Let x kind d body` | `x` | `body` |
| `Handler params metadata handler` | `params` | `handler` |
| `Let_cont_non_recursive k h body` | `k` | `body` |
| `Let_cont_recursive handlers body` | the continuations of `handlers` | `body` and every handler |
| `Code params my_closure ret exn metadata body` | `params`, `my_closure`, `ret`, `exn` | `body` |

A `Let_static` binds no variable and no continuation. Code is closed:
its body has no free name but what the code binds (decision D31).
Kinds, arities and metadata have no names in them: a renaming leaves
them as they are, and two terms that are the same up to bound names
have the same ones.

Two places name without binding. A trap action of an `Apply_cont`
names the continuation of an exception handler, and the exception
continuation of a call may come with extra arguments, which are
simples. The continuation and the simples are renamed, and are among
the free names.

A well-formed unit meets three conditions, which are checks:

- `names_in_scope`: no free variable, and no free continuation but the
  unit's return and exception continuations;
- `binders_distinct`: the names of one binder are distinct, and a
  recursive group has a handler;
- `code_closed`: all code, at any depth, is closed.

The last two ask more than the compiler checks (decision D32;
discrepancies N1-3 and N1-4). Each of the three is unchanged by a
renaming.

## 4. Renaming

A renaming is a permutation of the variables and one of the
continuations, each a list of swaps. It acts on everything that holds
names: terms, code, static constants, environments, continuation
closures (and a continuation kept with the extra arguments of a call:
the arguments are values, the continuation is renamed), the code of
the static environment, states, the result of a step and behaviours.
It renames bound and free occurrences alike, so it never captures. The
laws of a group hold of the action.

It is the action of the compiler's renamings that are permutations
within one unit (what the freshening of a binder uses). A renaming with
an import map is another operation and is not modelled (discrepancy
N1-8).

Execution commutes with renaming, with no condition:
`step (rename_state r s) = rename_result r (step s)`, and the same of
`run` and of `behaves`. So a rewrite that is correct with one choice of
names is correct with every other. The action leaves the heap of a
state as it is, since a cell holds values.

The same holds of the machine of `FlambdaHeapMachineSpec.v`, which runs the
primitives of the heap (`heap_step`, `heap_run`, `heap_behaves`, and
its refinements `heap_refines` and `heap_expr_refines`).

An immutable array may be a value that a primitive makes as the term
runs (`Value_immutable_array`, made by `Make_immutable_array`). It
holds values, so no variable and no continuation, and a permutation of
locations moves the locations of its elements
(`FlambdaLocationsSpec.permute_value`). `FlambdaNoArrayValueSpec.v` is
about the states that are free of arrays: a state that holds no array
value (in its variables, in what its continuations keep, in what its
symbols are bound to, in its heap) and may come to run nothing that
makes one (its expression, the handlers that its continuations keep,
its code). No one statement of it says that what was proved before the
array value says what it said. It has laws of those states and of the
values that hold no array value: a step keeps a state free of arrays;
the observation that reads an array answers of a value that is not an
array value as it did; a derivation of the relation of the machine
with a heap between values that hold no array value, under memories
that hold none, stays among such values; and a value that holds none
is lifted to values that hold none. With the clauses of the
definitions whose text did not change, which are read and not stated,
those give the statements that have one of the two relations in them
the reading that they had (`status/P1.md` has the reading, statement
by statement). Under renaming it has four statements. Whether the
primitive is somewhere in a term is the same of the term renamed
(`makes_array_renamed_statement`). A state is free of arrays exactly
when the state renamed is, for the machine and for the machine with a
heap (`state_free_of_arrays_renamed_statement`,
`heap_state_free_of_arrays_renamed_statement`). And the refinement of
expressions of the machine with a heap, restricted to the closed whole
source states that are free of arrays, is kept by renaming
(`heap_expr_refines_free_of_arrays_equivariant_statement`); the
statement of that refinement for all states does not give it, as its
premise is about more of them.

Under a permutation of locations the statement has another form. A
step that allocates chooses a location that the heap does not bind
(`FlambdaHeapSpec.fresh_location`), and no choice of a name commutes
with a permutation. So what a state moved by `p` steps to is what the
state steps to, moved by some `p'` that agrees with `p` on the
locations that the heap of the state binds; the same of a run and of
a behaviour (`FlambdaLocationsMachineSpec.v`). The premise is that the
heap of the state binds every location of the state, those of an
entry of the heap that a later write hides among them: a location
that is in the state and is not bound may be the one that an
allocation chooses, and `hidden_entry_statement`
(`FlambdaLocationsPrimitiveSpec.v`) is a heap where a hidden entry
makes the difference. A step keeps the premise. No step allocates
yet: a `Let` of a primitive is `eval_prim`, which commutes with the
action by an equation, as does every primitive of the heap but the
two that bind a location; so of the machine as it is `p'` is `p`. The
statements are in the form that stays true when a step runs the
primitives of the heap.

What a run leaves at an exit (`FlambdaExitWorldSpec.exit_world`: the
continuation, the values, the heap, and the static environment of the
last state) has the two actions too (`FlambdaExitWorldActionsSpec.v`).
A renaming renames the continuation and the code of the static
environment. A state leaves a world exactly when the state renamed
leaves the world renamed, and refinement with the worlds related
(`refines_world`) is kept with the relation renamed as the two states
are; so it is kept at the relation itself where the renaming keeps
the relation, as every renaming keeps `same_exit`. A permutation of
locations moves the values, the static environment and the heap. A
state whose heap binds every location of the state leaves a world
whose heap binds every location of the world, and the state moved by
`p` leaves the world moved by some `p'` that agrees with `p` on the
locations that the heap of the state binds. Nothing is stated of
`refines_world` under a permutation of locations: `same_exit`
compares values and heaps by equality, and one permutation applied to
both worlds keeps it, where two independent permutations need not.

What may follow a unit has the two actions as well
(`FlambdaObserversActionsSpec.v`, on the definitions of
`FlambdaObserversSpec.v`: `observer` and `observers`, what the
continuation of an exit may stand for; `observed_refines`, which
compares two results of a step, an exit by its continuation, by the
constants and the locations among its values and by the locations
that its heap binds; `worlds_observed`, that no observer tells two
worlds apart; `refines_under_observers`, the form in which a pass
proves it of two terms). A renaming renames jointly the continuation
of an exit, what a continuation stands for, a continuation
environment, the code of a static environment and the two terms.
Each of the five holds exactly when it holds of what is renamed, with
no premise: the tests of an observer read static bindings, symbols,
code ids and value slots, none of which a renaming moves, and an exit
is compared by its continuation, renamed on both sides, and by values
and a heap, which hold no variable and no continuation. So every
renaming keeps `worlds_observed`, and the comparison of two units up
to what may follow is that of the two renamed.

A class is a predicate on continuation values, and it is the whole of
what may follow a unit, not a condition added to `observer`:
`worlds_observed_by O` relates two worlds that exit by the same
continuation when everything in `O`, applied to the values of each
under its static environment and heap, comes to results that
`observed_refines` relates, and `refines_observed_by O` relates two
units when they come to such results from every continuation
environment each entry of which is in `O`, the hidden ones included;
at `observer slots` the two are `worlds_observed slots` and
`refines_under_observers slots`, by unfolding
(`FlambdaObserversSpec.v`).

For every class that a renaming keeps, in that what is in the class is
in it renamed, the two relations, and the comparison of two states by
the first (`refines_world`), hold exactly when they hold of what is
renamed (`worlds_observed_by_equivariant_statement`,
`refines_world_observed_by_equivariant_statement`,
`refines_observed_by_equivariant_statement`). The class is under an
outer universal and nothing else is asked of it: no observer is named.
The three statements above that have a universal over the observers on
each side (of `worlds_observed`, of the comparison of two states by
it, and of `refines_under_observers`) are these at the class
`observer slots`, which a renaming keeps
(`observer_equivariant_statement`). A term that the language gains is
a term that an observer may be written with, so `observer slots` has
more in it after such a change, and the three statements at that class
are then about more observers on each side: they are not the
statements that they were. The three of a class are about a predicate
that is given. Of a class of the continuation values that there were
they read as they did on inputs that are wholly of the tree before the
change: the worlds, the states, the static environment and the two
terms, with their code and with what they hold. The same class may be
asked of worlds and of units that have the new term, and of those
nothing is as it was by this.

The observers that make no array are such a class
(`FlambdaNoArrayValueObserversSpec.observer_free_of_arrays`: an
observer in which the primitive that makes an immutable array does not
occur; a renaming brings the primitive neither in nor out), and the
three are stated of it in that file
(`worlds_observed_free_of_arrays_equivariant_statement`,
`refines_world_observed_free_of_arrays_equivariant_statement`,
`refines_observed_free_of_arrays_equivariant_statement`).

Under a permutation of locations four things are stated, each of one
permutation applied to both sides: a value is a constant or a location
exactly when the value moved is; two lists of values have the same
constants and the same locations at the same places exactly when the
two lists moved have; the locations that a heap moved binds are those
that the heap binds, moved, in their order and as many times; and what
the way out compares of two worlds is the same of the two exactly when
it is the same of the two moved.
Nothing is stated of `observed_refines` nor of `worlds_observed`
under a permutation of locations. What is proved of a run gives each
of the two runs its exit up to a second permutation of its own, the
comparison is by equality, and a handler that is an observer may be
written with a primitive that allocates, though no step of the
machine of `FlambdaSemanticsSpec.v` allocates yet. Whether the two
hold is not said, either way.

## 5. The same up to bound names

There are two accounts, and they are related.

- `alpha_equivalent`: the least equivalence, closed under contexts,
  that relates a binder with its scope to its image under a renaming
  that fixes the free names of the two. Each rule asks that the code in
  what is renamed is closed; the rule for the binders of code asks
  nothing else. It is sound: alpha-equivalent terms behave alike under
  all environments. Alpha-equivalent terms have the same free names,
  and satisfy the same of the three conditions of section 3
  (`alpha_equivalent_code_closed_statement` and the statements beside
  it), so the conditions are conditions on the classes.
- `expr_eqb`: the comparison that the validator runs, by recursion on
  the two terms with a map of paired names. It is sound, it is an
  equivalence relation, it accepts every two alpha-equivalent terms,
  and its answer does not change when the two terms and the names
  paired so far are renamed (`expr_eqb_equivariant_statement`).

The comparison is the primary notion (it is a function and it is what
runs); the rules of `alpha_equivalent` are facts about it. The converse
(what the comparison accepts is alpha-equivalent) is for terms with
distinct binders and closed code, which are the terms that
well-formedness admits. It is proved
(`FlambdaCompareConverseSpec.expr_eqb_iff_alpha_equivalent_statement`):
on those terms the comparison and alpha-equivalence are one relation,
and the comparison decides it.

At each binder alpha-equivalence can also be said as the papers on
nominal sets say it: two binders are alpha-equivalent exactly when
two things hold. What is outside the scope agrees: the two `Let`s
have the same kind and the same defining expression, the two handlers
or the two pieces of code the same metadata, and the handlers of the
two non-recursive `Let_cont`s are themselves alpha-equivalent (the
handlers of a recursive group are in the scope of its continuations,
and are compared with them exchanged). And the scopes are
alpha-equivalent once the bound names of both are exchanged for the
same fresh names, for all but finitely many choices of them
(`FlambdaCompareConverseSpec.alpha_let_iff_statement`, and the
statements of `FlambdaAlphaFreshSpec.v` and `FlambdaAlphaGroupSpec.v`
for a `Let_cont`, a handler, code, a `Let_static` and a recursive
group). That is how the compiler compares two binders
(`Name_abstraction.pattern_match_pair`).

Outside those terms the comparison accepts more, and is still sound.
Two pairs of terms are accepted and proved not alpha-equivalent:

- `expr_eqb_repeated_parameter_statement`: a handler that repeats a
  parameter, against one that does not. The first has not distinct
  binders.
- `expr_eqb_open_code_statement`: a variable bound around code that
  mentions it, against the same term with the binder, and its uses
  outside the code, renamed. Both have distinct binders; their code is
  not closed, so no rule renames the binder.

Every statement that a bound name may be renamed has the premise that
the code in what is renamed is closed. The comparison, from which
they are proved, treats code as closed. Where code is not closed the
comparison and the rules part: `expr_eqb_open_code_statement` is a
pair that the comparison accepts and that is not alpha-equivalent. No
more than that is shown: the statements about behaviour are not
proved of terms whose code is not closed, and no such term is known
to refute them. Well-formedness admits none.

## 6. The interface

`FlambdaNameAbstractionSpec.v` is the interface of `nominal/` with the
laws the compiler leaves unsaid, for one sort of name at a time: a
type whose values contain names has an action, free names and an
equality (`=` where there is no binder, alpha-equivalence for
expressions); a name abstraction is a pair of bound names and a scope,
equal to another when each can be opened, by a permutation that fixes
its free names, to the same bound names and equal scopes.

Instances: `simple`, `named`, static constants and environments;
expressions whose code is closed, for each sort; and, in the types
library, the aliases, the types and the environment extensions. Each
binder of section 3 is shown to be a name abstraction: equal
abstractions give alpha-equivalent terms
(`FlambdaNominalInstancesSpec.v`; for the
recursive group, whose scope is its handlers and its body together,
`FlambdaAlphaGroupSpec.group_abstraction_statement`). The laws of
abstractions, and that opening two equal abstractions at the same
names gives equal scopes (`pattern_match_pair_statement`), then hold
of them with no further proof.

The quantifier "for all but finitely many names" is a derived view of
equal abstractions (`FlambdaFreshQuantifierSpec.v`,
`FlambdaFreshAbstractionSpec.v`), for one bound name and for a list of
distinct bound names. The definition by permutations stays. Section 5
says what the quantifier gives at the binders of terms.

States. The renamings act on the states of the machine
(`FlambdaEquivarianceSpec.v`: the identity, composition, inverses), a
step, a run and a behaviour commute with the action, and every state
has a finite support (`FlambdaStateSupportSpec.v`): so the states are
a nominal set and execution is equivariant between nominal sets. A
state is not an instance of the interface. The interface is for one
sort of name at a time and the machine is renamed in both; and its
`free_names` go with an equality under which a permutation that fixes
them changes nothing. For a state as it is written that equality is
`=`, and the names are then all that occur, bound ones included,
which is the support of `FlambdaStateSupportSpec.v` and not the free
names of anything. For a state up to bound names, the equality would
have to be defined through every handler held by a continuation and
all the code of the static environment, and the laws would hold only
of states whose code is closed, as for expressions. So the
equivariance of execution stays stated directly, for both sorts at
once.

Locations. Each thing that holds locations (a value, a cell, a heap,
an environment, a continuation, the static environment, a state) is
an instance of the interface for that one sort, with `=` for its
equality and all its locations for its names
(`FlambdaLocationsSpec.v`). Nothing binds a location, so the names of
a thing are its support, and there is no equality up to bound names
to define.

## 7. What is owed for each function on terms

A function on terms that models something the compiler does owes that
it commutes with renaming, in one of these forms:

- a transformation: `f (rename_expr r e) = rename_expr r (f e)`;
- one that takes names, or a map keyed by names: the same with that
  argument renamed too;
- a check: `c (rename_expr r e) = c e`, with any names it carries
  renamed;
- one that makes a name: it takes the names to avoid and uses
  `fresh_name`, and its statement relates two choices by the
  permutation that swaps them.

**Why a function that makes names cannot commute with every
renaming.** A source of new names is a function from the names to
avoid to a name outside them; `fresh_name` is one. No such function
commutes with every renaming. Take a list `l`, the name `n` that the
source gives for `l`, and a name `y` that is not in `l` and is not
`n`. The renaming that swaps `n` and `y` moves no name of `l`, so the
source gives `n` for the renamed list; and `n` renamed is `y`. The
same goes for a function that asks the source for a name and lets
that name show in its answer: swap the name with one that is neither
in the arguments nor in the answer; the arguments are as they were
and the answer is not. So such a function does not commute with every
renaming, whatever the source is. This is not a defect of
`fresh_name`, and another choice would not help: that no function
which respects renaming can pick a name out is what it is for names
to be compared by equality only.

Of this, what is a theorem is the first part, of the source alone
(`no_source_commutes_statement`, in
`types/rocq/FlambdaEnvRenameJoinSpec.v`). That the join of typing
environments with one source does not commute with every renaming is
the argument above applied to it. It is proved of one join of the
model, by computation
(`types/rocq/FlambdaJoinMeetRenameExampleSpec.v`): the join makes a
variable, and the renaming that exchanges it with a variable which is
nowhere in the inputs leaves the inputs as they are and changes the
answer. It is not proved of every source.

**What the caller of such a function owes.** `fresh_name` gives a
name outside the names that it is given, and no more. A name that a
function makes and leaves in its answer is a binder of that answer.
Two answers that were made apart may each hold a made name that is
spelt as the other's is, and whatever then reads the two side by side
takes the two for one name. The compiler does not meet this: its
names are distinct on every path. The model met it in the tests of
types, at joins of environments that joins had made. So the names to
avoid are every name that the answer may later stand beside, not only
the names of the arguments.

For the join of typing environments the premise is `avoids_known`
(`types/rocq/FlambdaJoinSpec.v`): the variables to avoid include
every variable that the source or a use knows. It is enough where
every use knows exactly the variables of the source
(`uses_know_the_source`, `types/rocq/FlambdaJoinInTermsSpec.v`): no
variable is then known to two uses without being one variable of the
source. Uses that are made from a source in which a meet may be
made, by going one scope further and then by steps that make a
variable equal to a `Simple` or give it a type (`made_from`, which
has the conditions of each step), are such, and with variables to
avoid that include those of the source they have the premise
(`uses_made_from_source_statement`,
`types/rocq/FlambdaJoinUsesSpec.v`): no step defines a variable. Of
inputs in which a use knows a variable that the source
does not, the part about valuations is not stated: the join takes a
variable that two uses know for one variable, and a caller that gives
it such inputs owes that it is one. The same is owed for the join
that the meet calls, by whoever shows of the model's meet what the
statements suppose of a meet.

**What such a function owes instead.** One of two things.

- *With the source renamed.* The source is an argument of the
  function, and a renaming acts on it as on any function, by
  conjugation: the renamed source, given a list, undoes the renaming
  on the list, asks the source, and renames what it answers. That is
  a source of new names again. The function then owes an equation of
  the second form above: with the renamed source, on the renamed
  arguments, it gives the renamed answer. Nothing is up to anything,
  nothing is asked of scopes or of code, the equation composes with
  the others of this section, and its proof is the recursion of a
  function that makes no name with one more case, the call of the
  source, which is the definition of the conjugate.
- *With one source on both sides.* The answer for the renamed
  arguments is the answer under a second renaming, which agrees with
  the first on the names that the function was given and may differ
  on the names made. It follows from the first and one more fact,
  that the choice does not matter: two sources give answers that
  differ by a renaming which fixes the names given.

Where the source can be an argument, state the first: a caller that
only has to push a renaming through the function needs no more, and
the second is derived from it.

For a rewrite of terms, the names made are bound in the result, and
the second says that the two results are the same up to bound names.
It is in `FlambdaValidatorEquivarianceSpec.v`, with what makes it
compose:

- `equivariant_up_to`: on a term whose code is closed, the result on
  the renamed term is the result under a second renaming, which
  renames the free names of the result as the first does. The two may
  differ on the names bound in the result;
- `keeps_scope`: on a term whose code is closed, the code of the
  result is closed and no name is free in the result that was not
  free in the term.

A rewrite that is not known to keep scope is made to by a test
(`FlambdaGuardScopeSpec.guard_scope`): it keeps scope whatever the
rewrite is, it is the rewrite where the rewrite keeps scope, and it
commutes with renaming where the rewrite does
(`FlambdaValidateScopeSpec.v`). The normalisation of the term before
applies so, once, between its rounds, the function of the model of
Simplify that rewrites a term from the types (below); that of a term
in which something was lifted applies it so too, once, between its
rounds and the same rounds again. What the two call is
`simplify_by_types_carried`
(`types/rocq/FlambdaSimplifyCarriedSpec.v`), which carries what is
known of the typing environment in place of a test of it at every
`Let`, and which is that function, of every term
(`simplify_by_types_carried_eq_statement`): so what is stated of
that function under renaming holds of what is called.
Where that function rewrites the handler of a continuation under what
is known at its one application, the application may be in the handler
of another continuation of the body (`find_one_use`,
`types/rocq/FlambdaSimplifyPassSpec.v`): it makes no name, it compares
names only to keep the rewritten handler where no binder passed on the
way to the application is free in it, and the environment and the
binders that it answers are renamed with the term
(`find_one_use_equivariant_statement`). With pruning, the test
`find_one_use_keeps_fit` answers the same on renamed arguments
(`find_one_use_keeps_fit_equivariant_statement`,
`types/rocq/FlambdaSimplifyPruningRenamingSpec.v`).

Rewrites that have both compose and iterate, a rewrite that commutes
with renaming has the first, and the comparison answers the same on
the results (`expr_eqb_up_to_statement`).

The rounds of a rewrite stop after the first that gives back the
term that it was given, up to bound names
(`FlambdaValidateSpec.iterate`, which asks the comparison). The four
statements of a rewrite applied in rounds have the texts that they
had, for every number of rounds (`iterate_equivariant_statement`,
`keeps_scope_iterate_statement`, `equivariant_up_to_iterate_statement`,
`equivariant_up_to_avoiding_iterate_statement`). Under the premises
of the three that are about renaming, as many rounds are made of the
term renamed. Where the rewrite commutes with renaming, the
comparison answers of the renamed term and its result what it
answers of the term and its result. Where it keeps scope and
commutes up to the names that it makes, of a term whose code is
closed, the result for the renamed term is the result under a second
renaming that does to the free names of the result what the first
does, and the comparison does not tell two such renamings apart
where the code of the result is closed
(`expr_eqb_renamed_alike_statement`; the rewrite keeps scope, so it
is). No statement says that what a rewrite of the validation gives
depends on its term up to bound names only; so none says that a
round which gave back its term would do so again.

A caller may hold names that the term does not mention, and need the
second renaming to do to them what the first does: `equivariant_up_to`
says nothing of a name that is not free in the result. Such a rewrite
is given those names to avoid, and owes `equivariant_up_to_avoiding`
(the same file, for continuations): the names given are renamed with
the term, the second renaming renames the free names of the result as
the first does, and it does to the names given what the first does.
It is kept by a rewrite that commutes with renaming and keeps scope,
applied before or after, and by the rewrite applied a number of times
with the same names; with no name given it is `equivariant_up_to`.

A rewrite, then a second one, and then a third only where the
comparison finds that the second changed the term, keeps scope and
commutes with renaming up to the names made where the three do
(`keeps_scope_unless_same_statement`,
`equivariant_up_to_unless_same_statement`,
`equivariant_up_to_avoiding_unless_same_statement`; in the last the
first and the third are given the same continuations to avoid, and
the second commutes with renaming by an equation): the comparison
answers of the two results for the renamed term what it answers of
them for the term, so the same branch is taken.

Where the names made stay free in the answer, the second is about
answers up to a renaming of the names made, and says less. The join
of typing environments is such a function: it defines the variables
that it makes in the environment that it answers. It owes the first,
and has it, by the third prover
(`types/rocq/FlambdaEnvRenameJoinSpec.v`): what a renaming does to
the source of new names and to the state of the join; that no source
commutes with every renaming; the equation for the two functions that
ask the source; and the equation of the join as a whole, for two sets
of parameters that the renaming relates, as the model's parameters
and the same with the source renamed are. With it
(`types/rocq/FlambdaJoinMeetRenameSpec.v`) the meet of the model with
its own join, and the join that Simplify calls, answer the renamed
answer with the renamed source. The traversal of types that the join
calls (`types/rocq/FlambdaJoinTypes.v`) makes no name and compares no
two variables itself: the variables that it finds in types it hands
to the functions that it is given. Each of its functions gives the
renamed answer on renamed arguments when those do (lemmas of
`types/rocq/FlambdaEnvRenameJoinTypesProofs.v`). The second is not
stated for the join: it needs that two sources give answers that
differ by a renaming of the variables made. Nothing calls it today.
Its caller would be a pass that calls the join on renamed inputs with
the model's one source, as a call of the join in the pass that
rewrites a term from the types would; that call is not made. The
fact that is missing is a simulation of two runs of the whole join
under a renaming that grows at each call of the source, with the
names to avoid containing every name of the state. The third
prover's estimate is a few thousand lines of proof. It stays owed.

Prove it by one recursion over expressions, handlers, bindings and
code. A function whose answer depends on the order or the number of a
name is a finding: record a discrepancy.

Proved: the machine (`step`, `run`, `behaves`); the free names, as
relations and as functions; `code_closed`, `binders_distinct`,
`names_in_scope`; `alpha_equivalent`; `expr_eqb`; `fold_constants`;
`inline_calls` (a direct call with more arguments than the code has
parameters is split where the test `over_applied_b` passes, the
continuation and its parameter taking the names of the return
continuation and of the closure variable of the code; no name is
made); and, in the fourth form, inlining at fresh names
(`FlambdaInlineFreshSpec.v`: the names made are bound in the result, so
two choices give alpha-equivalent terms).

Proved by the third prover, in `FlambdaValidatorEquivarianceSpec.v`:
the other rewrites of `FlambdaNormaliseSpec.v`,
`FlambdaUnusedStaticsSpec.v`, `FlambdaUnusedContsSpec.v`,
`FlambdaBlockLoadSpec.v` and `FlambdaUnboxSpec.v`; the checks of
`FlambdaWellFormedSpec.v`, those of exception handlers and of trap
actions among them; the check of kinds; the passes for loops
(`FlambdaRecHandlersSpec.v`, `FlambdaSinkContsSpec.v`,
`FlambdaSinkUsesSpec.v`, `FlambdaShortcutSpec.v`, `FlambdaUnusedParamsSpec.v`,
`FlambdaLoopAliasesSpec.v`, `FlambdaLoopifySpec.v`, those of the
second series, `FlambdaParamAliasesSpec.v`, `FlambdaForwardSpec.v`,
`FlambdaInlineTailSpec.v`, `FlambdaLoopLetsSpec.v`, and that of the
third, `FlambdaDeadParamsSpec.v`); the forwarding of an application
of a continuation whose handler only applies another, where it is an
equivalence: with no parameters (`FlambdaForwardJumpsSpec.v`), and
with arguments known to have the kinds of the parameters, which the
validation applies to the term after, in the bodies of code too
(`FlambdaForwardKindedSpec.v`; an application keeps its own trap
action there too, where the handler's own application has none, and
with it the continuation that the action names;
an arm of a `Switch` that applies, to no arguments and with no trap
action, a continuation that has no parameters and gives no kinds, and
whose handler, once the pass has been through it, is one application,
takes that application with its trap action (`forward_kinded_arm`, as
`FlambdaForwardSpec.forward_arm` for the term before): the
continuation that the action names is one that the body of the
`Let_cont` does not bind (`FlambdaForwardSpec.forwarding`), so it
means at the arm what it means in the handler;
`forward_kinded_equivariant_statement` and
`forward_kinded_in_code_equivariant_statement` have the texts that
they had);
an application that gives a number to the parameter that the handler
of its continuation tests, made to go where the arm of that number
goes (`FlambdaForwardArmSpec.v`: with nothing asked of the kinds,
`forward_to_arms`; where the arguments are known to have the kinds of
the parameters, `forward_to_arms_kinded`, and that in the bodies of
code, `forward_to_arms_in_code`; the pass compares names by equality
only, binds none and makes none: the application that it puts is that
of the arm with the arguments put for the parameters of the handler,
where the tests of `FlambdaForwardSpec.forwarding` pass of the arm, as
of a handler that would be that arm alone;
`forward_to_arms_equivariant_statement`,
`forward_to_arms_kinded_equivariant_statement` and
`forward_to_arms_in_code_equivariant_statement`, each an equation that
holds of every term; that the first keeps scope,
`FlambdaLoopsScopeSpec.forward_to_arms_keeps_scope_statement`; the
other two have no statement that they keep scope, and are applied to
the term after only, where no round makes a name); the binding, at the
head of its handler, of a parameter that is given one symbol or one
constant (`FlambdaBindConstParamsSpec.bind_const_params`: a parameter
of a non-recursive continuation, not an exception handler, that every
application gives one symbol or one constant is taken away and bound
at the head of the handler, by a `Let` of the parameter's own name; it
compares names by equality only and makes none;
`bind_const_params_equivariant_statement`, an equation with no
premise; that it keeps scope,
`FlambdaStaticsScopeSpec.bind_const_params_keeps_scope_statement`,
scope not counting the symbol that the `Let` may mention; the
normalisation of the term after applies it, just after the
applications are made to go to their arms); the
removal, in the body of code, of the unused `Let`s of its parameters
(`FlambdaCodeLetsSpec.v`); the removal of a parameter that its
handler does not use, of a continuation that a call returns to, the
call returning to a wrapper (`FlambdaWrapReturnsSpec.v`: the wrapper
is bound to the name of the continuation and its parameters have the
names of those of the handler, so no name is made; that it keeps
scope, `FlambdaStaticsScopeSpec.wrap_returns_keeps_scope_statement`);
the untagging of the parameters of continuations that hold a tagged
immediate and are only untagged or passed on, in all the
continuations that pass them at once (`FlambdaUntagParamsSpec.v`: a
parameter keeps its name and no name is made; an argument that a
`Let` in scope binds to the tagging of a variable becomes that
variable, which is in scope where the argument is;
`untag_params_equivariant_statement`; that it keeps scope,
`FlambdaStaticsScopeSpec.untag_params_keeps_scope_statement`); the
forwarding of applications with the calls taken on or not
(`FlambdaForwardSpec.forward_applications_with`, of which the pass
of the rounds is the case without the calls; an application that is
made to go where the handler of its continuation goes keeps its own
trap action where the handler's own application has none,
`forward_apply_cont_keeping`, and with it the continuation that the
action names:
`forward_applications_with_equivariant_statement`; that it keeps
scope,
`FlambdaLoopsScopeSpec.forward_applications_with_keeps_scope_statement`);
the sharing of a handler (`FlambdaSameHandlersSpec.share_handlers`: a
non-recursive continuation whose handler is the same, up to bound
names, as that of a continuation bound around it is made to apply that
one to its own parameters; the test is the comparison,
`FlambdaCompareSpec.handler_eqb` with no names paired, and it asks
that no name free in the handler is a variable or a continuation bound
since the outer one, or the outer one itself, so that each means there
what it means where the outer one is bound, and that its parameters
are distinct; it makes no name;
`share_handlers_equivariant_statement`, an equation that holds of
every term, the comparison answering the same of two handlers renamed;
that it keeps scope,
`FlambdaLoopsScopeSpec.share_handlers_keeps_scope_statement`; it is
applied to the term before only);
the
test for inlining a call that gives tupled code a tuple; the demotion
of exception handlers with the rounds that tidy after it
(`demote_and_tidy`); the normalisation of the term after and the
first phase of that of the term before, a round of which is the
round of the parameters followed by a step
(`FlambdaValidateSpec.first_phase_round`: after
`normalise_params_round`, in the term, in the handlers of its
recursive continuations and in the bodies of its code, the
parameters of a loop that every turn passes on are taken away where
the checks of `FlambdaUnaliasRecParamsSpec.unalias_rec_params` pass
(of which more with the tidying), the
continuations that a handler binds on its way to its `Switch` are
bound outside it, the applications are made to go to their arms and
are then forwarded, and the continuations left unused are removed;
it makes no name, commutes
with every renaming by an equation,
`first_phase_round_equivariant_statement`, and keeps scope,
`FlambdaValidateScopeSpec.first_phase_round_keeps_scope_statement`;
the first phase of the term before, that of a term in which
something was lifted and the rounds that tidy after the demotion are
rounds of it); the calls of the term before
taken on when its rounds are over, or the term as it is
(`FlambdaValidateSpec.with_returns`: taken or not, it commutes with
every renaming by an equation and keeps scope,
`with_returns_equivariant_statement`,
`with_returns_keeps_scope_statement`; it is the forwarding with the
calls and the removal of the continuations left unused, and makes no
name), with the test by which the check chooses, which it asks of
the term after: the test compares two rewrites of a term up to bound
names, and answers the same of the term renamed
(`keeps_a_return_equivariant_statement`); the placing of continuations
that the check makes of each term that it compares
(`FlambdaValidateSpec.placed`: the binding of a continuation goes
down past the bindings under it, and into the handler under it that
alone names the continuation, where the tests of
`FlambdaSinkUsesSpec.sink_to_use` pass; no name is made, and the
statements have the texts that they had), which also keeps scope,
and the
tidying in rounds that follows it (`tidied_before`, `tidied_after`:
an argument that is the number of the arm of a `Switch` that it is
in is made the scrutinee where the parameter can then be taken away
(`FlambdaGiveScrutineeSpec.give_scrutinees`, which makes no name,
looks for the scrutinee to give among those of the `Switch`es that
the application is under, the nearest first, which is an order of
the term and not of names,
commutes with every renaming by an equation,
`give_scrutinees_equivariant_statement`, and keeps scope,
`FlambdaStaticsScopeSpec.give_scrutinees_keeps_scope_statement`);
a parameter that every application gives one variable is taken away
where the checks of `FlambdaUnaliasParamsSpec.unalias_params` pass,
a parameter of the one continuation of a recursive `Let_cont` that
every application outside the handler gives one variable, and every
application in the handler gives the parameter itself, is taken away
where the other checks of
`FlambdaUnaliasRecParamsSpec.unalias_rec_params` pass, and the handler
then begins with a `Let` of it to that variable (the pass compares
names by equality only and makes none: the `Let` has the name of the
parameter, and the variable may have that name too, the `Let` then
binding the parameter to the variable from outside that it hid; it
commutes with every renaming by an equation,
`unalias_rec_params_equivariant_statement`, and keeps scope,
`FlambdaStaticsScopeSpec.unalias_rec_params_keeps_scope_statement`;
the tidying of each term applies it just after `unalias_params`),
the scrutinee of a `Switch` is made the number of the arm at the reads
that the header of `FlambdaArmNumbersSpec.v` lists, in the arm and in
the handler that only arms of that number apply (an argument of an
application of a continuation, a defining expression that is the
variable, an argument of a primitive, the scrutinee of a `Switch`; a
call, a set of closures, a projection and a static binding keep the
variable), and the variable that it is the untagging of is made the
tagged number at the last three of those reads
(`FlambdaArmNumbersSpec.numbers_in_arms`, which puts constants for
variables, compares names by equality only and makes none, commutes
with every renaming by an equation,
`numbers_in_arms_equivariant_statement`, and keeps scope,
`FlambdaStaticsScopeSpec.numbers_in_arms_keeps_scope_statement`), what
that has made a primitive on constants or a `Switch` on a constant is
folded (`fold_constants`);
in the term after, the continuations that a handler binds on its way
to its `Switch` are then bound outside it and the applications are
made to go to their arms where the kinds are known
(`lift_handler_conts`, `forward_to_arms_kinded`, as in the rounds of
the term after);
in the term before a `Switch` whose arms
are then all the same is made that arm, and one whose arms then apply
one continuation to arguments that are computed from their numbers is
made that application (`collapse_switches`,
`FlambdaMergeSwitchesSpec.merge_switches`, as in the rounds); a
continuation that this leaves unused is then removed, and one that it
leaves applied once has its handler put at its application
(`remove_unused_conts`, `sink_let_conts`, and `inline_tail_conts` or,
in the term after, `FlambdaInlineTailSpec.inline_tail_conts_kinded`);
all are rewrites of the rounds and none makes a name;
primitives are shared, aliases are followed and the `Let`s left
unused are removed; and then, once, the groups of static bindings at
the head of the body of a continuation binding are bound before it
(`FlambdaGroupsUpSpec.groups_up`), which commutes with every renaming
by an equation (`groups_up_equivariant_statement`) and keeps scope
(`FlambdaStaticsScopeSpec.groups_up_keeps_scope_statement`)).
`unalias_params` and the two tidyings commute with
every renaming, each by an equation, the tidyings for every number
of rounds (`unalias_params_equivariant_statement`,
`tidied_before_equivariant_statement`,
`tidied_after_equivariant_statement`); `unalias_params` and the
tidying of the term before keep scope
(`FlambdaStaticsScopeSpec.unalias_params_keeps_scope_statement`,
`tidied_before_keeps_scope_statement`). Every one is an equation
that holds of every term. Other functions that the check applies are
stated there as well: `with_lifting`, with the lifting renamed with
the term, `without_slots` and `with_known`, each an equation
(`with_lifting_equivariant_statement`,
`without_slots_equivariant_statement`,
`with_known_equivariant_statement`).
The test of the exits that `FlambdaLiftSpec.lift_unit` makes reads
names and makes none. `exits_plain` asks that a variable is given only
to a continuation in scope; `exits_plain_knowing`, which is the test
that `lift_unit` makes, allows as well a variable known to hold a
symbol or a constant: a parameter of a non-recursive handler that
every way of reaching the handler gives a symbol, a constant or such a
variable. It keeps the variables known and takes one away at a `Let`
of it and at a parameter of that name that is not known, the
parameters of a handler being taken in the order in which the machine
binds them (`params_known`), and all those of a recursive handler; the
body of code starts with none; what the scope of a `Let_cont` does to
the parameters of the continuations that it does not bind is read off
it first (`spoils`), and what is said there of a continuation is
dropped where a `Let_cont` binds that name (`forget_conts`). Each test
answers of a term renamed, with the scope and the variables known
renamed with it, what it answers of the term
(`exits_plain_equivariant_statement`,
`exits_plain_knowing_equivariant_statement`, in `FlambdaLiftSpec.v`);
nothing is stated of two terms that are the same up to their bound
names.
`FlambdaLiftObservedSpec.with_lifting_observed`, which the check
does not apply and which is `with_lifting` without the test of the
exits, is stated beside it: the same equation
(`with_lifting_observed_equivariant_statement`), and it keeps scope
(`FlambdaStaticsScopeSpec.with_lifting_observed_keeps_scope_statement`).
Of a lifting the two
continuations of its unit are renamed and the rest is passed as it
is: paths, symbols and code ids; for the lifting to a group around,
the places where groups are put (the groups themselves are read off
the term, `FlambdaLiftAroundSpec.groups_of`); and the tables to bind
at the head, of constants and symbols (where an element of a table
is a variable no table is bound,
`FlambdaUnusedHeadStaticsSpec.add_head_tables`). The
lifting to a group around, its check and the reading of its groups
commute with every renaming, the groups and what the check knows
being renamed with the term (`lift_around_equivariant_statement`,
`around_ok_equivariant_statement`, `groups_of_equivariant_statement`
in `FlambdaLiftAroundSpec.v`, with `rename_groups` and
`rename_known_constants`), and it keeps scope
(`FlambdaStaticsScopeSpec.lift_around_keeps_scope_statement`). Also
stated there: `lifted_in_turn`, which keeps
scope, and commutes with renaming up to the names made where the
continuations that it is given to avoid have among them those of the
units of the liftings (`lifted_in_turn_keeps_scope_statement`,
`lifted_in_turn_equivariant_up_to_statement`); after the last
lifting the validation reads the known value slots once more, with
every symbol that the unit binds taken to be in scope
(`FlambdaKnownSlotsSpec.read_known_slots_anywhere`), and then uses
what is known of the symbols once more, in rounds that make no name
(`FlambdaValidateSpec.demote_and_tidy_known`): the rounds commute
with every renaming by an equation with no premise
(`demote_and_tidy_known_equivariant_statement`); that they keep
scope is a lemma of `FlambdaValidatorEquivarianceProofs.v` and not a
statement, and the
two statements of `lifted_in_turn`, whose texts are as they were, are
of the function with that read and those rounds; and
`with_statics_first`, which reads the continuations of the units: it
commutes with a renaming of the term where the liftings are under
another renaming that does to those continuations what it does
(`with_statics_first_renamed_alike_statement`). The removal, from the
head of the term after, of the static bindings that nothing uses,
where its checks pass (`FlambdaValidateSpec.without_head_statics`), is
stated there too: an equation, for every set of names to remove
(`without_head_statics_equivariant_statement`); the checks read the
same of a term renamed, as a renaming moves no symbol and no code id.
`FlambdaImportsSpec.static_names_free_in`,
`FlambdaUnusedHeadStaticsSpec.head_free_in` and
`FlambdaValidateImportsSpec.head_removal_free_in` read symbols and
code ids only, and answer of a term renamed what they answer of the
term, with no premise (`FlambdaValidateImportsRenameSpec.v`:
`static_names_free_in_equivariant_statement`,
`head_free_in_equivariant_statement`,
`head_removal_free_in_equivariant_statement`). The tests that a
verdict holds of a unit run from a static environment
(`verdict_holds_from`, `verdict_knowing_holds_from`, and what they are
made of) compute again terms that `validate_unit` computes; where the
rounds make names the renamed computation has those terms under
another renaming, which a test on static names does not tell from the
first. Of what they are made of, the tests that go through no round
that makes a name have no premise that the code is closed
(`demote_and_tidy_known_free_in_equivariant_statement`,
`holds_from_equivariant_statement`, and
`statics_first_free_in_renamed_alike_statement`, where the term is
under a second renaming that does to the continuations of the units of
the liftings what the first does); those that go through such rounds
are stated where the code of the term is closed
(`known_free_in_equivariant_statement`,
`normalise_lifted_free_in_equivariant_statement`, and the same of the
tests of the rounds that also make direct the calls of the closures
that a `Let` makes, whether or not a lifting asks for it,
`known_calling_free_in_equivariant_statement` and
`normalise_lifted_calling_free_in_equivariant_statement`, and of the
tests of the rounds that also use, in the bodies of code, what
`my_closure` is known to be, `known_own_free_in_equivariant_statement`
and `normalise_lifted_own_free_in_equivariant_statement`;
`lifted_free_in_renamed_alike_statement`, where the term is under a
second renaming that does to the continuations to avoid what the first
does, the two continuations of the unit of each lifting being among
them; `lifted_holds_from_renamed_alike_statement`, where the term
before is under a second renaming that does to its free names, and to
the continuations of the units of the liftings and of the loops, what
the first does). Each of the tests of a verdict answers of the renamed
terms what it answers of the terms, where the code of the term before
is closed (`verdict_holds_from_equivariant_statement`,
`verdict_knowing_holds_from_equivariant_statement`, and with a second
renaming of the term before
`verdict_holds_from_renamed_alike_statement`); so, with the laws of
the two verdicts, whether a unit is validated with its imports does
not depend on the names of its variables and continuations, where the
code of the term before is closed
(`validate_unit_holds_from_equivariant_statement`,
`validate_unit_knowing_holds_from_equivariant_statement`). Nothing is
stated of `statics_free_of`, which reads a static environment and no
term. In these statements, and in those below of the second check and
of the validation asked with the known arrays read, what is given that
holds continuations is renamed with the terms, by the first renaming
where there are two: the liftings, the continuations given for the
loops, and the continuations to avoid. The static environment that a
test from the imports is given, the imports that the second check is
given, the names to remove and the value slots are the same on the two
sides: a renaming moves no symbol and no code id.
By the same prover, in the types library (`FlambdaEnvRenameSpec.v`):
the functions of typing environments.

The validation applies one rewrite that makes names, `rename_apart`,
in each round of the second phase of the term before. In the fourth
form, by the second and third provers: each rewrite of a round keeps
scope (`FlambdaKeepsScopeSpec.v`, `FlambdaValidateScopeSpec.v`,
`FlambdaLoopsScopeSpec.v` for the passes for loops,
`FlambdaStaticsScopeSpec.v`; for some passes a file of the pass,
named with the pass below); a round, the normalisation of the term
before and that of a term in which something was lifted keep scope
and commute with renaming up to the names that they make; and
`validate_unit` answers the same on a pair of terms and on the pair
renamed (`validate_unit_equivariant_statement`) where the code of the
term before is closed. Nothing is said of a term before whose code
is not closed. What the validation is given that holds continuations
is renamed with the terms: the liftings, and the continuation given
for each loop that it makes of the self tail calls of a function
(`FlambdaLoopifySpec.loopify_unit`, which it applies to the term
before ahead of every rewrite that makes names).

Where a round of the second phase has settled the term, the parameters
of continuations that hold a tagged immediate and are untagged are
bound again at the head of their handlers
(`FlambdaShadowParamsSpec.shadow_params`, which makes no name: the
`Let` that it puts binds the parameter's own name under the parameter,
so that the result has that name bound twice on one path, which
`binders_distinct` does not forbid, being about the names of one
binder; it commutes with every renaming by an equation,
`shadow_params_equivariant_statement`, and keeps scope,
`FlambdaStaticsScopeSpec.shadow_params_keeps_scope_statement`);
`rename_apart_avoiding` comes just after and gives every binder a name
of its own; and such a parameter gets a twin that holds the number
(`FlambdaTwinParamsSpec.twin_params`, which makes no name either: the
`Let` goes, its variable is the parameter and the parameter is one
more parameter, the twin; it compares names by equality only, commutes
with every renaming by an equation,
`twin_params_equivariant_statement`, and keeps scope,
`FlambdaStaticsScopeSpec.twin_params_keeps_scope_statement`). The
round asks the comparison twice, to decide whether the twins are
looked for and whether one was made, and applies
`rename_apart_avoiding` a second time where it looks for them
(`FlambdaValidateSpec.with_twins`); its two statements keep their
texts and are proved of the round with the twins
(`normalise_before_round_keeps_scope_statement`,
`normalise_before_round_equivariant_up_to_statement`).

The liftings are made in turn, each followed by a normalisation, and
each holds the two continuations of its unit. The term that a later
lifting is applied to need not mention them. The normalisation before
it commutes with renaming up to the names that it makes, and that
alone does not say that the lifting renamed by the first renaming is
the lifting renamed by the second. So each of those normalisations is
given the continuations of the units of all the liftings
(`FlambdaValidateSpec.unit_continuations`), its rounds make no name
that is one of them (`FlambdaRenameApartSpec.rename_apart_avoiding`,
which is `rename_apart` when it is given none), and it is stated with
`equivariant_up_to_avoiding`
(`normalise_lifted_equivariant_up_to_statement`). While the rounds
were not given those continuations the statement of `validate_unit`
had a second premise, on the terms that the liftings after the first
are applied to; it has none now.

The same holds where the term before is under a second renaming,
which does to its free names, and to the continuations that the
validation is given, what the first does
(`validate_unit_renamed_alike_statement`); the statement above is its
case of one renaming. That is the form for a caller that rewrites the
term before, ahead of the validation, with a rewrite that makes
names. In general: whoever rewrites a term with a rewrite that makes
names, ahead of a function that reads names which its caller holds,
gives the rewrite those names to avoid, and the law of the function
is stated with a second renaming that agrees with the first on them.

The second asking of the validation is such a caller
(`FlambdaValidateKnowingSpec.validate_unit_knowing`): it gives
`validate_unit` the two terms normalised by rounds in front of each
of which aliases are followed by a function that knows what symbols
are bound to. The rounds of the term before make names, and are given
the continuations of the units of the liftings and those given for
the loops to avoid. It answers the same on a pair of terms and on the
pair renamed where the code of the term before is closed
(`validate_unit_knowing_equivariant_statement`), by the statement
with a second renaming. That a unit validated at the second asking is
correct does not rest on this.

A second check of a unit
(`FlambdaValidateRelatedSpec.validate_unit_related`) follows the two
comparisons of lifted terms of `validate_unit`, with the liftings made
without the test of the exits (`lifted_in_turn_observed`, which ends
as `lifted_in_turn` does). Its two statements are about what may
follow a unit, over the whole class `observer slots` at the value
slots given, and over no smaller one. The check answers of the two
units renamed what it answers of the two units
(`FlambdaValidateRelatedRenameSpec.v`:
`validate_unit_related_equivariant_statement`), and the same where
the term before is under a second renaming that renames its free
names, and the continuations of the units of the liftings and of the
loops, as the first does
(`validate_unit_related_renamed_alike_statement`). Neither has a
premise that the code is closed: the check tests that of both units
itself and answers `Not_related` where it fails. Its liftings made
in turn keep scope and commute with renaming up to the names that
the rounds make (`lifted_in_turn_observed_keeps_scope_statement`,
`lifted_in_turn_observed_equivariant_up_to_statement`), as those of
`validate_unit` do, and their test on static names answers the same
of a term whose code is closed, under a second renaming that does to
the continuations to avoid what the first does, the two
continuations of the unit of each lifting being among them
(`lifted_observed_free_in_renamed_alike_statement`). The statements
are of the checking function as it is defined, not of what is proved
of a pair that it accepts. It makes names where `validate_unit`
does, in the rounds that follow a lifting, which are given the
continuations of the units of the liftings to avoid. The test tool
asks the check only of a unit
that it reports as not validated, and prints what the check accepts
on a line of its own, which has not the word "validated"; such a
unit stays not validated and is in no count of the units validated.

Proved by the owner of the meet and the third prover, in the types
library (`FlambdaMeetRenameSpec.v`): the meet of types, of expanded
types at every fuel, and the functions that add to a typing
environment. The meet makes no variable; where it chooses between two
names it is by their binding times, which a renaming keeps. The join
that it is given is a parameter. The first statements are for a join
that commutes with renaming; the join that forgets does, so with it
the meet commutes with no premise. The join that the model has makes
variables and is not such a join: the statements of the second part
of that file are for two joins that a renaming relates, and the
model's join and the same with the source of new names renamed are
two such (above).

Proved by the owners of the passes, each in the file of its pass: the
identities of primitives (`FlambdaIdentitiesSpec.v`), the lifting of
constants (`FlambdaLiftSpec.v`) and out of the code of a function
(`FlambdaLiftFromCodeSpec.v`), the binding of `my_closure`
(`FlambdaMyClosureSpec.v`), the check of ranges (`FlambdaRangeSpec.v`),
the sharing of primitives (`FlambdaCseSpec.v`) and of the
projections of a value slot (`FlambdaShareProjectionsSpec.v`; that
it keeps scope, `FlambdaShareProjectionsScopeSpec.v`), the recovery
of the comparison that a three-way result is tested for
(`FlambdaRecoverComparisonSpec.v`:
`recover_known_comparisons_equivariant_statement`, an equation that
holds of every term; that it keeps scope,
`FlambdaRecoverComparisonScopeSpec.v`), the
fold of a `Phys_equal` of a simple against itself or of two different
symbols (`FlambdaSameComparisonSpec.v`:
`fold_same_comparisons_equivariant_statement`, an equation that holds
of every term: the pass compares two simples by equality and asks
whether both are symbols, which no renaming moves; that it keeps
scope, `FlambdaSameComparisonScopeSpec.v`), the folding
at known symbols (`FlambdaKnownSymbolsSpec.v`; of a static block one
of whose fields is a variable, what is known has a constant in the
place of the variable, `unknown_field`, to which no load is folded:
so what is known holds no variable, a renaming leaves it as it is,
and the statement has the text that it had,
`fold_known_equivariant_statement`), the two
`Let`s that the copy of an inlined body ends with made one in the
term after (`FlambdaAliasLetsSpec.merge_alias_lets`: a `Let` of a
variable to the variable that the `Let` around it has just bound, at
the same kind, where the variable of the outer `Let` is not free in
the body of the inner one (which may bind a variable of that name
again and use it), is made one `Let` with it, before the aliases are
followed; the pass
compares variables by equality and makes no name;
`merge_alias_lets_renamed_statement`, an equation that holds of
every term; it has no statement that it keeps scope, and it is
applied to the term after only, where no round makes a name), the
read of a known block where the read is sure to have a value, and the
removal of an allocation of constants that nothing uses, in the term
after (`FlambdaReadsAfterSpec.load_known_blocks_surely`, the part of
`FlambdaBlockLoadSpec.load_known_blocks` that is sure to have a value:
`load_blocks_knowing_surely_equivariant_statement`, with what is known
of the blocks renamed with the term by `rename_known_blocks`, and
`load_known_blocks_surely_equivariant_statement`, both in
`FlambdaValidatorEquivarianceSpec.v`;
`FlambdaReadsAfterSpec.remove_constant_allocations`:
`remove_constant_allocations_equivariant_statement`, in its own file;
both compare variables by equality, make no name and bind nothing, and
under a binder the first forgets what is known of the name; they have
no statement that they keep scope, and are applied to the term after
only, where no round makes a name), the
reading of a known immutable array (`FlambdaArrayLoadSpec.v`:
`load_known_arrays_equivariant_statement`, an equation that holds of
every term; `load_arrays_knowing_equivariant_statement`, the same
with what is known renamed with the term by `rename_known_arrays`,
which renames the variables that are known to be arrays and the
variables that their elements are known to hold, and leaves the
symbols and the numbers of elements as they are; the pass makes no
name and compares variables and symbols by equality only; that it
keeps scope,
`FlambdaArrayLoadScopeSpec.load_known_arrays_keeps_scope_statement`;
no round of the validation applies it; `FlambdaValidateReadingSpec.v`
has it beside the validation: `validate_unit_reading` and
`validate_unit_knowing_reading` ask `validate_unit` and
`validate_unit_knowing` of the term before with its known arrays read
(`before_read`, `normalise_reading`: rounds of `reading_round`, each
of which makes the loops of the term, normalises it as the term before
is normalised, and reads the arrays in the term, in the handlers of
its recursive continuations and in the bodies of its code,
`read_arrays`; a round that reads nothing, by the comparison up to
bound names, gives the term back as it was); the reading makes no name
of its own, and the rounds of the normalisation in it are given, as
names that they are not to make, the continuations of the units of the
liftings and those given for the loops (`normalise_before_avoiding`),
as the second asking gives them to its rounds; the laws under renaming
are in `FlambdaValidateReadingRenameSpec.v`: the read commutes with
every renaming by an equation and keeps scope
(`read_arrays_equivariant_statement`,
`read_arrays_keeps_scope_statement`); the normalisation of a round
keeps scope and commutes with renaming up to the names that it makes,
the second renaming doing to the continuations given what the first
does (`normalise_before_avoiding_keeps_scope_statement`,
`normalise_before_avoiding_equivariant_up_to_statement`); a round, the
rounds and the term that the two functions give keep scope
(`reading_round_keeps_scope_statement`,
`normalise_reading_keeps_scope_statement`,
`before_read_keeps_scope_statement`), and of a renamed term whose code
is closed each gives its result under a second renaming, which does to
the free names of that result and to the continuations to avoid what
the first does (`reading_round_renamed_statement`;
`normalise_reading_renamed_statement`, where the continuations of the
loops are among those to avoid; `before_read_renamed_statement`, which
asks no such thing, the function giving them to avoid); and the two
functions answer the same on a pair of terms and on the pair renamed,
the liftings and the continuations given for the loops renamed with
it, where the code of the term before is closed
(`validate_unit_reading_equivariant_statement`,
`validate_unit_knowing_reading_equivariant_statement`)), the
removal of unused
value slots (`FlambdaUnusedSlotsSpec.v`), the reading of a value slot
that every set of closures of the unit gives the same symbol or
constant, outside the code of a stub (`FlambdaKnownSlotsSpec.v`), and
the same with every symbol that the unit binds taken to be in scope
(`read_known_slots_anywhere`:
`read_known_slots_anywhere_equivariant_statement`, an equation that
holds of every term, a renaming moving no symbol; that it keeps
scope,
`FlambdaStaticsScopeSpec.read_known_slots_anywhere_keeps_scope_statement`,
`keeps_scope` saying nothing of symbols; what it gives need not pass
`FlambdaStaticsInScopeSpec.statics_in_scope`: it may put a symbol
where the unit binds that symbol only later
(`read_known_slots_anywhere_out_of_scope_statement`)), the
sharing of the reads of two places of the static constants of one
group that hold one variable (`FlambdaSharedReadsSpec.v`:
`share_static_reads_equivariant_statement`, an equation that holds of
every term, what is known being read from the term and so renamed with
it; of the function that it is made of,
`share_reads_equivariant_statement`, with what is known and the reads
made renamed with the term by `rename_held` and `rename_reads_made`,
which rename the variables and leave the symbols and the places as
they are; that it keeps scope,
`share_static_reads_keeps_scope_statement`, which is `keeps_scope`
written out, that file coming before the one that defines it; that
what it gives passes `FlambdaStaticsInScopeSpec.statics_in_scope`
where the term does, `share_static_reads_in_scope_statement`, and of
the function that it is made of the same of `expr_statics_in_scope`,
at the symbols and the code ids in scope around the term,
`share_reads_in_scope_statement`; the pass makes no name: a read of a
place becomes the variable of an earlier read of a place that the
group gave the same variable, and an earlier read is forgotten where
its variable is bound again; a round of the normalisation of a term in
which something was lifted applies it, after the reading of known
value slots, and so does a round of those that follow the last
lifting), the
direct calls of the unit's code that is not a stub, made indirect in
the code of a stub
(`FlambdaStubCallsSpec.v`: `demote_stub_calls_equivariant_statement`,
an equation that holds of every term, the rewrite making no name and
a renaming leaving what code says of itself as it is; that it keeps
scope, `FlambdaStaticsScopeSpec.demote_stub_calls_keeps_scope_statement`),
the order of static bindings
(`FlambdaStaticsOrderSpec.v`), the use of the static closures in
sight (`FlambdaKnownClosuresSpec.v`), the
call of a closure that a `Let` in sight makes, made direct
(`FlambdaClosureCallsSpec.v`:
`call_made_closures_equivariant_statement`, an equation that holds of
every term; `call_closures_knowing_equivariant_statement`, the same
with what is known renamed with the term by `rename_made_closures`,
which renames the variables that are known to be closures and leaves
their function slots and code ids as they are; the pass binds no name
and makes none, compares variables by equality, and forgets what is
known of a variable where it is bound again; its result has the free
variables and the free continuations of the term, and its code is
closed where that of the term is,
`call_closures_knowing_names_statement`; that it keeps scope is a
lemma of `FlambdaValidatorEquivarianceProofs.v` and not a statement; a
lifting asks for it by its field `made_closures_called`, a bool that
no renaming moves; the normalisation that follows that lifting, in the
first check and in the second, then applies it at the end of each
round of its second phase, in the rounds before the rewrites from the
types and in those after them, in the term and in the bodies of its
code: the round of `FlambdaValidateSpec.normalise_lifted_calling` is
`normalise_lifted_round_calling`, a round of `normalise_lifted` and
then the pass, and the rounds that know `my_closure`, next, add to
that round after it; its first phase is as it was, and the
normalisation after the last lifting does not apply it; with the field
unset `normalise_lifted_calling` is `normalise_lifted` of the same
rounds, continuations to avoid, code kept and term,
`normalise_lifted_calling_unset_statement`), the use, in the body of
code, of what `my_closure` is known to be from its code
(`FlambdaClosureOfCodeSpec.know_own_closures`: in the body of code
`my_closure` is known to be a closure of that code, the body running
only where it is one, and an indirect call of a variable known to be a
closure of a code is made the direct call of that code; a variable
bound to the projection of a function slot from such a variable is
known in its turn where the unit makes a set of closures with the code
at the slot projected from and every such set has the same code at the
slot projected to (`sibling_code`); the projection of a value slot
from such a variable that names the function slot it projects from
(`Project_value_slot_from`) is made a symbol or a constant where the
unit makes a set of closures with the code at that function slot,
every such set gives the value slot that symbol or constant and, if it
is a symbol, it is in scope (`slot_held`, `read_code_slot`); a
`Project_value_slot`, which names no function slot, is left as it is;
nothing else is rewritten, and the pass is the identity of a unit in
which a group of static bindings does not bind its static sets of
closures whole (`groups_whole`);
`know_own_closures_equivariant_statement`, an equation with no
premise, and `know_own_closures_names_statement`, that where the code
of the unit is closed, the code of the result is closed and the result
has no free variable or continuation that the unit has not; it binds
nothing and makes no name; what it keeps of a variable is a code id,
which it forgets under a binder of the variable (the `Let` of it, and
`forget_code_vars` for the parameters of a handler); it leaves the
code alone where it is the code of a stub or `my_closure` is also a
parameter; what it puts in the place of a projection is a symbol or a
constant; it reads the sets of closures that the unit makes with what
they put in their value slots, outside the scope of the variables
there (`FlambdaSetsOfCodeSpec.sets_made`), and asks of them only that
a function slot has a given code, or that a value slot is given a
given symbol or constant (`fact_closed`; `closed_simple` in
`slot_held`), so nothing is concluded from the name of a variable read
outside its scope; in the rounds it comes after the round of
`normalise_lifted_calling`, where the lifting asks for it
(`FlambdaValidateSpec.normalise_lifted_round_own`,
`normalise_lifted_own`; the field `own_closures_known`, which
`rename_lifting` passes on, having no name in it), the
move of code to the head of
what it is in (`FlambdaFloatCodeSpec.v`), the binding first of the
code of a unit (`FlambdaHoistCodeSpec.v`) and of its static bindings
that need nothing from around them (`FlambdaHoistStaticsSpec.v`), the
copy of code for static sets of closures
(`FlambdaCopyCodeSpec.copy_static_code`: the code that a list pairs
with new code ids is bound again under them, and the static sets of
closures of the symbols given get the new ids; the new code ids are
given to the pass, as a lifting is given its symbols, and the copy is
made only where the check passes that the code ids to copy are
distinct, that the new ones are distinct and that the unit binds none
of the new (`copy_ok`, `binds_no_code`); the pass introduces no fresh
variable, continuation or symbol names: a copy retains the binders of
the code that it copies; a renaming moves no code id and no symbol, so
the rewrite, the check and the copies made in turn commute with every
renaming by an equation, what they are given being as it is
(`copy_expr_equivariant_statement`,
`binds_no_code_equivariant_statement`,
`with_copies_equivariant_statement`); a lifting holds the copies to
make, which are passed as they are when the lifting is renamed, and
`with_lifting` and `with_lifting_observed` end with them, their
statements keeping their texts; that the copies keep scope is a lemma
of `FlambdaStaticsScopeProofs.v` and not a statement), the
lifting out of the arms of a test (`FlambdaLiftFromArmsSpec.v`), the
inlining, in the body of code, of the calls of the other code of its
group (`FlambdaInlineSiblingsSpec.v`), and the same in the term
after, where the callee is a symbol that the group binds to a closure
of the code called
(`FlambdaKnownSiblingsSpec.inline_known_sibling_calls`: it compares
code ids and symbols and makes no name, the copy of a body having the
names that the body binds, so that a body copied twice binds its
names twice; `inline_known_sibling_calls_renamed_statement`, an
equation that holds of every term and every list of code to leave; it
has no statement that it keeps scope, and it is applied by a round of
the term after only, where no round makes a name), the unboxing of
the parameters
of continuations and of variants (`FlambdaUnboxParamsRenameSpec.v`,
`FlambdaUnboxVariantsRenameSpec.v`, each with the statement that the
pass keeps scope), the unboxing of the reads of a loop
(`FlambdaUnboxLoopsSpec.unbox_loops_knowing`: a read of a parameter
of a loop becomes a parameter that has the name of the variable that
the `Let` of the read bound, so no name is made; a parameter is
unboxed only if no read of it is left in the handler (`reads_left`:
no primitive that reads a box is applied to the parameter itself,
outside code and where the parameter is not bound again, names being
compared by equality; a read through another variable that holds
the box is not looked for); with what is known
renamed with the term, `unbox_loops_knowing_equivariant_statement` in
`FlambdaUnboxVariantsRenameSpec.v`; that it keeps scope, with what
is known of symbols alone,
`unbox_loops_knowing_symbols_keeps_scope_statement` in
`FlambdaUnboxInScopeRenameSpec.v`), the unboxing together of the
boxed numbers that continuations pass to one another
(`FlambdaUnboxTogetherSpec.unbox_together_knowing`: a parameter that
is unboxed keeps its name, in another place among the parameters of
its handler, and the `Let` of its unboxing keeps its variable, so no
name is made; the positions that it unboxes are computed from the
term, by tests of equality of names; with what is known renamed with
the term, `unbox_together_knowing_equivariant_statement` in
`FlambdaUnboxVariantsRenameSpec.v`; that it keeps scope, with what
is known of symbols alone,
`unbox_together_knowing_symbols_keeps_scope_statement` in
`FlambdaUnboxInScopeRenameSpec.v`; a round of the first phase of the
term before applies it, after `untag_params`), the merging of the
arms of a
`Switch` that apply
one continuation (`FlambdaMergeSwitchesSpec.v`), the merging of such
arms into a load from a static table that is in scope
(`FlambdaMergeTablesSpec.merge_tables`, which reads the tables off
the term and makes no name: where one table has the arms, the
variable of the scrutinee is bound again as the other merging binds
it; where there is a way for each argument, the `Let`s bind the
parameters of the continuation that the arms apply, which is bound
around the `Switch`, as
`FlambdaMergeSwitchesSpec.merge_switch_by_arguments` binds them, and
the scrutinee is not bound again:
`merge_tables_equivariant_statement`, an equation that holds of every
term; that it keeps scope, `FlambdaMergeTablesScopeSpec.v`, where the
symbols of the tables are not counted, `keeps_scope` saying nothing
of symbols), the binding, outside a handler, of the continuations that
the handler binds on its way to its `Switch`
(`FlambdaLiftContsSpec.lift_handler_conts`: a continuation that is
moved keeps its name and takes, under their own names, the
parameters of the handler that it may read, so no name is made;
`lift_handler_conts_equivariant_statement`, an equation that holds of
every term; that it keeps scope, `FlambdaLiftContsScopeSpec.v`;
two tests more keep the pass from a `Let_cont`,
`number_of_a_scrutinee` and `read_where_given`: they read the term,
compare names by equality only and make none, and the first counts
a variable that a `Switch` tests as the number of an arm, by its
name alone, in the handler of the continuation that the arm applies,
where that variable need not be in scope, as its header says; a test
can only keep the pass from a place, and the two statements have the
texts that they had; a round of the first phase of the term before
applies the pass, and so does the normalisation of the term after,
each just before the applications are made to go to their arms), the
demotion of the
exception handlers that no call names (`FlambdaDemoteSpec.v`, with
the list of continuations that it is given renamed; that it keeps
scope is `FlambdaDemoteScopeSpec.v`), the rewrite of the calls of
code that never returns (`FlambdaNeverReturnsSpec.v`, with the code
that is known renamed; it makes no name: it binds the call's own
return continuation again around the call, and parameters that have
the names of the code's own binders, so its result need not have
distinct binders; that it keeps scope is
`FlambdaNeverReturnsScopeSpec.v`), and the machine with a stack of
exception handlers (`FlambdaTrapSpec.v`). In the fourth form: giving
every binder a name of its own (`FlambdaRenameApartSpec.v`).

Proved by the owners of the model of Simplify, in the types library.
Each is an equation: on the renamed typing environment and the
renamed arguments the function gives the renamed answer. Some take a
function as a parameter and ask that it commutes with renaming. Where
that is asked of a join, it is in the sense of the paragraph on the
meet above: the join that forgets is such a join, and the join of
the model, which makes variables, is not.

- The function that rewrites a term from the types, and its parts
  (`FlambdaSimplifyPassRenamingSpec.v`,
  `FlambdaSimplifyPrimitivesRenamingSpec.v`,
  `FlambdaSimplifyHandlersSpec.v`, `FlambdaSimplifyApplySpec.v`,
  `FlambdaSimplifyStaticsSpec.v`, `FlambdaSimplifyCodeEnvSpec.v`,
  `FlambdaSimplifyValueSlotsSpec.v`), with no premise. From the
  environment that knows nothing it is a rewrite in the form of those
  of the validation.
- The same function with pruning
  (`FlambdaSimplifyPruningRenamingSpec.v`): for a rewrite of a
  `Switch` that commutes with renaming; with the meet that the
  contract is proved of, for a join that does; and with the join that
  forgets, with no premise.
- The rewrites of a `Switch` (`FlambdaSimplifySwitchSpec.v`,
  `FlambdaSimplifySwitchArmsSpec.v`, `FlambdaSimplifySwitchMergeSpec.v`):
  for a meet, and for a rewrite of an `Apply_cont` where one is
  taken, that commute with renaming; with the meet that the contract
  is proved of, for a join that does. The rewrite that merges the
  arms binds variables. It does not make them: they are an argument,
  and are renamed with the term. `FlambdaSimplifySwitchRewriteSpec.v`
  has the same with the variables kept as they are, for a renaming
  that fixes them, which is less. It also has that the rewrite of
  `FlambdaSimplifySwitchArmsSpec.v` (`simplify_switch_arms`, which
  merges nothing) is a rewrite of a `Switch` that commutes with
  renaming, for such a meet and such a rewrite of an `Apply_cont`:
  that is what the function with pruning asks of the rewrite that it
  is given. The rewrite that merges the arms, with its variables kept
  as they are, is not such a rewrite. `FlambdaSimplifyArmsSpec.v`
  has the arms of a `Switch` each rewritten under what its arm
  knows, with no meet (`simplify_arm_args`,
  `simplify_switch_and_arms`): each an equation with no premise.
  Nothing calls it yet.
- Following aliases in a term with the aliases of the library
  (`FlambdaEnvAliasesFollowSpec.v`; that it keeps scope,
  `FlambdaEnvAliasesFollowScopeSpec.v`), and the sets of aliases
  (`FlambdaEnvAliasesSetSpec.v`), with no premise. What is renamed
  with the term there is what is known of the names where a subterm
  is, not a typing environment; and a function that one of these is
  given (a filter, the binding times of names) is renamed with the
  other arguments.
- The accessors and the creators of types (`FlambdaTypeAccessSpec.v`,
  `FlambdaTypeCreatorsSpec.v`), with no premise.

Owed:

- each pass of the validator that is merged after this was written,
  by its owner or by the third prover, and `validate_unit` again with
  it;
- a rewrite that the validation is to apply owes `keeps_scope` with
  its equation: a round makes names, and `validate_unit` answers the
  same on renamed terms only because every rewrite of a round keeps
  scope;
- by their owners, the functions on types that have no such statement
  yet.

## 8. What a change to the model owes

- **A new term form.** Its case in `rename_expr` and the functions
  beside it, in the free names of `FlambdaBindersSpec.v`, in the three
  conditions if it binds, in `expr_eqb`, and in the proofs that go by
  cases on terms. If it binds, a row in the table of section 3, a rule
  of `alpha_equivalent`, and a statement that it is a name abstraction.
- **A new place in a term that names** (as a trap action does): it is
  renamed, it is among the free names, and the comparison pairs it as
  it pairs the other occurrences of that sort.
- **A new function on terms.** Section 7.
- **A new place for names.** A sort of its own, an action, and an
  instance of the interface, as the locations of the heap have
  (`FlambdaLocationsSpec.v`). Tell N1 before putting a variable or a
  continuation into a value.
- **A statement that renames.** The premise that code is closed, or a
  reason why it is not needed.
- **A fresh name.** `fresh_name`, never a count of your own. A pass
  that needs a new name does not make one: it puts a binder under a
  name that the term has and leaves the naming to
  `rename_apart_avoiding`, as `shadow_params` does for `twin_params`.
- **The name of an action.** The action of the renamings on a type is
  `rename_` followed by what the type is of (`rename_known_closures`,
  `rename_known_blocks`). A new one does not take a short name that a
  file which can be imported with it already uses for another type: a
  statement must not read differently according to the order of the
  imports. A few older actions share a short name (`rename_outcome`,
  `rename_answer`, and `rename_known` in the types library); a file
  that imports two of them writes the name in full.

## 9. Not done

- The older accounts of free names are not gone. The statements that
  a bound name may be renamed and the rules of `alpha_equivalent` are
  written with the one function, and their older forms are lemmas
  that proofs still use. What is left in the older style is in the
  last table of "Where it stands"; `requests/N1-one-free-names.md`
  says, statement by statement, what was replaced, what is left and
  why, and how to go back.
- The functions listed as owed in section 7.
- A step under a permutation of locations is proved of the machine as
  it is, in which no step allocates. The statements are in the form
  that stays true when one does (section 4); the proofs of the case
  of a `Let` are owed again then.
- The machine of regions (`FlambdaRegionsSpec.v`, with its invariant,
  `FlambdaRegionsInvariantSpec.v`, the erasure of regions,
  `FlambdaRegionsErasureSpec.v`, what is proved of its calls,
  `FlambdaRegionsCallsSpec.v`, and the check of the regions of a term,
  `FlambdaRegionsCheckSpec.v`). Nothing is stated of it under a renaming
  of variables and continuations, nor of the term with its regions
  dropped (`FlambdaRegionsSpec.erase`) under one (what is stated of
  `erase` is of runs, in `FlambdaRegionsErasureSpec.v`,
  `FlambdaRegionsCallsSpec.v` and `FlambdaRegionsCheckSpec.v`, and
  nothing in them is about a renaming). `erase` is not a function of
  terms up to bound names: it drops the `Let` of a `Local_allocation`
  that binds the name of its argument and makes an alias of any other,
  so of two terms that differ in that bound name the erased terms differ
  by a `Let`. No rewrite of the validation applies it; the test tool
  does, in its check of regions, where it compares `erase` of the term
  converted with its regions against the term validated, up to bound
  names (`expr_eqb`), and nothing says that a step of the machine of
  regions respects that equality. The step compares the names of two
  variables in one place (`marker_of`): the first argument of a
  `Local_allocation` is the variable that the `Let` above it binds, and
  the region that it names is not that variable. The machine makes names
  of a sort that no renaming moves: it draws symbols, one for the mark
  of each region and for each local allocation (for a set of closures,
  one for each function slot), from a counter that only goes up
  (`rs_next`, which starts at `rs_base`), and not by `fresh_name`; the
  next symbol is the successor of the last (`Pos.succ`, `advance`), so
  of two symbols drawn the smaller was not drawn after the other. It
  looks at the number of a symbol: a symbol that it drew is one at or
  above `rs_base` (`drawn`), and a value that may not be evaluated is a
  drawn symbol that nothing binds (`value_usable`), so what a step does
  depends on it. That the symbols drawn are none of the term or of the
  static environment is a test made before the run
  (`region_names_fresh`), and that they stay apart is a part of the
  invariant that a run keeps
  (`FlambdaRegionsInvariantSpec.symbols_fresh`,
  `region_run_keeps_invariant_statement`); both tests are written with
  the order of the numbers (below `rs_base`, below `rs_next`). The test
  of a state by which it is proved that code cannot end a region of its
  caller (`FlambdaRegionsCallsSpec.calls_kept`) compares the numbers of
  two symbols and takes the smaller to be the older (`frames_ordered`,
  `held_older`, `cont_held_older`); by `region_names_fresh`, a symbol of
  the term or of the static environment that the run starts with is
  below every symbol drawn. That a run does not depend on the base is
  not stated.
- The check of the regions of a term
  (`FlambdaRegionsCheckSpec.region_check`, with a certificate; it is a
  function of the model, not the check of the test tool, and nothing
  runs it) is by names. The certificate is keyed by the continuations
  that the term binds (two handlers bound under one name have one entry)
  and by code ids, and names the regions that are open at a handler by
  variables (`cert_handlers`, `he_open`; a `lifetime` may be `Of_region`
  of a variable); `region_at_jump` and `lifetime_at_jump` put the
  regions that a jump gives in the place of the parameters of the
  handler that are regions, a substitution and not the action of a
  renaming. The check compares variables and continuations by equality
  only, and it refuses a term in which the variable of a region in
  scope, open or not, is bound again (`not_a_region`), so it does not
  answer the same of two terms that differ by bound names. Its law under
  a renaming is owed (section 7, a check): that `region_check` of the
  renamed term with the certificate renamed is `region_check` of the
  term is not stated, and no action of the renamings on a certificate is
  defined. The property of states by which
  `region_check_fault_statement` is proved
  (`FlambdaRegionsCheckedSpec.v`) asks that the regions of the state of
  the check are in scope (`check_state_scoped`) and reads the order of
  the symbols drawn as time (`value_within`, `lifetime_holds`, with
  `FlambdaRegionsCallsSpec.frames_ordered`).

## 10. The compiler's side

Nothing in the compiler was changed. What follows is what was found by
reading `nominal/` and its users against the laws of section 6; the
labels are those of the discrepancies with the compiler.

Where the code follows the discipline: no caller outside
`name_abstraction.ml` reads an abstraction without `pattern_match`
(`descr` is not exported; `pattern_match_for_printing` is used by the
printers only); `Compare` opens both sides at one fresh name and then
compares names by equality. The canonical alias is chosen by binding
time, and CSE sorts by `stable_compare_simples`, which compares binding
times; but both fall back on the order of names where binding times do
not decide: two variables of one binding time are ordered by their
stamps, and among symbols, none of which is earlier than another, the
alias chosen is the least in the order of the map (N1-7).

What it would take for the compiler to obey the same laws:

1. **Make the precondition of a binder structural.** Check distinct
   parameters in `Bound_parameters.create` always, as
   `Bound_continuations.create` does, and the two continuations of
   `Bound_for_function`. Both eliminators then agree (N1-3).
2. **Make the pair eliminator total.** `pattern_match_pair` returns an
   error value when the two bindables differ in shape, for every
   bindable, and no caller tests first (N1-2).
3. **Check closed code against the body** when invariant checks are
   on: `Code0.create` compares the claim with the free names of the
   abstraction (N1-4).
4. **Put the remaining binders behind `Name_abstraction`**: the
   existential variables of an environment extension, and the
   variables a level defines where a level is cut or joined. Freshness
   for the environment then comes from `pattern_match` and not from
   the caller (N1-6).
5. **Take static patterns out of `Name_abstraction`.** They are not
   abstractions: they are never renamed. A `Let` of static constants as
   a constructor of its own makes `free_names` and `apply_renaming`
   agree (N1-5).
6. **Two types for the two things `Renaming.t` is**: permutations,
   which form a group and are what binders use; and import maps, which
   are injections that also prune value slots (N1-8).
7. **No term structure from the order of names.** Where an order is
   needed, use binding time or position. A rule that is easy to state
   and to search for: never fold over a `Variable.Map` or a
   `Variable.Set` to build a list that ends up in a term (N1-7).
8. **A test of equivariance.** Run Simplify on a unit and on the same
   unit with its names created in another order, and compare the two
   results up to bound names. A pass that depends on names shows as a
   difference; it needs no model. Run once by hand, on 81 units with
   unused bindings put first: no dependence on the order of a set or a
   map of names was seen; the names of lifted symbols contain the
   stamp of a variable, so symbols too have to be compared up to
   renaming (N1-7).
9. **Repair `compare.ml`**: the maps of slots (N1-1) and the missing
   case for primitives of four arguments, without which a term is
   judged different from itself (N1-9). Or replace its core by the
   comparison of section 5. That one is proved to be an equivalence
   relation for the terms of the model only: it would need the
   pairing of slots, symbols and code ids, and every construct and
   annotation of the compiler's terms that the model has not got, and
   what is proved here does not carry over to those without proof.
