# Core change request C2: kinds and well-formed terms

Owner: C2. Branch `flambda2-rocq/C2-epoch`: the whole repair, as a series
that C0 can apply. This is the third version of the request and replaces
the two before it. The series is rebased
over C3's epoch (sets of closures of several functions) and has the
cases for its forms; the text is revised for that, and for the review of
the series (section 10).

## In short

Terms gain kinds where the compiler has them. A `Let` gives the kind of
its variable; a handler gives the kinds of its parameters; code gives
the arity of its parameters and that of its results; an `Apply` gives
the two arities it is made at. An arity is the compiler's
`Flambda_arity.t` without subkinds, unboxed products included.

The machine looks at one thing only: a call is undefined unless it
agrees with the code it calls, by the three tests that Simplify makes
before it replaces a call by `Invalid`. Everything that a well-typed
program with GADTs and layout `any` is known to put into Flambda 2 is
then undefined on the machine, with no premise (section 4).

A new pair of files, `FlambdaWellKinded*`, has a check of the kinds of a
term as a boolean function, the same for a state, and nine statements,
all proved. The check is a sufficient condition and a diagnostic. It is
not a condition that the compiler's terms are known to meet, and no
statement about those terms takes it as a premise (section 3).

What the human has ruled, and where it is applied:

- A call at arities that are not those of its code is undefined, and so
  is any mismatch of kinds where it is executed ("For D9 and D10 for
  the moment, I think we should assume these produce Invalid"). Step B,
  section 1.
- A check may not be stricter than the compiler ("We cannot be more
  strict on kind checks than the compiler"), and the compiler is the
  compiler without `-flambda2-kind-checks` ("Treat everything as if this
  is not specified"). So nothing here is described by what that flag
  does, the check looks at no argument of a call or of a primitive, and
  it is a gate for nothing.
- What GADTs and `any` can produce must be undefined ("those cases MUST
  produce Invalid"). Section 4.
- Where something is undefined, or refused, and neither a ruling nor a
  GADT forces it, the `Spec` file has a comment for the human, marked as
  a review comment addressed to mshinwell, with the number of the
  decision. There are two, both for D10 (section 9).

What the human is asked to decide is in section 9: one question about
binders (D10), with what each answer costs. What the test tool does
with the check's answer (D11) is settled: as built.

## 1. The change to the terms and the machine

The series has the change in two steps, as C0 asked. Step A is the
annotations with the machine as it was. Step B is the tests at a call.

### Step A: the annotations

`FlambdaModel.v`:

    Inductive arity_component : Type :=
      | Singleton (k : flambda_kind)
      | Unboxed_product (components : list arity_component).
    Definition arity : Type := list arity_component.

with `unarize`, `unarize_per_parameter`, `arity_of_kinds`, `arity_eqb`,
each named after the function of `kinds/flambda_arity.ml` that it is.
Parameters and arguments stay unarized, as in the compiler: an unboxed
pair is two variables, and void is none.

    Let (x : variable) (kind : flambda_kind) (defining_expr : named)
        (body : expr)
    Handler (params : list variable) (metadata : handler_metadata)
        (handler : expr)
    Code (params : list variable) (my_closure : variable)
        (return_continuation exn_continuation : continuation)
        (metadata : code_metadata) (body : expr)

    Record handler_metadata := { hm_params_kinds : list flambda_kind }.
    Record code_metadata :=
      { cm_params_arity : arity; cm_result_arity : arity }.

and `apply_expr` gains `ap_args_arity` and `ap_return_arity`. The two
records are made by `make_handler_metadata` and `make_code_metadata` and
are bound whole in patterns, so that a later annotation (C5 asks for
`hm_is_exn_handler`) is a field and no term changes shape for it.

A handler keeps a list of kinds beside its parameters, as the compiler
pairs each parameter with a kind. Code has no such list: the kinds of
its parameters are its parameter arity unarized (`code_params_kinds`),
and those of the parameters of its return continuation its result arity
unarized.

The machine ignores all of it in step A. Every statement about a `Let`,
a handler, code or a call now holds whatever these are annotated with:
twenty-four statements gain a quantified kind, metadata or arity
(section 6 names them), and none changes its meaning. Renaming leaves the
annotations as they are; alpha-equivalent terms have the same ones (the
rules of `alpha_equivalent` and `alpha_structural` say so); `expr_eqb`
compares them. `let_params` takes the kinds of the parameters and binds
every parameter that has an argument, also where the list of kinds is
too short, so that the bindings are those the call makes whatever a
term is annotated with (the reviewer's point N1). The `Let`s that
`let_closures` and `let_projections` make (C3's) are at kind `Value`.

### Step B: a call agrees with its code, or is undefined

(Since the series for partial applications and over-applications,
which landed after this request, a call is made of pieces:
`shape_of_call` and `shape_of_closure_call` say what the call is, and
`call_code` enters the body of code. The three tests below are in
`call_code`, for a call with as many arguments as the code has
parameters; a call with fewer or more is tested in
`shape_of_closure_call`. Where this request says `call_function`, read
the piece that holds the test.)

`call_function` takes the two arities of the call and has no step unless
three things hold. Each is what Simplify tests before it replaces a call
by `Invalid`; the functions named are in
`simplify/simplify_apply_expr.ml`.

| The test in `call_function` | The compiler | The reason it gives |
|---|---|---|
| `map kind_of_value vs = unarize args_arity` | `simplify_apply_shared`: the kinds of the arguments against `Flambda_arity.unarize (Apply.args_arity apply)` | `Application_argument_kind_mismatch` |
| `unarize_per_parameter args_arity = unarize_per_parameter (code_params_arity c)` | `arity_mismatch`, on `Flambda_arity.unarize_per_parameter` of the two | `Direct_application_parameter_kind_mismatch` |
| `return_arity = code_result_arity c` | `simplify_direct_function_call`: `Flambda_arity.equal_ignoring_subkinds` | `Application_result_kind_mismatch` |

The first is the one place where the machine looks at the kind of a
value. Poison has the kind it carries, and an allocated value or a
symbol, bound or not, has kind `Value`, so poison of the right kind
passes; that is intended and the comment says so.

Why the arities are trees. The second version had flat lists of kinds.
The review (B1) showed that this erases what decides `Invalid`. Each of
these is an example in `FlambdaWellKindedProofs.v` that a flat arity
would have run and that is now undefined:

- arguments: code with one parameter that is an unboxed pair of values,
  called at an arity of two parameters, each a value
  (`product_two_parameters_undefined`). An OCaml program of the
  reviewer's produces it: a GADT indexed by `#(int * int) -> int` and
  `int -> int -> int`;
- results: code that returns `#(a * #(b * c))`, called by a call that
  expects `#(#(a * b) * c)` (`product_result_nesting_undefined`). A list
  of lists of kinds would not do here: the compiler compares result
  arities as they are nested.

And the one that shows the two tests differ: arguments nested
differently inside one parameter run (`product_argument_nesting_runs`),
as the compiler compares arguments parameter by parameter after
unarizing each.

What else changes in step B:

- `inline_known_calls`, whose statement is an equivalence, inlines a
  call only where it is sure to agree with the code (`call_agrees_b`):
  its arities are those of the code, and each argument is a constant or
  a symbol of the kind the call gives it. The inlined body does not test
  the kind of what a variable holds, and the call does. The text of
  `inline_known_calls_equivalent_statement` is unchanged. Section 7 says
  what this costs the validation.
- `inline_calls`, a refinement, is unchanged: a call that does not agree
  is undefined, and anything refines it.
- `expr_eqb` already compares the arities of calls and the metadata of
  code, from step A.
- `call_function_step`, a step lemma of `FlambdaSemanticsProofs.v`
  (since replaced by `call_code_step` and the lemmas on the other
  pieces of a call), gains the three facts, and `step_apply_closure`
  (C3's: the way in which an `Apply` does not go wrong) the three as
  premises.

What the tests do not cover. They are about the values of the arguments
and about two pairs of annotations. Nothing is tested when the body
returns: the third test is that the return arity of the call is the
result arity of the code, an equality of two annotations, and the values
that the body then gives its return continuation are not looked at.

## 2. Kinds of values

Every value has a kind (`FlambdaValueSpec.kind_of_value`, C1's): that of
its constant, and `Value` for everything allocated and for a symbol.
The model has no subkinds.

## 3. The check, and what it is not

`FlambdaWellKindedSpec.well_kinded slots kinds conts e` is a boolean
function. What it checks:

1. every variable and continuation used is in scope;
2. the kind of a `Let` is that of its defining expression; for a
   primitive, the kind of its result (`prim_result_kind`, C1's);
3. the arguments of an `Apply_cont` have the kinds of the parameters of
   the continuation;
4. the scrutinee of a `Switch` is a naked immediate;
5. the return continuation of an `Apply` takes what the call says it
   returns, unarized, and the exception continuation one `Value`;
6. a handler has as many kinds as parameters, and its parameters are
   distinct, as are those of code with `my_closure`;
7. the body of code is well-kinded on its own, under its parameters at
   the kinds of its parameter arity;
8. a value slot holds a `Simple` of the kind of the slot (a table given
   to the check), where a closure is made, by a `Let` or statically, of
   a set of one function or of several; and its projection has that
   kind;
9. the fields of a static block are values, and the contents of a
   static boxed number are a number that can be boxed;
10. a variable bound to a closure, or to the projection of a function
    slot, is a `Value`.

What it does not look at, each being undefined where it is executed: the
kinds of the arguments of a primitive; the kind of the callee of a call,
and of what a value slot or a function slot is projected from, and the
function slot that a projection names (D3); whether a call agrees with
its code (the three tests).

It is a sufficient condition and a diagnostic. The reasons it is not
more, which the header of the file gives:

- Each condition is one the compiler has a check of its own for, which
  stops it. But it checks only where Simplify visits. A `Switch` on a
  constant, whose arm that is not taken applies a continuation to an
  argument of another kind, compiles: Simplify removes the arm first.
  The check here looks at every branch and answers no (the reviewer's
  term, here as `tests/semantics/kinds_dead_cont.fl`;
  `example_dead_arm_refused` and `example_dead_arm_runs` in the Proofs
  file; discrepancy R2-11). So as a gate it would be stricter than the
  compiler.
- That what closure conversion builds passes the check is known by
  experiment only (section 4).

So the check is not proposed as a condition of `well_formed_unit`. C0's
note `requests/C0-well-formed.md` says the same, and has the three of
its conditions that a dead branch cannot break (names in scope, distinct
parameters, annotations of the right length) as rows of its own.

The statements (all proved; `make claims` checks them):

| Statement | What it says |
|---|---|
| `call_at_wrong_kinds_undefined` | A call that fails one of the three tests has no step, whatever the state. |
| `partial_or_over_application_undefined` | A call whose arguments have the kinds of its arity and whose arities pass the compiler's test for a partial application or an over-application has no step (an omission; section 8). |
| `well_kinded_unit_state` | A unit that passes the check starts in a well-kinded state. |
| `step_well_kinded`, `run_well_kinded` | A step, and a run, keep a state well-kinded. |
| `well_kinded_no_kind_fault` | A well-kinded state reads no unbound variable, applies no handler to the wrong number of arguments, and switches on a naked immediate. |
| `well_kinded_exit` | A well-kinded state exits only through the interface, with values of its kinds. |
| `well_kinded_unit_never_faults`, `well_kinded_unit_exits` | The same for a unit run on its own. |

None has a premise about the primitives: the two laws used are C1's
`eval_prim_kinds_statement` (a primitive that has a value has one of its
result kind) and `eval_prim_preserves_statement` (a primitive makes its
value out of its arguments), which replace the two hypotheses of the
prototype. None is about progress. And for the types: the predicate on
states takes the kinds of the values held now; it does not remember what
a binder said. What a consumer may use is the fact at the step: a
well-kinded state that binds a variable binds a value of the kind the
binder gives.

With no premise at all, the machine gives less, and it is to be read
exactly. At a call that steps, the values of the arguments have, in
order, the kinds of the parameter arity of the code, unarized, and the
callee is a `Value`. That is a fact about the values that are bound. It
is a fact about what a parameter, looked up in the body, holds only
where the parameters and `my_closure` are distinct names, which the
machine does not test: code with one parameter `x` and `my_closure` also
`x` binds the argument over the closure. The check asks for distinct
names (item 6), and so do C0's conditions on binders. And nothing is
given for what the body returns (section 1, the end of step B).

## 4. What GADTs and layout `any` put into Flambda 2

The experiments are not in this tree; the table says what the programs
show. Every program is accepted by the type checker and compiled by
the real compiler with its default flags and five other settings; none
stops the compiler.

Two ways of making the mismatch: a GADT indexed at layout `any`, matched
in a function that is inlined where the index is known; and an equality
witness between two value types of different representation, with an
inlined `cast`. In both, the term before Simplify is consistent: every
variable has one kind and every call is an indirect call at the arity of
its own arguments. The mismatch appears when Simplify brings together
two things that the source kept apart by an indirection.

| Form | Programs | Simplify | The machine |
|---|---|---|---|
| A call whose argument arity is not the code's parameter arity | one of X2's, one of mine | `Invalid` | undefined: test 2 |
| A call whose result arity is not the code's | one of X2's, one of mine | `Invalid` | undefined: test 3 |
| A call with another number of unarized arguments or results (an unboxed pair against one value) | one of X2's, the reviewer's | `Invalid` | undefined: tests 2 and 3, on the trees |
| An over-application whose callee does not return one value; a partial application whose result is not one value, or whose argument is of another kind | one of X2's | `Invalid` | undefined: test 2 (another number of parameters) |
| A load that does not agree with the block (a value from a mixed block, a naked float from a block of values, a field beyond the end) | two of X2's, one of mine | `Invalid`, by the types | the argument is a `Value`, so this is not a mismatch of kinds: C1's `Block_load` has no value on a field that is not there or not a value; mixed blocks are not in the fragment |
| Unboxing or untagging another sort of number; `Is_int`, `Get_tag` on a boxed float or a closure | two of X2's | `Invalid`, by the types | as above: C1's primitives have no value on what they are not for |
| An array primitive at another array kind | one of X2's | a load from an immutable array: `Invalid`; a load or store of immediates on an array of naked floats: kept; `Array_length`: folded with the primitive's own kind | arrays are not in the fragment; recorded for C1 as discrepancy C2-8 |

Not produced, in any program tried: a call whose argument is not of the
kind its own arity gives (test 1; the conversion from Lambda takes the
arity from the arguments); a primitive given an argument of another
kind; a `Let`, a continuation parameter, an `Apply_cont` argument, a
`Switch` scrutinee, a block field or a value slot at another kind. For
the last group the type checker refused each of eight attempts.

So every form found is undefined on the machine, by the three tests or
by C1's primitives, and the forms that the check answers no to were not
found. This is experiment and an argument (a value crosses from one
piece of code to another only through a call, a block or closure, or a
continuation of the same function), not a proof. A corpus of the
compiler's own units through the check would be the test of it. The
start of one is V2's fifteen units made from small OCaml sources: each
passes the check before Simplify and after.

If a program is found that the machine runs and Simplify makes
`Invalid`, that is a blocking error of the model.

## 5. The layers

- Undefined where executed, in the machine, with no premise: a call that
  does not agree with its code; a primitive on an argument of another
  kind (C1). Ruled.
- Well-formedness (C0's one definition): a condition that the
  compiler's terms always meet and that a dead branch cannot break. Of
  mine: names in scope, distinct parameters, annotations of the right
  length. They are rows 3, 5 and 9 of C0's note; my series does not
  touch `well_formed_unit`.
- The diagnostic: everything else that `well_kinded` asks. A unit that
  passes has the guarantees of section 3. A unit that fails may be a
  unit the compiler accepts.

What an executed mismatch does where the machine has no test (a value of
another kind bound by a `Let` or as a parameter of a handler; a block
field; a value slot) is the open question of section 9.

## 6. Which statements change

Step A, text only (a kind, a metadata or an arity is now quantified;
the meaning is that the statement holds whatever the annotation):
`apply_indirect_known`; `let_cont_apply_cont`,
`let_cont_apply_cont_one_param`, `let_cont_recursive_apply_cont`;
`rename_let`, `freshen_let`, `rename_handler_params`,
`rename_recursive_handler_params`, `unused_let`;
`project_static_closure`, `let_static_code_refines`;
`project_function_slot_let_closures`, `apply_sibling`,
`project_function_slot_static`, `project_value_slot_static` (C3's);
`step_values_all` (N1's: its premise about a `Let`);
`let_simple_canonical`, `let_refines_in`, `let_simple_refines_in`,
`let_cont_handler_refines_in`, `let_cont_recursive_handler_refines_in`,
`let_const_apply_cont`, `let_simple_apply_cont`, `inline_identity`.
That is twenty-four. C3's example of two functions that call each
other (`parity_code` and what uses it) is annotated at naked immediates,
so that its two statements about runs hold after step B as before.
The rules of `alpha_equivalent` and `alpha_structural` for a `Let` and a
handler carry the kind and the metadata, the same on the two sides.

Step B, meaning: every statement about `step`, `run`, `refines` and
`equivalent` is now about a machine on which more calls are undefined.
For a refinement, "the term before is undefined" covers more. One text
changes: `call_binds_my_closure` (C3's) names `call_function`, and so
quantifies the two arities of the call. Two functions change so that
their statements stay true as written: `inline_known_apply` (the guard
`call_agrees_b`) and the comparison, which was already changed in
step A.

`validate_unit_sound` is as it was: its statement has no premise about
kinds.

## 7. Fallout, measured

Step A: 44 files, 1306 lines added and 804 removed, all mechanical; the
largest are `FlambdaNormaliseProofs.v` and the files of the nominal
part. Step B: 15 files, 327 added and 93 removed; the proofs that
unfold `call_function` each gain a case split on the tests.
`FlambdaWellKinded*`: 809 and 1679 lines.

Validation. There were 54 examples: 38 in `tests/semantics/` (C3's
eleven included) and V2's 16 units made from OCaml sources, in
`tests/semantics/ocaml/`. 47 are converted (the other seven have what
the model has not got, or are not well-formed, as before). Each of the
47 prints what it printed, and one new line, "the check of kinds: passes
before, passes after". So every example passes the check on both sides,
the fifteen units that the compiler made from OCaml sources among them,
and none loses its validation: the counts of "validated" are what they
were (in `tests/semantics/`, 2 up to bound names, 16 once normalised, 5
with unused static bindings removed; of the units from OCaml sources,
9).

What the guard on `inline_known_calls` costs: nothing, on the examples
there are. I ran every example, the new ones included, with the guard
taken out of the extracted function, and none prints anything
different: in none does the term after Simplify keep a direct call of a
symbol whose arguments are not all constants or symbols. It will cost on
larger units. C0's list of variables of known kind is the repair, and is
a follow-up.

Ten examples are new:

| Example | What it shows | What the tool prints |
|---|---|---|
| `kinds_gadt.fl` | the shape of the GADT program: a call in a branch that is not run meets code of another arity once inlined | returns `[0: 5]` before and after; passes, passes |
| `kinds_wrong_arity.fl` | a call at an arity that is not the code's | undefined before (3 steps) and after; passes, passes |
| `kinds_naked.fl` | a call at naked immediates | returns `[0: 7]` before and after; passes, passes |
| `kinds_dead_prim.fl` | a primitive on an argument of another kind, in a branch that is not run | returns `[0: 5]` before and after; passes, passes |
| `kinds_opaque_bad.fl`, `kinds_is_int_bad.fl` | the reviewer's two: a primitive on an argument of another kind, executed | undefined before and after; passes, passes |
| `kinds_dead_cont.fl` | the reviewer's: an `Apply_cont` of the wrong kind in an arm that Simplify removes | returns `[0: 5]` before and after; fails before, passes after; validated once normalised (V2's removal of unused continuations takes the arm away) |
| `kinds_partial_application.fl` | an indirect partial application | "stops at a partial application or an over-application, which the model has not got"; not decided; no validation is attempted |
| `kinds_partial_direct.fl` | a direct one | not checked: the model has not got it |
| `kinds_tupled.fl` | the reviewer's: tupled code of one parameter, called indirectly with the tuple; the arities of the call and of the code agree, and the call is still not the one the model would run | not checked: the model has not got a tupled function |

## 8. Trust

`rocq/TRUST-LEDGER.md` gains the following, in the item on the
conversion of the test tool (the last commit):

- The converter gives each call and each code its two arities. A wrong
  one makes a run undefined that should not be, or the reverse, and the
  check of kinds does not catch it: code that never returns passes with
  any result arity (the reviewer's B4). So the arities are trusted. Two
  cross-checks run at each conversion and stop the tool: the arity as
  converted, unarized by the model, has the kinds that
  `Flambda_arity.unarize` gives; the parameter arity of code, unarized,
  has the kinds of the parameters of its body (discrepancy C2-6: nothing
  in the compiler's `Code0.create_with_metadata` checks this). What is
  not cross-checked: the grouping into products, which the tool rebuilds
  through `Flambda_arity.fold_per_parameter`.
- The table of the kinds of value slots is built from
  `Value_slot.kind` as slots are met.
- The variables of a `Let` of a set of several functions get the kind
  `Value` from `let_closures`; the tool refuses the binding if the
  compiler gives one of them another kind.
- Partial applications and over-applications are not in the model. A
  direct call that the compiler takes for one, of code that the unit
  binds, is refused by the converter (reported as unsupported), by the
  compiler's test of the arities: `partial_or_over_application`, which
  is read against `simplify_direct_function_call`. It is that test and
  no more: a call that passes it and whose arguments are not of the
  kinds of its arity is `Invalid` for the compiler, and a call with no
  argument of code that has parameters, which fails it, is a fatal error
  of the compiler and not `Invalid` (the comment at the definition says
  both). An indirect one cannot be told from the term. A run that stops
  at one is reported as such (`at_partial_or_over_application`, which
  asks also that the values of the arguments are of the kinds of the
  call's arity); the tool then says "not decided", whatever the other
  run does, and does not call `validate_unit`: what that would accept is
  true of the model, where the call is undefined, and of nothing else.
  The tool of the series as first sent printed "validated" there, which
  the review showed. A unit with such a call on a path that the run does not
  take is not detected: that is an omission, and the header of
  `FlambdaWellKindedSpec.v` says so.
- Tupled functions are not in the model: a call of unknown arity gives
  one a single argument, the tuple, whatever the arity of its code (a
  direct call and a call of known arity go by the arity of the code).
  The converter refuses code that is tupled (`Code0.is_tupled`). The
  arities do not show it: the reviewer's example has code of one
  parameter, so the call and the code agree, and without the refusal the
  machine ran it and returned the tuple where the compiler returns its
  field: a wrong result, not an undefined one.

## 9. Decisions, and the comments left for the human

Ruled: D9 (a call at other arities is undefined), D10 for calls and
primitives (undefined where executed), D12 (no check stricter than the
compiler; the compiler is the compiler without
`-flambda2-kind-checks`).

Settled by the orchestrator: D11, what the test tool does with the
check's answer. It prints it for the unit before and for the unit
after, and goes on. The comment before `well_kinded_unit` says so and is
no longer a question.

Open, with a review comment addressed to mshinwell in two `Spec` files
(`FlambdaSemanticsSpec.v` before `step`, and before `well_kinded`):

**D10, the rest.** A value of another kind than the one its binder
gives, bound by a `Let` or as a parameter of a handler; a field of a
static block or a value slot of another kind. The machine does not test
these. (The fields of a block that a primitive makes are tested, as
arguments of the primitive.) The compiler has a hardcoded check for each
and stops where Simplify visits the node, so Simplify never turns one
into `Invalid`, and no well-typed program is known to produce one.

- (a) as built: the machine does not test; the check answers no. The
  statements about well-kinded units have the check as their premise;
  the statements about the compiler's terms have none and say nothing
  of the kinds of what variables hold.
- (b) the machine tests the kind of a value where a variable is bound,
  and the mismatch is undefined where executed, as for calls.

The series lands (a). The orchestrator reads the ruling ("a mismatch of
kinds is undefined where it is executed") as (b), and the types need
the fact that (b) gives, so (b) is expected as a later epoch unless the
human says otherwise. What it is and what it costs, so that the choice
can be made:

*The change.* Two tests in the machine, and nothing in the terms. A
`Let x kind d body` has no step unless the value of `d` has kind `kind`.
The application of a handler has no step unless the values have the
kinds of its metadata, which takes in the count of arguments. A call
needs nothing more (step B). A value slot and a field need no test of
their own: what is read from one is tested by the `Let` that binds it.
An exit through a continuation that is not bound is the interface of
the term and is not tested.

*What it gives.* In every run, with no premise, a variable bound by a
`Let`, or as a parameter of a handler or of code, holds a value of the
kind of its binder (the parameters being distinct). The environment of
kinds that the check computes is then right for every closed term and
not only for those that pass, and conditions 2 and 3 of the check
(section 3) stop being conditions on terms: they say what is undefined.

*Statements that change.*

- `remove_unused_total_lets_equivalent`,
  `remove_unused_defined_lets_equivalent`: the functions must test the
  kind of what they remove. For a constant or a symbol that is a test on
  the term; for a variable it needs the kinds of the variables in scope,
  in place of the list of variables that
  `remove_unused_defined_lets` carries now.
- `thread_switches_equivalent`: the handler of no parameters that is
  threaded through must have no kinds either; a test on the term.
- `inline_known_calls_equivalent` gets simpler. The `Let`s of the
  inlined body test what the call tested, so the guard keeps the
  comparison of the arities and loses its condition on the arguments.
  The list of variables of known kind for the inliner (C0's sketch) is
  then not needed.
- `kind_fault` gains the two reasons, and
  `well_kinded_no_kind_fault` with it.
- The step lemmas: `step_let`, `step_apply_cont_handler`, and C3's
  three for a `Let` of its forms.
- `let_cont_apply_cont` (a handler applied where it is defined runs
  with its parameters bound) gains a premise: the values have the kinds
  of the handler. Its two companions make a `Let` of the parameter at
  the kind of the handler, which tests the same, and are to be checked.
- The eight statements of the types files about a `Let` are to be read
  again by their owner. The two I have read hold as written:
  `let_simple_canonical` keeps its `Let`, and `inline_identity` is a
  refinement whose two sides apply the same continuation to the same
  value.
- Unchanged as written: the refinements (`unused_let`,
  `remove_unused_lets`, `inline_tail_conts`, `collapse_switches`,
  `inline_calls`), the statements about renaming and alpha-equivalence,
  `fold_constants_equivalent` and `follow_aliases_equivalent` (the
  `Let` stays, with the same value), and `validate_unit_sound`.

*Proofs.* 71 proofs in 13 files step through a `Let` or the application
of a handler (35 through a `Let`, 49 through a handler; 9 go by cases on
every construct). Step B touched about a dozen. Most gain one case
split. The work that is not mechanical is the environment of kinds in
`FlambdaNormaliseProofs.v`, and showing in `FlambdaContextProofs.v`
that related environments hold values of the same kinds.

*Estimate.* Three to four times step B: 800 to 1200 lines, as an epoch
of its own. The conversion in the test tool does not change. Every
example is run again; I expect none to lose its validation, as the one
unit that fails the check today (`kinds_dead_cont.fl`, before Simplify)
fails in an arm that is not run.

Recommendation: (a) in this series; (b) as its own epoch, after C5's,
whose forms it would otherwise have to be repaired for twice. Section
12 is (b) as it was then built, on `flambda2-rocq/C2-binders`: read it
for what changes, and this section for the choice.

Not a decision but to be known: partial and over-applications are
undefined on the machine and should not be (section 8). Adding them to
the fragment is the repair; it is not in this request.

## 10. The review of the second version, answered

- B1 (flat arities erase what decides `Invalid`): arities are the
  compiler's trees; no product is rejected; examples for arguments and
  for results.
- B2 (the prototype's evaluator of primitives was wrong on
  `Opaque_identity` and `Is_int`): the series uses C1's `eval_prim`.
  The two negative cases are examples in the Proofs file
  (`example_opaque_undefined`, `example_is_int_undefined`) and in
  `tests/semantics/` (`kinds_opaque_bad.fl`, `kinds_is_int_bad.fl`):
  both pass the check and are undefined before Simplify. (Later: the
  test tool gave each `Opaque_identity` of a unit read from a text
  the kind of its argument, as the text did not write its kind, so a
  text could not say an opaque identity at another kind than its
  argument's. The text writes the kind since, unless it is the kind
  of values, and the tool keeps a kind as read that is not the kind
  of values: a text can say one again, at such a kind, and
  `kinds_opaque_written_bad.fl` does.
  `kinds_opaque_bad.fl` is since then a block made of the naked
  result of one, still a primitive on an argument of another kind,
  executed; the negative case of the primitive itself is the example
  of the Proofs file.)
- B3 (the check is stricter than the compiler on code that Simplify
  removes before it looks): the check is a diagnostic and a sufficient
  condition; headers and this text say "where it visits"; the dead-arm
  unit is an example on each side.
- B4 (the check does not certify the converter's arities): the claim is
  gone; section 8 says what is trusted.
- N1 (`let_params` total on malformed annotations): done in step A.
- N2: the comment on `kind_of_value` is C1's now; the count of
  statements with hypotheses about primitives is none; the prototype
  is not in this tree.
- For the types: section 3, last paragraph.
- C3's forms: done in the rebase over C3's epoch. The check has cases
  for `Closure_of_set` and `Project_function_slot` (a `Value`),
  `Project_value_slot_from` (the kind of the slot),
  `Static_const_closure_of_set`, and the two new closure values; a `Let`
  of `let_closures` or `let_projections` is at kind `Value`. C5's are
  to come with its epoch: the exception continuation of an `Apply` at
  `Value` followed by the kinds of its extra arguments, a case for the
  new form of `cont_value`, and `hm_is_exn_handler`.

The review of the series, as first sent (R2):

- `partial_or_over_application` is the test of arities and not all that
  the compiler asks: the comment says so, names the call with no
  argument (a fatal error of the compiler, not `Invalid`) and the
  tupled case; `at_partial_or_over_application` asks also that the
  values of the arguments are of the kinds of the call's arity, with an
  example on each side.
- The tool printed "not decided" and then "validated" for a pair whose
  runs stop at a partial application: it now attempts no validation
  there, and the ledger says what the wrapper covers.
- Tupled functions: refused by the converter; an example
  (`kinds_tupled.fl`, the reviewer's).
- The count of statements that change in step A was nineteen, not
  twenty; with C3's four and one of N1's it is twenty-four, named in
  section 6.
- The comment for D10 before `step` tells the fields of a static block
  from those of a block made by a primitive, which are tested.
- Section 3: what the machine gives with no premise needs distinct
  binders to be a fact about a lookup, and is about the arguments only;
  section 1: nothing is tested of the values returned.

## 11. The series

On `flambda2-rocq/C2-epoch`, each commit passing `make -C rocq -j16 all`
and `make -C rocq assumptions`:

1. step A: terms carry kinds and arities, and the machine ignores them;
2. step B: a call is undefined unless it agrees with the code called;
3. the check of kinds: `FlambdaWellKindedSpec.v`,
   `FlambdaWellKindedProofs.v`;
4. this request, the status file, my entry in `OWNERS.md`;
5. lines for the shared lists (`rocq/Makefile`, `FlambdaExtract.v`);
6. the test tool, the ledger, ten new examples and the expected files
   (every converted example gains a line), which C0 applies with the
   regenerated extracted files.

Step A can land without step B: it then needs the part of commit 6 that
converts the annotations, which is separable (the tests at a call and
the reports of partial applications are not).

## 12. The kind of a value is tested where it is bound (D10, the other option)

This section is the second series, on `flambda2-rocq/C2-binders`. It is
not in the series above, which landed with the machine testing kinds at
a call only. It is rebased over C5's exceptions ("Over the epoch of
exceptions" and the two sections after it, below), the first step of
C4's heap included. It is built: every file of the two Flambda 2
libraries is repaired, `make all` and the trust check pass, and the
examples are run again (below).

### The two tests

In `FlambdaSemanticsSpec.v`, and nowhere else. The terms do not change.

A `Let` has no step unless the value of its defining expression is of
the kind it gives its variable:

    | State (Let x kind defining_expr body) vars conts statics =>
      match eval_named statics vars defining_expr with
      | Some v =>
        if kind_eqb (kind_of_value v) kind
        then Step (State body ((x, v) :: vars) conts statics)
        else Undefined
      | None => Undefined
      end

The application of a handler has no step unless the values are of the
kinds that the handler gives its parameters:

    | Closure (Handler params metadata handler) def_vars def_conts
        recursive =>
      if kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata)
      then
        match bind_params params vs def_vars with
        | Some handler_vars => Step (State handler handler_vars ...)
        | None => Undefined
        end
      else Undefined
    | Extra_args c extra => apply_cont_value c (vs ++ extra) statics

The last case is C5's, unchanged: the extra arguments of an exception
continuation reach the handler after the exception, and the handler
tests them with it.

Nothing more is needed. A call tests its arguments already (step B),
and `my_closure` is bound to a closure, which is a `Value`. What is
read from a value slot or from a field is tested by the `Let` that
binds it. Poison of the right kind passes, as at a call.

What the tests do not reach:

- an exit through a continuation that is not bound. It is the interface
  of the term, which says nothing of kinds;
- what a field of a static block or a value slot holds, until a `Let`
  binds it to a variable;
- what a call returns, against the result arity of the code. The value
  is tested where it arrives: against the kinds of the handler that the
  return continuation stands for, if it stands for one. The comparison
  of the two return arities at the call (step B) is a comparison of
  annotations, as before;
- the extra arguments of the exception continuation of a call, against
  the kinds that the call gives them. They are tested where they
  arrive, as above.

The review comment before `step` is reworded: it now says that the two
tests are there, that nothing in the compiler says what such a term
means (the compiler stops on one it sees), and what the other reading
would give.

### What it gives

Three statements in `FlambdaWellKindedSpec.v`, with no premise on the
term or on the state. Each is about the step that binds:

| Statement | What it says |
|---|---|
| `let_binds_at_kind` | A `Let` that has a step binds its variable to the value of its defining expression, which is of the kind of the `Let`. |
| `handler_binds_at_kinds` | A handler that runs has its parameters bound to values of the kinds it gives them, one kind for each value. |
| `call_binds_at_kinds` | Where `call_code` has a step, the body of the code called runs with its parameters bound to values of their kinds, over an environment that binds `my_closure` to the closure called. (That `my_closure` stands for the closure in the body needs that no parameter has its name: `call_binds_my_closure`, C3's.) |
| `called_closure_at_kind` | The closure that a call gives `call_code` is of kind `Value`. |

The third is about `call_code`, the one place where the body of code
is entered, and not about an `Apply`: with partial applications and
over-applications an `Apply` that has a step need not enter a body
with all its arguments, and may enter none (the reviewer's point,
agreed with C3). It was first stated of `call_function`; the series
for partial applications and over-applications restated it of
`call_code`, which is what every call that runs code comes to (a call
with as many arguments as the code has parameters, the full call
inside an over-application, and the call of a partial application),
and added the fourth, about the closure.

They are not a statement that a state is well-kinded, and the request
does not claim one: `state_well_kinded` still needs the check as its
premise. What follows from the three is that a variable, where it is
read, holds a value of the kind of the binder that the term has in
scope for it, provided no other binder of the same name comes between
the step that bound it and the read. That proviso is the scoping of the
term (distinct binders, section 3), and nothing in the series proves
the consequence as one statement.

How a proof gets the fact, without unfolding the machine:

- `FlambdaSemanticsProofs.let_refines_step` and `let_equivalent_step`
  give the hypothesis about the body one more fact,
  `kind_of_value v = kind`. Both sides are a `Let` at the one kind.
- `FlambdaContextProofs.handler_refines_at` and
  `handler_refines_where` take the kinds of the handler, and give the
  hypothesis about the handler `map kind_of_value vs = kinds` beside
  `bind_params params vs def_vars = Some handler_vars`. So does the
  relation of `transfer`.

`kind_fault` has the two reasons more (`wrong_kind_bound`,
`wrong_kinds_of_arguments`), so `well_kinded_no_kind_fault` and
`well_kinded_unit_never_faults` say more with the same text.

### Statements that change

No statement of the validator gains a premise: `validate_unit_sound`
and `validate_unit_sound_everywhere` keep their text, and no pass
needs "the unit is well kinded". A pass that must stay an equivalence,
and whose rewrite is right only where a kind is the expected one,
tests that on the term, with the kinds of the variables in scope,
which the binders now guarantee. Where the test fails the pass leaves
the term as it is.

Statements whose text changes: three in `FlambdaControlFlowSpec.v`,
each by one premise, one of N3's in
`FlambdaValidatorEquivarianceSpec.v`, where another goes, and one of
N2's in `FlambdaKeepsScopeSpec.v`.

| Statement | The change |
|---|---|
| `let_cont_apply_cont` | new premise `map kind_of_value vs = hm_params_kinds metadata` |
| `let_cont_recursive_apply_cont` | new premise `map kind_of_value vs = hm_params_kinds metadata` |
| `let_cont_apply_cont_one_param` | new premise `hm_params_kinds metadata = [kind]` |
| `remove_unused_defined_lets_equivariant` | about `kinds_in_scope` in place of a list of variables |
| `remove_unused_defined_lets_keeps_scope` | about `kinds_in_scope` in place of a list of variables |
| `simple_has_kind_b_equivariant` | removed with its function |

- `let_cont_apply_cont`: its premises now say when the application has
  a step: the arguments have values, of the kinds of the handler, and
  as many as there are parameters. Without the new one the statement
  is false: the application is undefined, and the handler under its
  parameters need not be.
- `let_cont_recursive_apply_cont`: the same, for the same reason. Its
  right-hand side is the handler under its parameters, with no `Let`.
- `let_cont_apply_cont_one_param`: the `Let` of its right-hand side is
  at `kind`, which the statement quantified apart from the metadata.
  The `Let` tests of the value what the application tests only if
  `kind` is the kind that the handler gives. (The reviewer's example
  against the statement as it was: metadata of one naked immediate,
  `kind` `Value`, a tagged immediate as the argument.)
- `remove_unused_defined_lets_equivariant`: the function commutes with
  a renaming when the variables in scope are renamed with the term and
  their kinds are kept (`rename_kinds_in_scope`, which is new, beside
  `kinds_in_scope` in `FlambdaNormaliseSpec.v`). The same shape as
  before. `call_agrees_b_equivariant` keeps its text. N3 was told
  before the change.
- `remove_unused_defined_lets_keeps_scope`: the function keeps scope
  whatever it is given as the variables in scope, which are now given
  with their kinds. It says what it said: nothing is asked of what is
  given.

Statements whose text is the same and whose function changes: four,
all in `FlambdaNormaliseSpec.v`. Each is still an equivalence.

| Statement | What changes |
|---|---|
| `remove_unused_total_lets_equivalent` | `named_is_total` takes the kind of the `Let` |
| `remove_unused_defined_lets_equivalent` | the function carries the kinds of the variables in scope; `named_defined` takes the kind of the `Let` |
| `thread_switches_equivalent` | `thread_switch` asks that the handler give no kinds |
| `inline_known_calls_equivalent` | `call_agrees_b` loses its condition on the arguments |

- `named_is_total kind d`: a constant must be of that kind, and for a
  symbol the kind must be `Value`.
- `remove_unused_defined_lets` carries the variables in scope with the
  kinds of their binders (`kinds_in_scope`: a `Let` adds its variable at
  its kind, a handler its parameters at its kinds, by
  `params_in_scope`), in place of the list of variables.
  `named_defined kinds kind d`: a `Simple` must be of that kind
  (`kind_of_simple_in_scope`), and for a closure the kind must be
  `Value`. The statement is still about
  `remove_unused_defined_lets []`.
- `thread_switch` threads through a handler of no parameters only if it
  gives no kinds either. A handler with no parameters and one kind
  never runs, and an arm that applies it is undefined; threaded, the
  arm would be defined.
- `call_agrees_b` no longer asks that each argument be a constant or a
  symbol of the kind of the arity: the `Let`s of the inlined body test
  what the call tested. It keeps the two comparisons of arities, and
  asks that the code give as many kinds as it has parameters, without
  which a `Let` of the inlined body would be at a default kind. More
  calls are inlined, in the term before and in the term after.
  `simple_has_kind_b` is gone.

Lemmas that others use (in `Proofs` files, so not statements to
review): `step_let` and the three step lemmas for a `Let` of a closure
of a set or of a projection have the test;
`step_apply_cont_handler` has the premise on the kinds;
`let_params_run` has the premise that the values are of the kinds
given; `inline_equivalent` has the premise that the code gives as many
kinds as parameters in place of the one on the kinds of the values;
`unarize_per_parameter_eq` and `closure_of_kind` moved from
`FlambdaWellKindedProofs.v` to `FlambdaSemanticsProofs.v`. T2's
`handler_refines_under_where` takes the kinds of the handler and does
not use them: `let_cont_handler_refines_under` could now ask its
hypothesis at the kinds of the handler only, which is T2's to choose.

Unchanged as written, proofs repaired: every refinement (`unused_let`,
`remove_unused_lets`, `inline_tail_conts`, `collapse_switches`,
`inline_calls`, `load_known_blocks_refines`,
`unbox_known_numbers_refines`, `plug_refines`); the equivalences that
keep the `Let` they rewrite (`fold_constants`, `follow_aliases`) or
remove no `Let` (`remove_unused_conts`, `merge_statics`,
`drop_deleted_code`, `plug_equivalent`); the statements about
renaming, alpha-equivalence and the comparison (the comparison asks
the same kinds and the same metadata on the two sides, so the two
tests answer alike), `step_equivariant` among them (a renaming keeps
kinds and values); the statements about sets of closures (their
`Let`s are at `Value`, and closures are); the statements of the types
about a `Let` or a handler in `FlambdaTypesSpec.v`,
`FlambdaAliasesSpec.v` and `FlambdaTypingEnvRefinementSpec.v`; the
other statements of `FlambdaValidatorEquivarianceSpec.v`.

V3's statements are refinements on the term before, and keep their
text. V4's passes for loops are in "Over the passes for loops" below:
two of their definitions gain a test.

### Measured

The compiler builds on the new extraction with no change to the test
tool, which calls the passes through the validator. Before the epoch
of exceptions, the 48 examples of `tests/semantics/` and the 16 of
`tests/semantics/ocaml/` printed what they printed, to the letter. So
no example loses a validation to the test of `thread_switch` or to
the tests of the two functions that remove a `Let`, and there is no
choice to put to the human there. None gained one either: as section
7 measured, in none of them does the term after Simplify keep a direct
call of a symbol whose arguments are not all constants or symbols,
which is what the guard of `inline_known_calls` refused.

Among the examples added since, two are such: `raise_in_call.fl` and
`try_in_code.fl` call known code with a variable as the argument. They
go from "not validated" to "validated: the same once normalised", as
the call is now inlined on the two sides; their expected files change,
in the last commit of the series. Every other example prints what it
printed: 97 of the 99 that there are now (82 in `tests/semantics/`
and 17 in `tests/semantics/ocaml/`).

No example runs a `Let` or a handler on a value of another kind: the
compiler stops on such a term where Simplify visits it, so the test
tool cannot be given one that is executed. `kinds_dead_cont.fl` has
one in an arm that is not run, and prints what it printed.

### Over the epoch of exceptions

The series is replayed over C5's epoch. What that changed:

- `apply_cont_value` is C5's fixpoint, with the test in its case of a
  handler (above). `handler_binds_at_kinds` keeps its text: it is about
  a handler, and a continuation with extra arguments comes to one.
- `wrong_kinds_of_arguments`, the reason of `kind_fault`, follows the
  extra arguments as C5's `wrong_number_given` does: a new
  `wrong_kinds_given c given` compares, at the handler inside, the
  kinds given followed by those of the extra arguments.
  `well_kinded_no_kind_fault` keeps its text; its proof has one lemma
  more (`right_kinds_given`).
- The machine with a stack (`FlambdaTrapSpec.v`) is defined through
  `step` and `apply_cont_value`, so it has the two tests with no
  change, and `stack_agrees_statement` keeps its text. Its proofs get
  two case splits.
- `thread_switch` has C5's test (an arm with a trap action stays) and
  this series' (the handler gives no kinds).
- `step_apply_cont_target_handler`, C5's step lemma for a continuation
  that may have extra arguments, has the premise on the kinds, as
  `step_apply_cont_handler` has.
- `inline_equivalent` is the proof of this series with C5's condition
  that the call has no extra arguments.
- Nothing else of C5's needed more than the case splits: twelve, in
  proofs of `FlambdaSemanticsProofs.v`, `FlambdaScopeProofs.v`,
  `FlambdaRangeProofs.v`, `FlambdaEquivarianceProofs.v`,
  `FlambdaClosuresProofs.v`, `FlambdaWellFormedProofs.v`,
  `FlambdaWellKindedProofs.v`, `FlambdaContextProofs.v` and
  `FlambdaTrapProofs.v`.

### Over the generalised `transfer`, lifting, and the primitives added

The series is replayed again, over what landed after the epoch of
exceptions. Definitions and statements of others that change:

- `FlambdaContextWhereSpec.handler_refines_from` takes the kinds that
  the handler gives its parameters, and hands its premise
  `map kind_of_value vs = kinds` before `bind_params`, as
  `handler_refines_where` and `handler_refines_at` do. The two
  statements about a handler replaced
  (`let_cont_handler_refines_from_statement`,
  `let_cont_recursive_handler_refines_from_statement`) ask it at
  `hm_params_kinds metadata`. The three statements that the premises
  can be met give the fact to their hypothesis on the handler
  (`handler_refines_from_always_statement`,
  `handler_refines_from_in_range_statement`,
  `let_cont_handler_refines_in_range_statement`): each asks less of
  the handler than before. The paragraph of the header that said that
  nothing is given about the kinds of the parameters of a handler now
  says what is given. The owner of the file agreed to the change
  beforehand and reads it.
- In `FlambdaContextProofs.v`, the relation of `transfer` between
  handlers carries the kinds, and its two hypotheses about related
  handlers have the fact (`inv_related`, `carried_related`); so do the
  relations `forwards` and `backwards` of the lemmas from a predicate.

Proofs of others repaired, statements unchanged: lifting a `Let` to a
static binding (a lifted value has the kind of the value, which the
proof had as a lemma: the `Let` that is left is of a symbol, of kind
`Value`, and tests what the `Let` of the block tested), `my_closure`
bound to a symbol, the removal of unused value slots, the identities
of arithmetic (the `Let` is kept at its kind, with the same value),
following aliases under a typing environment: the same case splits,
and an argument more to an `intros` where a lemma hands the fact.

### Over the first step of the heap

The first step of the heap has a machine of its own beside `step`
(`FlambdaHeapMachineSpec.heap_step`): a copy of it with one component
more.
The two tests go into the copy, in two definitions of that file:

- `heap_apply_cont_value`, in its case of a handler: no step unless
  `kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata)`;
- `heap_step`, in its case of a `Let`: no step unless
  `kind_eqb (kind_of_value v) kind`, of the value that the defining
  expression has.

Without them `heap_step_agrees_statement`, which says that the two
machines step alike where no primitive of the heap runs, is false.
With them every statement of that file keeps its text. The relation
between two runs relates values of the same kind (the file had the
lemma, for the test of a call), so the two sides test alike. The eight
statements about rewrites need no premise more:
`delete_unused_let` (a `Let` whose test fails is undefined, which any
term refines), `exchange_lets` and `cse_adjacent` (the same values are
tested on the two sides), `load_known_field` (the same value at the
same kind), and the three liftings (a block or a closure and the
symbol it is lifted to are both of kind `Value`, so the `Let` of the
symbol tests what the `Let` of the allocation tested);
`delete_write_wrong` keeps its witness. About forty-five lines of its
proofs change, and four of the proofs that the machine with a heap
commutes with renaming. The owner of the file agreed to the change
beforehand; the reviewer reads the two definitions.

Proofs of others repaired with their statements unchanged, in what
landed between the two replays: sharing a primitive already computed,
folding what a unit binds its symbols to, the arms of a `Switch` that
a type excludes, a primitive replaced by a constant or by a `Simple`
from the types. In each the `Let` is kept at its kind with the same
value, so the two sides test alike.

### Over the passes for loops

The six passes that normalise loops landed before the series, which
is replayed over them. Two definitions under their statements change,
and the two statements keep their text:

- `FlambdaShortcutSpec.shortcut_switch` asks, as `thread_switch`
  does, that the handler whose `Apply_cont` it puts in an arm give no
  kinds. A handler of no parameters that gives a kind never runs, so
  an arm that applies it is undefined, and the arm made of its
  `Apply_cont` would not be: without the test
  `shortcut_switches_equivalent_statement` is false.
- `FlambdaLoopifySpec.loopable` asks that no name be two of the
  parameters of the code (`FlambdaMapSpec.distinct`). The looped code
  enters its loop by a jump to `self` with the parameters as its
  arguments, and the handler of `self` gives the kinds of the
  parameters. Were one name two parameters of different kinds, both
  would have the value of the second at the jump, which would have no
  step where the call had one: without the test
  `loopify_unit_refines_statement` is false. The compiler binds no
  name twice in one list of parameters, so no code that it makes is
  left alone for this.

`FlambdaUnusedParamsSpec.drop_unused_params` already removed the kind
of a parameter with the parameter (`keep_kinds`): the values kept are
of the kinds kept, and it needs no change. A self tail call that
`loopify_unit` makes a jump is a call that has a step, so its
arguments are of the kinds of the parameters of the code, which are
those that the handler of `self` gives.

Proofs of these passes repaired, statements unchanged. The relation
of the proof of `drop_unused_params` says that the second handler
gives the kinds kept. The proof of `loopify_unit` has two lemmas more
(distinct parameters, once bound, have the values they were bound to;
a call that agrees with its code gives values of the kinds of its
parameters). Two continuations that are alike, the handlers of a
recursive `Let_cont` rewritten one by one, a `Let_cont` sunk, the
aliases of a loop: the same case splits, and an argument more to an
`intros`. The proofs that `shortcut_switch` and `loopable` commute
with renaming have a line more each.

The statements that the rewrites of a round keep scope
(`FlambdaKeepsScopeSpec.v`) landed with them. One changes its text,
in the table above; the proof that `thread_switch` keeps scope has a
case split more. The other fifteen go through the rewrites and not
through the step, and are untouched.

### For a pass that is being written now

A rewrite that replaces the defining expression of a `Let` by another
of the same value, and keeps the `Let`, needs nothing. A rewrite that
removes a `Let`, or a handler with parameters, or moves a value past
one (a parameter replaced by the argument, a binding lifted to a static
one) changes what is tested: it stays a refinement in the direction
"the result is defined where the term was", and it is an equivalence
only where the kind is known on the term. A `Let` that a rewrite makes
must give the kind of the value it binds.

### Texts of other files that the series makes stale

The series changes no document or comment that is another owner's.
These say what the machine does with the kind of a binder, and are to
be reworded by their owners when it lands:

- `rocq/HACKING.md`, the paragraph "The terms carry the compiler's
  kinds": "Execution ignores all of it but at a call". To say instead:
  execution tests them where a variable is bound, at a call, at a `Let`
  and where a handler is applied, so the step that binds a variable
  binds a value of the kind of its binder
  (`FlambdaWellKindedSpec.let_binds_at_kind`,
  `handler_binds_at_kinds`, `call_binds_at_kinds`); a proof gets the
  fact from `let_refines_step`, `let_equivalent_step` and
  `handler_refines_at`; the two tests of the binders are decision D10,
  with a review comment before `step`. The paragraph after it lists
  the faults of `kind_fault`, which has two more. The advice on
  writing a rewrite could gain the paragraph "For a pass that is being
  written now" above.
- `DECISIONS.md`, the line of D10: "the machine does not test yet".
- `rocq/types/env/FlambdaTypingEnvRefinementSpec.v`, the paragraph "Kinds"
  of its header ("the machine looks at neither: it binds the value
  that it is given, of whatever kind") and the comments before
  `let_refines_under_statement` and
  `let_simple_refines_under_statement` ("is not read: the machine does
  not look at it"). The statements hold as written. With the series
  they could ask less: the value bound by a `Let` is of the kind the
  `Let` gives, and the values given to a handler are of the kinds of
  its parameters, so the premise on the kind of the value and the
  quantification over every choice of kinds are no longer needed
  where the kind is the binder's.

### The series, and what is left

One commit of the model and its proofs, which cannot be cut smaller
(the two tests break every proof that steps through a `Let` or a
handler), between the plan and this document. 27 files change, about
1020 lines added and 460 removed. It is rebased over the integration
branch as it is with C5's epoch of exceptions. The reviewer has read
the three new statements, the changed ones and the four functions of
`FlambdaNormaliseSpec.v` on the version before that rebase, and reads
what each rebase changed.

## What is unfinished

- No large corpus of the compiler's own units has been through the
  check or the machine. The evidence is programs written for the
  purpose, and V2's fifteen small units made from OCaml sources, which
  pass.
- The compiler's hardcoded checks were exercised by hand-written terms
  that are not in this tree, as the test tool does not catch a fatal
  error of the compiler.
- The cases for C5's forms wait for its epoch.
- Partial applications, over-applications and tupled functions are not
  in the fragment (section 8).
- The list of variables of known kind for `inline_known_calls` (C0's
  sketch) is not built; section 7 says what is lost without it. The
  second series (section 12) makes it unnecessary: the inlined body
  tests what the call tested.
- V2's newer units made from OCaml sources have not been run on the
  second series.
