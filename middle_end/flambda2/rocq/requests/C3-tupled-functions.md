# C3: tupled functions. A core change request

From C3, for C0 (who applies it), R2 (who reviews it), T3 (whose
canonicalisation of call kinds section 7, choice 1, is about), the
owners of the files in which a call kind gets a case (section 6) and
the human (section 7: the choices taken while he was away).

It is written over the series on partial applications and
over-applications (`requests/C3-partial-application.md`), whose
definitions it changes, and it lands after that series. It is three
series (section 9). The first is made, and this request comes with
it; the other two are not started.

## Why

`let f (x, y) = x + y` is a tupled function. Native code compiles it
to code with one parameter for each component of the tuple, and marks
the code; a call that does not know its callee still gives the
function one argument, the tuple. Such functions are ordinary in OCaml
sources. Today the code of the model has no mark, a call gives the
code its arguments as they are, and the test tool refuses every unit
that has tupled code.

## 1. What the compiler does

- **The front end** (`lambda/translcore.ml`,
  `transl_tupled_function`). A function whose one parameter is matched
  against tuple patterns, that is allocated on the heap, in native
  code, with no more components than `Lambda.max_arity` and every
  component of a value sort, becomes a function of kind `Tupled` with
  one parameter for each component.
- **Closure conversion** (`from_lambda/closure_conversion.ml`). The
  code has those parameters, all of kind `Value`, and
  `Code_metadata.is_tupled`. An application whose callee is known to
  be such a function is not made direct: it is an
  `Indirect_unknown_arity` call with the arguments as written, the
  tuple first (`close_exact_or_unknown_apply`). Where closure
  conversion expands a partial or an over-application itself
  (`close_apply`), it takes the function to have one parameter, a
  block of tag 0. It does not inline a tupled function.
- **Simplify** (`simplify/simplify_apply_expr.ml`). The comment of
  `simplify_function_call` gives the rule: the function declaration
  has one parameter, the tuple; the code has one for each field; a
  direct call and a call of known arity are in the convention of the
  code; a call of unknown arity is in the convention of the
  declaration. When it finds the code of the callee of an
  unknown-arity call and that code is tupled
  (`call_must_be_detupled`), `simplify_direct_tuple_application`
  rewrites the call: with `n` the number of parameters of the code,
  unarized, it binds `n` variables to the fields 0 to `n - 1` of the
  first argument, each by a `Block_load` whose access kind is
  `Values` with tag 0, size `n` and field kind `Any_value`
  (`Simplify_common.project_tuple`), and makes the call direct, with
  those variables followed by the arguments that remain, at the arity
  of `n` values followed by the arity of the call less its first
  parameter. If arguments remain it splits the call at once
  (`split_direct_over_application`). A call with no argument is a
  fatal error. It then simplifies the new term. It never makes a call
  of unknown arity one of known arity where a code that the callee may
  have is tupled.
- **The backend.** The closure of a tupled function has, for its
  generic entry point, `caml_tuplifyN` (`backend/cmm_helpers.ml`,
  `tuplify_function`), which reads the fields 0 to `N - 1` of its
  argument and calls the code with them; it tests neither the tag nor
  the size of the argument. The generic application (`caml_applyN`)
  never takes such a closure for a function of `N` parameters: it
  gives it the first argument and applies the result to the rest. A
  direct call and a call of known arity jump to the code.

So a closure of tupled code is called in one of two ways: with a
tuple, by a call that does not know it; with the fields, by a call
that Simplify has rewritten.

## 2. The design

- **Code is marked**: its metadata says whether it is tupled
  (`Code_metadata.is_tupled`).
- **Two conventions.** In the *convention of the code* a call gives the
  code its arguments as they are; that is every call of today. In the
  *generic convention* a call of a closure of tupled code gives it one
  argument, the tuple, followed by any others; for code that is not
  tupled the two are the same.
- **A generic call of tupled code** with the arguments `tuple :: rest`
  at the arity `first :: rest_arity`:
  - the arguments must be of the kinds of the arity of the call, as
    for every call, and `first` must be one `Value`;
  - the tuple must have the `n` fields that Simplify's projections
    read, `n` being the number of parameters of the code, unarized:
    each `Block_load` at tag 0, size `n` and `Any_value` has a value on
    it. So it is an immutable block of tag 0 with exactly `n` fields of
    kind `Value`;
  - the call is then the call, in the convention of the code, of the
    fields followed by `rest`, at the arity of `n` values followed by
    `rest_arity`: a full call when nothing remains, an over-application
    otherwise, by the rules of today;
  - with no argument it is undefined.
- **Which calls are generic.** `Indirect` and `Direct`: every call of
  the fragment, in the first series. The compiler's direct call of
  tupled code, which is in the convention of the code, is a third call
  kind, `Detupled`, added by the second series; until then the test
  tool refuses such a call (section 7, choice 1).
- **In the convention of the code, a call of tupled code with fewer
  arguments than parameters is undefined** (section 7, choice 2).

`step`, `call_code`, the shapes of a call and what is done with each
do not change. A partial application is never of a closure of tupled
code.

## 3. The definitions

Of the first series. `FlambdaModel.v`:

    Record code_metadata : Type := Code_metadata {
      cm_params_arity : arity;
      cm_result_arity : arity;
      cm_is_tupled : bool
    }.

    (* [make_code_metadata] makes code that is not tupled, as before;
       [make_tupled_code_metadata] makes tupled code. *)

    Definition code_is_tupled (c : code) : bool := ...

`code_metadata_eqb` compares the mark too.

`FlambdaSemanticsSpec.v`:

    (* The access kind of Simplify's projections of a tuple. *)
    Definition tuple_access (size : nat) : block_access_kind :=
      Block_access_values (Some 0%Z) (Some (Z.of_nat size)) Any_value.

    (* The fields [field], [field + 1], ... of a tuple, [count] of
       them, as those projections read them. *)
    Fixpoint tuple_fields_from (statics : static_env) (size : nat)
        (tuple : value) (field count : nat) : option (list value) :=
      match count with
      | O => Some []
      | S count =>
        let* v :=
          eval_prim statics
            (Unary
               (Block_load (tuple_access size) Immutable
                  (Z.of_nat field)))
            [tuple]
        in
        let* vs :=
          tuple_fields_from statics size tuple (S field) count
        in
        Some (v :: vs)
      end.

    Definition tuple_fields (statics : static_env) (size : nat)
        (tuple : value) : option (list value) :=
      tuple_fields_from statics size tuple 0 size.

    (* The arguments of a call of tupled code as the code takes them,
       with their arity. *)
    Definition detuple (statics : static_env) (size : nat)
        (vs : list value) (args_arity : arity)
        : option (list value * arity) :=
      match vs, args_arity with
      | tuple :: rest, first :: rest_arity =>
        if
          negb
            (kinds_eqb (map kind_of_value vs) (unarize args_arity)
             && kinds_eqb (unarize_component first) [Value])
        then None
        else
          let* fields := tuple_fields statics size tuple in
          Some
            (fields ++ rest,
             arity_of_kinds (repeat Value size) ++ rest_arity)
      | _, _ => None
      end.

    Definition arguments_for_code (statics : static_env) (c : code)
        (vs : list value) (args_arity : arity)
        : option (list value * arity) :=
      if code_is_tupled c
      then
        detuple statics (cardinal_unarized (code_params_arity c)) vs
          args_arity
      else Some (vs, args_arity).

`shape_of_closure_call` is cut in two. `shape_of_code_call closure c
vs args_arity return_arity` is its body of today, from the comparison
of the numbers on, with one more test: in the branch of a partial
application, code that is tupled gives no shape. And, in
`shape_of_closure_call`, where the code `c` of the closure has been
found:

    let* '(vs', args_arity') :=
      arguments_for_code statics c vs args_arity
    in
    shape_of_code_call closure c vs' args_arity' return_arity

Other definitions that gain a test:

- `inlinable` (`FlambdaApplySpec.v`), its boolean in the validator and
  its variant at fresh variables ask that the code is not tupled: the
  body is inlined with the arguments bound to the parameters, which a
  call of tupled code does not do;
- the boolean of that test in the normalisation
  (`FlambdaNormaliseSpec.inlinable_b`) has the same conjunct, so the
  validator's inlining of a call of known code
  (`inline_known_apply`, which asks `inlinable_b` and
  `call_agrees_b`) leaves a call of tupled code alone;
  `call_agrees_b` itself does not change.

The second series adds the call kind:

    Inductive call_kind : Type :=
      | Direct (code : code_id)
      | Detupled (code : code_id)
      | Indirect.

`Detupled` is as `Direct` wherever the call kind is read for the code
it names (free names, renaming, the comparison of terms, the code ids
that a term mentions, the canonicalisation by aliases, which leaves it
as it is), and in `shape_of_closure_call` it goes by
`shape_of_code_call` with the arguments as they are.

## 4. Statements that change

None changes for a unit with no tupled code. Where a statement is
about a call that gives the code its arguments as they are, it gets
the premise `code_is_tupled c = false`:

- `FlambdaPartialApplicationSpec.v`: `apply_partial_statement`,
  `partial_then_apply_statement`, `apply_over_statement`,
  `over_application_split_statement`;
- `FlambdaWellKindedSpec.call_at_wrong_kinds_undefined_statement`, in
  its cases that compare the arity of the call with that of the code;
- `inline_identity_statement` (`rocq/types/naive/FlambdaTypesSpec.v`, the
  stream of types): the premise `cm_is_tupled metadata = false`, which
  `inlinable` now asks.

`FlambdaClosuresSpec.call_binds_my_closure_statement` and
`well_formed_call_binds_my_closure_statement` do not change in text
and stay true: where the call has as many arguments as the code has
parameters or more, it runs the body with `my_closure` bound, whether
the code is tupled or not. (For tupled code they say less than they
could: a call with one argument runs the body of code with two
parameters. `apply_tupled_statement` says that.)

`FlambdaSemanticsSpec.step_values_all_statement` gets one more
hypothesis: the property holds of the fields of a tuple of which it
holds, under the static environment of the state. A call of tupled
code binds those fields, as a `Let` of a projection would.

Not changed: `apply_indirect_known_statement` (an indirect call of a
known closure is a direct call of its code), `apply_sibling_statement`
and the statement of the canonicalisation by aliases. That is what
choice 1 of section 7 buys.

## 5. New statements

In a new file, `FlambdaTupledSpec.v`, proved in
`FlambdaTupledProofs.v`. First series:

- `tuple_is_block_statement`: where `tuple_fields statics n v` gives
  fields and `n` is not 0, `v` is, or is a symbol bound to, an
  immutable block of tag 0 whose fields they are, `n` of them, all of
  kind `Value`; and conversely. (A tuple of no component is tested for
  nothing, as Simplify makes no projection of it.)
- `apply_tupled_statement`: a call, `Direct` or `Indirect`, of a
  closure of tupled code whose parameters are single values, with one
  argument, a tuple with a field for each parameter, is the full call
  of that code with the fields.
- `apply_tupled_over_statement`: with more arguments, of the kinds of
  its arity and at arities that pass the compiler's test, it is that
  full call, at the result arity of the code, returning to what is
  left of the application, which keeps the other arguments.
- `apply_tupled_undefined_statement`: with no argument, or an arity
  whose first parameter is not one `Value`, or a first argument that
  is not such a tuple, the call is undefined.
- `tupled_never_partial_statement`: no call of a closure of tupled
  code is a partial application (choice 2 of section 7).
- Examples: three units that bind tupled code and call a closure of it
  with a tuple that is a symbol, with one that a primitive makes
  (a direct call), and with a tuple and one more argument (an
  over-application whose result, read out of the tuple, is called);
  they run, are well-formed and pass the check of kinds
  (`examples_run_statement`, `examples_well_formed_statement`). Two
  more, given a triple and a block of tag 1, go wrong
  (`wrong_tuple_undefined_statement`).

Second series, with `Detupled`:

- `detuple_statement`: a call of a closure of tupled code that gives
  it a tuple is refined by the term that Simplify makes of it: the
  fields bound by `Let`s to variables that occur nowhere in the call,
  then the `Detupled` call of those variables followed by the other
  arguments. With `over_application_split_statement`, which then
  applies to the `Detupled` call, that is the whole of
  `simplify_direct_tuple_application`.
- `apply_detupled_partial_undefined_statement`: a `Detupled` call of
  tupled code with fewer arguments than parameters is undefined.

Third series: the inlining of a call that gives tupled code a tuple.
Its statements are proved, as leaf work after the first series. A
direct call that gives tupled code a tuple and nothing else is
refined by the body of the code with its parameters bound to the
fields of the tuple, each by the projection that Simplify makes
(`inline_tupled_statement`; `inlined_tupled_body`, under the
conditions `tupled_inlinable`, which are those of `inlinable` but for
the arguments); and the two are equivalent where the callee is a
closure of that code, the call returns at the result arity of the
code and the value of the tuple is no location
(`inline_tupled_equivalent_statement`), which is the form that a
rewrite of the term after Simplify needs. Of a location the two need
not be equivalent: the call, which reads no cell, is undefined, and
the projections read the cell that the heap binds the location to,
and may have their values. The refinement asks nothing of the tuple.
The validation uses the
refinement: `FlambdaNormaliseSpec.inline_apply`, which inlines the
direct calls of known code in the term before, has a second case, for
a call that gives known tupled code one argument
(`tupled_inlinable_b`, a check that is enough for
`tupled_inlinable`; `inlined_tupled_call`). The rewrite commutes with
renaming and keeps
scope as it did. With it the units in which Simplify inlines a tupled
function validate (`tests/semantics/tupled_apply_inlined.fl`,
`tupled_apply_inlined_twice.fl`, `kinds_tupled.fl`), where the
validation makes the call direct first: the callee is a symbol, or a
variable bound to a closure of a set of one function with no named
slot. `tests/semantics/ocaml/tupled_function.ml` is not there yet: its
code projects its own function slot, so its closures are converted as
closures of sets at named slots, and a call of a variable bound to
such a closure is not made direct
(`types/naive/FlambdaAliasesSpec.typing_env_under_let`). The
equivalence is not used yet (`inline_known_apply`, for a callee that
is a symbol).

## 6. Fallout

- `FlambdaModel.v`: the record, `make_tupled_code_metadata`,
  `code_is_tupled`; in the second series, the call kind.
- `FlambdaSemanticsSpec.v`, `FlambdaSemanticsProofs.v`: the
  definitions above; the lemmas about the pieces of a call
  (`shape_of_closure_call_cases` and what follows from it) go through
  `shape_of_code_call`, so the invariants proved through them
  (`call_function_lands`, the traversal of values, validity, scope,
  ranges, kinds) need one new fact each: what the property is of the
  fields of a tuple, which for every one of them is what it is of the
  fields of a block.
- Equivariance (`FlambdaEquivariance*`): `tuple_fields` and `detuple`
  give the same under a renaming. A renaming acts on the variables
  and the continuations of terms. Values hold neither: they name
  symbols, code ids and slots, which are not renamed. The renamed
  static environment binds the same symbols to the same values, and
  the primitives that read the tuple do not look at the code
  (`FlambdaPrimitiveProofs.eval_prim_same_static_values`). Arities
  have no names.
- The relations between two runs under two static environments
  (`FlambdaCompareProofs.v`, `FlambdaContextProofs.v`,
  `FlambdaUnusedStaticsProofs.v`): the primitives read the tuple alike
  under the two, which each of them has already for a `Let` of a
  primitive; and the two bind code with the same mark.
- The simulations in which the values of the two runs differ
  (`FlambdaLiftProofs.v`): a tuple and its lifted form have related
  fields, by the law that the primitives respect a relation that the
  observations respect (`arguments_for_code_related`). Those in which
  the values are the same and the static environments differ
  (`FlambdaUnusedSlotsProofs.v`, `FlambdaKnownSymbolsProofs.v`): the
  primitives read a value alike under the two
  (`arguments_for_code_mono`). The repairs in the series are
  stand-ins, for the owner of those files to read.
- The machine with a stack (`FlambdaTrap*`): nothing, as it goes by
  `shape_of_call`.
- The machine with a heap (`FlambdaHeap*`): no definition changes, as
  `heap_call_function` goes by `shape_of_call`; a tuple is an
  immutable block, never a location, and is read by `eval_prim`. The
  simulation between two runs relates the shapes of two calls of
  related values by unfolding the definition of a call: it needs that
  related tuples have related fields, which its relation gives by the
  same law of the primitives; and the invariant that every location
  of a state is allocated needs that the fields of a closed tuple are
  closed. The repair in the series (`FlambdaHeapMachineProofs.v`, two
  lemmas and one new one) is a stand-in, for the owner of those proofs
  to read. No statement of `FlambdaHeapMachineSpec.v` changes.
- Loopify (`FlambdaLoopifySpec.v`): a self tail call is made a jump
  that binds the parameters to the arguments of the call; of tupled
  code the call would take its first argument apart. So `loopable`
  gets the conjunct that the code is not tupled, as `inlinable` does.
  The change, and the repair of the proofs of loopify and of its
  equivariance, are stand-ins for the owner of those files to read.
- Second series: `FlambdaCompareSpec.v`, `FlambdaNormaliseSpec.v`,
  `FlambdaUnusedStaticsSpec.v`, `FlambdaInlineFreshSpec.v`, the
  canonicalisation by aliases, and the proofs that go by cases on a
  call kind: the case of `Detupled`.
- The comparison of code (`code_metadata_eqb`): the mark.
- The test tool: the converter gives code its mark. First series: a
  direct call of tupled code that the unit binds is refused, and so is
  a call of known arity in a unit that has tupled code (section 7,
  choice 3). A direct call of code that the unit does not bind stays
  as it is: the machine has no step for it, tupled or not. Second
  series: a direct call is `Detupled` if the code named is tupled,
  and `Direct` otherwise. The entry of the ledger on tupled functions
  is rewritten.
- The printer of the round trip (`tests/tools/flmodel_to_fexpr.ml`)
  writes the mark of code. In a unit that binds tupled code it does
  not write the model's direct call of that code, which fexpr would
  read as the compiler's; and it writes an indirect call without its
  arities, so that it is read as a call of unknown arity, where those
  arities are the ones that the parser then gives it, and not at all
  otherwise.

## 7. Choices taken, for the human to review

1. **`Direct` stays generic, and the compiler's direct call of tupled
   code is a third kind, `Detupled`.** In the compiler a call kind says
   two things: what is known of the callee, and which convention the
   call is in. For tupled code alone they come apart. If `Direct`
   meant the convention of the code, as in the compiler, "an indirect
   call of a known closure is a direct call of its code" would be
   false for tupled code. That statement is used with no knowledge of
   the code: the canonicalisation by aliases rewrites a call under
   every static environment, and the validator runs it on both terms.
   It would need a premise on the static environment, and so would
   every rewrite built on it, which today are stated under every
   environment. So the model separates the two: `Direct` and
   `Indirect` are both generic, and differ only in that `Direct` names
   the code; `Detupled` names the code and is in its convention. For
   code that is not tupled, which is all the code there was, nothing
   changes, and a `Detupled` call is a `Direct` call. The cost: the
   model's `Direct` is not the compiler's on tupled code, where the
   converter makes `Detupled` of it; and the model can write a direct
   call of tupled code with a tuple, which the compiler never makes
   (the canonicalisation makes them). The other choice: `Direct` in
   the convention of the code, a premise "the code is not tupled" on
   the three statements and on the canonicalisation, and the
   validator's rewrites stated from the states of a unit that has no
   tupled code, or told which code is tupled. To change later: the
   kind that the canonicalisation produces, and the tests of the
   validator's passes that look for `Direct`.
2. **A call in the convention of the code, of tupled code, with fewer
   arguments than parameters is undefined.** A partial application
   keeps the closure and the arguments, and is later called as an
   indirect call of the closure with all the arguments: for a closure
   of tupled code that would read the first of them as a tuple. The
   compiler makes no such call: a direct call of tupled code comes
   from `simplify_direct_tuple_application`, which makes it full or
   splits it. Simplify would make a wrapper of one
   (`simplify_direct_partial_application`), so the model is undefined
   where the compiler has a meaning, for a term it does not make. The
   other choice: a partial application that records the convention.
3. **One `Indirect` still** (choice 2 of the request on partial
   applications). A call of known arity is in the convention of the
   code, with no code named. Simplify makes one of an unknown-arity
   call only where no code that the callee may have is tupled, but it
   also makes one of a direct call whose code it has not simplified
   (`simplify_function_call_where_callee's_type_unavailable`), and
   that call may be of tupled code. So in a unit that has tupled code
   the converter refuses a call of known arity. The other choice: a
   fourth call kind, in the convention of the code with no code named.
4. **The tuple is what Simplify's projections read**: tag 0, exactly
   `n` fields, each of kind `Value`. The run time reads the first `n`
   fields of whatever it is given. This is the choice already taken
   for `Block_load` (the review comment at `eval_block_load`), and it
   makes the term that Simplify makes of the call the same as the
   call.
5. **An over-application of tupled code that does not return one
   `Value` is undefined at the call**, as for other code. In
   Simplify's term the full call is made first and what follows it is
   `Invalid`; the model is undefined sooner, which is the safe side.

## 8. What it gives, and what it does not

A unit with a tupled function runs in the model, before and after
Simplify, and the two runs are compared. It validates where nothing
but the canonicalisation is needed. Where Simplify has detupled a call
the validator needs `detuple_statement` as a rewrite of the term
before (with the variables of the term after), and where it has
inlined a tupled function, the inlining of section 5; neither is in
this series.

## 9. The series

Three.

A. Tupled code and the tuple. Made; this request comes with it.
   1. The mark of code and the machine, in one commit, since the mark
      is a field of a record that many proofs take apart:
      `tuple_fields`, `detuple`, `arguments_for_code`,
      `shape_of_code_call`, `shape_of_closure_call`; the hypothesis of
      the traversal of values; `inlinable`, `loopable` and the tests
      of the validator; the proofs repaired, those of other streams
      by stand-ins (section 6).
   2. `FlambdaTupledSpec.v` and its proofs; this request; the status.
   3. The guide and the ledger.
   4. Lines for the shared lists.
   5. The test tool and examples.
B. The call kind `Detupled`, `detuple_statement`, and the converter's
   direct calls of tupled code. It is a change of the size of an
   epoch: about forty-five files go by cases on a call kind, in
   definitions or in proofs, and each gets a case. It waits for the
   human's word on choice 1 of section 7, since the other choice
   needs no third kind.
C. The inlining of a call that gives tupled code a tuple, for the
   validator. The statements are proved (`inline_tupled_statement`,
   `inline_tupled_equivalent_statement`, after the first series), and
   the validator's inlining of direct calls has the case
   (`FlambdaNormaliseSpec.inline_apply`).

What the first series gives on its own: a unit that binds tupled code
is run and compared where Simplify leaves the calls of that code as
they are, or where there is none, which is the common case of a
module that defines such a function for others to call. Where
Simplify knows the function at a call it makes the call direct, and
the unit is reported as not checked until the second series.
