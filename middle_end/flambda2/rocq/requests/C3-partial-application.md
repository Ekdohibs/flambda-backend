# C3: partial applications and over-applications. A core change request

From C3, for C0 (who applies it), R2 (who reviews it), C2 (whose series
on kinds and arities it is written over, and whose check it touches)
and the human (section 7: the choices taken while he was away).

It was written as a request and a design, with the definitions
compiled in a prototype that is not in this tree, before the series.
The series has since been made and merged. Section 9 says
where it differs from sections 2 to 5, which are left as they were
written: the definitions have another shape; a direct call is a generic
application too (choice 1 of section 7 was first taken the other way);
and a call that gives a partial application no argument is undefined.

## Why

An OCaml application of a function to fewer arguments than it has
parameters, or to more, is ordinary: `let add x y = x + y in let inc =
add 1 in inc 2`; `let f x = fun y -> x + y in f 1 2`. Today the machine
has no step at such a call, the test tool says so and decides nothing,
and no statement speaks of a unit that executes one (C2's series names
the calls: `partial_or_over_application`). Closures are in most of the
OCaml examples, and these two forms are what is still missing of them.

## 1. What the compiler does

- **Closure conversion** (`from_lambda/closure_conversion.ml`). An
  application whose callee it knows nothing of becomes an `Apply` with
  the call kind `Indirect_unknown_arity` and the arguments as written,
  however many (`close_exact_or_unknown_apply`, the case with no
  approximation). That is every application outside classic mode. Only
  where it has a closure approximation of the callee (`close_apply`)
  does it compare the numbers, and then it expands the application
  itself: `wrap_partial_application` makes a wrapper function,
  `wrap_over_application` a full call followed by an application of the
  result.
- **Simplify** (`simplify/simplify_apply_expr.ml`,
  `simplify_direct_function_call`), once the type of the callee gives
  its code, whatever the call kind was. After `simplify_apply_shared`
  has made `Invalid` a call whose arguments are not of the kinds of its
  own arity, it goes by the numbers of arguments and of parameters, not
  unarized:
  - the arguments must agree in kind with the parameters as far as the
    shorter of the two goes (`arity_mismatch`), or the call is
    `Invalid`;
  - as many: a full call, whose return arity must be the code's;
  - more: the code must return one `Value`, or `Invalid`; then
    `Simplify_common.split_direct_over_application`: a full call of the
    code with the first arguments, whose return continuation is a new
    handler that applies the result to the remaining arguments, with
    `Indirect_unknown_arity`, the return continuation and the exception
    continuation of the original call;
  - fewer, but not none: the call must expect one `Value`, or
    `Invalid`; then `simplify_direct_partial_application`: new code,
    whose parameters are the missing ones and whose body calls the
    callee directly with all of them, and a closure of that code whose
    value slots hold the callee and the arguments given (those that are
    variables); the closure goes to the return continuation;
  - none, of code that has parameters: a fatal error.
- **The backend** (`to_cmm/to_cmm_expr.ml`, the three cases of
  `Function`). An `Indirect_unknown_arity` call is the generic
  application (`C.indirect_call`, at run time `caml_applyN`): it calls
  the closure in full if it has as many parameters as there are
  arguments, and otherwise applies the arguments one after another,
  through the closures that the run time makes for partial
  applications. A `Direct` call and an `Indirect_known_arity` call are
  full calls; the backend stops if the arguments of a direct call are
  not those of the code.

So in the term that Simplify is given, an application is an indirect
call of unknown arity with any number of arguments; in the term it
gives back, a call whose callee it knew is exact, a partial application
of a known function is a closure of new code, an over-application of a
known function is two calls, and a call whose callee it did not know is
as it was.

## 2. The design

Constructors are added and none is changed.

- A **partial application** is a value of its own: the closure, the
  arguments it has been applied to so far, and the arity they were
  given at. It is not a closure of code (`closure_of` and
  `closure_in_set` have no value on it), so a direct call of it and a
  projection from it are undefined; it is an allocated value of kind
  `Value`.
- What is left of an **over-application** is a continuation of its
  own: the remaining arguments, their arity, the return arity of the
  call, and the return and exception continuations of the call.
  Applying it to one value calls that value, indirectly, with the
  remaining arguments. The machine makes up no term and no name.
- An **indirect call** is the generic application. Its callee is a
  closure, or a partial application, which stands for its closure
  applied to the arguments so far followed by those of the call, at
  the two arities appended. With `provided` arguments and `params`
  parameters, not unarized:
  - as many: the full call of today, with C2's three tests;
  - otherwise the arguments must be of the kinds of the arity the call
    is made at (C2's first test, which the compiler makes before
    anything else), and the arities must pass C2's
    `partial_or_over_application`, which is the compiler's test as
    section 1 has it;
  - more arguments: the full call with the first of them (the flat
    list of values cut at the unarized size of the parameters), made at
    the result arity of the code, returning to what is left of the
    over-application; the exception continuation is the call's;
  - fewer: nothing is called; the partial application goes to the
    return continuation of the call;
  - anything else is undefined, as today.
- A **direct call** stays exact: a closure of the code named, with as
  many arguments as parameters. A direct call with another number of
  arguments is undefined, as today (section 7, choice 1).

What is not tested, as in the compiler: at a partial application
nothing is compared with the result arity of the code; at an
over-application nothing is compared with the return arity of the call
until the result is applied, when it is the return arity of that call.

`step` is not changed: a call is still `call_function` of the callee,
the arguments and what the two continuations of the `Apply` stand for.

## 3. The definitions

Over C2's series. They compiled in the prototype, which is not in
this tree.

### `FlambdaModel.v`

Two functions on arities, named after the compiler's, beside
`unarize`:

```
Definition cardinal_unarized (a : arity) : nat := length (unarize a).

Definition partially_apply (a : arity) (provided : nat) : arity :=
  skipn provided a.
```

`kinds_agree_on_prefix`, `is_one_param_of_kind_value` and
`partial_or_over_application` move here from
`FlambdaWellKindedSpec.v`, text unchanged: they speak of arities only,
and `call_function` now uses them (agreed with C2).

### `FlambdaValueSpec.v`

```
Inductive value : Type :=
  ...
  | Value_partial_application (callee : value) (args : list value)
      (args_arity : arity).

Definition partial_application_of (v : value)
    : option (value * list value * arity) :=
  match v with
  | Value_partial_application callee args args_arity =>
    Some (callee, args, args_arity)
  | _ => None
  end.
```

`is_allocated` has one more disjunct, `is_some (partial_application_of
v)`: at run time a partial application is a closure, so `Is_int`
without `variant_only` and `Is_null` answer no on it. Every other
reading of a value has no value on it: the matches end with a wildcard.

### `FlambdaSemanticsSpec.v`

```
Inductive cont_value : Type :=
  | Closure ...
  | Outer (k : continuation)
  | Over_application (remaining : list value)
      (remaining_arity return_arity : arity) (return_to exn_to : cont_value).

(* A full call: today's body of [call_function], from the test that the
   code binds nothing static on, with [closure] as [my_closure]. *)
Definition call_code (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity) (return_to exn_to : cont_value)
    (statics : static_env) : step_result := ...

Definition applied_closure (callee : value) : value * list value * arity :=
  match partial_application_of callee with
  | Some applied => applied
  | None => (callee, [], [])
  end.

Definition call_function_returning (return_partial : value -> step_result)
    (callee : value) (kind : call_kind) (vs : list value)
    (args_arity return_arity : arity) (return_to exn_to : cont_value)
    (statics : static_env) : step_result :=
  let '(closure, applied, applied_arity) :=
    match kind with
    | Direct _ => (callee, [], [])
    | Indirect => applied_closure callee
    end
  in
  let vs := applied ++ vs in
  let args_arity := applied_arity ++ args_arity in
  match closure_of statics closure with
  | Some (called, _) =>
    let expected :=
      match kind with
      | Direct named => Pos.eqb called named
      | Indirect => true
      end
    in
    match expected, lookup called (static_code statics) with
    | true, Some c =>
      let provided := length args_arity in
      let params := length (code_params_arity c) in
      if Nat.eqb provided params
      then call_code closure c vs args_arity return_arity return_to exn_to
             statics
      else
        match kind with
        | Direct _ => Undefined
        | Indirect =>
          if
            negb
              (kinds_eqb (map kind_of_value vs) (unarize args_arity)
               && partial_or_over_application args_arity return_arity
                    (code_params_arity c) (code_result_arity c))
          then Undefined
          else if Nat.ltb params provided
          then
            let size := cardinal_unarized (code_params_arity c) in
            call_code closure c (firstn size vs) (firstn params args_arity)
              (code_result_arity c)
              (Over_application (skipn size vs)
                 (partially_apply args_arity params) return_arity return_to
                 exn_to)
              exn_to statics
          else
            return_partial (Value_partial_application closure vs args_arity)
        end
    | _, _ => Undefined
    end
  | None => Undefined
  end.

Fixpoint apply_cont_value (c : cont_value) (vs : list value)
    (statics : static_env) {struct c} : step_result :=
  match c with
  | Outer k => Exit k vs
  | Closure (Handler params _ handler) def_vars def_conts recursive =>
    ... (* as today *)
  | Over_application remaining remaining_arity return_arity return_to exn_to =>
    match vs with
    | [v] =>
      call_function_returning
        (fun partial => apply_cont_value return_to [partial] statics) v
        Indirect remaining remaining_arity return_arity return_to exn_to
        statics
    | _ => Undefined
    end
  end.

Definition call_function (callee : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) : step_result :=
  call_function_returning
    (fun partial => apply_cont_value return_to [partial] statics) callee kind
    vs args_arity return_arity return_to exn_to statics.
```

Why `apply_cont_value` is recursive. A partial application is given to
the return continuation of its call in the same step, as a value is
given to a continuation by `Apply_cont`. That continuation may be what
is left of an over-application (the call was the last thing done by a
function that was itself over-applied), and then the partial
application is called at once with the remaining arguments, which may
again be too few. The recursion is on the continuation: what is left
of an over-application holds its return continuation, which is
smaller. Rocq accepts the definition as it is written here.

A partial application whose callee is not a closure (another partial
application, a number) can be written in an environment; the machine
makes none, as it always puts the closure in, and calling one is
undefined.

### `FlambdaWellKindedSpec.v` (C2's, cases agreed with C2)

`value_well_kinded` of a partial application: its callee and its
arguments are well-kinded. Nothing is asked of the kinds of the
arguments against the parameters: the completed call makes the three
tests on the whole. `arity_of_cont` of what is left of an
over-application is `[Value]`; `cont_well_kinded` asks that the
remaining values, the return continuation and the exception
continuation are well-kinded, the first at `unarize return_arity` and
the second at `[Value]`. `wrong_number_of_arguments` counts one
argument for it. `partial_or_over_application_undefined_statement`
goes; `at_partial_or_over_application` stays as the test, for the tool,
that a state is at a call which the machine runs as one.

### Other definitions that gain a case

`FlambdaUnusedStaticsSpec.v`: `value_avoids` (the callee and the
arguments), `cont_free_of` (the remaining values and the two
continuations). `FlambdaSemanticsSpec.v`: `cont_values_all` (the
traversal of where a state holds values): the remaining values, and
the two continuations. `FlambdaClosuresSpec.v`: `value_valid` (the
callee and the arguments are valid). The types
(`rocq/types/grammar/FlambdaTypeMeaning*`) read a value through observations
and need no case in a definition; their proofs by cases on a value
need one more branch.

## 4. Which statements change

- `call_at_wrong_kinds_undefined_statement` (C2's): true as it is of a
  direct call. Of an indirect call it needs one more premise: that the
  call has as many arguments as the code has parameters, or does not
  pass `partial_or_over_application`.
- `partial_or_over_application_undefined_statement` (C2's): false, and
  removed; it said that the fragment left these calls out.
- `step_values_all_statement` (N1's traversal): one more hypothesis,
  that `P` holds of a partial application whose callee and arguments
  satisfy it. Its clients (the validity of values, and whatever is
  written on it meanwhile) give it.
- Every statement that quantifies over all environments now speaks of
  environments with the two new forms in them, and is to be proved of
  them: the congruences, the simulations of the validator's passes,
  the renaming.
- The inlining statements (`FlambdaApplySpec.v`) are about a call that
  agrees with the code, which has as many arguments as parameters:
  their text stands.
- New statements, with the change: an over-application is equivalent
  to the full call followed by the application of the result (the
  rewrite that Simplify makes, and the one the validator needs); an
  indirect call of a partial application is the call of its closure
  with the arguments put together; a partial application of a partial
  application is one partial application.

## 5. Fallout

Measured on the prototype: the `Spec` files build; the first proof file
that fails is `FlambdaSemanticsProofs.v`.

The definitions cost little: the cases above, and one more branch in
each proof by cases on a value that names the constructors
(twenty-three such patterns, in six files; done on the prototype).

The proofs about a call are the cost. Two lemmas that most of them rest
on are false as stated and have to be restated:

- `call_function_step` ("a call that has a step runs the body of the
  code of the callee with the arguments bound"). Now a call that has a
  step runs a body with the first of the arguments and another return
  continuation, or with the arguments of a partial application put
  first; or it is the step that its return continuation makes of a
  partial application.
- `call_function_cases` ("a call is undefined whatever its
  continuations, or runs a body that does not depend on them"). Now
  there are four cases, still decided without looking at the
  continuations: undefined; a body with the return continuation given;
  a body with what is left of an over-application; the return
  continuation applied to a partial application.

Every proof that a step keeps something, or that two runs stay
related, goes through one of the two, or unfolds `call_function` or
`apply_cont_value`: about thirty places in eleven files
(`FlambdaWellKindedProofs.v`, `FlambdaWellFormedProofs.v`,
`FlambdaContextProofs.v`, `FlambdaCompareProofs.v`,
`FlambdaUnusedStaticsProofs.v`, `FlambdaNominalProofs.v`,
`FlambdaNormaliseProofs.v`, `FlambdaApplyProofs.v`,
`FlambdaControlFlowProofs.v`, `FlambdaClosuresProofs.v`,
`rocq/types/naive/FlambdaAliasesProofs.v`), and the proof of the traversal
of values. Each gets the same shape: a lemma about
`call_function_returning` that assumes what is wanted of
`return_partial`; the fact about `apply_cont_value` by induction on the
continuation, with that lemma at what is left of an over-application;
then the fact about `call_function`. The relations between the
continuations of two runs get a case: two over-applications are
related when their remaining values are and their continuations are.

The test tool: the conversion is not changed (an application is
already converted with its arguments and its arities). The printer and
the observation of a value at exit get the case of a partial
application (a function, as a closure is); the tool stops refusing a
direct partial or over-application before the run and stops reporting
a run that ends at one. The ledger's entry on what the tool does about
these calls goes.

## 6. What it gives, and what it does not

- A unit that applies an unknown function to too few or too many
  arguments runs, before Simplify and after.
- An over-application of a known function validates with one rewrite,
  which is an equivalence: it is exactly the split that Simplify makes.
- A partial application of a known function runs on both sides, and
  the two runs compute the same; but it does not validate. Simplify's
  term makes a closure of new wrapper code where the term before makes
  a partial application, and the two are different values: the terms
  are not equivalent as long as what a unit ends with is compared with
  `=`. It needs the comparison of values up to a relation, which is
  also what the code ids of a simplified set need. Until then the tool
  says "the same, not validated" of such a unit.
- Tupled functions stay out: the converter refuses tupled code. So do
  the allocation modes and regions that the compiler threads through
  both expansions.

## 7. Choices taken, for the human to review

Each is to have a review comment for the human at the definition, in
the series.

1. **Every call is a generic application, a direct one too.** This
   was first taken the other way (a direct call with another number of
   arguments than parameters undefined) and changed while the series
   was made. Simplify reads a call of any kind as a generic application
   once it knows the callee; the backend reads a direct call as a full
   call, and stops on one that is not. With a direct call exact and an
   indirect one generic, "an indirect call of a known closure is a
   direct call of its code" holds of full calls only. That is three
   statements of the model (`apply_indirect_known_statement`,
   `apply_sibling_statement`, and the canonicalisation by aliases,
   which the validator runs and which has no arity of the code to test
   with). So a direct call differs from an indirect one only in that
   its callee must be a closure of the code named, which a partial
   application is not. The machine then gives a meaning to direct calls
   that the backend refuses. No term that the compiler makes is known
   to have one, and nothing checks it: discrepancy C3-9, and a review
   comment at `shape_of_closure_call`. The other choice is to make a
   direct call that is not full undefined, and to give those three
   statements a premise on the number of arguments.
2. **One `Indirect` for both of the compiler's indirect call kinds.**
   The model does not tell `Indirect_known_arity` from
   `Indirect_unknown_arity`. The backend makes a full call of a
   known-arity one, so a known-arity call with another number of
   arguments is run by the model as a generic application and would
   not be by the compiler. No such term is known: closure conversion
   makes no call of known arity, and Simplify marks a call so only
   where its types say that every code the callee may have has the
   arity of the call, or where it makes an indirect call of a direct
   one. Nothing checks afterwards that a call so marked is full, which
   is what the model would need of the compiler (discrepancy C3-9, and
   the review comment at `shape_of_closure_call`). The other choice is
   a third call kind, exact like `Direct`; C2 agrees to one `Indirect`.
3. **A partial application is not a closure of code.** `closure_of`
   has no value on it, so the types say nothing of it but that it is a
   value, a direct call of it is undefined, and at exit it is compared
   with `=` like every value (the callee and the arguments). In
   Simplify's terms it is a closure of wrapper code; see section 6.
4. **A partial application is given to its return continuation in the
   step of the call**, and the continuation left by an over-application
   calls in the step in which it is applied. The other choice is a
   step of its own for each, through a state made up by the machine
   with names of its own; it keeps `apply_cont_value` from being
   recursive and makes the proofs about a call shorter, but the
   machine's steps then do not commute with a renaming of names, which
   the work on names relies on.

5. **A call that gives a partial application no argument is
   undefined.** The test of arities is made on the arguments kept
   followed by those of the call, so without this a partial
   application called with no argument would be a partial application
   again. The compiler stops on a call with no argument of code that
   has parameters, and the closure that it makes of a partial
   application is of code with the parameters that are left. The other
   choice is to let such a call return the partial application
   unchanged.
6. **Inlining asks that the call is full.** `inlinable` has one more
   condition: as many arguments as the code has parameters, counted by
   parameter. It already asked for as many values as the code has
   variables; with a parameter that is an unboxed product of nothing
   the two differ, and a call with as many values could be a partial
   application or an over-application, which does not run the body
   once. The validator's test (`inlinable_b`) makes the same
   comparison.

## 8. The series, once C2's is on the integration branch

1. The arities in `FlambdaModel.v` (the two functions, the three moved
   definitions); nothing else changes meaning.
2. The value and the continuation, with every definition that gains a
   case and every proof by cases repaired; `call_function` not yet
   changed, so the new forms are inert and every statement stands.
3. `call_function` and `apply_cont_value`; the two lemmas restated;
   the proofs about a call, file by file.
4. The new statements of section 4, and the validator's rewrite for an
   over-application.
5. The test tool and examples (`let inc = add 1`, `f 1 2` with `f`
   returning a function, a function passed and applied to too many
   arguments), last, as they need the regenerated extracted files.

Steps 1 and 2 are small and can be reviewed and applied first. Step 3
is the epoch.

## 9. The series as made

On `flambda2-rocq/C3-partial`, over the series on kinds and the epoch
of exceptions: the arities (step 1); the value, the continuation and
the machine (step 2; the two constructors were a commit of their own,
made by no step, until the series was put over the epoch of
exceptions, whose proofs about a continuation it would have had to
repair twice); the statements (step 3); then the test tool and its
examples. What differs from sections 2 to 5:

**Exceptions.** A call evaluates the extra arguments of its exception
continuation and gives its callee that continuation with them
(`with_extra_args`). What is left of an over-application keeps that
continuation as it is, extra arguments included, so the full call and
the call of its result raise to the same handler with the same extra
arguments: the compiler's split gives both calls the exception
continuation of the over-application. A partial application raises
nothing, and the extra arguments of a call that makes one are
evaluated and dropped. Nothing here has a trap action. The machine
with a stack of handlers (`FlambdaTrapSpec.v`) says what the two sorts
of call do there: the stream of exceptions changed it, with an entry
of the stack for the full call inside an over-application.

**The definitions** (`FlambdaSemanticsSpec.v`). Section 3's
`applied_closure` and `call_function_returning` are not there. What a
call is and what it does are two functions:

- `call_code`: the full call of a closure of code, with the three
  tests of the series on kinds;
- `call_shape`: `Full_call`, `Over_call` or `Partial_call`, with what
  each needs;
- `shape_of_closure_call`: what the call of a closure is, or nothing
  where it is undefined: full with as many arguments as parameters;
  otherwise, the kinds of the arguments and the compiler's test of
  arities permitting, an over-application or a partial application;
- `shape_of_call`: the same of any callee; an indirect call of a
  partial application, with one argument at least, is that of the
  closure it keeps with all the arguments;
- `run_call`: runs a shape, given what is done with a partial
  application;
- `apply_cont_value`, recursive on the continuation as in section 3,
  and `call_function`, which is `run_call` of `shape_of_call` with the
  partial application given to the return continuation.

**The call kinds.** A direct call is a generic application too
(section 7, choice 1). A call that gives a partial application no
argument is undefined (choice 5).

**Statements changed**, beyond section 4:

- `call_at_wrong_kinds_undefined_statement` is restated by cases:
  arguments not of the kinds of the call's arity; a full call at
  arities that are not the code's; a call that is not full and fails
  the compiler's test. `partial_or_over_application_undefined_statement`
  and `at_partial_or_over_application` are removed (section 3 kept the
  second for the tool, which no longer needs it).
- `step_values_all_statement`: the new hypothesis is an equivalence,
  `P` of a partial application if and only if `P` of the value it calls
  and of its arguments, since a call both makes partial applications
  and takes them apart.
- `call_binds_my_closure_statement` and
  `well_formed_call_binds_my_closure_statement` ask that the call is not
  a partial application, which runs no code.
- `inlinable` has the condition of choice 6, and so
  `inline_identity_statement` (in the types' first model) a premise.
- `relates_observations`, the premise of the law that primitives
  compute alike on values that the observations read alike, has one
  more clause: what is related to a partial application is a partial
  application. A partial application is allocated, and `Is_int` looks.
- `calls_code_binding_statics` looks through a partial application at
  the closure it keeps.
- `cont_in_range`, `cont_in_scope` and the relations between the
  continuations of two runs have their case for what is left of an
  over-application.

**New statements** (`FlambdaPartialApplicationSpec.v`): a call with
fewer arguments gives its return continuation a partial application; an
indirect call of a partial application is the call of its closure with
all the arguments, and a direct one, or one with no argument, or one
whose own arguments are not of the kinds of its own arity, is
undefined (the last is tested of the call itself, before the arguments
that the partial application keeps are put in front: R2's finding, with
an example of a partial application that no call makes, in which the
test of all the arguments together would pass); a partial application
whose result is at once called
refines the one call with all the arguments; a call with more arguments
is a full call that returns to what is left of the application, and
what is left, applied, is an indirect call of the value returned; so
an over-application is refined by the term that splits it, the full
call followed, in a handler, by the call of its result
(`over_application_split_statement`: the term of Simplify's
`split_direct_over_application`, without the regions); two units, one
of each sort, run, are well-formed and pass the check of kinds; two
over-applications that raise, one in the full call and one in the call
of its result, leave by the exception continuation with its extra
argument; a call whose extra arguments have no value is undefined,
even where it would make a partial application.

**The proofs.** Section 5's two lemmas about a call are replaced by
lemmas about the pieces (`call_code_step`, `call_code_cases`,
`shape_of_closure_call_cases`, `shape_of_call_cases`), and each
invariant or relation is proved of `call_code`, of `run_call` given
what is wanted of the partial application, of `apply_cont_value` by
induction on the continuation, and then of `call_function`. An
invariant that asks something of the terms only goes through one
generic lemma about where a call lands (`call_function_lands`).

**Not done.**

- The rewrite of the validator's normalisation that splits an
  over-application of a function it knows, which is what such a unit
  needs in order to validate. The statement that justifies it is
  there (`over_application_split_statement`). The two runs are not
  step for step the same (the handler takes a step that what is left
  of the application does not, and a partial application may reach an
  exit through several of them at once), so its proof is a simulation
  in which one step, or an exit, is answered by several steps, with a
  relation between the two sorts of continuation. Section 6 called it
  one rewrite and an equivalence; it is a refinement, since the
  over-application evaluates all its arguments, and tests their kinds,
  before the full call.
- `kind_fault` does not see a handler given the wrong number of
  arguments by a call that returns a partial application, or by what
  is left of an over-application. A well-kinded state has neither
  fault; the test that says so only looks at an `Apply_cont`.
- The partial application of a function that Simplify knows validates
  only once values are compared up to a relation (section 6).
