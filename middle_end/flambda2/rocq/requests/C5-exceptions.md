# C5, exceptions: core change request

Stream C5 of `PARALLEL-PLAN.md` (section 6): trap actions on
`Apply_cont`, exception continuations with extra arguments, `raise`.
From C5, for C0 (who applies it), R1 (who reviews the `Spec` diff and
section 4) and the human (section 8).

The prototype is not in this tree. It was made on the integration
branch at b134cfe0ca, before C2 and C3. It was whole: every `.v` file
built, `make -C rocq all` and `make -C rocq assumptions` passed, and
the semantics tests passed with seven new examples. It was not for
merging; the reviewer read its diff of the `Spec` files and of the
model.

The series for the epoch, over C3 and C2, is merged. Sections 1 to 9
describe the prototype and the plan as they were reviewed; section 10
says what the series is and where it differs from the prototype.

Paths below are in `middle_end/flambda2/rocq/` unless they say
otherwise; compiler files are in `middle_end/flambda2/`.

## Summary

Terms. An `Apply_cont` carries its trap action (`Push k`, or `Pop k`
with a raise kind), and an `Apply` the extra arguments of its exception
continuation. A raise is not a new form: it is an `Apply_cont` whose
trap action pops the continuation it applies, as in the compiler. The
compiler's `is_exn_handler` flag belongs in C2's record
`handler_metadata` and is added there, after C2.

Machine. It does not execute trap actions. A raise goes to the
continuation it names, and an exception of a call goes to the exception
continuation the call names, applied to the exception followed by the
extra arguments. That is how Simplify reads a term. So `state`, `Exit`,
`behaves`, `refines` and `call_function` are untouched; `step` on
`Apply` evaluates the extra arguments, and `cont_value` gains one form,
`Extra_args c extra`, for what the callee raises to. Nothing new is
undefined.

Discipline. The backend's code has a stack of handlers and goes by it:
a raise goes to the handler installed last whatever the term names.
The two readings agree on the terms the front end builds, and nothing
in Flambda 2 checks that (discrepancy C5-1). A new file,
`FlambdaTrapSpec.v`, has the check as a boolean function of the term,
proposed to C0 as conditions of well-formedness (section 7). That the
stack is not in the machine is decision 1, for the human.

Left for after the epoch, as leaf work: a machine with the stack and
the theorem that it agrees with this one on a unit that passes the
check; the demotion of a handler as a rewrite the validator knows; and
inlining a call that has extra arguments.

## 1. The change to the `Spec` files

### `FlambdaModel.v`

```coq
Inductive raise_kind : Type := Regular | Reraise | No_trace.

Inductive trap_action : Type :=
  | Push (exn_handler : continuation)
  | Pop (exn_handler : continuation) (kind : option raise_kind).
```

```diff
 Record apply_cont : Type := Apply_cont_expr {
   ac_continuation : continuation;
-  ac_args : list simple
+  ac_args : list simple;
+  ac_trap_action : option trap_action
 }.

 Record apply_expr : Type := Apply_expr {
   ...
-  ap_exn_continuation : continuation
+  ap_exn_continuation : continuation;
+  ap_exn_extra_args : list simple
 }.
```

with four small definitions: `trap_action_handler` (the continuation a
trap action names), `apply_cont_continuations` (the continuation
applied and that one), `no_trap_action`, `no_exn_extra_args`, and
`is_raise`, which is `Apply_cont_expr.is_raise`. The exception
continuation of a call stays two fields where the compiler has an
`Exn_continuation.t`: every reader of `ap_exn_continuation` is then
untouched.

### `FlambdaSemanticsSpec.v`

```diff
 Inductive cont_value : Type :=
   | Closure (h : continuation_handler) (vars : env)
       (conts : list (continuation * cont_value))
       (recursive : list (continuation * continuation_handler))
-  | Outer (k : continuation).
+  | Outer (k : continuation)
+  | Extra_args (c : cont_value) (extra : list value).

-Definition apply_cont_value (c : cont_value) (vs : list value)
+Fixpoint apply_cont_value (c : cont_value) (vs : list value)
     (statics : static_env) : step_result :=
   match c with
   ...
+  | Extra_args c extra => apply_cont_value c (vs ++ extra) statics
   end.

+Definition with_extra_args (c : cont_value) (extra : list value)
+    : cont_value :=
+  match extra with
+  | [] => c
+  | _ :: _ => Extra_args c extra
+  end.

   | State (Apply ap) vars conts statics =>
-    match eval_simple vars (ap_callee ap), eval_simples vars (ap_args ap) with
-    | Some callee, Some vs =>
+    match
+      eval_simple vars (ap_callee ap), eval_simples vars (ap_args ap),
+      eval_simples vars (ap_exn_extra_args ap)
+    with
+    | Some callee, Some vs, Some extra =>
       call_function callee (ap_call_kind ap) vs
         (resolve (ap_continuation ap) conts)
-        (resolve (ap_exn_continuation ap) conts) statics
-    | _, _ => Undefined
+        (with_extra_args (resolve (ap_exn_continuation ap) conts) extra)
+        statics
+    | _, _, _ => Undefined
     end
```

The rule of `step` for `Apply_cont` does not change: it does not read
`ac_trap_action`. `call_function` does not change. With no extra
argument, `with_extra_args` gives the caller's continuation as it is,
so a call of today steps as it did.

Why a form of `cont_value`. The callee's exception continuation takes
the exception alone, and the caller's handler takes it followed by the
extra arguments. Something has to hold the values of those between the
call and the raise. The alternatives were a handler made up at the call
(it needs a fresh variable for the exception, which the model has no
way to pick), or a field in every `Closure`. `Extra_args` holds values
and no name.

What is undefined, and the CRs. Nothing is undefined that was not: an
extra argument that is an unbound variable makes the call undefined, as
any unbound variable does, and a handler applied to the wrong number of
values is undefined as any continuation is (so a raise that does not
supply the extra arguments its handler takes). The extra arguments are
evaluated at the call, raised or not, as to_cmm does
(`to_cmm_expr.ml`, `translate_apply`). So there is no new CR in the
`Spec` files. If decision 1 goes the other way, every mismatch with the
stack is a new undefined case and each gets its CR.

One case is defined and should not be (R1's question; discrepancy
C5-5). A callee that raises to a way out of the term through a call
with extra arguments exits with the exception followed by them: a way
out takes any number of values, as it does today when an `Apply_cont`
applies it. The exception continuation of a unit, or of code, takes
the exception alone. The machine cannot test that, since a way out has
no arity, and after C2 it still cannot: C2's machine tests kinds in
`call_function` only, and the number of values only where a handler
binds its parameters. `Extra_args` holds values and keeps no kinds;
where it is around a handler, a wrong number of values is undefined at
the raise. No check of kinds is relied on for this. What rules the
case out is a condition of well-formedness (section 7): no call gives
extra arguments to a continuation that the unit, or the body of its
code, does not bind. That is an invariant of what the front end and
the transformations build (extra arguments go only to handlers of
`try`, and inlining wraps), and Simplify stops on a call that breaks
it, by accident (C5-5). A way out applied directly to the wrong number
of values stays defined: that gap is older than this request and is
not closed by it; it closes when a way out has an arity. There is a CR
at `with_extra_args` that says so.

The kinds that C2 gives the extra arguments (below) are not tested by
the machine, like the kinds of every binder (C2's CR for D10). Simplify
does not look at the simples either: at the use it gives the handler
unknown types of the declared kinds (C5-5). Should D10 come to be
"undefined where executed", the test for extra arguments belongs in
`step` at the `Apply`, where the values and the declared kinds are
both at hand.

### Names: renaming, free names, comparison, scope

No new binder. A trap action names a continuation, which is a use of
it:

- `FlambdaNominalSpec.v`: `rename_trap_action`; `rename_apply_cont`
  renames it; `rename_apply` renames the extra arguments;
  `var_free_in_expr` has the extra arguments; `cont_free_in_expr` is
  `k ∈ apply_cont_continuations ac` for an `Apply_cont` and for each
  arm of a `Switch`. (`Name_occurrences` counts the two sorts of
  occurrence apart; the model has one notion of free.)
- `FlambdaCompareSpec.v`: `apply_cont_eqb` compares the trap actions,
  the continuations they name up to the pairs of bound names and the
  raise kinds exactly (`Trap_action.compare`); `apply_eqb` compares the
  extra arguments.
- `FlambdaNormaliseSpec.v`: `expr_in_scope` asks that the continuation
  of a trap action and the extra arguments be in scope.
- `FlambdaWellFormedSpec.v`: `free_continuations` has the continuation
  of a trap action. `raises_to k e`, which is how the model recognises
  an exception handler, is also true when a trap action of `e` names
  `k`; so the handler of a continuation that its body installs as an
  exception handler is not at toplevel
  (`simplify_let_cont_expr.ml`: `at_unit_toplevel && not
  is_exn_handler`).

Equivariance of all of these is proved (`FlambdaAlphaProofs.v`:
`rename_trap_action_compose`, `apply_cont_continuations_equivariant`,
and the cases of the existing statements).

### Other `Spec` files

- `FlambdaControlFlowSpec.v`: the four statements about a `Let_cont`
  applied at once are about an `Apply_cont` with no trap action
  (`Apply_cont_expr k args None`).
- `FlambdaApplySpec.v`: `apply_indirect_known_statement` is for any
  extra arguments. `inlinable` has one more conjunct,
  `ap_exn_extra_args ap = []`.
- `FlambdaNormaliseSpec.v`: `inlinable_b` tests it.
  `inline_cont_at_tail` does not inline an application that has a trap
  action, `thread_arm` leaves alone an arm that has one, and
  `arms_apply` counts a continuation that a trap action names (so a
  `Let_cont` is not dropped while one does). On this machine the
  rewrites would be sound without these tests; with them a
  normalisation never drops a trap action or the binding of a handler
  that one names, and so keeps the discipline (C0's rule that a
  function on terms keeps `well_formed_unit`). They are what Simplify
  does: it does not inline a continuation applied with a trap action
  (`is_applied_with_traps`).
- `FlambdaUnusedStaticsSpec.v`: `apply_avoids` has the extra arguments;
  `cont_free_of` has the case of `Extra_args`.
- `rocq/types/naive/FlambdaAliasesSpec.v`: `canonical_apply_cont` keeps the
  trap action; `canonical_apply` follows the aliases of the extra
  arguments. `FlambdaTypesSpec.v`: `None` and `[]` in the statements
  that build an `Apply_cont` or an `Apply`.

### The new file pair, `FlambdaTrapSpec.v` and `FlambdaTrapProofs.v`

Mine. The first is to be reviewed; its header says what is checked
and against which OCaml. In outline:

- `trap_stack`: the handlers installed, as continuation names, the last
  installed first. `demands`: for each continuation an expression may
  go to and does not bind, the stack with which it gets there.
- `apply_cont_demands ts ac`: under the stack `ts`, a `Push k` asks
  that the handler `k` run under `ts` and goes on under `k :: ts`; a
  `Pop k` needs `k` on top and goes on under the rest. A raise is the
  case where the continuation applied is `k`.
- `apply_demands ts ap`: the exception continuation of a call is on
  top of `ts`, and runs under what is below it.
- `trap_demands ts e : option demands`, by recursion on the term. A
  continuation is entered with one stack, found from its applications
  in the body of its `Let_cont` (and in the handlers of its group, for
  a recursive one), and its handler is checked under that. It cannot be
  the stack where the `Let_cont` stands: the front end defines the
  continuation of the body of a `try` outside the `try` and enters it
  with a `Push`, and Simplify lifts continuations out of handlers.
- `code_traps_well_formed`: the body of code starts with its exception
  continuation as the one handler installed, returns so, and raises to
  it so. `traps_well_formed unit_exn e`: the same for a unit.
- `exn_handlers_well_formed e`: a continuation used as an exception
  handler is not applied as an ordinary continuation and has a
  parameter for the exception; none is recursive.
- `ways_out_well_formed exn e`, for the body of code and for a unit:
  of the continuations it does not bind, only the exception
  continuation is used as an exception handler, and no call gives one
  of them extra arguments. (Added after R1's review, which found that
  a unit could install its return continuation as a handler: R1-22.)

`FlambdaTrapProofs.v` has twenty-five examples, each a term the check
accepts or refuses and why. There is no theorem about the check yet
(section 9).

Where they end up. C0's rule is that the checks of well-formedness are
all in `FlambdaWellFormedSpec.v`, so that one file is the definition
of a well-formed term. So at the epoch the two checks, with what they
are made of, go into that file, written by me and applied by C0, each
with its row and its comment; `FlambdaTrapSpec.v` keeps what is proved
of them. The prototype has them in a file of its own because it is
not for merging.

## 2. The design

### What the compiler does

The front end (`from_lambda/lambda_to_flambda.ml`, the case `Ltrywith`;
`closure_conversion.ml`, `close_raise0`, `close_exn_continuation`)
makes, for `try body with exn -> handler`:

    cont body_k push(handler_k)
      where body_k = <body, its result going to poptrap_k>
      where poptrap_k (r) = cont after_k pop(handler_k) (r)
      where handler_k exn (exn, <mutable variables>) = <handler>

A `raise e` is `cont k pop(<kind> k) (e, <extra>)`, with `k` the
innermost handler or the exception continuation of the function. A
call names `k` as its exception continuation, with the same extra
arguments. A jump out of the body goes through continuations that pop.
The extra arguments are the mutable variables of the function that the
handler reads, which closure conversion has made parameters. No
primitive raises: a bounds check or a division check is a `Switch` and
a raise.

Simplify reads the names. It joins the types of a handler's parameters
over the raises and calls that name it. When no such use needs a real
raise, it demotes the handler to an ordinary continuation and erases
the trap actions that name it (`simplify_let_cont_expr.ml`,
`prepare_dacc_for_handlers`; `simplify_common.ml`,
`clear_demoted_trap_action`). Inlining renames the exception
continuation of the code to that of the call
(`simplify_shared/inlining_helpers.ml`), through a wrapper handler if
the call has extra arguments.

The backend reads the stack. `to_cmm_expr.ml`, `translate_raise`: an
`Apply_cont` of an exception handler is a raise to whatever is
installed last. `translate_apply` does not look at the handler the call
names. `backend/select_utils.ml`: a `Pop` removes the top whatever it
names; a continuation reached with two stacks is a fatal error, and so
is a `Pop` on an empty stack. Nothing else is checked, anywhere.

### An example where the two differ

    cont body1 push(outer)
      where body1 =
        (cont body2 push(inner)
           where body2 =
             let bucket = %block.[`0`].`toplevel` (a, 7) in
             cont outer pop(notrace outer) (bucket)
           where inner exn (e : val) = cont leave pop(outer) (1))
      where outer exn (e : val) = cont leave (2)

By the names the raise goes to `outer`, and the unit returns 2. By the
stack it goes to `inner`, and the unit returns 1. Simplify makes it the
constant 2 (run: `tests/semantics/raise_over_handler.fl` without the
check). The front end does not build this term, and the check refuses
it. C5-1 has the details.

### The choice

The machine goes by the names, and the discipline that makes the stack
agree is a property of the term, checked apart. The reasons:

- It is the reading under which Simplify is right, and the only one in
  which a rewrite of exceptions can be stated without a premise: with a
  stack in the machine, the handler that a raise reaches is known only
  from the discipline.
- The human's guidance for a term that the compiler cannot build and
  the model accepts is that well-formedness is too weak. The term above
  is such a term. The backend's own two checks are static, and fail on
  code that never runs.
- A stack in the machine has to say when the handler on top is the one
  named. Continuation values have no identity: two closures of one
  handler made in two turns of a loop are equal when their environments
  are. Comparing closures structurally needs a decidable equality on
  values, environments and terms, and numbering them makes every proof
  that relates two runs relate two numberings.
- It adds no field to the state. C4's heap adds one, and each such
  field touches every statement that builds a state.

What it gives up: `refines` does not say that a pass keeps the trap
actions right. That is carried by the pass keeping `well_formed_unit`
(C0's rule for every function on terms), and by the validator, which
checks the term after Simplify (the test tool fails on one that is not
well-formed, and says which condition).

### What is assumed of the exception

Nothing. It is whatever value the raise passes first. An exception of
OCaml is a block whose first field is its constructor, and a
constructor is an `Immutable_unique` block compared by `Phys_equal`;
both are C4's (its staging has `Phys_equal` on locations first and
static unique blocks later), so matching an exception against a
constructor is not checkable until then. The predefined exceptions are
symbols of another unit, which the model has not got. The examples
raise blocks made by `Make_block` and read their fields.

## 3. How it meets C2, C3, C4 and N1

C2 lands first. Agreed with its owner (messages of 2026-11-11):

- `hm_is_exn_handler : bool` is a field of `handler_metadata`, false in
  what `make_handler_metadata` builds, with a second function for an
  exception handler. C5 adds it in its epoch. Then
  `FlambdaTrapSpec.exn_handlers_well_formed` reads the flag where the
  prototype looks for uses (`raises_to`): a continuation that a trap
  action or a call uses as an exception handler has the flag, and one
  that has it is not applied as an ordinary continuation. C0's
  `handler_at_toplevel` can read it too. The prototype does not have
  the flag because the record is not on its base.
- `ap_exn_extra_args : list (simple * flambda_kind)`, as the compiler
  has it. The machine uses the first components. C2's rule 5 becomes:
  the exception continuation of an `Apply` has arity `Value` followed
  by those kinds; its checker compares each kind with that of the
  simple.
- An exception handler's first parameter has kind `Value`: C2's checker,
  which is a diagnostic and no premise, may test it now that the flag
  exists. It does not in this epoch (`handler_well_kinded` does not
  read the flag). Whether it is also a condition of well-formedness is
  decision 4.
- `cont_well_kinded` gains the case of `Extra_args`; C2 adds it when it
  rebases.
- C2's step B changes `call_function`; this request does not, so the
  two do not meet there.

C3. Its request changes neither `apply_expr`, `code`, `call_function`
nor `cont_value`, and this one changes neither `named`, `value` nor
`static_const`. The two meet only in the lists of constructors that
proofs go through.

C4. No field of the state comes from here and `Exit` is as it was, so
there is no joint sweep unless decision 1 goes the other way. C4 has a
case to add wherever it goes through `cont_value`: `Extra_args` holds
values, so locations.

N1. For the pilot: the new forms have their renaming and free names,
proved to commute. Owed, when N1 asks: that `traps_well_formed` and
`exn_handlers_well_formed` are unchanged by a renaming of the term and
of the unit's exception continuation. They compare names with
`Pos.eqb` only and keep no name in a value. `traps_well_formed` does
keep names in a list while it goes through the term; it refuses a
binder of a name that is on that list, so a name there means one
binding.

## 4. Statements that change meaning

No statement of `FlambdaSemanticsSpec.v` changes its text. `behaves`,
`equivalent`, `refines` and `Exit` are as they were. What a reader
should know:

1. Every statement quantified over an `apply_cont`, an `apply_expr`, an
   `expr` or a `cont_env` now also speaks of trap actions, extra
   arguments and `Extra_args`. All are proved again. Trap actions make
   no difference to any behaviour, so for instance
   `switch_same_arms_statement` holds of arms that have them.
2. `cont_free_in_expr`: a continuation that only a trap action names is
   free. So `unused_let_cont`-style statements do not remove the
   binding of a handler that is only pushed and popped. Removing it
   with its trap actions is the demotion, which is not a statement yet.
3. `inlinable`, hence `inline_statement`, `inline_let_code_statement`
   and the validator's `inline_calls`: a call with extra arguments is
   not inlined. The inlined body would raise to the caller's handler
   with the exception alone.
4. `let_cont_apply_cont_statement`, its one-parameter form,
   `let_cont_recursive_apply_cont_statement` and `loop_forever`: about
   applications with no trap action.
5. `expr_eqb_sound_statement`: the comparison is stricter (trap
   actions, raise kinds, extra arguments), so it accepts less.
6. `well_formed_never_faults_statement`: unchanged in text;
   `raises_to` inside `well_formed_unit` is wider (section 1).
7. In `FlambdaSemanticsProofs.v`, for whoever proves through the step
   lemmas: `step_apply` has a third match;
   `step_apply_no_extra_args` is the old form, under
   `ap_exn_extra_args ap = []`. `apply_cont_value` is a `Fixpoint`: a
   proof about applying a continuation rewrites with
   `apply_cont_value_target` and goes by `cont_target_cases` (a
   handler, or a way out, under its extra arguments), or uses
   `step_apply_cont_target_handler` and `step_apply_cont_target_exit`.
   `step_apply_cont_handler` and `step_apply_cont_exit` keep their
   statements.

## 5. Fallout

Measured on the prototype: 23 existing files, 725 lines added and 188
removed, of which the `Spec` and `Model` files are 207 added and 58
removed. `FlambdaTrapSpec.v` is 401 lines. By owner:

- C0, core: `FlambdaModel.v`, `FlambdaSemanticsSpec.v` and its
  `Proofs` (the step lemma, and the lemmas of item 7 above),
  `FlambdaContextProofs.v` (`closure_related` gains a constructor;
  the two cases of `transfer`), `FlambdaWellFormedSpec.v` and its
  `Proofs` (`cont_names` and `cont_ok` gain a case; two cases of
  `inv_step`).
- C0, kept for N1, S3, S4: `FlambdaNominal*` and `FlambdaAlpha*` (a
  constructor of `closure_renamed`, `apply_cont_value_renamed` by
  induction, the equivariance lemmas), `FlambdaControlFlow*` (five
  `None`), `FlambdaApply*` (the conjunct).
- C0, kept for V1: `FlambdaCompare*` (a constructor of `cont_matches`,
  `apply_cont_value_match` by induction), `FlambdaNormalise*` (six
  small repairs), `FlambdaUnusedStatics*` (a constructor of
  `cont_erased`).
- T1 and T3: `rocq/types/naive/FlambdaTypes*` and `FlambdaAliases*`: nine
  `None`, one `[]`, three lines of proof.
- C2, C3, C4, N1 on their branches: each relation or predicate on
  `cont_value` gains the case of `Extra_args`, and each place that
  applies `Apply_cont_expr` or `Apply_expr` gains an argument.
- The test tool (C0's until V2 and V3 have owners): the conversion of
  an `Apply_cont` and of an `Apply` no longer refuses a trap action and
  extra arguments (two small functions); printing; the tool says which
  condition of well-formedness a unit fails.

The pattern of the repairs: four proofs relate the continuation values
of two runs by an inductive relation, and each gains the constructor
"extra arguments around related continuations" and goes by induction
where it went by cases.

## 6. Trust

No axiom, no `Parameter`, nothing `Admitted`. The trusted conversion
grows by the conversion of a trap action and of the raise kind, one
constructor to one, of the extra arguments, each with its kind (the
prototype dropped the kinds, having none), and of the flag of a
handler. I have not edited the ledger. The item proposed for it (R1
has read the text; the converter itself is to be compared at the last
commit of the series):

> - **Exceptions**, in the same file (`convert_trap_action`,
>   `convert_raise_kind`, the extra arguments in `convert_apply`, and
>   the flag in the conversion of a handler; about 25 lines). Trusted:
>   that a trap action is the compiler's, with the continuation it
>   names and its raise kind, constructor for constructor; that the
>   extra arguments of the exception continuation of a call are the
>   compiler's, each with its kind, in their order; and that a handler
>   has the flag exactly when the compiler's has `is_exn_handler`. Of
>   these the machine reads only the extra arguments, so a wrong one
>   changes a run. A wrong trap action, raise kind, flag or kind
>   changes no run: it changes what the validator's comparison accepts
>   and what the checks of well-formedness say of the unit
>   (`traps_well_formed`, `exn_handlers_well_formed`), so the tool
>   could pass a unit that breaks the discipline or refuse one that
>   keeps it. Not cross-checked.

## 7. Conditions of well-formedness, proposed to C0

For the rows of `requests/C0-well-formed.md` ("exception handlers ...
With C5"). By the rule of D32, each that is stricter than the compiler
checks says so at its definition; the header of `FlambdaTrapSpec.v`
does.

Each line: the condition; the check that has it; where the compiler
establishes it.

- No exception handler is recursive. `exn_handlers_well_formed`.
  Creation function: `Let_cont_expr.create_recursive` fails.
- An exception handler has a first parameter.
  `exn_handlers_well_formed`. Pass: to_cmm fails
  (`let_cont_exn_handler`, `To_cmm_env.add_exn_handler`).
- With C2, and only if the human says so (decision 4): that
  parameter is of kind `Value`. Nothing: `add_exn_handler` tests that
  there is a parameter and not its kind (C5-7). So this one would be
  stricter than the compiler, about a kind. It is not part of what is
  asked to be applied at the epoch.
- An exception handler is applied only by a raise, and so is the
  exception continuation of code and of the unit.
  `exn_handlers_well_formed`. Pass, in part: to_cmm fails on an
  `Apply_cont` of an exception handler that has no `Pop`. The comment
  of `trap_action.mli` states the rule.
- Of the continuations that code, or the unit, does not bind, only
  the exception continuation is used as an exception handler: not the
  return continuation. `ways_out_well_formed`. For code, pass: to_cmm
  fails on a trap action that names the return continuation of a
  function (`translate_jump_to_continuation`,
  `To_cmm_env.get_cmm_continuation`). For the unit, nothing: to_cmm
  gives the unit's return continuation a handler of its own
  (`to_cmm.ml`, `unit0`) and lets the trap action through (R1-22).
- No call gives extra arguments to a continuation that code, or the
  unit, does not bind. `ways_out_well_formed`. When visited, and by
  accident: Simplify stops with `Invalid_argument "fold_left_map2"`
  (`Continuation_uses_env.add_continuation_use`), before the check of
  arities that has a message (C5-5). This is not a claim that a
  mismatch of arities can be ruled out statically: it is about who may
  be given extra arguments, which the front end and inlining decide,
  and a way out applied directly to the wrong number of values is left
  as it is.
- A `Pop`, a raise and the exception continuation of a call name the
  handler installed last. `traps_well_formed`. Nothing (C5-1); the
  front end builds it so.
- A continuation is entered with one stack of handlers, and there is
  no `Pop` on an empty stack. `traps_well_formed`. Pass:
  `backend/select_utils.ml`, `set_traps` and `combine_traps`.
- Code returns with the stack it started with, and raises to its
  exception continuation so. `traps_well_formed`. Nothing in
  Flambda 2.
- The continuation that a trap action names is in scope. C0's row 3,
  through `cont_free_in_expr`. Pass: `Upwards_env`, as for any
  continuation.

Two things the check does that a reviewer should weigh. A handler that
nothing enters is not checked, since it has no stack to be checked
under; the backend does not compile it either. And the check is by
names, so it refuses a term that binds again a name that is on the
stack; a unit of the compiler binds each name once.

What it rules out in a run, which C0's note asks of each condition:
nothing on this machine, which does not look at trap actions. It rules
out a difference between this machine and one that goes by the stack:
that is the proved statement that is false without the condition, and
`raise_over_handler.fl` is the term that fails it. Section 9 has the
statement as work after the epoch. Until it is proved the condition on
the stack has only examples for it, so it should not land long before
that leaf; the conditions on exception handlers do not depend on it
(the first of them is not stricter than the compiler, which has it by
a creation function that always checks).

## 8. Decisions for the human

1. **Is the stack of handlers part of the machine?**
   (a) No. The machine goes by the names; the discipline is a
   condition of well-formedness, with a theorem to follow that a
   machine with the stack agrees on well-formed units.
   (b) Yes. The state gains the stack; a `Pop`, a raise or a call that
   does not name the handler on top is undefined, with a CR on each.
   Recommendation: (a), for the reasons of section 2. Under (b) the
   static check is still needed, since the backend fails on code that
   never runs; the state gains a field, to be swept with C4's; and the
   machine needs a notion of which handler is which.
   Meanwhile: (a).
2. **Backtraces are not observed.** The raise kind is in the term and
   the validator compares it, and it changes nothing. So the model
   allows any raise to become a jump, where Simplify does so only for
   `notrace`, or when debugging information is off (C5-4).
   Recommendation: not observed. Meanwhile: so.
3. **The discipline is stricter than anything the compiler checks.**
   Adopt `traps_well_formed` and `exn_handlers_well_formed` as
   conditions of well-formedness, with the comment that D32 asks for;
   or only what the backend checks (one stack for each continuation,
   no `Pop` on an empty stack). Recommendation: adopt both. A unit of
   the compiler that fails is then a finding. The evidence so far is
   small: six examples written after what `-drawfexpr` prints for
   `try`, raise and a mutable variable, before and after Simplify. The
   converter refuses most real units, so no corpus has been through
   it.
4. **Must the first parameter of an exception handler be declared of
   kind `Value`?** After C2 a handler declares the kinds of its
   parameters. The compiler tests that an exception handler has a
   parameter and not its kind (C5-7), so the condition would be
   stricter than the compiler, and it is about a kind. D32 lets the
   model be stricter than the compiler but not on kinds, because real
   programs break agreement of kinds. This is a condition on what one
   handler declares, which the front end always declares so, and not
   an agreement between two places of a program; whether D32's
   exclusion covers it is the human's to say.
   (a) Ask for it, as a condition of well-formedness, with the comment
   of D32.
   (b) Do not: a first parameter, of any kind.
   Recommendation: (a), as no program is known to break it.
   Meanwhile: (b). The condition is not written and is not applied at
   the epoch without a ruling.

## 9. The plan for the epoch

After C3, C2 and N1's change (the renaming that reaches into code, and
one function for free names), in that order, as C0 has scheduled them.
Each commit passes `make -C rocq -j16 all` and `assumptions`. On that
base the free continuations of a term come from N1's function, which
gains the continuation of a trap action, in place of C0's
`free_continuations`; and exception handlers are read from C2's
record, in place of `raises_to`.

1. The terms and the machine, with every repair of section 5: the
   first commit of the prototype rebased, with `ap_exn_extra_args` as
   pairs. Mechanical but for the four relations on continuation
   values.
2. `hm_is_exn_handler`, `raises_to` reading it, and the header of
   `FlambdaModel.v`.
3. The two checks, reading the flag, in `FlambdaWellFormedSpec.v`
   with their rows, and the conjuncts in `well_formed_unit`, applied
   by C0; `FlambdaTrapSpec.v` and `FlambdaTrapProofs.v` with the
   examples and their lines in the shared lists. The check of the
   stack may wait here for the leaf below.
4. The test tool and the examples, applied by C0 with the regenerated
   extracted files.

A day to rebase and adapt, a day for C0 to apply and validate, and the
reviewer's reading of the `Spec` diff, which is small outside the new
file.

Leaf work after it:

- Mine. A machine with a stack of handlers and of calls, which
  performs trap actions, raises to the handler installed last and
  sends the exceptions of a call there; and the statement that a unit
  that passes the two checks has the same behaviours on both machines.
  It is what the conditions of section 7 buy, and until it is proved
  the check is justified by examples only.
- Mine or N1's: the two checks commute with renaming.
- S3: demoting an exception handler (removing its trap actions and
  making its raises jumps) as a rewrite the validator knows.
  `try_raise.fl` is not validated for want of it.
- S4: inlining a call with extra arguments, through the wrapper that
  `Inlining_helpers.wrap_inlined_body_for_exn_extra_args` builds; and
  that the wrapper is equivalent to the call.

## What the prototype proves, and what it leaves

Run on the prototype at its tip: `make -C rocq -j16 all` (the claims pass
and no statement is lost), `make -C rocq assumptions`,
`make -s boot-compiler` and the semantics tests (34 examples).

The examples, in `tests/semantics/`:

- `try_raise.fl`: a `try` in the unit, a raise in one arm, a pop on
  the way out in the other. Simplify demotes the handler and removes
  every trap action; both runs return the same.
- `raise_in_call.fl`: the raise is in a function called from the body
  of the `try`. Validated: the same once normalised, trap actions
  compared.
- `exn_extra_args.fl`: the same with an extra argument, which the
  handler adds to a field of the exception.
- `try_in_code.fl`: a function whose body is a `try` around an
  indirect call, with a handler that reraises to the function's
  exception continuation or returns; called three times, the third
  inside a `try` of the unit. Validated.
- `uncaught.fl`: a raise to the exception continuation of the unit;
  both runs raise the same block.
- `raise_over_handler.fl`: the example of section 2; the tool refuses
  it as not well-formed.
- `push_unit_return.fl` (R1's): the unit installs its return
  continuation as an exception handler and raises to it; the tool
  refuses it as not well-formed. Before the repair it was reported as
  validated.

Left: `make -s test` and `make -s fmt` have not been run (the tool was
formatted with ocamlformat). The flag and the kinds of the extra
arguments wait for C2. There is no theorem about the check. The checks
have not been run on a unit that the compiler made from OCaml source.

## 10. The series for the epoch

Merged. Its base was the integration branch at c77399d527, which
has C3's and C2's epochs, the conditions on names
in `well_formed_unit` and the renaming that enters code. The series
was written on C2's series before those last two and replayed as the
branch moved; in the last replay, by N1, the owners of the files that
had landed since repaired them in the series (`status/C5.md` names
them). Every commit passes `make -C rocq all` and
`make -C rocq assumptions`; the last needs the extracted files
regenerated.

1. **The terms and the machine.** The first commit of the prototype,
   over C2 and C3. What differs from the prototype:
   - `ap_exn_extra_args : list (simple * flambda_kind)`, as agreed with
     C2 (section 3), placed after `ap_exn_continuation` and before the
     two arities. `exn_extra_simples ap` is the list of the simples;
     the machine, the free names, the scope check and everything else
     that wants a list of simples go through it. Renaming and the
     following of aliases act on the first components. `apply_eqb`
     compares the simples up to the pairs of bound names and the kinds
     exactly.
   - C2's checker (`FlambdaWellKindedSpec.v`), agreed with C2: the
     continuation of a trap action is in scope
     (`trap_action_in_scope`); the extra arguments are of the kinds
     written beside them; the exception continuation of a call takes
     `Value` followed by those kinds; `arity_of_cont` and
     `cont_well_kinded` have the case of `Extra_args` (the arity of the
     continuation inside without its last kinds; the values kept are
     well-kinded and of those last kinds); `simples_read` has the
     extra arguments; `wrong_number_of_arguments` counts the values
     kept. No statement of C2's is weakened.
   - C3's invariant of sets of closures
     (`FlambdaClosuresProofs.cont_sets_ok`) has the case of
     `Extra_args`; `apply_sibling_statement` is for any extra
     arguments.
   - What the integration branch gained while the series was written,
     repaired in the replay: the statement that a property of values
     is kept along a run (`cont_values_all` goes into `Extra_args`,
     and the hypothesis on a call has the values of the extra
     arguments); the scope and the ranges of a state (`cont_in_scope`,
     `cont_in_range`), and `apply_in_range`, which asks the extra
     arguments; the free names of `FlambdaBindersSpec.v` (the extra
     arguments in `free_variables_of_apply`, the continuation of a
     trap action in `free_continuations_of_expr`); the test
     `cont_free_b` of `FlambdaUnusedContsSpec.v`, for which a
     continuation that only a trap action names is used; the action
     of renamings on states (`rename_cont_value` renames inside an
     `Extra_args`), with the equivariance of a step; and
     `apply_cont_canonical_statement` of
     `FlambdaTypingEnvRefinementSpec.v`, now for any trap action.
2. **The flag.** `hm_is_exn_handler : bool` in `handler_metadata`;
   `make_handler_metadata` gives false and `make_exn_handler_metadata`
   true; `handler_metadata_eqb` compares it; `handler_is_exn_handler`
   reads it. The header of `FlambdaModel.v`.
3. **The checks**, in `FlambdaWellFormedSpec.v` as C0's rule asks
   (section 1, "Where they end up"), with the examples in
   `FlambdaTrapProofs.v`. There is no `FlambdaTrapSpec.v` in the
   series: it comes with the first statement about the checks.
   `traps_well_formed` and `ways_out_well_formed` are those of the
   prototype. `exn_handlers_well_formed` goes by the flag where the
   prototype went by uses:
   - a handler with the flag is not applied as an ordinary
     continuation in the body of its `Let_cont`, and has a parameter
     for the exception;
   - a handler without it is not used there as an exception handler
     (no trap action names its continuation and no call has it as its
     exception continuation);
   - no handler of a recursive `Let_cont` has the flag, and none is
     used as an exception handler.
   The second is new: the prototype could not say it. It is stricter
   than the compiler, which never compares the flag with the uses
   (C5-8). Two more conditions, found while the machine with a stack
   was designed: the exception continuation of code, or of the unit,
   is named by calls and by the trap actions of raises only, in
   `ways_out_well_formed` (to_cmm stops on another trap action that
   names it; C5-9); and `trap_demands` tests, after its passes over the
   handlers of a recursive group, that every handler that is entered
   has been checked, where the prototype relied on the number of
   passes. The stack means what it says only where the return and
   exception continuations of code are two names and the
   continuations of a recursive `Let_cont` are distinct: both are in
   `binders_distinct` (row 5), which a well-formed unit is asked
   first, so these checks do not ask them again. Thirty-six examples:
   the twenty-five of the prototype, four for the flag and seven for
   these. Also here: the CR at `with_extra_args`, the paragraph on
   exceptions in the header of `FlambdaSemanticsSpec.v`, and this
   file.
4. **The shared lists**: `FlambdaTrapProofs.v`; the checks and
   `make_exn_handler_metadata` among what is extracted.
5. **The conjuncts.** `well_formed_unit` asks
   `unit_exn_handlers_well_formed` and `traps_well_formed` of the
   unit (decision 3, taken as "adopt both" while the human is away),
   and `handler_at_toplevel` reads the flag where it read `raises_to`:
   with the first conjunct, a continuation used as an exception
   handler has the flag, so nothing that was refused is accepted. C0
   applies this commit, or writes its own.
6. **The test tool and the examples.** The conversion of a trap
   action, of the extra arguments with their kinds, and of the flag;
   printing; the messages. The seven examples of the prototype.

Rows for the table of `requests/C0-well-formed.md`, in its form:

| | Condition | Owner | Where the compiler establishes it |
|-|-----------|-------|-----------------------------------|
| 11a | No handler of a recursive `Let_cont` is an exception handler | C5 | Creation function: `Let_cont_expr.create_recursive` fails |
| 11b | An exception handler has a first parameter | C5 | When visited: to_cmm fails (`To_cmm_env.add_exn_handler`) |
| 11c | A handler that is an exception handler is not applied as an ordinary continuation: by an `Apply_cont` that is not a raise, or as the return continuation of a call. Likewise the exception continuation of code and of the unit | C5 | When visited, in part: to_cmm fails on an `Apply_cont` of an exception handler that has no `Pop` (`To_cmm_expr.translate_raise`). Nothing for the return continuation of a call (C5-8) |
| 11d | A handler that is not an exception handler is not named by a trap action nor as the exception continuation of a call | C5 | Nothing (C5-8). The front end marks the handlers it installs; Simplify removes the trap actions when it clears the mark |
| 11e | Of the continuations that code, or the unit, does not bind, only the exception continuation is used as an exception handler | C5 | Code, when visited: to_cmm fails on a trap action that names the return continuation of a function (`To_cmm_env.get_cmm_continuation`). The unit: nothing (R1-22) |
| 11f | No call gives extra arguments to a continuation that code, or the unit, does not bind | C5 | When visited, by accident: Simplify stops with `Invalid_argument` (`Continuation_uses_env.add_continuation_use`; C5-5) |
| 11g | The exception continuation of code, or of the unit, is named by calls, as their exception continuation, and by the trap actions of raises, and by no other trap action | C5 | When visited: to_cmm stops, as it does not find that continuation among those a trap action of a jump can name (`To_cmm_env.get_cmm_continuation`; C5-9) |
| 12a | A `Pop`, a raise and the exception continuation of a call name the handler installed last | C5 | Nothing (C5-1); the front end builds it so |
| 12b | A continuation is entered with one stack of handlers; no `Pop` on an empty stack | C5 | When visited, in the backend: `backend/select_utils.ml`, `set_traps` and `combine_traps` |
| 12c | Code returns with the stack it started with, and raises to its exception continuation so; the unit likewise | C5 | Nothing in Flambda 2 |

That the continuation a trap action names is in scope is row 3. That
the return and exception continuations of code are two names, and the
continuations of a recursive `Let_cont` distinct, is row 5: rows 12a
to 12c are for a term that passes it.

Not in the series:

- That the first parameter of an exception handler is of kind `Value`
  (decision 4): not written, as it waits for a ruling.
- The free continuations of a term from N1's function
  (`FlambdaBindersSpec.free_continuations_of_expr`), in place of C0's
  `free_continuations`: `ways_out_well_formed` uses C0's function.
  Both have the continuation of a trap action from the first commit,
  and `FlambdaFreeNamesSpec.v` states that they give one list on a
  term whose code is closed.
- The leaf work of section 9. The first item of it, the machine with
  a stack and the statement that the two machines agree on a unit
  whose binders are distinct and that passes `traps_well_formed`, is
  done since (`FlambdaTrapSpec.v`): so
  the condition on the stack no longer has only examples for it.
