# C0: the body of a `try` at the top of a unit. A request

Owner: C0. For the orchestrator and the human. Words only: this note
changes no definition and no proof. R2 read it against the compiler
and against the model, and what it found is in. Discrepancy T5b-4;
section 8 is on X5-9, a second shape that the proposal does not
cover.

## 1. The case

A unit with a `try` at its top, whose body makes a function that
captures nothing and gives it to a call that stays:

    let[@inline never] apply f x = f x
    let result =
      (try apply (fun v -> v + 1) (opaque 1) with Not_found -> 0)
      + apply (fun v -> v) 2

Simplify makes the closure a symbol and binds it where it was made, in
the continuation that is the body of the `try`. With `k_body` for that
continuation, `k_exn` for the handler of the `try`, `k_ret` for where
the result of the body goes and `k_join` for what follows the `try`:

    cont k_body push(k_exn)
      where k_body = let $closure = ... in
                     apply ... -> k_ret * k_exn
      where k_ret (r) = cont k_join pop(k_exn) (r)
      where k_exn exn (e) = ...

The compiler takes that term back (its own tool that runs Simplify on
a term written as text accepts it) and the program runs. The test tool
stops: "the term after Simplify is NOT WELL-FORMED (a static binding
that is not at toplevel, or a name bound twice)". It gives no verdict;
it is not wrong by it.

## 2. The two rules

The compiler (`simplify/simplify_let_cont_expr.ml`, `at_unit_toplevel`
for the handler of a non-recursive continuation): the handler is at
toplevel when its `Let_cont` is, it is not an exception handler, and
the continuations that the body uses are among the continuation
itself and the exception continuation of the unit. The uses are those
recorded while the body is simplified, without those of the
continuations bound inside it. One is recorded for the continuation
that an `Apply_cont` applies (`simplify_apply_cont_expr.ml`), for that
of each arm of a `Switch` that is kept (`simplify_switch_expr.ml`),
and for the exception continuation of a call and its return
continuation where it has one (`simplify_apply_expr.ml`). None is
recorded for the handler that a trap action names. Simplify does not
go into a handler that nothing applies, so what is in one is not a
use either. The comment there gives the aim: the handler
postdominates the body.

The model (`FlambdaWellFormedSpec.handler_at_toplevel`): the handler
is not an exception handler and `free_continuations` of the body are
among the same two. Of an `Apply_cont`, `free_continuations` takes
`FlambdaModel.apply_cont_continuations`: the continuation applied and
the one of the trap action. Its own comment says "the continuations an
expression may apply, or give to a call", which is what the compiler
goes by. The function gives more than its comment says.

Above, the body of the binding of `k_body` is `cont k_body
push(k_exn)`. For the compiler it uses `k_body` alone and the handler
is at toplevel. For the model `k_exn` is free in it and the handler is
not.

The compiler's rule is the one that fits the aim. Where an exception
goes is said by the term: a raise applies its handler and a call names
its exception continuation, and both are uses. A `push` or a `pop`
moves no control. The machine of the model does not execute a trap
action at all (FlambdaSemanticsSpec.v: "the machine has no stack of
handlers"); the second machine (FlambdaTrapSpec.v) has the stack, and
on a term that passes `traps_well_formed` it goes where the first
goes. So the model's rule is stricter than the compiler's for no
reason of its own: it counts a name as a jump.

## 3. What I propose

Done, with decision D60: the function is
`FlambdaWellFormedSpec.continuations_used`, and
`handler_at_unit_toplevel` (the name since of `handler_at_toplevel`)
asks its condition of it. `continuations_used_free_statement` says
that the new list is within the old one, and
`try_body_at_unit_toplevel_statement` is the unit of section 1 as a
computed example. The same stage answers section 8 otherwise than
this note proposed: `well_formed_unit` no longer asks this rule, but
`statics_outside_loops_and_code`. So, of what the last paragraph of
this section lists as not changed, a static binding in an exception
handler is now taken by `well_formed_unit`, and is still not at
toplevel by Simplify's rule; and the sentence of the header that
this section says stays word for word now speaks of the
continuations that the body uses. The rest of the section is as it
was written.

One function more in FlambdaWellFormedSpec.v, beside
`free_continuations` and by the same recursion: the continuations
that an expression applies or gives to a call, and does not bind
itself. It differs in one place: of an `Apply_cont`, and of an arm of
a `Switch`, it takes the continuation applied and not that of the
trap action. `handler_at_toplevel` asks its condition of that list.

The list is read off the text of a term. It has every use that the
compiler records for that term and may have more: it goes into a
handler that nothing applies and counts an arm that Simplify would
remove. So where the two still differ the model is the stricter one,
as it is today. What goes is the name in a trap action.

`free_continuations` stays as it is. Four other things read it and
mean every name: `ways_out_well_formed` (same file),
`FlambdaHoistCodeSpec.statics_at_toplevel_of_unit`,
`FlambdaFreeNamesSpec.free_continuations_agree_statement` and
`well_formed_free_continuations_equivariant_statement` of
FlambdaValidatorEquivarianceSpec.v. Its comment is corrected to say
what it gives.

The new list is within the old one, and the condition is that a list
is among two names. So a handler that is at toplevel today stays so:
`statics_at_toplevel` and `well_formed_unit` accept every term that
they accept today, and others. The sentence on toplevel in the header
of the file ("when all its body can do is to apply that continuation
or raise to the exception continuation of the unit") stays word for
word.

Not changed by it: a static binding in an exception handler stays
refused (C5-11; decided overnight, entry 23 of status/C0.md), and so
does a group of code alone away from toplevel (C0-3).

## 4. The landed statements that the definition reaches

Counted on this tree by the search of the changed definitions: 46
statements, 1 that names `handler_at_toplevel` and 45 through
`statics_at_toplevel`, `well_formed_unit` and the functions that test
it (`hoist_code`, `hoist_statics`, `lift_unit`, `lift_to_head_unit`,
`lift_from_code_unit`, and through them `with_code_first`,
`with_statics_first`, `with_lifting`, `lifted_in_turn`,
`validate_unit`, `validate_unit_knowing`). No statement gains a
hypothesis. By what happens to each:

- **The conclusion is weakened, four statements.** Three say of
  given terms that they pass the check, and the check asks less:
  `FlambdaClosuresSpec.parity_well_formed_statement` and
  `examples_well_formed_statement` of FlambdaPartialApplicationSpec.v
  and of FlambdaTupledSpec.v. The fourth,
  `FlambdaStaticsInScopeSpec.rebound_unit_exn_reads_unbound_statement`,
  says that there is a unit that passes `statics_at_toplevel` and
  every other condition but one and reads a symbol that is not
  bound: a unit that passes the new check is less to ask for. The
  three terms, and the unit that the proof of the fourth gives, pass
  the check of today too; the statements would no longer say so.
- **The conclusion says more, two.**
  `FlambdaClosuresSpec.partial_static_set_statement` says of a given
  term that it fails `well_formed_unit`; the term has no `Let_cont`,
  so nothing changes for it.
  `branch_round_binding_reads_unbound_statement` of
  FlambdaStaticsInScopeSpec.v says that there is a unit that fails
  `statics_at_toplevel`, passes the other conditions and reads a
  symbol that is not bound: one that fails the new check is more to
  ask for. The unit that its proof gives fails the new check too,
  because an arm of its `Switch` applies another continuation. To be
  computed again.
- **The premise is that a unit is well-formed, sixteen.** The
  premise is weaker, so they say the same of more units: eleven of
  FlambdaClosuresSpec.v
  (`set_of_code_bound`, `sibling_of_code`, `sibling_symbol_of_code`,
  `well_formed_call_binds_my_closure`, `well_formed_closure_slot`,
  `well_formed_code_binders`, `well_formed_exit_valid`,
  `well_formed_makes_closures`, `well_formed_sets_bound`,
  `well_formed_sets_inv`, `well_formed_values_valid`),
  `well_formed_in_range` (FlambdaRangeSpec.v),
  `statics_bound_in_run` (FlambdaStaticsInScopeSpec.v),
  `well_formed_names_bound` and `well_formed_never_faults`
  (FlambdaWellFormedSpec.v), and `lift_simulates` (FlambdaLiftSpec.v),
  each with `_statement`.
- **The premise is that a hoisting or a lifting answers, five.**
  `hoist_code_program_equivalent_statement` (FlambdaHoistCodeSpec.v),
  `hoist_statics_program_equivalent_statement`
  (FlambdaHoistStaticsSpec.v), `lift_unit_program_refines_statement`
  (FlambdaLiftSpec.v), `lift_to_head_unit_program_refines_statement`
  (FlambdaLiftFromArmsSpec.v) and
  `lift_from_code_unit_program_refines_statement`
  (FlambdaLiftFromCodeSpec.v). The check is the only test of these
  functions that changes, so each answers what it answers today
  where it answers today, and answers on other terms too: a term in
  which a static binding is in a handler whose body names another
  continuation in a trap action only. The statements say the same
  of more terms.
- **The premise is a verdict of the validation, six.**
  `validate_unit_sound_statement` and
  `validate_unit_sound_everywhere_statement` (FlambdaValidateSpec.v),
  the same two of `validate_unit_knowing`
  (FlambdaValidateKnowingSpec.v), and
  `validate_unit_sound_imports_statement` and
  `validate_unit_knowing_sound_imports_statement`
  (FlambdaValidateImportsSpec.v). The validation compares the two
  terms after rewrites that go through the functions above
  (`with_code_first`, `with_statics_first`, `with_lifting`,
  `lifted_in_turn`), which now rewrite a term that they left as it
  was. So it is another function on such terms: nothing says that
  a pair accepted today is accepted then, nor the reverse. The
  sentences are the same and are to be proved of it.
- **Equations under a renaming and rewrites that keep the scope,
  ten.** `validate_unit_equivariant`,
  `validate_unit_knowing_equivariant`, `validate_unit_renamed_alike`,
  `lifted_in_turn_equivariant_up_to`, `lifted_in_turn_keeps_scope`,
  `with_code_first_equivariant`, `with_lifting_equivariant`,
  `with_statics_first_renamed_alike`, `with_lifting_keeps_scope`,
  `with_statics_first_keeps_scope`, each with `_statement`. The
  same sentences, of functions that rewrite more terms.
- **The statement is the equation of the check under a renaming,
  three**: `handler_at_toplevel_equivariant`,
  `statics_at_toplevel_equivariant`, `well_formed_unit_equivariant`
  (FlambdaValidatorEquivarianceSpec.v). To be proved of the new
  function.

No verdict of the tool has been measured with the change, which is
not written; whether every unit that is validated today stays so is
part of that measure.

## 5. The proof that rests on the rule of today

By name, two files of proofs open the definition:
FlambdaValidatorEquivarianceProofs.v, for the three equations, and
FlambdaStaticsInScopeProofs.v, in the step lemma behind
`statics_bound_in_run`. The others go by lemmas.

The second is the one that needs thought. Its invariant says, of
every continuation free in the expression of a state, that the
continuation is good under the scope that the expression is still to
bind. In the case of a `Let_cont` whose handler is at toplevel, the
handler's own bindings leave that scope, and the proof gets rid of
every other free continuation of the body by the rule: it is the
exception continuation of the unit. With the new rule a continuation
that only a trap action names can be free there, and nothing says it
is good under the smaller scope. The invariant has to range over the
continuations applied or given to a call, in the invariant of a state
and in what is asked of a closure. I have read the cases and expect
them to go through as they are, since each takes the invariant at the
continuation that the step applies; I have not done it.

## 6. The scope of the symbol after the `try`

The rule of scope (FlambdaStaticsInScopeSpec.v) has: what the body of
a `Let_cont` binds is in scope in its handlers. Its justification in
that header is what the body can do before it enters a handler, and
it reads the same with the new function: a handler binds something
static only if its body can do nothing but apply its continuation or
leave the unit.

In the term of section 1 the symbol is bound in the handler of
`k_body`, which is in the body of the `Let_cont`s of `k_ret`, of
`k_exn` and of `k_join`. So it is in scope in the three handlers. It
is bound when each is entered: the body of each goes down to `cont
k_body`, the handler of `k_body` binds the symbol before its call,
`k_ret` and `k_exn` are reached only by that call, and `k_join` only
from their handlers. The
handler of `k_ret` is not at toplevel under either rule (the body of
its `Let_cont` gives `k_exn` to a call), nor that of `k_exn` (it is
marked as an exception handler); that of `k_join` is, under both. A
body of a `try` that could raise to `k_exn` before the binding has
the binding in the handler of a continuation whose body gives `k_exn`
to a call or applies it, and that handler is at toplevel under
neither rule. A body that leaves the unit by its exception
continuation before the binding enters no handler of that
`Let_cont`, which the header of that file allows already. The clause
stays right only of a unit that passes row 1 and
`interface_conts_not_rebound`, as that header says; both stay
conditions of the statement.

## 7. How many units

In this tree none: no unit of the corpus and none of the generated
ones has that output (the seven expected files that say "not
well-formed" are terms written by hand to fail other conditions, and
one a static binding in a loop). Outside it, in the units written to
try exceptions (70, at two settings of the flags), one, at both
settings; in 124 units of real code, none (the tool does not read 70
of them). It takes a `try` at the top of a unit whose body makes
something that Simplify lifts to a symbol where it is made.

## 8. A second shape, which the proposal does not cover (X5-9)

A unit that divides by a number it does not know, at its top:

    let result = opaque 17 / opaque 5

The compiler's own term after Simplify, with `k_zero` for the handler
of the division by zero and `k_div` for the rest of the unit:

    switch (divisor <> 0) | 0 -> k_zero | 1 -> k_div
      where k_div = let result = ... in
                    let $module = Block 0 (result) in
                    cont done ($module)
      where k_zero = cont error pop(error) (Division_by_zero)

The front end marks `k_zero` cold. For this handler, which takes no
argument and is used from an arm of a `Switch`, Simplify keeps the
binding: a cold continuation that it does not inline has its handler
unknown to its uses (`simplify/env/upwards_env.ml`,
`add_non_inlinable_continuation`), so the arm is not replaced by the
handler. (Cold does not forbid inlining everywhere: a continuation
used once may still be inlined.) So
the arm stays a jump, and in the output the binding of `k_zero` is
around that of `k_div`. The body of the `Let_cont` of `k_div` is the
`Switch`, which applies `k_zero`: neither `k_div` nor the exception
continuation of the unit.

This is not the case of section 1. There the compiler takes its
output back; here it does not: given that term, Simplify stops ("Let
binding symbols is only allowed at the toplevel of compilation
units"), by the rule of section 2, as it does for C5-11. The model
refuses the term by the same rule, today and with the function of
section 3, which counts `k_zero` as it is applied. The test tool does
not meet it as it is run today: the text of the term before has no
mark of cold, the tool's Simplify puts the raise in the arm, and the
unit is validated. A tool given the compiler's own two terms meets it
on two units of the corpus, `div_unknown` and `string_get16_checked`,
at both settings of the flags. It can arise wherever a cold branch
stays bound around a continuation that binds something static; only
those two units are established here.

What would accept it: the body of the `Let_cont` may also apply a
continuation bound around it whose handler can only leave the unit,
as `k_zero` does. The handler of `k_div` is then still entered on
every path of the body that does not leave the unit, which is what
the rule is for. It is more than one function:

- `statics_at_toplevel` has to carry down which continuations in
  scope only leave the unit, so `handler_at_toplevel` takes one
  argument more and the statement of its renaming changes its text.
- The justification of the clause of scope of section 6 fails under
  the wider rule. The arm for zero enters the handler of `k_zero`
  before the handler of `k_div` has bound the module block (the run
  shown, with 5, does not take it), and by the clause the block is
  in scope there: a handler that only leaves the unit could load the
  symbol first. So that the body can only leave is not enough. The
  rule of scope needs an exception, and the blunt one (nothing that
  the body binds is in scope in a handler that only leaves the unit)
  refuses units that pass today, where the binding is on the spine
  of the body before the jump: that strengthens a premise of the
  statements that take the check of scope. Which names are safe in
  such a handler is for the separate request to say. In the example
  the handler names a predefined exception and nothing of the unit.
- The invariant of section 5 needs a third sort of continuation, good
  under the scope as it was where the continuation was bound.

C5 observes that the wider rule has two parts that could be kept
apart. That a static binding is executed once at most holds of the
handler of any non-recursive continuation that is neither in code nor
in the handler of a recursive one, whatever its body applies: nothing
bound inside a `Let_cont` runs that `Let_cont` again. What the list
is needed for is the scope alone, the second point above, and that
part could go in the check of scope.

And it is of another kind than section 3: the model would accept a
term that Simplify refuses when it is given it. I do not propose it
with the change of section 3. It is the question of C5-11 again, in
the shape that is common, and it comes up when the tool is run on the
compiler's own two terms.

## 9. What I ask

1. The orchestrator: whether to make the change, given the four
   statements whose conclusion is weakened. The other course is the
   one taken for C5-11: keep the rule, and have the tool report such
   a unit under the label T5b-4 without checking it.
2. If yes: the definition and the header are mine; the invariant of
   FlambdaStaticsInScopeProofs.v is C2's; the three equations are
   N3's; the unit of section 1 becomes an example of the corpus (V2).
   R2 reads the new function against the compiler's uses, as the
   header of the file asks for `statics_at_toplevel`.
3. The orchestrator, for X5-9: one of these. Keep the rule and have
   the tool report such a unit under the label without checking it,
   as for C5-11; a unit that divides at its top is then not checked
   when the tool is given the compiler's own terms. Or the rule of
   section 8, as a series of its own with its list of statements,
   after the change of section 3. Or have the tool put a cold handler
   that only raises back in the arm before the check: a verdict is
   then of the rewritten term, unless the rewrite is proved to relate
   it to the compiler's term in the direction the verdict needs, or
   a new trust is authorised for it and read by R1; a sentence in
   the ledger is not enough. I recommend the first now and the
   second when the tool
   runs on the compiler's own terms. A fourth, from C5:
   `validate_unit_sound_statement` has no premise of well-formedness,
   so the tool could report the label and still run and validate such
   a unit where the validation needs no lifting (a lifting tests the
   well-formedness of its result and refuses by itself). The tool and
   the trust ledger say today that a term after that is not
   well-formed ends the check; that wording would change. It would
   not go round what is asked of a unit with imports: that statement
   is of a unit run on its own, the unit of section 8 names a
   predefined exception, and its verdict would still be one that the
   theorem from the imports covers (`holds_from_imports`), printed as
   such. Neither C5 nor I propose it now.
