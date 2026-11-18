# Controls of the scope of static names in a handler

The test tool asks two tests of each term before it prints a verdict
(`../../../tools/flsemantics.ml`, "The check of static names"). The
second is the check of static names,
`FlambdaStaticsInScopeSpec.statics_in_scope`: every symbol and every
code id that a unit mentions is in scope where it is mentioned. Since
decision D60 the handler of a continuation has in scope only what is
bound on every way to it (`exits`, `handler_scope` and
`recursive_scope` of that file).

The 29 texts of this directory are controls of that rule, written by
hand. 15 must be refused and 14 must pass. The first words of the
comment of a text say which, MUST BE REFUSED or MUST PASS, and the
comment says why. `scope.expected` is what the tool says of them.

## The runs

`run.sh` runs the tool on each text, each time in a directory that
holds the texts given and nothing else:

- alone: the text is the term before, and the tool's Simplify makes
  the term after if the guard lets the term through;
- both: the text is the term before and, by the option `-after`, the
  term after; Simplify is not run;
- and for a text that must be refused, beside a plain unit of the
  same name, which binds one block and nothing else: as the term
  after of the plain unit, and as its term before.

What the tool says of a text that must be refused: as the term before,
"not checked: the term mentions a symbol where it is not in scope" (or
"a code id"), with the status 0; as the term after, "the term after
Simplify mentions a symbol that is NOT IN SCOPE" (or "a code id"),
with the status 1. No run of such a text has the line of the check, a
run of the machine or a verdict.

What the tool says of a text that must pass: "the check of static
names: passes before, passes after", in 26 of the 28 runs. The other
two are `symbol_handler_never_entered.fl` and
`symbol_join_other_name_on_one_way_not_mentioned.fl`, each alone: the
guard lets the term through, and the compiler's Simplify, which takes
a static binding at the toplevel of a unit only, stops with a fatal
error and the status 2, as it does on `../static_in_arm.fl`.

A text that passes the check is not thereby one that the validation
accepts. The verdicts of the fourteen are in `scope.expected` as they
came. Where one text is given as both terms, "not validated" is no
evidence that the two terms differ, if the text binds code: the dune
file of the directory above says why, of `after_same_text`.

`run.sh` holds each text to the first words of its comment: of a text
that must be refused, every run has to say that a name is not in
scope, and of a text that must pass, no run may say so. Otherwise it
prints "NOT AS ITS COMMENT SAYS" and fails, whatever `scope.expected`
has.

## The controls

In the table a join is a handler with two entries, and the first and
the second are the first two rows.

| Control | What it is | Must |
|---|---|---|
| `code_join_one_entry_binds.fl` | A join entered from the handlers of two arms; one binds the code; the join allocates a closure of it | be refused |
| `code_join_bound_at_head_of_body.fl` | The code is bound at the head of the body of the join's `Let_cont`, before the test; the join allocates a closure of it | pass |
| `code_loop_one_entry_binds.fl` | A recursive handler with two entries from outside; one binds the code; the loop allocates a closure of it | be refused |
| `code_loop_bound_at_head_of_body.fl` | The code is bound at the head of the body of the recursive `Let_cont`, before the test | pass |
| `code_join_one_entry_binds_called_directly.fl` | As the first, and the mention is a direct call of the code | be refused |
| `code_join_bound_at_head_of_body_called_directly.fl` | As the second, and the mention is a direct call of the code | pass |
| `code_in_code_one_entry_binds.fl` | As the first, and the mention is in the body of code that the join binds | be refused |
| `code_join_other_name_on_one_way.fl` | One code is bound on both ways, another on one way; the join mentions both | be refused |
| `code_join_other_name_on_one_way_not_mentioned.fl` | As the one before, and the join mentions only the code that both ways bind | pass |
| `code_through_inner_join_one_entry_binds.fl` | The join is entered through an inner handler that has two entries, one of which binds the code | be refused |
| `code_through_inner_join_handler_binds.fl` | The inner handler, through which every way goes, binds the code | pass |
| `code_bound_after_call.fl` | The code is bound after a call that raises; the exception handler of the call mentions it | be refused |
| `code_bound_before_call.fl` | The code is bound before the call | pass |
| `code_handler_never_entered.fl` | A handler that nothing applies mentions code that one arm of a live body binds | pass |
| `symbol_join_bound_at_head_of_body.fl` | The symbol is bound at the head of the body of the join's `Let_cont` | pass |
| `symbol_loop_bound_at_head_of_body.fl` | The symbol is bound at the head of the body of the recursive `Let_cont` | pass |
| `symbol_join_one_entry_binds_run_binds.fl` | One of two entries binds; the run takes the entry that binds | be refused |
| `symbol_join_one_entry_binds_in_static_field.fl` | As the one before; the mention is a field of a static block | be refused |
| `symbol_join_one_entry_binds_as_argument.fl` | As the one before; the mention is an argument given to a continuation | be refused |
| `symbol_join_other_name_on_one_way.fl` | `$S` is bound on both ways, `$T` on one; the join reads both | be refused |
| `symbol_join_other_name_on_one_way_not_mentioned.fl` | As the one before, and the join reads `$S` alone | pass |
| `symbol_through_inner_join_one_entry_binds.fl` | The join is entered through an inner handler that has two entries, one of which binds | be refused |
| `symbol_through_inner_join_handler_binds.fl` | The inner handler, through which every way goes, binds | pass |
| `symbol_switch_arm_never_taken.fl` | The arm that does not bind is never taken (the scrutinee is a constant): it counts all the same | be refused |
| `symbol_bound_before_loop_used_after_loop.fl` | Bound on the one way into a loop; read in the handler that the loop leaves to | pass |
| `symbol_loop_one_entry_binds_used_after_loop.fl` | A loop with two entries, one of which binds; read in the handler that the loop leaves to | be refused |
| `symbol_raise_before_binding_on_one_way.fl` | One arm raises to the handler with nothing bound, the other binds and leaves; the handler reads the symbol | be refused |
| `symbol_pushed_handler_never_applied.fl` | A handler of exceptions that is pushed, popped and never raised to reads a symbol of the body | pass |
| `symbol_handler_never_entered.fl` | A handler that nothing applies reads a symbol that one arm of a live body binds | pass |

The directory two levels up has eight more, all for a symbol, which
the tool is given alone: `statics_scope_used_on_the_other_path.fl`,
`statics_scope_bound_in_other_arm.fl`,
`statics_scope_used_before_bound.fl`,
`statics_scope_loop_one_entry_binds.fl` and
`statics_scope_bound_after_call.fl` are refused;
`statics_scope_handler_after_both_arms.fl`,
`statics_scope_loop_bound_before.fl` and
`statics_scope_bound_before_call.fl` pass. No text here repeats one of
them: where one is near, its comment says what differs.

## What the texts tell apart

- From a rule that gives a handler all that the body binds: every text
  that must be refused. In each the name is bound somewhere in the
  body of the `Let_cont` and not on every way to the handler.
- From a rule that gives a handler only what is in scope around its
  `Let_cont`: the texts that pass with the binding inside the body
  (`*_bound_at_head_of_body*.fl`,
  `*_through_inner_join_handler_binds.fl`,
  `*_join_other_name_on_one_way_not_mentioned.fl`,
  `code_bound_before_call.fl`,
  `symbol_bound_before_loop_used_after_loop.fl`).
- From a rule under which a handler has none of the names once one way
  lacks one: `*_join_other_name_on_one_way_not_mentioned.fl`, which
  pass. They have the bindings of `*_join_other_name_on_one_way.fl`,
  which are refused for the name that one way lacks, and they mention
  only the name that every way binds.
- From a rule that counts only the way that the run takes: the refused
  texts whose run takes the way that binds, and
  `symbol_switch_arm_never_taken.fl`, whose other arm no run can
  take.
- The kinds of mention: a closure allocated, a direct call, a mention
  in the body of code, a load, a field of a static block, an argument
  of a continuation.
- The ways in: an arm of a test, an inner handler, the entry of a
  loop, the exit of a loop, the exception continuation of a call, a
  raise. A push of a trap is not a way in
  (`symbol_pushed_handler_never_applied.fl`).
- A handler that is never entered, under a body that is live
  (`*_handler_never_entered.fl`).

## What they do not test

- They are read by the reader of texts. It gives each name of a
  continuation a continuation of its own, so no text has a
  continuation bound again under the same number, and it refuses a
  recursive group of two handlers (`andwhere`), so no text has one.
- They test the check as the tool asks it of a term. The tool also
  uses the scope of a call for the list of code ids of a call of known
  arity; the controls of that are terms of the model made by hand, in
  `../../known_arity_lists/`.
- The check is a lower bound read off the text. That it refuses a term
  does not show that a run of the term reads a name that is not bound:
  in `symbol_join_one_entry_binds_run_binds.fl`, for one, the run takes
  the way that binds.
