# N3, a second verdict of the check of regions, about ends: status

Owner: N3. What this file describes is merged. The machine of regions
and the check of regions, beside which this stands, are their owner's
(`status/C2.md`). The statuses of the other work of the same stream
are `status/N3.md` and `status/N3-types.md`.

On 18 November 2026: the sections "Done" and "What is not said" are
as the tree is at this commit, and "The test tool" gives its counts
with their dates and the directories that they are of. The other
sections were written on 15 November 2026, with the work, and say
what was so then; they are not kept current.

The task: the check of regions does not say that a region is ended
(decision D48). A second test of the same term, with the same
certificate, that refuses a term in which a plain region is open
where code returns, or is forgotten at a jump unless the handler may
be entered stale (below); and a proof of what the run of a term that
passes it leaves on the stack.

Files (mine): `FlambdaRegionsEndsSpec.v`, which two reviewers read
for its meaning (below, "Agreed with others"), and eight files of
proofs, which need no review:
`FlambdaRegionsEndsBasicsProofs.v`, `FlambdaRegionsEndsStableProofs.v`,
`FlambdaRegionsEndsViewProofs.v`,
`FlambdaRegionsEndsHandlersProofs.v`,
`FlambdaRegionsEndsJumpProofs.v`, `FlambdaRegionsEndsLetProofs.v`,
`FlambdaRegionsEndsCallProofs.v` and `FlambdaRegionsEndsProofs.v`. In
the test tool, one block ("The ends of regions") and one statement in
`check_regions`, with the words of the owners of the tool and of that
function. No definition and no statement of another file changed.

## Done

Eight statements, all proved, each closed under the global context.

- **The check**, `ends_check cert opts ret exn statics e`. It follows
  the regions that are open, by their variables, as the check of
  regions does, and knows which are of a `try`. By the ordinary test
  of a jump to a handler, every open region that is not of a `try`
  is one of those that the handler is entered with: none is
  forgotten. A jump to a handler that is entered stale has a second
  test (below). Where the return continuation of code, or of the
  unit, is applied, no region is open but those of a `try`. A jump
  to a continuation that nothing binds, and that is neither of the
  two that the check is given, is refused. Of the certificate it
  reads one thing: the regions that each handler is entered with.
- **What is left open on purpose, and what is stale.** The end of a
  region pops every frame above the one that it ends, and two things
  leave regions open for the end of an older region to pop: a raise,
  whose handler ends the region of its `try`; and the jump that the
  front end makes where control leaves several regions at once, to
  the continuation that ends the outermost of them. So a handler may
  be entered stale (`enters_stale`): an exception handler is, and,
  without the option, so is a handler whose body ends a region
  before any other primitive of regions and before anything that is
  not a `Let` or a `Let_static` (`ends_first`). A jump to a handler
  that is entered stale also passes if the handler is entered with
  the oldest of the open regions, whatever is above them. Such a
  handler is checked in a state that is stale: frames that no open
  region accounts for may be above its newest open region. The end
  of an open region, where it makes a step, pops them, and the state
  of the check is not stale after it. A stale state may not begin a
  region, jump to a handler that is not entered stale, or return.
  That is strict, on purpose;
  decision D55, with a `CR` at `ends_check`, says what the choice is
  and what would make it wrong.
- **The option** `eo_own_end`. With it, the regions that the check
  has open above a region that is ended must be of a `try`, and only
  an exception handler is entered stale. It is a test of the regions
  that the check follows: what is proved of it is of one step, from
  a state that is not stale.
- **The property of states**, `state_ends`, beside
  `FlambdaRegionsCheckedSpec.state_checked` and not using it: a state
  of the check under which the expression passes; the open regions
  bound to marks that are on the stack; the continuations as their
  entries say; and the plain frames drawn since the running code was
  entered are those of its open regions, the same of each caller for
  the frames drawn between its entry and its call. It asks of the
  machine only that the frames are in the order of their marks.
- **The statements**, each under the premises that the file gives
  it. `ends_initial_statement`: the first state of a term that passes
  has the property. `ends_code_entry_statement`: so has the body of
  code that passes, in a state whose frames are ordered, whose static
  environment has only code that passes and which binds no
  continuation. `region_step_keeps_ends_statement`: a step keeps the
  property, and the frames drawn before what is run was entered.
  `ends_exit_statement`: an exit through the return continuation
  from a state that has the property leaves no plain frame drawn
  since. `ends_own_end_statement`: with the option, a step from a
  state that has the property, with a state of the check that is not
  stale, counts no plain frame cut by the end of another region.
  `ends_unit_exit_statement`: the run of a term that passes, from its
  first state, where it exits through its return continuation, leaves
  no plain frame. `ends_code_return_statement`: the body of code that
  passes, run on its own from such a state as that of
  `ends_code_entry_statement`, where it exits through its return
  continuation, leaves the plain frames that the stack had.
  `validated_and_ends_statement`: with the premises of
  `FlambdaRegionsCheckSoundSpec.validated_and_checked_exits_agree_statement`
  and this check of the term after too, an exit of the term before
  through the return continuation is one of the term after that
  leaves no plain frame; the verdict of the validation is one of
  those premises, and is not enough alone.

## How the proofs go

- `FlambdaRegionsEndsViewProofs.v` takes a step of the machine of
  regions apart, once (`region_step_viewed`): what a step is, said by
  what it leaves. No other file of these reads
  `FlambdaRegionsSpec.region_step`.
- `FlambdaRegionsEndsBasicsProofs.v` and
  `FlambdaRegionsEndsStableProofs.v` are about lists, marks and
  frames, and about what is said of a continuation being kept where
  the frames below are kept.
- `FlambdaRegionsEndsHandlersProofs.v` and
  `FlambdaRegionsEndsJumpProofs.v`: the state in which a handler is
  entered has the property, given the marks of the regions that it is
  entered with; and those marks, for a handler that is entered stale
  with the oldest of the open regions and for one that is entered by
  the ordinary test.
- `FlambdaRegionsEndsLetProofs.v`: the steps that are not a jump and
  not a call.
- `FlambdaRegionsEndsCallProofs.v`: a jump, a return and a raise, by
  one induction over the three relations of the continuations
  (`landings`), and a call.
- `FlambdaRegionsEndsProofs.v`: the eight theorems.

Two things the proofs had to find, neither of which touches a
statement. The machine of regions applies a handler that is given an
open region by a rule of its own
(`FlambdaRegionsSpec.region_apply_cont`), whose test of kinds counts
a mark of the stack as a region (`region_kind_of_value`) where that
of `FlambdaSemanticsSpec.apply_cont_value` counts it as a value; and
a tail call can return to a handler of a caller that way. So the
induction is about `apply_unkinded`, which is `apply_cont_value`
without its test: a function of these proofs, not a rule of the
machine. And an exception handler is applied to the exception
followed by the extra arguments of the call, where the check reads a
constant and then the extra arguments: the state in which a handler
is entered is asked of the values only where an argument is a
variable (`args_given`).

## The test tool

Where the tool prints the line of the check of regions with what the
two runs came to, it prints one more: of each of the two terms
converted with its regions kept, whether the extracted `ends_check`
passes it, and whether it passes it with the option. The model has
no function that makes a certificate, so the tool guesses the one
part of it that the check reads (`guess_ends_certificate`: for each
handler, the oldest regions that are open at every place which
applies it, by the names that the handler has for them, in a bounded
number of passes). The guess is not trusted: `ends_unit_exit_statement`
is of any certificate, and a poor guess can only refuse more. The
reason of a refusal is not printed. Each term has its own guess, and
nothing that is proved relates the two answers of a line.
`rocq/TRUST-LEDGER.md`, "The ends of regions", says what the line
rests on, what of the tool is trusted for it, and what it does not
say.

On 15 November 2026 the line was in 146 of the expected files of the
tree: the 26 examples of `tests/semantics` that have the line of the
check of regions with two runs, four files of
`tests/semantics/regions` (the three of counts, and
`region_param_loop.regions-only`), and 116 units made from OCaml
sources. On 18 November 2026 it was in 154, counted in these
directories and in no other: the same 26 of `tests/semantics` itself
and the same four of `tests/semantics/regions`, three examples of
`tests/semantics/guard`, and 121 units made from OCaml sources (109
of `tests/semantics/ocaml` itself, six of its `library` and six of
its `pairs`). A directory of tests that entered the tree after that
count is not in it. The rest of this paragraph is of those
directories on that day. Of the 121 units from OCaml, and of the
three examples of `guard`, both terms are
accepted, with and without the option. So are both terms of each of
the eight controls of the cells that the conversion of the tool makes
(`region_cell_*`), those three included whose run the check of
regions fails: what it finds there (a value used after its region is
gone, a local value held by an allocation on the heap, a store of a
value that does not live as long as its cell) is not a matter of
ends. Of the other examples, two are refused before and after
(`region_ended_twice`, `region_param_given_after_end`: a region that
is not open is ended, or given to a continuation), and two are
accepted before and refused after (`region_param_loop`,
`region_param_ended_by_handler`): Simplify deletes the end of a
region that is a parameter of a continuation, and the counts of
`tests/semantics/regions` show the region left by the run of the
term after. `region_inner_ended_by_outer`, in that directory, is the
control of the jump out of several regions: both of its terms are
accepted, both are refused with the option, and its counts show that
the run leaves nothing.

## What is not said

- Nothing of an exit by a raise: a raise leaves its regions open.
- Nothing of the frames of a `try`, nor of what they hold. The front
  end ends the region of a `try` in the handler only, so a `try`
  whose body returns leaves its frame on the stack.
- Nothing of a run that does not exit: a loop that passes may still
  allocate in one region for ever.
- Nothing that relates what the check answers of two terms: the
  check and `ends_unit_exit_statement` are of one term, and an
  answer is of the term that it was asked of.
  `validated_and_ends_statement` takes an exit of the term before to
  an exit of the term after, under its other premises; it asks this
  check of the term after only, and its conclusion is of the frames
  that the term after leaves.
- `ends_code_return_statement` is of the body of code run on its own.
  That the run of a term which calls the code is that run, until the
  code applies one of its two continuations, is not stated.
- That the frame of a region is popped by its own end is said, with
  the option, only of a step from a state that is not stale. Whether
  a state is stale is the check's word and not the machine's, so
  nothing is said with the option of a whole run.
- That a term is refused says nothing of its runs.

## Agreed with others

- With the owner of the machine of regions and of the check of
  regions: the check stands beside `region_check` and is not a mode
  of it; the property stands apart from `state_checked`. One lemma of
  these files takes a step of the machine apart, so that a change to
  the step is ported in `FlambdaRegionsEndsViewProofs.v` alone, by
  its owner's call. It was ported once, when the machine came to
  list the cells of local allocations and to run a `Let` on the
  heap: that file changed, and beside it only the cases of a result
  that the machine no longer has, and the last statement, which lost
  the alternative that its twin of the check of regions lost. In the
  tool, the guess fills only the regions that a handler is entered
  with and the check of regions is not asked; when the tool asks the
  check of regions, it is to compute one certificate for both
  checks, and the guess of this block goes.
- With the two reviewers who read the `Spec` file for its meaning
  before the proofs: the property has two bounds on the entry
  counters that it records (the entry of the running code is not
  after the counter; each saved caller was entered no earlier than
  the unit or the body of code whose run is considered), because the
  statement of a step assumes the property of a state and not a run
  that reached it. Both were found by reading; the header of the
  file says so.

## Found

- A first form of the check let only an exception handler be entered
  stale. It refused, whatever the certificate, the jump that the
  front end makes on purpose where control leaves several regions at
  once (`from_lambda/lambda_to_flambda.ml`, the comment in the case
  of `Lregion`), though the run of such a term leaves no plain
  frame. The owner of the units made from OCaml found it, by running
  the tool on the 840 generated units of the count of
  `tests/semantics/ocaml/census`: the first form refused the term
  before Simplify of 17 of them, at the default flags and with
  `-O3`. Simplify keeps the jump where the local value is used and
  the handler is applied from two places. With the check as it is,
  that jump is no longer a reason to refuse a term without the
  option, and is one with it. Of the control
  `tests/semantics/regions/region_inner_ended_by_outer.fl` the line
  says "accepted" of both terms, and "refused" of both with the
  option.
- With the option, the check refuses what Simplify makes where it
  splits an application with too many arguments
  (`simplify/simplify_common.ml`, `split_direct_over_application`):
  two regions are begun, the second by a begin that is not a ghost
  though it is ended by an end that is one, so the end of the first
  ends the second. Decision D48 names it already.

## Open

- The exact rule for a stale state, which would record where among
  the open regions the frames that were left are (decision D55).
- A function that gives the reason of a refusal, with the statement
  that it gives none exactly where the check passes, for the tool to
  print.
- The statement that ties `ends_code_return_statement` to a call
  inside a run.
