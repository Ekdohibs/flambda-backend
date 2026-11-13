# A count over generated units

The examples of the directory above are written by hand, and most are
written to validate. This is the other measure: units made at random,
with what the test tool says of each.

    sh census/run.sh <ocamlopt> <flsemantics.exe> 1 150 census-output

run from the directory above, makes for each seed from 1 to 150 one
unit of each profile (`gen.py`), makes its term (`from_ocaml.sh`), runs
the tool on it and prints the counts. Nothing here is run by the tests.

A unit that has the line of the second check under "not validated"
(`../README.md`, "What the verdicts mean") is counted as not
validated, and what that line says of it is counted apart, after the
counts. Measured with the tool of this tree, at the default flags
and with `-O3`: no unit of the 600 or of the 240 below has the line,
and none has the line of a verdict that was asked of the term before
with its known arrays read (the same section of that README).

## What a generated unit is

One to three inlined functions of two integers and a boolean, each
returning an integer, a boolean, an option, a pair, a variant or a
record; then one to three values that call each of them and compute
with the results. The expressions are arithmetic and comparisons on
integers, boolean operators, conditionals, constructors and matches on
the variant, on options and on pairs, record fields, `let`, and
closures applied at once. There is no recursion, no loop, nothing
mutable, no float, no string and no exception, but in the two
profiles of the last section.

The profiles: `known` (every input is a constant), `unknown` (about a
quarter of the integer constants are behind `opaque`), `functions`
(as `unknown`, and the unit keeps one or two functions that are not
inlined). With `_scalar` the values of the unit are integers and
booleans only; without it, about four in ten are an option, a pair or
a variant, that is, a block that the unit keeps.

## The counts of 2026-11-12

Seeds 1 to 100 for the first three profiles, 1 to 150 for the two
scalar ones: 600 units. No unit on which the two runs differ, none
undefined before Simplify, none on which the tool gave no answer, none
that the conversion refused.

| Profile | Units | Validated | Not validated |
|---------|-------|-----------|---------------|
| `known_scalar` | 150 | 150 | 0 |
| `unknown_scalar` | 150 | 150 | 0 |
| `known` | 100 | 100 | 0 |
| `unknown` | 100 | 100 | 0 |
| `functions` | 100 | 99 | 1 |

599 of the 600 are validated (593 before the liftings to a group
around, below; 592 on 2026-11-12). On the same 600
terms, earlier that day: 509 before the term after lost the
continuations that are applied once
(`FlambdaValidateSpec.inline_once_applied_after`); 555
with that, and still 555 with the unused `Let`s of the parameters of
code removed from the term after; 577 with the tool guessing again
what was lifted where its first guess gave a symbol to a block that
nothing uses, and with the lifting out of the code of functions; 578
with the second asking (`unknown`, seed 30); and four more with the
constants lifted to the head of the unit and the parameters of
continuations unboxed where branches meet (`functions`, seeds 1, 55
and 69; `unknown_scalar`, seed 122). Sharing the loads marked
immutable changes none, and nor does the demotion of exception
handlers, which the generated units have not got; with the rounds
gone through again after it, the two units that were validated at the
second asking are validated at the first. Two more with two arms of
a `Switch` that give one continuation 0 and 1 made the tagging of
what is tested (`FlambdaMergeSwitchesSpec.merge_switches`;
`functions`, seed 5, and `unknown_scalar`, seed 38), and four more
with the applications of a continuation of no parameters whose
handler is one jump forwarded in the term after (by
`FlambdaForwardJumpsSpec.forward_jumps` then, by
`FlambdaForwardKindedSpec.forward_kinded` since; `functions`, seeds
14, 27 and 72, the last at the second asking, and `unknown_scalar`,
seed 100). Sharing the projections of value slots changes none, and nor
does the unboxing of parameters giving, for an untagging under a
test, the number that was tagged. One more with that unboxing
giving a constant where a symbol of a constant block would be
untagged (`unknown_scalar`, seed 1). Two more with the rewrites
that Simplify makes from the types applied once to the term before
(`FlambdaSimplifyPassSpec.simplify_by_types`; `unknown_scalar`,
seeds 57 and 61). Reading the value slots that every closure of the
unit gives one symbol changes none. One more with the
continuations of each term placed once the rounds are over
(`FlambdaValidateSpec.placed`; `functions`, seed 65), and the unit
that was validated at the second asking is validated at the first.
The sharing of primitives knowing, under an untagging, that the
variable untagged is the tagging of the result changes none, and nor
does the test against zero of a three-way comparison made the
comparison of its operands
(`FlambdaRecoverComparisonSpec.recover_known_comparisons`), nor the
tidying of each term once its continuations are placed
(`FlambdaValidateSpec.tidied_before`), nor the static blocks in
scope known where a parameter is unboxed
(`FlambdaUnboxInScopeSpec.v`), nor the second asking following
aliases in the bodies of code. These 600 units import nothing, so
the tool prints of each what it printed. Nor does the removal of
unused static bindings from the head of the term after
(`FlambdaUnusedHeadStaticsSpec.v`), the liftings going into the
handlers of recursive continuations, the unboxing of a join going by
the box that each arm has, or the conversion of a call that never
returns: of each of the 600 the tool prints what it printed. One
more on 2026-11-13 (`unknown_scalar`, seed 144), with the arms of a
`Switch` merged in more ways, the number of the arm passed as it is
among them (`FlambdaMergeSwitchesSpec.merge_switches`, T1's), and the
`Switch`es of the term after threaded
(`FlambdaNormaliseSpec.thread_switches`, in
`FlambdaValidateSpec.inline_once_applied_after`). Of the other 599
the tool prints what it printed, with everything else that the
validation has gained since: a call returning where its result is
passed on, a continuation going down as far as the bindings under it
allow, the applications with known arguments forwarded in the term
after, the code of a stub left as Simplify leaves it, the fold of a
comparison of a value with itself, the load of a byte of a known
string left, the unused `Let`s removed after the value slots, and
the groups of static bindings put above the continuation bindings.
Six more later on 2026-11-13, with the liftings to a group of
static bindings around (`FlambdaLiftAroundSpec.lift_around_unit`,
V3's; `functions`, seeds 6 and 22; `known`, seeds 37 and 46;
`unknown`, seeds 2 and 72). Of the other 594 the tool prints what
it printed, with what else the validation and the tool have gained
since: the length of an array among the primitives that are not
computed twice, the modes of the parameters of code made unknown
before the tool's Simplify runs, the rewrites from the types in a
term in which something was lifted, a join given what a `Let` in
scope has read of its argument, a call of code that never returns
given the form that the term after has, the tables bound at the
head of the term before, an unused parameter removed from a
continuation that a call returns to, and the verdict asked of
`FlambdaValidateImportsSpec.holds_from`.
Between the tool of 87cdbf3e01 and the tool of this tree, of each of
the 600 the tool prints what it printed, byte for byte, with what
has come between: the load from a table in place of a `Switch`, the
test from the imports of the verdict that lifts, the calls of the
term before taken on when its rounds are over, the parameters of
loops untagged and given their scrutinee, what a handler reads in
two places made one parameter, the reads of a loop unboxed, an
`Opaque_identity` given the kind of its argument in a unit read
from a text, the refusal of a phantom binding and of a unit that
does not give
its return continuation its own symbol, and the rounds that stop.
No unit that was validated at one of these points is not at a later
one. The
day before, 336 were validated, 256 were not and 8 were not checked;
of the units that keep a function, none was.

What the 7 units that were not validated before those liftings
lacked, read off the two terms that the tool showed of each: a block
that Simplify makes a symbol and the term before still made with a
`Let`. The block is made in a handler and Simplify binds the symbol
outside it, or the guess of the tool did not find it (`functions`,
seeds 6, 18 and 22; `known`, seeds 37 and 46; `unknown`, seeds 2
and 72). In four of the 7 the term before kept a continuation that
the term after has not: a handler only makes a block and goes on
(Simplify makes the block a symbol and has no use for the handler).
Six of the 7 are validated now.

The unit that is left (`functions`, seed 18) makes three blocks of
one shape, each with one field that is a variable. The guess of the
tool gives a symbol of the term after to the first `Let` that makes
a block of its shape, and a field that is a variable agrees with
anything: the three `Let`s get the three symbols in another order
than Simplify gave them, and the two terms that are compared differ
only by which symbol each block has. Symbols are not bound names,
so the comparison does not match them. That is in the tool only:
nothing is believed of its guesses, and a better one needs no
proof.

Three of the 7 were validated already where the check was given the
pair that the compiler itself makes, converted with the change that
the measure makes to the conversion (`functions`, seed 6; `known`,
seeds 37 and 46; measured by X5 with the tool as it is at
ba85db2822, see `../README.md`, "What a validation does not say",
for the change and what it does not show). The tool's Simplify,
which reads the term from a text that has no debugging information,
shares the block that it lifts with an equal constant that the unit
binds already, where the compiler's Simplify makes a second symbol
(discrepancy X5-10). The validator had no such sharing. With the
lifting to a group around, a `Let` that makes what a symbol in
scope is bound to becomes that symbol, and the three are validated
of the tool's pair too.

The eighth unit (`unknown_scalar`, seed 144) had two things. Two
arms of a `Switch` gave one continuation two arguments, each the
number of the arm, where the term after has no such continuation;
and below it a `Switch` was on a variable that a `Switch` around it
had tested. It is validated since 2026-11-13 (above).

Five of the 600 have the constant `poison` after Simplify, which
unboxes a variant where two branches meet and gives the fields that a
branch has not got that constant. The conversion makes it the poison
of the model, so they are run and compared; all five are validated
(`functions`, seeds 27 and 55; `unknown_scalar`, seeds 1, 100 and
144).

So, for units of this kind: when the result is a number, all are
validated, whether the inputs are known or not; of the units that
keep a function, 99 of 100.

## Two more profiles: exceptions, loops

`exceptions` is `unknown` with handlers: `try ... with _ ->`, `raise
Not_found`, `raise_notrace`, and `/` and `mod`, which raise where the
divisor is zero, under a handler; an inlined function may raise, and
is then called under a handler. `loops` is `unknown` with `for` loops
(up and down) and `while` loops on references, local recursive
functions (whose recursive call is a tail call or not), and one
recursive function that the unit keeps and exports. With `_scalar`
the values of the unit are integers and booleans, as above. The
bounds of the loops, and the number of times that a recursive
function is first asked to go round by the unit itself, are chosen
from 0 to 6, and the generator nests loops twice at most. The
function that the unit exports takes any number.

    sh census/run.sh <ocamlopt> <flsemantics.exe> 1 60 census-output \
      exceptions exceptions_scalar loops loops_scalar

Seeds 1 to 60 of each, 240 units, on 2026-11-13. No unit on which the
two runs differ, none undefined before Simplify, none on which the
tool gave no answer. `run.sh` allows a unit 60 seconds of processor
time, user and system together. Until now it allowed 60 seconds of
the clock, which measured what else the machine was running:
`loops_scalar`, seed 55, which was without an answer in 60 seconds
the day before, then took 16 seconds of processor time, then 4,
then 7 with the tool of 87cdbf3e01, and takes 3.2 now, the most of the
240. The 240 took 17.8 seconds of processor time together in one run
(13.4 user, 4.4 system), and 18.3 in a second run of the same tool (13.9
and 4.4). Other builds ran on the machine during both: in two runs made
one after the other in the same conditions, the tool of ca84b4345c took
18.8 and the tool of this tree 17.9. The tool of this tree asks more of
each unit that is not validated: the second check, as the tool of
ca84b4345c did, and the validation of the term before with its known
arrays read; 10 of the 240 are not validated, where 25 were. When the
search of the second check came, in two runs made one after the other,
the tool of 93703d211b, which had the model and the tool of ca84b4345c
but for that search, took 14.2 (10.2 of user time) and the tool of
ca84b4345c 18.3 (14.0). When it was measured at that commit, the tool of
ca84b4345c took 16.7 and 16.7. They took 73 (69 and 4) with the tool of
87cdbf3e01, 51 (47 and 5) with the tool of ecb8c49042, and 204 the
day before that (198 and 6). Of the tools built between ecb8c49042
and 87cdbf3e01, one built before the rewrites from the types went
into a term in which something was lifted took 58 seconds, and one
built after them, and after the reads in scope given to a join,
took 81, in one run of each; two runs of one tool have differed by
a tenth on this machine. Between the tool of 87cdbf3e01 and this
one, a run of rounds of the validation was made to stop after the
first round that gives back its term
(`FlambdaValidateSpec.iterate`), where every run made all its fifty
rounds. That change was measured alone, on the tree where it was
made, with the tool before it and the tool after: 79 seconds and 12
for the 240, 30 and 16 for the 600, and no output of the 840
differs between the two. What else came between the two tools was
not measured for its cost on these units one change at a time. The 600
took 19.1 seconds in the first of those two runs (8.0 user, 11.1 system)
and 18.8 in the second (8.0 and 10.8), and 19.5 in the run made just
after that of the tool of ca84b4345c, which took 19.0; when it was
measured at that commit, the tool of ca84b4345c took 17.1 and 14.7. They
took 33 with the tool of 87cdbf3e01. More than half of what the 600 take
now is the system's part. A tool that
a signal other than that of the limit ends
is reported as that, with its status, and counted as no answer.

| Profile | Units | Validated | Validated with its imports | Validated with nothing imported only | Not validated | Not checked |
|---------|-------|-----------|----------------------------|--------------------------------------|---------------|-------------|
| `exceptions` | 60 | 33 | 27 | 0 | 0 | 0 |
| `exceptions_scalar` | 60 | 35 | 23 | 0 | 2 | 0 |
| `loops` | 60 | 56 | 0 | 0 | 2 | 2 |
| `loops_scalar` | 60 | 54 | 0 | 0 | 6 | 0 |

A unit that names a predefined exception has an import, and the
tool runs it from its imports (`../README.md`, "What the verdicts
mean"): 51 of the 120 units with exceptions. The validator accepts
50 of them, as it did before the tool ran units so, and all 50 are
validated with their imports. 36 are accepted with a verdict that
holds from the imports whatever they are. The other 14, all in the
profile that keeps a function, are accepted once closures are
lifted, a verdict that holds from the imports where it passes a
test (`FlambdaValidateImportsSpec.lifted_holds_from`, C5's;
`../README.md`), which the 14 pass. With the tool of 87cdbf3e01,
which had no such test, they were validated with nothing imported
only; of the other 226 units the tool of ea56783099 printed what
that one printed, byte for byte. One of the two units with
exceptions that are not validated has an import. The validator accepts
228 of the 240; 178 of them import nothing. Fifteen are validated since
it accepted 213, none lost. One (`loops`, seed 21) by the applications
of the two terms that go to the arm of the number they give, the
continuations that the handler binds on its way to its `Switch` bound
outside it first, and a parameter of a non-recursive continuation that
is no exception handler, which every application gives one constant,
bound at the head of its handler in the term after
(`FlambdaForwardArmSpec.forward_to_arms`,
`FlambdaLiftContsSpec.lift_handler_conts`,
`FlambdaBindConstParamsSpec.bind_const_params`; V4's, N2's and N3's,
wired together: `../README.md` has a unit of each). Fourteen by the
copies of code that a lifting of the validation may now ask for, and
that it makes where their check passes, for a static set of closures
that Simplify gives new code where the term before has one code
(`FlambdaCopyCodeSpec.copy_static_code`, C3's; `../README.md` has a
unit): `loops`, seeds 6, 9, 12, 20 and 28; `loops_scalar`, seeds 1, 11,
19, 22, 25, 35, 50, 57 and 60. They are the units of the 240 of which,
at the default flags, the tool of 722ada2bee, which has all of this but
the copies, prints another thing than the tool of this tree. Of the
other 225 units, and of each of the 600, the tool of this tree prints at
the default flags what the tool of ca84b4345c printed, byte for byte.
Six were validated when it went from 207 to 213,
none lost: three by the liftings to a group of static bindings
around (`FlambdaLiftAroundSpec.lift_around_unit`, V3's; `loops`,
seeds 7, 17 and 27), one by the removal of an unused parameter
of a continuation that a call returns to
(`FlambdaWrapReturnsSpec.wrap_returns`, N3's; `loops`, seed 34),
and two by the placing that takes a continuation into the handler
under it that alone names it (`FlambdaSinkUsesSpec.sink_to_use`,
V4's; `loops`, seed 40, and `loops_scalar`, seed 17). When
it accepted 211, with the first four of the six, the tool printed
of the two units that are not checked other names of variables in
the message that says what the model has not got, as the unit is
built again with fresh names where the modes of the parameters of
code are made unknown; of the other 234 it printed what it had
printed when it accepted 207. Between the tool of ea56783099, which
accepted 211, and the tool of ca84b4345c, the two units of V4's
placing are the only ones of the 240 of which the tool prints
anything else, and of each of the 600 it prints what it printed,
byte for byte. It accepted 195 the day before the 207, and twelve
units were validated between the two, none lost
(`exceptions_scalar`, seed 16;
`loops`, seeds 5, 15, 25, 35 and 37; `loops_scalar`, seeds 5, 9, 18,
24, 28 and 34). `middle_end/flambda2/rocq/status/V4.md` has the
measures, on these 240 units, of three rewrites that came that day:
seven units by a call that returns where the handler of its return
continuation passes its results on
(`FlambdaForwardSpec.forward_applications`), two by a continuation
going down as far as the bindings under it allow
(`FlambdaSinkUsesSpec.sink_conts_to_uses`: the way out of a loop
that only the handler of the loop names, and the continuation after
an inlined test), and one by the applications with known arguments
forwarded in the term after, the term before no longer guessing
what Simplify left (`FlambdaForwardKindedSpec.forward_kinded`). The
twelve were not attributed one by one here. `exceptions`, seed 50,
is validated at the first asking where it was at the second. Of the
other 227 units the tool prints what it printed. The day before
that it accepted 189, before the liftings went into the handlers
of recursive continuations (`FlambdaLiftFromCodeSpec.v`, V3's): six
units with loops, in which a local function that captures nothing
is made in a loop and is a symbol in the term after, are validated
since (`loops`, seeds 23, 29, 44 and 50; `loops_scalar`, seeds 15
and 38). Of the other 234 units the tool prints what it printed,
with the removal of unused static bindings from the head of the term
after, the unboxing of a join going by the box that each arm has,
and the conversion of a call that never returns as well.

The two that are not checked have a reference that a local function
captures: it stays a block with a field that is written
(`Block_set`).

Before each term that is compared had its continuations placed
(`FlambdaValidateSpec.placed`), 16 of the 120 units with loops were
validated, and in 57 of the 102 that were not the two terms had,
once the names are erased, the same lines in another order. The
common case is the return continuation of a call that does not use
its parameter, as in

    let rec f n acc =
      if n <= 0 then acc else (let _ = f (n - 1) acc in n)

The term before binds the continuation and then has the `Let` of
`n - 1`; the term after has the `Let` first. Simplify cannot remove
the parameter of a return continuation, so it makes a wrapper of
the continuation at the call, below the `Let`s of the arguments,
and the handler, which is then applied once, ends up there. The
placing moves the continuation of the term before down to the call.

What the 46 units with loops that were not validated lacked, before
the liftings went into the handlers of recursive continuations. One
unit of each kind was read; the counts are of the units whose two
terms, as the tool showed them, differed by the same kinds of line.
The six units that were validated next are of the first kind, and
the eleven others of the first kind were not read again. The third
kind is what a call returning where its result is passed on removes;
seven units are validated by it (above). Eight units with loops are
not validated now; they were not sorted again.

- A closure of no value slot that the term before makes by a `Let`
  and the term after has as a symbol (17 units). A local recursive
  function that captures nothing, in the code of a function that
  the unit keeps or in a handler: Simplify binds its closure with
  its code, and where the function around it is inlined twice the
  two `Let`s are one symbol.
- Code that the term after has twice or more (10 units). An inlined
  function with a local recursive function that captures one of its
  parameters, called twice. Simplify copies the code at each
  inlining, makes each closure a symbol and puts the constant of its
  value slot in its copy of the code; the term before has one piece
  of code and two closures.
- The return continuation of a call, whose handler only applies
  another continuation (7 units): the term before keeps it, and in
  the term after the call returns to the other.
- A closure with a value slot that is a variable, which Simplify
  makes a symbol in the handler where the variable is bound (4
  units).
- Eight others. One has a `Let` on the other side of the binding of
  a recursive continuation, one has static bindings in another
  place, and six differ by continuations or by a `Switch`, and were
  not sorted further.

Of the three units with exceptions that are not validated, one has
the same lines in another order: a `Let` is on the other side of the
binding of a continuation that the placing does not move. One has a
comparison in the term after whose result nothing uses, which the
term before has lost. One has, in the term after, a record that
Simplify made a symbol and a load of a field of that symbol, where
the validator has the field: the record is the result of a `try`
whose handler gives another record, and the handler turns out not to
be reached.
