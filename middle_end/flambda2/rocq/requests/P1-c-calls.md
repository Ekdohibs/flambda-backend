# P1: calls of C functions in the model. A request

Owner: P1. For the human, and for C0 and C3, whose files the change
would be in. No call of a C function has a meaning in the model. With
the loads from strings in, the next thing a plain program on strings
meets is not a primitive: it is a call of a C function. This says
which calls the front end makes, what Simplify does with them, two
ways to give them a meaning, what is written so far (a table of
functions that nothing uses, `FlambdaExtcallSpec.v`), and how it
would be wired.

## What there is

An `Apply` whose call kind is `C_call` is refused by the conversion
(the unit is "not checked"). A unit with one that `-drawfexpr` prints
does not even parse back (`requests/V2-missing-constructs.md`, items
10 and 4).

## The calls that the front end makes of plain OCaml

Tried on the compiler built from this tree, each from a few lines of
source with no library:

| Source | Call | Allocates |
|--------|------|-----------|
| `match s with "a" -> ... \| "bc" -> ...` on a string | `caml_string_notequal`, and `caml_string_compare` for a match with many cases | no |
| `=` and `<>` at type `string` | `caml_string_equal`, `caml_string_notequal` | no |
| `<`, `<=`, `>`, `>=`, `compare` at type `string` | `caml_string_lessthan` and the like, `caml_string_compare` | no |
| `=`, `<`, `compare` and the like at a type that is not known to be `int` | `caml_equal`, `caml_lessthan`, `caml_compare` and the like | may raise |
| `exception E` | `caml_fresh_oo_id` | no; has an effect |

With a library there are many more (output, `caml_array_make`,
conversions between numbers and strings). The first three rows are
pure functions of two immutable strings.

By count, over twenty-two ordinary units on strings and integers
written for this (equality, order and `match` on strings; `mem`,
`assoc` and a sort on lists of strings and of integers; `max`, `min`,
`abs`, `compare`; equality of an option, a pair and a list; tests on
characters; an exception), each with the declarations of the standard
library that it uses and the functions written out: eighteen meet a
call of a C function and four do not.

| Function | Calls | Units | On what |
|----------|-------|-------|---------|
| `caml_string_notequal` | 13 | 3 | strings (`<>`, and each case of a `match`) |
| `caml_equal` | 5 | 5 | integers in one unit (a polymorphic `equal`); an option, two lists of integers, a list of strings in the others |
| `caml_compare` | 4 | 4 | integers in one unit (`mem`); strings in two (`mem`, `assoc`); a pair in one |
| `caml_string_compare` | 3 | 3 | strings (`compare`, a `match` with many cases) |
| `caml_string_equal`, `caml_string_lessthan`, `caml_string_lessequal`, `caml_string_greaterequal` | 1 each | 1 each | strings |
| `caml_greaterequal`, `caml_lessequal` | 1 each | 1 | integers (a polymorphic `max` and `min`) |
| `caml_fresh_oo_id` | 1 | 1 | nothing: it returns a new number, kept in the block of an exception that the unit declares |

`compare` and the six relations at the type `int`, `char` or `bool`
are primitives and not calls; `caml_int_compare` is never called. So
of the eighteen units, seven call only comparisons of strings; in
five the polymorphic comparison is given two tagged immediates or two
strings; in five it is given blocks (an option, a pair, lists), which
the table below has not got; one needs `caml_fresh_oo_id`.

## What Simplify does with a C call

`simplify/simplify_extcall.ml`. It knows eight names. For
`caml_compare`, `caml_equal`, `caml_notequal`, `caml_lessequal`,
`caml_lessthan`, `caml_greaterequal` and `caml_greaterthan`: where
the types prove that both arguments are tagged immediates, the call
becomes `Int_comp` at the tagged immediates (`Phys_equal` for the two
equalities) and a `Tag_immediate`; where they prove that both are
boxed numbers of one kind, two `Unbox_number`, the comparison at that
kind and a `Tag_immediate`. For `caml_array_make` it gives the result
a type. Every other call is left as it is, with a result of unknown
type. So a model in which these seven have no meaning cannot check
those rewrites, and one in which they have a meaning has to agree
with the integer comparisons on immediates and on boxed integers.

## Two ways

- (A) A table of known functions. The machine has, for some names, a
  function from the values of the arguments to a result, and an
  `Apply` of such a name goes to the return continuation with it. The
  conversion still refuses a call of a name that is not in the table.
  First in the table: the seven comparisons of strings, which are
  total on two strings (`runtime/str.c`: the bytes are compared as
  `memcmp` does up to the shorter length, then the lengths; the
  results are -1, 0 and 1, or a boolean). Then the seven polymorphic
  comparisons on what Simplify specialises (two tagged immediates;
  two boxed integers of one kind; floats when the model has them),
  and, if wanted, on strings and on immutable blocks by structure.
  Where the real function raises, the model must raise too, to the
  exception continuation: on a closure it may raise (`compare` of a
  value with itself answers before it looks, `runtime/compare.c`),
  and a path of the source that raises must not become one on which
  anything goes. Where the table simply has no answer (values of
  different shapes, which the real function orders by the layout of
  the runtime), that is a domain the model has not got, and is kept
  apart from a raise: the check of the unit stops, as it does for an
  unknown name, and the call is not made undefined.
- (B) Calls as events. A run is observed by the calls it makes (the
  name and the arguments) as well as by how it ends, and the result
  of a call is given by the environment. This covers every call at
  once, and is what a statement about whole programs needs in the
  end. It makes the machine depend on an environment, it needs the
  heap to say what of the arguments is observed and what a call may
  change, and Simplify's seven rewrites still need (A) for those
  names.

## Recommendation

(A), in two steps, each small. First the seven comparisons of
strings: with them, a `match` on a string and the equality of two
strings are converted and run. Second the seven polymorphic
comparisons on immediates and boxed integers, so that the rewrites of
`simplify_extcall.ml` can be checked (they are a pass for the
validator, like the identities of arithmetic). A name that is not in
the table, and a call whose signature the table has not got (what
the `Apply` declares of its arguments and its result: their kinds,
unboxed arguments, `noalloc`, a call that does not return), stay
refused by the conversion. That is about what the call declares. It
is not a check of the kinds of the values that arrive: a value of
another kind than the call declares is undefined where the call is
executed, as for a primitive (D10), and no statement gets a static
premise on kinds from the table. `caml_fresh_oo_id` is not a
function of its argument, so it is not for the table: it has a
section of its own below. (B) is for later, with the whole-program
statement.

A limit that the strings bring with them: a string of the model is
immutable, and one made from bytes by an erased conversion is outside
it (X5-3). The comparisons of strings inherit that limit as the loads
do.

## What is written: the table, which nothing uses

`FlambdaExtcallSpec.v`, with its proofs in `FlambdaExtcallProofs.v`:
fourteen functions on the values of the model, `eval_extcall statics
f vs`, where `f` is `Caml_string_comparison c` or `Caml_comparison c`
and `c` one of `compare` and the six relations.
- The comparisons of strings have a value on two strings: the order
  of their bytes read without a sign, then of their lengths
  (`compare_bytes`), answered as -1, 0 or 1 or as a boolean.
- The polymorphic comparisons have a value on two tagged immediates
  (the order of the integers) and on two strings (the same as the
  comparisons of strings).
- Everywhere else there is no value. The header of the file says, for
  each, what the C function does there (it reads the memory of
  whatever it is given; it orders blocks by structure; it raises on a
  closure; it polls on a long comparison), and a CR at each
  definition says that "no value" is a decision.
- Its statements: the order of strings is a total order, and is
  equality of the bytes where it answers "equal"; on two strings the
  polymorphic comparison is the comparison of strings; on two tagged
  immediates it is the primitive that Simplify puts in its place,
  tagged; and the laws that every primitive obeys (the value depends
  on the static environment only through the symbols among the
  arguments, is kept when the environment grows, is of kind `Value`
  on two arguments of kind `Value`, is -1, 0 or 1, and is the same on
  arguments that the observations read alike).
- It agrees with the C functions themselves: `requests/P1-extcall.py`
  makes an OCaml program that calls each of the fourteen, by its name
  in the runtime, on every pair of twenty-one strings and of twelve
  integers, and from what the program prints a Rocq file that checks
  `eval_extcall` on every line.

The table is not extracted and no file but its own mentions it.

## How it would be wired

For C3, which owns calls, and the human. None of this is done, and
none of it is in the trust ledger.

1. The term. A third `call_kind`, a call of a C function, that
   carries the function as a constructor of the table (`extcall`) and
   not as a symbol. So a term of the model cannot call a function
   that the table has not got, and well-formedness needs no table of
   names. The compiler's callee is a symbol of no unit
   (`*extern*.caml_string_equal`), which a well-formed term may not
   mention (D8): the model's call has no callee. The conversion maps
   the name to the constructor and refuses every other name, as it
   refuses every C call today.
2. What the call declares. Its arity is that of the table: two
   arguments of kind `Value`, one result of kind `Value`
   (`eval_extcall_kinds_statement`). The conversion refuses a call
   that declares otherwise (unboxed arguments or results, a call that
   does not return). That the front end declares the comparisons of
   strings `noalloc` and the polymorphic ones not changes nothing of
   execution, and is not in the model's term.
3. The step, where the table has a value. The machine evaluates the
   arguments as for any call, and applies the return continuation of
   the call to the value of `eval_extcall`: the step of a `Let` of a
   pure primitive followed by an `Apply_cont`. The exception
   continuation is not taken, and no trap action is involved: on the
   domain of the table the C functions raise nothing.
   Where the table has no value, what the machine does is not
   designed. "No step", as for a primitive, would make the call
   undefined, and R2 does not accept that: the table is partial
   because it is unfinished, not because the C function is undefined
   there. It needs an outcome of its own, "outside the model", or a
   table that is total on what a well-formed unit can pass, before a
   call is given a step at all. This page is a sketch for C3 and is
   not approved.
4. The laws. Such a step keeps what the machine's invariants
   ask of a primitive, by the statements of the table that mirror
   those of `FlambdaPrimitiveSpec.v`: numbers stay in range, a value
   is kept when the static environment grows, no symbol bound to
   nothing is read, and lifting a value to a symbol does not change
   the answer.
5. The heap. The functions of the table have no effect and no
   coeffect: they read immutable strings and constants. A string
   that an erased conversion made from bytes is outside the model
   (X5-3), here as for the loads.
6. Simplify. It leaves a comparison of strings as it is, so the
   validator compares the two calls as it compares two `Apply`. It
   replaces a polymorphic comparison of two tagged immediates by a
   primitive and a `Tag_immediate`; the validator would do the same
   to the term before where it can see that both arguments are tagged
   immediates, by `specialised_for_tagged_immediates_statement`, which
   says that the two have the same result. If a call came to be an
   event that a run shows, the rewrite would also need that such an
   event may be dropped.
   (Where Simplify knows it from a type that the validator cannot
   read off the term, the unit is not validated.) The same for two
   boxed integers needs the table to have them.

CR mshinwell: what a call on values that the table has not got should
be. The table has no result there; it does not say that the program
is undefined. If the machine had no step, a program that compares two
lists with `=` would be undefined in the model, and then any rewrite
of it would be accepted. The ways out: extend the table before any
pass is trusted on such a call (blocks by structure, boxed integers;
a raise, with the exception that `caml_compare` makes, where it
raises); or give a run an outcome "outside the model", which the test
tool already has for a partial application, so that the check stops
instead of accepting. R2 asks that a domain the model has not got be
kept apart from a raise and from undefined, and has not approved the
step above.

## The open choice: what a call shows of its arguments

Decision D50. Nothing of this section is built, and none of it is
agreed as a design. Every count, cost and gain in it is an estimate:
by a text search of the tree of c7fa0612cd, or counted by another
stream on units that are not in this tree. None is a verified
coverage.

The orchestrator asked for a plan of way (A) with only what the
table answers today, in which a call that the table does not answer
ends the run in a way of its own; the plan was held, by the two
reviewers who read it, on one question: what such an end shows of
the arguments of the call. This section sets the question out, with
the ways to answer it and what each would cost. The choice is the
human's.

### What is settled, and cheap

- The table (`FlambdaExtcallSpec.v`): fourteen comparisons at the
  signature of two values and a value. The seven on strings answer
  on two strings; the seven polymorphic ones answer on two strings
  or on two tagged immediates. Elsewhere the table has no answer.
- A test of the table against the functions of the runtime on
  generated arguments, and the repair of the reader of the text of
  terms, which does not read a C call as the compiler prints it
  (V2-1), are needed whatever is chosen. Neither is in this tree.
- The machine side, as directions that the reviewer of the changes
  to the core and the owner of calls found acceptable as scoped. It
  is not a design that has passed, and it is not built. A call kind
  for the fourteen functions. One form for such a call, a
  restriction of the records of the model with a review comment of
  its own: the conversion writes the constant 0 for the callee, and
  the machine requires the callee to evaluate to the tagged
  immediate 0; the arities are those of the signature. A step in
  five cases: a record that is not of that form, undefined; an
  argument of another kind, undefined (D10); an argument that is a
  symbol bound to nothing, undefined (D8), and poison too, with the
  qualification that what primitives are on poison is open; the
  table answers, and the call returns the answer; otherwise the run
  ends there, with the function, the arguments and the heap. That
  end is not undefined, not a raise and not divergence. A name or a
  signature outside the table stays refused by the conversion.
- What that end costs by itself: a fourth way for a step to end. By
  the text search, which is not a checked inventory: about 95 proof
  files take the result of a step apart and gain a branch; 16
  definitions and 12 statements gain a case; one statement
  (`step_values_all_statement`) gains a premise. It would be the
  widest change of constructors that the model has had.

### The choice

A program that compares two lists reaches a call that the table
does not answer, and its run ends there in the model. A statement
that relates the term before Simplify to the term after has to say
something of the two ends. The pair that shows the difficulty:

    before:  let static $s = Block 0 (0)  $t = Block 0 (0)
             in  caml_equal ($s, $t)
    after:   let static $s = Block 0 (0)  $t = Block 0 (1)
             in  caml_equal ($s, $t)

Both runs end at the call with the arguments `[$s; $t]` and the
same heap. As values the arguments are equal: the same two symbols.
What the symbols stand for differs, and the real function, which
reads the blocks, answers true before and false after. (The
reviewers' pair is `[$s; $s]`, with `$s` bound to `Block 0 (0)` and
then to `Block 0 (1)`: there the real answers are the same, and
what is lost is only what the argument stands for.) It is an
example on paper: no pass is known to make it. At an exit such a
pair is told apart by the comparisons that are under every
continuation environment (the refinement of terms, and the forms
with observers), because a continuation may read the block; the
comparison of two units each run on its own does not look at it.
Nothing follows a call that ends the run, so no continuation
environment reaches it.

Common to O1, O2 and O3 below: the cost of the fourth end, above.
For the gain there is one figure, counted by another stream on 199
units written for the purpose, taken without their `main` (not in
this tree): 51 reach the comparison of the two terms today, 60
would with the comparisons of strings, 68 with the polymorphic ones
too. So 17 units more would reach the comparison. That is not 17
verdicts (of the 51, 43 are validated at the default flags), and it
is not a bound for other programs. Of the 200 whole units, with
their `main`, none would: each also needs another unit when it
runs. A list of 2,689 other units has no C call in the terms that
are converted today; 168 of its outputs are of terms that are not
read, and a C call may be among those.

- **O1. Equal arguments, and everything reachable from them equal
  as text**, through the two static environments and heaps. It
  claims that the two runs stop at the same function on the same
  data, code included. A lifted value or a rewritten function among
  the arguments is a difference. Cost: a definition of "reachable
  and equal". With the way a lifting is proved today (a check that
  the term gives only symbols and constants where values are
  compared by equality), most units that give a variable to a C
  call would be refused; that is a limit of that way of proving,
  not of the option. Gain: a handful of the 17 is expected, unless
  that changes.
- **O1', a variant: the arguments read the same to every depth.** A
  value is read through its own static environment and heap: an
  immediate, a string by its bytes, a block by its tag and fields,
  a boxed number, an immutable array, and a function only as "a
  function". The comparisons do not read code. A lifted block reads
  as it did, so the comparisons that lift stay available, and the
  pair above is told apart. What it forgets: which function, and
  whether two arguments are the same allocation. The runtime can
  see that: `caml_compare` answers at once on two pointers that are
  the same, and raises on two different functions, so `compare f f`
  may answer 0 where `compare f g` raises. Cyclic and mutable data
  are not specified. So it is not established that this reading
  keeps what the comparisons of the runtime do. Cost, an estimate:
  one new `Spec` file (about 250 lines, proofs 400 to 600) and, for
  each relation that a proof takes through a call, one law of the
  reading. No theory about contexts. Gain: up to the 17.
- **O2. The arguments compared by observers**, as the values of an
  exit are: every term that could follow, applied to them. It tells
  apart two functions that behave differently, which O1' does not;
  against O1 it is not simply more, since equality of text is the
  stricter of the two. Cost: the comparison is defined through
  itself (an observer may reach such a call), so it needs a
  well-founded design; the relation `refines` of the core would
  depend on observers; and every proof that a pass refines would
  owe a fact about contexts for which the tree has no theorem
  (about 310 statements name a relation between two runs, by the
  text search; not all of them are about passes). Gain: as O1'.
- **O3. Nothing claimed of a run that reaches such a call**, and
  the tool says so on the line of the unit. Cost: the least of the
  three, as no law says what is kept at the end. What a wrong
  compiler could then get away with: anything, on every run that
  reaches such a call, including what the run does before the
  call. The statements are about whole runs, so a run of the term
  before that ends this way would license any term after, as an
  undefined run does. (That is what the statements would allow; it
  is not a claim that the validation as it is would accept every
  such pair.)
- **O4. Nothing in the machine until calls are events** (way (B):
  a run is observed by the calls that it makes, and the environment
  gives the results). What carries over: the table, its test, the
  repair of the reader, the form of the call in the term, the case
  in which the table answers. What does not: the fourth end and its
  sweep, unless the design of events keeps it as "the environment
  gives no answer". The question of this section does not go away:
  with events it is what a call shows of its arguments, and O1, O1'
  and O2 are three answers to it.

### Recommendation

O4 now; the two reviewers who read this section for its accuracy
recommend the same. The gain of the first step is at most 17 units
of the 199, none of the whole programs, and its price is the widest
sweep that the model has had, part of which events would pay again.
The table, its test and the repair of the reader go on. What a call
shows of its arguments is to be decided, by the human, before the
machine is given any call, whether calls get their meaning from the
table or from events. O1' is the candidate that this stream would
study first, with the limits said above; neither reviewer endorses
it in advance. What stops those programs is elsewhere as well: by
the same count, all 200 whole units use another unit when they run,
and 93 the heap.

Meanwhile no call of a C function has a meaning in the model, the
conversion refuses every one, and nothing uses the table.

## `caml_fresh_oo_id`, for an exception that a unit declares

Not in the table: it is not a function of its argument. This is for
C5, which is writing how exceptions are matched and what a symbol of
another unit is, and for the human. The wording is agreed with C5,
whose own request says only what the matching needs of the call and
leaves the form of the answer to this one.

Of `exception E` the front end makes a call `caml_fresh_oo_id (0)`
and a block of tag 248 that is `Immutable_unique`, whose first field
is the name, a string, and whose second is what the call returned.
That block is the exception, or the first field of the exception if
it has arguments, and a handler finds it by `Phys_equal`: by the
identity of the block, which is that of its allocation, and never by
the number. (For an exception with arguments the handler loads field
0 of what it caught and compares that; the field 1 that it then loads
is the argument, not the number: `lambda/matching.ml`,
`combine_extension_constructor`.)

The runtime (`runtime/obj.c`) returns the next of a counter as a
tagged immediate, without looking at its argument. The numbers are
taken in chunks of 1024 for each domain: two calls never answer alike
in a program, and consecutive calls answer consecutive numbers only
within a chunk. The number is read by the polymorphic comparison and
the hash of two exceptions or objects (`runtime/compare.c` and
`hash.c`, at `Object_tag`), and by nothing that a `match` or a
handler compiles to.

So what the model needs of the answer is little:
1. it is a value of kind `Value`, in range, so that the block that
   holds it is well formed;
2. the run before Simplify and the run after get the same answers
   where they make the calls in the same order, which they do:
   Simplify neither removes, copies nor reorders a call of a C
   function that has effects;
3. it is new. Only code that reads the number could see that, and
   the code that declares, raises and matches an exception does not
   read it. (The model has comparisons of integers, and a program can
   load the field and compare it; it is such a program, not the
   matching, that the form of the answer has to be right for.) The
   counter of the runtime is a machine word, so "never alike" is up
   to its wrapping round, which a model that says "new" leaves out
   and should say so.

Three ways to have that.
- A counter in the state of the machine, its first number a parameter
  of the run, as the symbols that a unit imports would be. Simple; but
  it promises consecutive numbers, which the runtime does not.
- An answer that the term never reads: a constant of a form of its
  own, the n-th fresh number of the run. It can be stored in a block,
  passed and returned. No primitive has a value on it: not the
  arithmetic, not a comparison of integers, and not `Phys_equal`
  (two of them are different numbers in the runtime, but one against
  a constant could be either). The outcomes of two runs compare it by
  n, and that is the only comparison there is. It says no more than
  the runtime does.
- One opaque value that every call returns, with no count, on which
  again no primitive has a value. It is enough to declare, raise and
  match. But two calls then return the same thing, so a term and the
  same term with two calls made one have the same runs, and that
  rewrite would be accepted.

The second is the one recommended while nothing reads the number; C5
agrees. The first two need the machine to count the calls, so they go
with the state of the machine and not with the table. The declaration
of an exception needs, besides, the block with an identity bound at
toplevel and `Phys_equal` on two such blocks, which are the heap's
(C4), with C5's case of a block of the unit against a predefined
exception, which is false.

Asked of the human: (A) or (B) first; whether polymorphic comparison
by structure is wanted now or only what Simplify specialises; for
`caml_fresh_oo_id`, a counter or an answer that the term never reads;
and, before the machine is given any call, what a call shows of its
arguments (Decision D50; "The open choice", above).

Read as a design by R2 (2026-11-12), whose points are taken above: a
comparison of closures may raise and does not always; an unknown name
or a call signature that the table has not got stays unconverted,
which is not a static check of kinds; a raise is kept apart from a
domain the model has not got; the limit of X5-3; what
`caml_fresh_oo_id` needs. The section on the open choice was read
for its accuracy by two reviewers (2026-11-13), and not as a design.
It is a request, and no meaning of a call is agreed.
