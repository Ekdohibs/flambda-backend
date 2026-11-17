# C0: one definition of a well-formed term. A proposal

Owner: C0. For the human. Nothing is restructured by this note: it
says what the definition would be, what it would come with, in what
order, and what is asked of C1, C2, C3 and N1. Revised after R2's
review of it; R2 checked the table against the compiler's files and
ran two probes, and its findings are cited by number.

The compiler has no single definition of a well-formed Flambda 2 term.
This work is to end with one: `FlambdaWellFormedSpec.well_formed_unit`.
A condition on terms that a stream finds goes there and not into a
premise of that stream's statements; and a term that the compiler
cannot build, or fails on, and that the model accepts, is first a hole
in that definition.

## 1. Two layers, kept apart

- **Well-formed** (`well_formed_unit`): meant to be true of every unit
  the compiler builds, GADTs and layout `any` included, in the
  fragment the model has. That is the aim and not a result: row 1 is
  stricter than the compiler today, a unit that mentions another unit
  is left out, and no corpus has been through it. A statement may take
  it as a premise. A unit made by the compiler that fails it is a
  finding: the condition is too strict, or the compiler is wrong.
- **Agreement of kinds.** Real units have mismatches of kinds, so no
  statement about the compiler's terms takes agreement of kinds as a
  premise; a mismatch is undefined where it is executed, and that is
  the machine's business. C2's check (`well_kinded_unit`, in its own
  file) asks for the agreements that the compiler tests by a check
  that is always on. It is not a condition of well-formedness, and C2
  does not propose it as one: such a check stops the compiler only if
  Simplify visits the place, and R2 has a unit that the compiler
  compiles and the check refuses (a mismatch in an arm that Simplify
  removes first; R2-11). A corpus could refute "every unit passes" and
  could not establish it. The check is a sufficient condition, proved,
  for the absence of certain faults, and a diagnostic that the tool
  prints. It is never a gate.

The test for which layer a condition belongs to: can a unit that the
compiler builds break it, in a branch that never runs? If so it is not
well-formedness.

The compiler is taken throughout as it runs with no option that adds
checks of kinds: the only such checks are the ones that are always on.

## 2. The conditions of well-formedness

"Where the compiler establishes it" is one of:

- a **type** that cannot express the bad term;
- a **creation function** that always checks, so that no term has it,
  visited or not;
- a check made **when visited**: a pass stops when it reaches or opens
  the construct. It says nothing of code that no pass visits;
- an **invariant check**, which runs only with `-flambda2-invariants`;
- **nothing**.

Where a pass stops only by accident, on something else than the
condition, the row says so.

Only the first two establish a condition of every term. A row with
none of them is an unchecked assumption of the compiler, and is
recorded as a discrepancy (`rocq/HACKING.md`, "Discrepancies with the
compiler").

| | Condition | Owner | Where the compiler establishes it |
|-|-----------|-------|-----------------------------------|
| 1 | Static bindings are at toplevel. Since decision D60: outside the handlers of recursive continuations and outside the body of code (`statics_outside_loops_and_code`), which is what `well_formed_unit` asks; Simplify's rule of toplevel stays as `statics_at_unit_toplevel` and is asked of no unit | C0; there today | When visited: `simplify_named.ml`, the case `Static_consts`, for a group that binds a symbol. A group of code only may be anywhere; the model refuses it (C0-3). What toplevel is has no one statement (`Downwards_env.at_unit_toplevel`, set and cleared in `simplify_let_cont_expr.ml`) |
| 2 | A symbol or code id is bound once in the unit | C0; there today | No check of the whole unit found (C0-4). Within one group: invariant check (`Bound_static.create`). Within the handler being traversed, when visited: `Flow_acc.record_symbol_binding` and `record_code_id_binding` (`flow_acc.ml`, 179 to 208). `Typing_env.add_symbol_definition` has a note that redefinition is not checked |
| 3 | Every variable and continuation used is in scope; the body of code under its own binders alone (row 6) | C0, by N1's free names; in C2's checker today | When visited: `Typing_env` ("Variable ... not bound"), `Upwards_env.find_continuation`; and the end of `simplify.ml` for the unit. No creation function |
| 4 | Every symbol mentioned is bound (the human, D8), and every code id mentioned is bound or declared deleted, by the rule of scope in section 3 | C0 | When visited: `Typing_env` ("Symbol ... not bound"), for a symbol of the current unit; one of another unit comes from its `.cmx`. Code ids: `To_cmm_env` |
| 5 | The names of one binder are distinct: parameters of a handler; those of code, with `my_closure`; continuations of a recursive group, which has at least one; the two continuations of code | C0, by N1's rules; C2's checker has the first two | Continuations of a group, distinct and not none: creation function (`Bound_continuations.create`; the model accepts and runs an empty group, R2-17). Parameters: invariant check only, and a weaker one: it compares whole parameters (variable, subkind and debug uid), so two parameters with one variable and different subkinds pass it (R2-16). The two continuations of code: nothing found |
| 6 | Code is closed: no free variable but its parameters and `my_closure`, no free continuation but its two, which are binders (the human, D31) | C0, by N1's free names; today a premise of `FlambdaApplySpec.inlinable` | Creation function (`Code0.create`), but of the free names its caller supplies, not of ones it computes (C0-5) |
| 7a | A closure is at a function slot of its set | C3 | In the compiler the closures of a set are bound by position, so the bad term there is a pattern with more or fewer names than the set has functions. Not a type: the pattern is a plain list (`Bound_pattern.Set_of_closures`). For a `Let` the lengths are not checked where it is built (`Let.create` checks the constructors only), and list operations fail later (C3-4, R2-9). For a static set the function slots are checked when the group is taken apart (7d) |
| 7b | The function slots of a set are distinct | C3 | No check where a set is built: they are an association list whose stated precondition is not enforced and whose invariant function has no caller (C3-5). When visited, by accident: Simplify stops on an assertion for a set made by a `Let` (`simplify_set_of_closures.ml`, 822) and in `Flow_acc` for a static one |
| 7c | The functions of a set have each their own code id (the human, D28) | C3 | No check where a set is built. When visited: Simplify gives each old code id one new one, and `Flow_acc.record_code_id_binding` stops on the second binding of it ("code_id has been bound twice"; R2-7, C3-3) |
| 7d | A group binds a static set whole, each symbol at its slot (D4, option (a)) | C3 | The set is bound by one pattern. That the slots agree, in order, is checked when a group is taken apart (`Static_const.match_against_bound_static_pattern`, `static_const.ml`, 779 to 798; called by Simplify, the printer and other passes), not when it is created: `Let.create` checks only that the pattern and what it binds are of the same variety |
| 7e | The sets that one group binds have no code id in common; sets of different groups may (the human, on R2-15) | C3 | No check where a group is built. When visited: Simplify makes one map from old to new code ids for all the sets of a group (`Simplify_set_of_closures_context.compute_old_to_new_code_ids_all_sets`), so two sets with one code id bind one new id twice and `Flow_acc.record_code_id_binding` stops (R2-15). A set made by a `Let` has a map of its own, which is why the rule stops at a group |
| 7f | The value slots of a set are distinct | C0, agreed with C3 | Type: a map (`Simple.t Value_slot.Map.t`, `set_of_closures.ml`). The fexpr parser accepts a slot written twice and keeps the second (C3-8) |
| 8 | Every number of a term is one that the compiler's can be: the tag of a block made, by `Make_block` or statically, and the tag a `Block_load` names, in `[0, 251)` (D23); a constant in the range of its kind (the human, D13); the field and the size of a `Block_load`, the discriminants of a `Switch` and the bytes of a string in the range of their type | C1 (`numbers_in_range`) | Type, each of them: `Tag.Scannable.t`, whose creation function refuses others; `Target_ocaml_int.t`; `Numeric_types.Int8.t` and `Int16.t`, made by functions that wrap or fail; `Int32.t`, `Int64.t`, `Targetint_32_64.t`. The fexpr parser wraps an `int8` or `int16` literal that is out of range (C1-11) |
| 9a | A handler has as many kinds as parameters | C2, with its first step | Type: a parameter carries its kind (`Bound_parameter`) |
| 9b | Code has as many parameters as its declared arity has components, unarized | C2, with its first step | Nothing checks that the arity in the metadata of code agrees with its parameters (C2-6); the producers make one from the other |
| 9c | A call has as many arguments as its own arity has components, unarized | C2, with its first step | Creation function (`Apply_expr.create`, `apply_expr.ml`, 207 to 232) |
| 10 | The arms of a `Switch` have distinct discriminants | C0 | Type: a map |

I read the compiler's files for rows 1 to 6, 7d, 9c and the tags; R2
read them for rows 2, 5, 7a to 7d and 9, and corrected 2, 5, 7a, 7d
and 9; C3 gave the functions for 7a to 7f.

That an `Apply_cont` has as many arguments as the continuation has
parameters is not a condition. The compiler checks it when the use is
visited (`Continuation_uses.add_use`), a term it accepts can break it
in an arm that never runs, and the machine makes it undefined where it
is executed.

Rows 7f and 10 are there because the model's value slots and arms are
lists, of which the first of two wins. They could instead be left out,
with a sentence that well-formedness admits more than the compiler on
purpose; I include them, since each is one line and the rule of this
note is that a term the compiler cannot build is not well-formed.

From the types (T1, T2): nothing is asked of terms so far. T1 uses
`run_extends_statics`, which is a statement of this file, and its own
conditions are on types (below).

Known, and not sayable in the model yet. The header would list them,
so that the gaps show:

- exception handlers: never recursive (`Let_cont.create_recursive`
  checks), their arity, and trap actions, which a comment in
  `flambda.ml` says are not checked. With C5;
- a symbol or code id of another unit; that a unit binds its module
  symbol. The model has one unit;
- a new version of code and the code it replaces (C0-1). With S4.

Not well-formedness, and kept out:

- agreement of kinds (section 1), and everything else that is
  undefined only when executed;
- what holds only after Simplify (no `Switch` on a constant, no
  over-application of a direct call);
- conditions on types and typing environments. I ask T1 to lay its
  predicate on raw type structures out in the same way in the types
  library, under a name that says "type"; the header here points at it;
- freshness of bound names. In the compiler a binder is a name
  abstraction and opening it gives fresh names; N1's pilot is how the
  model gets the same. Row 5 is what is left of it on raw terms.

## 3. The shape of the definition

All of it is in `FlambdaWellFormedSpec.v` (C0's): the header, which is
the table and the three lists above; each check, a boolean function of
the term that uses nothing of the machine; and their conjunction.

    Record unit_interface := {
      unit_return : continuation;
      unit_exn : continuation
    }.

    Definition well_formed_unit (u : unit_interface) (e : expr) : bool :=
      statics_at_toplevel u e          (* 1 *)
      && static_names_bound_once e     (* 2 *)
      && names_in_scope u e            (* 3 *)
      && statics_in_scope e            (* 4 *)
      && binders_distinct e            (* 5 *)
      && code_closed e                 (* 6 *)
      && sets_well_formed e            (* 7 *)
      && tags_in_range e               (* 8 *)
      && annotations_well_formed u e   (* 9 *)
      && switch_arms_distinct e.       (* 10 *)

The record stands for `Flambda_unit.t`: a later field (the kinds of
the value slots for C2, the module symbol) then changes no call.
Today the check takes the exception continuation alone.

A stream writes its check and I apply it at its merge, with its row of
the header; a reviewer reads the row against the compiler. What a
stream proves about its condition may stay in its own files.

Row 4 is not a lookup in what the unit declares somewhere, nor a
plain lexical rule. Its scope, which the header will state:

- in a group, the closure symbols of the group are in scope in every
  definition of the group; its other symbols are in scope in the
  definitions that follow them, and in the closures and the code of
  the group. That is the order in which Simplify processes a group:
  closure symbols declared first, the other constants in order, the
  closures last (`simplify_static_const.ml`, 318 to 343);
- after a group, all it binds is in scope in the body;
- a symbol bound at toplevel in the body of a `Let_cont` is also in
  scope in the handler, when the handler is at toplevel (row 1 has
  the definition): the handler runs after the body. Simplify makes
  such terms: a constant lifted while the body is simplified is known
  in the handler and placed in the body (`simplify_let_cont_expr.ml`,
  `consts_lifted_after_fork`). Since decision D60 the clause is: what
  the body binds on every way to the handler is in scope in it
  (`FlambdaStaticsInScopeSpec.handler_scope`, `recursive_scope`),
  whether or not the handler is at toplevel;
- the body of code sees what is in scope where the code is bound. If
  closure conversion turns out to bind code before a symbol its body
  mentions, this is too strict, and the rule for code becomes "bound
  before a closure of the code is made"; to be settled against
  `close_program` when the row is written;
- a code id declared deleted is in scope and is not code: a
  `Bind_deleted_code` binds nothing, and a call of a closure of it
  stays undefined when executed (section 4, the list).

C1 has made every primitive have no value on a symbol given to it that
is not bound, with a law that says so. The law is about the arguments
themselves; that no such symbol is inside a field, or returned, is
this row's to give.

## 4. The theorems

In `FlambdaWellFormedSpec.v`, proved in its `Proofs` file from what
each stream proves:

- `well_formed_state : state -> Prop`, a definition to be reviewed:
  what a run keeps. One part for each row that says something of a
  run. It is about everything a state holds and not only the
  expression left to run: the values of the variables, down the fields
  of blocks and the value slots of closures; the handlers and the
  environments that continuations keep; the static values and the
  code; and, at an exit, the arguments. (For rows 1 and 2 it exists,
  as `inv` in the `Proofs` file; it moves to the `Spec` file.)
- A unit that passes starts in a well-formed state, and **a step keeps
  a state well-formed**.
- **A well-formed state does none of the things the conditions are
  there to exclude**, each a test on a state:
  - it does not read a variable, continuation, symbol or code id that
    is not bound (rows 3, 4 and 6; this is the theorem asked for with
    D8). Row 6 is needed for it: a call enters the body of code with
    its parameters and `my_closure` and nothing of where the code was
    defined, so only closed code reads no unbound variable there;
  - it does not bind a static name twice, call code whose body binds
    one, or define a recursive handler that does (rows 1 and 2;
    today's `static_fault`);
  - it does not make a closure at a slot its set has not got, and
    every static set in it is bound whole (row 7; C3 has both proved);
  - it does not make a block with a tag out of range (row 8).
- So no state of the run of a well-formed unit does any of them:
  today's `well_formed_never_faults_statement`, widened.

Each condition also comes with a unit of the examples that passes and
a term that fails, kept as a statement that shows what then goes wrong
(C3's `partial_static_set_statement` is the pattern). Rows 5, 7f, 9
and 10 exclude nothing at run time; each names instead one proved
statement that is false without it.

Later, and I think worth having: every way a step can be undefined is
either one of the tests above or is on one list, written as a function
of the state (`Invalid`; a primitive with no value; a `Switch` with no
arm; a call of what is not a closure, of deleted code, of other code,
at other arities; a projection at the wrong slot). That list would be
the model's whole account of undefined behaviour in one definition,
where today it is spread over the machine, and it is where the `CR`s
on undefined cases would sit together. The human has said yes to it
(section 7).

## 5. Rules for the streams

- Where a condition is stricter than what the compiler checks (it
  ought to be an invariant, and the compiler checks nothing, or less,
  or only with its invariant checks on), a comment at its definition
  in `FlambdaWellFormedSpec.v` says so: that the model is stricter
  here, what the compiler does (the file and the function, or
  nothing), and the number of the discrepancy. This is the human's
  rule (D32). Today that is rows 1, 2, 5, 6, 7b, 7c, 7e and 9b. (Row
  8 is not one of them: C1 found each of its ranges to hold in the
  compiler by a type, and its comment says which.) It is a plain
  comment and not a `CR`: a `CR` marks something undefined that may be
  wrong. It never applies to kinds, where the model is not stricter
  than the compiler.
- No `Spec` file keeps a condition on terms as its own premise if the
  compiler's terms always satisfy it. It is proposed as a row, with
  the place in the compiler, or "nothing".
- Agreement of kinds is never such a premise. C2's statements about
  well-kinded units take `well_kinded_unit` as their own premise;
  nothing else does.
- Scope and free names are computed by one function, N1's. There are
  four today (`expr_in_scope`, `free_continuations`, the relations of
  `FlambdaNominalSpec.v`, C2's check); rows 3 and 6 are not a fifth.
- The test tool says which condition failed. A unit that fails before
  Simplify is reported and counted, never passed over; one that fails
  after is an error, as now.
- A function on terms that models something Simplify does owes three
  things: it refines, it commutes with renaming (N1), and it keeps
  `well_formed_unit`. The third is asked of the S streams when they
  start, not of the validator's own normalisations now.
- `validate_unit_sound` keeps its statement: no premise is added.

## 6. Order, and what it costs

1. Done: rows 7a to 7e, with C3's epoch, as conjunct
   `sets_well_formed`.
2. Now, mine, comments only, for a reviewer: the comments that the
   rule of section 5 asks for at the conditions in the file today
   (rows 1, 2, 7b, 7c, 7e).
3. Mine, one reviewer, about a day: the header as the table; the
   record; the checks there today under the names above; rows 7f and
   10; `well_formed_state` and "a step keeps it" stated for them. No
   file moves and the tool changes by one line.
4. Mine, about two days: row 4 with its theorem. Symbols are not names
   that a binder renames, so it does not wait for N1. Then the
   classification of undefined behaviour, about two days.
5. With C1, a leaf: row 8, tags and constants. With C2's first step:
   row 9.
6. Rows 3, 5 and 6: N1 writes them on its one function for free
   names, with the invariant of a state that goes with rows 3 and 6,
   and I apply them. They come after C3's epoch and do not wait for
   C2.

## 7. Answered by the human

The three questions of this section were put to the human, who said
yes to each (D32): "I think the model can be stricter than the
compiler in terms of various things, yes. The compiler probably
doesn't thoroughly check some things that ought to be invariants."

- **Row 1 stays stricter than the compiler** (C0-3): a group that
  binds only code is refused away from toplevel, though the compiler
  lets it through.
- **Constants in range** (D13) are part of row 8, with the tags. C1
  has written the check (`numbers_in_range`), and it is a condition of
  `well_formed_unit`.
- **The classification of undefined behaviour** in one definition is
  wanted. It comes after C3's epoch and after row 4, since the list is
  what is left when the conditions are in.

Still open, and C2's to put: what an executed mismatch of kinds does
where the machine has no test today (a binder, an argument of an
`Apply_cont`, the scrutinee of a `Switch`, a field of a static block,
a value slot). It is in C2's request, with a `CR` in the `Spec` file.
No static condition is proposed for it.
