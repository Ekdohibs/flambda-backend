# The provers, asked in the environments of Simplify

`provers_of_compilations.ml`, its list of units
(`provers_of_compilations.units`) and its expected file. It is run by the
alias `runtest` of this directory, at the default flags and at `-O3`.

`provers_difftest.ml` asks the provers in environments that it makes from
generated equations. This tool takes the typing environments that Simplify
itself computes for a unit, and makes for each one an environment of the
model that is seen the same. It counts the environments for which that is
done, and each one for which it is not, with the reason. In the
environments for which it is done, it asks the provers that the model has
(`prover_table.ml`), and reification, about the names, and compares three
answers.

## It leans on the hook of decision D33

The environments are had through `Typing_env.For_testing.set_observer` and
`Typing_env.notify_test_observer`, which Simplify calls at the start of
each continuation handler that it simplifies and at the end of the unit.
The comment on them in `typing_env.mli` asks the human a question that is
still open: they were added for tests, "to be confirmed or reverted". If
they are reverted, this test goes with them, as `types_of_compilations.ml`
does. Nothing else of the compiler is changed or wanted for it.

## What a reported environment is

A snapshot that the observer keeps, gone through when Simplify has
returned. It is not a point where Simplify called a prover, and it is not
an environment in the middle of a handler, where Simplify asks most of its
questions. Simplify may report one continuation more than once, and
handlers that it then discards; at the end of a unit that does not return
it reports an environment that knows nothing. So the counts are of
reported environments, not of handlers that are kept, nor of points that
a run reaches.

## The replay

There is no conversion of environments. An environment of Simplify, E, is
replayed twice: into the model (M), and into a new environment of the
compiler (E'), which is a control. The steps are the same but for one
exception, in step 2.

1. The names. Those that the levels of E define, and its symbols; every
   name that the type of one of these mentions; every name that E has as
   an alias of one; and so on. A type may mention a name of another unit,
   which no level defines.
2. Each name is defined, in the order that `stable_compare_simples` gives
   in E (the symbols, then the variables in the order in which E bound
   them), with its kind
   and with the strongest name mode at which `mem` says that E has it
   (the three are normal, phantom and for types, in that order). For a
   variable bound outside the closure that Simplify is in, that is the
   mode for types, whatever mode it was bound with. The exception: in E'
   a symbol of another unit is not defined, as the compiler has such a
   symbol without a definition; M defines it like any symbol.
3. Each name that is not canonical is made an alias of its canonical
   element (`add_alias`).
4. Each name is given the type that `find` gives for it in E
   (`replace_equation`), but for a type that is plainly unknown, which the
   name has from its definition; for M, the type is taken to the model by
   the converter of this directory.

No equation is added and nothing is met, on either side. A first version
added one equation a name, and the gate below refused it on some hundreds
of environments, for two reasons that are of how an environment is made
and not of the model. A name keeps the alias type that it was given when
its target was canonical (`x : (= y)`, and later `y : (= 1)`); an equation
added afresh stores the canonical one. And a join records aliases beside
the types: two names are aliases and each keeps a type that is no alias;
equations made from the types have no such alias.

What a replay does not carry: the levels and their scopes, the code age
relation, the symbol projections, and what the compiler would read of
another unit. `provers.ml` and `reify.ml` read none of them from an
environment (symbol projections reach reification as an argument).

## The gate

A replay is used only where it is seen the same as E. `provers.ml` and
`reify.ml` read an environment by `find`, `mem`,
`get_canonical_simple_exn`, `get_canonical_simple_ignoring_name_mode`,
`get_alias_then_canonical_simple_exn`, `expand_head` and `machine_width`.
That list was made by reading the two files; it says nothing of other
clients of an environment.

- An environment whose `machine_width` is not the one this test is built
  for is a loss.
- Each of the other six is asked of every name, in E and in the replay:
  `find` with the kind of the name and without; `mem` at no name mode and
  at each of the three; the canonical element at each of the three; the
  canonical element whatever the mode; the canonical element of what the
  type of the name is an alias of, at each of the three; the expanded head
  of the alias of the name. And whether each environment is bottom.
- The answers are compared as values of the model: a type in the normal
  form of `model_type.ml`. A function of the model that has no answer is a
  difference; it is not the same thing as a canonical element that there
  is not. A fatal error where an environment is read is a difference.

The gate is finite. It asks of the names, not of every type that a prover
could be given, and not of constants. The first difference found is the
reason that is counted.

Three comparisons are made. M against E is the admission. E' against E
says whether the steps give E back in the compiler itself. E' against M
says, where they do not, whether the model went the way of the compiler.

## A control

The units of the tests may have no variable bound at the phantom mode, and
the mode of a variable that a closure puts out of scope matters only where
there is one. So the tool first makes an environment by hand
(`-control`): variables of the three modes, in the scope of a closure and
outside it, and a name in scope that is an alias of one outside. It prints
the mode at which the environment has each, which is to be the mode it was
bound with in scope and the mode for types outside, and the three
comparisons of the replays, which are to say "seen the same". Its output
is the head of the expected file. (A second control, of the statuses, is
described below.)

## What is lost, and what is counted apart

- A type that the model has not: an array type, or a simple with a
  coercion. The environment is lost, whichever name has the type.
- The model has one unit. A symbol of another unit is numbered and defined
  like any symbol, and goes through the gate; the compiler gives it, with
  no `.cmx` file, the type of any value that is not null. The admitted
  environments that have such a symbol among their names are counted, so
  that they can be told apart. A variable of another unit has not been
  seen.

## What an admission says and does not say

That M answers as E to the readers above, on the names above. It does not
say that M satisfies the premises of the model's theorems on environments,
and it does not say that E is right.

## The questions

In each environment that is admitted with both replays, every name is
asked about, with up to three inputs.

1. The type that Simplify has for the name in a term
   (`Typing_env.type_simple_in_term_exn` at the mode normal, which is what
   `Simplify_simple` gives a prover): the alias of the canonical simple.
   For a name that the environment has at the mode normal, and for which
   that function has an answer (where it raises `Not_found`, the name is
   not asked about with this input). This is the
   input that Simplify would supply for that name there. It is not a
   question that Simplify asked: an environment is reported at the start
   of a handler, and no prover is called there.
2. The alias of the name itself, whatever its mode. Types hold such
   aliases, and a prover is given them when it goes into a type.
3. The type that the environment has for the name (`find`): a head where
   the name is canonical, or an alias, which may be one stored before its
   target became an alias. Reification answers a simple for the alias of
   a name that is allowed, so it is on this input that it says what a
   type holds.

The type of the compiler is the compiler's own, in E and in E'; the
model's is made from it by the converter. A question belongs to its
environment: the same name with the same type is asked again in each
environment that has it.

The other arguments follow rules, the same for the three sides:

- the least name mode of a simple that a prover answers: normal for the
  first input, the mode for types for the other two;
- the kind (of `prove_equals_to_simple_of_kind`, and of the field of
  `meet_block_field_simple`): the kind of the name;
- the index of a field: the place of the name among the names, modulo 3;
- the function slot and the value slot: the first of each in the type,
  where its expanded head is a type of closures; otherwise a slot that no
  type has. The model takes the kind of the value slot as an argument: it
  is read off the slot (`Value_slot.kind`);
- the variable of `never_holds_locally_allocated_values`, which is asked
  of a variable and of no type: the name, where it is a variable; that
  prover is not asked about a symbol;
- the second type of `prove_physical_equality`: the type of the question
  before of the same input and the same kind, or the type itself.

Reification has three more arguments, and an environment does not have
what Simplify gives for them. They are a stated policy, the same for the
three sides, and not what Simplify gave where it reified:

- the variables that are allowed are those of the environment itself (E,
  E' or M);
- the projections from symbols are the variables that E says are
  (`Typing_env.find_symbol_projection`), given alike to the three;
- the variables at toplevel are all the variables; and reification is
  asked a second time with none.

## The three answers

Each question is answered by the compiler in E, by the compiler in E', and
by the model in M, and the answers are compared as values of one type
(`prover_answers.ml`).

- The three agree. A stop of the compiler that is caught and no answer of
  the model count as the same answer, as in `provers_difftest.ml`.
- The two answers of the compiler differ. The replay does not give the
  compiler's answer back, so there is nothing to compare the model with:
  a failure of this test.
- The two answers of the compiler agree and the model's is another. This
  is a candidate, to be looked into: the replay of the model, the
  converter of types and the conversion of the answers are on the way
  from E to the model's answer, as well as the prover.

An answer of the compiler that the model has no value for is counted as
not compared.

What agreement says: that on these questions the model's prover answers
as the compiler's, in an environment that is seen as Simplify's. It does
not say that the answer is right, and it does not say that the snapshots
satisfy the premises of the theorems proved of the model's provers.

## What is shown of a case, and how a run ends

A case is a triple of answers that is compared and on which the three do
not agree: one of the last two verdicts above. An answer of the compiler
that the model has no value for is counted as not compared, and is no
case. A case is shown, in every way of running the tool: the file, the
flags, the number of the environment among those that Simplify reported
and its continuation, the name, the input, both forms of the two types,
the arguments and the three answers. After the cases of an environment
come its names, in the order of the replay, each with what the
two replays are made from: its kind, its name mode, and its canonical
element and its type, as the model has them and as the compiler has them.

That is a record to read. It is not an environment that can be loaded: no
export of E, E' and M that runs without Simplify is written here. The
numbers that the model has for the slots and for the pieces of code are
not printed beside the compiler's (both forms of each type are). To have a
case again, the same text is simplified again at the same flags: the line
of a case names the file, without its directory (the list that the run
was given has the path), and says whether the flags are the default ones
or `-O3`, and the tool sets nothing else. This requires Simplify to
report the same environments in the same order at each run, as the
expected file of the alias has it for its units. Nothing makes a case
smaller.

That the tool does report a case where there is one is what the control
of the statuses, below, shows.

Where the check of a file returns, the run on that file ends with status
1 where a question is a candidate; else with 4 where the compiler answers
otherwise in its replay; else with 0. It is so with `-why` and with
`-why-with-names` too. A run on one file that stops before its check
returns, at an exception that nothing catches, ends as the system makes
it end. A file that has both
ends with 1, and is counted both among the units with a disagreement and
among those in which the compiler answers otherwise in its replay. A run
over files ends with 1 where a file has a candidate; else with 4 where a
file has a failure of the replay; else with 3 where the tool stopped on a
file; else with 0. Its last lines say each of the three that there is. A
run on a file whose status is not the one of what it counted, or that
counted no unit, is counted as a stop.

## A control of the statuses

No unit of the tests has a candidate or a failure of the replay, so the
alias would never go the ways above. The tool has two options for this
control and for nothing else: `-spoil-the-model PROVER` and
`-spoil-the-replay PROVER` replace each answer of a prover, in the model
or of the compiler in its replay, by another answer; a run with either
says so first. With `-control-of-the-statuses FILE.fl` the tool runs
itself on a file: with nothing spoiled, with a prover spoiled in the
model, with another spoiled in the compiler's replay, and with both, so
that the one file has a candidate and a failure of the replay. Each time
it is run on the file alone in the three ways, over the file as over a
list, and over the file and one that is in no directory, on which the
tool stops. It prints the status of each run and the two counts of units
of the run on the file alone, and fails if one is not the number written
in it; the counts of the cases of the two provers; how many
times the names of an environment are shown; and the last lines of the
runs over a list. Its output is in the expected file, after the first
control.

It is a control of the statuses, of the counts of units and of the names
being shown. It is not a control of the comparison of answers: the answer
that it puts in is one that differs.

## Figures

Of this directory's list (100 units, of which 95 are simplified), as the
expected file has them: 275 environments at the default flags and 285 at
`-O3`, all admitted, each with the compiler's replay seen as E too; 16,862
and 17,564 questions, and no disagreement.

By hand, on 17 November 2026, with the tool and the tree as they were
that day. These figures are not made again when the tree, the model or
the wording of the tool changes; those of the expected file are.

The environments:

| Units | Flags | Reported | Admitted | Arrays | Coercions | Gate |
|---|---|---|---|---|---|---|
| the tree's 740 files | default | 5,048 | 4,873 | 167 | 8 | 0 |
| the same | `-O3` | 5,371 | 5,188 | 175 | 8 | 0 |
| the testers' 2,689 texts | default | 39,248 | 37,213 | 1,995 | 40 | 0 |
| the same | `-O3` | 45,548 | 43,158 | 2,350 | 40 | 0 |

The tree's files are the `.fl` files that `tests/semantics` and
`tests/semantics/ocaml` had that day; the testers' texts are a list kept
outside the tree. "Reported" is the environments that Simplify reports;
"Arrays" and "Coercions" are those lost for a type that the model has
not; "Gate" is those that the gate refuses.

In each of the four, every admitted environment has the compiler's replay
seen as E too. Of the 740 files, 729 are simplified; of the 2,689 texts,
2,677 (12 are refused where the text is made a term). The other 11 files
of the tree are units on which Simplify stops, which are there to show
it.

The questions, in the admitted environments:

| Units | Flags | Questions | Answers compared | The three agree |
|---|---|---|---|---|
| the tree's 740 files | default | 231,730 | 14,550,471 | 14,550,471 |
| the same | `-O3` | 271,100 | 17,026,557 | 17,026,557 |
| the testers' 2,689 texts | default | 2,873,463 | 180,272,352 | 180,272,352 |
| the same | `-O3` | 3,734,030 | 234,254,457 | 234,254,457 |

An answer is of one of the 61 provers of the table or of reification,
which is asked twice; `never_holds_locally_allocated_values` is not asked
about a symbol. In the four runs there is no candidate and no question
on which the compiler answers otherwise in its replay; none of the
answers compared is of the class "no answer", which is a stop caught
where the compiler's prover or reification is asked; and there is no
answer that the model has no value for. The units on which Simplify
itself stops are counted above, among those that are not simplified.

Reification, on the testers' texts at the default flags: with every
variable at toplevel it lifts 148,351 times and cannot reify 482,114 times, of
2,873,463 questions; with none, 119,195 and 500,147.

## How to run

    dune build --root=. --workspace=duneconf/main.ws \
      @middle_end/flambda2/tests/semantics/types/runtest

    # by hand, from this directory in the build tree
    ./provers_of_compilations.exe [-O3] FILE.fl ...
    ./provers_of_compilations.exe [-O3] -why FILE.fl
    ./provers_of_compilations.exe [-O3] -why-with-names FILE.fl
    ./provers_of_compilations.exe -control
    ./provers_of_compilations.exe -control-of-the-statuses FILE.fl

`-why` shows each environment of a file that is not admitted with both
replays too (the first difference, with the name and the two answers).
`-why-with-names` adds the names of such an environment. `-control` and
`-control-of-the-statuses` run the controls alone.
