# The environments of Simplify, replayed into the Rocq model

`provers_of_compilations.ml`, its list of units
(`provers_of_compilations.units`) and its expected file. It is run by the
alias `runtest` of this directory, at the default flags and at `-O3`.

`provers_difftest.ml` asks the provers in environments that it makes from
generated equations. This tool takes the typing environments that Simplify
itself computes for a unit, and makes for each one an environment of the
model that is seen the same. Today it stops there: it counts the
environments for which that is done, and each one for which it is not,
with the reason. No prover is asked yet.

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
is the head of the expected file.

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

## Figures

Of this directory's list (100 units, of which 95 are simplified), as the
expected file has them: 275 environments at the default flags and 285 at
`-O3`, all admitted, each with the compiler's replay seen as E too.

By hand, on 16 November 2026, with the tool and the tree as they were
that day (these figures are not made again when the tree, the model or
the wording of the tool changes; those of the expected file are):

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

## How to run

    dune build --root=. --workspace=duneconf/main.ws \
      @middle_end/flambda2/tests/semantics/types/runtest

    # by hand, from this directory in the build tree
    ./provers_of_compilations.exe [-O3] FILE.fl ...
    ./provers_of_compilations.exe [-O3] -why FILE.fl
    ./provers_of_compilations.exe [-O3] -why-with-names FILE.fl

`-why` shows each environment of a file that is not admitted with both
replays: the first difference, with the name and the two answers.
`-why-with-names` adds the names of the environment, each with its name
mode, its canonical element and its type.
