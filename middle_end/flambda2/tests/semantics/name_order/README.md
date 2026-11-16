# What Simplify makes of a unit, and the order of its names

Simplify is run three times on each example `X.fl` of the directory
above, and the three results are compared up to the names that they
bind. From one run to the next the unit is the same, and the
variables and the continuations of the unit have other numbers, which
can change the order in which the compiler holds them. A difference
between two results is evidence to look into: that Simplify may take
something from that order.

    dune build --root=. --workspace=duneconf/main.ws \
      @middle_end/flambda2/tests/semantics/name_order/name_order

The test has this alias of its own and is not part of `runtest`. It
took about half a minute in the measured runs, at two jobs.
`dune promote` writes `name_order.expected` from what the run printed.

A new example of the directory above needs nothing here if its line is
"the same": the expected file has the two lines of control, the lines
of the units that are not "the same", and one last line with the
number of those lines of each kind. The number of the units and of
those that say "the same" are in `name_order.counts`, beside the
output of the run in the build directory; that file is compared with
nothing.

## How the names are given other numbers

A variable and a continuation each take the next number of a counter
when they are made, and their maps and sets (`Variable.Map`,
`Variable.Set`, and those of `Continuation`) hold them in the order of
their identifiers. The identifier is a hash of the unit and of that
number, or the next one that is free where another name has that hash
(`middle_end/flambda2/identifiers/int_ids.ml`, `continuation.ml`,
`middle_end/flambda2/algorithms/table_by_int_id.ml`).
Before it converts the text of the unit, the program of the test
(`../../tools/nameorder/flnameorder.ml`) makes K variables and K
continuations and throws them away. The names that the reader then
makes for the text have their numbers moved by K, which can change
their order in those maps and sets; it need not change the order of
every two names, nor the order in every map and set, of every unit.
The three runs are with K = 0, K = 1 and K = 1000 (`dune`). No file
of the compiler is changed for the test.

Simplify is given the unit that the reader made, as it is, as
`../../tools/fexprc.ml` gives it. The check of the directory above
does not: it refuses some units before Simplify, and changes what it
has read of the others (`as_read_from_text` of
`../../tools/flsemantics.ml`). So the unit that is simplified here
need not be the one that is simplified there, for the same text.

The first two lines of the expected file are a control, on a sample:
they check that K = 1 and K = 1000 change the order in which a set
holds eight variables, and eight continuations, made after the K that
are thrown away. If the order of a sample comes not to change, the
control no longer shows that the names are moved. That the order of
the samples changes does not show that an order changes in every
unit.

What is not moved:

- which of two names of the text was made first: the reader makes them
  in the same order whatever K is, and no other order of making is
  tried. Simplify sorts by the number, and not by the hash, where it
  adds parameters to a continuation for the variables that the handler
  uses under other names (`simplify/simplify_let_cont_expr.ml`,
  `extra_params_for_continuation_param_aliases`), and the environment
  of types falls back on it for two variables that have the same
  binding time (`types/env/typing_env.ml`). The names that Simplify
  makes afterwards are made in the order in which it works, which the
  order of the maps and sets may itself change;
- the symbols and the code of the text. A symbol is held in the order
  of a hash of its name, and a code ID in that of a hash of its name
  with a number of another counter. A symbol that Simplify lifts has,
  at the default options, the number of a variable in its name, so it
  is moved;
- function slots and value slots, which are held in the order in which
  they were made (`identifiers/slot.ml`);
- a map that keeps the order in which it was filled (`Lmap`,
  `algorithms/lmap.ml`).

## What compares the results

Each run of Simplify is a process of its own, and writes its result as
text. The result with K = 0 is compared with each of the two others,
each comparison in a process of its own, which reads the two texts and
gives the two units to the comparison of the compiler
(`Compare.flambda_units_and_differences`,
`middle_end/flambda2/compare/compare.ml`). The comparison pairs the
variables, the continuations, the symbols and the code that the two
units bind, so it does not see that a symbol or a code ID of the unit
has another number in its name; a symbol, and code, that neither unit
binds is to be the same in both. `run.sh` has the form of each line.

What is trusted: the comparison, and that the text of a unit says what
the unit is. The printer of texts, the reader and the comparison are
part of the test as Simplify is: what the text does not write, and
what the comparison does not read, is not compared. A result is first
compared with itself, read twice: where the comparison finds it to
differ from itself, the line says so and the unit is not compared.

## The lines that are not "the same"

A unit whose line is not "the same" has the line in the expected file,
so a new line, a line that changes and a line that goes fail the test
until the file is written again. The last line counts them by kind,
and says that each of the other units that the script was given was
run and says "the same": the script counts the units that it is given
and those that it runs, and says so if the two numbers are not the
same. Where what a line says is known and has a label, `labels` gives
the label of the unit, which is put at the end of its line; no line
has one now.

- "not simplified", eleven units: in each of the three runs Simplify
  stops with an exception. The script records that, and not what the
  exception says, which need not be the same in the three. Each is a
  unit that the check of the directory above does not give to
  Simplify, and says "not checked" of: a term that is not well-formed
  (`set_duplicate_slot`, `set_shared_code`,
  `set_static_group_one_function`, `set_static_group_shared_code`,
  `static_in_loop`), or a call of unknown arity that returns to a
  continuation whose parameters are not one value, where the text does
  not supply the arity of the result that is required (the six
  others).
- "not compared: the result with K = 0, read twice, differs: a code
  ID", three units: the text of the result names code that it does not
  bind, the reader makes a new code ID for such a name at each
  reading, and the comparison takes two code IDs that no unit binds
  for the same only if they are equal. In
  `kinds_call_unbound_written` it is the code of the direct call,
  which the unit does not bind either. In `closure_at_own_slot` and
  `closure_at_other_slot`, which are terms that Simplify made, it is
  the code that `newer_version_of` names: the unit binds it as deleted
  and Simplify, run again, drops the binding and keeps the mark. The
  texts of the three results are the same but for the numbers of the
  names and the places where the lines break; the test does not check
  that. The three lines have no label: the reader's rule, a new code
  ID that nothing binds for a name that the text does not bind, is
  the one that is recorded under X4-4 for code of another unit, and
  the code here is not of another unit.

## That the test can fail

No unit is known of which the test says "differs". That it would was
tried once by hand, with a change of Simplify that is not kept: with
the arguments of an integer addition put in the order of the
identifiers of two variables, of the 187 units, 24 that had said "the
same" come to say "differs: a variable" or "differs: a constant", for
one K or for both.

## What a difference would mean

A line "differs" says that the comparison tells the result with one K
from the result with K = 0, after each was printed and read. It is
evidence to look into for a dependence on the order of names; it does
not say where the dependence is, and the printer, the reader and the
comparison are to be looked at with Simplify. Nor does it establish
that the two units behave differently: the two may run alike, as the
two orders of the arguments of an addition do.

The test is over small units at the default options. The units that
were compared across the three numberings all say "the same". That
says nothing of the fourteen other units, of a unit that is not here,
or of another option.
