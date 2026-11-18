# Pairs of units

A unit that uses another unit reaches it, before Simplify, by loading
a field of the module block of that unit. The compiler's Simplify
reads the `.cmx` of the other unit and uses what it says: it puts the
symbol of a closure in the place of the load, makes the call direct,
and inlines the body where the `.cmx` has it. The units of the
directory above use no other unit. Here the second unit of each pair
uses the first, and the test tool is given the directory that has the
`.cmx` of the first, so that its Simplify does what the compiler's
does.

These are the first two steps of
`middle_end/flambda2/rocq/requests/C5-imports-with-contents.md`
(section 6, items 1 and 2): the option of the tool and the pairs,
and the prefix of static bindings by which the model is told what
the `.cmx` of the first unit says. A verdict here may rest on that
file; see "What is claimed".

## The files

`F.ml` is a first unit and `F_more.ml` a second unit that uses it: a
unit is the second unit of a pair if its name is the name of another
unit of this directory followed by an underscore and more (the
longest such name, if several fit). As in the directory above,
`X.fl` is the term that the compiler makes of `X.ml` for Simplify
and `X.expected` is what the tool prints of that term; both are kept.
A unit that has no `X.expected` is compiled, for the unit that uses
it, and its term is compared with `X.fl`; the tool is not run on it.
There are no interface files: a unit exports everything.

Five units are no part of a pair: each is a unit that no other uses
and that uses no other. They are here for their first line, the
flags that the rules of this directory give to the compiler and to
the tool and that the rules of the directory above have no place
for, or as the control of such a unit, which has no such line
("Units that are here for their first line", below). Two more that
no unit uses are counted with the first units ("The pairs", below).
No unit may have the name of another followed by an underscore and
more unless it is its second unit.

## How a pair is made and checked

    sh from_ocaml.sh <ocamlopt> F.ml F
    sh from_ocaml.sh <ocamlopt> F_more.ml F_more F

Each is, in a directory of its own, to which `F.cmi` and `F.cmx` are
copied first for the second unit,

    ocamlopt -nopervasives -nostdlib -stop-after middle_end -c \
      X.ml -drawfexpr-to X.fl

for `X.fl`, and then the same command without `-drawfexpr-to X.fl`,
which writes `X.cmi` and `X.cmx` as the compiler writes them where it
prints nothing: the compilation that prints the text gives other
numbers to the names that it makes afterwards, and a symbol that
Simplify lifts would have another name in the `.cmx`. The three are
kept in the directory of the build. Then, in that directory,

    flsemantics.exe -I . X.fl

The dune rules of this directory do so for every unit, compare the
term made with `X.fl` and what the tool prints with `X.expected`, and
fail on a difference:

    dune build --root=. --workspace=duneconf/main.ws \
      @middle_end/flambda2/tests/semantics/ocaml/pairs/runtest

The alias of `tests/semantics/` runs these rules with the others.

- As in the directory above, a unit is compiled with no standard
  library and no initially opened module, and declares the
  primitives it uses.
- The compiler finds `F.cmi` and `F.cmx` in the directory it is run
  in. The tool's Simplify looks for a `.cmx` in the directories given
  with `-I` and in no other: neither the standard library nor the
  directory it is run in. The rules of the directory above give it
  none, so what the tool prints of a unit there does not depend on
  any `.cmx`.
- A source may give flags of its own, on a line `(* flags: ... *)`:
  the unit is then compiled with them after the others, and the tool
  is given them when it is run on the unit. A second unit is not
  given the flags of its first.
- The `.cmx` of a first unit is made by the build, with the compiler
  that builds the tool. What the tool prints of a second unit depends
  on that file, which is not kept: like `X.fl`, it moves with the
  compiler.

## The load path

The tool is given the directory of the build as its load path, which
has the `.cmx` of every unit of this directory. A second unit names
its first unit only, and the tool's Simplify reads that one file. A
unit whose first unit is itself a second unit
(`never_called_again.ml`) reaches the first unit of that one through
code that it inlines, and the tool's Simplify reads that file too.

A source may ask, on a line of its own,

    (* load path: the first unit alone *)

that the tool be given the `.cmx` of its first unit and of no other
unit: `tool_alone.sh` then runs the tool with a load path of its own
that has that one file. `never_called_unread.ml` is
`never_called_again.ml` with that line: see "The pairs".

Both sources have `(* flags: -w -58 *)`: the compiler is given the
`.cmx` of the first unit only ("How a pair is made and checked"),
and warns that it has not that of `never.ml`.

A file that is on the load path is not thereby read. The `.cmx` of
`maker.ml` is on the load path of `maker_kept_unread.ml` and of
`maker_kept_read.ml` alike. The tool's Simplify reads it for the
second, which keeps a function of `maker.ml` (and has the flag, for
the same reason as above), and not for the first, which reaches
nothing of that unit; and what makes the prefix opens no file that
the tool's Simplify did not read. See "The pairs".

## What is claimed

Where its Simplify read a `.cmx`, the tool tells the model what the
file says. It makes one prefix of static bindings and puts it in
front of the term before and of the term after; the two terms so made
are the ones that are run and validated. The prefix binds, of the
first unit (`make_prefix` in `../../../tools/flsemantics.ml` has the
rule):

- the blocks that the two terms ask for, and the data that the file
  reaches from them through fields: a block of values whose fields
  are constants and symbols, and, for a symbol of a closure, the
  closure at its function slot, alone in its set and with no value
  slot;
- the code that the two terms ask for, and the code that such code
  asks for in turn, where the file has its body.

The prefix declares deleted, and binds to nothing, the code that it
has from the file of the code's own unit and does not bind: code
that is asked for and has no body in the file, or a body that the
model has not got, and the code of a closure that is bound where
nothing asks for its code. The declaration says that the code id is
one of a unit that the program is linked with and that its body is
not given. A call of such code is undefined, as it was when the code
id was bound to nothing and not declared; what the declaration
changes is that the code id is a static name in scope (the guard,
below). Nothing is declared of a unit whose `.cmx` the tool's
Simplify did not read, whatever the `.cmx` of another unit says of
its code: a closure of a unit that was read can be of code of a unit
that was not (`maker_kept_unread.ml`), and is then bound, as it was,
with its code neither bound nor declared.

A unit whose prefix names such code passes the guard only where the
tool's Simplify reads the `.cmx` of the unit of that code. Where the
unit, or code that it inlines, reaches something of that unit, giving
the tool the file is what does it (`never_called_again.ml` against
`never_called_unread.ml`). Where nothing reaches it, the file is not
read though the tool is given it, and no option of the tool makes
its Simplify read one: `maker_kept_unread.ml` is "not checked" as it
stands, and `maker_kept_read.ml`, which keeps a function of
`maker.ml`, passes.

The declarations are in the first group of the prefix, and a group
is one step of a run whatever it binds. A prefix that binds
something has as many groups as it had, and the runs as many steps.
A prefix that declares code and binds nothing is one group where the
unit had no prefix at all: each run takes one step more to end as it
did, and the lines of the unit are those of a unit with a prefix.
The tool counts the steps of a unit with its prefix in front of it
against those that it allows, so a run that ended at the last step
allowed is then "still running". No unit here has such a prefix;
`../../prefix_fault/` has a control of it, on units made by hand.

The tool prints the prefix after the name of the term, to be read
against `F.ml`:

    given the .cmx of F, a prefix of N static bindings:

then the bindings, with the names that the compiler gives; what is
bound without being asked for is counted on one line. The runs are
then "after the prefix".

- **validated with its imports, given the .cmx of N of them**: the
  validator accepted the two terms with the prefix in front, and its
  theorem is of those two terms: with no import left, the theorem of
  a unit run on its own; with imports left, the theorem from the
  imports, for the verdicts that it has (`../README.md`, "What the
  verdicts mean"). It is a statement about the unit as it is linked
  only if the `.cmx` is right about the unit that the program is
  linked with, and if the prefix says nothing that the file does not
  say. Both are trusted; the header of the tool says what the second
  is trusted with. The tool checks, before it uses a prefix, that it
  binds each name once, none that the unit binds and none that is
  still an import, and that it evaluates from the imports.
- **validated with nothing else imported, given the .cmx of N**,
  followed by **not validated with its imports**: the validator
  accepted the two terms with the prefix in front with a verdict
  that does not pass the test from the imports (`../README.md`,
  "What the verdicts mean"), and an import is left. What is proved
  is of the two terms run with no import at all.
- The other outcomes are those of the directory above, of the two
  terms with the prefix in front: "not validated"; "nothing to
  validate, as the term before is undefined", where the term before
  is undefined and the validator does not accept the pair; and,
  where it is undefined and the validator accepts the pair, a line
  after the verdict which says that the validation says nothing of
  the unit. No line begins "validated:" for a unit that has a
  prefix.
- What the prefix leaves out is not said to be absent. Code that has
  no body in the file is declared deleted and bound to nothing, and
  a call of it is undefined when it runs. A module block with a field
  of which the file says nothing is bound with a symbol of its own in
  that field ("The pairs"), but for one case: where such a field is
  loaded in the body of code or in the handler of a recursive
  continuation, the block is not bound. Its symbol then stays an
  import, a load from it has no value, and where nothing of the file
  can be bound or declared there is no prefix and the tool prints

      the tool's Simplify read the .cmx of: F

  in its place, as it does where the prefix is made from fewer units
  than were read.
- The guard of the tool (`../README.md`, "What a validation does not
  say") asks its two tests, where the two terms have a prefix, of
  the two terms with the prefix in front of them, as they are run.
  `statics_in_scope` asks that the code of a closure, and of a
  direct call, be bound or declared deleted, and the prefix declares
  deleted the code that it has from the file of the code's own unit
  and does not bind. Of 23 of the 25 units here for which a prefix
  is made the guard passes, and the line "the check of static names"
  says "passes" of both terms and ends ", each with the prefix in
  front of it". Of `never_called_unread` and of `maker_kept_unread`
  the check of static names fails of both terms: those two are
  controls, whose prefixes name code of a unit that the tool did not
  read ("The pairs"). The tool prints of each the one line

      X.fl: not checked: with the prefix made from the .cmx files in front of it, the term before Simplify mentions a code id where it is not in scope

  and neither runs the two terms nor asks the validation. The line
  is for a name of another unit, of which the prefix says nothing;
  it says nothing of those two units themselves.
- Nothing is said of the first unit itself: its initialisation is
  not run, and what its `.cmx` says of it is taken as it is.

The units of this directory are in none of the tables and counts of
`../README.md` and of `../census/README.md`, which are of units that
use no other unit. A verdict that is given a `.cmx` is counted apart
from the others.

## The pairs

| Second unit | First unit | What the second unit does | What the tool says of it |
|-------------|------------|---------------------------|--------------------------|
| `lengths_called_in_function.ml` | `lengths.ml` | a function calls a function of the first unit | a prefix of five bindings; both runs return the module block; the same; validated with its imports, given the `.cmx` |
| `lengths_called_at_toplevel.ml` | `lengths.ml` | a toplevel calls one | the same prefix; undefined before and after; validated, with the line that says the validation says nothing of a unit whose term before is undefined |
| `lengths_included.ml` | `lengths.ml` | a toplevel reads the fields of the first unit and keeps them (`include`) | the same prefix; both runs return a block of the two closures; the same; validated with its imports, given the `.cmx` |
| `lengths_kept_with_exception.ml` | `lengths.ml` | the same, and exports a predefined exception too | the same prefix, and one import left; both runs return a block of the two closures and the imported exception; the same; validated with its imports, given the `.cmx` |
| `never_called.ml` | `never.ml` | calls a function that is never inlined, whose code has no body in the `.cmx`, in a function and at toplevel | a prefix of three bindings; undefined before and after; validated, with the line that says the validation says nothing |
| `never_called_again.ml` | `never_called.ml` | a toplevel calls the function of `never_called.ml`, which calls that of `never.ml` | the tool reads the `.cmx` of both units: a prefix of five bindings, with the code of `add_two`, and the code of `add_one` declared deleted; undefined before and after; the check of static names passes; validated, with the line that says the validation says nothing |
| `never_called_unread.ml` | `never_called.ml` | the same unit; the tool is given the `.cmx` of `never_called.ml` alone | a control: the prefix binds the code of `add_two`, which names the code of `add_one`, of a unit that was not read and of which nothing is declared; not checked: with the prefix in front of it the term before mentions a code id where it is not in scope (the guard, above) |
| `maker_kept_unread.ml` | `maker_kept.ml` | a toplevel reads a field of the second unit that is a constant | a control: the tool reads the `.cmx` of `maker_kept.ml` and not that of `maker.ml`; the prefix binds the module block and the closure that is a field of it, whose code is of `maker.ml`, a unit that was not read, and is not declared; not checked: with the prefix in front of it the term before mentions a code id where it is not in scope (the guard, above) |
| `maker_kept_read.ml` | `maker_kept.ml` | a toplevel keeps the function of `maker.ml`, which the second unit gives again | the tool reads the `.cmx` of both units: a prefix of five bindings, the same module block and closure, the closure of `make`, and the code of both, bound from the file of `maker.ml`; the check of static names passes; both runs return the module block; the same; validated with its imports, given the `.cmx` of both |
| `hidden_read.ml` | `hidden.ml` | a toplevel reads a field of which the `.cmx` says nothing, beside one that it knows, and adds the two | a prefix of one binding, the module block with a symbol of which nothing is known in that field; undefined before and after; validated with its imports, given the `.cmx`, with the line that says the validation says nothing |
| `hidden_known_read.ml` | `hidden.ml` | a toplevel reads the field that the `.cmx` knows, and not the one of which it says nothing | a prefix of one binding, the module block with a symbol of which nothing is known in the other field, which is the one import left; both runs return the module block; the same; validated with its imports, given the `.cmx` |
| `hidden_included.ml` | `hidden.ml` | a toplevel reads the two fields and keeps them (`include`) | the same prefix; both runs return a block of the number and of the symbol of which nothing is known; the same; validated with its imports, given the `.cmx` |
| `hidden_kept.ml` | `hidden.ml` | a toplevel puts the two fields in a block that it makes | the same prefix; both runs return the module block, whose field is the block of the symbol and of the number; the same; validated with its imports, given the `.cmx` |
| `hidden_in_function.ml` | `hidden.ml` | a function that nothing calls loads the field of which the `.cmx` says nothing | a control: the load is in the body of code, so the block is not bound and there is no prefix; both runs return the module block; the same; validated with its imports |
| `hidden_in_loop.ml` | `hidden.ml` | a loop at toplevel loads that field each time round | a control: the load is in the handler of a recursive continuation, so the block is not bound and there is no prefix; undefined before and after; validated with its imports, with the line that says the validation says nothing |
| `hidden_tested.ml` | `hidden.ml` | a toplevel tests that field (`match`) | the prefix of one binding; undefined before and after; validated with its imports, given the `.cmx`, with the line that says the validation says nothing |
| `hidden_same.ml` | `hidden.ml` | a toplevel compares that field with itself | the prefix of one binding; undefined before; the run after returns the answer that the two are the same; validated with its imports, given the `.cmx`, with the line that says the validation says nothing |
| `hidden_joined.ml` | `hidden.ml` | a toplevel takes that field down one arm of a test and a number down the other, and adds to what it has | the prefix of one binding; undefined before and after; validated with its imports, given the `.cmx`, with the line that says the validation says nothing |
| `several_included.ml` | `several.ml` | a toplevel reads the four fields, of each of which the `.cmx` says nothing, and keeps them (`include`) | a prefix of one binding, the module block with four symbols of which nothing is known, the four imports left; both runs return a block of the four; the same; validated with its imports, given the `.cmx` |
| `several_compared.ml` | `several.ml` | a toplevel compares two of the fields | the same prefix; undefined before and after; nothing to validate |
| `several_loaded.ml` | `several.ml` | a toplevel loads a field of one of the fields | the same prefix; undefined before and after; validated with its imports, given the `.cmx`, with the line that says the validation says nothing |
| `several_called.ml` | `several.ml` | a toplevel calls one of the fields | the same prefix; undefined before and after; validated with its imports, given the `.cmx`, with the line that says the validation says nothing |
| `partly_tested.ml` | `partly.ml` | a function loads, from its argument, the field of a record of which the `.cmx` says nothing, and adds to it; the toplevel reads the other field and gives the function the record | a prefix of two bindings, the record with a symbol of which nothing is known in that field; undefined before and after, as the addition has no value on such a symbol; validated with its imports, given the `.cmx`, with the line that says the validation says nothing |
| `parity_called.ml` | `parity.ml` | calls a function that is one of a set of two, in a function and at toplevel | a prefix of five bindings; undefined before and after; validated, with the line that says the validation says nothing |
| `cycle_read.ml` | `cycle.ml` | a toplevel reads a block that the first unit made by calls of the runtime, and tests it | a prefix of one binding, the module block with two symbols of which nothing is known; undefined before and after; validated with its imports, given the `.cmx`, with the line that says the validation says nothing |
| `probed_called.ml` | `probed.ml` | calls, without inlining it, a function of the first unit whose body has a probe | a prefix of five bindings without the body, which the conversion refuses; the run is undefined before and after, and the validation says nothing of it |
| `probed_inlined.ml` | `probed.ml` | calls the same function and lets Simplify inline it | not checked: the term after Simplify has the call that is a probe, which the conversion refuses |
| `phantom_called.ml` | `phantom.ml` | calls a function of the first unit whose body, in the `.cmx`, has a phantom binding | a prefix of three bindings without the body, which the conversion refuses; the run before is undefined, the run after returns; nothing to validate |

Forty-six sources. Five are the units that are here for their
first line and their controls, which the last section is about. Of
the other forty-one, twelve are first units and use no other unit;
ten of them have an expected file: six are validated, as a unit
of the directory above is, `phantom.ml` and `probed.ml` are not
checked, and two that
no unit uses are here for the lists of calls of known arity:
`known_arity_lists_older_code.ml` has the same two runs and is not
validated, and `known_arity_lists_undefined_elsewhere.ml` is
undefined before and after. Of the twenty-nine second units, one has
no expected file (`maker_kept.ml`, which is compiled for the two
units that use it); eight are validated given a `.cmx`, five of them
with an import left; one is validated with its imports and has no
prefix (`hidden_in_function.ml`); sixteen are undefined in the model,
fifteen of them with a prefix, and two of the fifteen,
`phantom_called.ml` and `hidden_same.ml`, before Simplify only; and
three are not checked: the two controls that the guard stops, and
`probed_inlined.ml`, which the conversion refuses. `never_called.ml`
and `maker_kept.ml` are each the first unit of two of the
twenty-nine.

The prefix of the four units that use `lengths.ml` is the same: the
module block, the closures of `length_from` and of `length`, and the
code of `length`, whose body the `.cmx` has. That body calls the code
of `length_from` directly; `length_from` is recursive, its code has
no body in the `.cmx`, and so its code id is declared deleted and
bound to nothing. No import is left in the first three.

In `lengths_called_in_function.ml` the unit only makes its closure, so
both runs are defined. The term before loads `length` from the module
block of the first unit and calls what it loaded. In the term after,
Simplify has inlined `length` and left a direct call of the code of
`length_from`, with the symbol of its closure as the callee. With the
prefix in front the validator reads the load from the block, makes
the call direct, inlines the body of `length`, and finds the two
terms the same once the closure of the unit is lifted.

In `lengths_included.ml` the term before loads the two fields of the
module block and the term after names the two closure symbols; with
the prefix both runs are defined and return the block of the two.

`lengths_kept_with_exception.ml` is that unit with a predefined
exception exported too, which stays an import: the unit has a
prefix and an import. The validator accepts the two terms once the
unused static bindings are removed, at the head of the term after
too. That verdict has a theorem from the imports where no import is
a name that the groups at the head of the term after bind, once the
validation has normalised and tidied that term (`../README.md`,
"What the verdicts mean", has the test and its statements). The
exception is not such a name: the line begins "validated with its
imports, given the .cmx of 1 of them".

`lengths_called_at_toplevel.ml`, `never_called.ml` and
`parity_called.ml` call, as they are initialised, code that has no
body in the `.cmx`: `length_from` through `length`, a function that
is never inlined, a function of a recursive set. The call is
undefined in the model, before Simplify and after, and a validation
of a unit whose term before is undefined says nothing of it; the two
terms are compared all the same, and are accepted. The prefix of
`parity_called.ml` binds the closure of `odd`, which the unit does
not ask for, as it is a field of the module block: each of the two
closures is bound alone in a set of its own, which is less than the
`.cmx` says of them. Its prefix declares the code of both deleted:
that of `even`, which the unit calls, and that of `odd`, whose
closure is bound and whose code nothing asks for.

`never_called_again.ml` and `never_called_unread.ml` are one unit
under two names, and the two sides of a control of the rule by which
the prefix declares code deleted. The unit calls `add_two` of
`never_called.ml`, whose body is in the `.cmx` of that unit and
calls the code of `add_one` of `never.ml` directly; that code has no
body. Of `never_called_again.ml` the tool's Simplify reads the
`.cmx` of both units: the prefix binds the code of `add_two` and
declares the code of `add_one` deleted, and the check of static
names passes. Of `never_called_unread.ml` the tool is given the
`.cmx` of `never_called.ml` alone ("The load path"): the unit of
`never.ml` is not read, the prefix binds the code of `add_two`,
which names the code of `add_one`, and declares nothing of a unit
that was not read. The check of static names fails of both terms
with that prefix in front of them, as it should, as nothing is known
of that code: the guard stops the unit.

`maker_kept_unread.ml` and `maker_kept_read.ml` are the two sides of
a second control, of the code of a closure. `maker.ml` has a
function, to be inlined, that makes a closure of what it is given.
`maker_kept.ml` keeps such a closure, of a value that is not known:
its Simplify inlines the function and makes no new code for the
closure, so the closure is a symbol of `maker_kept.ml` and its code
is still that of `maker.ml`. `maker_kept_unread.ml` reads a constant
of `maker_kept.ml`: the tool's Simplify reads the `.cmx` of
`maker_kept.ml` and not that of `maker.ml`; the prefix binds the
module block and, as a field of it, that closure; nothing is
declared of its code, and the guard stops the unit.
`maker_kept_read.ml` keeps the function of `maker.ml`, which
`maker_kept.ml` gives again: the tool's Simplify reads both files,
the prefix binds the code of the closure from the file of
`maker.ml`, and the guard passes. Another control,
`../../prefix_fault/`, is of the test that the tool makes of a
prefix before it uses one: a prefix that declares deleted a code id
which a unit binds, or declares deleted itself, is refused. No
source and no text makes such a unit, so that control is of
prefixes and units made by hand.

The module block of `hidden.ml` has a field of which the `.cmx` says
nothing. The prefix binds the block with a symbol of its own in that
field: an import of which nothing is known, that a run can pass on
and cannot inspect. `hidden_known_read.ml` loads only the other
field: the load has the value that the `.cmx` gives, and the unit is
validated with that one import left. `hidden_included.ml` and
`hidden_kept.ml` load the field of which the `.cmx` says nothing and
pass it on, the first as a field of its own module block and the
second in a block that it makes: the load has the symbol for its
value, both runs are defined and return it, and the units are
validated with that import left. `several_included.ml` is the same
of a first unit all of whose fields are such.

The block is bound in those units because the load is outside code
and outside the handler of a recursive continuation: there the
validation reads the term after, as it reads the term before, with
what the unit binds its symbols to. Where such a load is in the body
of code or in such a handler the block is not bound (`make_prefix`
has the rule and what it is for). `hidden_in_function.ml` and
`hidden_in_loop.ml` are the controls of that: neither has a prefix.
The first only makes its closure when it runs, and is validated with
its imports; the run of the second is undefined before and after,
as the load from an import has no value.

A unit whose run inspects such a symbol is undefined in the model
before Simplify, and of such a unit no line is a validation:
whatever the tool prints of the comparison of the two terms, it
prints under it that the validation says nothing of the unit, or it
prints "nothing to validate". Where the tool prints "validated with
its imports" above the line "the term before is undefined in the
model, so the validation says nothing of this unit", that is not a
meaningful validation of the unit: the run before goes wrong in the
model, and the refinement then asks nothing of the term after.
`hidden_read.ml` adds to the field,
`hidden_tested.ml` tests it, `hidden_joined.ml` adds to what is the
field down one arm of a test and a number down the other,
`several_loaded.ml` loads a field of it and `several_called.ml`
calls it. The two others compare. `hidden_same.ml` compares the
field with itself: the model has no value for that comparison
(decision D41 of `middle_end/flambda2/rocq/DECISIONS.md`), Simplify
makes the comparison of a variable with itself its answer, so the
run after returns, and the validation makes it that answer in the
term before too
(`middle_end/flambda2/rocq/rewrites/lets/FlambdaSameComparisonSpec.v`,
a rewrite of the term before only, which refines), and accepts the
pair. `several_compared.ml` compares two such fields: both runs are
undefined, and the validator does not accept the pair.

`partly_tested.ml` shows that such a symbol is not a value that the
model computes with. The load of the field is in a function, from
its argument, which the making of the prefix does not see as a load
from the record: the record is bound, the function gets the symbol
of the field, and the addition has no value on it, so the run is
undefined before and after.

Of `cycle.ml` the `.cmx` says nothing of the two fields of the
module block, and nothing that Simplify can use. The prefix of
`cycle_read.ml` binds that block with a symbol of which nothing is
known in each field. The unit loads one of the two and tests it,
which has no value on such a symbol: the run is undefined before
and after, and the validator accepts the pair.

The first units `lengths.ml`, `never.ml`, `hidden.ml`, `several.ml`,
`partly.ml` and `parity.ml` use no other unit; the tool says of each
what it says of a unit of the directory above, and each is validated.

`cycle.ml` is compiled and the tool is not run on it. Its two blocks
name each other, and the compiler makes them by four calls of C
functions of the runtime (`caml_alloc_dummy`, `caml_update_dummy`).
The model has no call of a C function. (The tool could not read the term
at all while `-drawfexpr` printed such a call as `apply ccall (...)`
with no region, where the parser asks for one after `ccall`
(`middle_end/flambda2/parser/flambda_parser.mly`, rule `call_kind`); the
text writes the region now, `apply ccall &toplevel`.) The `.cmx` of that
unit says nothing of its two fields, which hold what the calls returned.

`probed.ml` is not checked: its term has a call that is a probe,
which the conversion refuses, and its expected file has that one
line. (When the unit came in, the text of a term did not say that a
call is a probe, and a verdict of the tool on it would have been of
a term that is not the compiler's: the unit had no expected file,
and `from_ocaml.sh` here refused a unit with a probe that had one.
The text says so now, `probed.fl` has the mark, and the script
refuses no unit.) Its `.cmx` has the body of `bump`,
in which the call of the handler of the probe is marked as one. The
program runs that handler only while the probe is enabled, and the
model has nothing of that, so the conversion refuses a call that is
a probe. In `probed_called.ml`, which forbids the inlining of `bump`,
the body is therefore left out of the prefix, as a body that the
model has not got is: the closure of `bump` is bound, its code is
declared deleted and bound to nothing, and the call is undefined
before and after. The
tool as it was when the prefix came in took the probe for a plain
call and printed, for this unit, a prefix of six bindings with the
body of `bump` and the code of the handler; both runs were undefined
then too, as the handler calls `note`, whose code has no body in the
`.cmx`. In `probed_inlined.ml` the tool's Simplify inlines `bump`,
the term after has the call that is a probe, and the unit is not
checked.

`phantom.ml` is compiled with `-g -flambda2-expert-phantom-lets`
(its first line). With them Simplify may keep an unused binding as
phantom instead of deleting it, and here keeps the load of a field
that nothing uses; the program runs nothing for such a binding: the
code made of the function loads nothing. The body of the function in
the `.cmx` has that binding. The model runs every `Let`, so the
conversion refuses a phantom binding, and the prefix of
`phantom_called.ml` has the closure and not the body. The call is
then of code that has no body, undefined in the model before
Simplify; after Simplify, which inlined the body and dropped the
binding, the unit returns 0, as the program does. The tool as it was
before took the binding for a `Let` that runs and printed, for this
unit, a prefix of three bindings with the body and its load; the run
before was undefined then too, at the load of a field of the number
0, which the program does not make. On `phantom.ml` itself the tool
is given the two flags, the term that its Simplify makes has the
phantom binding, and the unit is not checked.

`known_arity_lists_older_code.ml` is here for its first line and not
for a pair: it is compiled with `-O3`, which the rules of the
directory above have no way to ask for, and no unit uses it. With
`-O3` Simplify makes, in the function `run`, a call of known arity of
a callee that is one of two closures chosen by a test, and lists the
code of the two. It then inlines `run` where the unit calls it; the
closures of the copy are of newer versions of that code, and the
call of the copy still lists the code as `run` has it. The
compiler's interface allows it (`terms/call_kind.mli`: "only the
provided functions (or a simplified version thereof) can be
called"). The machine of the model reads a list as it stands, and
with the compiler's list alone it would call the term after
undefined; the conversion extends the list under "newer version
of", as far as the evidence and the scope allow (decision D59 of
`middle_end/flambda2/rocq/DECISIONS.md`; `rocq/TRUST-LEDGER.md`,
"The code ids that a call of known arity lists"), and for this unit
the two runs are then the same. The unit is not validated:
the validation has no rewrite that makes that call of known arity
in the term before.

`known_arity_lists_undefined_elsewhere.ml` is the same unit with one
thing more: its toplevel then calls a number, which is undefined in
the model whatever the lists are. The call of known arity inside
`run` lists the code of the two closures of `run`; the newer versions
of that code are bound later, for the copy of `run` at toplevel, out
of the scope of that call, so the conversion does not add them
there. The tool refuses a unit for a list only where a run stops at
a call of known arity whose list alone makes it undefined, with code
that the compiler's reading of the list may allow; here the run
stops elsewhere, so it is reported as of any unit: both runs
undefined, and nothing to validate.

## Units that are here for their first line

| Unit | Its first line | What it is | What the tool says of it |
|------|----------------|------------|--------------------------|
| `handler_kept_with_g.ml` | `(* flags: -g *)` | a function whose `try` catches, by name, the exception that its body raises | both runs return the module block; the same; not validated, with no line of the second check |
| `handler_gone_without_g.ml` | none | its control: the same source without the line | validated with its imports |
| `handler_gone_with_g_notrace.ml` | `(* flags: -g *)` | a control: the same with `raise_notrace` in the place of `raise` | validated with its imports |
| `handler_of_reraise_kept_with_g.ml` | `(* flags: -g *)` | a `try` inside a `try`: the body raises by `raise_notrace`, the inner handler raises again what it caught, and the outer handler catches that by name | both runs return the module block; the same; not validated, with no line of the second check |
| `handler_of_reraise_gone_without_g.ml` | none | its control: the same source without the line | validated with its imports |

Each has one import, the predefined exception that it raises, and
no prefix: it uses no other unit.

For a continuation of the sort `Normal_or_exn` that is used by a pop
of a trap, Simplify marks the use as one that it will not make a
jump (`escaping`) if debugging information is on and the raise of
the pop is a regular one, a raise again of what was caught, or has
no kind; it does not mark it so if debugging information is off,
nor, on or off, if the raise is one with no trace, which
`raise_notrace` makes. A use of a continuation of the sort `Return`
or `Toplevel_return` never has the mark (`apply_cont_use_kind` in
`middle_end/flambda2/simplify/simplify_common.ml`, whose comment
gives the backtrace as the reason; discrepancy C5-4). The model has
no backtraces, and gives a raise the same meaning whatever its
kind.

What the five units show of that, in the term that the compiler's
Simplify makes of each. With `-g` the term after of
`handler_kept_with_g.ml` keeps the handler, the push and the pop of
its trap, and the regular raise. Without `-g`
(`handler_gone_without_g.ml`), and with `-g` where the raise is by
`raise_notrace` (`handler_gone_with_g_notrace.ml`), the raise is a
jump and the term after has no trap. In
`handler_of_reraise_kept_with_g.ml` the first raise, by
`raise_notrace`, is a jump to the inner handler, with `-g`; the
raise again of what that handler caught is left, with the outer
handler and its trap; without `-g`
(`handler_of_reraise_gone_without_g.ml`) both are jumps. That is of
these five units: it is not said here that Simplify makes a jump of
every raise that has not the mark.

The validation removes a handler of exceptions in the term before
(`FlambdaDemoteSpec.demote_handlers`), and not in the term after:
where Simplify kept the handler the two terms differ by it, and the
unit is not validated.

The two expected files that say "not validated" say what the tool
of this tree prints of those units. They are no finding about the
compiler: the two runs are the same, and what stops the validation
is that limit of it. What is expected to turn them is the same
removal in the term after, which is not done
(`middle_end/flambda2/rocq/status/C5.md`). The units are here with
their controls so that their expected files change when it comes.
The flags are given to the tool as to the compiler, and
with `-g` the tool's Simplify keeps the handler as the compiler's
does; the directory above, whose rules give no flags, has no unit
compiled with `-g`.
