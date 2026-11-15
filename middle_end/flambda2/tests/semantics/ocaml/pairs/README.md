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

## How a pair is made and checked

    sh from_ocaml.sh <ocamlopt> F.ml F
    sh from_ocaml.sh <ocamlopt> F_more.ml F_more F

Each is, in a directory of its own, to which `F.cmi` and `F.cmx` are
copied first for the second unit,

    ocamlopt -nopervasives -nostdlib -stop-after middle_end -c \
      X.ml -drawfexpr-to X.fl

which writes `X.cmi` and `X.cmx` beside `X.fl`; the three are kept in
the directory of the build. Then, in that directory,

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
  no body in the file is bound to nothing, and a call of it is
  undefined when it runs. A module block with a field of which the
  file says nothing is not bound: its symbol stays an import, a load
  from it has no value, and where nothing of the file can be bound
  there is no prefix and the tool prints

      the tool's Simplify read the .cmx of: F

  in its place, as it does where the prefix is made from fewer units
  than were read.
- The line "the check of static names" (`../README.md`, "What a
  validation does not say") is, where the two terms have a prefix,
  of the two terms with the prefix in front of them, and then ends
  ", each with the prefix in front of it". Of 8 of the 10 units here
  that have a prefix it says "fails" of both terms
  (`lengths_called_at_toplevel`, `lengths_called_in_function`,
  `lengths_included`, `lengths_kept_with_exception`, `never_called`,
  `parity_called`, `phantom_called` and `probed_called`): the prefix
  binds a closure of the first unit and not its code, which
  `statics_in_scope` refuses whatever follows the prefix. So the
  line says nothing of those 8 units themselves. Of the 2 others
  (`hidden_known_read` and `partly_tested`), whose prefix binds no
  closure, it says "passes" of both terms.
- Nothing is said of the first unit itself: its initialisation is
  not run, and what its `.cmx` says of it is taken as it is.

The units of this directory are in none of the tables and counts of
`../README.md` and of `../census/README.md`, which are of units that
use no other unit. A verdict that is given a `.cmx` is counted apart
from the others.

## The pairs

| Second unit | First unit | What the second unit does | What the tool says of it |
|-------------|------------|---------------------------|--------------------------|
| `lengths_called_in_function.ml` | `lengths.ml` | a function calls a function of the first unit | a prefix of four bindings; both runs return the module block; the same; validated with its imports, given the `.cmx` |
| `lengths_called_at_toplevel.ml` | `lengths.ml` | a toplevel calls one | the same prefix; undefined before and after; validated, with the line that says the validation says nothing of a unit whose term before is undefined |
| `lengths_included.ml` | `lengths.ml` | a toplevel reads the fields of the first unit and keeps them (`include`) | the same prefix; both runs return a block of the two closures; the same; validated with its imports, given the `.cmx` |
| `lengths_kept_with_exception.ml` | `lengths.ml` | the same, and exports a predefined exception too | the same prefix, and one import left; both runs return a block of the two closures and the imported exception; the same; validated with its imports, given the `.cmx` |
| `never_called.ml` | `never.ml` | calls a function that is never inlined, whose code has no body in the `.cmx`, in a function and at toplevel | a prefix of two bindings; undefined before and after; validated, with the line that says the validation says nothing |
| `hidden_read.ml` | `hidden.ml` | a toplevel reads a field of which the `.cmx` says nothing, beside one that it knows | no prefix; undefined before and after; nothing to validate |
| `hidden_known_read.ml` | `hidden.ml` | a toplevel reads the field that the `.cmx` knows, and not the one of which it says nothing | a prefix of one binding, the module block with a symbol of which nothing is known in the other field, which is the one import left; both runs return the module block; the same; validated with its imports, given the `.cmx` |
| `partly_tested.ml` | `partly.ml` | a function loads, from its argument, the field of a record of which the `.cmx` says nothing, and adds to it; the toplevel reads the other field and gives the function the record | a prefix of two bindings, the record with a symbol of which nothing is known in that field; undefined before and after, as the addition has no value on such a symbol; validated with its imports, given the `.cmx`, with the line that says the validation says nothing |
| `parity_called.ml` | `parity.ml` | calls a function that is one of a set of two, in a function and at toplevel | a prefix of three bindings; undefined before and after; validated, with the line that says the validation says nothing |
| `cycle_read.ml` | `cycle.ml` | a toplevel reads a block that the first unit made by calls of the runtime | no prefix; undefined before and after; validated with its imports, with the line that says the validation says nothing |
| `probed_called.ml` | `probed.ml` | calls, without inlining it, a function of the first unit whose body has a probe | a prefix of three bindings without the body, which the conversion refuses; the run is undefined before and after, and the validation says nothing of it |
| `probed_inlined.ml` | `probed.ml` | calls the same function and lets Simplify inline it | not checked: the term after Simplify has the call that is a probe, which the conversion refuses |
| `phantom_called.ml` | `phantom.ml` | calls a function of the first unit whose body, in the `.cmx`, has a phantom binding | a prefix of two bindings without the body, which the conversion refuses; the run before is undefined, the run after returns; nothing to validate |

Twenty-one sources. Eight are first units; six of them have an
expected file: five are validated, as a unit of the directory above
is, and `phantom.ml` is not checked. Of the thirteen second units,
four are validated given a `.cmx`, two of them with an import left;
eight are undefined in the model, six of them with a prefix, and
one of the six, `phantom_called.ml`, before Simplify only; one is
not checked.

The prefix of the four units that use `lengths.ml` is the same: the
module block, the closures of `length_from` and of `length`, and the
code of `length`, whose body the `.cmx` has. That body calls the code
of `length_from` directly; `length_from` is recursive, its code has
no body in the `.cmx`, and so its code id is bound to nothing. No
import is left in the first three.

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
`.cmx` says of them.

In `hidden_read.ml` the module block of the first unit has a field of
which the `.cmx` says nothing, and the unit loads that field: the
block is not bound, the unit has no prefix, and the load from the
block has no value: the unit is as it was when the model was told
nothing of a `.cmx`. `hidden_known_read.ml` loads only the other
field of the same block. There the block is bound, with a symbol of
its own in the field of which the `.cmx` says nothing: an import of
which nothing is known, that a run can pass on and cannot inspect.
The load has the value that the `.cmx` gives, and the unit is
validated with that one import left.

`partly_tested.ml` shows that such a symbol is not a value that the
model computes with. The load of the field is in a function, from
its argument, which the making of the prefix does not see as a load
from the record: the record is bound, the function gets the symbol
of the field, and the addition has no value on it, so the run is
undefined before and after.

Of `cycle.ml` the `.cmx` says nothing that Simplify can use, and
nothing that the prefix can bind: the term after of `cycle_read.ml`
is the term before simplified as a unit of the directory above is,
and the validator accepts the pair; the unit is undefined all the
same, as it loads from an import.

The first units `lengths.ml`, `never.ml`, `hidden.ml`, `partly.ml`
and `parity.ml` use no other unit; the tool says of each what it says
of a unit of the directory above, and each is validated.

`cycle.ml` is compiled and the tool is not run on it. Its two blocks
name each other, and the compiler makes them by four calls of C
functions of the runtime (`caml_alloc_dummy`, `caml_update_dummy`).
The model has no call of a C function, and the tool cannot read the
term at all: `-drawfexpr` prints such a call as `apply ccall (...)`
with no region, and the parser asks for one after `ccall`
(`middle_end/flambda2/parser/print_fexpr.ml`, at `C_call`;
`middle_end/flambda2/parser/flambda_parser.mly`, rule `call_kind`).
So the `.cmx` of that unit says nothing of its two fields, which
hold what the calls returned.

`probed.ml` has no expected file: the text of its term does not say
that a call is a probe, so a verdict of the tool on it would be of a
term that is not the compiler's. `from_ocaml.sh` here refuses what
that of the directory above refuses, and what it refuses for a probe
or for a call of unknown arity (an unboxed product among the
arguments of such a call, or among the parameters of code of the
unit), only where the unit has an expected file: so `probed.ml` is
compiled for its second units, and would be refused if it were given
an expected file. Its `.cmx` has the body of `bump`,
in which the call of the handler of the probe is marked as one. The
program runs that handler only while the probe is enabled, and the
model has nothing of that, so the conversion refuses a call that is
a probe. In `probed_called.ml`, which forbids the inlining of `bump`,
the body is therefore left out of the prefix, as a body that the
model has not got is: the closure of `bump` is bound, its code is
bound to nothing, and the call is undefined before and after. The
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
