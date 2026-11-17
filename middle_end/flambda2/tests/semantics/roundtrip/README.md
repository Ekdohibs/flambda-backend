# The conversion of the test tool, through fexpr and back

`../../tools/flsemantics.exe -round-trip X.fl` converts the unit of
`X.fl` to a term of the Rocq model, prints that term as fexpr
(`../../tools/flmodel_to_fexpr.ml`), reads the text again with the
compiler's parser, converts what it read, and compares the two terms of
the model with the model's comparison (`../../tools/flroundtrip.ml`). It
does the same of the unit that Simplify makes of the one in the file.
The rules of this directory run it on every example of the directory
above, of `../ocaml` and of `../ocaml/library`; `dune` says how the
answers are checked.
A unit read from a text, the unit of the file and each unit read
back, is first normalised as the check of a unit normalises it
(`with_lost_kinds` in `../../tools/flsemantics.ml`): each
`Opaque_identity`, whose kind the text does not write, is given the
kind of its argument; and each direct call and call of unknown arity,
the arity of whose arguments the text does not write, is given one, a
direct call that of the parameters of its code and a call of unknown
arity the kinds of its arguments.

The conversion is trusted (`rocq/TRUST-LEDGER.md`). What this adds is a
second reading of each of its cases, in the other direction, and a
check that the two agree on every example: a case that the conversion
gets wrong in a way that is not the same both times is found. It does
not find what the conversion drops and the printer does not put back.

The two terms are compared with the value slots of each set of
closures in the order of their numbers. The model keeps them as a list
in the order of the compiler's map, which is the order in which the
compiler made the slots, and the text cannot keep that order: the
parser makes a slot where the text first names it, which may be in the
code of a function that projects only some of the slots, before the
set. After Simplify a closure that keeps a function of its module and
an argument is such a case. The machine looks a slot up by its name,
and what a lookup finds does not depend on where the slot stands among
the others; what each slot holds is compared. This is a choice of what
the round trip tells apart, not a proof that two such terms are
equivalent: a closure that is a value still holds its slots as a list
in the order of the term.

## Answers that are not "round trip the same"

Each has a file here with what the tool prints, and its reason below.
There are six: four of units of `../ocaml/library` and two of units
of `../ocaml`.

The four are `ocaml_library_int64_identities.expected`,
`ocaml_library_int64_laws_more.expected`,
`ocaml_library_nativeint_identities.expected` and
`ocaml_library_nativeint_laws_more.expected`. In each the round trip
is the same of the unit of the file, and the unit after Simplify is
"not read back: the parser of fexpr fails", with the exception
`Failure("int_of_string")`.

The term after Simplify of each has a naked 64-bit or native integer
at an end of its range as the argument of a continuation
(`k (-9223372036854775808L)`), where Simplify has removed the box of
two numbers that meet. The printer of fexpr writes such a number with
all its digits and its suffix. The parser gives the digits of a
constant that stands in an expression to `int_of_string` before it
looks at the suffix (`middle_end/flambda2/parser/flambda_parser.mly`,
`make_const_int`), and that fails outside 63 bits (discrepancy
C1-11). The exception comes out of the parser itself and is no error
that it reports; `read_back` (`../../tools/flroundtrip.ml`) makes its
line of it as it does of one. The term before of each of the four has
the same numbers as static data, which the parser reads another way
(`make_boxed_const_int`).

It is a limit of fexpr and not a fault of the conversion or of the
printer of the model, which writes the number that the term has; the
compiler's own text of the term after of each of the four stops the
parser at the same place. The round trip says nothing of the term
after of those four units. The check of a unit, as the rules of the
tests run it, reads no text of its term after, so their verdicts are
not touched. The other sixteen units of that directory are the same
before and after Simplify.

Until `read_back` made that line, the tool ended on the exception at
the term after of those four, with status 2 and no second line, and
no rule ran the round trip on that directory: it was seen when the
round trip was run there by hand, with the tool of the tree that
first had those units together with all that came in with them.

The two are `ocaml_made_array_store_of_integer.expected` and
`ocaml_made_array_checked_store.expected`. In each the round trip is
the same of the unit of the file, and of the unit after Simplify it
"DIFFERS", with the line "the two terms are printed the same: they
differ in what the printer does not show".

The term after Simplify of each has a store into an array that
carries the array kind of immediates and the set kind of values that
the collector may ignore. The front end writes a store into an array
of integers with both kinds those of values that the collector may
ignore; Simplify changes the array kind where the type of the array
says that its elements are immediates, and keeps the set kind
(`Simplify_common.specialise_array_kind`). The text of fexpr writes
the array kind of a store, and its set kind only for an array of
unboxed products; the parser gives a store whose set kind is not
written the one that goes with its array kind
(`middle_end/flambda2/parser/fexpr_prim.ml`, `array_set`; discrepancy
P1-9). So the unit read back has the set kind of immediates where the
unit printed has that of values that the collector may ignore, and
the texts of the two are the same.

It is a limit of what fexpr writes, and not a fault of the conversion
or of the printer of the model, which gives fexpr the array kind and
the set kind that the term has: the parser reads a set kind that is
written, for an array of any kind, so the text could carry it. The
check of a unit, as the rules of the tests run it, reads no text of
its term after, so the verdicts of the two units are not touched; the
term before of each, which is read from a text, has the two kinds
that go together.

There was another, `kinds_direct_naked.fl` after Simplify, until a unit
read from a text was given the arity of the arguments of its calls.
Simplify makes the call of `$Add_naked` a direct call of its code, at
the arity of two naked immediates. The printer writes that arity; the
parser of fexpr gives a direct call the arity of as many values as it
has arguments, whatever is written
(`middle_end/flambda2/parser/fexpr_to_flambda.ml`, the case of
`Function (Direct ...)`; discrepancies C2-5 and R2-12). So the unit
read back was not the unit printed. It is a limit of fexpr and it is
still there: the round trip no longer shows it, as the unit read back
is given the arity of the parameters of the code before it is
converted. What the round trip shows of such a call is that the
conversion and the printer agree once that is done.

## What cannot be printed

`flmodel_to_fexpr.ml` lists at its head what it refuses, with the
reason for each. No example meets one today.

## The constant `poison`

Simplify unboxes a variant where two branches meet, and gives the
fields that a branch has not got the constant `poison`; eight examples
of `../ocaml` that the tool converts have it after Simplify
(`option_join.fl` and seven whose names end in `_heap.fl`). The
conversion makes that constant of the model, which keeps its kind and
no name. The printer writes it as
`poison.KIND."poison"`, with the quotes in the name that it gives the
printer of fexpr: that printer writes the name of a poison constant as
it is (`middle_end/flambda2/parser/print_fexpr.ml`), and the parser
reads a string there (`middle_end/flambda2/parser/flambda_parser.mly`,
the rule `const`), so a poison constant printed with the name that the
compiler gave it is not read back (discrepancy C2-14).

A poison constant whose kind is a naked `int8`, a naked `int16` or
`mask` is printed all the same and is not read back today: the
printer of fexpr writes those three kinds and the parser has no rule
for them (`flambda_parser.mly`, the rule `naked_number_kind`;
discrepancy R6-1). No example has one.
