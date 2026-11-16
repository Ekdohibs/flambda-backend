# Units written against the standard library

The sources of the directory above are compiled with no standard
library and declare the primitives that they use; those here are
compiled with the standard library of this tree, with `Stdlib` opened
as the compiler opens it, so that a source says `Int32.add` or
`Iarray.get` as a program does. A unit is kept here only if the term
that the compiler gives to Simplify names no symbol of another unit,
which `../from_ocaml.sh` tests: what the source took from the library
is then in the term as primitives, the term needs nothing but itself,
and no `.cmx` file is read.

So a verdict here rests on nothing more than a verdict of the
directory above does, and says what one says there (`../README.md`,
"What the verdicts mean" and "What a validation does not say"). It
says nothing of the functions of the library: a unit that calls one
is refused.

## How a term is made

    sh ../from_ocaml.sh <ocamlopt> X.ml X.fl <directory>

with the directory of the standard library as the last argument. The
script then compiles the unit with that directory in the place of the
compiler's own standard library (`-nostdlib -I <directory>`, and not
`-nopervasives`), as far as the middle end, and with `X.mli` if it is
beside `X.ml`. It refuses what it refuses of a unit of the directory
above, and one thing more: a unit whose term names a symbol of
another unit, which the text writes with the name of that unit before
it. The script looks for that in the text of the term and does not
parse it: it may refuse a unit for a string constant that reads as
such a symbol, and lets through no unit that has one. A predefined
exception is not of another unit, and a unit that names one is kept:
the tool runs it from its imports, as it does in the directory
above. A function of the runtime, which the text writes as a symbol
of a unit of its own, is refused with the others: the tool reads no
term that calls one. `../script/library_function.ml` is a unit that
the script refuses so, and `../script/library_function_not.ml` the
same unit with a function of its own in the place of the library's.

The rules are written by `gen_rules.sh` and included by `run/dune`.
Those that make a term again and compare it with `X.fl` give the
script the standard library that the compiler of the context finds,
and are for the context `main` only, where that is the library that
the build of this tree made. A term depends on the interfaces of the
library: the rules make it again when one of them changes, and it
differs, or the unit is refused, if a primitive that the source uses
becomes a function there. The rules that run the tool on `X.fl` and
compare what it prints with `X.expected` are for every context, as
in the directory above: the tool is given no directory to read a
`.cmx` from, and runs at the default flags. As in the directory
above, the rule takes the status 1, and no other, of a unit that
`tool_fails.txt` here names.

To add a unit, add `X.ml` (and `X.mli`), `X.fl` and `X.expected`, and
nothing else. Each unit is looked at before it enters as the units of
the directory above are (`../README.md`, "What a validation does not
say"): for a kind that may be null where the text writes a kind, and
for the modes of the parameters of its code.

## The examples

Twenty sources: 9 validated, 11 validated with their imports. The
import of each of the 11 is a predefined exception, that of a read
out of the bounds of an array. Steps are those of the machine, before
and after Simplify.

Fourteen are on immutable arrays (sources by T5, each cut before the
part that prints the values: the comment at its head says what the
unit gives, which is what both runs return here). Six are on the
fixed-width integers (sources by T5b).

| Source | Steps | Verdict | What it has |
|--------|-------|---------|-------------|
| `b_match_only.ml` | 20, 14 | validated | a match of four constants, of which Simplify makes a table |
| `c_exported.ml` | 17, 12 | validated | an immutable array made as the unit runs and exported |
| `c_in_a_record.ml` | 45, 36 | validated with its imports | a record with a field that is an immutable array, made by one function and read by another |
| `e_empty_in_one_arm.ml` | 47, 28 | validated | one arm of a test gives the empty array and the other an array of two elements |
| `f_reader_given_as_an_argument.ml` | 43, 34 | validated with its imports | a function that is kept makes an array and gives it to the function that it is given |
| `g_boxed_int64s.ml` | 34, 27 | validated with its imports | an immutable array of boxed 64-bit integers |
| `int32_identities.ml` | 184, 51 | validated | `Int32` operations with one operand written in the function, or the same operand twice |
| `int32_laws_more.ml` | 302, 63 | validated | laws of the integers on `Int32` numbers: one that holds whatever the width, and three that fail at the ends of the range |
| `int64_identities.ml` | 184, 51 | validated | as `int32_identities.ml`, on `Int64` |
| `int64_laws_more.ml` | 302, 63 | validated | as `int32_laws_more.ml`, on `Int64` |
| `l_summed_by_a_recursive_function.ml` | 251, 178 | validated with its imports | a recursive function sums an array made by another function |
| `m_constant_and_variable.ml` | 45, 30 | validated with its imports | a literal with a variable between two constants |
| `nativeint_identities.ml` | 184, 51 | validated | as `int32_identities.ml`, on `Nativeint` |
| `nativeint_laws_more.ml` | 302, 63 | validated | as `int32_laws_more.ml`, on `Nativeint` |
| `r_constant_index.ml` | 41, 31 | validated with its imports | an array made by a function that is kept, read at two constant indices |
| `r_out_of_bounds_caught.ml` | 57, 39 | validated with its imports | a read past the end of an array, caught by a handler |
| `r_out_of_bounds_escapes.ml` | 32, 26 | validated with its imports | a read past the end with no handler: both runs raise the exception |
| `t_two_lengths.ml` | 61, 43 | validated with its imports | the arms make arrays of one and of three elements; the last element is read by the length |
| `v_of_closures.ml` | 53, 45 | validated with its imports | an immutable array of two closures over a parameter; one is read and applied |
| `v_two_parameters.ml` | 33, 23 | validated with its imports | a function that is kept returns the immutable array of its two parameters; both are read |

No unit here has a kind that may be null where the text writes a
kind. Four have code with a parameter whose mode is not heap
(`f_reader_given_as_an_argument`, `int32_laws_more`,
`int64_laws_more`, `nativeint_laws_more`), and the term before
Simplify of those four and of `e_empty_in_one_arm` has an allocation
mode that names a region; `../README.md` says what the tool and the
conversion make of both.

The text of the term after Simplify of four of the six on the
fixed-width integers (`int64_identities`, `int64_laws_more`,
`nativeint_identities`, `nativeint_laws_more`) is not read by the
reader of the text of terms, which stops on a naked integer outside
63 bits (discrepancy C1-11). The verdicts above are not touched: the
check of a unit reads no text of its term after. The round trip of
the conversion does, and says of the term after of those four that
it is not read back; its rules run on this directory and keep the
four answers (`../../roundtrip/README.md`).

The sources with the syntax of immutable arrays are not formatted
(`../.ocamlformat-ignore`).
