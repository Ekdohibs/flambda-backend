# The files of the standard library under the compiler's own run

For each file `stdlib/X.ml` that the build of the tree compiles to native
code, 79 of them, the rules of this directory compile the file as the build
compiles it, with the driver that checks the compiler's own run
(`../../tools/ownrun/flsemantics_ownrun.ml`; `../ocaml/README.md`, "The
compiler's own run"), and compare what the driver says of the compiler's
two terms, before and after Simplify, with a stored file,
`X.own_expected`. No source is kept here: the rules read the real files of
`stdlib/`. Nothing under `stdlib/` is changed for them, and no file of the
compiler's build.

At this commit the driver validates 26 of the 79 files, and "What a
verdict means for a file of the library" says what that is a statement
of. For the others the table says what stops each.

## How to run it

The rules are for the context `main`, where the driver is built. Two
aliases run them, and **`make runtest` runs neither**; nor does a build
of the compiler build or run anything of this directory ("What a rule
does"):

    dune build --workspace=duneconf/main.ws \
      @middle_end/flambda2/tests/semantics/stdlib/runtest-stdlib
    dune build --workspace=duneconf/main.ws \
      @middle_end/flambda2/tests/semantics/stdlib/runtest-stdlib-long

after `make boot-compiler`, which builds the library. The first has the
units whose rule takes at most 10 seconds of processor time; the second
has the others, which `long.txt` names (`map.ml` and `set.ml` at this
commit). "What it costs" has the times.

## When an alias fails

After a change of `stdlib/` or of the compiler, a stored file may no
longer be what the driver prints, and `dune promote` takes what it prints
now. Look at the difference first: if a line that began "validated" no
longer does, say so where the change is reviewed; a file that differs only
in the numbers of names or in places of the source says nothing new.

If the rule says instead that a comparison with the build's files failed
(`stdlib_run.sh` says which), there is nothing to promote: a file that
was written here, or what was printed here, is not the build's, and the
cause is to be found. "The library was not built from it as it is now"
says that the source is not the copy of it in the directory where the
library was built: the compiler is to be built again
(`make boot-compiler`) before the tests are run.

If the alias says that `run/dune.inc` is not what the generator writes
now (dune names the file and shows the difference), the rules by which
the build compiles the library have changed: another flag, another file
that a unit depends on. `dune promote` takes the new rules, and the
alias is run again.

**A file added to `stdlib/`** (`x.ml`, `x.mli` and the name in
`stdlib/StdlibModules`) has no stored file. The generator stops before
it writes the rules of that file, so that its output is not kept, and
the alias fails with one line:

    gen_rules: the build compiles x.ml, and there is no x.own_expected here

Add an empty `x.own_expected` and run the alias: it now fails at the
comparison of `run/dune.inc`, and `dune promote` takes the rules of the
new file. Run it again, the library built with the new file: the rule
of the new file fails at the empty stored file, and `dune promote` takes
what the driver prints of it.

A file taken away leaves its stored file, and the line is

    gen_rules: x.own_expected is of no unit that the build compiles

Remove the stored file, run the alias and promote the rules.

A file added with no `x.mli` gets from the build a rule of a shape that
no unit of the library has at this commit (its compilation writes its
own `.cmi`), and the generator refuses the shape, with the rule printed
after:

    gen_rules: it does not depend on the .cmi of the unit, in the rule of
    stdlib__X: (rule ...

So does any rule of the build that has not the shape that `gen_rules.ml`
knows: the line says what was missed, and `gen_rules.ml` and
`stdlib_run.sh` are then to be taught the shape (or, here, the file given
an interface). The three lines were got in a tree on 18 November 2026:
the first and the third by adding a file to the library's list, the
second by adding a stored file of no unit.

## What a rule does

The rules are in `run/dune.inc`, a file of the tree, which
`gen_rules.exe` wrote from the rules by which the build compiles the
library. Those are written by `stdlib/dune_rules/gen.ml` in the first
dune run of the tree (the context that builds the boot compiler and the
library). Under either alias, in the context `main`, the same generator,
given the same two files, writes them again as a text (`dune`, the rule
of `build_rules.txt`); `gen_rules.exe` writes the rules of this directory
from that text; and they are compared with `run/dune.inc`. So for an
alias to pass, the rules in `run/dune.inc` must be those that the
generator writes from the build's rules at that run, and none is written
by hand. (Where an alias fails at that comparison, the rules that ran
under it were those of `run/dune.inc` as it stood.)

**The rules are a file of the tree, and are not made when dune loads the
directory, so that a build of the compiler builds nothing and runs
nothing here, and cannot fail here.** No build of the compiler reads
the library's build for this directory; the two aliases do. A rule of
the build that has not the shape that
`gen_rules.ml` knows stops the generator, with the rule printed; so does
a file `stdlib/X.ml` that the build compiles and that has no stored file
here, and a stored file that is of no unit. Every stanza of the build's
rules must be of a kind that `gen_rules.ml` names, those that are of no
use here among them (the rules of the bytecode and of the archives, the
alias that names the library's files): one that is not stops the
generator too, so that a unit cannot be left out because its rule is
written in a new way. Each of those fails the alias that was asked, with
its message, and no build of the compiler.

One thing cannot be kept from dune: its own aliases `all` and `default`
name every file that the rules of a directory make, and `check` the
interface of every program, so `dune build @all` over this directory in
the context `main` runs every rule here, and a file that a rule of this
directory makes can be asked of dune by its name. No target of the Makefile asks
that: the build of the compiler asks `@install`, the files that it
names and the tools of `../../tools/`, and `make runtest` asks `runtest`,
which has nothing here. The
generator is a program of the context `main` only, so the first dune run
of the tree, which asks `check` over `middle_end/`, compiles nothing
here.

For one unit, `stdlib_run.sh` is given the compiler that built the
library (in the context `main` that is the compiler that dune runs,
`%{ocamlopt}`), the driver, the directory in which the library was built
(`%{ocaml_where}`), the real source, the names of the files that the
build's rule for the unit depends on, and the words of the build's
command. In a directory of its own it:

1. writes the source as the build gives it to the compiler (for every
   file but `sys.ml`, after a line that gives its name and the number 2),
   and fails if that is not, byte for byte, the copy of it in the
   directory where the library was built;
2. copies in the `.cmi` and `.cmx` files that the build's rule names, and
   no other: the unit's own `.cmi` among them, which is the one that the
   compilation reads (the build's command gives `-cmi-file`, and writes
   none);
3. runs **the compiler that built the library**, with the words of the
   build's command and two flags that print the terms before and after
   Simplify (`-drawflambda -dflambda`), to its end. **The `.cmx` and the
   `.o` that it writes must be the build's own, byte for byte.** That is
   a check of the compilation made here, in a directory of its own and
   with the two flags more: it wrote the two files that the build wrote;
4. runs **the driver** with the same words: the compilation goes to its
   end, and the check is made of its two terms;
5. fails unless **the two terms that the driver printed are those that
   the compiler of step 3 printed**, byte for byte;
6. fails unless **`ocamlobjinfo` (`tools/objinfo.ml`) prints the same of
   the `.cmx` that the driver wrote and of the build's own**, the line
   "CRC of implementation" left out;
7. prints what the driver said of the unit, which the rule compares with
   `X.own_expected`.

For a unit that fails one of the comparisons the rule fails and says
which, whatever the driver says of the unit, and nothing is offered to
promote.

What the three comparisons with the build's files show, and what they do
not:

- **The compilation made here wrote the build's `.cmx` and `.o`** (step
  3), byte for byte, with the compiler that built the library. That the
  words of the command are the build's does not come from this
  comparison: the generator takes them from the build's rules, and each
  alias compares what it writes with `run/dune.inc`. Nor does the
  comparison show that everything given to the compiler here was what
  the build gave it: a flag, a name, a directory, a variable of the
  environment or a file of another unit that differed and left both
  files as they are would not be seen. The two flags that print are not
  words of the build's command; step 3 shows at every run that they
  change neither file.
- **The two terms that the driver checked print as the two terms of
  that compilation** (step 5), byte for byte, numbers of names and all,
  or the rule fails. The driver is not the compiler that built the
  library. The library is built by the boot compiler, which is the
  sources of the tree compiled by the compiler that the tree is built
  with; the driver is those sources compiled by the boot compiler.
- **`ocamlobjinfo` prints the same of the driver's `.cmx` and of the
  build's** (step 6); the two files are not asked to be the same bytes,
  and the comparison is of that print and not of every value in the
  file. When this was measured (17 November
  2026, outside the tree) the two files were the same bytes for 24 of
  the 79 units; in the other 55 every value was the same and the
  driver's file had more of its blocks shared. Why the two compilers
  share differently was not looked for. What `ocamlobjinfo` prints of a
  `.cmx` has not got the list of its external symbols nor its flag
  `requires_metaprogramming`, so the comparison does not see a
  difference that is in those two alone; and it is a print, in which
  two values that print alike are taken for the same.
- **The object file that the driver writes is not compared.** At that
  measure it was the build's for all 79 once the debug sections were
  removed, and the one difference looked at (`std_exit`) was the version
  of the library that each compiler is linked with, which it writes in
  the debug information.

The command has the build's words, so it has not got
`-flambda2-expert-cmm-safe-subst`, which the compilations of this work
have elsewhere: nothing that is compiled here is run, and what is
written is compared with the build's. (That departure was ruled for this
directory by the orchestrator of this work, and can be overruled.)

The environment of the two runs is the build's where that shows in what
is written: `BUILD_PATH_PREFIX_MAP` gives the directory of the run the
name `/workspace_root`, as dune does for the build, so that the path of
a source in the terms is `/workspace_root/stdlib/build/X.ml` on any
machine; `OCAMLPARAM` and `OCAMLRUNPARAM` are not set.

## The 79 files

At this commit, by the stored files. (The counts and the table are made
from the stored files by a script that is not in the tree; they are of
this commit, and a commit that moves a stored file moves its row.)

- validated: 7;
- validated with its imports: 19, of which `stringLabels.ml` has a line
  under the verdict, "(asked of the term after with the exception
  handlers that no call can raise to made ordinary continuations)", and
  `set.ml` has two, "(with the calls of the closures that a Let makes
  made direct in the rounds after each lifting)" and "(with what
  my_closure is known to be from its code used in the rounds after each
  lifting)", which record what the liftings of the attempt that
  accepted the pair asked of their rounds (`../ocaml/README.md`, "What
  the verdicts mean", says what each is, under "validated");
  of the twenty-six that are validated, with or without their imports,
  five carry the mark of decision D65, a line under the verdict,
  "(decision D65: the guards of the liftings pass by the finer tests
  alone)": `camlinternalAtomic.ml`, `map.ml`, `semaphore.ml`, `set.ml`
  and `uchar.ml` ("What a verdict means for a file of the library"
  says what it is);
- compared, not validated: 4. Three have a line under the verdict:
  `option.ml`, `pqueue.ml` and `result.ml` have the line of a function
  slot named `partial_app_closure`, the name that Simplify gives the
  wrapper which it makes of a partial application;
- undefined in the model: none;
- not checked: 49. The first thing that the driver names is a primitive
  that the model has not got for 42 of them, a static constant that it
  has not got for 5 (in each, the block of an exception that the file
  declares), a call with no callee for `effect.ml`, and for
  `moreLabels.ml` a test that the tool makes of the code that the
  `.cmx` files of the other units give it.

The 79 sources have 29,925 lines; the twenty-six that are validated, with
or without their imports, have 4,151.

In the table, "what stops it" is the first thing that the driver names
of a unit that is not checked: there may be others behind it. For a unit
that is compared and not validated the cell is empty unless the driver
prints a line under the verdict. The line that the driver prints under
the verdict of `stringLabels.ml` is in the cell of its verdict. The
verdicts are in the words of `../ocaml/README.md`, "What the verdicts
mean".

| File | Unit | Lines | What the driver says | What stops it |
|---|---|---|---|---|
| `arg.ml` | `stdlib__Arg` | 410 | not checked | a static constant: `Unique_block` |
| `array.ml` | `stdlib__Array` | 540 | not checked | a primitive: `Array_set Naked_floats` |
| `arrayLabels.ml` | `stdlib__ArrayLabels` | 24 | validated with its imports |  |
| `atomic.ml` | `stdlib__Atomic` | 149 | not checked | a primitive: `Atomic_int_arith Field_index` |
| `backoff.ml` | `stdlib__Backoff` | 71 | not checked | a primitive: `Dls_get` |
| `bigarray.ml` | `stdlib__Bigarray` | 653 | not checked | a primitive: `Bigarray_length` |
| `bool.ml` | `stdlib__Bool` | 49 | validated with its imports |  |
| `buffer.ml` | `stdlib__Buffer` | 419 | not checked | a primitive: `String_length` |
| `bytes.ml` | `stdlib__Bytes` | 865 | not checked | a primitive: `Bytes_set` |
| `bytesLabels.ml` | `stdlib__BytesLabels` | 24 | validated with its imports |  |
| `callback.ml` | `stdlib__Callback` | 36 | not checked | a primitive: `Array_load Naked_floats` |
| `camlinternalAtomic.ml` | `camlinternalAtomic` | 62 | validated (decision D65: the guards of the liftings pass by the finer tests alone) |  |
| `camlinternalComprehension.ml` | `camlinternalComprehension` | 34 | validated |  |
| `camlinternalFormat.ml` | `camlinternalFormat` | 2997 | not checked | a primitive: `Bytes_set` |
| `camlinternalFormatBasics.ml` | `camlinternalFormatBasics` | 749 | validated |  |
| `camlinternalLazy.ml` | `camlinternalLazy` | 126 | not checked | a primitive: `Make_lazy` |
| `camlinternalMod.ml` | `camlinternalMod` | 94 | not checked | a primitive: `Make_lazy` |
| `camlinternalOO.ml` | `camlinternalOO` | 632 | not checked | a primitive: `Array_set Naked_floats` |
| `camlinternalQuote.ml` | `camlinternalQuote` | 3029 | not checked | a primitive: `String_length` |
| `char.ml` | `stdlib__Char` | 129 | not checked | a primitive: `Bytes_set` |
| `complex.ml` | `stdlib__Complex` | 83 | not checked | a primitive: `Make_block Naked_floats` |
| `condition.ml` | `stdlib__Condition` | 24 | validated with its imports |  |
| `digest.ml` | `stdlib__Digest` | 205 | not checked | a primitive: `Bytes_set` |
| `domain.ml` | `stdlib__Domain` | 666 | not checked | a primitive: `Atomic_compare_exchange Field_index` |
| `dynamic.ml` | `stdlib__Dynamic` | 32 | validated with its imports |  |
| `dynarray.ml` | `stdlib__Dynarray` | 1310 | not checked | a primitive: `Array_set Naked_floats` |
| `effect.ml` | `stdlib__Effect` | 571 | not checked | a call with no callee |
| `either.ml` | `stdlib__Either` | 86 | compared, not validated |  |
| `ephemeron.ml` | `stdlib__Ephemeron` | 1214 | not checked | a primitive: `Make_lazy` |
| `filename.ml` | `stdlib__Filename` | 397 | not checked | a primitive: `String_length` |
| `float.ml` | `stdlib__Float` | 631 | not checked | a primitive: `=.` |
| `format.ml` | `stdlib__Format` | 1613 | not checked | a primitive: `String_length` |
| `fun.ml` | `stdlib__Fun` | 43 | not checked | a static constant: `Unique_block` |
| `gc.ml` | `stdlib__Gc` | 243 | not checked | a primitive: `*.` |
| `hashtbl.ml` | `stdlib__Hashtbl` | 799 | not checked | a primitive: `Atomic_set Field_index` |
| `iarray.ml` | `stdlib__Iarray` | 369 | not checked | a primitive: `Array_set Naked_floats` |
| `in_channel.ml` | `stdlib__In_channel` | 210 | not checked | a primitive: `Bigarray_length` |
| `int.ml` | `stdlib__Int` | 64 | validated with its imports |  |
| `int32.ml` | `stdlib__Int32` | 119 | validated with its imports |  |
| `int64.ml` | `stdlib__Int64` | 116 | validated with its imports |  |
| `lazy.ml` | `stdlib__Lazy` | 84 | not checked | a primitive: `Make_lazy` |
| `lexing.ml` | `stdlib__Lexing` | 250 | not checked | a primitive: `String_length` |
| `list.ml` | `stdlib__List` | 618 | not checked | a primitive: `Array_set Values` |
| `listLabels.ml` | `stdlib__ListLabels` | 24 | validated with its imports |  |
| `map.ml` | `stdlib__Map` | 1027 | validated with its imports (decision D65: the guards of the liftings pass by the finer tests alone) (the second alias) |  |
| `marshal.ml` | `stdlib__Marshal` | 83 | not checked | a primitive: `String_length` |
| `modes.ml` | `stdlib__Modes` | 45 | validated |  |
| `moreLabels.ml` | `stdlib__MoreLabels` | 32 | not checked | with the prefix made from the `.cmx` files in front of it, the term before mentions a code id where it is not in scope |
| `mutex.ml` | `stdlib__Mutex` | 37 | validated with its imports |  |
| `nativeint.ml` | `stdlib__Nativeint` | 108 | validated with its imports |  |
| `obj.ml` | `stdlib__Obj` | 254 | not checked | a primitive: `Array_load Naked_floats` |
| `oo.ml` | `stdlib__Oo` | 23 | validated with its imports |  |
| `option.ml` | `stdlib__Option` | 47 | compared, not validated | the line under the verdict says that the term after has a function slot of the name that Simplify gives the wrapper of a partial application |
| `out_channel.ml` | `stdlib__Out_channel` | 79 | not checked | a primitive: `Bigarray_length` |
| `pair.ml` | `stdlib__Pair` | 36 | validated |  |
| `parsing.ml` | `stdlib__Parsing` | 216 | not checked | a primitive: `Array_load Naked_floats` |
| `pqueue.ml` | `stdlib__Pqueue` | 289 | compared, not validated | the line under the verdict says that the term after has a function slot of the name that Simplify gives the wrapper of a partial application |
| `printexc.ml` | `stdlib__Printexc` | 416 | not checked | a primitive: `Atomic_load Field_index` |
| `printf.ml` | `stdlib__Printf` | 52 | not checked | a primitive: `String_length` |
| `queue.ml` | `stdlib__Queue` | 177 | not checked | a static constant: `Unique_block` |
| `quote.ml` | `stdlib__Quote` | 71 | validated with its imports |  |
| `random.ml` | `stdlib__Random` | 387 | not checked | a primitive: `Bytes_set` |
| `repr.ml` | `stdlib__Repr` | 25 | validated with its imports |  |
| `result.ml` | `stdlib__Result` | 65 | compared, not validated | the line under the verdict says that the term after has a function slot of the name that Simplify gives the wrapper of a partial application |
| `scanf.ml` | `stdlib__Scanf` | 1528 | not checked | a primitive: `String_load` |
| `semaphore.ml` | `stdlib__Semaphore` | 89 | validated with its imports (decision D65: the guards of the liftings pass by the finer tests alone) |  |
| `seq.ml` | `stdlib__Seq` | 719 | not checked | a primitive: `Atomic_exchange Field_index` |
| `set.ml` | `stdlib__Set` | 1179 | validated with its imports (with the calls of the closures that a Let makes made direct in the rounds after each lifting) (with what my_closure is known to be from its code used in the rounds after each lifting) (decision D65: the guards of the liftings pass by the finer tests alone) (the second alias) |  |
| `stack.ml` | `stdlib__Stack` | 74 | not checked | a static constant: `Unique_block` |
| `stdLabels.ml` | `stdlib__StdLabels` | 30 | validated |  |
| `std_exit.ml` | `std_exit` | 22 | not checked | a primitive: `Atomic_load Field_index` |
| `stdlib.ml` | `stdlib` | 692 | not checked | a primitive: `Array_set Values` |
| `string.ml` | `stdlib__String` | 389 | not checked | a primitive: `Bytes_set` |
| `stringLabels.ml` | `stdlib__StringLabels` | 24 | validated with its imports (asked of the term after with the exception handlers that no call can raise to made ordinary continuations) |  |
| `sys.ml` | `stdlib__Sys` | 324 | not checked | a static constant: `Unique_block` |
| `type.ml` | `stdlib__Type` | 48 | not checked | a primitive: `Array_load Naked_floats` |
| `uchar.ml` | `stdlib__Uchar` | 112 | validated with its imports (decision D65: the guards of the liftings pass by the finer tests alone) |  |
| `unit.ml` | `stdlib__Unit` | 24 | validated |  |
| `weak.ml` | `stdlib__Weak` | 339 | not checked | a primitive: `Array_load Naked_floats` |

## What a verdict means for a file of the library

**A verdict here is about the file's initialisation in the model, run on
its own with its declared imports**, and with the prefix that the tool
makes from the `.cmx` files of the other units. The term that the
compiler makes of a file is a program that builds the module block of
the unit; for a file of the library that run is short: it allocates
closures and constants, and calls little. "Validated" says that the
validation accepts the pair of that term before and after Simplify, as
`../ocaml/README.md`, "What the verdicts mean", says of any unit: where
that run returns, refinement preserves what it exits with and its heap.
Of `stringLabels.ml` the pair that the validation accepts is of the term
before and of the term after with the exception handlers that no call
can raise to made ordinary continuations, as the validation makes them
in the term before; the driver asks that where it has refused the two
terms as they are, and says so under the verdict. What is proved is of
the term after as it was, with its handlers, and says nothing of
backtraces (the entry of that line in the same section).
`stringLabels.ml` has no `try`: the stored file shows three exception
handlers, and they are in the code of `contains`, `contains_from` and
`rcontains_from` of `String`, which the prefix has from the `.cmx` of
`String`; the build compiles the library with `-g`.

**A call of a C function does not stop a file.** The driver converts
such a call (decision D50; `../ocaml/README.md`, "What the conversion
does to a unit", "Calls of C functions"): a run of the model that
reaches one is undefined there. At this commit no stored file says that
a run is undefined at a call of a C function, and in the stored file of
each unit that the driver compares the run of the term before returns:
the initialisation of none of them, as the driver runs it, reaches such
a call. The stored file of a
unit that is compared says what is put in front of its two terms, if
anything is, in one or two blocks before the lines of the runs:

- "in front of the two terms, the declarations of N C functions:" (or
  "the declaration of 1 C function:"): the C functions that one of the
  two terms, or code that the prefix binds, calls, a line for each, with
  its name, "(a C function)" or "(a C function: noalloc)" in the stored
  files of this commit, and "declared with no code". 15 stored files
  have that block. In 4 of them (`bool.own_expected`,
  `condition.own_expected`, `int.own_expected` and `mutex.own_expected`)
  nothing else is in front of the terms: the lines of the two runs say
  "after the declarations of the C functions", and the line of the check
  of static names "each with the declarations of the C functions in
  front of it".
- "given the .cmx of ..., a prefix of N static bindings:", the prefix
  made from the `.cmx` files of the units named. 19 stored files have
  it, 11 of them after the block of the declarations; the lines of their
  two runs say "after the prefix".

The C functions that the two terms of a unit call are among its imports,
so the verdict of such a file, where the driver validates it, is that of
a unit that has imports: "validated with its imports", or, where there
is a prefix, "validated with its imports, given the .cmx of N of them":
the two terms are then the two units each with the prefix made from
those N files in front of it (`../ocaml/README.md`, "What the verdicts
mean", says so of the line "related once lifted", which can have the
same head). `rocq/TRUST-LEDGER.md`, "Calls of C functions", says what is
kept of a call of a C function and what a verdict of a function that
calls one rests on.

**The mark of decision D65.** Under the verdict of five files,
`camlinternalAtomic.ml`, `map.ml`, `semaphore.ml`, `set.ml` and
`uchar.ml`, the driver
prints "(decision D65: the guards of the liftings pass by the finer tests
alone)". A lifting of the validation asks two tests of the term that it
is given, of its exits and of its primitives of the heap. By decision D65
(yes, provisionally, by the human on 18 November 2026; to be come back
to) each test also passes a term that has no call outside the bodies of
its code and that passes with those bodies left out; the line says that
the term before, as the validation was given it, passes the finer tests
and not the tests as they were (`../ocaml/README.md`, "What the verdicts
mean", has the item of the line; `rocq/TRUST-LEDGER.md` has "The mark of
decision D65"). The functions of such a file may raise a variable or
store in cells, and the run of its initialisation enters none of them.
What the finer tests give away is said at their definitions
(`FlambdaLiftSpec.exits_plain_fine`,
`FlambdaHeapFreeSpec.free_of_heap_fine`): the statements of the liftings
are of that run, and say nothing of the bodies of the functions that it
does not enter, as "What it is not", below, says of any verdict here.
The mark is a sign: it is neither necessary nor sufficient for the
acceptance to rest on the finer tests. Of `camlinternalAtomic.ml`,
`semaphore.ml` and `uchar.ml` it is known besides, from the stored files
of the commit before the tool was given the finer tests, that each was
"compared, not validated", with the line "related once lifted" under
the verdict, and gained its verdict with them. `map.ml` and `set.ml`
have had the mark since the commits at which they were first validated:
the stored file of each, at the commit before that one, says "not
validated", with no line under the verdict, and the stored files do not
show whether the tests as they were would have been enough for either.

What it is not:

- **No theorem about the fields of the module block.** What the run exits
  with is the symbol that names the block. That the two runs return the
  same symbol is not a statement that the fields of the block, or the
  functions that the unit exports through them, behave the same.
- **No theorem about later calls of the functions of the file.** The
  validation compares the code of those functions, after rewrites whose
  guarantees have not all the same scope, and what it says of a function
  is what `../ocaml/README.md`, "What the validation of a unit that
  exports a function rests on", says, and no more. A verdict here is not
  a statement about a function of the library under every caller.
- **No theorem of linking.** "Validated with its imports" is about the
  imported values that the model declares for the unit, into which the
  model does not look, and about the prefix: the code that the `.cmx`
  files of the other units hold, put in front of both terms (the `.cmx`
  files of the library do not hold the body of every function). It is
  not a theorem over every implementation of the other units, nor about
  the program that links them.

"Not checked" says that the term has something that the model has not
got, a primitive first among them, or, for
`moreLabels.ml`, that a test which the tool makes before it checks
refuses the unit (the table says which). It is not a failure of the
compiler, and it says nothing of the rest of the file. "Compared, not
validated" says that the two runs came to the same result in the model
and that the validation does not accept the pair: no theorem of
refinement comes from this validation, and it is no failure of the
compiler either. A weaker verdict, which is proved apart, may be printed
under it, "related once lifted" (`../ocaml/README.md` says what that is
a statement of); no stored file has it at this commit. `option.ml`,
`pqueue.ml` and `result.ml` have under it a line in round brackets,
which is no verdict: the term after has a function slot of the name
that Simplify gives the wrapper which it makes of a partial
application, the term before has none of that name, and no rewrite of
the validation makes such a wrapper (the same section of
`../ocaml/README.md`). The tool goes by the name alone, and the line
does not say that nothing else stops the unit. "Undefined in the model"
says that the run of the term before is undefined in the model, so that
there is nothing to validate.

The compilation is the build's: with `-g`, with the flags of the
configuration (`ocamlopt_stdlib_flags.txt`), against the `.cmi` and
`.cmx` files of this build. Another configuration can give another term.
58 of the 79 sources carry `[@@@ocaml.flambda_o3]`, and are simplified
as with `-O3` whatever the flags are; the others are not checked here
with `-O3`.

What `../ocaml/README.md`, "What a validation does not say", says holds
here: the kinds below the kind of values, regions, what is trusted in the
conversion (`rocq/TRUST-LEDGER.md`); and the head comment of the driver
says what is trusted in the driver.

## What is not here

- The other way of checking a unit, by the text of its term and the test
  tool (`../ocaml/README.md`): no text of a file of the library is in the
  tree.
- `-O3` for the files that do not ask for it; any flags but the build's.
- The library as a whole program, and `otherlibs/`.
- The interface of a unit: the rules read the `.cmi` that the build made,
  and do not make it again.

## What a stored file holds, and what moves it

A stored file is the whole of what the driver prints of the unit: for
most units what it says of a unit that is not checked, with the
primitive that stops the unit, which is on one line in every stored
file of this commit (the driver prints a primitive with a margin of a
million columns, and a name that has a newline in it with its
newline), or the static constant, which takes several lines; for a
unit inside the model, what is put in front of its two terms (the
declarations of the C functions that are called, the prefix), the two
runs, the checks and the verdict. Its
first line begins with the name that the build gives the unit, and `.fl`
(`stdlib__List.fl:`). It is stored byte for byte: at this commit 28
lines of 21 of the files end with a space, as
the driver prints them (a line whose value begins on the next line; in
`modes.own_expected`, a block of empty blocks cut at a field), and
`git diff --check` reports those lines.

A stored file moves with a change of `stdlib/X.ml` or of a unit that X
uses; with a change of the compiler that gives Simplify another term or
makes it return another; with a change of the model, of the conversion or
of the validation; with a change of the build's flags or of the
configuration. **It can move when no verdict does**: a line "not
checked" prints a static constant with the numbers of its names
(`Pccall/1510N`), which move when a change before them makes one name
more or fewer (it prints a primitive with the names of its variables
alone, which do not), and 26 of the files hold a place in the source
(`/workspace_root/stdlib/build/sys.ml.in:225,0--15`), which moves with
the lines of the source. A file that is validated with its imports names
the code of the other unit by its numbers.

`stdlib/sys.ml` is not a file of the repository: `configure` writes it
from `stdlib/sys.ml.in`, with the version of the compiler in it. What the
driver prints of it must not hold what `configure` writes, or the stored
file would move with the version. It does not at this commit (the
stored file shows a static constant and a place in `sys.ml.in`), and a
rule of this
directory fails if `sys.own_expected` has the first line of `VERSION`.

The driver must exit with the status 0 of every unit that
`driver_status.txt` does not name; no unit is named there at this
commit. A unit whose check did not end in 30 minutes of processor time
would be in neither alias and would have no stored file (`not_run.txt`);
there is none at this commit.

## What it costs

No rule has a limit of time: a stored verdict must not depend on the
machine. A unit is in the first alias if its rule takes at most 10
seconds of processor time, and the first alias as a whole is to stay
under 2 minutes; the others are in the second alias, by `long.txt`; a
unit whose check did not end in 30 minutes would be in neither
(`not_run.txt`). That bound says which units are run, and no more: a
time is no verdict, and a unit is not left out of the two aliases,
and does not lose its stored file, because its time has passed 10
minutes while it stays under 30. Whether it is in the first alias or
in the second is still by the 10 seconds above. Which unit is where
is in those lists, and is changed by a commit, not decided when the
rules run.

Measured on 18 November 2026 in a tree built at f5591bb185, one job, on a
machine that other work shared (so the times place the rules, and are no
bound):

- **The first alias**, 77 units: about 75 seconds of processor time.
  Each rule runs the compiler that built the library to its end, then
  the driver, then `ocamlobjinfo` twice. The slowest are
  `camlinternalFormat.ml` (7.0 seconds), `pqueue.ml` (6.8),
  `camlinternalQuote.ml` (5.5), `stringLabels.ml` (4.3) and `result.ml`
  (3.1); 58 of the 77 take less than one second.
- **The second alias**, `map.ml` and `set.ml`: about 9 minutes 40
  seconds of processor time (about 5 minutes for `map.ml` and 4 minutes
  45 seconds for `set.ml`), nearly all of it in the driver's check.

Measured again on 18 November 2026 with the compiler, the driver and the
library of a build of the files of the commit "C0: the ledger has the two
lines of an acceptance of an undefined run" (the build was made at an
earlier placing of the same files in the line of the stage), one job, on
a machine that other work shared, each alias asked of dune once, as a
whole. The first alias
(`runtest-stdlib`, 77 units) took 68 seconds of processor time, user and
system, and 68 seconds of the clock. The second (`runtest-stdlib-long`:
the rules of `map.ml` and `set.ml`, one after the other) took 823
seconds of processor time, about 13 minutes 45 seconds, and 829 seconds
of the clock. The two units were not measured apart this time, and no
unit of the first alias alone: the times of single units above, and the
times beside the two names in `long.txt` (298 and 284 seconds), are
those of the measure of 18 November 2026 in the tree built at
f5591bb185. `long.txt` stays as it was; only its names are read.

Since the driver converts the calls of C functions (decision D50), ten
more files are compared where it stopped at a call before. The two
aliases were not measured again with that change. P1 measured the 79
files on another route on 19 November 2026 (the same files compiled with
the driver outside the rules of this directory, at the build's flags,
one process at a time): 829 seconds of processor time with the driver of
the stage before and 837 with the driver of this change. `map.ml` and
`set.ml` are counted in those sums by a second run of each alone, as
each had ended at the limit of 120 seconds of the run of the 79; the
other 77 files took about 25 seconds before and about 34 with the
change, and of the ten files the slowest took 4.3 seconds there
(`nativeint.ml`).

The rules depend on the driver, which links the whole compiler: any
change of the compiler runs all 79 again. A change of `stdlib/X.ml` runs
the rule of X and of every unit whose build's rule names a file of X.
