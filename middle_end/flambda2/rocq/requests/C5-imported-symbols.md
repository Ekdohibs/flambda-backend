# C5: symbols that a unit imports, and exceptions told apart by name

From C5, for C4 (identity and the heap), P1 (`Phys_equal`), C0
(well-formedness and the interface of a unit), V2 (the tool and the
corpus), V3 and the owners of the validator's passes (the fallout),
R1 (who reviews it) and the human (two decisions, at the end).
Section 6 says which of V2's units need what.

It is a request with a prototype, which comes with it. The prototype
is whole: every `.v` file builds, `make -C rocq all` and
`make -C rocq assumptions` pass. It changes files of other streams,
each change small; the list is in section 4, and the owner of each
file has agreed to the change to it.

This is the second version. The first made every import an object
with an identity, and so answered false of any import against any
allocated value of the unit. R1 found that this is not justified for
an import that is immutable data, which Simplify may share
(discrepancy R1-24). An import now has a class, and only an import
that is an identity is told apart from what the unit allocates.

Paths are in `middle_end/flambda2/rocq/` unless they say otherwise;
compiler files are from the root of the repository.

## 1. What stops

A handler that catches an exception by name is undefined in the model
(discrepancy V2-4). From `try ... with Not_found -> 2` the front end
makes

    where k exn (exn : val) =
      let prim = %phys_eq (exn, $`*predef*`.caml_exn_Not_found) in ...

and the model has two holes there: the symbol is bound by no unit, so
every operation on it is undefined (D8), and `Phys_equal` has no value
on two values that are not tagged immediates (D19). The same symbols
are in every division by a divisor that is not known, every checked
access and every match that is not exhaustive, each of which raises a
predefined exception. Of V2's units made from OCaml sources, none with
a handler that names an exception can be run to its end.

## 2. What the compiler has

- A symbol carries its compilation unit (`Symbol.compilation_unit`).
  The predefined exceptions are symbols of a unit of their own
  (`middle_end/flambda2/identifiers/symbol.ml`,
  `is_predefined_exception`), made by closure conversion
  (`from_lambda/closure_conversion.ml`, at `Pgetpredef`).
- A symbol of the unit being compiled must be defined before it is
  used; one of another unit is looked up in that unit's `.cmx`, and is
  of unknown type if there is none (`types/env/typing_env.ml`,
  `find_with_binding_time_and_mode'`).
- The object behind a predefined exception is made by the startup
  code: a block of tag 248 with two fields, the name and a negative
  number (`backend/cmm_helpers.ml`, `predef_exception`).
- A handler that matches exceptions compares, by `Phys_equal`, the
  exception caught with the symbol, for a constant exception; and the
  first field of the exception caught with the symbol, for an
  exception with arguments (`lambda/matching.ml`,
  `combine_extension_constructor`). An exception with arguments is a
  block of tag 0 whose first field is the exception.
- Simplify folds `Phys_equal` of two symbols to whether they are one
  symbol, and of a symbol against a constant to false
  (`types/provers.ml`, `prove_physical_equality`, its last match).
- An exception that a unit declares is another matter: the front end
  calls `caml_fresh_oo_id`, an external function, and makes a block of
  tag 248 that is `Immutable_unique`. That needs the heap and a model
  of the call, and is not in this request.
- `exception E = Not_found` makes no object and no symbol: `E` is
  another name for the object of `Not_found`. So two symbols of
  predefined exceptions are two objects.
- For a symbol of another unit that is immutable data, Simplify may
  know the contents from the other unit's `.cmx`, and the compiler
  shares equal immutable values (R2-21). Nothing here establishes that
  a block of this unit is never made the same object as such a
  symbol.

## 3. The design

**An imported symbol is bound, to something opaque, of a class.** A
new form of what a symbol points to, `Static_imported c`
(`FlambdaValueSpec.statically_allocated_value`), where `c` is an
`import_class`:

- `Import_identity`: an object with an identity, which is its symbol.
  It is for an exception: a predefined one, or one that another unit
  declares. Such an object is made once (by the runtime, or by the
  unit that declares it) and is never copied or shared.
- `Import_data`: anything else of another unit. The compiler may know
  its contents; the model does not.

`imported_of statics v` reads a value as an import (the answer is the
symbol and the class), `identity_of statics v` as an identity (the
answer is the symbol). None of the four observations reads an import
of either class, so it is not allocated in the sense of
`is_allocated`, and every law about allocated values stands as it is.

**`Phys_equal`** (`FlambdaPrimitiveSpec.eval_phys_equal`):

- an import of either class against a tagged immediate: false (a
  symbol is a pointer);
- two identities: whether they are one symbol;
- an identity against an allocated value of the unit (a block, a
  closure, a boxed number, a string, or a symbol bound to one), and
  against an import that is data: false. An identity is generative
  and unique, so nothing that the unit allocates, and nothing that
  Simplify shares or lifts, can be it; R2-21 is about immutable
  structure, which an identity is not;
- an import that is data against an allocated value, another import
  that is data, or itself: no value. It is as an immutable allocated
  value is today (D19);
- an identity against null, poison, a number that is not a value or a
  symbol that is not bound: no value, as a tagged immediate against
  them has none (D19);
- an import of either class against a location of the heap: false
  (`FlambdaHeapSpec.eval_phys_equal_location`). A
  location names a cell that this run allocated, and an import was
  there before the run.

The case of an identity against an allocated value is needed, and not
only convenient. The handler `with Not_found -> ...` first tests
`%phys_eq (exn, Not_found)`, and what it caught may be `Failure "x"`,
a block of tag 0. Without the case, most programs that catch a
constant exception would be undefined.

Two immutable allocated values stay without a value (D19):
nothing of that changes.

**Which symbols are identities** is said by whoever makes the static
environment of a unit: the conversion from the compiler's terms,
which is OCaml that the proofs trust. It is to give the class to the
predefined exceptions only, which it can tell from the symbol alone
(`Symbol.is_predefined_exception`), and `Import_data` to every other
symbol of another unit. An exception that another unit declares is
then data until something can check that it is one, and the
comparisons on it have no value, which is safe.

**Nothing else reads an import.** Of the unary and binary primitives
only `Opaque_identity`, which returns it, and `Phys_equal` have a
value on an imported symbol of either class
(`FlambdaImportsSpec.imported_not_read_statement`, proved). It can be
passed, returned, raised and stored in a block.

**A unit starts with its imports bound.** `imports_statics imports`
(`FlambdaImportsSpec.v`) is the static environment that binds each
symbol of the list to `Static_imported` of the class given with it. A
statement about a unit run on its own is then about that environment
and not the empty one: `unit_refines imports source target`. Proposed
to C0: a field `unit_imports : list (symbol * import_class)` of
`unit_interface`, the model's account of "the symbols whose
compilation unit is not this one"; the prototype passes the list
explicitly and does not touch the record.

**Well-formedness** (C0's rows 2 and 4, D8). Row 4 becomes: every
symbol mentioned is bound by the unit, by the rule of scope that the
row has, or is one of `unit_imports`. Row 2: no static binding of the
unit binds a symbol of `unit_imports`. The machine already makes the
second undefined (a static binding of a name that is bound is
undefined, `eval_static_bindings`).

**The law about related values** (`eval_prim_related_statement`). Its
premise `relates_observations` gains one part: if the first value is
an imported symbol, the second is the same imported symbol, of the
same class. The answer of `Phys_equal` on an import depends on its
class, and on an identity on which symbol it is, so the law is false
without it. The law still goes one way only.

**The relations of the heap** (C4). `states_related` asks that the
two states import the same symbols, each as the same class (both
directions: that relation is the one without poison, which is kept
symmetric). `names_related` asks the same of two memories; a symbol
that is left out is allocated in the second memory, so it is imported
in neither. `outcome_related` has no such clause: at an exit, an
import of the first run may still be related to the same symbol left
unbound by the second. So `outcome_related` alone is not a congruence
for what consumes an outcome: `Phys_equal` of an identity `s` with
itself is defined, and of `s` unbound it is not. What is owed before
the validator's theorem, or any use of outcomes in a context, goes
over to imports is a checked condition that the two units start from
the same imports, each of the same class, and the invariant that a
run keeps it (section 4). R2, who reviews the statements of the heap,
has read the two definitions.

**A unit's own exception** (the block of tag 248 marked
`Immutable_unique`) is an identity through C4's cells: two of them
are compared by location, and one against an identity that is
imported is false, a location against an import. Running that needs
the heap, which is in `state` since this was written, and what
section 6 says besides.

Statements proved in the prototype, in `FlambdaImportsSpec.v`:
`phys_equal_identities_statement`,
`phys_equal_imported_immediate_statement`,
`phys_equal_identity_other_statement`,
`phys_equal_data_import_statement`, `imported_not_read_statement`,
`expr_refines_unit_refines_statement`, and two examples after V2's
units, run on the machine from the imports:
`catches_by_name_statement` (the handler catches `Not_found`, and lets
`Division_by_zero` and a `Failure` that the unit bound statically go
by to the exception continuation of the unit) and
`catches_with_argument_statement` (the handler catches that `Failure`
and reads its argument).

## 4. The fallout, as measured on the prototype

Twenty-four files besides the two new ones. In fifteen the change is
a case for the new constructor, a pattern with one more branch or one
more line: `FlambdaClosuresSpec.v`, `FlambdaWellKindedSpec.v`,
`FlambdaUnusedStaticsSpec.v`, `FlambdaStaticsInScopeSpec.v`,
`FlambdaHoistCodeSpec.v` and `FlambdaKnownSlotsSpec.v` with their
proofs, and the proofs of `FlambdaSemantics`, `FlambdaMyClosure` and
`FlambdaKnownSymbols`. In each of those definitions an import is what
a string is: it holds no value, no code and no value slot. What has
content:

- `FlambdaValueSpec.v`: `import_class`, the constructor,
  `imported_of`, `is_imported` and `identity_of`.
- P1: `FlambdaPrimitiveSpec.v`, the definition of `eval_phys_equal`,
  its comment, a CR, and the premise of the law;
  `FlambdaPrimitiveProofs.v`, four proofs about `Phys_equal` redone
  (two of them now by one lemma, `eval_phys_equal_transfer`) and
  small lemmas about `imported_of` and `identity_of`.
  `FlambdaIdentitiesProofs.v`: one proof (a value is the same as
  itself).
- C4: `FlambdaHeapSpec.v`, one disjunct in
  `eval_phys_equal_location`; `FlambdaHeapMachineSpec.v`, one clause
  in each of `names_related` and `states_related`: the two memories
  import the same symbols, each as the same class.
  `FlambdaHeapProofs.v` and `FlambdaHeapMachineProofs.v`: the
  new part of the premise of the law, four places where a relation is
  built, and cases. C4's relation already had the form for a symbol
  of another unit (`Step_opaque_symbol`), which is kept as it is.
- V3: `FlambdaLiftProofs.v`, a constructor of `allocated_lifted` (an
  import is lifted to itself) and the new part of the premise.
  `FlambdaUnusedSlotsProofs.v`: the same part, and a lemma.
- The types: `FlambdaProversProofs.v`, the proof that `Phys_equal`
  answers whether two values are the same, with the new cases. No
  statement of the types changes.
- The test tool: one case where it prints what a symbol points to.

Not done in the prototype, and owed before this is a change:

- C0: the field, and rows 2 and 4.
- The test tool makes the list of the external symbols that the two
  units of a check mention (those of another compilation unit, and
  the predefined exceptions), and runs both from `imports_statics` of
  it. Today it runs every unit from the empty static environment, so
  an imported symbol is still bound to nothing there. Until the
  validator's theorem is from the imports, what the tool says of the
  validation of a unit that has imports is to be labelled as of the
  unit run with nothing imported.
- The validator. Its theorem for units is from the empty static
  environment (`FlambdaUnusedStaticsSpec.program_refines`). With
  imports it is to be `unit_refines imports`. The passes proved under
  all environments carry over as they are
  (`expr_refines_unit_refines_statement`), and so does the removal of
  unused static bindings, which is proved from any environments that
  are free of the names removed. Those proved from the empty
  environment only (the liftings, known symbols, known and unused
  value slots, the hoisting of code and of static bindings) need
  their statements made from an environment that binds imports: none
  removes, lifts or reads an import, so the invariants they keep hold
  of it by the cases above, but the statements are their owners'.
- The machine with a stack of handlers. `stack_agrees_statement`
  (`FlambdaTrapSpec.v`) starts both machines from the empty static
  environment; for a unit with imports it is to be stated from
  `imports_statics`. Mine to do.
- Outcomes. A checked condition that two units compared start from
  the same imports, each of the same class, and the invariant that a
  run keeps it, before `outcome_related` is used of units with
  imports (section 3).
- N1 and N3: an imported symbol has no name of a variable or a
  continuation in it, and symbols have no renaming action; nothing is
  needed. The equivariance files build unchanged.

## 5. What it leaves undefined

**A load from a constant exception.** A handler tests its constant
cases first: `combine_extension_constructor` puts the tests of all the
constant cases around the code for the cases with arguments. If none
of them is what was caught, and the handler has a case for an
exception with arguments (or one for any exception, after such a
case), it loads the first field of what it caught. When that is a
constant predefined exception the load is from an imported symbol,
which has no value: so

    try raise Not_found with Failure _ -> 2

is undefined in the model, and the compiled code reads the name of
`Not_found` there, finds it is not `Failure`, and raises it again. A
handler `Failure _ -> 2 | Not_found -> 3` is not this case: it tests
`Not_found` first and loads nothing from it. The last part of
`catches_with_argument_statement` is the case, with a CR.

**An import that is data**, against an allocated value, another
import that is data, or itself; and every load from it. (It has a
value against a tagged immediate, against an identity and, in the
machine with a heap, against a location: false in each.) A unit that
reads a value of another unit's module loads a field of that unit's
symbol, and that has no value.

The second step, which gives the first a meaning: an identity has
contents. C4's proposal, which I take: a predefined exception is
bound, in the static environment and not in the heap (it is never
written and is there before the unit runs), to a value with an
identity and contents: tag 248, the name and the number. `block_of`
does not read it, so that it stays outside `is_allocated` and two of
them are never related as equal blocks; a load that is not marked
`Mutable` reads it, and so do `Get_tag` and `Is_int`; nothing writes
it; it is compared by its symbol. `Static_imported Import_identity`
is that object without its contents. The relation between a location
of one run and such a symbol of the other, which is what validates a
unit that declares its own exception (a cell before Simplify, a
static block with an identity after), is C4's (`requests/C4-heap.md`,
section 2.7). This request is the part that needs no heap. An
exception that the unit declares needs the heap, which is in `state`
since this was written, and the call of `caml_fresh_oo_id` besides.

## 6. The units, and which come first

When this was written V2 had staged the units made from OCaml sources
that raise or catch. I made the term of each with `from_ocaml.sh` and
read it. There were seventeen with a `try`, a `raise`, a division or
a checked access (the corpus has more since; the list is of those).
**None declares an exception of its own**: all need the predefined
exceptions only, so all are in the first group, which is this
request.

- One mentions no symbol of another unit: `try_no_raise`.
- Twelve raise a predefined exception, or pass it on, and never look
  at it: `div_unknown`, `mod_zero`, `raise_in_function`,
  `raise_inlined`, `raise_notrace`, `raise_unused_exn`,
  `try_catch_all`, `try_div_zero`, `try_nested`, `try_reraise`,
  `try_result_unused`, `uncaught_raise`. The machine runs them today,
  the symbol being bound to nothing; they are refused by row 4 of
  well-formedness (D8) until the symbol is an import.
- Two more put the symbol in a static block of the unit and never
  look at it: `raise_unused_bucket`, `string_get_unknown`.
- Two compare by name, and are undefined today: `try_not_found` (the
  exception caught against the symbol) and `exn_with_argument` (the
  first field of a static block of the unit against the symbol, then
  a load of the argument). Both are defined with this request, and
  the two examples of `FlambdaImportsSpec.v` are after them.

Every symbol of another unit in the seventeen is a predefined
exception, so every import of theirs is an identity. None of the
seventeen is a case of section 5. (One of C4's sources for arrays
raises `Invalid_argument` at a failed bounds check, with the symbol
as a field of a static block: the same as the two above.)

**The second group: a unit that declares an exception**
(`exception E` then `try raise E with E -> ...`). The front end
makes, for each declaration,

    apply ccall noalloc ($`*extern*`.caml_fresh_oo_id : val -> val) (0)
      -> k * error
      where k (id) =
        let E = %block.immut_uniq.[`248`] ($name_string, id) in ...

and the handler compares the exception caught, or its first field,
with the variable `E` by `Phys_equal`. Besides this request it needs:

- C4's cells with an identity (the heap is in `state` since this was
  written; what follows is still to be run): the block is
  `Immutable_unique`, so it is a location, and `Phys_equal` of two
  locations is by their names. A location against an imported symbol
  is false (in the prototype here). Where Simplify lifts the block to
  a symbol, C4's static cells ("Later: symbols as addresses").
- Of P1's work on calls of C functions, for `caml_fresh_oo_id`: only
  that the call returns, to its return continuation, a value of kind
  `Value` that can be stored in the block. The code that the front
  end makes to declare, raise and match an exception never reads that
  field: it compares the address of the block. (For an exception with
  arguments it loads field 0 of the exception caught, and that field
  is the block of tag 248; then field 1, the argument. It never loads
  a field of the block of tag 248.) So the answer is only ever stored and passed
  on, never computed with, and the model should not know it as a
  number: if the call returned a tagged immediate that the model
  knows, a program that read the field would be given an answer that
  is not the machine's. The function is not pure (each call answers a
  new number), so it is not in P1's table of pure functions; P1's
  request for calls of C functions says what form the answer has (the
  n-th fresh number of the run, compared by n and with no value under
  any other primitive), agreed with this one. It needs the machine to
  count the calls, which is state, and goes with the heap.
- The symbol `caml_fresh_oo_id` itself is of the unit `*extern*`: it
  is the callee of a C call and not a value that the term handles, so
  it is not an import in the sense of this request.

## 7. Options not taken

- **An unbound symbol is an import.** No new form; `Phys_equal`
  answers on symbols that are bound to nothing. It breaks D8 (an
  operation on a symbol that is not bound is undefined), and the
  model could no longer tell an import from a symbol used out of its
  scope.
- **A declaration in the term**, as a static binding of a new sort
  put in front of the unit. No statement about units changes; but it
  is a form of term that the compiler has not got, and every pass
  that goes by cases on static bindings gets a case.
- **Contents at once**: the import as a static block with a mark of
  identity. It is the second step above, and it puts loads from
  blocks with an identity into the machine without a heap, which is
  C4's to design.
- **Identity by tag**: a static block of tag 248 has an identity. No
  new form at all; but the identity of a block is its mutability in
  the compiler (`Immutable_unique`), not its tag.
- **Every import an identity**: the first version of this request. It
  answers false of an import that is immutable data against a block
  of the unit, which nothing justifies (discrepancy R1-24).
- **No class, and no value for an import against an allocated
  value**: the smallest repair of the first version. It makes
  `with Not_found -> ...` undefined whenever what is caught is an
  exception with arguments.

## 8. Decisions for the human

1. **Is "an import is opaque" an acceptable first step**, given what
   section 5 leaves undefined? Recommended: yes, as it unblocks the
   handlers that catch a constant exception, which V2's corpus has,
   and is replaced, not undone, by the second step. Taken overnight.
2. **An import has a class, and `Phys_equal` of an identity against
   an allocated value of the unit, or against an import that is data,
   is false.** The compiler does not prove this in general (it folds
   two different symbols to false, and otherwise answers by the shape
   of the two types). It is true of the machine code. The reason
   given for the model is that an identity is generative and unique.
   It rests on the conversion giving the class to exceptions only,
   and on two symbols of that class being two objects. For an import
   that is data the comparison has no value. Taken overnight, after
   discrepancy R1-24; the CR is at `eval_phys_equal`.
3. **Which symbols are identities.** The predefined exceptions only,
   decided by the conversion from the symbol alone. An exception of
   another unit could be one, but the conversion does not read that
   unit's `.cmx` and cannot tell. Taken overnight.
