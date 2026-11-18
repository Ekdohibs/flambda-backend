# The semantic decisions of the model

What the Rocq model of Flambda 2 says a program means, wherever that was
a choice. `DECISIONS.md` lists the labels and reasons (D51 is unused);
each definition's review comment has the detail. A change to the tree
that adds a decision or changes one changes this file with it.

**Undefined** means that the run has no meaning in the model: refinement
gives no guarantee for it. That is cautious about results, not about how
much of the compiler's correctness is shown (D22). A line is **ruled**
(by the human; the date, or "by" a date where only the day it was
written down is known), **provisional** (the model's choice, the human's
to overrule) or **open** (the human's to decide; the model meanwhile
does what the line says). Dates are of 2026.

## 1. Values and kinds

- **D10, D16.** A value of the wrong kind is undefined where it is
  executed: as an argument of a call or of a primitive (ruled by 11
  Nov); where a `let` binds it or a handler is applied to it
  (provisional). Not tested: a field of a static block or a value slot
  where it is stored, a value given to a way out of the term (open).
- **D25.** Subkinds. A run is undefined where a parameter of a
  continuation or of code is bound to a value that has not the shape
  that its subkind states (ruled 18 Nov; not yet in the tree). A `let`
  binds with no subkind test. What the subkind of a field of a made
  block promises is open: meanwhile nothing; a field is any value.
- **D8.** An operation on a symbol bound to nothing is undefined (ruled
  by 11 Nov). `Is_null` and `Is_int` have no value on poison (open).
- **D5.** `Get_tag` has a value on blocks only (ruled by 11 Nov); a
  variant's `Is_int` on tagged immediates and blocks, not null (open).
- **D15.** A `Block_load` whose access kind disagrees with the block
  (its tag, its size, an `Immediate` field) is undefined (open).
- **D21.** A boxed number whose contents are poison is a boxed number;
  boxing and unboxing poison are undefined (open).
- **D38.** `Boolean_not` is undefined on all but tagged 0 and 1 (provisional).
- **D13, D24.** A well-formed term has its integer constants in range
  for their kinds. A block may have no fields (both ruled by 11 Nov).
- **D23.** The machine makes a block of any integer tag, and `Get_tag`
  gives it back. A well-formed term has tags in [0, 251); the meaning of
  a type relates a block to its tag only from 0 to 255 (provisional).
- **D41.** An import of which nothing is known has no value wherever it
  is inspected, `Phys_equal` against itself included (provisional).

## 2. Numbers

- **D17.** The target has 64 bits (open).
- **D14.** Division and modulo by zero are undefined; the least integer
  divided by -1 wraps (provisional: taken from the compiler).
- **D18.** A shift by an amount outside [0, bound) is undefined: the
  bound is the width of the kind, and 64 for tagged immediates (open).
- **D66.** The bit pattern of a NaN is not guaranteed to be preserved
  (ruled 18 Nov; not yet in the tree, where the floats of two exits are
  compared by their bits). Whether a run that observes those bits is
  undefined, or exits are compared up to which NaN it is, is open.
  Arithmetic on floats is to be built (approved 18 Nov).

## 3. Closures and calls

- **D1.** An exit observes a structural closure whole: its function
  slot, the code ids of its set, the values of its value slots; of a
  symbol it observes the symbol alone (open).
- **D2, D19.** An immutable block, boxed number, closure or array made
  as a structural value has no identity: equal ones are the same value,
  and `Phys_equal` of two is undefined. Cells, unique blocks among them,
  have locations, which `Phys_equal` compares. A symbol is its name
  (D57). Null and poison give no answer (open; goes with the heap).
- **D72.** A predefined exception is an identity: two are equal exactly
  when they are one symbol, and one is never equal to a value that the
  unit allocates (provisional).
- **D3.** A projection is undefined where the function slot that it
  projects from is not the closure's own (ruled by 11 Nov, "as probably
  right"), or the value slot is not in the closure (provisional; open).
- **D4, D28, D31.** Conditions of a well-formed term: one group binds a
  static set of closures whole, each symbol at its slot (open); the
  functions of a set have each their own code id, and the static sets of
  one group none in common; code is closed but for symbols, code ids and
  slots (the last two ruled by 11 Nov).
- **D20.** Undefined: making a closure at a function slot not of its
  set; a call of a closure whose code is not bound or is deleted (open).
- **D9.** A full call whose arities are not those of the code called is
  undefined (ruled by 11 Nov).
- **D58.** A call of known arity is defined only where it is a full
  call, of one group of arguments at least, of code that is bound and
  not tupled. A call of unknown arity is the generic application, and so
  is a direct call, which may be partial or over-applied, gives tupled
  code a tuple, and must name the code of the closure called
  (provisional; open until the developers answer).
- **D59.** The code ids that a call of known arity lists in the model's
  term are a promise: the call is undefined unless the callee's code is
  one of them. The compiler also allows newer versions of that code, a
  relation that the model has not got; the test tool extends the list as
  far as the metadata that it has and the scope of the call allow, and
  that extension is trusted (provisional; open as D58).

## 4. Blocks and the heap

- **D40.** Two exits are the same when their values and their heaps
  are equal, locations included: garbage and the order of allocation
  are seen. Intended: up to a correspondence of locations (open).
- **D57.** An exit carries no static environment: a symbol is a value,
  bound or not. Two tests of the unit, not refinement, say that a symbol
  at an exit is bound; what it is bound to is not compared (open).
- **D6.** A block that a primitive makes and a symbol bound to an
  equal static block are not the same at an exit (open).
- **D42.** Two boxes of one number are one value, so the validation may
  share them, not blocks or arrays (provisional for the proof; open).
- **D52.** A field read as immutable from a block that can be written,
  as in a record with a mutable field, is undefined (open).
- **D49.** A view of mutable data as immutable (bytes as a string, an
  array as an immutable array): a change of the data while the view is
  live is undefined (ruled by 13 Nov). The rule planned for the model
  differs; meanwhile such a load of an array is undefined (open).
- **D71.** Not modelled, on the human's word (by 12 Nov): the blocks
  that tail-modulo-cons fills by initialising stores, and the copy of a
  record by `Duplicate_block`. A term that has one is not checked.
- **D70.** Undefined: a store into a block not made mutable; an array
  access out of bounds; making a mutable array of no elements; the byte
  after the last of a string (provisional; open whether it reads as 0).
- **D56.** A verdict need not establish that the array kind of a store,
  or the kind of a length, agrees in the two terms: the machine reads
  neither (provisional).
- **D63, D64.** Atomics are to be built (approved 18 Nov) as cells of a
  model that has one thread, so a verdict is of runs with one thread; a
  pause (`Cpu_relax`) has no effect (provisional; not yet in the tree).
- **D67, D68.** Arrays that may hold floats, and bytes, are to be built
  (approved 18 Nov). How such an array is read, and what bytes hold
  before they are filled, are open: the model is to say least.
- **D40, D43, D57.** The block of an exception that a file defines is
  to be built (approved 18 Nov). Whether the block made at run time and
  the static block that Simplify makes of it are the same exit is open.

## 5. Exceptions

- **D73.** A raise is the application of an exception continuation. The
  machine has no stack of handlers and reads no trap action. A second
  machine has the stack, and is proved to agree on units with distinct
  binders that keep the discipline of traps. Some terms that break it
  are undefined there: a pop or a raise with no handler on top, a return
  with a handler installed above the one it began with (provisional).

## 6. Regions

- **D37, D44.** The machine of terms has no regions: a local allocation
  is ordinary (open). A second machine (the rest of this section) has
  them on one stack. A local allocation lives as long as the newest
  region when it is made, whatever region its mode names, for ever if
  none is open. The end of a region ends all begun since (provisional).
- **D45, D53.** Evaluating a value whose region is gone is undefined,
  whatever is done with it. The machine of regions runs the heap: a cell
  of a local allocation is dead once its frame is gone (provisional).
- **D46, D54.** Undefined: a `Let` of a heap allocation, or a static
  constant, with an argument that is a local value still in a region
  (not tested: the partial application that a call makes); a store of
  such a value into a heap cell; a store into a local cell of a value in
  a newer frame (provisional).
- **D47.** Undefined: the end of a region that is not on the stack, a
  local allocation that names one, giving one to a continuation. The end
  of a region that is not the newest is defined (provisional).
- **D48.** Not a fault and not found: leaving code with a region of
  its own still open, by a return or by a raise (provisional).
- **D55.** The verdict about ends: a unit that passes, run from its
  first state, leaves no frame of a plain region where it exits by its
  return continuation; a body of code that passes, run on its own under
  the stated entry conditions, leaves the plain frames it began with. It
  says nothing of an exit by an exception or of the regions of a `try`,
  and what it refuses is a choice (provisional).

## 7. C calls

- **D50.** A call of a C function of which the model says nothing
  makes the run undefined at the call (ruled 18 Nov; not yet in the
  tree, where every such call is still refused before the model).
- **D69.** Named C functions are to have their meaning in the model: one
  Rocq spec, under the names of the runtime, of what a call of each
  does. Not all of them: not the collector and the like. Two are given:
  `caml_create_bytes` is an allocation, `caml_fresh_oo_id` the increment
  of a counter (ruled 18 Nov; the design is being written).

## 8. What a verdict is about

- **D74.** A validation is of the initialisation of one unit, run on its
  own from the empty heap, with nothing imported or from its declared
  imports, as its verdict says: every exit of the term before is an exit
  of the term after, with the same continuation, values and heap (D40,
  D57), and the term after does nothing else unless the term before can
  go wrong. This verdict supplies no theorem about later calls of what
  the unit exports (provisional).
- **D75.** A verdict may rest on what the `.cmx` files of other units
  say: their symbols and code are bound in front of both terms, the same
  for both. That each file is right about its unit, and that the program
  is linked with that unit, is trusted (ruled 18 Nov, for now).
- **D43.** Where a lifting changes an exit there is a different verdict,
  one-way as D74 is: each exit of the term before is answered by one of
  the term after that is related to it under every observer, also in one
  direction. Where the observer goes wrong after the first exit, nothing
  is asked of its run after the second. An observer is a way out, or a
  handler with no environment of its own, no symbol, code id or static
  binding (so no import) and no projection of a removed slot. The
  verdict is printed apart, never counted as validated (open).
- **D60.** Well-formedness asks that static bindings be outside loops
  and code; a separate check of static names gives a handler only the
  names bound on every way to it (provisional for the interim; open).
- **D65.** Whether the guards of a lifting need look inside functions in
  a unit with no call outside them. Meanwhile they do; not to look would
  change which units are validated, not what a program means (open).

## 9. The types of Simplify

- **D27, D61.** The model's meet and join are n-way; the binary pair,
  the compiler's default, is not transcribed (ruled by 11 Nov; 17 Nov).
- **D26.** A typing environment sets aside what a type says of the shape
  of a name that holds poison, and the `Is_int` and `Get_tag` facts on
  it; it still asks the kind and the aliases (provisional).
- **D30, D36.** Where `==` in the compiler changes an answer, the
  structural behaviour is transcribed (ruled by 11 Nov). The model's
  equality of types is coarser than the compiler's (provisional).

## 10. Conditions of the proofs, not of a program's meaning

- **D7.** How a heap enters the statements is part of its design (open).
- **D11.** The validation does not run the check of kinds (provisional).
- **D12.** A check of kinds may refuse only what the compiler refuses
  with its default flags (ruled by 11 Nov).
- **D29.** A renaming acts on all that holds names, code too (ruled by 11 Nov).
- **D32.** Well-formedness may ask more than the compiler checks, where
  that ought to be an invariant; never for kinds (ruled by 11 Nov).
- **D33.** Tests may read Simplify's types (ruled in part by 11 Nov);
  the design of the observer is open.
- **D34.** `structural` as the compiler's default of join sharing (open).
- **D35.** What `Coercion.change_depth` compares its arguments by (open).
- **D39.** The freshness quantifier of names is derived (provisional).
- **D62.** Where a check of regions accepts the term before and not the term
  after, a switch of the test tool, off unless asked for, withholds the caption
  "validated"; the verdict stands (provisional; open; not yet in the tree).
