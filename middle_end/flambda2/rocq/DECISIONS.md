# Decisions and discrepancies: an index of the labels

The comments and documents of the Rocq model of Flambda 2 cite decisions
as `D<n>` (in a `CR`: "Decision D<n>") and discrepancies between the
model and the compiler by the stream that found them and a number
(`T1-4`). This file says in a line or two what each label stands for.
It is an index and not the record: a decision's state here is its state
when the line was last touched.

"Ruled" is a ruling of the human in charge; "decided" is a choice made
for the proof that the human may overrule; "open" is still the human's
to decide, and the model then takes the conservative choice named, with
a `CR` at the place in the `Spec` file.

## Decisions

- **D1.** What an exit observes of a closure. Meanwhile the whole value:
  its function slot, the code ids of its set, the values of its value
  slots. Open.
- **D2.** Identity of closures. Meanwhile none: two closures are the
  same when their contents are. Open; goes with the heap.
- **D3.** A projection that names a function slot other than the
  closure's own is undefined. Ruled (as probably right).
- **D4.** How a static set of closures is kept consistent. The model
  asks, as a condition of well-formedness, that one group binds a set
  whole, each symbol at its slot. Open.
- **D5.** On which values `Is_int` and `Get_tag` have a value: `Is_int`
  of a variant on tagged immediates and blocks, `Get_tag` on blocks.
  Ruled for `Get_tag` (no value on an immediate); `Is_int` outside its
  domain is open.
- **D6.** Whether a block made by `Make_block` and a symbol bound to an
  equal static block are the same at an exit. Open; goes with the heap.
- **D7.** How a heap enters the statements, which build states field by
  field. Open; part of the design of the heap.
- **D8.** An operation on a symbol that is not bound is undefined, and
  well-formedness rules such terms out. Ruled. What `Is_null` and the
  other primitives are on poison is open; they have no value.
- **D9.** A call whose arities are not those of the code called is
  undefined. Ruled.
- **D10.** A mismatch of kinds is undefined where it is executed. Ruled
  for calls and primitives. The machine also tests the kind of a value
  where a `Let` binds it and where a handler is applied to it: decided
  overnight, for the review of the human, with a review comment before
  `step` that gives the other reading. Not tested: the field of a
  static block or a value slot where it is stored, a value given to a
  way out of the term, a value that a call returns. Open in that part.
- **D11.** The validator does not run the check of kinds; the test tool
  prints the check's answer and goes on. Decided.
- **D12.** A static check of kinds may refuse only what the compiler
  refuses with its default flags, and no statement takes such a check
  as a premise about the compiler's terms. Ruled.
- **D13.** Integer constants are in range for their kind: a condition
  of well-formedness. Ruled.
- **D14.** Division by zero is undefined; the least integer divided by
  -1 wraps. Taken from the compiler; open to objection.
- **D15.** A `Block_load` that fails the checks of size, tag or
  `Immediate` has no value. Open.
- **D16.** A primitive given an argument of another kind has no value.
  Ruled, with D10.
- **D17.** The target has 64 bits. Open.
- **D18.** A shift by an amount outside `[0, width)` has no value. Open.
- **D19.** `Phys_equal` has no value on two allocated values. Open;
  goes with the heap.
- **D20.** Closures that no compiler term makes (a closure at a
  function slot that its set has not got, and the like): the model
  leaves them to well-formedness and gives the machine no step on
  them. Open.
- **D21.** A boxed number whose contents are poison is a boxed number
  and is allocated; boxing and unboxing poison have no value. Open.
- **D22.** A caution, not a question: "no value" removes a case from
  what refinement has to prove. It is conservative about results, not
  about how much of the compiler's correctness is established.
- **D23.** `Make_block` keeps an integer tag and the machine does not
  test it; tags in `[0, 251)` are a condition of well-formedness.
  Decided.
- **D24.** Blocks with no fields are allowed. Ruled.
- **D25.** What the kind of a `Make_block` field promises. The
  conversion refuses annotated blocks meanwhile. Open; goes with kinds.
- **D26.** Types of names that hold poison. Typing facts are about
  values that are not poison; an environment asks nothing of the type
  of a name whose value is poison; an `Is_int` or `Get_tag` relation
  is never true of a scrutinee that is poison; `Is_null` relates
  poison to 0; aliases are exact, poison included. Decided.
- **D27.** The meet and join transcribed first are those of
  `types/meet_and_n_way_join.ml` (the n-way join). Ruled.
- **D28.** The functions of a set of closures have distinct code ids,
  and so have the sets bound in one static binding group. Ruled.
- **D29.** The renaming action reaches into code, and is defined on
  everything that holds names. Ruled.
- **D30.** The model ignores physical equality (`==`) in the compiler.
  Where `==` changes a result, the behaviour transcribed is the
  structural one. Ruled.
- **D31.** Code is closed but for symbols, code ids and slots; its
  parameters, `my_closure` and its return and exception continuations
  are binders. A condition of well-formedness. Ruled.
- **D32.** The model's well-formedness may be stricter than what the
  compiler checks where the condition ought to be an invariant, with a
  comment at the condition; not for kinds. Ruled.
- **D33.** Access to the types library for tests: accessors on
  `Block_size`, and a way for a test to observe the typing
  environments of Simplify, without opening the abstraction of
  `Flambda2_types.t`. Ruled in part; the design of the observer is
  open.
- **D34.** Whether `structural` becomes the default of
  `-flambda2-join-sharing`. Open.
- **D35.** Whether `Coercion.change_depth` compares its arguments with
  `Rec_info_expr.equal` in place of `==`. Open.
- **D36.** The model's equality of types is coarser than the
  compiler's structural one (depth information, the string of a
  poison, relation entries bound to no name); the correspondence is
  stated only on types that do not differ there. Decided.
- **D37.** Regions and local allocations. Proposed: the conversion
  drops regions and reads a local allocation as an ordinary one.
  Open.
- **D38.** `Boolean_not` of a tagged immediate other than 0 and 1 is
  undefined. Simplify makes it `Invalid` where the argument cannot be 0
  or 1, the backend flips the low bit of any integer, and nothing checks
  that the front end gives it a boolean. Decided overnight, for the
  review of the human; forced by the `Invalid` of Simplify.
- **D39.** The freshness quantifier of nominal sets is a derived layer;
  the explicit forms stay primary. Decided.
- **D40.** What an exit observes of the heap. Meanwhile the heap itself:
  two exits are the same when their values and their heaps are equal,
  locations included. So different heaps are refused even when the
  difference is only garbage that is unreachable from the exit (the
  term after does not allocate a block that nothing at the exit
  reaches) or a consistent correspondence of locations (two blocks
  allocated in the other order, each with the location of the other):
  two defined exits that differ so are not related by this comparison.
  The comparison intended is up to a correspondence between the
  locations of two runs (the prototype's is
  `FlambdaHeapMachineSpec.outcome_related`, of outcomes that carry
  their static environments); it is not started without the human.
  Decided for the interim, as the strict side; open.
- **D41.** An import of which nothing is known (`Import_unknown`) has no
  value under `Phys_equal`, even against itself, where the program
  answers true; and nothing that inspects it has a value. The model says
  less than the program does there, never more. Decided, for the review
  of the human.
- **D42.** Whether the validation may make two allocations of an
  immutable value one, as the compiler's sharing of primitives does.
  Meanwhile for boxed numbers only (`FlambdaCseSpec.shareable` lists
  `Box_number`): in the model two boxes of one number are one value
  and `Phys_equal` has no value on two of them (D19), so the pass
  stays an equivalence there. Blocks and arrays are not shared. The
  other answer is that no allocation is shared; a pair of terms
  whose validation needs this sharing may then stay not validated.
  Decided for the proof; open.
- **D43.** What a verdict is to mean of a unit whose exits a lifting
  changes. The form that is in the model: an exit of the unit before
  is answered by an exit of the unit after that nothing which may
  follow a unit tells apart from it (`FlambdaValidateRelatedSpec.v`,
  proved, over the whole class `FlambdaObserversSpec.observer`). It
  does not establish `refines`, which asks of two exits the same
  values. What may follow names no symbol, so no import: of two exits
  that each carry a predefined exception it does not say that it is
  the same exception. Allowing what may follow to name imports would
  be a separate proposed strengthening; it is neither defined nor
  proved here. The alternative is structural, a relation of the
  values of the two exits and of every entry of the two memories; it
  is not defined here. Meanwhile the first is in the model beside the
  validation and changes none of it: the tool prints what it accepts
  on a line of its own under "not validated", and such a unit is in
  no count of the units validated. Open: the human's to choose.
- **D44.** What a region is on the machine that has them
  (`FlambdaRegionsSpec.v`). Chosen: what the program does. There is one
  stack; a local allocation lives as long as the newest region when it
  is made, whatever region its mode names; the end of a region ends
  every region begun since; a raise, a `Push` and a `Pop` are not
  events of the region machine. A local allocation is a fresh symbol
  bound to what was made, which is bound to nothing once its region is
  gone; one made with no region on the stack is bound for ever, and
  counted. The conversion from Lambda relies on the same: a region in
  tail position of another has no end of its own on that way out, and
  the end of the outer one ends it (`lambda_to_flambda.ml`, the case
  `Lregion`: "we will jump directly to the region closure continuation
  for the outermost open region"). The other choice: an allocation
  lives as long as the region that it names, and a value carries a
  mark of where it is. Then a term in which a name is wrong or an end
  is lost has a defined run in which a value lives that the program
  has freed, a region that the front end leaves to the end of another
  never ends, and every proof about values is done again. Decided, for
  the review of the human.
- **D45.** The step that evaluates a simple whose value was allocated
  in a region that is gone is undefined, whatever is then done with the
  value; so is one that evaluates the variable of a region anywhere but
  at its end, in the mode of an allocation, or as an argument given to
  a continuation at a parameter that is a region (a region that is
  open and is evaluated anywhere else is refused under a name of its
  own, as nothing freed is used there). A value made since a
  `try` was entered is dead once the handler has ended the region of
  the `try`, though the region that it names is open. The program asks
  less: a use of what was freed goes wrong where the freed space has
  been written again (a later local allocation, the poison of a
  debugging runtime), where the collector has stopped tracing and
  updating the fields of the freed block, or where a root that is
  still live makes the collector look at the freed space. The other
  choice is that only a primitive, a call or a `Switch` that reads the
  value is undefined: then a dead pointer may be kept, passed and
  stored. Decided, for the review of the human.
- **D46.** An allocation on the heap that holds a local value which is
  still in a region is undefined: a block, an array or a closure made
  by a `Let` whose mode is not `Local`, and every static constant that
  the term binds. The program makes it. It goes wrong if the value is
  reached through it after its region is gone, and it can go wrong if
  a collection runs while the local value is reachable only through
  the heap block: the collector does not follow that edge into the
  local stack. The other
  choice is to say nothing where it is made and leave it to D45 where
  it is used: then a heap block may point into the local stack. Not
  asked of the value that the machine makes for a partial application.
  Decided, for the review of the human.
- **D47.** The end of a region that is not on the stack (ended twice,
  or after the end of a region begun before it) is undefined, and so
  are a local allocation whose mode names such a region and an
  `Apply_cont` that gives such a region to a continuation, at a
  parameter that is a region (the program passes the number, and a
  handler that does nothing with it is harmless). The end of a
  region that is not the newest is defined, and ends those above it.
  The normal ways out of a `try` leave its mark; the end of a try
  region is the end of a region. The program has no such test: a
  region is the number that its begin read, and the end writes it
  back, whatever was done since. The other choice is to follow the
  program there too, with the pointer of the local stack as a number
  in the state: then a region ended twice is a defined run. Decided,
  for the review of the human.
- **D48.** What the machine of regions does not check. Leaving code
  with a region of its own still on the stack, by a return or by a
  raise, is not undefined, and is not found: the model's code does not
  say whether its function returns something local, and the handler of
  a `try` ends what a raise leaves. The exception that a raise carries
  is not asked to be on the heap at the raise: that is asked where it
  is built (D46) and where it is used (D45). A local allocation that
  names another region than the newest that is not of a `try` is not
  undefined either. Neither is found where it happens. The test tool
  counts what it can: the regions left when the run of a unit ends and
  the local allocations still in them; the regions ended by the end of
  another; and the allocations that name, by a variable, another
  region than that one (an allocation that names the region of its
  code is not counted, whatever is open). A count is not a fault by
  itself: the front end has a region ended by the end of another on
  purpose (D44), and Simplify, where it splits an application with too
  many arguments, begins the ghost region by a begin that is not a
  ghost (`simplify_common.ml`), which is one more region for the end
  of the first to end. A count is looked at where Simplify adds to
  it. Ghost regions, the parent of a region, the modes of calls and
  the `alloc_region`s are not in the terms that the machine runs. The
  other choice is that a region which nothing ends is a fault of the
  term; it needs the result mode of code in the model. Decided, for
  the review of the human.

- **D49.** A view of mutable data as immutable:
  `Bytes.unsafe_to_string` and `%array_to_iarray`. Ruled, in the
  source's terms: "Regarding Bytes.unsafe_to_string, the semantics of
  this function are intended to be that modification of the Bytes
  during the period when the returned String is live are UB. Same for
  the function producing an immutable array from a mutable one." Both
  conversions are erased before Flambda 2
  (`from_lambda/lambda_to_lambda_transforms.ml`): the term has no
  view, a load through one is a string load, or an array load marked
  `Immutable`, of the bytes or of the array itself, and the term does
  not say which view a load was through, nor where a view was made.
  Decided for the model, to be made with the bytes and the arrays of
  the heap: such a load of a cell that can be written is to read the
  cell as it is, and a later write to a cell that such a load has read
  is to be undefined. That is not the rule. For a lend it is stricter:
  it calls undefined a program that the rule defines, one whose view
  is dead before the write (use-case 2 of `stdlib/bytes.mli`). And it
  gives a meaning to a write made after a view and before the first
  read through it, which the rule makes undefined. The departure is
  deliberate: the compiler gives these loads no coeffect
  (`effects_and_coeffects_of_binary_primitive` of
  `terms/flambda_primitive.ml`), and the move of a load in X5-3 relies
  on what it read not changing. The tool is then to list the units
  concerned: for a unit whose run writes a cell that such a load has
  read, it is to make a second pair of runs in which the write is
  carried out, and to print one line more, which says that the unit
  does this and whether the two runs of the second pair, before and
  after Simplify, are the same or differ. The line is not to use the
  word by which the tool marks a difference of the runs it validates,
  and such a unit is to be in no count of the units validated. Not in
  the model yet: meanwhile an array load marked `Immutable` of an
  array that can be written has no value, and the model has no bytes
  (`FlambdaHeapSpec.exec_binary_on_cell`, with its `CR`). Open, and
  the human's: whether the rule is to be his, or one under which the
  move of X5-3 is right, that bytes of which a view has been read are
  never written again.
- **D50.** What a call of a C function shows of its arguments where
  the model has no answer for it. The table of
  `FlambdaExtcallSpec.v` answers fourteen comparisons, on two strings
  or on two tagged immediates; a program that compares two lists
  calls one of them outside that. Such a call is to be neither
  undefined nor a raise, so the run would end there, and a statement
  that relates the term before to the term after has to say what of
  the two ends is the same: the arguments as values, what is
  reachable from them, what they read as, or what every term that
  could follow sees of them. Meanwhile no call of a C function has a
  meaning in the model: the conversion refuses every one and nothing
  uses the table. Open, and the human's: `requests/P1-c-calls.md`,
  "The open choice", has the ways to answer and what each costs.
- **D53.** What the machine of regions does with the heap, once
  `step` runs a `Let` on the heap. Its frames list the symbols of
  local allocations and no cells, so a cell made in a region would
  never be found dead there. Decided for the stage that makes `step`
  run on the heap: the machine of regions does not run a `Let` of a
  primitive of the heap, nor of any other primitive but one of
  regions, one of whose arguments has a location for value
  (`FlambdaRegionsSpec.named_needs_heap`). The step then has a result
  of its own, `Region_let_not_run`, which is not a step, not an exit
  and not a fault; the run ends there. The test of the simples (D45)
  comes first. The tests of the form of the `Let` do not (the marker
  of a local allocation, D46): a local value put in a cell of the heap
  is neither found nor ruled out at such a `Let`. What "safe" then
  includes: `region_safe` is "never undefined", and it holds of a run
  that ends at such a `Let`, with nothing checked from there on. Its
  name and the names of the statements that conclude it stay; the
  comment and a `CR` at the definition say what they give. A run from
  the first state of a term with no primitive of the heap, in a static
  environment whose code has none and whose values hold no location,
  never ends so (`no_heap_primitive_no_let_not_run_statement`): of
  such a term the statements say what they said. The statements about
  exits gain the case "or the run comes to a `Let` that the machine
  does not run". The test tool is to print such a unit as not checked
  for regions from that `Let` on. The other choices: to call such a
  `Let` undefined for want of a step, which the term with its regions
  dropped is not; or to run it on the heap and say nothing of the
  cells of a region, which would pass as safe a term that reads a
  local cell after the end of its region. Lifted when the frames of
  the machine of regions list cells. If a symbol comes to name a cell
  before that, the test follows the premise of
  `exec_named_no_location_statement`. Decided, for the review of the
  human.

## Discrepancies with the compiler

A label is the stream that recorded the entry and a number. An entry
says where the compiler is silent, imprecise or inconsistent, or where
the model and the compiler part; it is not a claim of a bug unless it
says so. Only the labels that this tree cites are listed.

- **C0-1.** Simplify gives every new version of a piece of code a new
  code id, and closures carry it out of the unit.
- **C0-3.** Simplify refuses a `Let` that binds symbols away from the
  toplevel of the unit, but not one that binds only code; the model
  refuses both.
- **C0-4.** Nothing checks that a symbol or a code id is bound only once
  in a unit.
- **C0-5.** Code is checked to be closed against the free names its
  creator supplies.
- **C1-1.** Simplify replaces an integer division or remainder by zero
  with `Invalid`.
- **C1-2.** The least integer divided by -1 wraps to itself, and its
  remainder is 0.
- **C1-3.** Simplify replaces a primitive applied to an argument of
  another kind with `Invalid`.
- **C1-4.** Simplify takes the argument of a variant-only `Is_int` to be
  a variant that is not null; the machine computes the low bit of
  anything.
- **C1-5.** Simplify folds `Is_null` of a poison constant to 0 and
  `Is_int` of one to 1; the backend makes poison a tagged immediate.
- **C1-6.** Simplify replaces `Get_tag` of null or of a boxed number
  with `Invalid`, and leaves `Get_tag` of a tagged immediate alone.
- **C1-7.** Simplify uses the tag of a `Block_load`'s access kind only
  when it does not already know the field, and never its size.
- **C1-8.** Simplify turns a block made by `Make_block`, or a boxed
  number, into a static symbol; in the model the two are different
  values.
- **C1-9.** Simplify folds an arithmetic shift right by 63 of a negative
  naked immediate to 0, and of a tagged immediate to -1.
- **C1-10.** fexpr reads `%is_int` as the variant-only `Is_int`, and
  prints the other one the same way.
- **C1-11.** The reader of the text of terms fails on an integer constant
  beyond 63 bits where it stands in an expression, though the printer
  writes it, and wraps an out-of-range `int8` or `int16` literal without
  a word.
- **C1-12.** A constant of the model holds any integer; the compiler's
  constants are in the range of their kind by construction.
- **C1-13.** The kinds of the fields of a `Make_block` come from the
  types of the source, are used by Simplify for variables only, and are
  ignored by the backend.
- **C2-1.** Simplify replaces by `Invalid` a call made at arities that
  are not those of the code called.
- **C2-2.** With the default flags Simplify accepts ill-kinded code and
  makes it `Invalid`; a well-typed program with a GADT indexed at layout
  `any` needs that.
- **C2-3.** Simplify replaces by `Invalid` a call whose argument is not
  of the kind that the arity of the call itself gives; the machine runs
  it.
- **C2-4.** Some mismatches of kinds stop Simplify with a fatal error
  whatever the flags, while others of the same sort become `Invalid`.
- **C2-5.** The fexpr parser gives a direct call, and an indirect call
  with no annotation, the arity in which every argument and result is a
  `Value`; the printer leaves the arity of such calls out.
- **C2-6.** Nothing checks that the arities in a code's metadata are the
  kinds of its parameters.
- **C2-7.** A call with fewer or more arguments than the code has
  parameters is undefined in the model; in the compiler it is a partial
  application or an over-application.
- **C2-8.** Two array primitives treat an array of another kind than
  their own differently, and neither becomes `Invalid`.
- **C2-9.** An indirect call of a tupled function is made at one
  argument, the tuple, whatever the arity of the code; the model has no
  such mark on code.
- **C2-10.** Simplify reads the subkind of a parameter into its type
  and the backend into what the collector does not scan; nothing checks
  that the value bound is of it.
- **C2-11.** A use of a continuation whose argument cannot be of the
  subkind of the parameter is not made `Invalid`.
- **C2-12.** Simplify gives a parameter that it puts in the place of
  another no subkind, so the term after asks less of the value.
- **C2-13.** Nothing checks that a unit does not bind again the name of
  its return or exception continuation, and the model's reading of "at
  toplevel" goes by that name.
- **C2-14.** The printer of fexpr writes the name of a poison constant
  without quotes and the parser reads a string there, so the compiler's
  own fexpr text of a term with poison is not read back.
- **C2-17.** The handler of a probe is called only while the probe is
  enabled, which a mark on the call says and the text of a term does
  not; the model has no probe.
- **C2-18.** The kind that `Make_block` gives each field is read by
  Simplify and by nothing after it, and the text of a term does not
  write it.
- **C2-19.** Simplify, with `-g` and phantom lets on, keeps a `Let` whose
  variable is unused as a phantom binding where it would delete it;
  `to_cmm` runs nothing for such a `Let`, and the text of a term does
  not say the mode.
- **C2-20.** The reader of the text of a term gives a symbol of the
  unit and a symbol of another unit one identity when their linkage
  names are the same.
- **C2-21.** Other units read the data of a unit's module symbol, and
  the backend ignores what the unit gives its return continuation; the
  model observes what the unit gives it.
- **C2-22.** In a unit converted with unnamed function slots the test
  tool did not read the function slot that a projection of a value
  slot names; the backend computes the load from it. Every unit is now
  converted with named function slots.
- **C2-23.** The backend lays closures out by offsets made from the
  free names that Simplify returns beside the term; nothing compares
  those names with the term.
- **C2-24.** Nothing compares the kind of the variable of a `Let` with
  the kind of its defining expression; the backend does not read it,
  and the model tests the value bound against it.
- **C2-25.** The backend relies on what a term caches of itself (counts
  of occurrences, whether code uses its closure, whether a value slot
  is always an immediate); the conversion reads none of it.
- **C2-26.** The text of a term writes both forms of `Is_int` as
  `%is_int`, and its reader gives the form for variants.
- **C2-27.** The text of a term does not write the kind of
  `Opaque_identity`, and its reader gives the kind of values.
- **C2-28.** The reader of the text of a term reads the kind `int32` of
  `%invalid` and of `%optimised_out` as naked int8, and `int64` as naked
  int16; the printer has no word for naked int32 and int64 there and
  stops.
- **C2-29.** The text of a term does not write the shape of a static
  block, and its reader gives the shape of values only; Simplify stops
  with a fatal error on a constant record with an unboxed field that is
  read back.
- **C2-30.** The text of a term writes nothing for an argument or a
  parameter of no kind (the unboxed product of nothing); read back, a
  partial application or an over-application can be a full application,
  and a full application a partial one.
- **C3-1.** Simplify answers a projection that names the wrong function
  slot on one of its two paths and makes it `Invalid` on the other; the
  slot named is not checked where the term is built.
- **C3-2.** `to_jsir` translates both projections from the environment
  of the function being translated, whatever closure they are applied
  to.
- **C3-3.** Simplify simplifies the body of each function of a set
  knowing that `my_closure` is that closure of that very set.
- **C3-4.** The number of variables bound by a `Let` of a set of
  closures is not checked against the number of its functions where the
  `Let` is built.
- **C3-5.** A set of closures in which a function slot is given twice is
  built without complaint, and Simplify then fails on an unrelated
  assertion.
- **C3-6.** The compiler binds the symbols of a static set with one
  pattern; the model binds each with a binding of its own and has to
  check that they agree.
- **C3-7.** The model can write a closure at a function slot that its
  set has not got; the compiler cannot.
- **C3-8.** The model can write a set of closures in which a value slot
  is given twice; the compiler cannot, and the fexpr parser accepts one
  and keeps the second value.
- **C3-9.** The backend reads a direct call and a call of known arity as
  full calls; Simplify reads a call of any kind as a generic
  application.
- **C3-10.** `simplify_direct_tuple_application` takes the tuple and the
  remaining arity apart by two different counts, and splits an
  over-application without the test that the code returns one value.
- **C3-11.** At the default flags Simplify simplifies the code of a stub
  only where it is bound, knowing nothing of its own closure: a
  projection of a value slot from `my_closure` stays, and a direct call
  in the code may become indirect.
- **C4-1.** Simplify lifts an `Immutable_unique` block out of a function
  to one symbol.
- **C4-2.** Nothing checks that a field loaded as immutable is never
  written.
- **C4-3.** A comment says that physical equality is referential on
  immutable blocks; CSE and sharing do not keep it.
- **C4-4.** The copy of a large record stores, with `Assignment`, into
  fields that are then read as immutable.
- **C5-1.** Nothing in Flambda 2 checks that trap actions, raises and
  calls name the handler that is installed last; the backend goes by the
  stack and Simplify by the names.
- **C5-2.** `Apply_cont_expr.is_raise` and the backend disagree on which
  `Apply_cont` is a raise.
- **C5-3.** A `Pop` with no raise kind is "the same as No_trace" for the
  backend and is treated as a regular raise by Simplify.
- **C5-4.** With debugging information on, Simplify does not turn a
  regular raise or a reraise into a jump.
- **C5-5.** A call can give extra arguments to the exception
  continuation of its code or of the unit, which takes the exception
  alone; in the model a way out of a term takes any number of values.
- **C5-6.** Fexpr drops the extra arguments of a call whose exception
  continuation is `error`, without a word.
- **C5-7.** Nothing tests that the first parameter of an exception
  handler is of kind value.
- **C5-8.** Nothing checks that a continuation used as an exception
  handler is marked `is_exn_handler`, nor that a marked one is not the
  return continuation of a call.
- **C5-9.** `to_cmm` stops on a trap action of a jump that names the
  exception continuation of the function or of the unit.
- **C5-10.** A `try` whose handler Simplify has demoted keeps the end of
  its region, so the handler is never a shortcut and a branch that goes
  nowhere else stays.
- **C5-11.** Simplify's output can bind a symbol inside an exception
  handler, which Simplify refuses as input. In a term not taken from
  an OCaml source it binds one under a handler beside which it has
  kept an arm of a `Switch` that does not come back to that handler
  (an arm that it made invalid, or one that raises or is invalid): by
  the rule read in its source that handler is not at toplevel;
  Simplify was not run on it.
- **N1-1.** `Compare` writes the reverse maps of function slots and
  value slots into the forward maps.
- **N1-2.** `pattern_match_pair` on code and on recursive handlers stops
  the compiler where the lists of bound names differ in length.
- **N1-3.** That the names of a binder are distinct is checked for
  continuations always, and for parameters only with invariant checks
  on.
- **N1-4.** That code is closed is a claim of whoever creates it,
  checked only for the form of the claim.
- **N1-5.** A static pattern is in a name abstraction, is never renamed,
  and is subtracted from the free names.
- **N1-6.** Existential variables of an environment extension, and the
  variables a level defines, are binders that the nominal interface does
  not see.
- **N1-7.** Some of Simplify's output depends on the order of names.
- **N1-8.** A renaming with an import map is not a permutation, and
  renamings do not compose in general.
- **N1-9.** The comparison of terms of the compiler (`Compare`) has no
  case for a primitive of four arguments, so it judges a term that has
  one different from itself.
- **P1-1.** Simplify folds `Phys_equal` of a poison against another
  constant to 0, and the backend makes a poison of kind `Value` a tagged
  immediate.
- **P1-2.** Simplify makes `Boolean_not` of a tagged immediate other
  than 0 and 1 `Invalid`; the backend flips a bit of whatever it is
  given.
- **P1-3.** Outside the amounts it computes, Simplify folds a shift to
  zero while calling it undefined, and the backend shifts by the amount
  modulo 64.
- **P1-4.** The fexpr parser reads `%bytes_length` as `String_length
  String`, and the printer writes `String_length Bytes` as
  `%bytes_length`.
- **P1-5.** An unsafe load of several bytes at the end of a string reads
  the byte after the last, which is 0 on the machine and which the model
  has not got; the model is stricter.
- **P1-6.** Simplify makes `x *. 1.` and `x /. 1.` into `x`, which for a
  signalling NaN is not the machine's result.
- **P1-7.** Simplify makes `x *. -1.` and `x /. -1.` into the negation
  of `x`, which flips the sign of a NaN where the machine keeps it.
- **P1-8.** Which NaN a folded operation returns is the host's. Not
  reproduced on x86-64 for x86-64; open for cross compilation.
- **P1-9.** The fexpr printer writes `Array_load` and `Array_set` with
  their array kind only, so a load kind or a set kind that differs from
  it is lost.
- **P1-10.** Simplify stops with a fatal error on two array primitives
  that the terms let anyone build.
- **P1-11.** Simplify looks at the array kind of a load and of a store
  and not at that of `Array_length`, whose compiled code depends on it.
- **R1-1.** Unboxing records `Get_tag` on the paths where the value is
  an immediate, and supplies a dummy tag of zero.
- **R1-2.** The types library evaluates `Is_null` of poison as false.
- **R1-3.** The field annotations of `Make_block` constrain variables in
  Simplify, and the model does not keep them.
- **R1-4.** The metadata of a `Block_load` is not checked uniformly by
  Simplify or by code generation.
- **R1-5.** `Get_tag` of a boxed value is lowered to a read of the tag,
  but Simplify can discard it.
- **R1-6.** The old test converter merged the least tagged immediate
  with minus one.
- **R1-7.** A block with no fields that is made at run time has to be
  lifted before the backend.
- **R1-8.** The model accepts block tags outside the compiler's
  scannable range.
- **R1-9.** `Tag.Non_scannable.of_tag` uses the test of the scannable
  range.
- **R1-10.** The comment on the modulus in `One_bit_fewer` disagrees
  with its implementation.
- **R1-11.** What is observed of a boxed number that holds poison
  differs between the types (T1) and the primitives (C1).
- **R1-12.** The conversion of static constants discards the shape of a
  mixed block.
- **R1-15.** The n-way join drops a conditional extension of a variant
  that the binary join keeps.
- **R1-16.** The n-way join of immediates gives an empty set to
  `create_naked_immediates_non_empty` when every head reduces to bottom;
  traced, not shown reachable.
- **R1-22.** Nothing checks that a trap action does not name the return
  continuation of the unit: `to_cmm` gives that continuation a handler
  of its own and lets such a trap action through, where for the return
  continuation of a function it fails.
- **R1-24.** A model of imported symbols in which every import is an
  object with an identity answers false to `Phys_equal` of any import
  against any allocated value of the unit, which is not justified for an
  import that is immutable data: Simplify may share it. An import has a
  class, and only one that is an identity is told apart from what the
  unit allocates.
- **R2-3.** A static set of closures may be empty; a set bound by a
  `Let` may not.
- **R2-7.** Simplify fails on a set of closures in which two function
  slots share one code id.
- **R2-9.** The number of variables that a `Let` of a set of closures
  binds is checked only by later consumers of the lists.
- **R2-11.** Simplify removes a dead use of a continuation at the wrong
  kind before checking it.
- **R2-12.** Printing a term as fexpr and parsing it back loses the
  arities of calls.
- **R2-15.** Sharing code between static sets succeeds in separate
  groups but fails within one group.
- **R2-16.** The checks for duplicate parameters compare the whole
  parameter, not only its variable.
- **R2-17.** A recursive group of continuations must be non-empty in the
  compiler; the model can write an empty one.
- **R2-18.** Tail-modulo-cons initialises blocks made `Mutable` that are
  later read as immutable lists.
- **R2-19.** An `Immutable_unique` block is lifted also where the
  function binds an allocation region.
- **R2-20.** A load marked immutable is reused across a write to the
  same field.
- **R2-21.** Sharing immutable allocations changes the result of a raw
  physical comparison.
- **R3-2.** The reverse reduction of `Get_tag` discards the possibility
  of an immediate.
- **R3-7.** The n-way join can say what the code of a closure is when
  one of its inputs does not (with T1-4).
- **R3-15.** In the model, the immediates of a type include raw integers
  that are out of range for the compiler.
- **R6-1.** The printer of fexpr writes the naked kinds `int8`, `int16`
  and `mask`, and the parser has no rule for them, so a poison constant
  of one of those kinds that is printed is not read back.
- **T1-1.** After unboxing, the compiler's own types say that the
  arguments passed for a constant constructor are impossible (the
  `Get_tag` reduction of meet).
- **T1-2.** The compiler reads `Is_int` of a constant in two ways, and
  both are wider than what `Meet` deduces from the relation.
- **T1-3.** Unboxing gives types and relations to parameters that hold
  poison.
- **T1-4.** The n-way join of two closure types can name a code that one
  of them does not (the two implementations of join disagree on function
  types).
- **T1-5.** What the creation functions and raw constructors of types
  leave unchecked.
- **T1-6.** The model's grammar of types has a boxed number of every
  naked number kind.
- **T1-7.** More that the operations on types rely on and nothing
  checks: an exact set of closures has a function type for each closure;
  a set of strings is not empty; the kind of an alias to a constant.
- **T1-8.** The `Get_tag` reduction of meet takes a number that is no
  tag to be the tag of no block, and the machine has blocks of any tag.
- **T1-9.** A prover reads `Is_null` of a name of type `Null` as 1, and
  the relation of the model gives 0 when that name holds poison.
- **T1-10.** The type that a projection of a value slot gives its
  closure has an index with no function slot, so the fast path of the
  next projection never answers on it.
- **T2-1.** `Typing_env.replace_equation` on a name that is not bound
  records an equation that the environment cannot see.
- **T2-2.** The invariants of the typing environment are checked only
  when invariant checks are on, and one is not checked at all.
- **T2-3.** `Typing_env.find` in an environment that is bottom answers
  bottom for every name but a symbol with no equation.
- **T2-4.** A level drops an equation whose type is unknown, so `cut`
  can answer an older equation than the type the environment has.
- **T2-5.** After lifting, the environment speaks of a symbol that no
  run of the term as it was has bound.
- **T2-6.** The n-way join gives the handler a variable that only some
  uses define, with its type from those uses; on a run through another
  use nothing has that type.
- **T2-7.** Nothing checks that the names given to
  `Typing_env.add_alias` are bound, and the aliases take an unbound
  variable for one of another compilation unit.
- **T2-8.** `Typing_env.replace_equation` gives a variable a type of any
  kind; only its callers in `Meet_env` keep the kind that the variable
  was defined with.
- **T2-9.** `Typing_env.add_alias` changes the aliases only; that the
  name demoted then has an alias type is left to its two callers in
  `Meet_env`, and the n-way join does not see an alias that has no such
  equation.
- **T2-10.** Simplify keeps `x lxor 0`, `x + 0` and `x * 1` where the
  type of `x` is a set of two or more constants, and rewrites them to
  `x` where nothing is known of `x`; so `-O2` and `-O3`, which give a
  parameter such a type where two branches meet, leave arithmetic that
  the default flags remove.
- **T3-1.** Ordering two variables of the unit being compiled that have
  one binding time is a fatal error in the compiler's resolver; the
  model's resolver is total, and the typing environment's invariant asks
  for distinct binding times.
- **T3-2.** `Aliases.add` on two symbols demotes the first, and the
  result fails the compiler's own `invariant` (run with heavy checks
  only); the model follows `add` and proves the failure.
- **T3-3.** The compiler's `Aliases.invariant` never reads
  `canonical_elements` and never checks the name mode recorded for an
  alias; the model asks for both and proves that `add` keeps them.
- **T3-4.** `Alias_set.choose_opt` answers `None` for a set that has a
  constant and names (dead code; not modelled).
- **T3-5.** The order on name modes is total, and `aliases.ml` treats it
  as partial.
- **T3-6.** Asked for a `Phantom` name, `get_canonical_element_exn`
  prefers a `Phantom` alias to an earlier `Normal` one.
- **T3-7.** `find_best` and the tie in `find_earliest_alias` depend on
  the order of names; the model takes the first of a list.
- **T3-8.** The extra variables of an extension with extra variables are
  not a binding site in the compiler: `With_extra_variables.free_names`
  lists them and `apply_renaming` renames them.
- **T3-9.** `Aliases.add_alias_set` stops with a fatal error where two
  different constants come into one class; in the model that is `None`,
  proved absent when some environment respects the structure and gives
  the elements of the set the value of the name. Not reproduced from
  source.
- **T3-10.** Simplify stops with a fatal error on a static boxed number
  whose contents are a variable: `simplify_or_variable`
  (`simplify/simplify_static_const.ml`) asks for the type of the
  variable at the kind value, and the variable holds the naked number.
  Run from an fexpr at default flags; nothing in the compiler makes such
  a constant today. The model gives the symbol the type of the boxed
  number.
- **T3-11.** In the model a static constant that is not a closure may
  name itself, or a symbol that its group binds later to such a constant
  (the group may be recursive); Simplify stops there with a fatal error,
  the symbol not being bound in the typing environment
  (`simplify/simplify_static_const.ml`, `simplify_static_consts`:
  closure symbols are defined first, the others in order, each after its
  own constant). Run from fexprs at default flags. The model is wider
  than what the compiler accepts; no bug of the compiler is claimed.
- **T4-1.** After the n-way join, the meet defines a second time the
  variables that the join imported from a branch.
- **T4-2.** A meet that reads the type of a name holding poison can
  answer bottom, or add a false alias, with every other premise true.
- **T4-3.** The meet answers bottom for two function types of one slot
  with different code, also where the index does not say that the set
  has the slot.
- **T4-4.** The meet of code ids in the model has no relation of code
  age.
- **T4-5.** The meet deduces a type of blocks or of immediates for a
  scrutinee from a relation, also where the scrutinee holds poison.
- **T4-6.** The reduction from a `Get_tag` relation drops the numbers
  that are not tags of blocks.
- **T4-7.** `expand_head` answers the type of a constant for an alias
  whose canonical element is a constant, whatever the kind of the alias.
- **T4-8.** `is_int_for_scrutinee` says 1 for every constant scrutinee,
  `Null` included, and `Relation.of_const` says 0 for `Null`.
- **T4-9.** `meet_row_like`, where it keeps one case only, does not
  define in its result the variables that the meet of that case defined.
- **T4-10.** `add_concrete_equation_on_canonical_name` takes the name to
  be canonical after the meet that it runs in between.
- **T4-11.** In the model, `Typing_env.add_equations_on_params` may give
  a later parameter that the environment does not know a variable that
  the join made; whether the compiler can is not settled.
- **T4-13.** The term that `-drawfexpr` prints cannot say that a value
  may be null where it writes a kind, in a field of the shape of a
  block or in an arity of more than one component; the parser reads
  every kind of values that is written as never null, and Simplify, run
  on the term read back, can remove a test of null that the compiler
  keeps. The pair validated is then not the pair that the compiler
  made; the compiler's own code is right.
- **T4-15.** The text of a closure that holds a variable of an unboxed
  kind is not read back: the projection, which comes first in the text,
  makes the value slot at the kind of values, and the definition of the
  closure then stops the parser with a fatal error.
- **T4-21.** The model's meet can add equations on the same names again
  in opposite orders: where one environment is joined, the join adds
  the equations that the level has for variables of the target in the
  order in which they were added, and `direct_return` of the meet of
  the disjunction around it then cuts the level of its own scope and
  adds it last added first. The compiler's n-way meet adds them in the
  order of the names each time. So the model can answer bottom where
  the compiler's n-way meet does not, in each of the twenty-four orders
  of the identifiers that the test of the types goes through (the line
  of T7-14).
- **T5-1.** The assumption stated by the n-way join, that every name of
  the source is defined in every joined environment, is false of its
  callers, and nothing checks it.
- **T5-2.** The assumption that every symbol of a joined environment is
  defined in the source is checked of a set passed beside the
  environments, not of the environments.
- **T5-3.** The assumption that the binding times of the variables of
  the source are coherent across the joined environments is not checked.
- **T5-4.** The environment of a handler holds the types of constants
  lifted in other branches, which are not true on a path that did not
  define the symbol.
- **T5-5.** The n-way join gives a variable that only some uses define
  the join of its types at those uses, and nothing records that it means
  nothing at the others.
- **T5-6.** The results of the n-way join depend on the order of names.
- **T5-7.** A join through a name that holds poison is not an upper
  bound.
- **T5-8.** `Analysis.fold_variables_created_at_join` also visits
  variables that the join did not create.
- **T5-9.** The n-way join makes a variable of a kind for simples whose
  kinds it does not look at.
- **T5-10.** The n-way join takes every variable that a joined
  environment knows to have a value there, where it puts a relation back
  on a scrutinee and where it drops a bottom type.
- **T5-11.** Whether two equal constant immutable arrays are one symbol
  depends on their debugging information, which the text of a term has
  not, and on how the copies were made.
- **T5b-1.** The join of types has no bound on its depth that is always
  reached; on two inputs that are an alias and a type in turn, both the
  n-way join and the binary join overflow the stack.
- **T5b-2.** The n-way join gives an `is_null` variable to a join one of
  whose inputs is not null, and so says that `Is_null` of each value of
  that input is 0, poison included.
- **T5b-3.** The n-way join of types takes what `Expand_head` answers to
  be of the kind of the types joined, and nothing gives it that.
- **T5b-4.** The body of a `try` at the top of a unit is at toplevel
  for Simplify and not for the model.
- **T6-1.** A prover answers what the environment says of a name, which
  is not true of a name that holds poison (one case reaches a number
  that is not poison).
- **T6-2.** The grammar of the model has a type of naked numbers at the
  kind of the immediates, of which four provers answer that it has no
  value.
- **T6-4.** A value slot of the model has no kind;
  `meet_project_value_slot_simple` answers `Invalid` where the type of
  the slot is of another kind than that of the slot.
- **T6-7.** Unboxing a continuation parameter adds, at a use, a load
  that names the tag and the size, justified only by the subkind of a
  binder.
- **T7-1.** No function of the compiler makes the allocation mode
  `Local` of a type; the converter of types refuses it.
- **T7-2.** `Equal_types_for_debug.equal_type` overflows the stack on a
  type and itself when an extension on a name gives it a type with an
  extension on that name; the tests do not call it on such types.
- **T7-3.** The binary and the n-way join of two types keep different
  things (aliases, relations, extensions); neither is always the more
  precise.
- **T7-4.** The meet of a type with an `is_null` variable and a type
  that settles it forgets the variable.
- **T7-5.** The meet of a variant with a mutable block keeps the
  immediates. A loss of precision.
- **T7-6.** A meet forgets an inverse relation of a naked immediate. A
  loss of precision.
- **T7-7.** With the default setting of `-flambda2-join-sharing`, a join
  depends on whether equal types, or parts of them, are one value in
  memory; with the structural setting it does not.
- **T7-8.** Types that Simplify computes mention variables of kind
  `Rec_info` (in the `rec_info` of function types and in `Rec_info`
  heads) that the model's types cannot.
- **T7-9.** The n-way meet of two variants drops what the extension of
  the one case of blocks left says of a symbol: the join that the meet
  of the cases ends with propagates no type of a symbol. The binary
  meet keeps it. Of two closure types with one case, the binary meet
  puts such an equation in the environment and the n-way meet drops
  it. The same where the meet itself makes the equation, of a slot or
  a field that is an alias of a symbol met with a type that says
  more. A loss of precision, shown on built types.
- **T7-10.** Which alias a name keeps after a meet of variants depends
  on the order of names: where one of the two alternatives of a variant
  is bottom, the equations of the level that the meet cut are added
  again in the order of `Name.Map`. The model adds them in the order in
  which they were made. The type answered and the canonical elements are
  the same.
- **T7-11.** The model takes for one two variables that sibling joins or
  meets made: a join makes a variable one greater than the names that it
  is told to avoid, a meet one greater than those that its environment
  knows, and two environments built apart may each have been given the
  same one. The compiler's variables are distinct on every path. A hole
  in the contract of the model's meet and join, not a difference with
  the compiler.
- **T7-12.** The type "any value" has two forms: `any_value`
  (`Unknown`), and the head whose part that is not null is unknown and
  which may be null, with no variable for whether it is, which
  `unknown_with_subkind` and the traversal of types make. Both joins
  turn that head into `Unknown`; `is_obviously_unknown` is false of it,
  so a level keeps an equation that says nothing. The two mean the same;
  no wrong result is known.
- **T7-13.** A meet gives a type that is no alias to a name that the
  same meet has just made an alias
  (`Meet_env.add_concrete_equation_on_canonical_name`), which the
  compiler's debug check for concrete types on canonical names only
  refuses; whether the answer is bottom then depends on the order of the
  identifiers of the variables, which are hashes. The binary meet and
  the n-way meet alike; the model is bottom. T4-10, found by proof,
  reached by a run. From OCaml source (generated programs whose joins
  carry relations between values, not in this tree) the check is
  reached with the n-way join only: the compiler then stops; it was
  not reached with the binary join, and no program printed another
  result at any setting.
- **T7-14.** On a line of five equations on four values the model's meet
  answers bottom and neither meet of the compiler does, in each of the
  twenty-four orders of the identifiers of its four variables: a
  variable on which a case of a variant has an extension becomes an
  alias of the variant itself while an equation is being added to it.
  Bottom is the precise answer. Between the model
  and the n-way meet it is T4-21, found by a trace of the model: the
  model adds equations on the same names again in opposite orders, the
  compiler in the order of the names each time. With either order
  reversed the model is not bottom, sees what the meets see in twelve
  of the twenty-four orders, and loses the contradiction where it
  ignores an equation on a name to which an equation is being added;
  that the compiler loses it there is not established by a trace of
  the compiler.
- **T7-15.** With one case in the place of two on that line, the binary
  meet answers bottom and the n-way meet, and the model, leave the
  environment as it was. A loss of precision of the n-way meet: by a
  trace of the model, and by the source of the n-way meet, read and
  not traced, the join inside the meet finds the contradiction and
  marks its target bottom, and the meet of the variant then adds the
  empty level of that environment to the environment before the
  scope, which drops the mark.
- **T7-16.** What the n-way meet and the n-way join answer depends on
  the identifiers of the variables that they make themselves, which
  are hashes of stamps. With the variables of a program the same at
  every run, in one process, the answers differ from run to run:
  which of two aliases is the canonical element (the one with the
  smaller identifier), and, on a program of five values reduced from
  a generated one, whether the environment is bottom (31 runs of
  100; bottom is the precise answer, so the other is a loss of
  precision). On 38 of 1499 generated programs of joins after meets.
  Repeated executions of one executable on one input printed the
  same in the trials (100 processes of 100). The model does not look
  at the order of names. An inconsistency of the compiler; no wrong
  result is known. Of the family of T7-10 and T7-13, which are about
  the program's own variables. Found by runs that are not in this
  tree; which fold over names decides was not traced.
- **V2-1.** The fexpr printer prints a call of a C function, and a
  method call, without its allocation mode, in a form that the fexpr
  parser refuses.
- **V2-2.** The fexpr printer drops the kinds of the fields of a
  `Make_block`, and the parser reads back `any_value` for each.
- **V2-3.** The conversion gives a meaning to a local allocation read
  after its region has ended.
- **V2-4.** A handler that catches an exception by name is undefined in
  the model, and the predefined exceptions are symbols that no unit
  binds.
- **V2-5.** The argument of a raise that nothing reads is to be replaced
  by 0, but the flow analysis counts it as read in every handler.
- **V2-6.** The term that `-drawfexpr` prints has no mark for a functor,
  for `[@cold]`, for opaque code or for the modes of parameters; the
  parser gives each a default, and Simplify reads them. One case is
  known: for code with no inlining attribute, the decision made at the
  declaration compares the size of a functor with the thresholds for
  functors, and a functor no larger than `small_functor_size` is to be
  inlined; the tool's Simplify, to which the functor is a function,
  compares it with the thresholds for functions and may leave the call.
  The pair validated is then not the pair that the compiler made.
- **V3-2.** Simplify lifts a closed constant out of a function or a
  loop, and shares equal constants: the number of allocations changes.
- **V4-1.** Loopify makes a self tail call a jump without comparing the
  call's return mode and position with the function's.
- **V4-3.** Simplify makes no shortcut through an exception handler and
  keeps its exception parameter though unused, where the machine would
  allow both.
- **X2-8.** The reader of the text of terms stops with a fatal error on
  a variable whose name has a `/` in it: an operator with a `/` in its
  name, an anonymous function when the path of the source has a
  directory, or a variable that the reaper names from others.
- **X2-9.** The text of a term names code of another unit like code of
  its own; the reader stops on a closure over it and reads a direct
  call of it as a call of code that does not exist.
- **X2-10.** The kinds of unboxed 8-bit and 16-bit integers are printed
  in the text of a term and are not in the grammar of the reader.
- **X3-5.** The n-way join of two variants that carry only a relation
  answers unknown and drops the relation it has just joined.
- **X4-1.** The output of Simplify binds the same variable in two
  places: the names stored in a term are not unique.
- **X4-3.** Where every set of closures in what a check converted (the
  unit before Simplify, the unit after and the prefix) had one function
  and nothing there projected a function slot, the test tool left the
  function slot of a closure out of the terms of the model: a term
  after that put a closure at another function slot than the one from
  which its code projects was converted to the same term, and one such,
  written by hand, was validated. Every unit is now converted with
  named function slots.
- **X4-4.** The text of a term writes the code id of a direct call of
  code of another unit by its name alone, and the reader takes it for
  a code id of the unit read: the one of that name if the text binds
  one, and otherwise a new one that nothing binds.
- **X5-2.** The fields of a recursive module are read by immutable
  loads, and `CamlinternalMod.update_mod` overwrites them; a read from
  before the write is reused after it.
- **X5-3.** A load from a string has no coeffect,
  `Bytes.unsafe_to_string` is erased, and `to_cmm` moves the load after
  a write to the bytes.
- **X5-9.** Simplify leaves the module block bound in a handler that
  its own rule does not hold to be at toplevel, where a cold handler
  stays out of line.
- **X5-10.** Simplify shares a lifted block with an equal static
  constant only when the debugging information of their fields is
  equal too; the text of terms has none.
- **X5-11.** The reader of the text of terms gives every parameter of
  code the mode heap; Simplify, run on a term read from text, then
  proves a `local_` parameter a block of the heap and returns the block
  of the stack where the compiler copies it to the heap.
- **X5-12.** The text of a term gives no arity for a call of unknown
  arity, and the reader gives its result one value; where the result
  is an unboxed pair or an unboxed number, Simplify stops with a fatal
  error on the term read.
- **X5-13.** Simplify turns a switch whose arms are all one action into
  that action only when the arms have the same debugging information.
