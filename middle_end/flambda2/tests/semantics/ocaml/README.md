# Units made from OCaml sources

Each `X.ml` here is a small OCaml program. `X.fl` is the Flambda 2 term
that the compiler makes of it for Simplify, and `X.expected` is what the
test tool (`../../tools/flsemantics.ml`) says of that term: it runs
Simplify on it, runs the term before and the term after on the machine
extracted from the Rocq semantics, and asks the extracted validator
whether the term after refines the term before
(`rocq/HACKING.md`, "Testing Simplify against the semantics").

## How a term is made

    sh from_ocaml.sh <ocamlopt> X.ml X.fl

which is, in a directory of its own,

    ocamlopt -nopervasives -nostdlib -stop-after middle_end -c \
      [X.mli] X.ml -drawfexpr-to X.fl

and then

    flsemantics.exe X.fl

The dune rules of this directory do both for every `X.ml` (with the
compiler that builds the tool, which is built from the same sources),
compare the term made with `X.fl` and what the tool prints with
`X.expected`, and fail on a difference:

    dune build --root=. --workspace=duneconf/main.ws \
      @middle_end/flambda2/tests/semantics/runtest

The rules that make a term name the compiler among what they depend
on: a term is made again, and compared, when the compiler changes,
and not only when its source does.

- `-drawfexpr` prints the unit after the conversion from Lambda (closure
  conversion) and before Simplify. The tool runs Simplify itself, with
  the default flags, on what the parser reads back.
- `-nopervasives -nostdlib`: the unit refers to no other unit, so each
  source declares the primitives it uses (`external ( + ) : int -> int ->
  int = "%addint"`). This changes nothing in what the front end makes of
  an expression. The sources of `library/` are written against the
  standard library instead: given the directory of the library as a
  fourth argument, `from_ocaml.sh` compiles a unit with it and refuses
  the unit if its term names a symbol of another unit, so that the
  terms kept there need nothing but themselves either
  (`library/README.md`). The counts and the tables below are of this
  directory and not of that one.
- `X.mli` exports `result` only, in most units. A function that the
  unit exports is a field of the module block, and Simplify makes its
  closure a symbol; the units named `..._exported.ml` are about
  that.
- A function whose inlining an example is about has `[@inline]`: with
  the default flags Simplify inlines only the smallest functions.
- `from_ocaml.sh` refuses a unit of whose term the text loses
  something that the tool cannot give back. It compiles the unit once
  more, for what `-drawflambda` prints of the term before Simplify,
  and fails on three things. The test of being an integer that is for
  any value (`Is_int_generic`), which the text writes as the test for
  variants (label C2-26), unless `generic_is_int.txt` names the unit.
  A call that is a probe: the text does not say that it is one, and
  the program runs the handler of a probe only while the probe is
  enabled (label C2-17). A call of unknown arity, where such a call
  has an unboxed product among its arguments or code of the unit has
  one among its parameters, the product of nothing included: the
  text groups neither the arguments of a call nor the parameters of
  code, and writes nothing for one of no kind, so that a partial
  application or an over-application may be read as a full one
  (labels C2-5 and C2-30). `script/` has a unit for each of these
  refusals and, for most, the same unit without the reason for it,
  with what the script says of each; and two more for the refusal
  that is of `library/` only.
- The sources are formatted by `ocamlformat`, as `make fmt` wants, but
  for those that it does not read, which `.ocamlformat-ignore` names:
  the sources, here, in `script/` and in `library/`, that have an
  unboxed tuple, an unboxed number, a parameter marked `local_` or
  the literal of an immutable array.

## What the conversion does to a unit

The test tool converts the compiler's term to a term of the model
(`../../tools/flsemantics.ml`; it is trusted, and
`rocq/TRUST-LEDGER.md` lists it). Four things it does matter to a
unit made from OCaml.

- **Regions.** Where a block, a boxed number or a closure does not
  escape, the compiler allocates it on the stack, in a region that it
  opens and closes around the expression (`Begin_region`,
  `End_region`). The machine that a validation is about has no
  regions and frees nothing. The conversion drops the opening of a
  region, makes its closing return the unit value, and converts what
  is allocated in it as if it were on the heap. A region that is a
  parameter of a continuation is dropped, with the argument at its
  place in every application of the continuation, and so is a `Let`
  that binds a region to another; a region given anywhere else is
  refused, and so is a call that returns or raises to a continuation
  with such a parameter (`../region_param_ended_by_handler.fl` and
  the three beside it, written by hand: Simplify makes such a
  parameter where it lifts a handler out of another, with
  `-flambda2-match-in-match`). That is right for a unit in which
  nothing allocated in a region is used after the region has ended,
  which the modes of the type checker are there to ensure and
  nothing in the middle end checks. It is assumed of the unit after
  Simplify as of the unit before: a validation does not establish
  that Simplify keeps the uses of a local allocation within the life
  of its region (discrepancy V2-3). The tool looks at that apart, on
  the one run that it makes of each term. The two terms of a unit
  that was compared are converted once more with their regions kept;
  where either has a region that is not a ghost, or a local
  allocation, the tool tries to run each on a machine that has
  regions (`FlambdaRegionsSpec.v`), and prints the line "the check
  of regions: passes before, passes after", or what failed or was
  not run. No verdict depends on that line; `rocq/TRUST-LEDGER.md`, "The
  check of regions", says what it rests on and what it does not say. The
  tool prints the line of 109 of the 546 units of this directory, and it
  says "passes before, passes after" of each. Where that line says
  what the two runs came to, the tool prints one more, "the ends of
  regions ...": of each term, the answer of the model's `ends_check`
  (`FlambdaRegionsEndsSpec.v`) on a certificate that the tool
  guesses, and its answer with the option `eo_own_end` of the check
  set. "accepted" rests on
  `FlambdaRegionsEndsSpec.ends_unit_exit_statement`, which is of any
  certificate: a run of the term that exits through the return
  continuation leaves no plain region open. A poor guess can only
  refuse more. Each term has its own guessed certificate and nothing
  that is proved relates the two answers. With the option the check
  refuses what the conversion from Lambda makes on purpose where
  control leaves several regions at once: a jump to the continuation
  that ends the outermost of them, whose end pops the frames of the
  others (`rocq/DECISIONS.md`, D55). `rocq/TRUST-LEDGER.md`, "The
  ends of regions", says what the line does not say. It says "accepted"
  of both terms of each of the 109 units, with and without the option.
  The regions around a `try`
  are dropped in the same way: the
  handler ends the regions that the body had open when it raised, and
  nothing allocated in one of them is assumed to be used after that.
  The `Let` by which the handler ends the region is kept, bound to
  `%opaque` of the unit value: Simplify does not delete that primitive
  for being unused, so a handler that begins with it is not a
  continuation that only applies another, and the validator must not
  take it for one.
  `match_variant_local.ml` is `match_variant.ml` without the type
  annotation that keeps the argument of its function on the heap; both
  are validated. Other early examples have such an annotation for the
  same reason.
- **Depths.** The depth variables of recursion, and the coercions that
  mention them, are dropped.
- **Poison.** Where two branches meet with a variant, Simplify unboxes
  it and gives the fields that a branch has not got the constant
  `poison`. The conversion makes it the poison constant of the model
  (`FlambdaModel.Poison`), of the kind it carries; the unit is then run
  and compared like any other (`option_join.ml`).
- **Tupled functions.** `let f (a, b) = ...` may be compiled as code
  that takes the two components, with a mark that says a call of
  unknown arity gives it the pair. The model's code has the mark, and
  every call of the model gives such code its first argument as the
  pair; the arguments after it, if there are any, are given as they
  are and make the call an over-application
  (`FlambdaSemanticsSpec.arguments_for_code`). So a call of unknown
  arity is converted as it is (`tupled_function.ml`). A direct call,
  and a call of known arity, give the code the components, which the
  model has not got yet: a direct call of tupled code that the unit
  binds is refused, and so is a call of known arity in a unit that
  binds tupled code. A function with a pair pattern is not always
  tupled code: the code that `nested_tuple.ml` makes takes one
  parameter, the pair, which the call allocates.

## What the verdicts mean

- **validated**: the extracted `validate_unit` accepted the pair. By
  `FlambdaValidateSpec.validate_unit_sound`, the unit after Simplify
  refines the unit before, run on its own from empty environments. This
  is a theorem about this compilation. It rests on what
  `rocq/TRUST-LEDGER.md` lists (the conversion of the compiler's terms
  to the model's) and, for a unit made from OCaml, on the printing of
  `-drawfexpr` and the fexpr parser: what is validated is the term the
  parser reads back, which is the term the compiler simplifies only as
  far as printing and parsing lose nothing.
  What the theorem observes at the exit is the continuation, its
  arguments, here the symbol of the unit, and the heap, which has
  the cells that the run has made; the contents of the module block
  are not part of the statement.
  Where the verdict says that closures and blocks were lifted to
  symbols, the validation accepted at the step that lifts. Where the
  conditions of a lifting hold, that step makes static bindings, in
  the term before, of `Let`s that Simplify lifted (for one of its
  four liftings, `FlambdaLiftSpec.lift_unit`, two of the conditions
  are that the term it is given exits with constants and symbols
  only and has no primitive of the heap). The words name the step,
  and not what it lifted: where a condition of a lifting fails, that
  lifting leaves the term that it was given as it is, and the step
  may still apply its other liftings and rewrites (see the item on
  the line that names a primitive of the heap, below). The theorem
  is the same one. The line has those words too where a `Let` that
  was lifted makes an immutable array of values, which is lifted as
  one that makes a block is (`iarray_literal.ml`).
  Where the verdict ends "at the head of the term after too", the
  validator removed from the groups of static bindings at the head
  of the term after the code and the closures that nothing uses, as
  it removes them from the term before, and then found the two terms
  the same. That the term after refines the term without them is a
  law of a unit run on its own from empty environments
  (`FlambdaUnusedHeadStaticsSpec.v`, under the checks of
  `head_statics_removable`). The theorem is the same one.
  Where the verdict says "the tables of the term after bound in the
  term before", a lifting named static immutable arrays that the
  term after binds and the term before does not, the validation
  bound them at the head of the term before
  (`FlambdaUnusedHeadStaticsSpec.add_head_tables`, under its two
  checks). The line says that and no more. The tables are bound
  for a round of the term before, which makes a `Switch` whose arms
  give a continuation the elements of such an array the load from
  it (`FlambdaMergeTablesSpec.merge_tables`, T1's). The theorem is
  the same one.
  Where the verdict has a second line, "at the second asking", the
  validator refused the pair as it is, and the tool asked
  `validate_unit_knowing` (`FlambdaValidateKnowingSpec.v`): the same
  check of the two terms, each normalised first by rounds in front of
  which the aliases are followed knowing what the symbols are bound
  to. What it accepts is correct of the unit in the same way
  (`FlambdaValidateKnowingProofs.validate_unit_knowing_sound`).
  Where the verdict has the line "(with the calls of the closures that a
  Let makes made direct in the rounds after each lifting)", the liftings
  of the attempt that accepted the pair asked that of their rounds. The
  line records what was asked: it does not say that the rounds changed a
  term with it, nor that the pair is refused without it. The tool makes
  such an attempt where the earlier attempts, those of the same term
  before without that among them, had refused the pair and the term
  before has such a closure. A round that normalises the term after a
  lifting then makes direct an indirect call of a variable that a `Let`
  in sight binds to a closure at a function slot of a set, or to the
  projection of a function slot from such a variable
  (`FlambdaClosureCallsSpec.call_made_closures`, C3's, a refinement
  under all environments; the round is
  `FlambdaValidateSpec.normalise_lifted_round_calling`). What a lifting
  asks of its rounds is part of what the validator is given: the theorem
  keeps the text of its statement, and is proved of the validation with
  the rounds so chosen. The rounds that a lifting chooses so are those
  that follow the first phase, in their two runs, before and after the
  rewrites from the types; the first phase is as it was. The attempts
  of the term before with copies of code for the sets of closures
  that a `Let` makes (below) are made with such rounds from the
  first, where the term before has such a closure. Four units of this
  directory have the line: `captured_closure_given_to_inlined.ml`,
  `maker_of_mutual_inlined.ml`, `maker_of_two_inlined_twice.ml` and,
  under the line of the copies, `local_rec_inlined_twice.ml`.
  Where it has the line "(with what my_closure is known to be from its
  code used in the rounds after each lifting)", the liftings of the
  attempt that accepted the pair asked that too; the line records what
  was asked, as the other does. The tool makes such an attempt where the
  earlier attempts had refused the pair and an asking had found a term
  that the rewrite changes; where the term before has a `Let` of a
  closure at a function slot of a set, its liftings ask for the calls
  too, and the verdict has both lines. The rounds then use, in the body
  of code, what `my_closure` is known to be from its code. In the body
  of code that is no stub and whose `my_closure` is not also a
  parameter, `my_closure` is known to be a closure of that code; a
  variable bound to the projection from it of another function slot is
  known to be a closure of the code that the sets have at that slot,
  where the unit makes a set with the first code at its function slot
  and every such set has the same code at the other slot; and what is
  known of a variable is forgotten under a binder of it. An indirect
  call of a variable so known is made the direct call of its code, and
  the projection from it of a value slot that every such set gives one
  constant, or one symbol that is in scope, is made that constant or
  symbol. A second pass, over the result of the first
  (`FlambdaClosureOfCodeSpec.know_code_slots`, after
  `know_code_closures`), knows one thing more of a variable so known: a
  variable bound to the projection of a value slot from it, where the
  projection names the function slot, is known to be a closure of one
  code where the result of the first pass makes a set with the first
  code at that function slot and every such set gives the value slot a
  variable that is in sight there as a closure of that code
  (`FlambdaSlotsOfCodeSpec.sight_made`: a variable that a `Let` above
  binds to a closure that the `Let` makes, or to the projection of a
  function slot from a variable in sight; no other variable, and no
  symbol, whatever it is bound to). That pass changes calls only: an
  indirect call of a variable that it knows is made the direct call of
  its code. The rewrite is made only of a unit in which each group of
  static bindings that binds a set of closures binds all of it, and is
  the identity of any other
  (`FlambdaClosureOfCodeSpec.know_own_closures`, C3's, whose header has
  the conditions in full: a refinement of a unit run on its own, from
  its imports, and followed by what may follow a unit, which is the
  observers of `FlambdaObserversSpec.v` at the value slots given, run
  from the imports, and not under all environments; the rounds are
  `FlambdaValidateSpec.normalise_lifted_own`). The theorem keeps the
  text of its statement here too. The attempts of the term before
  with copies of code for the sets of closures that a `Let` makes
  (below) are made with these rounds from the first. Three units of this
  directory have the line: `captured_closure_given_to_inlined.ml`,
  `maker_of_mutual_inlined.ml` and, under the line of the copies,
  `local_rec_inlined_twice.ml`.
- **validated with its imports**: the unit names a symbol of another
  unit, here always a predefined exception, and the tool runs both
  terms from the static environment of their imports
  (`FlambdaImportsSpec.imports_statics`), where such a symbol is
  bound to an imported value whose contents are unknown; the two
  lines of the runs say "with its import". The validator accepted
  the pair, and its verdict passes a test with which the theorem
  from the imports is proved: the tool asks
  `FlambdaValidateImportsSpec.verdict_holds_from` of the verdict
  (`verdict_knowing_holds_from` of a verdict of the second asking;
  `validate_unit_sound_verdict_imports_statement` and
  `validate_unit_knowing_sound_verdict_imports_statement`). Three
  verdicts pass it whatever the imports are: the same up to bound
  names, once normalised, once the unused static bindings of the
  term before are removed. The verdict that removes unused static
  bindings from the head of the term after too passes it where no
  import is a name that the groups at the head of the term after
  bind, once the validation has normalised and tidied that term
  (`imp_head_inlined_at_one_call.ml` and
  `imp_head_include_functor_result.ml`). The verdict that lifts
  passes it where three things hold
  (`lifted_holds_from`): that same test of the head of the term
  after, where the checks of the removal of bindings from that head
  pass, whether a binding goes or not; no import is a name that the
  lifted term binds, at each step of the validation whose law from
  the imports asks it (the result of each lifting that is made; the
  term in which what the unit binds its symbols to is used, in each
  round in which it is; both terms whose static bindings are taken
  first, where the verdict is of that comparison); and, where a
  lifting is given a table to put at the head of the unit, no import
  is a symbol that the groups at the head of the unit bind after
  that step, whether the table is put or not. The test computes
  again the terms that the validation
  compares, with rounds that stop where those of the validation
  stop; it is of those terms, not of the two terms as the unit has
  them. It is sufficient and not necessary: a unit that fails it
  may be correct from its imports. The unit after then refines the
  unit before, each run from the same imports, whatever they are.
  No line of a unit that has imports begins "validated:".
- **validated with nothing imported**, followed by **not validated
  with its imports**: the validator accepted the pair and its
  verdict does not pass that test. What is proved is of the two
  units run from the empty static environment, where the imported
  symbol is bound to nothing. It says nothing of the two runs
  shown, and their comparison is a test and no more. Such a unit is
  counted apart. No unit of this directory has these lines with the
  tool of this tree: the six that had them when the verdict that
  lifts passed no test from the imports (`array_length_twice.ml`,
  `never_returns_in_function.ml`, `never_returns_twice.ml`,
  `string_get_checked.ml`, `try_exported.ml`,
  `try_result_matched.ml`) are validated with their imports (C5's
  statement of that verdict from the imports).
- **validated for the term as read, which says never null of a value
  that may be null** (discrepancy T4-13): the tool's verdict is
  "validated", and the tables count the unit apart. The text that
  `-drawfexpr` prints cannot say that a value may be null where it
  writes a kind, which it does in a field of the shape of a block
  and in an arity of more than one component: the parser reads every
  kind of values that is written as one that is never null
  (`middle_end/flambda2/parser/fexpr_to_flambda.ml`). As far as
  this loss goes, the model's term before is the compiler's all the
  same, as the conversion drops these kinds, and it is the term
  after that can differ from the compiler's, where Simplify used
  the kind: told that a value is never null, the tool's Simplify
  can remove a test of null that the compiler's keeps. (The other
  things that the printed term loses are under "What a validation
  does not say".) So what is validated is the tool's Simplify of
  the term as read. That a unit is in this row does not say that
  its term after differs from the compiler's: it says that nothing
  in the tree shows that it does not. Nothing in the tree finds these
  units, as the text is where the kind is lost: the list of the
  first thirteen was made twice outside it, by compiling each unit
  with a compiler whose printer writes the mark (T4) and by
  comparing the term in the compiler's memory with the term read
  back (X5), and the two lists are the same. A third way gives the
  same list: reading what `-drawflambda` prints, which marks such a
  kind, for the places where the text writes a kind
  (`middle_end/flambda2/rocq/status/V2.md` says what is read); every
  unit is looked at so before it enters this directory. The
  fourteenth, `ignored_second_of_pair.ml`, was found so when it
  entered: a function there returns two values, and the arity of
  its result, which the text writes where the code is declared and
  where it is called, has two kinds that may be null. For each of
  the first
  thirteen, the term after of the compiler's own run and the
  tool's, as printed, have since been found the same up to names
  (X5; see "What a validation does not say", which says what that
  comparison leaves out). That comparison is older than the
  fourteenth unit, and nothing is said here of its two terms after.
- **(asked of the term before normalised with the lengths and loads
  of its known immutable arrays read)**, a line under a verdict: the
  verdict is of another function than the validation, and the
  tables count the unit apart. The validation refused the pair at
  every attempt that the tool makes without the reading, the second
  asking included where it makes one, and the tool then asked
  `FlambdaValidateReadingSpec.validate_unit_reading` of the same
  two terms, and `validate_unit_knowing_reading` where it makes a
  second asking. Each makes another term of the term before
  (`before_read`): the term is normalised with, in each round, the
  length of an immutable array that it makes or binds to a symbol
  made a constant, and a load from one at a constant index within
  its bounds made the element
  (`FlambdaArrayLoadSpec.load_known_arrays`). The first function
  then asks `validate_unit` of that term and of the term after; the
  second asks `validate_unit_knowing` of them. What is proved of a
  pair that either accepts is of the term before as the tool
  converted it, not of the term so read, since the reading refines
  the term: the unit after refines the unit before, each run on its
  own (`validate_unit_reading_sound_statement`,
  `validate_unit_knowing_reading_sound_statement`). Acceptance
  gives no more than that. Of the two units run from their imports
  the same is proved where the test of the verdict from the imports
  passes too (`verdict_holds_from` for the first function,
  `verdict_knowing_holds_from` for the second), asked of the term
  that `before_read` gives, of the term after and of that verdict,
  with the arguments of the asking
  (`validate_unit_reading_verdict_imports_statement`,
  `validate_unit_knowing_reading_verdict_imports_statement`); the
  tool makes the test of that term. A unit that has the line is one
  that the validation refuses and in which something was read:
  where nothing is read, each of the two functions is the one that
  it asks, `validate_unit` or `validate_unit_knowing`, of the two
  terms as they are. Two units of this directory have the line,
  `iarray_exported.ml` and `iarray_length_of_any.ml`.
- **(asked of the term before with chosen calls inlined first)**, a line
  under a verdict: the verdict is of another function than the
  validation, and the tables count the unit apart. The validation
  refused the pair at every attempt that the tool makes of the term
  before as it is and of the term before with its known arrays read, the
  second asking included where it makes one, and the tool then asked
  `FlambdaValidateInliningSpec.validate_unit_inlining` of the same two
  terms, and `validate_unit_knowing_inlining` where it makes a second
  asking. Each makes another term of the term before (`before_inlined`),
  from a list of choices: for each choice in turn, the direct calls that
  stand at the places of the choice are rewritten as a round of the
  validation rewrites a call when it is given no code not to inline
  (`FlambdaInlineCallsAtSpec.inline_calls_at`; a place is where a call
  is in the term, or in the body of the code that it stands in), and the
  term is renamed apart. The first function then asks `validate_unit` of
  that term and of the term after; the second asks
  `validate_unit_knowing` of them. Why there is such a function: the
  validation is told the code that its rounds are not to inline, one
  list for the two terms and for every round (a call of that code may
  still be split, where the test of an over-application passes), and
  Simplify chooses by call. It inlines one call of a
  function and leaves another call of it, it unrolls a recursive
  function at a call that is marked for it, and it inlines a function
  everywhere but in the body of a stub. Then neither choice by code need
  give the term after: with that code in the list the term before keeps
  the call that Simplify inlined, and with it out of the list the term
  before loses the call that Simplify left. The validation rewrites both
  terms and may find them the same even so; the tool asks this function
  where it does not.
  The choices are a guess of the tool (`inlining_choices` in
  `middle_end/flambda2/tests/tools/flsemantics.ml`), and nothing proved
  rests on them. The first has the direct calls that the term before
  marks to be unrolled or always inlined (`[@unrolled n]`,
  `[@inlined]`); each choice after it has, for a call marked to be
  unrolled, the calls of the same code that the inlining of the choice
  before brought under it, as many times as the mark says. The first
  three choices also have, at each place where the term before calls
  directly code that the term after still calls directly somewhere and
  does not call at that place, the calls there of that code and of the
  code that the term before binds, that is recursive in neither term and
  that the term after does not call directly; a place here is the term
  outside all code, or the body of a piece of code that is not a stub
  and that the term after binds too. A call of code to itself is not
  taken there. One that is marked is chosen as any marked call is, and
  the inlining leaves it: in the body of code it does not know the code
  of the group that binds that code. There are never more than eight
  choices, whatever a call is marked for.
  What is proved of a pair that either function accepts is of the term
  before as the tool converted it, not of the term with the calls
  inlined, since the inlining refines the term whatever the places are
  (`inline_calls_at_refines_statement`,
  `inlined_in_turn_refines_statement`): the unit after refines the unit
  before, each run on its own (`validate_unit_inlining_sound_statement`,
  `validate_unit_knowing_inlining_sound_statement`). Acceptance gives no
  more than that. Of the two units run from their imports the same is
  proved where the test of the verdict from the imports passes too
  (`verdict_holds_from` for the first function,
  `verdict_knowing_holds_from` for the second), asked of the term that
  `before_inlined` gives, of the term after and of that verdict, with
  the arguments of the asking
  (`validate_unit_inlining_verdict_imports_statement`,
  `validate_unit_knowing_inlining_verdict_imports_statement`); the tool
  makes the test of that term. A unit that has the line is one that the
  validation refuses, with and without the reading, and that one of the
  two functions then accepts. The tool does not ask them where the guess
  gives no choice, nor where the term that `before_inlined` gives with
  nothing lifted is the term before name for name. That test is of the
  two terms as they are written: a term that differs need not have a
  call inlined, as each choice renames the term apart. No unit has this
  line and the line of the reading. Four units of this directory have
  the line in their expected files, which are made at the default flags,
  `f2_one_inlined_one_not.ml`, `rec_local_helper_called_outside.ml`,
  `rec_unrolled.ml` and `stub_called_twice.ml`.
- **(asked of the term before with copies of code for the sets of
  closures that a Let makes)**, a line under a verdict: the verdict is
  of another function than the validation, and the tables count the
  unit apart. The validation refused the pair at every attempt of the
  three kinds that the tool makes first (of the term before as it is,
  with its known arrays read, with chosen calls inlined first), and
  the tool then asked
  `FlambdaValidateCopyingSpec.validate_unit_copying` of the same two
  terms, and `validate_unit_knowing_copying` where it makes a second
  asking. Each makes another term of the term before
  (`before_copied`): the term before after the same initial
  preprocessing as `validate_unit` (`demote_stub_calls`,
  `cut_returns`, then `loopify_unit`); then the liftings given, each
  followed by its rounds; then one lifting more, with no round after
  it; and then, at given places, a `Let` that makes a set of closures,
  or a closure of a set, makes it of a copy of its code, the copy
  being bound under code ids that are given for it
  (`FlambdaCopyLetsSpec.with_let_copies`). The first function then
  asks `validate_unit` of that term and of the term after; the second
  asks `validate_unit_knowing` of them. Why there is such a function:
  Simplify gives new code to every set of closures that it simplifies,
  and a set that it meets twice, in the body of a function that stays
  and in a copy of that body that it inlines, has two newer versions
  of its code. Where the closure of the copy is lifted to a symbol, a
  lifting of the validation asks a copy for the static set
  (`FlambdaCopyCodeSpec.copy_static_code`). Where it is not, a value
  slot of the set holding a variable for one, the term after has a
  `Let` that makes its closures of code under a new id, where the
  term before has one code.
  The liftings, the places and the code ids are a guess of the tool,
  and nothing proved rests on them. The tool makes the attempts of
  this kind where the term after makes by a `Let`, in its code too, a
  set of closures of code that the term before does not bind. It
  guesses liftings in turn, as for any asking, until one that lifts
  nothing has copies beside it that change the term. The copies are
  guessed from the term that the liftings before that one and their
  rounds have made: each of its `Let`s that makes a set of closures
  is paired, by the place where it stands (outside code, or in the
  body of a piece of code) and in the order of the terms, with a
  `Let` of the term after that makes a set with the same function
  slots and, for each function, the same code or code that the term
  before does not bind; a `Let` whose code is not the same is to get
  the code ids of the term after. Where no such lifting is found, no
  function is asked. The attempts of this kind are made once, with
  what `my_closure` is known to be from its code used in the rounds
  after each lifting and, where the term before has a closure of the
  sort that those rounds know, with the calls of the closures that a
  `Let` makes made direct there: the lines of those rounds stand
  under this one. For these attempts the lists of the code not to
  inline have, beside the code that the term after calls directly,
  the code that it is a newer version of, and what that is a newer
  version of, eight times at most: the term before may call the
  older code where the term after calls the newer.
  What is proved of a pair that either function accepts is of the
  term before as the tool converted it, each unit run on its own from
  empty environments: the unit after refines the unit before
  (`validate_unit_copying_sound_statement`,
  `validate_unit_knowing_copying_sound_statement`), since
  `before_copied` refines the term before so run
  (`before_copied_program_refines_statement`). Nothing is said under
  all environments, of which a lifting and a copy are no
  refinements. A copy is made only where its tests pass
  (`FlambdaCopyLetsSpec.copy_let_code`); where one fails the term is
  as it was, and the function asked answers of that term. Of the two
  units run from their imports the same is proved where two tests
  pass, that of the liftings which `before_copied` applies
  (`before_copied_holds_from`) and that of the verdict of the
  function asked, on the term that `before_copied` gives
  (`validate_unit_copying_verdict_imports_statement`,
  `validate_unit_knowing_copying_verdict_imports_statement`); the
  verdict of a unit with imports then begins "validated with its
  imports", and "validated with nothing imported" where one of them
  fails. No unit has this line and the line of the reading or that of
  the inlining. One unit of this directory has the line in its
  expected file, which is made at the default flags,
  `local_rec_inlined_twice.ml`.
- **the same, not validated**: the two runs came to the same result,
  which is a test of one run and no proof. The validator could not make
  the two terms the same; the table says what it is missing.
  "The same", on the line that compares the two runs, is of what the
  tool observes of the two exits, and not of the two exits with
  their heaps. A value that is a location is observed as what its
  cell holds in the heap of the exit, a block with its tag or an
  array, 50 levels deep from each value that the run returns.
  So what is observed can be the same for an alias and for a copy,
  where the two cells hold the same at the exit, and for a cell and
  for a value with no identity that holds the same: a unit whose
  run before returns a mutable array and whose run after returns an
  immutable one with the same elements can show equal returns. The
  line does not say which of the cells above it a cycle comes back
  to, nor what is below the 50 levels. And where the printing
  enters a cycle at a block that is a cell in one run and a value
  with no identity in the other, the second may be unfolded
  further, and the line may then say that the two runs differ
  (`rocq/TRUST-LEDGER.md`, "The conversion of terms", has this in
  full). The verdict of the validation does not rest on what is
  observed; that line does.
- **(the term before and the term after have a primitive of the heap:
  each of the four liftings to symbols of the validation lifts nothing
  in a term that has one)**, a line under "not validated", and under a
  verdict that names a lifting. Each of the four liftings of the
  validation lifts nothing in the term that it is given where that term
  has a primitive of the heap (`FlambdaHeapFreeSpec.expr_free_of_heap`;
  a limit pending decision D40, and not a property that a unit is meant
  to have). The line says which of the two terms as converted, the
  prefix in front where there is one, has such a primitive: both, as
  above; or "the term before has a primitive of the heap, and the term
  after none"; or "the term after has a primitive of the heap, and the
  term before none". There is no line where neither has one. It is a
  fact of those two terms. It is not a fact of what each lifting was
  given, as a rewrite before a lifting may have removed the primitive;
  and under "not validated" it does not say why the unit is not
  validated, as the validation may refuse a unit for another reason.
  Under "not validated" it is printed where the run before that is shown
  is not undefined. Under a verdict "the same once closures and blocks
  are lifted to symbols", by whichever asking gave it and whatever the
  run before is, the line ends "; the line above names the step at which
  the validation accepted": that verdict is the step of the validation
  that accepted, which has other rewrites than its liftings, and a unit
  that has a primitive of the heap in both terms may be validated at it
  with nothing lifted. A unit that has a primitive of the heap and is
  validated at another step has no such line (`array_literal.ml`,
  `ref_counter.ml`): the theorem is the same one as for any unit
  validated at that step. Two units of this directory have the line in
  their expected files, under "not validated":
  `made_array_checked_store_in_loop.ml` and
  `pair_of_unknowns_after_inlined_pair.ml`.
- **related once lifted**, a line under "not validated" (it awaits
  the human's choice, decision D43): the validator refused the
  pair for every list of calls that the tool tries, and the tool
  then asked a second check of the same two terms,
  `FlambdaValidateRelatedSpec.validate_unit_related`. That check
  makes each lifting without two tests that a lifting of the
  validation makes of the term that it is given: that it exits
  with symbols and constants only
  (`FlambdaLiftSpec.exits_plain_knowing`), and that it has no
  primitive of the heap (`FlambdaHeapFreeSpec.expr_free_of_heap`).
  It keeps every other condition of a lifting
  (`FlambdaLiftObservedSpec.with_lifting_observed`), and then
  compares the two terms as the validator compares two lifted
  terms. The line of the tool names the first test, "without the
  liftings' symbols-and-constants-only exit test", and where either
  of the two terms has a primitive of the heap it names the second
  too, "and without their test of the heap". The line is printed
  for a unit that stores as for another, where the check accepts
  the pair; it is not printed of every unit that stores.
  What is proved of a pair that it accepts
  (`validate_unit_related_sound_statement`) is not that the unit
  after refines the unit before. It is one-way, and it is this:
  an exit of the unit before is answered by an exit of the unit
  after that no observer of a class tells apart from it
  (`FlambdaObserversSpec.worlds_observed`), and the unit after
  ends only if the unit before exits or goes wrong. Of a unit
  before that goes wrong it says nothing. It is no claim that the
  two units are equivalent under what may follow them.
  The class is the whole of `FlambdaObserversSpec.observer` at
  the value slots that the check was given to remove, and no
  smaller one: a way out, or a handler with no captured
  environment, in which there is no static binding, which names
  no symbol and no code id, and which projects none of those
  value slots (there are none unless the line ends "unused value
  slots removed"). Such a handler may read, store, allocate,
  project and call; one that makes an immutable array is in the
  class. So the line says nothing of the unit linked with another
  unit, which names its symbols.
  What two such exits have in common
  (`worlds_observed_way_out_statement`): they are by the same
  continuation; where a value of either is a constant or a
  location, the value at the same place of the other is the same;
  and the two heaps bind the same locations in the same order.
  What an allocated value of an exit holds, and what the cells of
  the heaps hold, need not be equal: an exit of the unit after
  may carry a symbol where the exit of the unit before carries
  the value that the symbol is bound to.
  "validated" requires equal exit values and equal heaps; this
  line does not require those equalities, so says less in that
  respect. Its observer comparison also uses the static
  environment at each exit. No implication between the two
  relations, or inclusion between the two checks' acceptances, is
  claimed.
  The head of the line says what the check was given. "with its
  imports": a list of imports that is not empty; the statement is
  of the two terms run from the static environment of those
  imports, and the function tests what the laws of its steps ask
  of them. "given the .cmx of N of them": the two terms are the
  two units each with the prefix made from those files in front
  of it, and the statement is of those two terms; that the files
  are right about the units linked, and how a linked program
  initialises them, are the limits that `rocq/TRUST-LEDGER.md`
  states of a verdict with a prefix. "with no remaining imports":
  the prefix left no import.
  An observer names no symbol, hence no imported exception
  symbol: it cannot compare a value with a specified predefined
  exception by naming that exception. No theorem that the two
  exits carry the same named predefined exception, alone or in a
  block, is claimed here. The check compares the terms, including
  their symbol names; that is not a proved same-exception
  guarantee for its answer.
  Such a unit is not validated and is counted so: no count of
  this file, of the census or of the tool changes for it. The
  units that have the line are in a table of their own below,
  those whose check was given imports apart from those with
  none. Whether a relation of this kind is what a verdict is to
  mean is the human's to choose, between this form and one that
  relates the values of the two exits and the two heaps
  structurally (decision D43; `rocq/TRUST-LEDGER.md`, "A check
  that does not establish `refines`"); until then these units are
  added to no count of units validated.
  The tool asks the check only under the line "not validated":
  where the run of the term before that is shown is reported
  undefined the line is another, and the check is not asked.
  That the run shown is not undefined does not establish that
  the unit before never goes wrong. The check is asked with no
  value slot to remove first. Of an import of which nothing is
  known (the class `Import_unknown`): where a run executes an
  inspection of such an import it goes wrong in the model, and
  the statement says nothing of that source run. The way-out
  consequence excludes a constant or a location opposite such an
  import; the full observer relation is still what is concluded.
- **stores into arrays by the array kind they carry (no part of the
  verdict): before ...; after ...**, a line after what the tool prints
  of the verdict: how many stores into arrays each of the two terms
  given to the validation has at each array kind (immediates; values
  that the collector may ignore; values), each term counted with the
  prefix made from the `.cmx` files of other units in front of it where
  there is one. The tool prints the line only where those numbers differ
  between the two terms. It is no part of the verdict. The validation
  writes the array kind of a store in one form where it folds constants
  (`FlambdaNormaliseSpec.fold_named`), the one that the set kind of the
  store determines (`FlambdaPrimitiveSpec.array_kind_of_set_kind`;
  decision D56): the machine does not read that kind, and a verdict does
  not say that the kinds agree in the two terms. Nor does the line: it
  gives numbers and no correspondence of the stores. Exchanging the
  kinds of two stores, and nothing else, leaves the numbers the same,
  and the numbers can differ merely because Simplify has removed a store
  or copied one; that no line is printed does not say that the kinds
  agree. `rocq/TRUST-LEDGER.md`, "The conversion of terms", has it. Two
  units of this directory have the line in their expected files:
  `made_array_checked_store.ml` and `made_array_store_of_integer.ml`.
- **nothing to validate, as the term before is undefined**: the machine
  has no step somewhere in the run of the term before, so anything
  refines it and a validation would say nothing. The compiler runs
  these programs: the model is what lacks something, and the table
  says what. Such a unit is not counted as validated; where the
  validator accepts one all the same, the tool says so on a line of its
  own.
- **not checked**: the conversion refused the unit, as it has something
  the model has not got, or something that the conversion does not
  make though the model has it. The line of the tool says "the model
  has not got" of both. The machine runs the primitives of the heap
  (`FlambdaSemanticsSpec.exec_named`, at a `Let`), and the conversion
  makes the four of `FlambdaPrimitiveModel.heap_primitive` in these
  cases: a block of values marked `Mutable` or `Immutable_unique`
  that has a field, with as many kinds of fields as arguments, where
  the kind of every field says no more than that it is a value (it
  is a cell of the heap; a block marked `Immutable` is a value with
  no identity, as it was); an array marked `Mutable` whose elements
  are values and that has an element; and a store into a block of
  values or into an array whose elements are values, but for the
  stores that it refuses. It refuses, and approximates nothing: a
  store marked `Initialization`, into a block or into an array; a
  store into a block of floats or a mixed block; an array of floats,
  of naked numbers, of vectors, of masks or of unboxed products, made
  or written; a store, into an array of the kind of any values, at a
  set kind of immediates or of values that the collector may ignore;
  and the three copies of a block or of an array (`Duplicate_block`,
  `Duplicate_array`, `Obj_dup`). The mode of an assignment is
  dropped. A load keeps the mark that the term gives it: a load not
  marked mutable of a block made mutable has no value in the model
  (decision D52, open), nor has an array load marked immutable of a
  mutable array (decision D49); a run that comes to one is
  undefined. Where the run before is undefined, a validation says
  nothing of the unit.
  `rocq/TRUST-LEDGER.md`, "The conversion of terms", has the cases
  and what is trusted of each.

## The examples

Each source begins with a comment that says what it shows and what
Simplify does to it. Steps are those of the machine, before and after
Simplify. The names in brackets are the streams whose work in progress
is the rewrite that is missing.

546 sources: 455 validated; 2 validated when asked of the term before
with its known immutable arrays read, which is another function than the
others are asked of; 4 validated when asked of the term before with
chosen calls inlined first, which is another function again; 1 validated
when asked of the term before with copies of code for the sets of
closures that a `Let` makes, which is another again; 14 validated for
the term as read, which says never null of a value that may be null
(T4-13); 59 validated with their imports; none validated with nothing
imported and not with its imports; 6 compared on one run and not
validated, 3 of them with the line of the second check (1 of those with
imports); 2 undefined in the model; 3 not checked.

### Validated

| Source | Steps | Source | Steps |
|--------|-------|--------|-------|
| `add3_exported.ml` | 9, 6 | `join_two_names_of_symbol.ml` | 24, 8 |
| `add_zero.ml` | 8, 2 | `kept_local_mutual_to_inlined.ml` | 145, 97 |
| `alias_kept_or_rebuilt.ml` | 49, 28 | `labelled_arguments.ml` | 13, 3 |
| `apply_exported.ml` | 8, 4 | `labelled_in_every_order.ml` | 108, 94 |
| `array_literal.ml` | 8, 3 | `labelled_optional.ml` | 38, 6 |
| `attr_inlined_never_at_call.ml` | 27, 20 | `late_three_values_interleaved.ml` | 430, 382 |
| `attr_local_in_match.ml` | 87, 59 | `least_join_table.ml` | 30, 35 |
| `attr_local_two_uses.ml` | 41, 27 | `len_compare_two.ml` | 41, 33 |
| `bitwise_unknown.ml` | 14, 7 | `len_of_constants.ml` | 32, 1 |
| `block_in_arm_of_unknown.ml` | 19, 7 | `len_of_unknown_twice.ml` | 18, 12 |
| `bool_not_unknown.ml` | 8, 3 | `list_constant.ml` | 39, 25 |
| `bool_ops.ml` | 47, 1 | `list_head.ml` | 41, 3 |
| `boxed_constant_made.ml` | 12, 4 | `list_heap.ml` | 54, 33 |
| `boxed_constant_twice.ml` | 16, 2 | `list_last.ml` | 76, 41 |
| `bswap_constants.ml` | 29, 2 | `list_sum.ml` | 61, 42 |
| `call_of_call_tested.ml` | 39, 42 | `load_through_join.ml` | 68, 37 |
| `call_result_passed_on.ml` | 79, 31 | `local_argument_partial_application.ml` | 102, 59 |
| `call_result_tested.ml` | 78, 73 | `local_block_matched.ml` | 93, 66 |
| `call_result_unused.ml` | 79, 50 | `local_block_matched_argument.ml` | 93, 66 |
| `capture_fun_and_var_run.ml` | 24, 20 | `local_call_exported.ml` | 8, 4 |
| `capture_fun_bound_in_handler_run.ml` | 31, 26 | `local_call_used.ml` | 10, 3 |
| `capture_rec_self.ml` | 9, 6 | `local_closure_applied_mutual.ml` | 158, 82 |
| `capture_two_funs.ml` | 15, 12 | `local_closure_given_partially.ml` | 58, 40 |
| `captured_closure_given_to_inlined.ml` | 104, 59 | `local_function.ml` | 29, 5 |
| `char_code.ml` | 13, 3 | `local_loop_inlined.ml` | 66, 41 |
| `char_comparisons.ml` | 56, 36 | `local_rec_below_a_let_and_in_an_arm.ml` | 58, 37 |
| `char_count_in_string.ml` | 654, 304 | `local_rec_in_tail_loop.ml` | 173, 108 |
| `char_digit_value.ml` | 87, 60 | `loop_exit_inlined.ml` | 82, 36 |
| `char_hex_digit_value.ml` | 144, 80 | `loop_for_local_pair_each_turn.ml` | 1749, 1336 |
| `char_match.ml` | 48, 1 | `loop_function_in_kept.ml` | 85, 50 |
| `char_of_int_in_loop.ml` | 269, 184 | `loop_function_twice.ml` | 149, 85 |
| `char_parse_number.ml` | 111, 92 | `loop_local_mutual_both_in_loop.ml` | 261, 165 |
| `char_ranges.ml` | 183, 94 | `loop_result_passed_round.ml` | 190, 110 |
| `char_ranges_inlined_constants.ml` | 112, 4 | `made_array_length.ml` | 15, 8 |
| `char_uppercase.ml` | 63, 38 | `made_array_length_in_two_calls.ml` | 57, 38 |
| `clamp.ml` | 49, 1 | `made_array_store_of_integer.ml` | 16, 9 |
| `clos_partial_local.ml` | 186, 160 | `maker_inlined_and_exported.ml` | 21, 10 |
| `closed_mutual_recursion.ml` | 66, 35 | `maker_of_mutual_inlined.ml` | 70, 40 |
| `closure_argument_exported.ml` | 12, 7 | `maker_of_two_inlined_twice.ml` | 311, 186 |
| `closure_argument_used.ml` | 14, 3 | `match_constant_table.ml` | 28, 22 |
| `closure_capture.ml` | 31, 4 | `match_guard.ml` | 52, 1 |
| `closure_captures_argument_inlined.ml` | 27, 21 | `match_int_dense.ml` | 74, 50 |
| `closure_captures_field_once.ml` | 24, 19 | `match_int_intervals.ml` | 235, 90 |
| `closure_from_pair_called.ml` | 44, 15 | `match_local_mutual_both_in_arm.ml` | 132, 85 |
| `closure_from_pair_exported.ml` | 9, 4 | `match_of_inlined_match.ml` | 68, 38 |
| `closure_from_pair_used.ml` | 16, 3 | `match_of_match.ml` | 86, 54 |
| `closure_in_arm_of_unknown.ml` | 23, 11 | `match_pair_option_built.ml` | 65, 32 |
| `closure_in_tuple.ml` | 33, 5 | `match_tuple_of_variants.ml` | 102, 71 |
| `closure_pair_shares_capture.ml` | 45, 40 | `match_unknown.ml` | 18, 10 |
| `closure_uses_unit_function_only.ml` | 39, 30 | `match_variant.ml` | 66, 7 |
| `compare_ints.ml` | 113, 1 | `match_variant_local.ml` | 72, 7 |
| `compare_order.ml` | 60, 35 | `max_exported.ml` | 8, 4 |
| `compare_signed.ml` | 31, 1 | `min_max.ml` | 25, 1 |
| `compare_tested.ml` | 15, 6 | `module_alias.ml` | 34, 5 |
| `compare_unknown.ml` | 9, 4 | `module_function_in_code.ml` | 58, 31 |
| `compose.ml` | 29, 4 | `mom_three_levels.ml` | 126, 77 |
| `compose_chains.ml` | 142, 115 | `mom_variant_to_variant.ml` | 50, 24 |
| `compose_exported.ml` | 8, 4 | `mutual_alias_exported.ml` | 78, 40 |
| `constant_exported.ml` | 8, 4 | `mutual_capture_exported.ml` | 123, 72 |
| `constant_folding.ml` | 31, 1 | `mutual_recursion.ml` | 130, 70 |
| `conv_truncate.ml` | 45, 1 | `nativeint_arith.ml` | 25, 8 |
| `count_after_flag.ml` | 275, 152 | `negate.ml` | 10, 5 |
| `data_closure_captures_string.ml` | 38, 33 | `nest_list_of_options.ml` | 177, 109 |
| `data_list_of_strings.ml` | 82, 61 | `nest_matrix_three_columns.ml` | 108, 81 |
| `data_option_of_string.ml` | 56, 41 | `nested_heap.ml` | 50, 29 |
| `data_string_in_module_and_functor.ml` | 37, 33 | `nested_ignored_tests.ml` | 37, 28 |
| `data_string_returned_through_join.ml` | 32, 28 | `nested_record.ml` | 23, 3 |
| `data_variant_argument.ml` | 89, 65 | `nested_tuple.ml` | 24, 4 |
| `either_heap.ml` | 58, 37 | `nomark_local_mutual.ml` | 145, 97 |
| `empty_array_in_record.ml` | 14, 8 | `not_inlined.ml` | 21, 17 |
| `empty_array_length.ml` | 9, 5 | `num_conversions.ml` | 27, 19 |
| `empty_array_tested.ml` | 14, 8 | `option_bind.ml` | 89, 1 |
| `empty_arrays_of_two_types.ml` | 12, 8 | `option_constant.ml` | 39, 25 |
| `empty_immutable_array.ml` | 9, 5 | `option_constant_loop.ml` | 175, 114 |
| `empty_type_refuted.ml` | 21, 6 | `option_constant_loop_fun.ml` | 379, 258 |
| `enum_to_int.ml` | 19, 1 | `option_default.ml` | 32, 3 |
| `exclave_after_local_work.ml` | 201, 177 | `option_heap.ml` | 47, 29 |
| `exclave_pair.ml` | 177, 157 | `option_in_pair.ml` | 67, 31 |
| `exported_after_effect.ml` | 15, 6 | `option_inlined.ml` | 36, 12 |
| `exported_and_used.ml` | 14, 6 | `option_join.ml` | 55, 29 |
| `exported_apply.ml` | 18, 9 | `option_map.ml` | 70, 4 |
| `exported_arms_after_let.ml` | 26, 21 | `option_matched_twice.ml` | 82, 43 |
| `exported_bool.ml` | 11, 7 | `optional_between_positional.ml` | 258, 178 |
| `exported_branch_calls_twice.ml` | 17, 11 | `optional_default_uses_earlier_argument.ml` | 200, 123 |
| `exported_branch_fun_const.ml` | 17, 12 | `optional_erased_for_higher_order.ml` | 172, 118 |
| `exported_branch_option.ml` | 14, 9 | `optional_given_omitted_none_some.ml` | 98, 69 |
| `exported_branch_pair.ml` | 15, 10 | `or_in_list_positions.ml` | 113, 69 |
| `exported_branch_then_const.ml` | 14, 9 | `or_null_known.ml` | 23, 1 |
| `exported_branch_then_fun.ml` | 14, 8 | `or_null_unknown.ml` | 16, 6 |
| `exported_branch_two_funs.ml` | 17, 11 | `or_pattern.ml` | 49, 1 |
| `exported_capture.ml` | 16, 5 | `or_two_variables_swapped.ml` | 61, 47 |
| `exported_chain.ml` | 14, 10 | `over_application.ml` | 19, 3 |
| `exported_compose_kept.ml` | 17, 13 | `over_application_three_stages.ml` | 101, 92 |
| `exported_const_list.ml` | 10, 7 | `over_application_unknown_function.ml` | 61, 58 |
| `exported_curried.ml` | 14, 6 | `over_inlined_maker_run.ml` | 23, 3 |
| `exported_fun_branch_fun.ml` | 17, 11 | `over_rec.ml` | 84, 52 |
| `exported_fun_given.ml` | 9, 7 | `overflow_wrap.ml` | 19, 1 |
| `exported_fun_in_arms.ml` | 20, 12 | `pair_built.ml` | 43, 21 |
| `exported_fun_in_one_arm.ml` | 17, 12 | `pair_computed.ml` | 45, 29 |
| `exported_fun_in_option.ml` | 9, 8 | `pair_constant_arm.ml` | 53, 23 |
| `exported_fun_returned.ml` | 9, 7 | `pair_constant_arm_toplevel.ml` | 31, 12 |
| `exported_function.ml` | 14, 6 | `pair_constants.ml` | 32, 23 |
| `exported_funs_given.ml` | 10, 10 | `pair_join.ml` | 31, 28 |
| `exported_funs_in_arms_used.ml` | 23, 14 | `pair_loop_in_loop.ml` | 333, 173 |
| `exported_funs_in_pair.ml` | 14, 11 | `pair_read_one_call_inlined.ml` | 31, 23 |
| `exported_list.ml` | 9, 5 | `pair_turned_in_loop.ml` | 192, 65 |
| `exported_local_fun.ml` | 8, 4 | `pair_two_constants.ml` | 25, 9 |
| `exported_local_never.ml` | 9, 7 | `pairs_for_a_suit.ml` | 20, 9 |
| `exported_match_pair_then_fun.ml` | 13, 7 | `param_copy_to_heap.ml` | 193, 171 |
| `exported_match_then_fun.ml` | 21, 12 | `partial_across_join.ml` | 148, 105 |
| `exported_match_three_funs.ml` | 33, 19 | `partial_application.ml` | 23, 3 |
| `exported_nested_branch.ml` | 21, 13 | `partial_exported.ml` | 13, 6 |
| `exported_nested_pair.ml` | 10, 7 | `partial_in_a_pair.ml` | 90, 26 |
| `exported_option_fun_in_arm.ml` | 18, 10 | `partial_one_argument_at_a_time.ml` | 116, 57 |
| `exported_pair_fun_in_arms.ml` | 21, 14 | `partial_two_to_six_parameters.ml` | 156, 112 |
| `exported_record.ml` | 23, 9 | `phys_equal_unknown.ml` | 9, 4 |
| `exported_record_of_funs.ml` | 14, 11 | `physical_equality.ml` | 32, 1 |
| `exported_record_update.ml` | 14, 6 | `pipeline.ml` | 23, 5 |
| `exported_records_in_arms.ml` | 21, 14 | `poly_abbreviation_patterns.ml` | 117, 68 |
| `exported_some_again.ml` | 26, 6 | `polymorphic_variant.ml` | 77, 4 |
| `exported_string_length.ml` | 8, 5 | `power.ml` | 147, 99 |
| `exported_submodule.ml` | 14, 8 | `project_record_two_levels.ml` | 43, 18 |
| `exported_table.ml` | 13, 8 | `project_variant_argument.ml` | 27, 11 |
| `exported_twice_kept.ml` | 14, 10 | `projections_exported.ml` | 11, 7 |
| `exported_two_branches.ml` | 23, 15 | `rec_helper_among_three.ml` | 29, 10 |
| `exported_two_functions.ml` | 22, 10 | `rec_helper_called_outside.ml` | 27, 13 |
| `exported_two_somes.ml` | 11, 5 | `record_exported.ml` | 11, 7 |
| `exported_two_tests.ml` | 26, 17 | `record_of_functions.ml` | 51, 10 |
| `exported_two_values.ml` | 18, 1 | `record_project.ml` | 37, 8 |
| `exported_variant.ml` | 17, 6 | `record_update.ml` | 27, 4 |
| `fact_exported.ml` | 8, 3 | `ref_counter.ml` | 23, 23 |
| `factorial.ml` | 70, 46 | `ref_local.ml` | 132, 79 |
| `fct_applied_twice_same_argument.ml` | 92, 89 | `ref_local_in_for_loop.ml` | 296, 180 |
| `fct_local_mutual_both_to_inlined.ml` | 139, 96 | `ref_straight.ml` | 13, 4 |
| `fct_of_functor.ml` | 97, 93 | `reraise_in_kept_function.ml` | 47, 35 |
| `fib.ml` | 2041, 1417 | `result_join.ml` | 64, 33 |
| `field_twice_inlined.ml` | 27, 17 | `result_map.ml` | 44, 3 |
| `first_class_module.ml` | 22, 4 | `returned_closure_of_pair.ml` | 71, 50 |
| `flag_and_count_in_loop.ml` | 156, 90 | `same_constant_after_loop.ml` | 15, 7 |
| `flag_and_test_in_loop.ml` | 157, 99 | `same_field_chain_twice.ml` | 29, 18 |
| `flag_cleared_once_in_while.ml` | 217, 133 | `same_int64_sum_twice.ml` | 29, 14 |
| `flag_guards_update.ml` | 240, 130 | `same_projection_across_call.ml` | 25, 13 |
| `flag_or_test_in_loop.ml` | 142, 87 | `self_through_closure.ml` | 189, 98 |
| `flag_or_test_then_count.ml` | 291, 177 | `seq_unit.ml` | 13, 2 |
| `flag_read_twice_in_loop.ml` | 441, 269 | `shape2_heap.ml` | 61, 36 |
| `flag_set_in_loop.ml` | 153, 78 | `shape_constant.ml` | 24, 11 |
| `flag_set_in_loop_first.ml` | 153, 78 | `shape_heap.ml` | 54, 30 |
| `flag_tested_in_loop.ml` | 237, 134 | `shared_read_slot_and_field.ml` | 15, 8 |
| `flip_applied_in_full_and_in_part.ml` | 101, 53 | `shared_read_two_closures.ml` | 15, 9 |
| `float_const.ml` | 15, 4 | `shift_edges.ml` | 23, 1 |
| `fold_passes_function_on.ml` | 67, 37 | `shift_unknown.ml` | 17, 7 |
| `for_down.ml` | 79, 40 | `shifts.ml` | 22, 8 |
| `for_if.ml` | 165, 96 | `sibling_given_as_value.ml` | 75, 38 |
| `for_nested.ml` | 171, 95 | `sibling_through_alias.ml` | 74, 38 |
| `for_sum.ml` | 135, 81 | `sign.ml` | 27, 1 |
| `for_two_refs.ml` | 90, 52 | `stack_argument_result_returned.ml` | 27, 20 |
| `fst_snd.ml` | 16, 3 | `stack_pair.ml` | 167, 146 |
| `fun_local_mutual_to_inlined.ml` | 145, 97 | `string_get16.ml` | 10, 5 |
| `fun_local_three_mutual_to_inlined.ml` | 219, 137 | `string_get16_be.ml` | 16, 6 |
| `function_passed_on_by_loop.ml` | 78, 39 | `string_get32.ml` | 23, 11 |
| `functions_in_record_called.ml` | 23, 17 | `string_get_const.ml` | 9, 4 |
| `functor_apply.ml` | 23, 1 | `string_length_const.ml` | 8, 1 |
| `functor_argument_block_in_slot.ml` | 50, 14 | `string_length_unknown.ml` | 9, 5 |
| `gadt_equal_indices_pair.ml` | 53, 40 | `string_sum.ml` | 100, 82 |
| `gadt_existential_match.ml` | 38, 8 | `submodule.ml` | 27, 1 |
| `gadt_known.ml` | 35, 4 | `succ_pred.ml` | 12, 5 |
| `gadt_refutation_clause.ml` | 67, 50 | `sums_of_two_sorts.ml` | 341, 182 |
| `get_after_own_bounds_test.ml` | 56, 37 | `table_and_array_alike.ml` | 24, 20 |
| `guard_falls_to_later_case.ml` | 151, 92 | `tag_known_in_arm.ml` | 91, 52 |
| `guard_on_or_pattern.ml` | 132, 84 | `tail_call_after_region.ml` | 171, 153 |
| `helper_called_through_slot.ml` | 33, 12 | `tail_loop.ml` | 128, 83 |
| `helper_inlined_in_closure.ml` | 28, 17 | `tail_nontail_annot.ml` | 175, 157 |
| `hidden_helper_inlined_twice.ml` | 46, 22 | `test_boolean_twice.ml` | 39, 18 |
| `higher_order.ml` | 32, 5 | `three_arms_same_sum.ml` | 24, 5 |
| `hof_closure_in_record.ml` | 49, 22 | `toplevel_closure_applied_mutual.ml` | 84, 40 |
| `hof_option_of_closure.ml` | 125, 47 | `tree_root.ml` | 90, 4 |
| `hof_two_closures_through_join.ml` | 65, 54 | `try_no_raise.ml` | 14, 5 |
| `iarray_element_read_twice.ml` | 25, 19 | `try_returned_exported.ml` | 11, 9 |
| `iarray_in_function.ml` | 21, 14 | `tuple_match.ml` | 75, 1 |
| `iarray_length_of_any_not_inlined.ml` | 19, 14 | `tuple_return.ml` | 20, 5 |
| `iarray_literal.ml` | 11, 5 | `tuple_swap.ml` | 19, 4 |
| `iarray_literal_measured_one_call_inlined.ml` | 27, 18 | `tuple_through_kept_identity.ml` | 19, 12 |
| `iarray_measured_one_call_inlined.ml` | 31, 23 | `tupled_function.ml` | 21, 2 |
| `iarray_of_variable_measured_one_call_inlined.ml` | 32, 24 | `tupled_partial_application.ml` | 91, 67 |
| `iarray_of_variable_read_one_call_inlined.ml` | 32, 24 | `twice_inlined.ml` | 19, 6 |
| `iarray_read_in_function.ml` | 18, 13 | `twice_nested_same_scrutinee.ml` | 60, 25 |
| `iarray_read_one_call_inlined.ml` | 31, 23 | `two_flags_and_or_in_loop.ml` | 319, 213 |
| `id_exported.ml` | 8, 4 | `two_flags_set_in_loop.ml` | 179, 81 |
| `identity.ml` | 22, 3 | `two_ignored_tests.ml` | 40, 34 |
| `if_unknown.ml` | 13, 6 | `two_local_mutual_both_in_inner_function.ml` | 93, 63 |
| `ignore_value.ml` | 11, 2 | `two_options_matched.ml` | 180, 83 |
| `ignored_call_of_three.ml` | 28, 24 | `two_tables_of_one_match.ml` | 50, 30 |
| `ignored_call_or_constant.ml` | 28, 22 | `two_tags_shared_field.ml` | 69, 49 |
| `ignored_two_calls.ml` | 34, 26 | `unboxed_float_argument.ml` | 18, 13 |
| `include_functor_result.ml` | 41, 23 | `unboxed_float_argument_unknown.ml` | 13, 9 |
| `indirect_call.ml` | 13, 9 | `unboxed_float_over_application.ml` | 19, 13 |
| `inl_callee_region_twice.ml` | 477, 392 | `unboxed_pair.ml` | 13, 4 |
| `inline_function.ml` | 20, 4 | `unboxed_pair_argument.ml` | 10, 3 |
| `inline_record.ml` | 45, 7 | `unboxed_pair_parameter_partial.ml` | 74, 43 |
| `inlined_at_one_call.ml` | 21, 14 | `unboxed_pair_with_float_argument.ml` | 16, 11 |
| `inlined_test_join.ml` | 34, 7 | `unit_argument.ml` | 18, 1 |
| `int32_arith.ml` | 27, 8 | `unsafe_get_const_and_unknown.ml` | 20, 14 |
| `int32_boxed_once_for_two_results.ml` | 18, 5 | `var_twelve_mixed.ml` | 191, 106 |
| `int32_overflow.ml` | 40, 1 | `variant_blocks_join.ml` | 20, 8 |
| `int32_unknown.ml` | 17, 7 | `variant_constants.ml` | 44, 30 |
| `int64_boxed.ml` | 22, 8 | `variant_match_exported.ml` | 8, 4 |
| `int64_built.ml` | 61, 43 | `variant_nested_match.ml` | 213, 151 |
| `int64_clamp.ml` | 50, 33 | `variant_two_blocks.ml` | 44, 30 |
| `int64_compare.ml` | 48, 1 | `while_const_after.ml` | 70, 45 |
| `int64_compare_tested.ml` | 18, 8 | `while_countdown.ml` | 61, 34 |
| `int64_join.ml` | 26, 1 | `while_flag.ml` | 150, 75 |
| `int64_max.ml` | 51, 37 | `while_flag_and_two_tests.ml` | 208, 130 |
| `int64_of_int.ml` | 25, 7 | `while_flag_chooses_test.ml` | 250, 130 |
| `int64_option_heap.ml` | 63, 38 | `while_flag_comparison_or_constant.ml` | 123, 67 |
| `int64_overflow.ml` | 33, 1 | `while_flag_is_comparison.ml` | 111, 57 |
| `int64_passed_on.ml` | 40, 26 | `while_flag_or_two_tests.ml` | 214, 147 |
| `int64_shift.ml` | 21, 8 | `while_flag_starts_as_comparison.ml` | 96, 51 |
| `int64_sum_in_loop.ml` | 204, 89 | `while_flag_then_test.ml` | 160, 83 |
| `int64_sum_on_some_turns.ml` | 269, 154 | `while_flag_toplevel.ml` | 156, 79 |
| `int64_to_int32.ml` | 10, 6 | `while_if_inside.ml` | 161, 106 |
| `int_extreme_literals.ml` | 130, 79 | `while_loop.ml` | 78, 45 |
| `int_of_char.ml` | 8, 3 | `while_nested.ml` | 188, 127 |
| `int_split_intervals.ml` | 94, 63 | `while_stop_flag.ml` | 170, 121 |
| `int_switch.ml` | 31, 1 | `while_sum_down.ml` | 133, 80 |
| `is_even.ml` | 29, 1 | `while_two_loops.ml` | 126, 81 |
| `join_pair_equal_fields.ml` | 50, 28 | `while_two_tests.ml` | 180, 123 |
| `join_phys_equal_arms.ml` | 25, 17 | `while_unused_counter.ml` | 78, 40 |
| `join_tags_then_shared_field.ml` | 69, 46 |  |  |

### Validated when asked of the term before with its known arrays read

| Source | Steps |
|--------|-------|
| `iarray_exported.ml` | 10, 3 |
| `iarray_length_of_any.ml` | 14, 4 |

### Validated when asked of the term before with chosen calls inlined

| Source | Steps |
|--------|-------|
| `f2_one_inlined_one_not.ml` | 59, 35 |
| `rec_local_helper_called_outside.ml` | 38, 24 |
| `rec_unrolled.ml` | 64, 18 |
| `stub_called_twice.ml` | 26, 15 |

### Validated when asked of the term before with copies of code

| Source | Steps |
|--------|-------|
| `local_rec_inlined_twice.ml` | 128, 93 |

### Validated for the term as read (T4-13)

| Source | Steps | Source | Steps |
|--------|-------|--------|-------|
| `curry.ml` | 36, 3 | `let_polymorphism.ml` | 27, 1 |
| `exported_fun_to_map.ml` | 12, 9 | `list_length.ml` | 64, 44 |
| `exported_length.ml` | 8, 3 | `list_map.ml` | 132, 95 |
| `exported_made.ml` | 14, 6 | `list_rev.ml` | 70, 47 |
| `exported_option_library.ml` | 14, 10 | `option_map_exported.ml` | 8, 4 |
| `ignored_second_of_pair.ml` | 30, 22 | `string_const.ml` | 21, 2 |
| `length_exported.ml` | 8, 3 | `swap_exported.ml` | 8, 4 |

### Validated with its imports

| Source | Steps | Source | Steps |
|--------|-------|--------|-------|
| `a_captured_after_the_loop.ml` | 350, 177 | `raise_division_inlined.ml` | 83, 29 |
| `a_for_body_direct_around.ml` | 338, 167 | `raise_handler_raises_again_inlined.ml` | 90, 31 |
| `a_for_notrace_direct_value.ml` | 332, 169 | `raise_in_function.ml` | 24, 11 |
| `a_for_prim_direct_around.ml` | 346, 193 | `raise_inlined.ml` | 27, 11 |
| `a_for_rethrow_direct_around.ml` | 420, 219 | `raise_notrace.ml` | 22, 10 |
| `a_known_on_one_path.ml` | 188, 81 | `raise_unused_bucket.ml` | 23, 10 |
| `a_while_body_direct_around.ml` | 324, 167 | `raise_unused_exn.ml` | 22, 10 |
| `a_while_notrace_closure_two_around.ml` | 396, 215 | `string_get16_checked.ml` | 29, 17 |
| `array_length_twice.ml` | 21, 18 | `string_get_checked.ml` | 21, 4 |
| `collatz_local_refs.ml` | 3218, 1827 | `string_get_in_branch.ml` | 33, 24 |
| `dead_division_in_dead_arm.ml` | 57, 9 | `string_get_twice.ml` | 39, 24 |
| `dead_raise_in_dead_arm.ml` | 84, 27 | `string_get_two_indices.ml` | 47, 34 |
| `div_const.ml` | 26, 5 | `string_get_unknown.ml` | 28, 16 |
| `div_unknown.ml` | 16, 8 | `string_length_after_join.ml` | 40, 28 |
| `divisor_tested_twice.ml` | 31, 16 | `try_catch_all.ml` | 22, 10 |
| `empty_array_read_caught.ml` | 32, 13 | `try_div_zero.ml` | 23, 12 |
| `exn_with_argument.ml` | 37, 10 | `try_exported.ml` | 8, 4 |
| `exported_function_raises.ml` | 40, 23 | `try_in_each_turn.ml` | 179, 107 |
| `handler_arm_never_taken.ml` | 38, 10 | `try_in_function.ml` | 25, 10 |
| `imp_head_closure_uses_unit_function_only.ml` | 39, 30 | `try_nested.ml` | 30, 14 |
| `imp_head_include_functor_result.ml` | 41, 23 | `try_not_found.ml` | 28, 10 |
| `imp_head_inlined_at_one_call.ml` | 21, 14 | `try_reraise.ml` | 29, 14 |
| `imp_knowing_local_mutual.ml` | 148, 100 | `try_result_matched.ml` | 81, 52 |
| `made_array_checked_store.ml` | 23, 7 | `try_result_unused.ml` | 13, 3 |
| `mod_zero.ml` | 22, 6 | `try_test_outside.ml` | 22, 10 |
| `name_then_wildcard.ml` | 26, 11 | `uncaught_raise.ml` | 10, 4 |
| `never_returns_in_function.ml` | 44, 37 | `value_arm_constant.ml` | 54, 37 |
| `never_returns_twice.ml` | 23, 16 | `while_for_with_raise.ml` | 203, 106 |
| `none_raises_after_inlining.ml` | 13, 10 | `wrong_name_uncaught.ml` | 25, 8 |
| `partial_uncaught.ml` | 20, 17 |  |  |

### Validated with nothing imported, and not with its imports

None. The six units that were here (`array_length_twice.ml`,
`never_returns_in_function.ml`, `never_returns_twice.ml`,
`string_get_checked.ml`, `try_exported.ml`, `try_result_matched.ml`)
are validated with their imports, and are in the table above.

### What the validation of a unit that exports a function rests on

Of the validated units, those that export a function are validated
through the lifting of closures and blocks to symbols in the term
before. What each shows, and what its validation uses besides:

| Source | What Simplify does | What the validation uses |
|--------|--------------------|--------------------------|
| `exported_function.ml` | inlines the call; the exported closure becomes a symbol | the closure lifted to a symbol in the term before |
| `exported_two_functions.ml` | two exported functions, the second calling the first: inlines the calls, makes both closures symbols, and removes from the second the value slot that held the first | the closures lifted to symbols and the unused value slot removed in the term before |
| `exported_table.ml` | an exported function that reads a field of a constant pair of the module, the field being a function: the code of the function, which stays, returns the symbol of that function, and the value slot that held the pair is removed | the closures and the pair lifted to symbols, the projection of the pair from the closure and the load from the pair folded in the code, and the unused value slot removed |
| `exported_capture.ml` | an exported function that uses a value of the module computed when the module runs: its closure becomes a symbol that holds the value in a value slot, its code is bound after the value is computed, and where the function is inlined the slot is read as the variable | the closure lifted to a symbol, the value slot read from the static closure, and the bindings of code moved up in both terms |
| `exported_apply.ml` | an exported function given to another exported function: inlines both calls; the call of the argument, once the first is inlined, is a call of the symbol of the function | the closures lifted to symbols and the indirect call of the symbol made a direct call of its code, which is then inlined |
| `exported_made.ml` | a pair made by an exported function that is called when the module runs: inlines the call, and makes the pair that the inlined call makes a symbol | in two liftings: the closure of the function in the term before as given, then the pair in the term that this gives once normalised |
| `exported_record.ml` | a record made by one exported function and read by another, both called when the module runs with an argument that is not known; the closures of the two functions become symbols | closures and blocks lifted to symbols in the term before |
| `exported_variant.ml` | a value of a variant type made when the module runs, with an argument that is not known, and an exported function that tests it, called on that value; the block and the closure become symbols | closures and blocks lifted to symbols in the term before |
| `exported_list.ml` | a list of three elements whose head is not known until the module runs; its cells become symbols | closures and blocks lifted to symbols in the term before |
| `exported_local_fun.ml` | an exported function with a function of its own inside, which it calls; its closure becomes a symbol | closures and blocks lifted to symbols in the term before |
| `exported_curried.ml` | an exported function of two arguments, called with both when the module runs | closures and blocks lifted to symbols in the term before |
| `exported_nested_pair.ml` | a constant pair of pairs, and an exported function that reads a field of a field of it | closures and blocks lifted to symbols in the term before, and the loads from the constants of the module folded in the code of the function |
| `exported_funs_in_pair.ml` | two functions in a pair, and an exported function that calls the first | closures and blocks lifted to symbols in the term before, and unused value slots removed |
| `exported_chain.ml` | three exported functions, each calling the one before twice | closures and blocks lifted to symbols in the term before, and unused value slots removed |
| `exported_string_length.ml` | a constant string of the module, and an exported function that returns its length | closures and blocks lifted to symbols in the term before, and the length of the constant computed in the code of the function |
| `exported_record_update.ml` | a constant record, an exported function that makes a record from another with one field changed, and a call of it on the constant when the module runs | closures and blocks lifted to symbols in the term before |
| `exported_bool.ml` | an exported predicate, and an exported function that uses it on each of its two arguments with `&&` | closures and blocks lifted to symbols in the term before, and unused value slots removed |
| `exported_after_effect.ml` | a value that is not known until the module runs, then an exported function, then a call of the function on the value | closures and blocks lifted to symbols in the term before |
| `exported_twice_kept.ml` | a function that takes a function and is always inlined, given an exported function that is never inlined: the two calls of the argument, in the code of the function that gives it, stay, as direct calls | closures and blocks lifted to symbols in the term before, and unused value slots removed. It takes two things together: the calls of the argument are left alone on both sides, as the code called is in the list of code not to inline (the term after still calls it), and in the term before the indirect calls of the symbol of the argument are made direct in the code, as Simplify leaves them (`FlambdaKnownSymbolsSpec.fold_known_apply`). Without the second it is not validated: the two terms then differ in the kind of those calls only. With every call inlined on both sides in place of the first it is not validated either: the term after is left with a continuation applied once, which its normalisation does not inline |
| `exported_compose_kept.ml` | the composition of two exported functions that are never inlined, by a function that is always inlined: the two calls stay, in the code of the composition, as direct calls | as `exported_twice_kept.ml` |
| `exported_branch_then_fun.ml` | a value chosen by a test that is not resolved when the module runs, then an exported function that uses it: the closure of the function becomes a symbol that holds the value in a value slot, and the symbol and its code are bound in the handler of the continuation that the test goes to | the closure lifted to a symbol in the term before and all the code of the term after bound first |
| `exported_match_then_fun.ml` | a value taken out of an option that is not known, then a predicate that uses it: as the one before, after a match | as the one before |
| `exported_branch_two_funs.ml` | a value chosen by a test, then two functions, the second calling the first: inlines the call; both closures become symbols, bound with their code in the handler | the closures lifted to symbols, the unused value slot removed and the code of the term after bound first |
| `exported_fun_branch_fun.ml` | a function, then a value chosen by a test, then a function that uses both: the code of the second function, with the call of the first inlined, is bound in the handler | as the one before |
| `exported_two_branches.ml` | two values, each chosen by a test, with a function after each, the second function calling the first: the code of each function is bound in the handler of its test, the second handler inside the first | as the one before |
| `exported_branch_pair.ml` | one of two constant pairs chosen by a test, then a function that reads it: the pairs become symbols, and the closure a symbol that holds the one chosen | the closure lifted to a symbol in the term before and the code of the term after bound first |
| `exported_branch_calls_twice.ml` | as `exported_branch_two_funs.ml`, the second function calling the first twice: inlines both calls, and reads the value slot of the first closure once | as `exported_branch_two_funs.ml`; with the two calls inlined in the term before, the value slot is projected twice, and the second projection is made the variable of the first (`FlambdaShareProjectionsSpec.share_known_projections`) |
| `exported_some_again.ml` | a value that is not known until the module runs, put in an option that a function which is always inlined takes apart, and kept in another option: the first option is gone, and the second is a symbol | the closure and the second option lifted to symbols: the test tool gives the symbol to the first block of its shape, which is the first option, finds it unused in the end, and guesses again without it |
| `exported_fun_given.ml` | a function that gives, to the function it is given, a function which needs nothing from around it: the closure given is a symbol, bound beside the code of the function that gives it | the closure lifted out of the code of the function in the term before (`FlambdaLiftFromCodeSpec.lift_from_code_unit`) |
| `exported_fun_returned.ml` | a function that returns a function of its own which needs nothing from around it: the closure returned is a symbol | as the one before |
| `exported_funs_given.ml` | a function that gives two such functions: both closures are symbols | the two closures lifted out of the code; they are bound in the order of their `Let`s, which is not the order in which Simplify binds them, and the static bindings of a group are compared whatever their order |
| `exported_fun_in_option.ml` | a function that returns an option of such a function: the closure and the option are symbols | the closure lifted out of the code in a first lifting and the option in a second, which puts it before the closure; the order again |
| `exported_fun_to_map.ml` | a recursive function on lists, which is not inlined, and a function that gives it a function which needs nothing from around it: the closure given is a symbol | the closure lifted out of the code of the function that gives it, the calls of the recursive function left as they are, and the unused value slot removed |
| `exported_local_never.ml` | a function with a function of its own inside, which is never inlined and needs nothing from around it: the closure of the inner function is a symbol | the closure lifted out of the code of the outer function, and the calls of the inner function left as they are |
| `exported_option_library.ml` | three functions on options, as a small library has them (`map`, `default`, `is_some`): nothing is known of their arguments, and the three closures are symbols | closures and blocks lifted to symbols in the term before |
| `exported_length.ml` | the length of a list, a recursive function that is not a loop: its closure is a symbol and it calls itself through it | closures and blocks lifted to symbols in the term before |
| `exported_const_list.ml` | a constant list and a function that reads its head: the match is resolved in the code of the function | closures and blocks lifted to symbols in the term before |
| `exported_record_of_funs.ml` | a record of two functions and a function that calls one of them through the record: the call is of the symbol of the function, and is inlined | closures and blocks lifted to symbols in the term before, and unused value slots removed |
| `exported_submodule.ml` | a module inside the module, and a function outside it that calls a function of the inner module twice: the calls are inlined | closures and blocks lifted to symbols in the term before, and unused value slots removed |
| `exported_two_somes.ml` | two options, each of a value that is not known until the module runs: each is a symbol with a variable in its field | the two blocks lifted to symbols in the term before |
| `exported_branch_option.ml` | an option chosen by a test when the module runs, and a function that reads it with a default | the closure lifted to a symbol in the term before and the code of the term after bound first |
| `exported_branch_then_const.ml` | a value chosen by a test, then a constant pair and a function that reads the pair and the value: the two loads from the pair and their sum are folded | as the one before |
| `exported_branch_fun_const.ml` | a test when the module runs, then two functions that do not use what it chose, the second calling the first twice | as the one before, with the unused value slot removed |
| `exported_match_pair_then_fun.ml` | a pair that is not known until the module runs, taken apart, and a function that uses both parts | the closure lifted to a symbol in the term before, with the two parts in its value slots |
| `exported_nested_branch.ml` | a value chosen by two tests, one inside the other, and a function that uses it: its code is bound in the handler of the continuation that the three arms go to | the closure lifted to a symbol in the term before and the code of the term after bound first |
| `exported_fun_in_arms.ml` | a function chosen by a test when the module runs, each arm making a function that needs nothing from around it: the two closures are symbols | the closures lifted to the head of the unit in the term before, and the two terms compared with their static bindings that mention no variable bound first |
| `exported_fun_in_one_arm.ml` | a test when the module runs, between a function defined before it and a function made in the other arm: the closure of the arm is a symbol | as the one before |
| `exported_arms_after_let.ml` | a function that makes a function in each arm of a test, and the same test made when the module runs on a value computed before it: the four closures of the arms are symbols | the closures lifted out of the code and to the head of the unit in the term before, and the two terms compared with their static bindings that mention no variable bound first |
| `exported_two_tests.ml` | two tests when the module runs, one after the other: the first chooses a number, the second a function, whose closures are symbols | as `exported_fun_in_arms.ml` |
| `exported_funs_in_arms_used.ml` | a function chosen by a test when the module runs, and a function defined after the test that calls it twice: the closures of the arms are symbols | as `exported_fun_in_arms.ml`, with the closure of the function defined after the test lifted where it stands |
| `exported_option_fun_in_arm.ml` | an option chosen by a test when the module runs, whose contents is a function made in the arm: the closure and the block that holds it are symbols | two liftings to the head of the unit in the term before: the closure, then the block, which mentions no variable once the closure is a symbol; and the two terms compared with their static bindings that mention no variable bound first |
| `exported_pair_fun_in_arms.ml` | a pair chosen by a test when the module runs, of a number and a function made in the arm: the closures and the pairs are symbols | as the one before |
| `exported_records_in_arms.ml` | a record chosen by a test when the module runs, of a function made in the arm and a number: the closures and the records are symbols | as the one before |
| `exported_match_three_funs.ml` | a function chosen by a match of three cases when the module runs: the three closures are symbols, bound below the handler of the inner test, which mentions two of them | as `exported_fun_in_arms.ml` |

### The same on one run, not validated

| Source | Steps | What the validator lacks |
|--------|-------|--------------------------|
| `made_array_checked_store_in_loop.ml` | 92, 50 | the length of an array that a `Let` makes before a loop, taken at each turn of the loop for the check of an index: Simplify has the number from the allocation, and the two passes of the validation that know the length of an array made by a `Let` do not carry what they know into the handler of a recursive continuation, which the loop is (P1) |
| `pair_kept.ml` | 35, 33 | one branch gives a parameter of the function, and Simplify loads its fields at that use, which only the subkind of the parameter justifies; the model has no subkinds (T6-7). Item 2 of the list below says what the comparison of the two terms with the two calls inlined is left with |
| `pair_loop.ml` | 192, 85 | the same loads of the fields of a parameter of the function, which only its subkind justifies, at the entry of a loop, whose parameter Simplify unboxes from its subkind without looking at the uses (T6-7) |
| `pair_of_unknowns_after_inlined_pair.ml` | 22, 10 | not validated; both terms have a primitive of the heap, and each of the four liftings of the validation lifts nothing in a term given to it that has one (a limit pending decision D40; "What the verdicts mean", the line that names the primitive); related once lifted (the tables below), with the second search of the liftings |
| `raise_parameter.ml` | 11, 7 | code raises a variable that is not known to hold a symbol or a constant (a parameter of the code), and a lifting is made only where the unit gives symbols, constants and variables known to hold one to the continuations that it may leave through (`FlambdaLiftSpec.exits_plain_knowing`, and the CR at `exits_plain`); related once lifted (the tables below) |
| `throw_variable_kept.ml` | 49, 31 | as `raise_parameter.ml` |

### Related once lifted, and not validated

These units are among those of the table above: each is "not
validated", and the second check answers of it. They are counted
in no row of the validated.

The check given no import:

| Source | What it is | The line |
|--------|------------|----------|
| `pair_of_unknowns_after_inlined_pair.ml` | a function that is always inlined, which counts its calls in a reference and returns a pair, called once with a constant; then a pair of two values that are not known | related once lifted; no value slots removed |
| `raise_parameter.ml` | a function that raises its parameter, and one that catches whatever the first raises; neither is called | related once lifted; unused value slots removed |

The check given imports:

| Source | What it is | The line |
|--------|------------|----------|
| `throw_variable_kept.ml` | a function that raises the exception that it is given, and stays a function; two handlers by name (source by T5b) | related once lifted, with its imports; no value slots removed |

`pair_kept.ml`, of the table above, is a unit that the check refuses:
what the tool prints of it ends with the line "not validated". Of the
terms written by hand in the directory above, `../uncaught.fl` has
the line, with no import and no value slot removed: it leaves by its
exception continuation with a block that it made.

Measured when the line was added, with the tool of cd71844a36,
whose conversion made no primitive of the heap: on the sources that
that tool was run on, in the tree and outside it, every unit that
had the line had an import, apart from the examples added then to
exercise the no-import case, two more of their shape that are not
in the tree, and that term written by hand. The import was a
predefined exception or, in a unit that uses another unit, an
exception or another symbol of that unit. A declaration of a fresh
exception uses the runtime call `caml_fresh_oo_id`. A term in which
that call remains has no verdict: the tool does not convert C calls.
Those examples exercised the line with no import. That measure is
not one of the tool of this tree, whose conversion makes the
primitives of the heap. A unit that stores or makes a cell can have
the line with no import: each lifting of the validation lifts
nothing in a term given to it that has such a primitive, and the
second check may relate the two terms all the same.
`pair_of_unknowns_after_inlined_pair.ml` has it so; so has a
generated unit of `census/` (`loops`, seed 19), and so have two
terms written by hand in the directory above
(`../heap_store_stops_lifting.fl`, `../heap_store_stops_guess_again.fl`;
C2's). How many units outside the tree have it so is not counted
here.

### Undefined in the model

| Source | Why the term before has no meaning |
|--------|------------------------------------|
| `empty_array_phys_equal.ml` | two empty arrays compared by `==`: the front end makes one symbol of the two, so the comparison is of an allocated value with itself, to which the model gives no value (C4) |
| `iarray_phys_equal.ml` | an immutable array compared with itself by `==`: the model gives the comparison of an allocated value with itself no value (C4) |

### Not checked

| Source | What the model has not got, or the conversion does not make |
|--------|----------------------------|
| `float_add.ml` | floating-point arithmetic |
| `tupled_recursive.ml` | after Simplify the unit calls tupled code directly, giving it the components; the model has not got that call (C3) |
| `tupled_unknown_pair.ml` | as `tupled_recursive.ml` |

## What stops a unit from validating, most common first

1. Code that raises a variable that is not known to hold a symbol
   or a constant (two units: `raise_parameter.ml`,
   `throw_variable_kept.ml`; in each the code raises its parameter).
   Simplify makes the closures of such a unit symbols, and the
   validation makes a lifting only where the unit gives symbols,
   constants and variables known to hold one to the continuations
   that it may leave through (`FlambdaLiftSpec.exits_plain_knowing`,
   and the CR at `exits_plain`). A handler that raises again what
   it caught passes that test where everything that can reach it is
   a symbol or a constant (`exported_function_raises.ml` and
   `reraise_in_kept_function.ml`, which were here, are validated).
   The second check answers of each of the two ("Related once
   lifted, and not validated", above); they are counted as not
   validated.
2. A pair that is a parameter of the function, unboxed where two
   branches meet (two units). One branch gives the parameter, and
   Simplify loads its fields at that use, which only the subkind of
   the parameter justifies; the model has no subkinds, so the code
   of the term after is less defined than that of the term before
   and no rewrite of it can be proved (`pair_kept.ml`; discrepancy
   T6-7). `pair_loop.ml` is the same at the entry of a loop, whose
   parameter Simplify unboxes from its subkind without looking at
   the uses. In `pair_kept.ml` the function is not recursive, so
   the validation also compares the two terms with its two calls
   inlined. There the term before is left with the block of the
   module alone. The term after has, before that block, the two
   pairs that the function returns, which nothing uses, each with
   a field that is a symbol, and it keeps its static bindings: an
   allocation that nothing uses is removed from the term after
   only where its fields are constants
   (`FlambdaReadsAfterSpec.remove_constant_allocations`).
   `pair_join.ml`, which was the third unit, is validated by that
   comparison: its entry in the list below.

What the model has not got, and the conversion refuses: floating-point
arithmetic (`float_add.ml`), and a direct call of tupled code,
which gives the code the components (`tupled_recursive.ml`,
`tupled_unknown_pair.ml`): Simplify makes one of the call of a tupled
function that it knows and does not inline, so the term after is
refused where the term before is not. What the model has, and the
conversion makes in the cases that "not checked" lists ("What the
verdicts mean"): the primitives of the heap. A mutable array made of its
elements (`array_literal.ml`), and a reference that stays a block, with
the stores into it (`ref_counter.ml`), are converted, and the two units
are validated with nothing lifted. No unit of this directory has a case
of those primitives that the conversion refuses. A static immutable
array of values and the array with no element are converted
(`../static_array.fl` and the four beside it, `empty_array_length.ml`,
`empty_array_tested.ml`, `empty_array_in_record.ml`,
`empty_arrays_of_two_types.ml`, `empty_immutable_array.ml`; P1's). An
immutable array made as the term runs is converted too, where its
elements are values (`iarray_in_function.ml` and
`iarray_read_in_function.ml`, which are validated: the array is made of
the arguments of a function that is not inlined); it is a value with no
identity, where a mutable one is a cell.
The conversion refuses a call that is a probe as well; no unit of
this directory has one (`pairs/README.md`), and `from_ocaml.sh`
refuses a unit that has one.

What was on this list and is not any more, with the rewrite that
removed it:

- The continuation that makes the module block, applied once inside
  the handler that two branches go to (`if_unknown.ml`): a
  continuation is moved to the handler that alone uses it
  (`FlambdaSinkContsSpec.v`).
- A function that the unit keeps, whose closure becomes a symbol
  (`exported_function.ml`, `factorial.ml`): the validator lifts the
  `Let`s that Simplify lifted (`FlambdaLiftSpec.v`), and uses what the
  unit binds its symbols to (`FlambdaKnownSymbolsSpec.v`).
- Loops whose handler has something to fold (`while_loop.ml`,
  `for_sum.ml`): the rewrites of a round go into the handlers of
  recursive continuations (`FlambdaRecHandlersSpec.v`).
- The identities of arithmetic (`add_zero.ml`):
  `FlambdaIdentitiesSpec.v`.
- A primitive computed twice by the front end and once by Simplify, a
  boxed number unboxed at each use above all: `FlambdaCseSpec.v`.
- A function inlined twice at arguments that are not known
  (`twice_inlined.ml`, `local_function.ml`, `labelled_optional.ml`).
  The two copies of its body bind the same names in the term before,
  one in the scope of the other, and an alias of a name that is bound
  again is not followed, as it must not be; Simplify gives each copy
  names of its own. The validator gives every binder of the term
  before a name of its own once the calls are inlined
  (`FlambdaRenameApartSpec.v`).
- The order in which a group of static bindings binds the code of
  functions that call each other: Simplify builds the group again, in
  an order of its own, and the validator puts the bindings of the term
  before in the order of the term after (`FlambdaStaticsOrderSpec.v`;
  `../statics_order.fl`, written by hand, is validated by that alone).
- A local function given to a function that is inlined, inside a
  function that the unit keeps (`closure_argument_exported.ml`): the
  call becomes direct in the body of code, whose own group has the
  code called; the validator inlines between the code of one group
  (`FlambdaInlineSiblingsSpec.v`).
- A closure of a set that the term does not use
  (`returned_closure_of_pair.ml`): in the model it is the `Let` of a
  projection, which the term before loses by a refinement; the term
  after now loses it by an equivalence
  (`FlambdaUnusedProjectionsSpec.v`).
- A recursive function that the unit keeps and calls (`fib.ml`,
  `list_map.ml`, `factorial.ml`): each round inlined the call once
  more on both sides. The validator is told the code that it is not
  to inline (`FlambdaNormaliseSpec.inline_calls`), and the test tool
  names the recursive code and the code that the term after still
  calls.
- A function that is never inlined, given to one that always is
  (`exported_twice_kept.ml`): the same list, and the indirect call of
  a symbol that the unit binds to a closure made direct
  (`FlambdaKnownSymbolsSpec.v`).
- Closures made symbols where a value slot is a variable
  (`closed_mutual_recursion.ml`), or where the function is then called
  through a variable (`indirect_call.ml`): the function slots of a
  static set are known whatever its value slots hold
  (`FlambdaKnownSymbolsSpec.v`), and liftings are made in turn.
- An application of a continuation whose handler only applies
  another, a trap action included (`div_unknown.ml`,
  `string_get16_checked.ml`): it goes there directly
  (`FlambdaForwardSpec.v`).
- A parameter of a loop that every application gives the same thing
  (`while_const_after.ml`, `while_nested.ml`): it is bound at the head
  of the handler (`FlambdaParamAliasesSpec.v`), and the parameters of
  loops that are left unused are removed
  (`FlambdaUnusedParamsSpec.v`).
- Exceptions (`try_no_raise.ml`, `mod_zero.ml`, `raise_inlined.ml`;
  twelve units). A `try` installs a handler and removes it. Where
  Simplify finds that nothing can raise to the handler any more, or
  that what is raised is known, it makes the handler an ordinary
  continuation and drops the trap actions. The validator does the
  same at the end of the normalisation of the term before, and after
  the last lifting, and goes through its rounds again
  (`FlambdaDemoteSpec.demote_handlers`,
  `FlambdaValidateSpec.demote_and_tidy`). The conversion keeps, for
  that, the `Let` by which a handler ends its region (see above).
- A continuation whose handler is one jump, bound above a `Let` and
  applied by an arm of a `Switch`, once a call is inlined
  (`option_inlined.ml`): the rounds of the first phase, which send
  the arm where the jump goes, are gone through again after the
  second (`FlambdaValidateSpec.demote_and_tidy`).
- A continuation of no parameters that Simplify leaves with a
  `Switch` on a parameter it removed, every application giving that
  parameter the same constant (`variant_blocks_join.ml`): Simplify
  binds the constant at the head of the handler and the `Switch`
  stays under that `Let`. The folding of constants of the validator
  makes the handler one jump in both terms; the term before has its
  applications forwarded, and the term after has them forwarded by
  an equivalence (`FlambdaForwardKindedSpec.forward_kinded`, in
  `FlambdaValidateSpec.inline_once_applied_after`).
- Two arms of a `Switch` that give one continuation 0 and 1 (two
  generated units of `census/`): Simplify makes of them the tagging
  of what is tested, and so does the validator in both phases of the
  term before (`FlambdaMergeSwitchesSpec.merge_switches`).
- A field of a block loaded twice, once for each pattern of a match
  that looks at it (`list_last.ml`): a load marked immutable is one
  of the primitives that are not computed twice (`FlambdaCseSpec.v`).
- The tagging of a number that is the untagging of a variable
  (`variant_two_blocks.ml`, `variant_nested_match.ml`). A match on a
  variant gives each constant constructor its own number: the front
  end untags the scrutinee to test it, and the arms of those
  constructors pass their numbers as tagged constants, where
  Simplify makes them pass the scrutinee. The merging of the arms of
  a `Switch` makes of those arms the tagging of what is tested
  (`FlambdaMergeSwitchesSpec.merge_switches`). The sharing of
  primitives knows, under the `Let` of an untagging, that the
  variable untagged is the tagging of the result
  (`FlambdaCseSpec.computed_under_let`), so that tagging becomes the
  variable.
- A primitive computed before a `Switch` or a call and again in the
  handler that follows (`string_get_unknown.ml`,
  `string_get_in_branch.ml`, `string_get_twice.ml`,
  `string_get_two_indices.ml`, `string_length_after_join.ml`: the
  index of an access to a string converted for the bounds test and
  again for the load, the length read for each test;
  `field_twice_inlined.ml`: a field loaded before a call and again
  when it returns). Simplify computes it once and, where a
  continuation is reached from several places, gives it to the
  handler as an argument. Once its continuations are placed, each
  term is tidied: a parameter that every application gives one
  variable is taken away, and the sharing of primitives then sees in
  the handler what the `Let` above it computed
  (`FlambdaValidateSpec.tidied_before`, `tidied_after`, N3's, with
  `FlambdaUnaliasParamsSpec.v`).
- A constant block of the unit given to a join in a function or in
  a loop (`option_constant.ml`, `option_constant_loop.ml`,
  `option_constant_loop_fun.ml`, `list_constant.ml`,
  `pair_constants.ml`, `variant_constants.ml`): the unboxing of a
  parameter where branches meet knows, in the code and in the loops
  of the term, the static blocks and boxed numbers of the static
  bindings in scope (`FlambdaUnboxInScopeSpec.v`, T6's).
- A closure that the body of code binds by a `Let` and gives to an
  inlined function, in a unit with two functions that call each
  other (`local_closure_applied_mutual.ml`): the second asking
  follows aliases in the bodies of code too
  (`FlambdaValidateKnowingSpec.follow_aliases_knowing`, T3's).
- A test against zero of the result of `compare`
  (`compare_tested.ml`, `compare_order.ml`,
  `int64_compare_tested.ml`). `compare a b < 0` reaches Simplify as
  a three-way comparison, the tagging of its result and a test of
  that against 0. Simplify makes the test a comparison of the two
  operands, and the three-way comparison goes where nothing else
  uses it. The validator does the same on the term before, with the
  compiler's table case for case
  (`FlambdaRecoverComparisonSpec.recover_known_comparisons`, T5b's).
- Two equal projections of a value slot, which a function inlined
  twice leaves in the term before (`exported_branch_calls_twice.ml`):
  the second is made the variable of the first
  (`FlambdaShareProjectionsSpec.share_known_projections`). Simplify
  finds that variable in the type of the closure, as it does for a
  field loaded twice.
- A variant, a pair or a boxed number unboxed where two branches
  meet, each branch building it (`option_join.ml`, `shape_heap.ml`,
  `pair_built.ml`, `int64_option_heap.ml`): Simplify gives the
  continuation the fields, with `poison` for those that a branch has
  not got, and the validator does the same in the first phase of the
  term before (`FlambdaUnboxVariantsSpec.unbox_variant_params`).
- A variant that is a constant block on one path and a constant
  constructor on the other, unboxed where they meet
  (`shape_constant.ml`): Simplify gives the continuation whether the
  value is an immediate and the field, which is that of the symbol
  of the constant block on one path and `poison` on the other. Under
  a test, the untagging of a symbol has no value, so the unboxing of
  parameters gives a constant in its place
  (`FlambdaUnboxVariantsSpec.no_value_on`).
- A boolean untagged where the two tests of the condition of a loop
  meet (`while_two_tests.ml`, a `while` with `&&`): the first test
  gives the number 0 and the second its comparison, with nothing
  tagged. Under a test, the unboxing of parameters gives, for an
  untagging, the number that a `Let` in scope tagged
  (`FlambdaUnboxVariantsSpec.made_number`).
- A self tail call made a jump (`tail_loop.ml`, `list_length.ml`,
  `list_rev.ml`): where the two terms are not the same as given, the
  validator makes loops of the self tail calls of the code it is told
  (`FlambdaLoopifySpec.v`), and the tool names the code that the front
  end marks for it. The recursive code is in the list of the code
  not to inline; in these units the call of the function stays on
  both sides, and the two bodies are compared.
- A parameter of a loop that is only passed round the loop
  (`while_unused_counter.ml`): removed (`FlambdaDeadParamsSpec.v`).
- A continuation applied once that Simplify leaves in the term after,
  or that is applied once after the validation has inlined a call
  that Simplify left (`../kinds_partial_application.fl`; 46 of the
  generated units of `census/`): the term after has it moved to where
  it is applied and inlined there, where the checks of the passes
  permit, by equivalences
  (`FlambdaValidateSpec.inline_once_applied_after`).
- In the code of a function of which Simplify makes a loop, a `Let`
  of a parameter of the code that nothing uses (`string_sum.ml`): the
  term after loses it, with the parameters of the code in scope at
  their kinds (`FlambdaCodeLetsSpec.v`).
- A call of a variable bound to a closure of a set at a named function
  slot, which the rounds do not make direct (`tupled_function.ml`,
  whose tupled functions project their own function slot): where the
  validation fails, the tool asks it again with the aliases followed
  by the function that knows more of what is bound
  (`FlambdaValidateKnowingSpec.v`), which makes the call direct; a
  call that gives tupled code its tuple is then inlined
  (`FlambdaNormaliseSpec.inline_apply`). The verdict then has a
  second line that says so. With the rewrites from the types in the
  normalisation of the term before (the next point), that unit is
  validated at the first asking.
- A `Switch` on a variable, in a handler that only one arm of a
  `Switch` on the same variable applies (two generated units of
  `census/`): Simplify, which knows in the arm what the variable is,
  resolves the second. The rewrites that Simplify makes from the
  types are applied to the term before once, between two runs of the
  rounds of its second phase
  (`FlambdaSimplifyPassSpec.simplify_by_types`, computed by
  `FlambdaSimplifyCarriedSpec.simplify_by_types_carried`, in
  `FlambdaValidateSpec.normalise_before`): a handler that is applied
  once is rewritten under what is known where it is applied.
- A closure that calls a function of its module, made by a function
  that stays a function (`capture_fun_and_var_run.ml`,
  `capture_two_funs.ml`, `capture_rec_self.ml`, `over_rec.ml`): the
  closure holds the function in a value slot, and Simplify, which
  knows what the slot holds, makes the code of the closure call the
  symbol of the function. The validator replaces the projection of a
  value slot that every set of closures of the unit gives the same
  symbol or constant by that symbol or constant, where the symbol is
  in scope (`FlambdaKnownSlotsSpec.read_known_slots`, in each round
  of the normalisation of a term in which something was lifted).
- The return continuation of a call, bound above the `Let`s of the
  arguments of the call in the term before and below them in the
  term after (`call_result_unused.ml`, `call_result_tested.ml`; one
  generated unit of `census/`, and 56 of its units with loops).
  Where Simplify changes the parameter of the continuation (a result
  that the handler does not use is not passed; a boolean that an
  `if` tests is untagged), the call can no longer return to it:
  Simplify makes the call return to a new continuation, bound at the
  call, that applies the old one. The comparison is of the bindings
  in their order. The first comparison is of the two terms as they
  are given. Each term of the later comparisons has its
  continuations placed once the rounds are over: a `Let_cont` that
  is not recursive goes down to the call that returns to its
  continuation, or to the test of which an arm names it, where the
  guards of the pass permit. It passes a `Let` whose variable its
  handler does not use, a static binding, and the binding of another
  continuation that is not recursive if neither handler names the
  other; failing that, it goes into the handler of the other where
  nothing under the two bindings names it and the parameters of
  that handler are not used in its own; anything else between the
  two leaves it where it was
  (`FlambdaSinkUsesSpec.sink_conts_to_uses`, an equivalence, in
  `FlambdaValidateSpec.placed`).
- Code and a closure that Simplify leaves bound at the head of the
  term after though nothing uses them (`inlined_at_one_call.ml`,
  `closure_uses_unit_function_only.ml`,
  `include_functor_result.ml`): it left a call of the function,
  which the normalisation has since inlined in both terms. The
  validator removes from the groups at the head of the term after
  the static bindings that nothing uses, where the checks of
  `FlambdaUnusedHeadStaticsSpec.head_statics_removable` pass, and
  says so in its verdict. `closure_uses_unit_function_only.ml` is
  validated at an earlier comparison since: the tool fits the
  closure that the code makes to a symbol whose closure has fewer
  value slots, those that nothing reads once the code reads the
  symbol, which Simplify drops, and the two terms are the same once
  closures are lifted and unused value slots removed (C3's, in the
  guess of the tool; the validation checks the lifting as it does
  any other).
- A function that captures nothing, made in a loop
  (`loop_function_in_kept.ml`, `local_rec_in_tail_loop.ml`,
  `loop_function_twice.ml`): Simplify makes its closure a symbol,
  bound beside the code that has the loop, and has one symbol for
  two `Let`s of the same closure. The liftings out of code and to
  the head of the unit go into the handlers of recursive
  continuations, and two `Let`s of one constant share a symbol
  (`FlambdaLiftFromCodeSpec.v`, `FlambdaLiftFromArmsSpec.v`, V3's);
  the tool guesses the `Let`s to lift out of loops.
- A pair, or an option of a pair, that is a constant on an arm of a
  test and is taken apart where the arms meet
  (`pair_constant_arm.ml`, `pair_constant_arm_toplevel.ml`,
  `pair_two_constants.ml`, where it is one on both arms,
  `match_pair_option_built.ml`). A
  parameter of a join that is a pair or a variant is unboxed as
  Simplify unboxes it (`FlambdaUnboxVariantsSpec.v`, T6's): an arm
  is given the field of its block wherever the block is there, and
  the constant for nothing to read only where the block has another
  tag or is not there at all, whatever else the arm shows of the
  path that the handler takes. The parameters for the fields that
  the handler loads one below another are those of the last field
  first.
- The wrapper of a partial application whose closure stays
  (`partial_exported.ml`). The front end makes the wrapper, as code
  marked `stub`, with a direct call of the function through a value
  slot of the wrapper's own closure. Simplify simplifies the code of
  a stub where the code is bound and not where its closure is made
  (`middle_end/flambda2/simplify/simplify_static_const.ml`), so it
  does not know what the slot holds, and it makes of the direct call
  an indirect call of known arity
  (`middle_end/flambda2/simplify/simplify_apply_expr.ml`;
  discrepancy C3-11). The validator does to the term before what
  Simplify does: the direct calls in the code of a stub are made
  indirect (`FlambdaStubCallsSpec.demote_stub_calls`, a refinement,
  C3's), and value slots are not read in that code
  (`FlambdaKnownSlotsSpec.read_slots`). `../stub_by_hand.fl` and
  `../stub_by_hand_unmarked.fl`, written by hand, are one unit with
  its code marked as a stub and not: both are validated.
- The load of a character of a string that is a constant of the
  unit, at a constant index (`string_get_checked.ml`). The validator
  computed it where Simplify keeps the load
  (`simplify_string_or_bigstring_load` of
  `middle_end/flambda2/simplify/simplify_binary_primitive.ml`
  answers that it knows nothing of the result), so the two terms
  differed the other way. The fold at known symbols leaves that load
  now (`FlambdaKnownSymbolsSpec.reads_string`, N3's); the length of
  a known string is still folded, as Simplify computes it. The unit
  has an import, and the comparison that accepts the pair is one
  that lifts: it is validated with its imports, since that verdict
  has a test from the imports (C5's).
- The comparison of a predefined exception with itself
  (`try_not_found.ml`). A handler that catches an exception by name
  compares what was raised with the predefined exception, a symbol
  that the unit imports. Where Simplify knows what was raised, the
  two are the same symbol: it folds the comparison to true and drops
  the arm that raises again. The validator folds a `Phys_equal` of a
  simple against itself to its answer in the term before
  (`FlambdaSameComparisonSpec.fold_same_comparisons`). It is a
  refinement: where the machine gives the comparison of a value with
  itself an answer it is that one
  (`FlambdaPrimitiveSpec.phys_equal_same_statement`, C5's), and
  where it gives none the term before is undefined.
- The comparison of two different predefined exceptions
  (`handler_arm_never_taken.ml`, `name_then_wildcard.ml`,
  `wrong_name_uncaught.ml`; T5b's sources). A handler has an arm
  for an exception by name, and what is raised is another
  predefined exception: the comparison is of two different symbols
  that the unit imports. Simplify answers of two different symbols
  that they differ (`prove_physical_equality` of
  `middle_end/flambda2/types/provers.ml`): it folds the comparison
  to false and drops the arm. The pass that folds a `Phys_equal` of
  a simple against itself folds one of two different symbols to
  the other answer, in the term before
  (`FlambdaSameComparisonSpec.fold_same_comparison`). It is a
  refinement: where the machine gives the comparison of two
  different symbols an answer it is that one
  (`FlambdaPrimitiveSpec.phys_equal_symbols_statement`, C5's), and
  where it gives none the term before is undefined. The three
  units were not validated before the pass had that case, and are
  validated with their imports.
- The argument of an exception that is raised at one place, in a
  branch (`exn_with_argument.ml`). The handler reads the length of
  the string that the exception carries, a string that the unit
  binds, and Simplify has the length. The exception is a static
  block, and the one application of the handler is in the handler
  of a continuation that an arm of a `Switch` leads to. The
  rewrites from the types rewrote a handler under what is known at
  its one application only where that application ends the body of
  its `Let_cont`; they now gather the applications in the handlers
  of the continuations that the body defines, and where there is one
  they rewrite the handler under what is known there
  (`FlambdaSimplifyPassSpec.find_one_use`, T2's), as Simplify knows
  the one use of a continuation whatever the flags. The parameter of
  the handler is then the block that was raised, the loads from it
  are the symbols that it holds, and the validation computes the
  length of the string where it uses what the unit binds its symbols
  to. The unit is validated with its imports.
- A call whose result is only passed on
  (`call_result_passed_on.ml`): where the handler of the return
  continuation of a call applies another continuation to its
  parameters as they are, Simplify makes the call return there, and
  so does the validation for the term before, once, when its rounds
  are over (`FlambdaValidateSpec.with_returns`, with
  `FlambdaForwardSpec.forward_applications_with`, V4's). It leaves
  the calls of the term before where the term after still has such a
  call. In the unit that showed it, Simplify left the call because
  the handler ended the region of a block on the caller's stack
  before it passed the result on, which the conversion drops.
- A continuation that the two terms bind on either side of a `Let`
  (`inlined_test_join.ml`, `loop_exit_inlined.ml`): the placing of
  continuations moves a binding down as far as the bindings under
  it allow, where it moved one only if it reached a use
  (`FlambdaSinkUsesSpec.sink_conts_to_uses`, V4's).
- A continuation that the term after binds in the handler of another
  (`call_of_call_tested.ml`): the result of a call is the argument of
  a second call, whose boolean result is tested. The front end binds
  the continuation that tests above the one that the first call
  returns to; Simplify, which untags the boolean, makes the second
  call return to a new continuation, bound at that call, in the
  handler of the first. The placing of continuations takes a binding
  into the handler of a continuation that is not recursive where
  only that handler names it (`FlambdaSinkUsesSpec.sink_to_use`,
  V4's).
- An application that has a trap action and whose continuation only
  applies another (`value_arm_constant.ml`): the value arm of a
  `match .. with exception` gives a constant, and the application
  that pops the handler goes through a continuation that passes the
  constant on. Simplify makes it go there directly and keeps the
  pop. The forwarding of the term before left an application that
  has a trap action; it now takes it on with that trap action, where
  the application of the handler has none
  (`FlambdaForwardSpec.forward_apply_cont_keeping`, V4's).
- An application that Simplify leaves going through a continuation
  whose handler is one jump (`local_block_matched.ml`,
  `local_block_matched_argument.ml`, `loop_result_passed_round.ml`,
  `try_result_matched.ml`): the term before had it forwarded or not
  by a guess at what Simplify would leave. The term after has such
  an application forwarded where its arguments are known to have
  the kinds of the parameters, an equivalence
  (`FlambdaForwardKindedSpec.forward_kinded`, V4's), and the term
  before has every one forwarded.
- A helper that a function which is kept holds in a value slot, and
  that Simplify inlines there (`helper_inlined_in_closure.ml`,
  `hidden_helper_inlined_twice.ml`): Simplify removes the slot, the
  closure of the helper and its code. The removal of unused value
  slots is followed by the removal of the `Let`s that nothing uses
  any more, in the term and in its code
  (`FlambdaValidateSpec.without_slots`, V3's), so that the code goes
  with the unused static bindings.
- Blocks of two tags that come to one load
  (`join_tags_then_shared_field.ml`, `two_tags_shared_field.ml`).
  Where both arms of a test give a join a block, of two tags, and
  the handler loads a field that the constructors share without a
  `Switch` on the tag, Simplify keeps the two `Make_block`, the
  parameter and the load, since its parameters for the fields of a
  variant are by tag; the unboxing of the term before now leaves
  them too (`FlambdaUnboxVariantsSpec.loads_of_two_tags`, T6's),
  where it made the load a parameter.
- A call of a function that never returns (`never_returns_twice.ml`,
  `never_returns_in_function.ml`). Simplify finds that the code
  called does not return, and gives the call no return continuation;
  the conversion makes that a call that returns to a continuation
  whose handler is `Invalid`. In the term before, a direct call of
  code in sight whose body does not mention its return continuation
  is put under a binding of its own return continuation to such a
  handler (`FlambdaNeverReturnsSpec.cut_returns`, C5's). Both units
  have an import, the exception that the function raises, and are
  validated once closures are lifted, with their imports.
- A block or a closure that Simplify binds to a symbol outside the
  handler in which the source makes it
  (`block_in_arm_of_unknown.ml`, `closure_in_arm_of_unknown.ml`:
  made in the arm of a test from a value that is computed before
  the test, and bound before the test), and a boxed number that
  Simplify computes and has one symbol for
  (`boxed_constant_twice.ml`, `boxed_constant_made.ml`). A lifting
  names the `Let`s to make aliases of symbols that a group of
  static bindings around them binds, and the `Let_cont`s before
  which such a group is put
  (`FlambdaLiftAroundSpec.lift_around_unit`, V3's). The tool
  guesses both from its second lifting on: a `Let` that makes just
  what a symbol in scope is bound to becomes that symbol.
- A helper that a function which is kept calls through a value slot
  of its own closure, and that Simplify inlines there
  (`helper_called_through_slot.ml`). Simplify removes the closure
  of the helper, so the term after has no symbol for it, and the
  tool gave symbols only to the closures that a symbol of the term
  after fits: the slot held a variable, and the calls stayed
  indirect. The tool now gives such a closure symbols of its own,
  which neither term binds or mentions (V3's). The validation does
  the rest as it did: the slot holds a symbol, the projection in
  the code is that symbol, the call is direct and is inlined, the
  slot is dropped, and the symbols go with the unused static
  bindings.
- The length of an array taken twice (`array_length_twice.ml`: a
  recursive function over an array takes the length for its test,
  and a checked read takes it again). `Array_length` is one of the
  primitives that are not computed twice
  (`FlambdaCseSpec.shareable`), as it is in Simplify. The unit has
  an import and is validated once closures are lifted, with its
  imports.
- A box that a join is given after the term has unboxed it
  (`int64_max.ml`, `int64_clamp.ml`: the greater of two boxed
  integers, where the comparison unboxes both and each arm gives
  the join one of the boxes). Simplify gives the join the number
  that was unboxed. Where a handler begins by unboxing its
  parameter, the unboxing of parameters gives, for an argument that
  a `Let` in scope has unboxed, the variable of that `Let`
  (`FlambdaUnboxParamsSpec.projection_of`, with
  `FlambdaCseSpec.computed_by`; T6's).
- A result that nothing uses, of a test that has a call in one of
  its branches (`ignored_call_or_constant.ml`,
  `ignored_two_calls.ml`, `ignored_call_of_three.ml`,
  `ignored_second_of_pair.ml`, `two_ignored_tests.ml`,
  `nested_ignored_tests.ml`; the source is `let _ = if c then f x
  else 8 in ...`). Simplify removes the parameter of the
  continuation after the test. A call gives its return continuation
  all its results, so the call then returns to a new continuation,
  bound at the call, that applies the old one to what is left. The
  removal of unused parameters stopped at a continuation that a
  call returns to. A round of the term before now ends by removing
  such a parameter and binding, at each call that returns there, a
  wrapper that drops the result
  (`FlambdaWrapReturnsSpec.wrap_returns`, N3's), where the
  continuation is not an exception handler and is applied or
  returned to twice at least.
- The rewrites that Simplify makes from the types (above), in a
  term in which something was lifted. They were applied in the
  normalisation of the term before only, and a term in which
  something was lifted is normalised from the term as it was read,
  so it had none of them. It now has them once too, between its
  rounds (`FlambdaValidateSpec.normalise_lifted`, T2's).
  `local_closure_applied_mutual.ml` and
  `toplevel_closure_applied_mutual.ml`, which were validated at the
  second asking, are validated at the first; no unit of this
  directory was stopped by it alone.
- A match that Simplify makes a table of (`match_constant_table.ml`,
  `match_int_dense.ml`, `match_int_intervals.ml`,
  `match_tuple_of_variants.ml`). The arms of a `Switch` give one
  continuation constants that follow no rule. Simplify binds the
  constants to a static immutable array and makes the `Switch` a
  load from it at the scrutinee. The validation does it in two
  halves. A lifting may name tables, each a symbol with the
  elements of an immutable array, and the validation binds them in
  one group at the head of the term before
  (`FlambdaUnusedHeadStaticsSpec.add_head_tables`), only where two
  checks pass: every element of every table is a constant or a
  symbol (`tables_plain`), and the removal of exactly those symbols
  is permitted of the term with the group
  (`head_statics_removable`). The unit with the group refines the
  unit without it, each run on its own
  (`add_head_tables_program_refines_statement`). The tool names, in
  its first lifting, the symbols that the term after binds outside
  code to such an array and uses, and that the term before does not
  bind. Then a round of the first phase of the term before makes a
  `Switch` whose arms all give one continuation the elements of a
  table that is in scope, in the order of the arms, the load from
  that table at the scrutinee
  (`FlambdaMergeTablesSpec.merge_tables`, T1's). Only a table of
  values is one there: the model has no static array of naked
  numbers. The line of the verdict says "the tables of the term
  after bound in the term before".
- A boolean that Simplify keeps untagged in a parameter of a loop
  (`while_flag.ml`, and seven units that came with the rewrites:
  `flag_set_in_loop.ml`, `flag_set_in_loop_first.ml`,
  `two_flags_set_in_loop.ml`, `flag_and_count_in_loop.ml`,
  `while_flag_then_test.ml`, `while_stop_flag.ml`,
  `while_flag_toplevel.ml`). A flag kept in a reference and set in
  a loop is a parameter of the loop and of the continuations around
  it, and Simplify makes it a naked immediate in all of them. A
  round of the first phase of the term before untags, where they
  are given, the parameters that hold a tagged immediate and are
  only untagged or passed on, in all the continuations that pass
  them at once (`FlambdaUntagParamsSpec.untag_params`, N3's). Where
  a loop goes on while the flag is not set, Simplify gives the
  continuation that the two arms of the test of the flag come to a
  parameter which each arm gives the number of the arm. The
  tidying that follows the placing of continuations makes each such
  number the scrutinee and takes the parameter away, in both terms
  (`FlambdaGiveScrutineeSpec.give_scrutinees`, then
  `FlambdaUnaliasParamsSpec.unalias_params`; N3's).
- A parameter of a join that is read in two places
  (`option_matched_twice.ml`, `two_options_matched.ml`,
  `option_in_pair.ml`): an option made on the two arms of a test
  and matched twice after they meet, so that the test of the option
  and the load of its contents are each made in two places. After
  Simplify the join has one parameter for each thing read, where
  the unboxing of the term before made one for each place. Where
  it makes a read of a parameter a parameter, the other `Let`s of
  the handler that read the same parameter by the same primitive
  now become `Let`s of the variable of the read
  (`FlambdaUnboxVariantsSpec.with_reads_shared`, T6's). Loads are
  shared only where the tags that are known for them do not
  differ.
- A loop that is given a box made at every turn
  (`int64_sum_in_loop.ml`, `pair_turned_in_loop.ml`,
  `pair_loop_in_loop.ml`): a reference that holds a boxed number or
  a pair, entered with a constant and set at every turn. After
  Simplify the loop has the number, or the fields, for parameters.
  A round of the first phase of the term before makes a read of a
  parameter of a loop, which its handler comes to before any test,
  a parameter, where the entry and every turn give something that
  the term shows (`FlambdaUnboxLoopsSpec.unbox_loops_knowing`,
  T6's). A parameter is unboxed so only if no read of it is then
  left in its handler (`unbox_loop_param_all` there, by
  `reads_left`). A read that is left is a primitive that reads a
  box applied to the parameter itself, outside code and where the
  parameter is not bound again; a read of another variable that
  holds the same value is not looked for. So a loop that tests its
  parameter and loads from it under the test is left as it was.
- A call that returns to a continuation which ends a region and
  passes the result on (`stack_argument_result_returned.ml`, X5's
  source: a function allocates a pair on its stack, gives it to
  another function and returns what that one returns). The
  conversion drops regions, so in the model the handler only passes
  its parameter on; for Simplify it ends a region, and the term
  after has the call returning to it. The rounds of the term before
  made such a call return to the continuation that the handler
  applies, and the term after cannot be given the same. The calls
  of the term before are now taken on when its rounds are over
  (`FlambdaValidateSpec.with_returns`), and left where the term
  after, normalised and tidied, keeps a call that returns so
  (`keeps_a_return`; V4's). The unit was validated before the
  forwarding of applications took a call on, and is again.
- Local functions that call each other, each given to a function
  that is inlined (`fun_local_mutual_to_inlined.ml`,
  `kept_local_mutual_to_inlined.ml`,
  `fct_local_mutual_both_to_inlined.ml`,
  `loop_local_mutual_both_in_loop.ml`,
  `match_local_mutual_both_in_arm.ml`,
  `two_local_mutual_both_in_inner_function.ml`,
  `fun_local_three_mutual_to_inlined.ml`, `nomark_local_mutual.ml`;
  sources by T3). In the code of the function around them the first
  asking makes the call of the first direct; the second is reached
  by a projection of its function slot from the first, and that
  call is left where the term after has a direct call. Each is
  validated at the second asking, which follows aliases in the
  bodies of code knowing what the symbols are bound to
  (`FlambdaValidateKnowingSpec.follow_aliases_knowing`). Seven more
  units of this directory are validated at the second asking:
  `imp_knowing_local_mutual.ml` (C5's), which is the first of
  these with a function that raises a predefined exception exported
  too, so that the verdict of a second asking is asked of a unit
  with an import; `closure_in_tuple.ml` and `over_application.ml`,
  which were validated at the first asking until the tool converted
  every unit with named function slots; `least_join_table.ml` (a
  source of N2's), `over_inlined_maker_run.ml`, and
  `flag_or_test_then_count.ml` and `two_flags_and_or_in_loop.ml`
  (N3's; all three below). `raise_handler_raises_again_inlined.ml`
  (a source of T3's), which was validated at the second asking, is
  validated at the first, by a lifting: its handler raises again
  what it caught, and a lifting is now made of such a unit (item 1
  of the list above).
  Two more, `f2_one_inlined_one_not.ml` and
  `rec_local_helper_called_outside.ml`, are validated, at the default
  flags, at the second asking of the function that inlines chosen calls
  of the term before first ("What the verdicts mean").
  By a
  measure of T3's with the tool of
  fda8114815, over
  4,166 terms (the units of this directory at two settings of the
  flags, and sets that are not in this tree), 66 are validated at
  the second asking, 10 of them only with the part that follows
  aliases in the bodies of code; nothing in the tree makes the
  measure.
- The result of a function that the validation inlines at two
  calls of the term after (`a_captured_after_the_loop.ml`,
  `a_while_notrace_closure_two_around.ml`; sources by T5). Simplify
  inlines none of the four calls of the function; the validation
  inlines them in both terms.
  A copy of the body then ends with a `Let` of the result of its
  call to the variable that the body returns, and the aliases could
  not be followed past the next copy, which binds the same names:
  the term after is not renamed apart, and the term before is. Such
  a `Let` is now made one with the `Let` around it that binds the
  variable, before the aliases are followed
  (`FlambdaAliasLetsSpec.merge_alias_lets`, an equivalence that
  makes no name). It is made only where the body returns the last
  thing that it computes. Each of the two units is validated with
  its imports.
- A loop that tests a flag before it sets it
  (`flag_tested_in_loop.ml`, `flag_cleared_once_in_while.ml`,
  `count_after_flag.ml`, `flag_guards_update.ml`; N3's). Where the
  loop does not set the flag it passes it on as it was read, and
  Simplify passes the constant that the test found it to be. The
  tidying that follows the placing of continuations now puts, in
  both terms, the number of an arm for the scrutinee in the
  arguments of that arm and of the handler that only that arm
  applies (`FlambdaArmNumbersSpec.numbers_in_arms`). In the last two
  units the flag is set under a second test, whose arms then give
  their own numbers; Simplify has one application of the
  continuation to the result of that test, and the tidying of the
  term before makes it by merging the arms
  (`FlambdaMergeSwitchesSpec.merge_switches`), which the rounds
  apply before the scrutinee is a number there. The tidying of the
  term after does not merge the arms.
- A flag that a loop sets to a comparison
  (`while_flag_is_comparison.ml`,
  `while_flag_comparison_or_constant.ml`,
  `while_flag_starts_as_comparison.ml`; N3's). Where the source
  sets the flag (`go := !i < n`), the loop is given a variable that
  a `Let` just above binds to the tagging of the comparison, and
  Simplify gives the untagged parameter the comparison itself. The
  untagging of parameters asked for a tagged constant or another
  untagged parameter at such a position; it now takes a variable in
  scope that a `Let` binds to the tagging of a variable too, and
  gives the variable that was tagged
  (`FlambdaUntagParamsSpec.untag_params`).
- A function of a module of the unit that is called through the
  block of the module (`module_function_in_code.ml`; V3's). The
  module holds a value computed when the unit runs and a function,
  and another function calls that function with the module opened:
  in its code the term before loads the function from the block of
  the module, a static block one of whose fields is a variable, and
  the term after calls the symbol of the function. The validation
  knew a static block only where none of its fields was a variable.
  It now folds a load that is not marked mutable from such a block
  to the constant or the symbol that the field holds, and leaves a
  load of a field that holds a variable
  (`FlambdaKnownSymbolsSpec.fold_block_load`).
- A loop written as a local function that Simplify inlines
  (`local_loop_inlined.ml`; V4's): an exported function whose body
  is a local tail-recursive function, marked to be inlined, and its
  one call. Simplify makes a loop of the local function and inlines
  it. Both lists of the code not to inline that the tool tried had
  the local function, as recursive code, so the term before kept
  the call where the term after has the loop. The tool has a third
  list, in which code that is recursive by its self tail calls
  alone is not recursive ("What a check costs").
- A match that Simplify makes two tables of, or a table and a
  computation (`two_tables_of_one_match.ml`, `pairs_for_a_suit.ml`;
  sources by P1, the rewrite T1's). The arms of the `Switch` give
  one continuation two arguments: in the first unit each is loaded
  from a static array of its own, and in the second one is loaded
  from a static array and the other is the number of the arm plus
  1. The rewrite that makes such a `Switch` a load asked for one
  table that has the arms. Where there is none it now looks for a
  way for each argument: one of the ways of the merging of arms
  where there is one, and else the first table in scope that has
  the argument across the arms
  (`FlambdaMergeTablesSpec.merge_tables`).
- A value slot that holds a symbol which is not in scope where the
  slot is read (`capture_fun_bound_in_handler_run.ml`). The closure
  of a function holds a value chosen by a test when the module
  runs, so its symbol is bound in the handler of that test; the
  code of a closure that calls the function is bound at the head of
  the unit and reads it from a value slot, where Simplify puts the
  symbol. The validation read a slot only where the symbol that it
  holds is in scope. After the last lifting it now reads the known
  value slots once more with every symbol that the unit binds taken
  to be in scope (`FlambdaKnownSlotsSpec.read_known_slots_anywhere`,
  C3's): no lifting comes after, and what follows tests the scope
  of static names where it relies on it.
- A function that returns a function, called with all the arguments
  at once (`over_inlined_maker_run.ml`; C3's). Once the function is
  inlined, the closure that it returns is called with one argument
  more than its code takes. Simplify splits a direct call that
  gives known code more arguments than it has parameters into the
  full call and the call of its result, and inlines the full call.
  The inlining of the term before left such a call as it was; it
  now splits it the same way where the tests of `over_applied_b`
  pass (`FlambdaNormaliseSpec.split_over_application`), whether or
  not the code is among those whose calls are not inlined. The unit
  is validated at the second asking.
- The block of a structure given to a functor, held in a value slot
  (`functor_argument_block_in_slot.ml`; V3's). A functor is applied
  to a structure that holds a partial application; the closure of
  the functor's function holds the block of the structure, and
  Simplify resolves every load from it and keeps no block. The
  tool's guess of what was lifted gave a `Let` that makes a block a
  symbol only where the term after binds a block of that shape. It
  now gives a symbol that neither term has to a block whose fields
  are symbols, constants or variables that stand for them, which a
  closure of the term before holds in a value slot, where the guess
  has no symbol of the term after left to take for it and the term
  after makes no block of that tag and number of fields by a `Let`
  (the term after may still bind a symbol to a block of that
  shape). It is a guess, which the validation checks.
- A flag tested with other tests, or read twice in a loop
  (`while_flag_and_two_tests.ml`, `while_flag_or_two_tests.ml`,
  `flag_read_twice_in_loop.ml`, `while_flag_chooses_test.ml`;
  N3's). In the first two the number that an arm gives is that of
  an arm of a `Switch` further up than the nearest one: the
  scrutinee to give is now looked for among those of the `Switch`es
  above, the nearest first
  (`FlambdaGiveScrutineeSpec.give_scrutinees`). In the other two
  each arm has a continuation of its own, bound above the `Switch`
  one after the other, and the number of the arm was known only in
  the handler bound nearest the `Switch`: it is now known through
  the continuations that are not recursive bound between, where
  their handlers do not mention the continuation of the arm
  (`arm_fact` there, which `FlambdaArmNumbersSpec.number_in_arms`
  uses too).
- A boxed number that continuations pass to one another
  (`int64_sum_on_some_turns.ml`, `int64_passed_on.ml`,
  `sums_of_two_sorts.ml`; T6's). A reference to a boxed 64-bit
  integer is added to on some turns of a loop only, so the loop and
  the join after the test in its body are each given the parameter
  of the other; or the join of one test gives a box to the join of
  another, which unboxes it. Simplify gives all of them the number.
  The unboxing of joins and the reads of a loop take one
  continuation at a time, and left both. A round of the first phase
  of the term before now unboxes together the parameters that hold
  a boxed number and are only unboxed or given to one another
  (`FlambdaUnboxTogetherSpec.unbox_together_knowing`).
- Arms that do the same thing (`gadt_existential_match.ml`,
  `three_arms_same_sum.ml`; V4's, from sources of T3's). The
  front end gives each arm a continuation of its own, with the same
  handler; Simplify keeps one handler, and the `Switch` whose arms
  all go to it is gone. A round of the term before can now make a
  non-recursive continuation whose handler is that of a
  continuation bound around it apply that one, where the tests of
  the pass allow (`FlambdaSameHandlersSpec.share_handlers`, an
  equivalence; the header of that file says which handlers it
  takes and which it leaves), after which the arms of these units
  go to one continuation.
- An immutable array of values, which Simplify makes a symbol
  (`iarray_literal.ml`, `iarray_exported.ml`; P1's sources). The
  term before makes the array when the module runs, and the term
  after binds a symbol to a static immutable array in its place.
  The validation now lifts the `Let` that makes such an array as it
  lifts one that makes a block (`FlambdaLiftSpec.lift_named`; P1's
  and V3's), under the same test of the exits, and the verdict has
  the words that it has for a block. Immutable arrays of values
  only: no mutable array and no array of naked numbers. The second
  unit also loads from the array at a constant index, which
  Simplify makes the element. The validation reads the length and
  the elements of an array that the term before knows
  (`FlambdaArrayLoadSpec.load_known_arrays`, P1's), and no longer
  in its rounds: with the read in the rounds there were pairs that
  the validation accepted without it and refused with it
  (`status/P1.md`). Five units of this directory are such pairs
  (`iarray_measured_one_call_inlined.ml`,
  `iarray_read_one_call_inlined.ml`,
  `iarray_literal_measured_one_call_inlined.ml`,
  `iarray_of_variable_measured_one_call_inlined.ml`,
  `iarray_of_variable_read_one_call_inlined.ml`; P1's): an
  immutable array made by one function and measured or read by
  another, in a unit with a function that Simplify inlines at one
  call and keeps at another. The validation makes the two terms of
  such a unit the same by inlining every call of known code on both
  sides, and the length or the load is then a primitive in both.
  The five are validated, with no read. The read is made by a
  function beside the validation, which the tool asks of a unit
  that the validation refuses
  (`FlambdaValidateReadingSpec.validate_unit_reading`; "What the
  verdicts mean" says what its line claims): the second unit is
  validated so.
- A number boxed twice (`int32_boxed_once_for_two_results.ml`;
  V2's). Two exported results are one boxed 32-bit integer: the
  front end boxes it once for each, and Simplify boxes it once for
  the two. The sharing of primitives now takes the boxing of a
  number (`Box_number` in `FlambdaCseSpec.shareable`): a later
  boxing of the same naked number is the variable of the first,
  where the first is still known. In the model a boxed number is
  its contents, so the pass stays an equivalence; the header of
  that file has the question that this leaves for the human
  (decision D42). The primitives that make a block or an array are
  not shared.
- A flag that a loop tests and keeps (`flag_and_test_in_loop.ml`,
  `flag_or_test_in_loop.ml`, `flag_or_test_then_count.ml`,
  `two_flags_and_or_in_loop.ml`; N3's). A loop sets a flag kept in
  a reference to itself and a test, or to itself or a test. After
  Simplify the continuation where the arms meet has the flag twice:
  as a value, which goes round the loop and is returned, and as a
  number, which the next turn tests. A round of the second phase of
  the term before that has left the term as it was, up to bound
  names, now gives such a parameter a twin and untags the twin
  (`FlambdaShadowParamsSpec.shadow_params`, the renaming apart,
  `FlambdaTwinParamsSpec.twin_params`). The last two units are
  validated at the second asking.
- A test made again in the arm of the first
  (`divisor_tested_twice.ml`, `tag_known_in_arm.ml`; V4's and N3's,
  found by T5 and by T2). A quotient and a remainder by one divisor
  each test the divisor; a match on a variant has, in the arm of
  one constructor, a second match on the same value. Simplify knows
  the answer of the second test in the arm of the first. The number
  that a `Switch` has decided for its scrutinee is now put for the
  variable, in the arm and in the handler that only that arm
  applies, at the reads that the header of `FlambdaArmNumbersSpec.v`
  lists (`number_in_arms`): among them an argument of a primitive
  and the scrutinee of a second `Switch`. The tidying of each term
  folds what that has made constant
  (`FlambdaValidateSpec.tidy_expr_before`, `tidy_expr_after`).
- A function that returns what a `try` gives, and has a region
  (`try_returned_exported.ml`; V4's, found by C4 and T5 in a unit
  of T5b's). The continuation that follows the `try` ends the
  region and passes its parameter on, so that its handler is one
  application in the model, which has no regions; the body of the
  `try` applies it with the trap action that pops the handler. The
  forwarding of the term before made that application go where the
  handler goes; that of the term after left an application that has
  a trap action. It now forwards it, where the application of the
  handler has none and the arguments are known to have the kinds of
  the parameters, and the application keeps its own
  (`FlambdaForwardKindedSpec.forward_kinded_apply_cont`).
- A function that Simplify inlines where the unit calls it and
  leaves in the code of another function of its group
  (`rec_helper_called_outside.ml`, T3's source, and
  `rec_helper_among_three.ml`; V2's). Functions are defined
  together with `let rec .. and ..`, none of them recursive, and
  the first calls the second. Simplify inlines the second where the
  unit calls it and leaves the call of it in the code of the first.
  A round of the term before inlined that call too, by its pass for
  the code of a group
  (`FlambdaInlineSiblingsSpec.inline_sibling_calls`); a round of the
  term after had no such pass, and the code of the first was not
  the same in the two terms. A round of the term after now inlines,
  in the body of code, the calls of the other code of its group
  whose callee is a symbol that the group binds to a closure of the
  code called
  (`FlambdaKnownSiblingsSpec.inline_known_sibling_calls`, an
  equivalence). In the second unit a third function of the group
  has the type of the second and another body. The limit that
  `status/V2.md` described stayed then: what the rounds of the
  validation do not inline is chosen by code and not by call. The
  entry of `f2_one_inlined_one_not.ml`, below, says for which
  calls it is lifted.
- A value of the unit read from two places
  (`shared_read_slot_and_field.ml`, `shared_read_two_closures.ml`;
  V3's). A value that is computed when the unit runs is held in two
  places of the static constants of the unit: a value slot of a
  closure and a field of the block of the module, or a slot of each
  of two closures. The code of a function reads both places, and
  Simplify makes the second read the value of the first. In a
  lifted term the validation now makes the second of two reads,
  from symbols, of a variable that one group of static bindings put
  in two places the variable of the first
  (`FlambdaSharedReadsSpec.share_static_reads`, a refinement for a
  unit run on its own and from its imports).
- A match where two branches meet, each giving a known number
  (`load_through_join.ml`; V4's). One branch of a test builds a
  value with the first constructor and the other with the second,
  and the value is matched where the two meet. Simplify gives the
  continuation where they meet the tag and the fields, and leaves
  the match there as a `Switch` on the tag, to which each branch
  gives a number; the arm of a third constructor, whose number no
  branch gives, reads fields that nothing gives. The continuations
  that a handler which tests its parameter binds on its way to its
  `Switch` are now bound outside the handler
  (`FlambdaLiftContsSpec.lift_handler_conts`, N2's, an
  equivalence), and an application that gives a number to the
  parameter that its handler tests goes where the arm of that
  number goes: in each round of the first phase of the term before
  (`FlambdaForwardArmSpec.forward_to_arms`, a refinement;
  `FlambdaValidateSpec.first_phase_round`) and in the term after
  (`forward_to_arms_kinded`, an equivalence, where the arguments
  are known to have the kinds of the parameters). Where every
  application then goes to its arm, in both terms, the handler is
  removed from both, and the arm that no application comes to goes
  with it.
- A parameter of a continuation that every path gives one constant
  (`same_constant_after_loop.ml`; V4's unit, N3's pass). A
  condition, or else a comparison with what a loop leaves in a
  reference that it never changes. Simplify finds that the
  reference still holds what it held before the loop and leaves the
  comparison where it was, so that the continuation which the two
  paths come to keeps its parameter; once the validation has folded
  the comparison, both paths give that parameter the same constant.
  A parameter of a non-recursive continuation that is no exception
  handler, which every application gives one symbol, or one
  constant, is now taken away and bound to it at the head of the
  handler, in the term after
  (`FlambdaBindConstParamsSpec.bind_const_params`, an equivalence);
  the term before has the same from
  `FlambdaParamAliasesSpec.bind_aliased_params`, which refines.
- A closure made by an inlined call of a function that is also
  exported (`maker_inlined_and_exported.ml`; C3's). A function
  marked for inlining makes a closure; the unit exports the
  function and calls it with a constant. Simplify gives the closure
  of the inlined copy new code, under a new code id, which reads no
  value slot, and keeps the code that reads the slot for the
  closures that the function makes when another unit calls it. The
  term before has one code for both. A lifting of the validation
  may now ask for copies of code, and it ends by making those that
  it asks for, each if its check passes
  (`FlambdaCopyCodeSpec.with_copies`, the last step of
  `FlambdaValidateSpec.with_lifting`). A copy binds code of the
  term again under a new code id, and gives the new id to the
  static sets of closures that the lifted term binds to given
  symbols (`copy_static_code`; a refinement, for a unit run on its
  own and from its imports). A lifting that asks for no copy does
  what it did, and a copy whose check fails is left out: `copy_ok`
  asks that the code ids to copy are distinct, that the new ones
  are distinct, and that the term binds none of the new. Which code
  and which symbols is a guess of the tool, read off the term
  after, and nothing is believed of it: the refinement is of every
  copy that passes that check. The copies at the end of a lifting
  are for static sets only: a closure still made by a `Let` in the
  term they are given keeps its code ids (`status/C3.md`). For such
  a `Let` the tool can ask another function
  (`local_rec_inlined_twice.ml`, below).
- A pair read where two paths meet, in a function that is called
  twice and that the unit does not export (`pair_join.ml`; P1's
  rewrite). The function is marked never to be inlined. Simplify
  keeps its two calls, and in its code unboxes the parameter of the
  continuation where the two paths meet: at the path that gives the
  pair it loads the two fields, which only the subkind of the
  parameter justifies (T6-7; item 2 above). The unit is validated,
  and that says nothing of the unboxing. The validation inlines the
  two calls in the two terms; nothing then calls the code of the
  function, and the code is removed from both terms with the unused
  static bindings, so the code in which Simplify unboxes is never
  compared. What is compared is the copies of its body. In the term
  after, the copy of the path that gives the pair loads the two
  fields of a pair of constants that a static binding holds, and a
  round of the term after now reads what a known block holds where
  the read is sure to have a value
  (`FlambdaReadsAfterSpec.load_known_blocks_surely`, an
  equivalence); the term before has the same from
  `FlambdaBlockLoadSpec.load_known_blocks`, which refines.
- A pair made by one function and read by another, in a unit that has a
  function which Simplify inlines at one call and keeps at another
  (`pair_read_one_call_inlined.ml`; P1's). Told to leave the calls of
  recursive code only, the validation inlines the calls of the two
  functions in the two terms. The term before then loses the load and
  the pair by refinements. The term after has them as primitives: a
  round of the term after reads the field of the pair, as in the entry
  before, and now removes an allocation that nothing uses where every
  field is a constant immediate
  (`FlambdaReadsAfterSpec.remove_constant_allocations`, an equivalence).
- A handler by name in a unit that keeps code
  (`while_for_with_raise.ml`, a source of T3's, and
  `try_in_each_turn.ml`, a source of T5b's; P1's units, V3's test with
  P1). A lifting was made only where the unit gave symbols and constants
  to the continuations that it, or its code, may leave through, and a
  handler by name raises again, as a variable, what it does not catch.
  The test of a lifting now counts as a symbol or a constant a variable
  known to hold one: a parameter of a non-recursive handler that every
  way of reaching the handler gives a symbol, a constant or such a
  variable (a call that returns to the handler makes its parameters
  unknown; what a call raises to it is not looked at where the call
  stands, the test asks it of the code of the unit that is called). The
  test is `FlambdaLiftSpec.exits_plain_knowing`, which accepts whatever
  `exits_plain` accepts. The two units are validated with their imports
  by the comparison of the lifted terms. `exported_function_raises.ml`
  and `reraise_in_kept_function.ml`, which were not validated, are
  validated by that comparison too, the first with its imports (item 1
  of the list above); eleven units that were validated with their
  imports by another comparison are validated with their imports by that
  one.
- The length of an array whose elements may be anything
  (`iarray_length_of_any.ml`, `iarray_length_of_any_not_inlined.ml`;
  P1's; the two units have an immutable one). The front end makes of it
  the primitive `Array_length` at `Float_array_opt_dynamic`, which
  carries no kind of array. The model has it
  (`FlambdaPrimitiveSpec.array_length_of`: the number of elements,
  whatever kind the primitive carries); the conversion had no case for
  it, and such a unit was not checked. It has the case now. A load from
  such an array is still refused where it is one of floats. The first
  unit, in which the function is inlined, is validated when asked of the
  term before with its known arrays read ("What the verdicts mean"); the
  second, in which it is not, by the validation itself.
- A closure that a `Let` makes, called through a variable
  (`maker_of_two_inlined_twice.ml`; C3's). A function marked for
  inlining makes two closures and is called twice. One of the two
  functions is recursive and stays: Simplify gives each of its two
  closures a symbol and code of its own. The other is a loop marked for
  inlining, which Simplify inlines at its two calls, and the term after
  has no closure of it. In the term before, once the maker is inlined,
  each closure of the loop is made by a `Let` and read from a pair, and
  its call is indirect. A lifting gives the closures of the first
  function their symbols and the copies of their code, and leaves those
  of the loop to their `Let`s, the term after having no symbol for them.
  A lifting may now ask its rounds to make direct an indirect call of a
  variable that a `Let` in sight binds to a closure at a function slot
  of a set, or to the projection of a function slot from such a variable
  (`FlambdaClosureCallsSpec.call_made_closures`, a refinement under all
  environments); the next round may then inline the call, and in this
  unit it does, as Simplify did. The tool makes an attempt whose
  liftings ask for it where the earlier attempts had refused the unit
  and the term before has such a closure, and the verdict has a line
  that says what was asked ("What the verdicts mean").
- Two local functions that call each other, made by a function that is
  inlined and exported (`maker_of_mutual_inlined.ml`; C3's). Simplify
  inlines the maker at its call. The two functions of that copy capture
  a value that the unit computes once, so Simplify gives them symbols
  and code of their own, and the maker keeps the code that there was. A
  lifting gives the two closures of the copy their symbols and copies of
  the code. That makes indirect the two calls in the code that the maker
  keeps: each named code that is copied, and its callee is a variable,
  the other function projected from `my_closure`. A lifting may now ask
  its rounds to use, in the body of code, what `my_closure` is known to
  be from its code, under the conditions that "What the verdicts mean"
  gives: a call of it, or of another closure of its set where every set
  of the unit that has that code at that function slot has the same code
  at the other slot, is made direct, and a value slot that every such
  set gives one constant, or one symbol that is in scope, is read
  (`FlambdaClosureOfCodeSpec.know_own_closures`: a refinement of a unit
  run on its own, from its imports, and followed by what may follow a
  unit, which is the observers of `FlambdaObserversSpec.v` at the value
  slots given, run from the imports, and not under all environments).
  The tool makes an attempt whose liftings ask for it where the earlier
  attempts had refused the unit and an asking has found a term that the
  rewrite changes, and the verdict has a line that says what was asked,
  under the line of the calls of the closures that a `Let` makes, which
  those liftings ask for too, the term before having such a `Let`.
- A parameter of a loop that every turn passes on as it is
  (`function_passed_on_by_loop.ml`, `fold_passes_function_on.ml`; V4's
  units, N3's pass). A recursive function that passes on to itself the
  function that it is given is inlined where it is called with a
  closure. Its self tail call is a jump, so what is inlined is a loop
  that has the parameters of the function, and the parameter that holds
  the closure is given the variable of its own name from outside, and
  itself at every turn. Simplify inlines the closure in the loop. A
  parameter of a recursive continuation, the only one of its `Let_cont`
  and no exception handler, is now taken away and bound at the head of
  the handler to the variable that the loop is given for it, where every
  application in the body of the `Let_cont` gives that one variable,
  every application in the handler gives the parameter itself, no binder
  on the way to an application binds what that application gives,
  neither is one of the other parameters, and nothing mentions the
  continuation but those applications (no call returns or raises to it,
  no trap action names it, nothing binds it again); the variable may
  have the name of the parameter, as here
  (`FlambdaUnaliasRecParamsSpec.unalias_rec_params`, an equivalence).
  The pass is applied in each round of the first phase of the term
  before and in the tidying of both terms. The call of the parameter is
  then a call of the closure, which the validation makes direct and
  inlines.
- An option unboxed where the ends of an inlined function meet, one arm
  of whose test only raises (`none_raises_after_inlining.ml`; V4's). A
  function that returns an option is inlined where its result is matched
  and `None` fails an assertion. Simplify unboxes the option: the three
  ends of the function come to one continuation with two parameters,
  whether the result is `None` and the field of `Some`, and its handler
  tests the first. In the term after, once the application that gives
  `Some` goes to its arm, the others give one constant for the parameter
  that is tested; it is bound at the head of the handler and the
  `Switch` folded, and the handler is the raise alone, with its trap
  action, of a continuation of no parameters that arms apply; the term
  before has the raise in those arms. In the term after, an arm of a
  `Switch` that applies, to no arguments and with no trap action of its
  own, a continuation that has no parameters and gives no kinds, and
  whose handler is one application, now becomes that application, with
  the trap action of the handler
  (`FlambdaForwardKindedSpec.forward_kinded_arm`, as
  `FlambdaForwardSpec.forward_arm` does for the term before; an
  equivalence in the model, whose machine does not execute trap
  actions). The same series also makes the tidying of the term after
  bind outside a handler the continuations that it binds, and make the
  applications go to its arms, as its rounds do
  (`FlambdaLiftContsSpec.lift_handler_conts`,
  `FlambdaForwardArmSpec.forward_to_arms_kinded`, equivalences): the
  rounds may leave a continuation that the first refuses and that the
  tidying then makes one that it accepts. This directory has no unit of
  that, which was seen at `-O3` only.
- A call that Simplify inlines where it stands, of code that it still
  calls elsewhere (`f2_one_inlined_one_not.ml`,
  `rec_local_helper_called_outside.ml` and `rec_unrolled.ml`, T3's
  sources, and `stub_called_twice.ml`, T5b's; V4's). In the first a
  functor is applied twice, once marked to be inlined and once marked
  never to be, and Simplify does as each mark says. In the second two
  functions are defined together inside a function, neither of them
  recursive: Simplify inlines the second where the function calls it and
  leaves the call of it in the code of the first. In the third a
  recursive function is unrolled three times at one call
  (`[@unrolled 3]`), and the call that the third inlining brings is
  left; its list is a list of integers, so that the text writes no kind
  that may be null (with a list of any type the unit would be one of
  those validated for the term as read, T4-13). In the fourth the
  wrapper of a partial application, which is the code of a stub, is
  inlined where the unit applies it and left in the wrapper of a partial
  application of it, which is the code of a stub too. The validation is
  told the code that it is not to inline: with the code in the list the
  term before keeps the call that Simplify inlined, and without it the
  term before loses the call that Simplify left. Every attempt that the
  tool makes of these four units with such a list refuses them. The tool
  now asks, of a unit that the validation refuses, a function that
  inlines the calls at chosen places of the term before and then asks
  the validation (`FlambdaValidateInliningSpec.validate_unit_inlining`;
  "What the verdicts mean" says what is chosen and what is proved). At
  the default flags, which are those of the tests of this directory, the
  four units are validated so, the first two at the second asking of
  that function, and are counted apart. The limit that `status/V2.md`
  described, that what is not inlined is chosen by code and not by call,
  is so lifted for the calls that the guess of the tool finds, and the
  guess looks for no other; the list of the code not to inline, which
  the rounds of the validation are given, is as it was.
- A test made again in an arm of itself (`test_boolean_twice.ml`, T3's
  source; V4's). A boolean is tested, and in each arm of the test it is
  tested again, once negated. Simplify knows the boolean in each arm:
  the second tests are gone, and each arm of the first returns its
  number. In the validation the number that a `Switch` has decided for
  its scrutinee is put for the variable in the arm, and the tidying of
  each term folds what that has made constant (the entry of
  `divisor_tested_twice.ml`, above). An arm may then go through a
  continuation whose handler is one application. A round forwards such
  an application after its own fold; the tidying comes after the rounds,
  and no round follows it. The tidying of each term now forwards after
  its fold: that of the term before by
  `FlambdaForwardSpec.forward_applications`, a refinement, and that of
  the term after by `FlambdaForwardKindedSpec.forward_kinded`, an
  equivalence, where the arguments are known to have the kinds of the
  parameters (`FlambdaValidateSpec.tidy_expr_before`,
  `tidy_expr_after`). The unit is validated by the comparison of the
  lifted terms.
  The same series changes one test more. A continuation that a handler
  binds on its way to its `Switch` is bound outside the handler where
  every mention of it is an application
  (`FlambdaLiftContsSpec.lift_handler_conts`, N2's; the entry of
  `load_through_join.ml`, above), and a call may now return to it too,
  where it takes no parameter of the handler (`only_applied`).
  `hof_option_of_closure.ml` (T3's source) enters with that change, and
  the test of it in this directory does not exercise the change: at the
  default flags, which are those of the tests, the unit is validated
  with and without it, and its expected file is the same with the two
  tools. What the change does for the unit is with `-O3`, where Simplify
  knows the closure at the place where the arms of the test meet and
  inlines its call: there the tool built without the change does not
  validate the unit and the tool of this tree does. No test of the tree
  shows that. It is seen in the comparison of the two tools, whole
  outputs, that is made when a series is closed, a measure which
  `status/V2.md` records with the count of the units with `-O3`.
- A set of closures that a `Let` makes, given new code where the
  function that makes it is inlined (`local_rec_inlined_twice.ml`;
  C3's). A function that is always inlined makes a recursive function
  local to it, which uses its parameter, and calls it; another
  function calls the first twice. Simplify inlines the two calls.
  Each copy of the body makes its closure of the local function where
  it stands, as the closure holds a variable, and Simplify gives each
  copy new code: the term after has the code of the local function
  three times, once for the closure that the code of the first
  function makes and once for each of the two that the code of the
  second makes. The term before has that code once, and with the two
  calls inlined its three `Let`s make their closures of it. The tool
  now asks, of a unit that the attempts of the three kinds before
  refuse, a function that gives such `Let`s copies of their code
  under the code ids of the term after, and then asks the validation
  (`FlambdaValidateCopyingSpec.validate_unit_copying`; "What the
  verdicts mean" says what is guessed and what is proved). At the
  default flags, which are those of the tests of this directory, the
  unit is validated so, and is counted apart.
- A match that Simplify makes a table of, in a unit that has an
  immutable array with the same elements (`table_and_array_alike.ml`;
  V3's). A function matches four constructors to four integers that
  follow no rule, and a second function reads an immutable array of
  the source that has the same four. The term after binds two symbols
  to static arrays with the same elements, the table and the array of
  the source. The liftings are a guess of the tool: a symbol that the
  term after binds and the term before does not goes to the first
  `Let` of the term before that the tests of the guess find to make
  what the symbol is bound to, and a symbol of an immutable array that
  is left is bound as a table at the head of the term before (the
  entry of `match_constant_table.ml`, above). The symbol of the table
  went to the `Let` of the array; the two terms were then the same but
  for those two symbols exchanged, and the unit was not validated.
  Where the function asked refuses the first guess of an asking, and
  that guess gave a `Let` the symbol of an immutable array of which
  the term after has a twin (another symbol that the term after binds
  to an array with the same elements, and that the term before does
  not bind), the tool now guesses the liftings again with such pairs
  left out, three times at most, each time from the guess before, and
  asks the function of each new guess. It gives the first answer that
  is a verdict; where there is none, the answer is that of the first
  guess. Only the `Let`s that a lifting lifts where they stand or to
  the head of the unit are looked at, and only an asking that has no
  search kept from an earlier one guesses again so. A guess is not
  trusted: the verdict is that of the function asked. No definition
  and no statement changes for this, and the tool prints no new line.
  The unit is validated, at the default flags and with `-O3`; its
  verdict has the words of the tables bound in the term before ("What
  the verdicts mean").
- An element of an immutable array read twice
  (`iarray_element_read_twice.ml`; V3's, the proof P1's). A function
  that Simplify does not inline loads one element of its argument
  twice, and Simplify shares the two loads. The sharing of primitives
  now takes the load of an element of an immutable array (`Array_load`
  marked `Immutable` or `Immutable_unique` in
  `FlambdaCseSpec.shareable`): a later `Let` of the same primitive
  applied to the same arguments has the variable of the first as its
  defining expression, while the first is still among what the pass
  has recorded. The pass compares the primitive and the arguments as
  they are written, and not what array or what element they stand for.
  In the model such a load has no value on an array that has an
  identity, which a store can change (decision D49, as it stands), so
  where the first has a value the array is an immutable array or a
  symbol bound to one, and the second has the same value; the comment
  at the definition says what would take the case out. A load marked
  `Mutable` is not shared. The unit is validated, at the default flags
  and with `-O3`.
- A guess of the liftings that the function asked refuses, in a unit
  of which another guess is accepted
  (`local_rec_below_a_let_and_in_an_arm.ml`,
  `pair_of_unknowns_after_inlined_pair.ml`; V4's). The liftings are a
  guess of the tool (the entry of `table_and_array_alike.ml`, above).
  In the first unit a recursive function that captures nothing is made
  below a value that is not known, and its code calls it by name.
  Simplify binds its closure to a symbol at the head of the unit. The
  guess lifted the closure where its `Let` stands, below the `Let` of
  the value, which stays, and the code that names the symbol is bound
  above that: the two terms differed by the place of one binding. In
  the second unit a call that Simplify inlines makes a pair of
  constants, which the term before as it is given does not make, and
  the unit then makes a pair of two values that are not known. The
  guess gave the symbol of the pair of constants to the `Let` of the
  second pair, whose fields agree with anything.
  Where the search of an asking ends in a refusal, and the asking is
  not one that asks nothing, the tool now makes the search once more
  with two choices of the guess changed; any liftings that it proposes
  are checked by the same function. Where the guess gives the `Let`,
  outside code, of a closure with no value slot a symbol of the term
  after, that `Let` is given to the lifting to the head of the unit
  and is not lifted where it stands. Where it chooses the symbol of a
  `Let` of a block that has a field and whose fields are all variables
  of which nothing is known, it prefers, among the symbols that fit,
  one whose block has a variable among its fields, and takes the first
  that fits if there is none. A verdict of the second search is the
  answer of the asking; where it is refused too, the answer is the
  refusal that there was. A verdict that the search of an asking gave
  before is still the answer of that asking, as the second search is
  made only after a refusal; a unit that was validated may now be
  validated by an earlier asking. A guess is not trusted: the verdict
  is that of the function asked. No definition and no statement
  changes for this, and the tool prints no new line. At the default
  flags, which are those of the tests of this directory, the first
  unit is validated. The second is not validated. Both its terms have
  a primitive of the heap, and a lifting of the validation lifts
  nothing in a term given to it that has one ("What the verdicts
  mean"). The second check, which printed no line of it, now prints
  "related once lifted", which is no validation of the unit.
- A closure that a recursive function captures and gives to a function
  that is inlined (`captured_closure_given_to_inlined.ml`; C3's). A
  function that the unit exports makes a closure that holds its
  argument, and then a recursive function that holds that closure and
  gives it to a function that applies what it is given twice and is
  always inlined; it returns the recursive function. Simplify inlines
  the two calls of the closure in the code of the recursive function.
  The term before, once the rounds have inlined the function that
  applies twice, has two indirect calls of what a value slot of the
  recursive function holds. The rounds that use what `my_closure` is
  known to be from its code now know the code of the closure that such a
  value slot holds, under the conditions that "What the verdicts mean"
  gives: every set that is made with the code of the recursive function
  gives the slot a variable that is in sight there as a closure of one
  code, here the variable that the `Let` of the closure binds, so the
  two calls are made direct. The unit is validated, at the default flags
  and with `-O3`, and the verdict has the two lines of what its liftings
  asked of their rounds.
- The length of a mutable array that a `Let` in scope has made, and the
  array kind that a store carries (`made_array_length.ml`,
  `made_array_checked_store.ml`, `made_array_store_of_integer.ml`,
  `made_array_length_in_two_calls.ml`,
  `made_array_checked_store_in_loop.ml`; P1's, decision D56). A mutable
  array is made of values that Simplify does not know. Simplify knows
  its length from the allocation: it has the number where the term
  before takes the length, and it removes a check of an index that takes
  it. The sharing of primitives of the validation now knows, under a
  `Let` of a variable to the allocation of a mutable array, that the
  variable holds an array made of so many arguments, where none of them
  is the variable itself; a later `Let` of the length of that variable
  that it visits while that is known becomes a `Let` of the number,
  where the number is a tagged immediate
  (`FlambdaCseSpec.made_length_named`), and the rounds of the term after
  apply that rule alone (`FlambdaMadeLengthsSpec.fold_made_lengths`).
  Nothing is known of the elements, which a store may change, nor of an
  array that no `Let` in scope makes: a parameter, what a call returns,
  what is read from a block or from a closure. Where the front end
  writes a store of an integer as one into an array of values that the
  collector may ignore, Simplify, which knows from the allocation that
  the elements are integers, writes it as one into an array of
  immediates: the validation writes the array kind of a store in one
  form, and the tool prints the line of the stores by array kind ("What
  the verdicts mean"). The first three units are validated, at the
  default flags and with `-O3`, and are not by the tool of 4ba710c501.
  `made_array_length_in_two_calls.ml` is validated without the rule too
  at the default flags, which are those of the tests of this directory:
  it is here so that the rule stays in the rounds of both terms; with
  `-O3` the tool of 4ba710c501 does not validate it, and the tool of
  this tree does. `made_array_checked_store_in_loop.ml` is not
  validated: the array is made before a loop and the check of the index
  is in the loop, and neither pass carries what it knows of the
  allocation into the handler of a recursive continuation, so the term
  before keeps the length in its loop where the term after has the
  number.

A count over 600 units made at random is in `census/README.md`, with
one over 240 more that have exceptions, loops and recursive functions.

## Units that test the arithmetic

Some units have no unknown input: Simplify computes the whole result,
and the machine computes it from the term before, so the two runs
agreeing says that the model's primitives and Simplify's compute the
same thing there. They are written for the edges: arithmetic that
overflows at 63, 32 and 64 bits (`overflow_wrap.ml`,
`int32_overflow.ml`, `int64_overflow.ml`), shifts at the first and the
last bit (`shift_edges.ml`), conversions that lose bits or extend a
sign (`conv_truncate.ml`), comparisons of negative numbers
(`compare_signed.ml`). Each result is exported by itself and not in a
tuple, as a tuple of constants is made a symbol and the unit would
then wait for lifting.

## What a check costs

The tool gives the validator fifty rounds. A round applies each
normalisation once to the whole term, and the normalisations are at
worst quadratic in the size of the term; the rounds are there because
one rewrite makes room for another, and most units are done in three or
four: a run of rounds stops after the first one that gives back the
term it was given, up to bound names (`FlambdaValidateSpec.iterate`).
A round inlines every direct call of known code that passes the
checks of the inlining, but for the code that the tool names to the
validator, a guess that is not trusted; a call that gives such code
more arguments than it has parameters may still be split into the
full call and the call of its result, where
`FlambdaNormaliseSpec.over_applied_b` passes. The
tool tries three lists in turn: the code that is recursive (that calls
itself, at once or through other code, in either term) with the code
that the term after still calls directly; then the recursive code
alone; then, where some code of the first list is not in it, the
first list with the recursive code of the term before taken from the
term that the validation normalises: the direct calls of its stubs
made indirect, its calls of code that never returns cut, its loops
made (V4's: code whose self tail calls have all become jumps is not
recursive there, and Simplify inlines it like any other code;
`local_loop_inlined.ml`). Without such a list
each round unrolled a recursive function once more on both sides, and
`fib.ml` did not end. With such a list a round inlines only code that
is not recursive in the term that it is given, so the terms stop
growing.
Each attempt may go on to guess what Simplify lifted, in four
liftings at most, and which value slots it removed, and where one of
the first two fails it is made again at the second asking. The third
is asked once: a second asking normalises the term before as it is
given, before its loops are made, and each of its rounds would inline
code that is recursive by its self tail calls into itself once more.
Where every attempt refuses the unit, the attempts are made once more of
the term before with its known immutable arrays read and, where those
refuse it too, once more of the term before with chosen calls inlined
first, unless the guess of the tool gives no choice or the term that its
choices give is the term before name for name ("What the verdicts
mean").
Where those refuse it as well, and the term after makes by a `Let`, in
its code too, a set of closures of code that the term before does not
bind, attempts are made of the term before with copies of code for
such sets, once, with the rounds that the lines under the verdict
name (the same section).
Where the function asked refuses the first guess of an asking that
has no search kept from an earlier one, and that guess gave a `Let`
the symbol of an immutable array which has a twin in the term after,
the liftings are guessed again and the function asked again, three
times at most (the entry of `table_and_array_alike.ml`, in the list
above).
Where the search of an asking that asks something ends in a refusal
after that, the search is made once more with two choices of the guess
changed (the entry of `local_rec_below_a_let_and_in_an_arm.ml`, in the
list above).
The 546 units here were checked in 11.7 seconds of processor time
together in one run (6.0 user, 5.7 system), and in 11.9 in a second run
of the same tool (6.1 and 5.8): less than half of each is the system's
part. Other builds ran on the machine during both. In two runs made one
after the other with the same inputs and flags, on the same 546 units,
the tool of 4ba710c501 took 11.9 seconds and the tool of this tree 11.8.
On the 541 units that the directory had at 4ba710c501, in two such runs,
the tool of 8e6e4ba1b2 took 10.7 seconds and the tool of 4ba710c501 took
11.2. That time, and those that `census/README.md` gives for the tool of
4ba710c501 where it says "when it was measured there", were measured
with an executable that is, byte for byte, the one that was built at
4ba710c501. On the 538 units that the directory had at 698cf003e5, in
two such runs,
the tool of that commit took 11.8 seconds and the tool of 8e6e4ba1b2
took 11.4. That time, and those that `census/README.md` gives for the
tool of 8e6e4ba1b2 where it says "when it was measured there", were
measured with the build that C2 made of its series of the conversion
that makes cells when that series was closed, and not with a build made
at 8e6e4ba1b2: between the two the source of the tool and one Rocq file
change in comments only, and no other Rocq file changes. On the 536
units that the directory had at 255f6a36c2, in two such runs,
the tool of that commit took 10.5 seconds and the tool of 698cf003e5
took 10.5. That time, and those that `census/README.md` gives for the
tool of 698cf003e5 where it says "when it was measured there", were
measured with the build that V3 made of its series of the loads of an
immutable array when that series was closed, and not with a build made
at 698cf003e5: between the two the source of the tool and the Rocq files
do not change. On the 529 units that the
directory had at 5a514806c3, in two such runs, the tool of that commit
took 10.3 seconds and the tool of 9c6e3fc02c took 10.2. The times given
for the tool of 9c6e3fc02c, here and in `census/README.md`, were
measured with the build that C3 made of its series of the copies of
code, when that series was closed, and not with a build made at
9c6e3fc02c: the source of the tool is the same, and between the two
the Rocq files change in comments only, but for two files that the
tool is not extracted from. On the 516 units that the
directory had at b006dd4b0d, in two such runs, the tool of that commit
took 10.5 seconds and the tool of 24cd3c9252 took 10.6; when it was
measured at that commit, the tool of b006dd4b0d took 16.1 and 14.0.
These are totals of processor time on a machine whose processors the
runs shared with other builds under one quota: they are not a comparison
of the speed of two tools run alone. The 347 units that the directory
had at 87cdbf3e01 took 47
seconds with the tool of that commit, 28 of them for the four units
with a match that Simplify makes a table of. Two things have come
since, each measured with the tool before it and the tool after it
on the tree of the change. The four units are validated by the load
from a table (above): 26 to 28 seconds for the four before, less
than one after. And a run of rounds stops, where it made all its
fifty rounds whatever the term: the 358 units that the directory
then had took 20.6 seconds before and 7.4 after. Neither changed
what the tool prints of a unit that the directory had, but for the
four. `census/README.md` has the times of the generated units.

`FLSEMANTICS_PRINT_TERMS=1` makes the tool print, of a unit that is not
validated, two terms that the validation compared, for each list, and
how the term before was made: it is not always the last pair that was
compared.

`FLSEMANTICS_REGIONS_ONLY`, set to anything, makes the tool not ask
the validation where it would ask it, which is once it has run the
two terms and compared the runs: there it prints "the validation:
not asked (FLSEMANTICS_REGIONS_ONLY is set)" in the place of the
line of the calls once those of known code are inlined, of the
verdict and of the lines under it, and the runs, the check of kinds, the
check of static names, the comparison of the two runs, the line of the
check of regions and the
line of the ends of regions as without the variable. Where the tool
stops before it asks the
validation (a unit that is not checked, a term after that is not
well-formed, a fault of the prefix made from `.cmx` files), the
variable changes nothing of what it prints. What the tool prints
with the variable set is no verdict on the unit. No test of this
directory sets the variable; one of `../regions/` does.

## What a validation does not say

- A unit is validated on the reading that a string is immutable.
  `Bytes.unsafe_to_string`, `Bytes.unsafe_of_string` and the
  conversions between arrays and immutable arrays are erased before
  Flambda 2 (`from_lambda/lambda_to_lambda_transforms.ml`), so the test
  tool cannot see them; it refuses a unit in which a primitive on bytes
  arrives, but the verdict on a unit that reads a string says nothing
  of a string that is the block of bytes written while it is read
  (X5-3).
- Regions: see above.
- The refinement that a validation concludes does not by itself say
  that a symbol which the term after exits with is bound. It
  compares the two exits by their continuation, their values and
  their heap, and a symbol is a value whether it is bound or not
  (decision D57). The statements of `FlambdaExitBoundSpec.v` add it
  for two terms run from the same imports and the empty heap (the
  header of that file has the two states), where the run of the
  term after refines that of the term before and the term after
  passes three tests: if the term before exits, the values that the
  term after exits with, and those in the cells of its heap, mention
  only symbols that its static environment binds when it exits.
  That a symbol is bound is not that it is bound to what the term
  before bound it to: the two static environments are not compared.
  The tool asks the first of the tests, `well_formed_unit`, of the
  term after before it validates anything. What the two others
  answer of each of the two terms (`statics_in_scope`, given the
  imports of the unit, and `interface_conts_not_rebound`, both of
  `FlambdaStaticsInScopeSpec.v`) it prints on the line "the check of
  static names", after the line of the check of kinds. Nothing is
  refused for that line and no verdict depends on it. The tool
  prints it of 543 of the 546 units of this directory (not of the 3
  that are not checked), and it says "passes before, passes after"
  of each.
- What is validated is the term the parser reads back from what
  `-drawfexpr` prints (see "validated" above), as the tool's Simplify
  simplifies it. The printed term has no mark for a functor, for
  `[@cold]`, for opaque code or for the modes of parameters, and the
  parser gives each a default
  (`middle_end/flambda2/parser/fexpr_to_flambda.ml`); for the modes
  of parameters the tool puts the least that can be said in the
  place of the default (below, with X5-11). Simplify reads
  them. So the tool's Simplify can decide otherwise than the
  compiler's on the same unit. One case is known. For code with no
  inlining attribute, the decision made at its declaration
  (`make_decision0` of
  `middle_end/flambda2/simplify_shared/function_decl_inlining_decision.ml`)
  compares the size of the code with the thresholds for functors
  (`small_functor_size`, `large_functor_size`) if the code is marked
  as a functor, and with those for functions if not; a functor no
  larger than `small_functor_size` is to be inlined. The tool's
  Simplify, to which the functor is a function, compares it with
  the thresholds for functions and may leave the call. The pair
  validated is then not the pair that the compiler made (discrepancy
  V2-6). For code with `[@inline]` or `[@inline never]` that
  decision does not look at the mark of a functor; the other marks
  are lost for such code as for any other.
- The printed term loses kinds and marks as well (found by X5, who
  compared the term in the compiler's memory with the term read
  back; nothing in the tree makes that comparison). Four of the
  losses are of something that Simplify reads, and with each the
  tool's Simplify knows less than the compiler's:
  - the kind of the result of an application, where the arity of the
    result is not written: the subkind is lost and the result is read
    as one that may be null, and the type of the result of a call
    that is not inlined comes from that arity where nothing more
    precise is known of the result (`T.unknown_types_from_arity` in
    `middle_end/flambda2/simplify/simplify_apply_expr.ml`);
    `identity.ml` has it;
  - the kinds of the fields of a `Make_block`, whose subkinds are
    lost (discrepancy V2-2): Simplify meets the type of each argument
    that is a variable with the kind of its field
    (`simplify_make_block` in
    `middle_end/flambda2/simplify/simplify_variadic_primitive.ml`);
    `exported_list.ml`;
  - the mark `[cold]` of a continuation handler, such as the handler
    of a failed bounds check, which is dropped (`DE.mark_as_cold` in
    `middle_end/flambda2/simplify/simplify_let_cont_expr.ml`);
    `string_get_checked.ml`;
  - debugging information (discrepancy X5-10). The fields of a
    static block carry it (`Simple.With_debuginfo.t`, in
    `middle_end/flambda2/terms/static_const.ml`); the order on static
    constants compares it; and the constants that may be shared are
    looked up by that order (`find_shareable_constant` in
    `middle_end/flambda2/simplify/env/downwards_acc.ml`, asked by
    `lift` in `middle_end/flambda2/simplify/lifting/reification.ml`).
    A block that Simplify lifts has the debugging information of its
    allocation on each field, a constant that the front end makes
    has none, and the text of terms has none at all. So the tool's
    Simplify shares a lifted block with an equal constant that the
    unit binds already, where the compiler's makes a second symbol.
    No unit of this directory changes verdict by it; three generated
    units did (`census/README.md`).

  One more loss is of something that Simplify does not read: whether
  a value slot always holds an immediate, which the parser reads as
  false (`add3_exported.ml`). The kind of the slot is kept.
  `middle_end/flambda2/to_cmm/to_cmm_set_of_closures.ml` reads the
  mark, and the tool does not run that pass.

  With one more the tool's Simplify is told more than the compiler's:
  a value that may be null is read as one that never is, wherever the
  text writes its kind. That is discrepancy T4-13, and the 14 units
  that have such a kind are counted apart (see "What the verdicts
  mean"). None of these losses makes a verdict wrong of the pair that
  the tool validates: they are a distance between that pair and the
  compiler's.

  With one more still the reader says more than the compiler's term
  does, and the tool takes it back before its Simplify runs. The
  text has no modes for the parameters of code, and the reader gives
  every parameter the mode of a value on the heap (`param_modes` in
  `middle_end/flambda2/parser/fexpr_to_flambda.ml`). Simplify, run on
  a term read so, can prove that a parameter which may be on the
  stack is a block of the heap, and return that block where the
  compiler makes a copy of it on the heap (discrepancy X5-11). The
  compiler, which does not read its terms from text, is not
  affected. It was seen on a unit that is not in the tree (by X5;
  nothing in the tree reproduces it): compiled through the text,
  the program read a block of the stack that a later call had
  written over, and compiled directly it was right. So the tool
  gives every parameter of every piece of code that the unit binds
  the least that the compiler can say of where a value is
  (`Alloc_mode.For_types.unknown ()`: heap or stack, and heap in a
  compiler configured without stack allocation, where nothing is on
  the stack), before anything else is done with the unit
  (`with_unknown_param_modes` in
  `middle_end/flambda2/tests/tools/flsemantics.ml`). Of where a
  parameter of that code is, the modes then tell the tool's Simplify
  no more than those of the compiler's term tell the compiler's,
  and less where the compiler's term says heap and the compiler
  allocates on the stack: a loss like those above, a distance and
  not an error. That is of these modes only.
  What else the reader gives a default for stays as the reader gives
  it (the marks named above), and code of another unit, which
  Simplify reads from a `.cmx` file, has the modes that the compiler
  wrote there. No verdict rests on the reader's mode heap for a
  parameter of code that the unit binds, and no unit is counted
  apart for it. In the compiler's own term (`param_modes` in what
  `-drawflambda` prints) 20 of the 331 units that this directory
  had when the tool was changed have a parameter whose mode is not
  heap, each of them heap or stack: `curry`, `inline_record`,
  `labelled_optional`, `let_polymorphism`, `list_head`,
  `list_length`, `list_map`, `list_sum`, `match_guard`,
  `match_variant_local`, `nested_record`, `nested_tuple`,
  `option_bind`, `option_map`, `polymorphic_variant`,
  `project_record_two_levels`, `record_update`, `result_map`,
  `same_field_chain_twice` and `tree_root`.
  Of the 215 units that have entered since, 32 have:
  `stack_argument_result_returned`, whose source says `local_` of a
  parameter, the first source of this directory that does, and 31 that
  entered after it: `attr_local_in_match`, `clos_partial_local`,
  `exclave_after_local_work`, `exclave_pair`,
  `flip_applied_in_full_and_in_part`, `guard_falls_to_later_case`,
  `guard_on_or_pattern`, `hof_option_of_closure`,
  `inl_callee_region_twice`, `late_three_values_interleaved`,
  `local_argument_partial_application`, `local_closure_given_partially`,
  `loop_for_local_pair_each_turn`, `nest_list_of_options`,
  `optional_between_positional`,
  `optional_default_uses_earlier_argument`,
  `optional_erased_for_higher_order`,
  `optional_given_omitted_none_some`, `or_in_list_positions`,
  `or_two_variables_swapped`, `param_copy_to_heap`,
  `poly_abbreviation_patterns`, `reraise_in_kept_function`,
  `stack_pair`, `tail_call_after_region`, `tail_nontail_annot`,
  `try_in_each_turn`, `try_returned_exported`,
  `tupled_partial_application`, `twice_nested_same_scrutinee` and
  `var_twelve_mixed`. Twelve of the 31 say `local_` in the source. Each
  unit is looked at so before it enters. By the same print, the term
  before Simplify of 64 of the 546 units of this directory has an
  allocation mode that names a region, on
  an allocation or on a call ("What the conversion does to a unit" says
  what is made of regions). What was run for the
  change of the tool: the tests of this directory before and after
  it, in which no expected file has another verdict or another
  count of steps, those of the 20 units among them (two files have
  other names of variables in the message of a primitive that the
  model has not got, as the unit is built again with fresh names
  for what it binds); and the tool on the 840 generated units of
  `census/` before and after it, which have the verdicts they had.

  The model erases the regions of allocation and their lifetimes
  (the conversion makes of a local allocation one on the heap,
  discrepancy V2-3): equal outcomes of the two runs, or a
  validation, whether plain or counted apart, do not establish that
  a block which is returned is still alive.
- Of a primitive the text loses two more things. It does not write
  the kind of an `Opaque_identity`, and the reader gives every one
  the kind of values. The tool gives each the kind of its argument,
  as the reader has it, before anything else is done with a unit
  read from a text (`with_lost_kinds` in
  `middle_end/flambda2/tests/tools/flsemantics.ml`; discrepancy
  C2-27). That is the kind which the compiler's term had on two
  conditions: the kind that the reader has for the argument is the
  kind of that argument in the compiler's term, and the primitive
  there had the kind of its argument. Nothing checks either;
  without those conditions this step makes no claim to recover the
  compiler's kind. By C2's count of what
  `-drawflambda` prints, which nothing in the tree makes, every
  `Opaque_identity` of the 347 units that the directory had at
  87cdbf3e01 is of a value; the tool with this step prints of each
  of them what it printed without it. And the text writes the two
  tests of being an integer
  alike, the one that is for variants and the one that is for any
  value (`Is_int_generic` in what `-drawflambda` prints); the reader
  gives the one for variants. On a value that is not a variant, a
  closure for one, the term read is then undefined in the model
  where the program answers 0 (discrepancy C2-26). The tool does
  not put that right. No unit of this directory has the test that
  is for any value, and `from_ocaml.sh` fails on a unit that has it
  and that `generic_is_int.txt` does not name; that list is empty.
- Of a call the text loses the arity of its arguments, where the
  call is direct or of unknown arity, and the reader gives such a
  call one value for each argument: a unit that gives a function an
  unboxed number was undefined in the model as read (discrepancy
  C2-5). In the same step as for an `Opaque_identity`, the tool
  gives a direct call the kinds of the parameters of the code that
  it names, which is what the conversion from Lambda made the arity
  of such a call, and a call of unknown arity the kinds of its
  arguments as the reader has them, on the two conditions said
  above (`unboxed_float_argument.ml` and the three units beside it,
  each undefined as read before). What it cannot give back is the
  grouping: in the compiler's term an argument that is an unboxed
  product is one component of several kinds, or of none, which an
  application counts as one against the parameters of the code, and
  the text has as many arguments as kinds, at a call and in the
  parameters of code alike. A direct call has the parameters of its
  code. Of a call of unknown arity nothing tells: read flat, a
  partial application that stops before a parameter of no kind is
  a full application, and so is an over-application whose last
  argument has no kind (discrepancy C2-30; the tool printed an
  integer where the program has a closure, and a closure where it
  has an integer, of two units that are not here). So
  `from_ocaml.sh` fails on a unit that has a call of unknown arity
  if such a call has a product among its arguments or code of the
  unit has one among its parameters; no unit of this directory is
  one. The text does not say that a call is a probe either
  (discrepancy C2-17), and `from_ocaml.sh` fails on a unit that has
  one.
- Two things that the tool refuses, to stay clear of two ways in
  which the term of the model may not be the program. Neither
  refusal shows that it is not: each refuses more than it need. A
  binding whose variables are phantom, which Simplify may keep with
  `-g` where it would delete an unused binding, and which the
  program does not run: the conversion refuses it, where it
  converted it as a `Let` that the model runs (discrepancy C2-19;
  `pairs/README.md` has a pair). A body taken from a `.cmx` that
  has one is left out of the prefix, and a unit whose term after
  has one is not checked. The text does not say that mode, so the
  refusal does not see a phantom binding in a term read from a
  text. And a unit of which, as either term is written, some use of
  the return continuation gives it something else than the symbol
  of the unit, or to which a call returns: other units read the
  data of that symbol, and the model observes what the unit gives
  its return continuation (discrepancy C2-21). It is a test of the
  form of the term: a unit that gives the continuation its own
  symbol through a variable, or from a call, fails it. No unit of
  this directory fails it.
- That distance has been measured for the units of this directory
  (by X5, with the tool and the 259 units as they are at ba85db2822;
  nothing in the tree makes the measure). The check of the tool was
  given the two terms of the compiler's own run as they are in its
  memory: the term given to Simplify and the term that the compiler's
  Simplify returned, taken through the two hooks of
  `driver/compiler_hooks.mli` (`Raw_flambda2`, `Flambda2`), with no
  text written or read and no second run of Simplify. The conversion
  was the tool's with one change: the kinds of the fields of a
  `Make_block` were dropped, as the text drops them. (As it stands,
  `convert_variadic_primitive` of
  `middle_end/flambda2/tests/tools/flsemantics.ml` takes only fields
  of any value; of the terms in memory it refuses 58 of the 259
  units, for a subkind in 51 and for a field that is only never null
  in 7; 57 of the 58 are converted with the change.) What follows is
  what the check answers with that changed conversion, which is in no
  tree. It is no proof that dropping those kinds is faithful, and the
  change is not made to the tool, which refuses such a block on
  purpose: Simplify takes an argument that is a variable to be of
  the subkind of its field, and makes the primitive `Invalid` where
  it cannot be (the comment of `convert_variadic_primitive`). At the
  default flags, of the 259 units:
  - 252 are converted in both ways, and for each of them the tool
    prints of the compiler's pair the lines that it prints of its
    own: the verdict, the comparison that succeeded, the steps of
    the two runs, the check of kinds. 242 are validated, 8 are not,
    2 are undefined before Simplify.
  - 5 are refused by the conversion in both ways.
  - 2 are validated of the tool's pair, and of the compiler's pair
    the term after Simplify is not well-formed: `div_unknown.ml` and
    `string_get16_checked.ml`. The compiler's Simplify keeps the
    handler of the raise, which the front end marks `[cold]`, as a
    handler of its own (`add_non_inlinable_continuation` in
    `middle_end/flambda2/simplify/env/upwards_env.ml` keeps the
    handler of a cold continuation unknown to its uses). The binding
    of the module block is then in a handler whose body can leave
    through another continuation, which
    `FlambdaWellFormedSpec.statics_at_toplevel` refuses, and which
    Simplify refuses too when it is given its own output
    (discrepancy X5-9, a second shape of C5-11). The tool's
    Simplify, to which the handler is not cold, puts the raise in
    the arm of the `Switch`, and the tool validates that term.

  With `-O3`: 228 validated, 22 not, 2 undefined before, the same 5
  and the same 2, and again the same lines for all 252.

  The two terms after Simplify, as `-dfexpr-to` writes them, compared
  as texts up to names, are the same for 253 of the 259 units (252
  with `-O3`). The others are the two above and units in which only a
  kind that is written differs (`call_result_tested.ml`,
  `partial_exported.ml`, `tupled_recursive.ml`,
  `tupled_unknown_pair.ml`; with `-O3`,
  `exported_funs_in_arms_used.ml` and `returned_closure_of_pair.ml`
  in place of the first).

  For the 13 units of T4-13 that the directory had then, with both
  sets of flags, the two terms
  after Simplify as printed are the same up to names, and with the
  changed conversion the check prints the same lines of the
  compiler's pair as of the tool's. The comparison of the terms is
  of texts: it does not compare the kinds and the marks that the
  text does not write. Nothing is claimed of dropping the kinds of
  fields. The 13 stay counted apart: the measure is not made in the
  tree, and is of the units and the tool of that commit. The
  fourteenth unit of T4-13 is not among the 259.

What no example here validates, because the model has not got it:
arithmetic on floats, calls of C functions (polymorphic comparison among
them), an exception that the unit declares (it is made by a call of a C
function), objects, laziness. And no example here has one of the cases
of the primitives of the heap that the conversion refuses though the
model has the primitive: a store marked `Initialization`, a block of
floats or a mixed block that is written, an array that is not of
values, a copy of a block or of an array ("not checked", in "What the
verdicts mean").
(`fst` and `snd` are in the model: the standard library declares them
with `%field0_immut` and `%field1_immut`, a load that cannot change,
with a tag and a size that are not known: `fst_snd.ml`. It is `!`, and
`%field0` written by hand, that load a field marked as one that may
change.)

## The compiler's own run

`../../tools/ownrun/flsemantics_ownrun.ml` is a second way to check a
unit, beside the first and not in its place: every count in this file,
but those of this section, is of the first.

    sh own_run.sh <flsemantics_ownrun.exe> X.ml

which is, in a directory of its own,

    flsemantics_ownrun.exe -nopervasives -nostdlib -stop-after middle_end \
      -c [X.mli] X.ml

The program is the compiler: its driver (`driver/oxcaml_main.ml`) with
two hooks registered (`driver/compiler_hooks.mli`). The hook
`Raw_flambda2` is given the term that the compiler gives to Simplify,
and the hook `Flambda2` the term that the compiler's own Simplify
returned (`flambda_to_flambda0` in `middle_end/flambda2/flambda2.ml`).
The two terms are given, as they are in memory, to the check of the
tool (`Flsemantics.check`), which prints what it prints in the first
way, under the name `X.fl`.

It differs from the first way in three things:

- No text. The first way checks the term that the reader makes of
  `X.fl`, the text that `-drawfexpr` printed. Here the term before is
  the one in the compiler's memory, which has what the text does not
  write ("What a validation does not say"). The test made of the text
  for a call of unknown arity whose result is not one value
  (`result_arity_lost`) is not made: the term has the arity. Nor is
  the term changed as a term read from a text is before it is checked
  (`as_read_from_text`, which gives each `Opaque_identity` the kind
  of its argument, each direct call and each call of unknown arity
  an arity for its arguments, and every parameter of code the mode
  unknown): the term in memory has its kinds, its arities and its
  modes.
- No second Simplify. The first way runs Simplify itself on the term
  read. Here the term after is the one that the compiler's Simplify
  returned, in the compilation that made the term before.
- The load path. The first way looks for the `.cmx` files of other
  units only in the directories given by `-I` (`init_load_path`). The
  program has the compiler's load path (`Compmisc.init_path`): the
  current directory unless `-nocwd` is given, the directories of `-I`
  and `-H`, and the standard library unless `-nostdlib` is given.
  `own_run.sh` gives `-nostdlib` and no `-I`, and runs the program in
  a directory that has only the unit. So the rule of a unit of this
  directory finds no `.cmx` file of another unit, where nothing else
  adds to the compiler's load path (`OCAMLPARAM`); a run by hand
  reads what the compiler would read.

The rest is the first way's, the same code and the same extracted
model: the conversion, the test that the term before is well-formed,
the two runs and their comparison, the validator, and the units asked
about (those whose `.cmx` file the compilation has read).

So a verdict printed in this way is of the pair that the compiler
made: of two terms, as a verdict of the first way is, with the same
labels and the theorems that those labels have. Like a verdict of the
first way it rests on the conversion. It does not rest on the
printer, on the reader or on the tool's run of Simplify being the
compiler's; in their place it rests on the two hooks being given the
terms that the compiler goes on with. What a verdict of the first way
does not say ("What a validation does not say"), this one does not
say either, but for the losses of the text. And it says nothing of
what the compiler then makes of the term after: the reaper, where
that is enabled, the translation to Cmm, the back end and the link
are not looked at. Of a unit on which the compiler does not run
Simplify (`-Oclassic`, or an attribute of the unit) the program says
"not checked".

What it finds of the 347 units of this directory at 87cdbf3e01. It
was measured with the program of this section on the tree at that
commit, at the default flags; no `X.own_expected` is in the tree, so
no rule of the tree makes the measure again. The units of `pairs/`
are not run: `own_run.sh` does not give the program the `.cmx` file
of a first unit.

- Of the 325 units that the first way validates at that commit (283,
  the 14 counted apart for T4-13 and 28 with their imports, by the
  tables that this file has there), this one validates 241 and says
  "not checked" of 78, for one clause of the conversion. Of a
  `Make_block` it takes only
  fields of any value (in `convert_variadic_primitive` of
  `../../tools/flsemantics.ml`, the test that the kind of each field is
  `Flambda_kind.With_subkind.any_value`), and the term in memory has
  the kinds of the fields, which the text has not. With that clause
  changed to ask only that each field is a value, a change that is in
  no tree and of which nothing is proved, it validates 319 of the 325,
  and no unit that the first way does not.
  `empty_array_phys_equal.ml`, whose run is undefined in the model in
  both ways, is not among the 325, and prints in this way what it
  prints in the first.
- The 6 others have the handler of a failed check (of a divisor, of an
  index), which the front end marks `[cold]`; the text does not carry
  the mark. The compiler's Simplify keeps a cold handler as a handler
  of its own; the tool's, to which it is not cold, puts the raise in
  the arm of the `Switch`. Of `div_unknown.ml`,
  `string_get16_checked.ml` and `string_get_twice.ml` the compiler's
  term after then binds the module block in a handler whose body can
  leave through another continuation, and is not well-formed
  (discrepancy X5-9, above): the program exits 1, as the tool does of
  such a term, so these three can have no `X.own_expected` as long as
  that stands. Of `string_get_in_branch.ml`,
  `string_get_two_indices.ml` and `string_length_after_join.ml` the
  two runs agree, and the validator does not accept the pair in which
  the handler is kept: "not validated".

A unit that has `X.own_expected` beside it is checked in this way by
the rules of this directory, which compare what the program prints
with that file; a unit that has none is checked in the first way
only.

The program checks one unit in a process, as the tool does. Given a
second, it says of it "not checked: a second unit in one process",
checks nothing from then on, and exits with 2; `../own_run/` has a
control of that, in which the compiler does not run Simplify on the
first unit.
