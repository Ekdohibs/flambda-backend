# Trust ledger of the Rocq developments

What the theorems rest on that no proof covers: every axiom, and every
piece of OCaml that is trusted. Each has its reason. A `Parameter` is an
axiom too: it assumes that something of its type exists.

The first part of each section is checked. `make -C rocq assumptions`
fails if

- Rocq reports an axiom that this file does not list, or this file lists
  one that Rocq no longer reports (the report is the one of `rocq check`
  on every module of `rocq/Makefile`, printed under "Global assumptions");
- a statement of a Spec file of the Flambda 2 libraries depends on an
  axiom that is not in the section "Flambda 2" below (`Print Assumptions`
  on each theorem that `make claims` checks).

The trusted OCaml cannot be checked by a tool. It is listed for the
reviewers, who read it against the compiler.

The rules, for the Flambda 2 work (`middle_end/flambda2/rocq/`
`PARALLEL-PLAN.md`, sections 4, 5 and 7):

- Nothing is `Admitted`, and there is no axiom that is not here.
- No workstream adds to this file without a reviewer's sign-off. The
  reviewers (R1, R2) own it.
- It should shrink. Each entry of the Flambda 2 section says what would
  remove it.

The format, which the check relies on: an axiom is a list item that
begins with its fully qualified name between backquotes, and nothing else
begins so. The second check takes the Flambda 2 axioms to be those
between the heading "Flambda 2" and the next heading of its level.

## Flambda 2

The libraries `OxCamlFlambda2` (`middle_end/flambda2/rocq/`) and
`OxCamlFlambda2Types` (`middle_end/flambda2/types/rocq/`).

### Axioms

None. No statement of these libraries depends on an axiom: none uses
classical logic, Rocq's primitive integers or the models of OxCaml's
types below. (There were seven, all about the primitives, which the model
left abstract. The primitives are now data,
`middle_end/flambda2/rocq/FlambdaPrimitiveModel.v`, and what they compute
is a definition, `FlambdaPrimitiveSpec.v`.)

### Trusted OCaml

None of this is used by a proof. It is what connects the proofs to the
compiler in the tests of `middle_end/flambda2/tests/semantics/`: when the
tool reports a pair of terms as validated, the theorem
(`FlambdaValidateSpec.validate_unit_sound`) is about the two model terms,
and what follows is why they stand for what Simplify was given and what
it gave back.

- **The term that Simplify is given is read from text.** The tool
  does not take its two terms from the compiler's own run. The
  compiler writes the unit as it is before Simplify as text
  (`-drawfexpr-to`); the tool reads that text with the compiler's
  parser (`parser/fexpr_to_flambda.ml`), changes the unit read in
  the one way that is said below, and runs Simplify on the result. A
  verdict is about the conversions to the model of that pair, the
  unit read so changed and what Simplify makes of it, under what the
  entries below trust of the conversion. It is not about the terms
  of the compiler's own run. The text does not carry everything of a
  term. It has no modes for the parameters of code, and the reader
  gives every parameter the mode heap (`param_modes`; discrepancy
  X5-11). Simplify on a term as the reader makes it can replace a
  copy on the heap of a parameter that is local by the parameter
  itself. X5 found a program that is right when it is compiled
  directly, and that reads a block of the stack which a later call
  has written over when it is compiled through the text; nothing in
  this tree reproduces it. That is a failure of the path through the
  text, and not of the compiler's own run of Simplify.
  The tool does not give Simplify the unit as the reader makes it.
  It gives every parameter of every piece of code that the unit
  binds the mode that says least under the compiler's setting for
  allocation: heap or stack where the compiler allocates on the
  stack, heap where it does not (`with_unknown_param_modes`, by
  `Alloc_mode.For_types.unknown`). In a check (`check_file`) the
  unit so changed is the one that is checked for well-formedness,
  that Simplify is given, and that is converted as the unit before.
  In the round trip of the conversion (`simplify_for_round_trip`) it
  is the unit that Simplify is given; the first conversion of the
  round trip is of the unit as it was read, and the model has no
  modes, so the two convert to the same term up to bound names.
  From the modes of those parameters the tool's Simplify then learns
  no more than the compiler's own does, and may learn less: where
  the compiler allocates on the stack, a parameter that the
  compiler's term says is on the heap has in the tool's the mode
  that says nothing, so the term after that is checked may keep
  what the compiler's own run removes. That is said of these modes
  alone, and not of anything else that Simplify infers. Trusted of
  that step: that the unit which it builds again is the unit read up
  to the variables and the continuations that it binds outside the
  bodies of code, which may be made fresh, and but for those modes;
  symbols and code ids are not made fresh. The free names that the
  compiler keeps with a `Let`, a `Let_cont` or a handler are given
  as not known, as the reader gives them; a piece of code has its
  own body and its own free names, as they were read. It replaces
  this one default of the reader, for the parameters of the code
  that this unit binds.
  The other defaults of the reader stay as they are. One is
  `first_complex_local_param`, which the reader gives as: no
  parameter is local and complex; Simplify reads it for where the
  closure of a partial application is allocated
  (`simplify/simplify_apply_expr.ml`), and with the reader's value
  that closure is on the heap. The modes that the text does write,
  of an allocation and of the result of a call, are read as written;
  and code of another unit that Simplify reads from a `.cmx` file
  has the modes that the compiler wrote there.
  The text writes a code id, a function slot and a value slot by
  its name alone, without the unit that made it (`parser/fexpr.ml`; a
  symbol is written with its unit). The reader takes such a name to
  be the one of that name that it has already made for the unit that
  it is reading, and makes a new one of that unit where there is
  none: it does not fail (`fresh_or_existing_code_id` and its
  fellows in `parser/fexpr_to_flambda_commons.ml`). So a direct call
  of code of another unit, read from a text, loses where its code
  comes from. Where the text binds code of that name, the call is
  read as a call of that code; nothing here rules that out. Where it
  binds none, the call has a new code id with no code, and is
  undefined in the model where the compiler's call is defined.
  The second was observed, of the text of a term after: the compiler
  writes such a call, in the term after of the second unit of a
  pair, as a direct call of a bare name beside the symbol of the
  other unit, which keeps its unit; and a check that reads its term
  after from that text, in a program not in this tree, says "not
  validated" of a pair that the tool validates (X4-4). The fault
  observed is of a right term refused. In an ordinary check
  (`check_file`) the tool reads from a text only the term before,
  and makes the term after in memory; the round trip of the
  conversion also reads back a text of the term after, which the
  tool's own printer of the terms of the model writes, and nothing
  is said of it here. Of the texts that the scripts of
  `tests/semantics/ocaml` make, at the settings that they use: each
  is of a term before Simplify compiled outside classic mode, where
  the conversion from Lambda knows nothing of the code of another
  unit and names none (`from_lambda/closure_conversion_aux.ml`).
  Nothing is said here of a text that is given to the tool by hand,
  or that anything else made: the tool reads it as it is. Of the
  compiler's text of a term after, of a pair not in this tree,
  given by hand as a unit, in which the call is in code that the
  run does not reach, the tool printed a validation: it is of the
  unit read, whose code calls a code id that nothing binds.
  A closure over code of another unit is written with the bare
  name of that code as well, and there the reader does stop: it
  looks for the code of every closure among the code that the text
  binds, and raises `Not_found` where there is none
  (`bind_all_code_ids_and_function_slots` in
  `parser/fexpr_to_flambda.ml`; X2-9, which is of the direct call
  above as well). The compiler writes such a
  closure in a term after, where Simplify has inlined a function of
  another unit that makes one; a term before that is compiled
  outside classic mode has none, so no text that those scripts make
  at those settings has one. A source of `pairs/` may ask for
  classic mode on its line of flags, and none of this tree does. Of
  one that did, not in this tree, the script made the text, with a
  direct call of code of the other unit in it, and the tool, given
  the flags of the source as the rules of `pairs/` give them, ended
  on the compiler's fatal error that classic mode is not supported
  where the middle end is run alone (`flambda2.ml`), with status 2.
  Given by hand the compiler's text of such a term after, of a
  pair not in this tree, the tool ended on that exception, which
  it does not catch, with status 2 and no verdict. In the round
  trip of the conversion of the same unit, read from its term
  before, the text that the tool writes of its own term after has
  the closure, and the tool said in words that it is not read back.
  Two more texts that the compiler prints are not read, and the
  tool gives no verdict of the unit. The printer writes a variable
  as its name, a `/` and its stamp; the reader cuts what it reads
  at every `/` and stops with a fatal error where that gives more
  than two pieces (`chop_stamp` in
  `parser/fexpr_to_flambda_commons.ml`; X2-8). A variable has a `/`
  in its own name where it is an operator with a `/` in its name
  that the unit binds, as a function or as a parameter; where it
  is of an anonymous function and the source was given with a
  directory; and where the reaper made its name from those of
  others, which is of a term after. The scripts compile a copy of
  the source in a directory of its own, under its bare name, so
  the second does not arise through them. Of a unit that binds
  such an operator they make the text, and the tool ended on the
  fatal error, which it does not catch, with status 2; `/` and
  `/.` were run.
  And the printer writes the kinds of unboxed 8-bit and 16-bit
  integers as `int8` and `int16`, which the grammar of the reader
  has not got as the kind of a number
  (`parser/flambda_parser.mly`, `naked_number_kind`; X2-10): of a
  unit with a parameter of such a kind the scripts make the text,
  and the tool printed a syntax error with the file and the line
  and ended with status 1. That is also the status of a check that
  fails: a status that is not 0 is not by itself a verdict. A rule
  of the tests fails on either, and what the tool printed says
  which it is. The three were observed of units not in this tree.
  Each is a stop and not a wrong verdict: nothing is printed of the
  two runs or of a validation. The loss is the reader's: the model
  has the two kinds (`FlambdaKindModel.Kind_naked_int8`,
  `Kind_naked_int16`) and its variables are numbers
  (`FlambdaModel.variable`), and the driver of the compiler's own
  run, which reads no text, validates the unit with the operator
  and the unit with the parameter. All of this was seen with the
  scripts and the tool of 2abcee6155, and was not run again on what
  has come since.
  A fourth text that the reader does not read is of a number
  (C1-11), and units of this tree have it. The printer writes a
  naked integer of 64 bits, or of the width of the machine, with
  all its digits and the suffix `L` or `n` (`parser/print_fexpr.ml`).
  The reader gives the digits of a constant that stands in an
  expression to `int_of_string` before it looks at the suffix
  (`parser/flambda_parser.mly`, `make_const_int`), and that fails
  outside the 63 bits of its own integers, with an exception that
  nothing catches. A number that is static data is read another way
  (`make_boxed_const_int`). Simplify can put such a number bare in
  an argument, as it does where two boxed numbers meet and it
  removes the box. Four units of `tests/semantics/ocaml/library`
  have one so in their term after (`int64_identities.ml`,
  `int64_laws_more.ml`, `nativeint_identities.ml`,
  `nativeint_laws_more.ml`). The terms before of these four have
  the numbers as static data: their texts are read, and each is
  validated, the check making its term after in memory and reading
  no text of it. The compiler's text of the term after of each is
  not read. And the round trip of the conversion, which reads back
  a text of the term after, says "round trip the same" of the unit
  of the file and "not read back" of the unit after Simplify, with
  the exception of the parser named: its rules run on that
  directory and keep the four answers as expected files
  (`tests/semantics/roundtrip`). Until the tool made that line of an
  exception of the parser, it ended on the exception there, with
  status 2, and no rule ran the round trip on that directory: that
  was seen with the tool of the tree that first had those units
  together with all that came in with them. Nothing of what is said
  of the round trip holds of the term after of those four units.
  Two primitive annotations are also lost by the text; the tool
  reconstructs the second from the argument and leaves the first. The
  printer writes both forms
  of `Is_int` as `%is_int`, and the reader gives `variant_only = true`,
  even where the compiler had the generic form
  (`parser/fexpr_prim.ml`; C2-26). A generic test of a closure gives
  0; the form for variants has no value on that closure in the model.
  The printer writes `Opaque_identity` as `%opaque` without its kind,
  and the reader gives the kind of values, even where the compiler had
  a naked number (C2-27). The model's opaque identity has no value on
  an argument of another kind. The conversion from a term in memory
  keeps both the variant flag and the opaque identity's kind; the text
  loses them first.
  A unit read from a text is normalised before Simplify and the
  conversion see it (`with_lost_kinds`, called in `check_file`; the
  round trip of the conversion applies it to the unit of the file and
  to each unit that it reads back): every `Opaque_identity` is given
  the kind of its argument as the reader has it, which is the kind of
  a constant, that of values for a symbol, and for a variable the
  kind that the reader made it with, from the kind written at a
  parameter or from the defining expression of its `Let`. A `Let`
  whose variable the reader made at the kind of values for want of
  that kind binds a new variable of the kind given; the bodies of
  code are gone into; a unit that has no `Opaque_identity` at another
  kind than its argument's is not gone through and is the unit as
  read. The compiler takes the argument of the primitive to have the
  kind that the primitive carries
  (`Flambda_primitive.arg_kind_of_unary_primitive`). So the kind
  given to an `Opaque_identity` is the compiler's kind where the kind
  now obtained for its argument is the kind of that argument in the
  compiler's term and the annotation of the primitive there agrees
  with it. The first need not hold: the compiler's term may bind a
  variable at another kind than that of its defining expression,
  where the reader takes the kind of the expression (C2-24; the
  paragraph on the kind of the variable of a `Let`, in the entry on
  what the conversion of terms leaves out), and an `Opaque_identity`
  above may have had another kind than its argument's. Those two
  conditions suffice to recover the lost kind. Where either fails, no
  recovery of the compiler's kind is claimed: the primitive there,
  the variable bound to it and what is bound below to them may be
  given kinds that the compiler's term had not, even where the kind
  of values that the reader gave was the compiler's. The text does
  not record the lost annotation, so nothing in it shows which is the
  case. The step is not
  a check of the compiler's term. What the compiler prints with
  `-drawflambda` has the kind, and no rule of the tests reads it.
  Trusted of this step, as of the one that sets the modes of
  parameters: that the unit which it builds again is the unit read,
  but for those kinds and for the variables and continuations that
  are made fresh. Where the kind of the argument comes from a nullary
  primitive at one of the two kinds that the reader's table of kinds
  gives for two words each (`parser/fexpr_prim.ml`, `kind`), the unit
  is not checked (`kinds_opaque_misread.fl`, in `tests/semantics/`).
  The units there whose names begin with `kinds_opaque_` show the
  step: before it, a unit that applies the primitive to a naked
  number was undefined in the model as read, the tool's Simplify made
  it `Invalid`, and the tool printed a validation that said nothing.
  The flag of `Is_int` is not restored. Two examples not in this tree
  show that loss: the compiler's own term is defined, but the term
  read before Simplify is undefined in the model. In one of them the
  tool's Simplify makes the unit `Invalid`. In both the tool prints a
  validation and the warning that it says nothing of the unit because
  the term before is undefined. That verdict is of the pair
  obtained through the text, not the compiler's own pair. Where this
  loss makes the source undefined in the model, every target refines
  it there; this does not say that the validator accepts every such
  pair. Neither comparing the two model runs nor proving refinement of
  that pair recovers the meaning lost from the program's term.
  The text loses the arity of the arguments of a direct call and of a
  call of unknown arity as well, and the reader gives such a call one
  value for each argument (C2-5); a call of known arity has its arity
  in the text. A unit that gives a function an unboxed number was
  therefore undefined in the model as read, and the tool's Simplify
  made a direct call of it `Invalid`. The same pass over a unit read
  gives each such call an arity. A direct call is given the kinds of
  the parameters of the code that it names, where the unit binds that
  code and the call has as many arguments as the code has parameters.
  The conversion from Lambda makes a call direct only where the arity
  of its arguments is, parameter for parameter, that of the code
  (`from_lambda/closure_conversion.ml`,
  `close_exact_or_unknown_apply`, which makes any other `Invalid`),
  and with the flags of the tests it knows no code but the unit's
  own. So that is the arity which the compiler's call had, subkinds
  and the grouping of unboxed products aside. Where the unit does not
  bind the code, or the numbers differ, and an argument is not a
  value, the unit is not checked (`kinds_call_unbound.fl`). A call of
  unknown arity is given one kind for each argument, the kind that
  the term read has for it. That is the arity which the compiler's
  call had, subkinds aside, on three conditions. For each argument,
  the kind obtained
  for it is the kind of that argument in the compiler's term, which
  can fail as it can for the argument of an `Opaque_identity`,
  above. The arity of the compiler's call gave each argument the
  kind of that argument: the compiler does not see to it, and
  Simplify makes `Invalid` a call of which it is not so. And that
  arity had one singleton component for each argument, with no
  component that is an unboxed product.
  The tool checks none of the three. Together they suffice, and they
  are not necessary; where one fails no recovery of the compiler's
  arity is claimed. Where the kind of an argument comes from a word
  that the reader may misread, the unit is not checked
  (`kinds_call_misread.fl`).
  The arity so given is flat, one component for each argument that
  the text writes, as the reader's was. The compiler's arity groups
  the kinds of an argument that is an unboxed product, which may be
  the product of nothing, and then the text does not write the
  argument at all; and an application counts the groups against the
  parameters of the code, to tell a full application from a partial
  application or an over-application. The text has no grouping, at a
  call or in the parameters that it writes for code, so the third
  condition is one that the tool cannot test. A direct call is given
  the parameters of its code, which are flattened alike. Of a call
  of unknown arity, as far as the grouping goes and that alone (the
  kinds are as said above, and the text loses the other things that
  this entry lists): the call runs in the term read as in the
  compiler's if neither the call nor the code that it reaches has a
  group. If one of them has, it may not. A partial application that
  stops before a parameter of no kind is read as a full application,
  and so is an over-application whose last argument has no kind
  (C2-30): the tool as it was printed, of two units not in this
  tree, an integer where the program has a closure and a closure
  where the program has an integer, with a bare "validated". The
  loss is the text's: a term in memory has the component, the
  conversion makes of it an unboxed product of no component, which
  is one component of the arity of the model, and the model's
  application counts it; the driver of the compiler's own run, of
  the same two units, has the closure and the integer that the
  programs have. The loss shows the other way too. Code of another
  unit comes from its `.cmx` with the component, and a call that
  gives it an argument of no kind is written without that argument:
  a full application can then be run as a partial application. That
  was observed of a unit not in this tree, which the compiler
  compiles rightly and which gives a function of another unit its
  last argument, of no kind: the tool printed "DIFFERENT" and "not
  validated", with status 1, where the driver of the compiler's own
  run validates it: a false alarm.
  A measure was made outside this tree, of 112 units from sources
  that apply functions with unboxed products among their parameters
  or their arguments. It is a report of those units and proves
  nothing of others. What it compares is the module block that a run
  is printed to return with the one that the program has, in which a
  function shows only as a function and not by what it does; and a
  unit is counted by the worse of what was printed of it at the
  default settings and at `-O3`. Through the text, the tool printed
  of 12 a run that is not the program's with no line that says so, a
  bare "validated" of 11, each with a component of no kind at the
  edge of an application; of the units in which every product has a
  field of some kind, it printed no such run. Through the compiler's
  own two terms it printed none (right on 99, undefined on 7 for
  want of code in a `.cmx`, not run on 6). The refusal of the
  scripts, below, refuses 69 of the 112, those 12 and the unit of
  the false "DIFFERENT" among them. And
  where the call and the code group the same kinds differently,
  which is ill-kinded in the compiler's term, the term read may be
  defined where the compiler's is not. All of that was so before
  this step of a call whose arguments are all values, which the
  reader has always read flat; what the step would add is the same
  of a call with an argument that is not a value, which was
  undefined as read.
  The scripts that make the texts of `tests/semantics/ocaml` refuse
  the units in which that can happen, where the tool is to be run on
  the unit (`from_ocaml.sh` there and in `pairs/`; `script/` there
  has a unit for each refusal and, for most, the same unit without
  the reason for it). From what `-drawflambda` prints of the term, a
  unit that has a call of unknown arity is refused if such a call
  has a group among its arguments, or if code that the unit binds
  has one among its parameters. So in a unit that they let through,
  no call of unknown arity has a group, and the code of the unit,
  which is all that such a call can reach of what is read from the
  text, has none: there the grouping is the compiler's. (Code of
  another unit is not read from a text: it is taken from the `.cmx`
  of that unit, with the arity that the compiler gave it.) No unit
  of that directory was refused when the rules came. The refusal is
  in the scripts and not in the tool: the tool cannot tell that a
  text is of such a unit, and reads any text that it is given. So
  the refusal does not protect a text that is given to the tool by
  hand, or that anything but those scripts made: of such a text what
  is said above of a group stands.
  Trusted of this step, beside what is trusted of the pass: that the
  conversion from Lambda keeps to that rule for a direct call, which
  is read from its source and tested by nothing here; and of the
  refusals of the scripts, that the compiler prints a call and code
  as they expect: the arity of the arguments of a call, then that of
  its result, then its kind; the arity of the parameters of code,
  then their modes; a product with `#(`, or `void` for the product
  of nothing. A unit in whose print those fields are not found is
  refused.
  What is still not claimed: the model erases the regions of
  allocations and their lifetimes (the entry "What the conversion of
  terms leaves out"). Equal outcomes in the model do not establish
  that a block returned is still alive, and no validation, whether
  plain or counted apart, is a check of that. Other things that the
  text loses, the kind of a value that may be null among them
  (T4-13), are described in
  `middle_end/flambda2/tests/semantics/ocaml/README.md`, "What a
  validation does not say"; the classification of the units in that
  file says which are counted apart, and for which of these reasons.
- **The compiler's own run, in a second program.**
  `middle_end/flambda2/tests/tools/ownrun/flsemantics_ownrun.ml` is
  the compiler's driver with two hooks registered
  (`driver/compiler_hooks.mli`), and gives the two terms of the
  compiler's own run, as they are in memory, to the check of the
  tool (`Flsemantics.check`). No rule of the tree runs it on a unit
  of the examples; one control runs it
  (`middle_end/flambda2/tests/semantics/own_run/`). A verdict that
  it prints does not rest on the entry above: there is no text, no
  reader and no second run of Simplify. In its place it rests on
  the two hooks: that `Raw_flambda2` is given the term that the
  compiler gives to Simplify and `Flambda2` the term that the
  compiler's Simplify returned, and that these are the terms that
  the compiler goes on with (`flambda_to_flambda0` in
  `middle_end/flambda2/flambda2.ml`). The entries below are trusted
  of it as they are of the tool, with one difference: the `.cmx`
  files of other units are looked for on the compiler's load path
  (`Compmisc.init_path`), not only in the directories given with
  `-I`. Such a verdict says nothing of what the compiler then makes
  of the term after: the reaper where it is enabled, the
  translation to Cmm, the back end and the link. The header of the
  file and the section "The compiler's own run" of
  `middle_end/flambda2/tests/semantics/ocaml/README.md` say the
  rest.
- **The conversion of terms**, in
  `middle_end/flambda2/tests/tools/flsemantics.ml` (`convert_expr` and
  the functions around it, about 230 lines): from the compiler's Flambda
  terms to the terms of `FlambdaModel.v`. It is trusted to give a term
  with the meaning of the one it is given, and to refuse (`Unsupported`)
  what the model has not got, never to approximate it. To be read case by
  case against `terms/`; that reading is not done. A second reading in
  the other direction is: `flmodel_to_fexpr.ml`, beside the tool, prints
  a term of the model back as fexpr, and `flsemantics.exe -round-trip`
  (`flroundtrip.ml`) converts a unit, prints it, reads the text again
  with the compiler's parser, converts that, and compares the two terms
  with the model's comparison, for the unit of each example and for the
  unit that Simplify makes of it
  (`middle_end/flambda2/tests/semantics/roundtrip/`). This finds a case
  that the conversion gets wrong in a way that is not the same both
  times (two arguments exchanged, a number changed). It does not find
  what the conversion drops and the printer does not put back, nor a
  mistake made the same way in both directions, and it does not look at
  what the conversion does to the two units of a check together. The
  printer and the round trip are OCaml and add nothing to what is
  trusted: they change no theorem and no answer of the validation. A
  mistake in them can make this check miss a mistake of the conversion,
  report a difference that is not one, or fail. Of a set of closures the
  conversion is trusted
  for three things that no statement covers. The order of the
  declarations: the functions of a set are given in the order of
  `Function_declarations.funs_in_order`, and the variables of a `Let`
  are paired with them by position. The symbols of a static set: each
  closure symbol of the compiler's pattern is given with the function
  slot it is bound at; the tool refuses a pattern whose function slots
  are not those of the set, in order, and `well_formed_unit` checks the
  result again. And the presentation of a set: the model has two, and
  every set is converted in the one that names its function slots,
  whatever the unit has. What the compiler's two patterns expand to is not
  in the OCaml: it is `let_closures` and `bind_closures`, in Rocq, with
  statements.
- **The kinds and arities that the conversion gives**, in the same file
  (`convert_arity`, `convert_params_kinds`, `convert_bound_var`,
  `find_value_slot`, and the arities in `convert_code` and
  `convert_apply`, about 70 lines). A `Let` gets the kind of its
  variable, a handler the kinds of its parameters, code and a call their
  two arities with their unboxed products, and the value slots met their
  kinds. The variables that a `Let` binds to the closures of a set of
  several functions get the kind `Value` from `let_closures`; the tool
  refuses the binding if one of the compiler's variables has another
  kind. The machine makes a call undefined that does not agree with its
  code (`FlambdaSemanticsSpec.call_function`), so a wrong arity makes a
  run undefined that should not be, or the reverse. The check of kinds
  (`FlambdaWellKindedSpec.v`) does not catch that: code that never
  returns passes it with any result arity. Trusted: that the arities of
  calls and of code, the kinds of parameters and the kinds of value
  slots are the compiler's. Cross-checked as the tool runs, and fatal to
  it: an arity as converted, unarized by the model, has the kinds of the
  compiler's `Flambda_arity.unarize`; the parameter arity of code,
  unarized, has the kinds of the parameters of its body (nothing in the
  compiler checks that where code is made). Not cross-checked: how an
  arity is grouped into unboxed products, which is rebuilt through
  `Flambda_arity.fold_per_parameter`.
- **Tupled functions** are in the model as code that is marked
  (`FlambdaModel.cm_is_tupled`), and the converter gives code the mark
  of `Code0.is_tupled`. Every call of the model gives tupled code its
  first argument as a tuple, which the machine takes apart; any
  arguments after it are given as they are, and make the call an
  over-application (`FlambdaSemanticsSpec.arguments_for_code`): that
  is the compiler's call of unknown arity. The compiler's direct call
  and its call of
  known arity give code its arguments as the code takes them, and the
  model has not got that of tupled code. So the converter refuses
  (`Unsupported`) a direct call of tupled code that the unit binds, and
  a call of known arity in a unit that binds tupled code; that is
  OCaml, in the same file, and is trusted. Trusted too: that a tuple
  given to tupled code that has parameters has the tag 0 and as many
  fields as the code has parameters. The machine makes any other call
  of such code undefined; the run time reads the first fields of
  whatever it is given (the review comment at
  `FlambdaSemanticsSpec.tuple_fields`). Tupled code with no parameter
  reads no field: its first argument is evaluated, and its kind is
  tested as that of any argument is, but neither its tag nor its
  fields are looked at
  (`middle_end/flambda2/tests/semantics/tupled_apply_zero.fl`).
- **Partial applications and over-applications** are in the model
  (`FlambdaSemanticsSpec.shape_of_closure_call`), and the converter
  gives a call its arguments and its two arities as they are, whatever
  its kind. Two things are trusted. A partial application is a value of
  the model that the compiler has not got as such (it makes a closure
  of a wrapper): at the end of a run the tool observes it as it
  observes a closure (`observe`, `Function`), without looking into it,
  so two runs that end with functions that differ are not told apart,
  as for closures. And the model takes every call for a generic
  application: a direct call, or an indirect one of known arity, with
  fewer or more arguments than its code has parameters is a partial
  application or an over-application in the model, where the backend
  stops on the first and compiles the second as a full call
  (discrepancy C3-9; the CR at `shape_of_closure_call`). The tool does
  not refuse such a call. Of a unit with one `validate_unit` may answer
  yes: its statement then holds of the model, and says nothing of what
  the backend makes of the unit there.
- **What the conversion of terms leaves out**, in the same file. These
  are the exceptions to "never to approximate", each with a `CR
  mshinwell` where it is done. *Depths*: the depth variables of
  recursion and the coercions that mention them are dropped
  (`depth_variables`), so the model sees nothing of what Simplify does
  with them. *Regions* (decision D37): the bindings of `Begin_region`
  are dropped, `End_region` returns the unit value and does nothing,
  and what is allocated in a region is converted as if it were
  allocated on the heap (`region_variables`). This is trusted of both
  units of a check, the one after Simplify as much as the one before:
  it is right for a unit in which nothing allocated in a region is used
  after the region has ended. A validation therefore does not
  establish that Simplify keeps the uses of a local allocation within
  the life of its region: a term that uses one later has, in the model,
  the meaning it would have with the allocation on the heap
  (discrepancy V2-3). The regions around a `try` (`Begin_try_region`,
  `End_try_region`) are dropped in the same way. There a region also
  ends when a raise leaves it open and the handler ends it, and the
  same is trusted of both units on that way out: nothing allocated in
  such a region is used in the handler after that, or later. One thing
  is kept of the region of a handler: the `Let` of its `End_try_region`
  (the one that is not a ghost) binds `Opaque_identity` of the unit
  value, where that of an `End_region` binds the unit value. The value
  bound is the same, so the unit means in the model what it meant;
  what changes is that the normalisations of the validator do not
  fold that `Let`, nor remove it for being unused
  (`FlambdaNormaliseSpec.prim_may_fold`), in either term, as Simplify
  does not delete the primitive for being unused (its effects are
  arbitrary, and `Flambda_primitive.is_end_region`, by which the end
  of an unused region is deleted, is of `End_region` alone). In the
  model as in the compiler it still goes with a handler or with code
  that is removed whole. It is a mark that something is done there,
  not a model of what is released. Without it the handler of a `try`
  that Simplify has demoted was, in the term before, a continuation
  that only applies another, and the validator forwarded to it where
  the compiler, whose handler ends the region first, has no shortcut
  (discrepancy C5-10).
  A region variable used as a value is refused. *Tupled functions*: a direct
  call of tupled code that the unit binds is refused, and so is a call
  of known arity in a unit that binds tupled code (the entry above);
  it is what keeps a call that gives code the components from being
  run as the model's call, which gives it the tuple. *Probes*: a call that
  is a probe is refused (`convert_apply`). The program runs the handler
  of a probe only while the probe is enabled, and the model has no
  notion of a probe being enabled. The refusal goes by the mark that a
  term in memory has on such a call. The text of a term has no such
  mark and its reader gives none, so the refusal does not detect a
  probe whose mark the text has lost: a unit that has a probe of its
  own is read, from its text, as a unit with a plain call, and is
  checked as one. The scripts that make the texts of
  `tests/semantics/ocaml` refuse a unit whose term has a call that is
  a probe, by the mark that `-drawflambda` prints of such a call,
  where the tool is to be run on the unit (`from_ocaml.sh` there and
  in `pairs/`; `script/` there has a unit that is refused and the
  same unit without the probe); the tool reads any text that it is
  given. The mark is met in a body taken from the `.cmx` of
  another unit, which is then left out of the prefix, as code whose
  body the model has not got (the entry on what those files say), the
  unit being still checked; and in the term that Simplify makes of a
  unit, where it inlines such a body, the unit being then not checked.
  *Phantom bindings*: a binding whose variables are phantom is refused
  (`convert_let`, before any of its cases). The program runs nothing
  for such a binding (`To_cmm_expr.let_expr` compiles its body alone),
  and the model runs every `Let`: converted, the binding could be
  undefined in the model where the program is defined. On a run where
  the term before is undefined, any term after refines it in the
  model, so the theorem of a validation would say nothing of the term
  after there; this is not to say that the validator accepts every
  such pair. With `-g` and phantom lets on
  (`-flambda2-expert-phantom-lets`, which
  `-gdwarf-may-alter-codegen-experimental` also sets), Simplify may
  keep an unused binding as phantom instead of deleting it; the term
  that the compiler gives to Simplify has none. The refusal goes by
  the mode that a term in memory has on its bound variables. The text
  of a term does not say the mode and its reader gives the normal
  one, so the refusal cannot detect a phantom binding whose mode a
  text has lost, as for probes. The texts of the tests are of terms
  before Simplify as the compiler prints them, which have none; the
  tool reads any text that it is given. The mode is met in a body
  taken from the `.cmx` of a unit compiled with those flags, which is
  then left out of the prefix, the unit being still checked; and in
  the term that Simplify makes of a unit when the tool is itself
  given them, the unit being then not checked (discrepancy C2-19). A
  binding whose variables are for types only is refused in the same
  place; the backend stops on one.
  *The symbol of the unit*: `convert_unit` does not read
  `Flambda_unit.module_symbol`. The tool observes the values given to
  the unit's return continuation, using the static environment at exit.
  Other units read the data of the unit's symbol; the backend's handler
  for the return of the unit ignores its argument and returns the unit
  value (`to_cmm/to_cmm.ml`). Returning the unit's symbol makes the tool
  start its observation at that symbol. The tool checks a unit only
  where, in the term before and in the term after as they are written,
  every use of the return continuation of the unit gives it the symbol
  of the unit and nothing else, and no call returns to it
  (`return_not_of_own_symbol`, asked in `check` of each of the two
  terms); a unit of which either term fails is not checked, and the
  line printed says which term and why. The test is of the form of the
  term: a unit that returns its own symbol through a variable or from
  a call is refused all the same. A hand-written text that returns
  another symbol was validated before this test and is refused by it
  (`returns_other_symbol.fl`, in `tests/semantics/`). The check that a
  term survives a round trip through its text does not ask it.
  Returning the right symbol does not by itself establish agreement
  with what a linked program observes.
  *The function slot of a closure and of a value-slot projection*:
  every unit is converted with named function slots. A closure is at
  the function slot that the Flambda term given to the conversion
  has for it, and
  `convert_prim` keeps `project_from` of `Project_value_slot`: in the
  model a projection that names another function slot than that of
  the closure has no value (decision D3). Simplify makes such a
  projection `Invalid`, and the backend reads the function slot to
  compute the load's offset (`to_cmm/to_cmm_primitive.ml`).
  It was not so before this change, and what the tool reported until
  then is not made good by it. Where every set of closures in what a
  check converted (the unit before Simplify, the unit after and the
  prefix) had one function, and nothing there projected a function
  slot, the conversion left the function slots out: it dropped
  `project_from`, and did not read the function slot at which a set
  had its one function. The model could then not test that a closure
  had the function slot named, and two terms that differed in such a
  slot only were one term of the model. Both consequences were
  observed. A hand-written text, not in this tree, that names the
  wrong slot returned a value before Simplify in the model and was
  undefined after it: a correct Simplify was faulted. And a term
  after that put a closure at another function slot was validated,
  as follows.
  In a set of edits of terms after, made with the tool of bddccfade8
  at the default flags and not in this tree, 302 edits put a closure
  at another function slot. The 276 in units converted with unnamed
  function slots were all validated. Where a native route was
  available, the compilation of the edited text was attempted, by a
  route that removes the `Let`s that only rename a variable and runs
  Simplify on the text again, so that what follows is not a test of
  the backend on the edited term as it stands: for 155 the native
  output and the assembly are those of the unedited term, for 61 the
  compilation stops and makes no object, one gives the same native
  output with different assembly; 59 are in units that have no native
  route. As they were measured then, none of the 302 was shown to be
  another program. The other 26 are in units converted with named
  function slots (a set of two functions, or a function slot
  projected): 14 were not validated and 12 not checked (X4-3).
  One edit of this sort, made since, is validated and compiles to a
  program that stops where the unit prints 3. It was written by hand
  in the text of a term after and is not in this tree; it is not a
  term that the compiler made. The unit has two functions, each
  alone in its set: one that is not inlined, and a closure given to
  it that reads a variable of the unit. The term after has that
  closure at its own function slot, from which its code projects the
  value slot; the edit, of one word, puts it at the function slot of
  the other function. All of what follows is at the default
  optimisation settings. The check of the tool of ea56783099
  (`Flsemantics.check`), given the term before and the edited text
  as the term after, with no Simplify run on that text, says
  "validated", and both runs of the model return the result of the
  unit, 3. A native route that takes `newer_version_of` out of a
  text, runs the compiler's Simplify on it again and links the
  object with the `.cmx` of the unit compiled from its source with
  `-opaque` gives this: the unit from its source prints 3; the text
  of the term after, unedited, prints 3; the edited text is compiled
  and linked, unlike the 61 above, and the program stops as it runs,
  with "Invalid code" and status 134. Simplify, run again on the
  edited text, makes the projection of the value slot invalid. As
  with the 302, that is not a test of the backend on the edited term
  as it stands. So a "validated" of a check converted with unnamed
  function slots did not establish that a closure of the term after
  is at the function slot from which its code projects.
  With the conversion as it is, the unedited text and the edited one
  are two terms of the model. `closure_at_own_slot.fl` and
  `closure_at_other_slot.fl`, in `tests/semantics/`, are the text of
  that term after and the same text with that one word changed (the
  names of the unit and of the anonymous function are others). Given
  to the tool as units, the first returns 3 and the second is
  undefined in the model before Simplify; with the conversion as it
  was, both returned 3 and were validated.
  The check of the term before of that unit against the edited text
  as the term after, with no Simplify run on that text, was made
  again with the conversion as it is, by a program not in this tree
  that calls `Flsemantics.check`: the run before returns 3, the run
  after is undefined, and the tool says "DIFFERENT" and "not
  validated", with status 1; against the unedited text it says
  "validated".
  The tool reads one text and makes the term after itself, so that
  pair is not among the tests of this tree.
  What the change moved in `tests/semantics`: no verdict. Of the
  units from sources, seven are validated by another route (by the
  lifting of closures where it was by the removal of unused static
  bindings, or at the second asking) and one at the first asking
  where it was at the second; the eight second units of `pairs/`
  print the function slot of each closure of their prefix; two texts
  by hand change their route.
  No claim is made that the validator refuses every pair in which a
  function slot is wrong. What is claimed is that the conversion
  keeps the named function slots of the Flambda term that it is
  given, up to the numbering that the two terms of a check share.
  Whether that term is the compiler's is another matter: of a term
  read from a text it is subject to what this entry says that the
  text loses. The change removes the loss that the unnamed
  presentation made, and no other.
  *What Simplify returns beside the term*: the tool converts
  `Flambda2.flambda_result.flambda`, not the other fields of the result.
  In particular it does not compare the slot offsets with the term. The
  offsets used by the backend are computed using the free names that
  Simplify reports (`Flambda2.finalize_offsets`). A projection of a slot
  whose offset says that it is dead becomes invalid code in the backend
  (`to_cmm/to_cmm_primitive.ml`); the model has no such offset table. A
  verdict does not establish that the reported names and offsets agree
  with the term, so that kind of error in the term after's compilation
  is not checked. This is a limit read from the code, not an exhibited
  wrong result of Simplify.
  *The kind of the variable of a `Let`*: `convert_bound_var` uses
  `Variable.kind`, and the machine tests the value bound against that
  kind. The conversion does not compare it with the kind of the defining
  expression. The reader of a text computes it from that expression
  (`parser/fexpr_to_flambda.ml`); the backend translates the expression,
  rather than testing the bound variable's kind. If those kinds disagree
  in a term in memory whose defining expression the program can execute,
  the model's run can be undefined where the program's is defined. No
  such compiler-produced term is exhibited here. The conversion and the
  validation do not establish that the two kinds agree.
  *Facts stored with the term and its names*: the conversion does not
  read cached occurrence counts of bound variables and continuations,
  the mark that a continuation is applied with a trap action, the marks
  saying whether code uses its closure, or the mark that a value slot is
  always an immediate. The backend uses these for the placing of
  expressions and handlers, the closure argument of a call and the
  parameters of code, and the representation and scanning of closure
  fields (`to_cmm/to_cmm_expr.ml`, `to_cmm/to_cmm_set_of_closures.ml`).
  A validation does not check that the facts agree with the term and its
  values. No term with such an incorrect fact is exhibited here; their
  being copied or computed by the compiler is not a proof of that
  agreement supplied by the validation.
- **Calls that never return**, in the same file (`convert_apply`). A
  call whose result continuation is `Never_returns`, whatever its call
  kind among those that the conversion takes (a method call, a C call
  and an effect are refused before that), is converted to a call that
  returns to a continuation bound around it and used nowhere else,
  whose handler is `Invalid` and whose parameters are of the kinds of
  the return arity of the call, which the call keeps; the continuation
  has a number that no continuation of the two units of the check has,
  and its parameters have numbers that no variable of them has (a
  number of one sort may be that of a name of another). The conversion
  interprets
  `Never_returns` as making a normal return undefined, and this
  interpretation is trusted. The backend inserts no trap for it: it
  emits the call with nothing after it, as it does a call in tail
  position (`to_cmm/to_cmm_expr.ml`, `apply_expr`). Nothing is assumed
  of the analysis by which Simplify finds that a callee never returns.
  If a run of the converted term after returns from such a call, it
  becomes undefined. Where the term before has a defined outcome,
  completed executions then expose the difference, and a sound
  validation cannot accept that pair. A validation is not a proof that
  the callee does not return in native code. The term of the model has
  a continuation that the compiler's term has not, and the round trip
  of the conversion prints it.
- **The conversion of primitives**, in the same file (the section
  "Primitives"): one function for each arity of primitive that the model
  has (`convert_unary_primitive`, `convert_binary_primitive` and, once
  the model has `Make_block`, `convert_variadic_primitive`), with the
  conversions of what a primitive carries (`convert_kind`,
  `convert_standard_int`, the signedness, the operations and the
  comparisons; later `convert_boxable_number` and `convert_or_unknown`).
  It is about 95 lines with the primitives of the first stage and 145
  with those of the third. It says which primitive of
  `FlambdaPrimitiveModel.v` stands for one of the compiler's. It is a
  renaming of constructors, with the fields the model keeps. Trusted:
  that it sends a primitive to the one of the same name, and refuses a
  primitive, or a value of a field, that the model has not got. Of the
  primitives that allocate or load, that is: a block that is not
  immutable or not of values; and, for `Make_block`, a kind of a field
  that says more than `any_value`, as the model has no subkinds and
  Simplify uses them. A `Make_array` is converted where it is marked
  `Immutable` and is of one of the three array kinds of values, to
  the model's `Make_immutable_array` at that kind, which is not the
  primitive of the same name: the model's carries no mutability and
  makes a value with no identity. One marked `Mutable` is refused.
  An allocation in a region is not refused: it is
  converted as one on the heap, which is not a renaming of constructors
  and is one of the exceptions of the entry above (*Regions*). What a
  primitive computes is not here: it is
  `FlambdaPrimitiveSpec.eval_prim`, to be read against
  `terms/flambda_primitive.mli` and `simplify/simplify_*_primitive.ml`.
- **The conversion of numeric constants**, in the same file (the section
  "Numbers" and `convert_const`, about 75 lines; with the boxed numbers
  also `convert_float32`, `convert_float` and the boxed numbers of
  `convert_static_const`, about 105 lines in all): from the compiler's
  integers of each width, and then the bit patterns of its floats, to the
  integer that a constant of the model carries. Trusted: that each number
  is converted to itself, at every width and at its bounds. (It was not
  so: until this conversion went through `int64`, `z_of_int` converted
  the least tagged or naked immediate to -1, and what the tool reported
  before then is not made good by the repair.) Constants of the vector
  and mask kinds are refused, so the tool does not exercise what the
  model says of boxing and unboxing at those kinds. A poison constant
  is converted to the poison constant of the model, of the kind that it
  carries; the string that says where the compiler made it is dropped.
  The poison of the model is the abstract one of decision D26: a
  constant with a kind and no bits. The backend gives the compiler's
  poison a number made from that string (`String.hash` of it, in
  `const` and `const_static` of `to_cmm/to_cmm_shared.ml`); the
  conversion follows the decision and not those bits.
- **The numbering of names**, in the same file (`Numbering`, about 30
  lines): each variable, continuation, symbol, code id, value slot and
  function slot of the compiler gets a number, the same throughout a
  unit. Trusted: that the compiler's names are fresh, so that one name
  stands for one thing. A function slot of the compiler may get the
  number 1, which is also the model's `unnamed_function_slot`, the
  function slot that the model's other presentation of a set gives
  its one function. No conversion makes a term in that presentation,
  so in a converted unit that number is the function slot that was
  given it and nothing else.
- **The identification of names across the two terms**, in the same
  file (`convert_both` with `code_id_used`, about 70 lines). The two
  units are converted with one numbering, the return and exception
  continuations of the unit after standing for those of the unit before.
  And Simplify gives a new code id to each new version of code: the tool
  gives the new id the number of the old one where the old one is bound
  as deleted and not otherwise used, and then leaves out the
  declaration that the old one is deleted, so that the term of the
  model does not declare deleted a code id that it binds. Trusted: that
  this identifies only what is the same continuation and the same
  function. The second part
  goes when the comparison relates code ids itself (workstream S4; there
  are notes in `flsemantics.ml` and `FlambdaCompareSpec.v`).
- **Symbols that a unit does not define**, in the same file (`imports`,
  about 20 lines, and `check`, which runs the two units from them). The
  conversion lists the external symbols that the two units of a check
  mention: the symbols of another compilation unit, the predefined
  exceptions among them, which are of a unit of their own. A symbol of
  the unit's own compilation unit that the unit does not define is not
  one of them, and stays bound to nothing. Both units are run from the
  static environment that binds each listed symbol to something
  imported (`FlambdaImportsSpec.imports_statics`), an object whose
  contents the model does not know. A symbol that the units mention
  is given the class `Import_identity` if it is a predefined exception
  (`Symbol.is_predefined_exception`) and `Import_data` otherwise. A
  symbol that the prefix of the next entry makes for a field of which
  a `.cmx` file says nothing is none that the units mention, and is
  given the class `Import_unknown`. Three
  things are trusted. The list is the same for the two units: a symbol
  that one of them mentions is declared for both. A predefined
  exception is an object made once, by the startup code, that nothing
  the unit allocates and no other symbol is: the model answers false to
  `Phys_equal` of it against an allocated value of the unit and against
  an import that is data, and answers by the symbols for two of them;
  the compiler does not prove the first of these (the CR at
  `eval_phys_equal`). And the model does not look into an import. It
  can be passed, returned, stored (`Make_block` takes one as a field)
  and given to `Opaque_identity`, which returns it. `Phys_equal`
  answers false of an import of the first two classes against a tagged
  immediate, compares an identity as just said, and has no value of an
  import that is data against an immutable allocated value, another
  import that is data or itself. Of an import of the class
  `Import_unknown` it has no value against anything, itself included,
  where the program has an answer: the model says less than the
  program does (decision D41). That is
  `FlambdaPrimitiveSpec.eval_phys_equal`, which the machine that the
  tool runs uses (`FlambdaSemanticsSpec.step`); it has no value on a
  location of the heap, against an import as against anything.
  `Phys_equal` of the machine with a heap (`FlambdaHeapSpec.exec_prim`),
  which the tool does not run, answers false of an import of the
  first two classes against a location bound by that heap, and has no
  value of one of the third. No other unary or
  binary primitive has a value on an import. It is not a closure, so a
  call of it is undefined. A run that executes a load from an import,
  or a call of one, is therefore undefined in the model where the
  compiler gives it a meaning. Merely containing such a load or call
  does not make the run undefined.
- **What the `.cmx` files of other units say**, in the same file
  (`make_prefix`, `units_read`, `prefix_fault`, `with_prefix`, and
  `check`, which puts the prefix in front of the two units). This
  entry awaits the human's confirmation (decision 1 of
  `middle_end/flambda2/rocq/requests/C5-imports-with-contents.md`).
  Until it is given, a verdict that rests on a prefix is counted
  apart, and is not a verdict of the kind that the lines "validated"
  and "validated with its imports" give.
  Where the tool is given directories of `.cmx` files and its run of
  Simplify read the file of another unit, the two units of a check
  are given one prefix of static bindings, the same in front of
  both, which binds symbols and code ids of other units to what
  those files say of them. The two terms that are run and validated
  are the prefix followed by each unit, from the imports that are
  left: a symbol that the prefix binds is not an import. The
  theorems are of those two terms. Where the prefix evaluates from
  the static environment of the imports, which the tool tests, a
  refinement of the two terms is a refinement of the two units, each
  run from the static environment that the prefix makes
  (`FlambdaImportsSpec.prefix_refines_statement`). Two things are
  trusted, and they are apart.
  The first is of the linked program. A load, a projection or a call
  that a run makes through the prefix has the answer that the prefix
  gives it, and no theorem says that the unit which the program is
  linked with gives the same. That rests on each `.cmx` file being
  right about its unit, which the compiler itself assumes when it
  reads one, and on the program being linked with the unit that the
  file was made from. The tool names the units from whose files the
  prefix is made, and prints the digest of each file where
  `FLSEMANTICS_PRINT_DIGESTS` is set; nothing compares a digest with
  a linked program. What the prefix binds is what the symbols of the
  other unit are once that unit is initialised. Its initialisation is
  not run: what it does before the unit of the check starts, an
  exception that it raises included, is in neither term.
  The second is of the conversion: that `make_prefix` says of a name
  nothing that the file of its unit does not say. What a file says is
  read through the compiler: its loader of `.cmx` files
  (`Flambda_cmx`), and its reduction of the type of a symbol to a
  block, a closure, another symbol, a constant or nothing
  (`Flambda_cmx.load_symbol_approx`, which classic mode uses), with
  the code for which the file has a body. The loader is asked only
  about the units whose files the tool's Simplify read
  (`units_read`): making the prefix opens no other file, and of
  those files it may read the body of code that Simplify did not
  look at. Nothing that Simplify concluded is used. The loader, the
  reduction and the conversion of what they give (the entry "The
  conversion of terms" for a body) are trusted. The prefix says less
  than the files, and what it leaves out is trusted to make a run
  undefined or a unit refused, and never to give a run a defined
  answer that the files do not give:
  - a closure is bound alone, at its function slot of a set that has
    this one function and no value slot, so a projection of a value
    slot of it, or to another function of its set, has no value;
  - code that is not asked for, code that has no body in the file,
    tupled code, code with a call of known arity where one of the
    two units has tupled code, and code whose body the model has not
    got, are bound to nothing, so a call of them is undefined. A
    body with a call that is a probe is one that the model has not
    got: the conversion refuses such a call (the entry "What the
    conversion of terms leaves out"). What is asked for is in the
    comment above `make_prefix`: the code of a closure that is bound
    only as a field of a block is not;
  - a symbol is bound to a block only where the file says that it is
    an immutable block of values each of whose fields is a constant,
    a symbol, or a field of which the file says nothing, as of a
    value that the other unit computes when it is initialised. Such
    a field is unknown and not absent. It is given a symbol of its
    own, which no name of the compiler has, one for each such field
    of each block, made when the block is bound, and that symbol is
    declared an import of which nothing is known (`Import_unknown`).
    The model gives such an import no value in a comparison, a test,
    a load or a call: a run may pass it on, store it and return it,
    and is undefined if it does more. So nothing is made up for the
    field, and the one answer that a run gets which the file does
    not give is a value that says nothing.
    One block with such a field is still not bound: a block one of
    whose fields of that sort is loaded, by either unit or by a body
    that the prefix binds, from something that the making of the
    prefix sees to stand for the block; the prefix is made again
    without it until no such block is left, and one prefix goes in
    front of both units. That test decides only which blocks are
    bound, and is trusted for nothing: binding fewer blocks is the
    prefix saying less, and a load that the test misses leaves a
    block bound with what the file says of it and no more. Which
    blocks are bound can change whether a unit is validated, one
    way or the other; a load that is missed adds no fact to what
    the file says, and nothing is claimed of the linked program on
    the strength of the test. It is there because the term
    after is not read with what the unit binds its symbols to inside
    code: the normalisation of the term before reads the load as the
    symbol of the field, the term after keeps the load, and the two
    are then not the same.
    A block with a field that is an approximation given in place of
    a symbol, or a constant that the model has not got, is not
    bound. Neither is a block bound that has a field which is a
    symbol that its own file says is another symbol or a constant:
    Simplify puts that symbol or constant where the load was, and a
    symbol of which nothing is known in the field would make the two
    runs differ where Simplify is right.
    `pairs/hidden_known_read.ml` is a unit that the rule turns,
    `pairs/hidden_read.ml` one whose block stays unbound, and
    `pairs/partly_tested.ml` one in which a function inspects such a
    field through its argument, which the test does not see as a
    load of the block: the block is bound and the run has no value.
    Measured outside this tree, on units that use another unit, at
    the default settings and at `-O3` (190 runs): seven runs come to
    "validated with its imports, given the .cmx", two of them from a
    run before that was undefined; none goes to "not validated" or
    to "DIFFERENT"; and no expected file of this tree moves. Every
    field that is given a symbol there is one of which the file says
    nothing; the measure has no field of the other sorts, so nothing
    is known from it of how often a block stays unbound for them;
  - a symbol of which the file says nothing stays an import;
  - a symbol that is asked for (one that either unit names, that a
    body which the prefix binds names, or that is at a field which
    a load reads) and that its file says is another symbol, or a
    constant, is refused, and the unit is not checked.

  The round trip of the conversion does not reach the prefix: the
  text of terms cannot declare what a prefix binds. What does check
  it: before a prefix is used the tool tests that it binds each name
  once, none that either unit binds and none that is still an
  import, that it evaluates from the static environment of the
  imports, and that the term after with the prefix in front of it is
  well-formed; and the prefix is printed for each unit that has one,
  to be read against the other unit.
  The lines. A unit whose two terms have a prefix in front of them
  is reported apart, whatever imports it has left. The line begins
  "validated with its imports, given the .cmx of N of them" where a
  theorem is of the two runs shown: the prefix leaves no import, or
  `FlambdaValidateImportsSpec.verdict_holds_from` accepts the
  verdict (`verdict_knowing_holds_from`, for the second asking).
  Otherwise it begins "validated with nothing else imported,
  given the .cmx of N", and is followed by a line that begins "not
  validated with its imports", as for a unit with no prefix (the
  entry "What the tool reports"). Both rest on the two things above.
  The tool's Simplify looks for a `.cmx` file in the directories
  given with `-I` and in no other, whatever other options of the
  compiler the tool is given (`init_load_path`). Where it is given
  none it reads no `.cmx` file, no unit has a prefix, and nothing of
  this entry is assumed: a symbol of another unit is an import, as
  the entry "Symbols that a unit does not define" says.
- **The extraction**: Rocq's extraction to OCaml, by
  `middle_end/flambda2/rocq/FlambdaExtract.v`. It has no directive that
  realises a definition of the model by hand-written OCaml. The extracted
  modules are in `middle_end/flambda2/tests/semantics/extracted/`.
- **What the tool reports**, in the same file (`check`, `run`,
  `observe`): that it calls `validate_unit` on the two terms converted
  or, where that does not validate them, `validate_unit_knowing` on the
  same two terms, and prints the answer as it is. The comparison of the
  two runs that it also prints is a test of one run, and no theorem is
  about it.
  For a unit that has imports (the entry "Symbols that a unit does not
  define") the two runs are from its imports. The theorem of the
  validator is of the two units run from the empty static environment,
  where an external symbol is bound to nothing. Of the two units run
  from their imports there is a theorem for a verdict that passes a
  test of the model: `FlambdaValidateImportsSpec.verdict_holds_from`
  (`validate_unit_sound_verdict_imports_statement`), and
  `verdict_knowing_holds_from` for the second asking
  (`validate_unit_knowing_sound_verdict_imports_statement`). Three
  verdicts pass it whatever the imports are: the same up to bound
  names, the same once normalised, and the same once the unused
  static bindings of the term before are removed as well. The fourth,
  the same once unused static bindings are removed from the head of
  the term after too, passes it where no import is a symbol that the
  groups at the head of the term after bind, once the validation has
  normalised and tidied that term. What a lifting lifts is a closure,
  an immutable block, a boxed number or an immutable array of values
  that the unit makes as it runs: in the term lifted it is a symbol
  bound to a static constant of its parts lifted, which for the array
  is a static immutable array of its elements lifted (a part that is
  itself lifted is a symbol there). The fifth, the same once
  something is lifted, passes it: where that holds of the head of the
  term
  after, if the checks of the removal of bindings from that head
  pass, whether a binding goes or not; where no import is a name
  that the term binds at each step of the validation of the lifted
  term whose law from the imports asks it (the result of each lifting
  that is made; the term in which what the unit binds its symbols to
  is used, in each round in which it is, after each lifting and,
  after the last, after the read of the known value slots and the
  demotion of the exception handlers; and both terms whose static
  bindings are taken first, where the verdict is of that comparison);
  and where, for a lifting that is given a table to put at the head
  of the unit, no import is a symbol that the groups at the head of
  the unit bind once that step is done, whether the table is put or
  not. The test computes
  those terms a second time, as the validation does; it is
  sufficient, not necessary. The rounds that follow the demotion
  are those of `FlambdaValidateSpec.demote_and_tidy_known`, and
  their test is
  `FlambdaValidateImportsSpec.demote_and_tidy_known_free_in`: of
  each of those rounds that is executed, the one at which they stop
  included, it asks that no import is a name that the term binds
  (`static_names_free_in`), of the term that the round gives
  `fold_known_unit` (the term of the round once
  `use_known_closures` is applied to it), where `fold_known_unit`
  gives a term. Before the demotion the validation reads the known
  value slots of the term that the last lifting leaves, with every
  symbol that the unit binds taken to be in scope
  (`FlambdaKnownSlotsSpec.read_known_slots_anywhere`). For that
  read the test asks nothing: its law from the imports
  (`read_known_slots_anywhere_refines_imports_statement`) has no
  premise. The demotion and the rounds are given the term read, and
  their test is asked of that term. Nothing was asked after the
  last lifting when the validation did not use, after the demotion,
  what the unit binds its symbols to; and when it used that and did
  not read first, the test was asked of the term that the last
  lifting leaves. So for this verdict the test is not the one that
  it was before either change, and the validation that gives the
  verdict is not either. No inclusion of the verdicts accepted
  from the imports before either change is claimed or proved, in
  either direction: a unit that had this verdict from its imports
  could lose it, and a unit that had not could have it. The
  theorem is of no verdict that does
  not pass the test, and of the test with the verdict that the
  validator gives of the same arguments. The tool
  asks those two functions, as they are
  extracted, and no line that it prints for a unit that has imports
  begins "validated:". Trusted of that call: that what the tool gives
  the function is the static environment of the imports that the two
  runs are from (`imports_statics`, as it is extracted), with the
  arguments and the verdict of the asking whose verdict it reports.
  The call is written beside each asking, with the arguments of that
  asking, and the tool reads the one of the asking that it made last;
  where a unit is validated, the asking made last is the one whose
  verdict is reported. Where the test accepts the verdict, the line
  begins "validated with its imports", and the verdict is of the two
  runs that the tool shows. Where the validator accepts and the test
  does not, what is validated is
  the unit run with nothing imported: the line begins "validated
  with nothing imported" and is followed by a line that begins "not
  validated with its imports", and the comparison of the two runs
  from the imports is then an execution of both terms that is
  compared, not a validated verdict. In that case the tool also runs
  the unit before from the empty static environment, to say where
  the validation with nothing imported is vacuous.
- **A check that does not establish `refines` is in the model, and
  what it accepts the tool prints on a line of its own, never as
  validated.** This entry awaits the human's confirmation (decision
  D43). `FlambdaValidateRelatedSpec.validate_unit_related`
  accepts units whose exits a lifting changes, which `validate_unit`
  refuses. What is proved of it is another relation than `refines`,
  and neither is said to give the other: an exit of the unit before
  is answered by an exit of the unit after that nothing which may
  follow a unit tells apart from it
  (`FlambdaObserversSpec.worlds_observed`), and the unit after ends
  only if the unit before exits or goes wrong. Of a unit before that
  goes wrong nothing is said. What may follow a unit is a way out, or
  a handler with no environment of its own that binds nothing static,
  names no symbol and no code id, and projects none of the value
  slots that the check removes: the whole of that class
  (`FlambdaObserversSpec.observer`), a handler that makes an
  immutable array included, and no smaller one. It is not claimed
  that the two exits have the same values, nor that the two memories
  hold the same; nothing is claimed of a unit linked with another,
  which names its symbols. What may follow names no import either:
  of two exits that each carry an imported symbol, a predefined
  exception for one, alone or inside a block, it is not claimed that
  it is the same symbol. The other form of the contract relates the
  values of the two exits and every entry of the two memories
  structurally; the choice between the two is the human's. The tool
  asks the check only of a unit that it reports as not validated: the
  validation has refused the unit for every list of calls that the
  tool tries, and the run of the term before that the tool shows is
  not reported undefined. The function does not test that; the tool
  does. The tool asks first with no value slot removed, and with the
  value
  slots that it guesses only if that is refused; the class of what
  may follow is then the smaller one, and the line says which was
  asked. What the check accepts is printed under the line "not
  validated", on a line that begins "related once lifted" and has not
  the word "validated". The head of the line says what the check was
  given: whether its list of imports was empty, and whether a prefix
  made from the `.cmx` files of other units stood in front of the
  two terms, with the imports that the prefix left or with none
  left. The statement is of the two terms as they were given, the
  prefix in front where there is one, run from the list of imports
  given; with a prefix the line rests on the entry on the `.cmx`
  files, within its limits, as a line of a verdict does. That list
  may have an import of which nothing is known (the class
  `Import_unknown`): the statement is of every list of imports, and
  of such an import at an exit it claims no more than of another.
  The unit
  stays not validated: it is in no count of the units validated, of
  this tree, of the census or of the tool, and
  the units that have the line are in a table of their own, those
  with an import apart from those with none. That stands until the
  human has chosen. The line rests on
  `validate_unit_related_sound_statement` and, as every line of the
  tool does, on the conversion of the two terms and on the imports
  that the tool gives the function. What the tool guesses for the
  check (the calls to leave, the liftings, the value slots, the
  rounds) is not trusted: the statement is of every such guess.
- **The conversion of types**, in
  `middle_end/flambda2/tests/semantics/types/type_converter.ml` with
  `model_numbers.ml` and `model_names.ml`: between `Type_grammar.t` and
  `flambda_type`, both ways. Trusted: that each case is the
  correspondence its comment states; that it refuses what one side lacks
  and never approximates; that on the way to the model it drops only the
  `rec_info` of a function type, the expression of a `Rec_info` head,
  the string of `Poison`, the cache of free names, and a relation that
  the compiler's map binds to the empty set of names (so the round trip
  from the compiler is the identity only up to these); that a numbering
  is one-to-one for each sort of name; that a number is read as the file
  says (an integer by its signed value, and the bits of a float, a mask
  or a vector as an integer in two's complement of as many bits, which
  is the reading of `const_in_range` in FlambdaPrimitiveSpec.v and of
  the converter of terms) and one out of range is refused; that the
  size of a case of blocks is read
  and made faithfully through `Row_like_for_blocks.get_singleton` and
  `create`; and that the names `Flambda2_types__...` of the build are the
  modules of the types library.
  Also trusted, of the two functions that the round trip from the
  compiler uses: that `type_to_model_with_what_is_dropped` is
  `type_to_model` and gives besides the `rec_info` of every function
  type of the type, with the code id of that function type, and the
  variables that those and the expressions of the `Rec_info` heads of
  the type mention; that `type_of_model_with_rec_infos` is
  `type_of_model` but that a function type gets the first `rec_info`
  listed for its code id, if there is one; and that the two references
  through which they work are as they were when either returns or
  raises, so that `type_to_model` and `type_of_model` are the
  conversions described above.
- **The tests of types**, in the same directory (`model_type.ml`,
  `type_generator.ml`, `types_difftest.ml`). No theorem rests on them;
  what a reader of their output trusts:
  - `Model_type.normalise` and `same`: that two types of the model are
    identified only up to how maps and sets are written;
  - the comparison of what two joins give (the binary and the n-way; one
    join on arguments that are one value and on two copies), and of two
    meets with what they teach in the shared and unshared comparison:
    it is of text, not `Model_type.same`. `print_joined` (the type of
    the parameter and what the join says of each variable of its own
    that this type reaches; nothing of what the joined environment says
    of the names of the tests, so two joins that differ only there
    print alike), `print_joined_with_names` (the same with the types of
    the names of the tests and every variable of the join's own that
    any of these reaches; the shared parts use it) and
    `print_with_own_variables` (the types of an answer) print the result
    written in one way, and `rename_own_variables` names the variables
    that are not of the tests `e0`, `e1`, ... in the order in which the
    whole text mentions them, once for the whole text.
    Trusted: that the printer of `model_type.ml` shows every part of a
    type, and that the renaming touches nothing but those variables, so
    that equal text is the same type up to the names of those variables.
    Unequal text may be two writings of one type: it is then counted as
    a difference, never as agreement;
  - the adapters that call the compiler: that `call_in_the_compiler`
    calls the function named with the arguments converted; that `binary`
    and `n_way` are the two implementations; that `join_env` is the join
    at a continuation of one parameter of the kind given with one use
    for each type, each use knowing besides what `known` says and
    binding the variables of `locals`; that `joined_of_env` reads from
    the joined environment the type of the parameter and of each
    variable not of the tests that this type reaches, or that the types
    given as `reached_from` reach; that
    `types_of_names` reads what an environment says of each name; that
    `or_fatal_error` reports the first line that the compiler wrote
    before it stopped; that the structural setting of the flag for the
    sharing of joins is set before any join is run, and that
    `with_join_sharing` puts the compiler's default in its place only
    for the two runs whose titles name it; that `shared_parts` builds
    its types with the compiler's `More_type_creators.immutable_block`,
    `Type_grammar.create_variant` and `Env_extension.create` around
    parts that the converter made, and calls a case not shared only
    where the two copies are not one value;
  - the semantic checks: that they ask `type_satisfied_b` (extracted,
    proved to decide `type_satisfied`) and nothing else whether a value
    has a type; the generators of values and worlds are trusted for
    nothing;
  - the variables that a join makes of its own (`has_joined`): a value
    is said to have what a join gives only when, with the parameter
    holding the value, each such variable is given a value that has the
    type the joined environment gives it, or no value, and the value has
    the type of the parameter. The search is finite: when it finds
    nothing the case is reported as not run, never as a failure. What
    the joined environment says of other names and of aliases is not
    checked. A failure is reported only when there is no such variable;
  - the reporting: a comparison counts as agreeing only cases that were
    run and whose premise some value tried satisfies; a failure of the
    compiler or a refusal of the converter is "not run"; a disagreement
    that is recorded is shown with its label and does not fail the test;
    any other disagreement makes the test fail;
  - the chosen joins: that the values shown are tried as the lines say;
    nothing is concluded from "not for the values tried".
  - the files `type_comparisons.ml` and `types_of_compilations.ml`
    beside those named;
  - `same_text`: two printed answers are taken to be the same when
    they have the same words in the same order. Trusted: that the
    printer of `model_type.ml` brackets every part of a type and
    writes a string as its bytes, so that where its lines are broken
    says nothing;
  - the examples of the join (`examples_of_joins`): that an example
    marked with the name of a statement has the inputs and the answer
    of that statement of
    `middle_end/flambda2/types/rocq/FlambdaJoinTypesExamplesSpec.v` as
    the file was when the example was written (the variables, slots
    and code ids are those of the tests); that one marked as having
    another answer has the inputs of the statement; a difference is
    shown and does not fail the test;
  - the equality and the predicate: that `Type_equal.type_eqb` and
    `Well_formed.type_well_formed` are the extracted functions, and
    that the predicate is asked of the type with its maps written in
    one way where a map of the input may bind a key twice;
  - the round trip from the compiler: that the type which comes back
    is made with the `rec_info`s that the conversion to the model
    recorded, and only when no code id has two that differ for
    `Type_grammar.equal`; that the two types are then compared whole,
    as printed, by `Equal_types_for_debug.equal_type` and by
    `Type_grammar.equal`;
  - the free variables: that where what the converter dropped mentions
    no variable the two sets are compared as they are, and that the
    other comparison adds exactly the variables recorded as dropped;
  - the types of compilations: that `names_of` lists the variables
    that the levels of an environment define after the initial scope
    and the symbols that it defines (names of other units are not gone
    through); that the type compared is `Typing_env.find` of the name;
    that a type of a name is left out only if `Type_grammar.equal`
    says it is one already compared for that name; that Simplify
    reports to the observer where the types library says it does; and
    that the tool runs itself once for each unit listed and adds up
    what each run counted.
  - the examples of the meet (`examples_of_meets`): that an example
    has the name of a statement of FlambdaMeetMatchExamplesSpec.v,
    FlambdaMeetTagExamplesSpec.v, FlambdaMeetProjectionExamplesSpec.v
    or FlambdaMeetExamplesSpec.v
    (`middle_end/flambda2/types/rocq/`), and the steps, the two types
    and what that statement says of the type answered and of the
    variables, as the file was when the example was written; that
    where the statement speaks of two meets the example says which;
    that `scenario` numbers the variables from 1 in the order given
    and makes the environment by the calls of the steps, the
    relations by `Flambda2_types.Typing_env` with the flag that
    chooses the meet set to the implementation run; that the
    environment read through the observer is the one those calls
    made; a difference is printed and counted, and does not add to
    the disagreements of the test;
  - the chosen meets: nothing but that what is printed is what was
    seen;
  - the creation functions: that a call on which the model's function
    answers `None` is counted as agreeing when the compiler's raises
    a fatal error or `Invalid_argument`, whatever the message; that
    `kind_for_const` is compared through the unknown type of the
    kind;
  - `must_be_singleton`: that the model's outer `None` is compared
    with a fatal error of the compiler.
  - the model run: that `the_model` is the extracted
    `FlambdaJoinMeet.model_meet`, `model_add_equation`, `model_join`
    and the three functions of `FlambdaMeet.Flambda2_types` that add
    a relation, given the records of choices named, the depth of
    joins of the compiler's flag and fuel 200; that
    `scenario_in_the_model`, `env_of_model` and `join_in_the_model`
    make in the model the environments that `scenario`,
    `create_names` and `join_env` make in the compiler, call for
    call, the variables having the numbers of the numbering, but
    that the model adds each equation by a call of its own where
    `join_env` adds those of a use inside one `use_meet_env`, which
    is the same with the compiler's default strategy for the
    environment of a meet and is not claimed for the others;
  - what is seen (`seen_in_the_model`, `seen_in_the_compiler`,
    `print_seen`, `same_seen`): that the names asked are those that
    the scenario defines, and the parameter of a join; that a name
    is asked nothing of unless the environment defines it; that the
    type is `find` and the canonical element
    `get_canonical_simple_exn` at the name mode of types; that the
    other variables are those that the type answered and the types
    of those names reach, of which only the type is asked; that the
    variables numbered above the threshold are
    renamed in the order in which the text comes to them, and two
    texts are the same if `same_text` says so; that two sides which
    both give no answer are counted as the same whatever their
    reasons;
  - the names asked of a join (`names_asked_of_a_join`): that they
    are, on both sides, the names of the tests, the variables that
    the source defines for types only, each variable that a use
    binds, once, and the parameter;
  - the generated environments: that `generate_equations` and
    `generate_aliases` give a variable at most one equation, whose
    type mentions only variables numbered after it.
  - the types that are alike (`generate_alike`): that the two are a
    type of the generator and, on either side, that type again or
    one of the types of the same kind that the shrinker gives for
    it;
  - the types one step from a common type (`generate_siblings`):
    that each of the three is one of the types of the same kind that
    the shrinker gives for one type of the generator, or that type
    where it gives none;
  - the scenarios in the reverse order (`reversed`, in
    `scenarios_in_the_model`): that the definitions of the variables
    are moved before the other steps, in the reverse order, and
    nothing else of the scenario changes;
  - the inputs counted for T7-9 (`on_a_symbol`): that it says whether
    an extension of one of the two types met has an equation on a
    symbol of the tests (`has_an_extension_on`);
  - the generated inputs `generate_meet_of_relations`,
    `generate_meet_of_closures` and `generate_join_of_relations`:
    that they are what the comments at their definitions say; for
    the last, that the two equations of a relation at a use are
    those that `add_is_int_relation`, `add_get_tag_relation` and
    `add_is_null_relation` of the compiler add, in their order;
  - `up_to_the_alias_kept`: that it changes nothing of what is seen
    but the type of a name whose type is an alias, into the alias of
    the canonical element seen for that name, and only where the
    canonical element of what the alias is to, as seen (a constant:
    itself; a name: that of its line), is that element; that a
    difference between the model and the n-way meet which it removes
    is recorded under the label of T7-10 and is not a disagreement
    of the test; that the cases of `control_of_the_alias_kept` are
    what their names say;
  - the inputs counted for T7-9 by an alias of a symbol
    (`has_a_part`, `has_an_alias_of_a_symbol`,
    `an_alias_of_a_symbol`): that it says whether one of the two
    types met, or a type that an equation before the meet gives a
    name, is or has within it an alias of a symbol;
  - what is read of the analysis of a join (`analysis_of_the_model`,
    `analysis_of_the_compiler`): that the compiler's is read by
    `Analysis.simple_refined_at_join` with `fold_values_at_uses` and
    by `fold_variables_created_at_join` with
    `fold_definitions_at_uses`, and the model's by the extracted
    `simple_refined_at_join_of` with `values_at_uses` and
    `variables_created_at_join` with `definitions_at_uses`; that the
    names asked are those asked of the target which the target
    defines, and the variables made that what is seen reaches; that
    the uses are numbered by their positions on both sides;
  - its text (`print_analysis`): that the variables made have the
    names that the text of what is seen gave them, and that those
    created which it does not reach are ordered by what is read of
    them; that two analyses are compared only where what is seen of
    the two targets is the same;
  - the comparison of two readings (`compare_analyses`,
    `analysis_reading`): that it compares only where both joins
    answer a target and what is seen of the two is the same; that a
    reading missing on one side is then a disagreement, but for the
    one case counted as not run, in which the compiler's reader
    stops and a reader of the model has no answer;
  - `control_of_the_analysis`: that its two joins are what its
    comment says, and that the reader which stops is put in place of
    the compiler's and of nothing else.
  - the groups of static constants (`generate_static_group`,
    `static_const_of_model`): that the generator is what the comment
    at its definition says; that the constant of the compiler is the
    model's, an immutable block of the shape of values for a block,
    the boxed number of the kind of the constant for a boxed
    constant, a boxed float where the contents are a variable;
  - the two sides (`static_group_in_the_model`,
    `env_before_a_static_group`, `static_group_in_the_compiler`,
    `static_group_in_simplify`): that the model's environment is the
    one of `env_of_model` with the aliases added by the model's
    `add_equation`, and the compiler's is made by the functions that
    the types library exports, the names of the tests defined in
    the same order and the aliases added by its `add_equation`
    under the flag that chooses the meet; that Simplify is given
    that environment in the environment of the top level of a unit
    with no code, and is run on each prefix of the group; that its
    typing environment is read through the observer;
  - what is compared (`static_group_verdict`,
    `verdict_of_a_static_group`, `outcome_of_a_static_group`,
    `static_groups`): that the constant at which Simplify stops is
    the last of the first prefix on which it stops; that the
    environments are compared after the constants before it, or
    after the group, first and on every path, what the model itself
    sees and what the compiler sees both being read; that
    a difference is recorded under T3-10 only as the comment at
    `static_group_in_the_model` says, the model being made to give
    the symbol of a boxed number whose contents are a variable of
    kind value the type that its environment has for the variable;
    that a stop at a constant to which the model gives a type is
    recorded under T3-10 only where the constant is a boxed number
    of the variable that holds a naked float and that comparison
    passes;
    that `why_untyped` tells by the text of a block, and by the
    symbols bound before it, whether it names a symbol that is not
    defined or has a field that is a naked immediate;
  - `control_of_the_static_groups`: that its groups are what its
    lines say, and that where a line says so one side of a verdict
    is put there by hand.
  - what is seen, as text (`print_seen`, with
    `Model_type.normalise_by`): that the variables numbered above
    the threshold are named in the order in which a first text
    comes to them, and that the text answered has the sets of names
    and the maps keyed by names in the order of those names, and
    the lines of those variables in that order; that
    `normalise_by` differs from `normalise` only in the order of
    the variables;
  - the names to avoid (`names_for_a_join_to_avoid`, and `made` in
    `sides_of_joins`): that a join of the model is told to avoid
    the variables of the test, those that its source and its uses
    know and, in the joins after joins, those of the targets of
    every join run before it;
  - the joins after joins (`joined_step`, `run_joined_steps`,
    `sides_of_joins`): that the two sides run the same steps, a
    definition at the normal name mode, one scope further, an
    equation added by `add_equation` with the n-way meet, and a
    join after the scope of the environment so far of the
    environments that the lists of steps make from it, the uses
    numbered by their positions; that what is looked at is what is
    seen of the variables of the program;
  - the three joins of the shape of T5-10
    (`joins_of_the_shape_of_t5_10`): that its steps are those of
    `Recovered` of FlambdaJoinMeetExamplesSpec.v as that file was
    when it was written;
  - the generated programs (`generate_joined_steps`,
    `generated_joins_after_joins`): that the generator is what the
    comment at its definition says; that a difference of the model
    with the repaired test is recorded under T5-10 only where the
    model with the compiler's test sees what the compiler sees at
    every place looked at, and both sides of the run with the
    repaired test are read at every place
    (`outcome_of_the_repaired_test`);
  - the two controls (`control_of_the_names_to_avoid`,
    `control_of_the_repaired_test`): that their programs are those
    that the generator makes from the seeds named, and that where a
    line says so a reading of a run is replaced by hand
    (`with_a_reading_replaced`); that the model and the compiler
    are said to be told apart by a run only where both answer and
    every place looked at is read on both sides
    (`told_apart_where_read`).
  - the types kept in `.cmx` files (`environment_of_cmx`,
    `check_cmx`, in `types_of_compilations.ml`): that a file is
    loaded by the compiler's own loader and resolver
    (`Flambda_cmx.create_loader`, `load_cmx_file_contents`), the
    directory given being the whole load path and the current unit
    another one; that the environment is the one that
    `Typing_env.create` makes with that resolver, read through the
    observer; that the names asked are the symbols and the
    variables that `Serializable.ids_for_export` gives and whose
    compilation unit is the unit, a symbol at the kind of values
    and a variable at its own kind; that the units are those of
    the `.cmx` files of the directory, each named as its file with
    the first letter in upper case;
  - the rule of the test: that the directory is the one in which
    the compiler of the context `main` finds the standard library,
    and that the files there are those that the build of this tree
    made with its own compiler;
  - `check_environments`: that it does what `check_file` did for
    the environments of a file (the expected file of that test did
    not change);
  - `top_of_a_type`: that what is counted is the constructor at the
    top of the model's type, and for a head whether it may be null
    and whether it then has a variable;
  - the control (`control_of_the_tops`, and the rules of the
    directory `cmx_control`): that the four types are those that its
    lines say; that the three files are made as the rules say, by
    the compiler of the context, the second with `-opaque`, and the
    third by writing a line of text.
  - the meet of T7-13 (`the_meet_of_t7_13`,
    `run_the_meet_of_t7_13`, `the_meet_of_t7_13_in_every_order`):
    that the scenario is what the comment at its definition says,
    with the number of fields of the last block a parameter; that
    each run makes new variables, and that the order given is that
    of `Variable.compare` on the three of that run, taken when the
    scenario tells them, before its first step (`?made` of
    `scenario`); that the
    compiler's check is the flag
    `Oxcaml_flags.Flambda2.Debug.concrete_types_only_on_canonicals`,
    set for the runs that say so and put back; that a run is read
    as refused by the check where the compiler stops with a fatal
    error whose first words are those of
    `Typing_env.replace_equation`; that `v2` is said to have a type
    that is no alias and `v1` for its canonical element by
    `Typing_env.find` and `get_canonical_simple_exn` on the
    environment answered.
  - what is seen, as text (`print_seen`,
    `without_the_names_above`): that where the first text comes to
    two variables numbered above the threshold inside one set, it
    has them in the order of the texts of their types with every
    such variable written alike, and in the order of their numbers
    where those texts are the same; that a set which the two sides
    then write differently is reported as a difference;
  - the comparisons (`run` of `type_comparisons.ml`): that an
    input on which a comparison disagrees is shrunk only where it
    is shown (one of each recorded entry, and the first three of
    the others); the counts are of the inputs as generated;
  - the steps (`run_joined_steps`): that a join is given every
    use, also one whose environment is bottom; that a use is said
    to be bottom by `is_bottom` of the model's environment at its
    end, and that nothing which is run depends on the answer;
  - what the side of the model tells (`events_of_joins`,
    `sides_of_joins`): that the variables said to be made by a
    meet or by a join of the model are those that the environment
    after it has and the environments before it have not
    (`variables_of_env`); that a variable is said to be made twice
    where two such tellings of one run have it (`run_after_meets`);
  - the variable for each use (`a_variable_for_each_use`): that
    the model's first environment, and its environment after each
    `Further_scope`, are given a variable of kind value, for types
    only, numbered 1000, 2000 and so on, which no type mentions and
    the compiler's environments have not; that the comparisons run
    with it ask that no variable is made twice;
  - the generator (`generate_joined_steps ~meets_of_cases`): that
    the variants which it adds mention only the values 7 to 12,
    that no step gives one of those a type, and that without the
    argument the programs are those of the tenth series;
  - the outcomes (`outcome_with_a_variable_for_each_use`,
    `outcome_without_a_variable_for_each_use`,
    `the_check_of_the_compiler_passes`): that a run without a
    variable for each use is recorded under T7-11 only where the
    two do not see the same or only one of the two answers, a
    variable is made twice in it, and the run of the same program
    with a variable for each use agrees; that a run in which
    neither answers is counted as not run; that the compiler's
    check is the flag
    `Oxcaml_flags.Flambda2.Debug.concrete_types_only_on_canonicals`,
    set for that comparison and put back, and that any stop of the
    compiler under it is a disagreement.
  - the line (`the_line_of_t7_13`, `immediates_of_the_line`,
    `variant_of_the_line`): that its steps are the definitions and the
    equations that the comment before them describes, in each of the
    three ways; that `scenario` runs them in the compiler by the calls
    that it runs for the examples of the meet, and
    `scenario_in_the_model` in the model;
  - the runs (`run_the_line`, `the_line_of_t7_13_in_every_order`): that
    the order of a run is that of `Variable.compare` on its three
    variables, taken before its first step; that a run is said to stop
    at the compiler's check where the text of the fatal error begins as
    that of `Typing_env.replace_equation` does; that the check is the
    flag
    `Oxcaml_flags.Flambda2.Debug.concrete_types_only_on_canonicals`, set
    on or off for each batch of runs and put back; that the line fails
    unless each of the six orders is seen, with one answer in each: an
    order in which two distinct answers are seen, an order that is
    missing, or an order that is not one of the six, fails it;
  - the names with a type that is no alias and a canonical element that
    is not themselves (`concrete_and_not_canonical`): that they are read
    from what is seen (`seen_in_the_compiler`, `seen_in_the_model`) of
    the three names, the type being `find` and the canonical element
    `get_canonical_simple_exn` at the name mode of types; that there is
    no answer where no environment is seen or a name is seen without one
    of the two.
  - the two lines (`the_line_of_values`): that their steps are the
    definitions and the equations that the comment before them
    describes; that `scenario` runs them in the compiler and
    `scenario_in_the_model` in the model, as for the examples of the
    meet;
  - the runs (`run_a_line_of_values`, `the_lines_of_t7_14_and_t7_15`):
    that the order of a run is that of `Variable.compare` on its four
    variables, taken before its first step; that an environment is said
    to be bottom by `is_bottom` of what is seen of it
    (`seen_is_bottom`), and that nothing is said where no environment is
    seen; that the line asked fails unless each of the twenty-four
    orders is seen, with one answer in each: an order in which two
    distinct answers are seen, an order that is missing, a run that
    stops, or an order that is not one of the twenty-four, fails it.

Not trusted, though it is OCaml: what the tool proposes to
`validate_unit` and to `validate_unit_knowing`. Each proposal is
checked by an extracted function, and `validate_unit_sound` and
`validate_unit_knowing_sound` hold whatever is proposed.
- Which static bindings to remove as unused (`unused_statics`, of the
  term before and, with them, of the term after): checked by
  `drop_statics` for the term before, and for the term after by
  `drop_statics` under the checks of
  `FlambdaUnusedHeadStaticsSpec.head_statics_removable`
  (`without_head_statics`: where a check fails the term after is left
  as it is).
- Which `Let`s Simplify made static bindings, under which symbols, and
  in which code `my_closure` is a symbol (`lifting_guess`, once for
  each lifting of the list): the `Let`s outside code and the code ids
  are checked by `lift_unit`; the `Let`s in the bodies of code (the
  field `lifted_out_of_code` of the guess, which `lift_unit` does not
  read) by `lift_from_code_unit`; the `Let`s of the handlers that are
  not at toplevel, to bind at the head of the unit (the field
  `lifted_to_head`), by `lift_to_head_unit`.
- Which value slots Simplify removed from the static sets of closures
  (`unused_slots_guess`): checked by `drop_unused_slots`.
- Which code Simplify made a loop of, with a name for the loop of each
  (`loopify_guess`: the code that the front end marks for loopify, as
  the conversion notes it): checked by `loopify_unit`, which leaves
  alone the code that is not `loopable` with the name given.

The guesses are made from the two terms as `convert_both` gives them,
or from what extracted functions of the validation make of them, which
the tool computes again for the purpose: the term before with its loops
made, normalised, with the liftings made so far, and with its
continuations placed and tidied; the term after normalised; and for
`validate_unit_knowing` the same of the two terms normalised by
`normalise_knowing`. A symbol or a code id taken from the term after is
used in a guess with the number assigned by the conversion above. That
identification of names is trusted, as said there; a guess adds nothing
to it.

**The mark of a stub** (`FlambdaModel.cm_is_stub`) is not trusted for
soundness either. The converter gives code the compiler's flag
(`Code.stub`), but no theorem assumes that this flag is right. The
machine and the checks of well-formedness do not read it. The
comparison reads it and is stricter for a fixed pair of terms. Two
rewrites of the validation read it:
`FlambdaKnownSlotsSpec.read_known_slots`, through `read_slots` under
its checks, and `FlambdaStubCallsSpec.demote_stub_calls`. Their
refinement statements hold whatever the flag says: the first for a
unit run on its own, the second under all environments. A wrong flag
can change which pairs are validated, but does not make an accepted
pair unsound in the model. The absence of lost verdicts measured at
the default flags on the sets named in
`middle_end/flambda2/rocq/status/C3.md` is a measurement of the
series, not a theorem that every formerly accepted pair is still
accepted. That status also records losses with
`-flambda2-simplify-stubs`.

## The SSA loop passes

The libraries `OxCamlStdlib` (`rocq/`), `OxCamlBackend` (`backend/rocq/`),
`OxCamlSsa` (`backend/ssa/rocq/`) and `OxCamlLoop`
(`backend/ssa/loop/rocq/`). `ROCQ-REVIEW-GUIDE.md` (sections 2.3 and 2.4)
says what each group is to be checked against; the reasons here are
short. The trusted OCaml of these libraries is the extraction directives
of their `*Extract.v` files and the hand-written rewrites of the passes,
which that guide covers (sections 2.3, 4.7 and 5.5).

### `Int64` (`rocq/OxCamlInt64.v`)

Rocq has no 64-bit machine integer. `Int64` is an abstract type with its
operations, and each axiom is the documented behaviour of one OxCaml
operation; `int32`, `int64` and `nativeint` are the types of constants
the models carry and do not compute with.

- `OxCamlStdlib.OxCamlInt64.Int64.abs_spec`
- `OxCamlStdlib.OxCamlInt64.Int64.add_spec`
- `OxCamlStdlib.OxCamlInt64.Int64.compare_spec`
- `OxCamlStdlib.OxCamlInt64.Int64.div_spec`
- `OxCamlStdlib.OxCamlInt64.Int64.equal_spec`
- `OxCamlStdlib.OxCamlInt64.Int64.mul_spec`
- `OxCamlStdlib.OxCamlInt64.Int64.neg_spec`
- `OxCamlStdlib.OxCamlInt64.Int64.of_int_spec`
- `OxCamlStdlib.OxCamlInt64.Int64.rem_spec`
- `OxCamlStdlib.OxCamlInt64.Int64.shift_right_logical_spec`
- `OxCamlStdlib.OxCamlInt64.Int64.shift_right_spec`
- `OxCamlStdlib.OxCamlInt64.Int64.sub_spec`
- `OxCamlStdlib.OxCamlInt64.Int64.to_Z_bounded`
- `OxCamlStdlib.OxCamlInt64.Int64.to_Z_max_int`
- `OxCamlStdlib.OxCamlInt64.Int64.to_Z_min_int`
- `OxCamlStdlib.OxCamlInt64.Int64.to_Z_minus_one`
- `OxCamlStdlib.OxCamlInt64.Int64.to_Z_one`
- `OxCamlStdlib.OxCamlInt64.Int64.to_Z_zero`
- `OxCamlStdlib.OxCamlInt64.Int64.to_int_exact`
- `OxCamlStdlib.OxCamlInt64.int32`
- `OxCamlStdlib.OxCamlInt64.int64`
- `OxCamlStdlib.OxCamlInt64.int64_abs`
- `OxCamlStdlib.OxCamlInt64.int64_add`
- `OxCamlStdlib.OxCamlInt64.int64_compare`
- `OxCamlStdlib.OxCamlInt64.int64_div`
- `OxCamlStdlib.OxCamlInt64.int64_equal`
- `OxCamlStdlib.OxCamlInt64.int64_max_int`
- `OxCamlStdlib.OxCamlInt64.int64_min_int`
- `OxCamlStdlib.OxCamlInt64.int64_minus_one`
- `OxCamlStdlib.OxCamlInt64.int64_mul`
- `OxCamlStdlib.OxCamlInt64.int64_neg`
- `OxCamlStdlib.OxCamlInt64.int64_of_int`
- `OxCamlStdlib.OxCamlInt64.int64_one`
- `OxCamlStdlib.OxCamlInt64.int64_rem`
- `OxCamlStdlib.OxCamlInt64.int64_shift_right`
- `OxCamlStdlib.OxCamlInt64.int64_shift_right_logical`
- `OxCamlStdlib.OxCamlInt64.int64_sub`
- `OxCamlStdlib.OxCamlInt64.int64_to_Z`
- `OxCamlStdlib.OxCamlInt64.int64_to_int`
- `OxCamlStdlib.OxCamlInt64.int64_zero`
- `OxCamlStdlib.OxCamlInt64.nativeint`
- `OxCamlStdlib.OxCamlInt64.nativeint_to_int64`

### Overflow tests (`rocq/IntArith.v`)

Four tests of whether a sum or a product fits its type, each a few lines
of OCaml in `rocq/IntArithExtract.v` that are easier to read than to
model, with an axiom saying that it answers the question.

- `OxCamlStdlib.IntArith.add_fits`
- `OxCamlStdlib.IntArith.add_fits64`
- `OxCamlStdlib.IntArith.add_fits64_spec`
- `OxCamlStdlib.IntArith.add_fits_spec`
- `OxCamlStdlib.IntArith.mul_fits`
- `OxCamlStdlib.IntArith.mul_fits64`
- `OxCamlStdlib.IntArith.mul_fits64_spec`
- `OxCamlStdlib.IntArith.mul_fits_spec`

### List functions (`rocq/OxCamlList.v`)

Two functions of OxCaml's `List` that are mapped onto OxCaml's own
instead of being extracted, with what the models use of them.

- `OxCamlStdlib.OxCamlList.compare_length_with`
- `OxCamlStdlib.OxCamlList.sort_uniq`
- `OxCamlStdlib.OxCamlList.sort_uniq_In`
- `OxCamlStdlib.OxCamlList.sort_uniq_NoDup`

### Operations of the backend (`backend/rocq/Operation.v`)

The payloads of `Operation.t` that no model looks inside are abstract
types. `specific_linear` is the target's reading of its specific
operations as linear combinations of their arguments, the one thing
taken on trust about a target's semantics.

- `OxCamlBackend.Operation.addressing_mode`
- `OxCamlBackend.Operation.alloc`
- `OxCamlBackend.Operation.float_operation`
- `OxCamlBackend.Operation.float_width`
- `OxCamlBackend.Operation.int128_operation`
- `OxCamlBackend.Operation.intop_atomic`
- `OxCamlBackend.Operation.load`
- `OxCamlBackend.Operation.memory_chunk`
- `OxCamlBackend.Operation.name_for_debugger`
- `OxCamlBackend.Operation.probe_is_enabled`
- `OxCamlBackend.Operation.reinterpret_cast`
- `OxCamlBackend.Operation.signedness`
- `OxCamlBackend.Operation.specific_linear`
- `OxCamlBackend.Operation.specific_operation`
- `OxCamlBackend.Operation.static_cast`
- `OxCamlBackend.Operation.symbol`
- `OxCamlBackend.Operation.test`
- `OxCamlBackend.Operation.vec128_bits`
- `OxCamlBackend.Operation.vec256_bits`
- `OxCamlBackend.Operation.vec512_bits`

### The SSA graph (`backend/ssa/rocq/SsaModel.v`)

Blocks, instruction identifiers and sets of blocks are the compiler's
own data, abstract in the models: the types, the observations the models
make of a block, equality tests with the axioms that they decide
identity, and sets with the axioms of membership.

- `OxCamlSsa.SsaModel.all_successors`
- `OxCamlSsa.SsaModel.block`
- `OxCamlSsa.SsaModel.block_body`
- `OxCamlSsa.SsaModel.block_eqb`
- `OxCamlSsa.SsaModel.block_eqb_spec`
- `OxCamlSsa.SsaModel.blockset`
- `OxCamlSsa.SsaModel.dominates`
- `OxCamlSsa.SsaModel.instr_id`
- `OxCamlSsa.SsaModel.instr_id_eqb`
- `OxCamlSsa.SsaModel.instr_id_eqb_spec`
- `OxCamlSsa.SsaModel.op_def`
- `OxCamlSsa.SsaModel.op_def_find`
- `OxCamlSsa.SsaModel.param`
- `OxCamlSsa.SsaModel.params`
- `OxCamlSsa.SsaModel.predecessors`
- `OxCamlSsa.SsaModel.raise_kind`
- `OxCamlSsa.SsaModel.set_add`
- `OxCamlSsa.SsaModel.set_empty`
- `OxCamlSsa.SsaModel.set_for_all`
- `OxCamlSsa.SsaModel.set_for_all_spec`
- `OxCamlSsa.SsaModel.set_mem`
- `OxCamlSsa.SsaModel.set_mem_add`
- `OxCamlSsa.SsaModel.set_mem_empty`
- `OxCamlSsa.SsaModel.set_mem_singleton`
- `OxCamlSsa.SsaModel.set_singleton`
- `OxCamlSsa.SsaModel.strict_dominators`
- `OxCamlSsa.SsaModel.terminator`
- `OxCamlSsa.SsaModel.value_eqb`
- `OxCamlSsa.SsaModel.value_eqb_spec`

## Rocq's own

### Primitive 63-bit integers

Rocq's `int` is OxCaml's `int`. Its operations are primitives of Rocq's
kernel, and their specifications are axioms of Rocq's library. Only the
SSA libraries use them.

- `Corelib.Numbers.Cyclic.Int63.PrimInt63.add`
- `Corelib.Numbers.Cyclic.Int63.PrimInt63.addc`
- `Corelib.Numbers.Cyclic.Int63.PrimInt63.addcarryc`
- `Corelib.Numbers.Cyclic.Int63.PrimInt63.addmuldiv`
- `Corelib.Numbers.Cyclic.Int63.PrimInt63.asr`
- `Corelib.Numbers.Cyclic.Int63.PrimInt63.compare`
- `Corelib.Numbers.Cyclic.Int63.PrimInt63.compares`
- `Corelib.Numbers.Cyclic.Int63.PrimInt63.div`
- `Corelib.Numbers.Cyclic.Int63.PrimInt63.diveucl`
- `Corelib.Numbers.Cyclic.Int63.PrimInt63.diveucl_21`
- `Corelib.Numbers.Cyclic.Int63.PrimInt63.divs`
- `Corelib.Numbers.Cyclic.Int63.PrimInt63.eqb`
- `Corelib.Numbers.Cyclic.Int63.PrimInt63.head0`
- `Corelib.Numbers.Cyclic.Int63.PrimInt63.int`
- `Corelib.Numbers.Cyclic.Int63.PrimInt63.land`
- `Corelib.Numbers.Cyclic.Int63.PrimInt63.leb`
- `Corelib.Numbers.Cyclic.Int63.PrimInt63.lesb`
- `Corelib.Numbers.Cyclic.Int63.PrimInt63.lor`
- `Corelib.Numbers.Cyclic.Int63.PrimInt63.lsl`
- `Corelib.Numbers.Cyclic.Int63.PrimInt63.lsr`
- `Corelib.Numbers.Cyclic.Int63.PrimInt63.ltb`
- `Corelib.Numbers.Cyclic.Int63.PrimInt63.ltsb`
- `Corelib.Numbers.Cyclic.Int63.PrimInt63.lxor`
- `Corelib.Numbers.Cyclic.Int63.PrimInt63.mod`
- `Corelib.Numbers.Cyclic.Int63.PrimInt63.mods`
- `Corelib.Numbers.Cyclic.Int63.PrimInt63.mul`
- `Corelib.Numbers.Cyclic.Int63.PrimInt63.mulc`
- `Corelib.Numbers.Cyclic.Int63.PrimInt63.sub`
- `Corelib.Numbers.Cyclic.Int63.PrimInt63.subc`
- `Corelib.Numbers.Cyclic.Int63.PrimInt63.subcarryc`
- `Corelib.Numbers.Cyclic.Int63.PrimInt63.tail0`
- `Corelib.Numbers.Cyclic.Int63.Sint63Axioms.asr_spec`
- `Corelib.Numbers.Cyclic.Int63.Sint63Axioms.compare_spec`
- `Corelib.Numbers.Cyclic.Int63.Sint63Axioms.div_spec`
- `Corelib.Numbers.Cyclic.Int63.Sint63Axioms.leb_spec`
- `Corelib.Numbers.Cyclic.Int63.Sint63Axioms.ltb_spec`
- `Corelib.Numbers.Cyclic.Int63.Sint63Axioms.mod_spec`
- `Corelib.Numbers.Cyclic.Int63.Uint63Axioms.add_spec`
- `Corelib.Numbers.Cyclic.Int63.Uint63Axioms.addc_def_spec`
- `Corelib.Numbers.Cyclic.Int63.Uint63Axioms.addcarryc_def_spec`
- `Corelib.Numbers.Cyclic.Int63.Uint63Axioms.addmuldiv_def_spec`
- `Corelib.Numbers.Cyclic.Int63.Uint63Axioms.compare_def_spec`
- `Corelib.Numbers.Cyclic.Int63.Uint63Axioms.div_spec`
- `Corelib.Numbers.Cyclic.Int63.Uint63Axioms.diveucl_21_spec`
- `Corelib.Numbers.Cyclic.Int63.Uint63Axioms.diveucl_def_spec`
- `Corelib.Numbers.Cyclic.Int63.Uint63Axioms.eqb_correct`
- `Corelib.Numbers.Cyclic.Int63.Uint63Axioms.eqb_refl`
- `Corelib.Numbers.Cyclic.Int63.Uint63Axioms.head0_spec`
- `Corelib.Numbers.Cyclic.Int63.Uint63Axioms.land_spec`
- `Corelib.Numbers.Cyclic.Int63.Uint63Axioms.leb_spec`
- `Corelib.Numbers.Cyclic.Int63.Uint63Axioms.lor_spec`
- `Corelib.Numbers.Cyclic.Int63.Uint63Axioms.lsl_spec`
- `Corelib.Numbers.Cyclic.Int63.Uint63Axioms.lsr_spec`
- `Corelib.Numbers.Cyclic.Int63.Uint63Axioms.ltb_spec`
- `Corelib.Numbers.Cyclic.Int63.Uint63Axioms.lxor_spec`
- `Corelib.Numbers.Cyclic.Int63.Uint63Axioms.mod_spec`
- `Corelib.Numbers.Cyclic.Int63.Uint63Axioms.mul_spec`
- `Corelib.Numbers.Cyclic.Int63.Uint63Axioms.mulc_spec`
- `Corelib.Numbers.Cyclic.Int63.Uint63Axioms.of_to_Z`
- `Corelib.Numbers.Cyclic.Int63.Uint63Axioms.sub_spec`
- `Corelib.Numbers.Cyclic.Int63.Uint63Axioms.subc_def_spec`
- `Corelib.Numbers.Cyclic.Int63.Uint63Axioms.subcarryc_def_spec`
- `Corelib.Numbers.Cyclic.Int63.Uint63Axioms.tail0_spec`

### Classical logic

The excluded middle, used by the SSA libraries for case splits on whether
a register is defined. No statement of the Flambda 2 libraries uses it,
and the second check above keeps it so.

- `Stdlib.Logic.Classical_Prop.classic`
