(******************************************************************************
 *                                  OxCaml                                    *
 * -------------------------------------------------------------------------- *
 *                               MIT License                                  *
 *                                                                            *
 * Copyright (c) 2026 Jane Street Group LLC                                   *
 * opensource-contacts@janestreet.com                                         *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included    *
 * in all copies or substantial portions of the Software.                     *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
 * DEALINGS IN THE SOFTWARE.                                                  *
 ******************************************************************************)

(* Checks Simplify against the Rocq semantics of Flambda 2 on one example.

   This file has no command line of its own and does nothing when it is linked:
   flsemantics_main.ml is the program, and calls [main]. A program that has the
   two terms of a unit already calls [check], which prints what the command line
   prints of them and returns what the check came to ([checked]) where the
   command line exits.

   An [.fl] file is parsed to a Flambda unit, which Simplify then rewrites. (The
   text has no modes for the parameters of code: the unit that is checked, and
   that Simplify is given, is the unit read with, in the place of the mode of
   each, the least that the compiler can say of where a value is,
   [with_unknown_param_modes].) The unit before and the unit after are each
   converted to a term of the Rocq model
   (middle_end/flambda2/rocq/FlambdaModel.v) and run on the machine extracted
   from FlambdaSemanticsSpec.v. What the two runs come to is printed and
   compared: Simplify is right on the example if, unless the run before is
   undefined, the run after ends the same way.

   This is a test and not a proof. What is trusted, beyond the semantics itself:
   the conversion below, which must be read against the model. What a primitive
   computes is the model's to say (FlambdaPrimitiveSpec.v); the conversion only
   says which primitive of the model stands for one of the compiler's. A
   construct or a primitive the conversion does not know is reported as
   unsupported, and never run as if it were undefined, as that would make the
   check pass for nothing.

   Kinds. The conversion gives each [Let] the kind of its variable, each handler
   the kinds of its parameters, code its two arities and each call the two it is
   made at, with their unboxed products, and it keeps the kinds of the value
   slots it meets. The machine makes a call undefined that does not agree with
   its code (FlambdaSemanticsSpec.call_code), so a mistake here in the arities
   of a call or of code would make a run undefined that should not be: that is
   trusted too. Two things are cross-checked as the conversion goes: an arity as
   converted, unarized by the model, has the kinds that the compiler's own
   unarizing gives; and the parameter arity of code, unarized, has the kinds of
   the parameters of its body. The check of kinds of FlambdaWellKindedSpec.v is
   run on each of the two units and its answers are printed; nothing depends on
   them.

   Exceptions. The conversion gives each [Apply_cont] its trap action, with the
   continuation it names and its raise kind; each call the extra arguments of
   its exception continuation, with their kinds; and each handler whether it is
   an exception handler. The machine reads the extra arguments only, so a
   mistake there changes a run. The rest is read by the comparison of the
   validator and by the checks of well-formedness, which say whether a unit
   keeps its exception handlers apart and follows the discipline of trap
   actions: a mistake there changes what those accept.

   Partial applications and over-applications are in the model: a call with
   fewer or more arguments than the code of its callee has parameters is one,
   whatever its kind (FlambdaSemanticsSpec.shape_of_closure_call), and an
   application is converted with its arguments and its arities as they are. A
   partial application is a value of the model that the compiler has not got as
   such; at the end of a run it is observed as a function is.

   Tupled functions are in the model, as code that is marked: every call of the
   model gives such code its first argument as the tuple, and any arguments
   after it as they are, for an over-application, as a call of unknown arity
   does in the compiler (FlambdaSemanticsSpec.shape_of_closure_call). The
   compiler's direct call and its call of known arity give code its arguments as
   the code takes them, and the model has not got that of tupled code yet. So a
   direct call of tupled code is reported as unsupported, and so is a call of
   known arity, which names no code, in a unit that has tupled code.

   Calls that never return. A call whose result continuation is [Never_returns]
   is converted to a call that returns to a continuation bound around it, whose
   handler is [Invalid] ([convert_apply]). That reads [Never_returns] as making
   a normal return undefined, which is trusted; the backend puts nothing after
   such a call. Nothing is assumed of how Simplify finds that a callee never
   returns: if a run of the term after returns from such a call it becomes
   undefined, and where the term before has a defined outcome the two runs, once
   completed, then differ. The term of the model has a continuation that the
   compiler's term has not. The term before has, for the same call, the return
   continuation that the front end gave it; the validation gives it the same
   form by a rewrite that is proved (FlambdaNeverReturnsSpec.v), where the code
   called is bound in sight, its body does not mention its return continuation,
   and the other checks of that rewrite pass.

   What the text loses of a call. The text of a term does not give the arity of
   a call of unknown arity, which is read with one value for each argument and
   one value for its result, nor the arity of the arguments of a direct call,
   which is read with one value for each (label C2-5); and it does not say that
   a call is a probe. A unit with a call of unknown arity that returns to a
   continuation whose parameters are not one value is not checked
   ([result_arity_lost]): the term read is not the compiler's, and Simplify may
   stop on it. The arity of the arguments is given back to a unit read from a
   text ([with_lost_kinds], which says from what and where it cannot be): a
   direct call has that of the parameters of the code that it names, and a call
   of unknown arity one kind for each argument. A method call is read with one
   value for its result too; the conversion refuses a method call, and the
   reader does not read one as the compiler prints it. A call that is a probe is
   refused by the conversion ([convert_apply]), as the program runs its handler
   only while the probe is enabled. The term read from a text never has the
   mark, whatever the source had: such a unit is read as one with a plain call,
   and it is for whatever makes the text to refuse it (the scripts of
   tests/semantics/ocaml do). A body taken from the .cmx of another unit may
   have the mark, and is then left out of the prefix, so that a call of it is
   undefined.

   What the text loses of a primitive. The text does not write the kind of an
   [Opaque_identity], and its reader gives every one the kind of values. A unit
   read from a text is normalised before anything else is done with it: each
   [Opaque_identity] is given the kind of its argument ([with_lost_kinds], which
   says what is done to the variables bound to one). The kind so given is the
   one that the text lost where the kind obtained for the argument is its kind
   in the compiler's term and the primitive there had that kind
   ([with_lost_kinds] says where the first may fail). Those two conditions
   suffice; where either fails no recovery of the compiler's kind is claimed,
   the term read may not be the compiler's, and nothing here finds that out: the
   step is not a check of the kinds of the compiler's terms. The text writes the
   two forms of [Is_int] alike, and its reader gives the form that is for
   variants; that is not put right here. The scripts of tests/semantics/ocaml
   refuse a unit whose term had the other form; a text from elsewhere is checked
   as it is read (label C2-26).

   Phantom bindings. A binding whose variables are phantom is refused by the
   conversion ([convert_let]): the program runs nothing for it, and the model
   runs every [Let]. With [-g] and phantom lets on, Simplify may keep an unused
   binding as phantom instead of deleting it; the term that the compiler gives
   to Simplify has none. The text of a term does not say the mode of a binding,
   so the refusal cannot detect a phantom binding whose mode a text has lost, as
   for probes. A body taken from the .cmx of another unit may have one, and is
   then left out of the prefix; the term that Simplify makes of a unit may,
   where the tool is given those flags, and the unit is then not checked.

   Imports. The external symbols that the two units of a check mention, those of
   another compilation unit, the predefined exceptions among them, are listed
   once for both ([imports]), each with a class: an identity for a predefined
   exception, data for any other. The two units are run from the static
   environment that binds each of them to something imported
   (FlambdaImportsSpec.imports_statics), and the two runs are compared. The
   theorem of the validator is of two units run from the empty static
   environment, where such a symbol is bound to nothing. Of two units run from
   their imports there is a theorem for a verdict that passes a test of the
   model (FlambdaValidateImportsSpec.verdict_holds_from, and
   verdict_knowing_holds_from for the second asking, which are given the static
   environment of the imports and what the asking was given). Three verdicts
   pass it whatever the imports are. The one that removes unused static bindings
   from the head of the term after too passes it where no import is a symbol
   that the groups at the head of the term after bind, once the validation has
   normalised and tidied that term. The one that lifts passes it where three
   things hold: no import is a symbol that the groups at the head of that term
   bind, where the checks of the removal of bindings from that head pass,
   whether a binding goes or not; no import is a name that the term lifted
   binds, at each step of the validation whose law from the imports asks that;
   and, where a lifting is given a table to put at the head of the unit, no
   import is a symbol that the groups at the head of the unit bind after that
   step, whether the table is put or not. The test computes the terms that the
   validation compares a second time. For a verdict of the second asking the two
   terms are those that the second asking gives the first. So for a unit that
   has imports no line begins "validated:". Where the verdict passes the test,
   the line begins "validated with its imports", and it is of the two runs
   shown. Where the validator accepts and the test does not, the line begins
   "validated with nothing imported", which is what the theorem of the validator
   says of it, and is followed by a line that begins "not validated with its
   imports"; the comparison of the two runs shown is then a test and no more,
   and the unit before is run from the empty static environment as well, to say
   whether the validation with nothing imported is of a term that is undefined.

   The .cmx of other units. The compiler's Simplify reads the .cmx of another
   unit when it meets a symbol of that unit, and uses what the file says: it
   puts a closure symbol in the place of a load from the module block of the
   unit, makes a call direct, inlines a body. The tool's Simplify looks for a
   .cmx in the directories given with [-I] and in no other, whatever other
   options of the compiler the tool is given ([init_load_path]). The rules of
   the tests give none, but for the pairs of units of
   tests/semantics/ocaml/pairs/. Given none, it reads no .cmx and knows nothing
   of another unit, so the term after that is checked, of a unit that uses
   another unit, is then not the one that the compiler makes; and the model is
   told nothing of another unit either: a symbol of one is an import whose
   contents are unknown, the symbol of a closure included, and a code id of one
   is bound to nothing, so that a call of it is undefined when it runs.

   What the .cmx files say. Where the tool's Simplify read the .cmx of another
   unit ([units_read]), the model is told what the files say: the two units of a
   check are given one prefix of static bindings, the same in front of both,
   which says of the symbols and the code that the two units ask for what those
   files say ([make_prefix] has the rule). The two terms run and validated are
   the prefix followed by each unit, and a symbol that the prefix binds is not
   an import. So the verdict is of the unit run from the static environment that
   the files describe, and it is right of the linked program only if they are
   right about the units that the program is linked with: that, and
   [make_prefix], which says in static bindings what the compiler's reduction of
   the files says, are trusted. The prefix is printed before the two runs, with
   the units whose files it was made from, and the line of the verdict says of
   how many units they are: "validated with its imports, given the .cmx of N of
   them", where the theorems apply to the two runs shown (the prefix leaves no
   import, or the verdict is one of those that hold from the imports), and
   otherwise "validated with nothing else imported, given the .cmx of N",
   followed by the line that begins "not validated with its imports". Where
   there is no prefix, as nothing that the files say can be bound, or where it
   is made from fewer units than were read, a line names the units read. A unit
   for which no .cmx was read has no prefix and no such line.

   The known arrays read. Where every attempt of the validation has refused a
   unit, the attempts are made again, for the same lists of calls in the same
   order and with both askings, of FlambdaValidateReadingSpec.v:
   validate_unit_reading in the place of [validate_unit] and
   validate_unit_knowing_reading in the place of [validate_unit_knowing], each
   given the two terms as converted. They ask the two functions of the term
   after and of the term before that FlambdaValidateReadingSpec.before_read
   gives: its loops made and then, in rounds, the length of an immutable array
   that it makes or binds to a symbol made a constant and a load from one at a
   constant index made the element (FlambdaArrayLoadSpec.load_known_arrays,
   which refines, is not an equivalence, and is not in the rounds of
   [validate_unit]), and the term normalised as [validate_unit] normalises the
   term before. What they accept is validated:
   validate_unit_reading_sound_statement and the three statements beside it are
   of the term before as it was given. The tool computes the term that
   before_read gives itself as well: the guesses are made from it, and the test
   of the verdict from the imports is made of it, as those statements have that
   test. The line under the verdict then begins "(asked of the term before".
   This comes before the second check, which is asked only where these attempts
   refuse the unit too, and of the two terms as converted.

   The second check. Where every attempt of the validation that the tool makes
   has refused a unit, and the run of the term before that is shown is not
   reported undefined, the tool asks
   FlambdaValidateRelatedSpec.validate_unit_related of the same two terms, with
   the imports. The liftings are guessed again, each made without its test that
   the unit exits with symbols and constants only
   (FlambdaLiftObservedSpec.with_lifting_observed in the place of
   FlambdaLiftSpec's lifting with [exits_plain]; the other tests of a lifting,
   of well-formedness, of fresh names and of closed code, and the comparisons,
   are those of the function), for the same lists of calls in the same order;
   the check is asked with no value slot to remove and, if it refuses, with the
   slots guessed. What it accepts is not said to be validated: the line "not
   validated" stays, [validate] gives [Not_validated], and one line more begins
   "related once lifted". What is proved of such a pair is
   validate_unit_related_sound_statement, of the two terms that the check was
   given, run from the static environment of the imports that it was given: an
   exit of the first is answered by an exit of the second that no observer of
   the class of that statement tells apart from it, and the second ends only if
   the first exits or goes wrong. The values of the two exits need not be the
   same. The function tests itself what the statement asks of the imports, so
   there is no test of the verdict beside it; the head of the line says whether
   the list of imports was empty. Where a prefix was made from the .cmx of other
   units, the two terms are the two units each with the prefix in front of it,
   and the statement is of those: that the files are right about the units
   linked, and what a linked program does when it initialises them, are limits
   of every verdict with a prefix, and of this line too.

   Stubs. The conversion gives code the compiler's flag that says whether it is
   a stub ([Code.stub]). The machine does not read it and the checks of
   well-formedness do not; the comparison of the validator does, so two terms
   that differ in it are not the same. *)

(* The matches below name the few constructs and primitives the model has and
   send every other to [unsupported]; they are meant to stay as they are when
   the compiler gains a constructor. *)
[@@@ocaml.warning "-4"]

open Import
open Flambda2_kinds
open Flambda2_numbers
open Flambda2_bound_identifiers
open Flambda2_term_basics
module Maps = Flambda2_rocq_semantics.FlambdaMapSpec
module Kinds = Flambda2_rocq_semantics.FlambdaKindModel
module Primitives = Flambda2_rocq_semantics.FlambdaPrimitiveModel
module Model = Flambda2_rocq_semantics.FlambdaModel
module Values = Flambda2_rocq_semantics.FlambdaValueSpec
module Imports = Flambda2_rocq_semantics.FlambdaImportsSpec
module Semantics = Flambda2_rocq_semantics.FlambdaSemanticsSpec
module Regions = Flambda2_rocq_semantics.FlambdaRegionsSpec
module Compare_terms = Flambda2_rocq_semantics.FlambdaCompareSpec
module Normalise = Flambda2_rocq_semantics.FlambdaNormaliseSpec
module Unused_statics = Flambda2_rocq_semantics.FlambdaUnusedStaticsSpec
module Well_formed = Flambda2_rocq_semantics.FlambdaWellFormedSpec
module Well_kinded = Flambda2_rocq_semantics.FlambdaWellKindedSpec
module Binders = Flambda2_rocq_semantics.FlambdaBindersSpec
module Validate = Flambda2_rocq_semantics.FlambdaValidateSpec
module Validate_knowing = Flambda2_rocq_semantics.FlambdaValidateKnowingSpec
module Validate_imports = Flambda2_rocq_semantics.FlambdaValidateImportsSpec
module Lift = Flambda2_rocq_semantics.FlambdaLiftSpec
module Lift_observed = Flambda2_rocq_semantics.FlambdaLiftObservedSpec
module Validate_related = Flambda2_rocq_semantics.FlambdaValidateRelatedSpec
module Validate_reading = Flambda2_rocq_semantics.FlambdaValidateReadingSpec
module My_closure = Flambda2_rocq_semantics.FlambdaMyClosureSpec
module Unused_slots = Flambda2_rocq_semantics.FlambdaUnusedSlotsSpec
module Float_code = Flambda2_rocq_semantics.FlambdaFloatCodeSpec
module Loopify = Flambda2_rocq_semantics.FlambdaLoopifySpec
module Stub_calls = Flambda2_rocq_semantics.FlambdaStubCallsSpec
module Never_returns = Flambda2_rocq_semantics.FlambdaNeverReturnsSpec
module Statics_order = Flambda2_rocq_semantics.FlambdaStaticsOrderSpec
module Lift_from_code = Flambda2_rocq_semantics.FlambdaLiftFromCodeSpec
module Lift_from_arms = Flambda2_rocq_semantics.FlambdaLiftFromArmsSpec
module Numbers = Flambda2_rocq_semantics.BinNums
module Value_approximation = Flambda2_classic_mode_types.Value_approximation
module P = Flambda_primitive

exception Unsupported of (Format.formatter -> unit)

let unsupported fmt = Format.kdprintf (fun msg -> raise (Unsupported msg)) fmt

(* ---- Numbers ---------------------------------------------------------- *)

let rec positive_of_int n : Numbers.positive =
  if n <= 1
  then Coq_xH
  else if n land 1 = 0
  then Coq_xO (positive_of_int (n lsr 1))
  else Coq_xI (positive_of_int (n lsr 1))

let rec nat_of_int n : Flambda2_rocq_semantics.Datatypes.nat =
  if n <= 0 then O else S (nat_of_int (n - 1))

let int_of_nat (n : Flambda2_rocq_semantics.Datatypes.nat) =
  let rec count acc (n : Flambda2_rocq_semantics.Datatypes.nat) =
    match n with O -> acc | S n -> count (acc + 1) n
  in
  count 0 n

let rec int_of_positive (p : Numbers.positive) =
  match p with
  | Coq_xH -> 1
  | Coq_xO p -> 2 * int_of_positive p
  | Coq_xI p -> (2 * int_of_positive p) + 1

let int_of_z (z : Numbers.coq_Z) =
  match z with
  | Z0 -> 0
  | Zpos p -> int_of_positive p
  | Zneg p -> -int_of_positive p

(* The integers of 64 bits: a positive number of the model is read as, and
   written from, the unsigned reading of an [int64]. *)
let rec positive_of_int64 n : Numbers.positive =
  if Int64.equal n 1L
  then Coq_xH
  else if Int64.equal (Int64.logand n 1L) 0L
  then Coq_xO (positive_of_int64 (Int64.shift_right_logical n 1))
  else Coq_xI (positive_of_int64 (Int64.shift_right_logical n 1))

let z_of_int64 n : Numbers.coq_Z =
  if Int64.equal n 0L
  then Z0
  else if Int64.compare n 0L > 0
  then Zpos (positive_of_int64 n)
  else Zneg (positive_of_int64 (Int64.neg n))

(* Through [int64], as the opposite of the least [int] is not an [int]. *)
let z_of_int n = z_of_int64 (Int64.of_int n)

let rec int64_of_positive (p : Numbers.positive) =
  match p with
  | Coq_xH -> 1L
  | Coq_xO p -> Int64.mul 2L (int64_of_positive p)
  | Coq_xI p -> Int64.add (Int64.mul 2L (int64_of_positive p)) 1L

let int64_of_z (z : Numbers.coq_Z) =
  match z with
  | Z0 -> 0L
  | Zpos p -> int64_of_positive p
  | Zneg p -> Int64.neg (int64_of_positive p)

(* ---- Names ------------------------------------------------------------ *)

(* Each sort of name of the compiler is numbered from 1 as it is met. The names
   the compiler gives when it opens a binder are fresh, so a name stands for one
   thing throughout a unit. *)
module Numbering (Id : sig
  type t

  val compare : t -> t -> int
end) =
struct
  module Map = Map.Make (Id)

  let numbers = ref Map.empty

  let next = ref 1

  (* The name that each number was first given to. *)
  let named : (int, Id.t) Hashtbl.t = Hashtbl.create 64

  let find id =
    match Map.find_opt id !numbers with
    | Some n -> positive_of_int n
    | None ->
      let n = !next in
      incr next;
      numbers := Map.add id n !numbers;
      Hashtbl.replace named n id;
      positive_of_int n

  (* The name that has a number: the first, where [alias] gave the number to a
     second; none for a number that [fresh] made. *)
  let name_of number = Hashtbl.find_opt named (int_of_positive number)

  (* A number that no name has, or will have. *)
  let fresh () =
    let n = !next in
    incr next;
    positive_of_int n

  (* Gives [id] the number of [same_as]. *)
  let alias id ~same_as =
    ignore (find same_as : Numbers.positive);
    numbers := Map.add id (Map.find same_as !numbers) !numbers

  (* Forgets every name numbered so far. *)
  let reset () =
    numbers := Map.empty;
    next := 1;
    Hashtbl.reset named

  (* The names numbered so far, each with its number. *)
  let bindings () =
    List.map (fun (id, n) -> id, positive_of_int n) (Map.bindings !numbers)
end

module Variables = Numbering (Variable)
module Continuations = Numbering (Continuation)
module Symbols = Numbering (Symbol)
module Code_ids = Numbering (Code_id)
module Value_slots = Numbering (Value_slot)
module Function_slots = Numbering (Function_slot)

(* The kinds of the value slots met ([Value_slot.kind]): the table that the
   check of kinds is given. *)
let slot_kinds : (Numbers.positive * Flambda_kind.t) list ref = ref []

let find_value_slot slot =
  let n = Value_slots.find slot in
  if not (List.mem_assoc n !slot_kinds)
  then slot_kinds := (n, Value_slot.kind slot) :: !slot_kinds;
  n

(* ---- Primitives ------------------------------------------------------- *)

(* The primitives the model has (FlambdaPrimitiveModel.v), with the fields it
   keeps of each. One that is not there, or that has a field with a value that
   is not there, is unsupported. *)
let convert_kind (kind : Flambda_kind.t) : Kinds.flambda_kind =
  match kind with
  | Value -> Value
  | Naked_number Naked_immediate -> Naked_number Kind_naked_immediate
  | Naked_number Naked_float32 -> Naked_number Kind_naked_float32
  | Naked_number Naked_float -> Naked_number Kind_naked_float
  | Naked_number Naked_int8 -> Naked_number Kind_naked_int8
  | Naked_number Naked_int16 -> Naked_number Kind_naked_int16
  | Naked_number Naked_int32 -> Naked_number Kind_naked_int32
  | Naked_number Naked_int64 -> Naked_number Kind_naked_int64
  | Naked_number Naked_nativeint -> Naked_number Kind_naked_nativeint
  | Naked_number Naked_vec128 -> Naked_number Kind_naked_vec128
  | Naked_number Naked_vec256 -> Naked_number Kind_naked_vec256
  | Naked_number Naked_vec512 -> Naked_number Kind_naked_vec512
  | Naked_number Naked_mask -> Naked_number Kind_naked_mask
  | Region -> Region
  | Rec_info -> Rec_info

(* An arity of the model is the compiler's without its subkinds: one component
   to a parameter, each a kind or an unboxed product of components. It is
   rebuilt through the fold that the compiler gives, a product from the
   components folded so far, the last first. What the model then makes of it
   unarized must be what the compiler makes of the arity unarized. *)
type component_so_far =
  | Kind of Flambda_kind.t
  | Members of Model.arity_component list

let convert_arity (arity : _ Flambda_arity.t) : Model.arity =
  let component (so_far : component_so_far) : Model.arity_component =
    match so_far with
    | Kind kind -> Singleton (convert_kind kind)
    | Members members -> Unboxed_product (List.rev members)
  in
  let converted =
    Flambda_arity.fold_per_parameter arity ~init:(Members [])
      ~singleton:(fun _ kind -> Kind (Flambda_kind.With_subkind.kind kind))
      ~unboxed_product:(fun ~outer ~inner ->
        match outer with
        | Members members -> Members (component inner :: members)
        | Kind _ ->
          Misc.fatal_error "flsemantics: a kind with members, in an arity")
    |> List.map component
  in
  let unarized =
    List.map
      (fun kind -> convert_kind (Flambda_kind.With_subkind.kind kind))
      (Flambda_arity.unarize arity)
  in
  if Model.unarize converted <> unarized
  then
    Misc.fatal_errorf "flsemantics: the arity %a is not converted faithfully"
      Flambda_arity.print arity;
  converted

let convert_standard_int (kind : Flambda_kind.Standard_int.t) :
    Kinds.standard_int =
  match kind with
  | Tagged_immediate -> Standard_tagged_immediate
  | Naked_immediate -> Standard_naked_immediate
  | Naked_int8 -> Standard_naked_int8
  | Naked_int16 -> Standard_naked_int16
  | Naked_int32 -> Standard_naked_int32
  | Naked_int64 -> Standard_naked_int64
  | Naked_nativeint -> Standard_naked_nativeint

(* The integers among the kinds that [Num_conv] converts between: the model has
   not got the conversions to and from the floats. *)
let convert_standard_int_or_float (kind : Flambda_kind.Standard_int_or_float.t)
    : Kinds.standard_int option =
  match kind with
  | Tagged_immediate -> Some Standard_tagged_immediate
  | Naked_immediate -> Some Standard_naked_immediate
  | Naked_int8 -> Some Standard_naked_int8
  | Naked_int16 -> Some Standard_naked_int16
  | Naked_int32 -> Some Standard_naked_int32
  | Naked_int64 -> Some Standard_naked_int64
  | Naked_nativeint -> Some Standard_naked_nativeint
  | Naked_float32 | Naked_float -> None

let convert_boxable_number (kind : Flambda_kind.Boxable_number.t) :
    Kinds.boxable_number =
  match kind with
  | Naked_float32 -> Boxable_naked_float32
  | Naked_float -> Boxable_naked_float
  | Naked_int32 -> Boxable_naked_int32
  | Naked_int64 -> Boxable_naked_int64
  | Naked_nativeint -> Boxable_naked_nativeint
  | Naked_vec128 -> Boxable_naked_vec128
  | Naked_vec256 -> Boxable_naked_vec256
  | Naked_vec512 -> Boxable_naked_vec512
  | Naked_mask -> Boxable_naked_mask

let convert_signed_or_unsigned (signedness : P.signed_or_unsigned) :
    Primitives.signedness =
  match signedness with Signed -> Signed | Unsigned -> Unsigned

let convert_signedness (signedness : Scalar.Signedness.t) :
    Primitives.signedness =
  match signedness with Signed -> Signed | Unsigned -> Unsigned

let convert_or_unknown convert (known : _ Flambda2_lattices.Or_unknown.t) =
  match known with Known a -> Some (z_of_int (convert a)) | Unknown -> None

let convert_unary_primitive (p : P.unary_primitive) :
    Primitives.unary_primitive option =
  match p with
  | Block_load { kind = Values { tag; size; field_kind }; mut; field } ->
    let field_kind : Primitives.block_access_field_kind =
      match field_kind with Any_value -> Any_value | Immediate -> Immediate
    in
    let kind : Primitives.block_access_kind =
      Block_access_values
        ( convert_or_unknown Tag.Scannable.to_int tag,
          convert_or_unknown Target_ocaml_int.to_int size,
          field_kind )
    in
    let mut : Primitives.mutability =
      match mut with
      | Mutable -> Mutable
      | Immutable -> Immutable
      | Immutable_unique -> Immutable_unique
    in
    Some (Block_load (kind, mut, z_of_int (Target_ocaml_int.to_int field)))
  | Block_load _ -> None
  | Is_int { variant_only } -> Some (Is_int variant_only)
  | Is_null -> Some Is_null
  | Get_tag -> Some Get_tag
  | String_length String -> Some String_length
  | String_length Bytes -> None
  | Array_length (Array_kind Immediates) ->
    Some (Array_length (Array_kind Array_immediates))
  | Array_length (Array_kind Gc_ignorable_values) ->
    Some (Array_length (Array_kind Array_gc_ignorable_values))
  | Array_length (Array_kind Values) ->
    Some (Array_length (Array_kind Array_values))
  | Array_length Float_array_opt_dynamic ->
    Some (Array_length Float_array_opt_dynamic)
  | Opaque_identity { middle_end_only = _; kind } ->
    Some (Opaque_identity (convert_kind kind))
  | Int_arith (kind, Not) ->
    Some (Unary_int_arith (convert_standard_int kind, Not))
  | Int_arith (kind, Swap_byte_endianness) ->
    Some (Unary_int_arith (convert_standard_int kind, Swap_byte_endianness))
  | Num_conv { src; dst } -> (
    match
      convert_standard_int_or_float src, convert_standard_int_or_float dst
    with
    | Some src, Some dst -> Some (Num_conv (src, dst))
    | None, _ | _, None -> None)
  | Boolean_not -> Some Boolean_not
  | Unbox_number kind -> Some (Unbox_number (convert_boxable_number kind))
  | Box_number (kind, (Heap _ | Local _)) ->
    (* CR mshinwell: a box made in a region is converted as one made on the
       heap; see [region_variables]. *)
    Some (Box_number (convert_boxable_number kind))
  | Untag_immediate -> Some Untag_immediate
  | Tag_immediate -> Some Tag_immediate
  | Reinterpret_64_bit_word Tagged_int63_as_unboxed_int64 ->
    Some (Reinterpret_64_bit_word Tagged_int63_as_unboxed_int64)
  | Reinterpret_64_bit_word Unboxed_int64_as_tagged_int63 ->
    Some (Reinterpret_64_bit_word Unboxed_int64_as_tagged_int63)
  | Reinterpret_64_bit_word
      (Unboxed_int64_as_unboxed_float64 | Unboxed_float64_as_unboxed_int64) ->
    None
  | _ -> None

let convert_int_arith_op (op : P.binary_int_arith_op) :
    Primitives.binary_int_arith_op =
  match op with
  | Add -> Add
  | Sub -> Sub
  | Mul -> Mul
  | Div signedness -> Div (convert_signedness signedness)
  | Mod signedness -> Mod (convert_signedness signedness)
  | And -> And
  | Or -> Or
  | Xor -> Xor

let convert_int_shift_op (op : P.int_shift_op) : Primitives.int_shift_op =
  match op with Lsl -> Lsl | Lsr -> Lsr | Asr -> Asr

let convert_int_comparison (cmp : P.signed_or_unsigned P.comparison) :
    Primitives.int_comparison =
  match cmp with
  | Eq -> Int_eq
  | Neq -> Int_neq
  | Lt signedness -> Int_lt (convert_signed_or_unsigned signedness)
  | Gt signedness -> Int_gt (convert_signed_or_unsigned signedness)
  | Le signedness -> Int_le (convert_signed_or_unsigned signedness)
  | Ge signedness -> Int_ge (convert_signed_or_unsigned signedness)

let convert_binary_primitive (p : P.binary_primitive) :
    Primitives.binary_primitive option =
  match p with
  | String_or_bigstring_load (String, width) -> (
    (* The loads of an integer; not that of a float, nor those of vectors. *)
    match width with
    | Eight -> Some (String_or_bigstring_load Eight)
    | Eight_signed -> Some (String_or_bigstring_load Eight_signed)
    | Sixteen -> Some (String_or_bigstring_load Sixteen)
    | Sixteen_signed -> Some (String_or_bigstring_load Sixteen_signed)
    | Thirty_two -> Some (String_or_bigstring_load Thirty_two)
    | Sixty_four -> Some (String_or_bigstring_load Sixty_four)
    | Single | One_twenty_eight _ | Two_fifty_six _ | Five_twelve _ | Mask ->
      None)
  | String_or_bigstring_load ((Bytes | Bigstring), _) -> None
  | Array_load
      ( ((Immediates | Gc_ignorable_values | Values) as kind),
        ((Immediates | Gc_ignorable_values | Values) as load_kind),
        mut ) ->
    let kind : Primitives.array_kind =
      match kind with
      | Immediates -> Array_immediates
      | Gc_ignorable_values -> Array_gc_ignorable_values
      | _ -> Array_values
    in
    let load_kind : Primitives.array_load_kind =
      match load_kind with
      | Immediates -> Load_immediates
      | Gc_ignorable_values -> Load_gc_ignorable_values
      | _ -> Load_values
    in
    let mut : Primitives.mutability =
      match mut with
      | Mutable -> Mutable
      | Immutable -> Immutable
      | Immutable_unique -> Immutable_unique
    in
    Some (Array_load (kind, load_kind, mut))
  | Phys_equal Eq -> Some (Phys_equal Phys_eq)
  | Phys_equal Neq -> Some (Phys_equal Phys_neq)
  | Int_arith (kind, op) ->
    Some (Int_arith (convert_standard_int kind, convert_int_arith_op op))
  | Int_shift (kind, op) ->
    Some (Int_shift (convert_standard_int kind, convert_int_shift_op op))
  | Int_comp (kind, Yielding_bool cmp) ->
    let behaviour : Primitives.int_comparison_behaviour =
      Yielding_bool (convert_int_comparison cmp)
    in
    Some (Int_comp (convert_standard_int kind, behaviour))
  | Int_comp (kind, Yielding_int_like_compare_functions signedness) ->
    Some
      (Int_comp
         ( convert_standard_int kind,
           Yielding_int_like_compare_functions
             (convert_signed_or_unsigned signedness) ))
  | _ -> None

(* The compiler's [Make_block] carries the kind of each field, one for each
   argument, with its subkind, and Simplify takes an argument that is a variable
   to be of the subkind of its field: where it cannot be, the primitive becomes
   [Invalid] ([Simplify_variadic_primitive]). The model's [Make_block] has no
   such annotation and makes a block of whatever values it is given. So the only
   annotation converted is the one that says nothing, [any_value] for every
   field; a block kind that says more of a field is not in the model.

   A [Make_array] is converted when it is [Immutable] and its elements are of
   kind [Value]: it is the model's [Make_immutable_array], which makes a value
   with no identity. A [Mutable] one is an allocation with an identity, which
   the conversion does not make, and [Immutable_unique] is not a mutability of
   arrays. *)
let convert_variadic_primitive (p : P.variadic_primitive) ~num_args :
    Primitives.variadic_primitive option =
  match p with
  | Make_block (Values (tag, kinds), Immutable, (Heap _ | Local _))
  (* CR mshinwell: a block made in a region is converted as one made on the
     heap; see [region_variables]. *)
    when List.compare_length_with kinds num_args = 0
         && List.for_all
              (fun kind ->
                Flambda_kind.With_subkind.equal kind
                  Flambda_kind.With_subkind.any_value)
              kinds ->
    Some (Make_block (Block_kind_values (z_of_int (Tag.Scannable.to_int tag))))
  (* CR mshinwell: an array made in a region is converted as one made on the
     heap; see [region_variables]. *)
  | Make_array (Immediates, Immutable, (Heap _ | Local _)) ->
    Some (Make_immutable_array Array_immediates)
  | Make_array (Gc_ignorable_values, Immutable, (Heap _ | Local _)) ->
    Some (Make_immutable_array Array_gc_ignorable_values)
  | Make_array (Values, Immutable, (Heap _ | Local _)) ->
    Some (Make_immutable_array Array_values)
  | _ -> None

(* ---- Conversion of terms ---------------------------------------------- *)

(* A float is its bit pattern, read as a signed integer. *)
let convert_float32 f : Model.const =
  Naked_float32
    (z_of_int64
       (Int64.of_int32 (Numeric_types.Float32_by_bit_pattern.to_bits f)))

let convert_float f : Model.const =
  Naked_float (z_of_int64 (Numeric_types.Float_by_bit_pattern.to_bits f))

let convert_const (c : Reg_width_const.t) : Model.const =
  match Reg_width_const.descr c with
  | Naked_immediate i -> Naked_immediate (z_of_int (Target_ocaml_int.to_int i))
  | Tagged_immediate i ->
    Tagged_immediate (z_of_int (Target_ocaml_int.to_int i))
  | Null -> Null
  | Naked_int8 i -> Naked_int8 (z_of_int (Numeric_types.Int8.to_int i))
  | Naked_int16 i -> Naked_int16 (z_of_int (Numeric_types.Int16.to_int i))
  | Naked_int32 i -> Naked_int32 (z_of_int64 (Int64.of_int32 i))
  | Naked_int64 i -> Naked_int64 (z_of_int64 i)
  | Naked_nativeint i ->
    Naked_nativeint (z_of_int64 (Targetint_32_64.to_int64 i))
  | Naked_float32 f -> convert_float32 f
  | Naked_float f -> convert_float f
  | Poison (kind, _) -> Poison (convert_kind kind)
  | Naked_vec128 _ | Naked_vec256 _ | Naked_vec512 _ | Naked_mask _ ->
    unsupported "constant %a" Reg_width_const.print c

(* CR mshinwell: A hack, so that a unit made from an OCaml source can be
   converted: the model has nothing of [Rec_info]. Every function the front end
   makes begins with [let next_depth = rec_info (succ my_depth)], and a
   recursive call coerces its callee by [depth my_depth -> next_depth]. The
   conversion drops all three: the [Let] of a [Rec_info] (in [convert_let]), the
   [my_depth] of code (in [convert_code]), and every coercion (in
   [convert_simple], which was so before). So the model sees none of what
   Simplify does with depths: unrolling, and the inlining of recursive functions
   up to a depth. What keeps this from passing a unit for nothing: the variables
   dropped are noted here, and one used as a value is refused
   ([convert_variable]); without that it would be a variable that nothing binds,
   the unit before Simplify would be undefined, and anything refines an
   undefined unit. To be replaced when the model has depths and coercions. *)
let depth_variables : Variable.Set.t ref = ref Variable.Set.empty

let drop_depth_variable var =
  depth_variables := Variable.Set.add var !depth_variables

(* CR mshinwell: Regions are dropped, and what is allocated in one is converted
   as if it were allocated on the heap (decision D37). The model has no stack of
   regions and frees nothing. So the conversion drops the [Let] of a
   [Begin_region], makes the [Let] of an [End_region] bind the unit value that
   the primitive returns (in [convert_let]), and forgets the allocation mode of
   a block, a boxed number, a set of closures and a call, and the region of
   code. This is right for a unit in which nothing allocated in a region is used
   after the region has ended, which is what the type checker's modes are for,
   and which nothing in the middle end checks. It is an assumption about both
   units of a check: a validation says nothing about whether Simplify keeps
   every use of a local allocation inside its region, since a term that uses a
   block after its region has ended has, here, the meaning it would have if the
   block were on the heap. What keeps a unit from passing for nothing: the
   region variables dropped are noted here, and one used as a value is refused
   ([convert_variable]). A region is not used as a value where it is given to a
   continuation at a parameter of the kind [Region], or bound by a [Let] to
   another variable of that kind: the parameter, the argument and the [Let] are
   dropped, and the parameter and the variable are noted as region variables
   ([region_params]). The regions of exception handlers ([Begin_try_region],
   [End_try_region]) are dropped in the same way. One is opened where a handler
   is installed, and the handler ends it, which ends every region that the body
   had open when it raised. So for a unit with a handler the assumption is
   wider: nothing allocated in a region is used after the region has ended,
   whether it ends at its [End_region] or because a raise left it open and the
   handler ended it; and a validation says nothing about whether Simplify keeps
   it so on either way out. One thing is kept of the region of a handler: the
   [Let] of its [End_try_region] (the one that is not a ghost) binds
   [Opaque_identity] of the unit value, and not the unit value. The value is the
   same; but the normalisations of the validator do not fold [Opaque_identity],
   nor remove its [Let] for being unused, as Simplify does not delete an
   [End_try_region] for being unused ([Flambda_primitive.is_end_region] is of
   [End_region] alone); the one and the other still go with a handler or with
   code that is removed whole. So a handler that Simplify has demoted is not, in
   the model, a continuation that only applies another, which it is not for
   Simplify. To be replaced if the model gains regions. *)
let region_variables : Variable.Set.t ref = ref Variable.Set.empty

let drop_region_variable var =
  region_variables := Variable.Set.add var !region_variables

(* The conversion for the run on the machine of regions (FlambdaRegionsSpec.v),
   which is made after the check and changes nothing of it: the terms that are
   run, compared and validated are converted with [keep_regions] unset, as
   above. While it is set:

   A [Begin_region] or a [Begin_try_region] that is not a ghost is the model's
   primitive of that name, with no argument (the parent is dropped), and its
   variable is kept ([kept_regions]).

   An [End_region] or an [End_try_region] that is not a ghost, of a region that
   is kept, is the model's primitive on the variable of the region.

   An allocation whose mode is [Local] (a block, an immutable array, a boxed
   number, a set of closures) is followed by a [Let] of the same variable to
   [Local_allocation], applied to the variable and to the variable of the region
   that the mode names. The region of the code that the allocation is in
   ([code_regions]) is named by nothing: no variable is bound to it when the
   code runs.

   What is a ghost is converted as above: the [Let] of the begin is dropped and
   that of the end binds the unit value. So the end that is a ghost of a region
   that is kept ends nothing, as when the program runs.

   A region may be a parameter of a continuation, and be given to it as an
   argument: see [region_params], below. Another variable that a [Let] binds to
   a region is given the number of the region's own variable, and its [Let] is
   dropped: the two are one variable of the model.

   Refused: a kept region used anywhere else (an argument of a call, a value
   slot, a field), as a region dropped is; an allocation in a region that is
   neither kept nor the region of its code; the end that is not a ghost of a
   region that is not kept (one that a ghost began, the region of the code),
   which would otherwise be dropped without a word; and a local set of closures
   for which [let_closures] makes no [Let].

   The modes of calls, the [alloc_region]s and the result mode of code are not
   converted. [regions_met] says whether a conversion with [keep_regions] set
   met something of the first three paragraphs, or refused something for
   them. *)
let keep_regions : bool ref = ref false

let kept_regions : Variable.Set.t ref = ref Variable.Set.empty

let code_regions : Variable.Set.t ref = ref Variable.Set.empty

let regions_met : bool ref = ref false

(* A region as a parameter of a continuation. Simplify makes one where it lifts
   a handler out of another: the region that the handler ends, or allocates in,
   is given to it as an argument.

   [region_params] has each continuation that a [Let_cont] binds, with, for its
   parameters in order, whether each is of the kind [Region]. It is filled
   before anything that the continuation is in scope of is converted.

   With regions dropped, such a parameter is dropped, and noted as a region
   variable; and the argument at its place is dropped from every [Apply_cont] of
   the continuation, in the arms of a [Switch] too. The argument must be a
   region variable. This is what FlambdaRegionsSpec.erase makes of the term
   converted with regions kept, and [check_regions] tests that it is.

   With regions kept, such a parameter is kept, as a kept region, and the
   argument at its place is the variable of the region. A region variable that
   is dropped with regions kept may be given too: a ghost, and also the regions
   of code and of the unit, though the compiler is not known to give those. Its
   variable is then noted ([ghost_regions]), and once the unit is converted the
   parameters that such a variable reaches are dropped with their arguments
   ([drop_ghost_regions]).

   Refused, with regions kept or dropped: a region variable given anywhere but
   at such a place (to a parameter of another kind, to the return or exception
   continuation of code or of the unit), as any use of one as a value; anything
   but a region variable at such a place; and a call that returns or raises to a
   continuation with such a parameter. *)
let region_params : bool list Continuation.Map.t ref =
  ref Continuation.Map.empty

let ghost_regions : Model.variable list ref = ref []

let is_region_param (param : Bound_parameter.t) =
  match Flambda_kind.With_subkind.kind (Bound_parameter.kind param) with
  | Region -> true
  | Value | Naked_number _ | Rec_info -> false

let has_region_param k =
  match Continuation.Map.find_opt k !region_params with
  | Some places -> List.exists Fun.id places
  | None -> false

(* A variable where it is used as a value. *)
let convert_variable var =
  if Variable.Set.mem var !depth_variables
  then unsupported "the depth variable %a used as a value" Variable.print var;
  if Variable.Set.mem var !region_variables
  then unsupported "the region variable %a used as a value" Variable.print var;
  if Variable.Set.mem var !kept_regions
  then (
    regions_met := true;
    unsupported "the region variable %a used as a value" Variable.print var);
  Variables.find var

(* CR mshinwell: the coercions are dropped; see [depth_variables]. *)
let convert_simple (s : Simple.t) : Model.simple =
  Simple.pattern_match' s
    ~var:(fun var ~coercion:_ -> Model.Var (convert_variable var))
    ~symbol:(fun symbol ~coercion:_ -> Model.Symbol (Symbols.find symbol))
    ~const:(fun c -> Model.Const (convert_const c))

let convert_simples = List.map convert_simple

let convert_value_slots (slots : Simple.t Value_slot.Map.t) =
  Value_slot.Map.bindings slots
  |> List.map (fun (slot, s) -> find_value_slot slot, convert_simple s)

(* The model has two presentations of a set of closures (FlambdaModel.v): a set
   of one function whose function slot is not named, and a set of any number of
   functions at named function slots. Every unit is converted with the second,
   whatever its sets: a closure is at the function slot that the compiler's term
   has for it, and a projection of a value slot names the function slot that the
   compiler's primitive names ([project_from]), which the model compares with
   that of the closure.

   This conversion made the first presentation of a unit all of whose sets had
   one function and which projected no function slot, and no longer does. In
   that presentation the model had neither the function slot of a closure nor
   the one that a projection of a value slot names, so that two terms which
   differed in one of them only were one term of the model: a term after that
   put a closure at another function slot than the one from which its code
   projects was validated (labels C2-22 and X4-3).

   What this file does below with [Set_of_closures], [Project_value_slot] and
   [Static_const_set_of_closures] of the model is for terms in the first
   presentation, and no conversion makes one.

   CR mshinwell: the first presentation is to be deleted from the model, and
   with it what is left for it here. *)

(* The functions of a set, in order, and its value slots. *)
let convert_set_of_closures (set : Set_of_closures.t) =
  let funs =
    Function_declarations.funs_in_order (Set_of_closures.function_decls set)
  in
  let function_decls =
    List.map
      (fun ( function_slot,
             (code_id : Function_declarations.code_id_in_function_declaration)
           ) ->
        match code_id with
        | Code_id { code_id; _ } ->
          Function_slots.find function_slot, Code_ids.find code_id
        | Deleted _ -> unsupported "a closure of deleted code")
      (Function_slot.Lmap.bindings funs)
  in
  (match function_decls with
  | [] -> unsupported "a set of closures of no function"
  | _ :: _ -> ());
  function_decls, convert_value_slots (Set_of_closures.value_slots set)

let convert_raise_kind (kind : Trap_action.Raise_kind.t) : Model.raise_kind =
  match kind with
  | Regular -> Regular
  | Reraise -> Reraise
  | No_trace -> No_trace

let convert_trap_action (action : Trap_action.t) : Model.trap_action =
  match action with
  | Push { exn_handler } -> Push (Continuations.find exn_handler)
  | Pop { exn_handler; raise_kind } ->
    Pop
      (Continuations.find exn_handler, Option.map convert_raise_kind raise_kind)

(* Tupled code ([let f (a, b) = ...]) has one parameter for each component of
   the tuple. A call of unknown arity gives it the tuple, which the compiler
   takes apart on the way in, and so does every call of the model. A direct
   call, and a call of known arity, give it the components, and the model has
   not got that. What is noted of the conventions of calls as a unit is
   converted: the code that is tupled, the code that a direct call names, and
   whether a call of known arity was met ([check_calls_of_tupled_code]). *)
let tupled_code : Code_id.t list ref = ref []

let directly_called_code : Code_id.t list ref = ref []

let known_arity_call_met : bool ref = ref false

(* The region variable that a simple is, if it is one, kept or dropped. *)
let region_variable_of (s : Simple.t) : Variable.t option =
  Simple.pattern_match' s
    ~var:(fun var ~coercion:_ ->
      if
        Variable.Set.mem var !kept_regions
        || Variable.Set.mem var !region_variables
      then Some var
      else None)
    ~symbol:(fun _ ~coercion:_ -> None)
    ~const:(fun _ -> None)

(* An argument given to a parameter that is a region; see [region_params]. *)
let convert_region_argument (arg : Simple.t) : Model.simple option =
  match region_variable_of arg with
  | Some var when Variable.Set.mem var !kept_regions ->
    Some (Model.Var (Variables.find var))
  | Some var ->
    if !keep_regions
    then (
      let var = Variables.find var in
      ghost_regions := var :: !ghost_regions;
      Some (Model.Var var))
    else None
  | None ->
    if !keep_regions then regions_met := true;
    unsupported "%a, which is not a region, given to a parameter that is one"
      Simple.print arg

(* The arguments of an [Apply_cont] of a continuation whose parameters are
   regions at [places], in order as [convert_simples] converts them. *)
let rec convert_arguments (places : bool list) (args : Simple.t list) :
    Model.simple list =
  match places, args with
  | [], args -> convert_simples args
  | _ :: _, [] -> []
  | false :: places, arg :: args ->
    let arg = convert_simple arg in
    arg :: convert_arguments places args
  | true :: places, arg :: args -> (
    match convert_region_argument arg with
    | Some arg -> arg :: convert_arguments places args
    | None -> convert_arguments places args)

let convert_apply_cont (ac : Apply_cont_expr.t) : Model.apply_cont =
  let k = Apply_cont_expr.continuation ac in
  let args = Apply_cont_expr.args ac in
  let ac_continuation = Continuations.find k in
  let ac_args =
    match Continuation.Map.find_opt k !region_params with
    | Some places -> convert_arguments places args
    | None -> convert_simples args
  in
  { ac_continuation;
    ac_args;
    ac_trap_action =
      Option.map convert_trap_action (Apply_cont_expr.trap_action ac)
  }

let convert_params (params : Bound_parameters.t) =
  List.map Variables.find (Bound_parameters.vars params)

(* The kinds of parameters, which the model keeps beside them. *)
let convert_params_kinds (params : Bound_parameters.t) =
  List.map
    (fun param ->
      let kind = Bound_parameter.kind param in
      convert_kind (Flambda_kind.With_subkind.kind kind))
    (Bound_parameters.to_list params)

let convert_bound_var (bound_var : Bound_var.t) =
  let var = Bound_var.var bound_var in
  Variables.find var, convert_kind (Variable.kind var)

(* The code that is a newer version of other code, with that code, and the code
   that is bound as deleted, noted as a unit is converted. *)
let newer_versions : (Code_id.t * Code_id.t) list ref = ref []

let deleted_code : Code_id.t list ref = ref []

(* The code ids, declared deleted, whose number has been given to a newer
   version of the code ([convert_both]). The number then stands for the code
   that the unit binds, so the declaration is left out: the term of the model
   does not declare deleted a code id that it binds. It is empty except while
   [convert_both] converts the unit after for the second time, so that no other
   conversion leaves a declaration out. *)
let superseded_code : Code_id.t list ref = ref []

(* The code that the front end marks for loopify
   ([Loopify_attribute.should_loopify]), noted as a unit is converted. *)
let loopify_marked : Code_id.t list ref = ref []

(* Forgets everything that the conversion has noted so far, so that a unit is
   converted as if it were the first: the numbers given to names and what is
   kept beside the terms. Whoever gives the conversion more state adds it
   here. *)
let reset_conversion () =
  Variables.reset ();
  Continuations.reset ();
  Symbols.reset ();
  Code_ids.reset ();
  Value_slots.reset ();
  Function_slots.reset ();
  slot_kinds := [];
  depth_variables := Variable.Set.empty;
  newer_versions := [];
  deleted_code := [];
  superseded_code := [];
  loopify_marked := [];
  region_variables := Variable.Set.empty;
  kept_regions := Variable.Set.empty;
  code_regions := Variable.Set.empty;
  region_params := Continuation.Map.empty;
  ghost_regions := [];
  tupled_code := [];
  directly_called_code := [];
  known_arity_call_met := false

(* The region that the mode of an allocation names, if the mode is [Local]. *)
let local_region (mode : Alloc_mode.For_allocations.t) : Variable.t option =
  match mode with Heap _ -> None | Local { region; _ } -> Some region

let local_region_of_prim (prim : P.t) : Variable.t option =
  match prim with
  | Variadic (Make_block (_, _, mode), _)
  | Variadic (Make_array (_, _, mode), _)
  | Unary (Box_number (_, mode), _) ->
    local_region mode
  | Nullary _ | Unary _ | Binary _ | Ternary _ | Quaternary _ | Variadic _ ->
    None

(* [Local_allocation] of the variable [var], which the [Let] before it binds to
   an allocation in [region]. *)
let local_allocation (var : Model.variable) (region : Variable.t) : Model.named
    =
  regions_met := true;
  let named : Model.simple list =
    if Variable.Set.mem region !code_regions
    then []
    else if Variable.Set.mem region !kept_regions
    then [Var (Variables.find region)]
    else
      unsupported
        "an allocation in %a, which is not a region that the term begins"
        Variable.print region
  in
  Prim (Region_primitive Local_allocation, Var var :: named)

let rec convert_expr (e : Flambda.Expr.t) : Model.expr =
  match Flambda.Expr.descr e with
  | Let let_expr ->
    Flambda.Let_expr.pattern_match let_expr ~f:(fun pattern ~body ->
        convert_let pattern (Flambda.Let_expr.defining_expr let_expr) body)
  | Let_cont (Non_recursive { handler; _ }) ->
    Flambda.Non_recursive_let_cont_handler.pattern_match handler
      ~f:(fun k ~body ->
        let handler = Flambda.Non_recursive_let_cont_handler.handler handler in
        note_region_params k ~invariant_params:Bound_parameters.empty handler;
        let handler =
          convert_handler ~invariant_params:Bound_parameters.empty handler
        in
        Model.Let_cont_non_recursive
          (Continuations.find k, handler, convert_expr body))
  | Let_cont (Recursive handlers) ->
    Flambda.Recursive_let_cont_handlers.pattern_match handlers
      ~f:(fun ~invariant_params ~body handlers ->
        let handlers =
          Flambda.Continuation_handlers.to_map handlers
          |> Continuation.Lmap.bindings
        in
        List.iter
          (fun (k, handler) -> note_region_params k ~invariant_params handler)
          handlers;
        let handlers =
          List.map
            (fun (k, handler) ->
              Continuations.find k, convert_handler ~invariant_params handler)
            handlers
        in
        Model.Let_cont_recursive (handlers, convert_expr body))
  | Apply apply -> convert_apply apply
  | Apply_cont ac -> Model.Apply_cont (convert_apply_cont ac)
  | Switch switch ->
    let arms =
      Target_ocaml_int.Map.bindings (Switch_expr.arms switch)
      |> List.map (fun (discriminant, arm) ->
          ( z_of_int (Target_ocaml_int.to_int discriminant),
            convert_apply_cont (Switch_arm.action arm) ))
    in
    Model.Switch (convert_simple (Switch_expr.scrutinee switch), arms)
  | Invalid _ -> Model.Invalid

(* The parameters of [handler] that are regions, noted for the continuation [k]
   that it is bound to; see [region_params]. The handler is opened as for
   printing, with the names that it has: only the kinds are read. *)
and note_region_params k ~invariant_params
    (handler : Flambda.Continuation_handler.t) =
  Flambda.Continuation_handler.pattern_match_for_printing handler
    ~f:(fun params ~handler:_ ->
      let params = Bound_parameters.append invariant_params params in
      region_params
        := Continuation.Map.add k
             (List.map is_region_param (Bound_parameters.to_list params))
             !region_params)

and convert_handler ~invariant_params (handler : Flambda.Continuation_handler.t)
    : Model.continuation_handler =
  let is_exn_handler = Flambda.Continuation_handler.is_exn_handler handler in
  Flambda.Continuation_handler.pattern_match handler ~f:(fun params ~handler ->
      let params = Bound_parameters.append invariant_params params in
      (* The parameters that are regions; see [region_params]. *)
      let params =
        let regions, others =
          List.partition is_region_param (Bound_parameters.to_list params)
        in
        if !keep_regions
        then (
          List.iter
            (fun param ->
              kept_regions
                := Variable.Set.add (Bound_parameter.var param) !kept_regions)
            regions;
          params)
        else (
          List.iter
            (fun param -> drop_region_variable (Bound_parameter.var param))
            regions;
          Bound_parameters.create others)
      in
      let kinds = convert_params_kinds params in
      Model.Handler
        ( convert_params params,
          (if is_exn_handler
           then Model.make_exn_handler_metadata kinds
           else Model.make_handler_metadata kinds),
          convert_expr handler ))

and convert_let (pattern : Bound_pattern.t) (defining_expr : Flambda.Named.t)
    body : Model.expr =
  (* The program runs nothing for a binding whose variables are phantom: the
     backend compiles its body alone ([To_cmm_expr.let_expr]). The model runs
     every [Let], so such a binding is refused, whatever it binds and before any
     case below: converted, it could be undefined in the model where the program
     is defined. With [-g] and phantom lets on, Simplify may keep an unused
     binding as phantom instead of deleting it. The text of a term does not say
     the mode of a binding, so a term read from a text has none, and the refusal
     cannot detect one whose mode a text has lost. A body taken from a .cmx may
     have one, and is then not bound, as any body that the conversion refuses
     ([make_prefix]). A binding whose variables are for types only is not for a
     term at all. *)
  (match Bound_pattern.name_mode pattern with
  | Normal -> ()
  | Phantom -> unsupported "a phantom binding"
  | In_types -> unsupported "a binding of a variable that is for types only");
  match pattern, defining_expr with
  | Singleton bound_var, Simple s
    when Option.is_some (region_variable_of s)
         && Flambda_kind.equal
              (Variable.kind (Bound_var.var bound_var))
              Flambda_kind.region ->
    (* A variable of the kind [Region] bound to a region variable: another name
       for the region. The [Let] is dropped. With regions dropped the variable
       is noted as a region variable. With regions kept it is given the number
       of the region, and is kept or not as the region is; see
       [region_params]. *)
    let var = Bound_var.var bound_var in
    (match region_variable_of s with
    | Some region when Variable.Set.mem region !kept_regions ->
      Variables.alias var ~same_as:region;
      kept_regions := Variable.Set.add var !kept_regions
    | Some region ->
      if !keep_regions then Variables.alias var ~same_as:region;
      drop_region_variable var
    | None -> assert false);
    convert_expr body
  | Singleton bound_var, Simple s ->
    let var, kind = convert_bound_var bound_var in
    Model.Let (var, kind, Simple (convert_simple s), convert_expr body)
  | ( Singleton bound_var,
      Prim
        ( Variadic
            ( (( Begin_region { ghost = false }
               | Begin_try_region { ghost = false } ) as p),
              _ ),
          _ ) )
    when !keep_regions ->
    (* For the machine of regions; see [keep_regions]. *)
    regions_met := true;
    let var = Bound_var.var bound_var in
    kept_regions := Variable.Set.add var !kept_regions;
    let p : Primitives.region_primitive =
      match p with Begin_try_region _ -> Begin_try_region | _ -> Begin_region
    in
    Model.Let
      ( Variables.find var,
        Kinds.Region,
        Prim (Region_primitive p, []),
        convert_expr body )
  | ( Singleton bound_var,
      Prim
        ( Variadic
            ((Begin_region { ghost = _ } | Begin_try_region { ghost = _ }), _),
          _ ) ) ->
    (* CR mshinwell: the binding of a region is dropped, and with it the region
       it is given as its parent, if any; see [region_variables]. *)
    drop_region_variable (Bound_var.var bound_var);
    convert_expr body
  | ( Singleton bound_var,
      Prim
        ( Unary
            ((End_region { ghost = _ } | End_try_region { ghost = _ }), region),
          _ ) ) ->
    (* CR mshinwell: the end of a region does nothing but return the unit value;
       that of the region of a handler returns it through [Opaque_identity], to
       be left where it is. See [region_variables]. *)
    let is_region =
      Simple.pattern_match' region
        ~var:(fun var ~coercion:_ -> Variable.Set.mem var !region_variables)
        ~symbol:(fun _ ~coercion:_ -> false)
        ~const:(fun _ -> false)
    in
    (* For the machine of regions; see [keep_regions]. *)
    let kept_region =
      Simple.pattern_match' region
        ~var:(fun var ~coercion:_ ->
          if Variable.Set.mem var !kept_regions then Some var else None)
        ~symbol:(fun _ ~coercion:_ -> None)
        ~const:(fun _ -> None)
    in
    if not (is_region || Option.is_some kept_region)
    then (
      if !keep_regions then regions_met := true;
      unsupported "the end of %a, which is not a region" Simple.print region);
    (* For the machine of regions: an end that runs must not be dropped. *)
    let is_ghost =
      match defining_expr with
      | Prim (Unary (End_region { ghost }, _), _)
      | Prim (Unary (End_try_region { ghost }, _), _) ->
        ghost
      | _ -> true
    in
    if !keep_regions && (not is_ghost) && Option.is_none kept_region
    then (
      regions_met := true;
      unsupported
        "the end, which is not a ghost, of %a, which no begin that is not a \
         ghost made"
        Simple.print region);
    let var, kind = convert_bound_var bound_var in
    let unit : Model.simple = Const (Tagged_immediate (z_of_int 0)) in
    let defining_expr : Model.named =
      match defining_expr, kept_region with
      | Prim (Unary (End_region { ghost = false }, _), _), Some region ->
        Prim (Region_primitive End_region, [Var (Variables.find region)])
      | Prim (Unary (End_try_region { ghost = false }, _), _), Some region ->
        Prim (Region_primitive End_try_region, [Var (Variables.find region)])
      | Prim (Unary (End_try_region { ghost = false }, _), _), None ->
        Prim (Unary (Opaque_identity (convert_kind Flambda_kind.value)), [unit])
      | _, _ -> Simple unit
    in
    Model.Let (var, kind, defining_expr, convert_expr body)
  | Singleton bound_var, Prim (prim, _)
    when !keep_regions && Option.is_some (local_region_of_prim prim) ->
    (* For the machine of regions; see [keep_regions]. *)
    let var, kind = convert_bound_var bound_var in
    let allocation = convert_prim prim in
    let local =
      match local_region_of_prim prim with
      | Some region -> local_allocation var region
      | None -> assert false
    in
    Model.Let
      (var, kind, allocation, Model.Let (var, kind, local, convert_expr body))
  | Singleton bound_var, Prim (prim, _) ->
    let var, kind = convert_bound_var bound_var in
    Model.Let (var, kind, convert_prim prim, convert_expr body)
  | Set_of_closures bound_vars, Set_of_closures (set, mode) -> (
    let function_decls, slots = convert_set_of_closures set in
    (* [let_closures] gives each variable the kind [Value]. *)
    let bound_vars =
      List.map
        (fun bound_var ->
          match convert_bound_var bound_var with
          | var, Kinds.Value -> var
          | _, (Naked_number _ | Region | Rec_info) ->
            unsupported "the binding %a" Bound_pattern.print pattern)
        bound_vars
    in
    match
      Model.let_closures bound_vars function_decls slots (convert_expr body)
    with
    | Some e -> (
      (* For the machine of regions: the [Local_allocation] goes after the [Let]
         of the first closure, from which [let_closures] projects the others;
         see [keep_regions]. *)
      match e, local_region mode with
      | Let (var, kind, closure, projections), Some region when !keep_regions ->
        Model.Let
          ( var,
            kind,
            closure,
            Model.Let (var, kind, local_allocation var region, projections) )
      | _, Some _ when !keep_regions ->
        regions_met := true;
        unsupported "the local set of closures %a, which binds nothing"
          Bound_pattern.print pattern
      | _, _ -> e)
    | None -> unsupported "the binding %a" Bound_pattern.print pattern)
  | Static bound_static, Static_consts group ->
    let bindings =
      Flambda.Static_const_group.match_against_bound_static group bound_static
        ~init:[]
        ~code:(fun bindings code_id code ->
          (match Code0.newer_version_of code with
          | Some older -> newer_versions := (code_id, older) :: !newer_versions
          | None -> ());
          if Loopify_attribute.should_loopify (Code0.loopify code)
          then loopify_marked := code_id :: !loopify_marked;
          let id = Code_ids.find code_id in
          Model.Bind_code (id, convert_code code) :: bindings)
        ~deleted_code:(fun bindings code_id ->
          deleted_code := code_id :: !deleted_code;
          if List.exists (Code_id.equal code_id) !superseded_code
          then bindings
          else Model.Bind_deleted_code (Code_ids.find code_id) :: bindings)
        ~set_of_closures:(fun bindings ~closure_symbols set ->
          let function_decls, slots = convert_set_of_closures set in
          let closure_symbols =
            List.map
              (fun (function_slot, symbol) ->
                Function_slots.find function_slot, Symbols.find symbol)
              (Function_slot.Lmap.bindings closure_symbols)
          in
          if
            not
              (List.equal ( = )
                 (List.map fst closure_symbols)
                 (List.map fst function_decls))
          then unsupported "the binding %a" Bound_pattern.print pattern;
          List.rev_append
            (Model.bind_closures closure_symbols function_decls slots)
            bindings)
        ~block_like:(fun bindings symbol const ->
          Model.Bind_symbol (Symbols.find symbol, convert_static_const const)
          :: bindings)
    in
    Model.Let_static (List.rev bindings, convert_expr body)
  | Singleton bound_var, Rec_info _ ->
    (* CR mshinwell: the binding of a depth variable is dropped; see
       [depth_variables]. *)
    drop_depth_variable (Bound_var.var bound_var);
    convert_expr body
  | Singleton _, (Set_of_closures _ | Static_consts _)
  | Set_of_closures _, _
  | Static _, _ ->
    unsupported "the binding %a" Bound_pattern.print pattern

and convert_prim (prim : P.t) : Model.named =
  match prim with
  | Unary (Project_value_slot { project_from; value_slot }, closure) ->
    Project_value_slot_from
      ( Function_slots.find project_from,
        find_value_slot value_slot,
        convert_simple closure )
  | Unary (Project_function_slot { move_from; move_to }, closure) ->
    Project_function_slot
      ( Function_slots.find move_from,
        Function_slots.find move_to,
        convert_simple closure )
  | Unary (p, arg) -> (
    match convert_unary_primitive p with
    | Some p -> Prim (Unary p, [convert_simple arg])
    | None -> unsupported "primitive %a" P.print prim)
  | Binary (p, arg1, arg2) -> (
    match convert_binary_primitive p with
    | Some p -> Prim (Binary p, [convert_simple arg1; convert_simple arg2])
    | None -> unsupported "primitive %a" P.print prim)
  | Variadic (p, args) -> (
    match convert_variadic_primitive p ~num_args:(List.length args) with
    | Some p -> Prim (Variadic p, convert_simples args)
    | None -> unsupported "primitive %a" P.print prim)
  | Nullary _ | Ternary _ | Quaternary _ ->
    unsupported "primitive %a" P.print prim

and convert_or_variable : type a.
    (a -> Model.const) -> a Or_variable.t -> Model.simple =
 fun convert contents ->
  match contents with
  | Const c -> Const (convert c)
  | Var (var, _) -> Var (convert_variable var)

(* A static block of the model is a tag and fields that are values, so a block
   with a mixed shape, whose flat suffix holds naked numbers, is not in the
   model.

   The empty array of values is the model's immutable array with no element. Its
   length is 0 and no load or store is within its bounds, so that the compiler's
   is mutable and the model's is not changes no run that keeps to the bounds: a
   load from it has no value in the model, and a store into an array is not
   converted. The model does not say that [[||] == [||]]: [Phys_equal] has no
   value on two allocated values, these as any others
   (FlambdaPrimitiveSpec.eval_phys_equal), so a run that compares them is
   undefined, where the compiler makes one symbol of the empty arrays of a unit
   and answers that they are the same. The compiler's constant is also the empty
   array of floats; no primitive of arrays of floats is converted, so nothing
   reads it as one. An empty array of naked numbers of another kind or of
   unboxed products is not in the model. *)
and convert_static_const (const : Static_const.t) : Model.static_const =
  match const with
  | Block (tag, Immutable, Value_only, fields) ->
    Static_const_block
      ( z_of_int (Tag.Scannable.to_int tag),
        List.map
          (fun field -> convert_simple (Simple.With_debuginfo.simple field))
          fields )
  | Immutable_value_array fields ->
    Static_const_immutable_value_array
      (List.map
         (fun field -> convert_simple (Simple.With_debuginfo.simple field))
         fields)
  | Empty_array Values_or_immediates_or_naked_floats ->
    Static_const_immutable_value_array []
  | Immutable_string s ->
    Static_const_immutable_string
      (List.init (String.length s) (fun i -> z_of_int (Char.code s.[i])))
  | Boxed_float32 contents ->
    Static_const_boxed_number (convert_or_variable convert_float32 contents)
  | Boxed_float contents ->
    Static_const_boxed_number (convert_or_variable convert_float contents)
  | Boxed_int32 contents ->
    Static_const_boxed_number
      (convert_or_variable
         (fun i : Model.const -> Naked_int32 (z_of_int64 (Int64.of_int32 i)))
         contents)
  | Boxed_int64 contents ->
    Static_const_boxed_number
      (convert_or_variable
         (fun i : Model.const -> Naked_int64 (z_of_int64 i))
         contents)
  | Boxed_nativeint contents ->
    Static_const_boxed_number
      (convert_or_variable
         (fun i : Model.const ->
           Naked_nativeint (z_of_int64 (Targetint_32_64.to_int64 i)))
         contents)
  | _ -> unsupported "static constant %a" Static_const.print const

and convert_code (code : Flambda.Function_params_and_body.t Code0.t) :
    Model.code =
  (* A call of unknown arity gives a tupled function the tuple as its first
     argument, whatever the arity of its code: the model's code says whether it
     is tupled. *)
  let is_tupled = Code0.is_tupled code in
  if is_tupled then tupled_code := Code0.code_id code :: !tupled_code;
  Flambda.Function_params_and_body.pattern_match (Code0.params_and_body code)
    ~f:(fun
        ~return_continuation
        ~exn_continuation
        params
        ~body
        ~my_closure
        ~is_my_closure_used:_
        ~my_alloc_mode
        ~my_depth
        ~free_names_of_body:_
      ->
      (* CR mshinwell: the depth of the code is dropped; see
         [depth_variables]. *)
      drop_depth_variable my_depth;
      (* CR mshinwell: the regions of the code are dropped; see
         [region_variables]. *)
      (match (my_alloc_mode : Alloc_mode.For_applications.t) with
      | Not_alloc_stack { alloc_region } -> drop_region_variable alloc_region
      | Maybe_alloc_stack { alloc_region; region; ghost_region } ->
        drop_region_variable alloc_region;
        drop_region_variable region;
        drop_region_variable ghost_region;
        (* For the machine of regions; see [keep_regions]. *)
        if !keep_regions
        then code_regions := Variable.Set.add region !code_regions);
      (* The arity that calls are compared with is that of the metadata of the
         code; unarized, it must give the kinds of the parameters. *)
      let params_arity = convert_arity (Code0.params_arity code) in
      let result_arity = convert_arity (Code0.result_arity code) in
      if Model.unarize params_arity <> convert_params_kinds params
      then unsupported "code whose parameters are not those of its arity";
      Model.Code
        ( convert_params params,
          Variables.find my_closure,
          Continuations.find return_continuation,
          Continuations.find exn_continuation,
          (let metadata =
             if is_tupled
             then Model.make_tupled_code_metadata params_arity result_arity
             else Model.make_code_metadata params_arity result_arity
           in
           if Code0.stub code
           then Model.stub_code_metadata metadata
           else metadata),
          convert_expr body ))

and convert_apply (apply : Apply_expr.t) : Model.expr =
  (* The program runs the handler of a probe only while the probe is enabled,
     and the model has nothing of that: a call that is a probe is refused. The
     text of a term does not say that a call is a probe, so no unit read from
     its text has one. A body taken from a .cmx may have one; it is then not
     bound, as any body that the conversion refuses ([make_prefix]). *)
  (match Apply_expr.probe apply with
  | None -> ()
  | Some _ -> unsupported "a call that is a probe");
  let callee =
    match Apply_expr.callee apply with
    | Some callee -> convert_simple callee
    | None -> unsupported "a call with no callee"
  in
  let args_arity = convert_arity (Apply_expr.args_arity apply) in
  let return_arity = convert_arity (Apply_expr.return_arity apply) in
  let call_kind : Model.call_kind =
    match Apply_expr.call_kind apply with
    | Function { function_call = Direct code_id; _ } ->
      directly_called_code := code_id :: !directly_called_code;
      Direct (Code_ids.find code_id)
    | Function { function_call = Indirect_known_arity _; _ } ->
      known_arity_call_met := true;
      Indirect
    | Function { function_call = Indirect_unknown_arity; _ } -> Indirect
    | Method _ | C_call _ | Effect _ ->
      unsupported "the call %a" Apply_expr.print apply
  in
  let exn_continuation = Apply_expr.exn_continuation apply in
  (* See [region_params]. *)
  if
    has_region_param (Exn_continuation.exn_handler exn_continuation)
    ||
    match Apply_expr.continuation apply with
    | Return k -> has_region_param k
    | Never_returns -> false
  then
    unsupported
      "a call that returns or raises to a continuation with a parameter that \
       is a region";
  let returning_to continuation : Model.apply_expr =
    { ap_callee = callee;
      ap_call_kind = call_kind;
      ap_args = convert_simples (Apply_expr.args apply);
      ap_continuation = continuation;
      ap_exn_continuation =
        Continuations.find (Exn_continuation.exn_handler exn_continuation);
      ap_exn_extra_args =
        List.map
          (fun (simple, kind) ->
            ( convert_simple simple,
              convert_kind (Flambda_kind.With_subkind.kind kind) ))
          (Exn_continuation.extra_args exn_continuation);
      ap_args_arity = args_arity;
      ap_return_arity = return_arity
    }
  in
  match Apply_expr.continuation apply with
  | Return k -> Model.Apply (returning_to (Continuations.find k))
  | Never_returns ->
    (* CR mshinwell: a call that never returns, of whatever kind that is
       converted (a method call, a C call and an effect are refused above), is
       made a call that returns to a continuation bound around it and used
       nowhere else, whose handler is [Invalid]: a return from the call is
       undefined. The call keeps its return arity, and the parameters of the
       continuation are of the kinds of that arity. The continuation has a
       number that no continuation of the two units of the check has, and its
       parameters have numbers that no variable of them has. See the header. *)
    let k = Continuations.fresh () in
    let kinds = Model.unarize return_arity in
    let params = List.map (fun _ -> Variables.fresh ()) kinds in
    let handler =
      Model.Handler (params, Model.make_handler_metadata kinds, Model.Invalid)
    in
    Model.Let_cont_non_recursive (k, handler, Model.Apply (returning_to k))

(* ---- Running ---------------------------------------------------------- *)

(* What is observed of a value: what it is, through the symbols it points to. A
   closure is not looked into: its code differs before and after. A partial
   application is observed as a closure is: Simplify makes a closure of one
   whose function it knows. An immutable array that a symbol is bound to is
   observed by its elements, and is not a block: the two runs do not agree by an
   array in one and a block of tag 0 in the other. *)
type observation =
  | Tagged of int
  | Naked of string * int64
  | Null
  | Block of int * observation list
  | Array of observation list
  | Boxed of observation
  | String of string
  | Function
  | Dangling
  | Imported of int
  | Location of int
  | Too_deep

let rec observe depth (statics : Values.static_env) (v : Values.value) :
    observation =
  if depth = 0
  then Too_deep
  else
    match v with
    | Value_const (Tagged_immediate i) -> Tagged (int_of_z i)
    | Value_const (Naked_immediate i) -> Naked ("imm", int64_of_z i)
    | Value_const (Naked_int8 i) -> Naked ("int8", int64_of_z i)
    | Value_const (Naked_int16 i) -> Naked ("int16", int64_of_z i)
    | Value_const (Naked_int32 i) -> Naked ("int32", int64_of_z i)
    | Value_const (Naked_int64 i) -> Naked ("int64", int64_of_z i)
    | Value_const (Naked_nativeint i) -> Naked ("nativeint", int64_of_z i)
    | Value_const (Naked_float32 bits) -> Naked ("float32", int64_of_z bits)
    | Value_const (Naked_float bits) -> Naked ("float", int64_of_z bits)
    | Value_const Null -> Null
    | Value_const _ -> Naked ("number", 0L)
    | Value_closure _ | Value_closure_of_set _ | Value_partial_application _ ->
      Function
    | Value_block (tag, fields) ->
      Block (int_of_z tag, List.map (observe (depth - 1) statics) fields)
    | Value_immutable_array elements ->
      Array (List.map (observe (depth - 1) statics) elements)
    | Value_boxed_number c ->
      Boxed (observe (depth - 1) statics (Values.Value_const c))
    | Value_location l -> Location (int_of_positive l)
    | Value_symbol symbol -> (
      match Maps.lookup symbol statics.static_values with
      | None -> Dangling
      | Some (Static_closure _ | Static_closure_of_set _) -> Function
      | Some (Static_block (tag, fields)) ->
        Block (int_of_z tag, List.map (observe (depth - 1) statics) fields)
      | Some (Static_boxed_number v) -> Boxed (observe (depth - 1) statics v)
      | Some (Static_immutable_array elements) ->
        Array (List.map (observe (depth - 1) statics) elements)
      | Some (Static_string bytes) ->
        String
          (String.init (List.length bytes) (fun i ->
               Char.chr (int_of_z (List.nth bytes i))))
      | Some (Static_imported _) -> Imported (int_of_positive symbol))

(* The name of the symbol of a number, for printing an import. *)
let imported_name n =
  match
    List.find_opt
      (fun (_, number) -> int_of_positive number = n)
      (Symbols.bindings ())
  with
  | Some (symbol, _) -> Linkage_name.to_string (Symbol.linkage_name symbol)
  | None -> string_of_int n

let rec print_observation ppf (o : observation) =
  match o with
  | Tagged i -> Format.fprintf ppf "%d" i
  | Naked (kind, i) -> Format.fprintf ppf "#%Ld:%s" i kind
  | Null -> Format.fprintf ppf "null"
  | Block (tag, fields) ->
    Format.fprintf ppf "@[<hov 1>[%d:@ %a]@]" tag
      (Format.pp_print_list
         ~pp_sep:(fun ppf () -> Format.fprintf ppf ";@ ")
         print_observation)
      fields
  | Array elements ->
    Format.fprintf ppf "@[<hov 2>[|%a|]@]"
      (Format.pp_print_list
         ~pp_sep:(fun ppf () -> Format.fprintf ppf ";@ ")
         print_observation)
      elements
  | Boxed o -> Format.fprintf ppf "@[<hov 1>(boxed@ %a)@]" print_observation o
  | String s -> Format.fprintf ppf "%S" s
  | Function -> Format.fprintf ppf "<fun>"
  | Dangling -> Format.fprintf ppf "<unbound symbol>"
  | Imported n -> Format.fprintf ppf "<imported %s>" (imported_name n)
  | Location n -> Format.fprintf ppf "<location %d>" n
  | Too_deep -> Format.fprintf ppf "..."

type outcome =
  | Returns of observation list
  | Raises of observation list
  | Exits_elsewhere
  | Undefined
  | Still_running

let print_outcome ppf (outcome : outcome) =
  let print_values ppf values =
    Format.pp_print_list
      ~pp_sep:(fun ppf () -> Format.fprintf ppf ",@ ")
      print_observation ppf values
  in
  match outcome with
  | Returns values ->
    Format.fprintf ppf "@[<hov 2>returns@ %a@]" print_values values
  | Raises values ->
    Format.fprintf ppf "@[<hov 2>raises@ %a@]" print_values values
  | Exits_elsewhere -> Format.fprintf ppf "applies an unbound continuation"
  | Undefined -> Format.fprintf ppf "undefined"
  | Still_running -> Format.fprintf ppf "still running when the fuel ran out"

let fuel = 1_000_000

(* The calls that are not in code: those of the unit itself, which inlining
   removes. *)
let rec calls_outside_code (e : Model.expr) =
  let in_handler (Model.Handler (_, _, handler)) = calls_outside_code handler in
  match e with
  | Let (_, _, _, body) | Let_static (_, body) -> calls_outside_code body
  | Let_cont_non_recursive (_, handler, body) ->
    in_handler handler + calls_outside_code body
  | Let_cont_recursive (handlers, body) ->
    List.fold_left
      (fun calls (_, handler) -> calls + in_handler handler)
      (calls_outside_code body) handlers
  | Apply _ -> 1
  | Apply_cont _ | Switch _ | Invalid -> 0

(* The code that calls itself, at once or through other code, by direct calls.
   The validation is told not to inline the calls of such code
   ([FlambdaNormaliseSpec.inline_calls]): Simplify leaves them, and to inline
   one is to leave another to inline in the next round. This is a guess that is
   not trusted: the validation is sound whatever code it is told to keep. *)
let rec direct_callees (e : Model.expr) : Model.code_id list =
  let in_handler (Model.Handler (_, _, handler)) = direct_callees handler in
  match e with
  | Let (_, _, _, body) | Let_static (_, body) -> direct_callees body
  | Let_cont_non_recursive (_, handler, body) ->
    in_handler handler @ direct_callees body
  | Let_cont_recursive (handlers, body) ->
    List.concat_map (fun (_, handler) -> in_handler handler) handlers
    @ direct_callees body
  | Apply { ap_call_kind = Direct code; _ } -> [code]
  | Apply { ap_call_kind = Indirect; _ } | Apply_cont _ | Switch _ | Invalid ->
    []

(* Every piece of code that an expression binds, with its body. *)
let rec code_bound (e : Model.expr) : (Model.code_id * Model.expr) list =
  let in_handler (Model.Handler (_, _, handler)) = code_bound handler in
  match e with
  | Let (_, _, _, body) -> code_bound body
  | Let_static (bindings, body) ->
    List.concat_map
      (fun (binding : Model.static_binding) ->
        match binding with
        | Bind_code (id, Code (_, _, _, _, _, body)) ->
          (id, body) :: code_bound body
        | Bind_deleted_code _ | Bind_symbol _ -> [])
      bindings
    @ code_bound body
  | Let_cont_non_recursive (_, handler, body) ->
    in_handler handler @ code_bound body
  | Let_cont_recursive (handlers, body) ->
    List.concat_map (fun (_, handler) -> in_handler handler) handlers
    @ code_bound body
  | Apply _ | Apply_cont _ | Switch _ | Invalid -> []

(* Whether an expression has, in the code that it binds too, a [Let] of a
   closure at a function slot of a set. *)
let rec lets_closure_of_set (e : Model.expr) : bool =
  let in_handler (Model.Handler (_, _, handler)) =
    lets_closure_of_set handler
  in
  match e with
  | Let (_, _, Closure_of_set _, _) -> true
  | Let (_, _, _, body) -> lets_closure_of_set body
  | Let_static (bindings, body) ->
    List.exists
      (fun (binding : Model.static_binding) ->
        match binding with
        | Bind_code (_, Code (_, _, _, _, _, body)) -> lets_closure_of_set body
        | Bind_deleted_code _ | Bind_symbol _ -> false)
      bindings
    || lets_closure_of_set body
  | Let_cont_non_recursive (_, handler, body) ->
    in_handler handler || lets_closure_of_set body
  | Let_cont_recursive (handlers, body) ->
    List.exists (fun (_, handler) -> in_handler handler) handlers
    || lets_closure_of_set body
  | Apply _ | Apply_cont _ | Switch _ | Invalid -> false

(* The code that an expression binds and that is a stub. *)
let rec stub_code (e : Model.expr) : Model.code_id list =
  let in_handler (Model.Handler (_, _, handler)) = stub_code handler in
  match e with
  | Let (_, _, _, body) -> stub_code body
  | Let_static (bindings, body) ->
    List.concat_map
      (fun (binding : Model.static_binding) ->
        match binding with
        | Bind_code (id, Code (_, _, _, _, metadata, _)) ->
          if metadata.cm_is_stub then [id] else []
        | Bind_deleted_code _ | Bind_symbol _ -> [])
      bindings
    @ stub_code body
  | Let_cont_non_recursive (_, handler, body) ->
    in_handler handler @ stub_code body
  | Let_cont_recursive (handlers, body) ->
    List.concat_map (fun (_, handler) -> in_handler handler) handlers
    @ stub_code body
  | Apply _ | Apply_cont _ | Switch _ | Invalid -> []

let recursive_code (e : Model.expr) : Model.code_id list =
  let calls =
    List.map (fun (id, body) -> id, direct_callees body) (code_bound e)
  in
  let reaches_itself id =
    let seen = ref [] in
    let rec from caller =
      match List.assoc_opt caller calls with
      | None -> false
      | Some callees ->
        List.exists
          (fun callee ->
            callee = id
            || (not (List.mem callee !seen))
               &&
               (seen := callee :: !seen;
                from callee))
          callees
    in
    from id
  in
  List.filter reaches_itself (List.map fst calls)

(* The code that an expression calls directly, in the code it binds too. *)
let called_directly (e : Model.expr) : Model.code_id list =
  direct_callees e
  @ List.concat_map (fun (_, body) -> direct_callees body) (code_bound e)

type converted =
  { body : Model.expr;
    return_continuation : Model.continuation;
    exn_continuation : Model.continuation
  }

(* The model's calls give tupled code a tuple. The compiler's direct call, and
   its call of known arity, give code its arguments as the code takes them: for
   tupled code that is another call, which the model has not got. A direct call
   names its code. A call of known arity does not, so it is refused wherever the
   unit has tupled code at all. Code that the unit does not bind cannot be run
   by the model, tupled or not, so a direct call of it stays. *)
let check_calls_of_tupled_code () =
  let is_tupled code_id = List.exists (Code_id.equal code_id) !tupled_code in
  if List.exists is_tupled !directly_called_code
  then unsupported "a direct call of a tupled function";
  match !tupled_code with
  | [] -> ()
  | _ :: _ ->
    if !known_arity_call_met
    then
      unsupported "a call of known arity in a unit that has a tupled function"

let convert_unit (unit : Flambda_unit.t) : converted =
  tupled_code := [];
  directly_called_code := [];
  known_arity_call_met := false;
  (* CR mshinwell: the region of the unit is dropped; see [region_variables]. *)
  drop_region_variable (Flambda_unit.toplevel_my_alloc_region unit);
  let body = convert_expr (Flambda_unit.body unit) in
  check_calls_of_tupled_code ();
  { body;
    return_continuation =
      Continuations.find (Flambda_unit.return_continuation unit);
    exn_continuation = Continuations.find (Flambda_unit.exn_continuation unit)
  }

(* What other units read of a unit is the data of its own symbol
   ([Flambda_unit.module_symbol]); the backend does not use what the unit gives
   its return continuation, whose handler returns the unit value
   ([To_cmm.unit0]). The model observes what the unit gives its return
   continuation ([run]). So that the observation begins at the symbol that other
   units read, a unit is checked only if, as it is written, every use of its
   return continuation gives it that symbol and nothing else, and no call
   returns to it ([check]). This is a test of the form of the term, and enough
   for what is wanted: a unit that fails it may still return its own symbol,
   through a variable or from a call, and is not checked all the same. What is
   given back is the reason, for a unit that fails. Code is not looked into: it
   cannot name the return continuation of the unit. The test is of a unit as the
   compiler has it and not of a term of the model read back from a text, which
   has no symbol of its own, so it is not part of [convert_unit]. *)
let return_not_of_own_symbol (unit : Flambda_unit.t) (converted : converted) =
  let return = converted.return_continuation in
  (* The number of the symbol, if the conversion has met it. *)
  let own =
    List.find_map
      (fun (symbol, number) ->
        if Symbol.equal symbol (Flambda_unit.module_symbol unit)
        then Some number
        else None)
      (Symbols.bindings ())
  in
  let other_than_own =
    Some
      "a use of the return continuation of the unit gives it something other \
       than the symbol of the unit alone"
  in
  let in_apply_cont (ac : Model.apply_cont) =
    if ac.ac_continuation <> return
    then None
    else
      match ac.ac_args, own with
      | [Symbol symbol], Some own when symbol = own -> None
      | _, _ -> other_than_own
  in
  let first_of reasons = List.find_map (fun reason -> reason) reasons in
  let rec in_expr (e : Model.expr) =
    match e with
    | Let (_, _, _, body) | Let_static (_, body) -> in_expr body
    | Let_cont_non_recursive (_, handler, body) ->
      first_of [in_handler handler; in_expr body]
    | Let_cont_recursive (handlers, body) ->
      first_of
        (List.map (fun (_, handler) -> in_handler handler) handlers
        @ [in_expr body])
    | Apply ap ->
      if ap.ap_continuation <> return
      then None
      else Some "a call returns to the return continuation of the unit"
    | Apply_cont ac -> in_apply_cont ac
    | Switch (_, arms) ->
      first_of (List.map (fun (_, ac) -> in_apply_cont ac) arms)
    | Invalid -> None
  and in_handler (Model.Handler (_, _, handler)) = in_expr handler in
  in_expr converted.body

let run ~(statics : Values.static_env) (unit : converted) : outcome * int =
  let rec go steps (state : Semantics.state) =
    if steps >= fuel
    then Still_running, steps
    else
      match Semantics.step state with
      | Step state -> go (steps + 1) state
      | Undefined -> Undefined, steps
      | Exit (k, values, _heap) ->
        let values = List.map (observe 50 state.state_statics) values in
        ( (if k = unit.return_continuation
           then Returns values
           else if k = unit.exn_continuation
           then Raises values
           else Exits_elsewhere),
          steps )
  in
  go 0
    { state_expr = unit.body;
      state_vars = [];
      state_conts = [];
      state_statics = statics;
      state_heap = Values.empty_heap
    }

(* The external symbols that the units converted so far mention, each with its
   class: the symbols of another compilation unit, the predefined exceptions
   among them. The test is the compiler's own of a symbol that it looks up in
   another unit ([Typing_env]). A symbol of the unit being checked that the unit
   does not define is not one, and stays bound to nothing. The two units of a
   check are converted with one numbering of symbols, so this is one list for
   both. A predefined exception is an identity, and anything else is data: the
   conversion cannot tell an exception that another unit declares from data. A
   symbol that the prefix binds is not an import, and nor is one that got its
   number from a body which the prefix did not bind in the end ([~bound],
   [~unmentioned]; see [make_prefix]). A symbol that the prefix made for a field
   of which the files say nothing ([~hidden]) is an import of which nothing is
   known. *)
let imports ~(bound : Model.symbol list) ~(unmentioned : Model.symbol list)
    ~(hidden : Model.symbol list) : (Model.symbol * Values.import_class) list =
  let left_out : (int, unit) Hashtbl.t = Hashtbl.create 64 in
  List.iter
    (fun number -> Hashtbl.replace left_out (int_of_positive number) ())
    (bound @ unmentioned);
  List.filter_map
    (fun (symbol, number) ->
      if
        Current_unit.is_current (Symbol.compilation_unit symbol)
        || Hashtbl.mem left_out (int_of_positive number)
      then None
      else
        Some
          ( number,
            if Symbol.is_predefined_exception symbol
            then Values.Import_identity
            else Values.Import_data ))
    (Symbols.bindings ())
  @ List.map (fun symbol -> symbol, Values.Import_unknown) hidden

(* ---- Printing model terms ---------------------------------------------- *)

(* For looking at what the normalisation made of the two terms, when an example
   is not validated (set FLSEMANTICS_PRINT_TERMS). *)
let print_list print ppf items =
  Format.pp_print_list
    ~pp_sep:(fun ppf () -> Format.fprintf ppf ",@ ")
    print ppf items

(* A name is printed as its number. Where [names_shown] is set, a symbol, a code
   id and a slot are printed as the compiler's name of them, and a variable and
   a continuation as the number of its first appearance since
   [forget_shown_names]: what is printed then does not depend on the numbers
   that the conversion gave. That is for the prefix ([print_prefix]). *)
let names_shown = ref false

let shown_variables : (int, int) Hashtbl.t = Hashtbl.create 64

let shown_continuations : (int, int) Hashtbl.t = Hashtbl.create 16

let forget_shown_names () =
  Hashtbl.reset shown_variables;
  Hashtbl.reset shown_continuations

let shown_number table n =
  let n = int_of_positive n in
  if not !names_shown
  then n
  else
    match Hashtbl.find_opt table n with
    | Some shown -> shown
    | None ->
      let shown = Hashtbl.length table + 1 in
      Hashtbl.replace table n shown;
      shown

let name_in_unit unit name = Compilation_unit.name_as_string unit ^ "." ^ name

let shown_symbol sym =
  match Symbols.name_of sym with
  | Some symbol when !names_shown ->
    Linkage_name.to_string (Symbol.linkage_name symbol)
  | Some _ | None -> Printf.sprintf "$%d" (int_of_positive sym)

let shown_code_id id =
  match Code_ids.name_of id with
  | Some code_id when !names_shown ->
    Linkage_name.to_string (Code_id.linkage_name code_id)
  | Some _ | None -> Printf.sprintf "code%d" (int_of_positive id)

let shown_function_slot slot =
  match Function_slots.name_of slot with
  | Some function_slot when !names_shown ->
    name_in_unit
      (Function_slot.get_compilation_unit function_slot)
      (Function_slot.to_string function_slot)
  | Some _ | None -> Printf.sprintf "f%d" (int_of_positive slot)

(* A value slot, where it is given a value and where it is projected. *)
let shown_value_slot ~projected slot =
  match Value_slots.name_of slot with
  | Some value_slot when !names_shown ->
    name_in_unit
      (Value_slot.get_compilation_unit value_slot)
      (Value_slot.to_string value_slot)
  | Some _ | None ->
    Printf.sprintf (if projected then "slot%d" else "%d") (int_of_positive slot)

let print_var ppf x = Format.fprintf ppf "v%d" (shown_number shown_variables x)

let print_simple ppf (s : Model.simple) =
  match s with
  | Var x -> print_var ppf x
  | Symbol sym -> Format.pp_print_string ppf (shown_symbol sym)
  | Const (Tagged_immediate i) -> Format.fprintf ppf "%d" (int_of_z i)
  | Const (Naked_immediate i) -> Format.fprintf ppf "#%d" (int_of_z i)
  | Const Null -> Format.fprintf ppf "null"
  | Const _ -> Format.fprintf ppf "<const>"

let print_simples = print_list print_simple

let print_cont ppf k =
  Format.fprintf ppf "k%d" (shown_number shown_continuations k)

let print_value_slots ppf slots =
  print_list
    (fun ppf (slot, s) ->
      Format.fprintf ppf "%s = %a"
        (shown_value_slot ~projected:false slot)
        print_simple s)
    ppf slots

let print_function_decls ppf function_decls =
  print_list
    (fun ppf (slot, code) ->
      Format.fprintf ppf "%s = %s" (shown_function_slot slot)
        (shown_code_id code))
    ppf function_decls

let string_of_standard_int (kind : Kinds.standard_int) =
  match kind with
  | Standard_tagged_immediate -> "tagged_imm"
  | Standard_naked_immediate -> "imm"
  | Standard_naked_int8 -> "int8"
  | Standard_naked_int16 -> "int16"
  | Standard_naked_int32 -> "int32"
  | Standard_naked_int64 -> "int64"
  | Standard_naked_nativeint -> "nativeint"

let string_of_signedness (signedness : Primitives.signedness) =
  match signedness with Signed -> "" | Unsigned -> "unsigned_"

let print_prim ppf (p : Primitives.prim) =
  let unary (p : Primitives.unary_primitive) =
    match p with
    | Block_load (_, mut, field) ->
      Printf.sprintf "Block_load%s %d"
        (match mut with
        | Mutable -> " mut"
        | Immutable -> ""
        | Immutable_unique -> " immut_uniq")
        (int_of_z field)
    | Is_int variant_only -> if variant_only then "Is_int" else "Is_int_generic"
    | Is_null -> "Is_null"
    | Get_tag -> "Get_tag"
    | Array_length _ ->
      (* The converter makes no primitive of arrays. *)
      "Array_length"
    | String_length -> "String_length"
    | Opaque_identity _ -> "Opaque_identity"
    | Unary_int_arith (kind, op) ->
      Printf.sprintf "Int_arith %s %s"
        (string_of_standard_int kind)
        (match op with Not -> "not" | Swap_byte_endianness -> "bswap")
    | Num_conv (src, dst) ->
      Printf.sprintf "Num_conv %s %s"
        (string_of_standard_int src)
        (string_of_standard_int dst)
    | Boolean_not -> "Boolean_not"
    | Unbox_number _ -> "Unbox_number"
    | Box_number _ -> "Box_number"
    | Untag_immediate -> "Untag_immediate"
    | Tag_immediate -> "Tag_immediate"
    | Reinterpret_64_bit_word Tagged_int63_as_unboxed_int64 ->
      "Reinterpret_64_bit_word int63_as_int64"
    | Reinterpret_64_bit_word Unboxed_int64_as_tagged_int63 ->
      "Reinterpret_64_bit_word int64_as_int63"
  in
  let int_arith (op : Primitives.binary_int_arith_op) =
    match op with
    | Add -> "add"
    | Sub -> "sub"
    | Mul -> "mul"
    | Div signedness -> string_of_signedness signedness ^ "div"
    | Mod signedness -> string_of_signedness signedness ^ "mod"
    | And -> "and"
    | Or -> "or"
    | Xor -> "xor"
  in
  let int_comp (behaviour : Primitives.int_comparison_behaviour) =
    match behaviour with
    | Yielding_bool Int_eq -> "eq"
    | Yielding_bool Int_neq -> "neq"
    | Yielding_bool (Int_lt signedness) ->
      string_of_signedness signedness ^ "lt"
    | Yielding_bool (Int_gt signedness) ->
      string_of_signedness signedness ^ "gt"
    | Yielding_bool (Int_le signedness) ->
      string_of_signedness signedness ^ "le"
    | Yielding_bool (Int_ge signedness) ->
      string_of_signedness signedness ^ "ge"
    | Yielding_int_like_compare_functions signedness ->
      string_of_signedness signedness ^ "compare"
  in
  match p with
  | Unary p -> Format.pp_print_string ppf (unary p)
  | Binary (Array_load _) ->
    (* The converter makes no primitive of arrays. *)
    Format.pp_print_string ppf "Array_load"
  | Binary (String_or_bigstring_load width) ->
    Format.fprintf ppf "String_load %s"
      (match width with
      | Eight -> "8"
      | Eight_signed -> "8s"
      | Sixteen -> "16"
      | Sixteen_signed -> "16s"
      | Thirty_two -> "32"
      | Sixty_four -> "64")
  | Binary (Phys_equal Phys_eq) -> Format.pp_print_string ppf "Phys_equal eq"
  | Binary (Phys_equal Phys_neq) -> Format.pp_print_string ppf "Phys_equal neq"
  | Binary (Int_arith (kind, op)) ->
    Format.fprintf ppf "Int_arith %s %s"
      (string_of_standard_int kind)
      (int_arith op)
  | Binary (Int_shift (kind, op)) ->
    Format.fprintf ppf "Int_shift %s %s"
      (string_of_standard_int kind)
      (match op with Lsl -> "lsl" | Lsr -> "lsr" | Asr -> "asr")
  | Binary (Int_comp (kind, behaviour)) ->
    Format.fprintf ppf "Int_comp %s %s"
      (string_of_standard_int kind)
      (int_comp behaviour)
  | Variadic (Make_block (Block_kind_values tag)) ->
    Format.fprintf ppf "Make_block %d" (int_of_z tag)
  | Variadic (Make_immutable_array kind) ->
    Format.fprintf ppf "Make_immutable_array %s"
      (match kind with
      | Array_immediates -> "immediates"
      | Array_gc_ignorable_values -> "gc_ignorable_values"
      | Array_values -> "values")
  | Heap _ ->
    (* The converter makes no primitive of the heap. *)
    Format.pp_print_string ppf "<primitive of the heap>"
  | Region_primitive p ->
    Format.pp_print_string ppf
      (match p with
      | Begin_region -> "Begin_region"
      | Begin_try_region -> "Begin_try_region"
      | End_region -> "End_region"
      | End_try_region -> "End_try_region"
      | Local_allocation -> "Local_allocation")

let print_named ppf (d : Model.named) =
  match d with
  | Simple s -> print_simple ppf s
  | Prim (p, args) ->
    Format.fprintf ppf "@[<hov 2>%a@ (%a)@]" print_prim p print_simples args
  | Set_of_closures (code, slots) ->
    Format.fprintf ppf "@[<hov 2>closure %s@ {%a}@]" (shown_code_id code)
      print_value_slots slots
  | Closure_of_set (slot, function_decls, slots) ->
    Format.fprintf ppf "@[<hov 2>closure %s of@ {%a}@ {%a}@]"
      (shown_function_slot slot) print_function_decls function_decls
      print_value_slots slots
  | Project_value_slot (slot, closure) ->
    Format.fprintf ppf "%a.%s" print_simple closure
      (shown_value_slot ~projected:true slot)
  | Project_value_slot_from (project_from, slot, closure) ->
    Format.fprintf ppf "%a.(%s).%s" print_simple closure
      (shown_function_slot project_from)
      (shown_value_slot ~projected:true slot)
  | Project_function_slot (move_from, move_to, closure) ->
    Format.fprintf ppf "%a.(%s -> %s)" print_simple closure
      (shown_function_slot move_from)
      (shown_function_slot move_to)

let print_trap_action ppf (action : Model.trap_action option) =
  match action with
  | None -> ()
  | Some (Push k) -> Format.fprintf ppf "@ push(%a)" print_cont k
  | Some (Pop (k, kind)) ->
    Format.fprintf ppf "@ pop(%s%a)"
      (match kind with
      | None -> ""
      | Some Regular -> "regular "
      | Some Reraise -> "reraise "
      | Some No_trace -> "notrace ")
      print_cont k

let print_apply_cont ppf (ac : Model.apply_cont) =
  Format.fprintf ppf "@[<hov 2>cont %a%a@ (%a)@]" print_cont ac.ac_continuation
    print_trap_action ac.ac_trap_action print_simples ac.ac_args

let rec print_expr ppf (e : Model.expr) =
  match e with
  | Let (x, _, d, body) ->
    Format.fprintf ppf "@[<hov 2>let %a =@ %a in@]@ %a" print_var x print_named
      d print_expr body
  | Let_static (bindings, body) ->
    Format.fprintf ppf "@[<v 2>let static@ %a@]@ in@ %a"
      (Format.pp_print_list print_binding)
      bindings print_expr body
  | Let_cont_non_recursive (k, handler, body) ->
    Format.fprintf ppf "@[<v 2>let_cont %a %a@]@ in@ %a" print_cont k
      print_handler handler print_expr body
  | Let_cont_recursive (handlers, body) ->
    Format.fprintf ppf "@[<v 2>let_cont rec@ %a@]@ in@ %a"
      (Format.pp_print_list (fun ppf (k, handler) ->
           Format.fprintf ppf "%a %a" print_cont k print_handler handler))
      handlers print_expr body
  | Apply ap ->
    Format.fprintf ppf "@[<hov 2>apply%s %a@ (%a)@ -> %a * %a%a@]"
      (match ap.ap_call_kind with
      | Direct code -> Printf.sprintf " direct(%s)" (shown_code_id code)
      | Indirect -> "")
      print_simple ap.ap_callee print_simples ap.ap_args print_cont
      ap.ap_continuation print_cont ap.ap_exn_continuation
      (fun ppf extra ->
        match extra with
        | [] -> ()
        | _ :: _ -> Format.fprintf ppf "@ (%a)" print_simples extra)
      (List.map fst ap.ap_exn_extra_args)
  | Apply_cont ac -> print_apply_cont ppf ac
  | Switch (scrutinee, arms) ->
    Format.fprintf ppf "@[<v 2>switch %a@ %a@]" print_simple scrutinee
      (Format.pp_print_list (fun ppf (i, ac) ->
           Format.fprintf ppf "| %d -> %a" (int_of_z i) print_apply_cont ac))
      arms
  | Invalid -> Format.fprintf ppf "invalid"

and print_handler ppf (Model.Handler (params, metadata, handler)) =
  Format.fprintf ppf "%s(%a) =@ @[<v>%a@]"
    (if metadata.hm_is_exn_handler then "exn " else "")
    (print_list print_var) params print_expr handler

and print_binding ppf (binding : Model.static_binding) =
  match binding with
  | Bind_code (id, Code (params, my_closure, ret, exn, metadata, body)) ->
    Format.fprintf ppf "@[<v 2>%s%s%s (%a) %a -> %a * %a =@ %a@]"
      (shown_code_id id)
      (if metadata.cm_is_tupled then " tupled" else "")
      (if metadata.cm_is_stub then " stub" else "")
      (print_list print_var) params print_var my_closure print_cont ret
      print_cont exn print_expr body
  | Bind_deleted_code id -> Format.fprintf ppf "%s deleted" (shown_code_id id)
  | Bind_symbol (sym, Static_const_set_of_closures (code, slots)) ->
    Format.fprintf ppf "@[<hov 2>%s = closure %s@ {%a}@]" (shown_symbol sym)
      (shown_code_id code) print_value_slots slots
  | Bind_symbol
      (sym, Static_const_closure_of_set (slot, symbols, function_decls, slots))
    ->
    Format.fprintf ppf "@[<hov 2>%s = closure %s of@ {%a}@ at {%a}@ {%a}@]"
      (shown_symbol sym) (shown_function_slot slot) print_function_decls
      function_decls
      (print_list (fun ppf (slot, sym) ->
           Format.fprintf ppf "%s = %s" (shown_function_slot slot)
             (shown_symbol sym)))
      symbols print_value_slots slots
  | Bind_symbol (sym, Static_const_block (tag, fields)) ->
    Format.fprintf ppf "@[<hov 2>%s = Block %d@ (%a)@]" (shown_symbol sym)
      (int_of_z tag) print_simples fields
  | Bind_symbol (sym, Static_const_boxed_number contents) ->
    Format.fprintf ppf "%s = boxed %a" (shown_symbol sym) print_simple contents
  | Bind_symbol (sym, Static_const_immutable_string _) ->
    Format.fprintf ppf "%s = <string>" (shown_symbol sym)
  | Bind_symbol (sym, Static_const_immutable_value_array elements) ->
    Format.fprintf ppf "@[<hov 2>%s = Array@ (%a)@]" (shown_symbol sym)
      print_simples elements

(* ---- Static bindings that are not used --------------------------------- *)

(* The symbols and code ids that a term binds and that nothing live refers to:
   not the term outside its static bindings, and not a static binding that is
   itself live. This is only a guess of what may be removed:
   FlambdaUnusedStaticsSpec.drop_statics checks it before removing anything. *)
type static_name =
  | Symbol_name of Model.symbol
  | Code_name of Model.code_id

let static_names (e : Model.expr) :
    (static_name option * static_name) list * static_name list =
  (* What each binding refers to, and what the term refers to outside its
     bindings ([None]). *)
  let references : (static_name option * static_name) list ref = ref [] in
  let bound : static_name list ref = ref [] in
  let refer owner name = references := (owner, name) :: !references in
  let in_simple owner (s : Model.simple) =
    match s with
    | Symbol sym -> refer owner (Symbol_name sym)
    | Var _ | Const _ -> ()
  in
  let in_simples owner = List.iter (in_simple owner) in
  let in_slots owner = List.iter (fun (_, s) -> in_simple owner s) in
  let rec in_expr owner (e : Model.expr) =
    match e with
    | Let (_, _, d, body) ->
      (match d with
      | Simple s
      | Project_value_slot (_, s)
      | Project_value_slot_from (_, _, s)
      | Project_function_slot (_, _, s) ->
        in_simple owner s
      | Prim (_, args) -> in_simples owner args
      | Set_of_closures (code, slots) ->
        refer owner (Code_name code);
        in_slots owner slots
      | Closure_of_set (_, decls, slots) ->
        List.iter (fun (_, code) -> refer owner (Code_name code)) decls;
        in_slots owner slots);
      in_expr owner body
    | Let_static (bindings, body) ->
      List.iter
        (fun (binding : Model.static_binding) ->
          match binding with
          | Bind_code (id, Code (_, _, _, _, _, code_body)) ->
            bound := Code_name id :: !bound;
            in_expr (Some (Code_name id)) code_body
          | Bind_deleted_code _ -> ()
          | Bind_symbol (sym, const) -> (
            bound := Symbol_name sym :: !bound;
            let owner = Some (Symbol_name sym) in
            match const with
            | Static_const_set_of_closures (code, slots) ->
              refer owner (Code_name code);
              in_slots owner slots
            | Static_const_closure_of_set (_, symbols, function_decls, slots) ->
              List.iter (fun (_, sym) -> refer owner (Symbol_name sym)) symbols;
              List.iter
                (fun (_, code) -> refer owner (Code_name code))
                function_decls;
              in_slots owner slots
            | Static_const_block (_, fields) -> in_simples owner fields
            | Static_const_boxed_number contents -> in_simple owner contents
            | Static_const_immutable_value_array elements ->
              in_simples owner elements
            | Static_const_immutable_string _ -> ()))
        bindings;
      in_expr owner body
    | Let_cont_non_recursive (_, Handler (_, _, handler), body) ->
      in_expr owner handler;
      in_expr owner body
    | Let_cont_recursive (handlers, body) ->
      List.iter
        (fun (_, Model.Handler (_, _, handler)) -> in_expr owner handler)
        handlers;
      in_expr owner body
    | Apply ap -> (
      in_simple owner ap.ap_callee;
      in_simples owner ap.ap_args;
      in_simples owner (List.map fst ap.ap_exn_extra_args);
      match ap.ap_call_kind with
      | Direct code -> refer owner (Code_name code)
      | Indirect -> ())
    | Apply_cont ac -> in_simples owner ac.ac_args
    | Switch (scrutinee, arms) ->
      in_simple owner scrutinee;
      List.iter
        (fun (_, (ac : Model.apply_cont)) -> in_simples owner ac.ac_args)
        arms
    | Invalid -> ()
  in
  in_expr None e;
  !references, !bound

let unused_statics (e : Model.expr) : Unused_statics.dropped =
  let references, bound = static_names e in
  let rec live_from live =
    let live' =
      List.fold_left
        (fun live (owner, name) ->
          let owner_is_live =
            match owner with None -> true | Some owner -> List.mem owner live
          in
          if owner_is_live && not (List.mem name live)
          then name :: live
          else live)
        live references
    in
    if List.length live' = List.length live then live else live_from live'
  in
  let live = live_from [] in
  let unused = List.filter (fun name -> not (List.mem name live)) bound in
  { dropped_symbols =
      List.filter_map
        (function Symbol_name sym -> Some sym | Code_name _ -> None)
        unused;
    dropped_code =
      List.filter_map
        (function Code_name id -> Some id | Symbol_name _ -> None)
        unused
  }

(* ---- Lets that Simplify made static bindings --------------------------- *)

(* Which [Let]s of the term before to make static bindings, and of which
   symbols. This is only a guess, read off the term after:
   FlambdaLiftSpec.lift_unit checks what it needs of it, and the comparison of
   the two terms the rest. The symbols to give are those that the term after
   binds and the term before does not. Each goes to the first [Let] that makes
   what the symbol is bound to: a closure of the same code, a block of the same
   tag whose fields agree, an immutable array of as many elements that agree, a
   boxed number. Two fields agree if the first is a variable of which nothing is
   known, or if they are the same symbol or constant; a variable bound to a
   symbol or a constant, or whose own [Let] is lifted here, counts as that
   symbol or constant. A [Let] one of whose parts is a variable of which nothing
   is known does not take a symbol bound to a constant that some [Let] of the
   term makes of symbols and constants alone: the symbol is left for that one. A
   [Let] is named by its path, as FlambdaLiftSpec.lift_expr counts it. A symbol
   is not given to a [Let] that [banned] pairs it with: the caller asks again,
   with the pair banned, where a symbol went to a [Let] whose block nothing uses
   in the end.

   In a handler that is not at toplevel
   (FlambdaWellFormedSpec.handler_at_toplevel: the arm of a test, say) and in
   the handler of a recursive continuation, where no static binding may stand, a
   symbol goes to the first [Let] that makes just what the symbol is bound to
   and mentions no variable: FlambdaLiftFromArmsSpec.lift_to_head binds it at
   the head of the unit. That puts the unit under a group of static bindings,
   and the validation makes it first, so the paths of the other [Let]s outside
   code are then given as they are under that group.

   A symbol that no [Let] outside code takes goes to the first [Let] in the body
   of code that makes just what the symbol is bound to and mentions no variable,
   in the handler of a recursive continuation too:
   FlambdaLiftFromCodeSpec.lift_from_code lifts it out of the code. In this and
   in the case before it, a closure may have more value slots than the closure
   that the symbol is bound to, which Simplify has dropped: see where the two
   are matched.

   In both of these, a later [Let] that makes just what a symbol given earlier
   in the same place is bound to (at the head of the unit, or in the body of the
   same code) is given that symbol too: where inlining has made two [Let]s of
   one closure that captures nothing, Simplify has one symbol for both, and
   FlambdaLiftFromCodeSpec.once_each puts one binding for the two.

   A closure that no symbol of the term after fits by its code, outside code or
   in one of the two places above, is given a symbol whose set of closures has
   the same function slots, value slots that agree and, for each function whose
   code is not the same, code that the term before does not bind. Simplify gives
   a set of closures new code where it simplifies it; where it has met one set
   twice, in a function that stays and in a copy of the body of that function
   that it inlined, the term after has the code of the term before for the first
   and new code for the second. The guess then asks for a copy of the old code
   under each new code id, for the symbols of that set
   (FlambdaCopyCodeSpec.copy_static_code, the last step of
   FlambdaValidateSpec.with_lifting). Where a function is inlined at two places,
   each of the two closures has new code of its own, and the guess asks for two
   copies of the one code, each for its symbol. Nothing is believed of this
   either: FlambdaCopyCodeSpec.copy_ok checks what a copy needs, and the
   comparison of the two terms whether the copy, once normalised, is the code
   that the term after has.

   A closure outside code that no symbol of the term after fits, whose value
   slots hold symbols, constants or variables that the guess has made aliases of
   symbols, and that the term after does not make by a [Let] either, is given
   symbols that neither of the two terms given here binds or mentions. The case
   aimed at is a function of which Simplify has inlined every call and removed
   the closure; where another closure holds it in a value slot and calls it
   through the slot, the slot then holds a symbol, which the code that reads the
   slot is given (FlambdaKnownClosuresSpec.use_known_closures), and the call is
   direct and can be inlined. Once it is, nothing uses the symbols, and the
   removal of unused static bindings takes them away.

   The guess also says in which code [my_closure] is a symbol: the code of each
   closure that the term after binds statically, if the term before, once
   lifted, makes no other closure of that code
   (FlambdaMyClosureSpec.only_closure_of).

   With [around], it also says which [Let]s outside code to make aliases of
   symbols that a group of static bindings around them binds
   (FlambdaLiftAroundSpec.lift_around_unit), in the term that the liftings above
   give. A [Let] that makes just what a symbol bound around it is bound to, with
   no variable of the constant bound since, is made an alias of that symbol: so
   two [Let]s of one constant, or a [Let] and a constant that the unit binds
   already, are one symbol. And in a handler that is not at toplevel, or the
   handler of a recursive continuation, a symbol that is still to give goes to
   the first [Let] that makes what the symbol is bound to and mentions no
   variable bound in the handler: its binding is put before the [Let_cont] of
   the outermost such handler around the [Let]. A group is given by the paths of
   the [Let]s whose bindings make it. A closure of a set whose function slots
   are named is so treated with the closures of its set: the [Let] is made an
   alias of the symbols of the set (FlambdaLiftSpec.Lift_to_closure_symbols),
   which are given together and bound by the bindings of that one [Let]. This
   part is a guess as the others are: FlambdaLiftAroundSpec.around_ok checks the
   aliases and the groups, and the comparison of the two terms the rest. The
   caller asks for this from the second lifting on, where the term is
   normalised. *)
let lifting_guess ~unit ~(banned : (Lift.path * Model.symbol) list) ~around
    (before : Model.expr) (after : Model.expr) : Lift.lifting =
  (* The symbols bound outside code, in the order in which they are bound when
     the term runs: the body of a [Let_cont] before its handler. *)
  let rec bound_symbols (e : Model.expr) =
    match e with
    | Let (_, _, _, body) -> bound_symbols body
    | Let_static (bindings, body) ->
      List.filter_map
        (fun (binding : Model.static_binding) ->
          match binding with
          | Bind_symbol (sym, const) -> Some (sym, const)
          | Bind_code _ | Bind_deleted_code _ -> None)
        bindings
      @ bound_symbols body
    | Let_cont_non_recursive (_, Handler (_, _, handler), body) ->
      bound_symbols body @ bound_symbols handler
    | Let_cont_recursive (_, body) -> bound_symbols body
    | Apply _ | Apply_cont _ | Switch _ | Invalid -> []
  in
  let bound_before = List.map fst (bound_symbols before) in
  let to_give =
    ref
      (List.filter
         (fun (sym, _) -> not (List.mem sym bound_before))
         (bound_symbols after))
  in
  (* The path of what is at a path in the unit, once the unit is under a group
     of static bindings. *)
  let rec under_group (path : Numbers.positive) : Numbers.positive =
    match path with
    | Coq_xH -> Coq_xO Coq_xH
    | Coq_xO path -> Coq_xO (under_group path)
    | Coq_xI path -> Coq_xI (under_group path)
  in
  (* A path of [banned] is as an earlier guess gave it: the path of the [Let],
     or that path under the group of the lifting to the head. *)
  let take path matches =
    match
      List.find_opt
        (fun (sym, const) ->
          matches const
          && (not (List.mem (path, sym) banned))
          && not (List.mem (under_group path, sym) banned))
        !to_give
    with
    | None -> None
    | Some (sym, const) ->
      to_give := List.filter (fun (sym', _) -> sym' <> sym) !to_give;
      Some (sym, const)
  in
  (* The variables met that stand for a symbol or a constant: those bound to
     one, and those whose [Let] is lifted. *)
  let known_simples : (Model.variable * Model.simple) list ref = ref [] in
  let fields_agree fields fields' =
    List.length fields = List.length fields'
    && List.for_all2
         (fun (field : Model.simple) (field' : Model.simple) ->
           let field =
             match field with
             | Var x -> (
               match List.assoc_opt x !known_simples with
               | Some known -> known
               | None -> field)
             | Symbol _ | Const _ -> field
           in
           match field, field' with
           | Var _, _ -> true
           | (Symbol _ | Const _), Var _ -> false
           | (Symbol _ | Const _), (Symbol _ | Const _) -> field = field')
         fields fields'
  in
  (* The constants that a [Let] of the term before makes of symbols and
     constants alone, anywhere. A symbol that the term after binds to one of
     them is left for such a [Let]: at toplevel, a [Let] with a variable among
     its parts that stands for no symbol and no constant, which agrees with
     anything there, does not take it. The case aimed at is a boxed constant
     that a rewrite of Simplify makes, among boxes of numbers computed when the
     unit runs: the first [Let] of a box met would take the symbol of the
     constant. The list is made once, from the term before as it is given: a
     part counts if it is a symbol or a constant there, not a variable that the
     guess comes to know stands for one; and the bodies of code are gone into,
     so a constant that only code makes holds back a [Let] at toplevel too. *)
  let made_exactly : Model.static_const list =
    let made = ref [] in
    let plain (s : Model.simple) =
      match s with Symbol _ | Const _ -> true | Var _ -> false
    in
    let rec in_term (e : Model.expr) =
      match e with
      | Let (_, _, d, body) ->
        (match d with
        | Prim (Variadic (Make_block (Block_kind_values tag)), fields)
          when List.for_all plain fields ->
          made := Model.Static_const_block (tag, fields) :: !made
        | Prim (Variadic (Make_immutable_array _), elements)
          when List.for_all plain elements ->
          made := Model.Static_const_immutable_value_array elements :: !made
        | Prim (Unary (Box_number _), [contents]) when plain contents ->
          made := Model.Static_const_boxed_number contents :: !made
        | _ -> ());
        in_term body
      | Let_static (bindings, body) ->
        List.iter
          (fun (binding : Model.static_binding) ->
            match binding with
            | Bind_code (_, Code (_, _, _, _, _, code_body)) ->
              in_term code_body
            | Bind_deleted_code _ | Bind_symbol _ -> ())
          bindings;
        in_term body
      | Let_cont_non_recursive (_, Handler (_, _, handler), body) ->
        in_term handler;
        in_term body
      | Let_cont_recursive (handlers, body) ->
        List.iter
          (fun (_, Model.Handler (_, _, handler)) -> in_term handler)
          handlers;
        in_term body
      | Apply _ | Apply_cont _ | Switch _ | Invalid -> ()
    in
    in_term before;
    !made
  in
  let kept_from parts (const : Model.static_const) =
    List.mem const made_exactly
    && List.exists
         (fun (part : Model.simple) ->
           match part with
           | Var x -> not (List.mem_assoc x !known_simples)
           | Symbol _ | Const _ -> false)
         parts
  in
  (* A symbol that neither [before] nor [after], as they are given here, binds
     or mentions. *)
  let next_symbol =
    let greatest e =
      let references, bound = static_names e in
      List.fold_left
        (fun n name ->
          match name with
          | Symbol_name sym -> max n (int_of_positive sym)
          | Code_name _ -> n)
        0
        (bound @ List.map snd references)
    in
    ref (1 + max (greatest before) (greatest after))
  in
  let fresh_symbol () =
    let sym = positive_of_int !next_symbol in
    incr next_symbol;
    sym
  in
  (* The function declarations of the closures that the term after makes by a
     [Let], anywhere, and the tag and the number of fields of the blocks that it
     makes by one. *)
  let made_after, blocks_made_after =
    let made = ref [] in
    let blocks = ref [] in
    let rec in_term (e : Model.expr) =
      match e with
      | Let (_, _, d, body) ->
        (match d with
        | Closure_of_set (_, decls, _) -> made := decls :: !made
        | Prim (Variadic (Make_block (Block_kind_values tag)), fields) ->
          blocks := (tag, List.length fields) :: !blocks
        | _ -> ());
        in_term body
      | Let_static (bindings, body) ->
        List.iter
          (fun (binding : Model.static_binding) ->
            match binding with
            | Bind_code (_, Code (_, _, _, _, _, code_body)) ->
              in_term code_body
            | Bind_deleted_code _ | Bind_symbol _ -> ())
          bindings;
        in_term body
      | Let_cont_non_recursive (_, Handler (_, _, handler), body) ->
        in_term handler;
        in_term body
      | Let_cont_recursive (handlers, body) ->
        List.iter
          (fun (_, Model.Handler (_, _, handler)) -> in_term handler)
          handlers;
        in_term body
      | Apply _ | Apply_cont _ | Switch _ | Invalid -> ()
    in
    in_term after;
    !made, !blocks
  in
  (* The variables that a closure of the term before, made by a [Let] or bound
     to a symbol, anywhere, holds in a value slot. *)
  let held_in_slots =
    let held = ref [] in
    let of_slots (slots : (Model.value_slot * Model.simple) list) =
      List.iter
        (fun ((_, s) : Model.value_slot * Model.simple) ->
          match s with Var x -> held := x :: !held | Symbol _ | Const _ -> ())
        slots
    in
    let rec in_term (e : Model.expr) =
      match e with
      | Let (_, _, d, body) ->
        (match d with
        | Closure_of_set (_, _, slots) | Set_of_closures (_, slots) ->
          of_slots slots
        | _ -> ());
        in_term body
      | Let_static (bindings, body) ->
        List.iter
          (fun (binding : Model.static_binding) ->
            match binding with
            | Bind_code (_, Code (_, _, _, _, _, code_body)) ->
              in_term code_body
            | Bind_symbol
                ( _,
                  ( Static_const_closure_of_set (_, _, _, slots)
                  | Static_const_set_of_closures (_, slots) ) ) ->
              of_slots slots
            | Bind_deleted_code _ | Bind_symbol _ -> ())
          bindings;
        in_term body
      | Let_cont_non_recursive (_, Handler (_, _, handler), body) ->
        in_term handler;
        in_term body
      | Let_cont_recursive (handlers, body) ->
        List.iter
          (fun (_, Model.Handler (_, _, handler)) -> in_term handler)
          handlers;
        in_term body
      | Apply _ | Apply_cont _ | Switch _ | Invalid -> ()
    in
    in_term before;
    !held
  in
  (* For the copies of code: the code that the term before binds; whether two
     sets have the same function slots and, where the code is not the same, code
     of the second that the term before does not bind; and whether each value
     slot of the second set is one of the first that agrees with it. The copies
     asked for, each a list of code ids paired with new ones and the symbols of
     the set that gets the new ones. *)
  let code_bound_before =
    let _, bound = static_names before in
    List.filter_map
      (function Code_name id -> Some id | Symbol_name _ -> None)
      bound
  in
  let newly_coded decls decls' =
    List.length decls = List.length decls'
    && List.for_all2
         (fun (slot, code) (slot', code') ->
           slot = slot'
           && (code = code' || not (List.mem code' code_bound_before)))
         decls decls'
  in
  let slots_agree slots slots' =
    List.for_all
      (fun (slot', field') ->
        match List.assoc_opt slot' slots with
        | Some field -> fields_agree [field] [field']
        | None -> false)
      slots'
  in
  let copies = ref [] in
  let copy ids symbols =
    if not (List.mem (ids, symbols) !copies)
    then copies := (ids, symbols) :: !copies
  in
  let copied_ids decls decls' =
    List.concat
      (List.map2
         (fun (_, code) (_, code') ->
           if code = code' then [] else [code, code'])
         decls decls')
  in
  let lets = ref [] in
  let lift x path (target : Lift.lift_to) =
    lets := (path, target) :: !lets;
    match target with
    | Lift_to_symbol sym ->
      known_simples := (x, Model.Symbol sym) :: !known_simples
    | Lift_to_closure_symbols _ -> ()
  in
  (* The [Let]s that mention no variable, in the body of code or in a handler
     that is not at toplevel. [given] has the symbols given in the same place,
     each with its constant: a [Let] that no symbol is left for takes one of
     those, if it makes just that constant. The handlers of a group of recursive
     continuations have the paths of FlambdaLiftSpec.rec_handlers_map. *)
  let plain (s : Model.simple) =
    match s with Symbol _ | Const _ -> true | Var _ -> false
  in
  let plain_slots slots = List.for_all (fun (_, s) -> plain s) slots in
  let rec in_code ((code_lets, given) as where) (path : Numbers.positive)
      (e : Model.expr) =
    match e with
    | Let (_, _, d, body) ->
      let lift_out target = code_lets := (path, target) :: !code_lets in
      let take path matches : (Model.symbol * Model.static_const) option =
        match take path matches with
        | Some (sym, const) ->
          given := (sym, const) :: !given;
          Some (sym, const)
        | None -> List.find_opt (fun (_, const) -> matches const) !given
      in
      (* A closure fits a symbol of the term after if it is of the same code, or
         declarations, and the value slots of the closure that the symbol is
         bound to are among its own, each with the same symbol or constant.
         Simplify drops the slots that nothing reads once the code reads the
         symbol, so the closure of the term after may have fewer;
         FlambdaValidateSpec.without_slots removes them from the term before
         after the lifting. Nothing is believed of this: the validation checks
         the lifting, and a wrong guess is a unit that is not validated. *)
      (match d with
      | Set_of_closures (code, slots) when plain_slots slots -> (
        match
          match
            take path (fun (const : Model.static_const) ->
                match const with
                | Static_const_set_of_closures (code', slots') ->
                  code = code'
                  && List.for_all (fun slot' -> List.mem slot' slots) slots'
                | _ -> false)
          with
          | Some _ as taken -> taken
          | None -> (
            (* No symbol has this code: one whose closure has code that the term
               before does not bind, with a copy asked for. *)
            match
              take path (fun (const : Model.static_const) ->
                  match const with
                  | Static_const_set_of_closures (code', slots') ->
                    (not (List.mem code' code_bound_before))
                    && List.for_all (fun slot' -> List.mem slot' slots) slots'
                  | _ -> false)
            with
            | Some (sym, Static_const_set_of_closures (code', _)) as taken ->
              copy [code, code'] [sym];
              taken
            | taken -> taken)
        with
        | Some (sym, _) -> lift_out (Lift.Lift_to_symbol sym)
        | None -> ())
      | Closure_of_set (_, decls, slots) when plain_slots slots -> (
        match
          match
            take path (fun (const : Model.static_const) ->
                match const with
                | Static_const_closure_of_set (_, _, decls', slots') ->
                  decls = decls'
                  && List.for_all (fun slot' -> List.mem slot' slots) slots'
                | _ -> false)
          with
          | Some _ as taken -> taken
          | None -> (
            match
              take path (fun (const : Model.static_const) ->
                  match const with
                  | Static_const_closure_of_set (_, _, decls', slots') ->
                    newly_coded decls decls'
                    && List.for_all (fun slot' -> List.mem slot' slots) slots'
                  | _ -> false)
            with
            | Some (_, Static_const_closure_of_set (_, symbols, decls', _)) as
              taken ->
              copy (copied_ids decls decls') (List.map snd symbols);
              taken
            | taken -> taken)
        with
        | Some (_, Static_const_closure_of_set (_, symbols, _, _)) ->
          to_give
            := List.filter
                 (fun (sym, _) ->
                   not (List.exists (fun (_, sym') -> sym = sym') symbols))
                 !to_give;
          lift_out (Lift.Lift_to_closure_symbols symbols)
        | Some _ | None -> ())
      | Prim (Variadic (Make_block (Block_kind_values tag)), fields)
        when List.for_all plain fields -> (
        match
          take path (fun (const : Model.static_const) ->
              match const with
              | Static_const_block (tag', fields') ->
                tag = tag' && fields = fields'
              | _ -> false)
        with
        | Some (sym, _) -> lift_out (Lift.Lift_to_symbol sym)
        | None -> ())
      | Prim (Variadic (Make_immutable_array _), elements)
        when List.for_all plain elements -> (
        match
          take path (fun (const : Model.static_const) ->
              match const with
              | Static_const_immutable_value_array elements' ->
                elements = elements'
              | _ -> false)
        with
        | Some (sym, _) -> lift_out (Lift.Lift_to_symbol sym)
        | None -> ())
      | Prim (Unary (Box_number _), [contents]) when plain contents -> (
        match
          take path (fun (const : Model.static_const) ->
              match const with
              | Static_const_boxed_number contents' -> contents = contents'
              | _ -> false)
        with
        | Some (sym, _) -> lift_out (Lift.Lift_to_symbol sym)
        | None -> ())
      | _ -> ());
      in_code where (Numbers.Coq_xO path) body
    | Let_static (_, body) -> in_code where (Numbers.Coq_xO path) body
    | Let_cont_non_recursive (_, Handler (_, _, handler), body) ->
      in_code where (Numbers.Coq_xO path) body;
      in_code where (Numbers.Coq_xI path) handler
    | Let_cont_recursive (handlers, body) ->
      in_code where (Numbers.Coq_xO path) body;
      in_rec_handlers where path handlers
    | Apply _ | Apply_cont _ | Switch _ | Invalid -> ()
  and in_rec_handlers where (path : Numbers.positive) handlers =
    ignore
      (List.fold_left
         (fun q (_, Model.Handler (_, _, handler)) ->
           in_code where (Numbers.Coq_xO q) handler;
           Numbers.Coq_xI q)
         (Numbers.Coq_xI path) handlers
        : Numbers.positive)
  in
  let to_head = ref [] in
  let to_head_given = ref [] in
  (* A symbol, a constant, or a variable met that stands for one. *)
  let known_simple (s : Model.simple) =
    match s with
    | Symbol _ | Const _ -> true
    | Var x -> List.mem_assoc x !known_simples
  in
  let rec in_expr (path : Numbers.positive) (e : Model.expr) =
    match e with
    | Let (x, _, d, body) ->
      (match d with
      | Simple ((Symbol _ | Const _) as known) ->
        known_simples := (x, known) :: !known_simples
      | Set_of_closures (code, slots) -> (
        match
          match
            take path (fun (const : Model.static_const) ->
                match const with
                | Static_const_set_of_closures (code', _) -> code = code'
                | _ -> false)
          with
          | Some _ as taken -> taken
          | None -> (
            (* No symbol to give has this code: one whose closure has code that
               the term before does not bind, with a copy asked for. *)
            match
              take path (fun (const : Model.static_const) ->
                  match const with
                  | Static_const_set_of_closures (code', slots') ->
                    (not (List.mem code' code_bound_before))
                    && slots_agree slots slots'
                  | _ -> false)
            with
            | Some (sym, Static_const_set_of_closures (code', _)) as taken ->
              copy [code, code'] [sym];
              taken
            | taken -> taken)
        with
        | Some (sym, _) -> lift x path (Lift_to_symbol sym)
        | None -> ())
      | Closure_of_set (slot, decls, slots) -> (
        match
          match
            take path (fun (const : Model.static_const) ->
                match const with
                | Static_const_closure_of_set (_, _, decls', _) ->
                  decls = decls'
                | _ -> false)
          with
          | Some _ as taken -> taken
          | None -> (
            (* The same of a set whose function slots are named: a copy for each
               function whose code is not the same. *)
            match
              take path (fun (const : Model.static_const) ->
                  match const with
                  | Static_const_closure_of_set (_, _, decls', slots') ->
                    newly_coded decls decls' && slots_agree slots slots'
                  | _ -> false)
            with
            | Some (_, Static_const_closure_of_set (_, symbols, decls', _)) as
              taken ->
              copy (copied_ids decls decls') (List.map snd symbols);
              taken
            | taken -> taken)
        with
        | Some (_, Static_const_closure_of_set (_, symbols, _, _)) ->
          (* The symbols of the other closures of the set go with it. *)
          to_give
            := List.filter
                 (fun (sym, _) ->
                   not (List.exists (fun (_, sym') -> sym = sym') symbols))
                 !to_give;
          (match List.assoc_opt slot symbols with
          | Some sym -> known_simples := (x, Model.Symbol sym) :: !known_simples
          | None -> ());
          lift x path (Lift_to_closure_symbols symbols)
        | Some _ -> ()
        | None ->
          (* A closure whose value slots hold symbols, constants or variables
             made aliases of symbols, for which the guess has no symbol left to
             take (the term after may bind one to such a closure: it is then one
             that the term before binds, that another [Let] took, or that an
             earlier attempt banned), and that the term after does not make by a
             [Let]. The case aimed at is a function of which Simplify has
             inlined every call. It is given symbols that neither of the two
             terms given here has, so that a value slot that holds it holds a
             symbol. *)
          if
            List.for_all (fun (_, s) -> known_simple s) slots
            && not (List.mem decls made_after)
          then (
            let symbols =
              List.map (fun (slot, _) -> slot, fresh_symbol ()) decls
            in
            (match List.assoc_opt slot symbols with
            | Some sym ->
              known_simples := (x, Model.Symbol sym) :: !known_simples
            | None -> ());
            lift x path (Lift_to_closure_symbols symbols)))
      | Prim (Variadic (Make_block (Block_kind_values tag)), fields) -> (
        match
          take path (fun (const : Model.static_const) ->
              match const with
              | Static_const_block (tag', fields') ->
                tag = tag'
                && fields_agree fields fields'
                && not (kept_from fields const)
              | _ -> false)
        with
        | Some (sym, _) -> lift x path (Lift_to_symbol sym)
        | None ->
          (* A block whose fields are symbols, constants or variables made
             aliases of symbols, that a closure of the term before holds in a
             value slot, for which the guess has no symbol left to take (as for
             the closure above: the term after may still bind one to a block of
             that shape), and of whose tag and number of fields the term after
             makes no block by a [Let]. The case aimed at is the block of a
             structure given to a functor: the closure of a function of the
             functor's result holds it (a closure that an earlier lifting has
             bound to a symbol, with the variable in its slot), and Simplify has
             resolved every load from it. It is given a symbol that neither of
             the two terms given here has, so that the value slot holds a
             symbol. The lifted term has no use for that symbol once the loads
             are resolved, which is as it should be: the search does not take it
             for a symbol given to the wrong [Let]. *)
          if
            List.for_all known_simple fields
            && List.mem x held_in_slots
            && not (List.mem (tag, List.length fields) blocks_made_after)
          then lift x path (Lift_to_symbol (fresh_symbol ())))
      | Prim (Variadic (Make_immutable_array _), elements) -> (
        match
          take path (fun (const : Model.static_const) ->
              match const with
              | Static_const_immutable_value_array elements' ->
                fields_agree elements elements'
                && not (kept_from elements const)
              | _ -> false)
        with
        | Some (sym, _) -> lift x path (Lift_to_symbol sym)
        | None -> ())
      | Prim (Unary (Box_number _), [contents]) -> (
        match
          take path (fun (const : Model.static_const) ->
              match const with
              | Static_const_boxed_number contents' ->
                fields_agree [contents] [contents']
                && not (kept_from [contents] const)
              | _ -> false)
        with
        | Some (sym, _) -> lift x path (Lift_to_symbol sym)
        | None -> ())
      | _ -> ());
      in_expr (Numbers.Coq_xO path) body
    | Let_static (_, body) -> in_expr (Numbers.Coq_xO path) body
    | Let_cont_non_recursive (k, (Handler (_, _, handler) as h), body) ->
      in_expr (Numbers.Coq_xO path) body;
      if Well_formed.handler_at_toplevel unit.Well_formed.unit_exn k h body
      then in_expr (Numbers.Coq_xI path) handler
      else in_code (to_head, to_head_given) (Numbers.Coq_xI path) handler
    | Let_cont_recursive (handlers, body) ->
      in_expr (Numbers.Coq_xO path) body;
      in_rec_handlers (to_head, to_head_given) path handlers
    | Apply _ | Apply_cont _ | Switch _ | Invalid -> ()
  in
  in_expr Lift.root before;
  let to_head = List.rev !to_head in
  let lets =
    match Lift_from_code.lifted_out_bindings to_head Lift.root before with
    | [] -> List.rev !lets
    | _ :: _ ->
      List.rev_map (fun (path, target) -> under_group path, target) !lets
  in
  let out_of_code = ref [] in
  let rec in_groups (e : Model.expr) =
    match e with
    | Let (_, _, _, body) -> in_groups body
    | Let_static (bindings, body) ->
      List.iter
        (fun (binding : Model.static_binding) ->
          match binding with
          | Bind_code (id, Code (_, _, _, _, _, code_body)) -> (
            let code_lets = ref [] in
            in_code (code_lets, ref []) Lift.root code_body;
            match !code_lets with
            | [] -> ()
            | _ :: _ -> out_of_code := (id, List.rev !code_lets) :: !out_of_code
            )
          | Bind_deleted_code _ | Bind_symbol _ -> ())
        bindings;
      in_groups body
    | Let_cont_non_recursive (_, Handler (_, _, handler), body) ->
      in_groups body;
      in_groups handler
    | Let_cont_recursive (_, body) -> in_groups body
    | Apply _ | Apply_cont _ | Switch _ | Invalid -> ()
  in
  in_groups before;
  let out_of_code = List.rev !out_of_code in
  let lifted =
    Lift.lift_expr lets Lift.root
      (Lift_from_code.lift_from_code out_of_code [] Lift.root
         (Lift_from_arms.lift_to_head to_head before))
  in
  (* In the code of a stub Simplify knows nothing of [my_closure]: it simplifies
     that code where it is bound and not where its set of closures is (see
     FlambdaStubCallsSpec.v). So the code of a stub is not among the code whose
     [my_closure] is to be bound to the symbol of its closure. *)
  let stubs = stub_code lifted in
  let my_closures =
    List.filter_map
      (fun (sym, (const : Model.static_const)) ->
        let code =
          match const with
          | Static_const_set_of_closures (code, _) -> Some code
          | Static_const_closure_of_set (slot, _, decls, _) ->
            List.assoc_opt slot decls
          | Static_const_block _ | Static_const_boxed_number _
          | Static_const_immutable_string _
          | Static_const_immutable_value_array _ ->
            None
        in
        match code with
        | Some code
          when (not (List.mem code stubs))
               && My_closure.only_closure_of code sym lifted ->
          Some (code, sym)
        | Some _ | None -> None)
      (bound_symbols after)
  in
  let first : Lift.lifting =
    { lifting_unit = unit;
      lifted_lets = lets;
      lifted_my_closures = my_closures;
      lifted_out_of_code = out_of_code;
      lifted_to_head = to_head;
      lifted_around = [];
      groups_put = [];
      tables_put = [];
      code_copied = List.rev !copies;
      made_closures_called = false
    }
  in
  (* [known] has the symbols that the groups around a place bind, each with its
     constant, less those whose constant mentions a variable bound since. [arm]
     is the path of the [Let_cont] of the outermost handler around the place
     that is not at toplevel, with the variables bound since. *)
  let lifted = Validate.with_lifting first before in
  let bound_lifted = List.map fst (bound_symbols lifted) in
  let left =
    ref (List.filter (fun (sym, _) -> not (List.mem sym bound_lifted)) !to_give)
  in
  let aliases = ref [] in
  let places : (Lift.path * Lift.path list) list ref = ref [] in
  let put at_path let_path =
    let before_it =
      match List.assoc_opt at_path !places with
      | Some paths -> paths
      | None -> []
    in
    places
      := (at_path, before_it @ [let_path])
         :: List.filter (fun (path', _) -> path' <> at_path) !places
  in
  let mentions bound (const : Model.static_const) =
    List.exists
      (fun x -> List.mem x bound)
      (Binders.free_variables_of_static_const const)
  in
  let forget bound known =
    List.filter (fun (_, const) -> not (mentions bound const)) known
  in
  (* What a [Let] of [d] is lifted to, where a symbol bound to [const] is to
     stand for it: that symbol, or, for a closure of a set whose function slots
     are named, the symbols of the closures of the set. *)
  let target_of (d : Model.named) sym (const : Model.static_const) :
      Lift.lift_to =
    match d, const with
    | Closure_of_set _, Static_const_closure_of_set (_, symbols, _, _) ->
      Lift_to_closure_symbols symbols
    | _, _ -> Lift_to_symbol sym
  in
  (* The symbols that stand for the [Let] of [d] lifted to [target], each with
     its constant: one, but for a set of several closures. *)
  let consts_of (d : Model.named) (target : Lift.lift_to) =
    match Lift.lift_named target d with
    | Some (bindings, _) ->
      let consts =
        List.filter_map
          (fun (binding : Model.static_binding) ->
            match binding with
            | Bind_symbol (sym, const) -> Some (sym, const)
            | Bind_code _ | Bind_deleted_code _ -> None)
          bindings
      in
      if List.length consts = List.length bindings then Some consts else None
    | None -> None
  in
  let makes (d : Model.named) (const : Model.static_const) =
    match d, const with
    | ( Prim (Variadic (Make_block (Block_kind_values tag)), fields),
        Static_const_block (tag', fields') ) ->
      tag = tag' && fields_agree fields fields'
    | ( Prim (Variadic (Make_immutable_array _), elements),
        Static_const_immutable_value_array elements' ) ->
      fields_agree elements elements'
    | ( Prim (Unary (Box_number _), [contents]),
        Static_const_boxed_number contents' ) ->
      fields_agree [contents] [contents']
    | Set_of_closures (code, slots), Static_const_set_of_closures (code', slots')
      ->
      code = code'
      && List.map fst slots = List.map fst slots'
      && fields_agree (List.map snd slots) (List.map snd slots')
    | ( Closure_of_set (slot, decls, slots),
        Static_const_closure_of_set (slot', _, decls', slots') ) ->
      slot = slot' && decls = decls'
      && List.map fst slots = List.map fst slots'
      && fields_agree (List.map snd slots) (List.map snd slots')
    | _ -> false
  in
  let rec scan known arm (path : Numbers.positive) (e : Model.expr) =
    match e with
    | Let (x, _, d, body) ->
      let known =
        match
          List.find_map
            (fun (sym, const) ->
              let target = target_of d sym const in
              match consts_of d target with
              | Some consts
                when List.mem (sym, const) consts
                     && List.for_all (fun bound -> List.mem bound known) consts
                ->
                Some target
              | Some _ | None -> None)
            known
        with
        | Some target ->
          aliases := (path, target) :: !aliases;
          known
        | None -> (
          match arm with
          | Some (arm_path, bound)
            when not
                   (List.exists
                      (fun x -> List.mem x bound)
                      (Binders.free_variables_of_named d)) -> (
            match List.find_opt (fun (_, const) -> makes d const) !left with
            | Some (sym, const) -> (
              let target = target_of d sym const in
              match consts_of d target with
              | Some consts ->
                left
                  := List.filter
                       (fun (sym', _) -> not (List.mem_assoc sym' consts))
                       !left;
                aliases := (path, target) :: !aliases;
                put arm_path path;
                consts @ known
              | None -> known)
            | None -> known)
          | Some _ | None -> known)
      in
      let arm =
        Option.map (fun (arm_path, bound) -> arm_path, x :: bound) arm
      in
      scan (forget [x] known) arm (Numbers.Coq_xO path) body
    | Let_static (bindings, body) ->
      let known =
        List.filter_map
          (fun (binding : Model.static_binding) ->
            match binding with
            | Bind_symbol (sym, const) -> Some (sym, const)
            | Bind_code _ | Bind_deleted_code _ -> None)
          bindings
        @ known
      in
      scan known arm (Numbers.Coq_xO path) body
    | Let_cont_non_recursive (k, (Handler (params, _, handler) as h), body) ->
      let arm' =
        match arm with
        | Some (arm_path, bound) -> Some (arm_path, params @ bound)
        | None ->
          if Well_formed.handler_at_toplevel unit.Well_formed.unit_exn k h body
          then None
          else Some (path, params)
      in
      scan (forget params known) arm' (Numbers.Coq_xI path) handler;
      scan known arm (Numbers.Coq_xO path) body
    | Let_cont_recursive (handlers, body) ->
      scan known arm (Numbers.Coq_xO path) body;
      ignore
        (List.fold_left
           (fun q (_, Model.Handler (params, _, handler)) ->
             let arm' =
               match arm with
               | Some (arm_path, bound) -> Some (arm_path, params @ bound)
               | None -> Some (path, params)
             in
             scan (forget params known) arm' (Numbers.Coq_xO q) handler;
             Numbers.Coq_xI q)
           (Numbers.Coq_xI path) handlers
          : Numbers.positive)
    | Apply _ | Apply_cont _ | Switch _ | Invalid -> ()
  in
  if around then scan [] None Lift.root lifted;
  { first with lifted_around = List.rev !aliases; groups_put = !places }

(* ---- Tables that Simplify made ----------------------------------------- *)

(* The symbols that the term after binds to an immutable array of constants and
   symbols, outside code, that the term before does not bind and that the term
   after uses. Simplify makes such a table of a [Switch] whose arms give
   constants, and loads from it ([Simplify_switch_expr]); the term before has
   the [Switch] and no table. This is a guess:
   FlambdaUnusedHeadStaticsSpec.add_head_tables binds the tables at the head of
   the term before only where its checks pass, and the comparison of the two
   terms says the rest. *)
let tables_guess (before : Model.expr) (after : Model.expr) :
    (Model.symbol * Model.simple list) list =
  let _, bound_before = static_names before in
  let unused_after = unused_statics after in
  let plain (s : Model.simple) =
    match s with Symbol _ | Const _ -> true | Var _ -> false
  in
  let tables = ref [] in
  let rec in_expr (e : Model.expr) =
    match e with
    | Let (_, _, _, body) -> in_expr body
    | Let_static (bindings, body) ->
      List.iter
        (fun (binding : Model.static_binding) ->
          match binding with
          | Bind_symbol (sym, Static_const_immutable_value_array elements)
            when List.for_all plain elements
                 && (not (List.mem (Symbol_name sym) bound_before))
                 && not (List.mem sym unused_after.dropped_symbols) ->
            tables := (sym, elements) :: !tables
          | Bind_symbol _ | Bind_code _ | Bind_deleted_code _ -> ())
        bindings;
      in_expr body
    | Let_cont_non_recursive (_, Handler (_, _, handler), body) ->
      in_expr handler;
      in_expr body
    | Let_cont_recursive (handlers, body) ->
      List.iter
        (fun (_, Model.Handler (_, _, handler)) -> in_expr handler)
        handlers;
      in_expr body
    | Apply _ | Apply_cont _ | Switch _ | Invalid -> ()
  in
  in_expr after;
  List.rev !tables

(* Whether a term begins with the group of static bindings of the tables given,
   as FlambdaUnusedHeadStaticsSpec.add_head_tables leaves it where it binds
   them. *)
let tables_bound (tables : (Model.symbol * Model.simple list) list)
    (e : Model.expr) =
  match tables, e with
  | [], _ -> false
  | _ :: _, Let_static (bindings, _) ->
    bindings
    = List.map
        (fun (sym, elements) ->
          Model.Bind_symbol (sym, Static_const_immutable_value_array elements))
        tables
  | ( _ :: _,
      ( Let _ | Let_cont_non_recursive _ | Let_cont_recursive _ | Apply _
      | Apply_cont _ | Switch _ | Invalid ) ) ->
    false

(* ---- Value slots that Simplify removed --------------------------------- *)

(* The value slots that the static sets of closures of the term before have and
   those of the term after have not, and that nothing in the term before
   projects. This is a guess: FlambdaUnusedSlotsSpec.drop_unused_slots checks
   that nothing projects them, and the comparison of the two terms the rest. *)
let unused_slots_guess (before : Model.expr) (after : Model.expr) :
    Model.value_slot list =
  let rec static_slots (e : Model.expr) =
    match e with
    | Let (_, _, _, body) -> static_slots body
    | Let_static (bindings, body) ->
      List.concat_map
        (fun (binding : Model.static_binding) ->
          match binding with
          | Bind_symbol (_, Static_const_set_of_closures (_, slots))
          | Bind_symbol (_, Static_const_closure_of_set (_, _, _, slots)) ->
            List.map fst slots
          | Bind_symbol
              ( _,
                ( Static_const_block _ | Static_const_boxed_number _
                | Static_const_immutable_string _
                | Static_const_immutable_value_array _ ) )
          | Bind_code _ | Bind_deleted_code _ ->
            [])
        bindings
      @ static_slots body
    | Let_cont_non_recursive (_, Handler (_, _, handler), body) ->
      static_slots body @ static_slots handler
    | Let_cont_recursive (_, body) -> static_slots body
    | Apply _ | Apply_cont _ | Switch _ | Invalid -> []
  in
  let kept = static_slots after in
  List.sort_uniq compare (static_slots before)
  |> List.filter (fun slot ->
      (not (List.mem slot kept)) && Unused_slots.slots_unused [slot] before)

(* ---- The check -------------------------------------------------------- *)

(* Whether a code id is used: called, made a closure of, or bound to code. Being
   bound as deleted is not a use. *)
let rec code_id_used (id : Model.code_id) (e : Model.expr) =
  let in_handler (Model.Handler (_, _, handler)) = code_id_used id handler in
  let in_binding (binding : Model.static_binding) =
    match binding with
    | Bind_code (bound, Code (_, _, _, _, _, body)) ->
      bound = id || code_id_used id body
    | Bind_deleted_code _ -> false
    | Bind_symbol (_, Static_const_set_of_closures (code, _)) -> code = id
    | Bind_symbol (_, Static_const_closure_of_set (_, _, function_decls, _)) ->
      List.exists (fun (_, code) -> code = id) function_decls
    | Bind_symbol
        ( _,
          ( Static_const_block _ | Static_const_boxed_number _
          | Static_const_immutable_string _
          | Static_const_immutable_value_array _ ) ) ->
      false
  in
  match e with
  | Let (_, _, Set_of_closures (code, _), body) ->
    code = id || code_id_used id body
  | Let (_, _, Closure_of_set (_, function_decls, _), body) ->
    List.exists (fun (_, code) -> code = id) function_decls
    || code_id_used id body
  | Let
      ( _,
        _,
        ( Simple _ | Prim _ | Project_value_slot _ | Project_value_slot_from _
        | Project_function_slot _ ),
        body ) ->
    code_id_used id body
  | Let_static (bindings, body) ->
    List.exists in_binding bindings || code_id_used id body
  | Let_cont_non_recursive (_, handler, body) ->
    in_handler handler || code_id_used id body
  | Let_cont_recursive (handlers, body) ->
    List.exists (fun (_, handler) -> in_handler handler) handlers
    || code_id_used id body
  | Apply { ap_call_kind = Direct code; _ } -> code = id
  | Apply { ap_call_kind = Indirect; _ } | Apply_cont _ | Switch _ | Invalid ->
    false

(* ---- What the .cmx files say of other units ---------------------------- *)

(* The names of units, in order. *)
let unit_names (units : (Compilation_unit.t * Digest.t) list) =
  List.sort String.compare
    (List.map (fun (unit, _) -> Compilation_unit.name_as_string unit) units)

(* The prefix: static bindings that say, of symbols and code of other units,
   what the [.cmx] files of those units say. It is put in front of both terms of
   a check ([with_prefix]), which are then run and validated as they are: a
   verdict is of the unit run from the static environment that the prefix makes.
   What is trusted of it, beyond that the [.cmx] files are right about the units
   that the program is linked with: that the prefix says nothing that the files
   do not say, and that whatever it leaves out can make a run undefined or a
   unit refused, and never gives a run a defined answer that the linked unit
   would not give. The round trip of the conversion does not reach it; it is
   printed for each unit that has one ([print_prefix]), to be read against the
   other unit.

   What the files say is read through the compiler: its loader of [.cmx] files
   ([Flambda_cmx]), and its own reduction of the type of a symbol to what a
   static binding can say ([Flambda_cmx.load_symbol_approx], which classic mode
   uses). Nothing that Simplify concluded is used. The loader is a second one,
   made here as that of Simplify is made, since that one cannot be reached. It
   is asked only about the units that the run of Simplify read ([units_read]),
   whose descriptions it has from the tables of the compiler: a unit that the
   run did not read is never asked about, so making the prefix opens no file
   that the run did not read. Of a file that the run read it may read more: the
   compiler reads the body of code from the file when the body is first looked
   at, and the prefix may bind code that Simplify did not look at. The loader
   renames the variables and the continuations of the code that it imports, and
   opening a binder of the compiler makes fresh names, so the prefix is
   converted once and is the same term in front of both terms.

   What is bound, for a symbol or a code id whose unit was read, is one of four
   things.

   A symbol that is an immutable block of values, each of whose fields is a
   constant, a symbol, or a field of which the file says nothing, where it is
   asked for or is a field of a block that is bound: the block. A field that is
   a symbol is that symbol, whether or not it is bound in turn. A field of which
   the file says nothing is a symbol of its own, which no name of the compiler
   has: one for each such field of each block, made when the block is bound, and
   declared as an import of which nothing is known ([imports]). A block one of
   whose such fields is loaded, by either term or by a body that the prefix
   binds, from something that stands for the block (below) is not bound. Nor is
   a block with any other field (an approximation given in place of a symbol, a
   constant that the model has not got), or with a field that is a symbol which
   its own file says is another symbol or a constant. So the prefix says less
   than the file and never more: a block that is not bound stays an import, and
   a field that is given a symbol is one of which the file says nothing. The
   test of the loads decides only which blocks are bound, which is what a
   validation can find, and is trusted for nothing: a load that it misses leaves
   a block bound, with what the file says of it and no more.

   A symbol that is a closure, where it is asked for or is a field of a block
   that is bound: the closure, at its function slot, of a set that has this one
   function and no value slot. The reduction gives no more of a closure. A
   projection of a value slot of it, or to another function of its set, has no
   value in the model, so a run that needs what is left out is undefined.

   A code id that is asked for and whose code has a body in the file: the code,
   converted as the code of a unit is. Code that has its metadata only is not
   bound, and neither is code whose body the model has not got, tupled code (no
   direct call of tupled code is in the model, see [tupled_code]), or code with
   a call of known arity where one of the two units has tupled code. A call of
   code that is not bound is undefined.

   A symbol of which the file says nothing: nothing.

   A symbol that is asked for and that the file says is another symbol, or a
   constant, is refused, and the unit is not checked: no static binding says
   either, and the class of data could be wrong of the first. What is not bound
   stays as it was: a symbol is an import ([imports]), and a code id is bound to
   nothing.

   What is asked for is the least set of names that has: (a) every symbol and
   every code id that either term names; (b) the code id of a closure that is
   asked for; (c) every symbol and every code id that a body which the prefix
   binds names; (d) the symbol at a field of a block that the prefix binds,
   where either term, or a body that the prefix binds, loads that field from
   something that stands for the block. What stands for a block: its symbol; a
   variable that a [Let] binds to its symbol; and a variable that a [Let] binds
   to a load, of a field that is a symbol, from something that stands for a
   block that the prefix binds, which stands for that symbol. A parameter stands
   for nothing, and neither does a variable that a [Let] binds to another
   variable.

   So what the prefix binds is the data that the files reach, through fields,
   from the blocks that are asked for, and the bodies of the code that is asked
   for: a unit that names the module block of a large unit has in front of it
   the blocks and the closures reached from that block, each closure with no
   code, and the bodies of the functions that the two terms reach, and not the
   bodies of the whole unit. The data is bound without being asked for because a
   run reads it through names that the rule above does not follow (a list of the
   other unit, through the parameter of the loop that goes down it), and because
   a symbol left an import is one more import: a verdict holds of a unit with
   imports for fewer of the comparisons of the validator. Being asked for
   matters for code alone.

   The bindings are in groups ([Let_static]), to be well-formed
   (FlambdaWellFormedSpec.sets_well_formed): two closures of the same code are
   not bound by one group. The first group has the code, a closure of each code,
   and the blocks, each after those of its fields that are blocks where the
   blocks are not cyclic; the closures left are in groups after it. The machine
   looks up neither a field that is a symbol when it binds a block, nor code
   when it binds a closure, so the groups make the same static environment in
   any order. *)
type prefix =
  { prefix_groups : Model.static_binding list list;
    (* The units from whose [.cmx] files something is bound, with the digests of
       the files. *)
    prefix_units : (Compilation_unit.t * Digest.t) list;
    (* The symbols that the prefix binds. *)
    prefix_symbols : Model.symbol list;
    (* The closures and the blocks bound that nothing asks for, each with the
       block that is asked for from which it is reached. *)
    prefix_unasked : (Model.symbol * Model.symbol) list;
    (* The symbols that have a number and that neither the two terms nor the
       prefix mention: those of a body that was not bound in the end. *)
    prefix_unmentioned : Model.symbol list;
    (* The symbols made for the fields, of the blocks bound, of which the files
       say nothing. Each is an import of which nothing is known. *)
    prefix_hidden : Model.symbol list
  }

let no_prefix =
  { prefix_groups = [];
    prefix_units = [];
    prefix_symbols = [];
    prefix_unasked = [];
    prefix_unmentioned = [];
    prefix_hidden = []
  }

(* What the [.cmx] file of its unit says of a symbol. *)
type approximation = Code_or_metadata.t Value_approximation.t

(* The prefix made with some blocks refused: [refused] are the blocks, by the
   numbers of their symbols, that are not to be bound with a symbol for a field
   of which the file says nothing. [found] is given the blocks not refused one
   of whose such fields a term loads; if there is one, the prefix returned is
   not to be used ([make_prefix]). *)
let make_prefix_refusing ~(refused : int list) ~(found : int list ref)
    ~(units : (Compilation_unit.t * Digest.t) list)
    ~(loader : Flambda_cmx.loader Lazy.t) ~units_have_tupled_code
    (terms : Model.expr list) : prefix =
  (* The unit whose file the loader reads for a name of [unit], if the run read
     it. *)
  let read unit =
    let file =
      Compilation_unit.which_cmx_file unit
        ~accessed_by:(Current_unit.get_cu_exn ())
    in
    List.find_opt (fun (read, _) -> Compilation_unit.equal read file) units
  in
  (* What the file of its unit says of a symbol, with the unit: nothing for a
     symbol whose unit was not read. *)
  let approximations :
      ((Compilation_unit.t * Digest.t) * approximation) option Symbol.Map.t ref
      =
    ref Symbol.Map.empty
  in
  let approximation symbol :
      ((Compilation_unit.t * Digest.t) * approximation) option =
    match Symbol.Map.find_opt symbol !approximations with
    | Some approx -> approx
    | None ->
      let approx =
        Option.map
          (fun unit ->
            unit, Flambda_cmx.load_symbol_approx (Lazy.force loader) symbol)
          (read (Symbol.compilation_unit symbol))
      in
      approximations := Symbol.Map.add symbol approx !approximations;
      approx
  in
  let asked_symbols : (int, unit) Hashtbl.t = Hashtbl.create 64 in
  let asked_code : (int, unit) Hashtbl.t = Hashtbl.create 64 in
  (* The blocks bound, by the numbers of their symbols, with their fields. *)
  let blocks : (int, Model.simple list) Hashtbl.t = Hashtbl.create 16 in
  (* The closures bound, by the numbers of their symbols, with their code. *)
  let closures : (int, Model.code_id) Hashtbl.t = Hashtbl.create 64 in
  (* The symbols that are fields of the blocks bound. *)
  let fields_of_blocks : (int, unit) Hashtbl.t = Hashtbl.create 64 in
  let block_bindings = ref [] in
  let closure_bindings = ref [] in
  let code_bindings = ref [] in
  let bodies = ref [] in
  let used = ref [] in
  let changed = ref false in
  let use ((unit, _) as read) =
    if not (List.exists (fun (u, _) -> Compilation_unit.equal u unit) !used)
    then used := read :: !used
  in
  let bind_code unit id code_id =
    let loader = Lazy.force loader in
    ignore
      (Flambda_cmx.load_cmx_file_contents loader
         (Code_id.get_compilation_unit code_id)
        : _ option);
    match
      Exported_code.find (Flambda_cmx.get_imported_code loader ()) code_id
    with
    | None -> ()
    | Some code -> (
      match Code_or_metadata.view code with
      | Metadata_only _ -> ()
      | Code_present code -> (
        tupled_code := [];
        directly_called_code := [];
        known_arity_call_met := false;
        match convert_code code with
        | exception Unsupported _ -> ()
        | Model.Code (_, _, _, _, _, body) as converted ->
          let is_tupled = match !tupled_code with [] -> false | _ -> true in
          if
            (not is_tupled)
            && not (!known_arity_call_met && units_have_tupled_code)
          then (
            code_bindings := Model.Bind_code (id, converted) :: !code_bindings;
            bodies := body :: !bodies;
            use unit)))
  in
  let ask_code id =
    let n = int_of_positive id in
    if not (Hashtbl.mem asked_code n)
    then (
      Hashtbl.replace asked_code n ();
      changed := true;
      match Code_ids.name_of id with
      | None -> ()
      | Some code_id -> (
        match read (Code_id.get_compilation_unit code_id) with
        | None -> ()
        | Some unit -> bind_code unit id code_id))
  in
  (* Binds a closure, if it is not bound yet, and gives its code. *)
  let bind_closure unit sym ~function_slot ~code_id =
    match Hashtbl.find_opt closures (int_of_positive sym) with
    | Some code -> code
    | None ->
      let slot = Function_slots.find function_slot in
      let code = Code_ids.find code_id in
      let const : Model.static_const =
        Static_const_closure_of_set (slot, [slot, sym], [slot, code], [])
      in
      Hashtbl.replace closures (int_of_positive sym) code;
      closure_bindings
        := (code, Model.Bind_symbol (sym, const)) :: !closure_bindings;
      use unit;
      code
  in
  (* Whether a field is one that a static block can have. *)
  let said (field : approximation) =
    match field with
    | Value_const c -> (
      match convert_const c with
      | exception Unsupported _ -> false
      | (_ : Model.const) -> true)
    | Value_symbol symbol -> (
      match Option.map snd (approximation symbol) with
      | None -> true
      | Some (Unknown _ | Closure_approximation _ | Block_approximation _) ->
        true
      | Some (Value_symbol _ | Value_const _) -> false)
    | Unknown _ | Closure_approximation _ | Block_approximation _ -> false
  in
  (* The blocks that are fields of the blocks bound and are still to be looked
     at, and the symbols of the blocks looked at so far: a block is looked at
     once, however many fields reach it, so the walk ends on data that is shared
     or cyclic. *)
  let blocks_to_bind = Queue.create () in
  let blocks_seen : (int, unit) Hashtbl.t = Hashtbl.create 16 in
  (* The symbols made for the fields of which the files say nothing, each with
     the block that it is a field of, and the blocks one of whose such fields a
     term loads. *)
  let hidden = ref [] in
  let hidden_of : (int, int) Hashtbl.t = Hashtbl.create 16 in
  let read_hidden = ref [] in
  let bind_one_block unit sym tag (fields : approximation list) =
    let n = int_of_positive sym in
    (* Whether a field is given a symbol of its own. *)
    let hides (field : approximation) =
      match field with
      | Unknown _ -> not (List.mem n refused)
      | Value_const _ | Value_symbol _ | Closure_approximation _
      | Block_approximation _ ->
        false
    in
    if
      (not (Hashtbl.mem blocks_seen n))
      && List.for_all (fun field -> said field || hides field) fields
    then (
      let fields =
        List.map
          (fun (field : approximation) : Model.simple ->
            match field with
            | Unknown _ when hides field ->
              let made = Symbols.fresh () in
              hidden := made :: !hidden;
              Hashtbl.replace hidden_of (int_of_positive made) n;
              Symbol made
            | Value_const c -> Const (convert_const c)
            | Value_symbol symbol ->
              let field = Symbols.find symbol in
              Hashtbl.replace fields_of_blocks (int_of_positive field) ();
              (* A field that is a closure or a block is bound, asked for or
                 not. *)
              (match approximation symbol with
              | Some
                  ( unit,
                    Closure_approximation
                      { code_id; function_slot; code = _; symbol = _ } ) ->
                ignore
                  (bind_closure unit field ~function_slot ~code_id
                    : Model.code_id)
              | Some (unit, Block_approximation (tag, Value_only, fields, _)) ->
                Queue.add
                  (unit, field, tag, Array.to_list fields)
                  blocks_to_bind
              | None
              | Some
                  ( _,
                    ( Unknown _ | Value_symbol _ | Value_const _
                    | Block_approximation (_, Mixed_record _, _, _) ) ) ->
                ());
              Symbol field
            | Unknown _ | Closure_approximation _ | Block_approximation _ ->
              assert false)
          fields
      in
      Hashtbl.replace blocks n fields;
      let tag = z_of_int (Tag.Scannable.to_int tag) in
      block_bindings
        := ( sym,
             fields,
             Model.Bind_symbol (sym, Static_const_block (tag, fields)) )
           :: !block_bindings;
      use unit);
    Hashtbl.replace blocks_seen n ()
  in
  let bind_block unit sym tag fields =
    bind_one_block unit sym tag fields;
    while not (Queue.is_empty blocks_to_bind) do
      let unit, sym, tag, fields = Queue.pop blocks_to_bind in
      bind_one_block unit sym tag fields
    done
  in
  let ask_symbol sym =
    let n = int_of_positive sym in
    if not (Hashtbl.mem asked_symbols n)
    then (
      Hashtbl.replace asked_symbols n ();
      changed := true;
      match Option.bind (Symbols.name_of sym) approximation with
      | None | Some (_, Unknown _) -> ()
      | Some (_, Value_symbol _) ->
        unsupported
          "the symbol %s, which the .cmx file of its unit says is another \
           symbol"
          (shown_symbol sym)
      | Some (_, Value_const _) ->
        unsupported
          "the symbol %s, which the .cmx file of its unit says is a constant"
          (shown_symbol sym)
      | Some
          ( unit,
            Closure_approximation
              { code_id; function_slot; code = _; symbol = _ } ) ->
        ask_code (bind_closure unit sym ~function_slot ~code_id)
      | Some (unit, Block_approximation (tag, Value_only, fields, _alloc_mode))
        ->
        bind_block unit sym tag (Array.to_list fields)
      | Some (_, Block_approximation (_, Mixed_record _, _, _)) -> ())
  in
  (* What the variables bound by the [Let]s around stand for. *)
  let stands_for : (int, Model.symbol) Hashtbl.t = Hashtbl.create 64 in
  let in_simple (s : Model.simple) =
    match s with Symbol sym -> ask_symbol sym | Var _ | Const _ -> ()
  in
  let in_simples = List.iter in_simple in
  let in_slots slots = List.iter (fun (_, s) -> in_simple s) slots in
  let in_function_decls = List.iter (fun (_, code) -> ask_code code) in
  let rec in_expr (e : Model.expr) =
    match e with
    | Let (x, _, d, body) ->
      (match d with
      | Simple s -> (
        in_simple s;
        match s with
        | Symbol sym -> Hashtbl.replace stands_for (int_of_positive x) sym
        | Var _ | Const _ -> ())
      | Prim (Unary (Block_load (_, _, field)), [arg]) -> (
        in_simple arg;
        let block =
          match arg with
          | Symbol sym -> Some sym
          | Var y -> Hashtbl.find_opt stands_for (int_of_positive y)
          | Const _ -> None
        in
        let fields =
          Option.bind block (fun block ->
              Hashtbl.find_opt blocks (int_of_positive block))
        in
        let field = int_of_z field in
        match fields with
        | Some fields when field >= 0 -> (
          match List.nth_opt fields field with
          | Some (Symbol sym) ->
            (match Hashtbl.find_opt hidden_of (int_of_positive sym) with
            | Some block ->
              if not (List.mem block !read_hidden)
              then read_hidden := block :: !read_hidden
            | None -> ());
            ask_symbol sym;
            Hashtbl.replace stands_for (int_of_positive x) sym
          | Some (Var _ | Const _) | None -> ())
        | Some _ | None -> ())
      | Prim (_, args) -> in_simples args
      | Project_value_slot (_, s)
      | Project_value_slot_from (_, _, s)
      | Project_function_slot (_, _, s) ->
        in_simple s
      | Set_of_closures (code, slots) ->
        ask_code code;
        in_slots slots
      | Closure_of_set (_, function_decls, slots) ->
        in_function_decls function_decls;
        in_slots slots);
      in_expr body
    | Let_static (bindings, body) ->
      List.iter
        (fun (binding : Model.static_binding) ->
          match binding with
          | Bind_code (_, Code (_, _, _, _, _, body)) -> in_expr body
          | Bind_deleted_code _ -> ()
          | Bind_symbol (_, Static_const_set_of_closures (code, slots)) ->
            ask_code code;
            in_slots slots
          | Bind_symbol
              ( _,
                Static_const_closure_of_set (_, symbols, function_decls, slots)
              ) ->
            List.iter (fun (_, sym) -> ask_symbol sym) symbols;
            in_function_decls function_decls;
            in_slots slots
          | Bind_symbol (_, Static_const_block (_, fields)) -> in_simples fields
          | Bind_symbol (_, Static_const_boxed_number contents) ->
            in_simple contents
          | Bind_symbol (_, Static_const_immutable_string _) -> ()
          | Bind_symbol (_, Static_const_immutable_value_array elements) ->
            in_simples elements)
        bindings;
      in_expr body
    | Let_cont_non_recursive (_, Handler (_, _, handler), body) ->
      in_expr handler;
      in_expr body
    | Let_cont_recursive (handlers, body) ->
      List.iter
        (fun (_, Model.Handler (_, _, handler)) -> in_expr handler)
        handlers;
      in_expr body
    | Apply ap -> (
      in_simple ap.ap_callee;
      in_simples ap.ap_args;
      in_simples (List.map fst ap.ap_exn_extra_args);
      match ap.ap_call_kind with Direct code -> ask_code code | Indirect -> ())
    | Apply_cont ac -> in_simples ac.ac_args
    | Switch (scrutinee, arms) ->
      in_simple scrutinee;
      List.iter (fun (_, (ac : Model.apply_cont)) -> in_simples ac.ac_args) arms
    | Invalid -> ()
  in
  (* The two terms and the bodies bound so far are gone through until nothing
     more is asked for: each time round the blocks that the loads are read
     against are those bound so far. *)
  let rec settle () =
    changed := false;
    List.iter
      (fun e ->
        Hashtbl.reset stands_for;
        in_expr e)
      (terms @ List.rev !bodies);
    if !changed then settle ()
  in
  match units with
  | [] -> no_prefix
  | _ :: _ ->
    settle ();
    found
      := List.filter (fun block -> not (List.mem block refused)) !read_hidden;
    (* The closures, by groups in none of which two are of the same code. *)
    let closure_groups =
      List.fold_left
        (fun groups (code, binding) ->
          let rec place groups =
            match groups with
            | [] -> [[code, binding]]
            | group :: groups ->
              if List.mem_assoc code group
              then group :: place groups
              else (group @ [code, binding]) :: groups
          in
          place groups)
        []
        (List.rev !closure_bindings)
      |> List.map (List.map snd)
    in
    (* The blocks, each after those of its fields that are blocks. *)
    let all_blocks = List.rev !block_bindings in
    let ordered_blocks =
      let bound : (int, Model.simple list * Model.static_binding) Hashtbl.t =
        Hashtbl.create 16
      in
      List.iter
        (fun (sym, fields, binding) ->
          Hashtbl.replace bound (int_of_positive sym) (fields, binding))
        all_blocks;
      let placed : (int, unit) Hashtbl.t = Hashtbl.create 16 in
      let ordered = ref [] in
      (* The blocks whose fields are being gone through, the innermost first,
         each with the fields still to go. *)
      let entered = Stack.create () in
      let enter sym =
        let n = int_of_positive sym in
        if not (Hashtbl.mem placed n)
        then
          match Hashtbl.find_opt bound n with
          | Some (fields, binding) ->
            Hashtbl.replace placed n ();
            Stack.push (binding, ref fields) entered
          | None -> ()
      in
      List.iter
        (fun (sym, _, _) ->
          enter sym;
          while not (Stack.is_empty entered) do
            let binding, fields = Stack.top entered in
            match !fields with
            | [] ->
              ignore (Stack.pop entered : _ * _);
              ordered := binding :: !ordered
            | field :: rest -> (
              fields := rest;
              match (field : Model.simple) with
              | Symbol field -> enter field
              | Var _ | Const _ -> ())
          done)
        all_blocks;
      List.rev !ordered
    in
    let first_closures, later_closures =
      match closure_groups with [] -> [], [] | first :: later -> first, later
    in
    let groups =
      List.filter
        (fun group -> match group with [] -> false | _ :: _ -> true)
        ((List.rev !code_bindings @ first_closures @ ordered_blocks)
        :: later_closures)
    in
    let symbols =
      List.concat_map (List.concat_map Well_formed.symbols_of_binding) groups
    in
    let asked sym = Hashtbl.mem asked_symbols (int_of_positive sym) in
    (* What is bound and not asked for, with the block asked for from which it
       is reached through blocks that are not asked for. *)
    let unasked =
      let reached : (int, unit) Hashtbl.t = Hashtbl.create 64 in
      let unasked = ref [] in
      List.iter
        (fun (root, fields, _) ->
          if asked root
          then (
            let to_visit = Stack.create () in
            Stack.push fields to_visit;
            while not (Stack.is_empty to_visit) do
              List.iter
                (fun (field : Model.simple) ->
                  match field with
                  | Symbol sym ->
                    let n = int_of_positive sym in
                    let fields = Hashtbl.find_opt blocks n in
                    if
                      (Option.is_some fields || Hashtbl.mem closures n)
                      && (not (asked sym))
                      && not (Hashtbl.mem reached n)
                    then (
                      Hashtbl.replace reached n ();
                      unasked := (sym, root) :: !unasked;
                      Option.iter
                        (fun fields -> Stack.push fields to_visit)
                        fields)
                  | Var _ | Const _ -> ())
                (Stack.pop to_visit)
            done))
        all_blocks;
      List.rev !unasked
    in
    { prefix_groups = groups;
      prefix_units =
        List.sort
          (fun (unit, _) (unit', _) ->
            String.compare
              (Compilation_unit.name_as_string unit)
              (Compilation_unit.name_as_string unit'))
          !used;
      prefix_symbols = symbols;
      prefix_unasked = unasked;
      prefix_unmentioned =
        List.filter_map
          (fun (_, sym) ->
            if asked sym || Hashtbl.mem fields_of_blocks (int_of_positive sym)
            then None
            else Some sym)
          (Symbols.bindings ());
      prefix_hidden = List.rev !hidden
    }

(* The prefix is made with no block refused, and made again with the blocks
   found added to those refused until none is found. Both terms are gone through
   each time, and one prefix goes in front of both, so a block is refused for
   both or for neither; a block once refused stays so; and in the prefix
   returned no block that is bound has such a field that a term loads from
   something that stands for the block. *)
let make_prefix ~units ~loader ~units_have_tupled_code terms =
  let rec again refused =
    let found = ref [] in
    let prefix =
      make_prefix_refusing ~refused ~found ~units ~loader
        ~units_have_tupled_code terms
    in
    match !found with [] -> prefix | _ :: _ as found -> again (found @ refused)
  in
  again []

(* A unit with the prefix in front of it. *)
let with_prefix (prefix : prefix) (unit : converted) : converted =
  { unit with
    body =
      List.fold_right
        (fun group body -> Model.Let_static (group, body))
        prefix.prefix_groups unit.body
  }

(* What is wrong with a prefix, if something is, given the imports that are left
   and the two units that it goes in front of: it must bind each symbol and each
   code id once, none that either unit binds and none that is still an import,
   and it must evaluate from the static environment of the imports. Each of
   these fails only where [make_prefix] is wrong, as no source makes a unit that
   binds a name of another unit. *)
let prefix_fault (prefix : prefix)
    ~(imports : (Model.symbol * Values.import_class) list) (before : converted)
    (after : converted) : string option =
  let names_of names =
    List.concat_map (List.concat_map names) prefix.prefix_groups
  in
  let symbols = names_of Well_formed.symbols_of_binding in
  let code = names_of Well_formed.code_ids_of_binding in
  let set_of names =
    let set : (int, unit) Hashtbl.t = Hashtbl.create 64 in
    List.iter (fun name -> Hashtbl.replace set (int_of_positive name) ()) names;
    set
  in
  let twice names = Hashtbl.length (set_of names) < List.length names in
  let of_units names =
    set_of
      (Well_formed.static_names names before.body
      @ Well_formed.static_names names after.body)
  in
  let among set name = Hashtbl.mem set (int_of_positive name) in
  if twice symbols || twice code
  then Some "binds a symbol or a code id twice"
  else if
    List.exists (among (of_units Well_formed.symbols_of_binding)) symbols
    || List.exists (among (of_units Well_formed.code_ids_of_binding)) code
  then Some "binds a symbol or a code id that the unit binds"
  else if List.exists (among (set_of (List.map fst imports))) symbols
  then Some "binds a symbol that is still among the imports"
  else
    match
      List.fold_left
        (fun statics group ->
          Option.bind statics (Semantics.eval_static_bindings [] group))
        (Some (Imports.imports_statics imports))
        prefix.prefix_groups
    with
    | Some (_ : Values.static_env) -> None
    | None ->
      Some "does not evaluate from the static environment of the imports"

(* The prefix, for reading it against the units that it is made from: the units,
   and the bindings with the names that the compiler gives ([names_shown]). The
   closures and the blocks that are bound without being asked for are counted,
   by the block asked for from which they are reached, on one line after the
   group that binds them; they are printed one by one, with the rest, where
   FLSEMANTICS_PRINT_TERMS is set. The digests of the [.cmx] files are printed
   only where FLSEMANTICS_PRINT_DIGESTS is set: the file of a test is made by
   the build, and its digest moves with the compiler. *)
let print_prefix (prefix : prefix) =
  match prefix.prefix_groups with
  | [] -> ()
  | _ :: _ ->
    let digests = Sys.getenv_opt "FLSEMANTICS_PRINT_DIGESTS" <> None in
    let every_binding = Sys.getenv_opt "FLSEMANTICS_PRINT_TERMS" <> None in
    let bindings = List.length (List.concat prefix.prefix_groups) in
    Format.printf "  given the .cmx of %s, a prefix of %d static binding%s:@."
      (String.concat ", "
         (List.map
            (fun (unit, digest) ->
              Compilation_unit.name_as_string unit
              ^ if digests then " (" ^ Digest.to_hex digest ^ ")" else "")
            prefix.prefix_units))
      bindings
      (if bindings = 1 then "" else "s");
    let unasked : (int, Model.symbol) Hashtbl.t = Hashtbl.create 64 in
    if not every_binding
    then
      List.iter
        (fun (sym, block) ->
          Hashtbl.replace unasked (int_of_positive sym) block)
        prefix.prefix_unasked;
    let number n what =
      Printf.sprintf "%d %s%s" n what (if n = 1 then "" else "s")
    in
    names_shown := true;
    Fun.protect
      ~finally:(fun () -> names_shown := false)
      (fun () ->
        List.iteri
          (fun i group ->
            if i > 0 then Format.printf "    and then@.";
            (* The blocks asked for from which something of this group that
               nothing asks for is reached, in the order of the first such
               binding of each, with the numbers of closures and of blocks so
               reached. *)
            let counted : (int, int ref * int ref) Hashtbl.t =
              Hashtbl.create 16
            in
            let blocks = ref [] in
            List.iter
              (fun (binding : Model.static_binding) ->
                let unasked =
                  match binding with
                  | Bind_symbol (sym, const) ->
                    Option.map
                      (fun block -> block, const)
                      (Hashtbl.find_opt unasked (int_of_positive sym))
                  | Bind_code _ | Bind_deleted_code _ -> None
                in
                match unasked with
                | Some (block, const) -> (
                  let closures, others =
                    match Hashtbl.find_opt counted (int_of_positive block) with
                    | Some counts -> counts
                    | None ->
                      let counts = ref 0, ref 0 in
                      Hashtbl.replace counted (int_of_positive block) counts;
                      blocks := block :: !blocks;
                      counts
                  in
                  match const with
                  | Static_const_set_of_closures _
                  | Static_const_closure_of_set _ ->
                    incr closures
                  | Static_const_block _ | Static_const_boxed_number _
                  | Static_const_immutable_string _
                  | Static_const_immutable_value_array _ ->
                    incr others)
                | None ->
                  forget_shown_names ();
                  Format.printf "    @[<v>%a@]@." print_binding binding)
              group;
            List.iter
              (fun block ->
                let closures, others =
                  Hashtbl.find counted (int_of_positive block)
                in
                Format.printf "    and %s of %s that nothing asks for@."
                  (String.concat " and "
                     ((if !closures > 0
                       then [number !closures "closure"]
                       else [])
                     @ if !others > 0 then [number !others "block"] else []))
                  (shown_symbol block))
              (List.rev !blocks))
          prefix.prefix_groups)

(* The two units are converted with one numbering of names, the continuations of
   the second unit standing for those of the first, so that the verified
   comparison can be made of the two bodies as they are.

   Simplify gives a new code id to each new version of a piece of code, and
   binds the old one as deleted. The comparison wants code ids to be the same,
   so the new id is given the number of the old where that changes nothing:
   where the unit after binds the old one as deleted and does not otherwise use
   it. The unit after is converted once to find that out, and again with the
   numbers settled; the second time, the declaration that such an old one is
   deleted is left out, as its number now stands for the new code.

   The prefix is made last, of what the two terms as converted ask for, and with
   their numbering ([make_prefix]); [units] are the units whose [.cmx] file the
   run of Simplify read. The two units are given back without it.

   CR mshinwell: This identification is part of what is trusted, and will most
   likely not be enough: it does nothing for code that has no older version, or
   whose older version is still used. The comparison should relate code ids
   itself; see the CR in middle_end/flambda2/rocq/FlambdaCompareSpec.v for what
   that takes. *)
let convert_both ~(units : (Compilation_unit.t * Digest.t) list)
    (before : Flambda_unit.t) (after : Flambda_unit.t) =
  let loader =
    lazy (Flambda_cmx.create_loader ~get_module_info:Flambda2.get_module_info)
  in
  superseded_code := [];
  let has_tupled_code () = match !tupled_code with [] -> false | _ -> true in
  let before_converted = convert_unit before in
  let before_has_tupled_code = has_tupled_code () in
  Continuations.alias
    (Flambda_unit.return_continuation after)
    ~same_as:(Flambda_unit.return_continuation before);
  Continuations.alias
    (Flambda_unit.exn_continuation after)
    ~same_as:(Flambda_unit.exn_continuation before);
  newer_versions := [];
  deleted_code := [];
  superseded_code := [];
  let first_pass = convert_unit after in
  List.iter
    (fun (newer, older) ->
      if
        List.exists (Code_id.equal older) !deleted_code
        && not (code_id_used (Code_ids.find older) first_pass.body)
      then (
        Code_ids.alias newer ~same_as:older;
        superseded_code := older :: !superseded_code))
    !newer_versions;
  let after_converted =
    Fun.protect
      ~finally:(fun () -> superseded_code := [])
      (fun () -> convert_unit after)
  in
  let prefix =
    make_prefix ~units ~loader
      ~units_have_tupled_code:(before_has_tupled_code || has_tupled_code ())
      [before_converted.body; after_converted.body]
  in
  before_converted, after_converted, prefix

(* What the check of kinds answers for a unit (FlambdaWellKindedSpec.v), given
   the kinds of the value slots met in the two units. *)
let well_kinded (unit : converted) =
  Well_kinded.well_kinded_unit
    (List.map (fun (slot, kind) -> slot, convert_kind kind) !slot_kinds)
    unit.return_continuation unit.exn_continuation unit.body

(* Static bindings at toplevel only, each name bound once, the sets of closures
   whole, the numbers in range, exception handlers apart from the other
   continuations, and the discipline of trap actions
   (FlambdaWellFormedSpec.v). *)
let well_formed (unit : converted) =
  Well_formed.well_formed_unit
    { unit_return = unit.return_continuation; unit_exn = unit.exn_continuation }
    unit.body

(* A condition of well-formedness that a unit which is not well-formed fails. *)
let why_not_well_formed (unit : converted) =
  let exn = unit.exn_continuation in
  if not (Well_formed.numbers_in_range unit.body)
  then
    (* The compiler's numbers are in range by their types, so this is a fault of
       the conversion. *)
    "a number out of its range: the tag of a block, a constant, the field or \
     the size of a block load, a discriminant or a byte of a string"
  else if not (Binders.names_in_scope unit.return_continuation exn unit.body)
  then "a variable or a continuation that is used where nothing binds it"
  else if not (Binders.binders_distinct unit.body)
  then
    "a name bound twice by one binder, the closure of code among its \
     parameters, or a recursive group of no continuation"
  else if not (Binders.code_closed unit.body)
  then "code that uses a variable or a continuation that it does not bind"
  else if not (Well_formed.sets_well_formed unit.body)
  then
    "a closure at a function slot that its set has not got, a function slot or \
     a code id given twice in a set, the same code in two static sets of one \
     group, or a static set of closures that is not bound whole"
  else if not (Well_formed.unit_exn_handlers_well_formed exn unit.body)
  then
    "an exception handler that is applied as an ordinary continuation, takes \
     no parameter or is recursive; a continuation that is used as an exception \
     handler and is not marked as one; a continuation that the unit or code \
     does not bind and that is used as an exception handler without being its \
     exception continuation, or is given extra arguments; or the exception \
     continuation of the unit or of code named by a trap action that is not \
     that of a raise"
  else if not (Well_formed.traps_well_formed exn unit.body)
  then
    "a trap action, a raise or a call that names a handler other than the one \
     installed last, or a continuation entered with two stacks of handlers"
  else "a static binding that is not at toplevel, or a name bound twice"

(* What a unit is given by where it is run: the continuation that it returns to
   and its exception continuation. *)
let interface (unit : converted) : Well_formed.unit_interface =
  { unit_return = unit.return_continuation; unit_exn = unit.exn_continuation }

(* The text of a term does not give the arity of a call of unknown arity, and
   such a call is read with one value for its result (the CRs of
   parser/fexpr_to_flambda.ml; label C2-5). Where the result of the compiler's
   call was something else, an unboxed pair or an unboxed number, the term read
   is not the compiler's, and Simplify, which is entitled to, stops on it with a
   fatal error. This says whether the text has such a call that can be told: a
   call of unknown arity that returns to a continuation whose parameters are not
   one value, where the continuation is bound in sight by a [Let_cont] or is the
   return continuation of the code that the call is in. A call is of unknown
   arity as the reader has it: one with no arity written, or with the arity of
   its result written and not that of its arguments, which the reader drops. The
   test goes by the identifiers of the text as it was parsed and makes no name
   of the compiler. *)
let result_arity_lost (unit : Fexpr.flambda_unit) : bool =
  let one_value (kinds : Fexpr.kind_with_subkind option list) =
    match kinds with
    | [(None | Some (Value _))] -> true
    | [Some (Naked_number _ | Region | Rec_info)] | [] | _ :: _ :: _ -> false
  in
  (* [others] are the names of the continuations in sight whose parameters are
     not one value; a name that is bound again hides the one before. *)
  let bind (name : Fexpr.continuation_id) kinds others =
    let others =
      List.filter (fun other -> not (String.equal other name.txt)) others
    in
    if one_value kinds then others else name.txt :: others
  in
  let rec expr others (e : Fexpr.expr) =
    match e with
    | Let { bindings = _; value_slots = _; body } -> expr others body
    | Let_cont { recursive; body; bindings } ->
      let invariant_params =
        match recursive with
        | Nonrecursive -> []
        | Recursive invariant_params -> invariant_params
      in
      let with_bindings =
        List.fold_left
          (fun others (binding : Fexpr.continuation_binding) ->
            bind binding.name
              (List.map
                 (fun (param : Fexpr.kinded_parameter) -> param.kind)
                 (invariant_params @ binding.params))
              others)
          others bindings
      in
      let in_handlers =
        match recursive with
        | Nonrecursive -> others
        | Recursive _ -> with_bindings
      in
      List.exists
        (fun (binding : Fexpr.continuation_binding) ->
          expr in_handlers binding.handler)
        bindings
      || expr with_bindings body
    | Let_symbol { bindings; value_slots = _; body } ->
      List.exists
        (fun (binding : Fexpr.symbol_binding) ->
          match binding with
          | Code code ->
            let result : Fexpr.kind_with_subkind option list =
              match code.ret_arity with
              | None -> [None]
              | Some arity -> List.map Option.some arity
            in
            expr
              (bind code.params_and_body.ret_cont result [])
              code.params_and_body.body
          | Data _ | Deleted_code _ | Closure _ | Set_of_closures _ -> false)
        bindings
      || expr others body
    | Apply
        { call_kind = Function Indirect;
          arities = None | Some { params_arity = None; ret_arity = _ };
          continuation = Return (Named k);
          _
        } ->
      List.mem k.txt others
    | Apply _ | Apply_cont _ | Invalid -> false
    | Switch { scrutinee = _; cases } ->
      List.exists
        (fun ((_, case) : int * Fexpr.apply_or_inlined_cont) ->
          match case with
          | Inlined_goto e -> expr others e
          | Named_cont _ -> false)
        cases
  in
  expr [] unit.body

let result_arity_lost_message =
  "a call of unknown arity returns to a continuation whose parameters are not \
   one value; the text does not give the arity of such a call, which is read \
   with one value for its result"

(* The term before is checked before Simplify sees it, which is entitled to fail
   on one that is not well-formed. *)
let well_formed_before ~filename (before : Flambda_unit.t) =
  match convert_unit before with
  | exception Unsupported _ -> true
  | before ->
    let ok = well_formed before in
    if not ok
    then
      Format.printf "%s: not checked: the term is not well-formed (%s)@."
        filename
        (why_not_well_formed before);
    ok

(* The comparison of the two runs is a test of those runs; this is a verdict
   that has a theorem. The theorem is of the two units run from the empty static
   environment and, for some verdicts, of the two units run from their imports
   (the header, under "Imports"). The two runs compared start from the imports,
   so for a unit that has imports the verdict is of those two runs only where
   its label says so. The whole check is [validate_unit], extracted from
   FlambdaValidateSpec.v, where it is proved that what it accepts is correct for
   the unit run on its own from empty environments. It normalises the two terms,
   compares them up to bound names and, failing that, removes from the term
   before the static bindings named here and compares again. Which names those
   are is a guess that it checks. Failing that, it removes those of them that
   the term after binds at its head from the term after too, under checks of its
   own, and compares again; a unit accepted so has a verdict of its own. Failing
   that too, it is asked again, with the [Let]s that Simplify seems to have made
   static bindings, in one or more liftings: it lifts them in the term before as
   it was given, if it may, normalises the result, reads the value slots of
   static closures and uses what the unit binds its symbols to, removes the
   value slots of its static sets of closures that the term after has not got,
   moves the bindings of code of both terms up, and compares again. What is
   lifted and which slots are guesses too. Before any of this, where the two
   terms are not the same as they are given, [validate_unit] makes indirect the
   direct calls in the code of the stubs of the term before, as Simplify does
   where it simplifies that code (FlambdaStubCallsSpec.v), puts each direct call
   of code that never returns, under the checks of that rewrite, under a binding
   of its return continuation to a handler that is [Invalid]
   (FlambdaNeverReturnsSpec.v), and makes jumps of the self tail calls of the
   code named here, as Simplify does for the code that the front end marks for
   loopify; which code that is, with a name for each loop, is a guess again, and
   the guesses above are made of the term that this gives. Failing all of that,
   the whole is asked a second time, of [validate_unit_knowing], extracted from
   FlambdaValidateKnowingSpec.v, where it is proved that what it accepts is
   correct for the unit in the same way. It is [validate_unit] of the two terms
   normalised by rounds in front of each of which the aliases are followed
   knowing what the symbols are bound to; the guesses are made again, from those
   two terms. *)

(* The code that the front end marks for loopify, each with a name for its loop
   that no continuation of the two units has. *)
let loopify_guess () =
  List.map
    (fun code_id -> Code_ids.find code_id, Continuations.fresh ())
    (List.sort_uniq Code_id.compare !loopify_marked)

let validate ~(imports : (Model.symbol * Values.import_class) list)
    ~(prefix_units : int) ~(outcome_before : outcome) (before : converted)
    (after : converted) =
  let recursive = recursive_code before.body @ recursive_code after.body in
  let inlined = Normalise.inline_calls recursive [] before.body in
  Format.printf
    "  calls outside code once those of known code are inlined: %d@."
    (calls_outside_code inlined);
  let rounds = nat_of_int 50 in
  let nothing_lifted : Lift.lifting =
    { lifting_unit = interface before;
      lifted_lets = [];
      lifted_my_closures = [];
      lifted_out_of_code = [];
      lifted_to_head = [];
      lifted_around = [];
      groups_put = [];
      tables_put = [];
      code_copied = [];
      made_closures_called = false
    }
  in
  (* The code to loopify. *)
  let loops = loopify_guess () in
  (* Whether the liftings of the last asking bound tables at the head of the
     term before: read for the line of a verdict that lifts. Each asking sets it
     to false first. *)
  let tables_were_bound = ref false in
  (* Whether the term before has a [Let] of a closure at a function slot of a
     set, in its code too. Where it has none, no rewrite of the validation makes
     one, so the rounds that make direct the calls of the closures that a [Let]
     makes (FlambdaValidateSpec.normalise_lifted_calling) are the rounds without
     that, and the askings are not made again with them. It is a shortcut of
     this search and nothing rests on it. *)
  let closures_to_call = lets_closure_of_set before.body in
  (* The validation asked once, with the code whose calls it is told to leave.
     [validate_unit] is the function asked, which has the two terms of the unit
     already and takes the other guesses; [before] and [after] are the two terms
     that it compares once it has made the loops and normalised, from which the
     guesses are made. It gives the verdict; a term before that the function
     asked compared, with how it was made, to show where a unit that is not
     validated differs; whether value slots were removed from it; and the term
     after as it was compared with it. Which pair: where nothing is guessed to
     lift, that of the fourth comparison (the term before normalised, with its
     unused static bindings removed, and the term after with those of its head
     removed); where something is lifted and nothing to the head of the unit,
     that of the fifth (the bindings of code of both terms moved up); where
     something is lifted to the head, that of the sixth, which is the last. Each
     term shown has its continuations placed and is tidied, as [validate_unit]
     does to each term that it compares after the first comparison
     ([tidied_before], [tidied_after]). The liftings are guessed from the terms
     as the rounds leave them; the value slots and the static bindings to remove
     from a lifted term are guessed from that term placed and tidied.
     [with_lifting] is the function that applies a lifting in the search: that
     of the validator, unless the function asked makes its liftings by another.
     [calling] says whether the liftings ask for
     FlambdaValidateSpec.normalise_lifted_calling, which call closures that a
     [Let] makes directly. *)
  let ask ~calling ?(with_lifting = Validate.with_lifting) validate_unit
      (before : converted) (after : converted) kept =
    (* No table is bound until a first lifting of this asking binds one: a
       verdict of this asking that is reached with no lifting proposed must not
       read what an asking before it left. *)
    tables_were_bound := false;
    (* The term before with the direct calls of the code of its stubs made
       indirect, its direct calls of code that never returns which pass the
       checks of that rewrite put under a binding of their return continuation
       to a handler that is [Invalid], and its self tail calls made jumps: the
       term that the function asked normalises and lifts in. *)
    let looped =
      Loopify.loopify_unit loops
        (Never_returns.cut_returns []
           (Stub_calls.demote_stub_calls before.body))
    in
    let normalised_after = Validate.normalise_after rounds kept after.body in
    let tidied_after = Validate.tidied_after rounds normalised_after in
    (* Whether the calls of the term before are taken on when its rounds are
       over, as [validate_unit] decides it: not where the term after, as it is
       compared, has a call that would be. *)
    let taken = not (Validate.keeps_a_return tidied_after) in
    let normalised_before =
      Validate.with_returns taken (Validate.normalise_before rounds kept looped)
    in
    (* The names that the term after binds and does not use are added to those
       of the term before, where the term before can do without them too: the
       one list is for both terms (FlambdaValidateSpec.without_head_statics). *)
    let unused_after = unused_statics tidied_after in
    let both (before_term : Model.expr) (unused : Unused_statics.dropped) :
        Unused_statics.dropped =
      let more =
        { Unused_statics.dropped_symbols =
            unused.dropped_symbols
            @ List.filter
                (fun sym -> not (List.mem sym unused.dropped_symbols))
                unused_after.dropped_symbols;
          dropped_code =
            unused.dropped_code
            @ List.filter
                (fun id -> not (List.mem id unused.dropped_code))
                unused_after.dropped_code
        }
      in
      match Unused_statics.drop_statics more before_term with
      | Some _ -> more
      | None -> unused
    in
    let unused =
      both
        (Validate.tidied_before rounds normalised_before)
        (unused_statics normalised_before)
    in
    let validation : Validate.validation =
      validate_unit unused [] [nothing_lifted]
    in
    let not_lifted =
      ( validation,
        Validate.without_statics unused
          (Validate.tidied_before rounds normalised_before),
        "normalised",
        false,
        Validate.without_head_statics unused tidied_after )
    in
    match validation with
    | Not_validated -> (
      (* The liftings, in turn, as [validate_unit] makes them. The first is
         guessed from the term before after its self tail calls are made jumps;
         each one after it from the term that those before it give, normalised:
         for the [Let]s that inlining a call has brought out of a function, and
         for those to make aliases of symbols bound around them, which are
         guessed from the second lifting on. The tables to bind at the head of
         the unit are guessed for the first. A second is always tried; one after
         it only if it lifts something; and there are four at most. After each,
         as [validate_unit] does, the value slots of static closures are read
         and what the unit binds its symbols to is used in every round of the
         normalisation. The rounds are told, as [validate_unit] tells them, not
         to make a name that is one of the two continuations of the unit: every
         lifting has the unit of the term before, and a name made is the one
         after the greatest of those to avoid and of the term, so the names made
         here are those that [validate_unit] makes whatever the number of
         liftings it is given. [banned] has, for the liftings by their place in
         the sequence, the [Let]s that a symbol is not to go to. *)
      let avoid =
        let unit = interface before in
        [unit.unit_return; unit.unit_exn]
      in
      let rec liftings_from banned stage made lifted remaining =
        let lifting =
          lifting_guess ~unit:(interface before)
            ~banned:
              (List.filter_map
                 (fun (stage', path, sym) ->
                   if stage' = stage then Some (path, sym) else None)
                 banned)
            ~around:(match made with [] -> false | _ :: _ -> true)
            lifted normalised_after
        in
        let lifting = { lifting with made_closures_called = calling } in
        (* The first lifting also binds, at the head of the unit, the tables
           that the term after has and the term before has not. *)
        let lifting =
          match made with
          | [] ->
            (* A symbol that this lifting gives to a [Let] is not a table: an
               immutable array of the source that is lifted is given its binding
               by this lifting already. *)
            let given =
              List.filter_map
                (fun (_, (target : Lift.lift_to)) ->
                  match target with
                  | Lift_to_symbol sym -> Some sym
                  | Lift_to_closure_symbols _ -> None)
                (lifting.lifted_lets @ lifting.lifted_to_head
                @ List.concat_map snd lifting.lifted_out_of_code)
            in
            { lifting with
              tables_put =
                List.filter
                  (fun (sym, _) -> not (List.mem sym given))
                  (tables_guess lifted normalised_after)
            }
          | _ :: _ -> lifting
        in
        let stop =
          match
            ( remaining,
              made,
              lifting.lifted_lets,
              lifting.lifted_my_closures,
              lifting.lifted_out_of_code,
              lifting.lifted_to_head,
              lifting.lifted_around )
          with
          | 0, _, _, _, _, _, _ | _, _ :: _, [], [], [], [], [] -> true
          | _, _, _, _, _, _, _ -> false
        in
        if stop
        then List.rev made, lifted
        else
          (* As [validate_unit] applies a lifting: to the head of the unit, out
             of code, where the [Let]s stand, then to groups around them, and
             last the tables. After the first lifting, the sequence ends here if
             the term comes back as the same value in memory. That comparison of
             two pointers is a shortcut of this search, which is bounded and not
             trusted. It says nothing of the checks of a lifting: a lifting
             whose checks pass may give the same term back, and then the search
             stops; one that gives a term rebuilt but equal may continue. *)
          let lifted' = with_lifting lifting lifted in
          (match made with
          | [] -> tables_were_bound := tables_bound lifting.tables_put lifted'
          | _ :: _ -> ());
          if lifted' == lifted && match made with [] -> false | _ :: _ -> true
          then List.rev made, lifted
          else
            liftings_from banned (stage + 1) (lifting :: made)
              (Validate.normalise_lifted_calling calling rounds avoid kept
                 lifted')
              (remaining - 1)
      in
      (* The guess gives a symbol to the first [Let] that makes a block of its
         shape, and a field that is a variable agrees with anything. So the
         symbol may go to a block that the normalisation then finds no use for
         (the argument of a match that is resolved, say), where the term after
         binds it to a block made later. Such a symbol is not used in the term
         lifted: the sequence is guessed again with that [Let] banned for it, a
         few times at most. A symbol that the term after does not bind is one
         that the guess made for a constant that Simplify has no use for, and is
         not such a symbol. *)
      let bound_after = snd (static_names normalised_after) in
      let rec guessed banned again =
        let liftings, lifted = liftings_from banned 0 [] looped 4 in
        (* After the last lifting [validate_unit] reads the known value slots
           with every symbol of the unit in scope, demotes the exception
           handlers that no call can raise to and uses what is known of the
           symbols again: the case of no lifting left of
           [Validate.lifted_in_turn], which is called here for that case, so
           that the term guessed from is the one that [validate_unit] computes.
           Then it takes the calls on where it does for the term that is not
           lifted, places the continuations and tidies. *)
        let lifted =
          Validate.tidied_before rounds
            (Validate.with_returns taken
               (Validate.lifted_in_turn rounds avoid [] kept lifted))
        in
        let slots = unused_slots_guess lifted normalised_after in
        let lifted = Validate.without_slots slots lifted in
        let unused = unused_statics lifted in
        let wrong =
          List.concat
            (List.mapi
               (fun stage (lifting : Lift.lifting) ->
                 List.filter_map
                   (fun (path, (target : Lift.lift_to)) ->
                     match target with
                     | Lift_to_symbol sym
                       when List.mem sym unused.dropped_symbols
                            && List.mem (Symbol_name sym) bound_after ->
                       Some (stage, path, sym)
                     | Lift_to_symbol _ | Lift_to_closure_symbols _ -> None)
                   lifting.lifted_lets)
               liftings)
        in
        match wrong with
        | _ :: _ when again > 0 -> guessed (wrong @ banned) (again - 1)
        | _ -> liftings, lifted, slots, unused
      in
      match guessed [] 5 with
      | [], _, _, _ -> not_lifted
      | (_ :: _ as liftings), lifted, slots, unused ->
        let unused = both lifted unused in
        let tidied_after = Validate.without_head_statics unused tidied_after in
        let validation = validate_unit unused slots liftings in
        let slots_removed = match slots with [] -> false | _ :: _ -> true in
        if
          List.exists
            (fun (lifting : Lift.lifting) ->
              match lifting.lifted_to_head, lifting.lifted_around with
              | [], [] -> false
              | _ :: _, _ | _, _ :: _ -> true)
            liftings
        then
          (* Something was lifted to the head of the unit or to a group around:
             the pair shown is that of the last comparison, of the two terms
             with their static bindings that mention no variable bound first. An
             earlier comparison may be the one that holds: the verdict is that
             of the function asked alone. *)
          ( validation,
            Validate.with_statics_first liftings
              (Validate.without_statics unused lifted),
            "lifted and normalised, with the static bindings that mention no \
             variable first",
            slots_removed,
            Validate.with_statics_first liftings tidied_after )
        else
          ( (* The pair shown is that of the fifth comparison: the bindings of
               code of both terms are moved up. *)
            validation,
            Float_code.float_code (Validate.without_statics unused lifted),
            "lifted and normalised",
            slots_removed,
            Float_code.float_code (Validate.with_code_first tidied_after) ))
    | Same_up_to_bound_names | Same_once_normalised
    | Same_once_unused_statics_removed | Same_once_head_statics_removed
    | Same_once_lifted ->
      not_lifted
  in
  (* One attempt, with the code whose calls the validation is told to leave:
     [validate_unit] is asked of the two terms and, where it fails,
     [validate_unit_knowing] is asked of the same two terms. That one compares
     the two terms normalised with what their symbols are bound to used, each by
     its own round, so the guesses for it are made from those, which are
     computed here a second time for the purpose. The round of the term before
     avoids the continuations of the units of the liftings and those of the
     loops; every lifting made here has the unit of the term before, and the
     names that the round makes depend only on the greatest of the names to
     avoid and of the term, so the term that the guesses are made from is the
     one that [validate_unit_knowing] compares. Nothing rests on that but
     whether a guess fits: the verdict is that of [validate_unit_knowing] on the
     two terms as converted. *)
  let asked_again = ref false in
  (* Whether the model says that the verdict of the asking made last holds of
     the unit run from its imports:
     FlambdaValidateImportsSpec.verdict_holds_from for [validate_unit] and
     verdict_knowing_holds_from for [validate_unit_knowing], each given the
     static environment of the imports, the arguments that the asking was given,
     and its verdict. It is set where the asking is made, beside it, and
     computed only where it is read. *)
  let statics_of_imports = Imports.imports_statics imports in
  let last_asking_holds = ref (fun () -> false) in
  let attempt ~calling ?(again = true) ~reading kept =
    (* [reading]: the attempt is of the term before with its known arrays read.
       The functions asked are the two of FlambdaValidateReadingSpec.v, each
       given the term before as converted, as the two of an attempt that is not
       [reading] are. They give [validate_unit] and [validate_unit_knowing], for
       the term before, the term that FlambdaValidateReadingSpec.before_read
       makes of it, given the rounds, the loops, the liftings and the list of
       this attempt. That term is computed here too. [term] is it with nothing
       lifted, for the guesses, which are made from it: the liftings are not
       guessed yet, and they change the term only by the continuations that its
       rounds are told not to name, which are those of the unit of the term
       before whatever the liftings are. [read liftings] is it for the liftings
       of an asking, for the test of the verdict of that asking from the
       imports, which the statements of that file have of that term. Of an
       attempt that is not [reading] both are the term before as converted. *)
    let read liftings =
      if reading
      then
        { before with
          body =
            Validate_reading.before_read rounds loops liftings kept before.body
        }
      else before
    in
    let term = read [nothing_lifted] in
    let validate_unit, validate_unit_knowing =
      if reading
      then
        ( Validate_reading.validate_unit_reading,
          Validate_reading.validate_unit_knowing_reading )
      else Validate.validate_unit, Validate_knowing.validate_unit_knowing
    in
    let first =
      ask ~calling
        (fun unused slots liftings ->
          let validation =
            validate_unit rounds unused slots loops liftings kept before.body
              after.body
          in
          (last_asking_holds
             := fun () ->
                  Validate_imports.verdict_holds_from statics_of_imports rounds
                    unused slots loops liftings kept (read liftings).body
                    after.body validation);
          validation)
        term after kept
    in
    match first with
    | Not_validated, _, _, _, _ when again -> (
      let knowing round (unit : converted) =
        { unit with
          body = Validate_knowing.normalise_knowing round rounds unit.body
        }
      in
      let avoid =
        Validate.unit_continuations [nothing_lifted] @ List.map snd loops
      in
      match
        ask ~calling
          (fun unused slots liftings ->
            let validation =
              validate_unit_knowing rounds unused slots loops liftings kept
                before.body after.body
            in
            (last_asking_holds
               := fun () ->
                    Validate_imports.verdict_knowing_holds_from
                      statics_of_imports rounds unused slots loops liftings kept
                      (read liftings).body after.body validation);
            validation)
          (knowing (Validate.normalise_before_round_avoiding avoid kept) term)
          (knowing (Validate.normalise_after_round kept) after)
          kept
      with
      | Not_validated, _, _, _, _ -> first
      | ( ( Same_up_to_bound_names | Same_once_normalised
          | Same_once_unused_statics_removed | Same_once_head_statics_removed
          | Same_once_lifted ),
          _,
          _,
          _,
          _ ) as second ->
        asked_again := true;
        second)
    | ( ( Not_validated | Same_up_to_bound_names | Same_once_normalised
        | Same_once_unused_statics_removed | Same_once_head_statics_removed
        | Same_once_lifted ),
        _,
        _,
        _,
        _ ) ->
      first
  in
  (* Which calls the validation is told to leave: three guesses, tried in turn.
     First the recursive code and the code that the term after still calls
     directly, which Simplify chose not to inline. Then the recursive code
     alone, for a function that Simplify inlines at one call and not at another:
     there every call inlined on both sides is what makes the two terms the
     same. Then the first guess again, with the recursive code of the term
     before taken from the term that the validation normalises: the direct calls
     of its stubs made indirect, its calls of code that never returns cut, its
     loops made. Code whose self tail calls have all become jumps is not
     recursive there, and Simplify inlines it where it would inline any other:
     that is the case that the guess is for. The third is tried only where some
     code of the first list is not in it. It is asked once: the second asking
     normalises the term before as it is given, before its loops are made, and
     each of its rounds would inline such code into itself once more. *)
  let called_after = called_directly after.body in
  let first_guess = recursive @ called_after in
  let once_looped_guess =
    recursive_code
      (Loopify.loopify_unit loops
         (Never_returns.cut_returns []
            (Stub_calls.demote_stub_calls before.body)))
    @ recursive_code after.body @ called_after
  in
  let another_guess =
    List.exists
      (fun code_id -> not (List.mem code_id once_looped_guess))
      first_guess
  in
  (* The attempts in turn: of the term before as it is, or with its known arrays
     read ([reading]); and without or with the calls of the closures that a
     [Let] makes made direct in the rounds that follow each lifting ([calling]:
     FlambdaValidateSpec.normalise_lifted_calling, which the liftings of every
     asking then ask for). *)
  let attempts_of ~calling ~reading =
    let attempt = attempt ~calling in
    let first = attempt ~reading first_guess in
    match first with
    | Not_validated, _, _, _, _ -> (
      let second = attempt ~reading recursive in
      let two =
        [ ( "leaving the calls of recursive code and of the code that the term \
             after calls",
            first );
          "leaving the calls of recursive code", second ]
      in
      match second with
      | Not_validated, _, _, _, _ when another_guess ->
        let third = attempt ~again:false ~reading once_looped_guess in
        ( two
          @ [ ( "leaving the calls of the code that is recursive once the \
                 loops are made and of the code that the term after calls",
                third ) ],
          third )
      | ( ( Not_validated | Same_up_to_bound_names | Same_once_normalised
          | Same_once_unused_statics_removed | Same_once_head_statics_removed
          | Same_once_lifted ),
          _,
          _,
          _,
          _ ) ->
        two, second)
    | ( ( Same_up_to_bound_names | Same_once_normalised
        | Same_once_unused_statics_removed | Same_once_head_statics_removed
        | Same_once_lifted ),
        _,
        _,
        _,
        _ ) ->
      [], first
  in
  (* The attempts of one kind: without the calls of the closures that a [Let]
     makes made direct first, so that a unit that is validated without them is
     reported as it was; and, where every one of those fails and the term before
     has a closure of the sort that those rounds know ([closures_to_call]), all
     of them again with those calls made direct. Where that validates, its
     verdict is the one reported, and [closures_were_called] says so for the
     line under the verdict. If not, the answer is that of the attempts without.

     The two kinds: the attempts of the term before as it is; and, where they
     all refuse the unit, the same of the term before with its known arrays
     read. [read] says that the verdict is of an attempt of the second kind. So
     the order is: the attempts; those with the calls made direct; those of the
     term read; those of the term read with the calls made direct. The three
     lists of calls to leave are, for both kinds, those made above from the term
     before as converted: they are guesses. Where both kinds refuse the unit,
     the pairs of terms of both, without the calls made direct, are kept to be
     shown, and the second check, below, is asked of the two terms as
     converted. *)
  let read = ref false in
  let closures_were_called = ref false in
  let attempts, (validation, _, _, slots_removed, _) =
    let validates (_, ((validation : Validate.validation), _, _, _, _)) =
      match validation with
      | Not_validated -> false
      | Same_up_to_bound_names | Same_once_normalised
      | Same_once_unused_statics_removed | Same_once_head_statics_removed
      | Same_once_lifted ->
        true
    in
    let in_turn ~reading =
      let without = attempts_of ~calling:false ~reading in
      if validates without || not closures_to_call
      then without
      else
        let holds = !last_asking_holds in
        let tables = !tables_were_bound in
        let calling = attempts_of ~calling:true ~reading in
        if validates calling
        then (
          closures_were_called := true;
          calling)
        else (
          last_asking_holds := holds;
          tables_were_bound := tables;
          without)
    in
    match in_turn ~reading:false with
    | attempts, (Not_validated, _, _, _, _) -> (
      match in_turn ~reading:true with
      | attempts_read, ((Not_validated, _, _, _, _) as refused) ->
        ( attempts
          @ List.map
              (fun (which, pair) ->
                which ^ ", of the term before with its known arrays read", pair)
              attempts_read,
          refused )
      | ( _,
          ( ( Same_up_to_bound_names | Same_once_normalised
            | Same_once_unused_statics_removed | Same_once_head_statics_removed
            | Same_once_lifted ),
            _,
            _,
            _,
            _ ) ) as accepted ->
        read := true;
        accepted)
    | ( _,
        ( ( Same_up_to_bound_names | Same_once_normalised
          | Same_once_unused_statics_removed | Same_once_head_statics_removed
          | Same_once_lifted ),
          _,
          _,
          _,
          _ ) ) as accepted ->
      accepted
  in
  (* Whether the verdict is of the two runs that were shown. It is for a unit
     that imports nothing. For a unit that has imports it is where the model
     says that the verdict holds of the unit run from its imports
     ([last_asking_holds]; the statements are
     validate_unit_sound_verdict_imports_statement and
     validate_unit_knowing_sound_verdict_imports_statement). Where the unit is
     validated, the asking made last is the one whose verdict is reported: [ask]
     asks a second time only where its first asking does not validate, and gives
     the answer of the second; [attempt] asks [validate_unit_knowing] only where
     [validate_unit] does not validate, and not at all for the third attempt;
     the second attempt is made only where the first does not validate, and the
     third only where the first two do not. Where the model does not say so, and
     the validator accepts, what is validated is the unit with nothing imported.
     No line begins "validated:" for a unit that has imports. Where the two
     terms have a prefix in front of them, made from the [.cmx] files of
     [prefix_units] units, the verdict is of the two terms with it, and the line
     says of how many units the files were used: it rests on those files being
     right, so such a unit is counted apart whatever imports it has left. *)
  let of_the_runs_shown =
    match imports with [] -> true | _ :: _ -> !last_asking_holds ()
  in
  let validated =
    match prefix_units, imports with
    | 0, [] -> "validated"
    | 0, _ :: _ ->
      if of_the_runs_shown
      then "validated with its imports"
      else "validated with nothing imported"
    | n, _ ->
      if of_the_runs_shown
      then
        Printf.sprintf
          "validated with its imports, given the .cmx of %d of them" n
      else
        Printf.sprintf
          "validated with nothing else imported, given the .cmx of %d" n
  in
  (* The run of the term before that the validation is of. *)
  let outcome_before =
    if of_the_runs_shown
    then outcome_before
    else
      match validation with
      | Not_validated -> outcome_before
      | Same_up_to_bound_names | Same_once_normalised
      | Same_once_unused_statics_removed | Same_once_head_statics_removed
      | Same_once_lifted ->
        fst (run ~statics:Values.empty_static_env before)
  in
  Format.printf "  %s@."
    (match validation with
    | Same_up_to_bound_names -> validated ^ ": the same up to bound names"
    | Same_once_normalised -> validated ^ ": the same once normalised"
    | Same_once_unused_statics_removed ->
      validated
      ^ ": the same once normalised and unused static bindings removed"
    | Same_once_head_statics_removed ->
      validated
      ^ ": the same once normalised and unused static bindings removed, at the \
         head of the term after too"
    | Same_once_lifted ->
      let lifted =
        if !tables_were_bound
        then
          ": the same once closures and blocks are lifted to symbols, the \
           tables of the term after bound in the term before"
        else ": the same once closures and blocks are lifted to symbols"
      in
      if slots_removed
      then
        validated ^ lifted
        ^ ", the terms normalised and unused value slots removed"
      else validated ^ lifted ^ " and the terms normalised"
    | Not_validated ->
      if outcome_before = Undefined
      then "nothing to validate, as the term before is undefined"
      else "not validated");
  if !asked_again
  then
    Format.printf
      "  (at the second asking, of the two terms normalised with what their \
       symbols are bound to used)@.";
  if !read
  then
    Format.printf
      "  (asked of the term before normalised with the lengths and loads of \
       its known immutable arrays read)@.";
  if !closures_were_called
  then
    Format.printf
      "  (with the calls of the closures that a Let makes made direct in the \
       rounds after each lifting)@.";
  (* The second check, FlambdaValidateRelatedSpec.validate_unit_related, of the
     two terms and the imports, for each list of calls that the attempts above
     were made with, in their order. The liftings are guessed as for the
     validator, each applied without its test of the exits
     (FlambdaLiftObservedSpec.with_lifting_observed), which is how the function
     asked applies them; they do not ask for the calls of the closures that a
     [Let] makes to be made direct. For each guess the function is asked with no
     value slot to remove and, if it refuses and slots are guessed, with those.
     The answer is that of the first list that is accepted with no value slot
     removed or, if there is none, of the first that is accepted with the slots:
     whether slots were given, and whether the liftings of that asking bound
     tables; or nothing, if no list is accepted. [ask] takes an answer in the
     type of the verdicts of the validator and searches on where it is
     [Not_validated]: [Same_once_lifted] stands for [Related_once_lifted]
     between the function given to [ask] and the match on what [ask] returns,
     here and nowhere else, and is no verdict of the validator. *)
  let related () : (bool * bool) option =
    let asked kept =
      let with_slots = ref false in
      let answer, _, _, _, _ =
        ask ~calling:false ~with_lifting:Lift_observed.with_lifting_observed
          (fun unused slots liftings ->
            let related slots =
              match
                Validate_related.validate_unit_related imports rounds unused
                  slots loops liftings kept before.body after.body
              with
              | Related_once_lifted -> true
              | Not_related -> false
            in
            if related []
            then (
              with_slots := false;
              Validate.Same_once_lifted)
            else
              match slots with
              | [] -> Validate.Not_validated
              | _ :: _ ->
                if related slots
                then (
                  with_slots := true;
                  Validate.Same_once_lifted)
                else Validate.Not_validated)
          before after kept
      in
      match answer with
      | Same_once_lifted -> Some (!with_slots, !tables_were_bound)
      | Not_validated | Same_up_to_bound_names | Same_once_normalised
      | Same_once_unused_statics_removed | Same_once_head_statics_removed ->
        None
    in
    let rec search lists with_the_slots =
      match lists with
      | [] -> with_the_slots
      | kept :: lists -> (
        match asked kept, with_the_slots with
        | (Some (false, _) as accepted), _ -> accepted
        | (Some (true, _) as accepted), None -> search lists accepted
        | Some (true, _), Some _ | None, _ -> search lists with_the_slots)
    in
    search
      (first_guess :: recursive
      :: (if another_guess then [once_looped_guess] else []))
      None
  in
  (* The line of the second check, under the line "not validated" and nowhere
     else: not where the run of the term before that is shown is undefined,
     where the line is another. Its head says what the check was given: imports
     or none, and whether the two terms have a prefix in front of them. *)
  (match validation with
  | Not_validated when outcome_before <> Undefined -> (
    match related () with
    | None -> ()
    | Some (with_slots, tables) ->
      let head =
        match prefix_units, imports with
        | 0, [] -> "related once lifted"
        | 0, _ :: _ -> "related once lifted, with its imports"
        | n, _ :: _ ->
          Printf.sprintf
            "related once lifted, with its imports, given the .cmx of %d of \
             them"
            n
        | n, [] ->
          Printf.sprintf
            "related once lifted, with no remaining imports, given the .cmx of \
             %d of them"
            n
      in
      Format.printf
        "  %s: the same once values are lifted to symbols without the \
         liftings' symbols-and-constants-only exit test%s%s@."
        head
        (if tables
         then ", the tables of the term after bound in the term before"
         else "")
        (if with_slots
         then ", the terms normalised and unused value slots removed"
         else ", and the terms normalised; no value slots removed");
      Format.printf
        "  (in the model, an exit before is matched by one after under the \
         restricted observer class; exit values may differ)@.")
  | Not_validated | Same_up_to_bound_names | Same_once_normalised
  | Same_once_unused_statics_removed | Same_once_head_statics_removed
  | Same_once_lifted ->
    ());
  (* A term before that is undefined is refined by anything: the validation is
     right and says nothing. *)
  (match validation with
  | Not_validated -> ()
  | Same_up_to_bound_names | Same_once_normalised
  | Same_once_unused_statics_removed | Same_once_head_statics_removed
  | Same_once_lifted ->
    if not of_the_runs_shown
    then
      Format.printf
        "  not validated with its imports: this verdict is outside what is \
         proved of runs with imports@.";
    if outcome_before = Undefined
    then
      if of_the_runs_shown
      then
        Format.printf
          "  the term before is undefined in the model, so the validation says \
           nothing of this unit@."
      else
        Format.printf
          "  with nothing imported the term before is undefined in the model, \
           so that validation says nothing of this unit@.");
  (match validation with
  | Not_validated when Sys.getenv_opt "FLSEMANTICS_PRINT_TERMS" <> None ->
    (* A pair of terms that the validation compared, for each guess of the calls
       to leave ([ask] says which pair): the static bindings of the term before
       are in the order of those of the term after. *)
    List.iter
      (fun (which, (_, compared, how, _, normalised_after)) ->
        Format.printf
          "@[<v 2>  %s:@ @[<v 2>the term before, %s:@ %a@]@ @[<v 2>the term \
           after, as it was compared:@ %a@]@]@."
          which how print_expr
          (Statics_order.order_statics_as normalised_after compared)
          print_expr normalised_after)
      attempts
  | Not_validated | Same_up_to_bound_names | Same_once_normalised
  | Same_once_unused_statics_removed | Same_once_head_statics_removed
  | Same_once_lifted ->
    ());
  validation

(* ---- The .cmx of other units ------------------------------------------- *)

(* Where the tool's Simplify looks for the .cmx of another unit: the directories
   given with [-I] (or [-Ix], which is [-I] with a promise about the .cmx
   files), in the order they are given, each as it is written (a [+dir] is not
   made a directory of the standard library), and no other directory and no
   other file. The load path is emptied and those directories are added to it
   one by one. It is set neither by [Compmisc.init_path], which the driver uses,
   nor by [Load_path.init]. The first adds the standard library, the current
   directory, the directories of [-H] and of [-thread], and those that
   OCAMLPARAM gives. The second reads, whatever directories it is given, the
   files of [-H-file] and [-Hx-file] and the manifests of [-I-manifest] and
   [-H-manifest]. The tool accepts the options of the compiler, and none of
   those is to reach the load path. *)
let init_load_path () =
  Load_path.reset ();
  List.iter
    (fun ({ path; cmx_guaranteed } : Clflags.visible_include) ->
      Load_path.add_dir (Visible { cmx_guaranteed }) path)
    (List.rev !Clflags.include_dirs);
  Env.reset_cache ~preserve_persistent_env:false

(* The units whose .cmx file the compilation has read so far, each with the
   digest of the file. They are what the compiler would record as the imports of
   the .cmx file of the unit being compiled ([Compilenv]), less those for which
   it found no file or which are opaque, which have no digest. [Compilenv] gives
   them only as part of the description of the unit that it would write, so that
   is built here, of an empty module, and nothing else of it is used. It is
   asked once, after Simplify has run: the units are then those that Simplify
   read. With no directory given there is none. *)
let units_read () : (Compilation_unit.t * Digest.t) list =
  let infos =
    Compilenv.build_unit_info
      ~main_module_block_format:
        (Lambda.Mb_struct
           { mb_repr = Lambda.Module_value_only { field_count = 0 } })
      ~static_data:(Slambda.CU_data.empty ())
  in
  List.filter_map
    (fun import ->
      match Import_info.crc import with
      | Some digest -> Some (Import_info.cu import, digest)
      | None -> None)
    infos.Cmx_format.ui_imports_cmx

(* What a check came to. The lines that it printed say more: this is for a
   caller that goes on after the check, and for the status with which the
   command line exits. *)
type checked =
  | Not_checked
      (** No verdict. From [check]: the model has not got something of the unit,
          or one of the two terms fails the test that each use of its return
          continuation gives it the symbol of the unit, a test of the syntax
          that refuses a call that returns there and another name for the symbol
          too. From [check_file] also: the text has lost something of the unit,
          or the term before Simplify is not well-formed ([well_formed_before]).
      *)
  | Failed
      (** The text could not be read, the term after Simplify is not
          well-formed, the prefix made from the .cmx files has a fault, or the
          two runs differ. The command line exits 1. *)
  | Compared of Validate.validation
      (** The two runs were made and no difference between them was found: they
          gave the same, or the run before is undefined, or one of them was not
          over when the steps allowed ran out, as the lines printed say. With
          it, the verdict of the validator, which may be that the unit is not
          validated; the last lines printed say with what it is of. *)

(* Raised where a check has failed, after the line that says how is printed. *)
exception Check_failed

(* The two terms of the last unit that [check_or_raise] ran and compared, as it
   ran them: converted with regions dropped, the prefix in front. For
   [check_regions]. *)
let terms_compared : (converted * converted) option ref = ref None

let check_or_raise ~filename ~(units : (Compilation_unit.t * Digest.t) list)
    (before : Flambda_unit.t) (after : Flambda_unit.t) :
    Validate.validation option =
  terms_compared := None;
  match convert_both ~units before after with
  | exception Unsupported what ->
    Format.printf "%s: not checked: the model has not got %s@." filename
      (Format.asprintf "%t" what);
    None
  | before_converted, after_converted, _
    when Option.is_some (return_not_of_own_symbol before before_converted)
         || Option.is_some (return_not_of_own_symbol after after_converted) ->
    (match
       ( return_not_of_own_symbol before before_converted,
         return_not_of_own_symbol after after_converted )
     with
    | Some reason, _ -> Format.printf "%s: not checked: %s@." filename reason
    | None, Some reason ->
      Format.printf "%s: not checked: after Simplify, %s@." filename reason
    | None, None -> ());
    None
  | _, after, _ when not (well_formed after) ->
    Format.printf "%s: the term after Simplify is NOT WELL-FORMED (%s)@."
      filename
      (why_not_well_formed after);
    raise Check_failed
  | before, after, prefix ->
    let imports =
      imports ~bound:prefix.prefix_symbols
        ~unmentioned:prefix.prefix_unmentioned ~hidden:prefix.prefix_hidden
    in
    (* The prefix is checked before it is used: a fault here is one of the
       conversion, and neither of the unit nor of Simplify. *)
    (match prefix_fault prefix ~imports before after with
    | Some fault ->
      Format.printf
        "%s: not checked: the prefix made from the .cmx files %s, which is a \
         fault of the conversion@."
        filename fault;
      raise Check_failed
    | None -> ());
    let before = with_prefix prefix before in
    let after = with_prefix prefix after in
    terms_compared := Some (before, after);
    (match prefix.prefix_groups with
    | [] -> ()
    | _ :: _ ->
      if not (well_formed after)
      then (
        Format.printf
          "%s: not checked: with the prefix made from the .cmx files in front \
           of it the term after Simplify is not well-formed (%s)@."
          filename
          (why_not_well_formed after);
        raise Check_failed));
    let statics = Imports.imports_statics imports in
    let outcome_before, steps_before = run ~statics before in
    let outcome_after, steps_after = run ~statics after in
    let with_imports =
      (match prefix.prefix_groups with
        | [] -> ""
        | _ :: _ -> ", after the prefix")
      ^
      match imports with
      | [] -> ""
      | [_] -> ", with its import"
      | _ :: _ :: _ ->
        Printf.sprintf ", with its %d imports" (List.length imports)
    in
    Format.printf "%s:@." filename;
    (* The units read are named on a line of their own, unless the line of the
       prefix names them all. *)
    (match units with
    | [] -> ()
    | _ :: _ ->
      if unit_names units <> unit_names prefix.prefix_units
      then
        Format.printf "  the tool's Simplify read the .cmx of: %s@."
          (String.concat ", " (unit_names units)));
    print_prefix prefix;
    Format.printf "  before Simplify%s (%d steps): %a@." with_imports
      steps_before print_outcome outcome_before;
    Format.printf "  after Simplify%s (%d steps): %a@." with_imports steps_after
      print_outcome outcome_after;
    let passes ok = if ok then "passes" else "fails" in
    Format.printf "  the check of kinds: %s before, %s after@."
      (passes (well_kinded before))
      (passes (well_kinded after));
    Format.printf "  calls outside code: %d before, %d after@."
      (calls_outside_code before.body)
      (calls_outside_code after.body);
    let verdict, ok =
      match outcome_before, outcome_after with
      | Undefined, _ -> "anything goes, as the run before is undefined", true
      | Still_running, _ | _, Still_running -> "not decided", true
      | _, _ ->
        if outcome_before = outcome_after
        then "the same", true
        else "DIFFERENT", false
    in
    Format.printf "  %s@." verdict;
    let validation =
      validate ~imports
        ~prefix_units:(List.length prefix.prefix_units)
        ~outcome_before before after
    in
    if not ok then raise Check_failed;
    Some validation

(* ---- The check of regions --------------------------------------------- *)

(* The parameters of continuations that a region which is a ghost reaches are
   dropped from a term converted with regions kept, with the arguments at their
   places; see [region_params]. Such a parameter is one that some [Apply_cont]
   gives a variable noted in [ghost_regions], or gives another such parameter.
   The machine of regions has nothing for a ghost, and nothing binds the
   variable of one, so what is left has, at a parameter that is a region, only
   regions that the term begins.

   Refused: a parameter that is given a ghost by one [Apply_cont] and something
   else by another; and a ghost, or a parameter dropped, that is still used
   somewhere once that is done (the end that is not a ghost of such a parameter,
   an allocation in it). *)
let drop_ghost_regions (e : Model.expr) : Model.expr =
  let ghosts : (int, unit) Hashtbl.t = Hashtbl.create 16 in
  List.iter
    (fun var -> Hashtbl.replace ghosts (int_of_positive var) ())
    !ghost_regions;
  if Hashtbl.length ghosts = 0
  then e
  else
    let is_ghost_variable var = Hashtbl.mem ghosts (int_of_positive var) in
    let is_ghost (s : Model.simple) =
      match s with
      | Var var -> is_ghost_variable var
      | Symbol _ | Const _ -> false
    in
    (* The parameters of each continuation that a [Let_cont] binds, and every
       [Apply_cont]. A continuation has one number in a converted term. *)
    let params : (int, Model.variable list) Hashtbl.t = Hashtbl.create 16 in
    let uses : Model.apply_cont list ref = ref [] in
    let rec collect (e : Model.expr) =
      match e with
      | Let (_, _, _, body) -> collect body
      | Let_static (bindings, body) ->
        List.iter
          (fun (binding : Model.static_binding) ->
            match binding with
            | Bind_code (_, Code (_, _, _, _, _, body)) -> collect body
            | Bind_deleted_code _ | Bind_symbol _ -> ())
          bindings;
        collect body
      | Let_cont_non_recursive (k, handler, body) ->
        collect_handler k handler;
        collect body
      | Let_cont_recursive (handlers, body) ->
        List.iter (fun (k, handler) -> collect_handler k handler) handlers;
        collect body
      | Apply_cont ac -> uses := ac :: !uses
      | Switch (_, arms) -> List.iter (fun (_, ac) -> uses := ac :: !uses) arms
      | Apply _ | Invalid -> ()
    and collect_handler k (handler : Model.continuation_handler) =
      match handler with
      | Handler (handler_params, _, body) ->
        Hashtbl.replace params (int_of_positive k) handler_params;
        collect body
    in
    collect e;
    let params_of (ac : Model.apply_cont) =
      Hashtbl.find_opt params (int_of_positive ac.ac_continuation)
    in
    let changed = ref true in
    while !changed do
      changed := false;
      List.iter
        (fun (ac : Model.apply_cont) ->
          match params_of ac with
          | None -> ()
          | Some handler_params ->
            let rec reach handler_params args =
              match handler_params, args with
              | param :: handler_params, arg :: args ->
                if is_ghost arg && not (is_ghost_variable param)
                then (
                  Hashtbl.replace ghosts (int_of_positive param) ();
                  changed := true);
                reach handler_params args
              | [], _ | _, [] -> ()
            in
            reach handler_params ac.ac_args)
        !uses
    done;
    let no_ghost (simples : Model.simple list) =
      if List.exists is_ghost simples
      then
        unsupported
          "a region that is a ghost, or a parameter that is given one, used \
           otherwise than as an argument at a parameter that is a region"
    in
    let rec drop_at : 'a. Model.variable list -> 'a list -> 'a list =
     fun handler_params l ->
      match handler_params, l with
      | [], l -> l
      | _ :: _, [] -> []
      | param :: handler_params, a :: l ->
        if is_ghost_variable param
        then drop_at handler_params l
        else a :: drop_at handler_params l
    in
    let rewrite_apply_cont (ac : Model.apply_cont) : Model.apply_cont =
      let ac_args =
        match params_of ac with
        | None -> ac.ac_args
        | Some handler_params ->
          let rec check handler_params args =
            match handler_params, args with
            | param :: handler_params, arg :: args ->
              if is_ghost_variable param && not (is_ghost arg)
              then
                unsupported
                  "a parameter that is a region, given a ghost by one \
                   [Apply_cont] and something else by another";
              check handler_params args
            | [], _ | _, [] -> ()
          in
          check handler_params ac.ac_args;
          drop_at handler_params ac.ac_args
      in
      no_ghost ac_args;
      { ac with ac_args }
    in
    let rec rewrite (e : Model.expr) : Model.expr =
      match e with
      | Let (x, kind, defining_expr, body) ->
        no_ghost (Regions.simples_of_named defining_expr);
        Let (x, kind, defining_expr, rewrite body)
      | Let_static (bindings, body) ->
        let bindings =
          List.map
            (fun (binding : Model.static_binding) : Model.static_binding ->
              no_ghost (Regions.simples_of_binding binding);
              match binding with
              | Bind_code
                  (id, Code (code_params, my_closure, ret, exn, metadata, body))
                ->
                Bind_code
                  ( id,
                    Code
                      (code_params, my_closure, ret, exn, metadata, rewrite body)
                  )
              | Bind_deleted_code _ | Bind_symbol _ -> binding)
            bindings
        in
        Let_static (bindings, rewrite body)
      | Let_cont_non_recursive (k, handler, body) ->
        Let_cont_non_recursive (k, rewrite_handler handler, rewrite body)
      | Let_cont_recursive (handlers, body) ->
        Let_cont_recursive
          ( List.map (fun (k, handler) -> k, rewrite_handler handler) handlers,
            rewrite body )
      | Apply ap ->
        no_ghost
          ((ap.ap_callee :: ap.ap_args) @ List.map fst ap.ap_exn_extra_args);
        e
      | Apply_cont ac -> Apply_cont (rewrite_apply_cont ac)
      | Switch (scrutinee, arms) ->
        no_ghost [scrutinee];
        Switch
          (scrutinee, List.map (fun (i, ac) -> i, rewrite_apply_cont ac) arms)
      | Invalid -> e
    and rewrite_handler (handler : Model.continuation_handler) :
        Model.continuation_handler =
      match handler with
      | Handler (handler_params, metadata, body) ->
        Handler
          ( drop_at handler_params handler_params,
            { metadata with
              hm_params_kinds = drop_at handler_params metadata.hm_params_kinds
            },
            rewrite body )
    in
    rewrite e

(* What the run of a unit on the machine of regions (FlambdaRegionsSpec.v) came
   to, with the number of steps made and the state before the step that was not
   made, or the state that the run left by. *)
type region_outcome =
  | Region_passes
      (** The run left through a continuation that is not bound, and no step
          before was refused. *)
  | Region_fails of Regions.region_fault
      (** A step was refused by one of the rules of regions. *)
  | Region_not_run of string
      (** Nothing is said: the reason is the conversion's, or the run had no
          step for a reason that is not of regions, or it was not over when the
          steps allowed ran out. *)

let run_regions ~(statics : Values.static_env) (unit : converted) :
    region_outcome * int * Regions.region_state option =
  (* A symbol that no symbol of the two units or of the prefix has: the first
     that the machine draws. The test that it is so is the model's. *)
  let base = Symbols.fresh () in
  if not (Regions.region_names_fresh base unit.body statics)
  then
    ( Region_not_run
        "the symbols that the machine would draw are not apart from those of \
         the term",
      0,
      None )
  else
    let rec go steps (state : Regions.region_state) =
      if steps >= fuel
      then Region_not_run "no answer in the steps allowed", steps, Some state
      else
        match Regions.region_step state with
        | Region_step state -> go (steps + 1) state
        | Region_exit (_, _, state) -> Region_passes, steps, Some state
        | Region_undefined No_plain_step ->
          ( Region_not_run "the run is undefined for another reason",
            steps,
            Some state )
        | Region_undefined Not_of_the_conversion ->
          ( Region_not_run
              "a primitive of regions is not as the conversion makes it",
            steps,
            Some state )
        | Region_undefined
            (( Dead_value_used | Heap_holds_local | Region_not_open
             | Region_used_as_value ) as fault) ->
          Region_fails fault, steps, Some state
    in
    go 0 (Regions.initial_region_state unit.body statics base)

let region_outcome_shown (outcome : region_outcome) ~steps =
  match outcome with
  | Region_passes -> "passes"
  | Region_fails fault ->
    Printf.sprintf "fails (%s, at step %d)"
      (match fault with
      | Dead_value_used ->
        "a value allocated in a region is used after the region is gone"
      | Heap_holds_local -> "an allocation on the heap holds a local value"
      | Region_not_open ->
        "a region that is not open is ended, named or given to a continuation"
      | Region_used_as_value -> "a region is used as a value"
      | Not_of_the_conversion | No_plain_step -> assert false)
      steps
  | Region_not_run why -> Printf.sprintf "not run (%s)" why

(* The counts of a run, on a line of its own, where the environment asks for
   them ([FLSEMANTICS_REGION_COUNTS]): they are for a measure, and are in no
   expected file except those of tests/semantics/regions/, whose rules set the
   variable. *)
let print_region_counts which ~steps (state : Regions.region_state option) =
  match state with
  | None -> ()
  | Some state ->
    let counts = state.rs_counts in
    let frames_left kind =
      List.length
        (List.filter
           (fun (frame : Regions.frame) -> frame.frame_kind = kind)
           state.rs_frames)
    in
    let locals_left =
      List.fold_left
        (fun n (frame : Regions.frame) -> n + List.length frame.frame_locals)
        0 state.rs_frames
    in
    Format.printf
      "  the counts of regions %s: steps=%d ends=%d local_allocations=%d \
       local_uses=%d other_region=%d cut_by_end_region=%d \
       cut_by_end_try_region=%d no_frame=%d plain_left=%d try_left=%d \
       locals_left=%d@."
      which steps
      (int_of_nat counts.count_ends)
      (int_of_nat counts.count_local_allocations)
      (int_of_nat counts.count_local_uses)
      (int_of_nat counts.count_other_region)
      (int_of_nat counts.count_cut_by_end_region)
      (int_of_nat counts.count_cut_by_end_try_region)
      (int_of_nat counts.count_no_frame)
      (frames_left Plain_mark) (frames_left Try_mark) locals_left

(* The check of regions, after everything else: the two terms are converted
   again with regions and the modes of allocations kept ([keep_regions]), and
   each is run on the machine of regions from the imports. It changes no
   verdict.

   It is made only of a unit that was compared: [check] calls it where
   [check_or_raise] has returned a validation, and not of a unit that is not
   checked or whose check failed. It then prints one line if the second
   conversion met something of regions (a region that is not a ghost, or an
   allocation whose mode is [Local]: [regions_met]), or if it was refused. The
   first conversion of the same terms was not refused, and the two differ only
   where [keep_regions] is read, so a refusal of the second is one for regions;
   whatever its reason, the line gives it. A unit of which the second conversion
   went to its end and met nothing of regions has none, and gets no line.

   Before anything is run, the two conversions are compared. The model's [erase]
   (FlambdaRegionsSpec.v) drops the regions from a term that has them. Of each
   of the two terms, [erase] of the term converted with regions kept must be the
   term that [check_or_raise] ran and compared ([terms_compared]), up to bound
   names (FlambdaCompareSpec.expr_eqb). So what is run here is the term that was
   validated, with its regions in it, and not another. Where that fails the line
   says so, and nothing is run: it is a fault of the conversion.

   And each term, with the imports, must be one that [erase] is meant for (the
   model's [erasable]): no call of it returns or raises to a continuation with a
   parameter that is a region, which the conversion refuses already, and it has
   no poison constant of the kind [Region]. Where that fails the line says so,
   and nothing is run.

   It resets the conversion and converts both terms again. So once [check] has
   returned, the tables of the conversion ([Variables], [Symbols] and what
   [reset_conversion] clears) are those of this second conversion, and whatever
   is linked with this file and reads them after [check] reads those.

   CR mshinwell: the line says of each term whether its run on the machine of
   FlambdaRegionsSpec.v reached its end with no value allocated in a region used
   after the region was gone, no allocation on the heap holding such a value,
   and no region ended or named that was not open (decisions D44 to D48). It is
   of the one run that the machine makes, from the imports and with no argument:
   a path that the run does not take is not looked at. It is of the term and not
   of the code: the backend may still move a load after the end of a region. And
   "passes before, passes after" relates the two runs in nothing: the validation
   still drops regions (decision D37). *)
let check_regions ~(units : (Compilation_unit.t * Digest.t) list)
    (before : Flambda_unit.t) (after : Flambda_unit.t) : unit =
  let compared = !terms_compared in
  regions_met := false;
  keep_regions := true;
  let converted =
    Fun.protect
      ~finally:(fun () -> keep_regions := false)
      (fun () ->
        reset_conversion ();
        match
          let before, after, prefix = convert_both ~units before after in
          let kept (unit : converted) =
            let unit = with_prefix prefix unit in
            { unit with body = drop_ghost_regions unit.body }
          in
          kept before, kept after, prefix
        with
        | exception Unsupported what -> Error (Format.asprintf "%t" what)
        | before, after, prefix -> Ok (before, after, prefix))
  in
  if !regions_met || Result.is_error converted
  then
    match converted with
    | Error what ->
      Format.printf
        "  the check of regions: not run (the model has not got %s)@." what
    | Ok (before, after, prefix) ->
      let imports =
        imports ~bound:prefix.prefix_symbols
          ~unmentioned:prefix.prefix_unmentioned ~hidden:prefix.prefix_hidden
      in
      let statics = Imports.imports_statics imports in
      (* The term that is run is to be, with its regions dropped by the model's
         [erase], the term that was compared, up to bound names. *)
      let erased_is_compared (kept : converted) (dropped : converted) =
        Compare_terms.expr_eqb [] [] (Regions.erase [] kept.body) dropped.body
      in
      let the_terms_compared =
        match compared with
        | Some (dropped_before, dropped_after) ->
          erased_is_compared before dropped_before
          && erased_is_compared after dropped_after
        | None -> false
      in
      if not the_terms_compared
      then
        Format.printf
          "  the check of regions: not run (a term converted with its regions \
           kept is not, once they are dropped from it, the term that was \
           compared, which is a fault of the conversion)@."
      else if
        not
          (Regions.erasable before.body statics
          && Regions.erasable after.body statics)
      then
        Format.printf
          "  the check of regions: not run (a call returns or raises to a \
           continuation with a parameter that is a region, or there is a \
           poison constant of the kind of regions)@."
      else
        let outcome_before, steps_before, state_before =
          run_regions ~statics before
        in
        let outcome_after, steps_after, state_after =
          run_regions ~statics after
        in
        Format.printf "  the check of regions: %s before, %s after@."
          (region_outcome_shown outcome_before ~steps:steps_before)
          (region_outcome_shown outcome_after ~steps:steps_after);
        if Sys.getenv_opt "FLSEMANTICS_REGION_COUNTS" <> None
        then (
          print_region_counts "before" ~steps:steps_before state_before;
          print_region_counts "after" ~steps:steps_after state_after)

(* The check of the two terms of a unit, the first as Simplify was given it and
   the second as Simplify returned it. It prints, under the name given, what the
   command line prints of a unit, and exits nowhere.

   It does not test that the first term is well-formed. That is for the caller,
   before Simplify is run, which is entitled to fail on a term that is not:
   [well_formed_before], which prints the line "not checked" where it says no,
   and after which the caller is to make no check. [check_file] does so. A first
   term that is not well-formed and is given here all the same is run and
   compared as it is: nothing here refuses it. *)
let check ~filename ~(units : (Compilation_unit.t * Digest.t) list)
    (before : Flambda_unit.t) (after : Flambda_unit.t) : checked =
  match check_or_raise ~filename ~units before after with
  | exception Check_failed -> Failed
  | None -> Not_checked
  | Some validation ->
    check_regions ~units before after;
    Compared validation

(* The text of fexpr has no syntax for the mode of a parameter of code, and the
   reader gives every parameter the mode heap ([Fexpr_to_flambda]). In the
   compiler's own term a parameter may have the mode that allows the stack, and
   Simplify reads the mode: it is what the type of the parameter says of where
   the value is ([Simplify_set_of_closures]). So that the tool's Simplify takes
   nothing from the reader's mode, every parameter of every piece of code that
   the unit read binds is given, before anything else is done with the unit, the
   least that the compiler can say of where a value is
   ([Alloc_mode.For_types.unknown ()]: heap or stack, and heap in a compiler
   configured without stack allocation, where nothing is on the stack). Only
   these modes are changed: the other defaults of the reader stay, and code of
   another unit, which Simplify reads from a .cmx file, has the modes that the
   compiler wrote there. The bodies of code are not gone into: a unit that binds
   code there is not well-formed, and no verdict is given of it
   ([well_formed_before]). A [Let], a [Let_cont] and a handler are built again
   as the reader builds them, with fresh names for what they bind. *)
let with_unknown_param_modes_of_code (code : Code.t) : Code.t =
  let code_metadata =
    Code_metadata.with_param_modes
      (List.map
         (fun (_ : Alloc_mode.For_types.t) -> Alloc_mode.For_types.unknown ())
         (Code.param_modes code))
      (Code.code_metadata code)
  in
  Code.create_with_metadata
    ~params_and_body:(Code.params_and_body code)
    ~free_names_of_params_and_body:(Code.free_names_of_params_and_body code)
    ~code_metadata

let rec with_unknown_param_modes_of_expr (e : Flambda.Expr.t) : Flambda.Expr.t =
  match Flambda.Expr.descr e with
  | Let let_expr ->
    Flambda.Let_expr.pattern_match let_expr ~f:(fun pattern ~body ->
        let named : Flambda.Named.t =
          match Flambda.Let_expr.defining_expr let_expr with
          | Static_consts group ->
            Flambda.Named.create_static_consts
              (Flambda.Static_const_group.map group ~f:(fun const ->
                   match Flambda.Static_const_or_code.to_code const with
                   | Some code ->
                     Flambda.Static_const_or_code.create_code
                       (with_unknown_param_modes_of_code code)
                   | None -> const))
          | (Simple _ | Prim _ | Set_of_closures _ | Rec_info _) as named ->
            named
        in
        Flambda.Expr.create_let
          (Flambda.Let_expr.create pattern named
             ~body:(with_unknown_param_modes_of_expr body)
             ~free_names_of_body:Unknown))
  | Let_cont (Non_recursive { handler; _ }) ->
    Flambda.Non_recursive_let_cont_handler.pattern_match handler
      ~f:(fun k ~body ->
        Flambda.Let_cont_expr.create_non_recursive k
          (with_unknown_param_modes_of_handler
             (Flambda.Non_recursive_let_cont_handler.handler handler))
          ~body:(with_unknown_param_modes_of_expr body)
          ~free_names_of_body:Unknown)
  | Let_cont (Recursive handlers) ->
    Flambda.Recursive_let_cont_handlers.pattern_match handlers
      ~f:(fun ~invariant_params ~body handlers ->
        Flambda.Let_cont_expr.create_recursive ~invariant_params
          (Continuation.Lmap.map with_unknown_param_modes_of_handler
             (Flambda.Continuation_handlers.to_map handlers))
          ~body:(with_unknown_param_modes_of_expr body))
  | Apply _ | Apply_cont _ | Switch _ | Invalid _ -> e

and with_unknown_param_modes_of_handler
    (handler : Flambda.Continuation_handler.t) : Flambda.Continuation_handler.t
    =
  let is_exn_handler = Flambda.Continuation_handler.is_exn_handler handler in
  let is_cold = Flambda.Continuation_handler.is_cold handler in
  Flambda.Continuation_handler.pattern_match handler ~f:(fun params ~handler ->
      Flambda.Continuation_handler.create params
        ~handler:(with_unknown_param_modes_of_expr handler)
        ~free_names_of_handler:Unknown ~is_exn_handler ~is_cold)

let with_unknown_param_modes (unit : Flambda_unit.t) : Flambda_unit.t =
  Flambda_unit.create_from_metadata_and_body
    (Flambda_unit.metadata unit)
    (with_unknown_param_modes_of_expr (Flambda_unit.body unit))

(* The text of fexpr does not write the kind of an [Opaque_identity], and the
   reader gives every one the kind of values ([Fexpr_prim]); the compiler makes
   one of each component of an unboxed argument, at the kind of that component
   ([Lambda_to_flambda_primitives]). The compiler takes the argument of the
   primitive to have the kind that the primitive carries
   ([Flambda_primitive.arg_kind_of_unary_primitive]), so in a unit read from a
   text every [Opaque_identity] is given here the kind of its argument, as the
   reader has it: the kind of a constant, that of values for a symbol, and for a
   variable the kind that the reader made it with, which is the kind written at
   a parameter or the kind of the defining expression of a [Let]
   ([Fexpr_to_flambda]: [Flambda.Named.kind]).

   The primitive is not all that the reader got wrong for want of that kind: the
   variable that the [Let] of such a primitive binds was made with the kind of
   values, and so was each variable bound after it to that variable, or to an
   [Opaque_identity] of it. So the unit is gone through from the top, and each
   [Let] is put right after its defining expression: where the kind of the
   defining expression is no longer the kind of the variable that the reader
   bound to it, the [Let] binds a new variable of that kind, with the same name,
   and the body is renamed to it before it is gone into, so that every later
   use, and every later defining expression, sees a variable of the right kind.
   For an [Opaque_identity] nothing else is changed: the kinds of parameters and
   of value slots are those that the text wrote, and so are the arities but for
   the arity of the arguments of a direct call and of a call of unknown arity,
   which the same pass gives (below). The handlers of continuations and the
   bodies of code are gone into. A unit that has no such primitive and no such
   call to put right is not gone through at all, and is the unit as it was read
   ([kind_lost_in_expr]).

   This is a normalisation of a unit read from a text. The kind that it gives an
   [Opaque_identity] is the kind that the compiler's term had there, and that
   the text lost, where the kind obtained for the argument is the kind of that
   argument in the compiler's term and the primitive there had that kind. The
   first need not hold: the compiler's term may bind a variable at another kind
   than that of its defining expression, where the reader takes the kind of the
   expression (the conversion does not test that either: [convert_bound_var]),
   and an [Opaque_identity] above may have had another kind than its argument's.
   Where both hold all the way down, the kinds are the compiler's from the top
   down. The two conditions suffice and are not necessary. Where either fails,
   no recovery of the compiler's kind is claimed: the primitive there, the
   variable bound to it and what is bound below to them may be given kinds that
   the compiler's term had not, even where the kind of values that the reader
   gave was the compiler's, and nothing here finds that out. It is not a check
   of kinds and refuses no term for its kinds; of the compiler's own terms
   nothing is compared. It is applied to a unit read from a text only: not to
   what Simplify makes, nor to a body taken from the .cmx of another unit, which
   have the kinds that the compiler gave them. The text loses whether the
   primitive is for the middle end only as well; that is left as the reader
   gives it, as neither the model nor Simplify reads it.

   The reader has one more kind wrong that is known: its table of kinds for the
   nullary primitives that carry one gives two of the kinds for two words each
   ([Fexpr_prim.kind]), so that a text does not say which kind such a primitive
   had. Where the kind of the argument of an [Opaque_identity] is one of those
   two and comes from such a primitive, directly or by [Let]s that bind a
   variable to a variable, the unit is not checked. (The conversion refuses
   every nullary primitive today, so such a unit is not checked for that reason
   either.)

   The arity of the arguments of a call. The text does not write it for a direct
   call, nor for a call of unknown arity, and the reader gives such a call one
   value for each argument ([Fexpr_to_flambda]; label C2-5). The machine has no
   step for a call whose arguments are not of the kinds of its arity, or whose
   arity is not that of the code called, and Simplify makes such a call
   [Invalid]: so a unit that gives a function an unboxed number was undefined as
   read. In the same pass each such call is given an arity
   ([arity_of_arguments]); a call of known arity has its arity in the text and
   is left alone.

   A direct call is given the arity of the parameters of the code that it names,
   each kind without a subkind, where the unit binds that code and the call has
   as many arguments as the code has parameters. In the term that the compiler
   gives to Simplify that is the arity which the call had, subkinds and the
   grouping of unboxed products aside: the conversion from Lambda makes a call
   direct only where the arity of its arguments is, parameter for parameter,
   that of the code ([Closure_conversion.close_exact_or_unknown_apply], which
   makes any other [Invalid]), and with the flags of the tests it knows no code
   but the unit's own. The kinds of the arguments are not looked at for this:
   where an argument is not of the kind of its parameter, the call is ill-kinded
   in the term read as it was written. Where the unit does not bind the code, or
   the numbers differ, nothing gives the arity: the call is left as it was read
   if every argument is a value, and otherwise the unit is not checked.

   A call of unknown arity is given one kind for each argument, the kind that
   the term read has for it. That is the arity that the compiler's call had,
   subkinds aside, on three conditions, none of which this pass checks. For each
   argument, the kind obtained for it is the kind of that argument in the
   compiler's term: that can fail as it can for the argument of an
   [Opaque_identity], above. The arity of the compiler's call gave each argument
   the kind of that argument: the compiler does not see to it, and Simplify
   makes [Invalid] a call of which it is not so. And that arity had one
   singleton component for each argument, with no component that is an unboxed
   product. The three together suffice and are not necessary; where one fails no
   recovery of the compiler's arity is claimed. Where the kind of an argument
   comes from one of the two words that the reader may misread, the unit is not
   checked.

   The arity so given is flat, one component for each argument that the text
   writes, as the reader's was. The compiler's arity may group arguments: an
   argument that is an unboxed product is one component of several kinds, or of
   none, and then the text does not write the argument at all; and an
   application counts the components against the parameters of the code to tell
   a full application from a partial application or an over-application. The
   text has no grouping, neither at a call nor in the parameters that it writes
   for code: nothing in it tells a call that gives an unboxed pair from one that
   gives two arguments, nor says that a call, or a piece of code, has one more
   component, of no kind. So the grouping is not given back, here or anywhere,
   and the third condition is one that this pass cannot test. A direct call and
   its code are flattened alike, the call having the parameters of the code. For
   a call of unknown arity, the term read is the compiler's with every group
   flattened, at the call and in the code of the unit. As far as the grouping
   goes, and that alone (the kinds are as said above, and the text loses other
   things), such a call runs in the term read as in the compiler's if neither
   the call nor the code that it reaches has a group. If one of them has, it may
   not. A partial application that stops before a parameter of no kind is read
   as a full application, and so is an over-application whose last argument has
   no kind (label C2-30). And where the call and the code group the same kinds
   differently, which is ill-kinded in the compiler's term, the term read may be
   defined where the compiler's is not. All of that was so before this step of a
   call whose arguments are all values, which the reader has always read flat;
   what the step would add is the same of a call with an argument that is not a
   value, which was undefined as read.

   The scripts that make the texts of tests/semantics/ocaml refuse the units in
   which that can happen, from what the compiler prints of the term: a unit that
   has a call of unknown arity is refused if such a call has a group among its
   arguments, or if code that the unit binds has one among its parameters. In a
   unit that they let through, no call of unknown arity has a group, and the
   code of the unit, which is all that such a call can reach of what is read
   from the text, has none: there the grouping is the compiler's. (Code of
   another unit is not read from a text: it is taken from the .cmx of that unit,
   with the arity that the compiler gave it.) This pass cannot tell that a text
   is of such a unit: the refusal of the scripts does not protect a text that is
   given to the tool by hand, which is read as it is. A term in memory has the
   component of no kind, and [convert_arity] keeps it.

   Like the rest of the pass this is a normalisation of a unit read from a text
   and not a check of kinds: it refuses a unit only where it cannot tell what to
   give, and compares nothing of the compiler's own terms. *)
exception Lost_kind_not_known of string

let opaque_kind_not_known_message =
  "the text does not say the kind of an Opaque_identity, and the kind of its \
   argument is one that the reader may have misread: it comes from a nullary \
   primitive whose kind the text writes with a word that the reader takes for \
   two kinds"

let arity_of_unknown_call_not_known_message =
  "the text does not give the arity of the arguments of a call of unknown \
   arity, and the kind of one of its arguments is one that the reader may have \
   misread: it comes from a nullary primitive whose kind the text writes with \
   a word that the reader takes for two kinds"

let arity_of_direct_call_not_known_message =
  "the text does not give the arity of the arguments of a direct call, one of \
   which is not a value, and nothing else gives it: the unit does not bind the \
   code that the call names, or that code has not as many parameters as the \
   call has arguments"

let kind_that_the_reader_may_misread (kind : Flambda_kind.t) =
  Flambda_kind.equal kind Flambda_kind.naked_int8
  || Flambda_kind.equal kind Flambda_kind.naked_int16

let variable_of_misread_kind ~misread (simple : Simple.t) =
  Simple.pattern_match' simple
    ~var:(fun var ~coercion:_ -> Variable.Set.mem var misread)
    ~symbol:(fun _ ~coercion:_ -> false)
    ~const:(fun _ -> false)

(* The kinds of the parameters, unarized, of each piece of code that a unit
   binds, wherever it binds it. The unit is looked at as for printing and
   nothing is built ([kind_lost_in_expr] says why that matters). *)
let rec params_of_code_in_expr (e : Flambda.Expr.t)
    (found : Flambda_kind.t list Code_id.Map.t) :
    Flambda_kind.t list Code_id.Map.t =
  match Flambda.Expr.descr e with
  | Let let_expr ->
    let found =
      params_of_code_in_named (Flambda.Let_expr.defining_expr let_expr) found
    in
    Flambda.Let_expr.pattern_match_for_printing let_expr ~f:(fun _ ~body ->
        params_of_code_in_expr body found)
  | Let_cont (Non_recursive { handler; _ }) ->
    let found =
      params_of_code_in_handler
        (Flambda.Non_recursive_let_cont_handler.handler handler)
        found
    in
    Flambda.Non_recursive_let_cont_handler.pattern_match_for_printing handler
      ~f:(fun _ ~body -> params_of_code_in_expr body found)
  | Let_cont (Recursive handlers) ->
    Flambda.Recursive_let_cont_handlers.pattern_match_for_printing handlers
      ~f:(fun ~invariant_params:_ ~body handlers ->
        List.fold_left
          (fun found handler -> params_of_code_in_handler handler found)
          (params_of_code_in_expr body found)
          (Continuation.Lmap.data
             (Flambda.Continuation_handlers.to_map handlers)))
  | Apply _ | Apply_cont _ | Switch _ | Invalid _ -> found

and params_of_code_in_named (named : Flambda.Named.t)
    (found : Flambda_kind.t list Code_id.Map.t) :
    Flambda_kind.t list Code_id.Map.t =
  match named with
  | Static_consts group ->
    List.fold_left
      (fun found const ->
        match Flambda.Static_const_or_code.to_code const with
        | Some code ->
          let found =
            Code_id.Map.add (Code.code_id code)
              (List.map Flambda_kind.With_subkind.kind
                 (Flambda_arity.unarize (Code.params_arity code)))
              found
          in
          Flambda.Function_params_and_body.pattern_match_for_printing
            (Code.params_and_body code)
            ~f:(fun
                ~return_continuation:_
                ~exn_continuation:_
                _
                ~body
                ~my_closure:_
                ~is_my_closure_used:_
                ~my_alloc_mode:_
                ~my_depth:_
                ~free_names_of_body:_
              -> params_of_code_in_expr body found)
        | None -> found)
      found
      (Flambda.Static_const_group.to_list group)
  | Simple _ | Prim _ | Set_of_closures _ | Rec_info _ -> found

and params_of_code_in_handler (handler : Flambda.Continuation_handler.t)
    (found : Flambda_kind.t list Code_id.Map.t) :
    Flambda_kind.t list Code_id.Map.t =
  Flambda.Continuation_handler.pattern_match_for_printing handler
    ~f:(fun _ ~handler -> params_of_code_in_expr handler found)

(* What the pass gives a call for the arity of its arguments. *)
type arity_of_arguments =
  | As_read  (** The arity that the call has is the one to give it. *)
  | Given of [`Complex] Flambda_arity.t
  | Not_known of string  (** Nothing gives it; why. *)

(* [code_params] has the parameters of the code that the unit binds
   ([params_of_code_in_expr]); [misread], the variables whose kind may be one
   that the reader misread. *)
let arity_of_arguments ~code_params ~misread (apply : Flambda.Apply.t) :
    arity_of_arguments =
  let args = Flambda.Apply.args apply in
  let given (kinds : Flambda_kind.t list) =
    let as_read =
      List.map Flambda_kind.With_subkind.kind
        (Flambda_arity.unarize (Flambda.Apply.args_arity apply))
    in
    if List.equal Flambda_kind.equal kinds as_read
    then As_read
    else
      Given
        (Flambda_arity.create_singletons
           (List.map Flambda_kind.With_subkind.anything kinds))
  in
  match Flambda.Apply.call_kind apply with
  | Function { function_call = Direct code_id; _ } -> (
    match Code_id.Map.find_opt code_id code_params with
    | Some params when List.compare_lengths params args = 0 -> given params
    | Some _ | None ->
      if List.for_all (fun arg -> Flambda_kind.is_value (Simple.kind arg)) args
      then As_read
      else Not_known arity_of_direct_call_not_known_message)
  | Function { function_call = Indirect_unknown_arity; _ } -> (
    match given (List.map Simple.kind args) with
    | Given _ when List.exists (variable_of_misread_kind ~misread) args ->
      Not_known arity_of_unknown_call_not_known_message
    | (As_read | Given _ | Not_known _) as arity -> arity)
  | Function { function_call = Indirect_known_arity _; _ }
  | Method _ | C_call _ | Effect _ ->
    As_read

let rec with_lost_kinds_of_expr ~code_params ~misread (e : Flambda.Expr.t) :
    Flambda.Expr.t =
  match Flambda.Expr.descr e with
  | Let let_expr ->
    Flambda.Let_expr.pattern_match let_expr ~f:(fun pattern ~body ->
        let named =
          with_lost_kinds_of_named ~code_params ~misread
            (Flambda.Let_expr.defining_expr let_expr)
        in
        let pattern, body, misread =
          match (pattern : Bound_pattern.t) with
          | Set_of_closures _ | Static _ -> pattern, body, misread
          | Singleton bound_var ->
            let var = Bound_var.var bound_var in
            let kind = Flambda.Named.kind named in
            let var, pattern, body =
              if Flambda_kind.equal kind (Variable.kind var)
              then var, pattern, body
              else
                let user_visible =
                  if Variable.user_visible var then Some () else None
                in
                let new_var =
                  Variable.create ?user_visible (Variable.name var) kind
                in
                ( new_var,
                  Bound_pattern.singleton
                    (Bound_var.create new_var
                       (Bound_var.debug_uid bound_var)
                       (Bound_var.name_mode bound_var)),
                  Flambda.Expr.apply_renaming body
                    (Flambda2_nominal.Renaming.add_variable
                       Flambda2_nominal.Renaming.empty var new_var) )
            in
            let misread =
              match named with
              | Prim (Nullary (Invalid kind | Optimised_out kind), _)
                when kind_that_the_reader_may_misread kind ->
                Variable.Set.add var misread
              | Simple simple when variable_of_misread_kind ~misread simple ->
                Variable.Set.add var misread
              | Simple _ | Prim _ | Set_of_closures _ | Static_consts _
              | Rec_info _ ->
                misread
            in
            pattern, body, misread
        in
        Flambda.Expr.create_let
          (Flambda.Let_expr.create pattern named
             ~body:(with_lost_kinds_of_expr ~code_params ~misread body)
             ~free_names_of_body:Unknown))
  | Let_cont (Non_recursive { handler; _ }) ->
    Flambda.Non_recursive_let_cont_handler.pattern_match handler
      ~f:(fun k ~body ->
        Flambda.Let_cont_expr.create_non_recursive k
          (with_lost_kinds_of_handler ~code_params ~misread
             (Flambda.Non_recursive_let_cont_handler.handler handler))
          ~body:(with_lost_kinds_of_expr ~code_params ~misread body)
          ~free_names_of_body:Unknown)
  | Let_cont (Recursive handlers) ->
    Flambda.Recursive_let_cont_handlers.pattern_match handlers
      ~f:(fun ~invariant_params ~body handlers ->
        Flambda.Let_cont_expr.create_recursive ~invariant_params
          (Continuation.Lmap.map
             (with_lost_kinds_of_handler ~code_params ~misread)
             (Flambda.Continuation_handlers.to_map handlers))
          ~body:(with_lost_kinds_of_expr ~code_params ~misread body))
  | Apply apply -> (
    match arity_of_arguments ~code_params ~misread apply with
    | As_read -> e
    | Given args_arity ->
      Flambda.Expr.create_apply
        (Flambda.Apply.with_args apply (Flambda.Apply.args apply) ~args_arity)
    | Not_known why -> raise (Lost_kind_not_known why))
  | Apply_cont _ | Switch _ | Invalid _ -> e

and with_lost_kinds_of_named ~code_params ~misread (named : Flambda.Named.t) :
    Flambda.Named.t =
  match named with
  | Prim (Unary (Opaque_identity { middle_end_only; kind }, arg), dbg) ->
    if variable_of_misread_kind ~misread arg
    then raise (Lost_kind_not_known opaque_kind_not_known_message);
    let kind_of_arg = Simple.kind arg in
    if Flambda_kind.equal kind kind_of_arg
    then named
    else
      Flambda.Named.create_prim
        (Unary (Opaque_identity { middle_end_only; kind = kind_of_arg }, arg))
        dbg
  | Static_consts group ->
    Flambda.Named.create_static_consts
      (Flambda.Static_const_group.map group ~f:(fun const ->
           match Flambda.Static_const_or_code.to_code const with
           | Some code ->
             Flambda.Static_const_or_code.create_code
               (with_lost_kinds_of_code ~code_params ~misread code)
           | None -> const))
  | Simple _ | Prim _ | Set_of_closures _ | Rec_info _ -> named

and with_lost_kinds_of_handler ~code_params ~misread
    (handler : Flambda.Continuation_handler.t) : Flambda.Continuation_handler.t
    =
  let is_exn_handler = Flambda.Continuation_handler.is_exn_handler handler in
  let is_cold = Flambda.Continuation_handler.is_cold handler in
  Flambda.Continuation_handler.pattern_match handler ~f:(fun params ~handler ->
      Flambda.Continuation_handler.create params
        ~handler:(with_lost_kinds_of_expr ~code_params ~misread handler)
        ~free_names_of_handler:Unknown ~is_exn_handler ~is_cold)

and with_lost_kinds_of_code ~code_params ~misread (code : Code.t) : Code.t =
  let params_and_body =
    Flambda.Function_params_and_body.pattern_match (Code.params_and_body code)
      ~f:(fun
          ~return_continuation
          ~exn_continuation
          params
          ~body
          ~my_closure
          ~is_my_closure_used:_
          ~my_alloc_mode
          ~my_depth
          ~free_names_of_body
        ->
        Flambda.Function_params_and_body.create ~return_continuation
          ~exn_continuation params
          ~body:(with_lost_kinds_of_expr ~code_params ~misread body)
          ~free_names_of_body ~my_closure ~my_alloc_mode ~my_depth)
  in
  Code.create_with_metadata ~params_and_body
    ~free_names_of_params_and_body:(Code.free_names_of_params_and_body code)
    ~code_metadata:(Code.code_metadata code)

(* Whether a unit has an [Opaque_identity] whose kind is not the kind of its
   argument, or a call that is not to keep the arity of its arguments, as the
   reader has them: whether the pass above has anything to do. The first such
   primitive or call from the top is one that this finds, as nothing above it is
   changed. The unit is looked at as a term is for printing
   ([pattern_match_for_printing]) and nothing is built, so a unit that has
   neither is left as it was read. Opened so, the bound names of a term are not
   made fresh, unless the tool is given the flag that makes them fresh where a
   term is printed ([Flambda_features.freshen_when_printing]), which the tests
   do not give. So with the flags of the tests looking at a unit uses up no
   name, and the names which the tool prints of a unit that has neither are
   those that it printed before this step was there. The tests that this look
   makes, whether an [Opaque_identity] and its argument have different kinds and
   what [arity_of_arguments] says of a call, are the same with either setting of
   the flag. *)
let rec kind_lost_in_expr ~code_params (e : Flambda.Expr.t) : bool =
  match Flambda.Expr.descr e with
  | Let let_expr ->
    kind_lost_in_named ~code_params (Flambda.Let_expr.defining_expr let_expr)
    || Flambda.Let_expr.pattern_match_for_printing let_expr ~f:(fun _ ~body ->
        kind_lost_in_expr ~code_params body)
  | Let_cont (Non_recursive { handler; _ }) ->
    kind_lost_in_handler ~code_params
      (Flambda.Non_recursive_let_cont_handler.handler handler)
    || Flambda.Non_recursive_let_cont_handler.pattern_match_for_printing handler
         ~f:(fun _ ~body -> kind_lost_in_expr ~code_params body)
  | Let_cont (Recursive handlers) ->
    Flambda.Recursive_let_cont_handlers.pattern_match_for_printing handlers
      ~f:(fun ~invariant_params:_ ~body handlers ->
        kind_lost_in_expr ~code_params body
        || List.exists
             (kind_lost_in_handler ~code_params)
             (Continuation.Lmap.data
                (Flambda.Continuation_handlers.to_map handlers)))
  | Apply apply -> (
    match arity_of_arguments ~code_params ~misread:Variable.Set.empty apply with
    | As_read -> false
    | Given _ | Not_known _ -> true)
  | Apply_cont _ | Switch _ | Invalid _ -> false

and kind_lost_in_named ~code_params (named : Flambda.Named.t) : bool =
  match named with
  | Prim (Unary (Opaque_identity { middle_end_only = _; kind }, arg), _) ->
    not (Flambda_kind.equal kind (Simple.kind arg))
  | Static_consts group ->
    List.exists
      (fun const ->
        match Flambda.Static_const_or_code.to_code const with
        | Some code ->
          Flambda.Function_params_and_body.pattern_match_for_printing
            (Code.params_and_body code)
            ~f:(fun
                ~return_continuation:_
                ~exn_continuation:_
                _
                ~body
                ~my_closure:_
                ~is_my_closure_used:_
                ~my_alloc_mode:_
                ~my_depth:_
                ~free_names_of_body:_
              -> kind_lost_in_expr ~code_params body)
        | None -> false)
      (Flambda.Static_const_group.to_list group)
  | Simple _ | Prim _ | Set_of_closures _ | Rec_info _ -> false

and kind_lost_in_handler ~code_params (handler : Flambda.Continuation_handler.t)
    : bool =
  Flambda.Continuation_handler.pattern_match_for_printing handler
    ~f:(fun _ ~handler -> kind_lost_in_expr ~code_params handler)

let with_lost_kinds (unit : Flambda_unit.t) : (Flambda_unit.t, string) result =
  let body = Flambda_unit.body unit in
  let code_params = params_of_code_in_expr body Code_id.Map.empty in
  if not (kind_lost_in_expr ~code_params body)
  then Ok unit
  else
    match
      with_lost_kinds_of_expr ~code_params ~misread:Variable.Set.empty body
    with
    | exception Lost_kind_not_known why -> Error why
    | body ->
      Ok
        (Flambda_unit.create_from_metadata_and_body
           (Flambda_unit.metadata unit)
           body)

(* A unit read from a text, as it is checked and as Simplify is given it. *)
let as_read_from_text (unit : Flambda_unit.t) : (Flambda_unit.t, string) result
    =
  Result.map with_unknown_param_modes (with_lost_kinds unit)

let check_file filename : checked =
  match Parse_flambda.parse_fexpr filename with
  | Error e ->
    Test_utils.dump_error e;
    Failed
  | Ok unit -> (
    let unit_info = Parse_flambda.make_unit_info ~filename in
    let comp_unit = Unit_info.modname unit_info in
    (* As the driver does for a unit that it compiles: [Compilenv] reads the
       .cmx of the other units for the current one. *)
    Compilenv.reset unit_info;
    let fl2 = Fexpr_to_flambda.conv comp_unit unit in
    let filename_shown = Filename.basename filename in
    if result_arity_lost unit
    then (
      Format.printf "%s: not checked: %s@." filename_shown
        result_arity_lost_message;
      Not_checked)
    else
      match as_read_from_text fl2.unit with
      | Error why ->
        Format.printf "%s: not checked: %s@." filename_shown why;
        Not_checked
      | Ok before ->
        if not (well_formed_before ~filename:filename_shown before)
        then Not_checked
        else
          let null_formatter =
            Format.make_formatter (fun _ _ _ -> ()) (fun () -> ())
          in
          let result =
            Flambda2.flambda_to_flambda ~ppf_dump:null_formatter
              ~prefixname:(Filename.chop_extension filename)
              ~machine_width:Sixty_four ~code_slot_offsets:fl2.code_slot_offsets
              before
          in
          check ~filename:filename_shown ~units:(units_read ()) before
            result.flambda)

(* The conversion, for the check that takes a term of the model through fexpr
   and back (flroundtrip.ml). *)
let conversion_for_round_trip : Flroundtrip.conversion =
  let names numbered name =
    List.map (fun (id, number) -> name id, number) (numbered ())
  in
  { convert =
      (fun unit ->
        match convert_unit unit with
        | exception Unsupported what -> Error (Format.asprintf "%t" what)
        | { body; return_continuation; exn_continuation } ->
          Ok { body; return_continuation; exn_continuation });
    as_read = with_lost_kinds;
    reset = reset_conversion;
    value_slot_kind =
      (fun slot -> Option.map convert_kind (List.assoc_opt slot !slot_kinds));
    symbols =
      (fun () ->
        names Symbols.bindings (fun symbol ->
            Linkage_name.to_string (Symbol.linkage_name symbol)));
    code_ids = (fun () -> names Code_ids.bindings Code_id.name);
    value_slots = (fun () -> names Value_slots.bindings Value_slot.name);
    function_slots =
      (fun () -> names Function_slots.bindings Function_slot.name)
  }

(* Simplify, for that check: run as [check_file] runs it, and on the same
   units. *)
let simplify_for_round_trip ~filename (text : Fexpr.flambda_unit)
    (parsed : Fexpr_to_flambda.conv_result) =
  if result_arity_lost text
  then Error result_arity_lost_message
  else
    match as_read_from_text parsed.unit with
    | Error why -> Error why
    | Ok before ->
      let well_formed_or_not_converted =
        match convert_unit before with
        | exception Unsupported _ -> true
        | before -> well_formed before
      in
      if not well_formed_or_not_converted
      then Error "the unit is not well-formed"
      else
        let null_formatter =
          Format.make_formatter (fun _ _ _ -> ()) (fun () -> ())
        in
        let result =
          Flambda2.flambda_to_flambda ~ppf_dump:null_formatter
            ~prefixname:(Filename.chop_extension filename)
            ~machine_width:Sixty_four
            ~code_slot_offsets:parsed.code_slot_offsets before
        in
        Ok result.flambda

(* The command line: the options of the compiler, [-round-trip], and one [.fl]
   file. It exits 1 where the check has failed ([checked]). *)
let main () =
  let module Options =
    Oxcaml_args.Make_optcomp_options (Oxcaml_args.Default.Optmain)
  in
  Clflags.set_target Clflags.Target.Native;
  let round_trip = ref false in
  let files = ref [] in
  let defer_file file =
    match Filename.extension file with
    | ".fl" -> files := file :: !files
    | ext -> Misc.fatal_errorf "Unrecognized extension %s" ext
  in
  Compenv.warnings_for_discarded_params := true;
  Compenv.set_extra_params (Some Oxcaml_args.Extra_params.read_param);
  Compenv.readenv Format.err_formatter Before_args;
  Clflags.add_arguments __LOC__
    (( "-round-trip",
       Arg.Set round_trip,
       " Check that the conversion to the Rocq model survives a round trip \
        through fexpr, for the unit and for what Simplify makes of it" )
    :: (Arch.command_line_options @ Options.list));
  Clflags.Opt_flag_handler.set Oxcaml_flags.opt_flag_handler;
  Compenv.parse_arguments (ref Sys.argv) defer_file "flsemantics";
  Location.read_clflags_from_env ();
  init_load_path ();
  (* One unit to a process: the symbols of a unit stay in the tables of the
     compiler, so a second unit that used the name of one would be refused for a
     reason that is not its own. *)
  match !files with
  | [] -> ()
  | [file] -> (
    if !round_trip
    then (
      reset_conversion ();
      Flroundtrip.check_file conversion_for_round_trip
        ~simplify:(simplify_for_round_trip ~filename:file)
        file)
    else
      match check_file file with
      | Failed -> exit 1
      | Not_checked | Compared _ -> ())
  | _ :: _ :: _ ->
    Misc.fatal_error
      "flsemantics checks one .fl file: run it once for each file"
