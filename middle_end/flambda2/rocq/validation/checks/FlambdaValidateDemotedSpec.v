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

(* The validation asked of the term after with its exception handlers
   that no call can raise to made ordinary continuations.

   This file is to be reviewed: [after_demoted],
   [validate_unit_demoted], [validate_unit_knowing_demoted],
   [validate_unit_reading_demoted] and
   [validate_unit_knowing_reading_demoted], which are extracted and run
   by the validation of middle_end/flambda2/tests/semantics/; and the
   statements.

   Why. The normalisation of the term before ends by making ordinary
   continuations of the exception handlers that no call can raise to
   (FlambdaDemoteSpec.demote_handlers, in
   FlambdaValidateSpec.demote_and_tidy), as Simplify does to a [try]
   all of whose raises it sees: the term after then has a jump where
   the term before had a raise. The term after is not given that
   rewrite there. On a unit compiled with debugging information
   Simplify keeps such a handler where a raise that reaches it is of a
   kind that it takes for a use that escapes
   (simplify/simplify_common.ml, [apply_cont_use_kind];
   FlambdaDemoteSpec.v says which kinds; discrepancies C5-3 and C5-4).
   The term before, normalised, can then have a jump where the term
   after still has the handler, its trap actions and the raise; and
   that can prevent the comparison of FlambdaValidateSpec.validate_unit,
   which tells a trap action from none, from succeeding.

   So the term after is given the rewrite too, here, in functions
   beside:

   - [after_demoted] is the rewrite of the term after, with no name in
     scope, as [demote_and_tidy] applies it to the term before;
   - [validate_unit_demoted] is FlambdaValidateSpec.validate_unit of
     the term before and of that term;
     [validate_unit_knowing_demoted] is
     FlambdaValidateKnowingSpec.validate_unit_knowing of the same; and
     [validate_unit_reading_demoted] and
     [validate_unit_knowing_reading_demoted] are the two functions of
     FlambdaValidateReadingSpec.v of the same.

   What they accept is correct for the unit whose term after is the
   one GIVEN, with its handlers as they are. The rewrite is an
   equivalence on the machine of FlambdaSemanticsSpec.v, under all
   environments
   (FlambdaDemoteSpec.demote_handlers_equivalent_statement). So
   [expr_refines (after_demoted after) after], the rewritten term for
   the source and the term as given for the target
   ([after_demoted_refines_statement]); and a refinement of the term
   before by the rewritten term is one by the term as given, of the
   unit run on its own from empty environments
   ([demoted_program_refines_statement]) and of the unit run from its
   imports ([demoted_unit_refines_statement]). The two statements of
   each function are the two statements of the function that it asks,
   at the rewritten term, and then one of the two above. The second
   is given the test of FlambdaValidateImportsSpec.v on the two terms
   that the function asked was given: the term before, or
   FlambdaValidateReadingSpec.before_read of it, and
   [after_demoted after]; and the verdict of that asking.

   What is not said.
   - The statements are about the machine that goes by names, which
     executes no trap action and does not read the mark of a handler.
     Nothing is said of backtraces.
   - Nothing here uses the machine with a stack of handlers
     (FlambdaTrapSpec.v), or the condition of the rewrite
     (FlambdaDemoteSpec.demotable). The rewritten term is what the
     comparison is given, and nothing else: the term that the
     statements conclude about is the term after as given, which
     corresponds to the compiler's own term after, unchanged.
   - The statements have no premise about the form of the term after.
     A caller that asks checks of the term after (that it is well
     formed, that it runs) asks them of the term as given: nothing
     here lets a check of [after_demoted after] stand for one of
     [after].

   Which handlers the rewrite demotes. The handler of a non-recursive
   [Let_cont] of [k] that is marked as an exception handler is demoted
   where the body of the [Let_cont] has no call whose exception
   continuation is [k], [k] is not bound again in that body, and [k]
   is not the name of a continuation in whose scope the [Let_cont]
   stands (the rewrite starts with none, adds those that the term
   binds around the [Let_cont], and in code starts from the return and
   exception continuations of the code): whatever the kinds of the
   raises to it. The two tests of names pass where each continuation
   of the term is bound once, as the names that the compiler makes
   are; that is said of the compiler and is no statement here. So a
   handler that Simplify keeps for the kinds of the raises to it
   alone, no call having it for exception continuation, is demoted
   here. That is what the source says of that case. It is not stated,
   here or anywhere, that every difference that debugging information
   makes to what Simplify gives is of that form; and where the
   handlers are demoted and the two terms still differ, the pair is
   refused as before.

   A caller asks the function of FlambdaValidateSpec.v,
   FlambdaValidateKnowingSpec.v or FlambdaValidateReadingSpec.v first,
   and the one here only of a unit that the first does not validate:
   nothing that the first accepts is lost. What a caller guesses for
   the function here (the names to drop, the loops, the liftings, the
   code to keep) is to be guessed for the two terms that are compared,
   the second of which is [after_demoted after]; the statements hold
   whatever is guessed.

   A control, computed. The term after is
   FlambdaDemoteSpec.try_raise true, a [try raise with _ -> 2] in
   which the handler is installed and is reached by a raise of kind
   [Regular], a kind for which Simplify keeps the handler where the
   unit is compiled with debugging information; the term before is
   that term with a [Let] in its handler.
   FlambdaValidateSpec.validate_unit refuses
   the pair and [validate_unit_demoted] accepts it
   ([kept_handler_validated_statement]); and with a term before that
   leaves with 3, which is another program, [validate_unit_demoted]
   refuses it too ([kept_handler_wrong_refused_statement]).

   The proofs are in FlambdaValidateDemotedProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaValueSpec
  FlambdaSemanticsSpec FlambdaWellFormedSpec FlambdaUnusedStaticsSpec
  FlambdaLoopifySpec FlambdaLiftSpec FlambdaDemoteSpec
  FlambdaValidateSpec FlambdaValidateKnowingSpec FlambdaImportsSpec
  FlambdaValidateImportsSpec FlambdaValidateReadingSpec.
Import ListNotations.

(* ======================================================================== *)
(* The term after, its handlers demoted                                     *)
(* ======================================================================== *)

(* The rewrite with no name in scope, as
   FlambdaValidateSpec.demote_and_tidy applies it to the term before. *)
Definition after_demoted (after : expr) : expr := demote_handlers [] after.

(* ======================================================================== *)
(* The functions                                                            *)
(* ======================================================================== *)

Definition validate_unit_demoted (rounds : nat) (unused : dropped)
    (slots : list value_slot) (G : loopified) (Ls : list lifting)
    (kept : list code_id) (before after : expr) : validation :=
  validate_unit rounds unused slots G Ls kept before (after_demoted after).

(* The same of the second asking. *)
Definition validate_unit_knowing_demoted (rounds : nat) (unused : dropped)
    (slots : list value_slot) (G : loopified) (Ls : list lifting)
    (kept : list code_id) (before after : expr) : validation :=
  validate_unit_knowing rounds unused slots G Ls kept before
    (after_demoted after).

(* The same of the two functions that read the known arrays of the term
   before. *)
Definition validate_unit_reading_demoted (rounds : nat) (unused : dropped)
    (slots : list value_slot) (G : loopified) (Ls : list lifting)
    (kept : list code_id) (before after : expr) : validation :=
  validate_unit_reading rounds unused slots G Ls kept before
    (after_demoted after).

Definition validate_unit_knowing_reading_demoted (rounds : nat)
    (unused : dropped) (slots : list value_slot) (G : loopified)
    (Ls : list lifting) (kept : list code_id) (before after : expr)
    : validation :=
  validate_unit_knowing_reading rounds unused slots G Ls kept before
    (after_demoted after).

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* The term after as given refines the term that the rewrite makes of
   it, under all environments: the rewritten term is the source, the
   term as given is the target. *)
Definition after_demoted_refines_statement : Prop :=
  ∀ after : expr, expr_refines (after_demoted after) after.

(* So a refinement by the rewritten term is one by the term as given:
   of the unit run on its own from empty environments, and of the unit
   run from its imports. *)
Definition demoted_program_refines_statement : Prop :=
  ∀ before after : expr,
    program_refines before (after_demoted after) ->
    program_refines before after.

Definition demoted_unit_refines_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (before after : expr),
    unit_refines imports before (after_demoted after) ->
    unit_refines imports before after.

(* A unit that is validated here refines the unit it came from, run on
   its own from empty environments. The term after of the conclusion is
   the one given. *)
Definition validate_unit_demoted_sound_statement : Prop :=
  ∀ (rounds : nat) (unused : dropped) (slots : list value_slot)
    (G : loopified) (Ls : list lifting) (kept : list code_id)
    (before after : expr),
    validate_unit_demoted rounds unused slots G Ls kept before after
    <> Not_validated ->
    program_refines before after.

Definition validate_unit_knowing_demoted_sound_statement : Prop :=
  ∀ (rounds : nat) (unused : dropped) (slots : list value_slot)
    (G : loopified) (Ls : list lifting) (kept : list code_id)
    (before after : expr),
    validate_unit_knowing_demoted rounds unused slots G Ls kept before
      after
    <> Not_validated ->
    program_refines before after.

Definition validate_unit_reading_demoted_sound_statement : Prop :=
  ∀ (rounds : nat) (unused : dropped) (slots : list value_slot)
    (G : loopified) (Ls : list lifting) (kept : list code_id)
    (before after : expr),
    validate_unit_reading_demoted rounds unused slots G Ls kept before
      after
    <> Not_validated ->
    program_refines before after.

Definition validate_unit_knowing_reading_demoted_sound_statement : Prop :=
  ∀ (rounds : nat) (unused : dropped) (slots : list value_slot)
    (G : loopified) (Ls : list lifting) (kept : list code_id)
    (before after : expr),
    validate_unit_knowing_reading_demoted rounds unused slots G Ls kept
      before after
    <> Not_validated ->
    program_refines before after.

(* A unit whose verdict here the test of FlambdaValidateImportsSpec.v
   accepts at the static environment of its imports, of the two terms
   that the function asked was given (the term before and
   [after_demoted after]), refines the unit it came from, each run on
   its own from those imports. The term after of the conclusion is the
   one given. *)
Definition validate_unit_demoted_imports_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (rounds : nat)
    (unused : dropped) (slots : list value_slot) (G : loopified)
    (Ls : list lifting) (kept : list code_id) (before after : expr),
    verdict_holds_from (imports_statics imports) rounds unused slots G Ls
      kept before (after_demoted after)
      (validate_unit_demoted rounds unused slots G Ls kept before after)
    = true ->
    unit_refines imports before after.

(* The same of the second asking. *)
Definition validate_unit_knowing_demoted_imports_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (rounds : nat)
    (unused : dropped) (slots : list value_slot) (G : loopified)
    (Ls : list lifting) (kept : list code_id) (before after : expr),
    verdict_knowing_holds_from (imports_statics imports) rounds unused
      slots G Ls kept before (after_demoted after)
      (validate_unit_knowing_demoted rounds unused slots G Ls kept before
         after)
    = true ->
    unit_refines imports before after.

(* The same of the two functions that read the known arrays of the term
   before: the test is of the term that the function asked was given
   for the term before (FlambdaValidateReadingSpec.before_read) and of
   [after_demoted after]. *)
Definition validate_unit_reading_demoted_imports_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (rounds : nat)
    (unused : dropped) (slots : list value_slot) (G : loopified)
    (Ls : list lifting) (kept : list code_id) (before after : expr),
    verdict_holds_from (imports_statics imports) rounds unused slots G Ls
      kept (before_read rounds G Ls kept before) (after_demoted after)
      (validate_unit_reading_demoted rounds unused slots G Ls kept before
         after)
    = true ->
    unit_refines imports before after.

Definition validate_unit_knowing_reading_demoted_imports_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (rounds : nat)
    (unused : dropped) (slots : list value_slot) (G : loopified)
    (Ls : list lifting) (kept : list code_id) (before after : expr),
    verdict_knowing_holds_from (imports_statics imports) rounds unused
      slots G Ls kept (before_read rounds G Ls kept before)
      (after_demoted after)
      (validate_unit_knowing_reading_demoted rounds unused slots G Ls kept
         before after)
    = true ->
    unit_refines imports before after.

(* ======================================================================== *)
(* A control                                                                *)
(* ======================================================================== *)

Section Control.

Let unit_exn : continuation := 1%positive.
Let leave : continuation := 2%positive.
Let handler : continuation := 3%positive.
Let body : continuation := 4%positive.
Let x : variable := 1%positive.
Let y : variable := 2%positive.

(* [try raise with _ -> result] with its handler installed, as
   FlambdaDemoteSpec.try_raise true is, but for a [Let] in the handler:
   so the pairs below are not of two terms that are the same as they
   are given. *)
Definition try_raise_let (result : Z) : expr :=
  Let_cont_non_recursive handler
    (Handler [x] (make_exn_handler_metadata [Value])
       (Let y Value (Simple (Const (Tagged_immediate result)))
          (Apply_cont (Apply_cont_expr leave [Var y] None))))
    (Let_cont_non_recursive body
       (Handler [] (make_handler_metadata [])
          (Apply_cont
             (Apply_cont_expr handler [Const (Tagged_immediate 0%Z)]
                (Some (Pop handler (Some Regular))))))
       (Apply_cont (Apply_cont_expr body [] (Some (Push handler))))).

(* The validation of a pair, by the function here where [demoted] and by
   FlambdaValidateSpec.validate_unit where not: ten rounds, nothing
   removed, a lifting that lifts nothing, and no code to be left where
   it is called. *)
Definition validate_try (demoted : bool) (before after : expr)
    : validation :=
  (if demoted then validate_unit_demoted else validate_unit) 10
    (Dropped [] []) [] []
    [Lifting (Unit_interface leave unit_exn) [] [] [] [] [] [] [] [] false
       false]
    [] before after.

(* The term after keeps its handler. The rewrite makes of it the term
   that Simplify leaves where it demotes; the validation of
   FlambdaValidateSpec.v refuses the pair and the one here accepts it;
   the two units exit with 2. *)
Definition kept_handler_validated_statement : Prop :=
  after_demoted (try_raise true) = try_raise false
  ∧ validate_try false (try_raise_let 2) (try_raise true) = Not_validated
  ∧ validate_try true (try_raise_let 2) (try_raise true)
    = Same_once_normalised
  ∧ run 10
      (State (try_raise_let 2) [] [] empty_static_env empty_heap no_reads)
    = Exit leave [Value_const (Tagged_immediate 2)] empty_heap no_reads
  ∧ run 10
      (State (try_raise true) [] [] empty_static_env empty_heap no_reads)
    = Exit leave [Value_const (Tagged_immediate 2)] empty_heap no_reads.

(* A term before that leaves with 3 is another program, and the pair is
   not validated here either. *)
Definition kept_handler_wrong_refused_statement : Prop :=
  validate_try true (try_raise_let 3) (try_raise true) = Not_validated
  ∧ run 10
      (State (try_raise_let 3) [] [] empty_static_env empty_heap no_reads)
    = Exit leave [Value_const (Tagged_immediate 3)] empty_heap no_reads
  ∧ run 10
      (State (try_raise true) [] [] empty_static_env empty_heap no_reads)
    = Exit leave [Value_const (Tagged_immediate 2)] empty_heap no_reads.

End Control.
