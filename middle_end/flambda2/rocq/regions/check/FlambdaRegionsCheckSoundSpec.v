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

(* The check of regions (FlambdaRegionsCheckSpec.v) is sound: the
   regions of a term that passes it add no fault to the term, under
   two premises: the term is one that the erasure is meant for
   ([erasable]), and the symbols that the machine draws are none of
   the term or of the static environment, which holds no location
   ([region_names_fresh]: "the freshness", below).

   This file is to be reviewed: all of it. It adds nothing to the
   machine, to the check or to the property of states.
   FlambdaRegionsCheckSoundProofs.v proves its six statements.

   Every statement here is of runs from the first state of a term in
   a static environment ([initial_region_state]), under those two
   premises, which each statement writes out: of the one term where
   one term is run, of the term after where only its run on the
   machine of regions is spoken of, of both terms where an exit of
   the term before is.

   WHAT IT ADDS TO THE STATEMENT OF THE CHECK

   FlambdaRegionsCheckSpec.region_check_fault_statement is that the
   run of a term that passes the check, from its first state and with
   symbols to draw that are none of the term or of the static
   environment, which holds no location, stops without a step only
   where [step] has none. For the regions of such a term to add no
   fault to it (FlambdaRegionsCheckSpec.regions_add_no_fault), what
   that leaves
   is that where the run stops so, the run of the term with its
   regions dropped goes wrong too ([no_plain_step_erased_statement]).
   With it the check is sound ([region_check_sound_statement]), under
   the premises of freshness and [erasable].

   WITH THE VALIDATION OF A RUN OF SIMPLIFY

   The two statements of FlambdaRegionsCheckSpec.v about a run of
   Simplify take [regions_add_no_fault] of the term after as a
   premise. The last four statements here are those two with the
   check in the place of that premise, each in two forms.

   In the first form ([refined_and_checked_safe_statement],
   [refined_and_checked_exits_agree_statement]) a premise is
   [refines] of the two terms with their regions dropped, from their
   first states in the static environment: a relation of the model,
   and not a verdict of the validation. If it holds, the term before
   does not go wrong with its regions dropped, and the term after
   passes the check, then, under the freshness and [erasable] of the
   term after, the term after, with its regions, never stops without
   a step; and, under the freshness and [erasable] of both terms, an
   exit of the term before with its regions is an exit of the term
   after with its regions.

   The second form ([validated_and_checked_safe_statement],
   [validated_and_checked_exits_agree_statement]) is of a unit with
   its imports, run from the static environment of the imports
   ([FlambdaImportsSpec.imports_statics]), and has in the place of
   [refines] what the test tool computes. It has two terms more, the
   two that the validation is given, and three premises on them. The
   verdict that [FlambdaValidateSpec.validate_unit] gives of those
   two terms passes the model's test of it with the imports
   ([FlambdaValidateImportsSpec.verdict_holds_from], asked with the
   same arguments): it is the premise of
   FlambdaValidateImportsSpec.validate_unit_sound_verdict_imports_statement,
   which gives [FlambdaImportsSpec.unit_refines] of the two terms.
   And each of the two is, up to bound names, [erase] of the term
   with its regions ([FlambdaCompareSpec.expr_eqb], which is sound:
   [expr_eqb_sound_statement]). With these that premise of the first
   form holds: [erase_statics] leaves the static environment of the
   imports, which has no code, as it is.

   What is between the test tool and the second form is not a
   statement of the tree, and is three things. The tool reads the two
   terms of a unit; it converts each twice, once with its regions
   dropped, for the validation, and once with them kept; and it calls
   the functions of the model that the first three premises name, on
   those terms: the validation and the test of its verdict on the two
   terms made with regions dropped and, before it runs a unit on the
   machine of regions, the comparison of [erase] of each term made
   with regions kept with the term that the validation was given.
   That it does is not proved: it is in rocq/TRUST-LEDGER.md (the
   entries "The check of regions" and "What the tool reports").

   What has no statement of the second form. A verdict of the second
   asking ([validate_unit_knowing]) and one of the two validators
   that read (FlambdaValidateReadingSpec.v): for each a statement of
   the same kind gives [unit_refines] of the two terms that the
   validator is given
   ([validate_unit_knowing_sound_verdict_imports_statement],
   [validate_unit_reading_verdict_imports_statement],
   [validate_unit_knowing_reading_verdict_imports_statement]), from
   which the like of the second form would follow as it does here;
   it is not stated. The route that the tool takes for a unit that
   imports nothing, which asks no test of the verdict: the second
   form is of such a unit too, if its premise on the verdict holds,
   but nothing here derives that premise merely from an accepted
   verdict; what is proved of an accepted verdict is of the two terms
   run from the empty static environment
   (FlambdaValidateSpec.validate_unit_sound_statement). A line of the
   tool that begins "related once lifted", which is no such verdict:
   what is proved of that check (FlambdaValidateRelatedSpec.v) is
   another relation than [refines]. And the check itself, which the
   tool does not ask (FlambdaRegionsCheckSpec.v).

   [no_plain_step_erased_statement] is about the erasure and not
   about the check. It is of any term that the erasure is meant for
   ([erasable]), and it asks of the states of the run two things that
   a state of FlambdaRegionsCheckedSpec.v is proved to have:
   [reads_live], with its tests of region arguments and symbols read
   as values, and a jump to a handler
   gives every kind [Region] of the handler a parameter and an
   argument ([regions_given]).

   Without the first it is false: a partial application whose callee
   is a symbol that is gone may be evaluated, and calling it has no
   step, where the term with its regions dropped calls a closure; a
   value that is no mark, given at the place of a region, has no step
   for the kinds, where the term with its regions dropped has neither
   the parameter nor the argument.

   Without the second it is false too, and a first text of the
   statement, which asked [reads_live] alone, was. A handler with a
   parameter of the kind [Region] that is applied to no argument has
   no step on the machine of regions: it has a kind and no argument
   for it. A handler with no parameter whose kinds have a [Region],
   applied to an open region, has none either: it has no parameter
   to bind the mark to. In both, the term with its regions dropped
   has neither the kind nor the argument, goes on and exits. The
   erasure mends a number of parameters, of kinds or of arguments
   that is wrong by the regions that it drops, and by nothing else:
   of a handler with no kind [Region] it changes neither the
   parameters nor the kinds (its body is erased as every expression
   is), and it drops no argument of a jump to it. The check refuses
   both terms. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaValueSpec
  FlambdaSemanticsSpec FlambdaCompareSpec FlambdaUnusedStaticsSpec
  FlambdaLoopifySpec FlambdaLiftSpec FlambdaImportsSpec FlambdaValidateSpec
  FlambdaValidateImportsSpec FlambdaRegionsSpec FlambdaRegionsErasureSpec
  FlambdaRegionsCheckSpec FlambdaRegionsCheckedSpec.
Import ListNotations.

(* ======================================================================== *)
(* Statements                                                               *)
(* ======================================================================== *)

(* Of a run whose states satisfy [reads_live] and in every state of
   which the regions of the handler that is applied are given: where
   it stops because [step] has no step, the run of the term with its
   regions dropped goes wrong. *)
Definition no_plain_step_erased_statement : Prop :=
  ∀ (e : expr) (statics : static_env) (base : symbol) (fuel : nat),
    region_names_fresh base e statics = true ->
    erasable e statics = true ->
    (∀ (fuel' : nat) (rs : region_state),
       region_run fuel' (initial_region_state e statics base)
       = Region_step rs ->
       reads_live rs ∧ regions_given rs = true) ->
    region_run fuel (initial_region_state e statics base)
    = Region_undefined No_plain_step ->
    behaves (erased_initial_state e statics) Goes_wrong.

(* The check is sound: the regions of a term that passes it, with any
   certificate, add no fault to the term. *)
Definition region_check_sound_statement : Prop :=
  ∀ (cert : certificate) (statics : static_env) (base : symbol) (e : expr),
    region_check cert statics e = true ->
    region_names_fresh base e statics = true ->
    erasable e statics = true ->
    regions_add_no_fault e statics base.

(* Of a run of Simplify, with the check. The first premise is
   [refines] of the two terms with their regions dropped: a relation
   of the model, and not a verdict of the validation
   ([validated_and_checked_safe_statement] has in its place what the
   test tool computes). If it holds, the term before, with its
   regions dropped, does not go wrong, and the term after passes the
   check, with any certificate, then the term after, with its
   regions, never stops without a step. It is
   FlambdaRegionsCheckSpec.checked_after_safe_statement with the
   check of the term after, the freshness ([region_names_fresh]) and
   [erasable] of the term after in the place of its
   premise [regions_add_no_fault]. Nothing is asked of the regions of
   the term before. *)
Definition refined_and_checked_safe_statement : Prop :=
  ∀ (cert : certificate) (statics : static_env) (base : symbol)
    (before after : expr),
    refines (erased_initial_state before statics)
      (erased_initial_state after statics) ->
    ~ behaves (erased_initial_state before statics) Goes_wrong ->
    region_check cert statics after = true ->
    region_names_fresh base after statics = true ->
    erasable after statics = true ->
    region_safe (initial_region_state after statics base).

(* And then the two runs compute the same: an exit of the term before
   with its regions is an exit of the term after with its regions,
   through the same continuation, with values that are erased to the
   same values where they may be evaluated or are locations. It is
   FlambdaRegionsCheckSpec.checked_exits_agree_statement with the
   check of the term after in the place of its premise
   [regions_add_no_fault]. *)
Definition refined_and_checked_exits_agree_statement : Prop :=
  ∀ (cert : certificate) (statics : static_env) (base : symbol)
    (before after : expr),
    region_names_fresh base before statics = true ->
    region_names_fresh base after statics = true ->
    erasable before statics = true ->
    erasable after statics = true ->
    refines (erased_initial_state before statics)
      (erased_initial_state after statics) ->
    ~ behaves (erased_initial_state before statics) Goes_wrong ->
    region_check cert statics after = true ->
    ∀ (fuel : nat) (k : continuation) (vs : list value) (rs : region_state),
      region_run fuel (initial_region_state before statics base)
      = Region_exit k vs rs ->
      ∃ (fuel' : nat) (ws : list value) (rs' : region_state)
        (us : list value),
        region_run fuel' (initial_region_state after statics base)
        = Region_exit k ws rs'
        ∧ Forall2 (value_erased_if_usable rs) vs us
        ∧ Forall2 (value_erased_if_usable rs') ws us.

(* The same of a unit with its imports, with what the test tool
   computes in the place of [refines]. [validated_before] and
   [validated_after] are the two terms that the validation is given.
   If the verdict that [validate_unit] gives of them passes the
   model's test of it with the imports ([verdict_holds_from]: the
   premise of
   FlambdaValidateImportsSpec.validate_unit_sound_verdict_imports_statement),
   each of them is, up to bound names, [erase] of the term with its
   regions ([expr_eqb]), the term before, with its regions dropped,
   does not go wrong, and the term after passes the check, with any
   certificate, then the term after, with its regions, run from the
   static environment of the imports, never stops without a step.
   The other premises are those of
   [refined_and_checked_safe_statement] at that static environment.
   Nothing is asked of the regions of the term before. *)
Definition validated_and_checked_safe_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (rounds : nat)
    (unused : dropped) (slots : list value_slot) (G : loopified)
    (Ls : list lifting) (kept : list code_id) (cert : certificate)
    (base : symbol) (before after validated_before validated_after : expr),
    let statics := imports_statics imports in
    verdict_holds_from statics rounds unused slots G Ls kept
      validated_before validated_after
      (validate_unit rounds unused slots G Ls kept validated_before
         validated_after)
    = true ->
    expr_eqb [] [] (erase [] before) validated_before = true ->
    expr_eqb [] [] (erase [] after) validated_after = true ->
    ~ behaves (erased_initial_state before statics) Goes_wrong ->
    region_check cert statics after = true ->
    region_names_fresh base after statics = true ->
    erasable after statics = true ->
    region_safe (initial_region_state after statics base).

(* And then the two runs compute the same, as in
   [refined_and_checked_exits_agree_statement] and with its other
   premises at the static environment of the imports: an exit of the
   term before with its regions is an exit of the term after with its
   regions, through the same continuation, with values that are
   erased to the same values where they may be evaluated or are
   locations. *)
Definition validated_and_checked_exits_agree_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (rounds : nat)
    (unused : dropped) (slots : list value_slot) (G : loopified)
    (Ls : list lifting) (kept : list code_id) (cert : certificate)
    (base : symbol) (before after validated_before validated_after : expr),
    let statics := imports_statics imports in
    verdict_holds_from statics rounds unused slots G Ls kept
      validated_before validated_after
      (validate_unit rounds unused slots G Ls kept validated_before
         validated_after)
    = true ->
    expr_eqb [] [] (erase [] before) validated_before = true ->
    expr_eqb [] [] (erase [] after) validated_after = true ->
    ~ behaves (erased_initial_state before statics) Goes_wrong ->
    region_check cert statics after = true ->
    region_names_fresh base before statics = true ->
    region_names_fresh base after statics = true ->
    erasable before statics = true ->
    erasable after statics = true ->
    ∀ (fuel : nat) (k : continuation) (vs : list value) (rs : region_state),
      region_run fuel (initial_region_state before statics base)
      = Region_exit k vs rs ->
      ∃ (fuel' : nat) (ws : list value) (rs' : region_state)
        (us : list value),
        region_run fuel' (initial_region_state after statics base)
        = Region_exit k ws rs'
        ∧ Forall2 (value_erased_if_usable rs) vs us
        ∧ Forall2 (value_erased_if_usable rs') ws us.
