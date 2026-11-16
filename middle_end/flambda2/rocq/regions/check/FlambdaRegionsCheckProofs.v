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

(* Proofs of the three statements of FlambdaRegionsCheckSpec.v. That the
   run of a term that passes the check, from its first state and with
   symbols to draw that are none of the term or of the static environment,
   stops without a step only where [step] has none ([region_check_fault])
   is from the statements of FlambdaRegionsCheckedSpec.v. The two
   statements about the validation of a run of Simplify take
   [regions_add_no_fault] of the term after as a premise; they are from the
   laws of the erasure (FlambdaRegionsErasureProofs.v) and use nothing of
   the check. Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaRegionsSpec
  FlambdaRegionsInvariantSpec FlambdaRegionsErasureSpec
  FlambdaRegionsErasureProofs FlambdaRegionsCallsSpec FlambdaRegionsCheckSpec
  FlambdaRegionsCheckedSpec.
From OxCamlFlambda2 Require Import FlambdaRegionsCheckedProofs.
Import ListNotations.

(* ======================================================================== *)
(* The runs of a state that has the property                                *)
(* ======================================================================== *)

(* A run keeps the property. *)
Lemma region_run_keeps_checked (cert : certificate)
    (codes : list (code_id * code)) (fuel : nat) :
  ∀ rs rs' : region_state,
    state_checked cert codes rs ->
    region_run fuel rs = Region_step rs' ->
    state_checked cert codes rs'.
Proof.
  induction fuel as [ | fuel IH]; intros rs rs' Hchecked; cbn [region_run].
  - intros [= <-]. exact Hchecked.
  - destruct (region_step rs) as [rs1 | k vs rs1 | fault] eqn:Hstep;
      try discriminate.
    apply IH.
    exact (region_step_keeps_checked cert codes rs rs1 Hchecked Hstep).
Qed.

(* A run from a state that has the property stops without a step only
   where [step] has none. *)
Lemma region_run_fault (cert : certificate) (codes : list (code_id * code))
    (fuel : nat) :
  ∀ (rs : region_state) (fault : region_fault),
    state_checked cert codes rs ->
    region_run fuel rs = Region_undefined fault ->
    fault = No_plain_step.
Proof.
  induction fuel as [ | fuel IH]; intros rs fault Hchecked; cbn [region_run];
    [ discriminate | ].
  destruct (region_step rs) as [rs1 | k vs rs1 | fault1] eqn:Hstep;
    try discriminate.
  - apply IH.
    exact (region_step_keeps_checked cert codes rs rs1 Hchecked Hstep).
  - intros [= <-]. exact (checked_fault cert codes rs fault1 Hchecked Hstep).
Qed.

(* ======================================================================== *)
(* The check                                                                *)
(* ======================================================================== *)

(* The run of a term that passes the check, from its first state and
   with symbols to draw that are none of the term or of the static
   environment, stops without a step only where [step] has none: not
   for decisions D45, D46 or D47, not for a region used as a value,
   not for a primitive of regions that is not as the conversion makes
   it. *)
Theorem region_check_fault : region_check_fault_statement.
Proof.
  intros cert statics base e fuel fault Hcheck Hfresh.
  exact
    (region_run_fault cert (all_code statics e) fuel _ fault
       (checked_initial cert statics base e Hcheck Hfresh)).
Qed.

(* ======================================================================== *)
(* With the validation of a run of Simplify                                 *)
(* ======================================================================== *)

Theorem checked_after_safe : checked_after_safe_statement.
Proof.
  intros statics base before after [_ Hback] Hdefined Hcheck fuel fault Hrun.
  pose proof (Hcheck fuel fault Hrun) as Hwrong.
  assert (Hne : Goes_wrong <> Diverges) by discriminate.
  destruct (Hback Goes_wrong Hne Hwrong) as [H | H]; exact (Hdefined H).
Qed.

Theorem checked_exits_agree : checked_exits_agree_statement.
Proof.
  intros statics base before after Hfresh Hfresh' Herasable Herasable'
    Hrefines Hdefined Hcheck fuel k vs rs Hrun.
  pose proof
    (checked_after_safe statics base before after Hrefines Hdefined Hcheck)
    as Hsafe.
  destruct Hrefines as [Hexits _].
  destruct
    (erasure_exits before statics base fuel k vs rs Hfresh Herasable Hrun)
    as (us & hp & Hbehaves & Hvs).
  pose proof (Hexits k us hp Hbehaves) as Hbehaves'.
  destruct
    (erasure_adds_no_exit after statics base k us hp Hfresh' Herasable'
       Hbehaves')
    as [fuel' [[fault Hfault] | (ws & rs' & Hrun' & Hws)]].
  - contradiction (Hsafe fuel' fault Hfault).
  - exists fuel', ws, rs', us. auto.
Qed.
