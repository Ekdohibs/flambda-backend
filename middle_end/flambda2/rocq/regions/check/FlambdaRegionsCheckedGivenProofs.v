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

(* Lemmas for the proofs of the statements of FlambdaRegionsCheckedSpec.v:
   in a state that has the property, a jump to a handler gives every kind
   [Region] of the handler a parameter and an argument. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec FlambdaRegionsSpec
  FlambdaRegionsInvariantSpec FlambdaRegionsErasureSpec
  FlambdaRegionsCallsSpec FlambdaRegionsCheckSpec FlambdaRegionsCheckedSpec.
From OxCamlFlambda2 Require Import FlambdaRegionsCheckedBasicsProofs
  FlambdaRegionsCheckedHeadProofs.
Import ListNotations.

(* The parameters, the kinds and the arguments of a jump that gives
   its regions are as many. *)
Lemma region_args_lengths (cs : check_state) :
  ∀ (params : list variable) (kinds : list flambda_kind)
    (args : list (option simple * lifetime))
    (given : list (variable * variable)),
    region_args cs params kinds args = Some given ->
    length kinds = length params ∧ length args = length params.
Proof.
  induction params as [ | p params IH]; intros kinds args given.
  - destruct kinds; [ | cbn [region_args]; discriminate ].
    destruct args; [ | cbn [region_args]; discriminate ].
    intros _. split; reflexivity.
  - destruct kinds as [ | kind kinds]; [ cbn [region_args]; discriminate | ].
    destruct args as [ | [s l] args]; [ cbn [region_args]; discriminate | ].
    cbn [region_args].
    destruct (region_args cs params kinds args) as [rest | ] eqn:Hrest;
      [ | simpl; discriminate ].
    intros _. destruct (IH kinds args rest Hrest) as [Hkinds Hargs].
    cbn [length]. split; congruence.
Qed.

Lemma forallb_negb_skipn (n : nat) :
  ∀ bs : list bool,
    forallb negb bs = true -> forallb negb (skipn n bs) = true.
Proof.
  induction n as [ | n IH]; intros bs H; [ exact H | ].
  destruct bs as [ | b bs]; [ reflexivity | ].
  cbn [forallb] in H. apply andb_true_iff in H as [_ H].
  exact (IH bs H).
Qed.

(* In a state that has the property, a jump to a handler gives every
   kind [Region] of the handler a parameter and an argument. A handler
   of the state that runs is jumped to with as many arguments as it
   has parameters and kinds; a continuation of a caller takes no
   region. *)
Lemma checked_gives_regions (cert : certificate)
    (codes : list (code_id * code)) (rs : region_state) :
  state_checked cert codes rs -> regions_given rs = true.
Proof.
  intros
    (_ & _ & _ & _ & _ & _ & _ & _ & _ & cs & ms & Hcheck & _ & _ & Hconts
     & Houter).
  unfold regions_given.
  destruct (state_expr (rs_state rs))
    as [x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ] eqn:He;
    try reflexivity.
  cbn [check_expr] in Hcheck.
  set (k := ac_continuation ac) in *.
  unfold jump, apply_cont_args in Hcheck.
  destruct (lookup k (cs_conts cs)) as [ce | ] eqn:Hk.
  - specialize (Hconts k ce Hk).
    destruct ce as [params kinds entry | | local | ].
    + inversion Hconts; subst.
      destruct (region_args cs params (hm_params_kinds metadata) _)
        as [given | ] eqn:Hgiven; [ | discriminate ].
      destruct (region_args_lengths cs _ _ _ _ Hgiven) as [Hkinds Hargs].
      rewrite length_map in Hargs.
      unfold region_params. cbn [handler_kinds].
      rewrite !skipn_all2 by (rewrite length_map; lia).
      reflexivity.
    + discriminate.
    + pose proof (proj1 (callers_take_no_region cert codes rs) _ _ _ Hconts)
        as Hno.
      destruct (resolve k (state_conts (rs_state rs)))
        as [[params metadata handler] vars conts recursive | | | ];
        try reflexivity.
      cbn [takes_no_region] in Hno.
      apply andb_true_iff.
      split; apply forallb_negb_skipn; exact Hno.
    + pose proof (proj1 (callers_take_no_region cert codes rs) _ _ _ Hconts)
        as Hno.
      destruct (resolve k (state_conts (rs_state rs)))
        as [[params metadata handler] vars conts recursive | | | ];
        try reflexivity.
      cbn [takes_no_region] in Hno.
      apply andb_true_iff.
      split; apply forallb_negb_skipn; exact Hno.
  - unfold resolve. rewrite (Houter k Hk). reflexivity.
Qed.
