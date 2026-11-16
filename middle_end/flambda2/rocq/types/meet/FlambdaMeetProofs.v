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



(* Proofs of the statements of FlambdaMeetSpec.v. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaClosuresSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaTypeWellFormedSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaEnvAliasesSpec FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvProofs FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvInvariantProofs FlambdaTypingEnvKindedSpec
  FlambdaMeetGrammar FlambdaExpandHead FlambdaMeetEnv FlambdaMeet
  FlambdaMeetSpec.
Import ListNotations.

(* ======================================================================== *)
(* What the contracts give                                                  *)
(* ======================================================================== *)

Theorem meet_answers_bottom : meet_answers_bottom_statement.
Proof.
  intros meet Hsound statics valuation tenv t1 t2 v Hinputs Hstatics
    Hsatisfied Hmeet [Ht1 Ht2].
  destruct
    (Hsound statics valuation tenv t1 t2 v _ Hinputs Hstatics Hsatisfied Ht1
       Ht2 Hmeet)
    as [t [tenv' [valuation' [Hresult _]]]].
  discriminate Hresult.
Qed.

Theorem add_equation_no_poison : add_equation_no_poison_statement.
Proof.
  intros add_equation Hsound statics valuation tenv tenv' n ty v Hfits Hty
    Hstatics Hsatisfied Hpoison Hvalue Hv Hadd.
  apply (Hsound statics valuation tenv tenv' n ty v Hfits Hty); try assumption.
  apply satisfied_no_poison_strictly; assumption.
Qed.

Theorem fits_types : fits_types_statement.
Proof.
  intros tenv n ty binding_time_and_mode
    ([Hwf _] & Htypes & (Hkinded & _ & _) & Hequations & _) Hlookup.
  apply typing_env_well_formed_iff in Hwf.
  destruct Hwf as ((_ & _ & _ & Hclosed & _) & _).
  split; [ exact (Htypes n ty binding_time_and_mode Hlookup) | ]. split.
  { intros x Hin. rewrite mem_var_is_bound.
    exact (Hclosed n ty binding_time_and_mode x Hlookup Hin). }
  split.
  - exact (Hkinded n ty binding_time_and_mode Hlookup).
  - exact (Hequations n ty binding_time_and_mode Hlookup).
Qed.

(* ======================================================================== *)
(* The step of the compiler's meet that is not sound                        *)
(* ======================================================================== *)

Theorem get_tag_reduction_unsound : get_tag_reduction_unsound_statement.
Proof.
  exists empty_static_env, [], (Value_const (Tagged_immediate 7)), 0%Z.
  split; [ | split ].
  - left. exists 7%Z. reflexivity.
  - intros [k Hpoison]. discriminate Hpoison.
  - intros Hblock.
    apply blocks_with_these_tags_satisfied in Hblock
      as [tag [fields [Hblock _]]].
    discriminate Hblock.
Qed.

Lemma lookup_among_unknown_cases {A : Type} (tag : Z) (tags : list Z) :
  tag ∈ tags ->
  Keyed.lookup Z.eqb tag (map (fun tag => (tag, @Or_unknown.Unknown A)) tags)
  = Some Or_unknown.Unknown.
Proof.
  induction tags as [ | tag' tags IH]; simpl; [ intros [] | ].
  intros [-> | Hin].
  - rewrite Z.eqb_refl. reflexivity.
  - destruct (Z.eqb tag tag'); [ reflexivity | exact (IH Hin) ].
Qed.

(* A number among some numbers that is a tag is among the tags that the
   step computes. *)
Lemma tag_among_tags (n : Z) (numbers : list Z) :
  n ∈ numbers -> is_tag n = true ->
  n ∈ Keyed.dedup Z.eqb (filter is_tag numbers).
Proof.
  intros Hin Htag.
  apply (keyed_among_in Z Z.eqb z_eqb_decides_equality).
  rewrite (keyed_among_dedup Z Z.eqb z_eqb_decides_equality).
  apply (keyed_among_in Z Z.eqb z_eqb_decides_equality).
  apply filter_In. split; assumption.
Qed.

Theorem get_tag_reduction_repaired : get_tag_reduction_repaired_statement.
Proof.
  intros statics vars w n numbers alloc_mode Hrelation Hin.
  unfold immediate_or_blocks_with_these_tags.
  apply (create_variant_satisfied statics vars w).
  destruct Hrelation as [[i ->] | [[fields Hblock] Hrange]].
  - left. exists i. split; [ reflexivity | ]. split; exact I.
  - right. exists n, fields. split; [ exact Hblock | ]. split; [ | exact I ].
    apply row_like_for_blocks_find.
    assert (Htag : is_tag n = true).
    { unfold FlambdaTypeMeaningSpec.is_tag in Hrange. unfold is_tag.
      apply andb_true_iff. split; apply Z.leb_le; lia. }
    rewrite (lookup_among_unknown_cases n _
               (tag_among_tags n numbers Hin Htag)).
    exact I.
Qed.
