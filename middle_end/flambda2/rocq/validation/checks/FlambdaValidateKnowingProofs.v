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

(* Proofs of the statements of FlambdaValidateKnowingSpec.v. Nothing
   here needs review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaNormaliseSpec FlambdaNormaliseProofs
  FlambdaUnusedStaticsSpec FlambdaNominalSpec FlambdaValidateSpec
  FlambdaValidateProofs FlambdaKeepsScopeSpec FlambdaKeepsScopeProofs
  FlambdaValidatorEquivarianceSpec FlambdaValidatorEquivarianceProofs
  FlambdaValidateKnowingSpec.
From OxCamlFlambda2Types Require FlambdaEnvAliasesFollowSpec
  FlambdaEnvAliasesFollowProofs.
From OxCamlFlambda2Types Require Import FlambdaEnvAliasesFollowScopeSpec
  FlambdaEnvAliasesFollowScopeProofs.
Import ListNotations.

(* The three laws of [follow_aliases_knowing]: those of the function
   that follows aliases, taken into the bodies of code. *)
Lemma follow_aliases_knowing_equivalent (e : expr) :
  expr_equivalent e (follow_aliases_knowing e).
Proof.
  unfold follow_aliases_knowing.
  eapply expr_equivalent_trans;
    [ apply FlambdaEnvAliasesFollowProofs.follow_aliases_equivalent | ].
  apply rewrite_code_equivalent.
  exact FlambdaEnvAliasesFollowProofs.follow_aliases_equivalent.
Qed.

Lemma follow_aliases_knowing_keeps_scope : keeps_scope follow_aliases_knowing.
Proof.
  exact
    (keeps_scope_compose _ _ follow_aliases_knowing_nothing_keeps_scope
       (rewrite_code_keeps_scope _
          follow_aliases_knowing_nothing_keeps_scope)).
Qed.

Lemma rename_follow_aliases_knowing :
  rewrite_equivariant follow_aliases_knowing.
Proof.
  intros r e. unfold follow_aliases_knowing.
  rewrite (FlambdaEnvAliasesFollowProofs.rename_follow_aliases_empty r e).
  exact
    (rewrite_code_equivariant _
       FlambdaEnvAliasesFollowProofs.rename_follow_aliases_empty r _).
Qed.

Theorem normalise_knowing_refines : normalise_knowing_refines_statement.
Proof.
  intros round rounds e Hround. revert e. unfold normalise_knowing.
  assert (Hone : ∀ e : expr, expr_refines e (knowing_round round e)).
  { intros e. unfold knowing_round.
    eapply expr_refines_trans; [ | apply Hround ].
    apply expr_equivalent_refines.
    apply follow_aliases_knowing_equivalent. }
  induction rounds as [ | rounds IH]; intros e; cbn [iterate];
    [ apply expr_refines_refl | ].
  cbv zeta.
  match goal with
  | |- context [if ?same then _ else _] => destruct same; [ apply Hone | ]
  end.
  exact (expr_refines_trans _ _ _ (Hone e) (IH _)).
Qed.

Theorem normalise_knowing_equivalent : normalise_knowing_equivalent_statement.
Proof.
  intros round rounds e Hround. revert e. unfold normalise_knowing.
  assert (Hone : ∀ e : expr, expr_equivalent e (knowing_round round e)).
  { intros e. unfold knowing_round.
    eapply expr_equivalent_trans; [ | apply Hround ].
    apply follow_aliases_knowing_equivalent. }
  induction rounds as [ | rounds IH]; intros e; cbn [iterate];
    [ apply expr_equivalent_refl | ].
  cbv zeta.
  match goal with
  | |- context [if ?same then _ else _] => destruct same; [ apply Hone | ]
  end.
  exact (expr_equivalent_trans _ _ _ (Hone e) (IH _)).
Qed.

Theorem validate_knowing_sound : validate_knowing_sound_statement.
Proof.
  intros round_before round_after rounds before after Hbefore Hafter
    Hnormalised.
  apply (refines_trans _ _ _
    (expr_refines_program _ _
       (normalise_knowing_refines round_before rounds before Hbefore))).
  apply (refines_trans _ _ _ Hnormalised).
  apply expr_refines_program, expr_equivalent_refines, expr_equivalent_sym.
  exact (normalise_knowing_equivalent round_after rounds after Hafter).
Qed.

Theorem validate_knowing_sound_everywhere
    : validate_knowing_sound_everywhere_statement.
Proof.
  intros round_before round_after rounds before after Hbefore Hafter
    Hnormalised.
  apply (expr_refines_trans _ _ _
    (normalise_knowing_refines round_before rounds before Hbefore)).
  apply (expr_refines_trans _ _ _ Hnormalised).
  apply expr_equivalent_refines, expr_equivalent_sym.
  exact (normalise_knowing_equivalent round_after rounds after Hafter).
Qed.

(* ======================================================================== *)
(* The second asking                                                        *)
(* ======================================================================== *)

Theorem validate_unit_knowing_sound : validate_unit_knowing_sound_statement.
Proof.
  intros rounds unused slots G Ls kept before after Hvalidated.
  apply (validate_knowing_sound
           (normalise_before_round_avoiding
              (unit_continuations Ls ++ map snd G) kept)
           (normalise_after_round kept) rounds before after
           (normalise_before_round_avoiding_refines
              (unit_continuations Ls ++ map snd G) kept)
           (normalise_after_round_equivalent kept)).
  exact (validate_unit_sound rounds unused slots G Ls kept _ _ Hvalidated).
Qed.

Theorem validate_unit_knowing_sound_everywhere
    : validate_unit_knowing_sound_everywhere_statement.
Proof.
  intros rounds unused slots G Ls kept before after Hvalidated.
  apply (validate_knowing_sound_everywhere
           (normalise_before_round_avoiding
              (unit_continuations Ls ++ map snd G) kept)
           (normalise_after_round kept) rounds before after
           (normalise_before_round_avoiding_refines
              (unit_continuations Ls ++ map snd G) kept)
           (normalise_after_round_equivalent kept)).
  exact (validate_unit_sound_everywhere rounds unused slots G Ls kept _ _
           Hvalidated).
Qed.

(* ======================================================================== *)
(* Renaming and scope                                                       *)
(* ======================================================================== *)

Lemma knowing_round_keeps_scope (round : expr -> expr) :
  keeps_scope round -> keeps_scope (knowing_round round).
Proof.
  intros Hround.
  exact (keeps_scope_compose _ round follow_aliases_knowing_keeps_scope
    Hround).
Qed.

Theorem normalise_knowing_keeps_scope
    : normalise_knowing_keeps_scope_statement.
Proof.
  intros round rounds Hround.
  exact (keeps_scope_iterate rounds (knowing_round round)
    (knowing_round_keeps_scope round Hround)).
Qed.

Theorem normalise_knowing_equivariant
    : normalise_knowing_equivariant_statement.
Proof.
  intros round rounds Hround r e. unfold normalise_knowing.
  apply (iterate_equivariant rounds (knowing_round round)).
  intros r' e'. unfold knowing_round.
  rewrite (rename_follow_aliases_knowing r' e').
  apply Hround.
Qed.

Theorem normalise_knowing_equivariant_up_to
    : normalise_knowing_equivariant_up_to_statement.
Proof.
  intros round rounds Hscope Hround.
  apply (equivariant_up_to_iterate rounds (knowing_round round));
    [ exact (knowing_round_keeps_scope round Hscope) | ].
  apply (equivariant_up_to_compose _ round
    follow_aliases_knowing_keeps_scope
    Hscope);
    [ | exact Hround ].
  apply rewrite_equivariant_up_to.
  exact rename_follow_aliases_knowing.
Qed.

(* The following of aliases comes before the round and commutes with
   renaming, so the renaming that the round gives is kept; the rounds
   are given the same continuations. *)
Theorem normalise_knowing_equivariant_up_to_avoiding
    : normalise_knowing_equivariant_up_to_avoiding_statement.
Proof.
  intros round rounds Hscope Hround.
  exact
    (equivariant_up_to_avoiding_iterate rounds
       (fun avoid => knowing_round (round avoid))
       (fun avoid => knowing_round_keeps_scope (round avoid) (Hscope avoid))
       (equivariant_up_to_avoiding_before round _
          follow_aliases_knowing_keeps_scope rename_follow_aliases_knowing
          Hround)).
Qed.

(* The second asking on renamed terms. The continuations that the round
   of the term before is given for the renamed arguments are those for
   the arguments, renamed. That round gives its result under a second
   renaming, which does to the free names of the result and to those
   continuations what the first does; the round of the term after
   commutes with the renaming; and [validate_unit] answers the same of
   a term before that is under such a second renaming. *)
Theorem validate_unit_knowing_equivariant
    : validate_unit_knowing_equivariant_statement.
Proof.
  intros r rounds unused slots G Ls kept before after Hclosed.
  unfold validate_unit_knowing.
  rewrite unit_continuations_renamed, loop_continuations_renamed, <- map_app.
  set (avoid := unit_continuations Ls ++ map snd G).
  destruct
    (normalise_knowing_equivariant_up_to_avoiding
       (fun avoid => normalise_before_round_avoiding avoid kept) rounds
       (fun avoid => normalise_before_round_avoiding_scope avoid kept)
       (normalise_before_round_avoiding_up_to kept) r avoid before Hclosed)
    as (r1 & Halike & Havoid & Heq).
  cbv beta in Halike, Heq. rewrite Heq.
  rewrite
    (normalise_knowing_equivariant (normalise_after_round kept) rounds
       (normalise_after_round_equivariant kept) r after).
  apply validate_unit_renamed_alike.
  - exact
      (proj1
         (normalise_knowing_keeps_scope
            (normalise_before_round_avoiding avoid kept) rounds
            (normalise_before_round_avoiding_scope avoid kept) before
            Hclosed)).
  - exact Halike.
  - exact Havoid.
Qed.
