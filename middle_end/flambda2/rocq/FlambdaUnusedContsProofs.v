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


(* Proofs of the statements of FlambdaUnusedContsSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaContextSpec FlambdaContextProofs FlambdaNominalSpec
  FlambdaNominalProofs FlambdaNormaliseProofs FlambdaUnusedContsSpec.
Import ListNotations.

(* The test says yes of every continuation that is free. *)
Fixpoint cont_free_b_sound (k : continuation) (e : expr) {struct e} :
  cont_free_in_expr k e -> cont_free_b k e = true
with cont_free_in_handler_b_sound (k : continuation)
    (h : continuation_handler) {struct h} :
  cont_free_in_handler k h -> cont_free_in_handler_b k h = true.
Proof.
  - destruct e
      as [ x kind d body | bindings body | bound h body | handlers body | ap
         | ac | scrutinee arms | ];
      simpl.
    + apply (cont_free_b_sound k body).
    + apply (cont_free_b_sound k body).
    + intros [Hk | [Hne Hk]]; apply orb_true_iff;
        [ left; apply (cont_free_in_handler_b_sound k h), Hk | right ].
      apply andb_true_iff. split.
      * apply negb_true_iff, Pos.eqb_neq, Hne.
      * apply (cont_free_b_sound k body), Hk.
    + intros [_ [Hk | Hk]]; apply orb_true_iff;
        [ left | right; apply (cont_free_b_sound k body), Hk ].
      induction handlers as [ | [k' h] handlers IH]; simpl in Hk |- *;
        [ contradiction | ].
      apply orb_true_iff. destruct Hk as [Hk | Hk];
        [ left; apply (cont_free_in_handler_b_sound k h), Hk
        | right; exact (IH Hk) ].
    + intros [-> | ->]; apply orb_true_iff;
        [ left | right ]; apply Pos.eqb_refl.
    + intros Hk.
      exact (proj2 (among_In k (apply_cont_continuations ac)) Hk).
    + intros Hk. apply Exists_exists in Hk. destruct Hk as (arm & Hin & Hk).
      apply existsb_exists. exists arm. split; [ exact Hin | ].
      exact (proj2 (among_In k (apply_cont_continuations (snd arm))) Hk).
    + contradiction.
  - destruct h as [params metadata handler]. simpl.
    apply (cont_free_b_sound k handler).
Qed.

(* Each layer is a context around what the induction gives; a [Let_cont]
   whose continuation the test finds unused then goes. *)
Fixpoint remove_unused_conts_equivalent_aux (e : expr) :
  expr_equivalent e (remove_unused_conts e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    simpl; try (intros vars conts statics hp rd; apply equivalent_refl).
  - exact (plug_equivalent (Let_body x kind d Hole) _ _ (not_many _)
      (remove_unused_conts_equivalent_aux body)).
  - exact (plug_equivalent (Let_static_body bindings Hole) _ _ (not_many _)
      (remove_unused_conts_equivalent_aux body)).
  - pose proof (plug_equivalent
      (Let_cont_non_recursive_body k (Handler params metadata handler) Hole)
      _ _ (not_many _) (remove_unused_conts_equivalent_aux body)) as Hbody.
    simpl in Hbody.
    destruct (cont_free_b k (remove_unused_conts body)) eqn:Hfree.
    + apply (expr_equivalent_trans _ _ _ Hbody).
      exact (plug_equivalent
        (Let_cont_non_recursive_handler k params metadata Hole
           (remove_unused_conts body))
        _ _ (not_many _) (remove_unused_conts_equivalent_aux handler)).
    + apply (expr_equivalent_trans _ _ _ Hbody), unused_let_cont.
      intros Hk. apply cont_free_b_sound in Hk. congruence.
  - exact (plug_equivalent (Let_cont_recursive_body handlers Hole)
        _ _ (not_many _)
      (remove_unused_conts_equivalent_aux body)).
Qed.

Theorem remove_unused_conts_equivalent :
  remove_unused_conts_equivalent_statement.
Proof. exact remove_unused_conts_equivalent_aux. Qed.
