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

(* Proofs of the statements of FlambdaExitWorldActionsSpec.v. This file
   needs no review: the statements are there.

   A world is what the last state of a run comes to with the static
   environment of that state, so each statement about [exits_world] is
   that of FlambdaEquivarianceSpec.v or of FlambdaLocationsMachineSpec.v
   about a run, then about a step, with what the action does to the
   static environment of a state. For a permutation of locations the
   run and the last step each give a permutation; the two agree on the
   locations that the heap of the last state binds, the locations of
   its static environment are among those, and a static environment
   moved by two permutations that agree on its locations is the
   same. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaNominalSpec FlambdaNominalProofs FlambdaAlphaSpec
  FlambdaAlphaProofs FlambdaEquivarianceSpec
  FlambdaEquivarianceProofs FlambdaHeapSpec FlambdaExitWorldSpec
  FlambdaExitWorldProofs FlambdaLocationsSpec FlambdaLocationsProofs
  FlambdaLocationsPrimitiveSpec FlambdaLocationsMachineSpec
  FlambdaLocationsMachineProofs FlambdaExitWorldActionsSpec.
Import ListNotations.

(* ======================================================================== *)
(* A renaming                                                               *)
(* ======================================================================== *)

Lemma rename_static_env_inverse_l (r : renaming) (statics : static_env) :
  rename_static_env (inverse_renaming r) (rename_static_env r statics)
  = statics.
Proof.
  rewrite <- rename_static_env_compose.
  rewrite (rename_static_env_ext _ identity_renaming);
    [ apply rename_static_env_id | | ].
  - apply compose_inverse_l_variable.
  - apply compose_inverse_l_continuation.
Qed.

Lemma rename_static_env_inverse_r (r : renaming) (statics : static_env) :
  rename_static_env r (rename_static_env (inverse_renaming r) statics)
  = statics.
Proof.
  rewrite <- rename_static_env_compose.
  rewrite (rename_static_env_ext _ identity_renaming);
    [ apply rename_static_env_id | | ].
  - apply compose_inverse_r_variable.
  - apply compose_inverse_r_continuation.
Qed.

Theorem rename_exit_world_inverse : rename_exit_world_inverse_statement.
Proof.
  intros r [k vs statics hp]. unfold rename_exit_world. simpl. split.
  - rewrite rename_continuation_inverse_l, rename_static_env_inverse_l.
    reflexivity.
  - rewrite rename_continuation_inverse_r, rename_static_env_inverse_r.
    reflexivity.
Qed.

(* One direction, for every renaming; the other is this one for the
   inverse. *)
Lemma exits_world_renamed (r : renaming) (s : state) (w : exit_world) :
  exits_world s w -> exits_world (rename_state r s) (rename_exit_world r w).
Proof.
  intros (n & last & Hrun & Hstep & Hstatics).
  exists n, (rename_state r last). split; [ | split ].
  - rewrite run_equivariant, Hrun. reflexivity.
  - rewrite step_equivariant, Hstep. reflexivity.
  - simpl. rewrite <- Hstatics. reflexivity.
Qed.

Theorem exits_world_equivariant : exits_world_equivariant_statement.
Proof.
  intros r s w. split; [ | apply exits_world_renamed ].
  intros Hexits.
  apply (exits_world_renamed (inverse_renaming r)) in Hexits.
  rewrite (proj1 (rename_state_inverse r s)) in Hexits.
  rewrite (proj1 (rename_exit_world_inverse r w)) in Hexits.
  exact Hexits.
Qed.

(* A world that a renamed state leaves is the renaming of one that the
   state leaves. *)
Lemma exits_world_of_renamed (r : renaming) (s : state) (w : exit_world) :
  exits_world (rename_state r s) w ->
  exits_world s (rename_exit_world (inverse_renaming r) w).
Proof.
  intros Hexits. apply (exits_world_equivariant r).
  rewrite (proj2 (rename_exit_world_inverse r w)). exact Hexits.
Qed.

Theorem refines_world_equivariant : refines_world_equivariant_statement.
Proof.
  intros r R source target [Hexits Hends]. split.
  - intros w Hw. apply exits_world_of_renamed in Hw.
    destruct (Hexits _ Hw) as (w' & Hw' & Hrelated).
    exists (rename_exit_world r w'). split.
    + apply exits_world_renamed. exact Hw'.
    + unfold rename_world_relation.
      rewrite (proj1 (rename_exit_world_inverse r w')). exact Hrelated.
  - intros b Hb Hbehaves. apply behaves_renamed in Hbehaves.
    destruct (Hends (rename_behaviour (inverse_renaming r) b))
      as [Hwrong | (w & Hw)].
    + destruct b; [ discriminate | discriminate | exact Hb ].
    + exact Hbehaves.
    + left. apply behaves_renamed. exact Hwrong.
    + right. exists (rename_exit_world r w). apply exits_world_renamed.
      exact Hw.
Qed.

Theorem refines_world_kept : refines_world_kept_statement.
Proof.
  intros r R source target Hkept Hrefines.
  apply (refines_world_weaken (rename_world_relation r R) R).
  - intros w w' Hrelated. unfold rename_world_relation in Hrelated.
    apply Hkept in Hrelated.
    rewrite !(proj2 (rename_exit_world_inverse r _)) in Hrelated.
    exact Hrelated.
  - apply refines_world_equivariant. exact Hrefines.
Qed.

Theorem same_exit_equivariant : same_exit_equivariant_statement.
Proof.
  intros r [k vs statics hp] [k' vs' statics' hp']. unfold same_exit.
  simpl.
  split; intros (Hk & Hvs & Hhp); (split; [ | split; assumption ]).
  - apply (f_equal (rename_continuation (inverse_renaming r))) in Hk.
    rewrite !rename_continuation_inverse_l in Hk. exact Hk.
  - rewrite Hk. reflexivity.
Qed.

(* ======================================================================== *)
(* A permutation of locations                                               *)
(* ======================================================================== *)

Lemma locations_of_statics_in_state (s : state) (l : location) :
  l ∈ locations_of_static_env (state_statics s) -> l ∈ locations_of_state s.
Proof.
  intros Hl. unfold locations_of_state. apply in_or_app. right.
  apply in_or_app. right. apply in_or_app. left. exact Hl.
Qed.

Theorem exits_world_keeps_locations_bound :
  exits_world_keeps_locations_bound_statement.
Proof.
  intros s [k vs statics hp rd] Hbound (n & last & Hrun & Hstep & Hstatics).
  simpl in *.
  pose proof (run_keeps_locations_bound n s Hbound) as Hkeeps.
  rewrite Hrun in Hkeeps. simpl in Hkeeps.
  destruct Hkeeps as [Hlast Hkept].
  pose proof (step_keeps_locations_bound last Hlast) as Hexit.
  rewrite Hstep in Hexit. simpl in Hexit.
  destruct Hexit as (Hhp & Hvs & Hrd & Hkept').
  split.
  - intros l Hl. simpl in Hl.
    apply in_app_or in Hl as [Hl | Hl]; [ exact (Hvs l Hl) | ].
    apply in_app_or in Hl as [Hl | Hl];
      [ | apply in_app_or in Hl as [Hl | Hl];
          [ exact (Hhp l Hl) | exact (Hrd l Hl) ] ].
    apply Hkept', Hlast, locations_of_statics_in_state.
    rewrite Hstatics. exact Hl.
  - intros l Hl. exact (Hkept' l (Hkept l Hl)).
Qed.

Theorem exits_world_permuted : exits_world_permuted_statement.
Proof.
  intros p s [k vs statics hp] Hbound (n & last & Hrun & Hstep & Hstatics).
  simpl in *.
  destruct (run_permuted p n s Hbound) as (p1 & Hagree1 & Hmoved).
  rewrite Hrun in Hmoved. simpl in Hmoved.
  pose proof (run_keeps_locations_bound n s Hbound) as Hkeeps.
  rewrite Hrun in Hkeeps. simpl in Hkeeps.
  destruct Hkeeps as [Hlast Hkept].
  destruct (step_permuted p1 last Hlast) as (p2 & Hagree2 & Hstep2).
  rewrite Hstep in Hstep2. simpl in Hstep2.
  exists p2. split.
  - intros l Hl. rewrite (Hagree2 l (Hkept l Hl)). exact (Hagree1 l Hl).
  - exists n, (permute_state p1 last). split; [ exact Hmoved | split ].
    + exact Hstep2.
    + simpl. rewrite Hstatics.
      apply (act_ext _ _ static_env_laws).
      intros l Hl. symmetry.
      apply Hagree2, Hlast, locations_of_statics_in_state.
      rewrite Hstatics. exact Hl.
Qed.

Theorem same_exit_permuted : same_exit_permuted_statement.
Proof.
  intros p [k vs statics hp] [k' vs' statics' hp'] (Hk & Hvs & Hhp).
  simpl in *. subst. unfold same_exit. simpl. repeat split.
Qed.

(* ======================================================================== *)
(* The two actions                                                          *)
(* ======================================================================== *)

Theorem rename_permute_exit_world : rename_permute_exit_world_statement.
Proof. intros r p [k vs [codes values] hp]. reflexivity. Qed.
