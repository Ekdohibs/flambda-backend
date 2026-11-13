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

(* The proofs of FlambdaReaperTupledSpec.v. The statements about the
   graphs, the solutions, the rebuilt terms and the runs are computed;
   those about refinement follow from the runs, as a state has one
   behaviour at most. Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaValueSpec
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaUnusedStaticsSpec
  FlambdaReaperGraphSpec FlambdaReaperTupledSpec.
Import ListNotations.

(* ======================================================================== *)
(* The graphs and the solutions                                             *)
(* ======================================================================== *)

Theorem keep_alive_facts : keep_alive_facts_statement.
Proof. split; vm_compute; reflexivity. Qed.

Theorem unit_well_formed : unit_well_formed_statement.
Proof. split; vm_compute; reflexivity. Qed.

Theorem with_keep_alive_solution_passes :
  with_keep_alive_solution_passes_statement.
Proof.
  intros is_local [<- | [<- | []]]; split; vm_compute; reflexivity.
Qed.

Theorem without_keep_alive_solution_passes :
  without_keep_alive_solution_passes_statement.
Proof.
  intros is_local [<- | [<- | []]]; split; vm_compute; reflexivity.
Qed.

(* The rule [any_usage_use], at the use of the fix: the code id of
   [ignore] has a use, and 11 has no [any_usage]. *)
Theorem keep_alive_facts_needed : keep_alive_facts_needed_statement.
Proof.
  intros is_local Hclosed.
  assert (Hused :
    s_has_usage (solution_of_tables without_keep_alive_solution)
      (Node_code_id ignore_code_id) = true)
    by (vm_compute; reflexivity).
  assert (Huse :
    g_use (graph_of_facts with_keep_alive_graph)
      (Node_code_id ignore_code_id) (Node_made 11))
    by (vm_compute; left; reflexivity).
  pose proof (any_usage_use _ _ _ Hclosed _ _ Hused Huse) as Hany.
  vm_compute in Hany. discriminate.
Qed.

(* ======================================================================== *)
(* The rebuilt terms and the runs                                           *)
(* ======================================================================== *)

Theorem with_keep_alive_rebuilt : with_keep_alive_rebuilt_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem without_keep_alive_rebuilt : without_keep_alive_rebuilt_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem with_keep_alive_rebuild_runs : with_keep_alive_rebuild_runs_statement.
Proof. split; exists 20; vm_compute; reflexivity. Qed.

Theorem unit_does_not_go_wrong : unit_does_not_go_wrong_statement.
Proof.
  intros Hwrong.
  destruct with_keep_alive_rebuild_runs as [Hexits _].
  discriminate (behaviour_unique _ _ _ Hwrong Hexits).
Qed.

(* Two states that exit alike refine each other: each has that one
   behaviour. *)
Lemma same_exit_refines (s s' : state) (k : continuation) (vs : list value)
    (hp : heap) :
  behaves s (Exits k vs hp) -> behaves s' (Exits k vs hp) -> refines s s'.
Proof.
  intros Hs Hs'. split.
  - intros k' vs' hp' Hs''.
    rewrite <- (behaviour_unique _ _ _ Hs Hs''). exact Hs'.
  - intros b _ Hb. left.
    rewrite (behaviour_unique _ _ _ Hb Hs'). exact Hs.
Qed.

Theorem with_keep_alive_refines : with_keep_alive_refines_statement.
Proof.
  destruct with_keep_alive_rebuild_runs as [Hunit Hrebuilt].
  exact (same_exit_refines _ _ _ _ _ Hunit Hrebuilt).
Qed.

Theorem without_keep_alive_rebuild_goes_wrong :
  without_keep_alive_rebuild_goes_wrong_statement.
Proof. exists 20. vm_compute. reflexivity. Qed.

(* The state is the one that the run has reached after four steps: the
   [Let_static], the [Let_cont], the call of [pass] and its return. *)
Theorem without_keep_alive_tuple_is_poison :
  without_keep_alive_tuple_is_poison_statement.
Proof.
  exists 4,
    (match run 4 (on_its_own rebuilt_without_keep_alive) with
     | Step s => s
     | _ => on_its_own rebuilt_without_keep_alive
     end).
  vm_compute. repeat split.
Qed.

(* The rebuilt unit goes wrong where the unit does not, which no
   refinement allows. *)
Theorem without_keep_alive_not_refinement :
  without_keep_alive_not_refinement_statement.
Proof.
  intros [_ Hback].
  destruct (Hback Goes_wrong ltac:(discriminate)
              without_keep_alive_rebuild_goes_wrong)
    as [Hwrong | Hwrong];
    exact (unit_does_not_go_wrong Hwrong).
Qed.
