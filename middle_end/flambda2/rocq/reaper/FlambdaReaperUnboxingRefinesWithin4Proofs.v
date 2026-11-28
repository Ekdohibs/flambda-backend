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

(* The proofs of FlambdaReaperUnboxingRefinesWithin4Spec.v. Nothing here
   needs review.

   [within_poison_analysis_within]: [facts_poison_ok] is
   [facts_within_poison] with the conditions of [poison_sites_met] on the
   fields and arguments of [F1] that [F] does not have. [box_poison]: the
   facts of the first rebuilt unit, its tables and the checks are
   computed; under a renaming with [renaming_ok], the alias of [x10] (a
   variable of the unit) to the constants stays one, which [F] does not
   have and the unit's tables with poison do not give, [x10] having a
   use. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaReaperGraphSpec
  FlambdaReaperTraversal FlambdaReaperSolution FlambdaReaperSolutionProofs
  FlambdaReaperCallingConventions FlambdaReaperFlowSpec
  FlambdaReaperUnboxingRefinesUnitFlowSpec
  FlambdaReaperUnboxingRefinesWithinSpec
  FlambdaReaperUnboxingRefinesWithin2Spec
  FlambdaReaperUnboxingRefinesWithin3Spec
  FlambdaReaperUnboxingRefinesWithin2Proofs
  FlambdaReaperUnboxingRefinesWithin4Spec
  FlambdaReaperUnboxingRefinesWithin4SimProofs.
Import ListNotations.

Theorem within_poison_analysis_within :
  within_poison_analysis_within_statement.
Proof.
  intros is_local r F1 F T Hu Hw Hp. unfold analysis_within.
  rewrite Hu. cbn [andb].
  unfold facts_within_poison in Hw. unfold poison_sites_met in Hp.
  unfold facts_poison_ok.
  repeat rewrite andb_true_iff in Hw. repeat rewrite andb_true_iff in Hp.
  destruct Hw as (((((((((((H1 & H2) & H3) & _) & _) & H6) & H7) & H8) & H9)
                     & H10) & H11) & H12).
  destruct Hp as [Hc Ha].
  rewrite H1, H2, H3, Hc, Ha, H6, H7, H8, H9, H10, H11, H12. reflexivity.
Qed.

Lemma renamed_consts_ok (r : list (node * node)) (e : expr) :
  renaming_ok r e = true -> renamed r Node_all_constants = Node_all_constants.
Proof.
  intros Hok. unfold renamed.
  destruct (find (fun p => node_eqb (fst p) Node_all_constants) r)
    as [[a b] | ] eqn:E; [ | reflexivity ].
  apply find_some in E as [Hin Heq]. cbn [fst] in Heq.
  apply node_eqb_eq in Heq. subst a.
  unfold renaming_ok in Hok. rewrite forallb_forall in Hok.
  specialize (Hok _ Hin). discriminate Hok.
Qed.

Lemma f_alias_rename_facts (r : list (node * node)) (F : facts) :
  f_alias (rename_facts r F)
  = map (fun p => (renamed r (fst p), renamed r (snd p))) (f_alias F).
Proof.
  unfold rename_facts. cbn [f_alias]. apply map_ext. intros [a b].
  reflexivity.
Qed.

Lemma box_facts1_alias :
  f_alias BoxExample.facts1 = [(Node_variable 10%positive, Node_all_constants)].
Proof. vm_compute. reflexivity. Qed.

Lemma box_alias_rejected :
  mem_pair (Node_variable 10%positive) Node_all_constants (f_alias
    BoxExample.facts)
  || s_alias
       (solution_of_tables
          (poisoned_tables BoxExample.facts BoxExample.tables))
       (Node_variable 10%positive) Node_all_constants
  = false.
Proof. vm_compute. reflexivity. Qed.

Theorem box_poison : box_poison_statement.
Proof.
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  intros r Hok. unfold analysis_within, facts_poison_ok.
  rewrite f_alias_rename_facts, box_facts1_alias. cbn [map fst snd].
  rewrite (renamed_variable r BoxExample.unit_expr 10%positive Hok)
    by (apply among_In; vm_compute; reflexivity).
  rewrite (renamed_consts_ok r BoxExample.unit_expr Hok).
  cbn [forallb]. rewrite box_alias_rejected. cbn [andb].
  apply andb_false_r.
Qed.
