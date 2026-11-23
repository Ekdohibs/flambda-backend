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

(* The proofs of FlambdaReaperFixFailsSlotsSpec.v. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperLeastSpec FlambdaReaperRebuild
  FlambdaReaperUnboxing FlambdaReaperFixFailsSlotsSpec
  FlambdaNameAbstractionSpec.
From OxCamlFlambda2 Require FlambdaReaperFlowProofs FlambdaReaperTypesProofs
  FlambdaReaperTypesTraversalSpec FlambdaReaperTypesTraversalProofs
  FlambdaReaperTypesTraversalUnaliasedProofs
  FlambdaReaperClosuresTablesProofs.
Import ListNotations.

Module TP := FlambdaReaperTypesProofs.
Module TTP := FlambdaReaperTypesTraversalProofs.
Module FP := FlambdaReaperFlowProofs.
Module TUP := FlambdaReaperTypesTraversalUnaliasedProofs.
Module CTP := FlambdaReaperClosuresTablesProofs.

Lemma is_bound_lookup (g : function_slot) {A : Type}
    (l : list (function_slot * A)) :
  is_bound g l = true -> ∃ a, lookup g l = Some a.
Proof.
  unfold is_bound. destruct (lookup g l) as [a | ]; [ | discriminate ].
  intros _. exists a. reflexivity.
Qed.

Lemma is_bound_of_in_fst (g : function_slot) {A : Type}
    (l : list (function_slot * A)) :
  g ∈ map fst l -> is_bound g l = true.
Proof.
  induction l as [ | [k a] l IH]; intros H; [ destruct H | ].
  unfold is_bound. cbn [lookup].
  destruct (Pos.eqb g k) eqn:E; [ reflexivity | ].
  cbn [map fst] in H.
  destruct H as [<- | H]; [ rewrite Pos.eqb_refl in E; discriminate | ].
  exact (IH H).
Qed.

Theorem changed_decls_all_renamed : changed_decls_all_renamed_statement.
Proof.
  intros is_local u e T dominated to_unbox tcr x slot decls vs r Hflow Hwfu
    Honce Hleast Hdom Hx Hlk g Hg.
  set (F := traverse_unit Auto u e) in *.
  pose proof (TTP.function_slots_of_sets_traversal u e Hflow Hwfu Honce)
    as Hsets.
  pose proof (TTP.function_slots_mutual_traversal u e Hflow Hwfu Honce)
    as Hmut.
  pose proof (TUP.closures_unaliased_traversal u e Hflow Hwfu Honce) as Hu.
  pose proof (TTP.changed_closures_one_allocation_unaliased is_local F T tcr
                Hleast Hsets Hmut Hu) as Hone.
  pose proof (TP.least_closed_tables is_local F T (or_introl Hleast)) as Hc.
  pose proof (CTP.closure_let_self u e x slot decls vs Hflow Hwfu Hx) as Hxx.
  unfold unit_changed_representations in Hlk.
  destruct (TTP.rinv_fold is_local F T (bound_code_my_closures e) dominated
              to_unbox tcr Hc Hsets Hmut Hdom Hone (fresh_name (slots_of e)))
    as (seen & Hinv1 & _).
  destruct (Hinv1 (Node_variable x) r slot (Node_variable x) Hxx Hlk)
    as (x0 & _ & Hcur & Hmap & _).
  pose proof (proj1 (TTP.sfs_iff F Hsets x0 (Node_variable x) (cr_current r))
                Hcur) as Hx0.
  (* The [Let] holds every closure of its set at its slot. *)
  destruct (is_bound_lookup slot decls) as (id0 & Hl).
  { assert (Hsw : sets_well_formed e = true).
    { unfold well_formed_unit in Hwfu.
      apply andb_true_iff in Hwfu as [Hwfu _].
      apply andb_true_iff in Hwfu as [Hwfu _].
      apply andb_true_iff in Hwfu as [Hwfu _].
      apply andb_true_iff in Hwfu as [_ Hwfu]. exact Hwfu. }
    pose proof (proj2 (proj1 CTP.let_ok e x _ Hx) Hsw) as Hn.
    cbn [named_sets_well_formed set_well_formed] in Hn.
    apply andb_true_iff in Hn as [Hn _]. apply andb_true_iff in Hn as [Hn _].
    exact Hn. }
  destruct (is_bound_lookup g decls Hg) as (idg & Hlg).
  destruct (FP.traverse_unit_facts Auto u e True Hflow (or_introl I))
    as (noted & deps & cds & Hedges & _).
  destruct (proj1 (CTP.let_edges _ noted deps cds _ _ _ True False) e None
              (unit_conts u) Hedges x _ Hx)
    as (ctx & Hset).
  cbn [FP.named_edges] in Hset.
  destruct (Hset id0 Hl) as (sites & Hsx & Hall).
  destruct (Hall slot id0 Hl) as (_ & Hfs & _).
  pose proof (Hfs g idg Hlg) as Hxg. rewrite Hsx in Hxg.
  pose proof Hsets as (_ & _ & Hcl).
  pose proof (proj2 (Hcl x0 (Node_variable x) (sites g) (cr_current r) g
                       Hx0) Hxg) as Hx0g.
  pose proof (proj2 (TTP.sfs_iff F Hsets x0 (sites g) g) Hx0g) as Hin.
  apply is_bound_of_in_fst. rewrite Hmap.
  exact (in_map fst _ _ Hin).
Qed.

Lemma rename_slots_fst (L : list function_slot) :
  ∀ n : positive, map fst (fst (rename_slots L n)) = L.
Proof.
  induction L as [ | fs L IH]; intros n; [ reflexivity | ].
  cbn [rename_slots]. specialize (IH (Pos.succ n)).
  destruct (rename_slots L (Pos.succ n)) as [rest n'] eqn:E.
  cbn [fst map] in *. rewrite IH. reflexivity.
Qed.

(* Every representation of the fold has a new slot for its current
   slot. *)
Definition cinv (acc : list (node * closure_repr)) : Prop :=
  ∀ (c : node) (r : closure_repr),
    (c, r) ∈ acc -> cr_current r ∈ map fst (cr_function_slots r).

Lemma cinv_fold (F : facts) (T : tables) (cm : finite_map variable)
    (dominated : list (node * node)) (to_unbox : list node)
    (is_local : field -> bool) (tcr : list node) (n0 : positive) :
  cinv
    (fst (changed_representations F T cm dominated to_unbox is_local tcr n0)).
Proof.
  unfold changed_representations.
  match goal with
  | |- context [List.fold_left ?h tcr _] => set (step := h)
  end.
  assert (Hgen :
    ∀ (l : list node) (acc : list (node * closure_repr)) (seen : list node)
      (n : positive),
      cinv acc -> cinv (fst (fst (List.fold_left step l (acc, seen, n))))).
  { induction l as [ | x l IH]; intros acc seen n Hinv; [ exact Hinv | ].
    cbn [List.fold_left].
    destruct (step (acc, seen, n) x) as [[acc' seen'] n'] eqn:Hstep.
    apply IH.
    unfold step in Hstep. cbv beta match in Hstep.
    destruct (mem_node x seen);
      [ injection Hstep as <- <- <-; exact Hinv | ].
    destruct (set_function_slots F x) as [ | p0 l0] eqn:Hsx;
      [ injection Hstep as <- <- <-; exact Hinv | ].
    destruct (mk_unboxed_fields _ _ _ _ _ _ _ _ _) as [fields n1].
    destruct (rename_slots (map fst (p0 :: l0)) n1) as [fss n2] eqn:Hren.
    injection Hstep as <- <- <-.
    pose proof (rename_slots_fst (map fst (p0 :: l0)) n1) as Hfst.
    rewrite Hren in Hfst. cbn [fst] in Hfst.
    intros c r Hin. apply in_app_or in Hin as [Hin | Hin];
      [ exact (Hinv c r Hin) | ].
    assert (Hin' :
      (c, r)
      ∈ flat_map
          (fun '(fs, f) =>
             flat_map
               (fun '(a, c) =>
                  if node_eqb a f
                  then [(c, Closure_repr fields fss fs)]
                  else [])
               dominated)
          (p0 :: l0)) by exact Hin.
    apply TTP.in_new in Hin' as (fs & f & Hfs & _ & ->).
    cbn [cr_function_slots cr_current]. rewrite Hfst.
    exact (in_map fst _ _ Hfs). }
  destruct (List.fold_left step tcr ([], [], n0)) as [[acc seen] n] eqn:Hf.
  cbn [fst].
  pose proof (Hgen tcr [] [] n0 (fun c r H => match H with end)) as H.
  rewrite Hf in H. exact H.
Qed.

Theorem changed_current_renamed : changed_current_renamed_statement.
Proof.
  intros is_local F T dominated to_unbox tcr e n r Hlk.
  apply (keyed_lookup_in node_eqb TP.node_eqb_decides) in Hlk.
  apply is_bound_of_in_fst.
  exact (cinv_fold F T (bound_code_my_closures e) dominated to_unbox
           is_local tcr (fresh_name (slots_of e)) n r Hlk).
Qed.
