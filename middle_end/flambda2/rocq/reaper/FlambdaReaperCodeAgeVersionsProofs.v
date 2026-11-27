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

(* Proofs of the statements of FlambdaReaperCodeAgeVersionsSpec.v.
   Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaWellFormedSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperFlowSpec FlambdaReaperRebuild
  FlambdaReaperRebuildSpec FlambdaReaperCallingConventions
  FlambdaReaperCallingConventionsSpec FlambdaReaperUnboxing
  FlambdaReaperLeastSpec FlambdaCodeAgeCallsSpec FlambdaCodeAgeCallsProofs
  FlambdaReaperCodeAgeSpec FlambdaReaperCodeAgeProofs
  FlambdaReaperCodeAgeVersionsSpec.
From OxCamlFlambda2 Require FlambdaReaperLeastProofs
  FlambdaReaperRebuildProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaCodeAgeRelation FlambdaCodeAgeRelationSpec.
Import ListNotations.

(* ======================================================================== *)
(* The conventions                                                          *)
(* ======================================================================== *)

Section RebuiltU.

Variable fixed : bool.
Variable F : facts.
Variable sol : solution.
Variable cannot_change : node -> bool.
Variable code_deps : finite_map code_dep.
Variable preserve_direct_calls : bool.
Variable unit_code_ids : list code_id.
Variable codes : finite_map code_metadata.
Variable unboxed : list (node * unboxed_fields).
Variable first_fresh : positive.
Variable changed : list (node * closure_repr).
Variable types : subkinds_rewriting.

Abbreviation rebuild_e :=
  (rebuild_expr_u fixed F sol cannot_change code_deps preserve_direct_calls
     unit_code_ids codes unboxed first_fresh changed types).
Abbreviation rebuild_b :=
  (rebuild_binding_u fixed F sol cannot_change code_deps
     preserve_direct_calls unit_code_ids codes unboxed first_fresh changed
     types).
Abbreviation rebuild_c :=
  (rebuild_code_u fixed F sol cannot_change code_deps preserve_direct_calls
     unit_code_ids codes unboxed first_fresh changed types).

(* Code whose code id is in the table keeps its convention. *)
Lemma rebuild_code_u_unchanged (id : code_id) (c : code) :
  cannot_change (Node_code_id id) = true ->
  same_convention c (rebuild_c id c).
Proof.
  intros H. destruct c as [params my_closure ret exn metadata body].
  cbn [rebuild_code_u]. cbv zeta.
  replace (code_change_u sol cannot_change code_deps unit_code_ids unboxed
             id metadata)
    with Not_changing_u.
  2: { unfold code_change_u, cannot_change_calling_convention.
       rewrite H, orb_true_r. reflexivity. }
  unfold same_convention, kept_code_metadata.
  destruct types as [t | ];
    [ destruct (lookup id (st_code_deps t)) | ]; simpl;
    (split; [ reflexivity | split; reflexivity ]).
Qed.

(* A code that a rebuilt binding binds is the rebuild of the code that the
   binding binds, by the same code id. *)
Lemma rebuild_binding_u_code (en : unbox_env) (b : static_binding)
    (id : code_id) (c : code) :
  Bind_code id c ∈ rebuild_b en b ->
  ∃ c0 : code, b = Bind_code id c0 ∧ c = rebuild_c id c0.
Proof.
  destruct b as [id' c0 | id' | s const]; cbn [rebuild_binding_u].
  - destruct (is_code_id_used sol id'); intros [H | []]; [ | discriminate H ].
    injection H as <- <-. exists c0. split; reflexivity.
  - intros [H | []]. discriminate H.
  - destruct (static_set_repr changed s const);
      [ intros [H | []]; discriminate H | ].
    destruct (static_const_is_used sol s const);
      [ intros [H | []]; discriminate H | intros [] ].
Qed.

(* The code of the statics of the first step of the rebuild of a unit that
   begins with its [Let_static]. *)
Lemma unit_statics_rebuild_u (en : unbox_env)
    (bindings : list static_binding) (body : expr) (id : code_id)
    (c : code) :
  lookup id
    (static_code (unit_statics (rebuild_e en (Let_static bindings body))))
  = Some c ->
  ∃ c0 : code, Bind_code id c0 ∈ bindings ∧ c = rebuild_c id c0.
Proof.
  cbn [rebuild_expr_u]. unfold unit_statics.
  match goal with
  | |- context [eval_static_bindings ?vars ?bs ?st] =>
    destruct (eval_static_bindings vars bs st) as [statics | ] eqn:He
  end;
    [ | intros H; cbn in H; discriminate H ].
  intros H.
  destruct (eval_static_bindings_codes _ _ _ _ He id c H) as [Hc | Hin];
    [ cbn in Hc; discriminate Hc | ].
  apply in_flat_map in Hin as (b & Hb & Hin).
  destruct (rebuild_binding_u_code _ _ _ _ Hin) as (c0 & -> & ->).
  exists c0. split; [ exact Hb | reflexivity ].
Qed.

End RebuiltU.

Theorem rebuild_unbox_sound_conventions :
  rebuild_unbox_sound_conventions_statement.
Proof.
  intros types fixed is_local F T cannot_change code_deps dominated to_unbox
    tcr preserve_direct_calls bindings body rel age_of v.
  cbv zeta. unfold rebuild_unit_unbox_repr_mode_with.
  destruct (unit_unboxed_fields F T dominated to_unbox
              (Let_static bindings body)) as [unboxed next].
  cbn beta iota. intros Hagree Hage newer older c c' _ Hl Hc Hc'.
  rewrite Hage in Hl.
  apply unit_statics_rebuild_u in Hc as (c0 & Hc0 & ->).
  apply unit_statics_rebuild_u in Hc' as (c0' & Hc0' & ->).
  assert (Hin : ∀ (id : code_id) (c1 : code),
            Bind_code id c1 ∈ bindings ->
            (id, c1) ∈ unit_codes (Let_static bindings body)).
  { intros id c1 H. cbn [unit_codes]. apply in_or_app. left.
    apply in_flat_map. exists (Bind_code id c1).
    split; [ exact H | left; reflexivity ]. }
  destruct (Hagree newer older c0 c0' Hl (Hin _ _ Hc0) (Hin _ _ Hc0'))
    as (Hp & Hr & Ht).
  destruct (age_fixed_table_links rel cannot_change newer older Hl)
    as [Hn Ho].
  destruct (rebuild_code_u_unchanged fixed F (solution_of_tables T)
              (fun n => mem_node n (age_fixed_table rel cannot_change))
              code_deps preserve_direct_calls
              (static_names code_ids_bound (Let_static bindings body))
              (bound_code_metadata (Let_static bindings body)) unboxed next
              (unit_changed_representations is_local F T dominated to_unbox
                 tcr (Let_static bindings body))
              types older c0 Ho)
    as (Hp1 & Hr1 & Ht1).
  destruct (rebuild_code_u_unchanged fixed F (solution_of_tables T)
              (fun n => mem_node n (age_fixed_table rel cannot_change))
              code_deps preserve_direct_calls
              (static_names code_ids_bound (Let_static bindings body))
              (bound_code_metadata (Let_static bindings body)) unboxed next
              (unit_changed_representations is_local F T dominated to_unbox
                 tcr (Let_static bindings body))
              types newer c0' Hn)
    as (Hp2 & Hr2 & Ht2).
  unfold same_convention. rewrite Hp1, Hr1, Ht1, Hp2, Hr2, Ht2.
  split; [ exact Hp | split; [ exact Hr | exact Ht ] ].
Qed.

(* ======================================================================== *)
(* A change of representation of the set of the older version               *)
(* ======================================================================== *)

(* Computes the results of the steps of the goal, once each: a state or
   [Undefined]. *)
Ltac compute_steps :=
  match goal with
  | |- result_refines_by ?m (?m ?s) (?m ?t) =>
    let r := fresh "r" in
    let r' := fresh "r" in
    let Hr := fresh "Hr" in
    let Hr' := fresh "Hr" in
    remember (m s) as r eqn:Hr; remember (m t) as r' eqn:Hr';
    vm_compute in Hr, Hr'; subst r r'
  end.

(* In the unit, with [s90], [f55] refines [f53]: both read 7 and return
   it. *)
Lemma repr_codes_refine :
  code_refines_for ReprExample.age_of
    (unit_statics ReprExample.unit_expr) (Value_symbol 90%positive)
    ReprExample.older ReprExample.newer.
Proof.
  apply code_refines_for_steps; try reflexivity.
  - split; [ reflexivity | split; reflexivity ].
  - intros vs vars return_to exn_to hp rd Hb.
    destruct vs as [ | a [ | ? ? ] ]; try discriminate Hb.
    injection Hb as <-. eexists. split; [ reflexivity | ].
    unfold result_refines_aged. compute_steps.
    cbn [result_refines_by]. apply result_refines_by_step.
    match goal with
    | |- result_refines_by ?m (?m ?s) (?m ?t) =>
      replace (m t) with (m s) by (cbn; reflexivity)
    end.
    apply result_refines_by_refl.
Qed.

(* In the rebuilt unit, [f53] with [s90] is undefined. *)
Lemma repr_codes_refine_rebuilt :
  code_refines_for ReprExample.age_of
    (unit_statics ReprExample.rebuilt) (Value_symbol 90%positive)
    ReprExample.older_rebuilt ReprExample.newer.
Proof.
  apply code_refines_for_steps; try reflexivity.
  - split; [ reflexivity | split; reflexivity ].
  - intros vs vars return_to exn_to hp rd Hb.
    destruct vs as [ | a [ | ? ? ] ]; try discriminate Hb.
    injection Hb as <-. eexists. split; [ reflexivity | ].
    unfold result_refines_aged. compute_steps. exact I.
Qed.

(* The facts of the unit, each computed once. *)
Lemma repr_least :
  least_solution Examples.no_local ReprExample.facts ReprExample.tables.
Proof.
  apply (FlambdaReaperLeastProofs.compute_least_sound _ _ 40).
  vm_compute. reflexivity.
Qed.

Lemma repr_runs :
  run_aged ReprExample.age_of 20 (ReprExample.initial ReprExample.unit_expr)
  = Exit 1%positive
      [Value_block 0
         [Value_const (Tagged_immediate 7); Value_const (Tagged_immediate 7);
          Value_symbol 90%positive]]
      empty_heap no_reads
  ∧ run_aged ReprExample.age_of 20 (ReprExample.initial ReprExample.rebuilt)
    = Exit 1%positive
        [Value_block 0
           [Value_const (Tagged_immediate 7);
            Value_const (Tagged_immediate 7); Value_symbol 90%positive]]
        empty_heap no_reads.
Proof. split; vm_compute; reflexivity. Qed.

Lemma repr_client_runs :
  run_aged ReprExample.age_of 3
    (ReprExample.client (unit_statics ReprExample.unit_expr) 53%positive)
  = Exit 1%positive [Value_const (Tagged_immediate 7)] empty_heap no_reads
  ∧ run_aged ReprExample.age_of 2
      (ReprExample.client (unit_statics ReprExample.rebuilt) 53%positive)
    = Undefined.
Proof. split; vm_compute; reflexivity. Qed.

Lemma repr_sound_before :
  sound_for ReprExample.age_of (unit_statics ReprExample.unit_expr)
    (Value_symbol 90%positive).
Proof.
  intros newer older c c' _ Hl Hc Hc'.
  apply one_link in Hl as [-> ->].
  vm_compute in Hc, Hc'. injection Hc as <-. injection Hc' as <-.
  exact repr_codes_refine.
Qed.

Lemma repr_sound_after :
  sound_for ReprExample.age_of (unit_statics ReprExample.rebuilt)
    (Value_symbol 90%positive).
Proof.
  intros newer older c c' _ Hl Hc Hc'.
  apply one_link in Hl as [-> ->].
  vm_compute in Hc, Hc'. injection Hc as <-. injection Hc' as <-.
  exact repr_codes_refine_rebuilt.
Qed.

(* The runs of the client stated on the machine itself ([run_by]), so
   that the kernel does not run them again to compare [run_aged] with
   [run_by]. *)
Lemma repr_client_not_refined :
  ~ refines_aged ReprExample.age_of
      (ReprExample.client (unit_statics ReprExample.unit_expr) 53%positive)
      (ReprExample.client (unit_statics ReprExample.rebuilt) 53%positive).
Proof.
  apply (not_refines_by_exit_undefined (step_aged ReprExample.age_of) 3 2
           _ _ 1%positive [Value_const (Tagged_immediate 7)] empty_heap
           no_reads);
    vm_compute; reflexivity.
Qed.

Theorem repr_client_counterexample :
  repr_client_counterexample_statement.
Proof.
  unfold repr_client_counterexample_statement.
  destruct repr_runs as [H0 H1].
  destruct repr_client_runs as [H2 H3].
  split; [ exact repr_least | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ exists 10; vm_compute; reflexivity | ].
  split; [ exact H0 | ].
  split; [ exact H1 | ].
  split; [ exact (refines_observably_aged_of_runs _ 20 20 _ _ _ _ _ _ _ _
                    H0 H1
                    (FlambdaReaperRebuildProofs.values_observed_refl _)) | ].
  split; [ apply conventions_agree_of_b; vm_compute; reflexivity | ].
  split; [ apply conventions_agree_of_b; vm_compute; reflexivity | ].
  split; [ exact repr_sound_before | ].
  split; [ exact repr_sound_after | ].
  split; [ vm_compute; reflexivity | ].
  split; [ exact H2 | ].
  split; [ exact H3 | ].
  split; [ exact repr_client_not_refined | ].
  split; vm_compute; reflexivity.
Qed.
