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

(* Proofs of the statements of FlambdaReaperRegionsCallsSpec.v.

   This file needs no review: what it proves is stated there. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaPrimitiveModel
  FlambdaModel FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec
  FlambdaSemanticsSpec FlambdaWellFormedSpec FlambdaNameAbstractionSpec
  FlambdaRenameApartSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperSolutionSpec FlambdaReaperFlowSpec
  FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaReaperLeastSpec FlambdaReaperLocalFieldsSpec FlambdaRegionsSpec
  FlambdaRegionsErasureSpec FlambdaRegionsCheckSpec FlambdaReaperRegionsSpec
  FlambdaReaperRegionsUnboxSpec FlambdaReaperRegionsFixedSpec
  FlambdaReaperRegionsCallsSpec FlambdaReaperRegionsRebuildProofs
  FlambdaReaperRegionsTraverseProofs FlambdaReaperRegionsCcRebuildProofs
  FlambdaReaperRegionsUnboxRebuildProofs
  FlambdaReaperRegionsUnboxTraverseProofs
  FlambdaReaperRegionsCallsRebuildProofs.
From OxCamlFlambda2 Require FlambdaReaperSolutionProofs
  FlambdaReaperLeastProofs FlambdaReaperRegionsProofs
  FlambdaReaperRegionsCcProofs FlambdaReaperRegionsUnboxProofs
  FlambdaReaperRegionsFixedProofs FlambdaReaperUnboxingRefinesProofs
  FlambdaReaperUnboxingRefinesCallsProofs.
Import ListNotations.

Module RUP := FlambdaReaperRegionsUnboxProofs.
Module RCP := FlambdaReaperUnboxingRefinesCallsProofs.
Module URP := FlambdaReaperUnboxingRefinesProofs.

(* ======================================================================== *)
(* The fragment read off the unboxed fields                                 *)
(* ======================================================================== *)

Lemma leaves_of_plain (F : facts) (T : tables) (cm : finite_map variable)
    (dominated : list (node * node)) (to_unbox : list node) (n0 : positive)
    (ub : list (node * unboxed_fields)) (next : positive) :
  graph_reads_plain F = true ->
  unboxed_fields_of_names F T cm dominated to_unbox n0 = (ub, next) ->
  ∀ (n : node) (fields : unboxed_fields),
    (n, fields) ∈ ub ->
    forallb kind_not_region (map snd (fields_leaves fields)) = true.
Proof.
  intros Hrp Hub n fields Hin.
  apply (unboxed_fields_kinds F T cm dominated to_unbox) with (next := n0)
    (x := n).
  - intros w f b' Hacc. unfold graph_reads_plain in Hrp.
    rewrite forallb_forall in Hrp. exact (Hrp _ Hacc).
  - rewrite Hub. exact Hin.
Qed.

Lemma sets_kept_of (F : facts) (T : tables) (cm : finite_map variable)
    (dominated : list (node * node)) (to_unbox : list node) (n0 : positive)
    (ub : list (node * unboxed_fields)) (next : positive) (e : expr) :
  unboxed_fields_of_names F T cm dominated to_unbox n0 = (ub, next) ->
  sets_not_unboxed dominated to_unbox e = true ->
  begins_kept ub (set_binders e).
Proof.
  intros Hub Hnu r Hr. unfold sets_not_unboxed in Hnu.
  rewrite forallb_forall in Hnu. specialize (Hnu r Hr).
  apply negb_true_iff in Hnu.
  destruct (unboxed_of_node ub (Node_variable r)) as [fields | ] eqn:Hu;
    [ exfalso | reflexivity ].
  apply (keyed_lookup_in node_eqb URP.node_eqb_decides) in Hu.
  assert (Hin : (Node_variable r, fields)
                ∈ fst (unboxed_fields_of_names F T cm dominated to_unbox n0))
    by (rewrite Hub; exact Hu).
  destruct (URP.unboxed_fields_of_names_dom F T cm dominated to_unbox n0 _ _
              Hin) as (a & _ & Hdom & Ha & _).
  unfold has_to_be_unboxed in Hnu. apply not_true_iff_false in Hnu.
  apply Hnu. apply existsb_exists. exists (a, Node_variable r).
  split; [ exact Hdom | ].
  rewrite FlambdaReaperSolutionProofs.node_eqb_refl, Ha. reflexivity.
Qed.

(* ======================================================================== *)
(* The commutation                                                          *)
(* ======================================================================== *)

(* The commutation, in either mode, for a solution of the erased unit
   that differs at the usages of the constants at most. *)
Lemma rebuild_unbox_calls_erase_commutes_with (fx : bool)
    (flag : reaper_preserve_direct_calls) (u : unit_interface) (e : expr)
    (T : tables) (sol' : solution) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node) (b : bool)
    (ub : list (node * unboxed_fields)) (next : positive) :
  regions_calls e = true ->
  regions_not_unboxed dominated to_unbox e = true ->
  sets_not_unboxed dominated to_unbox e = true ->
  graph_reads_plain (traverse_unit flag u e) = true ->
  (∀ n : node, n <> Node_all_constants ->
     s_has_usage sol' n = s_has_usage (solution_of_tables T) n) ->
  s_has_source sol' = s_has_source (solution_of_tables T) ->
  s_any_source sol' = s_any_source (solution_of_tables T) ->
  s_sources sol' = s_sources (solution_of_tables T) ->
  s_field_used sol' = s_field_used (solution_of_tables T) ->
  unboxed_fields_of_names (traverse_unit flag u e) T
    (bound_code_my_closures e) dominated to_unbox (fresh_name (names_of e))
  = (ub, next) ->
  erase []
    (rebuild_unit_unbox_repr_mode fx (fun _ => false) (traverse_unit flag u e)
       T cannot_change (unit_code_deps flag u e) dominated to_unbox [] b e)
  = rebuild_expr_u fx (traverse_unit flag u (erase [] e)) sol'
      (fun n => mem_node n cannot_change) (unit_code_deps flag u (erase [] e))
      b (static_names code_ids_bound (erase [] e))
      (bound_code_metadata (erase [] e)) ub next [] None empty_unbox_env
      (erase [] e).
Proof.
  intros Hcalls Hnu Hsnu Hrp Hu Hsrc Has Hss Hfu Hub.
  pose proof Hcalls as Hc'. unfold regions_calls in Hc'.
  apply andb_true_iff in Hc' as [Hpl Hsa].
  destruct (traverse_unit_rel flag u e Hpl) as (l & l' & HF & HF' & Hrel).
  destruct (traverse_unit_positions flag u e) as (l0 & HF0 & Hpos0).
  destruct Hrel as [(HC & HA & HP & _) _].
  rewrite (proj1 (static_names_erase code_ids_bound
                    ltac:(intros [ | | ]; reflexivity)) e []).
  rewrite (proj1 bound_code_metadata_erase e []).
  rewrite (FlambdaReaperRegionsCcProofs.unit_code_deps_erase flag u e Hpl).
  unfold rebuild_unit_unbox_repr_mode, unit_unboxed_fields.
  rewrite Hub.
  change (unit_changed_representations (fun _ => false)
            (traverse_unit flag u e) T dominated to_unbox [] e)
    with (@nil (node * closure_repr)).
  assert (Hpac : ∀ (x : node) (i : nat) (y : node),
            (x, i, y) ∈ f_parameter (traverse_unit flag u e) ->
            y <> Node_all_constants)
    by (rewrite HF0; apply FlambdaReaperRegionsProofs.parameters_not_consts,
          Hpos0).
  assert (HC' : f_constructor (traverse_unit flag u (erase [] e))
                = f_constructor (traverse_unit flag u e))
    by (rewrite HF, HF'; exact HC).
  assert (HA' : f_accessor (traverse_unit flag u (erase [] e))
                = f_accessor (traverse_unit flag u e))
    by (rewrite HF, HF'; exact HA).
  assert (HP' : f_parameter (traverse_unit flag u (erase [] e))
                = f_parameter (traverse_unit flag u e))
    by (rewrite HF, HF'; exact HP).
  exact (proj1 (rebuild_u_calls_erase_commutes fx (traverse_unit flag u e)
                  (traverse_unit flag u (erase [] e)) (solution_of_tables T)
                  sol' (fun n => mem_node n cannot_change)
                  (unit_code_deps flag u e) b (static_names code_ids_bound e)
                  (bound_code_metadata e) ub next HC' HA' HP' Hpac Hu Hsrc Has
                  Hss Hfu
                  (fun id cd Hl =>
                     FlambdaReaperRegionsCcProofs.unit_code_deps_ok flag u e id
                       cd (lookup_in _ _ _ Hl))
                  (leaves_of_plain _ T _ dominated to_unbox _ ub next Hrp
                     Hub))
           e empty_unbox_env empty_unbox_env Hpl Hsa
           (RUP.begins_kept_of _ _ _ _ _ _ _ _ e Hub Hnu)
           (sets_kept_of _ _ _ _ _ _ _ _ e Hub Hsnu)
           (uenv_rel_refl _) (conj eq_refl eq_refl)).
Qed.

Theorem rebuild_unbox_fixed_calls_erase_commutes :
  rebuild_unbox_fixed_calls_erase_commutes_statement.
Proof.
  intros flag u e T cannot_change D U b Hcalls Hnu Hsnu Hrp.
  pose proof Hcalls as Hc'. unfold regions_calls in Hc'.
  apply andb_true_iff in Hc' as [Hpl _].
  rewrite (RUP.unboxed_fields_erase flag u e T D U _ Hpl).
  destruct (unboxed_fields_of_names (traverse_unit flag u e) T
              (bound_code_my_closures e) D U (fresh_name (names_of e)))
    as [ub next] eqn:Hub.
  exact (rebuild_unbox_calls_erase_commutes_with true flag u e T
           (solution_of_tables T) cannot_change D U b ub next Hcalls Hnu
           Hsnu Hrp (fun _ _ => eq_refl) eq_refl eq_refl eq_refl eq_refl Hub).
Qed.

(* ======================================================================== *)
(* The refinement on the machine of regions                                 *)
(* ======================================================================== *)

Theorem rebuild_unbox_fixed_calls_regions_refines :
  rebuild_unbox_fixed_calls_regions_refines_statement.
Proof.
  intros flag is_local u e T cannot_change D U b cert base Hcalls Hnu Hsnu
    Hrp Hnames Hslots Hfrag Hdc Hrk Hwf Hexits Hcl Hlrs Hexact Hcc Hwid
    Hcu Hreads Hfe Hee Hfr Her Hcheck.
  pose proof Hcalls as Hc'. unfold regions_calls in Hc'.
  apply andb_true_iff in Hc' as [Hpl _].
  apply (FlambdaReaperRegionsProofs.regions_refine_of_erased cert base e);
    try assumption.
  unfold erased_initial_state.
  rewrite FlambdaReaperRegionsProofs.erase_statics_empty.
  destruct (unboxed_fields_of_names (traverse_unit flag u e) T
              (bound_code_my_closures e) D U (fresh_name (names_of e)))
    as [ub next] eqn:Hub.
  unfold rebuild_unit_unbox_fixed, rebuild_unit_unbox_repr_fixed.
  rewrite (rebuild_unbox_calls_erase_commutes_with true flag u e T
             (solution_of_tables T) cannot_change D U b ub next Hcalls Hnu
             Hsnu Hrp (fun _ _ => eq_refl) eq_refl eq_refl eq_refl eq_refl
             Hub).
  pose proof (RCP.rebuild_unbox_fixed_refines_observably_local flag
                is_local u (erase [] e) (alias_consts T) cannot_change D U b
                Hslots Hfrag Hdc Hrk Hwf Hexits Hcl Hlrs Hexact Hcc Hwid
                Hcu Hreads)
    as H.
  unfold program_refines_observably in H.
  unfold rebuild_unit_unbox_fixed, rebuild_unit_unbox_repr_fixed in H.
  rewrite (FlambdaReaperRegionsFixedProofs.alias_consts_rebuild true
             (fun _ => false)) in H.
  unfold rebuild_unit_unbox_repr_mode, unit_unboxed_fields in H.
  rewrite Hnames, (RUP.unboxed_fields_erase flag u e T D U _ Hpl), Hub
    in H.
  exact H.
Qed.

(* ======================================================================== *)
(* Example                                                                  *)
(* ======================================================================== *)

Import FlambdaReaperRegionsSpec.RegionsExamples RegionsCallsExamples.

Ltac ex_run :=
  eexists; vm_compute; reflexivity.

Theorem regions_calls_example : regions_calls_example_statement.
Proof.
  assert (Hleast : least_solution (fun _ => false)
                     (traverse_unit Auto interface call_in_region)
                     call_in_region_tables)
    by (apply (FlambdaReaperLeastProofs.check_least_sound _ _ 100);
        vm_compute; reflexivity).
  assert (Hreb : rebuild_unit_unbox_fixed
                   (traverse_unit Auto interface call_in_region)
                   call_in_region_tables
                   (derive_cannot_change
                      (traverse_unit Auto interface call_in_region)
                      call_in_region_tables)
                   (unit_code_deps Auto interface call_in_region)
                   (derive_dominated call_in_region_tables)
                   (least_to_unbox (fun _ => false)
                      (traverse_unit Auto interface call_in_region)
                      call_in_region_tables 10 false)
                   true call_in_region
                 = call_in_region_rebuilt)
    by (vm_compute; reflexivity).
  split; [ exact Hleast | ].
  do 2 (split; [ vm_compute; reflexivity | ]).
  split; [ exact Hreb | ].
  split; [ vm_compute; reflexivity | ].
  split; [ ex_run | ]. split; [ ex_run | ].
  rewrite <- Hreb.
  apply (rebuild_unbox_fixed_calls_regions_refines Auto (fun _ => false)
           interface call_in_region call_in_region_tables _ _ _ true
           call_in_region_cert base);
    try (vm_compute; reflexivity);
    try (rewrite Hreb; vm_compute; reflexivity).
  exact FlambdaReaperRegionsProofs.no_local_slots.
Qed.

(* ======================================================================== *)
(* The completion of the tables                                             *)
(* ======================================================================== *)

Theorem alias_consts_needs_consts : alias_consts_needs_consts_statement.
Proof.
  intros is_local F T x Hc Has Hau.
  pose proof (FlambdaReaperSolutionProofs.check_closed_sound _ _ _ Hc)
    as Hcl.
  assert (Ha : s_alias (solution_of_tables (alias_consts T))
                 (Node_variable x) Node_all_constants = true).
  { cbn [solution_of_tables s_alias alias_consts t_alias].
    apply FlambdaReaperSolutionProofs.mem_pair_In. apply in_or_app. right.
    unfold consts_aliases. apply in_flat_map. exists (Node_variable x).
    split; [ apply FlambdaReaperSolutionProofs.mem_node_In; exact Has
           | left; reflexivity ]. }
  exact (any_usage_alias _ _ _ Hcl (Node_variable x) Node_all_constants Hau
           Ha).
Qed.

Theorem alias_consts_example : alias_consts_example_statement.
Proof.
  split; [ | split; vm_compute; reflexivity ].
  apply (FlambdaReaperLeastProofs.check_least_sound _ _ 100).
  vm_compute. reflexivity.
Qed.
