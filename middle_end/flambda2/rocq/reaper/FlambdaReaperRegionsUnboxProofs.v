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

(* Proofs of the statements of FlambdaReaperRegionsUnboxSpec.v.

   This file needs no review: what it proves is stated there. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaPrimitiveModel
  FlambdaModel FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec
  FlambdaSemanticsSpec FlambdaWellFormedSpec FlambdaNameAbstractionSpec
  FlambdaRenameApartSpec FlambdaUnusedStaticsSpec FlambdaReaperGraphSpec
  FlambdaReaperTraversal FlambdaReaperSolution FlambdaReaperSolutionSpec
  FlambdaReaperFlowSpec FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperLeastSpec
  FlambdaReaperLocalFieldsSpec FlambdaRegionsSpec
  FlambdaRegionsErasureSpec FlambdaRegionsCheckSpec FlambdaReaperRegionsSpec
  FlambdaReaperRegionsUnboxSpec FlambdaReaperRegionsRebuildProofs
  FlambdaReaperRegionsTraverseProofs FlambdaReaperRegionsUnboxBaseProofs
  FlambdaReaperRegionsUnboxRebuildProofs
  FlambdaReaperRegionsUnboxTraverseProofs.
From OxCamlFlambda2 Require FlambdaReaperSolutionProofs
  FlambdaReaperLeastProofs FlambdaReaperFlowProofs
  FlambdaReaperUnboxingRefinesProofs FlambdaReaperRegionsProofs
  FlambdaReaperRegionsCcProofs FlambdaReaperRegionsUnboxKeepsProofs
  FlambdaReaperRebuildProofs.
Import ListNotations.

Module URP := FlambdaReaperUnboxingRefinesProofs.

(* ======================================================================== *)
(* The tables of the erased unit                                            *)
(* ======================================================================== *)

(* The facts that the unboxed fields read: the accessors and the
   constructors. *)
Definition reads_of (F : facts) : facts :=
  Facts [] [] (f_accessor F) (f_constructor F) [] [] [] [] [] [] [] [].

Lemma unboxed_fields_of_names_reads (F : facts) (T : tables)
    (cm : finite_map variable) (dominated : list (node * node))
    (to_unbox : list node) (next : positive) :
  unboxed_fields_of_names F T cm dominated to_unbox next
  = unboxed_fields_of_names (reads_of F) T cm dominated to_unbox next.
Proof. reflexivity. Qed.

(* The facts that the rules of the unboxing read: and the parameters. *)
Definition unboxing_reads_of (F : facts) : facts :=
  Facts [] [] (f_accessor F) (f_constructor F) [] (f_parameter F) [] [] []
    [] [] [].

Lemma derive_to_unbox_reads (is_local : field -> bool) (F : facts)
    (T : tables) (cannot_change : list node) (max_unbox_size : nat)
    (lambda_lifting : bool) :
  derive_to_unbox is_local F T cannot_change max_unbox_size lambda_lifting
  = derive_to_unbox is_local (unboxing_reads_of F) T cannot_change
      max_unbox_size lambda_lifting.
Proof. reflexivity. Qed.

Lemma bound_code_my_closures_erase :
  (∀ e : expr, ∀ c,
     bound_code_my_closures (erase c e) = bound_code_my_closures e)
  ∧ (∀ h : continuation_handler, ∀ c,
       handler_bound_code_my_closures (erase_handler c h)
       = handler_bound_code_my_closures h).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH c.
    destruct d as [s | p args | | | | | ]; cbn [erase];
      try (cbn [bound_code_my_closures]; apply IH).
    destruct p as [ | | | | r]; try (cbn [bound_code_my_closures]; apply IH).
    destruct (erase_region_primitive x r args);
      cbn [bound_code_my_closures]; apply IH.
  - intros bindings body _ IH c. cbn [erase bound_code_my_closures].
    rewrite IH. f_equal. induction bindings as [ | b bs IHb];
      [ reflexivity | ].
    cbn [map flat_map]. rewrite IHb. f_equal.
    destruct b as [id cd | | ]; [ | reflexivity | reflexivity ].
    destruct cd. reflexivity.
  - intros k h body IHh IH c. cbn [erase bound_code_my_closures].
    rewrite IHh, IH. reflexivity.
  - intros handlers body IHh IH c. cbn [erase bound_code_my_closures].
    rewrite IH. f_equal. rewrite flat_map_concat_map, map_map,
      flat_map_concat_map. f_equal. apply map_ext_in.
    intros [k h] Hin. cbn [fst snd]. rewrite (IHh k h Hin _). reflexivity.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - intros params md hbody IH c. cbn [erase_handler
      handler_bound_code_my_closures]. apply IH.
Qed.

Theorem unboxed_fields_erase : unboxed_fields_erase_statement.
Proof.
  intros flag u e T dominated to_unbox next Hpl.
  destruct (traverse_unit_rel flag u e Hpl) as (l & l' & HF & HF' & Hrel).
  destruct Hrel as [(HC & HA & _ & _) _].
  rewrite (proj1 bound_code_my_closures_erase).
  rewrite (unboxed_fields_of_names_reads (traverse_unit flag u (erase [] e))),
    (unboxed_fields_of_names_reads (traverse_unit flag u e)).
  unfold reads_of. rewrite HF, HF', HC, HA. reflexivity.
Qed.

Theorem to_unbox_erase : to_unbox_erase_statement.
Proof.
  intros flag u e is_local T cannot_change max_unbox_size lambda_lifting Hpl.
  assert (Hd : ∀ cc : list node,
             derive_to_unbox is_local (traverse_unit flag u (erase [] e)) T cc
               max_unbox_size lambda_lifting
             = derive_to_unbox is_local (traverse_unit flag u e) T cc
                 max_unbox_size lambda_lifting).
  { intros cc.
    destruct (traverse_unit_rel flag u e Hpl) as (l & l' & HF & HF' & Hrel).
    destruct Hrel as [(HC & HA & HP & _) _].
    rewrite (derive_to_unbox_reads _ (traverse_unit flag u (erase [] e))),
      (derive_to_unbox_reads _ (traverse_unit flag u e)).
    unfold unboxing_reads_of. rewrite HF, HF', HC, HA, HP. reflexivity. }
  split; [ apply Hd | ]. unfold least_to_unbox.
  rewrite (FlambdaReaperRegionsCcProofs.derive_cannot_change_erase flag u e T
             Hpl).
  apply Hd.
Qed.

Theorem local_unboxing_names : local_unboxing_names_statement.
Proof.
  intros F T dominated to_unbox e next.
  unfold local_unboxing, unit_unboxed_fields. apply local_expr_names.
Qed.

(* ======================================================================== *)
(* The variables of the regions are not unboxed                             *)
(* ======================================================================== *)

Lemma not_dominated_any_source (T : tables) (to_unbox : list node)
    (x : node) :
  mem_node x (t_any_source T) = true ->
  has_to_be_unboxed (derive_dominated T) to_unbox x = false.
Proof.
  intros Hx. unfold has_to_be_unboxed.
  apply not_true_iff_false. intros Hh.
  apply existsb_exists in Hh as ([a x'] & Hin & Htest).
  apply andb_true_iff in Htest as [Hxx _].
  apply FlambdaReaperSolutionProofs.node_eqb_eq in Hxx. subst x'.
  unfold derive_dominated in Hin. apply in_flat_map in Hin
    as ([x0 y] & _ & Hin).
  destruct (mem_node y (t_has_source T)
            && negb (multiple_allocation_points T x0)) eqn:Ht;
    [ | destruct Hin ].
  destruct Hin as [Heq | []]. injection Heq as -> ->.
  apply andb_true_iff in Ht as [_ Hm]. apply negb_true_iff in Hm.
  unfold multiple_allocation_points in Hm. rewrite Hx in Hm.
  discriminate Hm.
Qed.

Lemma regions_not_unboxed_any_source (T : tables) (to_unbox : list node)
    (e : expr) :
  (∀ r, r ∈ region_binders e ->
     mem_node (Node_variable r) (t_any_source T) = true) ->
  regions_not_unboxed (derive_dominated T) to_unbox e = true.
Proof.
  intros H. unfold regions_not_unboxed. apply forallb_forall.
  intros r Hr. apply negb_true_iff. apply not_dominated_any_source, H, Hr.
Qed.

Lemma closed_regions_any_source (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables) :
  regions_local e = true ->
  in_flow_fragment (erase [] e) = true ->
  closed is_local (graph_of_facts (traverse_unit flag u e))
    (solution_of_tables T) ->
  ∀ r, r ∈ region_binders e ->
    mem_node (Node_variable r) (t_any_source T) = true.
Proof.
  intros Hloc Hfrag Hcl r Hr. unfold regions_local in Hloc.
  apply andb_true_iff in Hloc as [_ Hcf].
  pose proof (in_flow_fragment_of_erase e Hcf Hfrag) as Hfrag'.
  destruct (FlambdaReaperFlowProofs.traverse_unit_facts flag u e True
              Hfrag' (or_introl I))
    as (noted & deps & cds & Hedges & _).
  pose proof (proj1 (begins_any_source _ _ _ _ _ _ _ _ _) e _ _ Hcf Hedges r
                Hr) as Hs.
  exact (any_source_input _ _ _ Hcl _ Hs).
Qed.

Theorem regions_not_unboxed_closed : regions_not_unboxed_closed_statement.
Proof.
  intros flag is_local u e T to_unbox Hloc Hfrag Hclosed.
  apply regions_not_unboxed_any_source.
  apply (closed_regions_any_source flag is_local u e T Hloc Hfrag).
  exact (FlambdaReaperSolutionProofs.check_closed_sound is_local _ T Hclosed).
Qed.

(* ======================================================================== *)
(* The commutation                                                          *)
(* ======================================================================== *)

Lemma begins_kept_of (F : facts) (T : tables) (cm : finite_map variable)
    (dominated : list (node * node)) (to_unbox : list node) (n0 : positive)
    (ub : list (node * unboxed_fields)) (next : positive) (e : expr) :
  unboxed_fields_of_names F T cm dominated to_unbox n0 = (ub, next) ->
  regions_not_unboxed dominated to_unbox e = true ->
  begins_kept ub (region_binders e).
Proof.
  intros Hub Hnu r Hr. unfold regions_not_unboxed in Hnu.
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

Lemma leaves_of (flag : reaper_preserve_direct_calls) (u : unit_interface)
    (e : expr) (T : tables) (cm : finite_map variable)
    (dominated : list (node * node)) (to_unbox : list node) (n0 : positive)
    (ub : list (node * unboxed_fields)) (next : positive) :
  closure_free e = true ->
  unboxed_fields_of_names (traverse_unit flag u e) T cm dominated to_unbox n0
  = (ub, next) ->
  ∀ (n : node) (fields : unboxed_fields),
    (n, fields) ∈ ub ->
    forallb kind_not_region (map snd (fields_leaves fields)) = true.
Proof.
  intros Hcf Hub n fields Hin.
  apply (unboxed_fields_kinds (traverse_unit flag u e) T cm dominated to_unbox
           (closure_free_reads_plain flag u e Hcf) n0 n).
  rewrite Hub. exact Hin.
Qed.

(* The commutation, for a solution of the erased unit that differs at the
   usages of the constants at most. *)
Lemma rebuild_unbox_mode_erase_commutes_with (fx : bool)
    (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) (T : tables) (sol' : solution)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox : list node) (b : bool)
    (ub : list (node * unboxed_fields)) (next : positive) :
  regions_local e = true ->
  regions_not_unboxed dominated to_unbox e = true ->
  (∀ n : node, n <> Node_all_constants ->
     s_has_usage sol' n = s_has_usage (solution_of_tables T) n) ->
  s_has_source sol' = s_has_source (solution_of_tables T) ->
  s_field_used sol' = s_field_used (solution_of_tables T) ->
  unboxed_fields_of_names (traverse_unit flag u e) T
    (bound_code_my_closures e) dominated to_unbox (fresh_name (names_of e))
  = (ub, next) ->
  erase []
    (rebuild_unit_unbox_repr_mode fx (fun _ => false) (traverse_unit flag u e)
       T cannot_change code_deps dominated to_unbox [] b e)
  = rebuild_expr_u fx (traverse_unit flag u (erase [] e)) sol'
      (fun n => mem_node n cannot_change) code_deps b
      (static_names code_ids_bound (erase [] e))
      (bound_code_metadata (erase [] e)) ub next [] None empty_unbox_env
      (erase [] e).
Proof.
  intros Hloc Hnu Hu Hsrc Hfu Hub.
  pose proof Hloc as Hloc'. unfold regions_local in Hloc'.
  apply andb_true_iff in Hloc' as [Hpl Hcf].
  unfold rebuild_unit_unbox_repr_mode, unit_unboxed_fields.
  rewrite Hub.
  change (unit_changed_representations (fun _ => false)
            (traverse_unit flag u e) T dominated to_unbox [] e)
    with (@nil (node * closure_repr)).
  exact (proj1 (rebuild_u_erase_commutes fx (traverse_unit flag u e)
                  (traverse_unit flag u (erase [] e)) (solution_of_tables T)
                  sol' (fun n => mem_node n cannot_change) code_deps b
                  (static_names code_ids_bound e)
                  (static_names code_ids_bound (erase [] e))
                  (bound_code_metadata e) (bound_code_metadata (erase [] e))
                  ub next Hu Hsrc Hfu
                  (leaves_of flag u e T _ dominated to_unbox _ ub next Hcf
                     Hub))
           e empty_unbox_env empty_unbox_env Hpl Hcf
           (begins_kept_of _ _ _ _ _ _ _ _ e Hub Hnu)
           (uenv_rel_refl _) (conj eq_refl eq_refl)).
Qed.

Lemma rebuild_unbox_erase_commutes_with (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) (T : tables) (sol' : solution)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox : list node) (b : bool)
    (ub : list (node * unboxed_fields)) (next : positive) :
  regions_local e = true ->
  regions_not_unboxed dominated to_unbox e = true ->
  (∀ n : node, n <> Node_all_constants ->
     s_has_usage sol' n = s_has_usage (solution_of_tables T) n) ->
  s_has_source sol' = s_has_source (solution_of_tables T) ->
  s_field_used sol' = s_field_used (solution_of_tables T) ->
  unboxed_fields_of_names (traverse_unit flag u e) T
    (bound_code_my_closures e) dominated to_unbox (fresh_name (names_of e))
  = (ub, next) ->
  erase []
    (rebuild_unit_unbox (traverse_unit flag u e) T cannot_change code_deps
       dominated to_unbox b e)
  = rebuild_expr_u false (traverse_unit flag u (erase [] e)) sol'
      (fun n => mem_node n cannot_change) code_deps b
      (static_names code_ids_bound (erase [] e))
      (bound_code_metadata (erase [] e)) ub next [] None empty_unbox_env
      (erase [] e).
Proof.
  exact (rebuild_unbox_mode_erase_commutes_with false flag u e T sol'
           cannot_change code_deps dominated to_unbox b ub next).
Qed.

Theorem rebuild_unbox_erase_commutes :
  rebuild_unbox_erase_commutes_statement.
Proof.
  intros flag u e T cannot_change code_deps dominated to_unbox b Hloc Hnu.
  pose proof Hloc as Hloc'. unfold regions_local in Hloc'.
  apply andb_true_iff in Hloc' as [Hpl _].
  rewrite (unboxed_fields_erase flag u e T dominated to_unbox _ Hpl).
  destruct (unboxed_fields_of_names (traverse_unit flag u e) T
              (bound_code_my_closures e) dominated to_unbox
              (fresh_name (names_of e)))
    as [ub next] eqn:Hub.
  exact (rebuild_unbox_erase_commutes_with flag u e T (solution_of_tables T)
           cannot_change code_deps dominated to_unbox b ub next Hloc Hnu
           (fun _ _ => eq_refl) eq_refl eq_refl Hub).
Qed.

(* ======================================================================== *)
(* What the rebuild keeps                                                   *)
(* ======================================================================== *)

Theorem rebuild_unbox_keeps_erasable : rebuild_unbox_keeps_erasable_statement.
Proof.
  intros flag u e T cannot_change code_deps dominated to_unbox b Hloc He.
  exact (FlambdaReaperRegionsUnboxKeepsProofs.rebuild_unbox_erasable flag u e
           T cannot_change code_deps dominated to_unbox b Hloc He).
Qed.

Theorem rebuild_unbox_keeps_fresh : rebuild_unbox_keeps_fresh_statement.
Proof.
  intros flag u e T cannot_change code_deps dominated to_unbox b base Hloc
    Hf.
  exact (FlambdaReaperRegionsUnboxKeepsProofs.rebuild_unbox_fresh flag u e T
           cannot_change code_deps dominated to_unbox b base Hloc Hf).
Qed.

(* ======================================================================== *)
(* The erased unit                                                          *)
(* ======================================================================== *)

Lemma closure_free_of_erase :
  (∀ e : expr, ∀ c, closure_free (erase c e) = true -> closure_free e = true)
  ∧ (∀ h : continuation_handler, ∀ c,
       handler_closure_free (erase_handler c h) = true ->
       handler_closure_free h = true).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH c Hcf.
    destruct (region_prim_of d) as [[p args] | ] eqn:Hrp.
    + apply region_prim_of_some in Hrp. subst d. cbn [erase] in Hcf.
      cbn [closure_free named_closure_free andb].
      destruct (erase_region_primitive x p args);
        [ cbn [closure_free] in Hcf; apply andb_true_iff in Hcf as [_ Hcf]
        | ];
        exact (IH c Hcf).
    + rewrite (erase_let_plain _ _ _ _ _ Hrp) in Hcf.
      cbn [closure_free] in Hcf |- *. apply andb_true_iff in Hcf as [Hd Hcf].
      rewrite Hd. exact (IH c Hcf).
  - intros bindings body _ IH c Hcf. cbn [erase closure_free] in Hcf |- *.
    apply andb_true_iff in Hcf as [Hbs Hcf]. rewrite (IH c Hcf), andb_true_r.
    rewrite forallb_forall in Hbs |- *. intros bd Hin.
    specialize (Hbs (erase_binding bd) (in_map _ _ _ Hin)).
    destruct bd; exact Hbs.
  - intros k h body IHh IH c Hcf. cbn [erase closure_free] in Hcf |- *.
    apply andb_true_iff in Hcf as [Hh Hcf].
    rewrite (IHh c Hh), (IH _ Hcf). reflexivity.
  - intros handlers body IHh IH c Hcf. cbn [erase closure_free] in Hcf |- *.
    apply andb_true_iff in Hcf as [Hhs Hcf]. rewrite (IH _ Hcf), andb_true_r.
    set (c' := map (fun '(k, h) => (k, region_params h)) handlers ++ c)
      in Hhs.
    rewrite forallb_forall in Hhs |- *. intros [k h] Hin.
    apply (IHh k h Hin c').
    apply (Hhs (k, erase_handler c' h)). apply in_map_iff.
    exists (k, h). split; [ reflexivity | exact Hin ].
  - intros ap c Hcf. exact Hcf.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - intros params md hbody IH c Hcf. cbn [erase_handler handler_closure_free]
      in Hcf |- *. exact (IH c Hcf).
Qed.

Lemma closure_free_projection_free :
  (∀ e : expr, closure_free e = true -> projection_free e = true)
  ∧ (∀ h : continuation_handler,
       handler_closure_free h = true -> handler_projection_free h = true).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH Hcf. cbn [closure_free] in Hcf.
    apply andb_true_iff in Hcf as [Hd Hcf]. cbn [projection_free].
    rewrite (IH Hcf), andb_true_r. destruct d; try discriminate Hd;
      reflexivity.
  - intros bindings body _ IH Hcf. cbn [closure_free] in Hcf.
    apply andb_true_iff in Hcf as [Hbs Hcf]. cbn [projection_free].
    rewrite (IH Hcf), andb_true_r. rewrite forallb_forall in Hbs |- *.
    intros bd Hin. specialize (Hbs bd Hin).
    destruct bd as [ | | s [ | | tag fields | | | ]]; try discriminate Hbs.
    reflexivity.
  - intros k h body IHh IH Hcf. cbn [closure_free] in Hcf.
    apply andb_true_iff in Hcf as [Hh Hcf]. cbn [projection_free].
    rewrite (IHh Hh), (IH Hcf). reflexivity.
  - intros handlers body IHh IH Hcf. cbn [closure_free] in Hcf.
    apply andb_true_iff in Hcf as [Hhs Hcf]. cbn [projection_free].
    rewrite (IH Hcf), andb_true_r. rewrite forallb_forall in Hhs |- *.
    intros [k h] Hin. exact (IHh k h Hin (Hhs _ Hin)).
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - intros params md hbody IH Hcf. exact (IH Hcf).
Qed.

Lemma drop_regions_in {A : Type} (places : list bool) (l : list A) (a : A) :
  a ∈ drop_regions places l -> a ∈ l.
Proof.
  revert l. induction places as [ | p places IH]; intros l Hin;
    [ exact Hin | ].
  destruct l as [ | b l]; [ destruct Hin | ].
  cbn [drop_regions] in Hin. destruct p.
  - right. exact (IH l Hin).
  - destruct Hin as [<- | Hin]; [ left; reflexivity | right; exact (IH l Hin) ].
Qed.

Lemma variables_apply_cont_erase (c : region_conts) (ac : apply_cont)
    (z : variable) :
  z ∈ variables_in_apply_cont (erase_apply_cont c ac) ->
  z ∈ variables_in_apply_cont ac.
Proof.
  unfold erase_apply_cont, variables_in_apply_cont, variables_in_simples.
  destruct (lookup (ac_continuation ac) c) as [places | ]; [ | exact id ].
  cbn [ac_args]. intros Hz.
  apply in_flat_map in Hz as (s & Hs & Hz). apply in_flat_map.
  exists s. split; [ exact (drop_regions_in _ _ _ Hs) | exact Hz ].
Qed.

Lemma erase_region_primitive_vars (x : variable) (p : region_primitive)
    (args : list simple) (d' : named) (z : variable) :
  erase_region_primitive x p args = Some d' ->
  z ∈ variables_in_named d' -> z ∈ variables_in_simples args.
Proof.
  intros Hd Hz. unfold erase_region_primitive in Hd.
  repeat match type of Hd with
  | context [match ?t with _ => _ end] => destruct t
  end;
    try discriminate Hd; injection Hd as <-;
    cbn [variables_in_named variables_in_simples variables_in_simple flat_map
           unit_simple app] in Hz |- *;
    first [ exact Hz
          | simpl in Hz; destruct Hz as [<- | []]; left; reflexivity
          | destruct Hz ].
Qed.

Lemma variables_erase :
  (∀ e : expr, ∀ c (z : variable),
     z ∈ variables_of (erase c e) -> z ∈ variables_of e)
  ∧ (∀ h : continuation_handler, ∀ c (z : variable),
       z ∈ variables_in_handler (erase_handler c h)
       -> z ∈ variables_in_handler h).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH c z Hz.
    destruct (region_prim_of d) as [[p args] | ] eqn:Hrp.
    + apply region_prim_of_some in Hrp. subst d. cbn [erase] in Hz.
      cbn [variables_of variables_in_named].
      destruct (erase_region_primitive x p args) as [d' | ] eqn:Hd;
        [ | right; apply in_or_app; right; exact (IH c z Hz) ].
      cbn [variables_of] in Hz. destruct Hz as [<- | Hz];
        [ left; reflexivity | right ].
      apply in_app_or in Hz as [Hz | Hz];
        [ | apply in_or_app; right; exact (IH c z Hz) ].
      apply in_or_app. left.
      exact (erase_region_primitive_vars x p args d' z Hd Hz).
    + rewrite (erase_let_plain _ _ _ _ _ Hrp) in Hz.
      cbn [variables_of] in Hz |- *. destruct Hz as [<- | Hz];
        [ left; reflexivity | right ].
      apply in_app_or in Hz as [Hz | Hz];
        [ apply in_or_app; left; exact Hz
        | apply in_or_app; right; exact (IH c z Hz) ].
  - intros bindings body Hcodes IH c z Hz. cbn [erase variables_of] in Hz |- *.
    apply in_app_or in Hz as [Hz | Hz];
      [ apply in_or_app; left | apply in_or_app; right; exact (IH c z Hz) ].
    apply in_flat_map in Hz as (bd & Hin & Hz).
    apply in_map_iff in Hin as (bd0 & <- & Hin).
    apply in_flat_map. exists bd0. split; [ exact Hin | ].
    destruct bd0 as [id cd | | ]; [ | exact Hz | exact Hz ].
    destruct cd as [params mc ret exn md cbody].
    cbn [erase_binding erase_code variables_in_binding variables_in_code]
      in Hz |- *.
    destruct Hz as [<- | Hz]; [ left; reflexivity | right ].
    apply in_app_or in Hz as [Hz | Hz]; apply in_or_app;
      [ left; exact Hz | right ].
    exact (Hcodes id _ Hin [] z Hz).
  - intros k h body IHh IH c z Hz. cbn [erase variables_of] in Hz |- *.
    apply in_app_or in Hz as [Hz | Hz]; apply in_or_app;
      [ left; exact (IHh c z Hz) | right; exact (IH _ z Hz) ].
  - intros handlers body IHh IH c z Hz. cbn [erase variables_of] in Hz |- *.
    apply in_app_or in Hz as [Hz | Hz]; apply in_or_app;
      [ left | right; exact (IH _ z Hz) ].
    apply in_flat_map in Hz as (kh & Hin & Hz).
    apply in_map_iff in Hin as ([k h] & <- & Hin).
    apply in_flat_map. exists (k, h). split; [ exact Hin | ].
    exact (IHh k h Hin _ z Hz).
  - intros ap c z Hz. exact Hz.
  - intros ac c z Hz. exact (variables_apply_cont_erase c ac z Hz).
  - intros scrutinee arms c z Hz. cbn [erase variables_of] in Hz |- *.
    apply in_app_or in Hz as [Hz | Hz]; apply in_or_app;
      [ left; exact Hz | right ].
    apply in_flat_map in Hz as (arm & Hin & Hz).
    apply in_map_iff in Hin as ([i ac] & <- & Hin).
    apply in_flat_map. exists (i, ac). split; [ exact Hin | ].
    exact (variables_apply_cont_erase c ac z Hz).
  - intros c z Hz. exact Hz.
  - intros params md hbody IH c z Hz.
    cbn [erase_handler variables_in_handler] in Hz |- *.
    apply in_app_or in Hz as [Hz | Hz]; apply in_or_app;
      [ left; exact (drop_regions_in _ _ _ Hz) | right; exact (IH c z Hz) ].
Qed.

Lemma fold_max_incl (l l' : list positive) :
  (∀ z, z ∈ l -> z ∈ l') ->
  (fold_right Pos.max 1 l <= fold_right Pos.max 1 l')%positive.
Proof.
  induction l as [ | a l IH]; intros H; cbn [fold_right]; [ lia | ].
  pose proof (URP.fold_max_le l' a (H a (or_introl eq_refl))) as Ha.
  assert (Hl : (fold_right Pos.max 1 l <= fold_right Pos.max 1 l')%positive)
    by (apply IH; intros z Hz; apply H; right; exact Hz).
  lia.
Qed.

Lemma fresh_name_erase_le (e : expr) :
  (fresh_name (names_of (erase [] e)) <= fresh_name (names_of e))%positive.
Proof.
  unfold fresh_name. apply (proj1 (Pos.succ_le_mono _ _)).
  apply fold_max_incl. intros z Hz. unfold names_of in Hz |- *.
  rewrite (proj1 FlambdaReaperRegionsProofs.continuations_erase) in Hz.
  apply in_app_or in Hz as [Hz | Hz]; apply in_or_app;
    [ left; exact (proj1 variables_erase e [] z Hz) | right; exact Hz ].
Qed.

(* ======================================================================== *)
(* The refinement on the machine of regions                                 *)
(* ======================================================================== *)

Lemma check_unboxed_reads_reads (F : facts) (T : tables)
    (dominated : list (node * node)) (to_unbox : list node) :
  check_unboxed_reads F T dominated to_unbox
  = check_unboxed_reads (reads_of F) T dominated to_unbox.
Proof. reflexivity. Qed.

Lemma coherent_consts_used (T : tables) :
  URP.coherent T
    (FlambdaReaperRegionsProofs.consts_used (solution_of_tables T)).
Proof.
  split; try reflexivity.
  - intros [ | | | | ] H; first [ reflexivity | exact H ].
  - intros [ | | | | ] H; first [ reflexivity | exact H ].
Qed.

Theorem least_rebuild_unbox_regions_refines :
  least_rebuild_unbox_regions_refines_statement.
Proof.
  intros flag is_local u e T max_unbox_size lambda_lifting b cert base Hpl
    Hslots Hfrag Hstatic Hrk Hwf Hexits Hleast Hlocal Hreads Hfe Hee Hcheck.
  set (D := derive_dominated T) in *.
  set (U := least_to_unbox is_local (traverse_unit flag u e) T max_unbox_size
              lambda_lifting) in *.
  set (sol := solution_of_tables T).
  set (sol' := FlambdaReaperRegionsProofs.consts_used sol).
  assert (Hcf' : closure_free (erase [] e) = true)
    by exact (proj1 (URP.local_closure_free _ _) _ _ _ Hlocal).
  assert (Hcf : closure_free e = true)
    by exact (proj1 closure_free_of_erase e [] Hcf').
  assert (Hloc : regions_local e = true)
    by (unfold regions_local; rewrite Hpl, Hcf; reflexivity).
  assert (Hc : closed is_local (graph_of_facts (traverse_unit flag u e)) sol)
    by exact (FlambdaReaperSolutionProofs.check_closed_sound is_local _ T
                (proj1 Hleast)).
  assert (Hnu : regions_not_unboxed D U e = true)
    by (apply regions_not_unboxed_any_source;
        exact (closed_regions_any_source flag is_local u e T Hloc Hfrag Hc)).
  apply (FlambdaReaperRegionsProofs.regions_refine_of_erased cert base e);
    try assumption.
  - apply rebuild_unbox_keeps_fresh; assumption.
  - apply rebuild_unbox_keeps_erasable; assumption.
  - unfold erased_initial_state.
    rewrite FlambdaReaperRegionsProofs.erase_statics_empty.
    destruct (unboxed_fields_of_names (traverse_unit flag u e) T
                (bound_code_my_closures e) D U (fresh_name (names_of e)))
      as [ub next] eqn:Hub.
    rewrite (rebuild_unbox_erase_commutes_with flag u e T sol'
               (derive_cannot_change (traverse_unit flag u e) T)
               (unit_code_deps flag u e) D U b ub next Hloc Hnu
               (fun n Hn => proj2 (FlambdaReaperRegionsProofs.cu_usage sol n
                                     Hn))
               eq_refl eq_refl Hub).
    (* The solution of the erased unit. *)
    destruct (traverse_unit_rel flag u e Hpl) as (l & l' & HF & HF' & Hrel).
    destruct (traverse_unit_positions flag u e) as (l0 & HF0 & Hpos0).
    assert (Hpos : FlambdaReaperRegionsProofs.positions_ok
                     (graph_of_facts (traverse_unit flag u e)))
      by (rewrite HF0;
          apply FlambdaReaperRegionsProofs.graph_positions, Hpos0).
    assert (Hgb : FlambdaReaperRegionsProofs.graph_below
                    (graph_of_facts (traverse_unit flag u e))
                    (graph_of_facts (traverse_unit flag u (erase [] e))))
      by (rewrite HF, HF';
          apply FlambdaReaperRegionsProofs.graph_below_of_rel, Hrel).
    assert (Hexact : field_used_exact is_local
                       (graph_of_facts (traverse_unit flag u e)) sol)
      by exact (FlambdaReaperSolutionProofs.check_field_used_exact_sound
                  is_local _ T
                  (FlambdaReaperLeastProofs.least_field_used_exact is_local _
                     T Hleast)).
    destruct (FlambdaReaperRegionsProofs.least_no_consts is_local _ T Hleast
                Hpos) as [Hna Hnr].
    assert (Hc' : closed is_local
                    (graph_of_facts (traverse_unit flag u (erase [] e))) sol')
      by exact (FlambdaReaperRegionsProofs.consts_used_closed is_local _ _ sol
                  Hc Hna Hnr Hpos Hgb).
    assert (Hexact' : field_used_exact is_local
                        (graph_of_facts (traverse_unit flag u (erase [] e)))
                        sol')
      by exact (FlambdaReaperRegionsProofs.consts_used_exact is_local _ _ sol
                  Hgb Hexact).
    assert (Hlrs' : local_reads_used is_local
                      (graph_of_facts (traverse_unit flag u (erase [] e)))
                      sol').
    { intros base' rel to Has Hacc Hl. exfalso.
      destruct (FlambdaReaperFlowProofs.traverse_unit_reads_no_slot flag u
                  (erase [] e) (proj1 closure_free_projection_free _ Hcf') to
                  rel base' Hacc) as [Hw Hfs].
      destruct (Hslots rel Hl) as [(w & ->) | (fs & ->)];
        [ exact (Hw w eq_refl) | exact (Hfs fs eq_refl) ]. }
    (* The unboxed fields of the erased unit. *)
    assert (Hub' : unboxed_fields_of_names (traverse_unit flag u (erase [] e))
                     T (bound_code_my_closures (erase [] e)) D U
                     (fresh_name (names_of e))
                   = (ub, next))
      by (rewrite unboxed_fields_erase; [ exact Hub | exact Hpl ]).
    assert (Hloc' : local_expr ub sol' [] [] (erase [] e) = true).
    { rewrite (proj1 (local_expr_ext ub sol sol' (fun _ => eq_refl))).
      rewrite (local_unboxing_names (traverse_unit flag u (erase [] e)) T D U
                 (erase [] e) (fresh_name (names_of e))) in Hlocal.
      rewrite Hub' in Hlocal. exact Hlocal. }
    assert (Hcheck' : check_unboxed_reads (traverse_unit flag u (erase [] e))
                        T D U = true).
    { rewrite check_unboxed_reads_reads, HF'.
      destruct Hrel as [(HC & HA & _ & _) _].
      unfold reads_of. rewrite HC, HA.
      change (check_unboxed_reads (reads_of (facts_of_list l)) T D U = true).
      rewrite <- HF, <- check_unboxed_reads_reads.
      exact (FlambdaReaperLeastProofs.least_unboxed_reads is_local _ T
               (derive_cannot_change (traverse_unit flag u e) T)
               max_unbox_size lambda_lifting Hleast Hreads). }
    apply (URP.refines_observably_refines _
             (State (rebuild_expr (traverse_unit flag u (erase [] e)) sol' b
                       (static_names code_ids_bound (erase [] e))
                       (fresh_name (names_of e)) None
                       FlambdaReaperRebuild.empty_env (erase [] e))
                [] [] empty_static_env empty_heap no_reads)).
    + exact (FlambdaReaperRegionsProofs.rebuild_refines_observably_gen
               flag is_local u (erase [] e) sol' b (fresh_name (names_of e))
               Hslots Hfrag
               (FlambdaReaperRebuildProofs.static_constants_any_kind_arguments
                  _ _ Hstatic)
               Hrk Hwf Hexits Hc' Hlrs' Hexact'
               (FlambdaReaperRegionsProofs.fresh_name_erase e)).
    + exact (URP.rebuild_unbox_refines_rebuild_sol None flag is_local u
               (erase [] e) T sol'
               (fun n => mem_node n
                           (derive_cannot_change (traverse_unit flag u e) T))
               (unit_code_deps flag u e) D U b (fresh_name (names_of e))
               (fresh_name (names_of e)) next ub Hslots Hfrag Hc'
               (coherent_consts_used T) (fresh_name_erase_le e) Hub' Hloc'
               Hcheck').
Qed.

(* ======================================================================== *)
(* Example                                                                  *)
(* ======================================================================== *)

Import FlambdaReaperRegionsSpec.RegionsExamples RegionsUnboxExamples.

Theorem regions_unbox_example : regions_unbox_example_statement.
Proof.
  assert (Hleast : least_solution (fun _ => false)
                     (traverse_unit Auto interface local_pair)
                     local_pair_tables)
    by (apply (FlambdaReaperLeastProofs.check_least_sound _ _ 100);
        vm_compute; reflexivity).
  assert (Hreb : rebuild_unit_unbox (traverse_unit Auto interface local_pair)
                   local_pair_tables
                   (derive_cannot_change
                      (traverse_unit Auto interface local_pair)
                      local_pair_tables)
                   (unit_code_deps Auto interface local_pair)
                   (derive_dominated local_pair_tables) local_pair_to_unbox
                   true local_pair
                 = local_pair_rebuilt)
    by (vm_compute; reflexivity).
  split; [ vm_compute; reflexivity | ]. split; [ exact Hleast | ].
  do 10 (split; [ vm_compute; reflexivity | ]).
  do 2 (split; [ vm_compute; reflexivity | ]).
  split; [ exact Hreb | ].
  split; [ vm_compute; reflexivity | ].
  split; [ FlambdaReaperRegionsProofs.ex_run | ].
  split; [ FlambdaReaperRegionsProofs.ex_run | ].
  rewrite <- Hreb. unfold local_pair_to_unbox.
  apply (least_rebuild_unbox_regions_refines Auto (fun _ => false) interface
           local_pair local_pair_tables 10 false true cert base);
    try exact Hleast; try (vm_compute; reflexivity);
    try (fold local_pair_to_unbox; rewrite Hreb; vm_compute; reflexivity).
  exact FlambdaReaperRegionsProofs.no_local_slots.
Qed.
