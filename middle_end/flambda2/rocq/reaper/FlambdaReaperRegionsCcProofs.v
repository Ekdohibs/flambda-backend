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

(* Proofs of the statements of FlambdaReaperRegionsCcSpec.v.

   This file needs no review: what it proves is stated there. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaPrimitiveModel
  FlambdaModel FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec
  FlambdaSemanticsSpec FlambdaWellFormedSpec FlambdaNameAbstractionSpec
  FlambdaRenameApartSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperSolutionSpec FlambdaReaperFlowSpec
  FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperCallingConventionsSpec
  FlambdaReaperLeastSpec FlambdaReaperLocalFieldsSpec FlambdaRegionsSpec
  FlambdaRegionsErasureSpec FlambdaRegionsCheckSpec
  FlambdaReaperRegionsSpec FlambdaReaperRegionsCcSpec
  FlambdaReaperRegionsRebuildProofs FlambdaReaperRegionsTraverseProofs
  FlambdaReaperRegionsKeepsProofs FlambdaReaperRegionsCcRebuildProofs
  FlambdaReaperRegionsProofs.
From OxCamlFlambda2 Require FlambdaReaperSolutionProofs
  FlambdaReaperLeastProofs FlambdaReaperFlowProofs
  FlambdaReaperCallingConventionsProofs FlambdaReaperRebuildProofs.
Import ListNotations.

(* ======================================================================== *)
(* The code deps and the table of the erased unit                           *)
(* ======================================================================== *)

(* The accumulators of the two traversals of a unit. *)
Lemma traverse_unit_acc_rel (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) :
  regions_plain [] e = true ->
  acc_rel (traverse_unit_acc flag u e) (traverse_unit_acc flag u (erase [] e)).
Proof.
  intros Hpl. unfold traverse_unit_acc.
  rewrite (proj1 projected_value_slots_erase e []).
  rewrite (proj1 (static_names_erase all_code_ids_of_binding
                    ltac:(intros [ | | ]; reflexivity)) e []).
  rewrite (proj1 (static_names_erase symbols_of_binding
                    ltac:(intros [ | | ]; reflexivity)) e []).
  rewrite (proj1 set_closures_erase e [] Hpl).
  cbn [fresh fst snd].
  apply (acc_rel_frame _ (fr_resolve _)).
  apply (acc_rel_frame _ (fr_add_fact _)).
  apply (proj1 traverse_rel e [] _ _ _ Hpl), acc_rel_refl.
Qed.

Lemma unit_code_deps_erase (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) :
  regions_plain [] e = true ->
  unit_code_deps flag u (erase [] e) = unit_code_deps flag u e.
Proof.
  intros Hpl. destruct (traverse_unit_acc_rel flag u e Hpl) as [Hcl _].
  unfold unit_code_deps.
  change (acc_code_deps (clear (traverse_unit_acc flag u (erase [] e)))
          = acc_code_deps (clear (traverse_unit_acc flag u e))).
  rewrite Hcl. reflexivity.
Qed.

(* The nodes of the code deps of a unit are not the constants. *)
Lemma unit_code_deps_ok (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) :
  ∀ (id : code_id) (cd : code_dep),
    (id, cd) ∈ unit_code_deps flag u e -> cd_ok cd.
Proof.
  unfold unit_code_deps, traverse_unit_acc. cbn [fresh fst snd].
  match goal with
  | |- ∀ id cd, (id, cd) ∈ acc_code_deps (resolve_delayed_deps ?syms
                  (add_fact ?f (traverse ?env e ?a))) -> _ =>
    assert (He : penv env); [ | assert (Ha : pinv a) ]
  end.
  - intros k ps Hin. cbn [FlambdaReaperTraversal.env_conts] in Hin.
    destruct Hin as [Heq | [Heq | []]]; injection Heq as _ <-;
      constructor; [ apply nc_made | constructor | apply nc_made
                   | constructor ].
  - split; [ | split; [ | split ] ].
    + intros fct Hin. cbn in Hin.
      destruct Hin as [<- | [<- | []]]; exact I.
    + intros id cd [].
    + intros d [].
    + intros d [].
  - refine (proj1 (proj2 (_ : pinv _))).
    apply pinv_resolve, pinv_add_fact; [ exact I | ].
    exact (proj1 pinv_traverse e _ _ He Ha).
Qed.

Lemma derive_cannot_change_erase (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) (T : tables) :
  regions_plain [] e = true ->
  derive_cannot_change (traverse_unit flag u (erase [] e)) T
  = derive_cannot_change (traverse_unit flag u e) T.
Proof.
  intros Hpl.
  destruct (traverse_unit_rel flag u e Hpl) as (l & l' & HF & HF' & Hrel).
  destruct Hrel as [(HC & HA & _ & _) _].
  unfold derive_cannot_change, derived_by_code_id_of_witness,
    derived_by_code_id_unknown_arity, code_ids_of_witness,
    cannot_change_witness_calling_convention, witness_other_source,
    witness_any_source_usage, code_id_read_with_usage.
  rewrite HF, HF', HC, HA. reflexivity.
Qed.

(* ======================================================================== *)
(* The code that changes returns no region                                  *)
(* ======================================================================== *)

Lemma kept_kinds_decisions (sol : solution) (nodes : list node)
    (kinds : list flambda_kind) (k : flambda_kind) :
  k ∈ kept_kinds (decisions_of sol nodes kinds) -> k ∈ kinds.
Proof.
  intros Hk. apply kept_kinds_in in Hk. unfold decisions_of in Hk.
  apply in_map_iff in Hk as [[n kind] [Heq Hin]].
  destruct (is_node_used sol n kind); [ | discriminate Heq ].
  injection Heq as ->. exact (in_combine_r _ _ _ _ Hin).
Qed.

Lemma code_results_hres (e : expr) :
  code_results_plain e = true ->
  ∀ (F : facts) (sol : solution) (cc : node -> bool)
    (cds : finite_map code_dep) (ucids : list code_id) (ap : apply_expr)
    (pd rd : list param_decision),
    apply_calling_convention_change F sol cc cds ucids
      (bound_code_metadata e) ap
    = Changing_calling_convention pd rd ->
    forallb kind_not_region (kept_kinds rd) = true.
Proof.
  intros Hr F sol cc cds ucids ap pd rd.
  unfold apply_calling_convention_change.
  destruct (code_id_actually_called F sol ap) as [id | ];
    [ | discriminate ].
  unfold get_calling_convention_change.
  destruct (lookup id (bound_code_metadata e)) as [md | ] eqn:Hl;
    [ | discriminate ].
  unfold code_calling_convention_change.
  destruct (cannot_change_calling_convention cc ucids id);
    [ discriminate | ].
  destruct (lookup id cds) as [cd | ]; [ | discriminate ].
  intros Heq. injection Heq as _ <-.
  unfold code_results_plain in Hr. rewrite forallb_forall in Hr.
  specialize (Hr _ (lookup_in _ _ _ Hl)). cbv beta iota in Hr.
  rewrite forallb_forall in Hr |- *. intros k Hk.
  exact (Hr k (kept_kinds_decisions _ _ _ _ Hk)).
Qed.

(* ======================================================================== *)
(* The commutation                                                          *)
(* ======================================================================== *)

Lemma rebuild_cc_erase_commutes_with (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) (sol sol' : solution) (b : bool)
    (cc : node -> bool) :
  regions_plain [] e = true -> code_results_plain e = true ->
  (∀ n : node, n <> Node_all_constants ->
     s_has_usage sol' n = s_has_usage sol n) ->
  s_has_source sol' = s_has_source sol ->
  s_any_source sol' = s_any_source sol ->
  s_sources sol' = s_sources sol ->
  s_field_used sol' = s_field_used sol ->
  erase []
    (rebuild_expr_cc (traverse_unit flag u e) sol cc (unit_code_deps flag u e)
       b (static_names code_ids_bound e) (bound_code_metadata e)
       (fresh_name (names_of e)) None FlambdaReaperRebuild.empty_env e)
  = rebuild_expr_cc (traverse_unit flag u (erase [] e)) sol' cc
      (unit_code_deps flag u (erase [] e)) b
      (static_names code_ids_bound (erase [] e))
      (bound_code_metadata (erase [] e)) (fresh_name (names_of e)) None
      FlambdaReaperRebuild.empty_env (erase [] e).
Proof.
  intros Hpl Hrp Hu Hsrc Has Hss Hfu.
  destruct (traverse_unit_rel flag u e Hpl) as (l & l' & HF & HF' & Hrel).
  destruct (traverse_unit_positions flag u e) as (l0 & HF0 & Hpos0).
  destruct Hrel as [(HC & HA & HP & _) _].
  rewrite (proj1 (static_names_erase code_ids_bound
                    ltac:(intros [ | | ]; reflexivity)) e []).
  rewrite (proj1 bound_code_metadata_erase e []).
  rewrite (unit_code_deps_erase flag u e Hpl).
  assert (Hpac : ∀ (x : node) (i : nat) (y : node),
            (x, i, y) ∈ f_parameter (traverse_unit flag u e) ->
            y <> Node_all_constants)
    by (rewrite HF0; apply parameters_not_consts, Hpos0).
  assert (HC' : f_constructor (traverse_unit flag u (erase [] e))
                = f_constructor (traverse_unit flag u e))
    by (rewrite HF, HF'; exact HC).
  assert (HA' : f_accessor (traverse_unit flag u (erase [] e))
                = f_accessor (traverse_unit flag u e))
    by (rewrite HF, HF'; exact HA).
  assert (HP' : f_parameter (traverse_unit flag u (erase [] e))
                = f_parameter (traverse_unit flag u e))
    by (rewrite HF, HF'; exact HP).
  assert (Hcds :
    ∀ (id : code_id) (cd : code_dep),
      lookup id (unit_code_deps flag u e) = Some cd ->
      ∀ n : node, n ∈ cd_params cd ++ cd_return cd ->
        n <> Node_all_constants).
  { intros id cd Hl n Hn.
    destruct (unit_code_deps_ok flag u e id cd (lookup_in _ _ _ Hl))
      as (C1 & _ & C3 & _).
    apply in_app_or in Hn as [Hn | Hn];
      [ exact (proj1 (Forall_forall _ _) C1 n Hn)
      | exact (proj1 (Forall_forall _ _) C3 n Hn) ]. }
  exact (proj1 (rebuild_cc_erase_commutes_gen _ _ _ _ b _
                  (fresh_name (names_of e)) cc _ _ HC' HA' HP' Hpac Hu Hsrc
                  Has Hss Hfu Hcds
                  (code_results_hres e Hrp (traverse_unit flag u e) sol cc
                     (unit_code_deps flag u e) (static_names code_ids_bound e)))
           e FlambdaReaperRebuild.empty_env FlambdaReaperRebuild.empty_env Hpl
           (env_rel_refl _)
           (pending_ok_nil FlambdaReaperRebuild.empty_env e eq_refl)).
Qed.

Theorem rebuild_cc_erase_commutes : rebuild_cc_erase_commutes_statement.
Proof.
  intros flag u e T cct b Hpl. unfold regions_cc_plain in Hpl.
  apply andb_true_iff in Hpl as [Hpl Hrp].
  unfold rebuild_unit_cc_from_tables, rebuild_unit_cc.
  exact (rebuild_cc_erase_commutes_with flag u e _ _ b _ Hpl Hrp
           (fun _ _ => eq_refl) eq_refl eq_refl eq_refl eq_refl).
Qed.

Theorem cc_tables_erase : cc_tables_erase_statement.
Proof.
  intros flag u e T Hpl. split;
    [ apply derive_cannot_change_erase | apply unit_code_deps_erase ];
    exact Hpl.
Qed.

(* ======================================================================== *)
(* What the rebuild keeps                                                   *)
(* ======================================================================== *)

Theorem rebuild_cc_keeps_erasable : rebuild_cc_keeps_erasable_statement.
Proof.
  intros F T cct cds b e Hpl He. unfold regions_cc_plain in Hpl.
  apply andb_true_iff in Hpl as [Hpl Hrp].
  unfold rebuild_unit_cc_from_tables, rebuild_unit_cc.
  apply rebuild_cc_erasable; [ | exact Hpl | exact He ].
  apply (code_results_hres e Hrp).
Qed.

Theorem rebuild_cc_keeps_fresh : rebuild_cc_keeps_fresh_statement.
Proof.
  intros F T cct cds b e base Hpl Hf. unfold regions_cc_plain in Hpl.
  apply andb_true_iff in Hpl as [Hpl Hrp].
  unfold rebuild_unit_cc_from_tables, rebuild_unit_cc.
  apply rebuild_cc_fresh; [ | exact Hpl | exact Hf ].
  apply (code_results_hres e Hrp).
Qed.

(* ======================================================================== *)
(* The table of the code that cannot change, on the erased graph            *)
(* ======================================================================== *)

(* A least solution has no usage of the constants but by the constants. *)
Lemma least_usages_consts (is_local : field -> bool) (F : facts)
    (T : tables) :
  least_solution is_local F T -> positions_ok (graph_of_facts F) ->
  ∀ a : node,
    s_usages (solution_of_tables T) a Node_all_constants = true ->
    a = Node_all_constants.
Proof.
  intros Hleast Hpos a Ha.
  destruct (least_no_consts is_local F T Hleast Hpos) as [Hna _].
  pose proof (FlambdaReaperLeastProofs.least_derivable is_local F T Hleast
                (Atom_usages a Node_all_constants) Ha) as D.
  assert (Hgen : ∀ at0 : atom,
            derivable is_local (graph_of_facts F) (solution_of_tables T) at0 ->
            match at0 with
            | Atom_usages x usage =>
              usage = Node_all_constants -> x = Node_all_constants
            | _ => True
            end).
  { intros at0 Hd. induction Hd; try exact I;
      try (intros Hx; exact Hx).
    match goal with
    | IH : ?w = Node_all_constants -> ?t = Node_all_constants,
      Hal : derivable _ _ _ (Atom_alias ?t ?f)
      |- ?w = Node_all_constants -> ?f = Node_all_constants =>
      intros Hw; specialize (IH Hw); subst t; exfalso;
      pose proof (FlambdaReaperLeastProofs.least_holds is_local F T Hleast _
                    Hal) as Hh;
      cbn [holds] in Hh; rewrite Hna in Hh; discriminate Hh
    end. }
  exact (Hgen _ D eq_refl).
Qed.

Lemma node_eqb_refl (n : node) : node_eqb n n = true.
Proof.
  unfold node_eqb. destruct (node_eq_dec n n) as [_ | Hn];
    [ reflexivity | contradiction Hn; reflexivity ].
Qed.

Lemma node_eqb_ne (n n' : node) : n <> n' -> node_eqb n n' = false.
Proof.
  intros H. unfold node_eqb. destruct (node_eq_dec n n') as [E | _];
    [ contradiction | reflexivity ].
Qed.

(* A table closed under the rules of the code that cannot change on the
   graph of the unit with its regions is on the erased graph and the
   solution with the constants used, with the constants among the
   witnesses that cannot change. *)
Lemma cc_closed_consts (G G' : graph) (S : solution) (w cc : node -> bool) :
  positions_ok G -> graph_below G G' ->
  (∀ a : node, s_usages S a Node_all_constants = true ->
     a = Node_all_constants) ->
  cc_closed G S w cc ->
  cc_closed G' (consts_used S)
    (fun n => w n || node_eqb n Node_all_constants) cc.
Proof.
  intros Hpos Hgb Hus Hc.
  assert (Hw : ∀ n, w n = true -> w n || node_eqb n Node_all_constants = true)
    by (intros n H; rewrite H; reflexivity).
  constructor.
  - intros x Hx. destruct (node_eq_dec x Node_all_constants) as [-> | Hne];
      [ rewrite node_eqb_refl, orb_true_r; reflexivity | ].
    apply Hw. apply (cc_witness_any_usage _ _ _ _ Hc).
    rewrite <- (proj1 (cu_usage S x Hne)). exact Hx.
  - intros a alias s v Husg Hsrc Hhs Hne Hacc Hhu. apply Hw.
    pose proof (proj1 (gb_accessor _ _ Hgb _ _ _) Hacc) as HaccG.
    pose proof (po_accessor _ Hpos _ _ _ HaccG) as Hv.
    rewrite (proj2 (cu_usage S v Hv)) in Hhu.
    exact (cc_witness_other_source _ _ _ _ Hc a alias s v Husg Hsrc Hhs Hne
             HaccG Hhu).
  - intros a alias v Husg Has Hacc Hhu. apply Hw.
    pose proof (proj1 (gb_accessor _ _ Hgb _ _ _) Hacc) as HaccG.
    pose proof (po_accessor _ Hpos _ _ _ HaccG) as Hv.
    rewrite (proj2 (cu_usage S v Hv)) in Hhu.
    exact (cc_witness_any_source_usage _ _ _ _ Hc a alias v Husg Has HaccG
             Hhu).
  - intros a s Hs Hhs Hne. apply Hw.
    exact (cc_witness_other_own_source _ _ _ _ Hc a s Hs Hhs Hne).
  - intros a usage Husg Hhu Hns.
    destruct (node_eq_dec usage Node_all_constants) as [-> | Hne].
    + rewrite (Hus a Husg), node_eqb_refl, orb_true_r. reflexivity.
    + apply Hw. rewrite (proj2 (cu_usage S usage Hne)) in Hhu.
      exact (cc_witness_not_own_source _ _ _ _ Hc a usage Husg Hhu Hns).
  - intros a Has. apply Hw. exact (cc_witness_any_source _ _ _ _ Hc a Has).
  - intros cw cid Hcons Hhu Hww.
    pose proof (proj1 (gb_constructor _ _ Hgb _ _ _) Hcons) as HconsG.
    pose proof (po_constructor _ Hpos _ _ _ HconsG) as Hne.
    rewrite (proj2 (cu_usage S cw Hne)) in Hhu.
    rewrite (node_eqb_ne _ _ Hne), orb_false_r in Hww.
    exact (cc_code_id_of_witness _ _ _ _ Hc cw cid HconsG Hhu Hww).
  - intros set cw cid Hcons1 Hhu Hcons2.
    pose proof (proj1 (gb_constructor _ _ Hgb _ _ _) Hcons1) as Hc1.
    pose proof (proj1 (gb_constructor _ _ Hgb _ _ _) Hcons2) as Hc2.
    pose proof (po_constructor _ Hpos _ _ _ Hc2) as Hne.
    rewrite (proj2 (cu_usage S cw Hne)) in Hhu.
    exact (cc_code_id_unknown_arity _ _ _ _ Hc set cw cid Hc1 Hhu Hc2).
Qed.

(* The witnesses read off a node have the same sources in the solution
   with the constants used. *)
Lemma widgets_consts (G G' : graph) (S : solution) :
  graph_below G G' ->
  (∀ a a' n y : node,
     g_accessor G a (Field_call_witness Known_arity_code_pointer) n ->
     g_accessor G a' (Field_call_witness Known_arity_code_pointer) n ->
     (s_any_source S a' = true -> s_any_source S a = true)
     ∧ (s_sources S a' y = true -> s_sources S a y = true)) ->
  ∀ a a' n y : node,
    g_accessor G' a (Field_call_witness Known_arity_code_pointer) n ->
    g_accessor G' a' (Field_call_witness Known_arity_code_pointer) n ->
    (s_any_source (consts_used S) a' = true ->
     s_any_source (consts_used S) a = true)
    ∧ (s_sources (consts_used S) a' y = true ->
       s_sources (consts_used S) a y = true).
Proof.
  intros Hgb H a a' n y Ha Ha'.
  exact (H a a' n y (proj1 (gb_accessor _ _ Hgb _ _ _) Ha)
           (proj1 (gb_accessor _ _ Hgb _ _ _) Ha')).
Qed.

(* ======================================================================== *)
(* The refinement on the machine of regions                                 *)
(* ======================================================================== *)

(* The refinement up to observation of the rebuild with changes of
   calling convention, for any solution closed under the rules, of which
   the fix of local fields and the post-processing hold, given as
   propositions, and any first fresh name that no continuation of the
   unit is (FlambdaReaperRebuildProofs.rebuild_cc_with_related_gen). *)
Lemma rebuild_cc_refines_observably_gen (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr)
    (sol : solution) (cannot_change cannot_change_witness : node -> bool)
    (b : bool) (ff : positive) :
  is_local_slots_only is_local ->
  in_flow_fragment e = true ->
  any_kind_arguments e = true ->
  results_kinded [] [] e = true ->
  well_formed_unit u e = true ->
  exits_one_value u e = true ->
  closed is_local (graph_of_facts (traverse_unit flag u e)) sol ->
  local_reads_used is_local (graph_of_facts (traverse_unit flag u e)) sol ->
  field_used_exact is_local (graph_of_facts (traverse_unit flag u e)) sol ->
  among ff (continuations_of e) = false ->
  cc_closed (graph_of_facts (traverse_unit flag u e)) sol
    cannot_change_witness cannot_change ->
  (∀ a a' n y : node,
     g_accessor (graph_of_facts (traverse_unit flag u e)) a
       (Field_call_witness Known_arity_code_pointer) n ->
     g_accessor (graph_of_facts (traverse_unit flag u e)) a'
       (Field_call_witness Known_arity_code_pointer) n ->
     (s_any_source sol a' = true -> s_any_source sol a = true)
     ∧ (s_sources sol a' y = true -> s_sources sol a y = true)) ->
  (∀ (w : node) (id id' : code_id),
     g_constructor (graph_of_facts (traverse_unit flag u e)) w
       Field_code_id_of_call_witness (Node_code_id id) ->
     g_constructor (graph_of_facts (traverse_unit flag u e)) w
       Field_code_id_of_call_witness (Node_code_id id') ->
     id' = id) ->
  refines_observably (State e [] [] empty_static_env empty_heap no_reads)
    (State (rebuild_expr_cc (traverse_unit flag u e) sol cannot_change
              (unit_code_deps flag u e) b (static_names code_ids_bound e)
              (bound_code_metadata e) ff None FlambdaReaperRebuild.empty_env
              e)
       [] [] empty_static_env empty_heap no_reads).
Proof.
  intros Hlocal Hfrag Hdc Hrk Hwf Hexits Hc Hlrs Hexact Hfresh Hccc Hwid
    Hcid.
  assert (Hkinded : FlambdaReaperRebuildProofs.calls_kinded [] e = true).
  { apply (proj1 FlambdaReaperRebuildProofs.calls_kinded_of_results e [] []
             [] Hrk).
    intros k l Hl. discriminate Hl. }
  pose proof (proj1 (FlambdaReaperRebuildProofs.any_kind_arguments_checked
                       (unit_codes e)) e Hdc) as Hcf.
  destruct (FlambdaReaperRebuildProofs.rebuild_cc_with_related_gen false
              false flag is_local u e sol cannot_change cannot_change_witness
              (unit_code_deps flag u e) b None False ff Hlocal Hfrag Hcf
              Hkinded Hwf Hexits Hc Hlrs Hexact Hfresh
              (fun _ _ _ _ _ _ _ _ _ _ _ _ => I)
              (FlambdaReaperRebuildProofs.no_types_deps _)
              (FlambdaReaperRebuildProofs.no_types_conts e) Hrk Hccc
              (fun id a a' n y _ => Hwid a a' n y)
              (fun id id' w _ => Hcid w id id') (fun _ _ => eq_refl)
              (fun Hf : False => False_ind _ Hf))
    as (noted & deps & cds & H0 & Hans).
  exact (proj1 (FlambdaReaperRebuildProofs.exits_rel_flow false _ _
                  (FlambdaReaperRebuildProofs.located_refines _ _ e _ H0
                     Hans))).
Qed.

Theorem least_rebuild_cc_regions_refines :
  least_rebuild_cc_regions_refines_statement.
Proof.
  intros flag is_local u e T b cert base Hpl0 Hslots Hlocal Hfrag Hdc Hrk
    Hwf Hexits Hleast Hwidgets Hfe Hee Hcheck.
  pose proof Hpl0 as Hpl1. unfold regions_cc_plain in Hpl1.
  apply andb_true_iff in Hpl1 as [Hpl Hrp].
  apply (regions_refine_of_erased cert); try assumption;
    [ apply rebuild_cc_keeps_fresh | apply rebuild_cc_keeps_erasable | ];
    try assumption.
  unfold erased_initial_state. rewrite erase_statics_empty.
  unfold rebuild_unit_cc_from_tables, rebuild_unit_cc.
  rewrite (rebuild_cc_erase_commutes_with flag u e (solution_of_tables T)
             (consts_used (solution_of_tables T)) b
             (fun n => mem_node n
                         (derive_cannot_change (traverse_unit flag u e) T))
             Hpl Hrp (fun n Hn => proj2 (cu_usage (solution_of_tables T) n Hn))
             eq_refl eq_refl eq_refl eq_refl).
  destruct (FlambdaReaperCallingConventionsProofs.check_cc_closed_sound
              (traverse_unit flag u e) T
              (derive_cannot_change (traverse_unit flag u e) T)
              (proj1 (FlambdaReaperCallingConventionsProofs
                      .derive_cannot_change_least _ _)))
    as (w & Hccl).
  pose proof (FlambdaReaperCallingConventionsProofs.check_apply_widgets_sound
                _ _ (FlambdaReaperLeastProofs.least_apply_widgets is_local
                       (traverse_unit flag u e) T Hslots Hleast Hwidgets))
    as Hwid.
  pose proof (FlambdaReaperCallingConventionsProofs
              .check_witness_code_ids_sound _
                (FlambdaReaperCallingConventionsProofs.traverse_witness_code_ids
                   flag u (erase [] e)))
    as Hcid.
  remember (traverse_unit flag u e) as F eqn:HFdef.
  remember (traverse_unit flag u (erase [] e)) as F' eqn:HF'def.
  remember (solution_of_tables T) as sol eqn:Hsoldef.
  destruct (traverse_unit_rel flag u e Hpl) as (l & l' & HF & HF' & Hrel).
  destruct (traverse_unit_positions flag u e) as (l0 & HF0 & Hpos0).
  assert (Hpos : positions_ok (graph_of_facts F))
    by (rewrite HFdef, HF0; apply graph_positions, Hpos0).
  assert (Hgb : graph_below (graph_of_facts F) (graph_of_facts F'))
    by (rewrite HFdef, HF'def, HF, HF'; apply graph_below_of_rel, Hrel).
  assert (Hc : closed is_local (graph_of_facts F) sol)
    by (rewrite Hsoldef;
        exact (FlambdaReaperSolutionProofs.check_closed_sound is_local F T
                 (proj1 Hleast))).
  assert (Hexact : field_used_exact is_local (graph_of_facts F) sol)
    by (rewrite Hsoldef;
        exact (FlambdaReaperSolutionProofs.check_field_used_exact_sound
                 is_local F T
                 (FlambdaReaperLeastProofs.least_field_used_exact is_local F T
                    Hleast))).
  destruct (least_no_consts is_local F T Hleast Hpos) as [Hna Hnr].
  pose proof (least_usages_consts is_local F T Hleast Hpos) as Hus.
  try rewrite <- Hsoldef in Hna; try rewrite <- Hsoldef in Hnr;
    try rewrite <- Hsoldef in Hus; try rewrite <- Hsoldef in Hccl;
    try rewrite <- Hsoldef in Hwid.
  assert (Hc' : closed is_local (graph_of_facts F') (consts_used sol))
    by exact (consts_used_closed is_local _ _ sol Hc Hna Hnr Hpos Hgb).
  assert (Hexact' : field_used_exact is_local (graph_of_facts F')
                      (consts_used sol))
    by exact (consts_used_exact is_local _ _ sol Hgb Hexact).
  assert (Hlrs' : local_reads_used is_local (graph_of_facts F')
                    (consts_used sol)).
  { intros base' rel to Has Hacc Hl. exfalso.
    destruct Hlocal as [Hno | Hpf]; [ rewrite Hno in Hl; discriminate Hl | ].
    rewrite HF'def in Hacc.
    destruct (FlambdaReaperFlowProofs.traverse_unit_reads_no_slot flag u
                (erase [] e) Hpf to rel base' Hacc) as [Hw Hfs].
    destruct (Hslots rel Hl) as [(w' & ->) | (fs & ->)];
      [ exact (Hw w' eq_refl) | exact (Hfs fs eq_refl) ]. }
  pose proof (cc_closed_consts _ _ sol _ _ Hpos Hgb Hus Hccl) as Hccl'.
  pose proof (widgets_consts _ _ sol Hgb Hwid) as Hwid'.
  rewrite HF'def in Hc', Hlrs', Hexact', Hccl', Hwid', Hcid.
  pose proof (rebuild_cc_refines_observably_gen flag is_local u (erase [] e)
                (consts_used sol)
                (fun n => mem_node n (derive_cannot_change F T))
                (fun n => w n || node_eqb n Node_all_constants) b
                (fresh_name (names_of e)) Hslots Hfrag Hdc Hrk Hwf Hexits Hc'
                Hlrs' Hexact' (fresh_name_erase e) Hccl' Hwid' Hcid)
    as Hbase.
  rewrite <- HF'def in Hbase. exact Hbase.
Qed.

Theorem least_rebuild_cc_auto_regions_refines :
  least_rebuild_cc_auto_regions_refines_statement.
Proof.
  intros is_local u e T b cert base Hpl Hslots Hlocal Hfrag Hdc Hrk Hwf
    Hexits Hleast Hfe Hee Hcheck.
  exact (least_rebuild_cc_regions_refines Auto is_local u e T b cert base Hpl
           Hslots Hlocal Hfrag Hdc Hrk Hwf Hexits Hleast
           (FlambdaReaperLeastProofs.traverse_widgets_read_only u e) Hfe Hee
           Hcheck).
Qed.

(* ======================================================================== *)
(* Example                                                                  *)
(* ======================================================================== *)

Import FlambdaReaperRegionsSpec.RegionsExamples
  FlambdaReaperRegionsCcSpec.RegionsCcExamples.

Theorem regions_cc_example : regions_cc_example_statement.
Proof.
  assert (Hleast : least_solution (fun _ => false)
                     (traverse_unit Auto interface call_in_region)
                     call_in_region_tables)
    by (apply (FlambdaReaperLeastProofs.check_least_sound _ _ 100);
        vm_compute; reflexivity).
  assert (Hreb : rebuild_unit_cc_from_tables
                   (traverse_unit Auto interface call_in_region)
                   call_in_region_tables
                   (derive_cannot_change
                      (traverse_unit Auto interface call_in_region)
                      call_in_region_tables)
                   (unit_code_deps Auto interface call_in_region) true
                   call_in_region
                 = call_in_region_rebuilt)
    by (vm_compute; reflexivity).
  split; [ vm_compute; reflexivity | ]. split; [ exact Hleast | ].
  do 9 (split; [ vm_compute; reflexivity | ]).
  split; [ exact Hreb | ].
  split; [ vm_compute; reflexivity | ].
  split; [ ex_run | ]. split; [ ex_run | ].
  rewrite <- Hreb.
  apply (least_rebuild_cc_auto_regions_refines (fun _ => false) interface
           call_in_region call_in_region_tables true call_in_region_cert
           base);
    try (vm_compute; reflexivity); try (rewrite Hreb; vm_compute; reflexivity).
  - exact no_local_slots.
  - left. intros f. reflexivity.
  - exact Hleast.
Qed.
