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

(* The refinement of the plain rebuild of
   FlambdaReaperRebuildProofs.v, for any first fresh name that no
   continuation of the unit is, and for any solution that is closed under
   the rules and of which the post-processing holds exactly, given as
   propositions.

   This file needs no review: it proves lemmas for
   FlambdaReaperRegionsProofs.v. Its proof is that of
   FlambdaReaperRebuildProofs.rebuild_with_refines_checked_local with
   those two things left free. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaPrimitiveModel FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec
  FlambdaPrimitiveSpec FlambdaPrimitiveProofs FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaUnusedStaticsSpec FlambdaBindersSpec
  FlambdaWellFormedSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperSolutionSpec
  FlambdaReaperSolutionProofs FlambdaReaperFlowSpec FlambdaReaperFlowProofs
  FlambdaNameAbstractionSpec FlambdaNameAbstractionProofs
  FlambdaFreshQuantifierSpec FlambdaRenameApartSpec
  FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperCallingConventionsSpec
  FlambdaReaperLocalFields FlambdaReaperLocalFieldsSpec
  FlambdaReaperRebuildProofs.
From OxCamlFlambda2 Require FlambdaHeapFreeSpec FlambdaHeapFreeProofs
  FlambdaReaperLocalFieldsProofs FlambdaStaticsInScopeSpec.
Import ListNotations.

Import FlambdaReaperFlowProofs.WithLocal.

Lemma rebuild_with_refines_gen (hb ex : bool)
    (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (sol : solution)
    (preserve_direct_calls : bool) (types : subkinds_rewriting)
    (ff : positive) :
  is_local_slots_only is_local ->
  in_fragment hb e = true ->
  checked ex (unit_codes e) e = true ->
  calls_kinded [] e = true ->
  well_formed_unit u e = true ->
  exits_one_value u e = true ->
  closed is_local (graph_of_facts (traverse_unit flag u e)) sol ->
  local_reads_used is_local (graph_of_facts (traverse_unit flag u e)) sol ->
  field_used_exact is_local (graph_of_facts (traverse_unit flag u e)) sol ->
  among ff (continuations_of e) = false ->
  types_sound_all (graph_of_facts (traverse_unit flag u e)) sol
    (traverse_unit flag u e) types ->
  (∀ t : subkinds_types,
     types = Some t -> st_code_deps t = unit_code_deps flag u e) ->
  conts_table_ok types e ->
  (ex = false -> results_kinded [] [] e = true) ->
  refines_heap_with (exits_rel (if hb then True else False) ex)
    (State e [] [] empty_static_env empty_heap no_reads)
    (State (rebuild_expr (traverse_unit flag u e) sol preserve_direct_calls
              (static_names code_ids_bound e) ff types
              FlambdaReaperRebuild.empty_env e)
       [] [] empty_static_env empty_heap no_reads).
Proof.
  intros Hlocal Hfrag Hcf Hkinded Hwf Hexits Hc Hlrs Hexact Hfresh
    Htypes Htypes_deps0 Htypes_conts Hrk_ex.
  rename preserve_direct_calls into b.
  destruct (traverse_unit_keeps flag u e True hb Hfrag (or_introl I))
    as (noted & deps & cds & He & Hconsts & Himported & Hunit & Hdeps & Hnd
        & Hkeys & Hkeep_alive & Hcds_eq).
  assert (Htypes_deps :
            ∀ t : subkinds_types, types = Some t -> st_code_deps t = cds)
    by (intros t Ht; rewrite (Htypes_deps0 t Ht), Hcds_eq; reflexivity).
  pose proof (traverse_unit_scrutinees ex (unit_codes e) flag u e Hcf)
    as Hscr.
  set (F := traverse_unit flag u e) in *.
  set (G := graph_of_facts F) in *.
  assert (Hclosed : code_closed e = true).
  { unfold well_formed_unit in Hwf.
    destruct (code_closed e); [ reflexivity | ].
    rewrite andb_false_r in Hwf. discriminate Hwf. }
  assert (Hagree0 : kinds_agree [] (env_conts FlambdaReaperRebuild.empty_env)
                      [] []).
  { intros k m Hl. discriminate Hl. }
  assert (Htab0 : conts_tabled types [] [] [] (handlers_params e))
    by exact (conts_table_top types e Htypes_conts).
  pose proof (conts_table_codes types e Htypes_conts) as Hcode_tabled0.
  (* The calls raise to exception handlers, with extra arguments only to
     handlers in scope. *)
  assert (Hraises : raises_kept [] e = true).
  { assert (Hue : unit_exn_handlers_well_formed (unit_exn u) e = true).
    { unfold well_formed_unit in Hwf.
      destruct (unit_exn_handlers_well_formed (unit_exn u) e);
        [ reflexivity | ].
      rewrite andb_false_r in Hwf. discriminate Hwf. }
    unfold unit_exn_handlers_well_formed in Hue.
    apply andb_true_iff in Hue as [Hue Hewf].
    apply andb_true_iff in Hue as [Hways _].
    apply (proj1 raises_kept_of_well_formed e [] Hewf);
      [ intros k Hl; discriminate Hl | ].
    intros k _ Hk. unfold ways_out_well_formed in Hways.
    rewrite forallb_forall in Hways. specialize (Hways k Hk).
    apply andb_true_iff in Hways as [_ Hg]. apply negb_true_iff in Hg.
    exact Hg. }
  assert (Hxagree0 : exn_agree (env_conts FlambdaReaperRebuild.empty_env)
                       []).
  { intros k. exact I. }
  pose proof (proj1 (unit_codes_checked G ex (unit_codes e)
                       ff) e None
                [] [] Hscr Hcf Hraises Hclosed Hkinded Hfresh)
    as Hcode_checks.
  (* What FlambdaReaperFlowProofs.flow_sound derives of the graph. *)
  pose proof (static_names_edges G noted deps cds (unit_kept e) (unit_codes e)
                (static_names symbols_of_binding e) True
                (if hb then True else False)) as [Hnames _].
  destruct (Hnames e None (unit_conts u) He) as [Hsym_names Hcode_names].
  assert (Hcds : ∀ (id : code_id) (cd : code_dep),
            lookup id cds = Some cd -> known_witness_edges G id cd).
  { intros id cd Hl.
    assert (Hid : id ∈ static_names code_ids_of_binding e).
    { apply Hkeys. apply (in_map fst cds (id, cd)).
      exact (lookup_in _ _ _ Hl). }
    destruct (Hcode_names id Hid) as (c & _ & Hcode).
    destruct c as [params mc ret exn metadata body].
    destruct Hcode as (cd' & Hin' & Hw & _).
    rewrite (lookup_nodup_in _ _ _ Hnd Hin') in Hl. injection Hl as <-.
    exact Hw. }
  assert (Hunit_syms : ∀ s : symbol,
            among s (static_names symbols_of_binding e) = true ->
            may_come_from sol (Node_symbol s) (Node_symbol s)).
  { intros s Hs. apply among_In in Hs.
    destruct (Hsym_names s Hs) as (const & Hb).
    destruct const as [ | slot syms decls vs | t fields | | | ];
      simpl in Hb; try contradiction.
    - destruct Hb as (Hslot & Hdep & _). unfold is_bound in Hslot.
      destruct (lookup slot decls) as [id | ] eqn:Hl; [ | discriminate Hslot ].
      exact (called_alloc is_local G sol cds Hc Hcds _ _
               (Hdeps _ (Hdep id eq_refl))).
    - destruct Hb as (_ & His & _).
      exact (alloc_self is_local G sol Hc _ _ _ His
               (as_hs _ _ _ Hc _ (any_source_input _ _ _ Hc _ Hconsts))).
    - exact (boxed_self is_local G sol Hc Hconsts _ _ Hb).
    - exact (any_source_self is_local G sol Hc _
               (alias_consts_any_source is_local G sol Hc Hconsts _ Hb)).
    - destruct elements as [ | el els ].
      + exact (any_source_self is_local G sol Hc _
                 (alias_consts_any_source is_local G sol Hc Hconsts _ Hb)).
      + destruct Hb as (_ & His & _).
        exact (alloc_self is_local G sol Hc _ _ _ His
                 (as_hs _ _ _ Hc _ (any_source_input _ _ _ Hc _ Hconsts))). }
  assert (Hsyms : ∀ s : symbol,
            noted s -> may_come_from sol (Node_symbol s) (Node_symbol s)).
  { intros s Hs.
    destruct (among s (static_names symbols_of_binding e)) eqn:Hunit_s;
      [ exact (Hunit_syms s Hunit_s) | ].
    apply (may_come_from_self is_local G sol _ Hc).
    exact (any_source_input _ _ _ Hc _ (Himported s Hs Hunit_s)). }
  (* The continuations that the unit names are its own. *)
  assert (Hfree : ∀ k : continuation,
            k ∈ free_continuations_of_expr e -> unit_cont u k).
  { assert (Hnames_in : names_in_scope (unit_return u) (unit_exn u) e = true).
    { unfold well_formed_unit in Hwf.
      destruct (names_in_scope (unit_return u) (unit_exn u) e);
        [ reflexivity | ].
      rewrite andb_false_r in Hwf. discriminate Hwf. }
    unfold names_in_scope, names_among in Hnames_in.
    apply andb_true_iff in Hnames_in as [_ Hks].
    intros k Hk.
    pose proof (proj1 (all_among _ _) Hks k (proj2 (among_In _ _) Hk)) as Hin.
    apply among_In in Hin. destruct Hin as [<- | [<- | []]];
      [ left | right ]; reflexivity. }
  assert (Hcc0 : cc_closed G sol (fun _ => true) (fun _ => true))
    by (constructor; intros; reflexivity).
  assert (Hcode_rets :
    ∀ (id : code_id) (params : list variable) (mc : variable)
      (ret exn : continuation) (md : code_metadata) (body : expr),
      ex = false ->
      (id, Code params mc ret exn md body) ∈ unit_codes e ->
      (ret <> exn -> no_raise_to ret body = true)
      ∧ (length params = length (unarize (cm_params_arity md)) ->
         ret_calls_ok
           (rev (with_kinds params (unarize (cm_params_arity md)))
            ++ [(mc, Value)])
           ret (unarize (cm_result_arity md)) body
         = true))
    by (intros id params mc ret exn md body Hex;
        exact (code_rets_of_unit u e Hwf (Hrk_ex Hex) id params mc ret exn md
                 body)).
  (* The rebuild where no code changes its calling convention. *)
  rewrite <- (rebuild_expr_cc_unchanged F sol (fun _ => true) [] b
                (static_names code_ids_bound e) (bound_code_metadata e)
                ff types (fun _ _ => eq_refl)).
  assert (Hslots_read : True -> local_reads_used is_local G sol)
    by (intros _; exact Hlrs).
  eapply (unit_refines is_local G sol noted deps cds (unit_kept e)
            (unit_codes e) (static_names symbols_of_binding e) u
            (if hb then True else False) True F b);
    eassumption || reflexivity || (intros; reflexivity)
    || (intros; apply Htypes)
    || (intros; discriminate)
    || (intros ? ? ? ? ? ? ? ? ? _ Hch;
        destruct (code_change_all_unchanged _ _ _ _ _ _ _ Hch))
    || exact (fun Hf : False => False_ind _ Hf)
    || exact (conj I (conj (ex_intro _ []
                              (conj (conj (fun k H => H) rets_nil_bound)
                                 (exits_bound_unit u e Hexits Hfree)))
                            (conj Hscr (conj Hcf (conj (conj Hfresh
                               (conj (ex_intro _ [] (conj Hkinded Hagree0))
                                  (conj (ex_intro _ []
                                           (conj Hraises Hxagree0))
                                     Htab0)))
                               (fun r K H => rets_nil_any _ r K H)))))).
Qed.

Lemma rebuild_refines_observably_gen (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr)
    (sol : solution) (b : bool) (ff : positive) :
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
  refines_observably (State e [] [] empty_static_env empty_heap no_reads)
    (State (rebuild_expr (traverse_unit flag u e) sol b
              (static_names code_ids_bound e) ff None
              FlambdaReaperRebuild.empty_env e)
       [] [] empty_static_env empty_heap no_reads).
Proof.
  intros Hlocal Hfrag Hdc Hrk Hwf Hexits Hc Hlrs Hexact Hfresh.
  assert (Hkinded : calls_kinded [] e = true).
  { apply (proj1 calls_kinded_of_results e [] [] [] Hrk).
    intros k l Hl. discriminate Hl. }
  exact (proj1 (exits_rel_flow false _ _
                  (rebuild_with_refines_gen false false flag is_local u e sol
                     b None ff Hlocal Hfrag
                     (proj1 (any_kind_arguments_checked (unit_codes e)) e Hdc)
                     Hkinded Hwf Hexits Hc Hlrs Hexact Hfresh
                     (fun _ _ _ _ _ _ _ _ _ _ _ _ => I)
                     (no_types_deps _) (no_types_conts e)
                     (fun _ => Hrk)))).
Qed.
