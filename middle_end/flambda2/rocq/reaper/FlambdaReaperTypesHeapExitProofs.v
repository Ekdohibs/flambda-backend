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


(* The exit of the simulation of the rebuild with the primitives of the
   heap: the static environments of the two runs where the unit exits are
   related ([statics_rel] of FlambdaReaperRebuildProofs.v, the locations
   of the first run related to those of the second by [kept_beta]), as
   [rebuild_exit_statics_local] has them without the heap. For
   FlambdaReaperTypesHeapProofs.v. This file needs no review. *)

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
  FlambdaReaperLocalFieldsProofs.
Import ListNotations.

Import FlambdaReaperFlowProofs.WithLocal.

(* The static environments of the two runs where the unit exits, with the
   heap, for some code deps and arguments of the rebuild that the
   relation reads only of code: the locations of the first run have the
   allocations [L]. *)
Definition exit_statics_rel_heap (G : graph) (sol : solution)
    (keep : value_slot -> bool) (L : location -> option node)
    (statics statics' : static_env) : Prop :=
  ∃ (cds : finite_map code_dep) (F : facts) (b : bool)
    (ids : list code_id) (first_fresh : positive) (cc : node -> bool)
    (code_deps : finite_map code_dep)
    (code_metadatas : finite_map code_metadata) (types : subkinds_rewriting),
    statics_rel G sol cds keep True F b ids first_fresh false True cc
      code_deps code_metadatas types L statics statics'.

(* Where a unit of the fragment with the heap exits, whose closures are
   made where the graph has only the facts of their sets, the rebuilt unit
   exits through the same continuation, the exits observed equal up to a
   correspondence of their locations, from a state whose static
   environment is related to that of the unit ([exit_statics_rel_heap]).
   With local fields, the tables passing the check of the fix. *)
Lemma rebuild_exit_statics_heap_local (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) :
  is_local_slots_only is_local ->
  in_heap_fragment e = true ->
  any_kind_arguments e = true ->
  results_kinded [] [] e = true ->
  well_formed_unit u e = true ->
  exits_one_value u e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  check_local_reads_used is_local (traverse_unit flag u e) T = true ->
  check_field_used_exact is_local (traverse_unit flag u e) T = true ->
  closures_exact (graph_of_facts (traverse_unit flag u e)) (unit_kept e) e ->
  ∀ (n : nat) (s : state) (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record),
    run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
    step s = Exit k vs hp rd ->
    ∃ (L : location -> option node) (m : nat) (t : state)
      (vs' : list value) (hp' : heap) (rd' : read_record),
      run m
        (State
           (rebuild_unit_from_tables (traverse_unit flag u e) T
              preserve_direct_calls e)
           [] [] empty_static_env empty_heap no_reads)
      = Step t
      ∧ step t = Exit k vs' hp' rd'
      ∧ exits_observed_equal vs hp vs' hp'
      ∧ exit_statics_rel_heap (graph_of_facts (traverse_unit flag u e))
          (solution_of_tables T) (unit_kept e) L (state_statics s)
          (state_statics t).
Proof.
  intros Hlocal Hfrag Hdc Hrk Hwf Hexits Hcheck Hlrs Hcheck_exact Hxe.
  assert (Hkinded : calls_kinded [] e = true).
  { apply (proj1 calls_kinded_of_results e [] [] [] Hrk).
    intros k' l Hl. discriminate Hl. }
  pose proof (proj1 (any_kind_arguments_checked (unit_codes e)) e Hdc)
    as Hcf.
  assert (Hrk_ex : false = false -> results_kinded [] [] e = true)
    by (intros _; exact Hrk).
  rename preserve_direct_calls into b.
  pose proof (check_closed_sound is_local _ _ Hcheck) as Hc.
  pose proof (check_field_used_exact_sound is_local _ _ Hcheck_exact)
    as Hexact.
  destruct (traverse_unit_keeps flag u e True true Hfrag (or_introl I))
    as (noted & deps & cds & He & Hconsts & Himported & Hunit & Hdeps & Hnd
        & Hkeys & Hkeep_alive & _).
  pose proof (traverse_unit_scrutinees false (unit_codes e) flag u e Hcf)
    as Hscr.
  set (F := traverse_unit flag u e) in *.
  set (G := graph_of_facts F) in *.
  set (sol := solution_of_tables T) in *.
  assert (Hclosed : code_closed e = true).
  { unfold well_formed_unit in Hwf.
    destruct (code_closed e); [ reflexivity | ].
    rewrite andb_false_r in Hwf. discriminate Hwf. }
  assert (Hfresh : among (fresh_name (names_of e)) (continuations_of e)
                   = false).
  { apply not_true_iff_false. intros E. apply among_In in E.
    pose proof (fresh_name_fresh (names_of e)) as Hff.
    apply not_true_iff_false in Hff. apply Hff. apply among_In.
    unfold names_of. apply in_or_app. right. exact E. }
  assert (Hagree0 : kinds_agree [] (env_conts FlambdaReaperRebuild.empty_env)
                      [] []).
  { intros k m Hl. discriminate Hl. }
  assert (Htab0 : conts_tabled None [] [] [] (handlers_params e))
    by (intros t Ht; discriminate Ht).
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
  pose proof (proj1 (unit_codes_checked G false (unit_codes e)
                       (fresh_name (names_of e))) e None
                [] [] Hscr Hcf Hraises Hclosed Hkinded Hfresh)
    as Hcode_checks.
  (* What FlambdaReaperFlowProofs.flow_sound derives of the graph. *)
  pose proof (static_names_edges G noted deps cds (unit_kept e) (unit_codes e)
                (static_names symbols_of_binding e) True
                (if true then True else False)) as [Hnames _].
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
      false = false ->
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
  intros n s k vs hp rd Hrun Hstep.
  unfold rebuild_unit_from_tables, rebuild_unit.
  (* The rebuild where no code changes its calling convention. *)
  rewrite <- (rebuild_expr_cc_unchanged F sol (fun _ => true) [] b
                (static_names code_ids_bound e) (bound_code_metadata e)
                (fresh_name (names_of e)) None (fun _ _ => eq_refl)).
  assert (Hx : ∃ (L : location -> option node) (m : nat) (t : state)
                 (vs' : list value) (hp' : heap) (rd' : read_record),
             run m
               (State
                  (rebuild_expr_cc F sol (fun _ => true) [] b
                     (static_names code_ids_bound e) (bound_code_metadata e)
                     (fresh_name (names_of e)) None
                     FlambdaReaperRebuild.empty_env
                     e)
                  [] [] empty_static_env empty_heap no_reads)
             = Step t
             ∧ step t = Exit k vs' hp' rd'
             ∧ exits_rel (if true then True else False) false vs hp vs' hp'
             ∧ related G sol noted deps cds (unit_kept e) (unit_codes e)
                 (static_names symbols_of_binding e) u
                 (if true then True else False) True F b
                 (static_names code_ids_bound e) (fresh_name (names_of e))
                 false True (fun _ => true) [] (bound_code_metadata e) None L s
                 t).
  { assert (Hslots_read : True -> local_reads_used is_local G sol)
      by (intros _;
          exact (FlambdaReaperLocalFieldsProofs.check_local_reads_used_sound
                   is_local _ _ Hlrs)).
    eapply (unit_exit_related is_local G sol noted deps cds (unit_kept e)
              (unit_codes e) (static_names symbols_of_binding e) u
              (if true then True else False) True F b);
      eassumption || reflexivity || (intros; reflexivity)
      || (intros; discriminate)
      || (intros ? ? ? ? ? ? ? ? ? _ Hch;
          destruct (code_change_all_unchanged _ _ _ _ _ _ _ Hch))
      || exact (fun _ : True => Hxe)
      || exact (fun H : False => H)
      || exact (conj I (conj (ex_intro _ []
                                (conj (conj (fun k H => H) rets_nil_bound)
                                   (exits_bound_unit u e Hexits Hfree)))
                              (conj Hscr (conj Hcf (conj (conj Hfresh
                                 (conj (ex_intro _ [] (conj Hkinded Hagree0))
                                    (conj (ex_intro _ []
                                             (conj Hraises Hxagree0))
                                       Htab0)))
                                 (fun r K H => rets_nil_any _ r K H)))))). }
  destruct Hx as (L & m & t & vs' & hp' & rd' & Hm & Ht & HE & Hrel).
  exists L, m, t, vs', hp', rd'.
  split; [ exact Hm | split; [ exact Ht | split; [ exact (proj1 HE) | ] ] ].
  destruct Hrel as [_ (ctx & en & ce & rets & _ & _ & _ & _ & Hst & _)].
  exists cds, F, b, (static_names code_ids_bound e),
    (fresh_name (names_of e)), (fun _ => true), [], (bound_code_metadata e),
    None.
  exact Hst.
Qed.
