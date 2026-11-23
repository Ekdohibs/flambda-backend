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
(* The proofs of FlambdaReaperUnboxingRefinesOver2Spec.v. Nothing here
   needs review.

   FlambdaReaperUnboxingRefinesCallsProofs.v relates the two rebuilt units
   on the fragment [calls_unboxing_reach_over_nr], given [flow_run] and
   [over_run] of the first ([rebuild_unbox_modes_refines_cc_over2]): the
   code that a step of the first run over-applies is in [ovr]. Here
   [over_run] is proved of the first rebuilt unit, with [over_reach] of its
   own analysis: FlambdaReaperFlowProofs.step_over_used gives, of each
   state of the run, that the code it over-applies is [over_used] (its last
   witness of unknown arity, read on past by the call, has a first result
   with a use), which [over_reach] reads off the facts and the tables
   ([over_used_reach]). *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaSemanticsProofs
  FlambdaWellFormedSpec FlambdaBindersSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaUnusedStaticsSpec FlambdaReaperUnboxingRefinesSpec
  FlambdaReaperUnboxingRefinesProofs FlambdaReaperLeast
  FlambdaReaperLocalFields FlambdaReaperLocalFieldsSpec
  FlambdaReaperSolutionSpec FlambdaReaperSolutionProofs
  FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaReaperUnboxingRefinesCallsSimProofs
  FlambdaReaperUnboxingRefinesOverSpec
  FlambdaReaperUnboxingRefinesOver1Spec
  FlambdaReaperUnboxingRefinesReachSpec FlambdaReaperUnboxingRefinesReach2Spec
  FlambdaReaperUnboxingRefinesOver2Spec
  FlambdaReaperUnboxingRefinesCallsProofs FlambdaReaperPartialGroupsProofs.
From OxCamlFlambda2 Require FlambdaReaperFlowProofs
  FlambdaReaperLocalFieldsProofs
  FlambdaReaperCallingConventionsSpec FlambdaReaperCallingConventionsProofs.
Import ListNotations.

Import FlambdaReaperFlowProofs.

Theorem calls_unboxing_over1_in_over2 :
  calls_unboxing_over1_in_over2_statement.
Proof.
  intros nr F T cannot_change code_deps dominated to_unbox F1 T1 e H.
  unfold calls_unboxing_over2_nr.
  apply calls_unboxing_in_over2_reach.
  apply (calls_unboxing_in_flow2_reach nr F T cannot_change code_deps
           dominated to_unbox (fun _ _ => true)); [ reflexivity | ].
  exact (calls_unboxing_in_flow_reach nr F T cannot_change code_deps
           dominated to_unbox (fun _ _ => true) e H).
Qed.

(* ======================================================================== *)
(* The code that the first rebuilt unit over-applies                        *)
(* ======================================================================== *)

(* [over_reach] is [over_used], read off the facts and the tables. *)
Lemma over_used_reach (F1 : facts) (T1 : tables) (id : code_id) :
  over_used (graph_of_facts F1) (solution_of_tables T1) id ->
  over_reach F1 T1 id = true.
Proof.
  intros (W & R & Hcode & Hret & Hused).
  unfold over_reach. apply existsb_exists.
  exists (W, Field_code_id_of_call_witness, Node_code_id id).
  split; [ exact Hcode | ].
  rewrite field_eqb_refl, node_eqb_refl. cbn [andb].
  apply existsb_exists. exists (W, Field_return_of_call (Normal 0), R).
  split; [ exact Hret | ].
  rewrite node_eqb_refl, field_eqb_refl. exact Hused.
Qed.

(* What a step over-applies, for a wider [OV]. *)
Lemma cont_over_ok_mono (OV OV' : code_id -> Prop) (S : static_env) :
  (∀ id : code_id, OV id -> OV' id) ->
  ∀ (c : cont_value) (vs : list value),
    cont_over_ok OV S c vs -> cont_over_ok OV' S c vs.
Proof.
  intros Himp c.
  induction c as [h vars conts recursive | k | c IH extra
                 | remaining remaining_arity return_arity return_to exn_to
                     IHret _]
    using cont_value_over_ind;
    intros vs; cbn [cont_over_ok]; [ exact (fun H => H) | exact (fun H => H)
                                   | apply IH | ].
  destruct vs as [ | v [ | v' vs]]; [ exact (fun H => H) | |
                                      exact (fun H => H) ].
  destruct (shape_of_call S v Indirect_unknown_arity remaining
              remaining_arity return_arity)
    as [[cl c vs' aa | cl c vs' rest aa ra | p] | ];
    cbn [shape_over_ok]; [ exact (fun H => H) | | apply IHret
                         | exact (fun H => H) ].
  intros H id slots Hclo. exact (Himp id (H id slots Hclo)).
Qed.

Lemma step_over_ok_mono (OV OV' : code_id -> Prop) (s : state) :
  (∀ id : code_id, OV id -> OV' id) ->
  step_over_ok OV s -> step_over_ok OV' s.
Proof.
  intros Himp. unfold step_over_ok.
  destruct (state_expr s); try exact (fun H => H).
  - intros H callee vs Hcallee Hvs.
    specialize (H callee vs Hcallee Hvs).
    destruct (shape_of_call _ _ _ _ _ _)
      as [[cl c vs' aa | cl c vs' rest aa ra | p] | ];
      cbn [shape_over_ok] in H |- *; [ exact I | | | exact I ].
    + intros id slots Hclo. exact (Himp id (H id slots Hclo)).
    + exact (cont_over_ok_mono OV OV' _ Himp _ _ H).
  - intros H vs Hvs. exact (cont_over_ok_mono OV OV' _ Himp _ _ (H vs Hvs)).
Qed.

(* Every state of the run of a unit of the fragment, with a solution closed
   for its facts, over-applies only code that is [over_used]. *)
Lemma run_over_used (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (n : nat) :
  is_local_slots_only is_local ->
  in_heap_fragment e = true ->
  closed is_local (graph_of_facts (traverse_unit flag u e))
    (solution_of_tables T) ->
  local_reads_used is_local (graph_of_facts (traverse_unit flag u e))
    (solution_of_tables T) ->
  match run n (State e [] [] empty_static_env empty_heap no_reads) with
  | Step s =>
    step_over_ok
      (over_used (graph_of_facts (traverse_unit flag u e))
         (solution_of_tables T))
      s
  | _ => True
  end.
Proof.
  intros Hlocal Hfrag Hc Hlrs.
  destruct (traverse_unit_facts_heap flag u e True Hfrag (or_introl I))
    as (noted & deps & cds & He0 & Hconsts & Himported & _ & Hdeps & Hnd
        & Hkeys).
  set (G := graph_of_facts (traverse_unit flag u e)) in *.
  set (sol := solution_of_tables T) in *.
  pose proof (static_names_edges G noted deps cds (unit_kept e)
                (unit_codes e) (static_names symbols_of_binding e)
                True True) as [Hnames _].
  destruct (Hnames e None (unit_conts u) He0) as [Hsym_names Hcode_names].
  assert (Hcds : ∀ (id : code_id) (cd : code_dep),
            lookup id cds = Some cd -> known_witness_edges G id cd).
  { intros id cd Hl.
    assert (Hid : id ∈ static_names code_ids_of_binding e).
    { apply Hkeys. apply (in_map fst (cds) (id, cd)).
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
    - destruct elements as [ | element elements ].
      + exact (any_source_self is_local G sol Hc _
                 (alias_consts_any_source is_local G sol Hc Hconsts _ Hb)).
      + destruct Hb as (_ & His & _).
        exact (alloc_self is_local G sol Hc _ _ _ His
                 (as_hs _ _ _ Hc _ (any_source_input _ _ _ Hc _ Hconsts))).
  }
  assert (Hsyms : ∀ s : symbol,
            noted s -> may_come_from sol (Node_symbol s) (Node_symbol s)).
  { intros s Hs.
    destruct (among s (static_names symbols_of_binding e)) eqn:Hunit;
      [ exact (Hunit_syms s Hunit) | ].
    apply (may_come_from_self is_local G sol _ Hc).
    exact (any_source_input _ _ _ Hc _ (Himported s Hs Hunit)). }
  destruct (located_exists n (State e [] [] empty_static_env empty_heap
                                no_reads))
    as (located & Hloc).
  pose proof (WithLocal.run_flow_invariant is_local G sol noted deps cds
                (unit_kept e) (unit_codes e) (static_names symbols_of_binding e)
                u True located True Hc Hlocal Hconsts Hsyms Hunit_syms Hdeps
                Hnd Hcds (fun _ => Hlrs) n _ Hloc
                (WithLocal.initial_flow_invariant G sol noted deps cds
                   (unit_kept e) (unit_codes e)
                   (static_names symbols_of_binding e) u True located True e
                   He0)
                ltac:(apply initial_heap_ok))
    as Hrun.
  destruct (run n (State e [] [] empty_static_env empty_heap no_reads))
    as [s | k vs hp rd | ]; [ | exact I | exact I ].
  destruct Hrun as [Hinv _].
  exact (WithLocal.step_over_used is_local G sol noted deps cds (unit_kept e)
           (unit_codes e) (static_names symbols_of_binding e) u True located
           True Hc Hlocal Hconsts Hsyms Hnd Hcds s Hinv).
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Theorem rebuild_unbox_over2_refines_cc :
  rebuild_unbox_over2_refines_cc_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc nr
    Hslots Hfrag Hclosed Hfrag1 Hclosed1 Hlrs1 Hcalls Hcheck.
  unfold first_rebuild in *.
  set (e1 := rebuild_unit_cc_from_tables (traverse_unit flag u e) T
               cannot_change (unit_code_deps flag u e) pdc e) in *.
  set (F1 := traverse_unit flag u e1) in *.
  apply (rebuild_unbox_modes_refines_cc_over2 flag is_local u e T
           cannot_change dominated to_unbox pdc nr (fun _ _ _ => true)
           (over_reach F1 T1) Hslots Hfrag Hclosed Hcalls Hcheck).
  - intros n s _ ap f v id slots _ _ _.
    split; intros _; reflexivity.
  - intros n s Hrun.
    pose proof (check_closed_sound is_local _ _ Hclosed1) as Hc1.
    pose proof (run_over_used flag is_local u e1 T1 n Hslots Hfrag1 Hc1
                  (FlambdaReaperLocalFieldsProofs.check_local_reads_used_sound
                     is_local _ _ Hlrs1))
      as Hover.
    fold F1 in Hover. unfold e1 in Hover. rewrite Hrun in Hover.
    apply (step_over_ok_mono _ _ s (over_used_reach F1 T1)). exact Hover.
Qed.

Theorem rebuild_unbox_over2_refines_observably_local :
  rebuild_unbox_over2_refines_observably_local_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc nr
    Hslots Hfrag Hany Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid Hfrag1
    Hclosed1 Hlrs1 Hcalls Hcheck.
  exact (refines_observably_refines _ _ _
           (FlambdaReaperCallingConventionsProofs
            .rebuild_cc_refines_observably_local
              flag is_local u e T cannot_change pdc Hslots Hfrag Hany
              Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid)
           (rebuild_unbox_over2_refines_cc flag is_local u e T T1
              cannot_change dominated to_unbox pdc nr Hslots Hfrag Hclosed
              Hfrag1 Hclosed1 Hlrs1 Hcalls Hcheck)).
Qed.

Theorem rebuild_unbox_over2_refines_observably :
  rebuild_unbox_over2_refines_observably_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc nr
    Hslots Hproj Hfrag Hany Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hfrag1
    Hclosed1 Hlrs1 Hcalls Hcheck.
  exact (rebuild_unbox_over2_refines_observably_local flag is_local u e T T1
           cannot_change dominated to_unbox pdc nr Hslots Hfrag Hany Hrk Hwf
           Hone Hclosed
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hslots Hproj)
           Hexact Hccc Hwid Hfrag1 Hclosed1 Hlrs1 Hcalls Hcheck).
Qed.

(* ======================================================================== *)
(* Example                                                                  *)
(* ======================================================================== *)

Module UE := FlambdaReaperUnboxing.Examples.

Theorem two_groups_example : two_groups_example_statement.
Proof.
  assert (Hcalls : calls_unboxing_over2 TwoGroupsExample.facts
                     TwoGroupsExample.tables TwoGroupsExample.cannot_change
                     TwoGroupsExample.code_deps [] []
                     TwoGroupsExample.facts1 TwoGroupsExample.tables1
                     TwoGroupsExample.unit_expr
                   = true)
    by (vm_compute; reflexivity).
  assert (Hfixed :
    rebuild_unit_unbox_fixed TwoGroupsExample.facts TwoGroupsExample.tables
      TwoGroupsExample.cannot_change TwoGroupsExample.code_deps [] [] true
      TwoGroupsExample.unit_expr
    = TwoGroupsExample.rebuilt)
    by (vm_compute; reflexivity).
  assert (Hprem :
    in_flow_fragment TwoGroupsExample.unit_expr = true
    ∧ any_kind_arguments TwoGroupsExample.unit_expr = true
    ∧ results_kinded [] [] TwoGroupsExample.unit_expr = true
    ∧ well_formed_unit UE.interface TwoGroupsExample.unit_expr = true
    ∧ exits_one_value UE.interface TwoGroupsExample.unit_expr = true
    ∧ check_closed UE.no_local TwoGroupsExample.facts TwoGroupsExample.tables
      = true
    ∧ check_field_used_exact UE.no_local TwoGroupsExample.facts
        TwoGroupsExample.tables
      = true
    ∧ check_cc_closed TwoGroupsExample.facts TwoGroupsExample.tables
        TwoGroupsExample.cannot_change
      = true
    ∧ FlambdaReaperCallingConventionsSpec.check_apply_widgets
        TwoGroupsExample.facts TwoGroupsExample.tables
      = true
    ∧ in_heap_fragment TwoGroupsExample.first = true
    ∧ check_closed UE.no_local TwoGroupsExample.facts1
        TwoGroupsExample.tables1
      = true
    ∧ check_local_reads_used UE.no_local TwoGroupsExample.facts1
        TwoGroupsExample.tables1
      = true
    ∧ check_unboxed_reads TwoGroupsExample.facts TwoGroupsExample.tables []
        []
      = true)
    by (vm_compute; repeat split).
  destruct Hprem as (Hfrag & Hany & Hrk & Hwf & Hone & Hclosed & Hexact
                     & Hccc & Hwid & Hfrag1 & Hclosed1 & Hlrs1 & Hreads).
  assert (Hslots : is_local_slots_only UE.no_local)
    by (intros f Hf; discriminate Hf).
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ exact Hcalls | ].
  split; [ exact Hfixed | ].
  rewrite <- Hfixed.
  (* The check as the statement has it, which the elaborator is slow to
     convert [Hcalls] to. *)
  assert (Hcalls' :
    calls_unboxing_over2_nr (fun _ => false)
      (traverse_unit Auto UE.interface TwoGroupsExample.unit_expr)
      TwoGroupsExample.tables TwoGroupsExample.cannot_change
      (unit_code_deps Auto UE.interface TwoGroupsExample.unit_expr) [] []
      (traverse_unit Auto UE.interface
         (first_rebuild Auto UE.interface TwoGroupsExample.unit_expr
            TwoGroupsExample.tables TwoGroupsExample.cannot_change true))
      TwoGroupsExample.tables1 TwoGroupsExample.unit_expr
    = true)
    by (vm_compute; reflexivity).
  exact (rebuild_unbox_over2_refines_observably Auto UE.no_local
           UE.interface TwoGroupsExample.unit_expr TwoGroupsExample.tables
           TwoGroupsExample.tables1 TwoGroupsExample.cannot_change [] [] true
           (fun _ => false) Hslots (or_introl (fun _ => eq_refl)) Hfrag Hany
           Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hfrag1 Hclosed1 Hlrs1 Hcalls'
           Hreads).
Qed.
