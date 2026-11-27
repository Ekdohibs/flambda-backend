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

(* Proofs of FlambdaReaperTypesParamsSpec.v. This file needs no review.

   The check read back ([read_by_only], with
   FlambdaReaperIndirectDeadFramesProofs.alias_only). A least solution of
   the rules is one of the rules with the fix of local fields
   (FlambdaReaperLeastProofs.least_local_of_least, under the premise of
   the rules before the fix), whose tuples the rules with the fix derive.
   Inverting those derivations: the parameter, into which only the alias
   from the node of the code dep puts anything, has its sources, any
   source and source through that alias only ([alias_into_param],
   [sources_into_param], ...); the node, which only that alias reads, has
   its usages, any usage and use through it only ([alias_from_node],
   [usages_from_node], ...). So what the value of the parameter may come
   from may come to the node ([may_come_from_param]), and the metadata of
   the node describes it (FlambdaReaperTypesProofs.metadata_for_name_at).
   The states are those of FlambdaReaperTypesCallsProofs.cc_states_used,
   and the type is rewritten at the node by
   FlambdaReaperTypesReturnsProofs.rewrite_result_type_at_sound_in. The
   example is computed. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaWellFormedSpec FlambdaReaperGraphSpec
  FlambdaReaperTraversal FlambdaReaperSolution FlambdaReaperSolutionSpec
  FlambdaReaperSolutionProofs FlambdaReaperFlowSpec FlambdaReaperLeastSpec
  FlambdaReaperLocalFields FlambdaReaperRebuild
  FlambdaReaperCallingConventions FlambdaReaperCallingConventionsSpec
  FlambdaReaperTypes FlambdaReaperTypesSpec FlambdaReaperTypesTermSpec
  FlambdaReaperTypesCallsSpec FlambdaReaperTypesReturnsSpec
  FlambdaReaperTypesUsedValuesSpec FlambdaReaperTypesCallsUsedSpec
  FlambdaReaperIndirectDeadFramesSpec FlambdaReaperTypesParamsSpec
  FlambdaRenameApartSpec.
From OxCamlFlambda2 Require FlambdaReaperRebuildProofs
  FlambdaReaperFlowProofs FlambdaReaperLeastProofs
  FlambdaReaperDeadFramesProofs FlambdaReaperIndirectDeadFramesProofs
  FlambdaNameAbstractionSpec FlambdaReaperTypesProofs
  FlambdaReaperTypesCallsProofs FlambdaReaperTypesReturnsProofs
  FlambdaReaperUnboxing FlambdaReaperUnboxingRefinesSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaEnvAliases.
From OxCamlFlambda2Types Require FlambdaTypingEnvProofs
  FlambdaTypeMeaningBoolProofs.
Import ListNotations.

Module RP := FlambdaReaperRebuildProofs.
Module LP := FlambdaReaperLeastProofs.
Module TP := FlambdaReaperTypesProofs.
Module CP := FlambdaReaperTypesCallsProofs.
Module RT := FlambdaReaperTypesReturnsProofs.
Module DF := FlambdaReaperDeadFramesProofs.
Module IDP := FlambdaReaperIndirectDeadFramesProofs.

Ltac unfold_rels :=
  cbn [graph_of_facts solution_of_tables holds g_alias g_use g_accessor
       g_constructor g_argument g_parameter g_propagate
       g_alias_if_any_source g_any_usage g_any_source g_any_source_if_used
       g_zero_alloc_source s_alias s_any_usage s_any_source s_has_usage
       s_has_source s_usages s_sources s_reading_field s_escaping_field
       s_field_used] in *.

(* ======================================================================== *)
(* The check, read back                                                     *)
(* ======================================================================== *)

(* What [only_read_by F r x] says, fact by fact. *)
Record read_by_only (F : facts) (r x : node) : Prop := {
  rb_alias : ∀ to_, (to_, r) ∈ f_alias F -> to_ = x;
  rb_propagate : ∀ c to_, (c, to_, r) ∈ f_propagate F -> to_ = x;
  rb_if_any_source : ∀ c to_, (c, to_, r) ∉ f_alias_if_any_source F;
  rb_constructor : ∀ b f, (b, f, r) ∉ f_constructor F;
  rb_argument_from : ∀ i b, (r, i, b) ∉ f_argument F;
  rb_argument_base : ∀ y i, (y, i, r) ∉ f_argument F;
  rb_accessor : ∀ to_ f, (to_, f, r) ∉ f_accessor F;
  rb_use : ∀ to_, (to_, r) ∉ f_use F;
  rb_any_usage : r ∉ f_any_usage F
}.

Lemma only_read_by_read_by_only (F : facts) (r x : node) :
  only_read_by F r x = true -> read_by_only F r x.
Proof.
  unfold only_read_by. intros H.
  apply andb_true_iff in H as [H Hany].
  apply andb_true_iff in H as [H Huse].
  apply andb_true_iff in H as [H Hacc].
  apply andb_true_iff in H as [H Harg].
  apply andb_true_iff in H as [H Hcons].
  apply andb_true_iff in H as [H Hias].
  apply andb_true_iff in H as [Halias Hprop].
  rewrite forallb_forall in Halias, Hprop, Hias, Hcons, Harg, Hacc, Huse.
  apply negb_true_iff in Hany.
  constructor.
  - intros to_ Hin. specialize (Halias _ Hin). cbn in Halias.
    rewrite node_eqb_refl in Halias. cbn in Halias.
    apply node_eqb_eq in Halias. exact Halias.
  - intros c to_ Hin. specialize (Hprop _ Hin). cbn in Hprop.
    rewrite node_eqb_refl in Hprop. cbn in Hprop.
    apply node_eqb_eq in Hprop. exact Hprop.
  - intros c to_ Hin. specialize (Hias _ Hin). cbn in Hias.
    rewrite node_eqb_refl in Hias. discriminate.
  - intros b f Hin. specialize (Hcons _ Hin). cbn in Hcons.
    rewrite node_eqb_refl in Hcons. discriminate.
  - intros i b Hin. specialize (Harg _ Hin). cbn in Harg.
    rewrite node_eqb_refl in Harg. discriminate.
  - intros y i Hin. specialize (Harg _ Hin). cbn in Harg.
    rewrite node_eqb_refl, orb_true_r in Harg. discriminate.
  - intros to_ f Hin. specialize (Hacc _ Hin). cbn in Hacc.
    rewrite node_eqb_refl in Hacc. discriminate.
  - intros to_ Hin. specialize (Huse _ Hin). cbn in Huse.
    rewrite node_eqb_refl in Huse. discriminate.
  - intros Hin. apply mem_node_In in Hin. congruence.
Qed.

(* The check at the parameter [p] of rank [i] of the code [id] of the
   unit, and at the node [r] of that rank of its code dep. *)
Lemma params_aliased_at (F : facts) (cds : finite_map code_dep) (e : expr)
    (id : code_id) (c : code) (cd : code_dep) (i : nat) (p : variable)
    (r : node) :
  params_aliased F cds e = true -> (id, c) ∈ unit_codes e ->
  lookup id cds = Some cd -> nth_error (code_params c) i = Some p ->
  nth_error (cd_params cd) i = Some r ->
  IDP.alias_only F (Node_variable p) r ∧ read_by_only F r (Node_variable p).
Proof.
  intros H Hin Hcd Hp Hr. unfold params_aliased in H.
  rewrite forallb_forall in H. specialize (H _ Hin). cbn beta iota in H.
  rewrite Hcd, forallb_forall in H.
  specialize (H _ (DF.combine_nth_in _ _ i p r Hp Hr)). cbn beta iota in H.
  apply andb_true_iff in H as [H1 H2].
  exact (conj (IDP.only_alias_into_alias_only _ _ _ H1)
           (only_read_by_read_by_only _ _ _ H2)).
Qed.

(* The least solution, of the rules with the fix of local fields. *)
Lemma least_local_with (local : bool) (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables) :
  is_local_slots_only is_local ->
  least_with local is_local (traverse_unit flag u e) T e ->
  least_solution_local is_local (traverse_unit flag u e) T.
Proof.
  intros Hlocal Hleast. destruct local; cbn [least_with] in Hleast;
    [ exact Hleast | ].
  destruct Hleast as [Hleast Hslots].
  exact (LP.least_local_of_least is_local _ T Hleast
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hlocal Hslots)).
Qed.

(* ======================================================================== *)
(* The parameter and the node of the code dep                               *)
(* ======================================================================== *)

Section Param.

Variable is_local : field -> bool.
Variable F : facts.
Variable T : tables.
Hypothesis Hleast : least_solution_local is_local F T.

Local Abbreviation G := (graph_of_facts F).
Local Abbreviation S := (solution_of_tables T).
Local Abbreviation D := (derivable_local is_local G S).

(* The parameter [x] and the node [r] of the code dep. *)
Variables x r : node.
Hypothesis HA : IDP.alias_only F x r.
Hypothesis HB : read_by_only F r x.

Ltac kill :=
  exfalso;
  first [ eapply (IDP.ao_if_any_source _ _ _ HA); eassumption
        | eapply (IDP.ao_accessor _ _ _ HA); eassumption
        | eapply (IDP.ao_parameter_to _ _ _ HA); eassumption
        | eapply (IDP.ao_parameter_base _ _ _ HA); eassumption
        | eapply (IDP.ao_constructor _ _ _ HA); eassumption
        | eapply (IDP.ao_any_source _ _ _ HA); eassumption
        | eapply (IDP.ao_zero_alloc _ _ _ HA); eassumption
        | eapply (IDP.ao_if_used _ _ _ HA); eassumption
        | eapply (IDP.ao_use _ _ _ HA); eassumption
        | eapply (rb_if_any_source _ _ _ HB); eassumption
        | eapply (rb_constructor _ _ _ HB); eassumption
        | eapply (rb_argument_from _ _ _ HB); eassumption
        | eapply (rb_argument_base _ _ _ HB); eassumption
        | eapply (rb_accessor _ _ _ HB); eassumption
        | eapply (rb_use _ _ _ HB); eassumption
        | eapply (rb_any_usage _ _ _ HB); eassumption ].

(* Only the alias from [r] puts something into [x]. *)
Lemma alias_into_param (from : node) : D (Atom_alias x from) -> from = r.
Proof.
  intros Hd. inversion Hd; subst; unfold_rels; try solve [ kill ].
  - eapply (IDP.ao_alias _ _ _ HA). eassumption.
  - eapply (IDP.ao_propagate _ _ _ HA). eassumption.
  - match goal with
    | H : derivable_local _ _ _ (Atom_reading_field _ _) |- _ =>
        inversion H; subst; unfold_rels
    end.
    kill.
Qed.

Lemma any_source_into_param : D (Atom_any_source x) -> D (Atom_any_source r).
Proof.
  intros Hd. inversion Hd; subst; unfold_rels; try solve [ kill ].
  match goal with
  | H : derivable_local _ _ _ (Atom_alias x ?f) |- _ =>
      rewrite <- (alias_into_param _ H)
  end.
  assumption.
Qed.

Lemma sources_into_param (a : node) :
  D (Atom_sources x a) -> D (Atom_sources r a).
Proof.
  intros Hd. inversion Hd; subst; unfold_rels; try solve [ kill ].
  match goal with
  | H : derivable_local _ _ _ (Atom_alias x ?f) |- _ =>
      rewrite <- (alias_into_param _ H)
  end.
  assumption.
Qed.

Lemma has_source_into_param : D (Atom_has_source x) -> D (Atom_has_source r).
Proof.
  intros Hd. inversion Hd; subst; unfold_rels; try solve [ kill ].
  - apply byl_has_source_any. apply any_source_into_param. assumption.
  - match goal with
    | H : derivable_local _ _ _ (Atom_alias x ?f) |- _ =>
        rewrite <- (alias_into_param _ H)
    end.
    assumption.
Qed.

(* Only the alias into [x] reads [r]. *)
Lemma alias_from_node (to_ : node) : D (Atom_alias to_ r) -> to_ = x.
Proof.
  intros Hd. inversion Hd; subst; unfold_rels; try solve [ kill ].
  - eapply (rb_alias _ _ _ HB). eassumption.
  - eapply (rb_propagate _ _ _ HB). eassumption.
  - match goal with
    | H : derivable_local _ _ _ (Atom_escaping_field _ _) |- _ =>
        inversion H; subst; unfold_rels
    end.
    kill.
Qed.

Lemma any_usage_from_node : D (Atom_any_usage r) -> D (Atom_any_usage x).
Proof.
  intros Hd. inversion Hd; subst; unfold_rels; try solve [ kill ].
  match goal with
  | H : derivable_local _ _ _ (Atom_alias ?t r) |- _ =>
      rewrite <- (alias_from_node _ H)
  end.
  assumption.
Qed.

Lemma has_usage_from_node : D (Atom_has_usage r) -> D (Atom_has_usage x).
Proof.
  intros Hd. inversion Hd; subst; unfold_rels; try solve [ kill ].
  - apply byl_has_usage_any. apply any_usage_from_node. assumption.
  - match goal with
    | H : derivable_local _ _ _ (Atom_alias ?t r) |- _ =>
        rewrite <- (alias_from_node _ H)
    end.
    assumption.
Qed.

Lemma usages_from_node (w : node) :
  D (Atom_usages r w) -> D (Atom_usages x w).
Proof.
  intros Hd. inversion Hd; subst; unfold_rels; try solve [ kill ].
  match goal with
  | H : derivable_local _ _ _ (Atom_alias ?t r) |- _ =>
      rewrite <- (alias_from_node _ H)
  end.
  assumption.
Qed.

Local Abbreviation LH := (LP.least_local_holds is_local F T Hleast).
Local Abbreviation LD := (LP.least_local_derivable is_local F T Hleast).

(* The node [r] has a use where [x] has one. *)
Lemma param_has_usage :
  s_has_usage S r = true -> s_has_usage S x = true.
Proof.
  intros H. exact (LH _ (has_usage_from_node (LD (Atom_has_usage r) H))).
Qed.

(* What the value of [x] may come from may come to [r]. *)
Lemma may_come_from_param (a : node) :
  may_come_from S a x -> may_come_from S a r.
Proof.
  intros (Hhs & Hsrc & Has & Hau & Hus).
  assert (Has_r : s_any_source S x = true -> s_any_source S r = true)
    by (intros H; exact (LH _ (any_source_into_param
                                  (LD (Atom_any_source x) H)))).
  split; [ exact (LH _ (has_source_into_param (LD (Atom_has_source x) Hhs)))
         | split; [ | split; [ | split ] ] ].
  - destruct Hsrc as [H | H]; [ left; exact (Has_r H) | right ].
    exact (LH _ (sources_into_param a (LD (Atom_sources x a) H))).
  - intros H. exact (Has_r (Has H)).
  - intros H. apply Hau.
    exact (LH _ (any_usage_from_node (LD (Atom_any_usage r) H))).
  - intros Hhu w Hw. apply Hus; [ exact (param_has_usage Hhu) | ].
    exact (LH _ (usages_from_node w (LD (Atom_usages r w) Hw))).
Qed.

(* A value rebuilt at [x] is rebuilt under the metadata of [r]. *)
Lemma param_rebuilt_under (sets : list set_of_closures) (v v' : value) :
  s_has_usage S r = true -> rebuilt_value G S v x v' ->
  rebuilt_under G S (metadata_for_name (plain_context is_local F T sets) r)
    v v'.
Proof.
  intros Hhu Hrv.
  pose proof (TP.metadata_for_name_at (plain_context is_local F T sets) r)
    as [Hat _].
  right. exists x. split; [ exact Hrv | split ].
  - intros a Ha Hhsa. exact (Hat a (may_come_from_param a Ha) Hhsa).
  - intros _. exact (param_has_usage Hhu).
Qed.

End Param.

(* What the statements read of the parameter and the node of the code
   dep. *)
Lemma param_node_facts (local : bool) (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (sets : list set_of_closures) (id : code_id) (i : nat) (p : variable)
    (r : node) :
  is_local_slots_only is_local ->
  least_with local is_local (traverse_unit flag u e) T e ->
  params_aliased (traverse_unit flag u e) (unit_code_deps flag u e) e
  = true ->
  at_param e id i p ->
  param_node flag u e id i = Some r ->
  s_has_usage (solution_of_tables T) r = true ->
  s_has_usage (solution_of_tables T) (Node_variable p) = true
  ∧ ∀ v v' : value,
      rebuilt_value (graph_of_facts (traverse_unit flag u e))
        (solution_of_tables T) v (Node_variable p) v' ->
      rebuilt_under (graph_of_facts (traverse_unit flag u e))
        (solution_of_tables T)
        (metadata_for_name
           (plain_context is_local (traverse_unit flag u e) T sets) r)
        v v'.
Proof.
  intros Hlocal Hleast Hcheck (c & Hin & Hp) Hrn Hhu.
  pose proof (least_local_with local flag is_local u e T Hlocal Hleast)
    as Hll.
  unfold param_node in Hrn.
  destruct (lookup id (unit_code_deps flag u e)) as [cd | ] eqn:Hcd;
    [ | discriminate Hrn ].
  destruct (params_aliased_at _ _ _ id c cd i p r Hcheck Hin Hcd Hp Hrn)
    as [HA HB].
  split; [ eapply param_has_usage; eassumption | ].
  intros v v' Hrv. eapply param_rebuilt_under; eassumption.
Qed.

Theorem param_value_at_node : param_value_at_node_statement.
Proof.
  intros local flag is_local u e T sets id i p r v v' Hlocal Hleast Hcheck
    Hat Hrn Hhu Hrv.
  destruct (param_node_facts local flag is_local u e T sets id i p r Hlocal
              Hleast Hcheck Hat Hrn Hhu)
    as [Hhp Hru].
  exact (conj Hhp (Hru v v' Hrv)).
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Lemma cc_param_gen (local : bool) (fx : fixes)
    (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (cc ccw : node -> bool) (code_deps : finite_map code_dep) (b : bool)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (valuation valuation' : env) (n : nat) (s : state) (id : code_id)
    (i : nat) (p : variable) (r : node) (type_var : variable) (v : value) :
  TP.fixes_sound fx ->
  used_premises local flag is_local u e T sets ->
  cc_closed (graph_of_facts (traverse_unit flag u e)) (solution_of_tables T)
    ccw cc ->
  (∀ (id : code_id) (a a' n y : node),
     cc (Node_code_id id) = false ->
     g_accessor (graph_of_facts (traverse_unit flag u e)) a
       (Field_call_witness Known_arity_code_pointer) n ->
     g_accessor (graph_of_facts (traverse_unit flag u e)) a'
       (Field_call_witness Known_arity_code_pointer) n ->
     (s_any_source (solution_of_tables T) a' = true ->
      s_any_source (solution_of_tables T) a = true)
     ∧ (s_sources (solution_of_tables T) a' y = true ->
        s_sources (solution_of_tables T) a y = true)) ->
  (∀ (id id' : code_id) (w : node),
     cc (Node_code_id id) = false ->
     g_constructor (graph_of_facts (traverse_unit flag u e)) w
       Field_code_id_of_call_witness (Node_code_id id) ->
     g_constructor (graph_of_facts (traverse_unit flag u e)) w
       Field_code_id_of_call_witness (Node_code_id id') ->
     id' = id) ->
  (∀ id : code_id,
     cc (Node_code_id id) = false ->
     lookup id code_deps = lookup id (unit_code_deps flag u e)) ->
  params_aliased (traverse_unit flag u e) (unit_code_deps flag u e) e
  = true ->
  param_premises flag u e T n s id i p r tenv canon valuation valuation'
    type_var v ->
  ∃ (m : nat) (t : state) (v' : value),
    run m
      (initial_state
         (rebuild_unit_cc_with None (traverse_unit flag u e)
            (solution_of_tables T) cc code_deps b e))
    = Step t
    ∧ lookup p (state_vars t) = Some v'
    ∧ type_satisfied (state_statics t) valuation' v'
        (rewrite_result_type_at fx
           (plain_context is_local (traverse_unit flag u e) T sets) tenv
           canon r type_var).
Proof.
  intros Hfx (Hlocal & Hleast & Hfrag & Hdc & Hrk & Hwf & Hexits & Honce
              & Hvals & Hsets)
    Hccc Hwid Hcid Hcd Hcheck
    (Hrun & Hs & Hat & Hx & Hrn & Hhu & Henv & Hfr & Hcopies & Hfind & Hl).
  destruct (CP.least_with_checks local flag is_local u e T Hlocal Hleast)
    as (Hle & Hcheck0 & Hlrs & Hcheck_exact).
  destruct (param_node_facts local flag is_local u e T sets id i p r Hlocal
              Hleast Hcheck Hat Hrn Hhu)
    as [Hhp Hru].
  destruct (CP.cc_states_used flag is_local u e T cc ccw code_deps b Hlocal
              Hfrag Hdc Hrk Hwf Hexits Hcheck0 Hlrs Hcheck_exact Honce Hvals
              Hccc Hwid Hcid Hcd n s Hrun Hs)
    as (m & t & Hm & Hst & Halloc & Hvars & _ & _).
  destruct (Hvars p v Hx Hhp) as (v' & Hx' & Hrv).
  exists m, t, v'. split; [ exact Hm | split; [ exact Hx' | ] ].
  apply (RT.rewrite_result_type_at_sound_in fx false Hfx is_local _ T sets
           tenv canon (state_statics s) (state_statics t) valuation
           valuation' r type_var v v');
    try assumption.
  - intros H. discriminate H.
  - exact (Hru v v' Hrv).
Qed.

(* The plain rebuild is the rebuild with changes of calling convention
   where no code changes. *)
Lemma plain_param_gen (local : bool) (fx : fixes) :
  TP.fixes_sound fx -> param_type_end_to_end_with local fx.
Proof.
  intros Hfx flag is_local u e T b sets tenv canon valuation valuation' n s
    id i p r type_var v Hprem Hcheck Hres.
  destruct (cc_param_gen local fx flag is_local u e T (fun _ => true)
              (fun _ => true) [] b sets tenv canon valuation valuation' n s
              id i p r type_var v Hfx Hprem
              ltac:(constructor; intros; reflexivity)
              ltac:(intros id0 a a' n0 y H; discriminate H)
              ltac:(intros id0 id' w H; discriminate H)
              ltac:(intros id0 H; discriminate H) Hcheck Hres)
    as (m & t & v' & Hm & Hx & Hsat).
  exists m, t, v'. split; [ | split; [ exact Hx | exact Hsat ] ].
  unfold rebuild_unit_from_tables, rebuild_unit.
  rewrite <- (RP.rebuild_expr_cc_unchanged _ _ (fun _ => true) [] b
                (static_names code_ids_bound e) (bound_code_metadata e)
                (FlambdaNameAbstractionSpec.fresh_name (names_of e)) None
                (fun _ _ => eq_refl)).
  exact Hm.
Qed.

Theorem param_type_end_to_end : param_type_end_to_end_statement.
Proof. exact (plain_param_gen false all_fixes TP.all_fixes_sound). Qed.

Theorem param_type_end_to_end_landed :
  param_type_end_to_end_landed_statement.
Proof. exact (plain_param_gen false landed_fixes TP.landed_fixes_sound). Qed.

Theorem param_type_end_to_end_local : param_type_end_to_end_local_statement.
Proof. exact (plain_param_gen true all_fixes TP.all_fixes_sound). Qed.

Theorem param_type_end_to_end_local_landed :
  param_type_end_to_end_local_landed_statement.
Proof. exact (plain_param_gen true landed_fixes TP.landed_fixes_sound). Qed.

Lemma cc_param_type_gen (local : bool) (fx : fixes) :
  TP.fixes_sound fx -> cc_param_type_end_to_end_with local fx.
Proof.
  intros Hfx flag is_local u e T cannot_change b sets tenv canon valuation
    valuation' n s id i p r type_var v Hprem Hccc Hwid Hcheck Hres.
  destruct (CP.cc_checks flag u e T cannot_change Hccc Hwid)
    as ((ccw & Hcl) & Hw & Hc).
  exact (cc_param_gen local fx flag is_local u e T _ ccw _ b sets tenv canon
           valuation valuation' n s id i p r type_var v Hfx Hprem Hcl Hw Hc
           (fun _ _ => eq_refl) Hcheck Hres).
Qed.

Theorem cc_param_type_end_to_end : cc_param_type_end_to_end_statement.
Proof. exact (cc_param_type_gen false all_fixes TP.all_fixes_sound). Qed.

Theorem cc_param_type_end_to_end_landed :
  cc_param_type_end_to_end_landed_statement.
Proof.
  exact (cc_param_type_gen false landed_fixes TP.landed_fixes_sound).
Qed.

Theorem cc_param_type_end_to_end_local :
  cc_param_type_end_to_end_local_statement.
Proof. exact (cc_param_type_gen true all_fixes TP.all_fixes_sound). Qed.

Theorem cc_param_type_end_to_end_local_landed :
  cc_param_type_end_to_end_local_landed_statement.
Proof.
  exact (cc_param_type_gen true landed_fixes TP.landed_fixes_sound).
Qed.

(* ======================================================================== *)
(* The example                                                              *)
(* ======================================================================== *)

Lemma param_example_gen (fx : fixes) :
  rewrite_result_type_at fx
    (plain_context FlambdaReaperUnboxing.Examples.no_local
       UnboxCallsExample.facts UnboxCallsExample.tables
       UnboxCallsExample.sets)
    UnboxCallsExample.tenv UnboxCallsExample.canon (Node_made 6) 90%positive
  = UnboxCallsExample.param_type ->
  param_example_with fx.
Proof.
  intros Hrw.
  split; [ exact (CP.unit_premises_used _ _ _ _ _ _ _
                    (proj1 CP.unbox_calls_example)) | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split.
  { exists CP.UnboxCallsRuns.state_at.
    split; [ exact CP.UnboxCallsRuns.run_at | ].
    split; [ vm_compute; discriminate | ].
    split.
    { exists FlambdaReaperUnboxingRefinesSpec.CallExample.callee.
      split; [ vm_compute; left; reflexivity | reflexivity ]. }
    split; [ vm_compute; reflexivity | ].
    split; [ vm_compute; reflexivity | ].
    split; [ vm_compute; reflexivity | ].
    split; [ apply CP.UnboxCallsRuns.tenv_satisfied | ].
    split.
    { intros n ty bt Hl.
      destruct (CP.UnboxCallsRuns.lookup_tenv n ty bt Hl) as [_ [-> | ->]];
        [ | apply TP.unknown_fragment ].
      split; [ vm_compute; reflexivity | ].
      intros s k w Hin Hw. vm_compute in Hin.
      CP.UnboxCallsRuns.in_cases Hin. injection Hin as <- <-.
      cbn in Hw. injection Hw as <-. left. eexists. reflexivity. }
    split.
    { intros s x k m Hcan. unfold value_of_name.
      exact (FlambdaTypingEnvProofs.get_canonical_simple_satisfied
               _ _ _ _ _ _ _ _ (CP.UnboxCallsRuns.tenv_satisfied
                                  (state_statics CP.UnboxCallsRuns.state_at))
               Hcan). }
    split; [ vm_compute; discriminate | reflexivity ]. }
  split; [ exact Hrw | ].
  exists
    (match run 5
             (initial_state
                (rebuild_unit_from_tables UnboxCallsExample.facts
                   UnboxCallsExample.tables true UnboxCallsExample.unit_expr))
     with
     | Step t => t
     | _ => initial_state UnboxCallsExample.unit_expr
     end).
  split; [ vm_compute; reflexivity | split; [ vm_compute; reflexivity | ] ].
  apply FlambdaTypeMeaningBoolProofs.type_satisfied_b_iff.
  vm_compute. reflexivity.
Qed.

Theorem param_example : param_example_statement.
Proof. apply param_example_gen. vm_compute. reflexivity. Qed.

Theorem param_example_landed : param_example_landed_statement.
Proof. apply param_example_gen. vm_compute. reflexivity. Qed.
