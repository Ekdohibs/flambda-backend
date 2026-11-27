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

(* Proofs of FlambdaReaperTypesReturnsSpec.v. This file needs no review.

   The body of the code has the edges of the flow with its return
   continuation at the nodes of the results of its code dep
   (FlambdaReaperWrapperProofs.unit_codes_edges): a return aliases the
   node of each result to the argument at its rank
   (FlambdaReaperDeadFramesProofs.applies_to_edges), whose symbols are
   noted ([applies_to_noted]) and so allocations of their own
   ([noted_self], as FlambdaReaperDeadFramesProofs.unit_inv has it). What
   the rebuild relates at the node of the argument it relates at that of
   the result ([rebuilt_value_alias]); the states are those of
   FlambdaReaperTypesCallsProofs.cc_states, and the type is rewritten as
   FlambdaReaperTypesProofs.rewrite_result_type_sound_in rewrites it, at
   the node of the result ([rewrite_result_type_at_sound_in]). *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaWellFormedSpec FlambdaReaperGraphSpec
  FlambdaReaperTraversal FlambdaReaperSolution FlambdaReaperSolutionSpec
  FlambdaReaperSolutionProofs FlambdaReaperFlowSpec FlambdaReaperFlowProofs
  FlambdaReaperLeastSpec FlambdaReaperLocalFields FlambdaReaperRebuild
  FlambdaReaperCallingConventions FlambdaReaperCallingConventionsSpec
  FlambdaReaperTypes FlambdaReaperTypesSpec FlambdaReaperTypesTermSpec
  FlambdaReaperTypesCallsSpec FlambdaReaperDeadFramesSpec
  FlambdaReaperTypesReturnsSpec FlambdaRenameApartSpec.
From OxCamlFlambda2 Require FlambdaReaperRebuildProofs
  FlambdaReaperWrapperProofs FlambdaReaperDeadFramesProofs
  FlambdaNameAbstractionSpec
  FlambdaReaperTypesProofs FlambdaReaperTypesCallsProofs.
From OxCamlFlambda2 Require FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesProofs FlambdaReaperLeastProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaEnvAliases.
From OxCamlFlambda2Types Require FlambdaTypingEnvProofs
  FlambdaTypeGrammarProofs FlambdaTypeMeaningBoolProofs
  FlambdaEnvAliasesProofs FlambdaBindingTime.
Import ListNotations.

Module RP := FlambdaReaperRebuildProofs.
Module TP := FlambdaReaperTypesProofs.
Module CP := FlambdaReaperTypesCallsProofs.
Module DF := FlambdaReaperDeadFramesProofs.

(* ======================================================================== *)
(* The flow at a return                                                     *)
(* ======================================================================== *)

(* What the rebuild relates at a node it relates at every alias of it. *)
Lemma rebuilt_value_alias (is_local : field -> bool) (G : graph)
    (sol : solution) (v : value) (to_ from : node) (v' : value) :
  closed is_local G sol -> s_alias sol to_ from = true ->
  rebuilt_value G sol v from v' -> rebuilt_value G sol v to_ v'.
Proof.
  intros Hc Hal.
  pose proof (may_come_from_alias is_local G sol Hc) as Hm.
  destruct v; cbn [rebuilt_value]; intros H; try exact H.
  - destruct H as [Has Heq].
    split; [ exact (any_source_alias _ _ _ Hc _ _ Hal Has) | exact Heq ].
  - destruct H as (sites & vs' & Heq & Ha & Hrest). exists sites, vs'.
    split; [ exact Heq | split; [ exact (Hm _ _ _ Hal Ha) | exact Hrest ] ].
  - destruct H as (Heq & Hmc & Hu).
    split; [ exact Heq | split; [ exact (Hm _ _ _ Hal Hmc) | ] ].
    intros Hhu. apply Hu. exact (has_usage_alias _ _ _ Hc _ _ Hhu Hal).
  - destruct H as (a & fields' & Ha & Hrest). exists a, fields'.
    split; [ exact (Hm _ _ _ Hal Ha) | exact Hrest ].
  - destruct H as (a & bn & from' & Ha & Hrest). exists a, bn, from'.
    split; [ exact (Hm _ _ _ Hal Ha) | exact Hrest ].
Qed.

(* The symbols of a return are noted. *)
Lemma applies_to_noted (G : graph) (noted : symbol -> Prop)
    (deps : closure_dep -> Prop) (cds : finite_map code_dep)
    (keep : value_slot -> bool) (codes : list (code_id * code))
    (symbols : list symbol) (slots_read heap_prims : Prop)
    (k : continuation) :
  ∀ e : expr, ∀ ctx ce,
    expr_edges G noted deps cds keep codes symbols slots_read heap_prims ctx
      ce e ->
    ∀ ac : apply_cont, ac ∈ applies_to k e ->
      simples_noted noted (ac_args ac).
Proof.
  assert (H :
    (∀ e : expr, ∀ ctx ce,
       expr_edges G noted deps cds keep codes symbols slots_read heap_prims
         ctx ce e ->
       ∀ ac : apply_cont, ac ∈ applies_to k e ->
         simples_noted noted (ac_args ac))
    ∧ (∀ h : continuation_handler, ∀ ctx ce,
         handler_edges G noted deps cds keep codes symbols slots_read
           heap_prims ctx ce h ->
         ∀ ac : apply_cont, ac ∈ handler_applies_to k h ->
           simples_noted noted (ac_args ac)));
    [ | exact (proj1 H) ].
  apply expr_handler_ind;
    cbn [expr_edges handler_edges applies_to handler_applies_to].
  - intros x kind d body IH ctx ce [_ H]. exact (IH ctx ce H).
  - intros bindings body _ IH ctx ce [_ H]. exact (IH ctx ce H).
  - intros k' h body IHh IH ctx ce [Hh H] ac Hac.
    apply in_app_or in Hac as [Hac | Hac]; [ exact (IHh ctx ce Hh ac Hac) | ].
    destruct (Pos.eqb k' k); [ destruct Hac | exact (IH _ _ H ac Hac) ].
  - intros handlers body IHh IH ctx ce [Hhs H] ac Hac.
    destruct (existsb (fun '(k', _) => Pos.eqb k' k) handlers);
      [ destruct Hac | ].
    apply in_app_or in Hac as [Hac | Hac]; [ | exact (IH _ _ H ac Hac) ].
    apply in_flat_map in Hac as ([k' h] & Hkh & Hac).
    exact (IHh k' h Hkh ctx _ (proj1 (every_in _ _ _) Hhs _ Hkh) ac Hac).
  - intros ap ctx ce _ ac [].
  - intros ac ctx ce [_ Hn] ac' Hac.
    destruct (Pos.eqb (ac_continuation ac) k); [ | destruct Hac ].
    destruct Hac as [<- | []]. exact Hn.
  - intros scrutinee arms ctx ce Harms ac Hac.
    apply filter_In in Hac as [Hac _].
    apply in_map_iff in Hac as ([d ac'] & <- & Hin).
    exact (proj2 (proj1 (every_in _ _ _) Harms _ Hin)).
  - intros ctx ce _ ac [].
  - intros params metadata handler IH ctx ce H. exact (IH ctx ce H).
Qed.

(* At a return of code that the unit binds, the node of the result of
   rank [i] is an alias of the argument of that rank, the node of every
   constant has any source, and every symbol of the return is an
   allocation of its own. *)
Lemma return_alias (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (id : code_id) (c : code) (ac : apply_cont) (i : nat) (arg : simple)
    (r : node) :
  in_flow_fragment e = true ->
  closed is_local (graph_of_facts (traverse_unit flag u e))
    (solution_of_tables T) ->
  (id, c) ∈ unit_codes e ->
  ac ∈ applies_to (code_return_continuation c) (code_body c) ->
  nth_error (ac_args ac) i = Some arg ->
  result_node flag u e id i = Some r ->
  g_alias (graph_of_facts (traverse_unit flag u e)) r (node_of_simple arg)
  ∧ s_any_source (solution_of_tables T) Node_all_constants = true
  ∧ ∀ sym : symbol,
      arg = Symbol sym ->
      may_come_from (solution_of_tables T) (Node_symbol sym)
        (Node_symbol sym).
Proof.
  intros Hfrag Hc Hin Hac Harg Hrn.
  destruct (RP.traverse_unit_keeps flag u e True false Hfrag (or_introl I))
    as (noted & deps & cds & He & Hconsts & Himported & _ & Hdeps & Hnd
        & Hkeys & _ & Hcds_eq).
  subst cds.
  set (G := graph_of_facts (traverse_unit flag u e)) in *.
  set (sol := solution_of_tables T) in *.
  set (cds := unit_code_deps flag u e) in *.
  pose proof (proj1 (FlambdaReaperWrapperProofs.unit_codes_edges G noted deps
                       cds (unit_kept e) (unit_codes e)
                       (static_names symbols_of_binding e) True False)
                e None (unit_conts u) He id c Hin) as Hcode.
  pose proof (any_source_input _ _ _ Hc _ Hconsts) as Has.
  unfold result_node in Hrn. fold cds in Hrn.
  destruct (lookup id cds) as [cd | ] eqn:Hcd; [ | discriminate Hrn ].
  destruct c as [params my_closure ret exn md body].
  cbn [code_edges] in Hcode.
  destruct Hcode as (cd' & Hin' & _ & _ & _ & _ & _ & _ & _ & Hne & _ & _
                     & Hbody).
  rewrite (lookup_nodup_in _ _ _ Hnd Hin') in Hcd. injection Hcd as <-.
  cbn [code_return_continuation code_body] in Hac.
  assert (Hk : lookup ret [(exn, [cd_exn cd']); (ret, cd_return cd')]
               = Some (cd_return cd')).
  { rewrite lookup_other; [ apply lookup_head | exact (fun H => Hne H) ]. }
  split; [ | split; [ exact Has | ] ].
  { exact (DF.applies_to_edges G noted deps _ _ _ _ True False ret
             (cd_return cd') body _ _ Hbody Hk ac Hac r arg
             (DF.combine_nth_in _ _ i r arg Hrn Harg)). }
  intros sym ->.
  pose proof (applies_to_noted G noted deps _ _ _ _ True False ret body _ _
                Hbody ac Hac (Symbol sym) (nth_error_In _ _ Harg))
    as Hnoted.
  cbn [simple_noted] in Hnoted.
  pose proof (static_names_edges G noted deps cds (unit_kept e)
                (unit_codes e) (static_names symbols_of_binding e) True False)
    as [Hnames _].
  destruct (Hnames e None (unit_conts u) He) as [Hsym_names Hcode_names].
  assert (Hcds : ∀ (id' : code_id) (cd : code_dep),
            lookup id' cds = Some cd -> known_witness_edges G id' cd).
  { intros id' cd Hl.
    assert (Hid : id' ∈ static_names code_ids_of_binding e).
    { apply Hkeys. apply (in_map fst cds (id', cd)).
      exact (lookup_in _ _ _ Hl). }
    destruct (Hcode_names id' Hid) as (c & _ & Hcode).
    destruct c as [params' mc ret' exn' metadata body'].
    destruct Hcode as (cd'' & Hin'' & Hw & _).
    rewrite (lookup_nodup_in _ _ _ Hnd Hin'') in Hl. injection Hl as <-.
    exact Hw. }
  destruct (among sym (static_names symbols_of_binding e)) eqn:Hunit.
  - apply among_In in Hunit.
    destruct (Hsym_names sym Hunit) as (const & Hb).
    destruct const as [ | slot syms decls vs | t fields | | | ];
      simpl in Hb; try contradiction.
    + destruct Hb as (Hslot & Hdep & _). unfold is_bound in Hslot.
      destruct (lookup slot decls) as [id' | ] eqn:Hl;
        [ | discriminate Hslot ].
      exact (called_alloc is_local G sol cds Hc Hcds _ _
               (Hdeps _ (Hdep id' eq_refl))).
    + destruct Hb as (_ & His & _).
      exact (alloc_self is_local G sol Hc _ _ _ His (as_hs _ _ _ Hc _ Has)).
    + exact (boxed_self is_local G sol Hc Hconsts _ _ Hb).
    + exact (any_source_self is_local G sol Hc _
               (alias_consts_any_source is_local G sol Hc Hconsts _ Hb)).
    + destruct elements as [ | element elements ].
      * exact (any_source_self is_local G sol Hc _
                 (alias_consts_any_source is_local G sol Hc Hconsts _ Hb)).
      * destruct Hb as (_ & His & _).
        exact (alloc_self is_local G sol Hc _ _ _ His
                 (as_hs _ _ _ Hc _ Has)).
  - apply (may_come_from_self is_local G sol _ Hc).
    exact (any_source_input _ _ _ Hc _ (Himported sym Hnoted Hunit)).
Qed.

(* ======================================================================== *)
(* The type at the node of a result                                         *)
(* ======================================================================== *)

(* FlambdaReaperTypesProofs.rewrite_result_type_sound_in at a node, for a
   value rebuilt under its metadata. *)
Lemma rewrite_result_type_at_sound_in (fx : fixes) (loose : bool) :
  TP.fixes_sound fx ->
  ∀ (is_local : field -> bool) (F : facts) (T : tables)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (statics statics' : static_env) (valuation valuation' : env)
    (r : node) (type_var : variable) (v v' : value),
    TP.least_either is_local F T ->
    is_local_slots_only is_local ->
    sets_of_graph (graph_of_facts F) sets ->
    typing_env_satisfied_strictly statics valuation tenv ->
    env_in_fragment_with loose valuation tenv ->
    statics_allocated (solution_of_tables T) statics ->
    statics_rebuilt (graph_of_facts F) (solution_of_tables T) statics
      statics' ->
    (loose = true ->
     static_sets_whole (solution_of_tables T) statics statics') ->
    copies_valued tenv canon valuation valuation' ->
    Typing_env.find tenv (Name_var type_var) None <> None ->
    lookup type_var valuation = Some v ->
    rebuilt_under (graph_of_facts F) (solution_of_tables T)
      (metadata_for_name (plain_context is_local F T sets) r) v v' ->
    type_satisfied statics' valuation' v'
      (rewrite_result_type_at fx (plain_context is_local F T sets) tenv canon
         r type_var).
Proof.
  intros (Hb & Hal & Hr & Hn) is_local F T sets tenv canon statics statics'
    valuation valuation' r type_var v v' Hleast Hlocal Hsets Henv Hfrag
    Halloc Hst Hwhole Hcopies Hfind Hl Hv.
  unfold rewrite_result_type_at, rewrite_concrete_type_of, kind_of_name.
  destruct (Typing_env.find tenv (Name_var type_var) None) as [ty0 | ] eqn:Hf;
    [ | contradiction ].
  unfold concrete_type_of. rewrite (TP.find_with_kind tenv _ ty0 Hf).
  pose proof (TP.find_fragment valuation tenv Hfrag _ _ ty0 Hf) as Hfr.
  pose proof (FlambdaTypingEnvProofs.find_satisfied_strictly statics
                valuation tenv (Name_var type_var) None ty0 v Henv Hf Hl)
    as Hsat.
  apply (proj2 (TP.rewrite_sound fx Hb Hal Hr Hn
                  is_local F T sets tenv canon statics statics' valuation
                  valuation' loose (TP.least_closed_tables is_local F T Hleast)
                  Hleast Hlocal Hsets Henv Hfrag Halloc Hst Hcopies Hwhole
                  (rewrite_fuel tenv ty0))
           _ ty0 v v' Hfr);
    [ | exact Hsat ].
  left. exact Hv.
Qed.

(* At a return, the argument has in the second run a value rebuilt under
   the metadata of the node of the result. *)
Lemma return_rebuilt (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (sets : list set_of_closures) (vars vars' : env) (id : code_id)
    (c : code) (ac : apply_cont) (i : nat) (arg : simple) (r : node)
    (v : value) :
  in_flow_fragment e = true ->
  closed is_local (graph_of_facts (traverse_unit flag u e))
    (solution_of_tables T) ->
  (id, c) ∈ unit_codes e ->
  ac ∈ applies_to (code_return_continuation c) (code_body c) ->
  nth_error (ac_args ac) i = Some arg ->
  result_node flag u e id i = Some r ->
  s_has_usage (solution_of_tables T) r = true ->
  eval_simple vars arg = Some v ->
  (∀ (x : variable) (w : value),
     lookup x vars = Some w ->
     s_has_usage (solution_of_tables T) (Node_variable x) = true ->
     ∃ w' : value,
       lookup x vars' = Some w'
       ∧ rebuilt_value (graph_of_facts (traverse_unit flag u e))
           (solution_of_tables T) w (Node_variable x) w') ->
  ∃ v' : value,
    eval_simple vars' arg = Some v'
    ∧ rebuilt_under (graph_of_facts (traverse_unit flag u e))
        (solution_of_tables T)
        (metadata_for_name
           (plain_context is_local (traverse_unit flag u e) T sets) r)
        v v'.
Proof.
  intros Hfrag Hc Hin Hac Harg Hrn Hhu Hev Hvars.
  destruct (return_alias flag is_local u e T id c ac i arg r Hfrag Hc Hin Hac
              Harg Hrn)
    as (Hal & _ & Hsym).
  pose proof (alias_input _ _ _ Hc _ _ Hal) as Hsal.
  pose proof (TP.metadata_for_name_at
                (plain_context is_local (traverse_unit flag u e) T sets) r)
    as [Hat Hfu].
  destruct arg as [x | sym | c0];
    cbn [eval_simple node_of_simple] in Hev, Hsal |- *.
  - destruct (Hvars x v Hev (has_usage_alias _ _ _ Hc _ _ Hhu Hsal))
      as (v' & Hx' & Hrv).
    exists v'. split; [ exact Hx' | ]. right. exists r.
    split; [ exact (rebuilt_value_alias is_local _ _ v r _ v' Hc Hsal Hrv) | ].
    split; [ exact Hat | intros _; exact Hhu ].
  - injection Hev as <-. exists (Value_symbol sym).
    split; [ reflexivity | ]. right. exists r.
    split; [ | split; [ exact Hat | intros _; exact Hhu ] ].
    cbn [rebuilt_value]. split; [ reflexivity | split ].
    + exact (may_come_from_alias is_local _ _ Hc _ _ _ Hsal
               (Hsym sym eq_refl)).
    + intros Hhu'. exact (has_usage_alias _ _ _ Hc _ _ Hhu' Hsal).
  - injection Hev as <-. exists (Value_const c0).
    split; [ reflexivity | ]. left. exists c0. split; reflexivity.
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Lemma cc_return_gen (local : bool) (fx : fixes)
    (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (cc ccw : node -> bool) (code_deps : finite_map code_dep) (b : bool)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (valuation valuation' : env) (n : nat) (s : state) (id : code_id)
    (i : nat) (arg : simple) (r : node) (type_var : variable) (v : value) :
  TP.fixes_sound fx ->
  unit_premises local flag is_local u e T sets ->
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
  return_premises false flag u e T n s id i arg r tenv canon valuation
    valuation' type_var v ->
  ∃ (m : nat) (t : state) (v' : value),
    run m
      (initial_state
         (rebuild_unit_cc_with None (traverse_unit flag u e)
            (solution_of_tables T) cc code_deps b e))
    = Step t
    ∧ eval_simple (state_vars t) arg = Some v'
    ∧ type_satisfied (state_statics t) valuation' v'
        (rewrite_result_type_at fx
           (plain_context is_local (traverse_unit flag u e) T sets) tenv
           canon r type_var).
Proof.
  intros Hfx (Hlocal & Hleast & Hfrag & Hdc & Hrk & Hwf & Hexits & Honce
              & Hvals & Hsets)
    Hccc Hwid Hcid Hcd
    (Hrun & Hs & (c & ac & Hin & Hse & Hac & Harg) & Hev & Hrn & Hhu & Henv
     & Hfr & Hcopies & Hfind & Hl).
  destruct (CP.least_with_checks local flag is_local u e T Hlocal Hleast)
    as (Hle & Hcheck & Hlrs & Hcheck_exact).
  destruct (CP.cc_states flag is_local u e T cc ccw code_deps b Hlocal Hfrag
              Hdc Hrk Hwf Hexits Hcheck Hlrs Hcheck_exact Honce Hvals Hccc
              Hwid Hcid Hcd n s Hrun Hs)
    as (m & t & Hm & Hst & Halloc & Hvars & _ & _).
  destruct (return_rebuilt flag is_local u e T sets (state_vars s)
              (state_vars t) id c ac i arg r v Hfrag
              (check_closed_sound is_local _ _ Hcheck) Hin Hac Harg Hrn Hhu
              Hev Hvars)
    as (v' & Hv' & Hru).
  exists m, t, v'. split; [ exact Hm | split; [ exact Hv' | ] ].
  apply (rewrite_result_type_at_sound_in fx false Hfx is_local _ T sets tenv
           canon (state_statics s) (state_statics t) valuation valuation' r
           type_var v v');
    try assumption.
  intros H. discriminate H.
Qed.

Lemma plain_return_gen (local : bool) (fx : fixes) :
  TP.fixes_sound fx -> code_result_type_end_to_end_with local fx.
Proof.
  intros Hfx flag is_local u e T b sets tenv canon valuation valuation' n s
    id i arg r type_var v Hprem Hres.
  destruct (cc_return_gen local fx flag is_local u e T (fun _ => true)
              (fun _ => true) [] b sets tenv canon valuation valuation' n s
              id i arg r type_var v Hfx Hprem
              ltac:(constructor; intros; reflexivity)
              ltac:(intros id0 a a' n0 y H; discriminate H)
              ltac:(intros id0 id' w H; discriminate H)
              ltac:(intros id0 H; discriminate H) Hres)
    as (m & t & v' & Hm & Hx & Hsat).
  exists m, t, v'. split; [ | split; [ exact Hx | exact Hsat ] ].
  unfold rebuild_unit_from_tables, rebuild_unit.
  rewrite <- (RP.rebuild_expr_cc_unchanged _ _ (fun _ => true) [] b
                (static_names code_ids_bound e) (bound_code_metadata e)
                (FlambdaNameAbstractionSpec.fresh_name (names_of e)) None
                (fun _ _ => eq_refl)).
  exact Hm.
Qed.

Theorem code_result_type_end_to_end : code_result_type_end_to_end_statement.
Proof. exact (plain_return_gen false all_fixes TP.all_fixes_sound). Qed.

Theorem code_result_type_end_to_end_landed :
  code_result_type_end_to_end_landed_statement.
Proof. exact (plain_return_gen false landed_fixes TP.landed_fixes_sound). Qed.

Theorem code_result_type_end_to_end_local :
  code_result_type_end_to_end_local_statement.
Proof. exact (plain_return_gen true all_fixes TP.all_fixes_sound). Qed.

Theorem code_result_type_end_to_end_local_landed :
  code_result_type_end_to_end_local_landed_statement.
Proof. exact (plain_return_gen true landed_fixes TP.landed_fixes_sound). Qed.

Lemma cc_return_type_gen (local : bool) (fx : fixes) :
  TP.fixes_sound fx -> cc_code_result_type_end_to_end_with local fx.
Proof.
  intros Hfx flag is_local u e T cannot_change b sets tenv canon valuation
    valuation' n s id i arg r type_var v Hprem Hccc Hwid Hres.
  destruct (CP.cc_checks flag u e T cannot_change Hccc Hwid)
    as ((ccw & Hcl) & Hw & Hc).
  exact (cc_return_gen local fx flag is_local u e T _ ccw _ b sets tenv canon
           valuation valuation' n s id i arg r type_var v Hfx Hprem Hcl Hw Hc
           (fun _ _ => eq_refl) Hres).
Qed.

Theorem cc_code_result_type_end_to_end :
  cc_code_result_type_end_to_end_statement.
Proof. exact (cc_return_type_gen false all_fixes TP.all_fixes_sound). Qed.

Theorem cc_code_result_type_end_to_end_landed :
  cc_code_result_type_end_to_end_landed_statement.
Proof.
  exact (cc_return_type_gen false landed_fixes TP.landed_fixes_sound).
Qed.

Theorem cc_code_result_type_end_to_end_local :
  cc_code_result_type_end_to_end_local_statement.
Proof. exact (cc_return_type_gen true all_fixes TP.all_fixes_sound). Qed.

Theorem cc_code_result_type_end_to_end_local_landed :
  cc_code_result_type_end_to_end_local_landed_statement.
Proof.
  exact (cc_return_type_gen true landed_fixes TP.landed_fixes_sound).
Qed.

(* ======================================================================== *)
(* The example                                                              *)
(* ======================================================================== *)

Module CodeResultRuns.

Import CodeResultExample.
Local Open Scope positive_scope.

Ltac in_cases H :=
  repeat match type of H with
         | _ ∨ _ => destruct H as [H | H]
         end;
  try discriminate H; try contradiction H.

(* The state of the run of the unit at the return of [x42]. *)
Definition state_at : state :=
  match run 7 (initial_state UnboxCallsExample.unit_expr) with
  | Step s => s
  | _ => initial_state UnboxCallsExample.unit_expr
  end.

Lemma run_at :
  run 7 (initial_state UnboxCallsExample.unit_expr) = Step state_at.
Proof. vm_compute. reflexivity. Qed.

Lemma lookup_tenv (n : name) (ty : flambda_type)
    (bt : FlambdaBindingTime.Binding_time.With_name_mode.t) :
  Keyed.lookup name_eqb n (Typing_env.names_to_types tenv) = Some (ty, bt) ->
  n = Name_var 91 ∧ (ty = result_type ∨ ty = unknown Value).
Proof.
  intros Hl.
  apply (FlambdaMapProofs.keyed_lookup_in name_eqb
           FlambdaTypeGrammarProofs.name_eqb_eq) in Hl.
  vm_compute in Hl.
  destruct Hl as [[= <- <- _] | [[= <- <- _] | []]]; split; auto.
Qed.

Lemma tenv_satisfied (statics : static_env) :
  typing_env_satisfied_strictly statics valuation tenv.
Proof.
  split; [ vm_compute; reflexivity | split ].
  - intros n ty bt w Hl Hw. right.
    destruct (lookup_tenv n ty bt Hl) as [-> [-> | ->]];
      vm_compute in Hw; injection Hw as <-;
      apply FlambdaTypeMeaningBoolProofs.type_satisfied_b_iff;
      vm_compute; reflexivity.
  - change (Typing_env.aliases tenv) with Aliases.empty.
    apply FlambdaEnvAliasesProofs.empty_respected.
Qed.

End CodeResultRuns.

Lemma code_result_example_gen (fx : fixes) :
  rewrite_result_type_at fx
    (plain_context FlambdaReaperUnboxing.Examples.no_local
       UnboxCallsExample.facts UnboxCallsExample.tables
       UnboxCallsExample.sets)
    CodeResultExample.tenv UnboxCallsExample.canon (Node_made 3) 91%positive
  = CodeResultExample.rewritten ->
  code_result_example_with fx.
Proof.
  intros Hrw.
  destruct FlambdaReaperUnboxingRefinesProofs.unboxed_call_example
    as (Hslots & Hfrag & Hstatic & Hrk & Hwf & Hone & _).
  split.
  { split; [ exact Hslots | split ].
    { split; [ apply (FlambdaReaperLeastProofs.check_least_sound _ _ 100);
               vm_compute; reflexivity | ].
      left. intros f. reflexivity. }
    split; [ exact Hfrag | split ].
    { exact (RP.static_constants_any_kind_arguments _ _ Hstatic). }
    split; [ exact Hrk | split; [ exact Hwf | split; [ exact Hone | ] ] ].
    split; [ vm_compute; reflexivity | split; [ vm_compute; reflexivity | ] ].
    exact CP.UnboxCallsRuns.sets_ok. }
  split; [ vm_compute; reflexivity | ].
  split.
  { exists CodeResultRuns.state_at.
    split; [ exact CodeResultRuns.run_at | ].
    split; [ vm_compute; discriminate | ].
    split.
    { exists FlambdaReaperUnboxingRefinesSpec.CallExample.callee,
        {| ac_continuation := 3%positive; ac_args := [Var 42%positive];
           ac_trap_action := None |}.
      split; [ vm_compute; left; reflexivity | ].
      split; [ vm_compute; reflexivity | ].
      split; [ vm_compute; left; reflexivity | reflexivity ]. }
    split; [ vm_compute; reflexivity | ].
    split; [ vm_compute; reflexivity | ].
    split; [ vm_compute; reflexivity | ].
    split; [ apply CodeResultRuns.tenv_satisfied | ].
    split.
    { intros n ty bt Hl.
      destruct (CodeResultRuns.lookup_tenv n ty bt Hl) as [_ [-> | ->]];
        [ | apply TP.unknown_fragment ].
      split; [ vm_compute; reflexivity | ].
      intros s k w Hin Hw. vm_compute in Hin.
      CodeResultRuns.in_cases Hin; injection Hin as <- <-;
        cbn in Hw; injection Hw as <-; left; eexists; reflexivity. }
    split.
    { intros s x k m Hcan. unfold value_of_name.
      exact (FlambdaTypingEnvProofs.get_canonical_simple_satisfied
               _ _ _ _ _ _ _ _ (CodeResultRuns.tenv_satisfied
                                  (state_statics CodeResultRuns.state_at))
               Hcan). }
    split; [ vm_compute; discriminate | reflexivity ]. }
  split; [ exact Hrw | ].
  exists
    (match run 7
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

Theorem code_result_example : code_result_example_statement.
Proof. apply code_result_example_gen. vm_compute. reflexivity. Qed.

Theorem code_result_example_landed : code_result_example_landed_statement.
Proof. apply code_result_example_gen. vm_compute. reflexivity. Qed.
