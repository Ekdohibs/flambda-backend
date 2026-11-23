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

(* The proofs of FlambdaReaperClosuresTablesSlotsSpec.v. Nothing here
   needs review.

   [vinv]: the constructors at a value slot of the variable of a [Let] of
   a closure of [e0] are of its value slots. The steps of the traversal
   keep it: those other than the sets of closures add no constructor at
   a slot ([CE.ns_ok]); a [Let] of a closure adds the facts of its group
   ([CE.record_set_group]), at allocations that are its variable, the
   variables of the projections of its closures and made nodes
   ([CE.closures_of_set_spec]); a static set at its symbol. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperLeastSpec FlambdaReaperRebuild
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaReaperUnboxingRefinesClosuresSpec FlambdaReaperClosuresTablesSpec
  FlambdaReaperClosuresTablesFieldsSpec FlambdaReaperClosuresTablesSlotsSpec.
From OxCamlFlambda2 Require FlambdaReaperFlowProofs FlambdaReaperTypesProofs
  FlambdaReaperTypesTraversalProofs FlambdaReaperClosuresExactProofs
  FlambdaReaperRebuildProofs FlambdaReaperClosuresTablesProofs
  FlambdaReaperClosuresTablesFieldsProofs.
From OxCamlFlambda2 Require Import FlambdaReaperSolutionProofs.
Import ListNotations.

Module TP := FlambdaReaperTypesProofs.
Module TTP := FlambdaReaperTypesTraversalProofs.
Module FP := FlambdaReaperFlowProofs.
Module CE := FlambdaReaperClosuresExactProofs.
Module RP := FlambdaReaperRebuildProofs.
Module CTP := FlambdaReaperClosuresTablesProofs.
Module FTP := FlambdaReaperClosuresTablesFieldsProofs.

Ltac split_bools :=
  repeat match goal with
  | H : (_ && _) = true |- _ => apply andb_true_iff in H as [? ?]
  | |- (_ && _) = true => apply andb_true_iff; split
  end.

(* ======================================================================== *)
(* The value slots of the traversal                                         *)
(* ======================================================================== *)

Section Slots.

Variable e0 : expr.
Hypothesis Hnl : NoDup (map fst (let_bindings e0)).

Definition vinv (a : acc) : Prop :=
  ∀ (x : variable) (slot : function_slot)
    (decls : list (function_slot * code_id))
    (vs : list (value_slot * simple)) (w : value_slot) (b : node),
    (x, Closure_of_set slot decls vs) ∈ let_bindings e0 ->
    Fact_constructor (Node_variable x) (Field_value_slot w) b ∈ acc_facts a ->
    ∃ s : simple, lookup w vs = Some s ∧ b = node_of_simple s.

Lemma vinv_ns (a a' : acc) : vinv a -> CE.ns_ok a a' -> vinv a'.
Proof.
  intros Hv (_ & H2 & _) x slot decls vs w b Hx Hin.
  exact (Hv x slot decls vs w b Hx (H2 _ Hin eq_refl)).
Qed.

Lemma vinv_fold {A : Type} (f : acc -> A -> acc) (l : list A) :
  (∀ (a : acc) (y : A), y ∈ l -> vinv a -> vinv (f a y)) ->
  ∀ a : acc, vinv a -> vinv (List.fold_left f l a).
Proof.
  induction l as [ | y l IH]; intros Hf a Ha; [ exact Ha | ].
  cbn [List.fold_left]. apply IH.
  - intros a' y' Hy'. apply Hf. right. exact Hy'.
  - apply Hf; [ left; reflexivity | exact Ha ].
Qed.

(* A value slot of the group [vs]. *)
Lemma vslot_in (env : traverse_env) (vs : list (value_slot * simple))
    (w : value_slot) (b : node) :
  (w, b)
    ∈ map (fun '(w, s) => (w, node_of_simple s))
        (filter (fun '(w, _) => should_keep_value_slot env w) (compact vs)) ->
  ∃ s : simple, lookup w vs = Some s ∧ b = node_of_simple s.
Proof.
  intros Hin. apply in_map_iff in Hin as ([w' s] & Heq & Hin).
  injection Heq as -> <-. apply filter_In in Hin as [Hin _].
  exists s. split; [ exact (proj1 (in_compact w s vs) Hin) | reflexivity ].
Qed.

Lemma vinv_closure (env : traverse_env) (y : variable)
    (slot' : function_slot) (decls' : list (function_slot * code_id))
    (vs' : list (value_slot * simple)) (a : acc) :
  env_set_closures env = set_closures e0 ->
  (y, Closure_of_set slot' decls' vs') ∈ let_bindings e0 ->
  vinv a -> vinv (traverse_closure_of_set env y slot' decls' vs' a).
Proof.
  intros Hsc Hy Hv. unfold traverse_closure_of_set.
  destruct (among slot' (map fst (compact decls'))); [ | exact Hv ].
  destruct (closures_of_set y slot' (set_closures_of env y) (compact decls')
              a)
    as [cl a1] eqn:Ecl.
  destruct (CE.closures_of_set_spec y slot' _ _ a cl a1 Ecl)
    as (Hf1 & _ & _ & Hcl).
  match goal with
  | |- vinv (record_set_of_closures_deps env ?cs ?fs _ _) =>
    destruct (CE.record_set_group env (Node_variable y) cs fs vs' a1)
      as (L & HL & HLf)
  end.
  intros x slot decls vs w b Hx Hin. rewrite HL in Hin.
  apply in_app_or in Hin as [Hin | Hin];
    [ | rewrite Hf1 in Hin; exact (Hv x slot decls vs w b Hx Hin) ].
  apply HLf in Hin.
  destruct Hin as (N & HN & [(f & b' & E & _) | (w' & b' & E & Hw)]);
    [ discriminate E | ].
  injection E as EN Ew Eb. subst N w' b'.
  cbn [CE.sg_bases] in HN. rewrite map_map in HN.
  apply in_map_iff in HN as ([[f N] id] & EN & Hf). cbn in EN. subst N.
  destruct (proj2 (Hcl f _ id Hf))
    as [Ex | [(y' & Hy' & Ey') | (k & Ek & _)]];
    [ | | discriminate Ek ].
  - injection Ex as ->.
    pose proof (CE.nodup_same _ _ _ _ Hnl Hx Hy) as E.
    injection E as _ _ <-. exact (vslot_in env vs w b Hw).
  - injection Ey' as <-. exfalso.
    unfold set_closures_of in Hy'. rewrite Hsc in Hy'.
    destruct (lookup y (set_closures e0)) as [projs | ] eqn:Hp;
      [ | discriminate Hy' ].
    destruct (proj1 CE.set_closures_projs e0 y projs (lookup_in _ _ _ Hp) f x
                (lookup_in _ _ _ Hy'))
      as (s & Hs).
    pose proof (CE.nodup_same _ _ _ _ Hnl Hx Hs) as E. discriminate E.
Qed.

Lemma vinv_static_binding (env : traverse_env) (bd : static_binding)
    (a : acc) :
  CE.binding_no_old_sets bd -> vinv a ->
  vinv (traverse_static_binding env bd a).
Proof.
  intros Hno Hv.
  destruct bd as [id c | id | s const];
    [ apply (vinv_ns _ _ Hv); apply CE.ns_traverse_static_binding; intros;
      discriminate
    | apply (vinv_ns _ _ Hv); apply CE.ns_traverse_static_binding; intros;
      discriminate | ].
  destruct const as [id vs | slot cs decls vs | tag fields | contents
                    | bytes | els];
    try (apply (vinv_ns _ _ Hv); apply CE.ns_traverse_static_binding;
         intros; discriminate).
  - destruct (Hno id vs eq_refl).
  - cbn [traverse_static_binding].
    match goal with
    | |- vinv (record_set_of_closures_deps env ?cs ?fs _ _) =>
      destruct (CE.record_set_group env (Node_symbol s) cs fs vs a)
        as (L & HL & HLf)
    end.
    intros x slot' decls' vs' w b Hx Hin. rewrite HL in Hin.
    apply in_app_or in Hin as [Hin | Hin];
      [ | exact (Hv x slot' decls' vs' w b Hx Hin) ].
    apply HLf in Hin.
    destruct Hin as (N & HN & [(f & b' & E & _) | (w' & b' & E & _)]);
      [ discriminate E | ].
    injection E as EN _ _. subst N.
    cbn in HN. destruct HN as [E | []]. discriminate E.
Qed.

Definition P_v (e : expr) : Prop :=
  ∀ env : traverse_env,
    env_set_closures env = set_closures e0 ->
    (∀ p, p ∈ let_bindings e -> p ∈ let_bindings e0) ->
    CE.no_old_sets e ->
    ∀ a : acc, vinv a -> vinv (traverse env e a).

Lemma add_rec_conts_sc (env : traverse_env)
    (handlers : list (continuation * continuation_handler)) :
  env_set_closures (add_rec_conts env handlers) = env_set_closures env.
Proof.
  unfold add_rec_conts. induction handlers as [ | [k h] hs IH];
    [ reflexivity | exact IH ].
Qed.

Lemma traverse_vinv :
  (∀ e : expr, P_v e)
  ∧ (∀ h : continuation_handler,
       match h with Handler _ _ handler => P_v handler end).
Proof.
  apply FP.expr_handler_ind; unfold P_v.
  - intros x k d body IH env Hsc Hsub [Hd Hold] a Hv.
    cbn [traverse]. apply IH; [ exact Hsc | | exact Hold | ].
    { intros p Hp. apply Hsub. right. exact Hp. }
    destruct d as [s | p args | id vs | slot decls vs | w c | pf w c
                  | mf mt c];
      try (apply (vinv_ns _ _ Hv); apply CE.ns_traverse_named; intros;
           discriminate).
    + destruct (Hd id vs eq_refl).
    + apply vinv_closure; [ exact Hsc | | exact Hv ].
      apply Hsub. left. reflexivity.
  - intros bindings body Hcodes IH env Hsc Hsub [Hbs Hold] a Hv.
    cbn [traverse]. apply IH; [ exact Hsc | | exact Hold | ].
    { intros p Hp. apply Hsub. cbn [let_bindings]. apply in_or_app.
      right. exact Hp. }
    apply vinv_fold.
    { intros a' bd Hbd Hv'. destruct bd as [id c | id | s const];
        cbn [traverse_binding]; [ | exact Hv' | exact Hv' ].
      pose proof (proj1 (FlambdaMapProofs.every_in _ _ _) Hbs _ Hbd)
        as [Hno _].
      pose proof (Hcodes id c Hbd) as Hc.
      destruct c as [params mc ret exn md cbody]. cbn in Hno, Hc.
      cbn [traverse_code].
      destruct (lookup id (acc_code_deps a')) as [cd | ]; [ | exact Hv' ].
      apply Hc; [ exact Hsc | | exact Hno | ].
      - intros p Hp. apply Hsub. cbn [let_bindings]. apply in_or_app.
        left. apply in_flat_map.
        exists (Bind_code id (Code params mc ret exn md cbody)).
        split; [ exact Hbd | exact Hp ].
      - exact (vinv_ns _ _ Hv' (CE.ns_code_aliases mc params cd a')). }
    apply vinv_fold.
    { intros a' bd Hbd Hv'. apply vinv_static_binding; [ | exact Hv' ].
      exact (proj1 (FlambdaMapProofs.every_in _ _ _) Hbs _ Hbd). }
    apply (vinv_ns _ _ Hv). apply CE.ns_fold. intros a' bd _.
    apply CE.ns_prepare_binding.
  - intros k h body IHh IH env Hsc Hsub [Hh Hold] a Hv.
    destruct h as [params md handler]. cbn [traverse traverse_handler].
    apply IH; [ exact Hsc | | exact Hold | ].
    { intros p Hp. apply Hsub. cbn [let_bindings]. apply in_or_app.
      right. exact Hp. }
    refine (vinv_ns _ _ _ (CE.ns_exn_handler_deps env _ _)).
    apply IHh; [ exact Hsc | | exact Hh | exact Hv ].
    intros p Hp. apply Hsub. cbn [let_bindings handler_let_bindings].
    apply in_or_app. left. exact Hp.
  - intros handlers body IHh IH env Hsc Hsub [Hhs Hold] a Hv.
    cbn [traverse].
    assert (Hsc' : env_set_closures (add_rec_conts env handlers)
                   = set_closures e0)
      by (rewrite add_rec_conts_sc; exact Hsc).
    apply IH; [ exact Hsc' | | exact Hold | ].
    { intros p Hp. apply Hsub. cbn [let_bindings]. apply in_or_app.
      right. exact Hp. }
    apply vinv_fold; [ | exact Hv ].
    intros a' [k h] Hkh Hv'. pose proof (IHh k h Hkh) as Hh.
    destruct h as [params md handler]. cbn [traverse_handler].
    apply Hh; [ exact Hsc' | | | exact Hv' ].
    + intros p Hp. apply Hsub. cbn [let_bindings]. apply in_or_app. left.
      apply in_flat_map. exists (k, Handler params md handler).
      split; [ exact Hkh | exact Hp ].
    + exact (proj1 (FlambdaMapProofs.every_in _ _ _) Hhs _ Hkh).
  - intros ap env _ _ _ a Hv.
    exact (vinv_ns _ _ Hv (CE.ns_traverse_apply _ _ _)).
  - intros ac env _ _ _ a Hv.
    exact (vinv_ns _ _ Hv (CE.ns_apply_cont_deps _ _ _)).
  - intros scrutinee arms env _ _ _ a Hv.
    exact (vinv_ns _ _ Hv (CE.ns_traverse_switch _ _ _ _)).
  - intros env _ _ _ a Hv. exact Hv.
  - intros params md handler IH. exact IH.
Qed.

End Slots.

(* What the facts of the unit need: no old sets, each variable bound
   once by a [Let], each symbol once. *)
Lemma unit_bound_once (u : unit_interface) (e : expr) :
  in_flow_fragment e = true -> variables_bound_once e = true ->
  CE.no_old_sets e ∧ NoDup (map fst (let_bindings e)).
Proof.
  intros Hflow Honce. split.
  - pose proof Hflow as Hf. unfold in_flow_fragment in Hf.
    apply andb_true_iff in Hf as [Hf _].
    exact (proj1 (CE.fragment_no_old_sets _ _ _) e Hf).
  - pose proof Honce as Ho. unfold variables_bound_once in Ho.
    apply distinct_NoDup in Ho. exact (NoDup_app_remove_r _ _ Ho).
Qed.

Theorem closure_value_slots : closure_value_slots_statement.
Proof.
  intros flag u e x slot decls vs w b Hflow Honce Hx Hc.
  destruct (unit_bound_once u e Hflow Honce) as [Hold Hnl].
  revert Hc. unfold traverse_unit.
  set (a0 := add_fact (Fact_any_source Node_all_constants) empty_acc).
  destruct (fresh a0) as [r a1] eqn:E1.
  destruct (fresh a1) as [z a2] eqn:E2.
  cbv zeta.
  match goal with
  | |- context [traverse ?env e ?a3] => set (env0 := env); set (a := a3)
  end.
  intros Hc. apply CE.constructor_of_list in Hc.
  assert (Hv0 : vinv e empty_acc).
  { intros x' slot' decls' vs' w' b' _ H. destruct H. }
  assert (H1 : CE.ns_ok a0 a1).
  { change a1 with (snd (r, a1)). rewrite <- E1. apply CE.ns_fresh. }
  assert (H2 : CE.ns_ok a1 a2).
  { change a2 with (snd (z, a2)). rewrite <- E2. apply CE.ns_fresh. }
  assert (H0 : CE.ns_ok empty_acc a0).
  { apply CE.ns_add_fact. cbn. auto. }
  assert (H3 : CE.ns_ok a2 a).
  { apply CE.ns_add_fact. cbn. auto. }
  assert (Hva : vinv e a).
  { exact (vinv_ns e _ _ (vinv_ns e _ _ (vinv_ns e _ _ (vinv_ns e _ _ Hv0 H0)
                                          H1) H2) H3). }
  assert (Hvt : vinv e (traverse env0 e a)).
  { apply (proj1 (traverse_vinv e Hnl) e env0 eq_refl (fun p Hp => Hp) Hold
             a Hva). }
  assert (H4 : CE.ns_ok (traverse env0 e a)
                 (add_fact (Fact_any_usage r) (traverse env0 e a))).
  { apply CE.ns_add_fact. cbn. auto. }
  exact (vinv_ns e _ _ (vinv_ns e _ _ Hvt H4) (CE.ns_resolve _ _) x slot
           decls vs w b Hx Hc).
Qed.

(* ======================================================================== *)
(* The fields of a representation                                           *)
(* ======================================================================== *)

(* The allocation of the variable of a [Let] of a closure holds exactly
   the facts of its set. *)
Lemma let_node_exact (u : unit_interface) (e : expr) (x : variable)
    (slot : function_slot) (decls : list (function_slot * code_id))
    (vs : list (value_slot * simple)) :
  in_flow_fragment e = true -> well_formed_unit u e = true ->
  variables_bound_once e = true ->
  (x, Closure_of_set slot decls vs) ∈ let_bindings e ->
  RP.node_exact (graph_of_facts (traverse_unit Auto u e)) (unit_kept e)
    (Node_variable x) decls.
Proof.
  intros Hflow Hwfu Honce Hx.
  destruct (unit_bound_once u e Hflow Honce) as [Hold Hnl].
  assert (Hns : NoDup (map fst (CE.sym_bindings e))).
  { rewrite (proj1 CE.sym_bindings_static_names e Hold).
    apply distinct_NoDup. pose proof Hwfu as Hwf.
    unfold well_formed_unit in Hwf.
    repeat (apply andb_true_iff in Hwf as [Hwf ?]).
    assumption. }
  destruct (CE.traverse_unit_groups Auto u e Hnl Hns Hold)
    as (Gs & acc & HF & Hinv & _).
  rewrite HF.
  exact (CE.closure_node_exact e Hnl Gs acc Hinv x slot decls vs Hx).
Qed.

Theorem changed_field_in_set : changed_field_in_set_statement.
Proof.
  intros is_local u e T cannot_change max_unbox_size lambda_lifting dominated
    to_unbox tcr x slot decls vs r w uf Hflow Hwfu Honce Hleast Htables Hx
    Hlk Hfu.
  destruct (FTP.field_origin is_local u e T cannot_change max_unbox_size
              lambda_lifting dominated to_unbox tcr x slot decls vs r Hflow
              Hwfu Hleast Htables Hx Hlk _ uf Hfu)
    as (x0 & w' & _ & from & Ew & _ & _ & Hfrom & g & Hx0x).
  injection Ew as <-.
  destruct (TTP.function_slots_mutual_traversal u e Hflow Hwfu Honce x0
              (Node_variable x) g Hx0x)
    as (g0 & Hxx0).
  destruct (let_node_exact u e x slot decls vs Hflow Hwfu Honce Hx)
    as (_ & _ & E3 & _).
  pose proof (proj1 (proj2 (E3 g0 x0 Hxx0) w from) Hfrom) as Hxw.
  destruct (closure_value_slots Auto u e x slot decls vs w from Hflow Honce
              Hx Hxw)
    as (a & Ha & _).
  rewrite Ha. discriminate.
Qed.

(* ======================================================================== *)
(* The fragment                                                             *)
(* ======================================================================== *)

Section With_mono.

Variable F : facts.
Variable sol : solution.
Variable cannot_change : node -> bool.
Variable code_deps : finite_map code_dep.
Variable unit_code_ids : list code_id.
Variable codes : finite_map code_metadata.
Variable unboxed : list (node * unboxed_fields).
Variable changed : list (node * closure_repr).
Variable my_closures : finite_map variable.
Variable nr : code_id -> bool.
Variable L0 : list (variable * named).

Variable cs cs' :
  variable -> flambda_kind -> named -> expr -> closure_repr -> bool.

(* [cs] implies [cs'] at the [Let]s among [L0]. *)
Hypothesis Hcs : ∀ (x : variable) (kind : flambda_kind) (d : named)
    (body : expr) (r : closure_repr),
  (x, d) ∈ L0 -> changed_name changed (Node_variable x) = Some r ->
  cs x kind d body r = true -> cs' x kind d body r = true.

Abbreviation frag kinds C e :=
  (repr_expr_with F sol cannot_change code_deps unit_code_ids codes unboxed
     changed my_closures nr cs kinds C e).
Abbreviation full kinds C e :=
  (repr_expr_with F sol cannot_change code_deps unit_code_ids codes unboxed
     changed my_closures nr cs' kinds C e).
Abbreviation frag_h kinds C ds h :=
  (repr_handler_with F sol cannot_change code_deps unit_code_ids codes
     unboxed changed my_closures nr cs kinds C ds h).
Abbreviation full_h kinds C ds h :=
  (repr_handler_with F sol cannot_change code_deps unit_code_ids codes
     unboxed changed my_closures nr cs' kinds C ds h).

Definition wmono_P (e : expr) : Prop :=
  ∀ (kinds : finite_map flambda_kind) (C : finite_map (list decision * bool)),
    (∀ p, p ∈ let_bindings e -> p ∈ L0) ->
    frag kinds C e = true -> full kinds C e = true.

Lemma with_mono :
  (∀ e : expr, wmono_P e)
  ∧ ∀ h : continuation_handler,
      match h with
      | Handler params md handler =>
        ∀ (kinds : finite_map flambda_kind)
          (C : finite_map (list decision * bool)) (ds : list decision),
          (∀ p, p ∈ let_bindings handler -> p ∈ L0) ->
          frag_h kinds C ds (Handler params md handler) = true ->
          full_h kinds C ds (Handler params md handler) = true
      end.
Proof.
  apply FP.expr_handler_ind; unfold wmono_P.
  - intros x kind d body IH kinds C Hsub H.
    cbn [repr_expr_with] in H |- *.
    apply andb_true_iff in H as [H Hb]. apply andb_true_iff in H as [Hk Hd].
    rewrite Hk, (IH _ C (fun p Hp => Hsub p (or_intror Hp)) Hb).
    rewrite andb_true_r. cbn [andb].
    destruct (fields_of unboxed x) as [fields | ];
      destruct (changed_name changed (Node_variable x)) as [r | ] eqn:Hch;
      try exact Hd.
    exact (Hcs x kind d body r (Hsub _ (or_introl eq_refl)) Hch Hd).
  - intros bindings body Hcodes IH kinds C Hsub H.
    cbn [repr_expr_with] in H |- *.
    apply andb_true_iff in H as [Hbs Hb].
    apply andb_true_iff. split.
    + rewrite forallb_forall in Hbs |- *. intros b Hbin.
      specialize (Hbs b Hbin).
      destruct b as [id c | id | s const];
        cbn [repr_binding_with] in Hbs |- *; [ | exact Hbs
                                                          | exact Hbs ].
      destruct c as [params mc ret exn md cbody].
      cbn [repr_code_with] in Hbs |- *.
      destruct (code_change_u sol cannot_change code_deps unit_code_ids
                  unboxed id md) as [ | pd rd];
        [ split_bools; discriminate | ].
      split_bools; try assumption.
      refine (Hcodes id (Code params mc ret exn md cbody) Hbin _ _ _ _);
        [ | eassumption ].
      intros p Hp. apply Hsub. cbn [let_bindings]. apply in_or_app. left.
      apply in_flat_map.
      exists (Bind_code id (Code params mc ret exn md cbody)).
      split; [ exact Hbin | exact Hp ].
    + refine (IH kinds C _ Hb). intros p Hp. apply Hsub.
      cbn [let_bindings]. apply in_or_app. right. exact Hp.
  - intros k h body IHh IH kinds C Hsub H. destruct h as [params md handler].
    cbn [repr_expr_with] in H |- *.
    apply andb_true_iff in H as [Hh Hb].
    apply andb_true_iff. split.
    + refine (IHh kinds C _ _ Hh). intros p Hp. apply Hsub.
      cbn [let_bindings handler_let_bindings]. apply in_or_app. left.
      exact Hp.
    + refine (IH _ _ _ Hb). intros p Hp. apply Hsub.
      cbn [let_bindings]. apply in_or_app. right. exact Hp.
  - intros handlers body IHh IH kinds C Hsub H.
    cbn [repr_expr_with] in H |- *.
    apply andb_true_iff in H as [Hhs Hb].
    apply andb_true_iff. split.
    + rewrite forallb_forall in Hhs |- *. intros [k h] Hkh.
      specialize (Hhs (k, h) Hkh). pose proof (IHh k h Hkh) as IHkh.
      destruct h as [params md handler]. cbv beta iota in Hhs |- *.
      refine (IHkh _ _ _ _ Hhs). intros p Hp. apply Hsub.
      cbn [let_bindings]. apply in_or_app. left. apply in_flat_map.
      exists (k, Handler params md handler). split; [ exact Hkh | exact Hp ].
    + refine (IH _ _ _ Hb). intros p Hp. apply Hsub.
      cbn [let_bindings]. apply in_or_app. right. exact Hp.
  - intros ap kinds C _ H. exact H.
  - intros ac kinds C _ H. exact H.
  - intros scrutinee arms kinds C _ H. exact H.
  - intros kinds C _ H. exact H.
  - intros params md handler IH kinds C ds Hsub H.
    cbn [repr_handler_with] in H |- *.
    split_bools; try assumption.
    exact (IH _ _ Hsub ltac:(eassumption)).
Qed.

End With_mono.
Theorem repr_unboxing_frag3_sound : repr_unboxing_frag3_sound_statement.
Proof.
  intros nr is_local u e T cannot_change max_unbox_size lambda_lifting
    code_deps dominated to_unbox tcr Hflow Hwfu Honce Hleast Htables H.
  unfold repr_unboxing_frag3_nr in H. unfold repr_unboxing_frag2_nr.
  apply andb_true_iff in H as [H Huses]. rewrite Huses, andb_true_r.
  refine (proj1 (with_mono _ _ _ _ _ _ _ _ _ _ (let_bindings e) _ _ _) e []
            [] (fun p Hp => Hp) H).
  intros x kind d body r Hx Hch Hfrag.
  destruct d as [s | p args | id vs | slot decls vs | w c | pf w c | mf mt c];
    cbn [changed_set_frag3_ok] in Hfrag; try discriminate Hfrag.
  unfold changed_name, changed_of_node in Hch.
  cbn [changed_set_frag2_ok].
  apply andb_true_iff in Hfrag as [Hfrag Hfields].
  rewrite Hfrag. cbn [andb].
  rewrite forallb_forall in Hfields |- *. intros [f uf] Hfu.
  specialize (Hfields _ Hfu).
  destruct f as [i k | w | fs | | | | cw | rc | ]; try reflexivity.
  cbn [repr_field_frag3_ok] in Hfields.
  destruct (lookup w vs) as [a | ] eqn:Hw; [ exact Hfields | ].
  exfalso.
  exact (changed_field_in_set is_local u e T cannot_change max_unbox_size
           lambda_lifting dominated to_unbox tcr x slot decls vs r w uf Hflow
           Hwfu Honce Hleast Htables Hx Hch Hfu Hw).
Qed.

(* ======================================================================== *)
(* The refinement                                                           *)
(* ======================================================================== *)

Theorem rebuild_repr_frag3_refines_cc :
  rebuild_repr_frag3_refines_cc_statement.
Proof.
  intros is_local u e T cannot_change max_unbox_size lambda_lifting dominated
    to_unbox tcr pdc nr Hloc Hflow Hwfu Honce Hleast Hcc Htables Hfrag Hreads.
  exact (FTP.rebuild_repr_frag2_refines_cc is_local u e T cannot_change
           max_unbox_size lambda_lifting dominated to_unbox tcr pdc nr Hloc
           Hflow Hwfu Honce Hleast Hcc Htables
           (repr_unboxing_frag3_sound nr is_local u e T cannot_change
              max_unbox_size lambda_lifting _ dominated to_unbox tcr Hflow
              Hwfu Honce Hleast Htables Hfrag)
           Hreads).
Qed.

Theorem rebuild_repr_frag3_observably :
  rebuild_repr_frag3_observably_statement.
Proof.
  intros is_local u e T cannot_change max_unbox_size lambda_lifting dominated
    to_unbox tcr pdc nr Hloc Hproj Hflow Hany Hres Hwfu Honce Hexits Hleast
    Hcc Hexact Hccc Hwidgets Htables Hfrag Hreads.
  exact (FTP.rebuild_repr_frag2_observably is_local u e T cannot_change
           max_unbox_size lambda_lifting dominated to_unbox tcr pdc nr Hloc
           Hproj Hflow Hany Hres Hwfu Honce Hexits Hleast Hcc Hexact Hccc
           Hwidgets Htables
           (repr_unboxing_frag3_sound nr is_local u e T cannot_change
              max_unbox_size lambda_lifting _ dominated to_unbox tcr Hflow
              Hwfu Honce Hleast Htables Hfrag)
           Hreads).
Qed.

Theorem rebuild_repr_frag3_local : rebuild_repr_frag3_local_statement.
Proof.
  intros is_local u e T cannot_change max_unbox_size lambda_lifting dominated
    to_unbox tcr pdc nr Hloc Hflow Hany Hres Hwfu Honce Hexits Hleast Hcc
    Hlocal Hexact Hccc Hwidgets Htables Hfrag Hreads.
  exact (FTP.rebuild_repr_frag2_local is_local u e T cannot_change
           max_unbox_size lambda_lifting dominated to_unbox tcr pdc nr Hloc
           Hflow Hany Hres Hwfu Honce Hexits Hleast Hcc Hlocal Hexact Hccc
           Hwidgets Htables
           (repr_unboxing_frag3_sound nr is_local u e T cannot_change
              max_unbox_size lambda_lifting _ dominated to_unbox tcr Hflow
              Hwfu Honce Hleast Htables Hfrag)
           Hreads).
Qed.
