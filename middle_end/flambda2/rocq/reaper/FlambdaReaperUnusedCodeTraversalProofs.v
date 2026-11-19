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

(* The traversal gives the code of a unit the shape of
   FlambdaReaperUnusedCodeProofs.v, with [Auto], where the unit is of
   [in_flow_fragment] and binds each variable once, so that a least solution
   meets [unused_code_dead] ([unused_code_dead_traversal]). Nothing here
   needs review.

   - [bound_once_split]: [param_variables] is a permutation of the
     [my_closure]s of [unit_codes] and of the other parameters
     ([other_params]); with the variables of the [Let]s they are distinct,
     so that a [my_closure] of the unit is of one code ([mc_inj]) and is no
     other variable ([mc_not_V0]). The closures that projections bind are
     variables of [Let]s ([set_closures_lets]).
   - [bounds]: what the facts of the unit put into the node of a variable,
     into the node [cd_my_closure] of a code dep and into the node
     [ad_apply_closure] of an apply dep, with the facts of the direct calls.
     With the known witnesses and the code deps of distinct code ids of
     FlambdaReaperWrapperProofs.unit_facts, they give the shape
     ([shape_of_bounds], [unused_code_dead_of_bounds]).
   - [vsafe]: each step of the traversal puts something only into the
     variables of [V0] (of the [Let]s, the handlers and the code, but the
     [my_closure]s of [unit_codes]), but [traverse_code], whose alias into
     its [my_closure] is the other case of [var_fact].
   - [inv]: the invariant of the accumulator. Nothing is put into a
     [cd_my_closure] before [resolve_delayed_deps], and into an
     [ad_apply_closure] only the alias on any source of its callee; these
     nodes are made and distinct; the call witnesses of the apply deps, the
     closures of the closure deps and the nodes of the continuations in
     scope ([menv]) are none of them. The steps of [KnownWitnesses] give it
     with an accumulator whose witnesses of known arity are these nodes
     ([ghost], [mstep_core]); [prepare_code] ([mstep_prepare_code]) and the
     call with [Auto] ([call_kind_spec], [mstep_apply]) make them;
     [mstep_traverse] is the traversal, [resolve_bounds] the delayed deps
     and [unit_bounds] the unit. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia Permutation.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaWellFormedSpec FlambdaReaperGraphSpec
  FlambdaReaperTraversal FlambdaReaperCallingConventions
  FlambdaReaperFlowSpec FlambdaReaperLeastSpec FlambdaReaperUnusedCodeProofs.
From OxCamlFlambda2 Require FlambdaMapProofs FlambdaReaperFlowProofs
  FlambdaReaperLeastProofs FlambdaReaperWrapperProofs
  FlambdaReaperWrapperAutoProofs FlambdaReaperFixFailsSpec.
Import ListNotations.

Abbreviation into_nodes := FlambdaReaperLeastProofs.AutoWidgets.into_nodes.
Abbreviation made_lt := FlambdaReaperLeastProofs.AutoWidgets.made_lt.
Abbreviation not_made := FlambdaReaperLeastProofs.AutoWidgets.not_made.
Abbreviation cid := Field_code_id_of_call_witness.
Abbreviation env_conts := FlambdaReaperTraversal.env_conts.

(* ======================================================================== *)
(* Variables bound once                                                     *)
(* ======================================================================== *)

Definition mc_of (p : code_id * code) : variable := code_my_closure (snd p).

(* The parameters other than the [my_closure]s of [unit_codes]: those of
   the handlers, and of the code with all that its body binds. *)
Definition binding_other_params (b : static_binding) : list variable :=
  match b with
  | Bind_code _ (Code params _ _ _ _ body) => params ++ param_variables body
  | Bind_deleted_code _ | Bind_symbol _ _ => []
  end.

Fixpoint other_params (e : expr) : list variable :=
  match e with
  | Let _ _ _ body => other_params body
  | Let_static bindings body =>
    flat_map binding_other_params bindings ++ other_params body
  | Let_cont_non_recursive _ h body =>
    handler_other_params h ++ other_params body
  | Let_cont_recursive handlers body =>
    flat_map (fun '(_, h) => handler_other_params h) handlers
    ++ other_params body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => []
  end
with handler_other_params (h : continuation_handler) : list variable :=
  match h with
  | Handler params _ handler => params ++ other_params handler
  end.

Definition binding_codes (b : static_binding) : list (code_id * code) :=
  match b with
  | Bind_code id c => [(id, c)]
  | Bind_deleted_code _ | Bind_symbol _ _ => []
  end.

Lemma perm_app4 {A : Type} (l1 l2 a b c d : list A) :
  Permutation l1 (a ++ b) -> Permutation l2 (c ++ d) ->
  Permutation (l1 ++ l2) ((a ++ c) ++ (b ++ d)).
Proof.
  intros H1 H2. eapply Permutation_trans; [ apply Permutation_app; eassumption
                                         | ].
  rewrite <- !app_assoc. apply Permutation_app_head.
  rewrite !app_assoc. apply Permutation_app_tail. apply Permutation_app_comm.
Qed.

Lemma perm_flat_map {A : Type} (f g h : A -> list variable) (l : list A) :
  (∀ x, x ∈ l -> Permutation (f x) (g x ++ h x)) ->
  Permutation (flat_map f l) (flat_map g l ++ flat_map h l).
Proof.
  induction l as [ | x l IH]; intros H; [ apply Permutation_refl | ].
  cbn [flat_map]. apply perm_app4; [ apply H; left; reflexivity | ].
  apply IH. intros y Hy. apply H. right. exact Hy.
Qed.

Lemma map_flat_map' {A B C : Type} (f : B -> C) (g : A -> list B)
    (l : list A) :
  map f (flat_map g l) = flat_map (fun x => map f (g x)) l.
Proof.
  induction l as [ | x l IH]; [ reflexivity | ].
  cbn [flat_map]. rewrite map_app, IH. reflexivity.
Qed.

Lemma params_perm :
  (∀ e, Permutation (param_variables e)
          (map mc_of (unit_codes e) ++ other_params e))
  ∧ (∀ h, Permutation (handler_param_variables h)
            (map mc_of (handler_unit_codes h) ++ handler_other_params h)).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind;
    cbn [param_variables unit_codes other_params handler_param_variables
         handler_unit_codes handler_other_params];
    try (intros; apply Permutation_refl).
  - intros x kind d body IH. exact IH.
  - intros bindings body _ IH. rewrite map_app. apply perm_app4; [ | exact IH ].
    replace (map mc_of (flat_map _ bindings))
      with (flat_map (fun b => map mc_of (binding_codes b)) bindings).
    + apply perm_flat_map. intros [id [params mc ret exn md body'] | id | s k];
        intros _; cbn; apply Permutation_refl.
    + rewrite map_flat_map'. apply flat_map_ext.
      intros [id c | id | s k]; reflexivity.
  - intros k h body IHh IH. rewrite map_app. apply perm_app4; assumption.
  - intros handlers body IHh IH. rewrite map_app.
    apply perm_app4; [ | exact IH ].
    replace (map mc_of (flat_map _ handlers))
      with (flat_map (fun '(_, h) => map mc_of (handler_unit_codes h))
              handlers).
    + apply perm_flat_map. intros [k h] Hin. exact (IHh k h Hin).
    + rewrite map_flat_map'. apply flat_map_ext. intros [k h]. reflexivity.
  - intros params md handler IH.
    eapply Permutation_trans; [ apply Permutation_app_head; exact IH | ].
    apply Permutation_app_swap_app.
Qed.

Lemma other_in_params (e : expr) (v : variable) :
  v ∈ other_params e -> v ∈ param_variables e.
Proof.
  intros H. apply (Permutation_in _ (Permutation_sym (proj1 params_perm e))).
  apply in_or_app. right. exact H.
Qed.

Lemma mc_in_params (e : expr) (id : code_id) (c : code) :
  (id, c) ∈ unit_codes e -> code_my_closure c ∈ param_variables e.
Proof.
  intros H. apply (Permutation_in _ (Permutation_sym (proj1 params_perm e))).
  apply in_or_app. left. exact (in_map mc_of _ (id, c) H).
Qed.

(* The variables of a unit other than the [my_closure]s of its code. *)
Definition V0 (e0 : expr) (v : variable) : Prop :=
  v ∈ map fst (let_bindings e0) ∨ v ∈ other_params e0.

Lemma bound_once_split (e : expr) :
  variables_bound_once e = true ->
  NoDup (map fst (let_bindings e) ++ map mc_of (unit_codes e)
         ++ other_params e).
Proof.
  intros H. unfold variables_bound_once in H.
  apply FlambdaMapProofs.distinct_NoDup in H.
  eapply Permutation_NoDup; [ | exact H ].
  apply Permutation_app_head. exact (proj1 params_perm e).
Qed.

Lemma nodup_map_inj {A B : Type} (f : A -> B) (l : list A) (x y : A) :
  NoDup (map f l) -> x ∈ l -> y ∈ l -> f x = f y -> x = y.
Proof.
  induction l as [ | z l IH]; intros Hnd Hx Hy Hf; [ destruct Hx | ].
  cbn [map] in Hnd. inversion Hnd as [ | z' l' Hz Hnd' ]; subst.
  destruct Hx as [<- | Hx], Hy as [<- | Hy]; auto.
  - exfalso. apply Hz. rewrite Hf. apply in_map. exact Hy.
  - exfalso. apply Hz. rewrite <- Hf. apply in_map. exact Hx.
Qed.

(* A [my_closure] of the unit is of one code, and is no other variable. *)
Lemma mc_inj (e : expr) (id id' : code_id) (c c' : code) :
  variables_bound_once e = true ->
  (id, c) ∈ unit_codes e -> (id', c') ∈ unit_codes e ->
  code_my_closure c = code_my_closure c' -> id = id' ∧ c = c'.
Proof.
  intros Hb H H' E. pose proof (bound_once_split e Hb) as Hnd.
  apply NoDup_app_remove_l, NoDup_app_remove_r in Hnd.
  pose proof (nodup_map_inj mc_of _ _ _ Hnd H H' E) as Heq.
  injection Heq as -> ->. auto.
Qed.

Lemma mc_not_V0 (e : expr) (id : code_id) (c : code) :
  variables_bound_once e = true ->
  (id, c) ∈ unit_codes e -> ~ V0 e (code_my_closure c).
Proof.
  intros Hb H [Hv | Hv]; pose proof (bound_once_split e Hb) as Hnd;
    assert (Hm : code_my_closure c ∈ map mc_of (unit_codes e))
      by exact (in_map mc_of _ (id, c) H).
  - apply (FlambdaReaperLeastProofs.BoundOnce.nodup_app_disjoint _ _ _ Hnd Hv).
    apply in_or_app. left. exact Hm.
  - apply NoDup_app_remove_l in Hnd.
    exact (FlambdaReaperLeastProofs.BoundOnce.nodup_app_disjoint _ _ _ Hnd Hm
             Hv).
Qed.

(* The closures of sets that projections bind are variables of [Let]s. *)
Lemma let_projections_lets (x : variable) (slot : function_slot)
    (others : list function_slot) (e : expr) (f : function_slot)
    (y : variable) :
  (f, y) ∈ let_projections_of x slot others e ->
  y ∈ map fst (let_bindings e).
Proof.
  revert e. induction others as [ | o others IH]; intros e H;
    [ destruct H | ].
  destruct e as [y' kind d body | | | | | | | ]; cbn in H; try destruct H.
  destruct d as [ | | | | | | mf mt [x' | | ]]; try destruct H.
  destruct (_ && _ && _ && _); [ | destruct H ].
  destruct H as [[= <- <-] | H]; [ left; reflexivity | ].
  right. exact (IH body H).
Qed.

Lemma after_local_allocation_lets (x : variable) (e : expr) (v : variable) :
  v ∈ map fst (let_bindings (after_local_allocation x e)) ->
  v ∈ map fst (let_bindings e).
Proof.
  unfold after_local_allocation. intros H.
  repeat match type of H with
  | context [match ?t with _ => _ end] => destruct t
  end;
  try exact H; cbn [let_bindings map]; right; exact H.
Qed.

Lemma set_closures_lets :
  (∀ e x others f y, (x, others) ∈ set_closures e -> (f, y) ∈ others ->
     y ∈ map fst (let_bindings e))
  ∧ (∀ h x others f y, (x, others) ∈ handler_set_closures h ->
       (f, y) ∈ others -> y ∈ map fst (handler_let_bindings h)).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind;
    cbn [set_closures handler_set_closures let_bindings handler_let_bindings];
    try (intros; contradiction).
  - intros x kind d body IH x' others f y H Hy.
    apply in_app_or in H as [H | H]; [ | right; exact (IH _ _ _ _ H Hy) ].
    destruct d as [ | | | slot decls vs | | | ]; try solve [ destruct H ].
    destruct H as [[= <- <-] | []]. right.
    unfold projected_set_closures in Hy.
    destruct decls as [ | [first id] rest]; [ destruct Hy | ].
    destruct (Pos.eqb first slot); [ | destruct Hy ].
    apply after_local_allocation_lets with (x := x).
    exact (let_projections_lets _ _ _ _ _ _ Hy).
  - intros bindings body IHc IH x others f y H Hy.
    rewrite map_app. apply in_or_app.
    apply in_app_or in H as [H | H]; [ left | right; exact (IH _ _ _ _ H Hy) ].
    apply in_flat_map in H as (b & Hb & H).
    rewrite map_flat_map'.
    apply in_flat_map. exists b. split; [ exact Hb | ].
    destruct b as [id c | | ]; cbn in H |- *; try solve [ destruct H ].
    destruct c as [params mc ret exn md body']. cbn in H |- *.
    exact (IHc id _ Hb _ _ _ _ H Hy).
  - intros k h body IHh IH x others f y H Hy. rewrite map_app.
    apply in_or_app. apply in_app_or in H as [H | H];
      [ left; exact (IHh _ _ _ _ H Hy) | right; exact (IH _ _ _ _ H Hy) ].
  - intros handlers body IHh IH x others f y H Hy. rewrite map_app.
    apply in_or_app. apply in_app_or in H as [H | H];
      [ left | right; exact (IH _ _ _ _ H Hy) ].
    apply in_flat_map in H as ([k h] & Hkh & H).
    rewrite map_flat_map'.
    apply in_flat_map. exists (k, h). split; [ exact Hkh | ].
    exact (IHh k h Hkh _ _ _ _ H Hy).
  - intros params md handler IH. exact IH.
Qed.

(* ======================================================================== *)
(* A node into which the facts of a list put only what is said             *)
(* ======================================================================== *)

Definition into_ok (A : node -> Prop) (P Q : node -> node -> Prop)
    (fct : fact) : Prop :=
  match fct with
  | Fact_alias _ from => A from
  | Fact_propagate c _ from => P c from
  | Fact_alias_if_any_source c _ from => Q c from
  | _ => False
  end.

Lemma into_only_of_list (l : list fact) (x : node) (A : node -> Prop)
    (P Q : node -> node -> Prop) :
  (∀ fct, fct ∈ l -> x ∈ into_nodes fct -> into_ok A P Q fct) ->
  into_only (FlambdaReaperTraversal.facts_of_list l) x A P Q.
Proof.
  intros H.
  assert (G : ∀ (B : Type) (g : fact -> list B) (y : B),
             y ∈ flat_map g l -> ∃ fct, fct ∈ l ∧ y ∈ g fct)
    by (intros B g y Hy; apply in_flat_map in Hy as (fct & H1 & H2); eauto).
  constructor; cbn [FlambdaReaperTraversal.facts_of_list f_alias f_use
    f_accessor f_constructor f_parameter f_propagate f_alias_if_any_source
    f_any_source f_any_source_if_used f_zero_alloc_source];
    [ intros from Hin | intros c from Hin | intros c from Hin
    | intros f b Hin | intros b i Hin | intros i t Hin | intros r from Hin
    | intros Hin | intros [] | intros c Hin | intros from Hin ];
    apply G in Hin as (fct & Hl & Hin); destruct fct; cbn in Hin;
    try contradiction; destruct Hin as [Heq | []];
    inversion Heq; subst; apply (H _ Hl); cbn; auto.
Qed.

(* ======================================================================== *)
(* The bounds, and the shape                                                *)
(* ======================================================================== *)

(* What a direct call of [d] makes, with its callee, its widget, its witness
   and the node its code id is read into. *)
Definition chain (l : list fact) (d : apply_dep)
    (callee widget witness called : node) : Prop :=
  Fact_alias_if_any_source callee (ad_apply_closure d) callee ∈ l
  ∧ Fact_alias_if_any_source callee widget (ad_apply_call_witness d) ∈ l
  ∧ alias_for_caller (ad_function_containing_apply_expr d) widget witness ∈ l
  ∧ Fact_accessor called cid witness ∈ l
  ∧ Fact_any_usage called ∈ l.

(* The facts [l] put into the node of a variable of the unit [e0], or the
   alias of [traverse_code] into the [my_closure] of its code; into the
   [my_closure] of a code dep, the propagate of a closure of its code and
   the aliases of its direct calls; into the closure of an apply dep, the
   alias on any source of the callee. *)
Record bounds (e0 : expr) (cds : finite_map code_dep) (ads : list apply_dep)
    (l : list fact) : Prop := {
  b_var : ∀ fct v, fct ∈ l -> Node_variable v ∈ into_nodes fct ->
    V0 e0 v
    ∨ ∃ id c cd, (id, c) ∈ unit_codes e0 ∧ code_my_closure c = v
      ∧ (id, cd) ∈ cds ∧ fct = Fact_alias (Node_variable v) (cd_my_closure cd);
  b_cdm : ∀ id cd fct, (id, cd) ∈ cds -> fct ∈ l ->
    cd_my_closure cd ∈ into_nodes fct ->
    (∃ base, fct = Fact_propagate (Node_code_id id) (cd_my_closure cd) base)
    ∨ ∃ d, d ∈ ads ∧ lookup (ad_apply_code_id d) cds = Some cd
      ∧ fct = alias_for_caller (ad_function_containing_apply_expr d)
                (ad_apply_closure d) (cd_my_closure cd);
  b_ad : ∀ d, d ∈ ads ->
    ∃ callee widget witness called,
      chain l d callee widget witness called
      ∧ ∀ fct, fct ∈ l -> ad_apply_closure d ∈ into_nodes fct ->
          fct = Fact_alias_if_any_source callee (ad_apply_closure d) callee;
  b_dep : ∀ d cd, d ∈ ads -> lookup (ad_apply_code_id d) cds = Some cd ->
    alias_for_caller (ad_function_containing_apply_expr d)
      (cd_known_arity_call_witness cd) (ad_apply_call_witness d) ∈ l
}.

Ltac fact_in :=
  cbn [FlambdaReaperTraversal.facts_of_list f_alias f_use f_accessor
       f_constructor f_parameter f_propagate f_alias_if_any_source
       f_any_source f_any_usage f_any_source_if_used];
  apply in_flat_map; eexists; split; [ eassumption | left; reflexivity ].

Lemma caller_alias_of (l : list fact) (caller : option code_id)
    (to_ from : node) :
  alias_for_caller caller from to_ ∈ l ->
  caller_alias (FlambdaReaperTraversal.facts_of_list l) caller to_ from.
Proof. destruct caller; cbn; intros H; fact_in. Qed.

Section Shape.

Variable e : expr.
Hypothesis Hbound : variables_bound_once e = true.
Variable cds : finite_map code_dep.
Variable ads : list apply_dep.
Variable l : list fact.
Hypothesis Hb : bounds e cds ads l.
Hypothesis Hnodup : NoDup (map fst cds).

Abbreviation F := (FlambdaReaperTraversal.facts_of_list l).

(* The direct calls of the code of [cd]. *)
Definition calls_of_dep (cd : code_dep) (dc : direct_call) : Prop :=
  ∃ d, d ∈ ads ∧ lookup (ad_apply_code_id d) cds = Some cd
    ∧ dc_caller dc = ad_function_containing_apply_expr d
    ∧ dc_closure dc = ad_apply_closure d
    ∧ dc_widget_if dc = ad_apply_call_witness d
    ∧ chain l d (dc_callee dc) (dc_widget dc) (dc_witness dc) (dc_called dc)
    ∧ ∀ fct, fct ∈ l -> ad_apply_closure d ∈ into_nodes fct ->
        fct = Fact_alias_if_any_source (dc_callee dc) (ad_apply_closure d)
                (dc_callee dc).

Lemma call_of_dep (cd : code_dep) (d : apply_dep) :
  d ∈ ads -> lookup (ad_apply_code_id d) cds = Some cd ->
  ∃ dc, calls_of_dep cd dc
    ∧ dc_caller dc = ad_function_containing_apply_expr d
    ∧ dc_closure dc = ad_apply_closure d.
Proof.
  intros Hd E. destruct (b_ad _ _ _ _ Hb d Hd)
    as (callee & widget & witness & called & Hch & Hinto).
  exists (Direct_call (ad_function_containing_apply_expr d) callee
            (ad_apply_closure d) widget (ad_apply_call_witness d) witness
            called).
  split; [ | split; reflexivity ].
  exists d. cbn. split; [ exact Hd | split; [ exact E | ] ].
  split; [ reflexivity | split; [ reflexivity | split; [ reflexivity | ] ] ].
  split; [ exact Hch | exact Hinto ].
Qed.

Lemma shape_of_bounds (id : code_id) (c : code) (cd : code_dep) :
  (id, c) ∈ unit_codes e -> (id, cd) ∈ cds ->
  (cd_known_arity_call_witness cd, cid, Node_code_id id) ∈ f_constructor F ->
  shape F id (code_my_closure c) (cd_my_closure cd)
    (cd_known_arity_call_witness cd) (calls_of_dep cd).
Proof.
  intros Hc Hcd Hkw.
  assert (Hcd_eq : ∀ cd', (id, cd') ∈ cds -> cd' = cd)
    by (intros cd' H';
        exact (FlambdaReaperLeastProofs.BoundOnce.nodup_map_fst _ _ _ _
                 Hnodup H' Hcd)).
  constructor.
  - apply into_only_of_list. intros fct Hin Hx.
    destruct (b_var _ _ _ _ Hb fct _ Hin Hx)
      as [Hv | (id' & c' & cd' & Hc' & Em & Hcd' & ->)];
      [ exfalso; exact (mc_not_V0 e id c Hbound Hc Hv) | ].
    destruct (mc_inj e id' id c' c Hbound Hc' Hc Em) as [-> ->].
    rewrite (Hcd_eq cd' Hcd'). reflexivity.
  - apply into_only_of_list. intros fct Hin Hx.
    destruct (b_cdm _ _ _ _ Hb id cd fct Hcd Hin Hx)
      as [(base & ->) | (d & Hd & E & ->)]; [ cbn; left; reflexivity | ].
    destruct (call_of_dep cd d Hd E) as (dc & Hdc & Hcaller & Hclosure).
    unfold alias_for_caller. cbn.
    destruct (ad_function_containing_apply_expr d) as [c' | ] eqn:Ec.
    + right. exists dc, c'. auto.
    + exists dc. auto.
  - intros dc (d & Hd & E & Hcaller & Hclosure & Hwif & Hch & Hinto).
    apply into_only_of_list. intros fct Hin Hx. rewrite Hclosure in Hx.
    rewrite (Hinto fct Hin Hx). cbn. auto.
  - exact Hkw.
  - intros dc (d & Hd & E & Hcaller & Hclosure & Hwif & Hch & Hinto).
    rewrite Hcaller, Hwif. apply caller_alias_of.
    exact (b_dep _ _ _ _ Hb d cd Hd E).
  - intros dc (d & Hd & E & Hcaller & Hclosure & Hwif & Hch & Hinto).
    destruct Hch as (_ & H & _). rewrite Hwif. fact_in.
  - intros dc (d & Hd & E & Hcaller & Hclosure & Hwif & Hch & Hinto).
    destruct Hch as (_ & _ & H & _). rewrite Hcaller.
    exact (caller_alias_of _ _ _ _ H).
  - intros dc (d & Hd & E & Hcaller & Hclosure & Hwif & Hch & Hinto).
    destruct Hch as (_ & _ & _ & H & _). fact_in.
  - intros dc (d & Hd & E & Hcaller & Hclosure & Hwif & Hch & Hinto).
    destruct Hch as (_ & _ & _ & _ & H). fact_in.
Qed.

End Shape.

(* [unused_code_dead] of a least solution, where the facts of the unit have
   the bounds. *)
Lemma unused_code_dead_of_bounds (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) :
  in_flow_fragment e = true -> variables_bound_once e = true ->
  is_local_slots_only is_local ->
  least_solution is_local (traverse_unit Auto u e) T ->
  bounds e (unit_code_deps Auto u e)
    (acc_apply_deps (traverse_unit_acc Auto u e))
    (acc_facts (traverse_unit_acc Auto u e)) ->
  FlambdaReaperFixFailsSpec.unused_code_dead (solution_of_tables T) e = true.
Proof.
  intros Hfrag Hbound Hslots Hleast Hb.
  destruct (FlambdaReaperWrapperProofs.unit_facts Auto u e True Hfrag
              (or_introl I))
    as (noted & deps & Hedges & _ & _ & Hnodup & _).
  pose proof (fun id c Hc =>
                proj1 (FlambdaReaperWrapperProofs.unit_codes_edges
                         _ _ _ _ _ _ _ _ _)
                  e None (unit_conts u) Hedges id c Hc) as Hcode_edges.
  apply (unused_code_dead_of_shapes is_local _ T Hleast Hslots e).
  intros id c Hc. pose proof (Hcode_edges id c Hc) as Hce.
  destruct c as [params mc ret exn md body].
  destruct Hce as (cd & Hcd & Hkw & _). destruct Hkw as (_ & _ & _ & Hk & _).
  exists (cd_my_closure cd), (cd_known_arity_call_witness cd),
    (calls_of_dep (unit_code_deps Auto u e)
       (acc_apply_deps (traverse_unit_acc Auto u e))
       (acc_facts (traverse_unit_acc Auto u e)) cd).
  rewrite traverse_unit_acc_facts.
  exact (shape_of_bounds e Hbound _ _ _ Hb Hnodup id _ cd Hc Hcd Hk).
Qed.

(* ======================================================================== *)
(* The variables that the steps of the traversal put something into         *)
(* ======================================================================== *)

Section Vars.

Variable e0 : expr.

Definition var_ok (x : node) : Prop := ∀ v, x = Node_variable v -> V0 e0 v.

(* A fact puts something into a variable of [V0], or is the alias of
   [traverse_code] into the [my_closure] of code of the unit. *)
Definition var_fact (cds : finite_map code_dep) (fct : fact) : Prop :=
  ∀ v, Node_variable v ∈ into_nodes fct ->
    V0 e0 v
    ∨ ∃ id c cd, (id, c) ∈ unit_codes e0 ∧ code_my_closure c = v
      ∧ (id, cd) ∈ cds ∧ fct = Fact_alias (Node_variable v) (cd_my_closure cd).

(* From [a] to [a'], the facts and the closure deps added are of such
   variables, and no apply dep is added. *)
Definition vsafe (cds : finite_map code_dep) (a a' : acc) : Prop :=
  (∀ fct, fct ∈ acc_facts a' -> fct ∈ acc_facts a ∨ var_fact cds fct)
  ∧ (∀ d, d ∈ acc_closure_deps a' ->
       d ∈ acc_closure_deps a ∨ var_ok (cld_let_bound_name_of_the_closure d))
  ∧ acc_apply_deps a' = acc_apply_deps a.

Lemma var_ok_made (k : positive) : var_ok (Node_made k).
Proof. intros v H. discriminate H. Qed.

Lemma vsafe_refl (cds : finite_map code_dep) (a : acc) : vsafe cds a a.
Proof. split; [ auto | split; auto ]. Qed.

Lemma vsafe_trans (cds : finite_map code_dep) (a a1 a2 : acc) :
  vsafe cds a a1 -> vsafe cds a1 a2 -> vsafe cds a a2.
Proof.
  intros (H1 & H2 & H3) (H4 & H5 & H6). split; [ | split; [ | congruence ] ].
  - intros fct Hin. destruct (H4 fct Hin) as [H | H]; [ | right; exact H ].
    exact (H1 fct H).
  - intros d Hin. destruct (H5 d Hin) as [H | H]; [ | right; exact H ].
    exact (H2 d H).
Qed.

Lemma vsafe_same (cds : finite_map code_dep) (a a' : acc) :
  acc_facts a' = acc_facts a -> acc_closure_deps a' = acc_closure_deps a ->
  acc_apply_deps a' = acc_apply_deps a -> vsafe cds a a'.
Proof.
  intros E1 E2 E3. split; [ | split; [ | exact E3 ] ].
  - intros fct Hin. left. rewrite <- E1. exact Hin.
  - intros d Hin. left. rewrite <- E2. exact Hin.
Qed.

Lemma vsafe_add_facts (cds : finite_map code_dep) (a : acc) (L : list fact) :
  (∀ fct, fct ∈ L -> var_fact cds fct) -> vsafe cds a (add_facts L a).
Proof.
  intros H. split; [ | split; reflexivity || (intros d Hd; left; exact Hd) ].
  intros fct Hin. cbn [add_facts acc_facts] in Hin.
  apply in_app_or in Hin as [Hin | Hin]; [ right; exact (H _ Hin)
                                        | left; exact Hin ].
Qed.

Lemma vsafe_add_fact (cds : finite_map code_dep) (a : acc) (fct : fact) :
  var_fact cds fct -> vsafe cds a (add_fact fct a).
Proof.
  intros H. apply vsafe_add_facts. intros fct' [<- | []]. exact H.
Qed.

Lemma vsafe_fresh (cds : finite_map code_dep) (a : acc) :
  vsafe cds a (snd (fresh a)).
Proof. apply vsafe_same; reflexivity. Qed.

Lemma vsafe_fresh_list (cds : finite_map code_dep) (n : nat) (a : acc) :
  vsafe cds a (snd (fresh_list n a)).
Proof.
  destruct (FlambdaReaperLeastProofs.AutoWidgets.fresh_list_spec n a)
    as (H1 & _ & H3 & H4 & _). apply vsafe_same; assumption.
Qed.

Lemma vsafe_note_simple (cds : finite_map code_dep) (s : simple) (a : acc) :
  vsafe cds a (note_simple s a).
Proof. destruct s; apply vsafe_same; reflexivity. Qed.

Lemma vsafe_note_simples (cds : finite_map code_dep) (l : list simple)
    (a : acc) :
  vsafe cds a (note_simples l a).
Proof.
  destruct (FlambdaReaperLeastProofs.AutoWidgets.note_simples_fields l a)
    as (_ & H1 & _ & H3 & H4). apply vsafe_same; assumption.
Qed.

Lemma vsafe_push_closure_dep (cds : finite_map code_dep) (d : closure_dep)
    (a : acc) :
  var_ok (cld_let_bound_name_of_the_closure d) ->
  vsafe cds a (push_closure_dep d a).
Proof.
  intros H.
  split; [ intros fct Hin; left; exact Hin | split; [ | reflexivity ] ].
  intros d' [<- | Hin]; [ right; exact H | left; exact Hin ].
Qed.

Lemma vsafe_fold {A : Type} (cds : finite_map code_dep) (f : acc -> A -> acc)
    (l : list A) (a : acc) :
  (∀ y b, y ∈ l -> vsafe cds b (f b y)) -> vsafe cds a (List.fold_left f l a).
Proof.
  revert a. induction l as [ | y l IH]; intros a H; cbn [List.fold_left];
    [ apply vsafe_refl | ].
  apply (vsafe_trans _ _ (f a y)); [ apply H; left; reflexivity | ].
  apply IH. intros y' b Hy'. apply H. right. exact Hy'.
Qed.

Ltac into_split H :=
  repeat match type of H with
  | In _ (_ ++ _) => apply in_app_or in H; destruct H as [H | H]
  | In _ (map _ _) =>
    let y := fresh "y" in let Hy := fresh "Hy" in
    apply in_map_iff in H; destruct H as [y [H Hy]]
  | In _ (flat_map _ _) =>
    let y := fresh "y" in let Hy := fresh "Hy" in
    apply in_flat_map in H; destruct H as [y [Hy H]]
  | In _ (_ :: _) => destruct H as [H | H]
  | In _ [] => destruct H
  | In _ (match ?t with _ => _ end) => destruct t
  end.

Ltac vf_tac :=
  let fct := fresh "fct" in let H := fresh "H" in
  intros fct H; into_split H;
  repeat match goal with y : prod _ _ |- _ => destruct y end;
  subst;
  let v := fresh "v" in let Hv := fresh "Hv" in
  intros v Hv; cbn [into_nodes In] in Hv;
  repeat match type of Hv with
  | _ ∨ _ => destruct Hv as [Hv | Hv]
  end;
  try contradiction; try discriminate Hv; left;
  try (injection Hv as <-; assumption);
  try (match goal with H : var_ok ?x |- _ => apply H; exact Hv end).

Lemma vsafe_cond_any_usage (cds : finite_map code_dep) (env : traverse_env)
    (s : simple) (a : acc) :
  vsafe cds a (cond_any_usage env s a).
Proof.
  unfold cond_any_usage.
  apply (vsafe_trans _ _ (note_simple s a)); [ apply vsafe_note_simple | ].
  destruct (env_current_code_id env); apply vsafe_add_fact;
    intros v Hv; cbn in Hv; try contradiction;
    destruct Hv as [Hv | []]; discriminate Hv.
Qed.

Lemma vsafe_cond_alias (cds : finite_map code_dep) (env : traverse_env)
    (from to_ : node) (a : acc) :
  var_ok to_ -> vsafe cds a (cond_alias env from to_ a).
Proof.
  intros H. unfold cond_alias.
  destruct (env_current_code_id env); apply vsafe_add_fact;
    intros v Hv; cbn in Hv; destruct Hv as [Hv | []]; left; exact (H v Hv).
Qed.

(* The parameters of the continuations in scope are of [V0], or made. *)
Definition venv (env : traverse_env) : Prop :=
  ∀ k ps x, (k, ps) ∈ env_conts env -> x ∈ ps -> var_ok x.

Lemma vsafe_find_cont (cds : finite_map code_dep) (env : traverse_env)
    (k : continuation) (n : nat) (a : acc) :
  venv env ->
  vsafe cds a (snd (find_cont env k n a))
  ∧ ∀ x, x ∈ fst (find_cont env k n a) -> var_ok x.
Proof.
  intros Henv. unfold find_cont.
  destruct (lookup k (env_conts env)) as [params | ] eqn:E.
  - split; [ apply vsafe_refl | ]. intros x Hx.
    exact (Henv k params x (FlambdaMapProofs.lookup_in _ _ _ E) Hx).
  - destruct (FlambdaReaperLeastProofs.AutoWidgets.fresh_list_spec n a)
      as (_ & _ & _ & _ & _ & Hm).
    pose proof (vsafe_fresh_list cds n a) as H1.
    destruct (fresh_list n a) as [params b] eqn:Eb. cbn [fst snd] in *.
    split.
    + apply (vsafe_trans _ _ _ _ H1). apply vsafe_add_facts. vf_tac.
    + intros x Hx. destruct (Hm x Hx) as (m & -> & _). apply var_ok_made.
Qed.

Lemma vsafe_block_deps (cds : finite_map code_dep) (base : node)
    (fields : list simple) (a : acc) :
  var_ok base -> vsafe cds a (block_deps base fields a).
Proof.
  intros Hb. unfold block_deps.
  apply (vsafe_trans _ _ (note_simples fields a));
    [ apply vsafe_note_simples | ].
  apply vsafe_add_facts. vf_tac.
Qed.

Lemma vsafe_traverse_prim (cds : finite_map code_dep) (env : traverse_env)
    (x : variable) (p : prim) (args : list simple) (a : acc) :
  V0 e0 x -> vsafe cds a (traverse_prim env x p args a).
Proof.
  intros Hx. assert (Hxo : var_ok (Node_variable x))
    by (intros v Hv; injection Hv as <-; exact Hx).
  unfold traverse_prim.
  assert (Hacc : ∀ f s b,
             vsafe cds b (add_fact (Fact_accessor (Node_variable x) f
                                      (node_of_simple s)) (note_simple s b))).
  { intros f s b. apply (vsafe_trans _ _ (note_simple s b));
      [ apply vsafe_note_simple | ].
    apply vsafe_add_fact. intros v Hv. cbn in Hv.
    destruct Hv as [Hv | []]. left. exact (Hxo v Hv). }
  assert (Hdefault :
    vsafe cds a
      (add_facts
         (Fact_any_source (Node_variable x)
          :: map (fun from => Fact_use (Node_variable x) from)
               (free_names_of_simples args))
         (if prim_has_arbitrary_effects p
          then cond_any_usage env (Var x) a else a))).
  { apply (vsafe_trans _ _ (if prim_has_arbitrary_effects p
                            then cond_any_usage env (Var x) a else a)).
    - destruct (prim_has_arbitrary_effects p);
        [ apply vsafe_cond_any_usage | apply vsafe_refl ].
    - apply vsafe_add_facts. vf_tac. }
  destruct p as [ u | b | v | h | r ];
    [ destruct u | destruct b | destruct v | destruct h | destruct r ];
    repeat match goal with
    | |- vsafe _ _ (match ?t with _ => _ end) => destruct t
    end;
    try exact Hdefault;
    try (apply vsafe_block_deps; exact Hxo);
    try apply Hacc;
    try apply vsafe_refl.
  - match goal with
    | |- vsafe _ _ (add_fact _ ?b) => apply (vsafe_trans _ _ b)
    end; [ apply Hacc | ].
    apply vsafe_add_fact. intros v Hv. cbn in Hv.
    destruct Hv as [Hv | []]. left. exact (Hxo v Hv).
  - apply (vsafe_trans _ _ (note_simple s a)); [ apply vsafe_note_simple | ].
    apply vsafe_add_fact. intros v Hv. cbn in Hv.
    destruct Hv as [Hv | []]. left. exact (Hxo v Hv).
Qed.

Lemma vsafe_record_set (cds : finite_map code_dep) (env : traverse_env)
    (closures : list (node * option code_id))
    (function_slots : list (function_slot * node))
    (value_slots : list (value_slot * simple)) (a : acc) :
  (∀ base code, (base, code) ∈ closures -> var_ok base) ->
  vsafe cds a (record_set_of_closures_deps env closures function_slots
                 value_slots a).
Proof.
  intros H. unfold record_set_of_closures_deps. apply vsafe_fold.
  intros [base code] b Hin. cbv beta iota.
  pose proof (H _ _ Hin) as Hb.
  apply (vsafe_trans _ _ (match code with
                          | Some id => push_closure_dep
                              {| cld_let_bound_name_of_the_closure := base;
                                 cld_closure_code_id := id |} b
                          | None => b
                          end)).
  { destruct code; [ apply vsafe_push_closure_dep; exact Hb
                   | apply vsafe_refl ]. }
  match goal with
  | |- vsafe _ ?b1 (add_facts _ (note_simples ?l ?b1)) =>
    apply (vsafe_trans _ _ (note_simples l b1)); [ apply vsafe_note_simples | ]
  end.
  apply vsafe_add_facts. vf_tac.
Qed.

Lemma vsafe_closures_of_set (cds : finite_map code_dep) (x : variable)
    (slot : function_slot) (others : list (function_slot * variable))
    (decls : list (function_slot * code_id)) (a : acc) :
  V0 e0 x -> (∀ f y, (f, y) ∈ others -> V0 e0 y) ->
  vsafe cds a (snd (closures_of_set x slot others decls a))
  ∧ ∀ f name id, (f, name, id) ∈ fst (closures_of_set x slot others decls a)
      -> var_ok name.
Proof.
  intros Hx Ho. revert a. induction decls as [ | [f id] decls IH]; intros a;
    cbn [closures_of_set].
  - cbn [fst snd]. split; [ apply vsafe_refl | intros f name id [] ].
  - match goal with
    | |- context [let '(_, _) := ?t in _] =>
      assert (Ht : vsafe cds a (snd t) ∧ var_ok (fst t));
      [ | destruct t as [name b] eqn:E ]
    end.
    { destruct (Pos.eqb f slot);
        [ cbn; split; [ apply vsafe_refl | intros v Hv; injection Hv as <- ;
                                          exact Hx ] | ].
      destruct (lookup f others) as [y | ] eqn:Ey;
        [ cbn; split; [ apply vsafe_refl | intros v0 Hv; injection Hv as <- ] |
          split; [ apply vsafe_fresh | apply var_ok_made ] ].
      exact (Ho f y (FlambdaMapProofs.lookup_in _ _ _ Ey)). }
    cbn [fst snd] in Ht. destruct Ht as [H1 Hname].
    destruct (IH b) as [H2 H3].
    destruct (closures_of_set x slot others decls b) as [rest c] eqn:E2.
    cbn [fst snd] in *. split; [ exact (vsafe_trans _ _ _ _ H1 H2) | ].
    intros f' name' id' [Heq | Hin].
    + injection Heq as <- <- <-. exact Hname.
    + exact (H3 _ _ _ Hin).
Qed.

Lemma vsafe_traverse_named (cds : finite_map code_dep) (env : traverse_env)
    (x : variable) (d : named) (a : acc) :
  V0 e0 x -> (∀ f y, (f, y) ∈ set_closures_of env x -> V0 e0 y) ->
  vsafe cds a (traverse_named env x d a).
Proof.
  intros Hx Hsets.
  assert (Hxo : var_ok (Node_variable x))
    by (intros v Hv; injection Hv as <-; exact Hx).
  assert (Hacc : ∀ f s b,
             vsafe cds b (add_fact (Fact_accessor (Node_variable x) f
                                      (node_of_simple s)) (note_simple s b))).
  { intros f s b. apply (vsafe_trans _ _ (note_simple s b));
      [ apply vsafe_note_simple | ].
    apply vsafe_add_fact. intros v Hv. cbn in Hv.
    destruct Hv as [Hv | []]. left. exact (Hxo v Hv). }
  unfold traverse_named.
  destruct d as [ s | p args | id vs | slot decls vs | w c | m w c
                | mf mt c ].
  - apply (vsafe_trans _ _ (note_simple s a)); [ apply vsafe_note_simple | ].
    apply vsafe_add_fact. intros v Hv. cbn in Hv.
    destruct Hv as [Hv | []]. left. exact (Hxo v Hv).
  - apply vsafe_traverse_prim. exact Hx.
  - apply vsafe_record_set. intros base code [Heq | []].
    injection Heq as <- _. exact Hxo.
  - unfold traverse_closure_of_set. cbv zeta.
    destruct (among slot _); [ | apply vsafe_refl ].
    match goal with
    | |- context [closures_of_set ?x ?s ?o ?d a] =>
      destruct (vsafe_closures_of_set cds x s o d a Hx Hsets) as [H1 H2];
      destruct (closures_of_set x s o d a) as [closures b] eqn:E
    end.
    cbn [fst snd] in *. apply (vsafe_trans _ _ _ _ H1).
    apply vsafe_record_set. intros base code Hin.
    apply in_map_iff in Hin. destruct Hin as [[[f name] id] [Heq Hin]].
    injection Heq as <- _. exact (H2 _ _ _ Hin).
  - apply Hacc.
  - apply Hacc.
  - apply Hacc.
Qed.

Lemma vsafe_traverse_static_binding (cds : finite_map code_dep)
    (env : traverse_env) (binding : static_binding) (a : acc) :
  vsafe cds a (traverse_static_binding env binding a).
Proof.
  unfold traverse_static_binding.
  destruct binding as [ | | s const]; [ apply vsafe_refl | apply vsafe_refl | ].
  assert (Hs : var_ok (Node_symbol s)) by (intros v Hv; discriminate Hv).
  destruct const as [ id vs | slot syms decls vs | tag fields
                    | contents | str | [ | f fields] ].
  - apply vsafe_record_set. intros base code [Heq | []].
    injection Heq as <- _. exact Hs.
  - apply vsafe_record_set. intros base code [Heq | []].
    injection Heq as <- _. exact Hs.
  - apply vsafe_block_deps. exact Hs.
  - unfold boxed_number_facts. apply vsafe_add_facts. vf_tac.
  - unfold add_fact. apply vsafe_add_facts. vf_tac.
  - unfold add_fact. apply vsafe_add_facts. vf_tac.
  - apply vsafe_block_deps. exact Hs.
Qed.

Lemma vsafe_create_known (cds : finite_map code_dep) (id : code_id)
    (params returns : list node) (exn : node) (a : acc) :
  (∀ x, x ∈ params -> var_ok x) ->
  vsafe cds a (snd (create_known_arity_call_witness id params returns exn a)).
Proof.
  intros Hp. unfold create_known_arity_call_witness. cbn [fresh snd].
  apply (vsafe_trans _ _ _ _ (vsafe_fresh cds a)). apply vsafe_add_facts.
  vf_tac. apply (Hp _ (FlambdaReaperLeastProofs.AutoWidgets.indexed_in
                         _ _ _ Hy)); exact Hv.
Qed.

Lemma vsafe_witness_deps (cds : finite_map code_dep) (id : code_id)
    (returns : list node) (exn : node) (pw : list (list node * node))
    (a : acc) :
  (∀ first w, (first, w) ∈ pw -> var_ok w ∧ ∀ x, x ∈ first -> var_ok x) ->
  vsafe cds a (unknown_arity_witness_deps id returns exn pw a).
Proof.
  revert a. induction pw as [ | [first w] rest IH]; intros a Hpw;
    cbn [unknown_arity_witness_deps]; [ apply vsafe_refl | ].
  destruct (Hpw first w (or_introl eq_refl)) as [Hw Hf].
  match goal with
  | |- vsafe _ _ (match rest with _ => _ end) =>
    set (b := add_facts _ a);
    assert (H1 : vsafe cds a b)
  end.
  { apply vsafe_add_facts. vf_tac.
    apply (Hf _ (FlambdaReaperLeastProofs.AutoWidgets.indexed_in _ _ _ Hy)).
    exact Hv. }
  apply (vsafe_trans _ _ _ _ H1).
  destruct rest as [ | [g nw] rest'].
  - apply vsafe_add_facts. vf_tac.
  - cbn [fresh]. cbv beta iota.
    match goal with
    | |- vsafe _ _ (unknown_arity_witness_deps _ _ _ _ ?c) =>
      apply (vsafe_trans _ _ c)
    end.
    + apply (vsafe_trans _ _ _ _ (vsafe_fresh cds b)). apply vsafe_add_facts.
      vf_tac.
    + apply IH. intros first' w' Hin. exact (Hpw first' w' (or_intror Hin)).
Qed.

Lemma vsafe_prepare_code (cds : finite_map code_dep) (id : code_id)
    (c : code) (a : acc) :
  vsafe cds a (prepare_code id c a).
Proof.
  unfold prepare_code.
  destruct (FlambdaReaperLeastProofs.AutoWidgets.fresh_list_spec
              (length (code_result_kinds c)) a) as (_ & _ & _ & _ & _ & _).
  pose proof (vsafe_fresh_list cds (length (code_result_kinds c)) a) as H1.
  destruct (fresh_list _ a) as [returns b1] eqn:E1. cbn [fst snd] in *.
  cbn [fresh]. cbv beta iota.
  match goal with
  | |- context [fresh_list ?n ?b] =>
    pose proof (vsafe_trans _ _ _ _
                  (vsafe_same cds b1 b eq_refl eq_refl eq_refl)
                  (vsafe_fresh_list cds n b)) as H2;
    destruct (FlambdaReaperLeastProofs.AutoWidgets.fresh_list_spec n b)
      as (_ & _ & _ & _ & _ & Hm);
    destruct (fresh_list n b) as [params b2] eqn:E2
  end.
  cbn [fst snd] in *. cbv beta iota.
  assert (Hp : ∀ x, x ∈ params -> var_ok x).
  { intros x Hx. destruct (Hm x Hx) as (m & -> & _). apply var_ok_made. }
  pose proof (vsafe_create_known cds id params returns
                (Node_made (acc_next b1)) b2 Hp) as H3.
  destruct (create_known_arity_call_witness _ _ _ _ b2) as [known b3] eqn:E3.
  cbn [fst snd] in *.
  assert (H4 : vsafe cds b3
                 (snd (create_unknown_arity_call_witnesses id (code_is_tupled c)
                         (code_params_arity c) params returns
                         (Node_made (acc_next b1)) b3))).
  { unfold create_unknown_arity_call_witnesses.
    destruct (code_is_tupled c).
    - unfold create_unknown_arity_tupled_call_witnesses. cbn [fresh snd].
      apply (vsafe_trans _ _ _ _ (vsafe_fresh cds _)).
      apply (vsafe_trans _ _ _ _ (vsafe_fresh cds _)).
      apply (vsafe_trans _ _ _ _ (vsafe_fresh cds _)).
      apply vsafe_add_facts. vf_tac.
      apply (Hp _ (FlambdaReaperLeastProofs.AutoWidgets.indexed_in
                     _ _ _ Hy)). exact Hv.
    - unfold create_unknown_arity_non_tupled_call_witnesses. cbv zeta.
      match goal with
      | |- context [fresh_list ?n ?b] =>
        pose proof (vsafe_fresh_list cds n b) as H5;
        destruct (fresh_list n b) as [ws b4] eqn:E4
      end.
      cbn [fst snd] in *. apply (vsafe_trans _ _ _ _ H5).
      apply vsafe_witness_deps. intros first w Hin. split.
      + destruct (FlambdaReaperLeastProofs.AutoWidgets.fresh_list_spec
                    (length (at_least_one_group
                               (group_by_parameter (code_params_arity c)
                                  params))) b3)
          as (_ & _ & _ & _ & _ & Hm').
        rewrite E4 in Hm'. cbn [fst] in Hm'.
        destruct (Hm' w (in_combine_r _ _ _ _ Hin)) as (m & -> & _).
        apply var_ok_made.
      + intros x Hx. apply in_combine_l in Hin.
        exact (Hp x (FlambdaReaperLeastProofs.AutoWidgets.group_in _ _ _ _ Hin
                       Hx)). }
  destruct (create_unknown_arity_call_witnesses _ _ _ _ _ _ b3)
    as [unknown b4] eqn:E4. cbn [fst snd] in *. cbv beta iota.
  apply (vsafe_trans _ _ b1); [ exact H1 | ].
  apply (vsafe_trans _ _ b2); [ exact H2 | ].
  apply (vsafe_trans _ _ b3); [ exact H3 | ].
  apply (vsafe_trans _ _ b4); [ exact H4 | ].
  split; [ | split; reflexivity || (intros d Hd; left; exact Hd) ].
  intros fct Hin. cbn [add_code_dep add_fact add_facts acc_facts] in Hin.
  destruct Hin as [<- | Hin]; [ right | left; exact Hin ].
  intros v Hv. cbn in Hv. destruct Hv as [Hv | []]. discriminate Hv.
Qed.

(* Without the condition on the apply deps, for calls. *)
Definition vsafe0 (cds : finite_map code_dep) (a a' : acc) : Prop :=
  (∀ fct, fct ∈ acc_facts a' -> fct ∈ acc_facts a ∨ var_fact cds fct)
  ∧ (∀ d, d ∈ acc_closure_deps a' ->
       d ∈ acc_closure_deps a ∨ var_ok (cld_let_bound_name_of_the_closure d)).

Lemma vsafe_vsafe0 (cds : finite_map code_dep) (a a' : acc) :
  vsafe cds a a' -> vsafe0 cds a a'.
Proof. intros (H1 & H2 & _). split; assumption. Qed.

Lemma vsafe0_trans (cds : finite_map code_dep) (a a1 a2 : acc) :
  vsafe0 cds a a1 -> vsafe0 cds a1 a2 -> vsafe0 cds a a2.
Proof.
  intros (H1 & H2) (H4 & H5). split.
  - intros fct Hin. destruct (H4 fct Hin) as [H | H]; [ | right; exact H ].
    exact (H1 fct H).
  - intros d Hin. destruct (H5 d Hin) as [H | H]; [ | right; exact H ].
    exact (H2 d H).
Qed.

Lemma vsafe_make_known (cds : finite_map code_dep) (env : traverse_env)
    (ap : apply_expr) (returns : list node) (exn : node) (a : acc) :
  (∀ x, x ∈ exn :: returns -> var_ok x) ->
  vsafe cds a (snd (make_known_arity_apply_widget env ap returns exn a))
  ∧ var_ok (fst (make_known_arity_apply_widget env ap returns exn a)).
Proof.
  intros Hin. unfold make_known_arity_apply_widget.
  cbn [fresh]. cbv beta iota zeta. cbn [fst snd].
  split; [ | unfold cond_alias; destruct (env_current_code_id env);
             apply var_ok_made ].
  match goal with
  | |- vsafe _ _ (cond_alias _ _ _ (add_facts _ (note_simples _ ?b))) =>
    apply (vsafe_trans _ _ b); [ apply vsafe_same; reflexivity | ];
    apply (vsafe_trans _ _ (note_simples (ap_args ap) b));
    [ apply vsafe_note_simples | ]
  end.
  match goal with
  | |- vsafe _ _ (cond_alias _ _ _ ?c) =>
    apply (vsafe_trans _ _ c); [ | apply vsafe_cond_alias, var_ok_made ]
  end.
  apply vsafe_add_facts. vf_tac.
  - apply (Hin _ (or_intror
                    (FlambdaReaperLeastProofs.AutoWidgets.indexed_in
                       _ _ _ Hy))).
    exact Hv.
  - apply (Hin _ (or_introl eq_refl)). exact Hv.
Qed.

Lemma vsafe_unknown_apply_deps (cds : finite_map code_dep)
    (returns : list node) (exn called : node)
    (aw : list (list simple * node)) (a : acc) :
  (∀ x, x ∈ exn :: called :: returns -> var_ok x) ->
  (∀ first w, (first, w) ∈ aw -> var_ok w) ->
  vsafe cds a (unknown_arity_apply_deps returns exn called aw a).
Proof.
  revert a. induction aw as [ | [first w] rest IH]; intros a Hin Haw;
    cbn [unknown_arity_apply_deps]; [ apply vsafe_refl | ].
  pose proof (Hin exn (or_introl eq_refl)) as He.
  pose proof (Hin called (or_intror (or_introl eq_refl))) as Hc.
  match goal with
  | |- vsafe _ _ (match rest with _ => _ end) =>
    set (b := add_facts _ a);
    assert (H1 : vsafe cds a b)
  end.
  { apply vsafe_add_facts. vf_tac. }
  apply (vsafe_trans _ _ _ _ H1).
  destruct rest as [ | [g nw] rest'].
  - apply vsafe_add_facts. vf_tac.
    apply (Hin _ (or_intror (or_intror
                    (FlambdaReaperLeastProofs.AutoWidgets.indexed_in
                       _ _ _ Hy)))).
    exact Hv.
  - cbn [fresh]. cbv beta iota.
    pose proof (Haw g nw (or_intror (or_introl eq_refl))) as Hnw.
    match goal with
    | |- vsafe _ _ (unknown_arity_apply_deps _ _ _ _ ?c) =>
      apply (vsafe_trans _ _ c)
    end.
    + apply (vsafe_trans _ _ _ _ (vsafe_fresh cds b)). apply vsafe_add_facts.
      vf_tac.
    + apply IH; [ exact Hin | ]. intros first' w' Hw'.
      exact (Haw _ _ (or_intror Hw')).
Qed.

Lemma vsafe_make_unknown (cds : finite_map code_dep) (env : traverse_env)
    (ap : apply_expr) (returns : list node) (exn : node) (a : acc) :
  (∀ x, x ∈ exn :: returns -> var_ok x) ->
  vsafe cds a (snd (make_unknown_arity_apply_widget env ap returns exn a))
  ∧ var_ok (fst (make_unknown_arity_apply_widget env ap returns exn a)).
Proof.
  intros Hin. unfold make_unknown_arity_apply_widget. cbn [fresh].
  cbv beta iota zeta.
  match goal with
  | |- context [fresh_list ?n ?b] =>
    assert (H1 : vsafe cds a b);
    [ | pose proof (vsafe_fresh_list cds n b) as H2;
        destruct (FlambdaReaperLeastProofs.AutoWidgets.fresh_list_spec n b)
          as (_ & _ & _ & _ & _ & Hm);
        destruct (fresh_list n b) as [ws b2] eqn:E2 ]
  end.
  { unfold add_fact. apply (vsafe_trans _ _ _ _ (vsafe_fresh cds a)).
    apply vsafe_add_facts. vf_tac. }
  cbn [fst snd] in *. cbv beta iota zeta.
  pose proof (vsafe_note_simples cds (ap_args ap) b2) as H3.
  match goal with
  | |- context [unknown_arity_apply_deps ?r ?ex ?cl ?aw (note_simples ?l b2)]
    =>
    pose proof (vsafe_unknown_apply_deps cds r ex cl aw (note_simples l b2))
      as H4
  end.
  destruct H4 as [H4a H4b].
  { intros x [<- | [<- | Hx]];
      [ exact (Hin _ (or_introl eq_refl)) | apply var_ok_made
      | exact (Hin _ (or_intror Hx)) ]. }
  { intros first w Hw. destruct (Hm w (in_combine_r _ _ _ _ Hw))
      as (m & -> & _). apply var_ok_made. }
  set (b4 := unknown_arity_apply_deps _ _ _ _ _) in *.
  assert (H45 : vsafe cds a (snd (fresh b4))).
  { apply (vsafe_trans _ _ _ _ H1). apply (vsafe_trans _ _ _ _ H2).
    apply (vsafe_trans _ _ _ _ H3).
    apply (vsafe_trans _ _ b4); [ exact (conj H4a H4b) | ].
    apply vsafe_fresh. }
  cbn [fresh fst snd] in *.
  destruct ws as [ | first rest]; cbn [fst snd].
  - split; [ exact H45 | apply var_ok_made ].
  - split; [ | apply var_ok_made ].
    apply (vsafe_trans _ _ _ _ H45). apply vsafe_cond_alias.
    destruct (Hm first (or_introl eq_refl)) as (m & -> & _).
    apply var_ok_made.
Qed.

Lemma vsafe0_add_apply (cds : finite_map code_dep) (env : traverse_env)
    (id : code_id) (callee call_widget : node) (b : bool) (a : acc) :
  var_ok call_widget ->
  vsafe0 cds a (add_apply env id callee call_widget b a).
Proof.
  intros Hcw. unfold add_apply. destruct b.
  - cbn [fresh]. cbv beta iota. split.
    + intros fct Hin. cbn [push_apply_dep acc_facts add_facts] in Hin.
      apply in_app_or in Hin as [Hin | Hin]; [ right | left; exact Hin ].
      revert fct Hin. vf_tac.
    + intros d Hd. left. exact Hd.
  - split; [ intros fct Hin; left; exact Hin | intros d Hd; left; exact Hd ].
Qed.

Lemma vsafe0_call_kind (cds : finite_map code_dep) (env : traverse_env)
    (ap : apply_expr) (exn_arg : node) (return_args : list node) (a : acc) :
  (∀ x, x ∈ exn_arg :: return_args -> var_ok x) ->
  vsafe0 cds a (traverse_call_kind env ap exn_arg return_args a).
Proof.
  intros Hin. unfold traverse_call_kind. cbv zeta.
  apply (vsafe0_trans _ _ (note_simple (ap_callee ap) a));
    [ apply vsafe_vsafe0, vsafe_note_simple | ].
  set (a1 := note_simple (ap_callee ap) a).
  destruct (ap_call_kind ap) as [id | | code_ids].
  - destruct (vsafe_make_known cds env ap return_args exn_arg a1 Hin)
      as [Hk Hcw].
    destruct (make_known_arity_apply_widget env ap return_args exn_arg a1)
      as [cw a2] eqn:E. cbn [fst snd] in *. cbv beta iota zeta.
    set (a3 := add_fact _ a2).
    assert (H3 : vsafe cds a1 a3).
    { apply (vsafe_trans _ _ _ _ Hk). unfold a3, add_fact.
      apply vsafe_add_facts. vf_tac. }
    apply (vsafe0_trans _ _ a3); [ apply vsafe_vsafe0, H3 | ].
    destruct (env_should_preserve_direct_calls env);
      [ | destruct (negb (among id (env_unit_code_ids env))) | ];
      try (apply vsafe_vsafe0, vsafe_refl); apply vsafe0_add_apply; exact Hcw.
  - destruct (vsafe_make_unknown cds env ap return_args exn_arg a1 Hin)
      as [Hk Hcw].
    destruct (make_unknown_arity_apply_widget env ap return_args exn_arg a1)
      as [cw a2] eqn:E. cbn [fst snd] in *. cbv beta iota.
    apply vsafe_vsafe0. apply (vsafe_trans _ _ _ _ Hk). unfold add_fact.
    apply vsafe_add_facts. vf_tac.
  - destruct (vsafe_make_known cds env ap return_args exn_arg a1 Hin)
      as [Hk Hcw].
    destruct (make_known_arity_apply_widget env ap return_args exn_arg a1)
      as [cw a2] eqn:E. cbn [fst snd] in *. cbv beta iota zeta.
    apply vsafe_vsafe0. apply (vsafe_trans _ _ _ _ Hk). unfold add_fact.
    apply vsafe_add_facts. vf_tac.
Qed.

Lemma vsafe0_traverse_apply (cds : finite_map code_dep) (env : traverse_env)
    (ap : apply_expr) (a : acc) :
  venv env -> vsafe0 cds a (traverse_apply env ap a).
Proof.
  intros Henv. unfold traverse_apply.
  destruct (vsafe_find_cont cds env (ap_continuation ap)
              (length (unarize (ap_return_arity ap))) a Henv) as [H1 H1o].
  destruct (find_cont env (ap_continuation ap) _ a) as [rets b1] eqn:E1.
  cbn [fst snd] in *. cbv beta iota.
  destruct (vsafe_find_cont cds env (ap_exn_continuation ap)
              (S (length (ap_exn_extra_args ap))) b1 Henv) as [H2 H2o].
  destruct (find_cont env (ap_exn_continuation ap) _ b1) as [eps b2] eqn:E2.
  cbn [fst snd] in *. cbv beta iota.
  apply (vsafe0_trans _ _ b1); [ apply vsafe_vsafe0, H1 | ].
  apply (vsafe0_trans _ _ b2); [ apply vsafe_vsafe0, H2 | ].
  destruct eps as [ | ep extra].
  - cbn [fresh]. cbv beta iota.
    match goal with
    | |- vsafe0 _ _ (traverse_call_kind _ _ _ _ ?b3) =>
      apply (vsafe0_trans _ _ b3);
        [ apply vsafe_vsafe0, vsafe_same; reflexivity | ]
    end.
    apply vsafe0_call_kind. intros x [<- | Hx];
      [ apply var_ok_made | exact (H1o x Hx) ].
  - cbv zeta beta iota.
    pose proof (vsafe_note_simples cds (exn_extra_simples ap) b2) as H3.
    set (b3 := note_simples _ b2) in *.
    match goal with
    | |- vsafe0 _ _ (traverse_call_kind _ _ _ _ ?b4) =>
      apply (vsafe0_trans _ _ b4)
    end.
    + apply vsafe_vsafe0. apply (vsafe_trans _ _ _ _ H3).
      apply vsafe_add_facts. vf_tac.
      match goal with
      | Hc : (_, _) ∈ combine _ _ |- _ =>
        apply (H2o _ (or_intror (in_combine_l _ _ _ _ Hc))); exact Hv
      end.
    + apply vsafe0_call_kind. intros x [<- | Hx];
        [ exact (H2o _ (or_introl eq_refl)) | exact (H1o x Hx) ].
Qed.

Lemma vsafe_apply_cont_deps (cds : finite_map code_dep) (env : traverse_env)
    (ac : apply_cont) (a : acc) :
  venv env -> vsafe cds a (apply_cont_deps env ac a).
Proof.
  intros Henv. unfold apply_cont_deps. cbv zeta.
  destruct (vsafe_find_cont cds env (ac_continuation ac) (length (ac_args ac))
              a Henv) as [H1 H1o].
  destruct (find_cont env (ac_continuation ac) _ a) as [params b1] eqn:E1.
  cbn [fst snd] in *. cbv beta iota zeta.
  apply (vsafe_trans _ _ _ _ H1).
  apply (vsafe_trans _ _ _ _ (vsafe_note_simples cds (ac_args ac) b1)).
  apply vsafe_add_facts. vf_tac.
  match goal with
  | Hc : (_, _) ∈ combine _ _ |- _ =>
    apply (H1o _ (in_combine_l _ _ _ _ Hc)); exact Hv
  end.
Qed.

Lemma vsafe_traverse_switch (cds : finite_map code_dep) (env : traverse_env)
    (scrutinee : simple) (arms : list (Z * apply_cont)) (a : acc) :
  venv env -> vsafe cds a (traverse_switch env scrutinee arms a).
Proof.
  intros Henv. unfold traverse_switch. cbv zeta.
  apply (vsafe_trans _ _ _ _ (vsafe_cond_any_usage cds env scrutinee a)).
  apply vsafe_fold. intros [i ac] b _. apply vsafe_apply_cont_deps.
  exact Henv.
Qed.

Lemma vsafe_exn_handler_deps (cds : finite_map code_dep) (env : traverse_env)
    (h : continuation_handler) (a : acc) :
  (∀ p, p ∈ handler_params h -> var_ok p) ->
  vsafe cds a (exn_handler_deps env h a).
Proof.
  intros Hp. unfold exn_handler_deps.
  destruct h as [[ | p ps] md body]; [ apply vsafe_refl | ].
  destruct (hm_is_exn_handler md); [ | apply vsafe_refl ].
  match goal with
  | |- vsafe _ _ (cond_any_usage _ _ ?b) => apply (vsafe_trans _ _ b)
  end; [ | apply vsafe_cond_any_usage ].
  apply vsafe_add_fact. intros v Hv. cbn in Hv. destruct Hv as [Hv | []].
  left. apply (Hp (Node_variable p) (or_introl eq_refl)). exact Hv.
Qed.

End Vars.

(* ======================================================================== *)
(* The invariant on the made nodes                                          *)
(* ======================================================================== *)

Module KW := FlambdaReaperWrapperAutoProofs.KnownWitnesses.

(* The nodes [cd_my_closure] of the code deps and [ad_apply_closure] of the
   apply deps. *)
Definition specials (a : acc) : list node :=
  map (fun p => cd_my_closure (snd p)) (acc_code_deps a)
  ++ map ad_apply_closure (acc_apply_deps a).

Definition good (a : acc) (x : node) : Prop :=
  made_lt x (acc_next a) ∧ x ∉ specials a.

(* An accumulator whose witnesses of known arity are the nodes [sp], for
   the steps of [KnownWitnesses]. *)
Definition gdep (x : node) : code_dep :=
  {| cd_params := []; cd_my_closure := x; cd_return := []; cd_exn := x;
     cd_known_arity_call_witness := x; cd_unknown_arity_call_witnesses := [] |}.

Definition ghost (n : positive) (sp : list node) : acc :=
  {| acc_next := n; acc_facts := [];
     acc_code_deps := map (fun x => (xH, gdep x)) sp;
     acc_apply_deps := []; acc_closure_deps := []; acc_symbols := [] |}.

Lemma kwit_ghost (n : positive) (sp : list node) (x : node) :
  KW.kwit (ghost n sp) x <-> x ∈ sp.
Proof.
  unfold KW.kwit. cbn [ghost acc_code_deps]. split.
  - intros (id & cd & H & ->). apply in_map_iff in H as (y & [= <- <-] & Hy).
    exact Hy.
  - intros H. exists xH, (gdep x). split; [ | reflexivity ].
    apply in_map_iff. exists x. auto.
Qed.

Lemma ok_ghost (n : positive) (sp : list node) (a : acc) (x : node) :
  (∀ y, y ∈ sp -> ∃ k, y = Node_made k ∧ (k < n)%positive) ->
  (n <= acc_next a)%positive ->
  KW.ok_node (ghost n sp) a x ->
  x ∉ sp ∧ made_lt x (acc_next a)
  ∧ (not_made x ∨ made_lt x n ∨ ∃ k, x = Node_made k ∧ (n <= k)%positive).
Proof.
  intros Hsp Hn [H | [[H1 H2] | (k & -> & H1 & H2)]].
  - split; [ | split; [ intros k E; destruct (H k E) | left; exact H ] ].
    intros Hx. destruct (Hsp x Hx) as (k & -> & _). exact (H k eq_refl).
  - split; [ intros Hx; apply H2, kwit_ghost; exact Hx | ].
    split; [ intros k E; specialize (H1 k E); cbn in H1; lia | ].
    right; left. exact H1.
  - cbn [ghost acc_next] in H1. split.
    + intros Hx. destruct (Hsp _ Hx) as (k' & [= <-] & Hk). lia.
    + split; [ intros k' [= <-]; exact H2 | right; right; eauto ].
Qed.

Section Inv.

Variable e0 : expr.

(* The direct call of the apply dep [d] in the facts [l], whose closure
   gets only the alias on any source of its callee. *)
Definition dep_chain (l : list fact) (d : apply_dep) : Prop :=
  ∃ callee widget witness called,
    chain l d callee widget witness called
    ∧ ∀ fct, fct ∈ l -> ad_apply_closure d ∈ into_nodes fct ->
        fct = Fact_alias_if_any_source callee (ad_apply_closure d) callee.

Record inv (a : acc) : Prop := {
  i_into : ∀ fct x,
    fct ∈ acc_facts a -> x ∈ into_nodes fct -> made_lt x (acc_next a);
  i_specials : ∀ x,
    x ∈ specials a -> ∃ k, x = Node_made k ∧ (k < acc_next a)%positive;
  i_cd_nodes : ∀ id cd x,
    (id, cd) ∈ acc_code_deps a -> x ∈ cd_exn cd :: cd_return cd ->
    good a x ∧ var_ok e0 x;
  i_cd_inj : ∀ id1 cd1 id2 cd2,
    (id1, cd1) ∈ acc_code_deps a -> (id2, cd2) ∈ acc_code_deps a ->
    cd_my_closure cd1 = cd_my_closure cd2 -> id1 = id2 ∧ cd1 = cd2;
  i_cd_ad : ∀ id cd d,
    (id, cd) ∈ acc_code_deps a -> d ∈ acc_apply_deps a ->
    cd_my_closure cd <> ad_apply_closure d;
  i_cd_into : ∀ id cd fct,
    (id, cd) ∈ acc_code_deps a -> fct ∈ acc_facts a ->
    cd_my_closure cd ∉ into_nodes fct;
  i_ad_wif : ∀ d,
    d ∈ acc_apply_deps a ->
    good a (ad_apply_call_witness d) ∧ var_ok e0 (ad_apply_call_witness d);
  i_ad_chain : ∀ d, d ∈ acc_apply_deps a -> dep_chain (acc_facts a) d;
  i_cl : ∀ d,
    d ∈ acc_closure_deps a ->
    (not_made (cld_let_bound_name_of_the_closure d)
     ∨ good a (cld_let_bound_name_of_the_closure d))
    ∧ var_ok e0 (cld_let_bound_name_of_the_closure d);
  i_var : ∀ fct, fct ∈ acc_facts a -> var_fact e0 (acc_code_deps a) fct
}.

(* The continuations in scope have good nodes of [V0], with [Auto] and the
   closures of the sets of the unit. *)
Record menv (env : traverse_env) (a : acc) : Prop := {
  me_conts : ∀ k ps x,
    (k, ps) ∈ env_conts env -> x ∈ ps -> good a x ∧ var_ok e0 x;
  me_auto : env_should_preserve_direct_calls env = Preserve_auto;
  me_sets : env_set_closures env = set_closures e0
}.

Definition grows (a a' : acc) : Prop :=
  (acc_next a <= acc_next a')%positive
  ∧ (∀ x, x ∈ specials a' ->
       x ∈ specials a ∨ ∃ k, x = Node_made k ∧ (acc_next a <= k)%positive)
  ∧ (∀ p, p ∈ acc_code_deps a -> p ∈ acc_code_deps a').

Lemma grows_refl (a : acc) : grows a a.
Proof. split; [ lia | split; auto ]. Qed.

Lemma grows_trans (a a1 a2 : acc) : grows a a1 -> grows a1 a2 -> grows a a2.
Proof.
  intros (H1 & H2 & H3) (H4 & H5 & H6). split; [ lia | split; [ | auto ] ].
  intros x Hx. destruct (H5 x Hx) as [H | (k & -> & Hk)].
  - destruct (H2 x H) as [H' | (k & -> & Hk)]; [ left; exact H' | ].
    right. exists k. split; [ reflexivity | lia ].
  - right. exists k. split; [ reflexivity | lia ].
Qed.

Lemma good_grows (a a' : acc) (x : node) :
  grows a a' -> good a x -> good a' x.
Proof.
  intros (Hn & Hs & _) [Hlt Hx].
  split; [ intros k E; specialize (Hlt k E); lia | ].
  intros Hx'. destruct (Hs x Hx') as [H | (k & -> & Hk)]; [ exact (Hx H) | ].
  specialize (Hlt k eq_refl). lia.
Qed.

Lemma menv_grows (env : traverse_env) (a a' : acc) :
  grows a a' -> menv env a -> menv env a'.
Proof.
  intros Hg [Hc Ha Hs]. constructor; [ | exact Ha | exact Hs ].
  intros k ps x Hk Hx. destruct (Hc k ps x Hk Hx) as [H1 H2].
  split; [ exact (good_grows _ _ _ Hg H1) | exact H2 ].
Qed.

Definition mstep (env : traverse_env) (a a' : acc) : Prop :=
  inv a -> menv env a -> inv a' ∧ grows a a'.

Lemma mstep_refl (env : traverse_env) (a : acc) : mstep env a a.
Proof. intros H _. split; [ exact H | apply grows_refl ]. Qed.

Lemma mstep_trans (env : traverse_env) (a a1 a2 : acc) :
  mstep env a a1 -> mstep env a1 a2 -> mstep env a a2.
Proof.
  intros H1 H2 Hinv Henv. destruct (H1 Hinv Henv) as [Hinv1 Hg1].
  destruct (H2 Hinv1 (menv_grows _ _ _ Hg1 Henv)) as [Hinv2 Hg2].
  split; [ exact Hinv2 | exact (grows_trans _ _ _ Hg1 Hg2) ].
Qed.

Lemma mstep_fold {A : Type} (env : traverse_env) (f : acc -> A -> acc)
    (l : list A) (a : acc) :
  (∀ y b, y ∈ l -> mstep env b (f b y)) -> mstep env a (List.fold_left f l a).
Proof.
  revert a. induction l as [ | y l IH]; intros a H; cbn [List.fold_left].
  - apply mstep_refl.
  - apply (mstep_trans _ _ (f a y)); [ apply H; left; reflexivity | ].
    apply IH. intros y' b Hy'. apply H. right. exact Hy'.
Qed.

Lemma mstep_env (env env' : traverse_env) (a a' : acc) :
  (inv a -> menv env a -> menv env' a) -> mstep env' a a' -> mstep env a a'.
Proof. intros H H' Hinv Henv. exact (H' Hinv (H Hinv Henv)). Qed.

Lemma menv_ghost (env : traverse_env) (a : acc) :
  menv env a -> KW.env_good env (ghost (acc_next a) (specials a)).
Proof.
  intros Henv k ps x Hk Hx. destruct (me_conts _ _ Henv k ps x Hk Hx)
    as [[Hlt Hs] _].
  split; [ exact Hlt | intros Hw; exact (Hs (proj1 (kwit_ghost _ _ _) Hw)) ].
Qed.

Lemma menv_venv (env : traverse_env) (a : acc) :
  menv env a -> venv e0 env.
Proof.
  intros Henv k ps x Hk Hx. exact (proj2 (me_conts _ _ Henv k ps x Hk Hx)).
Qed.

Lemma var_fact_mono (cds cds' : finite_map code_dep) (fct : fact) :
  (∀ p, p ∈ cds -> p ∈ cds') -> var_fact e0 cds fct -> var_fact e0 cds' fct.
Proof.
  intros Hc H v Hv.
  destruct (H v Hv) as [H' | (id & c & cd & H1 & H2 & H3 & H4)];
    [ left; exact H' | right ].
  exists id, c, cd. auto.
Qed.

Lemma cd_special (a : acc) (id : code_id) (cd : code_dep) :
  (id, cd) ∈ acc_code_deps a -> cd_my_closure cd ∈ specials a.
Proof.
  intros H. apply in_or_app. left. apply in_map_iff. exists (id, cd). auto.
Qed.

Lemma ad_special (a : acc) (d : apply_dep) :
  d ∈ acc_apply_deps a -> ad_apply_closure d ∈ specials a.
Proof. intros H. apply in_or_app. right. apply in_map. exact H. Qed.

(* A new apply dep of a call with [Auto]. *)
Definition newdep (a a' : acc) (d : apply_dep) : Prop :=
  ∃ k, ad_apply_closure d = Node_made k ∧ (acc_next a <= k)%positive
    ∧ ad_apply_call_witness d = Node_made (Pos.succ k)
    ∧ (Pos.succ k < acc_next a')%positive
    ∧ dep_chain (acc_facts a') d.

(* A step whose facts put something into nodes that [KnownWitnesses] says
   are not special in [a], and only into variables of [V0], with no code
   dep and no apply dep but that of a call. *)
Lemma mstep_core (a a' : acc) :
  inv a ->
  KW.ext (ghost (acc_next a) (specials a)) a a' ->
  vsafe0 e0 (acc_code_deps a) a a' ->
  (acc_apply_deps a' = acc_apply_deps a
   ∨ ∃ d, acc_apply_deps a' = d :: acc_apply_deps a ∧ newdep a a' d
     ∧ acc_closure_deps a' = acc_closure_deps a) ->
  inv a' ∧ grows a a'.
Proof.
  intros Hinv He [Vf Vc] Had.
  destruct He as [Hn (L & Hf & Hi & _) Hc _ Hcl].
  pose proof (fun x => ok_ghost (acc_next a) (specials a) a' x
                         (i_specials a Hinv) Hn) as Hok.
  assert (HL : ∀ fct x, fct ∈ acc_facts a' -> x ∈ into_nodes fct ->
             fct ∈ acc_facts a
             ∨ (x ∉ specials a ∧ made_lt x (acc_next a'))).
  { intros fct x Hin Hx. rewrite Hf in Hin.
    apply in_app_or in Hin as [Hin | Hin]; [ right | left; exact Hin ].
    destruct (Hok x (Hi _ _ Hin Hx)) as (H1 & H2 & _). auto. }
  assert (Hsp : ∀ x, x ∈ specials a' ->
             x ∈ specials a
             ∨ ∃ d, acc_apply_deps a' = d :: acc_apply_deps a
               ∧ newdep a a' d ∧ x = ad_apply_closure d).
  { intros x Hx. unfold specials in Hx. rewrite Hc in Hx.
    apply in_app_or in Hx as [Hx | Hx];
      [ left; apply in_or_app; left; exact Hx | ].
    destruct Had as [E | (d & E & Hd & _)]; rewrite E in Hx;
      [ left; apply in_or_app; right; exact Hx | ].
    destruct Hx as [<- | Hx]; [ right; exists d; auto | ].
    left. apply in_or_app. right. exact Hx. }
  assert (Hgrows : grows a a').
  { split; [ exact Hn | split ].
    - intros x Hx.
      destruct (Hsp x Hx) as [H | (d & _ & (k & Hk & Hle & _) & ->)];
        [ left; exact H | right; exists k; auto ].
    - intros p Hp. rewrite Hc. exact Hp. }
  assert (Hgood : ∀ x, good a x -> good a' x)
    by (intros x; exact (good_grows _ _ _ Hgrows)).
  assert (Hold : ∀ d, d ∈ acc_apply_deps a' ->
             d ∈ acc_apply_deps a ∨ (newdep a a' d
               ∧ acc_apply_deps a' = d :: acc_apply_deps a)).
  { intros d Hd. destruct Had as [E | (d' & E & Hd' & _)]; rewrite E in Hd;
      [ left; exact Hd | ].
    destruct Hd as [<- | Hd]; [ right; split; [ exact Hd' | exact E ]
                             | left; exact Hd ]. }
  split; [ constructor | exact Hgrows ].
  - intros fct x Hin Hx. destruct (HL fct x Hin Hx) as [H | [_ H]];
      [ | exact H ].
    intros k E. specialize (i_into a Hinv fct x H Hx k E). lia.
  - intros x Hx. destruct (Hsp x Hx)
      as [H | (d & _ & (k & Hk & _ & _ & Hlt & _) & ->)].
    + destruct (i_specials a Hinv x H) as (k & -> & Hk).
      exists k. split; [ reflexivity | lia ].
    + exists k. split; [ exact Hk | lia ].
  - intros id cd x Hin Hx. rewrite Hc in Hin.
    destruct (i_cd_nodes a Hinv id cd x Hin Hx) as [H1 H2].
    split; [ exact (Hgood _ H1) | exact H2 ].
  - intros id1 cd1 id2 cd2 H1 H2. rewrite Hc in H1, H2.
    exact (i_cd_inj a Hinv _ _ _ _ H1 H2).
  - intros id cd d Hin Hd. rewrite Hc in Hin.
    destruct (Hold d Hd) as [Hd' | [(k & Hk & Hle & _) _]];
      [ exact (i_cd_ad a Hinv _ _ _ Hin Hd') | ].
    destruct (i_specials a Hinv _ (cd_special _ _ _ Hin)) as (k' & Hk' & Hlt).
    rewrite Hk, Hk'. intros [= ->]. lia.
  - intros id cd fct Hin Hfct Hx. rewrite Hc in Hin.
    destruct (HL fct _ Hfct Hx) as [H | [H _]];
      [ exact (i_cd_into a Hinv _ _ _ Hin H Hx)
      | exact (H (cd_special _ _ _ Hin)) ].
  - intros d Hd. destruct (Hold d Hd)
      as [Hd' | [(k & Hk & Hle & Hw & Hlt & _) E]].
    + destruct (i_ad_wif a Hinv d Hd') as [H1 H2].
      split; [ exact (Hgood _ H1) | exact H2 ].
    + rewrite Hw. split; [ | apply var_ok_made ].
      split; [ intros k' [= <-]; exact Hlt | ].
      intros Hx. destruct (Hsp _ Hx) as [H | (d' & E' & _ & Hx')].
      * destruct (i_specials a Hinv _ H) as (k' & [= <-] & Hk'). lia.
      * rewrite E in E'. injection E' as <-. rewrite Hk in Hx'.
        injection Hx'. lia.
  - intros d Hd.
    destruct (Hold d Hd) as [Hd' | [(k & _ & _ & _ & _ & Hch) _]];
      [ | exact Hch ].
    destruct (i_ad_chain a Hinv d Hd')
      as (callee & widget & witness & called & Hchain & Hinto).
    exists callee, widget, witness, called. split.
    + destruct Hchain as (C1 & C2 & C3 & C4 & C5).
      rewrite Hf. repeat split; apply in_or_app; right; assumption.
    + intros fct Hin Hx. destruct (HL fct _ Hin Hx) as [H | [H _]];
        [ exact (Hinto fct H Hx) | exfalso; exact (H (ad_special a d Hd')) ].
  - intros d Hd.
    assert (Hd0 : d ∈ acc_closure_deps a
                  ∨ (var_ok e0 (cld_let_bound_name_of_the_closure d)
                     ∧ acc_apply_deps a' = acc_apply_deps a
                     ∧ KW.ok_node (ghost (acc_next a) (specials a)) a'
                         (cld_let_bound_name_of_the_closure d))).
    { destruct Had as [E | (d' & _ & _ & E)];
        [ | left; rewrite <- E; exact Hd ].
      destruct (Vc d Hd) as [H | Hv]; [ left; exact H | ].
      destruct (Hcl d Hd) as [H | Hb]; [ left; exact H | ].
      right. auto. }
    destruct Hd0 as [Hd' | (Hv & E & Hb)].
    + destruct (i_cl a Hinv d Hd') as [[H | H] H2];
        [ split; [ left; exact H | exact H2 ]
        | split; [ right; exact (Hgood _ H) | exact H2 ] ].
    + split; [ | exact Hv ].
      destruct (Hok _ Hb) as (H1 & H2 & [H3 | [H3 | (k & Hk & Hle)]]);
        [ left; exact H3 | right; exact (Hgood _ (conj H3 H1)) | right ].
      split; [ exact H2 | ]. intros Hx. unfold specials in Hx.
      rewrite Hc, E in Hx. exact (H1 Hx).
  - intros fct Hin. rewrite Hc. destruct (Vf fct Hin) as [H | H];
      [ exact (i_var a Hinv fct H) | exact H ].
Qed.

Lemma mstep_leaf (env : traverse_env) (a a' : acc) :
  (inv a -> menv env a ->
   KW.ext (ghost (acc_next a) (specials a)) a a'
   ∧ vsafe e0 (acc_code_deps a) a a') ->
  mstep env a a'.
Proof.
  intros H Hinv Henv. destruct (H Hinv Henv) as [He (V1 & V2 & V3)].
  exact (mstep_core a a' Hinv He (conj V1 V2) (or_introl V3)).
Qed.

End Inv.

(* ------------------------------------------------------------------------ *)
(* The steps of the traversal                                               *)
(* ------------------------------------------------------------------------ *)

Section Steps.

Variable e0 : expr.

Abbreviation inv := (inv e0).
Abbreviation menv := (menv e0).
Abbreviation mstep := (mstep e0).

Lemma ghost_le (a : acc) :
  (acc_next (ghost (acc_next a) (specials a)) <= acc_next a)%positive.
Proof. cbn. lia. Qed.

Lemma mstep_named (env : traverse_env) (x : variable) (d : named) (a : acc) :
  V0 e0 x -> (∀ f y, (f, y) ∈ set_closures_of env x -> V0 e0 y) ->
  mstep env a (traverse_named env x d a).
Proof.
  intros Hx Hs. apply mstep_leaf. intros _ _. split;
    [ apply KW.ext_traverse_named, ghost_le
    | apply vsafe_traverse_named; assumption ].
Qed.

Lemma mstep_static_binding (env : traverse_env) (b : static_binding)
    (a : acc) :
  mstep env a (traverse_static_binding env b a).
Proof.
  apply mstep_leaf. intros _ _. split;
    [ apply KW.ext_traverse_static_binding
    | apply vsafe_traverse_static_binding ].
Qed.

Lemma mstep_exn_handler (env : traverse_env) (h : continuation_handler)
    (a : acc) :
  (∀ p, p ∈ handler_params h -> var_ok e0 p) ->
  mstep env a (exn_handler_deps env h a).
Proof.
  intros Hp. apply mstep_leaf. intros _ _. split;
    [ apply KW.ext_exn_handler_deps | apply vsafe_exn_handler_deps, Hp ].
Qed.

Lemma mstep_apply_cont (env : traverse_env) (ac : apply_cont) (a : acc) :
  mstep env a (apply_cont_deps env ac a).
Proof.
  apply mstep_leaf. intros _ Henv. split;
    [ apply KW.ext_apply_cont_deps;
        [ exact (menv_ghost _ _ _ Henv) | apply ghost_le ]
    | apply vsafe_apply_cont_deps, (menv_venv _ _ _ Henv) ].
Qed.

Lemma mstep_switch (env : traverse_env) (s : simple)
    (arms : list (Z * apply_cont)) (a : acc) :
  mstep env a (traverse_switch env s arms a).
Proof.
  apply mstep_leaf. intros _ Henv. split;
    [ apply KW.ext_traverse_switch;
        [ exact (menv_ghost _ _ _ Henv) | apply ghost_le ]
    | apply vsafe_traverse_switch, (menv_venv _ _ _ Henv) ].
Qed.

(* The facts that [traverse_code] adds before its body. *)
Lemma mstep_code_facts (env : traverse_env) (id : code_id) (c : code)
    (cd : code_dep) (a : acc) :
  lookup id (acc_code_deps a) = Some cd ->
  (∀ p, p ∈ code_params c -> V0 e0 p) ->
  ((id, c) ∈ unit_codes e0 ∨ V0 e0 (code_my_closure c)) ->
  mstep env a
    (add_facts
       (Fact_alias (Node_variable (code_my_closure c)) (cd_my_closure cd)
        :: map (fun '(p, arg) => Fact_alias (Node_variable p) arg)
             (combine (code_params c) (cd_params cd)))
       a).
Proof.
  intros E Hp Hmc. apply mstep_leaf. intros _ _. split.
  - apply KW.ext_add_facts.
    + intros fct x H Hx. apply KW.ok_not_made.
      destruct H as [<- | H];
        [ | apply in_map_iff in H as ([p arg] & <- & _) ];
        cbn in Hx; destruct Hx as [<- | []]; intros k; discriminate.
    + intros b from [H | H]; [ discriminate H | ].
      apply in_map_iff in H as ([p arg] & H & _). discriminate H.
  - apply vsafe_add_facts. intros fct [<- | H] v Hv.
    + cbn in Hv. destruct Hv as [Hv | []]. injection Hv as <-.
      destruct Hmc as [Hc | Hv]; [ right | left; exact Hv ].
      exists id, c, cd. repeat split; auto.
      exact (FlambdaMapProofs.lookup_in _ _ _ E).
    + apply in_map_iff in H as ([p arg] & <- & Hin). cbn in Hv.
      destruct Hv as [Hv | []]. injection Hv as <-. left.
      exact (Hp p (in_combine_l _ _ _ _ Hin)).
Qed.

(* A node that is not made or made before [b]. *)
Definition low (b : acc) (x : node) : Prop :=
  not_made x ∨ made_lt x (acc_next b).

Lemma low_mono (b b' : acc) (x : node) :
  (acc_next b <= acc_next b')%positive -> low b x -> low b' x.
Proof.
  intros H [H1 | H1]; [ left; exact H1 | right ].
  intros k E. specialize (H1 k E). lia.
Qed.

Definition facts_low (b : acc) : Prop :=
  ∀ fct x, fct ∈ acc_facts b -> x ∈ into_nodes fct -> low b x.

Lemma find_cont_low (env : traverse_env) (k : continuation) (n : nat)
    (b : acc) :
  (∀ k' ps x, (k', ps) ∈ env_conts env -> x ∈ ps -> low b x) ->
  facts_low b ->
  facts_low (snd (find_cont env k n b))
  ∧ (∀ x, x ∈ fst (find_cont env k n b) -> low (snd (find_cont env k n b)) x)
  ∧ (acc_next b <= acc_next (snd (find_cont env k n b)))%positive
  ∧ acc_apply_deps (snd (find_cont env k n b)) = acc_apply_deps b
  ∧ acc_closure_deps (snd (find_cont env k n b)) = acc_closure_deps b.
Proof.
  intros Henv Hb. unfold find_cont.
  destruct (lookup k (env_conts env)) as [params | ] eqn:E.
  - cbn [fst snd]. split; [ exact Hb | split; [ | split; [ lia | auto ] ] ].
    intros x Hx. exact (Henv k params x (FlambdaMapProofs.lookup_in _ _ _ E)
                          Hx).
  - destruct (FlambdaReaperLeastProofs.AutoWidgets.fresh_list_spec n b)
      as (H1 & _ & H3 & H4 & H5 & H6).
    destruct (fresh_list n b) as [params b'] eqn:Eb. cbn [fst snd] in *.
    cbn [add_facts acc_facts acc_next acc_apply_deps acc_closure_deps].
    split; [ | split; [ | split; [ exact H5 | auto ] ] ].
    + intros fct x Hin Hx. apply in_app_or in Hin as [Hin | Hin].
      * apply in_map_iff in Hin as (p & <- & _). destruct Hx.
      * rewrite H1 in Hin. exact (low_mono _ _ _ H5 (Hb _ _ Hin Hx)).
    + intros x Hx. destruct (H6 x Hx) as (m & -> & _ & Hm). right.
      intros k' [= <-]. exact Hm.
Qed.

Ltac in_tac :=
  first [ left; reflexivity
        | right; in_tac
        | apply in_or_app; first [ left; in_tac | right; in_tac ] ].

Lemma ns_ads (l : list simple) (a : acc) :
  acc_apply_deps (note_simples l a) = acc_apply_deps a.
Proof.
  exact (proj1 (proj2 (proj2 (proj2
    (FlambdaReaperLeastProofs.AutoWidgets.note_simples_fields l a))))).
Qed.

Lemma ns_cls (l : list simple) (a : acc) :
  acc_closure_deps (note_simples l a) = acc_closure_deps a.
Proof.
  exact (proj2 (proj2 (proj2 (proj2
    (FlambdaReaperLeastProofs.AutoWidgets.note_simples_fields l a))))).
Qed.

Lemma ok_low (b : acc) (x : node) :
  low b x -> KW.ok_node (ghost (acc_next b) []) b x.
Proof.
  intros [H | H]; [ left; exact H | right; left ].
  split; [ exact H | intros Hw; apply (proj1 (kwit_ghost _ _ _)) in Hw;
                     destruct Hw ].
Qed.

Lemma low_ok (b b' : acc) (x : node) :
  (acc_next b <= acc_next b')%positive ->
  KW.ok_node (ghost (acc_next b) []) b' x -> made_lt x (acc_next b').
Proof.
  intros Hn H. refine (proj1 (proj2 (ok_ghost _ _ _ _ _ Hn H))).
  intros y [].
Qed.

(* A call of known arity: its widget, its witness and the node its code
   id is read into. *)
Lemma make_known_spec (env : traverse_env) (ap : apply_expr)
    (returns : list node) (exn : node) (b : acc) :
  facts_low b -> (∀ x, x ∈ exn :: returns -> low b x) ->
  let b1 := snd (make_known_arity_apply_widget env ap returns exn b) in
  let cw := fst (make_known_arity_apply_widget env ap returns exn b) in
  facts_low b1 ∧ made_lt cw (acc_next b1)
  ∧ (acc_next b <= acc_next b1)%positive
  ∧ acc_apply_deps b1 = acc_apply_deps b
  ∧ acc_closure_deps b1 = acc_closure_deps b
  ∧ (∀ fct, fct ∈ acc_facts b -> fct ∈ acc_facts b1)
  ∧ ∃ witness called,
      alias_for_caller (env_current_code_id env) cw witness ∈ acc_facts b1
      ∧ Fact_accessor called cid witness ∈ acc_facts b1
      ∧ Fact_any_usage called ∈ acc_facts b1.
Proof.
  intros Hb Hin. cbv zeta.
  destruct (KW.ext_make_known (ghost (acc_next b) []) env ap returns exn b)
    as [He Hcw]; [ cbn; lia | intros x Hx; apply ok_low, Hin, Hx | ].
  destruct He as [Hn (L & Hf & Hi & _) _ _ _].
  split; [ | split; [ exact (low_ok _ _ _ Hn Hcw) | split; [ exact Hn | ] ] ].
  { intros fct x H Hx. rewrite Hf in H. apply in_app_or in H as [H | H].
    - right. exact (low_ok _ _ _ Hn (Hi _ _ H Hx)).
    - exact (low_mono _ _ _ Hn (Hb _ _ H Hx)). }
  clear Hi Hcw.
  split; [ | split; [ | split; [ intros fct H; rewrite Hf; apply in_or_app;
                                 right; exact H | ] ] ].
  all: clear Hf; unfold make_known_arity_apply_widget; cbn [fresh];
    cbv beta iota zeta; cbn [fst snd]; unfold cond_alias;
    destruct (env_current_code_id env) as [id | ];
    cbn [add_fact add_facts acc_facts acc_apply_deps acc_closure_deps
         alias_for_caller];
    rewrite ?ns_ads, ?ns_cls; try reflexivity.
  all: eexists; eexists; split; [ left; reflexivity | split; in_tac ].
Qed.

Lemma vsafe_ads (cds : finite_map code_dep) (a a' : acc) :
  vsafe e0 cds a a' -> acc_apply_deps a' = acc_apply_deps a.
Proof. intros (_ & _ & H). exact H. Qed.

Lemma call_kind_spec (env : traverse_env) (ap : apply_expr) (ex : node)
    (rets : list node) (b : acc) :
  env_should_preserve_direct_calls env = Preserve_auto ->
  facts_low b -> (∀ x, x ∈ ex :: rets -> low b x) ->
  (∀ x, x ∈ ex :: rets -> var_ok e0 x) ->
  let b' := traverse_call_kind env ap ex rets b in
  acc_apply_deps b' = acc_apply_deps b
  ∨ ∃ d, acc_apply_deps b' = d :: acc_apply_deps b ∧ newdep b b' d
    ∧ acc_closure_deps b' = acc_closure_deps b.
Proof.
  intros Hauto Hb Hin Hvar. cbv zeta. unfold traverse_call_kind. cbv zeta.
  set (b0 := note_simple (ap_callee ap) b).
  assert (N : acc_next b0 = acc_next b ∧ acc_facts b0 = acc_facts b
              ∧ acc_apply_deps b0 = acc_apply_deps b
              ∧ acc_closure_deps b0 = acc_closure_deps b)
    by (unfold b0; destruct (ap_callee ap); cbn; auto).
  destruct N as (N1 & N2 & N3 & N4).
  assert (Hb0 : facts_low b0).
  { intros fct x H Hx. rewrite N2 in H. unfold low. rewrite N1.
    exact (Hb _ _ H Hx). }
  assert (Hin0 : ∀ x, x ∈ ex :: rets -> low b0 x)
    by (intros x Hx; unfold low; rewrite N1; exact (Hin x Hx)).
  clearbody b0.
  set (closure := node_of_simple (ap_callee ap)).
  destruct (ap_call_kind ap) as [id | | code_ids].
  - destruct (make_known_spec env ap rets ex b0 Hb0 Hin0)
      as (Hb1 & Hcw & Hn1 & Ha1 & Hc1 & Hf1 & (witness & called & W1 & W2
                                                 & W3)).
    destruct (make_known_arity_apply_widget env ap rets ex b0)
      as [cw b1] eqn:E. cbn [fst snd] in *. cbv beta iota zeta.
    rewrite Hauto. unfold add_apply. cbn [fresh]. cbv beta iota zeta.
    right. eexists. split; [ cbn; rewrite Ha1, N3; reflexivity | ].
    split; [ | cbn; rewrite Hc1, N4; reflexivity ].
    exists (acc_next b1). cbn [add_fact add_facts acc_next acc_facts
      push_apply_dep ad_apply_closure ad_apply_call_witness].
    split; [ reflexivity | split; [ lia | split; [ reflexivity
                                                 | split; [ lia | ] ] ] ].
    exists closure, cw, witness, called. split.
    + cbn [ad_apply_closure ad_apply_call_witness
           ad_function_containing_apply_expr].
      split; [ left; reflexivity | split; [ right; left; reflexivity | ] ].
      split; [ right; right; right; exact W1 | ].
      split; right; right; right; assumption.
    + cbn [ad_apply_closure]. intros fct H Hx.
      destruct H as [<- | [<- | [<- | H]]]; [ reflexivity | | | ];
        [ cbn in Hx; destruct Hx as [Hx | []]; rewrite Hx in Hcw;
          specialize (Hcw _ eq_refl); lia
        | cbn in Hx; destruct Hx as [Hx | []]; rewrite Hx in Hcw;
          specialize (Hcw _ eq_refl); lia | ].
      exfalso. destruct (Hb1 _ _ H Hx) as [Hm | Hm];
        [ exact (Hm _ eq_refl) | specialize (Hm _ eq_refl); lia ].
  - left. rewrite <- N3.
    destruct (vsafe_make_unknown e0 (acc_code_deps b) env ap rets ex b0 Hvar)
      as [Hk _].
    destruct (make_unknown_arity_apply_widget env ap rets ex b0)
      as [cw b1] eqn:E. cbn [fst snd] in *. cbv beta iota.
    cbn [add_fact add_facts acc_apply_deps]. exact (vsafe_ads _ _ _ Hk).
  - left. rewrite <- N3.
    destruct (make_known_spec env ap rets ex b0 Hb0 Hin0)
      as (_ & _ & _ & Ha1 & _).
    destruct (make_known_arity_apply_widget env ap rets ex b0)
      as [cw b1] eqn:E. cbn [fst snd] in *. cbv beta iota zeta.
    cbn [add_fact add_facts acc_apply_deps]. exact Ha1.
Qed.

Lemma newdep_weaken (a b b' : acc) (d : apply_dep) :
  (acc_next a <= acc_next b)%positive -> newdep b b' d -> newdep a b' d.
Proof.
  intros H (k & H1 & H2 & H3 & H4 & H5). exists k. repeat split; auto. lia.
Qed.

Lemma mstep_apply (env : traverse_env) (ap : apply_expr) (a : acc) :
  mstep env a (traverse_apply env ap a).
Proof.
  intros Hinv Henv.
  pose proof (KW.ext_traverse_apply (ghost (acc_next a) (specials a)) env ap a
                (menv_ghost _ _ _ Henv) (ghost_le a)) as He.
  pose proof (vsafe0_traverse_apply e0 (acc_code_deps a) env ap a
                (menv_venv _ _ _ Henv)) as Hv.
  apply (mstep_core e0 a _ Hinv He Hv). clear He Hv.
  assert (Henv_low : ∀ b, (acc_next a <= acc_next b)%positive ->
            ∀ k ps x, (k, ps) ∈ env_conts env -> x ∈ ps -> low b x).
  { intros b Hle k ps x Hk Hx. right.
    destruct (me_conts _ _ _ Henv k ps x Hk Hx) as [[Hlt _] _].
    intros k' E. specialize (Hlt k' E). lia. }
  assert (Ha : facts_low a)
    by (intros fct x H Hx; right; exact (i_into _ _ Hinv fct x H Hx)).
  unfold traverse_apply.
  destruct (find_cont_low env (ap_continuation ap)
              (length (unarize (ap_return_arity ap))) a
              (Henv_low a (Pos.le_refl _)) Ha)
    as (Hb1 & Hr1 & Hn1 & Ha1 & Hc1).
  destruct (vsafe_find_cont e0 (acc_code_deps a) env (ap_continuation ap)
              (length (unarize (ap_return_arity ap))) a
              (menv_venv _ _ _ Henv)) as [_ Hv1].
  destruct (find_cont env (ap_continuation ap) _ a) as [rets b1] eqn:E1.
  cbn [fst snd] in *. cbv beta iota.
  destruct (find_cont_low env (ap_exn_continuation ap)
              (S (length (ap_exn_extra_args ap))) b1
              (Henv_low b1 Hn1) Hb1)
    as (Hb2 & Hr2 & Hn2 & Ha2 & Hc2).
  destruct (vsafe_find_cont e0 (acc_code_deps a) env (ap_exn_continuation ap)
              (S (length (ap_exn_extra_args ap))) b1
              (menv_venv _ _ _ Henv)) as [_ Hv2].
  destruct (find_cont env (ap_exn_continuation ap) _ b1) as [eps b2] eqn:E2.
  cbn [fst snd] in *. cbv beta iota.
  assert (Hfin : ∀ ex b3,
            (acc_next a <= acc_next b3)%positive ->
            acc_apply_deps b3 = acc_apply_deps a ->
            acc_closure_deps b3 = acc_closure_deps a ->
            facts_low b3 -> (∀ x, x ∈ ex :: rets -> low b3 x) ->
            (∀ x, x ∈ ex :: rets -> var_ok e0 x) ->
            let a' := traverse_call_kind env ap ex rets b3 in
            acc_apply_deps a' = acc_apply_deps a
            ∨ ∃ d, acc_apply_deps a' = d :: acc_apply_deps a
              ∧ newdep a a' d ∧ acc_closure_deps a' = acc_closure_deps a).
  { intros ex b3 Hn3 Ha3 Hc3 Hb3 Hl3 Hvar3. cbv zeta.
    destruct (call_kind_spec env ap ex rets b3 (me_auto _ _ _ Henv) Hb3 Hl3
                Hvar3) as [H | (d & H1 & H2 & H3)];
      [ left; rewrite H; exact Ha3 | right ].
    exists d. rewrite <- Ha3, <- Hc3. split; [ exact H1 | ].
    split; [ exact (newdep_weaken _ _ _ _ Hn3 H2) | exact H3 ]. }
  destruct eps as [ | ep extra].
  - cbn [fresh]. cbv beta iota.
    apply Hfin; cbn [acc_next acc_apply_deps acc_closure_deps];
      [ lia | congruence | congruence | | | ].
    + intros fct x H Hx. apply (low_mono b2); [ cbn; lia | ].
      exact (Hb2 _ _ H Hx).
    + intros x [<- | Hx]; [ right; intros k [= <-]; cbn; lia | ].
      apply (low_mono b1); [ cbn; lia | exact (Hr1 x Hx) ].
    + intros x [<- | Hx]; [ apply var_ok_made | exact (Hv1 x Hx) ].
  - cbv zeta beta iota.
    destruct (FlambdaReaperLeastProofs.AutoWidgets.note_simples_fields
                (exn_extra_simples ap) b2) as (N1 & N2 & _ & N4 & N5).
    set (b3 := note_simples _ b2) in *. clearbody b3.
    apply Hfin; cbn [add_facts acc_next acc_apply_deps acc_closure_deps
                     acc_facts];
      [ lia | congruence | congruence | | | ].
    + intros fct x H Hx. unfold low. cbn [add_facts acc_next]. rewrite N1.
      apply in_app_or in H as [H | H].
      * apply in_map_iff in H as ([p s] & <- & Hps). cbn in Hx.
        destruct Hx as [<- | []].
        exact (Hr2 p (or_intror (in_combine_l _ _ _ _ Hps))).
      * rewrite N2 in H. exact (Hb2 _ _ H Hx).
    + intros x [<- | Hx]; unfold low; cbn [add_facts acc_next]; rewrite N1;
        [ exact (Hr2 ep (or_introl eq_refl)) | ].
      apply (low_mono b1); [ lia | exact (Hr1 x Hx) ].
    + intros x [<- | Hx]; [ exact (Hv2 ep (or_introl eq_refl))
                         | exact (Hv1 x Hx) ].
Qed.

End Steps.

(* ------------------------------------------------------------------------ *)
(* The step that makes a code dep                                           *)
(* ------------------------------------------------------------------------ *)

Section Prepare.

Variable e0 : expr.

Lemma fresh_eq (c : acc) (x : node) (b : acc) :
  fresh c = (x, b) ->
  x = Node_made (acc_next c) ∧ acc_next b = Pos.succ (acc_next c)
  ∧ acc_facts b = acc_facts c ∧ acc_code_deps b = acc_code_deps c
  ∧ acc_apply_deps b = acc_apply_deps c
  ∧ acc_closure_deps b = acc_closure_deps c.
Proof. unfold fresh. intros E. injection E as <- <-. cbn. repeat split. Qed.

Lemma mstep_prepare_code (env : traverse_env) (id : code_id) (c : code)
    (a : acc) :
  mstep e0 env a (prepare_code id c a).
Proof.
  intros Hinv _.
  pose proof (vsafe_prepare_code e0 (acc_code_deps a) id c a) as Hv.
  unfold prepare_code in *.
  destruct (FlambdaReaperLeastProofs.AutoWidgets.fresh_list_spec
              (length (code_result_kinds c)) a)
    as (R1 & R2 & R3 & R4 & R5 & R6).
  destruct (fresh_list (length (code_result_kinds c)) a)
    as [returns b1] eqn:E1. cbn [fst snd] in *.
  destruct (fresh b1) as [exn b2] eqn:E2.
  destruct (fresh_eq _ _ _ E2) as (X1 & X2 & X3 & X4 & X5 & X6).
  destruct (fresh b2) as [mc b3] eqn:E3.
  destruct (fresh_eq _ _ _ E3) as (M1 & M2 & M3 & M4 & M5 & M6).
  destruct (FlambdaReaperLeastProofs.AutoWidgets.fresh_list_spec
              (length (code_params_kinds c)) b3)
    as (P1 & P2 & P3 & P4 & P5 & P6).
  destruct (fresh_list (length (code_params_kinds c)) b3)
    as [params b4] eqn:E4. cbn [fst snd] in *.
  set (sp := mc :: specials a).
  set (g := ghost (acc_next b4) sp).
  assert (Hsp : ∀ y, y ∈ sp -> ∃ k, y = Node_made k
                  ∧ (k < acc_next b4)%positive).
  { intros y [<- | Hy]; [ exists (acc_next b2); split; [ exact M1 | lia ] | ].
    destruct (i_specials _ _ Hinv y Hy) as (k & -> & Hk).
    exists k. split; [ reflexivity | lia ]. }
  assert (Hspa : ∀ y, y ∈ specials a -> made_lt y (acc_next a)).
  { intros y Hy. destruct (i_specials _ _ Hinv y Hy) as (k & -> & Hk).
    intros k' [= <-]. exact Hk. }
  assert (Hpok : ∀ x, x ∈ params -> KW.ok_node g b4 x).
  { intros x Hx. destruct (P6 x Hx) as (m & -> & Hm1 & Hm2).
    right; left. split; [ intros k [= <-]; exact Hm2 | ].
    intros Hw. apply (proj1 (kwit_ghost _ _ _)) in Hw.
    destruct Hw as [Hw | Hw]; [ rewrite M1 in Hw; injection Hw; lia | ].
    specialize (Hspa _ Hw _ eq_refl). lia. }
  destruct (KW.ext_create_known g id params returns exn b4)
    as [H5 _]; [ cbn; lia | exact Hpok | ].
  destruct (create_known_arity_call_witness id params returns exn b4)
    as [known b5] eqn:E5. cbn [fst snd] in *.
  pose proof (KW.ext_next _ _ _ H5) as Hn5.
  destruct (KW.ext_create_unknown g id (code_is_tupled c) (code_params_arity c)
              params returns exn b5) as [H6 _];
    [ cbn; lia | intros x Hx; exact (KW.ok_node_mono _ _ _ _ Hn5 (Hpok x Hx))
    | ].
  destruct (create_unknown_arity_call_witnesses id (code_is_tupled c)
              (code_params_arity c) params returns exn b5)
    as [unknown b6] eqn:E6. cbn [fst snd] in *.
  set (b7 := add_fact (Fact_any_source (Node_code_id id)) b6) in *.
  assert (H7 : KW.ext g b6 b7).
  { apply KW.ext_add_fact; [ intros y [<- | []]; apply KW.ok_not_made;
                             intros k; discriminate
                           | intros ? ? ?; discriminate ]. }
  pose proof (KW.ext_trans _ _ _ _ (KW.ext_trans _ _ _ _ H5 H6) H7) as He.
  clearbody b7. clear H5 H6 H7.
  destruct He as [Hn (L & Hf & Hi & _) Hc _ Hcl].
  set (cd := {| cd_params := params; cd_my_closure := mc;
                cd_return := returns; cd_exn := exn;
                cd_known_arity_call_witness := known;
                cd_unknown_arity_call_witnesses := unknown |}) in *.
  destruct Hv as (Vf & Vc & Va). cbn [add_code_dep acc_apply_deps] in Va.
  assert (Hf4 : acc_facts b4 = acc_facts a) by congruence.
  assert (Hc4 : acc_code_deps b4 = acc_code_deps a) by congruence.
  assert (Hl4 : acc_closure_deps b4 = acc_closure_deps a) by congruence.
  assert (Hn4 : (acc_next a < acc_next b4)%positive) by lia.
  set (a' := add_code_dep id cd b7).
  assert (Hspec : specials a' = mc :: specials a).
  { unfold a', specials. cbn [add_code_dep acc_code_deps acc_apply_deps map
                               app snd cd_my_closure].
    rewrite Hc, Hc4, Va. reflexivity. }
  assert (Hna : acc_next a' = acc_next b7) by reflexivity.
  assert (Hmc : mc = Node_made (acc_next b2)) by exact M1.
  assert (Hmc_new : ∀ x, made_lt x (acc_next a) -> x <> mc).
  { intros x Hx ->. rewrite Hmc in Hx. specialize (Hx _ eq_refl). lia. }
  assert (HL : ∀ fct x, fct ∈ acc_facts a' -> x ∈ into_nodes fct ->
             fct ∈ acc_facts a ∨ (x ∉ sp ∧ made_lt x (acc_next a'))).
  { intros fct x Hin Hx. cbn [a' add_code_dep acc_facts] in Hin.
    rewrite Hf, Hf4 in Hin.
    apply in_app_or in Hin as [Hin | Hin]; [ right | left; exact Hin ].
    destruct (ok_ghost _ _ _ _ Hsp Hn (Hi _ _ Hin Hx)) as (H1 & H2 & _).
    auto. }
  assert (Hgrows : grows a a').
  { split; [ cbn; lia | split ].
    - intros x Hx. rewrite Hspec in Hx. destruct Hx as [<- | Hx];
        [ right; exists (acc_next b2); split; [ exact Hmc | lia ]
        | left; exact Hx ].
    - intros p Hp. cbn [a' add_code_dep acc_code_deps]. right.
      rewrite Hc, Hc4. exact Hp. }
  assert (Hgood : ∀ x, good a x -> good a' x)
    by (intros x; exact (good_grows _ _ _ Hgrows)).
  assert (Hcds : ∀ id' cd', (id', cd') ∈ acc_code_deps a' ->
             (id' = id ∧ cd' = cd) ∨ (id', cd') ∈ acc_code_deps a).
  { intros id' cd' H. cbn [a' add_code_dep acc_code_deps] in H.
    rewrite Hc, Hc4 in H. destruct H as [[= <- <-] | H]; auto. }
  assert (Hold_mc : ∀ id' cd', (id', cd') ∈ acc_code_deps a ->
             cd_my_closure cd' <> mc).
  { intros id' cd' H. apply Hmc_new, Hspa. exact (cd_special _ _ _ H). }
  split; [ constructor | exact Hgrows ].
  - intros fct x Hin Hx. destruct (HL fct x Hin Hx) as [H | [_ H]];
      [ | exact H ].
    intros k E. specialize (i_into _ _ Hinv fct x H Hx k E). cbn. lia.
  - intros x Hx. rewrite Hspec in Hx. destruct Hx as [<- | Hx].
    + exists (acc_next b2). split; [ exact Hmc | cbn; lia ].
    + destruct (i_specials _ _ Hinv x Hx) as (k & -> & Hk).
      exists k. split; [ reflexivity | cbn; lia ].
  - intros id' cd' x Hin Hx. destruct (Hcds _ _ Hin) as [[-> ->] | H].
    + unfold cd in Hx. cbn [cd_exn cd_return] in Hx.
      split; [ | destruct Hx as [<- | Hx];
                 [ rewrite X1; apply var_ok_made
                 | destruct (R6 x Hx) as (m & -> & _); apply var_ok_made ] ].
      assert (Hx' : ∃ m, x = Node_made m ∧ (acc_next a <= m)%positive
                    ∧ (m < acc_next b2)%positive).
      { destruct Hx as [<- | Hx];
          [ exists (acc_next b1); split; [ exact X1 | split; lia ] | ].
        destruct (R6 x Hx) as (m & -> & Hm1 & Hm2). exists m.
        split; [ reflexivity | split; lia ]. }
      clear Hx. destruct Hx' as (m & -> & Hm1 & Hm2).
      split; [ intros k [= <-]; cbn; lia | ].
      rewrite Hspec. intros [Hx | Hx];
        [ rewrite Hmc in Hx; injection Hx; lia | ].
      specialize (Hspa _ Hx _ eq_refl). lia.
    + destruct (i_cd_nodes _ _ Hinv _ _ _ H Hx) as [H1 H2].
      split; [ exact (Hgood _ H1) | exact H2 ].
  - intros id1 cd1 id2 cd2 H1 H2 E.
    destruct (Hcds _ _ H1) as [[-> ->] | H1'];
      destruct (Hcds _ _ H2) as [[-> ->] | H2']; auto.
    + exfalso. exact (Hold_mc _ _ H2' (eq_sym E)).
    + exfalso. exact (Hold_mc _ _ H1' E).
    + exact (i_cd_inj _ _ Hinv _ _ _ _ H1' H2' E).
  - intros id' cd' d Hin Hd. cbn [a' add_code_dep acc_apply_deps] in Hd.
    rewrite Va in Hd. destruct (Hcds _ _ Hin) as [[-> ->] | H].
    + intros E. cbn [cd cd_my_closure] in E. apply (Hmc_new _ (Hspa _
        (ad_special _ _ Hd))). exact (eq_sym E).
    + exact (i_cd_ad _ _ Hinv _ _ _ H Hd).
  - intros id' cd' fct Hin Hfct Hx. destruct (HL fct _ Hfct Hx) as [H | [H _]].
    + destruct (Hcds _ _ Hin) as [[-> ->] | H'].
      * apply (Hmc_new mc); [ | reflexivity ].
        exact (i_into _ _ Hinv fct _ H Hx).
      * exact (i_cd_into _ _ Hinv _ _ _ H' H Hx).
    + destruct (Hcds _ _ Hin) as [[-> ->] | H'];
        [ apply H; left; reflexivity | apply H; right ].
      exact (cd_special _ _ _ H').
  - intros d Hd. cbn [a' add_code_dep acc_apply_deps] in Hd. rewrite Va in Hd.
    destruct (i_ad_wif _ _ Hinv d Hd) as [H1 H2].
    split; [ exact (Hgood _ H1) | exact H2 ].
  - intros d Hd. cbn [a' add_code_dep acc_apply_deps] in Hd. rewrite Va in Hd.
    destruct (i_ad_chain _ _ Hinv d Hd)
      as (callee & widget & witness & called & Hchain & Hinto).
    exists callee, widget, witness, called.
    assert (Hsub : ∀ fct, fct ∈ acc_facts a -> fct ∈ acc_facts a').
    { intros fct H. cbn [a' add_code_dep acc_facts]. rewrite Hf, Hf4.
      apply in_or_app. right. exact H. }
    split.
    + destruct Hchain as (C1 & C2 & C3 & C4 & C5).
      repeat split; apply Hsub; assumption.
    + intros fct Hin Hx. destruct (HL fct _ Hin Hx) as [H | [H _]];
        [ exact (Hinto fct H Hx) | ].
      exfalso. apply H. right. exact (ad_special _ _ Hd).
  - intros d Hd. cbn [a' add_code_dep acc_closure_deps] in Hd.
    destruct (Vc d Hd) as [Hd' | Hvar].
    + destruct (i_cl _ _ Hinv d Hd') as [[H | H] H2];
        [ split; [ left; exact H | exact H2 ]
        | split; [ right; exact (Hgood _ H) | exact H2 ] ].
    + split; [ | exact Hvar ].
      destruct (Hcl d Hd) as [Hd' | Hb].
      * rewrite Hl4 in Hd'. destruct (i_cl _ _ Hinv d Hd') as [[H | H] _];
          [ left; exact H | right; exact (Hgood _ H) ].
      * destruct (ok_ghost _ _ _ _ Hsp Hn Hb) as (H1 & H2 & [H3 | _]);
          [ left; exact H3 | right ].
        split; [ exact H2 | rewrite Hspec; exact H1 ].
  - intros fct Hin. destruct (Vf fct Hin) as [H | H].
    + apply (var_fact_mono e0 (acc_code_deps a));
        [ intros p Hp; cbn [a' add_code_dep acc_code_deps]; right;
          rewrite Hc, Hc4; exact Hp | exact (i_var _ _ Hinv fct H) ].
    + apply (var_fact_mono e0 (acc_code_deps a));
        [ intros p Hp; cbn [a' add_code_dep acc_code_deps]; right;
          rewrite Hc, Hc4; exact Hp | exact H ].
Qed.

End Prepare.

(* ------------------------------------------------------------------------ *)
(* The traversal                                                            *)
(* ------------------------------------------------------------------------ *)

Section Trav.

Variable e0 : expr.

(* The variables of a part of the unit: those it binds are of [V0], but
   the [my_closure]s of the code of the unit. *)
Definition sub (e : expr) : Prop :=
  (∀ x d, (x, d) ∈ let_bindings e -> V0 e0 x)
  ∧ (∀ v, v ∈ other_params e -> V0 e0 v)
  ∧ (∀ id c, (id, c) ∈ unit_codes e ->
       (id, c) ∈ unit_codes e0 ∨ V0 e0 (code_my_closure c)).

Definition hsub (h : continuation_handler) : Prop :=
  (∀ x d, (x, d) ∈ handler_let_bindings h -> V0 e0 x)
  ∧ (∀ v, v ∈ handler_other_params h -> V0 e0 v)
  ∧ (∀ id c, (id, c) ∈ handler_unit_codes h ->
       (id, c) ∈ unit_codes e0 ∨ V0 e0 (code_my_closure c)).

Lemma set_closures_V0 (env : traverse_env) (x : variable)
    (f : function_slot) (y : variable) :
  env_set_closures env = set_closures e0 ->
  (f, y) ∈ set_closures_of env x -> V0 e0 y.
Proof.
  intros E H. unfold set_closures_of in H. rewrite E in H.
  destruct (lookup x (set_closures e0)) as [others | ] eqn:L; [ | destruct H ].
  left. exact (proj1 set_closures_lets e0 x others f y
                 (FlambdaMapProofs.lookup_in _ _ _ L) H).
Qed.

Lemma menv_add_cont (env : traverse_env) (k : continuation)
    (params : list variable) (a : acc) :
  inv e0 a -> menv e0 env a -> (∀ p, p ∈ params -> V0 e0 p) ->
  menv e0 (add_cont env k (map Node_variable params)) a.
Proof.
  intros Hinv Henv Hp.
  constructor; [ | exact (me_auto _ _ _ Henv) | exact (me_sets _ _ _ Henv) ].
  intros k' ps x [Heq | Hin] Hx.
  - injection Heq as <- <-. apply in_map_iff in Hx as (p & <- & Hp').
    split.
    + split; [ intros k0 E; discriminate E | ].
      intros Hs. destruct (i_specials _ _ Hinv _ Hs) as (k0 & E & _).
      discriminate E.
    + intros v [= <-]. exact (Hp _ Hp').
  - exact (me_conts _ _ _ Henv k' ps x Hin Hx).
Qed.

Lemma menv_add_rec_conts (env : traverse_env)
    (handlers : list (continuation * continuation_handler)) (a : acc) :
  inv e0 a -> menv e0 env a ->
  (∀ k h, (k, h) ∈ handlers -> ∀ p, p ∈ handler_other_params h -> V0 e0 p) ->
  menv e0 (add_rec_conts env handlers) a.
Proof.
  intros Hinv Henv. unfold add_rec_conts.
  induction handlers as [ | [k h] hs IH]; intros Hp; cbn [fold_right];
    [ exact Henv | ].
  destruct h as [params md hb]. cbn [handler_params].
  apply menv_add_cont; [ exact Hinv | | ].
  - apply IH. intros k' h' Hin. exact (Hp k' h' (or_intror Hin)).
  - intros p Hin. apply (Hp k (Handler params md hb) (or_introl eq_refl)).
    cbn. apply in_or_app. left. exact Hin.
Qed.

Lemma mstep_traverse :
  (∀ e, sub e -> ∀ env a, mstep e0 env a (traverse env e a))
  ∧ (∀ h, hsub h -> ∀ env a, mstep e0 env a (traverse_handler env h a)).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind;
    unfold sub, hsub;
    cbn [let_bindings other_params unit_codes handler_let_bindings
         handler_other_params handler_unit_codes].
  - intros x kind d body IH (Hl & Ho & Hu) env a. cbn [traverse].
    apply (mstep_trans _ _ _ (traverse_named env x d a)).
    + intros Hinv Henv. refine (mstep_named e0 env x d a _ _ Hinv Henv).
      * apply (Hl x d). left. reflexivity.
      * intros f y H. exact (set_closures_V0 env x f y (me_sets _ _ _ Henv) H).
    + apply IH. split; [ intros x' d' H; apply (Hl x' d'); right; exact H | ].
      split; [ exact Ho | exact Hu ].
  - intros bindings body IHc IH (Hl & Ho & Hu) env a. cbn [traverse]. cbv zeta.
    match goal with
    | |- mstep _ _ _ (traverse _ _ ?X) =>
      apply (mstep_trans _ _ _ X);
      [ | apply IH; split; [ intros x' d' H; apply (Hl x' d'); apply in_or_app;
                             right; exact H | ];
          split; [ intros v H; apply Ho; apply in_or_app; right; exact H
                 | intros id c H; apply Hu; apply in_or_app; right; exact H ] ]
    end.
    match goal with
    | |- mstep _ _ _ (List.fold_left _ _ ?Y) =>
      apply (mstep_trans _ _ _ Y)
    end.
    { match goal with
      | |- mstep _ _ _ (List.fold_left _ _ ?Y) =>
        apply (mstep_trans _ _ _ Y)
      end.
      - apply mstep_fold. intros [id c | id | s const] a' _;
          cbn [prepare_binding];
          [ apply mstep_prepare_code | apply mstep_refl | apply mstep_refl ].
      - apply mstep_fold. intros b a' _. apply mstep_static_binding. }
    + apply mstep_fold. intros [id c | id | s const] a' Hin;
        cbn [traverse_binding]; [ | apply mstep_refl | apply mstep_refl ].
      assert (Hbo : ∀ v, v ∈ binding_other_params (Bind_code id c) -> V0 e0 v).
      { intros v Hv. apply Ho. apply in_or_app. left. apply in_flat_map.
        exists (Bind_code id c). split; [ exact Hin | exact Hv ]. }
      assert (Hbl : ∀ x d, (x, d) ∈ code_let_bindings c -> V0 e0 x).
      { intros x d H. apply (Hl x d). apply in_or_app. left. apply in_flat_map.
        exists (Bind_code id c). split; [ exact Hin | exact H ]. }
      assert (Hbu : (id, c) ∈ unit_codes e0 ∨ V0 e0 (code_my_closure c)).
      { apply Hu. apply in_or_app. left. apply in_flat_map.
        exists (Bind_code id c). split; [ exact Hin | left; reflexivity ]. }
      pose proof (IHc id c Hin) as IHb.
      destruct c as [params mc ret exn md cbody]. cbn [traverse_code].
      destruct (lookup id (acc_code_deps a')) as [cd | ] eqn:Ecd;
        [ | apply mstep_refl ].
      eapply mstep_trans.
      * exact (mstep_code_facts e0 env id (Code params mc ret exn md cbody) cd
                 a' Ecd (fun p Hp => Hbo p (in_or_app _ _ _ (or_introl Hp)))
                 Hbu).
      * apply (mstep_env _ _ (code_env env id ret exn cd)).
        -- intros Hinv Henv. constructor.
           ++ intros k ps x Hk Hx.
              cbn [code_env FlambdaReaperTraversal.env_conts] in Hk.
              assert (Hcd : (id, cd) ∈ acc_code_deps
                         (add_facts
                            (Fact_alias (Node_variable mc) (cd_my_closure cd)
                             :: map (fun '(p, arg) =>
                                       Fact_alias (Node_variable p) arg)
                                  (combine params (cd_params cd)))
                            a'))
                by exact (FlambdaMapProofs.lookup_in _ _ _ Ecd).
              destruct Hk as [Hk | [Hk | []]]; injection Hk as <- <-;
                apply (i_cd_nodes _ _ Hinv id cd x Hcd);
                [ destruct Hx as [<- | []]; left; reflexivity
                | right; exact Hx ].
           ++ cbn. exact (me_auto _ _ _ Henv).
           ++ cbn. exact (me_sets _ _ _ Henv).
        -- apply IHb. cbn [code_body]. split; [ exact Hbl | split ].
           ++ intros v Hv. apply Hbo. cbn. apply in_or_app. right.
              exact (other_in_params _ _ Hv).
           ++ intros id' c' Hc'. right. apply Hbo. cbn. apply in_or_app.
              right. exact (mc_in_params _ _ _ Hc').
  - intros k h body IHh IH (Hl & Ho & Hu) env a. cbn [traverse]. cbv zeta.
    apply (mstep_trans _ _ _
             (exn_handler_deps env h (traverse_handler env h a))).
    + apply (mstep_trans _ _ _ (traverse_handler env h a)).
      * apply IHh. split; [ intros x d H; apply (Hl x d); apply in_or_app;
                            left; exact H | ].
        split; [ intros v H; apply Ho; apply in_or_app; left; exact H
               | intros id c H; apply Hu; apply in_or_app; left; exact H ].
      * apply mstep_exn_handler. destruct h as [params md hb].
        cbn [handler_params]. intros x Hx v ->.
        apply in_map_iff in Hx as (p & [= ->] & Hp). apply Ho.
        apply in_or_app. left. cbn. apply in_or_app. left. exact Hp.
    + apply (mstep_env _ _ (add_cont env k (handler_params h))).
      * intros Hinv Henv. destruct h as [params md hb]. cbn [handler_params].
        apply menv_add_cont; [ exact Hinv | exact Henv | ].
        intros p Hp. apply Ho. apply in_or_app. left. cbn.
        apply in_or_app. left. exact Hp.
      * apply IH. split; [ intros x d H; apply (Hl x d); apply in_or_app;
                           right; exact H | ].
        split; [ intros v H; apply Ho; apply in_or_app; right; exact H
               | intros id c H; apply Hu; apply in_or_app; right; exact H ].
  - intros handlers body IHh IH (Hl & Ho & Hu) env a. cbn [traverse].
    cbv zeta.
    apply (mstep_env _ _ (add_rec_conts env handlers)).
    + intros Hinv Henv. apply menv_add_rec_conts; [ exact Hinv | exact Henv | ].
      intros k h Hkh p Hp. apply Ho. apply in_or_app. left.
      apply in_flat_map. exists (k, h). auto.
    + match goal with
      | |- mstep _ _ _ (traverse _ _ ?Y) => apply (mstep_trans _ _ _ Y)
      end.
      * apply mstep_fold. intros [k h] a' Hin. apply (IHh k h Hin).
        split; [ intros x d H; apply (Hl x d); apply in_or_app; left;
                 apply in_flat_map; exists (k, h); auto | ].
        split; [ intros v H; apply Ho; apply in_or_app; left;
                 apply in_flat_map; exists (k, h); auto
               | intros id c H; apply Hu; apply in_or_app; left;
                 apply in_flat_map; exists (k, h); auto ].
      * apply IH. split; [ intros x d H; apply (Hl x d); apply in_or_app;
                           right; exact H | ].
        split; [ intros v H; apply Ho; apply in_or_app; right; exact H
               | intros id c H; apply Hu; apply in_or_app; right; exact H ].
  - intros ap _ env a. cbn [traverse]. apply mstep_apply.
  - intros ac _ env a. cbn [traverse]. apply mstep_apply_cont.
  - intros s arms _ env a. cbn [traverse]. apply mstep_switch.
  - intros _ env a. apply mstep_refl.
  - intros params md handler IH (Hl & Ho & Hu) env a. cbn [traverse_handler].
    apply IH. split; [ exact Hl | split; [ | exact Hu ] ].
    intros v H. apply Ho. apply in_or_app. right. exact H.
Qed.

End Trav.

(* ------------------------------------------------------------------------ *)
(* The delayed deps, and the unit                                           *)
(* ------------------------------------------------------------------------ *)

(* A fact of the closure dep [d]: the propagate into the [my_closure] of
   the code dep of its code id, or one into its closure and nodes made from
   [n]. *)
Definition cdep_fact (cds : finite_map code_dep) (n : positive)
    (d : closure_dep) (fct : fact) : Prop :=
  (∃ cd, lookup (cld_closure_code_id d) cds = Some cd
     ∧ fct = Fact_propagate (Node_code_id (cld_closure_code_id d))
               (cd_my_closure cd) (cld_let_bound_name_of_the_closure d))
  ∨ (∀ x, x ∈ into_nodes fct ->
       x = cld_let_bound_name_of_the_closure d
       ∨ ∃ k, x = Node_made k ∧ (n <= k)%positive).

Lemma cdep_fact_weaken (cds : finite_map code_dep) (n n' : positive)
    (d : closure_dep) (fct : fact) :
  (n <= n')%positive -> cdep_fact cds n' d fct -> cdep_fact cds n d fct.
Proof.
  intros Hn [H | H]; [ left; exact H | right ].
  intros x Hx. destruct (H x Hx) as [H' | (k & -> & Hk)]; [ left; exact H' | ].
  right. exists k. split; [ reflexivity | lia ].
Qed.

Lemma closure_deps_facts (cds : finite_map code_dep) (ds : list closure_dep)
    (b : acc) :
  let b' := List.fold_left (fun a d => add_closure_dep cds d a) ds b in
  acc_code_deps b' = acc_code_deps b ∧ acc_apply_deps b' = acc_apply_deps b
  ∧ (acc_next b <= acc_next b')%positive
  ∧ (∀ fct, fct ∈ acc_facts b -> fct ∈ acc_facts b')
  ∧ ∀ fct, fct ∈ acc_facts b' ->
      fct ∈ acc_facts b ∨ ∃ d, d ∈ ds ∧ cdep_fact cds (acc_next b) d fct.
Proof.
  revert b. induction ds as [ | d ds IH]; intros b; cbv zeta;
    cbn [List.fold_left].
  - split; [ reflexivity | split; [ reflexivity | split; [ lia | ] ] ].
    split; [ auto | intros fct H; left; exact H ].
  - assert (S1 : acc_code_deps (add_closure_dep cds d b) = acc_code_deps b
                 ∧ acc_apply_deps (add_closure_dep cds d b) = acc_apply_deps b
                 ∧ (acc_next b <= acc_next (add_closure_dep cds d b))%positive
                 ∧ (∀ fct, fct ∈ acc_facts b ->
                      fct ∈ acc_facts (add_closure_dep cds d b))
                 ∧ ∀ fct, fct ∈ acc_facts (add_closure_dep cds d b) ->
                     fct ∈ acc_facts b ∨ cdep_fact cds (acc_next b) d fct).
    { unfold add_closure_dep.
      destruct (lookup (cld_closure_code_id d) cds) as [cd | ] eqn:E.
      - cbn [add_facts acc_code_deps acc_apply_deps acc_next acc_facts].
        split; [ reflexivity | split; [ reflexivity | split; [ lia | ] ] ].
        split; [ intros fct H; right; right; apply in_or_app; right;
                 exact H | ].
        intros fct H. destruct H as [<- | [<- | H]];
          [ right; left; exists cd; split; [ exact E | reflexivity ]
          | right; right; intros x [<- | []]; left; reflexivity | ].
        apply in_app_or in H as [H | H]; [ | left; exact H ].
        right; right.
        destruct (cd_unknown_arity_call_witnesses cd); [ destruct H | ].
        destruct H as [<- | []]. intros x [<- | []]. left. reflexivity.
      - cbn [fresh]. cbv beta iota.
        cbn [add_facts acc_code_deps acc_apply_deps acc_next acc_facts].
        split; [ reflexivity | split; [ reflexivity | split; [ lia | ] ] ].
        split; [ intros fct H; apply in_or_app; right; exact H | ].
        intros fct H. apply in_app_or in H as [H | H]; [ | left; exact H ].
        right; right.
        destruct H as [<- | [<- | [<- | [<- | []]]]]; intros x Hx;
          cbn in Hx; destruct Hx as [<- | []];
          first [ left; reflexivity
                | right; exists (acc_next b); split; [ reflexivity | lia ] ]. }
    destruct S1 as (S1 & S2 & S3 & S4 & S5).
    destruct (IH (add_closure_dep cds d b)) as (T1 & T2 & T3 & T4 & T5).
    split; [ congruence | split; [ congruence | split; [ lia | split ] ] ].
    + intros fct H. exact (T4 _ (S4 _ H)).
    + intros fct H. destruct (T5 fct H) as [H' | (d' & Hd' & Hc)].
      * destruct (S5 fct H') as [H'' | Hc]; [ left; exact H'' | right ].
        exists d. split; [ left; reflexivity | exact Hc ].
      * right. exists d'. split; [ right; exact Hd' | ].
        exact (cdep_fact_weaken _ _ _ _ _ S3 Hc).
Qed.

Lemma resolve_facts (us : list symbol) (a : acc) :
  let A := resolve_delayed_deps us a in
  acc_code_deps A = acc_code_deps a ∧ acc_apply_deps A = acc_apply_deps a
  ∧ (∀ fct, fct ∈ acc_facts a -> fct ∈ acc_facts A)
  ∧ (∀ fct, fct ∈ flat_map (apply_dep_facts (acc_code_deps a))
                    (acc_apply_deps a) -> fct ∈ acc_facts A)
  ∧ ∀ fct, fct ∈ acc_facts A ->
      fct ∈ acc_facts a
      ∨ fct ∈ flat_map (apply_dep_facts (acc_code_deps a)) (acc_apply_deps a)
      ∨ (∃ d, d ∈ acc_closure_deps a
           ∧ cdep_fact (acc_code_deps a) (acc_next a) d fct)
      ∨ ∃ s, fct = Fact_any_source (Node_symbol s).
Proof.
  cbv zeta. unfold resolve_delayed_deps. cbv zeta.
  set (L1 := flat_map _ _).
  destruct (closure_deps_facts (acc_code_deps a)
              (acc_closure_deps (add_facts L1 a)) (add_facts L1 a))
    as (C1 & C2 & C3 & C4 & C5).
  set (b2 := List.fold_left _ _ (add_facts L1 a)) in *.
  cbn [add_facts acc_code_deps acc_apply_deps acc_facts acc_closure_deps
       acc_next] in *.
  split; [ exact C1 | split; [ exact C2 | split; [ | split ] ] ].
  - intros fct H. apply in_or_app. right. apply C4. apply in_or_app. right.
    exact H.
  - intros fct H. apply in_or_app. right. apply C4. apply in_or_app. left.
    exact H.
  - intros fct H. apply in_app_or in H as [H | H].
    + right; right; right. apply in_map_iff in H as (s & <- & _). eauto.
    + destruct (C5 fct H) as [H' | (d & Hd & Hc)].
      * apply in_app_or in H' as [H' | H']; [ right; left; exact H'
                                            | left; exact H' ].
      * right; right; left. exists d. auto.
Qed.

(* The facts that [apply_dep_facts] puts into a node: the alias into the
   [my_closure] of the code dep found, or into the call witness of the
   apply dep or a code id. *)
Lemma apply_dep_fact_cases (cds : finite_map code_dep) (d : apply_dep)
    (fct : fact) :
  fct ∈ apply_dep_facts cds d ->
  (∃ cd, lookup (ad_apply_code_id d) cds = Some cd
     ∧ fct = alias_for_caller (ad_function_containing_apply_expr d)
               (ad_apply_closure d) (cd_my_closure cd))
  ∨ ∀ x, x ∈ into_nodes fct ->
      x = ad_apply_call_witness d ∨ ∃ id, x = Node_code_id id.
Proof.
  unfold apply_dep_facts.
  destruct (lookup (ad_apply_code_id d) cds) as [cd | ] eqn:E;
    [ | destruct (ad_function_containing_apply_expr d) ];
    intros [<- | [<- | []]];
    try (left; exists cd; split; [ first [ exact E | reflexivity ]
                                 | reflexivity ]);
    right; intros x Hx; unfold alias_for_caller in Hx;
    try destruct (ad_function_containing_apply_expr d);
    cbn in Hx; try contradiction; destruct Hx as [<- | []]; eauto.
Qed.

Lemma into_alias_for_caller (caller : option code_id) (from to_ x : node) :
  x ∈ into_nodes (alias_for_caller caller from to_) -> x = to_.
Proof. destruct caller; cbn; intros [<- | []]; reflexivity. Qed.

Section Resolve.

Variable e0 : expr.
Variable a : acc.
Hypothesis Hinv : inv e0 a.

Lemma made_not_var (x : node) (v : variable) :
  x ∈ specials a -> x <> Node_variable v.
Proof.
  intros Hx ->. destruct (i_specials _ _ Hinv _ Hx) as (k & E & _).
  discriminate E.
Qed.

Lemma resolve_bounds (us : list symbol) :
  let A := resolve_delayed_deps us a in
  bounds e0 (acc_code_deps A) (acc_apply_deps A) (acc_facts A).
Proof.
  cbv zeta. destruct (resolve_facts us a) as (R1 & R2 & R3 & R4 & R5).
  rewrite R1, R2.
  assert (Hcl_ne : ∀ d x, d ∈ acc_closure_deps a -> x ∈ specials a ->
             x <> cld_let_bound_name_of_the_closure d).
  { intros d x Hd Hx E. destruct (i_cl _ _ Hinv d Hd) as [[H | [_ H]] _].
    - destruct (i_specials _ _ Hinv _ Hx) as (k & -> & _).
      rewrite <- E in H. exact (H k eq_refl).
    - rewrite <- E in H. exact (H Hx). }
  assert (Hwif_ne : ∀ d x, d ∈ acc_apply_deps a -> x ∈ specials a ->
             x <> ad_apply_call_witness d).
  { intros d x Hd Hx E. destruct (i_ad_wif _ _ Hinv d Hd) as [[_ H] _].
    rewrite <- E in H. exact (H Hx). }
  assert (Hfresh_ne : ∀ x k, x ∈ specials a ->
             (acc_next a <= k)%positive -> x <> Node_made k).
  { intros x k Hx Hk ->. destruct (i_specials _ _ Hinv _ Hx)
      as (k' & [= <-] & Hk'). lia. }
  constructor.
  - intros fct v Hin Hv.
    destruct (R5 fct Hin) as [H | [H | [(d & Hd & Hc) | (s & ->)]]].
    + exact (i_var _ _ Hinv fct H v Hv).
    + apply in_flat_map in H as (d & Hd & H). left.
      destruct (apply_dep_fact_cases _ _ _ H) as [(cd & E & ->) | Hx].
      * apply into_alias_for_caller in Hv. exfalso.
        apply (made_not_var (cd_my_closure cd) v); [ | exact (eq_sym Hv) ].
        exact (cd_special _ _ _ (FlambdaMapProofs.lookup_in _ _ _ E)).
      * destruct (Hx _ Hv) as [Hw | (id & Hw)]; [ | discriminate Hw ].
        exact (proj2 (i_ad_wif _ _ Hinv d Hd) v (eq_sym Hw)).
    + left. destruct Hc as [(cd & E & ->) | Hx].
      * cbn in Hv. destruct Hv as [Hv | []]. exfalso.
        apply (made_not_var (cd_my_closure cd) v); [ | exact Hv ].
        exact (cd_special _ _ _ (FlambdaMapProofs.lookup_in _ _ _ E)).
      * destruct (Hx _ Hv) as [Hb | (k & Hk & _)]; [ | discriminate Hk ].
        exact (proj2 (i_cl _ _ Hinv d Hd) v (eq_sym Hb)).
    + cbn in Hv. destruct Hv as [Hv | []]. discriminate Hv.
  - intros id cd fct Hcd Hin Hx.
    pose proof (cd_special _ _ _ Hcd) as Hs.
    destruct (R5 fct Hin) as [H | [H | [(d & Hd & Hc) | (s & ->)]]].
    + exfalso. exact (i_cd_into _ _ Hinv _ _ _ Hcd H Hx).
    + apply in_flat_map in H as (d & Hd & H).
      destruct (apply_dep_fact_cases _ _ _ H) as [(cd' & E & ->) | Hx'].
      * apply into_alias_for_caller in Hx.
        destruct (i_cd_inj _ _ Hinv _ _ _ _
                    (FlambdaMapProofs.lookup_in _ _ _ E) Hcd (eq_sym Hx))
          as [_ ->].
        right. exists d. auto.
      * exfalso. destruct (Hx' _ Hx) as [Hw | (id' & Hw)].
        -- exact (Hwif_ne d _ Hd Hs Hw).
        -- destruct (i_specials _ _ Hinv _ Hs) as (k & E & _).
           rewrite E in Hw. discriminate Hw.
    + destruct Hc as [(cd' & E & ->) | Hx'].
      * cbn in Hx. destruct Hx as [Hx | []].
        destruct (i_cd_inj _ _ Hinv _ _ _ _
                    (FlambdaMapProofs.lookup_in _ _ _ E) Hcd Hx)
          as [-> ->].
        left. eauto.
      * exfalso. destruct (Hx' _ Hx) as [Hb | (k & Ek & Hk)].
        -- exact (Hcl_ne d _ Hd Hs Hb).
        -- exact (Hfresh_ne _ k Hs Hk Ek).
    + exfalso. cbn in Hx. destruct Hx as [Hx | []].
      destruct (i_specials _ _ Hinv _ Hs) as (k & E & _).
      rewrite <- Hx in E. discriminate E.
  - intros d Hd. pose proof (ad_special _ _ Hd) as Hs.
    destruct (i_ad_chain _ _ Hinv d Hd)
      as (callee & widget & witness & called & Hch & Hinto).
    exists callee, widget, witness, called. split.
    + destruct Hch as (C1 & C2 & C3 & C4 & C5).
      repeat split; apply R3; assumption.
    + intros fct Hin Hx.
      destruct (R5 fct Hin) as [H | [H | [(d' & Hd' & Hc) | (s & ->)]]].
      * exact (Hinto fct H Hx).
      * exfalso. apply in_flat_map in H as (d' & Hd' & H).
        destruct (apply_dep_fact_cases _ _ _ H) as [(cd' & E & ->) | Hx'].
        -- apply into_alias_for_caller in Hx.
           exact (i_cd_ad _ _ Hinv _ _ _ (FlambdaMapProofs.lookup_in _ _ _ E)
                    Hd (eq_sym Hx)).
        -- destruct (Hx' _ Hx) as [Hw | (id' & Hw)].
           ++ exact (Hwif_ne d' _ Hd' Hs Hw).
           ++ destruct (i_specials _ _ Hinv _ Hs) as (k & E & _).
              rewrite E in Hw. discriminate Hw.
      * exfalso. destruct Hc as [(cd' & E & ->) | Hx'].
        -- cbn in Hx. destruct Hx as [Hx | []].
           exact (i_cd_ad _ _ Hinv _ _ _ (FlambdaMapProofs.lookup_in _ _ _ E)
                    Hd Hx).
        -- destruct (Hx' _ Hx) as [Hb | (k & Hk & Hle)].
           ++ exact (Hcl_ne d' _ Hd' Hs Hb).
           ++ exact (Hfresh_ne _ k Hs Hle Hk).
      * exfalso. cbn in Hx. destruct Hx as [Hx | []].
        destruct (i_specials _ _ Hinv _ Hs) as (k & E & _).
        rewrite <- Hx in E. discriminate E.
  - intros d cd Hd E. apply R4. apply in_flat_map. exists d.
    split; [ exact Hd | ]. unfold apply_dep_facts. rewrite E. left.
    reflexivity.
Qed.

End Resolve.

Lemma mstep_any_usage (e0 : expr) (env : traverse_env) (x : node) (a : acc) :
  mstep e0 env a (add_fact (Fact_any_usage x) a).
Proof.
  apply mstep_leaf. intros _ _. split.
  - apply KW.ext_add_fact; [ intros y [] | intros ? ? ?; discriminate ].
  - apply vsafe_add_fact. intros v [].
Qed.

(* The facts of the unit meet the bounds. *)
Lemma unit_bounds (u : unit_interface) (e : expr) :
  bounds e (unit_code_deps Auto u e)
    (acc_apply_deps (traverse_unit_acc Auto u e))
    (acc_facts (traverse_unit_acc Auto u e)).
Proof.
  unfold unit_code_deps, traverse_unit_acc. cbn [fresh]. cbv beta iota zeta.
  match goal with
  | |- context [traverse ?env e ?a] => set (env0 := env); set (a4 := a)
  end.
  assert (Ha4 : acc_next a4 = 3%positive
                ∧ acc_facts a4 = [Fact_any_usage (Node_made 2);
                                  Fact_any_source Node_all_constants]
                ∧ acc_code_deps a4 = [] ∧ acc_apply_deps a4 = []
                ∧ acc_closure_deps a4 = []) by (cbn; auto).
  destruct Ha4 as (N4 & F4 & C4 & A4 & L4).
  assert (Hs4 : specials a4 = []) by (unfold specials; rewrite C4, A4;
                                      reflexivity).
  assert (Hinv4 : inv e a4).
  { constructor; rewrite ?N4, ?F4, ?C4, ?A4, ?L4, ?Hs4; try contradiction.
    - intros fct x [<- | [<- | []]] Hx; cbn in Hx; try contradiction;
        destruct Hx as [<- | []]; intros k E; discriminate E.
    - intros fct [<- | [<- | []]] v Hv; cbn in Hv; try contradiction;
        destruct Hv as [Hv | []]; discriminate Hv. }
  assert (Henv4 : menv e env0 a4).
  { constructor; [ | reflexivity | reflexivity ].
    intros k ps x Hk Hx. cbn [env0 FlambdaReaperTraversal.env_conts] in Hk.
    destruct Hk as [Hk | [Hk | []]]; injection Hk as <- <-;
      destruct Hx as [<- | []]; cbn;
      (split; [ split; [ intros k' [= <-]; rewrite N4; lia
                       | rewrite Hs4; intros [] ]
              | apply var_ok_made ]). }
  assert (Hsub : sub e e).
  { split; [ intros x d H; left; exact (in_map fst _ (x, d) H) | split ].
    - intros v H. right. exact H.
    - intros id c H. left. exact H. }
  destruct (proj1 (mstep_traverse e) e Hsub env0 a4 Hinv4 Henv4)
    as [Hinv5 Hg5].
  set (a5 := traverse env0 e a4) in *.
  match goal with
  | |- context [resolve_delayed_deps ?us
                  (add_fact (Fact_any_usage ?x) a5)] =>
    destruct (mstep_any_usage e env0 x a5 Hinv5 (menv_grows _ _ _ _ Hg5 Henv4))
      as [Hinv6 _];
    exact (resolve_bounds e _ Hinv6 us)
  end.
Qed.

(* [unused_code_dead] of a least solution of the facts of a unit of the
   fragment that binds each variable once. *)
Lemma unused_code_dead_traversal (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) :
  in_flow_fragment e = true -> variables_bound_once e = true ->
  is_local_slots_only is_local ->
  least_solution is_local (traverse_unit Auto u e) T ->
  FlambdaReaperFixFailsSpec.unused_code_dead (solution_of_tables T) e = true.
Proof.
  intros Hfrag Hbound Hslots Hleast.
  exact (unused_code_dead_of_bounds is_local u e T Hfrag Hbound Hslots Hleast
           (unit_bounds u e)).
Qed.
