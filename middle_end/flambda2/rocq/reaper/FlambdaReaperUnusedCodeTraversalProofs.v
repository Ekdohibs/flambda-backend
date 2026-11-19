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

(* Toward the shape of FlambdaReaperUnusedCodeProofs.v from the traversal,
   with [Auto], for a unit of [in_flow_fragment] that binds each variable
   once. Nothing here needs review.

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
     ([shape_of_bounds]), so [unused_code_dead]
     ([unused_code_dead_of_bounds]) and the conclusion of
     [fix_never_fails_statement] ([fix_never_fails_of_bounds]).
   - [vsafe]: each step of the traversal puts something only into the
     variables of [V0] (of the [Let]s, the handlers and the code, but the
     [my_closure]s of [unit_codes]), but [traverse_code], whose alias into
     its [my_closure] is the other case of [var_fact]: the half on
     variables of the invariant that gives [bounds].

   Not proved: that [traverse_unit_acc Auto u e] meets [bounds] (see
   S7.md). *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia Permutation.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaWellFormedSpec FlambdaReaperGraphSpec
  FlambdaReaperTraversal FlambdaReaperCallingConventions
  FlambdaReaperFlowSpec FlambdaReaperLeastSpec FlambdaReaperUnusedCodeProofs.
From OxCamlFlambda2 Require FlambdaMapProofs FlambdaReaperFlowProofs
  FlambdaReaperLeastProofs FlambdaReaperWrapperProofs
  FlambdaReaperWrapperAutoProofs FlambdaReaperFixFailsSpec
  FlambdaReaperFixFailsProofs.
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
(* The fix never stops, given the bounds                                    *)
(* ======================================================================== *)

Import FlambdaReaperSolution FlambdaReaperRebuild FlambdaReaperUnboxing
  FlambdaReaperLeast FlambdaReaperWrapperSpec FlambdaReaperWrapperDerivedSpec
  FlambdaReaperFixFailsSpec.

(* [fix_never_fails_statement] with the bounds of the traversal and
   [variables_bound_once] in place of [unused_code_dead]. *)
Lemma fix_never_fails_of_bounds :
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (max_unbox_size : nat) (lambda_lifting preserve_direct_calls : bool)
    (tcr : list node),
    let F := traverse_unit Auto u e in
    let code_deps := unit_code_deps Auto u e in
    let cannot_change := derive_cannot_change F T in
    let dominated := derive_dominated T in
    let to_unbox :=
      derive_to_unbox_fixed is_local F T cannot_change max_unbox_size
        lambda_lifting
    in
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    least_solution is_local F T ->
    conts_counted (unit_counts u) e = true ->
    plain_kinds e = true ->
    exn_handlers_well_formed e = true ->
    unknown_calls_have_args e = true ->
    results_within_codes e = true ->
    variables_bound_once e = true ->
    bounds e code_deps (acc_apply_deps (traverse_unit_acc Auto u e))
      (acc_facts (traverse_unit_acc Auto u e)) ->
    rebuild_unit_unbox_repr_fixed_fails is_local F T cannot_change code_deps
      dominated to_unbox tcr preserve_direct_calls e
    = false.
Proof.
  intros is_local u e T max_unbox_size lambda_lifting pdc tcr F code_deps
    cannot_change dominated to_unbox Hslots Hloc Hfrag Hleast Hcount Hkinds
    Hwf Hhas Hres Hbound Hb.
  apply (FlambdaReaperFixFailsProofs.fix_never_fails is_local u e T
           max_unbox_size lambda_lifting pdc tcr); try assumption.
  exact (unused_code_dead_of_bounds is_local u e T Hfrag Hbound Hslots Hleast
           Hb).
Qed.
