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

(* Proofs of FlambdaReaperTypesParamsTraversalSpec.v. Nothing here needs
   review.

   [pinv]: the invariant of the accumulator of the traversal of the unit
   [e0]. Its facts have their nodes made before it; the nodes of its code
   deps are made, distinct within a code dep and between two; a fact
   reads a parameter of a code dep, or its witness of known arity, only
   if it is the alias of [traverse_code] into the parameter of the code
   of that rank ([pi_reads]); a fact puts something into a variable only
   if the variable is bound by a [Let] of [e0] or is a parameter of a
   handler or of the body of code ([hparams]), or the fact is an alias of
   [traverse_code] ([pi_vars], as FlambdaReaperUnusedCodeTraversalProofs
   .vsafe has it, of the expression [e1 e0] whose variables are those);
   the deps have no such node. The steps of the traversal but
   [prepare_code] and [traverse_code] add facts whose nodes are not made,
   are made in the step, or are made before it and are none of the
   parameters and witnesses of known arity of the code deps
   (FlambdaReaperIndirectDeadFramesTraversalProofs.Chains.ext, and [next]
   for the call of unknown arity), make no code dep
   (FlambdaReaperWrapperAutoProofs.KnownWitnesses.ext), and put something
   into the variables as [vsafe] says. [prepare_code] makes a code dep of
   fresh nodes, whose facts read none of its parameters or its witness of
   known arity ([prepare_spec]). The variables of the unit being bound
   once, the parameters of its code are none of those of [e1 e0]
   ([cparams_of], [hparams], [bound_split]); the facts of
   [resolve_delayed_deps] read none of the parameters ([resolve_reads]).
   The statements of FlambdaReaperTypesParamsTraversalSpec.v follow, at
   the end, from those of FlambdaReaperTypesParamsSpec.v. *)

From Stdlib Require Import ZArith List Bool Lia Permutation.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaWellFormedSpec FlambdaSemanticsSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperSolutionProofs
  FlambdaReaperCallingConventions FlambdaReaperFlowSpec
  FlambdaReaperLeastSpec FlambdaReaperIndirectDeadFramesSpec
  FlambdaReaperTypes FlambdaReaperTypesCallsSpec
  FlambdaReaperTypesCallsUsedSpec FlambdaReaperTypesParamsSpec
  FlambdaReaperTypesParamsTraversalSpec.
From OxCamlFlambda2 Require FlambdaMapProofs FlambdaReaperFlowProofs
  FlambdaReaperLeastProofs FlambdaReaperRebuildProofs
  FlambdaReaperWrapperAutoProofs FlambdaReaperUnusedCodeTraversalProofs
  FlambdaReaperIndirectDeadFramesTraversalProofs
  FlambdaReaperTypesProofs FlambdaReaperTypesParamsProofs.
Import ListNotations.

Module ParamsTraversal.

Module AW := FlambdaReaperLeastProofs.AutoWidgets.
Module KW := FlambdaReaperWrapperAutoProofs.KnownWitnesses.
Module UT := FlambdaReaperUnusedCodeTraversalProofs.
Module CH := FlambdaReaperIndirectDeadFramesTraversalProofs.Chains.

Abbreviation made_lt := AW.made_lt.
Abbreviation not_made := AW.not_made.
Abbreviation into_nodes := AW.into_nodes.
Abbreviation nodes_of := CH.nodes_of.
Abbreviation code_dep_nodes := CH.code_dep_nodes.
Abbreviation kw := cd_known_arity_call_witness.
Abbreviation env_conts := FlambdaReaperTraversal.env_conts.

(* ======================================================================== *)
(* The check, on a list of facts                                            *)
(* ======================================================================== *)

(* The nodes that a fact reads, as [only_read_by] looks at them. *)
Definition read_nodes (fct : fact) : list node :=
  match fct with
  | Fact_alias _ y | Fact_use _ y | Fact_accessor _ _ y
  | Fact_constructor _ _ y | Fact_propagate _ _ y
  | Fact_alias_if_any_source _ _ y | Fact_any_usage y => [y]
  | Fact_argument y _ b => [y; b]
  | Fact_parameter _ _ _ | Fact_any_source _ | Fact_any_source_if_used _ _ =>
    []
  end.

Lemma read_in_nodes (fct : fact) (x : node) :
  x ∈ read_nodes fct -> x ∈ nodes_of fct.
Proof.
  destruct fct; cbn; intros H; repeat destruct H as [<- | H]; cbn; auto;
    contradiction.
Qed.

Ltac read_conj r Hok :=
  first
    [ lazymatch goal with
      | |- negb (mem_node r ?l) = true =>
        let Em := fresh "Em" in
        destruct (mem_node r l) eqn:Em; [ | reflexivity ];
        exfalso; apply mem_node_In in Em;
        cbn [f_any_usage facts_of_list] in Em;
        let fct := fresh "fct" in let Hf := fresh "Hf" in
        let Hy := fresh "Hy" in
        destruct (CH.flat_in _ _ _ Em) as [fct [Hf Hy]];
        destruct fct; cbn in Hy; try contradiction;
        destruct Hy as [Hy | []]; subst;
        let Hw := fresh "Hw" in
        destruct (Hok _ Hf (or_introl eq_refl)) as [Hw | [? Hw]];
        discriminate Hw
      end
    | let t := fresh "t" in let Hy := fresh "Hy" in
      let fct := fresh "fct" in let Hf := fresh "Hf" in
      let Hy' := fresh "Hy" in
      apply forallb_forall; intros t Hy;
      destruct (CH.flat_in _ _ _ Hy) as [fct [Hf Hy']];
      destruct fct; cbn in Hy'; try contradiction;
      destruct Hy' as [<- | []]; cbv beta iota;
      repeat match goal with
      | |- context [node_eqb ?z r] =>
        let E := fresh "E" in
        destruct (node_eqb z r) eqn:E;
        [ apply node_eqb_eq in E; subst z | ]
      end; cbn [negb orb andb]; try reflexivity;
      let Hw := fresh "Hw" in
      first [ pose proof (Hok _ Hf (or_introl eq_refl)) as Hw
            | pose proof (Hok _ Hf (or_intror (or_introl eq_refl))) as Hw ];
      destruct Hw as [Hw | [? Hw]]; try discriminate Hw;
      injection Hw; intros; subst; rewrite ?node_eqb_refl; reflexivity ].

(* Nothing reads [r] but aliases and propagations into [x]. *)
Lemma only_read_by_of (L : list fact) (r x : node) :
  (∀ fct, fct ∈ L -> r ∈ read_nodes fct ->
     fct = Fact_alias x r ∨ ∃ c, fct = Fact_propagate c x r) ->
  only_read_by (facts_of_list L) r x = true.
Proof.
  intros Hok. unfold only_read_by.
  repeat (apply andb_true_intro; split).
  all: read_conj r Hok.
Qed.

(* ======================================================================== *)
(* The variables of the unit                                                *)
(* ======================================================================== *)

(* The parameters of the handlers, and of the bodies of code. *)
Definition binding_hparams (b : static_binding) : list variable :=
  match b with
  | Bind_code _ c => param_variables (code_body c)
  | Bind_deleted_code _ | Bind_symbol _ _ => []
  end.

Fixpoint hparams (e : expr) : list variable :=
  match e with
  | Let _ _ _ body => hparams body
  | Let_static bindings body =>
    flat_map binding_hparams bindings ++ hparams body
  | Let_cont_non_recursive _ h body => handler_hparams h ++ hparams body
  | Let_cont_recursive handlers body =>
    flat_map (fun '(_, h) => handler_hparams h) handlers ++ hparams body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => []
  end
with handler_hparams (h : continuation_handler) : list variable :=
  match h with
  | Handler params _ handler => params ++ hparams handler
  end.

(* The parameters of the code of the unit. *)
Definition cparams_of (codes : list (code_id * code)) : list variable :=
  flat_map (fun p => code_params (snd p)) codes.

Lemma flat_map_flat {A B C : Type} (f : B -> list C) (g : A -> list B)
    (l : list A) :
  flat_map f (flat_map g l) = flat_map (fun x => flat_map f (g x)) l.
Proof.
  induction l as [ | x l IH]; [ reflexivity | ].
  cbn [flat_map]. rewrite flat_map_app, IH. reflexivity.
Qed.

Lemma other_perm :
  (∀ e, Permutation (UT.other_params e)
          (cparams_of (unit_codes e) ++ hparams e))
  ∧ (∀ h, Permutation (UT.handler_other_params h)
            (cparams_of (handler_unit_codes h) ++ handler_hparams h)).
Proof.
  unfold cparams_of.
  apply FlambdaReaperFlowProofs.expr_handler_ind;
    cbn [UT.other_params unit_codes hparams UT.handler_other_params
         handler_unit_codes handler_hparams];
    try (intros; apply Permutation_refl).
  - intros x kind d body IH. exact IH.
  - intros bindings body _ IH. rewrite flat_map_app.
    apply UT.perm_app4; [ | exact IH ].
    rewrite flat_map_flat. apply UT.perm_flat_map.
    intros [id [params mc ret exn md body'] | id | s k] _; cbn;
      rewrite ?app_nil_r; apply Permutation_refl.
  - intros k h body IHh IH. rewrite flat_map_app.
    apply UT.perm_app4; assumption.
  - intros handlers body IHh IH. rewrite flat_map_app.
    apply UT.perm_app4; [ | exact IH ].
    rewrite flat_map_flat. apply UT.perm_flat_map.
    intros [k h] Hin. exact (IHh k h Hin).
  - intros params md handler IH.
    eapply Permutation_trans; [ apply Permutation_app_head; exact IH | ].
    apply Permutation_app_swap_app.
Qed.

(* The variables of a unit: those of its [Let]s, the [my_closure]s and the
   parameters of its code, and those of [hparams]. *)
Lemma bound_split (e : expr) :
  variables_bound_once e = true ->
  NoDup (map fst (let_bindings e) ++ map UT.mc_of (unit_codes e)
         ++ cparams_of (unit_codes e) ++ hparams e).
Proof.
  intros H. pose proof (UT.bound_once_split e H) as Hnd.
  eapply Permutation_NoDup; [ | exact Hnd ].
  apply Permutation_app_head. apply Permutation_app_head.
  exact (proj1 other_perm e).
Qed.

Lemma hparams_in_params :
  ∀ e v, v ∈ hparams e -> v ∈ param_variables e.
Proof.
  intros e v H.
  apply (Permutation_in _ (Permutation_sym (proj1 UT.params_perm e))).
  apply in_or_app. right.
  apply (Permutation_in _ (Permutation_sym (proj1 other_perm e))).
  apply in_or_app. right. exact H.
Qed.

Lemma in_cparams (codes : list (code_id * code)) (id : code_id) (c : code)
    (p : variable) :
  (id, c) ∈ codes -> p ∈ code_params c -> p ∈ cparams_of codes.
Proof.
  intros H Hp. apply in_flat_map. exists (id, c). auto.
Qed.

Lemma nodup_flat_inj {A B : Type} (f : A -> list B) (l : list A)
    (a a' : A) (x : B) :
  NoDup (flat_map f l) -> a ∈ l -> a' ∈ l -> x ∈ f a -> x ∈ f a' ->
  a = a'.
Proof.
  induction l as [ | y l IH]; intros Hnd Ha Ha' Hx Hx'; [ destruct Ha | ].
  cbn [flat_map] in Hnd.
  destruct Ha as [<- | Ha], Ha' as [<- | Ha']; auto.
  - exfalso.
    apply (FlambdaReaperLeastProofs.BoundOnce.nodup_app_disjoint _ _ x Hnd
             Hx).
    apply in_flat_map. exists a'. auto.
  - exfalso.
    apply (FlambdaReaperLeastProofs.BoundOnce.nodup_app_disjoint _ _ x Hnd
             Hx').
    apply in_flat_map. exists a. auto.
  - exact (IH (NoDup_app_remove_l _ _ Hnd) Ha Ha' Hx Hx').
Qed.

(* The expression whose variables are those of the [Let]s of [e0] and of
   [hparams e0]: for FlambdaReaperUnusedCodeTraversalProofs.V0. *)
Definition md0 : handler_metadata :=
  {| hm_params_kinds := []; hm_params_subkinds := [];
     hm_is_exn_handler := false |}.

Definition e1 (e0 : expr) : expr :=
  Let_cont_non_recursive xH
    (Handler (map fst (let_bindings e0) ++ hparams e0) md0 Invalid) Invalid.

Lemma V0_e1 (e0 : expr) (v : variable) :
  UT.V0 (e1 e0) v <-> v ∈ map fst (let_bindings e0) ∨ v ∈ hparams e0.
Proof.
  unfold UT.V0, e1. cbn. rewrite !app_nil_r. split.
  - intros [[] | H]. apply in_app_or. exact H.
  - intros H. right. apply in_or_app. exact H.
Qed.

Lemma var_fact_e1 (e0 : expr) (cds : finite_map code_dep) (fct : fact)
    (v : variable) :
  UT.var_fact (e1 e0) cds fct -> Node_variable v ∈ into_nodes fct ->
  v ∈ map fst (let_bindings e0) ∨ v ∈ hparams e0.
Proof.
  intros H Hv. destruct (H v Hv) as [H' | (id & c & cd & Hin & _)].
  - exact (proj1 (V0_e1 e0 v) H').
  - cbn in Hin. destruct Hin.
Qed.

Lemma var_ok_e1 (e0 : expr) (v : variable) :
  v ∈ map fst (let_bindings e0) ∨ v ∈ hparams e0 ->
  UT.var_ok (e1 e0) (Node_variable v).
Proof. intros H w [= <-]. exact (proj2 (V0_e1 e0 v) H). Qed.

(* ======================================================================== *)
(* The invariant                                                            *)
(* ======================================================================== *)

Section Inv.

Variable e0 : expr.

(* The parameters and the witnesses of known arity of the code deps. *)
Definition spec (a : acc) (x : node) : Prop :=
  ∃ id cd, (id, cd) ∈ acc_code_deps a ∧ (x ∈ cd_params cd ∨ x = kw cd).

(* [fct] is the alias of [traverse_code] into the parameter [p] of the
   code [c] of [e0] from the node [r] of its rank. *)
Definition param_alias (a : acc) (fct : fact) (r : node) : Prop :=
  ∃ id c cd p, (id, c) ∈ unit_codes e0 ∧ (id, cd) ∈ acc_code_deps a
    ∧ (p, r) ∈ combine (code_params c) (cd_params cd)
    ∧ fct = Fact_alias (Node_variable p) r.

(* [fct] is an alias of [traverse_code] of the code [c] of [e0]. *)
Definition code_alias (a : acc) (fct : fact) : Prop :=
  ∃ id c cd, (id, c) ∈ unit_codes e0 ∧ (id, cd) ∈ acc_code_deps a
    ∧ (fct = Fact_alias (Node_variable (code_my_closure c)) (cd_my_closure cd)
       ∨ ∃ p r, (p, r) ∈ combine (code_params c) (cd_params cd)
           ∧ fct = Fact_alias (Node_variable p) r).

Record pinv (a : acc) : Prop := {
  pi_nodes : ∀ fct x,
    fct ∈ acc_facts a -> x ∈ nodes_of fct -> made_lt x (acc_next a);
  pi_cd : ∀ id cd x,
    (id, cd) ∈ acc_code_deps a -> x ∈ code_dep_nodes cd ->
    ∃ k, x = Node_made k ∧ (k < acc_next a)%positive;
  pi_cd_nodup : ∀ id cd,
    (id, cd) ∈ acc_code_deps a -> NoDup (code_dep_nodes cd);
  pi_cd_disj : ∀ id cd id' cd' x,
    (id, cd) ∈ acc_code_deps a -> (id', cd') ∈ acc_code_deps a ->
    x ∈ code_dep_nodes cd -> x ∈ code_dep_nodes cd' -> id = id' ∧ cd = cd';
  pi_reads : ∀ fct x,
    fct ∈ acc_facts a -> x ∈ read_nodes fct -> spec a x ->
    param_alias a fct x;
  pi_ad : ∀ d x,
    d ∈ acc_apply_deps a -> x ∈ CH.apply_dep_nodes d -> CH.good (spec a) a x;
  pi_cl : ∀ d,
    d ∈ acc_closure_deps a ->
    CH.good (spec a) a (cld_let_bound_name_of_the_closure d);
  pi_vars : ∀ fct,
    fct ∈ acc_facts a -> UT.var_fact (e1 e0) [] fct ∨ code_alias a fct;
  pi_ad_var : ∀ d,
    d ∈ acc_apply_deps a -> UT.var_ok (e1 e0) (ad_apply_call_witness d);
  pi_cl_var : ∀ d,
    d ∈ acc_closure_deps a ->
    UT.var_ok (e1 e0) (cld_let_bound_name_of_the_closure d)
}.

Definition envok (env : traverse_env) (a : acc) : Prop :=
  CH.env_good (spec a) env a ∧ UT.venv (e1 e0) env
  ∧ env_set_closures env = set_closures e0.

Definition pgrows (a a' : acc) : Prop :=
  (acc_next a <= acc_next a')%positive
  ∧ (∀ p, p ∈ acc_code_deps a -> p ∈ acc_code_deps a')
  ∧ ∀ x, spec a' x ->
      spec a x ∨ ∃ k, x = Node_made k ∧ (acc_next a <= k)%positive.

Definition pstep (env : traverse_env) (a a' : acc) : Prop :=
  pinv a -> envok env a -> pinv a' ∧ pgrows a a'.

Lemma pgrows_refl (a : acc) : pgrows a a.
Proof. split; [ lia | split; [ auto | intros x H; left; exact H ] ]. Qed.

Lemma pgrows_trans (a a1 a2 : acc) :
  pgrows a a1 -> pgrows a1 a2 -> pgrows a a2.
Proof.
  intros (H1 & H2 & H3) (H4 & H5 & H6).
  split; [ lia | split; [ auto | ] ].
  intros x Hx. destruct (H6 x Hx) as [H | (k & -> & Hk)].
  - destruct (H3 _ H) as [H' | (k & -> & Hk)]; [ left; exact H' | ].
    right. exists k. split; [ reflexivity | lia ].
  - right. exists k. split; [ reflexivity | lia ].
Qed.

Lemma envok_grows (env : traverse_env) (a a' : acc) :
  pgrows a a' -> envok env a -> envok env a'.
Proof.
  intros (Hn & _ & Hs) (Hg & Hv & Hc).
  split; [ | split; [ exact Hv | exact Hc ] ].
  apply (CH.env_good_grows (spec a) (spec a') a a' env); [ | exact Hg ].
  split; [ exact Hn | exact Hs ].
Qed.

Lemma pstep_refl (env : traverse_env) (a : acc) : pstep env a a.
Proof. intros H _. split; [ exact H | apply pgrows_refl ]. Qed.

Lemma pstep_trans (env : traverse_env) (a a1 a2 : acc) :
  pstep env a a1 -> pstep env a1 a2 -> pstep env a a2.
Proof.
  intros H1 H2 Hinv Henv. destruct (H1 Hinv Henv) as [Hinv1 Hg1].
  destruct (H2 Hinv1 (envok_grows _ _ _ Hg1 Henv)) as [Hinv2 Hg2].
  split; [ exact Hinv2 | exact (pgrows_trans _ _ _ Hg1 Hg2) ].
Qed.

Lemma pstep_fold {A : Type} (env : traverse_env) (f : acc -> A -> acc)
    (l : list A) (a : acc) :
  (∀ y b, y ∈ l -> pstep env b (f b y)) -> pstep env a (List.fold_left f l a).
Proof.
  revert a. induction l as [ | y l IH]; intros a H; cbn [List.fold_left].
  - apply pstep_refl.
  - apply (pstep_trans _ _ (f a y)); [ apply H; left; reflexivity | ].
    apply IH. intros y' b Hy'. apply H. right. exact Hy'.
Qed.

Lemma pstep_env (env env' : traverse_env) (a a' : acc) :
  (pinv a -> envok env a -> envok env' a) -> pstep env' a a' ->
  pstep env a a'.
Proof. intros H H' Hinv Henv. exact (H' Hinv (H Hinv Henv)). Qed.

(* ------------------------------------------------------------------------ *)
(* Basic lemmas                                                             *)
(* ------------------------------------------------------------------------ *)

Lemma param_alias_mono (a a' : acc) (fct : fact) (x : node) :
  (∀ p, p ∈ acc_code_deps a -> p ∈ acc_code_deps a') ->
  param_alias a fct x -> param_alias a' fct x.
Proof.
  intros H (id & c & cd & p & H1 & H2 & H3 & H4).
  exists id, c, cd, p. auto.
Qed.

Lemma code_alias_mono (a a' : acc) (fct : fact) :
  (∀ p, p ∈ acc_code_deps a -> p ∈ acc_code_deps a') ->
  code_alias a fct -> code_alias a' fct.
Proof.
  intros H (id & c & cd & H1 & H2 & H3). exists id, c, cd. auto.
Qed.

Lemma spec_made (a : acc) (x : node) :
  pinv a -> spec a x -> ∃ k, x = Node_made k ∧ (k < acc_next a)%positive.
Proof.
  intros Hinv (id & cd & Hcd & [Hx | ->]).
  - apply (pi_cd _ Hinv id cd x Hcd). right; right; right.
    apply in_or_app. left. exact Hx.
  - exact (pi_cd _ Hinv id cd _ Hcd (CH.in_cd_known cd)).
Qed.

(* A node [ok] in a step is made before its end, and of no code dep. *)
Lemma ok_spec (a a' : acc) (x : node) :
  pinv a -> (acc_next a <= acc_next a')%positive ->
  CH.ok_node (spec a) a a' x ->
  made_lt x (acc_next a') ∧ ~ spec a x.
Proof.
  intros Hinv Hle [H | [[H1 H2] | (k & -> & H1 & H2)]].
  - split; [ intros k E; destruct (H k E) | ].
    intros Hs. destruct (spec_made a x Hinv Hs) as (k & E & _).
    exact (H k E).
  - split; [ intros k E; specialize (H1 k E); lia | exact H2 ].
  - split; [ intros k' [= <-]; exact H2 | ].
    intros Hs. destruct (spec_made a _ Hinv Hs) as (k' & [= <-] & Hk).
    lia.
Qed.

Lemma good_of_ok (a a' : acc) (x : node) :
  pinv a -> (acc_next a <= acc_next a')%positive ->
  (∀ y, spec a' y -> spec a y) ->
  CH.ok_node (spec a) a a' x -> CH.good (spec a') a' x.
Proof.
  intros Hinv Hle Hs Hok. destruct (ok_spec a a' x Hinv Hle Hok) as [H1 H2].
  split; [ exact H1 | intros H; exact (H2 (Hs x H)) ].
Qed.

Lemma good_mono (a a' : acc) (x : node) :
  (acc_next a <= acc_next a')%positive ->
  (∀ y, spec a' y -> spec a y) ->
  CH.good (spec a) a x -> CH.good (spec a') a' x.
Proof.
  intros Hle Hs [H1 H2].
  split; [ intros k E; specialize (H1 k E); lia | ].
  intros H. exact (H2 (Hs x H)).
Qed.

(* The nodes of the facts of a step, of none of the code deps, and no code
   dep: as FlambdaReaperIndirectDeadFramesTraversalProofs.Chains.ext, but
   for the reads of the field of unknown arity and the code deps. *)
Record next (S : node -> Prop) (a0 a a' : acc) : Prop := {
  nx_next : (acc_next a <= acc_next a')%positive;
  nx_facts : ∃ L, acc_facts a' = L ++ acc_facts a
    ∧ ∀ fct x, fct ∈ L -> x ∈ nodes_of fct -> CH.ok_node S a0 a' x;
  nx_apply_deps : ∀ d, d ∈ acc_apply_deps a' ->
    d ∈ acc_apply_deps a
    ∨ ∀ x, x ∈ CH.apply_dep_nodes d -> CH.ok_node S a0 a' x;
  nx_closure_deps : ∀ d, d ∈ acc_closure_deps a' ->
    d ∈ acc_closure_deps a
    ∨ CH.ok_node S a0 a' (cld_let_bound_name_of_the_closure d)
}.

Lemma next_of (S : node -> Prop) (a0 a a' : acc) :
  CH.ext S a0 a a' -> next S a0 a a'.
Proof.
  intros [Hn (L & Hf & Hok & _) _ Hap Hcl].
  constructor; [ exact Hn | exists L; auto | exact Hap | exact Hcl ].
Qed.

Lemma next_trans (S : node -> Prop) (a0 a a1 a2 : acc) :
  next S a0 a a1 -> next S a0 a1 a2 -> next S a0 a a2.
Proof.
  intros [Hn1 (L1 & Hf1 & Hi1) Hap1 Hcl1] [Hn2 (L2 & Hf2 & Hi2) Hap2 Hcl2].
  constructor; [ lia | | | ].
  - exists (L2 ++ L1). split; [ rewrite Hf2, Hf1, app_assoc; reflexivity | ].
    intros fct x Hin Hx. apply in_app_or in Hin as [Hin | Hin];
      [ exact (Hi2 _ _ Hin Hx) | ].
    exact (CH.ok_node_mono S _ _ _ _ Hn2 (Hi1 _ _ Hin Hx)).
  - intros d Hin. destruct (Hap2 _ Hin) as [Hin' | H]; [ | right; auto ].
    destruct (Hap1 _ Hin') as [H' | H']; [ left; auto | right ].
    intros x Hx. exact (CH.ok_node_mono S _ _ _ _ Hn2 (H' x Hx)).
  - intros d Hin. destruct (Hcl2 _ Hin) as [Hin' | H]; [ | right; auto ].
    destruct (Hcl1 _ Hin') as [H' | H']; [ left; auto | right ].
    exact (CH.ok_node_mono S _ _ _ _ Hn2 H').
Qed.

(* The code deps that a step of [KnownWitnesses] leaves. *)
Lemma kw_code_deps (a0 a a' : acc) :
  KW.ext a0 a a' -> acc_code_deps a' = acc_code_deps a.
Proof. intros [_ _ Hcd _ _]. exact Hcd. Qed.

Lemma kw_env_good (env : traverse_env) (a : acc) :
  CH.env_good (spec a) env a -> KW.env_good env a.
Proof.
  intros H k ps x Hk Hx. destruct (H k ps x Hk Hx) as [H1 H2].
  split; [ exact H1 | ].
  intros (id & cd & Hcd & ->). apply H2. exists id, cd. auto.
Qed.

(* A step of the traversal but [prepare_code] and [traverse_code]. *)
Lemma pclose (a a' : acc) :
  pinv a ->
  next (spec a) a a a' -> acc_code_deps a' = acc_code_deps a ->
  (∀ fct, fct ∈ acc_facts a' ->
     fct ∈ acc_facts a ∨ UT.var_fact (e1 e0) [] fct) ->
  (∀ d, d ∈ acc_apply_deps a' ->
     d ∈ acc_apply_deps a ∨ UT.var_ok (e1 e0) (ad_apply_call_witness d)) ->
  (∀ d, d ∈ acc_closure_deps a' ->
     d ∈ acc_closure_deps a
     ∨ UT.var_ok (e1 e0) (cld_let_bound_name_of_the_closure d)) ->
  pinv a' ∧ pgrows a a'.
Proof.
  intros Hinv [Hn (L & HL & Hok) Hap Hcl] Hcd Hv Hapv Hclv.
  assert (Hs : ∀ y, spec a' y -> spec a y)
    by (intros y (id & cd & H1 & H2); rewrite Hcd in H1; exists id, cd; auto).
  assert (Hmono : ∀ p, p ∈ acc_code_deps a -> p ∈ acc_code_deps a')
    by (intros p H; rewrite Hcd; exact H).
  split; [ | split; [ exact Hn | split; [ exact Hmono | ] ] ];
    [ | intros x H; left; exact (Hs x H) ].
  constructor.
  - intros fct x Hin Hx. rewrite HL in Hin.
    apply in_app_or in Hin as [Hin | Hin].
    + exact (proj1 (ok_spec a a' x Hinv Hn (Hok _ _ Hin Hx))).
    + intros k E. pose proof (pi_nodes _ Hinv fct x Hin Hx k E). lia.
  - intros id cd x H Hx. rewrite Hcd in H.
    destruct (pi_cd _ Hinv id cd x H Hx) as (k & -> & Hk).
    exists k. split; [ reflexivity | lia ].
  - intros id cd H. rewrite Hcd in H. exact (pi_cd_nodup _ Hinv id cd H).
  - intros id cd id' cd' x H H'. rewrite Hcd in H, H'.
    exact (pi_cd_disj _ Hinv id cd id' cd' x H H').
  - intros fct x Hin Hx Hsx. rewrite HL in Hin.
    apply in_app_or in Hin as [Hin | Hin].
    + exfalso. apply (proj2 (ok_spec a a' x Hinv Hn
                               (Hok _ _ Hin (read_in_nodes _ _ Hx)))).
      exact (Hs x Hsx).
    + exact (param_alias_mono a a' fct x Hmono
               (pi_reads _ Hinv fct x Hin Hx (Hs x Hsx))).
  - intros d x Hd Hx. destruct (Hap d Hd) as [Hd' | H].
    + exact (good_mono a a' x Hn Hs (pi_ad _ Hinv d x Hd' Hx)).
    + exact (good_of_ok a a' x Hinv Hn Hs (H x Hx)).
  - intros d Hd. destruct (Hcl d Hd) as [Hd' | H].
    + exact (good_mono a a' _ Hn Hs (pi_cl _ Hinv d Hd')).
    + exact (good_of_ok a a' _ Hinv Hn Hs H).
  - intros fct Hin. destruct (Hv fct Hin) as [H | H]; [ | left; exact H ].
    destruct (pi_vars _ Hinv fct H) as [H' | H']; [ left; exact H' | right ].
    exact (code_alias_mono a a' fct Hmono H').
  - intros d Hd. destruct (Hapv d Hd) as [H | H]; [ | exact H ].
    exact (pi_ad_var _ Hinv d H).
  - intros d Hd. destruct (Hclv d Hd) as [H | H]; [ | exact H ].
    exact (pi_cl_var _ Hinv d H).
Qed.

(* With the variables of [vsafe]. *)
Lemma pclose_vsafe (a a' : acc) :
  pinv a -> next (spec a) a a a' -> acc_code_deps a' = acc_code_deps a ->
  UT.vsafe (e1 e0) [] a a' -> pinv a' ∧ pgrows a a'.
Proof.
  intros Hinv Hn Hcd (Hf & Hcl & Hap).
  apply (pclose a a' Hinv Hn Hcd Hf); [ | exact Hcl ].
  intros d Hd. left. rewrite <- Hap. exact Hd.
Qed.

(* ------------------------------------------------------------------------ *)
(* The steps that make no code dep                                          *)
(* ------------------------------------------------------------------------ *)

Lemma pstep_named (env : traverse_env) (x : variable) (d : named) (a : acc) :
  x ∈ map fst (let_bindings e0) ->
  env_set_closures env = set_closures e0 ->
  pstep env a (traverse_named env x d a).
Proof.
  intros Hx Hsets Hinv _.
  apply (pclose_vsafe a _ Hinv).
  - apply next_of. apply CH.ext_traverse_named. lia.
  - exact (kw_code_deps _ _ _ (KW.ext_traverse_named a env x d a
                                 (Pos.le_refl _))).
  - apply UT.vsafe_traverse_named.
    + apply (proj2 (V0_e1 e0 x)). left. exact Hx.
    + intros f y H. apply (proj2 (V0_e1 e0 y)). left.
      unfold set_closures_of in H. rewrite Hsets in H.
      destruct (lookup x (set_closures e0)) as [others | ] eqn:L;
        [ | destruct H ].
      exact (proj1 UT.set_closures_lets e0 x others f y
               (FlambdaMapProofs.lookup_in _ _ _ L) H).
Qed.

Lemma pstep_static_binding (env : traverse_env) (b : static_binding)
    (a : acc) :
  pstep env a (traverse_static_binding env b a).
Proof.
  intros Hinv _. apply (pclose_vsafe a _ Hinv).
  - apply next_of. apply CH.ext_traverse_static_binding.
  - exact (kw_code_deps _ _ _ (KW.ext_traverse_static_binding a env b a)).
  - apply UT.vsafe_traverse_static_binding.
Qed.

Lemma pstep_apply_cont (env : traverse_env) (ac : apply_cont) (a : acc) :
  pstep env a (apply_cont_deps env ac a).
Proof.
  intros Hinv (Hg & Hv & _). apply (pclose_vsafe a _ Hinv).
  - apply next_of. apply CH.ext_apply_cont_deps; [ exact Hg | lia ].
  - exact (kw_code_deps _ _ _ (KW.ext_apply_cont_deps a env ac a
                                 (kw_env_good env a Hg) (Pos.le_refl _))).
  - exact (UT.vsafe_apply_cont_deps (e1 e0) [] env ac a Hv).
Qed.

Lemma pstep_switch (env : traverse_env) (s : simple)
    (arms : list (Z * apply_cont)) (a : acc) :
  pstep env a (traverse_switch env s arms a).
Proof.
  intros Hinv (Hg & Hv & _). apply (pclose_vsafe a _ Hinv).
  - apply next_of. apply CH.ext_traverse_switch; [ exact Hg | lia ].
  - exact (kw_code_deps _ _ _ (KW.ext_traverse_switch a env s arms a
                                 (kw_env_good env a Hg) (Pos.le_refl _))).
  - exact (UT.vsafe_traverse_switch (e1 e0) [] env s arms a Hv).
Qed.

Lemma pstep_exn_handler (env : traverse_env) (h : continuation_handler)
    (a : acc) :
  (∀ p, p ∈ handler_params h -> UT.var_ok (e1 e0) p) ->
  pstep env a (exn_handler_deps env h a).
Proof.
  intros Hp Hinv _. apply (pclose_vsafe a _ Hinv).
  - apply next_of. apply CH.ext_exn_handler_deps.
  - exact (kw_code_deps _ _ _ (KW.ext_exn_handler_deps a env h a)).
  - exact (UT.vsafe_exn_handler_deps (e1 e0) [] env h a Hp).
Qed.

(* ------------------------------------------------------------------------ *)
(* Calls                                                                    *)
(* ------------------------------------------------------------------------ *)

(* The variables of a step that adds apply deps. *)
Definition vsafe1 (a a' : acc) : Prop :=
  UT.vsafe0 (e1 e0) [] a a'
  ∧ ∀ d, d ∈ acc_apply_deps a' ->
      d ∈ acc_apply_deps a ∨ UT.var_ok (e1 e0) (ad_apply_call_witness d).

Lemma vsafe1_of (a a' : acc) : UT.vsafe (e1 e0) [] a a' -> vsafe1 a a'.
Proof.
  intros H. split; [ exact (UT.vsafe_vsafe0 _ _ _ _ H) | ].
  intros d Hd. left. destruct H as (_ & _ & Hap). rewrite <- Hap. exact Hd.
Qed.

Lemma vsafe1_trans (a a1 a2 : acc) :
  vsafe1 a a1 -> vsafe1 a1 a2 -> vsafe1 a a2.
Proof.
  intros [H1 H2] [H3 H4]. split; [ exact (UT.vsafe0_trans _ _ _ _ _ H1 H3) | ].
  intros d Hd. destruct (H4 d Hd) as [H | H]; [ | right; exact H ].
  exact (H2 d H).
Qed.

Lemma vsafe1_add_apply (env : traverse_env) (id : code_id)
    (callee call_widget : node) (b : bool) (a : acc) :
  UT.var_ok (e1 e0) call_widget ->
  vsafe1 a (add_apply env id callee call_widget b a).
Proof.
  intros Hcw. split; [ exact (UT.vsafe0_add_apply _ _ env id callee
                                call_widget b a Hcw) | ].
  intros d Hd. unfold add_apply in Hd. destruct b.
  - cbn [fresh] in Hd. cbv beta iota in Hd.
    cbn [push_apply_dep acc_apply_deps add_facts] in Hd.
    destruct Hd as [<- | Hd]; [ right; apply UT.var_ok_made | left; exact Hd ].
  - cbn [push_apply_dep acc_apply_deps] in Hd.
    destruct Hd as [<- | Hd]; [ right; exact Hcw | left; exact Hd ].
Qed.

Lemma vsafe1_call_kind (env : traverse_env) (ap : apply_expr)
    (exn_arg : node) (return_args : list node) (a : acc) :
  (∀ x, x ∈ exn_arg :: return_args -> UT.var_ok (e1 e0) x) ->
  vsafe1 a (traverse_call_kind env ap exn_arg return_args a).
Proof.
  intros Hin. unfold traverse_call_kind. cbv zeta.
  apply (vsafe1_trans _ (note_simple (ap_callee ap) a));
    [ apply vsafe1_of, UT.vsafe_note_simple | ].
  set (a1 := note_simple (ap_callee ap) a).
  destruct (ap_call_kind ap) as [id | | code_ids].
  - destruct (UT.vsafe_make_known (e1 e0) [] env ap return_args exn_arg a1
                Hin)
      as [Hk Hcw].
    destruct (make_known_arity_apply_widget env ap return_args exn_arg a1)
      as [cw a2] eqn:E. cbn [fst snd] in *. cbv beta iota zeta.
    set (a3 := add_fact _ a2).
    assert (H3 : UT.vsafe (e1 e0) [] a1 a3).
    { apply (UT.vsafe_trans _ _ _ _ _ Hk). unfold a3, add_fact.
      apply UT.vsafe_add_facts. intros fct [<- | []] v Hv. cbn in Hv.
      destruct Hv as [Hv | []]. left. exact (Hcw v Hv). }
    apply (vsafe1_trans _ a3); [ apply vsafe1_of, H3 | ].
    destruct (env_should_preserve_direct_calls env);
      [ | destruct (negb (among id (env_unit_code_ids env))) | ];
      try (apply vsafe1_of, UT.vsafe_refl); apply vsafe1_add_apply;
      exact Hcw.
  - destruct (UT.vsafe_make_unknown (e1 e0) [] env ap return_args exn_arg a1
                Hin)
      as [Hk Hcw].
    destruct (make_unknown_arity_apply_widget env ap return_args exn_arg a1)
      as [cw a2] eqn:E. cbn [fst snd] in *. cbv beta iota.
    apply vsafe1_of. apply (UT.vsafe_trans _ _ _ _ _ Hk). unfold add_fact.
    apply UT.vsafe_add_facts. intros fct [<- | []] v Hv. cbn in Hv.
    destruct Hv as [Hv | []]. left. exact (Hcw v Hv).
  - destruct (UT.vsafe_make_known (e1 e0) [] env ap return_args exn_arg a1
                Hin)
      as [Hk Hcw].
    destruct (make_known_arity_apply_widget env ap return_args exn_arg a1)
      as [cw a2] eqn:E. cbn [fst snd] in *. cbv beta iota zeta.
    apply vsafe1_of. apply (UT.vsafe_trans _ _ _ _ _ Hk). unfold add_fact.
    apply UT.vsafe_add_facts. intros fct [<- | []] v Hv. cbn in Hv.
    destruct Hv as [Hv | []]. left. exact (Hcw v Hv).
Qed.

Lemma vsafe1_traverse_apply (env : traverse_env) (ap : apply_expr)
    (a : acc) :
  UT.venv (e1 e0) env -> vsafe1 a (traverse_apply env ap a).
Proof.
  intros Henv. unfold traverse_apply.
  destruct (UT.vsafe_find_cont (e1 e0) [] env (ap_continuation ap)
              (length (unarize (ap_return_arity ap))) a Henv) as [H1 H1o].
  destruct (find_cont env (ap_continuation ap) _ a) as [rets b1] eqn:E1.
  cbn [fst snd] in *. cbv beta iota.
  destruct (UT.vsafe_find_cont (e1 e0) [] env (ap_exn_continuation ap)
              (S (length (ap_exn_extra_args ap))) b1 Henv) as [H2 H2o].
  destruct (find_cont env (ap_exn_continuation ap) _ b1) as [eps b2] eqn:E2.
  cbn [fst snd] in *. cbv beta iota.
  apply (vsafe1_trans _ b1); [ apply vsafe1_of, H1 | ].
  apply (vsafe1_trans _ b2); [ apply vsafe1_of, H2 | ].
  destruct eps as [ | ep extra].
  - cbn [fresh]. cbv beta iota.
    match goal with
    | |- vsafe1 _ (traverse_call_kind _ _ _ _ ?b3) =>
      apply (vsafe1_trans _ b3);
        [ apply vsafe1_of, UT.vsafe_same; reflexivity | ]
    end.
    apply vsafe1_call_kind. intros x [<- | Hx];
      [ apply UT.var_ok_made | exact (H1o x Hx) ].
  - cbv zeta beta iota.
    pose proof (UT.vsafe_note_simples (e1 e0) [] (exn_extra_simples ap) b2)
      as H3.
    set (b3 := note_simples _ b2) in *.
    match goal with
    | |- vsafe1 _ (traverse_call_kind _ _ _ _ ?b4) =>
      apply (vsafe1_trans _ b4)
    end.
    + apply vsafe1_of. apply (UT.vsafe_trans _ _ _ _ _ H3).
      apply UT.vsafe_add_facts. intros fct Hf v Hv.
      apply in_map_iff in Hf as ([p s] & <- & Hc). cbn in Hv.
      destruct Hv as [Hv | []]. left.
      exact (H2o _ (or_intror (in_combine_l _ _ _ _ Hc)) v Hv).
    + apply vsafe1_call_kind. intros x [<- | Hx];
        [ exact (H2o _ (or_introl eq_refl)) | exact (H1o x Hx) ].
Qed.

(* The call of unknown arity: its chain is of nodes made in the step
   (FlambdaReaperIndirectDeadFramesTraversalProofs.Chains.deps_spec). *)
Lemma next_call_unknown (G : node -> Prop) (a0 : acc) (env : traverse_env)
    (ap : apply_expr) (exn_arg : node) (return_args : list node) (a : acc) :
  ap_call_kind ap = Indirect_unknown_arity ->
  (acc_next a0 <= acc_next a)%positive ->
  (∀ x, x ∈ exn_arg :: return_args -> CH.ok_node G a0 a x) ->
  next G a0 a (traverse_call_kind env ap exn_arg return_args a)
  ∧ acc_code_deps (traverse_call_kind env ap exn_arg return_args a)
    = acc_code_deps a.
Proof.
  intros Hkind Hle Hok. unfold traverse_call_kind. rewrite Hkind.
  cbv zeta.
  destruct (AW.note_simple_fields (ap_callee ap) a)
    as (N1 & N2 & N3 & N4 & N5).
  set (closure := node_of_simple (ap_callee ap)).
  set (a1 := note_simple (ap_callee ap) a) in *. clearbody a1.
  unfold make_unknown_arity_apply_widget. cbn [fresh]. cbv beta iota zeta.
  set (called := Node_made (acc_next a1)).
  set (b1 := add_fact (Fact_any_usage called) _).
  set (groups := at_least_one_group _).
  assert (Hb1 : acc_next b1 = Pos.succ (acc_next a))
    by (cbn; rewrite N1; reflexivity).
  destruct (AW.fresh_list_spec (length groups) b1)
    as (F1 & F2 & F3 & F4 & F5 & F6).
  pose proof (CH.fresh_list_nodup (length groups) b1) as Hnd.
  pose proof (CH.fresh_list_length (length groups) b1) as Hlen.
  destruct (fresh_list (length groups) b1) as [ws b2] eqn:Ews.
  cbn [fst snd] in *.
  destruct (AW.note_simples_fields (ap_args ap) b2)
    as (M1 & M2 & M3 & M4 & M5).
  set (b3 := note_simples (ap_args ap) b2) in *. clearbody b3.
  destruct ws as [ | first rest].
  { exfalso. unfold groups, at_least_one_group in Hlen.
    destruct (group_by_parameter _ _); discriminate. }
  set (p := acc_next a) in *.
  assert (Hcalled : called = Node_made p)
    by (unfold called; rewrite N1; reflexivity).
  clearbody called. subst called.
  assert (Hws : ∀ w, w ∈ first :: rest ->
                  ∃ m, w = Node_made m ∧ (Pos.succ p <= m)%positive
                       ∧ (m < acc_next b2)%positive).
  { intros w Hw. destruct (F6 w Hw) as (m & -> & H1 & H2). exists m.
    rewrite Hb1 in H1. auto. }
  assert (Hsnd : map snd (combine groups (first :: rest)) = first :: rest)
    by (apply CH.map_snd_combine; symmetry; exact Hlen).
  assert (HE : ∀ x, x ∈ exn_arg :: return_args -> made_lt x p).
  { intros x Hx k Hk.
    destruct (Hok x Hx) as [H | [[H1 _] | (k' & -> & _ & H2)]].
    - destruct (H k Hk).
    - specialize (H1 k Hk). unfold p. lia.
    - injection Hk as <-. exact H2. }
  assert (Hsep : CH.sep_ok (exn_arg :: Node_made p :: return_args)
                   (map snd (combine groups (first :: rest))) (acc_next b3)).
  { rewrite Hsnd. split; [ exact Hnd | split ].
    - intros w Hw. destruct (Hws w Hw) as (m & -> & _ & H2).
      intros k Hk0. injection Hk0 as <-. rewrite M1. exact H2.
    - intros x Hx. split.
      + intros k Hk0. rewrite M1.
        destruct Hx as [<- | [<- | Hx]].
        * pose proof (HE _ (or_introl eq_refl) k Hk0). lia.
        * injection Hk0 as <-. lia.
        * pose proof (HE _ (or_intror Hx) k Hk0). lia.
      + intros Hw. destruct (Hws x Hw) as (m & -> & H1 & _).
        destruct Hx as [Hx | [Hx | Hx]].
        * pose proof (HE _ (or_introl eq_refl) m Hx). lia.
        * injection Hx as ->. lia.
        * pose proof (HE _ (or_intror Hx) m eq_refl). lia. }
  destruct (CH.deps_spec return_args exn_arg (Node_made p)
              (combine groups (first :: rest)) b3 Hsep)
    as (D & HD & Hcd & Had & Hcl & Hn4 & _ & _ & _ & _ & HD5).
  rewrite Hsnd in HD5.
  set (b4 := unknown_arity_apply_deps return_args exn_arg (Node_made p)
               (combine groups (first :: rest)) b3) in *.
  clearbody b4. cbv beta iota.
  set (an := Node_made (acc_next b4)).
  set (CA := match env_current_code_id env with
             | None => Fact_alias first an
             | Some id => Fact_propagate (Node_code_id id) first an
             end).
  set (b5 := Acc (Pos.succ (acc_next b4)) _ _ _ _ _).
  assert (Ea2 : acc_facts (cond_alias env an first b5) = CA :: acc_facts b4
                ∧ acc_next (cond_alias env an first b5)
                  = Pos.succ (acc_next b4)
                ∧ acc_code_deps (cond_alias env an first b5) = acc_code_deps b4
                ∧ acc_apply_deps (cond_alias env an first b5)
                  = acc_apply_deps b4
                ∧ acc_closure_deps (cond_alias env an first b5)
                  = acc_closure_deps b4).
  { unfold cond_alias, CA. destruct (env_current_code_id env); cbn;
      repeat split. }
  destruct Ea2 as (Ef & En & Ec & Eap & Ecl).
  set (a2 := cond_alias env an first b5) in *. clearbody a2.
  set (au := Fact_any_usage (Node_made p)).
  set (Lw := Fact_accessor an (Field_call_witness Unknown_arity_code_pointer)
               closure :: CA :: D ++ [au]).
  assert (Hfacts :
    acc_facts (add_fact (Fact_accessor an
                           (Field_call_witness Unknown_arity_code_pointer)
                           closure) a2)
    = Lw ++ acc_facts a).
  { cbn [add_fact add_facts acc_facts]. rewrite Ef, HD, M2, F1.
    unfold b1. cbn [add_fact add_facts acc_facts]. rewrite N2.
    unfold Lw. cbn [app]. rewrite <- app_assoc. reflexivity. }
  assert (Hdeps :
    acc_code_deps (add_fact (Fact_accessor an
                               (Field_call_witness Unknown_arity_code_pointer)
                               closure) a2)
    = acc_code_deps a
    ∧ acc_apply_deps (add_fact (Fact_accessor an
                                  (Field_call_witness
                                     Unknown_arity_code_pointer)
                                  closure) a2)
      = acc_apply_deps a
    ∧ acc_closure_deps (add_fact (Fact_accessor an
                                    (Field_call_witness
                                       Unknown_arity_code_pointer)
                                    closure) a2)
      = acc_closure_deps a).
  { cbn [add_fact add_facts acc_code_deps acc_apply_deps acc_closure_deps].
    rewrite Ec, Eap, Ecl, Hcd, Had, Hcl, M3, M4, M5, F2, F3, F4.
    unfold b1. cbn. rewrite N3, N4, N5. auto. }
  destruct Hdeps as (Dc & Dap & Dcl).
  assert (Hnext :
    acc_next (add_fact (Fact_accessor an
                          (Field_call_witness Unknown_arity_code_pointer)
                          closure) a2)
    = Pos.succ (acc_next b4)) by exact En.
  set (a' := add_fact _ a2) in *. clearbody a'.
  assert (Hb2 : (Pos.succ p <= acc_next b2)%positive)
    by (rewrite <- Hb1; exact F5).
  assert (Hb4 : (acc_next b2 <= acc_next b4)%positive)
    by (rewrite <- M1; exact Hn4).
  assert (Hfr : ∀ k, (p <= k)%positive -> (k < Pos.succ (acc_next b4))%positive
                 -> CH.ok_node G a0 a' (Node_made k)).
  { intros k H1 H2. apply CH.ok_made; [ unfold p in H1; lia | ].
    rewrite Hnext. exact H2. }
  split; [ | exact Dc ].
  constructor.
  - rewrite Hnext. unfold p in *. lia.
  - exists Lw. split; [ exact Hfacts | ].
    intros fct x Hf Hx. destruct Hf as [<- | [<- | Hf]].
    + cbn in Hx. destruct Hx as [<- | [<- | []]].
      * apply Hfr; unfold an; lia.
      * apply CH.ok_not_made. apply AW.not_made_simple.
    + unfold CA in Hx. destruct (Hws first (or_introl eq_refl))
        as (m0 & Hm0 & Hm1 & Hm2).
      destruct (env_current_code_id env); cbn in Hx;
        repeat destruct Hx as [<- | Hx]; try contradiction;
        first [ rewrite Hm0; apply Hfr; lia
              | apply Hfr; unfold an; lia
              | apply CH.ok_not_made; intros k; discriminate ].
    + apply in_app_or in Hf as [Hf | [<- | []]].
      * destruct (HD5 fct x Hf Hx) as [Hx' | [Hx' | [Hx' | Hx']]].
        -- destruct Hx' as [<- | [<- | Hx']].
           ++ apply (CH.ok_node_mono G a0 a); [ rewrite Hnext; lia | ].
              exact (Hok _ (or_introl eq_refl)).
           ++ apply Hfr; lia.
           ++ apply (CH.ok_node_mono G a0 a); [ rewrite Hnext; lia | ].
              exact (Hok _ (or_intror Hx')).
        -- destruct Hx' as (m & -> & H1 & H2). apply Hfr; rewrite ?M1 in H1;
             lia.
        -- destruct (Hws x Hx') as (m & -> & H1 & H2). apply Hfr; lia.
        -- apply CH.ok_not_made. exact Hx'.
      * cbn in Hx. destruct Hx as [<- | []]. apply Hfr; lia.
  - intros d Hd. left. rewrite <- Dap. exact Hd.
  - intros d Hd. left. rewrite <- Dcl. exact Hd.
Qed.

Lemma next_call_kind (G : node -> Prop) (a0 : acc) (env : traverse_env)
    (ap : apply_expr) (exn_arg : node) (return_args : list node) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  (∀ x, x ∈ exn_arg :: return_args -> CH.ok_node G a0 a x) ->
  next G a0 a (traverse_call_kind env ap exn_arg return_args a).
Proof.
  intros Hle Hok.
  assert (Hdec : ap_call_kind ap = Indirect_unknown_arity
                 ∨ ap_call_kind ap <> Indirect_unknown_arity)
    by (destruct (ap_call_kind ap);
        (left; reflexivity) || (right; discriminate)).
  destruct Hdec as [Hk | Hk].
  - exact (proj1 (next_call_unknown G a0 env ap exn_arg return_args a Hk Hle
                    Hok)).
  - apply next_of. exact (CH.ext_call_known G a0 env ap exn_arg return_args a
                            Hk Hle Hok).
Qed.

Lemma next_traverse_apply (G : node -> Prop) (env : traverse_env)
    (ap : apply_expr) (a : acc) :
  CH.env_good G env a -> next G a a (traverse_apply env ap a).
Proof.
  intros Henv. unfold traverse_apply.
  destruct (CH.ext_find_cont G a env (ap_continuation ap)
              (length (unarize (ap_return_arity ap))) a Henv)
    as [H1 H1o]; [ lia | ].
  pose proof (CH.ext_next _ _ _ _ H1) as Hn1.
  destruct (find_cont env (ap_continuation ap) _ a) as [rets b1] eqn:E1.
  cbn [fst snd] in *. cbv beta iota.
  destruct (CH.ext_find_cont G a env (ap_exn_continuation ap)
              (S (length (ap_exn_extra_args ap))) b1 Henv)
    as [H2 H2o]; [ lia | ].
  pose proof (CH.ext_next _ _ _ _ H2) as Hn2.
  destruct (find_cont env (ap_exn_continuation ap) _ b1) as [eps b2] eqn:E2.
  cbn [fst snd] in *. cbv beta iota.
  assert (Hpre : ∀ (exn_arg : node) (b3 : acc),
            next G a b2 b3 -> CH.ok_node G a b3 exn_arg ->
            next G a a (traverse_call_kind env ap exn_arg rets b3)).
  { intros exn_arg b3 H3 Hex.
    pose proof (nx_next _ _ _ _ H3) as Hn3.
    assert (H03 : next G a a b3)
      by exact (next_trans _ _ _ _ _
                  (next_of _ _ _ _ (CH.ext_trans _ _ _ _ _ H1 H2)) H3).
    apply (next_trans _ _ _ _ _ H03).
    apply next_call_kind; [ lia | ].
    intros x [<- | Hx]; [ exact Hex | ].
    apply (CH.ok_node_mono G a b1); [ lia | exact (H1o x Hx) ]. }
  destruct eps as [ | ep extra].
  - destruct (fresh b2) as [x b3] eqn:E3. cbv beta iota.
    destruct (CH.fresh_spec G a b2 x b3 E3) as (_ & _ & He & Ho); [ lia | ].
    exact (Hpre x b3 (next_of _ _ _ _ He) Ho).
  - cbv zeta beta iota.
    pose proof (CH.ext_note_simples G a (exn_extra_simples ap) b2) as H3.
    destruct (AW.note_simples_fields (exn_extra_simples ap) b2)
      as (N1 & _).
    set (b3 := note_simples _ b2) in *. clearbody b3.
    apply Hpre.
    + apply next_of. apply (CH.ext_trans _ _ _ _ _ H3).
      apply CH.ext_add_facts.
      * intros fct y H Hy. apply in_map_iff in H.
        destruct H as [[p s] [<- Hc]]. cbn in Hy.
        destruct Hy as [<- | [<- | []]]; [ | apply CH.ok_simple ].
        apply (CH.ok_node_mono G a b2); [ rewrite N1; lia | ].
        exact (H2o p (or_intror (in_combine_l _ _ _ _ Hc))).
      * intros w n H. apply in_map_iff in H.
        destruct H as [[p s] [H _]]. discriminate.
    + apply (CH.ok_node_mono G a b2);
        [ cbn [acc_next add_facts]; rewrite N1; lia | ].
      exact (H2o ep (or_introl eq_refl)).
Qed.

Lemma pstep_apply (env : traverse_env) (ap : apply_expr) (a : acc) :
  pstep env a (traverse_apply env ap a).
Proof.
  intros Hinv (Hg & Hv & _).
  destruct (vsafe1_traverse_apply env ap a Hv) as [[Hf Hcl] Hap].
  apply (pclose a _ Hinv); [ | | exact Hf | exact Hap | exact Hcl ].
  - exact (next_traverse_apply (spec a) env ap a Hg).
  - exact (kw_code_deps _ _ _ (KW.ext_traverse_apply a env ap a
                                 (kw_env_good env a Hg) (Pos.le_refl _))).
Qed.

(* ======================================================================== *)
(* The facts that [prepare_code] makes                                      *)
(* ======================================================================== *)

Ltac fsplit H :=
  repeat match type of H with
  | In _ (_ ++ _) => apply in_app_or in H; destruct H as [H | H]
  | In _ (map _ _) =>
    let y := fresh "y" in let Hy := fresh "Hy" in
    apply in_map_iff in H; destruct H as [y [H Hy]]
  | In _ (_ :: _) => destruct H as [H | H]
  | In _ [] => destruct H
  end;
  repeat match goal with y : prod _ _ |- _ => destruct y end;
  subst.

Lemma not_made_code_id (id : code_id) : not_made (Node_code_id id).
Proof. intros k. discriminate. Qed.

Lemma create_known_spec (id : code_id) (params returns : list node)
    (exn : node) (a : acc) :
  fst (create_known_arity_call_witness id params returns exn a)
  = Node_made (acc_next a)
  ∧ acc_next (snd (create_known_arity_call_witness id params returns exn a))
    = Pos.succ (acc_next a)
  ∧ (∃ L,
       acc_facts (snd (create_known_arity_call_witness id params returns
                         exn a))
       = L ++ acc_facts a
       ∧ ∀ fct x, fct ∈ L -> x ∈ read_nodes fct ->
           not_made x ∨ x ∈ exn :: returns)
  ∧ acc_code_deps (snd (create_known_arity_call_witness id params returns
                          exn a))
    = acc_code_deps a
  ∧ acc_apply_deps (snd (create_known_arity_call_witness id params returns
                           exn a))
    = acc_apply_deps a
  ∧ acc_closure_deps (snd (create_known_arity_call_witness id params returns
                             exn a))
    = acc_closure_deps a.
Proof.
  unfold create_known_arity_call_witness. cbn [fresh]. cbv beta iota zeta.
  cbn [fst snd add_facts acc_next acc_facts acc_code_deps acc_apply_deps
       acc_closure_deps].
  split; [ reflexivity | split; [ reflexivity | split; [ | auto ] ] ].
  eexists. split; [ reflexivity | ].
  intros fct x Hf Hx. fsplit Hf; cbn in Hx;
    repeat destruct Hx as [<- | Hx]; try contradiction;
    first [ left; apply not_made_code_id
          | right; left; reflexivity
          | right; right; eapply AW.indexed_in; eassumption ].
Qed.

Lemma witness_deps_spec (id : code_id) (returns : list node) (exn : node)
    (pw : list (list node * node)) (a : acc) :
  (acc_next a
   <= acc_next (unknown_arity_witness_deps id returns exn pw a))%positive
  ∧ (∃ L,
       acc_facts (unknown_arity_witness_deps id returns exn pw a)
       = L ++ acc_facts a
       ∧ ∀ fct x, fct ∈ L -> x ∈ read_nodes fct ->
           not_made x ∨ x ∈ exn :: returns ∨ x ∈ map snd pw
           ∨ CH.fresh_in (acc_next a)
               (acc_next (unknown_arity_witness_deps id returns exn pw a)) x)
  ∧ acc_code_deps (unknown_arity_witness_deps id returns exn pw a)
    = acc_code_deps a
  ∧ acc_apply_deps (unknown_arity_witness_deps id returns exn pw a)
    = acc_apply_deps a
  ∧ acc_closure_deps (unknown_arity_witness_deps id returns exn pw a)
    = acc_closure_deps a.
Proof.
  revert a. induction pw as [ | [first w] rest IH]; intros a;
    cbn [unknown_arity_witness_deps].
  - split; [ lia | split; [ | auto ] ].
    exists []. split; [ reflexivity | intros fct x [] ].
  - set (b := add_facts _ a).
    set (Lb := map (fun '(i, p) => Fact_parameter w i p) (indexed first)
               ++ [Fact_constructor w Field_code_id_of_call_witness
                     (Node_code_id id)]).
    assert (Hb : acc_facts b = Lb ++ acc_facts a ∧ acc_next b = acc_next a
                 ∧ acc_code_deps b = acc_code_deps a
                 ∧ acc_apply_deps b = acc_apply_deps a
                 ∧ acc_closure_deps b = acc_closure_deps a)
      by (cbn; auto).
    destruct Hb as (B1 & B2 & B3 & B4 & B5).
    assert (HLb : ∀ fct x, fct ∈ Lb -> x ∈ read_nodes fct -> not_made x).
    { intros fct x Hf Hx. unfold Lb in Hf. fsplit Hf; cbn in Hx;
        repeat destruct Hx as [<- | Hx]; try contradiction.
      apply not_made_code_id. }
    clearbody b.
    destruct rest as [ | [g nw] rest'].
    + cbn [add_facts acc_next acc_facts acc_code_deps acc_apply_deps
           acc_closure_deps].
      rewrite B2, B3, B4, B5. split; [ lia | split; [ | auto ] ].
      eexists. split; [ rewrite B1, app_assoc; reflexivity | ].
      intros fct x Hf Hx. apply in_app_or in Hf as [Hf | Hf].
      * fsplit Hf; cbn in Hx; repeat destruct Hx as [<- | Hx];
          try contradiction;
          first [ right; left; left; reflexivity
                | right; left; right; eapply AW.indexed_in; eassumption ].
      * left. exact (HLb fct x Hf Hx).
    + cbn [fresh]. cbv beta iota zeta.
      set (c := add_facts _ _).
      destruct (IH c) as (I1 & (L & I2 & I3) & I4 & I5 & I6).
      set (a' := unknown_arity_witness_deps id returns exn _ c) in *.
      assert (Hc : acc_next c = Pos.succ (acc_next b)
                   ∧ acc_facts c
                     = [Fact_constructor w (Field_return_of_call (Normal 0))
                          (Node_made (acc_next b));
                        Fact_constructor (Node_made (acc_next b))
                          (Field_call_witness Unknown_arity_code_pointer) nw]
                       ++ acc_facts b
                   ∧ acc_code_deps c = acc_code_deps b
                   ∧ acc_apply_deps c = acc_apply_deps b
                   ∧ acc_closure_deps c = acc_closure_deps b)
        by (cbn; auto).
      destruct Hc as (C1 & C2 & C3 & C4 & C5).
      clearbody c a'.
      split; [ rewrite C1 in I1; lia | ].
      split; [ | rewrite I4, I5, I6, C3, C4, C5; auto ].
      exists (L ++ [Fact_constructor w (Field_return_of_call (Normal 0))
                      (Node_made (acc_next b));
                    Fact_constructor (Node_made (acc_next b))
                      (Field_call_witness Unknown_arity_code_pointer) nw]
              ++ Lb).
      split; [ rewrite I2, C2, B1, !app_assoc; reflexivity | ].
      intros fct x Hf Hx. apply in_app_or in Hf as [Hf | Hf].
      * destruct (I3 fct x Hf Hx) as [H | [H | [H | H]]];
          [ left; exact H | right; left; exact H | right; right; left;
            right; exact H | ].
        right; right; right. destruct H as (k & -> & H1 & H2).
        exists k. split; [ reflexivity | rewrite C1, B2 in H1; lia ].
      * apply in_app_or in Hf as [Hf | Hf]; [ | left; exact (HLb _ _ Hf Hx) ].
        fsplit Hf; cbn in Hx; repeat destruct Hx as [<- | Hx];
          try contradiction.
        -- right; right; right. exists (acc_next b).
           split; [ reflexivity | rewrite C1 in I1; lia ].
        -- right; right; left. right. left. reflexivity.
Qed.

Lemma create_unknown_spec (id : code_id) (is_tupled : bool) (ar : arity)
    (params returns : list node) (exn : node) (a : acc) :
  let r := create_unknown_arity_call_witnesses id is_tupled ar params returns
             exn a in
  (acc_next a <= acc_next (snd r))%positive
  ∧ NoDup (fst r)
  ∧ (∀ x, x ∈ fst r -> CH.fresh_in (acc_next a) (acc_next (snd r)) x)
  ∧ (∃ L, acc_facts (snd r) = L ++ acc_facts a
       ∧ ∀ fct x, fct ∈ L -> x ∈ read_nodes fct ->
           not_made x ∨ x ∈ exn :: returns
           ∨ CH.fresh_in (acc_next a) (acc_next (snd r)) x)
  ∧ acc_code_deps (snd r) = acc_code_deps a
  ∧ acc_apply_deps (snd r) = acc_apply_deps a
  ∧ acc_closure_deps (snd r) = acc_closure_deps a.
Proof.
  cbv zeta. unfold create_unknown_arity_call_witnesses. destruct is_tupled.
  - unfold create_unknown_arity_tupled_call_witnesses. cbn [fresh].
    cbv beta iota zeta.
    cbn [fst snd add_facts acc_next acc_facts acc_code_deps acc_apply_deps
         acc_closure_deps].
    split; [ lia | split; [ repeat constructor; intros [] | split ] ].
    { intros x [<- | []]. exists (acc_next a). split; [ reflexivity | lia ]. }
    split; [ | auto ].
    eexists. split; [ reflexivity | ].
    intros fct x Hf Hx. fsplit Hf; cbn in Hx;
      repeat destruct Hx as [<- | Hx]; try contradiction;
      first [ left; apply not_made_code_id
            | right; left; left; reflexivity
            | right; left; right; eapply AW.indexed_in; eassumption
            | right; right; eexists; split; [ reflexivity | lia ] ].
  - unfold create_unknown_arity_non_tupled_call_witnesses.
    set (groups := at_least_one_group _).
    destruct (AW.fresh_list_spec (length groups) a)
      as (F1 & F2 & F3 & F4 & F5 & F6).
    pose proof (CH.fresh_list_nodup (length groups) a) as Hnd.
    pose proof (CH.fresh_list_length (length groups) a) as Hlen.
    destruct (fresh_list (length groups) a) as [ws b] eqn:Ews.
    cbn [fst snd] in *.
    destruct (witness_deps_spec id returns exn (combine groups ws) b)
      as (W1 & (L & W2 & W3) & W4 & W5 & W6).
    set (b' := unknown_arity_witness_deps _ _ _ _ b) in *. clearbody b'.
    cbn [fst snd].
    split; [ lia | split; [ exact Hnd | split ] ].
    { intros x Hx. destruct (F6 x Hx) as (m & -> & H1 & H2).
      exists m. split; [ reflexivity | lia ]. }
    split; [ | rewrite W4, W5, W6, F2, F3, F4; auto ].
    exists L. split; [ rewrite W2, F1; reflexivity | ].
    intros fct x Hf Hx.
    destruct (W3 fct x Hf Hx) as [H | [H | [H | (m & -> & H1 & H2)]]];
      [ left; exact H | right; left; exact H | | ].
    + right; right. rewrite CH.map_snd_combine in H; [ | symmetry; exact Hlen ].
      destruct (F6 x H) as (m & -> & H1 & H2).
      exists m. split; [ reflexivity | lia ].
    + right; right. exists m. split; [ reflexivity | lia ].
Qed.

Lemma nodup_app_intro {A : Type} (l l' : list A) :
  NoDup l -> NoDup l' -> (∀ x, x ∈ l -> x ∈ l' -> False) -> NoDup (l ++ l').
Proof.
  induction l as [ | y l IH]; intros H H' Hd; [ exact H' | ].
  inversion H as [ | y' l0 Hy Hl ]; subst. cbn [app]. constructor.
  - intros Hin. apply in_app_or in Hin as [Hin | Hin]; [ exact (Hy Hin) | ].
    exact (Hd y (or_introl eq_refl) Hin).
  - apply IH; [ exact Hl | exact H' | ]. intros x Hx Hx'.
    exact (Hd x (or_intror Hx) Hx').
Qed.

(* [prepare_code] makes a code dep of nodes made in the step, distinct,
   and facts that read none of its parameters or its witness of known
   arity. *)
Lemma prepare_spec (id : code_id) (c : code) (a : acc) :
  ∃ cd L,
    acc_code_deps (prepare_code id c a) = (id, cd) :: acc_code_deps a
    ∧ acc_facts (prepare_code id c a) = L ++ acc_facts a
    ∧ acc_apply_deps (prepare_code id c a) = acc_apply_deps a
    ∧ acc_closure_deps (prepare_code id c a) = acc_closure_deps a
    ∧ (acc_next a <= acc_next (prepare_code id c a))%positive
    ∧ NoDup (code_dep_nodes cd)
    ∧ (∀ x, x ∈ code_dep_nodes cd ->
         CH.fresh_in (acc_next a) (acc_next (prepare_code id c a)) x)
    ∧ (∀ fct x, fct ∈ L -> x ∈ read_nodes fct ->
         ~ (x ∈ cd_params cd ∨ x = kw cd)).
Proof.
  unfold prepare_code.
  destruct (AW.fresh_list_spec (length (code_result_kinds c)) a)
    as (R1 & R2 & R3 & R4 & R5 & R6).
  pose proof (CH.fresh_list_nodup (length (code_result_kinds c)) a) as Rnd.
  destruct (fresh_list _ a) as [returns b1] eqn:E1. cbn [fst snd] in *.
  cbn [fresh]. cbv beta iota.
  cbn [acc_next acc_facts acc_code_deps acc_apply_deps acc_closure_deps
       acc_symbols].
  set (n1 := acc_next b1).
  set (b3 := Acc (Pos.succ (Pos.succ n1)) _ _ _ _ _).
  destruct (AW.fresh_list_spec (length (code_params_kinds c)) b3)
    as (P1 & P2 & P3 & P4 & P5 & P6).
  pose proof (CH.fresh_list_nodup (length (code_params_kinds c)) b3) as Pnd.
  destruct (fresh_list _ b3) as [params b4] eqn:E4. cbn [fst snd] in *.
  cbv beta iota.
  destruct (create_known_spec id params returns (Node_made n1) b4)
    as (K1 & K2 & (LK & K3 & K4) & K5 & K6 & K7).
  destruct (create_known_arity_call_witness id params returns (Node_made n1)
              b4) as [known b5] eqn:E5. cbn [fst snd] in *. cbv beta iota.
  destruct (create_unknown_spec id (code_is_tupled c) (code_params_arity c)
              params returns (Node_made n1) b5)
    as (U1 & U2 & U3 & (LU & U4 & U5) & U6 & U7 & U8).
  destruct (create_unknown_arity_call_witnesses _ _ _ _ _ _ b5)
    as [unknown b6] eqn:E6. cbn [fst snd] in *. cbv beta iota zeta.
  cbn [add_code_dep add_fact add_facts acc_code_deps acc_facts acc_apply_deps
       acc_closure_deps acc_next].
  assert (Hb3 : acc_next b3 = Pos.succ (Pos.succ n1)) by reflexivity.
  assert (Hb3f : acc_facts b3 = acc_facts b1 ∧ acc_code_deps b3
                 = acc_code_deps b1 ∧ acc_apply_deps b3 = acc_apply_deps b1
                 ∧ acc_closure_deps b3 = acc_closure_deps b1)
    by (cbn; auto).
  destruct Hb3f as (G1 & G2 & G3 & G4).
  clearbody b3.
  eexists. exists (Fact_any_source (Node_code_id id) :: LU ++ LK).
  split; [ rewrite U6, K5, P2, G2, R2; reflexivity | ].
  split; [ rewrite U4, K3, P1, G1, R1; cbn [app]; rewrite <- app_assoc;
           reflexivity | ].
  split; [ rewrite U7, K6, P3, G3, R3; reflexivity | ].
  split; [ rewrite U8, K7, P4, G4, R4; reflexivity | ].
  assert (Hn1 : (acc_next a <= n1)%positive) by exact R5.
  split; [ lia | ].
  cbn [code_dep_nodes cd_my_closure cd_exn cd_known_arity_call_witness
       cd_params cd_return cd_unknown_arity_call_witnesses].
  rewrite K1.
  (* The ranges of the nodes made. *)
  assert (Hret : ∀ x, x ∈ returns ->
             ∃ m, x = Node_made m ∧ (acc_next a <= m)%positive
                  ∧ (m < n1)%positive) by exact R6.
  assert (Hpar : ∀ x, x ∈ params ->
             ∃ m, x = Node_made m ∧ (Pos.succ (Pos.succ n1) <= m)%positive
                  ∧ (m < acc_next b4)%positive)
    by (intros x Hx; rewrite <- Hb3; exact (P6 x Hx)).
  assert (Hunk : ∀ x, x ∈ unknown ->
             ∃ m, x = Node_made m ∧ (Pos.succ (acc_next b4) <= m)%positive
                  ∧ (m < acc_next b6)%positive)
    by (intros x Hx; rewrite <- K2; exact (U3 x Hx)).
  split; [ | split ].
  - (* distinct *)
    repeat apply NoDup_cons.
    + intros [H | [H | H]]; [ injection H as H; lia | injection H as H; lia
                              | ].
      apply in_app_or in H as [H | H];
        [ destruct (Hpar _ H) as (m & [= <-] & H1 & _); lia | ].
      apply in_app_or in H as [H | H];
        [ destruct (Hret _ H) as (m & [= <-] & _ & H2); lia | ].
      destruct (Hunk _ H) as (m & [= <-] & H1 & _); lia.
    + intros [H | H]; [ injection H as H; lia | ].
      apply in_app_or in H as [H | H];
        [ destruct (Hpar _ H) as (m & [= <-] & H1 & _); lia | ].
      apply in_app_or in H as [H | H];
        [ destruct (Hret _ H) as (m & [= <-] & _ & H2); lia | ].
      destruct (Hunk _ H) as (m & [= <-] & H1 & _); lia.
    + intros H. apply in_app_or in H as [H | H];
        [ destruct (Hpar _ H) as (m & [= <-] & _ & H2); lia | ].
      apply in_app_or in H as [H | H];
        [ destruct (Hret _ H) as (m & [= <-] & _ & H2); lia | ].
      destruct (Hunk _ H) as (m & [= <-] & H1 & _); lia.
    + apply nodup_app_intro; [ exact Pnd | | ].
      * apply nodup_app_intro; [ exact Rnd | exact U2 | ].
        intros x H H'. destruct (Hret _ H) as (m & -> & _ & H2).
        destruct (Hunk _ H') as (m' & [= <-] & H1 & _). lia.
      * intros x H H'. destruct (Hpar _ H) as (m & -> & H1 & H2).
        apply in_app_or in H' as [H' | H'].
        -- destruct (Hret _ H') as (m' & [= <-] & _ & H3). lia.
        -- destruct (Hunk _ H') as (m' & [= <-] & H3 & _). lia.
  - (* made in the step *)
    intros x Hx.
    assert (Hlt : (acc_next b4 < acc_next b6)%positive)
      by (rewrite K2 in U1; lia).
    destruct Hx as [<- | [<- | [<- | Hx]]];
      [ exists (Pos.succ n1) | exists n1 | exists (acc_next b4) | ];
      try (split; [ reflexivity | lia ]).
    apply in_app_or in Hx as [Hx | Hx];
      [ destruct (Hpar _ Hx) as (m & -> & H1 & H2) | ].
    + exists m. split; [ reflexivity | lia ].
    + apply in_app_or in Hx as [Hx | Hx].
      * destruct (Hret _ Hx) as (m & -> & H1 & H2).
        exists m. split; [ reflexivity | lia ].
      * destruct (Hunk _ Hx) as (m & -> & H1 & H2).
        exists m. split; [ reflexivity | lia ].
  - (* the reads *)
    intros fct x Hf Hx Hpk.
    assert (Hpk' : ∃ m, x = Node_made m ∧ (Pos.succ (Pos.succ n1) <= m)%positive
                        ∧ (m <= acc_next b4)%positive).
    { destruct Hpk as [Hp | ->].
      - destruct (Hpar _ Hp) as (m & -> & H1 & H2). exists m.
        split; [ reflexivity | lia ].
      - exists (acc_next b4). split; [ reflexivity | lia ]. }
    destruct Hpk' as (m & -> & H1 & H2).
    destruct Hf as [<- | Hf]; [ cbn in Hx; destruct Hx | ].
    apply in_app_or in Hf as [Hf | Hf].
    + destruct (U5 _ _ Hf Hx) as [H | [H | (m' & [= <-] & H3 & _)]].
      * exact (H m eq_refl).
      * destruct H as [[= <-] | H]; [ lia | ].
        destruct (Hret _ H) as (m' & [= <-] & _ & H3). lia.
      * rewrite K2 in H3. lia.
    + destruct (K4 _ _ Hf Hx) as [H | H].
      * exact (H m eq_refl).
      * destruct H as [[= <-] | H]; [ lia | ].
        destruct (Hret _ H) as (m' & [= <-] & _ & H3). lia.
Qed.

(* ------------------------------------------------------------------------ *)
(* [prepare_code] and [traverse_code]                                       *)
(* ------------------------------------------------------------------------ *)

Lemma spec_in_cd (a : acc) (id : code_id) (cd : code_dep) (x : node) :
  pinv a -> (id, cd) ∈ acc_code_deps a -> x ∈ code_dep_nodes cd ->
  spec a x -> x ∈ cd_params cd ∨ x = kw cd.
Proof.
  intros Hinv Hcd Hx (id' & cd' & Hcd' & Hs).
  assert (Hx' : x ∈ code_dep_nodes cd').
  { destruct Hs as [Hs | ->]; [ | apply CH.in_cd_known ].
    right; right; right. apply in_or_app. left. exact Hs. }
  destruct (pi_cd_disj _ Hinv id cd id' cd' x Hcd Hcd' Hx Hx') as [_ <-].
  exact Hs.
Qed.

(* The [my_closure], the exception and the results of a code dep are none
   of its parameters, nor its witness of known arity. *)
Lemma cd_roles (cd : code_dep) (x : node) :
  NoDup (code_dep_nodes cd) ->
  x = cd_my_closure cd ∨ x = cd_exn cd ∨ x ∈ cd_return cd ->
  ~ (x ∈ cd_params cd ∨ x = kw cd).
Proof.
  unfold code_dep_nodes. intros Hnd Hx Hs.
  inversion Hnd as [ | y1 l1 Hy1 Hnd1 ]; subst.
  inversion Hnd1 as [ | y2 l2 Hy2 Hnd2 ]; subst.
  inversion Hnd2 as [ | y3 l3 Hy3 Hnd3 ]; subst.
  destruct Hx as [-> | [-> | Hx]].
  - apply Hy1. destruct Hs as [Hs | ->]; [ | right; left; reflexivity ].
    right; right. apply in_or_app. left. exact Hs.
  - apply Hy2. destruct Hs as [Hs | ->]; [ | left; reflexivity ].
    right. apply in_or_app. left. exact Hs.
  - destruct Hs as [Hs | ->].
    + exact (FlambdaReaperLeastProofs.BoundOnce.nodup_app_disjoint _ _ x
               Hnd3 Hs (in_or_app _ _ _ (or_introl Hx))).
    + apply Hy3. apply in_or_app. right. apply in_or_app. left. exact Hx.
Qed.

Lemma pstep_prepare (env : traverse_env) (id : code_id) (c : code)
    (a : acc) :
  pstep env a (prepare_code id c a).
Proof.
  intros Hinv _.
  destruct (prepare_spec id c a)
    as (cd & L & S1 & S2 & S3 & S4 & S5 & S6 & S7 & S8).
  pose proof (CH.ext_prepare_code (spec a) a id c a (Pos.le_refl _))
    as [_ (L' & HL' & Hok & _) _ _ _].
  pose proof (UT.vsafe_prepare_code (e1 e0) [] id c a) as (V1 & _ & _).
  set (a' := prepare_code id c a) in *. clearbody a'.
  assert (Hmono : ∀ p, p ∈ acc_code_deps a -> p ∈ acc_code_deps a')
    by (intros p H; rewrite S1; right; exact H).
  assert (Hnew : ∀ x, spec a' x ->
            spec a x ∨ ((x ∈ cd_params cd ∨ x = kw cd)
                        ∧ CH.fresh_in (acc_next a) (acc_next a') x)).
  { intros x (id' & cd' & H1 & H2). rewrite S1 in H1.
    destruct H1 as [[= <- <-] | H1]; [ right | left; exists id', cd'; auto ].
    split; [ exact H2 | apply S7 ].
    destruct H2 as [H2 | ->]; [ | apply CH.in_cd_known ].
    right; right; right. apply in_or_app. left. exact H2. }
  assert (Hold : ∀ x, made_lt x (acc_next a) ->
            CH.fresh_in (acc_next a) (acc_next a') x -> False).
  { intros x Hx (k & -> & H1 & _). specialize (Hx k eq_refl). lia. }
  assert (Hgood : ∀ x, CH.good (spec a) a x -> CH.good (spec a') a' x).
  { intros x [H1 H2]. split; [ intros k E; specialize (H1 k E); lia | ].
    intros Hs. destruct (Hnew x Hs) as [H | [_ H]];
      [ exact (H2 H) | exact (Hold x H1 H) ]. }
  split; [ constructor | ].
  - intros fct x Hin Hx. rewrite S2 in Hin.
    apply in_app_or in Hin as [Hin | Hin].
    + assert (Hin' : fct ∈ L').
      { rewrite S2 in HL'. apply (app_inv_tail _ _ _) in HL'. rewrite <- HL'.
        exact Hin. }
      exact (proj1 (ok_spec a a' x Hinv S5 (Hok _ _ Hin' Hx))).
    + intros k E. pose proof (pi_nodes _ Hinv fct x Hin Hx k E). lia.
  - intros id' cd' x H Hx. rewrite S1 in H. destruct H as [[= <- <-] | H].
    + destruct (S7 x Hx) as (k & -> & _ & H2). exists k. auto.
    + destruct (pi_cd _ Hinv id' cd' x H Hx) as (k & -> & Hk).
      exists k. split; [ reflexivity | lia ].
  - intros id' cd' H. rewrite S1 in H. destruct H as [[= <- <-] | H];
      [ exact S6 | exact (pi_cd_nodup _ Hinv id' cd' H) ].
  - intros i1 c1 i2 c2 x H1 H2 Hx1 Hx2. rewrite S1 in H1, H2.
    destruct H1 as [[= <- <-] | H1], H2 as [[= <- <-] | H2].
    + split; reflexivity.
    + exfalso. destruct (pi_cd _ Hinv i2 c2 x H2 Hx2) as (k & -> & Hk).
      destruct (S7 _ Hx1) as (k' & [= <-] & H3 & _). lia.
    + exfalso. destruct (pi_cd _ Hinv i1 c1 x H1 Hx1) as (k & -> & Hk).
      destruct (S7 _ Hx2) as (k' & [= <-] & H3 & _). lia.
    + exact (pi_cd_disj _ Hinv i1 c1 i2 c2 x H1 H2 Hx1 Hx2).
  - intros fct x Hin Hx Hs. rewrite S2 in Hin.
    apply in_app_or in Hin as [Hin | Hin].
    + exfalso. destruct (Hnew x Hs) as [H | [H _]].
      * assert (Hin' : fct ∈ L').
        { rewrite S2 in HL'. apply (app_inv_tail _ _ _) in HL'.
          rewrite <- HL'. exact Hin. }
        exact (proj2 (ok_spec a a' x Hinv S5
                        (Hok _ _ Hin' (read_in_nodes _ _ Hx))) H).
      * exact (S8 fct x Hin Hx H).
    + destruct (Hnew x Hs) as [H | [_ H]].
      * exact (param_alias_mono a a' fct x Hmono
                 (pi_reads _ Hinv fct x Hin Hx H)).
      * exfalso. exact (Hold x (pi_nodes _ Hinv fct x Hin
                                  (read_in_nodes _ _ Hx)) H).
  - intros d x Hd Hx. rewrite S3 in Hd.
    exact (Hgood x (pi_ad _ Hinv d x Hd Hx)).
  - intros d Hd. rewrite S4 in Hd. exact (Hgood _ (pi_cl _ Hinv d Hd)).
  - intros fct Hin. destruct (V1 fct Hin) as [H | H]; [ | left; exact H ].
    destruct (pi_vars _ Hinv fct H) as [H' | H']; [ left; exact H' | right ].
    exact (code_alias_mono a a' fct Hmono H').
  - intros d Hd. rewrite S3 in Hd. exact (pi_ad_var _ Hinv d Hd).
  - intros d Hd. rewrite S4 in Hd. exact (pi_cl_var _ Hinv d Hd).
  - split; [ exact S5 | split; [ exact Hmono | ] ].
    intros x Hs. destruct (Hnew x Hs) as [H | [_ (k & -> & H1 & _)]];
      [ left; exact H | right; exists k; auto ].
Qed.

(* The aliases of [traverse_code] of the code [c] of [e0]. *)
Definition code_facts (c : code) (cd : code_dep) : list fact :=
  Fact_alias (Node_variable (code_my_closure c)) (cd_my_closure cd)
  :: map (fun '(p, arg) => Fact_alias (Node_variable p) arg)
       (combine (code_params c) (cd_params cd)).

Lemma pstep_code (env : traverse_env) (id : code_id) (c : code)
    (cd : code_dep) (a : acc) :
  (id, c) ∈ unit_codes e0 -> (id, cd) ∈ acc_code_deps a ->
  pstep env a (add_facts (code_facts c cd) a).
Proof.
  intros Hc Hcd Hinv _.
  assert (Hcdn : ∀ x, x ∈ code_dep_nodes cd ->
             ∃ k, x = Node_made k ∧ (k < acc_next a)%positive)
    by (intros x Hx; exact (pi_cd _ Hinv id cd x Hcd Hx)).
  assert (Hfacts : ∀ fct, fct ∈ code_facts c cd ->
             (fct = Fact_alias (Node_variable (code_my_closure c))
                      (cd_my_closure cd))
             ∨ ∃ p r, (p, r) ∈ combine (code_params c) (cd_params cd)
                 ∧ fct = Fact_alias (Node_variable p) r).
  { intros fct [<- | H]; [ left; reflexivity | right ].
    apply in_map_iff in H as ([p r] & <- & H). exists p, r. auto. }
  split; [ constructor | ].
  all: cbn [add_facts acc_facts acc_next acc_code_deps acc_apply_deps
            acc_closure_deps].
  - intros fct x Hin Hx. apply in_app_or in Hin as [Hin | Hin];
      [ | exact (pi_nodes _ Hinv fct x Hin Hx) ].
    destruct (Hfacts fct Hin) as [-> | (p & r & Hpr & ->)];
      cbn in Hx; destruct Hx as [<- | [<- | []]];
      try (intros k E; discriminate E).
    + destruct (Hcdn _ (CH.in_cd_my cd)) as (k & -> & Hk).
      intros k' [= <-]. exact Hk.
    + assert (Hr : r ∈ code_dep_nodes cd)
        by (right; right; right; apply in_or_app; left;
            exact (in_combine_r _ _ _ _ Hpr)).
      destruct (Hcdn r Hr) as (k & -> & Hk).
      intros k' [= <-]. exact Hk.
  - exact (pi_cd _ Hinv).
  - exact (pi_cd_nodup _ Hinv).
  - exact (pi_cd_disj _ Hinv).
  - intros fct x Hin Hx Hs. apply in_app_or in Hin as [Hin | Hin];
      [ | exact (pi_reads _ Hinv fct x Hin Hx Hs) ].
    destruct (Hfacts fct Hin) as [-> | (p & r & Hpr & ->)];
      cbn in Hx; destruct Hx as [<- | []].
    + exfalso.
      exact (cd_roles cd _ (pi_cd_nodup _ Hinv id cd Hcd)
               (or_introl eq_refl)
               (spec_in_cd a id cd _ Hinv Hcd (CH.in_cd_my cd) Hs)).
    + exists id, c, cd, p.
      split; [ exact Hc | split; [ exact Hcd | split; [ exact Hpr | ] ] ].
      reflexivity.
  - exact (pi_ad _ Hinv).
  - exact (pi_cl _ Hinv).
  - intros fct Hin. apply in_app_or in Hin as [Hin | Hin];
      [ | exact (pi_vars _ Hinv fct Hin) ].
    right. exists id, c, cd. split; [ exact Hc | split; [ exact Hcd | ] ].
    exact (Hfacts fct Hin).
  - exact (pi_ad_var _ Hinv).
  - exact (pi_cl_var _ Hinv).
  - split; [ cbn [add_facts acc_next]; lia | ].
    split; [ intros q H; exact H | intros x H; left; exact H ].
Qed.

(* The environment of the body of code. *)
Lemma envok_code_env (env : traverse_env) (id : code_id)
    (ret exn : continuation) (cd : code_dep) (a : acc) :
  pinv a -> (id, cd) ∈ acc_code_deps a -> envok env a ->
  envok (code_env env id ret exn cd) a.
Proof.
  intros Hinv Hcd (Hg & Hv & Hs).
  assert (Hok : ∀ x, x ∈ cd_exn cd :: cd_return cd -> CH.good (spec a) a x
                     ∧ UT.var_ok (e1 e0) x).
  { intros x Hx.
    assert (Hxn : x ∈ code_dep_nodes cd).
    { destruct Hx as [<- | Hx]; [ right; left; reflexivity | ].
      right; right; right. apply in_or_app. right. apply in_or_app. left.
      exact Hx. }
    destruct (pi_cd _ Hinv id cd x Hcd Hxn) as (k & -> & Hk).
    split; [ split | apply UT.var_ok_made ].
    - intros k' [= <-]. exact Hk.
    - intros Hsp. apply (cd_roles cd (Node_made k)
                           (pi_cd_nodup _ Hinv id cd Hcd)).
      + destruct Hx as [<- | Hx]; [ right; left; reflexivity | ].
        right; right. exact Hx.
      + exact (spec_in_cd a id cd _ Hinv Hcd Hxn Hsp). }
  split; [ | split; [ | exact Hs ] ].
  - intros k ps x Hk Hx. cbn [code_env env_conts] in Hk.
    destruct Hk as [Hk | [Hk | []]]; injection Hk as <- <-;
      apply Hok; [ destruct Hx as [<- | []]; left; reflexivity
                 | right; exact Hx ].
  - intros k ps x Hk Hx. cbn [code_env env_conts] in Hk.
    destruct Hk as [Hk | [Hk | []]]; injection Hk as <- <-;
      apply Hok; [ destruct Hx as [<- | []]; left; reflexivity
                 | right; exact Hx ].
Qed.

(* ------------------------------------------------------------------------ *)
(* The traversal                                                            *)
(* ------------------------------------------------------------------------ *)

(* The code of [e0] binds no static constant in its body. *)
Hypothesis Hnest :
  ∀ id c, (id, c) ∈ unit_codes e0 -> unit_codes (code_body c) = [].

(* A part of [e0]: its [Let]s and [hparams] are those of [e0], its code
   the code of [e0]. *)
Definition psub (e : expr) : Prop :=
  (∀ x d, (x, d) ∈ let_bindings e -> x ∈ map fst (let_bindings e0))
  ∧ (∀ v, v ∈ hparams e -> v ∈ hparams e0)
  ∧ (∀ id c, (id, c) ∈ unit_codes e -> (id, c) ∈ unit_codes e0).

Definition hsub (h : continuation_handler) : Prop :=
  (∀ x d, (x, d) ∈ handler_let_bindings h -> x ∈ map fst (let_bindings e0))
  ∧ (∀ v, v ∈ handler_hparams h -> v ∈ hparams e0)
  ∧ (∀ id c, (id, c) ∈ handler_unit_codes h -> (id, c) ∈ unit_codes e0).

Lemma envok_add_cont (env : traverse_env) (k : continuation)
    (params : list variable) (a : acc) :
  pinv a -> envok env a -> (∀ p, p ∈ params -> p ∈ hparams e0) ->
  envok (add_cont env k (map Node_variable params)) a.
Proof.
  intros Hinv (Hg & Hv & Hs) Hp.
  split; [ | split; [ | exact Hs ] ].
  - intros k' ps x [Heq | Hin] Hx; [ | exact (Hg k' ps x Hin Hx) ].
    injection Heq as <- <-. apply in_map_iff in Hx as (p & <- & _).
    split; [ intros k0 E; discriminate E | ].
    intros Hsp. destruct (spec_made a _ Hinv Hsp) as (k0 & E & _).
    discriminate E.
  - intros k' ps x [Heq | Hin] Hx; [ | exact (Hv k' ps x Hin Hx) ].
    injection Heq as <- <-. apply in_map_iff in Hx as (p & <- & Hp').
    apply var_ok_e1. right. exact (Hp p Hp').
Qed.

Lemma envok_add_rec_conts (env : traverse_env)
    (handlers : list (continuation * continuation_handler)) (a : acc) :
  pinv a -> envok env a ->
  (∀ k h, (k, h) ∈ handlers -> ∀ p, p ∈ handler_hparams h ->
     p ∈ hparams e0) ->
  envok (add_rec_conts env handlers) a.
Proof.
  intros Hinv Henv. unfold add_rec_conts.
  induction handlers as [ | [k h] hs IH]; intros Hp; cbn [fold_right];
    [ exact Henv | ].
  destruct h as [params md hb]. cbn [handler_params].
  apply envok_add_cont; [ exact Hinv | | ].
  - apply IH. intros k' h' Hin. exact (Hp k' h' (or_intror Hin)).
  - intros p Hin. apply (Hp k (Handler params md hb) (or_introl eq_refl)).
    cbn. apply in_or_app. left. exact Hin.
Qed.

Lemma pstep_traverse :
  (∀ e, psub e -> ∀ env a, pstep env a (traverse env e a))
  ∧ (∀ h, hsub h -> ∀ env a, pstep env a (traverse_handler env h a)).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind;
    unfold psub, hsub;
    cbn [let_bindings hparams unit_codes handler_let_bindings
         handler_hparams handler_unit_codes].
  - intros x kind d body IH (Hl & Hh & Hu) env a. cbn [traverse].
    apply (pstep_trans _ _ (traverse_named env x d a)).
    + intros Hinv Henv.
      exact (pstep_named env x d a (Hl x d (or_introl eq_refl))
               (proj2 (proj2 Henv)) Hinv Henv).
    + apply IH. split; [ intros x' d' H; apply (Hl x' d'); right; exact H | ].
      split; [ exact Hh | exact Hu ].
  - intros bindings body IHc IH (Hl & Hh & Hu) env a. cbn [traverse].
    cbv zeta.
    match goal with
    | |- pstep _ _ (traverse _ _ ?X) =>
      apply (pstep_trans _ _ X);
      [ | apply IH; split; [ intros x' d' H; apply (Hl x' d');
                             apply in_or_app; right; exact H | ];
          split; [ intros v H; apply Hh; apply in_or_app; right; exact H
                 | intros id c H; apply Hu; apply in_or_app; right;
                   exact H ] ]
    end.
    match goal with
    | |- pstep _ _ (List.fold_left _ _ ?Y) => apply (pstep_trans _ _ Y)
    end.
    { match goal with
      | |- pstep _ _ (List.fold_left _ _ ?Y) => apply (pstep_trans _ _ Y)
      end.
      - apply pstep_fold. intros [id c | id | s const] a' _;
          cbn [prepare_binding];
          [ apply pstep_prepare | apply pstep_refl | apply pstep_refl ].
      - apply pstep_fold. intros b a' _. apply pstep_static_binding. }
    apply pstep_fold. intros [id c | id | s const] a' Hin;
      cbn [traverse_binding]; [ | apply pstep_refl | apply pstep_refl ].
    assert (Hc0 : (id, c) ∈ unit_codes e0).
    { apply Hu. apply in_or_app. left. apply in_flat_map.
      exists (Bind_code id c). split; [ exact Hin | left; reflexivity ]. }
    assert (Hbl : ∀ x d, (x, d) ∈ code_let_bindings c ->
              x ∈ map fst (let_bindings e0)).
    { intros x d H. apply (Hl x d). apply in_or_app. left. apply in_flat_map.
      exists (Bind_code id c). split; [ exact Hin | exact H ]. }
    assert (Hbh : ∀ v, v ∈ param_variables (code_body c) -> v ∈ hparams e0).
    { intros v H. apply Hh. apply in_or_app. left. apply in_flat_map.
      exists (Bind_code id c). split; [ exact Hin | exact H ]. }
    pose proof (IHc id c Hin) as IHb.
    pose proof (Hnest id c Hc0) as Hnone.
    destruct c as [params mc ret exn md cbody]. cbn [traverse_code].
    destruct (lookup id (acc_code_deps a')) as [cd | ] eqn:Ecd;
      [ | apply pstep_refl ].
    pose proof (FlambdaMapProofs.lookup_in _ _ _ Ecd) as Hcd.
    eapply pstep_trans.
    + exact (pstep_code env id (Code params mc ret exn md cbody) cd a' Hc0
               Hcd).
    + apply (pstep_env _ (code_env env id ret exn cd)).
      * intros Hinv Henv. apply (envok_code_env env id ret exn cd _ Hinv);
          [ exact Hcd | exact Henv ].
      * apply IHb. cbn [code_body] in Hnone |- *.
        split; [ exact Hbl | split ].
        -- intros v Hv. apply Hbh. exact (hparams_in_params _ _ Hv).
        -- intros id' c' H. rewrite Hnone in H. destruct H.
  - intros k h body IHh IH (Hl & Hh & Hu) env a. cbn [traverse]. cbv zeta.
    apply (pstep_trans _ _
             (exn_handler_deps env h (traverse_handler env h a))).
    + apply (pstep_trans _ _ (traverse_handler env h a)).
      * apply IHh. split; [ intros x d H; apply (Hl x d); apply in_or_app;
                            left; exact H | ].
        split; [ intros v H; apply Hh; apply in_or_app; left; exact H
               | intros id c H; apply Hu; apply in_or_app; left; exact H ].
      * apply pstep_exn_handler. destruct h as [params md hb].
        cbn [handler_params]. intros x Hx v ->.
        apply in_map_iff in Hx as (p & [= ->] & Hp).
        apply (proj2 (V0_e1 e0 v)). right. apply Hh. apply in_or_app. left.
        cbn. apply in_or_app. left. exact Hp.
    + apply (pstep_env _ (add_cont env k (handler_params h))).
      * intros Hinv Henv. destruct h as [params md hb]. cbn [handler_params].
        apply envok_add_cont; [ exact Hinv | exact Henv | ].
        intros p Hp. apply Hh. apply in_or_app. left. cbn.
        apply in_or_app. left. exact Hp.
      * apply IH. split; [ intros x d H; apply (Hl x d); apply in_or_app;
                           right; exact H | ].
        split; [ intros v H; apply Hh; apply in_or_app; right; exact H
               | intros id c H; apply Hu; apply in_or_app; right; exact H ].
  - intros handlers body IHh IH (Hl & Hh & Hu) env a. cbn [traverse].
    cbv zeta.
    apply (pstep_env _ (add_rec_conts env handlers)).
    + intros Hinv Henv. apply envok_add_rec_conts; [ exact Hinv | exact Henv
                                                   | ].
      intros k h Hkh p Hp. apply Hh. apply in_or_app. left.
      apply in_flat_map. exists (k, h). auto.
    + match goal with
      | |- pstep _ _ (traverse _ _ ?Y) => apply (pstep_trans _ _ Y)
      end.
      * apply pstep_fold. intros [k h] a' Hin. apply (IHh k h Hin).
        split; [ intros x d H; apply (Hl x d); apply in_or_app; left;
                 apply in_flat_map; exists (k, h); auto | ].
        split; [ intros v H; apply Hh; apply in_or_app; left;
                 apply in_flat_map; exists (k, h); auto
               | intros id c H; apply Hu; apply in_or_app; left;
                 apply in_flat_map; exists (k, h); auto ].
      * apply IH. split; [ intros x d H; apply (Hl x d); apply in_or_app;
                           right; exact H | ].
        split; [ intros v H; apply Hh; apply in_or_app; right; exact H
               | intros id c H; apply Hu; apply in_or_app; right; exact H ].
  - intros ap _ env a. cbn [traverse]. apply pstep_apply.
  - intros ac _ env a. cbn [traverse]. apply pstep_apply_cont.
  - intros s arms _ env a. cbn [traverse]. apply pstep_switch.
  - intros _ env a. apply pstep_refl.
  - intros params md handler IH (Hl & Hh & Hu) env a. cbn [traverse_handler].
    apply IH. split; [ exact Hl | split; [ | exact Hu ] ].
    intros v H. apply Hh. apply in_or_app. right. exact H.
Qed.

End Inv.

(* ======================================================================== *)
(* The delayed deps                                                         *)
(* ======================================================================== *)

Lemma closure_deps_reads (cds : finite_map code_dep) (ds : list closure_dep)
    (b : acc) :
  (acc_next b
   <= acc_next
        (List.fold_left (fun a d => add_closure_dep cds d a) ds b))%positive
  ∧ ∃ D,
      acc_facts (List.fold_left (fun a d => add_closure_dep cds d a) ds b)
      = D ++ acc_facts b
      ∧ ∀ fct x, fct ∈ D -> x ∈ read_nodes fct ->
          (∃ d, d ∈ ds ∧ x = cld_let_bound_name_of_the_closure d)
          ∨ (∃ id cd, lookup id cds = Some cd
               ∧ (x = kw cd ∨ x ∈ cd_unknown_arity_call_witnesses cd))
          ∨ ∃ k, x = Node_made k ∧ (acc_next b <= k)%positive.
Proof.
  revert b. induction ds as [ | d ds IH]; intros b; cbn [List.fold_left].
  - split; [ lia | exists []; split; [ reflexivity | intros fct x [] ] ].
  - assert (S1 : (acc_next b <= acc_next (add_closure_dep cds d b))%positive
                 ∧ ∃ D1, acc_facts (add_closure_dep cds d b) = D1 ++ acc_facts b
                   ∧ ∀ fct x, fct ∈ D1 -> x ∈ read_nodes fct ->
                       x = cld_let_bound_name_of_the_closure d
                       ∨ (∃ id cd, lookup id cds = Some cd
                            ∧ (x = kw cd
                               ∨ x ∈ cd_unknown_arity_call_witnesses cd))
                       ∨ ∃ k, x = Node_made k ∧ (acc_next b <= k)%positive).
    { unfold add_closure_dep.
      destruct (lookup (cld_closure_code_id d) cds) as [cd | ] eqn:E.
      - cbn [add_facts acc_next acc_facts]. split; [ lia | ].
        eexists. split; [ reflexivity | ].
        intros fct x Hf Hx.
        destruct Hf as [<- | [<- | Hf]].
        + cbn in Hx. destruct Hx as [<- | []]. left. reflexivity.
        + cbn in Hx. destruct Hx as [<- | []]. right; left.
          exists (cld_closure_code_id d), cd. auto.
        + destruct (cd_unknown_arity_call_witnesses cd) as [ | w ws] eqn:Ew;
            [ destruct Hf | ].
          destruct Hf as [<- | []]. cbn in Hx. destruct Hx as [<- | []].
          right; left. exists (cld_closure_code_id d), cd.
          split; [ exact E | right; rewrite Ew; left; reflexivity ].
      - cbn [fresh]. cbv beta iota.
        cbn [add_facts acc_next acc_facts]. split; [ lia | ].
        eexists. split; [ reflexivity | ].
        intros fct x Hf Hx.
        destruct Hf as [<- | [<- | [<- | [<- | []]]]]; cbn in Hx;
          repeat destruct Hx as [<- | Hx]; try contradiction;
          first [ left; reflexivity
                | right; right; exists (acc_next b); split;
                  [ reflexivity | lia ] ]. }
    destruct S1 as (S1 & D1 & S2 & S3).
    destruct (IH (add_closure_dep cds d b)) as (T1 & D2 & T2 & T3).
    split; [ lia | ].
    exists (D2 ++ D1). split; [ rewrite T2, S2, app_assoc; reflexivity | ].
    intros fct x Hf Hx. apply in_app_or in Hf as [Hf | Hf].
    + destruct (T3 fct x Hf Hx) as [(d' & Hd' & H) | [H | (k & -> & Hk)]].
      * left. exists d'. split; [ right; exact Hd' | exact H ].
      * right; left. exact H.
      * right; right. exists k. split; [ reflexivity | lia ].
    + destruct (S3 fct x Hf Hx) as [H | [H | H]].
      * left. exists d. split; [ left; reflexivity | exact H ].
      * right; left. exact H.
      * right; right. exact H.
Qed.

Lemma resolve_reads (us : list symbol) (a : acc) :
  ∀ fct x, fct ∈ acc_facts (resolve_delayed_deps us a) ->
    x ∈ read_nodes fct ->
    fct ∈ acc_facts a
    ∨ (∃ d, d ∈ acc_apply_deps a ∧ x ∈ CH.apply_dep_nodes d)
    ∨ (∃ d, d ∈ acc_closure_deps a
         ∧ x = cld_let_bound_name_of_the_closure d)
    ∨ (∃ id cd, lookup id (acc_code_deps a) = Some cd
         ∧ (x = kw cd ∨ x ∈ cd_unknown_arity_call_witnesses cd))
    ∨ ∃ k, x = Node_made k ∧ (acc_next a <= k)%positive.
Proof.
  intros fct x. unfold resolve_delayed_deps. cbv zeta.
  set (L1 := flat_map _ _).
  destruct (closure_deps_reads (acc_code_deps a)
              (acc_closure_deps (add_facts L1 a)) (add_facts L1 a))
    as (C1 & D & C2 & C3).
  set (b2 := List.fold_left _ _ (add_facts L1 a)) in *.
  cbn [add_facts acc_facts acc_next acc_closure_deps] in *.
  intros Hf Hx. apply in_app_or in Hf as [Hf | Hf].
  - apply in_map_iff in Hf as (s & <- & _). cbn in Hx. destruct Hx.
  - rewrite C2 in Hf. apply in_app_or in Hf as [Hf | Hf].
    + destruct (C3 fct x Hf Hx) as [H | [H | H]];
        [ right; right; left; exact H | right; right; right; left; exact H
        | right; right; right; right; exact H ].
    + apply in_app_or in Hf as [Hf | Hf]; [ | left; exact Hf ].
      unfold L1 in Hf. apply in_flat_map in Hf as (d & Hd & Hf).
      unfold apply_dep_facts in Hf.
      destruct (lookup (ad_apply_code_id d) (acc_code_deps a))
        as [cd | ] eqn:E.
      * destruct Hf as [<- | [<- | []]]; unfold alias_for_caller in Hx;
          destruct (ad_function_containing_apply_expr d); cbn in Hx;
          destruct Hx as [<- | []].
        -- right; right; right; left. exists (ad_apply_code_id d), cd.
           split; [ exact E | left; reflexivity ].
        -- right; right; right; left. exists (ad_apply_code_id d), cd.
           split; [ exact E | left; reflexivity ].
        -- right; left. exists d. split; [ exact Hd | left; reflexivity ].
        -- right; left. exists d. split; [ exact Hd | left; reflexivity ].
      * right; left. exists d. split; [ exact Hd | ].
        destruct (ad_function_containing_apply_expr d);
          repeat destruct Hf as [Hf | Hf]; try contradiction; subst fct;
          cbn in Hx; try contradiction; destruct Hx as [Hx | []]; subst x;
          left; reflexivity.
Qed.

(* ======================================================================== *)
(* The unit                                                                 *)
(* ======================================================================== *)

Lemma nodup_flat_elem {A B : Type} (f : A -> list B) (l : list A) (a : A) :
  NoDup (flat_map f l) -> a ∈ l -> NoDup (f a).
Proof.
  induction l as [ | y l IH]; intros Hnd Ha; [ destruct Ha | ].
  cbn [flat_map] in Hnd. destruct Ha as [<- | Ha].
  - exact (NoDup_app_remove_r _ _ Hnd).
  - exact (IH (NoDup_app_remove_l _ _ Hnd) Ha).
Qed.

Lemma combine_fst_inj {A B : Type} (l1 : list A) (l2 : list B) (x : A)
    (y y' : B) :
  NoDup l1 -> (x, y) ∈ combine l1 l2 -> (x, y') ∈ combine l1 l2 -> y = y'.
Proof.
  revert l2. induction l1 as [ | a l1 IH]; intros [ | b l2] Hnd H H';
    cbn in H, H'; try contradiction.
  inversion Hnd as [ | a' l1' Ha Hnd' ]; subst.
  destruct H as [E | H], H' as [E' | H'].
  - inversion E; inversion E'; subst. reflexivity.
  - exfalso. inversion E; subst. exact (Ha (in_combine_l _ _ _ _ H')).
  - exfalso. inversion E'; subst. exact (Ha (in_combine_l _ _ _ _ H)).
  - exact (IH l2 Hnd' H H').
Qed.

Lemma combine_snd_inj {A B : Type} (l1 : list A) (l2 : list B) (x x' : A)
    (y : B) :
  NoDup l2 -> (x, y) ∈ combine l1 l2 -> (x', y) ∈ combine l1 l2 -> x = x'.
Proof.
  revert l2. induction l1 as [ | a l1 IH]; intros [ | b l2] Hnd H H';
    cbn in H, H'; try contradiction.
  inversion Hnd as [ | b' l2' Hb Hnd' ]; subst.
  destruct H as [E | H], H' as [E' | H'].
  - inversion E; inversion E'; subst. reflexivity.
  - exfalso. inversion E; subst. exact (Hb (in_combine_r _ _ _ _ H')).
  - exfalso. inversion E'; subst. exact (Hb (in_combine_r _ _ _ _ H)).
  - exact (IH l2 Hnd' H H').
Qed.

Lemma nodup_keys {A : Type} (l : finite_map A) (k : positive) (v v' : A) :
  NoDup (map fst l) -> (k, v) ∈ l -> (k, v') ∈ l -> v = v'.
Proof.
  intros Hnd H H'.
  pose proof (FlambdaReaperFlowProofs.lookup_nodup_in _ _ _ Hnd H) as E.
  rewrite (FlambdaReaperFlowProofs.lookup_nodup_in _ _ _ Hnd H') in E.
  injection E as ->. reflexivity.
Qed.

(* The code of a unit of the fragment binds no static constant in its
   body. *)
Lemma codes_no_statics (hb : bool) (codes : list (code_id * code))
    (symbols : list symbol) :
  (∀ e, flow_fragment hb codes symbols e = true ->
     ∀ id c, (id, c) ∈ unit_codes e -> unit_codes (code_body c) = [])
  ∧ (∀ h, handler_flow_fragment hb codes symbols h = true ->
       ∀ id c, (id, c) ∈ handler_unit_codes h ->
         unit_codes (code_body c) = []).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind;
    cbn [flow_fragment handler_flow_fragment unit_codes handler_unit_codes];
    try (intros; contradiction).
  - intros x kind d body IH H. apply andb_true_iff in H as [_ H].
    exact (IH H).
  - intros bindings body _ IH H id c Hin.
    apply andb_true_iff in H as [Hb H].
    apply in_app_or in Hin as [Hin | Hin]; [ | exact (IH H id c Hin) ].
    apply in_flat_map in Hin as (b & Hbin & Hin).
    destruct b as [id' c' | | ]; cbn in Hin; try contradiction.
    destruct Hin as [[= <- <-] | []].
    rewrite forallb_forall in Hb. specialize (Hb _ Hbin).
    cbn [binding_flow_fragment] in Hb.
    destruct c' as [params mc ret exn md body']. cbn in Hb |- *.
    repeat (apply andb_true_iff in Hb as [Hb ?]).
    apply (proj1 FlambdaReaperFlowProofs.unit_codes_of_no_statics).
    apply negb_true_iff. assumption.
  - intros k h body IHh IH H id c Hin. apply andb_true_iff in H as [Hh H].
    apply in_app_or in Hin as [Hin | Hin];
      [ exact (IHh Hh id c Hin) | exact (IH H id c Hin) ].
  - intros handlers body IHh IH H id c Hin.
    apply andb_true_iff in H as [Hh H].
    apply in_app_or in Hin as [Hin | Hin]; [ | exact (IH H id c Hin) ].
    apply in_flat_map in Hin as ([k h] & Hkh & Hin).
    rewrite forallb_forall in Hh. specialize (Hh _ Hkh).
    exact (IHh k h Hkh Hh id c Hin).
  - intros params md handler IH H. exact (IH H).
Qed.

Module FP := FlambdaReaperFlowProofs.

Lemma params_nodup (cd : code_dep) :
  NoDup (code_dep_nodes cd) -> NoDup (cd_params cd).
Proof.
  unfold code_dep_nodes. intros H.
  apply NoDup_cons_iff in H as [_ H]. apply NoDup_cons_iff in H as [_ H].
  apply NoDup_cons_iff in H as [_ H]. exact (NoDup_app_remove_r _ _ H).
Qed.

(* The code deps of the unit have distinct code ids
   (FlambdaReaperWrapperProofs.unit_facts, with the heap). *)
Lemma code_deps_nodup (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) :
  in_heap_fragment e = true ->
  NoDup (map fst
           (acc_code_deps
              (resolve_delayed_deps (static_names symbols_of_binding e)
                 (add_fact (Fact_any_usage (Node_made 1))
                    (traverse (FP.initial_env flag u e) e FP.initial_acc))))).
Proof.
  intros Hfrag. unfold in_heap_fragment in Hfrag.
  apply andb_true_iff in Hfrag as [Hfrag Hdist].
  destruct (proj1 (FP.traverse_makes (unit_codes e)
                     (static_names symbols_of_binding e)
                     (projected_value_slots e) True true) e Hfrag Hdist
              (fun w Hw => proj2 (FlambdaMapProofs.among_In w _) Hw)
              (or_introl I) (fun x Hx => Hx) (FP.initial_env flag u e) eq_refl
              FP.initial_acc)
    as [[_ (l & Hl & Hperm)] _].
  destruct (FP.resolve_delayed_deps_spec (static_names symbols_of_binding e)
              (add_fact (Fact_any_usage (Node_made 1))
                 (traverse (FP.initial_env flag u e) e FP.initial_acc)))
    as ([_ S6] & _).
  rewrite S6. cbn [add_fact add_facts acc_code_deps]. rewrite Hl.
  cbn [FP.initial_acc acc_code_deps]. rewrite app_nil_r.
  apply (Permutation_NoDup (Permutation_sym Hperm)).
  apply FlambdaMapProofs.distinct_NoDup. exact Hdist.
Qed.

Section Unit.

Variable flag : reaper_preserve_direct_calls.
Variable u : unit_interface.
Variable e : expr.
Hypothesis Hfrag : in_heap_fragment e = true.
Hypothesis Honce : variables_bound_once e = true.

Abbreviation env0 := (FP.initial_env flag u e).
Abbreviation a5 := (traverse env0 e FP.initial_acc).
Abbreviation a6 := (add_fact (Fact_any_usage (Node_made 1)) a5).
Abbreviation A := (resolve_delayed_deps (static_names symbols_of_binding e)
                     a6).

Lemma Hnest_unit :
  ∀ id c, (id, c) ∈ unit_codes e -> unit_codes (code_body c) = [].
Proof.
  apply (proj1 (codes_no_statics true (unit_codes e)
                  (static_names symbols_of_binding e))).
  unfold in_heap_fragment in Hfrag. apply andb_true_iff in Hfrag as [H _].
  exact H.
Qed.

Lemma pinv_initial : pinv e FP.initial_acc.
Proof.
  constructor; cbn [FP.initial_acc acc_facts acc_code_deps acc_apply_deps
                    acc_closure_deps acc_next].
  - intros fct x [<- | [<- | []]] Hx; cbn in Hx;
      destruct Hx as [<- | []]; intros k E; [ injection E as <-; lia
                                           | discriminate E ].
  - intros id cd x [].
  - intros id cd [].
  - intros id cd id' cd' x [].
  - intros fct x _ _ (id & cd & [] & _).
  - intros d x [].
  - intros d [].
  - intros fct [<- | [<- | []]]; left; intros v Hv; cbn in Hv;
      try contradiction; destruct Hv as [Hv | []]; discriminate Hv.
  - intros d [].
  - intros d [].
Qed.

Lemma envok_initial : envok e env0 FP.initial_acc.
Proof.
  split; [ | split; [ | reflexivity ] ].
  - intros k ps x Hk Hx. cbn [FP.initial_env env_conts unit_conts] in Hk.
    destruct Hk as [Hk | [Hk | []]]; injection Hk as <- <-;
      destruct Hx as [<- | []];
      (split; [ intros k' [= <-]; cbn; lia | intros (id & cd & [] & _) ]).
  - intros k ps x Hk Hx. cbn [FP.initial_env env_conts unit_conts] in Hk.
    destruct Hk as [Hk | [Hk | []]]; injection Hk as <- <-;
      destruct Hx as [<- | []]; apply UT.var_ok_made.
Qed.

Lemma pinv_a6 : pinv e a6 ∧ (3 <= acc_next a5)%positive.
Proof.
  assert (Hsub : psub e e).
  { split; [ intros x d H; exact (in_map fst _ (x, d) H) | split ].
    - intros v H. exact H.
    - intros id c H. exact H. }
  destruct (proj1 (pstep_traverse e Hnest_unit) e Hsub env0 FP.initial_acc
              pinv_initial envok_initial)
    as [Hinv5 (Hn5 & _ & Hs5)].
  cbn [FP.initial_acc acc_next] in Hn5.
  split; [ | exact Hn5 ].
  apply (pclose e a5 a6 Hinv5); [ | reflexivity | | | ].
  - apply next_of. apply CH.ext_add_fact; [ | intros ? ? ?; discriminate ].
    intros y [<- | []]. apply CH.ok_good. split.
    + intros k [= <-]. lia.
    + intros Hs.
      destruct (Hs5 _ Hs) as [(id & cd & [] & _) | (k & [= <-] & Hk)].
      cbn [FP.initial_acc acc_next] in Hk. lia.
  - intros fct [<- | H]; [ right; intros v [] | left; exact H ].
  - intros d Hd. left. exact Hd.
  - intros d Hd. left. exact Hd.
Qed.

Section Param.

Variable id : code_id.
Variable c : code.
Variable cd : code_dep.
Variable p : variable.
Variable r : node.
Hypothesis Hc : (id, c) ∈ unit_codes e.
Hypothesis Hcd : lookup id (acc_code_deps A) = Some cd.
Hypothesis Hpr : (p, r) ∈ combine (code_params c) (cd_params cd).

Lemma code_deps_A : acc_code_deps A = acc_code_deps a6.
Proof.
  destruct (UT.resolve_facts (static_names symbols_of_binding e) a6)
    as (R1 & _). exact R1.
Qed.

Lemma Hsplit :
  NoDup (map fst (let_bindings e) ++ map UT.mc_of (unit_codes e)
         ++ cparams_of (unit_codes e) ++ hparams e).
Proof. exact (bound_split e Honce). Qed.

Lemma p_cparams : p ∈ cparams_of (unit_codes e).
Proof. exact (in_cparams _ id c p Hc (in_combine_l _ _ _ _ Hpr)). Qed.

Lemma p_not_e1 : ~ (p ∈ map fst (let_bindings e) ∨ p ∈ hparams e).
Proof.
  pose proof Hsplit as Hnd. pose proof p_cparams as Hp.
  intros [H | H].
  - apply (FlambdaReaperLeastProofs.BoundOnce.nodup_app_disjoint _ _ p Hnd H).
    apply in_or_app. right. apply in_or_app. left. exact Hp.
  - apply NoDup_app_remove_l, NoDup_app_remove_l in Hnd.
    exact (FlambdaReaperLeastProofs.BoundOnce.nodup_app_disjoint _ _ p Hnd Hp
             H).
Qed.

Lemma p_not_var_ok : ~ UT.var_ok (e1 e) (Node_variable p).
Proof.
  intros H. apply p_not_e1. apply (proj1 (V0_e1 e p)). exact (H p eq_refl).
Qed.

Lemma p_not_mc (c' : code) (id' : code_id) :
  (id', c') ∈ unit_codes e -> code_my_closure c' <> p.
Proof.
  intros H E. pose proof Hsplit as Hnd.
  apply NoDup_app_remove_l in Hnd.
  apply (FlambdaReaperLeastProofs.BoundOnce.nodup_app_disjoint _ _ p Hnd).
  - rewrite <- E. exact (in_map UT.mc_of _ (id', c') H).
  - apply in_or_app. left. exact p_cparams.
Qed.

Lemma cparams_nodup : NoDup (cparams_of (unit_codes e)).
Proof.
  pose proof Hsplit as Hnd.
  apply NoDup_app_remove_l, NoDup_app_remove_l in Hnd.
  exact (NoDup_app_remove_r _ _ Hnd).
Qed.

Lemma codes_nodup : NoDup (map fst (unit_codes e)).
Proof.
  exact (FlambdaReaperRebuildProofs.unit_codes_nodup true e Hfrag).
Qed.

Lemma cd_in : (id, cd) ∈ acc_code_deps a6.
Proof. rewrite <- code_deps_A. exact (FlambdaMapProofs.lookup_in _ _ _ Hcd).
Qed.

(* What puts something into the parameter is the alias from its node. *)
Lemma into_param (fct : fact) :
  fct ∈ acc_facts A -> Node_variable p ∈ into_nodes fct ->
  fct = Fact_alias (Node_variable p) r.
Proof.
  intros Hf Hx. destruct pinv_a6 as [Hinv _].
  destruct (UT.resolve_facts (static_names symbols_of_binding e) a6)
    as (R1 & R2 & R3 & R4 & R5).
  destruct (R5 fct Hf) as [H | [H | [(d & Hd & Hcl) | (s & ->)]]].
  - destruct (pi_vars e _ Hinv fct H) as [Hv | (id' & c' & cd' & Hc' & Hcd'
                                                  & Hal)].
    + exfalso. apply p_not_e1. exact (var_fact_e1 e [] fct p Hv Hx).
    + destruct Hal as [-> | (p' & r' & Hpr' & ->)].
      * exfalso. cbn in Hx. destruct Hx as [Hx | []]. injection Hx as Hx.
        exact (p_not_mc c' id' Hc' Hx).
      * cbn in Hx. destruct Hx as [Hx | []]. injection Hx as ->.
        assert (Hp' : p ∈ code_params c')
          by exact (in_combine_l _ _ _ _ Hpr').
        pose proof (nodup_flat_inj (fun q => code_params (snd q))
                      (unit_codes e) (id', c') (id, c) p cparams_nodup Hc' Hc
                      Hp' (in_combine_l _ _ _ _ Hpr)) as E.
        injection E as -> ->.
        assert (Hnd : NoDup (map fst (acc_code_deps a6)))
          by (rewrite <- code_deps_A; exact (code_deps_nodup flag u e Hfrag)).
        rewrite (nodup_keys _ id cd' cd Hnd Hcd' cd_in) in Hpr'.
        rewrite (combine_fst_inj _ _ p r' r
                   (nodup_flat_elem _ _ (id, c) cparams_nodup Hc) Hpr' Hpr).
        reflexivity.
  - exfalso. apply in_flat_map in H as (d & Hd & H).
    destruct (UT.apply_dep_fact_cases _ _ _ H) as [(cd' & E & ->) | Hx'].
    + apply UT.into_alias_for_caller in Hx.
      destruct (pi_cd e _ Hinv _ cd' _ (FlambdaMapProofs.lookup_in _ _ _ E)
                  (CH.in_cd_my cd'))
        as (k & Ek & _).
      rewrite Ek in Hx. discriminate Hx.
    + destruct (Hx' _ Hx) as [Hw | (id' & Hw)]; [ | discriminate Hw ].
      apply p_not_var_ok. rewrite Hw. exact (pi_ad_var e _ Hinv d Hd).
  - exfalso. destruct Hcl as [(cd' & E & ->) | Hx'].
    + cbn in Hx. destruct Hx as [Hx | []].
      destruct (pi_cd e _ Hinv _ cd' _ (FlambdaMapProofs.lookup_in _ _ _ E)
                  (CH.in_cd_my cd'))
        as (k & Ek & _).
      rewrite Ek in Hx. discriminate Hx.
    + destruct (Hx' _ Hx) as [Hb | (k & Hk & _)]; [ | discriminate Hk ].
      apply p_not_var_ok. rewrite Hb. exact (pi_cl_var e _ Hinv d Hd).
  - exfalso. cbn in Hx. destruct Hx as [Hx | []]. discriminate Hx.
Qed.

(* What reads the node of the parameter is the alias into it. *)
Lemma reads_param (fct : fact) :
  fct ∈ acc_facts A -> r ∈ read_nodes fct ->
  fct = Fact_alias (Node_variable p) r.
Proof.
  intros Hf Hx. destruct pinv_a6 as [Hinv _].
  assert (Hrn : r ∈ code_dep_nodes cd)
    by (right; right; right; apply in_or_app; left;
        exact (in_combine_r _ _ _ _ Hpr)).
  assert (Hs : spec a6 r)
    by (exists id, cd; split; [ exact cd_in | left;
                                  exact (in_combine_r _ _ _ _ Hpr) ]).
  assert (Hnot : ∀ cd' id', (id', cd') ∈ acc_code_deps a6 ->
            r = kw cd' ∨ r ∈ cd_unknown_arity_call_witnesses cd' -> False).
  { intros cd' id' Hcd' Hr.
    assert (Hrn' : r ∈ code_dep_nodes cd').
    { destruct Hr as [-> | Hr]; [ apply CH.in_cd_known | ].
      apply CH.in_cd_unknown. exact Hr. }
    destruct (pi_cd_disj e _ Hinv id cd id' cd' r cd_in Hcd' Hrn Hrn')
      as [<- <-].
    pose proof (pi_cd_nodup e _ Hinv id cd cd_in) as Hnd.
    unfold code_dep_nodes in Hnd.
    inversion Hnd as [ | y1 l1 Hy1 Hnd1 ]; subst.
    inversion Hnd1 as [ | y2 l2 Hy2 Hnd2 ]; subst.
    inversion Hnd2 as [ | y3 l3 Hy3 Hnd3 ]; subst.
    assert (Hp : r ∈ cd_params cd) by exact (in_combine_r _ _ _ _ Hpr).
    destruct Hr as [-> | Hr].
    - apply Hy3. apply in_or_app. left. exact Hp.
    - apply (FlambdaReaperLeastProofs.BoundOnce.nodup_app_disjoint _ _ r Hnd3
               Hp).
      apply in_or_app. right. exact Hr. }
  destruct (resolve_reads _ a6 fct r Hf Hx)
    as [H | [(d & Hd & Hr) | [(d & Hd & Hr) | [(id' & cd' & E & Hr)
                                                | (k & Ek & Hk)]]]].
  - destruct (pi_reads e _ Hinv fct r H Hx Hs)
      as (id' & c' & cd' & p' & Hc' & Hcd' & Hpr' & ->).
    assert (Hrn' : r ∈ code_dep_nodes cd')
      by (right; right; right; apply in_or_app; left;
          exact (in_combine_r _ _ _ _ Hpr')).
    destruct (pi_cd_disj e _ Hinv id cd id' cd' r cd_in Hcd' Hrn Hrn')
      as [<- <-].
    pose proof (nodup_keys _ id c' c codes_nodup Hc' Hc) as ->.
    rewrite (combine_snd_inj _ _ p' p r
               (params_nodup cd (pi_cd_nodup e _ Hinv id cd cd_in)) Hpr' Hpr).
    reflexivity.
  - exfalso. destruct (pi_ad e _ Hinv d r Hd Hr) as [_ H]. exact (H Hs).
  - exfalso. rewrite Hr in Hs. destruct (pi_cl e _ Hinv d Hd) as [_ H].
    exact (H Hs).
  - exfalso. exact (Hnot cd' id' (FlambdaMapProofs.lookup_in _ _ _ E) Hr).
  - exfalso. destruct (pi_cd e _ Hinv id cd r cd_in Hrn) as (k' & Ek' & Hk').
    rewrite Ek in Ek'. injection Ek' as <-. lia.
Qed.

End Param.

End Unit.

Theorem params_aliased_traversal_proof (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) :
  in_heap_fragment e = true -> variables_bound_once e = true ->
  params_aliased (traverse_unit flag u e) (unit_code_deps flag u e) e = true.
Proof.
  intros Hfrag Honce. rewrite FP.traverse_unit_eq.
  change (unit_code_deps flag u e)
    with (acc_code_deps
            (resolve_delayed_deps (static_names symbols_of_binding e)
               (add_fact (Fact_any_usage (Node_made 1))
                  (traverse (FP.initial_env flag u e) e FP.initial_acc)))).
  unfold params_aliased. apply forallb_forall. intros [id c] Hc.
  match goal with
  | |- context [lookup id (acc_code_deps ?X)] =>
    destruct (lookup id (acc_code_deps X)) as [cd | ] eqn:Hcd;
      [ | reflexivity ]
  end.
  apply forallb_forall. intros [p r] Hpr. apply andb_true_intro. split.
  - apply CH.only_alias_into_of. intros fct Hf Hx. left.
    exact (into_param flag u e Hfrag Honce id c cd p r Hc Hcd Hpr fct Hf Hx).
  - apply only_read_by_of. intros fct Hf Hx. left.
    exact (reads_param flag u e Hfrag id c cd p r Hc Hcd Hpr fct Hf Hx).
Qed.

End ParamsTraversal.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Module PT := ParamsTraversal.
Module TP := FlambdaReaperTypesProofs.
Module PP := FlambdaReaperTypesParamsProofs.

Theorem params_aliased_traversal : params_aliased_traversal_statement.
Proof. exact PT.params_aliased_traversal_proof. Qed.

(* The premises of the statements give the check. *)
Lemma used_premises_aliased (local : bool)
    (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (sets : list set_of_closures) :
  used_premises local flag is_local u e T sets ->
  params_aliased (traverse_unit flag u e) (unit_code_deps flag u e) e
  = true.
Proof.
  intros (_ & _ & Hfrag & _ & _ & _ & _ & Honce & _).
  exact (params_aliased_traversal flag u e
           (FlambdaReaperFlowProofs.in_flow_fragment_in_heap_fragment e
              Hfrag)
           Honce).
Qed.

Lemma plain_param_traversal_gen (local : bool) (fx : fixes) :
  TP.fixes_sound fx -> param_type_end_to_end_traversal_with local fx.
Proof.
  intros Hfx flag is_local u e T b sets tenv canon valuation valuation' n s
    id i p r type_var v Hprem Hres.
  exact (PP.plain_param_gen local fx Hfx flag is_local u e T b sets tenv
           canon valuation valuation' n s id i p r type_var v Hprem
           (used_premises_aliased local flag is_local u e T sets Hprem) Hres).
Qed.

Lemma cc_param_traversal_gen (local : bool) (fx : fixes) :
  TP.fixes_sound fx -> cc_param_type_end_to_end_traversal_with local fx.
Proof.
  intros Hfx flag is_local u e T cannot_change b sets tenv canon valuation
    valuation' n s id i p r type_var v Hprem Hccc Hwid Hres.
  exact (PP.cc_param_type_gen local fx Hfx flag is_local u e T cannot_change
           b sets tenv canon valuation valuation' n s id i p r type_var v
           Hprem Hccc Hwid
           (used_premises_aliased local flag is_local u e T sets Hprem) Hres).
Qed.

Theorem param_type_end_to_end_traversal :
  param_type_end_to_end_traversal_statement.
Proof. exact (plain_param_traversal_gen false all_fixes TP.all_fixes_sound).
Qed.

Theorem param_type_end_to_end_traversal_landed :
  param_type_end_to_end_traversal_landed_statement.
Proof.
  exact (plain_param_traversal_gen false landed_fixes TP.landed_fixes_sound).
Qed.

Theorem param_type_end_to_end_traversal_local :
  param_type_end_to_end_traversal_local_statement.
Proof. exact (plain_param_traversal_gen true all_fixes TP.all_fixes_sound).
Qed.

Theorem param_type_end_to_end_traversal_local_landed :
  param_type_end_to_end_traversal_local_landed_statement.
Proof.
  exact (plain_param_traversal_gen true landed_fixes TP.landed_fixes_sound).
Qed.

Theorem cc_param_type_end_to_end_traversal :
  cc_param_type_end_to_end_traversal_statement.
Proof. exact (cc_param_traversal_gen false all_fixes TP.all_fixes_sound).
Qed.

Theorem cc_param_type_end_to_end_traversal_landed :
  cc_param_type_end_to_end_traversal_landed_statement.
Proof.
  exact (cc_param_traversal_gen false landed_fixes TP.landed_fixes_sound).
Qed.

Theorem cc_param_type_end_to_end_traversal_local :
  cc_param_type_end_to_end_traversal_local_statement.
Proof. exact (cc_param_traversal_gen true all_fixes TP.all_fixes_sound).
Qed.

Theorem cc_param_type_end_to_end_traversal_local_landed :
  cc_param_type_end_to_end_traversal_local_landed_statement.
Proof.
  exact (cc_param_traversal_gen true landed_fixes TP.landed_fixes_sound).
Qed.
