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

(* A term that keeps the scope of another: the lemmas that the proofs
   that a rewrite keeps scope are made of.

   Nothing here needs review, and no statement is this file's own. A
   rewrite keeps scope (FlambdaValidatorEquivarianceSpec.keeps_scope) if,
   on a term whose code is closed, the code of its result is closed and
   every variable or continuation that is free in the result is free in
   the term. It says nothing of symbols, code ids or slots. The
   statements that say so of a rewrite are in Spec files of their own;
   their proofs go by the relation [scoped] of this file, which holds
   of a term and one that keeps its scope, and by a lemma for each form
   of term.

   What is here: lists of names; [scoped] and what it is for handlers,
   code, groups of static bindings and groups of handlers, with a lemma
   for each form of term and for a binding or a continuation that is
   removed; the same relative to the variables that what is known may
   hold ([scoped_under]), for the rewrites that go by what is known; and
   the comparison up to bound names, which accepts a term only against
   one that keeps its scope ([expr_eqb_scoped]). *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaBindersSpec
  FlambdaFreeNamesSpec FlambdaFreeNamesProofs FlambdaApplySpec
  FlambdaTupledSpec FlambdaCompareSpec FlambdaNormaliseSpec
  FlambdaNormaliseProofs
  FlambdaUnusedContsSpec FlambdaUnusedContsProofs
  FlambdaValidatorEquivarianceSpec.
Import ListNotations.

(* ======================================================================== *)
(* Lists of names                                                           *)
(* ======================================================================== *)

Lemma in_without (a : positive) (bound names : list positive) :
  In a (without bound names) <-> In a names ∧ ~ In a bound.
Proof.
  unfold without. rewrite filter_In, negb_true_iff.
  split; intros [Hin Hbound]; (split; [ exact Hin | ]).
  - intros Ha. apply among_In in Ha. congruence.
  - destruct (among a bound) eqn:Hamong; [ | reflexivity ].
    apply among_In in Hamong. contradiction.
Qed.

Lemma incl_without (bound names names' : list positive) :
  incl names names' -> incl (without bound names) (without bound names').
Proof.
  intros Hincl a Ha. apply in_without in Ha. apply in_without.
  split; [ apply Hincl | ]; tauto.
Qed.

Lemma incl_is_empty {A : Type} (l l' : list A) :
  incl l l' -> Keyed.is_empty l' = true -> Keyed.is_empty l = true.
Proof.
  destruct l' as [ | b l']; [ | discriminate ].
  destruct l as [ | a l]; [ reflexivity | ].
  intros Hincl _. destruct (Hincl a (or_introl eq_refl)).
Qed.

Lemma is_empty_nil {A : Type} (l : list A) :
  Keyed.is_empty l = true -> l = [].
Proof. destruct l; [ reflexivity | discriminate ]. Qed.

Lemma incl_flat_map_filter {A : Type} (f : A -> list positive)
    (keep : A -> bool) (l : list A) :
  incl (flat_map f (filter keep l)) (flat_map f l).
Proof.
  intros a Ha. apply in_flat_map in Ha. destruct Ha as (b & Hb & Ha).
  apply filter_In in Hb. apply in_flat_map. exists b. tauto.
Qed.

(* ======================================================================== *)
(* A term that keeps the scope of another                                   *)
(* ======================================================================== *)

Ltac scope :=
  cbn [code_closed handler_code_closed binding_code_closed code_is_closed
       free_variables_of_expr free_variables_of_handler
       free_variables_of_binding free_variables_of_code
       free_continuations_of_expr free_continuations_of_handler
       free_continuations_of_binding free_continuations_of_code].

Ltac scope_in H :=
  cbn [code_closed handler_code_closed binding_code_closed code_is_closed
       free_variables_of_expr free_variables_of_handler
       free_variables_of_binding free_variables_of_code
       free_continuations_of_expr free_continuations_of_handler
       free_continuations_of_binding free_continuations_of_code] in H.

(* If the code of the first is closed, the code of the second is, and
   the free variables and continuations of the second are free in the
   first. *)
Definition scoped (e e' : expr) : Prop :=
  code_closed e = true ->
  code_closed e' = true
  ∧ incl (free_variables_of_expr e') (free_variables_of_expr e)
  ∧ incl (free_continuations_of_expr e') (free_continuations_of_expr e).

Definition scoped_handler (h h' : continuation_handler) : Prop :=
  handler_code_closed h = true ->
  handler_code_closed h' = true
  ∧ incl (free_variables_of_handler h') (free_variables_of_handler h)
  ∧ incl (free_continuations_of_handler h')
      (free_continuations_of_handler h).

Definition scoped_code (c c' : code) : Prop :=
  code_is_closed c = true -> code_is_closed c' = true.

Definition scoped_bindings (bindings bindings' : list static_binding) : Prop :=
  forallb binding_code_closed bindings = true ->
  forallb binding_code_closed bindings' = true
  ∧ incl (flat_map free_variables_of_binding bindings')
      (flat_map free_variables_of_binding bindings)
  ∧ incl (flat_map free_continuations_of_binding bindings')
      (flat_map free_continuations_of_binding bindings).

Definition scoped_handlers
    (handlers handlers' : list (continuation * continuation_handler))
    : Prop :=
  forallb (fun '(_, h) => handler_code_closed h) handlers = true ->
  forallb (fun '(_, h) => handler_code_closed h) handlers' = true
  ∧ incl (flat_map (fun '(_, h) => free_variables_of_handler h) handlers')
      (flat_map (fun '(_, h) => free_variables_of_handler h) handlers)
  ∧ incl
      (flat_map (fun '(_, h) => free_continuations_of_handler h) handlers')
      (flat_map (fun '(_, h) => free_continuations_of_handler h) handlers).

Lemma scoped_refl (e : expr) : scoped e e.
Proof. intros Hclosed. split; [ exact Hclosed | split; apply incl_refl ]. Qed.

Lemma scoped_trans (e e' e'' : expr) :
  scoped e e' -> scoped e' e'' -> scoped e e''.
Proof.
  intros Hfirst Hsecond Hclosed.
  destruct (Hfirst Hclosed) as (Hclosed' & Hv & Hk).
  destruct (Hsecond Hclosed') as (Hclosed'' & Hv' & Hk').
  split; [ exact Hclosed'' | split; eapply incl_tran; eassumption ].
Qed.

Lemma scoped_handler_refl (h : continuation_handler) : scoped_handler h h.
Proof. intros Hclosed. split; [ exact Hclosed | split; apply incl_refl ]. Qed.

Lemma scoped_bindings_refl (bindings : list static_binding) :
  scoped_bindings bindings bindings.
Proof. intros Hclosed. split; [ exact Hclosed | split; apply incl_refl ]. Qed.

Lemma scoped_handlers_refl
    (handlers : list (continuation * continuation_handler)) :
  scoped_handlers handlers handlers.
Proof. intros Hclosed. split; [ exact Hclosed | split; apply incl_refl ]. Qed.

Lemma scoped_let (x : variable) (kind kind' : flambda_kind) (d d' : named)
    (body body' : expr) :
  incl (free_variables_of_named d') (free_variables_of_named d) ->
  scoped body body' ->
  scoped (Let x kind d body) (Let x kind' d' body').
Proof.
  intros Hd Hbody Hclosed. scope_in Hclosed. scope.
  destruct (Hbody Hclosed) as (Hclosed' & Hv & Hk).
  split; [ exact Hclosed' | split; [ | exact Hk ] ].
  apply incl_app_app; [ exact Hd | apply incl_without, Hv ].
Qed.

Lemma scoped_let_static (bindings bindings' : list static_binding)
    (body body' : expr) :
  scoped_bindings bindings bindings' ->
  scoped body body' ->
  scoped (Let_static bindings body) (Let_static bindings' body').
Proof.
  intros Hbindings Hbody Hclosed. scope_in Hclosed. scope.
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hfirst Hrest].
  destruct (Hbindings Hfirst) as (Hfirst' & Hv & Hk).
  destruct (Hbody Hrest) as (Hrest' & Hv' & Hk').
  split; [ rewrite Hfirst', Hrest'; reflexivity | ].
  split; apply incl_app_app; assumption.
Qed.

Lemma scoped_let_cont (k : continuation) (h h' : continuation_handler)
    (body body' : expr) :
  scoped_handler h h' ->
  scoped body body' ->
  scoped (Let_cont_non_recursive k h body) (Let_cont_non_recursive k h' body').
Proof.
  intros Hh Hbody Hclosed. scope_in Hclosed. scope.
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hfirst Hrest].
  destruct (Hh Hfirst) as (Hfirst' & Hv & Hk).
  destruct (Hbody Hrest) as (Hrest' & Hv' & Hk').
  split; [ rewrite Hfirst', Hrest'; reflexivity | ].
  split; apply incl_app_app; try assumption. apply incl_without, Hk'.
Qed.

Lemma scoped_let_cont_recursive
    (handlers handlers' : list (continuation * continuation_handler))
    (body body' : expr) :
  map fst handlers' = map fst handlers ->
  scoped_handlers handlers handlers' ->
  scoped body body' ->
  scoped (Let_cont_recursive handlers body)
    (Let_cont_recursive handlers' body').
Proof.
  intros Hnames Hhandlers Hbody Hclosed. scope_in Hclosed. scope.
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hfirst Hrest].
  destruct (Hhandlers Hfirst) as (Hfirst' & Hv & Hk).
  destruct (Hbody Hrest) as (Hrest' & Hv' & Hk').
  split; [ rewrite Hfirst', Hrest'; reflexivity | ].
  split; [ apply incl_app_app; assumption | ].
  rewrite Hnames. apply incl_without, incl_app_app; assumption.
Qed.

Lemma scoped_handler_intro (params : list variable)
    (metadata metadata' : handler_metadata) (handler handler' : expr) :
  scoped handler handler' ->
  scoped_handler (Handler params metadata handler)
    (Handler params metadata' handler').
Proof.
  intros Hhandler Hclosed. scope_in Hclosed. scope.
  destruct (Hhandler Hclosed) as (Hclosed' & Hv & Hk).
  split; [ exact Hclosed' | split; [ apply incl_without, Hv | exact Hk ] ].
Qed.

Lemma scoped_code_intro (params : list variable) (my_closure : variable)
    (ret exn : continuation) (metadata metadata' : code_metadata)
    (body body' : expr) :
  scoped body body' ->
  scoped_code (Code params my_closure ret exn metadata body)
    (Code params my_closure ret exn metadata' body').
Proof.
  intros Hbody Hclosed. scope_in Hclosed. scope.
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hclosed Hrest].
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hv Hk].
  destruct (Hbody Hrest) as (Hrest' & Hv' & Hk').
  apply andb_true_iff. split; [ apply andb_true_iff; split | exact Hrest' ].
  - exact
      (incl_is_empty _ _ (incl_without (my_closure :: params) _ _ Hv') Hv).
  - exact (incl_is_empty _ _ (incl_without [ret; exn] _ _ Hk') Hk).
Qed.

Lemma code_is_closed_free (c : code) :
  code_is_closed c = true ->
  free_variables_of_code c = [] ∧ free_continuations_of_code c = [].
Proof.
  destruct c as [params my_closure ret exn metadata body].
  cbn [code_is_closed]. intros Hclosed.
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hclosed _].
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hv Hk].
  split; apply is_empty_nil; assumption.
Qed.

(* Closed code has no free continuation, and neither have symbols. *)
Lemma bindings_closed_no_continuations (bindings : list static_binding) :
  forallb binding_code_closed bindings = true ->
  flat_map free_continuations_of_binding bindings = [].
Proof.
  induction bindings as [ | binding bindings IH]; cbn [forallb flat_map];
    [ reflexivity | ].
  intros Hclosed. apply andb_true_iff in Hclosed.
  destruct Hclosed as [Hfirst Hrest]. rewrite (IH Hrest), app_nil_r.
  destruct binding as [id c | id | sym const]; try reflexivity.
  exact (proj2 (code_is_closed_free c Hfirst)).
Qed.

Lemma scoped_bindings_nil : scoped_bindings [] [].
Proof. apply scoped_bindings_refl. Qed.

Lemma scoped_bindings_cons_code (id id' : code_id) (c c' : code)
    (bindings bindings' : list static_binding) :
  scoped_code c c' ->
  scoped_bindings bindings bindings' ->
  scoped_bindings (Bind_code id c :: bindings) (Bind_code id' c' :: bindings').
Proof.
  intros Hc Hbindings Hclosed. cbn [forallb flat_map] in Hclosed |- *.
  scope_in Hclosed. scope.
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hfirst Hrest].
  destruct (Hbindings Hrest) as (Hrest' & Hv & Hk).
  destruct (code_is_closed_free c' (Hc Hfirst)) as [Hv' Hk'].
  rewrite (Hc Hfirst), Hrest', Hv', Hk'.
  split; [ reflexivity | split; apply incl_appr; assumption ].
Qed.

Lemma scoped_bindings_cons_same (binding : static_binding)
    (bindings bindings' : list static_binding) :
  scoped_bindings bindings bindings' ->
  scoped_bindings (binding :: bindings) (binding :: bindings').
Proof.
  intros Hbindings Hclosed. cbn [forallb flat_map] in Hclosed |- *.
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hfirst Hrest].
  destruct (Hbindings Hrest) as (Hrest' & Hv & Hk).
  rewrite Hfirst, Hrest'.
  split; [ reflexivity | ].
  split; (apply incl_app_app; [ apply incl_refl | assumption ]).
Qed.

Lemma scoped_handlers_nil : scoped_handlers [] [].
Proof. apply scoped_handlers_refl. Qed.

Lemma scoped_handlers_cons (k k' : continuation) (h h' : continuation_handler)
    (handlers handlers' : list (continuation * continuation_handler)) :
  scoped_handler h h' ->
  scoped_handlers handlers handlers' ->
  scoped_handlers ((k, h) :: handlers) ((k', h') :: handlers').
Proof.
  intros Hh Hhandlers Hclosed. cbn [forallb flat_map] in Hclosed |- *.
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hfirst Hrest].
  destruct (Hh Hfirst) as (Hfirst' & Hv & Hk).
  destruct (Hhandlers Hrest) as (Hrest' & Hv' & Hk').
  rewrite Hfirst', Hrest'.
  split; [ reflexivity | split; apply incl_app_app; assumption ].
Qed.

(* A rewrite keeps scope if its result keeps the scope of the term, and
   the other way round. *)
Lemma keeps_scope_intro (f : expr -> expr) :
  (∀ e : expr, scoped e (f e)) -> keeps_scope f.
Proof.
  intros Hkept e Hclosed. destruct (Hkept e Hclosed) as (Hclosed' & Hv & Hk).
  split; [ exact Hclosed' | ].
  split; intros name Hname; apply among_In; [ apply Hv | apply Hk ];
    apply among_In; exact Hname.
Qed.

Lemma keeps_scope_scoped (f : expr -> expr) :
  keeps_scope f -> ∀ e : expr, scoped e (f e).
Proof.
  intros Hkeeps e Hclosed. destruct (Hkeeps e Hclosed) as (Hclosed' & Hv & Hk).
  split; [ exact Hclosed' | ].
  split; intros name Hname; apply among_In; [ apply Hv | apply Hk ];
    apply among_In; exact Hname.
Qed.

(* The forms of term that most rewrites go through without a change. *)
Ltac scoped_under_binders rec :=
  first
    [ apply scoped_let; [ apply incl_refl | apply rec ]
    | apply scoped_let_static; [ apply scoped_bindings_refl | apply rec ]
    | apply scoped_let_cont; [ apply scoped_handler_intro, rec | apply rec ]
    | apply scoped_let_cont_recursive;
      [ reflexivity | apply scoped_handlers_refl | apply rec ]
    | apply scoped_refl ].

(* ======================================================================== *)
(* More lists of names                                                      *)
(* ======================================================================== *)

Lemma incl_flat_map_under {A B : Type} (names_of : A -> list positive)
    (names_of' : B -> list positive) (f : A -> B) (names : list positive)
    (l : list A) :
  (∀ a : A, incl (names_of' (f a)) (names_of a ++ names)) ->
  incl (flat_map names_of' (map f l)) (flat_map names_of l ++ names).
Proof.
  intros Hone. induction l as [ | a l IH]; cbn [map flat_map];
    [ intros y [] | ].
  intros y Hy. apply in_app_or in Hy. apply in_or_app.
  destruct Hy as [Hy | Hy]; [ apply Hone in Hy | apply IH in Hy ];
    apply in_app_or in Hy;
    (destruct Hy as [Hy | Hy]; [ left; apply in_or_app | right; exact Hy ]);
    [ left | right ]; exact Hy.
Qed.

Lemma incl_app_under (l l' m m' names : list positive) :
  incl l (m ++ names) ->
  incl l' (m' ++ names) ->
  incl (l ++ l') ((m ++ m') ++ names).
Proof.
  intros Hfirst Hsecond y Hy. apply in_app_or in Hy. apply in_or_app.
  destruct Hy as [Hy | Hy]; [ apply Hfirst in Hy | apply Hsecond in Hy ];
    apply in_app_or in Hy;
    (destruct Hy as [Hy | Hy]; [ left; apply in_or_app | right; exact Hy ]);
    [ left | right ]; exact Hy.
Qed.

Lemma flat_map_map_same {A B : Type} (names_of : A -> list positive)
    (names_of' : B -> list positive) (f : A -> B) (l : list A) :
  (∀ a : A, names_of' (f a) = names_of a) ->
  flat_map names_of' (map f l) = flat_map names_of l.
Proof.
  intros Hone. induction l as [ | a l IH]; cbn [map flat_map];
    [ reflexivity | rewrite Hone, IH; reflexivity ].
Qed.

Lemma forallb_filter {A : Type} (f keep : A -> bool) (l : list A) :
  forallb f l = true -> forallb f (filter keep l) = true.
Proof.
  intros Hall. apply forallb_forall. intros a Ha. apply filter_In in Ha.
  exact (proj1 (forallb_forall f l) Hall a (proj1 Ha)).
Qed.

(* ======================================================================== *)
(* A binder that is not used, and a Switch that goes to an arm              *)
(* ======================================================================== *)

(* The test of a free variable says yes of every free variable, and
   that of a free continuation of every free continuation, where the
   code is closed. *)
Lemma var_free_b_free (x : variable) (e : expr) :
  code_closed e = true ->
  In x (free_variables_of_expr e) ->
  var_free_b x e = true.
Proof.
  intros Hclosed Hx. apply var_free_b_sound.
  apply (proj2 (var_free_agrees x e Hclosed)), among_In, Hx.
Qed.

Lemma cont_free_b_free (k : continuation) (e : expr) :
  code_closed e = true ->
  In k (free_continuations_of_expr e) ->
  cont_free_b k e = true.
Proof.
  intros Hclosed Hk. apply cont_free_b_sound.
  apply (proj2 (cont_free_agrees k e Hclosed)), among_In, Hk.
Qed.

(* A [Let] whose variable is not used, removed. *)
Lemma scoped_let_unused (x : variable) (kind : flambda_kind) (d : named)
    (body body' : expr) :
  scoped body body' ->
  var_free_b x body' = false ->
  scoped (Let x kind d body) body'.
Proof.
  intros Hbody Hunused Hclosed. scope_in Hclosed. scope.
  destruct (Hbody Hclosed) as (Hclosed' & Hv & Hk).
  split; [ exact Hclosed' | split; [ | exact Hk ] ].
  intros y Hy. apply in_or_app. right. apply in_without.
  split; [ apply Hv, Hy | ]. intros [<- | []].
  rewrite (var_free_b_free x body' Hclosed' Hy) in Hunused. discriminate.
Qed.

(* A [Let_cont] whose continuation is not used, removed. *)
Lemma scoped_let_cont_unused (k : continuation) (h : continuation_handler)
    (body body' : expr) :
  scoped body body' ->
  cont_free_b k body' = false ->
  scoped (Let_cont_non_recursive k h body) body'.
Proof.
  intros Hbody Hunused Hclosed. scope_in Hclosed. scope.
  apply andb_true_iff in Hclosed. destruct Hclosed as [_ Hclosed].
  destruct (Hbody Hclosed) as (Hclosed' & Hv & Hk).
  split; [ exact Hclosed' | split; [ apply incl_appr, Hv | ] ].
  intros j Hj. apply in_or_app. right. apply in_without.
  split; [ apply Hk, Hj | ]. intros [<- | []].
  rewrite (cont_free_b_free k body' Hclosed' Hj) in Hunused. discriminate.
Qed.

(* A [Switch] that goes to one of its arms. *)
Lemma scoped_switch_arm (scrutinee : simple) (arms : list (Z * apply_cont))
    (i : Z) (ac : apply_cont) :
  In (i, ac) arms -> scoped (Switch scrutinee arms) (Apply_cont ac).
Proof.
  intros Hin _. scope. split; [ reflexivity | split ].
  - apply incl_appr. intros x Hx. apply in_flat_map. exists (i, ac).
    split; [ exact Hin | exact Hx ].
  - intros k Hk. apply in_flat_map. exists (i, ac).
    split; [ exact Hin | exact Hk ].
Qed.

Lemma find_arm_In (i : Z) (arms : list (Z * apply_cont)) (ac : apply_cont) :
  find_arm i arms = Some ac -> ∃ j : Z, In (j, ac) arms.
Proof.
  induction arms as [ | [j ac'] arms IH]; cbn [find_arm]; [ discriminate | ].
  destruct (Z.eqb i j).
  - intros Hfound. injection Hfound as ->. exists j. left. reflexivity.
  - intros Hfound. destruct (IH Hfound) as [j' Hin]. exists j'. right.
    exact Hin.
Qed.

(* ======================================================================== *)
(* Groups of static bindings                                                *)
(* ======================================================================== *)

Lemma scoped_bindings_filter (keep : static_binding -> bool)
    (bindings : list static_binding) :
  scoped_bindings bindings (filter keep bindings).
Proof.
  intros Hclosed.
  split; [ apply forallb_filter, Hclosed | split; apply incl_flat_map_filter ].
Qed.

Lemma scoped_bindings_trans
    (bindings bindings' bindings'' : list static_binding) :
  scoped_bindings bindings bindings' ->
  scoped_bindings bindings' bindings'' ->
  scoped_bindings bindings bindings''.
Proof.
  intros Hfirst Hsecond Hclosed.
  destruct (Hfirst Hclosed) as (Hclosed' & Hv & Hk).
  destruct (Hsecond Hclosed') as (Hclosed'' & Hv' & Hk').
  split; [ exact Hclosed'' | split; eapply incl_tran; eassumption ].
Qed.

Lemma scoped_bindings_code_first (bindings : list static_binding) :
  scoped_bindings bindings (code_first bindings).
Proof.
  intros Hclosed. unfold code_first.
  rewrite forallb_app, !flat_map_app, !forallb_filter by exact Hclosed.
  split; [ reflexivity | ].
  split; apply incl_app; apply incl_flat_map_filter.
Qed.

Lemma scoped_let_static_none (bindings : list static_binding)
    (body body' : expr) :
  scoped body body' -> scoped (Let_static bindings body) body'.
Proof.
  intros Hbody Hclosed. scope_in Hclosed. scope.
  apply andb_true_iff in Hclosed. destruct Hclosed as [_ Hclosed].
  destruct (Hbody Hclosed) as (Hclosed' & Hv & Hk).
  split; [ exact Hclosed' | split; apply incl_appr; assumption ].
Qed.

Lemma scoped_let_static_if_any (bindings bindings' : list static_binding)
    (body body' : expr) :
  scoped_bindings bindings bindings' ->
  scoped body body' ->
  scoped (Let_static bindings body) (let_static_if_any bindings' body').
Proof.
  intros Hbindings Hbody. destruct bindings' as [ | binding bindings'];
    cbn [let_static_if_any];
    [ apply scoped_let_static_none, Hbody
    | apply scoped_let_static; assumption ].
Qed.

(* Two groups of static bindings, one in the scope of the other, as
   one. *)
Lemma scoped_let_static_merge (bindings bindings' : list static_binding)
    (body : expr) :
  scoped (Let_static bindings (Let_static bindings' body))
    (Let_static (bindings ++ bindings') body).
Proof.
  intros Hclosed. scope_in Hclosed. scope.
  rewrite forallb_app, !flat_map_app, <- !app_assoc, <- andb_assoc.
  split; [ exact Hclosed | split; apply incl_refl ].
Qed.

(* ======================================================================== *)
(* Terms that bind nothing static, and parameters bound to arguments        *)
(* ======================================================================== *)

(* A term that binds nothing static binds no code. *)
Fixpoint binds_statics_code_closed (e : expr) :
  binds_statics e = false -> code_closed e = true
with handler_binds_statics_code_closed (h : continuation_handler) :
  handler_binds_statics h = false -> handler_code_closed h = true.
Proof.
  - destruct e
      as [x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [binds_statics handler_binds_statics]; scope; intros Hbinds;
      try reflexivity.
    + exact (binds_statics_code_closed body Hbinds).
    + apply orb_false_iff in Hbinds. destruct Hbinds as [Hgroup Hbody].
      rewrite (binds_statics_code_closed body Hbody), andb_true_r.
      induction bindings as [ | binding bindings IH];
        cbn [existsb forallb] in Hgroup |- *; [ reflexivity | ].
      apply orb_false_iff in Hgroup. destruct Hgroup as [Hfirst Hrest].
      rewrite (IH Hrest), andb_true_r.
      destruct binding;
        [ discriminate Hfirst | reflexivity | discriminate Hfirst ].
    + apply orb_false_iff in Hbinds. destruct Hbinds as [Hh Hbody].
      rewrite (handler_binds_statics_code_closed h Hh),
        (binds_statics_code_closed body Hbody).
      reflexivity.
    + apply orb_false_iff in Hbinds. destruct Hbinds as [Hgroup Hbody].
      rewrite (binds_statics_code_closed body Hbody), andb_true_r.
      induction handlers as [ | [k h] handlers IH];
        cbn [existsb forallb] in Hgroup |- *; [ reflexivity | ].
      apply orb_false_iff in Hgroup. destruct Hgroup as [Hfirst Hrest].
      rewrite (IH Hrest), (handler_binds_statics_code_closed h Hfirst).
      reflexivity.
  - destruct h as [params metadata handler].
    cbn [binds_statics handler_binds_statics]. scope.
    exact (binds_statics_code_closed handler).
Qed.

(* The parameters of a continuation or of code bound to arguments. *)
Lemma let_params_code_closed (params : list variable) :
  ∀ (kinds : list flambda_kind) (args : list simple) (body : expr),
    code_closed (let_params params kinds args body) = code_closed body.
Proof.
  induction params as [ | param params IH]; intros kinds [ | arg args] body;
    cbn [let_params]; scope; try reflexivity.
  apply IH.
Qed.

Lemma let_params_free_continuations (params : list variable) :
  ∀ (kinds : list flambda_kind) (args : list simple) (body : expr),
    free_continuations_of_expr (let_params params kinds args body)
    = free_continuations_of_expr body.
Proof.
  induction params as [ | param params IH]; intros kinds [ | arg args] body;
    cbn [let_params]; scope; try reflexivity.
  apply IH.
Qed.

Lemma let_params_free_variables (params : list variable) :
  ∀ (kinds : list flambda_kind) (args : list simple) (body : expr)
    (x : variable),
    length params = length args ->
    In x (free_variables_of_expr (let_params params kinds args body)) ->
    In x (free_variables_of_simples args)
    ∨ (In x (free_variables_of_expr body) ∧ ~ In x params).
Proof.
  induction params as [ | param params IH];
    intros kinds [ | arg args] body x Hlength; try discriminate;
    cbn [let_params]; scope.
  - intros Hx. right. split; [ exact Hx | intros [] ].
  - cbn [free_variables_of_named]. intros Hx. apply in_app_or in Hx.
    unfold free_variables_of_simples. cbn [flat_map].
    destruct Hx as [Hx | Hx]; [ left; apply in_or_app; left; exact Hx | ].
    apply in_without in Hx. destruct Hx as [Hx Hparam].
    apply IH in Hx; [ | simpl in Hlength; congruence ].
    destruct Hx as [Hx | [Hx Hparams]];
      [ left; apply in_or_app; right; exact Hx | right ].
    split; [ exact Hx | ].
    intros [Heq | Hin]; [ apply Hparam; left; exact Heq | exact (Hparams Hin) ].
Qed.

(* The parameters of tupled code bound to the fields of a tuple. *)
Lemma let_tuple_fields_code_closed (params : list variable) :
  ∀ (size field : nat) (tuple : simple) (body : expr),
    code_closed (let_tuple_fields params size field tuple body)
    = code_closed body.
Proof.
  induction params as [ | param params IH]; intros size field tuple body;
    cbn [let_tuple_fields]; scope; [ reflexivity | ].
  apply IH.
Qed.

Lemma let_tuple_fields_free_continuations (params : list variable) :
  ∀ (size field : nat) (tuple : simple) (body : expr),
    free_continuations_of_expr (let_tuple_fields params size field tuple body)
    = free_continuations_of_expr body.
Proof.
  induction params as [ | param params IH]; intros size field tuple body;
    cbn [let_tuple_fields]; scope; [ reflexivity | ].
  apply IH.
Qed.

Lemma let_tuple_fields_free_variables (params : list variable) :
  ∀ (size field : nat) (tuple : simple) (body : expr) (x : variable),
    In x
      (free_variables_of_expr
         (let_tuple_fields params size field tuple body)) ->
    In x (free_variables_of_simples [tuple])
    ∨ (In x (free_variables_of_expr body) ∧ ~ In x params).
Proof.
  induction params as [ | param params IH];
    intros size field tuple body x; cbn [let_tuple_fields]; scope.
  - intros Hx. right. split; [ exact Hx | intros [] ].
  - cbn [free_variables_of_named]. intros Hx. apply in_app_or in Hx.
    destruct Hx as [Hx | Hx]; [ left; exact Hx | ].
    apply in_without in Hx. destruct Hx as [Hx Hparam].
    apply IH in Hx.
    destruct Hx as [Hx | [Hx Hparams]]; [ left; exact Hx | right ].
    split; [ exact Hx | ].
    intros [Heq | Hin]; [ apply Hparam; left; exact Heq | exact (Hparams Hin) ].
Qed.

(* ======================================================================== *)
(* The rewrites that go by what is known                                    *)
(* ======================================================================== *)

(* What is known where a subterm is holds variables that are free in
   the term or bound around the subterm. The free variables of what the
   subterm becomes are free in the subterm or among them. *)
Definition scoped_under (names : list variable) (e e' : expr) : Prop :=
  code_closed e = true ->
  code_closed e' = true
  ∧ incl (free_variables_of_expr e') (free_variables_of_expr e ++ names)
  ∧ incl (free_continuations_of_expr e') (free_continuations_of_expr e).

Lemma scoped_under_none (e e' : expr) : scoped_under [] e e' -> scoped e e'.
Proof.
  intros Hunder Hclosed. destruct (Hunder Hclosed) as (Hclosed' & Hv & Hk).
  rewrite app_nil_r in Hv. split; [ exact Hclosed' | split; assumption ].
Qed.

Lemma scoped_under_refl (names : list variable) (e : expr) :
  scoped_under names e e.
Proof.
  intros Hclosed.
  split; [ exact Hclosed | split; [ apply incl_appl | ]; apply incl_refl ].
Qed.

Lemma scoped_under_let (names : list variable) (x : variable)
    (kind : flambda_kind) (d d' : named) (body body' : expr) :
  incl (free_variables_of_named d') (free_variables_of_named d ++ names) ->
  scoped_under (x :: names ++ free_variables_of_named d) body body' ->
  scoped_under names (Let x kind d body) (Let x kind d' body').
Proof.
  intros Hd Hbody Hclosed. scope_in Hclosed. scope.
  destruct (Hbody Hclosed) as (Hclosed' & Hv & Hk).
  split; [ exact Hclosed' | split; [ | exact Hk ] ].
  intros y Hy. apply in_app_or in Hy. destruct Hy as [Hy | Hy].
  - apply Hd, in_app_or in Hy. apply in_or_app.
    destruct Hy as [Hy | Hy];
      [ left; apply in_or_app; left; exact Hy | right; exact Hy ].
  - apply in_without in Hy. destruct Hy as [Hy Hx].
    apply Hv, in_app_or in Hy. apply in_or_app.
    destruct Hy as [Hy | [Heq | Hy]].
    + left. apply in_or_app. right. apply in_without. split; assumption.
    + destruct Hx. left. exact Heq.
    + apply in_app_or in Hy.
      destruct Hy as [Hy | Hy];
        [ right; exact Hy | left; apply in_or_app; left; exact Hy ].
Qed.

(* A group of static bindings that may change. *)
Lemma scoped_under_let_static_bindings (names : list variable)
    (bindings bindings' : list static_binding) (body body' : expr) :
  (forallb binding_code_closed bindings = true ->
   forallb binding_code_closed bindings' = true) ->
  incl (flat_map free_variables_of_binding bindings')
    (flat_map free_variables_of_binding bindings ++ names) ->
  incl (flat_map free_continuations_of_binding bindings')
    (flat_map free_continuations_of_binding bindings) ->
  scoped_under (names ++ flat_map free_variables_of_binding bindings) body
    body' ->
  scoped_under names (Let_static bindings body) (Let_static bindings' body').
Proof.
  intros Hcode Hgroup_v Hgroup_k Hbody Hclosed. scope_in Hclosed. scope.
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hgroup Hclosed].
  destruct (Hbody Hclosed) as (Hclosed' & Hv & Hk).
  rewrite (Hcode Hgroup), Hclosed'.
  split; [ reflexivity | split; [ | apply incl_app_app; assumption ] ].
  intros y Hy. apply in_app_or in Hy. apply in_or_app.
  destruct Hy as [Hy | Hy].
  - apply Hgroup_v, in_app_or in Hy.
    destruct Hy as [Hy | Hy];
      [ left; apply in_or_app; left; exact Hy | right; exact Hy ].
  - apply Hv, in_app_or in Hy.
    destruct Hy as [Hy | Hy]; [ left; apply in_or_app; right; exact Hy | ].
    apply in_app_or in Hy.
    destruct Hy as [Hy | Hy];
      [ right; exact Hy | left; apply in_or_app; left; exact Hy ].
Qed.

Lemma scoped_under_let_static (names : list variable)
    (bindings : list static_binding) (body body' : expr) :
  scoped_under (names ++ flat_map free_variables_of_binding bindings) body
    body' ->
  scoped_under names (Let_static bindings body) (Let_static bindings body').
Proof.
  apply scoped_under_let_static_bindings;
    [ intros Hcode; exact Hcode | apply incl_appl, incl_refl
    | apply incl_refl ].
Qed.

Lemma scoped_under_let_cont (names : list variable) (k : continuation)
    (params : list variable) (metadata : handler_metadata)
    (handler handler' body body' : expr) :
  scoped_under names handler handler' ->
  scoped_under names body body' ->
  scoped_under names
    (Let_cont_non_recursive k (Handler params metadata handler) body)
    (Let_cont_non_recursive k (Handler params metadata handler') body').
Proof.
  intros Hhandler Hbody Hclosed. scope_in Hclosed. scope.
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hfirst Hrest].
  destruct (Hhandler Hfirst) as (Hfirst' & Hv & Hk).
  destruct (Hbody Hrest) as (Hrest' & Hv' & Hk').
  rewrite Hfirst', Hrest'. split; [ reflexivity | split ].
  - intros y Hy. apply in_app_or in Hy. apply in_or_app.
    destruct Hy as [Hy | Hy].
    + apply in_without in Hy. destruct Hy as [Hy Hparams].
      apply Hv, in_app_or in Hy.
      destruct Hy as [Hy | Hy]; [ left | right; exact Hy ].
      apply in_or_app. left. apply in_without. split; assumption.
    + apply Hv', in_app_or in Hy.
      destruct Hy as [Hy | Hy];
        [ left; apply in_or_app; right; exact Hy | right; exact Hy ].
  - apply incl_app_app; [ exact Hk | apply incl_without, Hk' ].
Qed.

Lemma scoped_under_let_cont_recursive (names : list variable)
    (handlers : list (continuation * continuation_handler))
    (body body' : expr) :
  scoped_under names body body' ->
  scoped_under names (Let_cont_recursive handlers body)
    (Let_cont_recursive handlers body').
Proof.
  intros Hbody Hclosed. scope_in Hclosed. scope.
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hgroup Hclosed].
  destruct (Hbody Hclosed) as (Hclosed' & Hv & Hk).
  rewrite Hgroup, Hclosed'. split; [ reflexivity | split ].
  - intros y Hy. apply in_app_or in Hy. apply in_or_app.
    destruct Hy as [Hy | Hy]; [ left; apply in_or_app; left; exact Hy | ].
    apply Hv, in_app_or in Hy.
    destruct Hy as [Hy | Hy];
      [ left; apply in_or_app; right; exact Hy | right; exact Hy ].
  - apply incl_without, incl_app_app; [ apply incl_refl | exact Hk ].
Qed.

(* ======================================================================== *)
(* The comparison                                                           *)
(* ======================================================================== *)

(* Every name of the second list stands, by the names paired, where a
   name of the first does. *)
Definition matched (m : name_map) (names names' : list positive) : Prop :=
  ∀ a' : positive,
    In a' names' -> ∃ a : positive, In a names ∧ names_match m a a' = true.

Lemma matched_nil (m : name_map) (names : list positive) :
  matched m names [].
Proof. intros a' []. Qed.

Lemma matched_app (m : name_map) (names names2 names' names2' : list positive) :
  matched m names names' ->
  matched m names2 names2' ->
  matched m (names ++ names2) (names' ++ names2').
Proof.
  intros Hfirst Hsecond a' Ha'. apply in_app_or in Ha'.
  destruct Ha' as [Ha' | Ha'];
    [ destruct (Hfirst a' Ha') as (a & Ha & Hmatch)
    | destruct (Hsecond a' Ha') as (a & Ha & Hmatch) ];
    exists a; (split; [ apply in_or_app | exact Hmatch ]); tauto.
Qed.

Lemma matched_cons (m : name_map) (a a' : positive)
    (names names' : list positive) :
  names_match m a a' = true ->
  matched m names names' ->
  matched m (a :: names) (a' :: names').
Proof.
  intros Hmatch Hrest b' [<- | Hb'].
  - exists a. split; [ left; reflexivity | exact Hmatch ].
  - destruct (Hrest b' Hb') as (b & Hb & Hmatch').
    exists b. split; [ right; exact Hb | exact Hmatch' ].
Qed.

Lemma matched_one (m : name_map) (a a' : positive) :
  names_match m a a' = true -> matched m [a] [a'].
Proof.
  intros Hmatch. apply matched_cons; [ exact Hmatch | apply matched_nil ].
Qed.

Lemma matched_is_empty (m : name_map) (names names' : list positive) :
  matched m names names' ->
  Keyed.is_empty names = true ->
  Keyed.is_empty names' = true.
Proof.
  intros Hmatched Hempty. apply is_empty_nil in Hempty. subst names.
  destruct names' as [ | a' names']; [ reflexivity | ].
  destruct (Hmatched a' (or_introl eq_refl)) as (a & [] & _).
Qed.

(* A name that is not the one bound last on its side is matched by the
   names bound before. *)
Lemma names_match_other (m : name_map) (b b' a a' : positive) :
  names_match ((b, b') :: m) a a' = true ->
  a' <> b' ->
  a <> b ∧ names_match m a a' = true.
Proof.
  cbn [names_match]. destruct (Pos.eqb_spec a b) as [-> | Hne].
  - intros Heq Hne'. apply Pos.eqb_eq in Heq. contradiction.
  - destruct (Pos.eqb_spec a' b'); [ discriminate | ].
    intros Hmatch _. split; assumption.
Qed.

Lemma names_match_other_one (m : name_map) (b b' a a' : positive) :
  names_match ((b, b') :: m) a a' = true ->
  ~ In a' [b'] ->
  ~ In a [b] ∧ names_match m a a' = true.
Proof.
  intros Hmatch Hout.
  apply names_match_other in Hmatch;
    [ | intros ->; apply Hout; left; reflexivity ].
  destruct Hmatch as [Hne Hmatch]. split; [ | exact Hmatch ].
  intros [Heq | []]. apply Hne. symmetry. exact Heq.
Qed.

Lemma bind_names_other (xs : list positive) :
  ∀ (xs' : list positive) (m m' : name_map) (a a' : positive),
    bind_names xs xs' m = Some m' ->
    names_match m' a a' = true ->
    ~ In a' xs' ->
    ~ In a xs ∧ names_match m a a' = true.
Proof.
  induction xs as [ | x xs IH]; intros [ | x' xs'] m m' a a' Hbind Hmatch Hout;
    cbn [bind_names] in Hbind; try discriminate.
  - injection Hbind as <-. split; [ intros [] | exact Hmatch ].
  - destruct (IH xs' _ m' a a' Hbind Hmatch) as [Hxs Hmatch'].
    { intros Hin. apply Hout. right. exact Hin. }
    apply names_match_other in Hmatch';
      [ | intros ->; apply Hout; left; reflexivity ].
    destruct Hmatch' as [Hne Hmatch']. split; [ | exact Hmatch' ].
    intros [Heq | Hin]; [ apply Hne; symmetry; exact Heq | exact (Hxs Hin) ].
Qed.

Lemma combine_names_other (xs : list positive) :
  ∀ (xs' : list positive) (m : name_map) (a a' : positive),
    length xs = length xs' ->
    names_match (combine xs xs' ++ m) a a' = true ->
    ~ In a' xs' ->
    ~ In a xs ∧ names_match m a a' = true.
Proof.
  induction xs as [ | x xs IH]; intros [ | x' xs'] m a a' Hlength Hmatch Hout;
    try discriminate.
  - split; [ intros [] | exact Hmatch ].
  - cbn [combine app] in Hmatch.
    apply names_match_other in Hmatch;
      [ | intros ->; apply Hout; left; reflexivity ].
    destruct Hmatch as [Hne Hmatch].
    destruct (IH xs' m a a') as [Hxs Hmatch'];
      [ simpl in Hlength; congruence | exact Hmatch
      | intros Hin; apply Hout; right; exact Hin | ].
    split; [ | exact Hmatch' ].
    intros [Heq | Hin]; [ apply Hne; symmetry; exact Heq | exact (Hxs Hin) ].
Qed.

(* Under binders: the names bound on each side taken away. *)
Lemma matched_without (m m' : name_map)
    (bound bound' names names' : list positive) :
  (∀ a a' : positive,
    names_match m' a a' = true ->
    ~ In a' bound' ->
    ~ In a bound ∧ names_match m a a' = true) ->
  matched m' names names' ->
  matched m (without bound names) (without bound' names').
Proof.
  intros Hother Hmatched a' Ha'. apply in_without in Ha'.
  destruct Ha' as [Ha' Hout].
  destruct (Hmatched a' Ha') as (a & Ha & Hmatch).
  destruct (Hother a a' Hmatch Hout) as [Hout' Hmatch'].
  exists a. split; [ apply in_without; split; assumption | exact Hmatch' ].
Qed.

Lemma forall2b_matched {A B : Type} (eqb : A -> B -> bool)
    (names : A -> list positive) (names' : B -> list positive)
    (m : name_map) :
  (∀ (a : A) (b : B), eqb a b = true -> matched m (names a) (names' b)) ->
  ∀ (l : list A) (l' : list B),
    forall2b eqb l l' = true ->
    matched m (flat_map names l) (flat_map names' l').
Proof.
  intros Hone. induction l as [ | a l IH]; intros [ | b l'];
    cbn [forall2b flat_map]; try discriminate.
  - intros _. apply matched_nil.
  - intros Hall. apply andb_true_iff in Hall. destruct Hall as [Hfirst Hrest].
    apply matched_app; [ apply Hone, Hfirst | apply IH, Hrest ].
Qed.

Lemma simple_eqb_matched (vm : name_map) (s s' : simple) :
  simple_eqb vm s s' = true ->
  matched vm (free_variables_of_simple s) (free_variables_of_simple s').
Proof.
  destruct s as [x | sym | c], s' as [x' | sym' | c']; simpl;
    try discriminate; intros Hmatch;
    first [ apply matched_one; exact Hmatch | apply matched_nil ].
Qed.

Lemma simples_eqb_matched (vm : name_map) (args args' : list simple) :
  simples_eqb vm args args' = true ->
  matched vm (free_variables_of_simples args)
    (free_variables_of_simples args').
Proof. apply forall2b_matched. intros s s'. apply simple_eqb_matched. Qed.

Lemma value_slots_eqb_matched (vm : name_map)
    (slots slots' : list (value_slot * simple)) :
  value_slots_eqb vm slots slots' = true ->
  matched vm (free_variables_of_value_slots slots)
    (free_variables_of_value_slots slots').
Proof.
  apply
    (forall2b_matched _ (fun slot => free_variables_of_simple (snd slot))
       (fun slot => free_variables_of_simple (snd slot))).
  intros slot slot' Heq. apply andb_true_iff in Heq.
  apply simple_eqb_matched. exact (proj2 Heq).
Qed.

Ltac eqb_parts :=
  repeat
    match goal with
    | H : _ && _ = true |- _ =>
      apply andb_true_iff in H; destruct H as [? ?]
    end.

Lemma named_eqb_matched (vm : name_map) (d d' : named) :
  named_eqb vm d d' = true ->
  matched vm (free_variables_of_named d) (free_variables_of_named d').
Proof.
  destruct d, d'; cbn [named_eqb free_variables_of_named];
    try discriminate; intros Heq; eqb_parts;
    first
      [ apply simple_eqb_matched | apply simples_eqb_matched
      | apply value_slots_eqb_matched ];
    assumption.
Qed.

Lemma static_const_eqb_matched (vm : name_map) (const const' : static_const) :
  static_const_eqb vm const const' = true ->
  matched vm (free_variables_of_static_const const)
    (free_variables_of_static_const const').
Proof.
  destruct const, const';
    cbn [static_const_eqb free_variables_of_static_const];
    try discriminate; intros Heq; eqb_parts;
    first
      [ apply matched_nil
      | apply simple_eqb_matched; assumption
      | apply simples_eqb_matched; assumption
      | apply value_slots_eqb_matched; assumption ].
Qed.

Lemma apply_cont_eqb_matched (vm km : name_map) (ac ac' : apply_cont) :
  apply_cont_eqb vm km ac ac' = true ->
  matched vm (free_variables_of_apply_cont ac)
    (free_variables_of_apply_cont ac')
  ∧ matched km (apply_cont_continuations ac) (apply_cont_continuations ac').
Proof.
  unfold apply_cont_eqb. intros Heq.
  apply andb_true_iff in Heq. destruct Heq as [Heq Htrap].
  apply andb_true_iff in Heq. destruct Heq as [Hk Hargs].
  split; [ apply simples_eqb_matched, Hargs | ].
  unfold apply_cont_continuations.
  destruct (ac_trap_action ac) as [[j | j kind] | ],
    (ac_trap_action ac') as [[j' | j' kind'] | ];
    cbn [trap_action_eqb trap_action_handler] in Htrap |- *;
    try discriminate.
  - apply matched_cons; [ exact Hk | apply matched_one, Htrap ].
  - apply andb_true_iff in Htrap.
    apply matched_cons; [ exact Hk | apply matched_one, (proj1 Htrap) ].
  - apply matched_one, Hk.
Qed.

Lemma apply_eqb_matched (vm km : name_map) (ap ap' : apply_expr) :
  apply_eqb vm km ap ap' = true ->
  matched vm (free_variables_of_apply ap) (free_variables_of_apply ap')
  ∧ matched km [ap_continuation ap; ap_exn_continuation ap]
      [ap_continuation ap'; ap_exn_continuation ap'].
Proof.
  unfold apply_eqb. intros Heq.
  apply andb_true_iff in Heq. destruct Heq as [Heq _].
  apply andb_true_iff in Heq. destruct Heq as [Heq _].
  apply andb_true_iff in Heq. destruct Heq as [Heq _].
  apply andb_true_iff in Heq. destruct Heq as [Heq Hextra].
  apply andb_true_iff in Heq. destruct Heq as [Heq Hexn].
  apply andb_true_iff in Heq. destruct Heq as [Heq Hk].
  apply andb_true_iff in Heq. destruct Heq as [Heq Hargs].
  apply andb_true_iff in Heq. destruct Heq as [Hcallee _].
  split.
  - unfold free_variables_of_apply.
    apply matched_app; [ apply simple_eqb_matched, Hcallee | ].
    apply matched_app; apply simples_eqb_matched; assumption.
  - apply matched_cons; [ exact Hk | apply matched_one, Hexn ].
Qed.

(* What the comparison accepts: closed code against closed code, and
   free variables and continuations that are matched. *)
Fixpoint expr_eqb_scope (e : expr) :
  ∀ (vm km : name_map) (e' : expr),
    expr_eqb vm km e e' = true ->
    code_closed e = true ->
    code_closed e' = true
    ∧ matched vm (free_variables_of_expr e) (free_variables_of_expr e')
    ∧ matched km (free_continuations_of_expr e)
        (free_continuations_of_expr e')
with handler_eqb_scope (h : continuation_handler) :
  ∀ (vm km : name_map) (h' : continuation_handler),
    handler_eqb vm km h h' = true ->
    handler_code_closed h = true ->
    handler_code_closed h' = true
    ∧ matched vm (free_variables_of_handler h) (free_variables_of_handler h')
    ∧ matched km (free_continuations_of_handler h)
        (free_continuations_of_handler h')
with binding_eqb_scope (binding : static_binding) :
  ∀ (vm : name_map) (binding' : static_binding),
    binding_eqb vm binding binding' = true ->
    binding_code_closed binding = true ->
    binding_code_closed binding' = true
    ∧ matched vm (free_variables_of_binding binding)
        (free_variables_of_binding binding')
    ∧ free_continuations_of_binding binding' = []
with code_eqb_scope (c : code) :
  ∀ c' : code,
    code_eqb c c' = true -> code_is_closed c = true -> code_is_closed c' = true.
Proof.
  - destruct e
      as [x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros vm km e' Heq Hclosed;
      destruct e'
        as [x' kind' d' body' | bindings' body' | k' h' body'
           | handlers' body' | ap' | ac' | scrutinee' arms' | ];
      cbn [expr_eqb handler_eqb binding_eqb code_eqb] in Heq;
      try discriminate; scope_in Hclosed; scope.
    + apply andb_true_iff in Heq. destruct Heq as [Heq Hbody].
      apply andb_true_iff in Heq. destruct Heq as [_ Hd].
      destruct (expr_eqb_scope body _ _ body' Hbody Hclosed)
        as (Hclosed' & Hv & Hk).
      split; [ exact Hclosed' | split; [ | exact Hk ] ].
      apply matched_app; [ apply named_eqb_matched, Hd | ].
      apply (matched_without vm ((x, x') :: vm) [x] [x']); [ | exact Hv ].
      intros a a'. apply names_match_other_one.
    + assert (Hbindings :
        ∀ others : list static_binding,
          forall2b (binding_eqb vm) bindings others = true ->
          forallb binding_code_closed bindings = true ->
          forallb binding_code_closed others = true
          ∧ matched vm (flat_map free_variables_of_binding bindings)
              (flat_map free_variables_of_binding others)
          ∧ flat_map free_continuations_of_binding others = []).
      { clear Heq Hclosed.
        induction bindings as [ | binding bindings IH];
          intros [ | other others]; cbn [forall2b forallb flat_map];
          try discriminate.
        - intros _ _.
          split; [ reflexivity | split; [ apply matched_nil | reflexivity ] ].
        - intros Hall Hcode.
          apply andb_true_iff in Hall. destruct Hall as [Hfirst Hrest].
          apply andb_true_iff in Hcode.
          destruct Hcode as [Hcode_first Hcode_rest].
          destruct (binding_eqb_scope binding vm other Hfirst Hcode_first)
            as (Hc1 & Hv1 & Hk1).
          destruct (IH others Hrest Hcode_rest) as (Hc2 & Hv2 & Hk2).
          split; [ rewrite Hc1, Hc2; reflexivity | ].
          split; [ apply matched_app; assumption | ].
          rewrite Hk1, Hk2. reflexivity. }
      apply andb_true_iff in Heq. destruct Heq as [Hgroup Hbody].
      apply andb_true_iff in Hclosed. destruct Hclosed as [Hcode Hcode_body].
      destruct (Hbindings bindings' Hgroup Hcode) as (Hc1 & Hv1 & Hk1).
      destruct (expr_eqb_scope body vm km body' Hbody Hcode_body)
        as (Hc2 & Hv2 & Hk2).
      split; [ rewrite Hc1, Hc2; reflexivity | ].
      split; [ apply matched_app; assumption | ].
      rewrite Hk1. cbn [app]. intros a' Ha'.
      destruct (Hk2 a' Ha') as (a & Ha & Hmatch).
      exists a. split; [ apply in_or_app; right; exact Ha | exact Hmatch ].
    + apply andb_true_iff in Heq. destruct Heq as [Hh Hbody].
      apply andb_true_iff in Hclosed. destruct Hclosed as [Hcode Hcode_body].
      destruct (handler_eqb_scope h vm km h' Hh Hcode) as (Hc1 & Hv1 & Hk1).
      destruct (expr_eqb_scope body vm _ body' Hbody Hcode_body)
        as (Hc2 & Hv2 & Hk2).
      split; [ rewrite Hc1, Hc2; reflexivity | ].
      split; [ apply matched_app; assumption | ].
      apply matched_app; [ exact Hk1 | ].
      apply (matched_without km ((k, k') :: km) [k] [k']); [ | exact Hk2 ].
      intros a a'. apply names_match_other_one.
    + unfold combine_names in Heq.
      match type of Heq with
      | context [Nat.eqb ?n ?n'] => destruct (Nat.eqb n n') eqn:Hlength
      end;
        cbv beta iota in Heq; [ | discriminate Heq ].
      apply Nat.eqb_eq in Hlength.
      match type of Heq with
      | context [expr_eqb vm ?m body body'] => remember m as km' eqn:Hkm'
      end.
      assert (Hhandlers :
        ∀ others : list (continuation * continuation_handler),
          forall2b (fun '(_, h) '(_, h') => handler_eqb vm km' h h') handlers
            others
          = true ->
          forallb (fun '(_, h) => handler_code_closed h) handlers = true ->
          forallb (fun '(_, h) => handler_code_closed h) others = true
          ∧ matched vm
              (flat_map (fun '(_, h) => free_variables_of_handler h) handlers)
              (flat_map (fun '(_, h) => free_variables_of_handler h) others)
          ∧ matched km'
              (flat_map (fun '(_, h) => free_continuations_of_handler h)
                 handlers)
              (flat_map (fun '(_, h) => free_continuations_of_handler h)
                 others)).
      { clear Heq Hclosed Hlength Hkm'.
        induction handlers as [ | [j h] handlers IH];
          intros [ | [j' h'] others]; cbn [forall2b forallb flat_map];
          try discriminate.
        - intros _ _. split; [ reflexivity | split; apply matched_nil ].
        - intros Hall Hcode.
          apply andb_true_iff in Hall. destruct Hall as [Hfirst Hrest].
          apply andb_true_iff in Hcode.
          destruct Hcode as [Hcode_first Hcode_rest].
          destruct (handler_eqb_scope h vm km' h' Hfirst Hcode_first)
            as (Hc1 & Hv1 & Hk1).
          destruct (IH others Hrest Hcode_rest) as (Hc2 & Hv2 & Hk2).
          split; [ rewrite Hc1, Hc2; reflexivity | ].
          split; apply matched_app; assumption. }
      apply andb_true_iff in Heq. destruct Heq as [Hgroup Hbody].
      apply andb_true_iff in Hclosed. destruct Hclosed as [Hcode Hcode_body].
      destruct (Hhandlers handlers' Hgroup Hcode) as (Hc1 & Hv1 & Hk1).
      destruct (expr_eqb_scope body vm km' body' Hbody Hcode_body)
        as (Hc2 & Hv2 & Hk2).
      split; [ rewrite Hc1, Hc2; reflexivity | ].
      split; [ apply matched_app; assumption | ].
      apply (matched_without km km' (map fst handlers) (map fst handlers'));
        [ | apply matched_app; assumption ].
      intros a a' Hmatch Hout. rewrite Hkm' in Hmatch.
      exact
        (combine_names_other (map fst handlers) (map fst handlers') km a a'
           Hlength Hmatch Hout).
    + destruct (apply_eqb_matched vm km ap ap' Heq) as [Hv Hk].
      split; [ reflexivity | split; assumption ].
    + destruct (apply_cont_eqb_matched vm km ac ac' Heq) as [Hv Hk].
      split; [ reflexivity | split; assumption ].
    + apply andb_true_iff in Heq. destruct Heq as [Hscrutinee Harms].
      split; [ reflexivity | split ].
      * apply matched_app; [ apply simple_eqb_matched, Hscrutinee | ].
        apply
          (forall2b_matched _
             (fun arm => free_variables_of_apply_cont (snd arm))
             (fun arm => free_variables_of_apply_cont (snd arm)) vm)
          with (2 := Harms).
        intros arm arm' Harm. apply andb_true_iff in Harm.
        exact (proj1 (apply_cont_eqb_matched vm km _ _ (proj2 Harm))).
      * apply
          (forall2b_matched _ (fun arm => apply_cont_continuations (snd arm))
             (fun arm => apply_cont_continuations (snd arm)) km)
          with (2 := Harms).
        intros arm arm' Harm. apply andb_true_iff in Harm.
        exact (proj2 (apply_cont_eqb_matched vm km _ _ (proj2 Harm))).
    + split; [ reflexivity | split; apply matched_nil ].
  - destruct h as [params metadata handler]. intros vm km h' Heq Hclosed.
    destruct h' as [params' metadata' handler'].
    cbn [expr_eqb handler_eqb binding_eqb code_eqb] in Heq.
    scope_in Hclosed. scope.
    destruct (bind_names params params' vm) as [vm' | ] eqn:Hbind;
      [ | discriminate Heq ].
    apply andb_true_iff in Heq. destruct Heq as [_ Hhandler].
    destruct (expr_eqb_scope handler vm' km handler' Hhandler Hclosed)
      as (Hclosed' & Hv & Hk).
    split; [ exact Hclosed' | split; [ | exact Hk ] ].
    apply (matched_without vm vm' params params'); [ | exact Hv ].
    intros a a' Hmatch Hout.
    exact (bind_names_other params params' vm vm' a a' Hbind Hmatch Hout).
  - destruct binding as [id c | id | sym const]; intros vm binding' Heq Hclosed;
      destruct binding' as [id' c' | id' | sym' const'];
      cbn [expr_eqb handler_eqb binding_eqb code_eqb] in Heq;
      try discriminate; scope_in Hclosed; scope.
    + apply andb_true_iff in Heq. destruct Heq as [_ Hcode].
      pose proof (code_eqb_scope c c' Hcode Hclosed) as Hclosed'.
      destruct (code_is_closed_free c' Hclosed') as [Hv Hk].
      split; [ exact Hclosed' | ].
      split; [ rewrite Hv; apply matched_nil | exact Hk ].
    + split; [ reflexivity | split; [ apply matched_nil | reflexivity ] ].
    + apply andb_true_iff in Heq.
      split; [ reflexivity | ].
      split; [ apply static_const_eqb_matched, (proj2 Heq) | reflexivity ].
  - destruct c as [params my_closure ret exn metadata body].
    intros c' Heq Hclosed.
    destruct c' as [params' my_closure' ret' exn' metadata' body'].
    cbn [expr_eqb handler_eqb binding_eqb code_eqb] in Heq.
    scope_in Hclosed. scope.
    destruct (bind_names params params' [(my_closure, my_closure')])
      as [vm | ] eqn:Hbind; [ | discriminate Heq ].
    apply andb_true_iff in Heq. destruct Heq as [_ Hbody].
    apply andb_true_iff in Hclosed. destruct Hclosed as [Hclosed Hcode].
    apply andb_true_iff in Hclosed. destruct Hclosed as [Hno_v Hno_k].
    destruct (expr_eqb_scope body vm _ body' Hbody Hcode)
      as (Hclosed' & Hv & Hk).
    apply andb_true_iff. split; [ apply andb_true_iff; split | exact Hclosed' ].
    + refine (matched_is_empty [] _ _ _ Hno_v).
      apply (matched_without [] vm); [ | exact Hv ].
      intros a a' Hmatch Hout.
      destruct
        (bind_names_other params params' [(my_closure, my_closure')] vm a a'
           Hbind Hmatch)
        as [Hparams Hmatch'].
      { intros Hin. apply Hout. right. exact Hin. }
      apply names_match_other in Hmatch';
        [ | intros ->; apply Hout; left; reflexivity ].
      destruct Hmatch' as [Hne Hmatch']. split; [ | exact Hmatch' ].
      intros [Heq | Hin];
        [ apply Hne; symmetry; exact Heq | exact (Hparams Hin) ].
    + refine (matched_is_empty [] _ _ _ Hno_k).
      apply (matched_without [] [(ret, ret'); (exn, exn')]); [ | exact Hk ].
      intros a a' Hmatch Hout.
      apply names_match_other in Hmatch;
        [ | intros ->; apply Hout; left; reflexivity ].
      destruct Hmatch as [Hne Hmatch].
      apply names_match_other in Hmatch;
        [ | intros ->; apply Hout; right; left; reflexivity ].
      destruct Hmatch as [Hne' Hmatch]. split; [ | exact Hmatch ].
      intros [Heq | [Heq | []]]; [ apply Hne | apply Hne' ]; symmetry;
        exact Heq.
Qed.

(* So the comparison accepts a term only against one that keeps its
   scope. *)
Lemma expr_eqb_scoped (e e' : expr) :
  expr_eqb [] [] e e' = true -> scoped e e'.
Proof.
  intros Hsame Hclosed.
  destruct (expr_eqb_scope e [] [] e' Hsame Hclosed) as (Hclosed' & Hv & Hk).
  split; [ exact Hclosed' | ].
  split; intros name Hname;
    [ destruct (Hv name Hname) as (name' & Hin & Hmatch)
    | destruct (Hk name Hname) as (name' & Hin & Hmatch) ];
    cbn [names_match] in Hmatch; apply Pos.eqb_eq in Hmatch; subst name';
    exact Hin.
Qed.
