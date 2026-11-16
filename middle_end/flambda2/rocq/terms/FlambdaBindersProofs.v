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

(* Lemmas about the free names and the checks of FlambdaBindersSpec.v,
   and the proofs of its statements. Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaBindersSpec.
Import ListNotations.

(* [all_among] is the test, here. FlambdaMapProofs.v proves its statement
   under the same name; that is [all_among_iff]. *)
#[local] Abbreviation all_among := FlambdaMapSpec.all_among.

Lemma all_among_iff (names allowed : list positive) :
  all_among names allowed = true
  <-> (∀ name : positive,
        among name names = true -> among name allowed = true).
Proof. exact (FlambdaMapProofs.all_among names allowed). Qed.

(* ======================================================================== *)
(* Sets of names                                                            *)
(* ======================================================================== *)

Lemma all_among_app (names names' allowed : list positive) :
  all_among (names ++ names') allowed
  = all_among names allowed && all_among names' allowed.
Proof. apply forallb_app. Qed.

(* What is left of [names] without [bound] is allowed when [names] is
   allowed or bound. *)
Lemma all_among_without (bound names allowed : list positive) :
  all_among (without bound names) allowed
  = all_among names (bound ++ allowed).
Proof.
  induction names as [ | name names IH]; simpl; [ reflexivity | ].
  rewrite among_app.
  destruct (among name bound); simpl; rewrite IH; reflexivity.
Qed.

Lemma all_among_flat_map {A : Type} (f : A -> list positive) (l : list A)
    (allowed : list positive) :
  all_among (flat_map f l) allowed = true
  <-> ∀ a : A, a ∈ l -> all_among (f a) allowed = true.
Proof.
  induction l as [ | a l IH]; simpl.
  - split; [ intros _ a [] | reflexivity ].
  - rewrite all_among_app, andb_true_iff, IH. split.
    + intros [Ha Hl] a' [<- | Hin]; [ exact Ha | exact (Hl a' Hin) ].
    + intros Hall. split; [ apply Hall; left; reflexivity | ].
      intros a' Hin. apply Hall. right. exact Hin.
Qed.

Lemma all_among_mono (names allowed allowed' : list positive) :
  (∀ name : positive,
    among name allowed = true -> among name allowed' = true) ->
  all_among names allowed = true -> all_among names allowed' = true.
Proof.
  intros Hallowed Hnames.
  pose proof (proj1 (all_among_iff names allowed) Hnames) as Hall.
  apply all_among_iff. intros name Hname.
  exact (Hallowed name (Hall name Hname)).
Qed.

(* A name of a list that is allowed is allowed. *)
Lemma all_among_among (names allowed : list positive) (name : positive) :
  all_among names allowed = true -> among name names = true ->
  among name allowed = true.
Proof.
  intros Hnames. exact (proj1 (all_among_iff names allowed) Hnames name).
Qed.

(* Nothing is left of [names] without [bound] when all of them are
   bound. *)
Lemma without_empty (bound names : list positive) :
  Keyed.is_empty (without bound names) = all_among names bound.
Proof.
  induction names as [ | name names IH]; simpl; [ reflexivity | ].
  destruct (among name bound); simpl; [ exact IH | reflexivity ].
Qed.

Lemma is_empty_nil {A : Type} (l : list A) :
  Keyed.is_empty l = true -> l = [].
Proof. destruct l; [ reflexivity | discriminate ]. Qed.

Lemma among_rev (name : positive) (names : list positive) :
  among name (rev names) = among name names.
Proof.
  destruct (among name names) eqn:Hname.
  - apply among_In, in_rev. rewrite rev_involutive. apply among_In, Hname.
  - destruct (among name (rev names)) eqn:Hrev; [ | reflexivity ].
    apply among_In, in_rev, among_In in Hrev. congruence.
Qed.

(* ======================================================================== *)
(* The free names of each construct                                         *)
(* ======================================================================== *)

Lemma names_among_let (vs : list variable) (ks : list continuation)
    (x : variable) (kind : flambda_kind) (defining_expr : named)
    (body : expr) :
  names_among vs ks (Let x kind defining_expr body) = true ->
  all_among (free_variables_of_named defining_expr) vs = true
  ∧ names_among (x :: vs) ks body = true.
Proof.
  unfold names_among. simpl.
  rewrite all_among_app, all_among_without, !andb_true_iff. simpl. tauto.
Qed.

Lemma names_among_let_static (vs : list variable) (ks : list continuation)
    (bindings : list static_binding) (body : expr) :
  names_among vs ks (Let_static bindings body) = true ->
  names_among vs ks body = true.
Proof.
  unfold names_among. simpl. rewrite !all_among_app, !andb_true_iff. tauto.
Qed.

Lemma names_among_let_cont (vs : list variable) (ks : list continuation)
    (k : continuation) (h : continuation_handler) (body : expr) :
  names_among vs ks (Let_cont_non_recursive k h body) = true ->
  handler_names_among vs ks h = true ∧ names_among vs (k :: ks) body = true.
Proof.
  unfold names_among, handler_names_among. simpl.
  rewrite !all_among_app, all_among_without, !andb_true_iff. simpl. tauto.
Qed.

Lemma names_among_let_cont_recursive (vs : list variable)
    (ks : list continuation)
    (handlers : list (continuation * continuation_handler)) (body : expr) :
  names_among vs ks (Let_cont_recursive handlers body) = true ->
  Forall
    (fun binding =>
       handler_names_among vs (map fst handlers ++ ks) (snd binding) = true)
    handlers
  ∧ names_among vs (map fst handlers ++ ks) body = true.
Proof.
  unfold names_among, handler_names_among. simpl.
  rewrite all_among_without, !all_among_app, !andb_true_iff.
  rewrite !all_among_flat_map.
  intros [[Hvars Hbody_vars] [Hconts Hbody_conts]].
  split; [ | split; assumption ].
  apply Forall_forall. intros [k h] Hin.
  pose proof (Hvars (k, h) Hin) as Hh_vars.
  pose proof (Hconts (k, h) Hin) as Hh_conts. simpl in *.
  apply andb_true_iff. split; [ exact Hh_vars | exact Hh_conts ].
Qed.

Lemma handler_names_among_handler (vs : list variable)
    (ks : list continuation) (params : list variable)
    (metadata : handler_metadata) (handler : expr) :
  handler_names_among vs ks (Handler params metadata handler)
  = names_among (params ++ vs) ks handler.
Proof.
  unfold names_among, handler_names_among. simpl.
  rewrite all_among_without. reflexivity.
Qed.

Lemma names_among_apply (vs : list variable) (ks : list continuation)
    (ap : apply_expr) :
  names_among vs ks (Apply ap) = true ->
  among (ap_continuation ap) ks = true
  ∧ among (ap_exn_continuation ap) ks = true.
Proof.
  unfold names_among. simpl. rewrite !andb_true_iff. tauto.
Qed.

Lemma names_among_apply_cont (vs : list variable) (ks : list continuation)
    (ac : apply_cont) :
  names_among vs ks (Apply_cont ac) = true ->
  among (ac_continuation ac) ks = true.
Proof.
  unfold names_among. simpl. rewrite !andb_true_iff. tauto.
Qed.

(* An arm of a [Switch] uses names of the [Switch]. *)
Lemma names_among_switch_arm (vs : list variable) (ks : list continuation)
    (scrutinee : simple) (arms : list (Z * apply_cont)) (i : Z)
    (ac : apply_cont) :
  names_among vs ks (Switch scrutinee arms) = true ->
  (i, ac) ∈ arms ->
  names_among vs ks (Apply_cont ac) = true.
Proof.
  unfold names_among.
  cbn [free_variables_of_expr free_continuations_of_expr].
  rewrite all_among_app, !andb_true_iff.
  intros [[_ Hvars] Hconts] Hin. split.
  - exact (proj1 (all_among_flat_map _ arms vs) Hvars (i, ac) Hin).
  - exact (proj1 (all_among_flat_map _ arms ks) Hconts (i, ac) Hin).
Qed.

(* More names may be allowed. *)
Lemma names_among_mono (vs vs' : list variable) (ks ks' : list continuation)
    (e : expr) :
  (∀ x : variable, among x vs = true -> among x vs' = true) ->
  (∀ k : continuation, among k ks = true -> among k ks' = true) ->
  names_among vs ks e = true -> names_among vs' ks' e = true.
Proof.
  intros Hvs Hks. unfold names_among. rewrite !andb_true_iff.
  intros [Hvars Hconts].
  split; eapply all_among_mono; eassumption.
Qed.

Lemma handler_names_among_mono (vs vs' : list variable)
    (ks ks' : list continuation) (h : continuation_handler) :
  (∀ x : variable, among x vs = true -> among x vs' = true) ->
  (∀ k : continuation, among k ks = true -> among k ks' = true) ->
  handler_names_among vs ks h = true -> handler_names_among vs' ks' h = true.
Proof.
  intros Hvs Hks. unfold handler_names_among. rewrite !andb_true_iff.
  intros [Hvars Hconts].
  split; eapply all_among_mono; eassumption.
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Theorem closed_code : closed_code_statement.
Proof.
  intros [params my_closure ret exn body]. unfold names_among. simpl.
  rewrite !without_empty, !andb_true_iff. tauto.
Qed.

Lemma code_is_closed_no_free_names (c : code) :
  code_is_closed c = true ->
  free_variables_of_code c = [] ∧ free_continuations_of_code c = [].
Proof.
  destruct c as [params my_closure ret exn metadata body]. simpl.
  rewrite !andb_true_iff. intros [[Hvars Hconts] _].
  split; apply is_empty_nil; assumption.
Qed.

Theorem closed_code_free_names : closed_code_free_names_statement.
Proof.
  intros bindings. induction bindings as [ | binding bindings IH]; simpl.
  - intros _. split; reflexivity.
  - rewrite andb_true_iff. intros [Hbinding Hbindings].
    destruct (IH Hbindings) as [-> ->].
    destruct binding as [id c | id | s const]; simpl;
      [ | split; reflexivity | split; reflexivity ].
    destruct (code_is_closed_no_free_names c Hbinding) as [-> ->].
    split; reflexivity.
Qed.

Theorem unit_free_names : unit_free_names_statement.
Proof.
  intros unit_return unit_exn e. unfold names_in_scope, names_among.
  rewrite andb_true_iff, !all_among_iff. split.
  - intros [Hvars Hconts]. split.
    + intros x. destruct (among x (free_variables_of_expr e)) eqn:Hx;
        [ | reflexivity ].
      apply Hvars in Hx. discriminate.
    + intros k Hk. apply Hconts, among_In in Hk.
      destruct Hk as [<- | [<- | []]]; [ left | right ]; reflexivity.
  - intros [Hvars Hconts]. split.
    + intros x Hx. rewrite Hvars in Hx. discriminate.
    + intros k Hk. apply among_In.
      destruct (Hconts k Hk) as [-> | ->];
        [ left | right; left ]; reflexivity.
Qed.
