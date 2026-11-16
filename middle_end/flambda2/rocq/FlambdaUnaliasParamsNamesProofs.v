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

(* FlambdaUnaliasParamsSpec.unalias_params commutes with renaming and
   keeps scope: the lemmas from which
   FlambdaValidatorEquivarianceProofs.v and FlambdaStaticsScopeProofs.v
   prove the two statements. There is no statement here, and nothing
   here needs review.

   The pass compares names by equality only and makes none, so each of
   its functions gives, on a renamed term, the renamed result. For the
   scope: an application that loses an argument loses free variables
   and no continuation; the [Let] that the handler gains mentions the
   variable that the applications gave, which is free in the body of
   the [Let_cont] where an application gives it and no binder on the
   way binds it; and the parameter that the handler loses is bound by
   that [Let]. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaNominalSpec FlambdaNominalProofs
  FlambdaEquivarianceSpec FlambdaEquivarianceProofs FlambdaBindersSpec
  FlambdaValidatorEquivarianceSpec FlambdaScopedProofs
  FlambdaUnaliasParamsSpec FlambdaUnaliasParamsProofs
  FlambdaTermInductionProofs FlambdaEquivarianceTactic
  FlambdaScopedInductionProofs FlambdaScopeTactic
  FlambdaSpineWalkSpec FlambdaSpineWalkNamesProofs.
Import ListNotations.

(* ======================================================================== *)
(* The pass commutes with renaming                                          *)
(* ======================================================================== *)

(* The law of the pass is proved from the definitions alone
   ([equivariance], of FlambdaEquivarianceTactic.v): the tactic states
   and proves, as it goes, the laws of the twelve functions that the
   pass is made of. What is stated here besides is what the tactic
   cannot read from a type or what another file uses by name: the
   laws of two functions of lists, and six laws of functions of this
   pass that the proofs of other passes rewrite with, each stated from
   the type of its function ([equivariance_of]; [Check] shows the
   statement). *)

Lemma drop_nth_map {A B : Type} (f : A -> B) (i : nat) (l : list A) :
  drop_nth i (map f l) = map f (drop_nth i l).
Proof.
  unfold drop_nth. rewrite map_app, firstn_map, skipn_map. reflexivity.
Qed.
#[export] Hint Rewrite @drop_nth_map : equivariance.

(* [first_answer] under [map], and of two functions that agree on the
   elements of the list up to what is done to their answers: what the
   tactic needs to go under it. *)
Lemma first_answer_of_map {A A' B : Type} (f : A' -> option B) (g : A -> A')
    (l : list A) :
  first_answer f (map g l) = first_answer (fun a => f (g a)) l.
Proof.
  induction l as [ | a l IH]; [ reflexivity | ]. cbn. rewrite IH.
  reflexivity.
Qed.
#[export] Hint Rewrite @first_answer_of_map : equivariance.

Lemma first_answer_ext_among {A B B' : Type} (f' : A -> option B')
    (f : A -> option B) (h : B -> B') (l : list A) :
  (∀ a : A, In a l -> f' a = option_map h (f a)) ->
  first_answer f' l = option_map h (first_answer f l).
Proof.
  induction l as [ | a l IH]; intros Hf; [ reflexivity | ]. cbn.
  rewrite (Hf a (or_introl eq_refl)).
  destruct (f a); [ reflexivity | ]. cbn. apply IH.
  intros b Hb. apply Hf. right. exact Hb.
Qed.

#[local] Ltac eqv_extra ::=
  match goal with
  | |- context [first_answer ?f' ?l] =>
    match goal with
    | |- context [first_answer ?f l] =>
      tryif constr_eq f f' then fail
      else (eqv_no_renaming f;
            erewrite (first_answer_ext_among f' f _ l);
            [ | eqv_enter l; solve [ eqv_core ] ])
    end
  end.

Lemma trap_action_handler_renamed : equivariance_of trap_action_handler.
Proof. equivariance. Qed.
#[export] Hint Rewrite trap_action_handler_renamed : equivariance.

Lemma drop_arg_of_equivariant : equivariance_of drop_arg_of.
Proof. equivariance. Qed.
#[export] Hint Rewrite drop_arg_of_equivariant : equivariance.

Lemma all_give_var_equivariant : equivariance_of all_give_var.
Proof. equivariance. Qed.
#[export] Hint Rewrite all_give_var_equivariant : equivariance.

(* What [first_given_var] gives, renamed: the proof of another pass
   names it. *)
Definition rename_given (r : renaming) (given : option (option variable))
    : option (option variable) :=
  option_map (option_map (rename_variable r)) given.

Lemma first_given_var_equivariant : equivariance_of first_given_var.
Proof. equivariance. Qed.
#[export] Hint Rewrite first_given_var_equivariant : equivariance.

Lemma drop_arg_equivariant : equivariance_of drop_arg.
Proof. equivariance. Qed.
#[export] Hint Rewrite drop_arg_equivariant : equivariance.

Lemma unalias_param_equivariant : equivariance_of unalias_param.
Proof. equivariance. Qed.
#[export] Hint Rewrite unalias_param_equivariant : equivariance.

(* The result of [unalias_param], renamed: for the law of a pass that
   uses that function and is stated by hand. *)
Definition rename_unaliased (r : renaming)
    (result : option (continuation_handler * expr))
    : option (continuation_handler * expr) :=
  option_map (fun '(h, body) => (rename_handler r h, rename_expr r body))
    result.

Lemma unalias_node_equivariant : rewrite_equivariant unalias_node.
Proof. equivariance. Qed.

Lemma unalias_params_renamed : rewrite_equivariant unalias_params.
Proof. exact (walk_spine_equivariant _ unalias_node_equivariant). Qed.

(* ======================================================================== *)
(* The pass keeps scope                                                     *)
(* ======================================================================== *)

(* Each law is proved by the induction of FlambdaTermInductionProofs.v.
   Where a function only goes through a form of term, the tactic of
   FlambdaScopedInductionProofs.v closes the case. Where it tests its
   arguments or puts something in the term, [tests] and [names] of
   FlambdaScopeTactic.v take the tests apart and close the inclusions
   between lists of names. *)

Lemma in_nth_or_drop_nth {A : Type} (i : nat) :
  ∀ (l : list A) (p a : A),
    nth_error l i = Some p -> In a l -> a = p ∨ In a (drop_nth i l).
Proof.
  induction i as [ | i IH]; intros [ | b l] p a; cbn [nth_error];
    try discriminate.
  - intros [= ->] [-> | Hin]; [ left; reflexivity | right; exact Hin ].
  - intros Hnth [-> | Hin]; [ right; left; reflexivity | ].
    destruct (IH l p a Hnth Hin) as [-> | Hin'];
      [ left; reflexivity | right; right; exact Hin' ].
Qed.

Lemma drop_arg_of_free_variables (k : continuation) (i : nat)
    (ac : apply_cont) :
  incl (free_variables_of_apply_cont (drop_arg_of k i ac))
    (free_variables_of_apply_cont ac).
Proof.
  unfold drop_arg_of. goal_tests; [ | apply incl_refl ].
  intros a Ha. apply in_flat_map in Ha as (s & Hs & Ha).
  apply in_flat_map. exists s.
  split; [ exact (in_drop_nth i _ s Hs) | exact Ha ].
Qed.

Lemma drop_arg_of_continuations (k : continuation) (i : nat)
    (ac : apply_cont) :
  apply_cont_continuations (drop_arg_of k i ac) = apply_cont_continuations ac.
Proof. unfold drop_arg_of. goal_tests; reflexivity. Qed.

Lemma drop_arg_scoped (k : continuation) (i : nat) (e : expr) :
  scoped e (drop_arg k i e).
Proof.
  induction e using expr_deep_ind; cbn [drop_arg drop_arg_in_handler];
    scoped_congruence.
  - apply scoped_apply_cont; [ apply drop_arg_of_free_variables | ].
    rewrite drop_arg_of_continuations. apply incl_refl.
  - apply scoped_switch_map; intros ac';
      [ apply drop_arg_of_free_variables | ].
    rewrite drop_arg_of_continuations. apply incl_refl.
Qed.

(* The variable that the applications of [k] give is free in a term
   that applies [k]. *)
Lemma given_by_free (k : continuation) (i : nat) (x : variable) (ok : bool)
    (ac : apply_cont) (given : option variable) :
  given_var_by k i ac = Some given ->
  apply_cont_gives_var k i x ok ac = true ->
  ok = true ∧ In x (free_variables_of_apply_cont ac).
Proof.
  unfold given_var_by, apply_cont_gives_var. intros Hgiven Hgives.
  tests; (split; [ reflexivity | ]); apply in_flat_map; eexists;
    (split; [ eapply nth_error_In; eassumption | left; reflexivity ]).
Qed.

Definition gives_free (k : continuation) (i : nat) (x : variable) (e : expr)
    : Prop :=
  ∀ (ok : bool) (given : option variable),
    first_given_var k i e = Some given ->
    all_give_var k i x ok e = true ->
    ok = true ∧ In x (free_variables_of_expr e).

Lemma first_answer_In {A B : Type} (f : A -> option B) (l : list A) (b : B) :
  first_answer f l = Some b -> ∃ a : A, In a l ∧ f a = Some b.
Proof.
  induction l as [ | a l IH]; cbn [first_answer]; intros Hsome; tests.
  - exists a. split; [ left; reflexivity | assumption ].
  - destruct (IH Hsome) as (a' & Hin & Ha').
    exists a'. split; [ right; exact Hin | exact Ha' ].
Qed.

(* What is known of a term whose applications give [x], from what the
   induction knows of it, and of an application that gives it. *)
Ltac given_free_known :=
  repeat match goal with
  | H : first_answer _ _ = Some _ |- _ =>
    apply first_answer_In in H; destruct H as (? & ? & ?); list_tests; tests
  | h : continuation_handler |- _ =>
    destruct h;
    cbn [handler_first_given_var handler_all_give_var handler_body snd] in *
  | IH : gives_free ?k ?i ?x ?e,
    H1 : first_given_var ?k ?i ?e = Some _,
    H2 : all_give_var ?k ?i ?x _ ?e = true |- _ =>
    destruct (IH _ _ H1 H2) as [? ?]; clear IH; tests
  | H1 : given_var_by ?k ?i ?ac = Some _,
    H2 : apply_cont_gives_var ?k ?i ?x _ ?ac = true |- _ =>
    destruct (given_by_free _ _ _ _ _ _ H1 H2) as [? ?]; clear H2; tests
  end.

Lemma given_free (k : continuation) (i : nat) (x : variable) (e : expr) :
  gives_free k i x e.
Proof.
  induction e using expr_deep_ind; intros ok given;
    cbn [first_given_var all_give_var free_variables_of_expr
         handler_first_given_var handler_all_give_var
         free_variables_of_handler];
    intros Hfirst Hgive; tests; given_free_known.
  all: split; [ reflexivity | names ].
Qed.

Lemma unalias_param_scoped (i : nat) (k : continuation)
    (h : continuation_handler) (body : expr) (h' : continuation_handler)
    (body' : expr) :
  unalias_param i k h body = Some (h', body') ->
  scoped (Let_cont_non_recursive k h body)
    (Let_cont_non_recursive k h' body').
Proof.
  unfold unalias_param. intros Hsome Hclosed. tests.
  edestruct (given_free k i) as [_ Hx]; [ eassumption | eassumption | ].
  scope_all. tests.
  edestruct (drop_arg_scoped k i body) as (-> & Hv & Hk); [ eassumption | ].
  match goal with
  | H : nth_error params i = Some _ |- _ =>
    pose proof (fun a => in_nth_or_drop_nth i params _ a H)
  end.
  split; [ rewrite andb_true_r; assumption | split; names ].
Qed.

Lemma unalias_from_scoped (i : nat) :
  ∀ (k : continuation) (h : continuation_handler) (body : expr),
    scoped (Let_cont_non_recursive k h body)
      (let '(h', body') := unalias_from i k h body in
       Let_cont_non_recursive k h' body').
Proof.
  induction i as [ | i IH]; intros k h body; cbn [unalias_from];
    goal_tests; eauto using scoped_refl, scoped_trans, unalias_param_scoped.
Qed.

Lemma unalias_node_scoped (e : expr) : scoped e (unalias_node e).
Proof.
  unfold unalias_node. goal_tests;
    auto using scoped_refl, unalias_from_scoped.
Qed.

Lemma unalias_params_scoped (e : expr) : scoped e (unalias_params e).
Proof. exact (walk_spine_scoped _ unalias_node_scoped e). Qed.

Lemma unalias_params_scope : keeps_scope unalias_params.
Proof. apply keeps_scope_intro, unalias_params_scoped. Qed.
