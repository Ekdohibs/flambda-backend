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


(* Tactics for the laws of scope.

   A law of scope says that the free names of what a rewrite gives are
   free in the term. Its proof has three parts: the rewrite going
   through a form of term (FlambdaScopedInductionProofs.v, and the
   lemmas at the end of this file); what is known when the tests of
   the rewrite have passed; and inclusions between lists of names read
   as sets. The tactics here do the last two.

   - [tests] takes apart what is known when a function that tests its
     arguments has answered: a [match] or an [if] of a hypothesis goes
     by cases, the cases that answer [None] or [false] are closed, a
     conjunction of tests is split, and a test of equality of names
     becomes an equality. [goal_tests] goes by the cases of a [match]
     of the goal, and [list_tests] takes, from a test of every element
     of a list, the test of one.
   - [names] closes a goal about lists of names read as sets: [In],
     [incl], [++], [FlambdaMapSpec.without], the names of an element
     of a list among those of the list, with what is known of those
     forms and of tests on names. It goes by the cases of what is
     known and ends with [tauto] or [congruence]; it looks for no
     lemma, and closes the goal or fails. [names_cases] goes first by
     the cases of the equalities of names in the goal, where the goal
     needs that two names are equal or are not.
   - [scope_all] computes the closed code and the free names of a
     term whose head is known.

   After them, more of the congruence part. For a rewrite that puts
   nothing new: a handler whose new body may mention its parameters,
   and the handlers of a recursive [Let_cont] each rewritten by a
   function of its name. For a rewrite that carries names, which it
   may put in the term (FlambdaScopedProofs.scoped_under): the same
   for handlers, and the two forms of term that hold applications of
   continuations.

   There is no statement here, and nothing here needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaBindersSpec FlambdaScopedProofs
  FlambdaTermInductionProofs FlambdaScopedInductionProofs.
Import ListNotations.

(* ======================================================================== *)
(* The tests of a function that has answered                                *)
(* ======================================================================== *)

Ltac tests_step :=
  match goal with
  | H : None = Some _ |- _ => discriminate H
  | H : Some _ = None |- _ => discriminate H
  | H : false = true |- _ => discriminate H
  | H : true = false |- _ => discriminate H
  | H : true = true |- _ => clear H
  | H : Some _ = Some _ |- _ => injection H as H
  | H : (_, _) = (_, _) |- _ => injection H as ? ?
  | H : ?x = ?y |- _ => is_var x; subst x
  | H : ?y = ?x |- _ => is_var x; subst x
  | H : andb _ _ = true |- _ => apply andb_true_iff in H; destruct H
  | H : negb _ = true |- _ => apply negb_true_iff in H
  | H : Pos.eqb _ _ = true |- _ => apply Pos.eqb_eq in H
  | H : Pos.eqb _ _ = false |- _ => apply Pos.eqb_neq in H
  | H : context [match ?c with _ => _ end] |- _ =>
    first [ is_var c; destruct c | destruct c eqn:? ]
  end.

Ltac tests := repeat tests_step.

(* The same for a [match] of the goal; a pair that a function gives is
   left as it is. *)
Ltac goal_tests :=
  repeat match goal with
  | |- context [match ?c with _ => _ end] =>
    first
      [ is_var c; destruct c
      | lazymatch type of c with
        | prod _ _ => fail
        | _ => destruct c eqn:?
        end ]
  end.

(* What a test of every element of a list says of one element. *)
Ltac list_tests :=
  repeat match goal with
  | H : forallb _ ?l = true, Hin : In _ ?l |- _ =>
    rewrite forallb_forall in H; specialize (H _ Hin); cbn beta iota in H
  | H : Forall _ ?l, Hin : In _ ?l |- _ =>
    rewrite Forall_forall in H; specialize (H _ Hin); cbn beta iota in H
  end.

(* ======================================================================== *)
(* Lists of names as sets                                                   *)
(* ======================================================================== *)

Lemma in_first (a b : positive) (l : list positive) :
  In a (b :: l) <-> b = a ∨ In a l.
Proof. cbn. tauto. Qed.

Lemma in_none (a : positive) : In a [] <-> False.
Proof. cbn. tauto. Qed.

Lemma among_not_In (name : positive) (names : list positive) :
  among name names = false <-> ~ In name names.
Proof.
  rewrite <- among_In. destruct (among name names); split; congruence.
Qed.

Ltac names_hyps :=
  repeat match goal with
  | H : among _ _ = true |- _ => apply among_In in H
  | H : among _ _ = false |- _ => apply among_not_In in H
  | H : Pos.eqb _ _ = true |- _ => apply Pos.eqb_eq in H
  | H : Pos.eqb _ _ = false |- _ => apply Pos.eqb_neq in H
  | H : _ = false |- _ => apply not_true_iff_false in H
  | H : incl _ _ |- _ => unfold incl in H
  end.

(* The closed code and the free names of a term whose head is known,
   in the goal and in what is known. *)
Ltac scope_all :=
  cbn [code_closed handler_code_closed binding_code_closed code_is_closed
       free_variables_of_expr free_variables_of_handler
       free_variables_of_binding free_variables_of_code
       free_continuations_of_expr free_continuations_of_handler
       free_continuations_of_binding free_continuations_of_code
       free_variables_of_named free_variables_of_simple
       forallb flat_map map fst snd] in *.

Lemma incl_flat_map_element {A : Type} (f : A -> list positive) (a : A)
    (l : list A) :
  In a l -> incl (f a) (flat_map f l).
Proof. intros Hin b Hb. apply in_flat_map. exists a. tauto. Qed.

(* The names of an element of a list are among those of the list. *)
Ltac names_elements :=
  repeat match goal with
  | Hin : In ?a ?l |- context [flat_map ?f ?l] =>
    lazymatch goal with
    | _ : incl (f a) (flat_map f l) |- _ => fail
    | _ => pose proof (incl_flat_map_element f a l Hin)
    end
  end.

(* The element of the goal put into what is known of every element. *)
Ltac names_at a :=
  repeat match goal with
  | H : forall x : positive, _ |- _ => specialize (H a)
  | H : forall x : variable, _ |- _ => specialize (H a)
  | H : forall x : continuation, _ |- _ => specialize (H a)
  end.

(* What is known and the goal, as propositions about one name. *)
Ltac names_open :=
  names_elements; names_hyps; unfold incl;
  repeat match goal with
  | |- forall a, In a _ -> _ =>
    let a := fresh "a" in intros a ?; names_at a
  | |- _ -> _ => intros ?
  end;
  try match goal with |- In ?a _ => names_at a end;
  cbn beta iota in *; scope_all;
  repeat rewrite ?in_app_iff, ?in_without, ?in_first, ?in_none in *.

Ltac names_close := intuition (subst; solve [ tauto | congruence ]).

Ltac names := names_open; names_close.

(* The same, by the cases of the equalities of names in the goal. *)
Ltac names_cases :=
  names_open;
  repeat match goal with
  | |- context [?b = ?a] =>
    lazymatch goal with
    | _ : b = a |- _ => fail
    | _ : b <> a |- _ => fail
    | _ => destruct (Pos.eq_dec b a)
    end
  end;
  names_close.

(* ======================================================================== *)
(* More of the congruence part                                              *)
(* ======================================================================== *)

Lemma scoped_handler_under_params (params : list variable)
    (metadata metadata' : handler_metadata) (handler handler' : expr) :
  scoped_under params handler handler' ->
  scoped_handler (Handler params metadata handler)
    (Handler params metadata' handler').
Proof.
  intros Hunder Hclosed. scope_all.
  destruct (Hunder Hclosed) as (? & ? & ?).
  split; [ assumption | split; names ].
Qed.

Lemma scoped_scoped_under (names : list variable) (e e' : expr) :
  scoped e e' -> scoped_under names e e'.
Proof.
  intros Hscoped Hclosed. destruct (Hscoped Hclosed) as (? & ? & ?).
  split; [ assumption | split; names ].
Qed.

Lemma scoped_handlers_map_keyed
    (g : continuation -> continuation_handler -> continuation_handler)
    (handlers : list (continuation * continuation_handler)) :
  Forall (fun kh => scoped_handler (snd kh) (g (fst kh) (snd kh))) handlers ->
  scoped_handlers handlers (map (fun '(k, h) => (k, g k h)) handlers).
Proof.
  intros Hall. induction Hall as [ | [k h] handlers Hhead _ IH]; cbn [map];
    [ apply scoped_handlers_nil | apply scoped_handlers_cons; assumption ].
Qed.

Lemma map_fst_keyed {A B : Type} (g : A -> B -> B) (l : list (A * B)) :
  map fst (map (fun '(k, h) => (k, g k h)) l) = map fst l.
Proof. rewrite map_map. apply map_ext. intros [k h]. reflexivity. Qed.

(* ======================================================================== *)
(* The congruence part for a rewrite that carries names                     *)
(* ======================================================================== *)

Lemma scoped_under_more (names names' : list variable) (e e' : expr) :
  incl names names' -> scoped_under names e e' -> scoped_under names' e e'.
Proof.
  intros Hnames Hunder Hclosed. destruct (Hunder Hclosed) as (? & ? & ?).
  split; [ assumption | split; names ].
Qed.

Definition scoped_handler_under (names : list variable)
    (h h' : continuation_handler) : Prop :=
  handler_code_closed h = true ->
  handler_code_closed h' = true
  ∧ incl (free_variables_of_handler h') (free_variables_of_handler h ++ names)
  ∧ incl (free_continuations_of_handler h')
      (free_continuations_of_handler h).

Definition scoped_handlers_under (names : list variable)
    (handlers handlers' : list (continuation * continuation_handler))
    : Prop :=
  forallb (fun '(_, h) => handler_code_closed h) handlers = true ->
  forallb (fun '(_, h) => handler_code_closed h) handlers' = true
  ∧ incl (flat_map (fun '(_, h) => free_variables_of_handler h) handlers')
      (flat_map (fun '(_, h) => free_variables_of_handler h) handlers
       ++ names)
  ∧ incl
      (flat_map (fun '(_, h) => free_continuations_of_handler h) handlers')
      (flat_map (fun '(_, h) => free_continuations_of_handler h) handlers).

Lemma scoped_handlers_under_map_keyed (names : list variable)
    (g : continuation -> continuation_handler -> continuation_handler)
    (handlers : list (continuation * continuation_handler)) :
  Forall
    (fun kh => scoped_handler_under names (snd kh) (g (fst kh) (snd kh)))
    handlers ->
  scoped_handlers_under names handlers
    (map (fun '(k, h) => (k, g k h)) handlers).
Proof.
  intros Hall.
  induction Hall as [ | [k h] handlers Hhead _ IH]; intros Hclosed;
    scope_all; tests; [ split; [ reflexivity | split; names ] | ].
  edestruct Hhead as (-> & ? & ?); [ eassumption | ].
  edestruct IH as (-> & ? & ?); [ eassumption | ].
  split; [ reflexivity | split; names ].
Qed.

Lemma scoped_under_let_cont_handler (names : list variable)
    (k : continuation) (h h' : continuation_handler) (body body' : expr) :
  scoped_handler_under names h h' ->
  scoped_under names body body' ->
  scoped_under names (Let_cont_non_recursive k h body)
    (Let_cont_non_recursive k h' body').
Proof.
  intros Hh Hbody Hclosed. scope_all. tests.
  edestruct Hh as (-> & ? & ?); [ eassumption | ].
  edestruct Hbody as (-> & ? & ?); [ eassumption | ].
  split; [ reflexivity | split; names ].
Qed.

Lemma scoped_under_let_cont_recursive_handlers (names : list variable)
    (handlers handlers' : list (continuation * continuation_handler))
    (body body' : expr) :
  map fst handlers' = map fst handlers ->
  scoped_handlers_under names handlers handlers' ->
  scoped_under names body body' ->
  scoped_under names (Let_cont_recursive handlers body)
    (Let_cont_recursive handlers' body').
Proof.
  intros Hnames Hhandlers Hbody Hclosed. scope_all. tests.
  edestruct Hhandlers as (-> & ? & ?); [ eassumption | ].
  edestruct Hbody as (-> & ? & ?); [ eassumption | ].
  rewrite Hnames. split; [ reflexivity | split; names ].
Qed.

Lemma scoped_under_apply_cont (names : list variable) (ac ac' : apply_cont) :
  incl (free_variables_of_apply_cont ac')
    (free_variables_of_apply_cont ac ++ names) ->
  incl (apply_cont_continuations ac') (apply_cont_continuations ac) ->
  scoped_under names (Apply_cont ac) (Apply_cont ac').
Proof. intros Hv Hk _. scope_all. split; [ reflexivity | split; names ]. Qed.

Lemma scoped_under_switch_map (names : list variable)
    (g : Z -> apply_cont -> apply_cont) (scrutinee scrutinee' : simple)
    (arms : list (Z * apply_cont)) :
  incl (free_variables_of_simple scrutinee')
    (free_variables_of_simple scrutinee ++ names) ->
  (∀ d ac,
     incl (free_variables_of_apply_cont (g d ac))
       (free_variables_of_simple scrutinee
        ++ free_variables_of_apply_cont ac ++ names)) ->
  (∀ d ac,
     incl (apply_cont_continuations (g d ac)) (apply_cont_continuations ac)) ->
  scoped_under names (Switch scrutinee arms)
    (Switch scrutinee' (map (fun '(d, ac) => (d, g d ac)) arms)).
Proof.
  intros Hs Hv Hk _. scope_all. split; [ reflexivity | ].
  induction arms as [ | [d ac] arms [IHv IHk]]; scope_all;
    [ split; names | ].
  specialize (Hv d ac). specialize (Hk d ac). split; names.
Qed.

(* The names of the elements of a list each rewritten. *)
Lemma incl_flat_map_map {A : Type} (f : A -> list positive) (g : A -> A)
    (l : list A) :
  (∀ a : A, incl (f (g a)) (f a)) ->
  incl (flat_map f (map g l)) (flat_map f l).
Proof.
  intros Hg. induction l as [ | a l IH]; cbn [map flat_map];
    [ apply incl_refl | ].
  specialize (Hg a). names.
Qed.
