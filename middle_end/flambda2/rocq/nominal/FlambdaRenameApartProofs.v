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

(* Proofs of the statements of FlambdaRenameApartSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaBindersSpec FlambdaCompareSpec FlambdaCompareProofs
  FlambdaNominalSpec
  FlambdaNominalProofs FlambdaAlphaSpec FlambdaAlphaProofs
  FlambdaEquivarianceProofs FlambdaCompareAlphaSpec
  FlambdaCompareAlphaProofs FlambdaNameAbstractionSpec
  FlambdaFreshQuantifierSpec FlambdaFreshQuantifierProofs
  FlambdaRenameApartSpec.
Import ListNotations.

(* What [rename_apart_avoiding] gives is the term, or is the same as it
   up to bound names. *)
Theorem rename_apart_avoiding_equivalent :
  rename_apart_avoiding_equivalent_statement.
Proof.
  intros avoid e. unfold rename_apart_avoiding.
  destruct (expr_eqb [] [] e (canonical_avoiding avoid e)) eqn:Hsame.
  - exact (expr_eqb_sound _ _ Hsame).
  - intros vars conts statics hp rd. apply equivalent_refl.
Qed.

Theorem rename_apart_equivalent : rename_apart_equivalent_statement.
Proof. intros e. apply rename_apart_avoiding_equivalent. Qed.

(* ======================================================================== *)
(* Lists of new names                                                       *)
(* ======================================================================== *)

Lemma skipn_tl {A : Type} (n : nat) (l : list A) :
  skipn n (tl l) = skipn (S n) l.
Proof. destruct l; [ destruct n; reflexivity | reflexivity ]. Qed.

Lemma tl_map {A B : Type} (f : A -> B) (l : list A) :
  tl (map f l) = map f (tl l).
Proof. destruct l; reflexivity. Qed.

Lemma hd_map {A B : Type} (f : A -> B) (d : A) (l : list A) :
  hd (f d) (map f l) = f (hd d l).
Proof. destruct l; reflexivity. Qed.

Lemma in_tl {A : Type} (a : A) (l : list A) : In a (tl l) -> In a l.
Proof. destruct l; simpl; auto. Qed.

Lemma in_skipn {A : Type} (a : A) (n : nat) :
  ∀ l : list A, In a (skipn n l) -> In a l.
Proof.
  induction n as [ | n IH]; intros [ | b l] Hin; simpl in Hin |- *;
    try assumption.
  right. exact (IH l Hin).
Qed.

Lemma list_sum_cons (n : nat) (l : list nat) :
  list_sum (n :: l) = n + list_sum l.
Proof. reflexivity. Qed.

(* ======================================================================== *)
(* How many new names are taken                                             *)
(* ======================================================================== *)

Lemma bind_variables_rest (xs : list variable) :
  ∀ (r : renaming) (news : list positive),
    fst (fst (bind_variables r news xs)) = skipn (length xs) news.
Proof.
  induction xs as [ | x xs IH]; intros r news; cbn [bind_variables length];
    [ reflexivity | ].
  pose proof (IH (bind_variable r x (new_variable r news x)) (tl news))
    as Hrest.
  destruct
    (bind_variables (bind_variable r x (new_variable r news x)) (tl news)
       xs)
    as [[rest xs'] r'].
  cbn [fst] in Hrest |- *. rewrite Hrest. apply skipn_tl.
Qed.

Lemma bind_continuations_rest (ks : list continuation) :
  ∀ (r : renaming) (news : list positive),
    fst (fst (bind_continuations r news ks)) = skipn (length ks) news.
Proof.
  induction ks as [ | k ks IH]; intros r news;
    cbn [bind_continuations length]; [ reflexivity | ].
  pose proof
    (IH (bind_continuation r k (new_continuation r news k)) (tl news))
    as Hrest.
  destruct
    (bind_continuations (bind_continuation r k (new_continuation r news k))
       (tl news) ks)
    as [[rest ks'] r'].
  cbn [fst] in Hrest |- *. rewrite Hrest. apply skipn_tl.
Qed.

(* An expression takes as many new names as it has binders, from the
   front. *)
Fixpoint apart_rest (e : expr) :
  ∀ (r : renaming) (news : list positive),
    fst (apart r news e) = skipn (binders e) news
with apart_handler_rest (h : continuation_handler) :
  ∀ (r : renaming) (news : list positive),
    fst (apart_handler r news h) = skipn (binders_in_handler h) news
with apart_binding_rest (binding : static_binding) :
  ∀ (r : renaming) (news : list positive),
    fst (apart_binding r news binding)
    = skipn (binders_in_binding binding) news
with apart_code_rest (c : code) :
  ∀ (r : renaming) (news : list positive),
    fst (apart_code r news c) = skipn (binders_in_code c) news.
Proof.
  - destruct e
      as [x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros r news;
      cbn [apart apart_handler apart_binding apart_code binders
           binders_in_handler binders_in_binding binders_in_code].
    + pose proof
        (apart_rest body (bind_variable r x (new_variable r news x))
           (tl news))
        as Hbody.
      destruct
        (apart (bind_variable r x (new_variable r news x)) (tl news) body)
        as [rest body'].
      cbn [fst] in Hbody |- *. rewrite Hbody. apply skipn_tl.
    + assert (Hthread :
        ∀ news' : list positive,
          fst (thread (apart_binding r) news' bindings)
          = skipn (list_sum (map binders_in_binding bindings)) news').
      { induction bindings as [ | binding bindings IH]; intros news';
          cbn [thread map]; [ reflexivity | ].
        pose proof (apart_binding_rest binding r news') as Hfirst.
        destruct (apart_binding r news' binding) as [rest binding'].
        pose proof (IH rest) as Hrest.
        destruct (thread (apart_binding r) rest bindings)
          as [rest' bindings'].
        cbn [fst] in Hfirst, Hrest |- *.
        rewrite Hrest, Hfirst, skipn_skipn, list_sum_cons. f_equal.
        apply Nat.add_comm. }
      pose proof (Hthread news) as Hfirst.
      destruct (thread (apart_binding r) news bindings)
        as [rest bindings'].
      pose proof (apart_rest body r rest) as Hbody.
      destruct (apart r rest body) as [rest' body'].
      cbn [fst] in Hfirst, Hbody |- *.
      rewrite Hbody, Hfirst, skipn_skipn. f_equal. apply Nat.add_comm.
    + pose proof (apart_handler_rest h r news) as Hh.
      destruct (apart_handler r news h) as [rest h'].
      pose proof
        (apart_rest body
           (bind_continuation r k (new_continuation r rest k)) (tl rest))
        as Hbody.
      destruct
        (apart (bind_continuation r k (new_continuation r rest k)) (tl rest)
           body)
        as [rest' body'].
      cbn [fst] in Hh, Hbody |- *.
      rewrite Hbody, skipn_tl, Hh, skipn_skipn. f_equal.
      apply Nat.add_comm.
    + assert (Hthread :
        ∀ (r' : renaming) (news' : list positive),
          fst
            (thread (fun news kh => apart_handler r' news (snd kh)) news'
               handlers)
          = skipn
              (list_sum
                 (map (fun kh => binders_in_handler (snd kh)) handlers))
              news').
      { intros r'.
        induction handlers as [ | [k h] handlers IH]; intros news';
          cbn [thread map snd]; [ reflexivity | ].
        pose proof (apart_handler_rest h r' news') as Hfirst.
        destruct (apart_handler r' news' h) as [rest h'].
        pose proof (IH rest) as Hrest.
        destruct
          (thread (fun news kh => apart_handler r' news (snd kh)) rest
             handlers)
          as [rest' handlers'].
        cbn [fst] in Hfirst, Hrest |- *.
        rewrite Hrest, Hfirst, skipn_skipn, list_sum_cons. f_equal.
        apply Nat.add_comm. }
      pose proof (bind_continuations_rest (map fst handlers) r news) as Hks.
      destruct (bind_continuations r news (map fst handlers))
        as [[rest ks'] r'].
      pose proof (Hthread r' rest) as Hhandlers.
      destruct
        (thread (fun news kh => apart_handler r' news (snd kh)) rest
           handlers)
        as [rest' handlers'].
      pose proof (apart_rest body r' rest') as Hbody.
      destruct (apart r' rest' body) as [rest'' body'].
      cbn [fst] in Hks, Hhandlers, Hbody |- *.
      rewrite Hbody, Hhandlers, Hks, !skipn_skipn, length_map. f_equal.
      lia.
    + reflexivity.
    + reflexivity.
    + reflexivity.
    + reflexivity.
  - destruct h as [params metadata handler]. intros r news.
    cbn [apart apart_handler apart_binding apart_code binders
         binders_in_handler binders_in_binding binders_in_code].
    pose proof (bind_variables_rest params r news) as Hparams.
    destruct (bind_variables r news params) as [[rest params'] r'].
    pose proof (apart_rest handler r' rest) as Hbody.
    destruct (apart r' rest handler) as [rest' handler'].
    cbn [fst] in Hparams, Hbody |- *.
    rewrite Hbody, Hparams, skipn_skipn. f_equal. apply Nat.add_comm.
  - destruct binding as [id c | id | sym const]; intros r news;
      cbn [apart apart_handler apart_binding apart_code binders
           binders_in_handler binders_in_binding binders_in_code];
      [ | reflexivity | reflexivity ].
    pose proof (apart_code_rest c r news) as Hc.
    destruct (apart_code r news c) as [rest c']. exact Hc.
  - destruct c as [params my_closure ret exn metadata body]. intros r news.
    cbn [apart apart_handler apart_binding apart_code binders
         binders_in_handler binders_in_binding binders_in_code].
    match goal with
    | |- context [bind_variables ?r1 ?l ?xs] =>
      pose proof (bind_variables_rest xs r1 l) as Hparams;
      destruct (bind_variables r1 l xs) as [[rest params'] r2]
    end.
    match goal with
    | |- context [apart ?r4 ?l body] =>
      pose proof (apart_rest body r4 l) as Hbody;
      destruct (apart r4 l body) as [rest' body']
    end.
    cbn [fst] in Hparams, Hbody |- *.
    rewrite Hbody, !skipn_tl, Hparams, skipn_skipn, skipn_tl. f_equal.
    lia.
Qed.

(* The same for the bindings of a group and the handlers of a recursive
   [Let_cont], gone through in turn. *)
Lemma thread_bindings_rest (r : renaming) (bindings : list static_binding) :
  ∀ news : list positive,
    fst (thread (apart_binding r) news bindings)
    = skipn (list_sum (map binders_in_binding bindings)) news.
Proof.
  induction bindings as [ | binding bindings IH]; intros news;
    cbn [thread map]; [ reflexivity | ].
  pose proof (apart_binding_rest binding r news) as Hfirst.
  destruct (apart_binding r news binding) as [rest binding'].
  pose proof (IH rest) as Hrest.
  destruct (thread (apart_binding r) rest bindings) as [rest' bindings'].
  cbn [fst] in Hfirst, Hrest |- *.
  rewrite Hrest, Hfirst, skipn_skipn, list_sum_cons. f_equal. lia.
Qed.

Lemma thread_handlers_rest (r : renaming)
    (handlers : list (continuation * continuation_handler)) :
  ∀ news : list positive,
    fst
      (thread (fun news kh => apart_handler r news (snd kh)) news handlers)
    = skipn
        (list_sum (map (fun kh => binders_in_handler (snd kh)) handlers))
        news.
Proof.
  induction handlers as [ | [k h] handlers IH]; intros news;
    cbn [thread map snd]; [ reflexivity | ].
  pose proof (apart_handler_rest h r news) as Hfirst.
  destruct (apart_handler r news h) as [rest h'].
  pose proof (IH rest) as Hrest.
  destruct
    (thread (fun news kh => apart_handler r news (snd kh)) rest handlers)
    as [rest' handlers'].
  cbn [fst] in Hfirst, Hrest |- *.
  rewrite Hrest, Hfirst, skipn_skipn, list_sum_cons. f_equal. lia.
Qed.

(* ======================================================================== *)
(* A renaming keeps the number of binders and renames the names             *)
(* ======================================================================== *)

Fixpoint binders_renamed (r : renaming) (e : expr) :
  binders (rename_expr r e) = binders e
with binders_in_handler_renamed (r : renaming) (h : continuation_handler) :
  binders_in_handler (rename_handler r h) = binders_in_handler h
with binders_in_binding_renamed (r : renaming) (binding : static_binding) :
  binders_in_binding (rename_static_binding r binding)
  = binders_in_binding binding
with binders_in_code_renamed (r : renaming) (c : code) :
  binders_in_code (rename_code r c) = binders_in_code c.
Proof.
  - destruct e
      as [x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + rewrite (binders_renamed r body). reflexivity.
    + rewrite (binders_renamed r body). f_equal.
      induction bindings as [ | binding bindings IH]; simpl;
        [ reflexivity | ].
      rewrite (binders_in_binding_renamed r binding), IH. reflexivity.
    + rewrite (binders_in_handler_renamed r h), (binders_renamed r body).
      reflexivity.
    + rewrite (binders_renamed r body), length_map. f_equal. f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl;
        [ reflexivity | ].
      rewrite (binders_in_handler_renamed r h), IH. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    rewrite length_map, (binders_renamed r handler). reflexivity.
  - destruct binding as [id c | id | sym const]; simpl;
      [ exact (binders_in_code_renamed r c) | reflexivity | reflexivity ].
  - destruct c as [params my_closure ret exn metadata body]. simpl.
    rewrite length_map, (binders_renamed r body). reflexivity.
Qed.

Lemma variables_in_simple_renamed (r : renaming) (s : simple) :
  variables_in_simple (rename_simple r s)
  = map (rename_variable r) (variables_in_simple s).
Proof. destruct s; reflexivity. Qed.

Lemma variables_in_simples_renamed (r : renaming) (args : list simple) :
  variables_in_simples (map (rename_simple r) args)
  = map (rename_variable r) (variables_in_simples args).
Proof.
  unfold variables_in_simples.
  induction args as [ | s args IH]; simpl; [ reflexivity | ].
  rewrite map_app, IH, variables_in_simple_renamed. reflexivity.
Qed.

Lemma variables_in_slots_renamed (r : renaming)
    (slots : list (value_slot * simple)) :
  variables_in_slots
    (map (fun slot => (fst slot, rename_simple r (snd slot))) slots)
  = map (rename_variable r) (variables_in_slots slots).
Proof.
  unfold variables_in_slots.
  induction slots as [ | [slot s] slots IH]; simpl; [ reflexivity | ].
  rewrite map_app, IH, variables_in_simple_renamed. reflexivity.
Qed.

Lemma variables_in_named_renamed (r : renaming) (d : named) :
  variables_in_named (rename_named r d)
  = map (rename_variable r) (variables_in_named d).
Proof.
  destruct d; simpl;
    first
      [ apply variables_in_simple_renamed | apply variables_in_simples_renamed
      | apply variables_in_slots_renamed ].
Qed.

Lemma variables_in_static_const_renamed (r : renaming)
    (const : static_const) :
  variables_in_static_const (rename_static_const r const)
  = map (rename_variable r) (variables_in_static_const const).
Proof.
  destruct const; simpl;
    first
      [ apply variables_in_simple_renamed | apply variables_in_simples_renamed
      | apply variables_in_slots_renamed | reflexivity ].
Qed.

Lemma variables_in_apply_cont_renamed (r : renaming) (ac : apply_cont) :
  variables_in_apply_cont (rename_apply_cont r ac)
  = map (rename_variable r) (variables_in_apply_cont ac).
Proof. apply variables_in_simples_renamed. Qed.

Lemma variables_in_apply_renamed (r : renaming) (ap : apply_expr) :
  variables_in_apply (rename_apply r ap)
  = map (rename_variable r) (variables_in_apply ap).
Proof.
  unfold variables_in_apply. rewrite exn_extra_simples_rename.
  unfold rename_apply. cbn [ap_callee ap_args].
  rewrite !map_app, variables_in_simple_renamed,
    !variables_in_simples_renamed.
  reflexivity.
Qed.

Fixpoint variables_of_renamed (r : renaming) (e : expr) :
  variables_of (rename_expr r e) = map (rename_variable r) (variables_of e)
with variables_in_handler_renamed (r : renaming) (h : continuation_handler) :
  variables_in_handler (rename_handler r h)
  = map (rename_variable r) (variables_in_handler h)
with variables_in_binding_renamed (r : renaming) (binding : static_binding) :
  variables_in_binding (rename_static_binding r binding)
  = map (rename_variable r) (variables_in_binding binding)
with variables_in_code_renamed (r : renaming) (c : code) :
  variables_in_code (rename_code r c)
  = map (rename_variable r) (variables_in_code c).
Proof.
  - destruct e
      as [x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl.
    + rewrite map_app, variables_in_named_renamed,
        (variables_of_renamed r body).
      reflexivity.
    + rewrite map_app, (variables_of_renamed r body). f_equal.
      induction bindings as [ | binding bindings IH]; simpl;
        [ reflexivity | ].
      rewrite map_app, (variables_in_binding_renamed r binding), IH.
      reflexivity.
    + rewrite map_app, (variables_in_handler_renamed r h),
        (variables_of_renamed r body).
      reflexivity.
    + rewrite map_app, (variables_of_renamed r body). f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl;
        [ reflexivity | ].
      rewrite map_app, (variables_in_handler_renamed r h), IH. reflexivity.
    + apply variables_in_apply_renamed.
    + apply variables_in_apply_cont_renamed.
    + rewrite map_app, variables_in_simple_renamed. f_equal.
      induction arms as [ | [i ac] arms IH]; simpl; [ reflexivity | ].
      rewrite map_app, variables_in_apply_cont_renamed, IH. reflexivity.
    + reflexivity.
  - destruct h as [params metadata handler]. simpl.
    rewrite map_app, (variables_of_renamed r handler). reflexivity.
  - destruct binding as [id c | id | sym const]; simpl;
      [ exact (variables_in_code_renamed r c) | reflexivity
      | apply variables_in_static_const_renamed ].
  - destruct c as [params my_closure ret exn metadata body]. simpl.
    rewrite map_app, (variables_of_renamed r body). reflexivity.
Qed.

Fixpoint continuations_of_renamed (r : renaming) (e : expr) :
  continuations_of (rename_expr r e)
  = map (rename_continuation r) (continuations_of e)
with continuations_in_handler_renamed (r : renaming)
    (h : continuation_handler) :
  continuations_in_handler (rename_handler r h)
  = map (rename_continuation r) (continuations_in_handler h)
with continuations_in_binding_renamed (r : renaming)
    (binding : static_binding) :
  continuations_in_binding (rename_static_binding r binding)
  = map (rename_continuation r) (continuations_in_binding binding)
with continuations_in_code_renamed (r : renaming) (c : code) :
  continuations_in_code (rename_code r c)
  = map (rename_continuation r) (continuations_in_code c).
Proof.
  - destruct e
      as [x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl.
    + exact (continuations_of_renamed r body).
    + rewrite map_app, (continuations_of_renamed r body). f_equal.
      induction bindings as [ | binding bindings IH]; simpl;
        [ reflexivity | ].
      rewrite map_app, (continuations_in_binding_renamed r binding), IH.
      reflexivity.
    + rewrite map_app, (continuations_in_handler_renamed r h),
        (continuations_of_renamed r body).
      reflexivity.
    + rewrite map_app, (continuations_of_renamed r body). f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl;
        [ reflexivity | ].
      rewrite map_app, (continuations_in_handler_renamed r h), IH.
      reflexivity.
    + reflexivity.
    + apply apply_cont_continuations_renamed.
    + induction arms as [ | [i ac] arms IH]; cbn [map flat_map snd];
        [ reflexivity | ].
      rewrite map_app, apply_cont_continuations_renamed, IH. reflexivity.
    + reflexivity.
  - destruct h as [params metadata handler]. simpl.
    exact (continuations_of_renamed r handler).
  - destruct binding as [id c | id | sym const]; simpl;
      [ exact (continuations_in_code_renamed r c) | reflexivity
      | reflexivity ].
  - destruct c as [params my_closure ret exn metadata body]. simpl.
    rewrite (continuations_of_renamed r body). reflexivity.
Qed.

(* ======================================================================== *)
(* Two renamings that agree on the names of a term rename it alike          *)
(* ======================================================================== *)

Section Agreeing.

Variables r r' : renaming.

Lemma rename_simple_agree (s : simple) :
  (∀ x : variable,
    In x (variables_in_simple s) -> rename_variable r x = rename_variable r' x)
  ->
  rename_simple r s = rename_simple r' s.
Proof.
  destruct s as [x | sym | c]; simpl; intros Hv;
    [ rewrite (Hv x (or_introl eq_refl)) | | ]; reflexivity.
Qed.

Lemma rename_simples_agree (args : list simple) :
  (∀ x : variable,
    In x (variables_in_simples args) ->
    rename_variable r x = rename_variable r' x) ->
  map (rename_simple r) args = map (rename_simple r') args.
Proof.
  unfold variables_in_simples.
  induction args as [ | s args IH]; simpl; intros Hv; [ reflexivity | ].
  f_equal; [ apply rename_simple_agree | apply IH ]; intros x Hx;
    apply Hv, in_or_app; [ left | right ]; exact Hx.
Qed.

Lemma rename_slots_agree (slots : list (value_slot * simple)) :
  (∀ x : variable,
    In x (variables_in_slots slots) ->
    rename_variable r x = rename_variable r' x) ->
  map (fun slot => (fst slot, rename_simple r (snd slot))) slots
  = map (fun slot => (fst slot, rename_simple r' (snd slot))) slots.
Proof.
  unfold variables_in_slots.
  induction slots as [ | [slot s] slots IH]; simpl; intros Hv;
    [ reflexivity | ].
  f_equal; [ f_equal; apply rename_simple_agree | apply IH ]; intros x Hx;
    apply Hv, in_or_app; [ left | right ]; exact Hx.
Qed.

Lemma rename_named_agree (d : named) :
  (∀ x : variable,
    In x (variables_in_named d) -> rename_variable r x = rename_variable r' x)
  ->
  rename_named r d = rename_named r' d.
Proof.
  destruct d; simpl; intros Hv; f_equal;
    first
      [ apply rename_simple_agree | apply rename_simples_agree
      | apply rename_slots_agree ];
    exact Hv.
Qed.

Lemma rename_static_const_agree (const : static_const) :
  (∀ x : variable,
    In x (variables_in_static_const const) ->
    rename_variable r x = rename_variable r' x) ->
  rename_static_const r const = rename_static_const r' const.
Proof.
  destruct const; simpl; intros Hv; try reflexivity; f_equal;
    first
      [ apply rename_simple_agree | apply rename_simples_agree
      | apply rename_slots_agree ];
    exact Hv.
Qed.

Lemma rename_apply_cont_agree (ac : apply_cont) :
  (∀ x : variable,
    In x (variables_in_apply_cont ac) ->
    rename_variable r x = rename_variable r' x) ->
  (∀ k : continuation,
    In k (apply_cont_continuations ac) ->
    rename_continuation r k = rename_continuation r' k) ->
  rename_apply_cont r ac = rename_apply_cont r' ac.
Proof.
  intros Hv Hk. unfold rename_apply_cont.
  rewrite (Hk (ac_continuation ac) (or_introl eq_refl)). f_equal.
  - apply rename_simples_agree. exact Hv.
  - unfold apply_cont_continuations in Hk.
    destruct (ac_trap_action ac) as [[k | k kind] | ]; simpl in Hk |- *;
      [ | | reflexivity ];
      rewrite (Hk k (or_intror (or_introl eq_refl))); reflexivity.
Qed.

Lemma rename_exn_extra_args_agree (extra : list (simple * flambda_kind)) :
  (∀ x : variable,
    In x (variables_in_simples (map fst extra)) ->
    rename_variable r x = rename_variable r' x) ->
  map (fun arg => (rename_simple r (fst arg), snd arg)) extra
  = map (fun arg => (rename_simple r' (fst arg), snd arg)) extra.
Proof.
  unfold variables_in_simples.
  induction extra as [ | [s kind] extra IH]; simpl; intros Hv;
    [ reflexivity | ].
  f_equal; [ f_equal; apply rename_simple_agree | apply IH ]; intros x Hx;
    apply Hv, in_or_app; [ left | right ]; exact Hx.
Qed.

Lemma rename_apply_agree (ap : apply_expr) :
  (∀ x : variable,
    In x (variables_in_apply ap) -> rename_variable r x = rename_variable r' x)
  ->
  (∀ k : continuation,
    In k (continuations_in_apply ap) ->
    rename_continuation r k = rename_continuation r' k) ->
  rename_apply r ap = rename_apply r' ap.
Proof.
  intros Hv Hk. unfold rename_apply.
  rewrite (Hk (ap_continuation ap) (or_introl eq_refl)),
    (Hk (ap_exn_continuation ap) (or_intror (or_introl eq_refl))).
  f_equal;
    [ apply rename_simple_agree | apply rename_simples_agree
    | apply rename_exn_extra_args_agree ];
    intros x Hx; apply Hv, in_or_app;
    [ left | right; apply in_or_app; left
    | right; apply in_or_app; right ];
    exact Hx.
Qed.

Fixpoint rename_expr_agree (e : expr) :
  (∀ x : variable,
    In x (variables_of e) -> rename_variable r x = rename_variable r' x) ->
  (∀ k : continuation,
    In k (continuations_of e) ->
    rename_continuation r k = rename_continuation r' k) ->
  rename_expr r e = rename_expr r' e
with rename_handler_agree (h : continuation_handler) :
  (∀ x : variable,
    In x (variables_in_handler h) -> rename_variable r x = rename_variable r' x)
  ->
  (∀ k : continuation,
    In k (continuations_in_handler h) ->
    rename_continuation r k = rename_continuation r' k) ->
  rename_handler r h = rename_handler r' h
with rename_binding_agree (binding : static_binding) :
  (∀ x : variable,
    In x (variables_in_binding binding) ->
    rename_variable r x = rename_variable r' x) ->
  (∀ k : continuation,
    In k (continuations_in_binding binding) ->
    rename_continuation r k = rename_continuation r' k) ->
  rename_static_binding r binding = rename_static_binding r' binding
with rename_code_agree (c : code) :
  (∀ x : variable,
    In x (variables_in_code c) -> rename_variable r x = rename_variable r' x)
  ->
  (∀ k : continuation,
    In k (continuations_in_code c) ->
    rename_continuation r k = rename_continuation r' k) ->
  rename_code r c = rename_code r' c.
Proof.
  - destruct e
      as [x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; intros Hv Hk.
    + rewrite (Hv x (or_introl eq_refl)). f_equal.
      * apply rename_named_agree. intros y Hy. apply Hv. right.
        apply in_or_app. left. exact Hy.
      * apply (rename_expr_agree body);
          [ intros y Hy; apply Hv; right; apply in_or_app; right; exact Hy
          | exact Hk ].
    + f_equal.
      * assert (Hv' :
          ∀ y : variable,
            In y (flat_map variables_in_binding bindings) ->
            rename_variable r y = rename_variable r' y)
          by (intros y Hy; apply Hv, in_or_app; left; exact Hy).
        assert (Hk' :
          ∀ j : continuation,
            In j (flat_map continuations_in_binding bindings) ->
            rename_continuation r j = rename_continuation r' j)
          by (intros j Hj; apply Hk, in_or_app; left; exact Hj).
        clear Hv Hk.
        induction bindings as [ | binding bindings IH];
          simpl in Hv', Hk' |- *; [ reflexivity | ].
        f_equal.
        -- apply (rename_binding_agree binding); intros y Hy;
             [ apply Hv' | apply Hk' ]; apply in_or_app; left; exact Hy.
        -- apply IH; intros y Hy; [ apply Hv' | apply Hk' ];
             apply in_or_app; right; exact Hy.
      * apply (rename_expr_agree body); intros y Hy;
          [ apply Hv | apply Hk ]; apply in_or_app; right; exact Hy.
    + rewrite (Hk k (or_introl eq_refl)). f_equal.
      * apply (rename_handler_agree h); intros y Hy;
          [ apply Hv | apply Hk; right ]; apply in_or_app; left; exact Hy.
      * apply (rename_expr_agree body); intros y Hy;
          [ apply Hv | apply Hk; right ]; apply in_or_app; right; exact Hy.
    + f_equal.
      * assert (Hv' :
          ∀ y : variable,
            In y
              (flat_map (fun kh => variables_in_handler (snd kh)) handlers)
            ->
            rename_variable r y = rename_variable r' y)
          by (intros y Hy; apply Hv, in_or_app; left; exact Hy).
        assert (Hk' :
          ∀ j : continuation,
            In j
              (flat_map
                 (fun kh => fst kh :: continuations_in_handler (snd kh))
                 handlers)
            ->
            rename_continuation r j = rename_continuation r' j)
          by (intros j Hj; apply Hk, in_or_app; left; exact Hj).
        clear Hv Hk.
        induction handlers as [ | [k h] handlers IH];
          simpl in Hv', Hk' |- *; [ reflexivity | ].
        rewrite (Hk' k (or_introl eq_refl)). f_equal.
        -- f_equal. apply (rename_handler_agree h); intros y Hy;
             [ apply Hv' | apply Hk'; right ]; apply in_or_app; left;
             exact Hy.
        -- apply IH; intros y Hy; [ apply Hv' | apply Hk'; right ];
             apply in_or_app; right; exact Hy.
      * apply (rename_expr_agree body); intros y Hy;
          [ apply Hv | apply Hk ]; apply in_or_app; right; exact Hy.
    + f_equal. exact (rename_apply_agree ap Hv Hk).
    + f_equal. exact (rename_apply_cont_agree ac Hv Hk).
    + f_equal.
      * apply rename_simple_agree. intros y Hy. apply Hv, in_or_app. left.
        exact Hy.
      * assert (Hv' :
          ∀ y : variable,
            In y
              (flat_map (fun arm => variables_in_apply_cont (snd arm)) arms)
            ->
            rename_variable r y = rename_variable r' y)
          by (intros y Hy; apply Hv, in_or_app; right; exact Hy).
        clear Hv.
        induction arms as [ | [i ac] arms IH];
          cbn [map flat_map fst snd] in Hv', Hk |- *; [ reflexivity | ].
        f_equal.
        -- f_equal. apply rename_apply_cont_agree; intros y Hy;
             [ apply Hv' | apply Hk ]; apply in_or_app; left; exact Hy.
        -- apply IH; intros y Hy; [ apply Hk | apply Hv' ];
             apply in_or_app; right; exact Hy.
    + reflexivity.
  - destruct h as [params metadata handler]. simpl. intros Hv Hk. f_equal.
    + apply map_ext_in. intros y Hy. apply Hv, in_or_app. left. exact Hy.
    + apply (rename_expr_agree handler);
        [ intros y Hy; apply Hv, in_or_app; right; exact Hy | exact Hk ].
  - destruct binding as [id c | id | sym const]; simpl; intros Hv Hk.
    + f_equal. exact (rename_code_agree c Hv Hk).
    + reflexivity.
    + f_equal. apply rename_static_const_agree. exact Hv.
  - destruct c as [params my_closure ret exn metadata body]. simpl.
    intros Hv Hk.
    rewrite (Hv my_closure (or_introl eq_refl)),
      (Hk ret (or_introl eq_refl)),
      (Hk exn (or_intror (or_introl eq_refl))).
    f_equal.
    + apply map_ext_in. intros y Hy. apply Hv. right. apply in_or_app. left.
      exact Hy.
    + apply (rename_expr_agree body);
        [ intros y Hy; apply Hv; right; apply in_or_app; right; exact Hy
        | intros j Hj; apply Hk; right; right; exact Hj ].
Qed.

End Agreeing.

(* ======================================================================== *)
(* The candidate of a renamed term                                          *)
(* ======================================================================== *)

Lemma rename_variable_bind_variable (r : renaming) (x x' y : variable) :
  rename_variable (bind_variable r x x') y
  = swap (rename_variable r x) x' (rename_variable r y).
Proof. reflexivity. Qed.

Lemma rename_continuation_bind_continuation (r : renaming)
    (k k' j : continuation) :
  rename_continuation (bind_continuation r k k') j
  = swap (rename_continuation r k) k' (rename_continuation r j).
Proof. reflexivity. Qed.

Lemma rename_variable_swap (p : renaming) (a b x : variable) :
  rename_variable p (swap a b x)
  = swap (rename_variable p a) (rename_variable p b) (rename_variable p x).
Proof. apply permute_swap. Qed.

Lemma rename_continuation_swap (p : renaming) (a b k : continuation) :
  rename_continuation p (swap a b k)
  = swap (rename_continuation p a) (rename_continuation p b)
      (rename_continuation p k).
Proof. apply permute_swap. Qed.

Lemma map_combine {A B A' B' : Type} (f : A -> A') (g : B -> B')
    (l : list A) :
  ∀ l' : list B,
    map (fun '(a, b) => (f a, g b)) (combine l l')
    = combine (map f l) (map g l').
Proof.
  induction l as [ | a l IH]; intros [ | b l']; simpl; try reflexivity.
  rewrite IH. reflexivity.
Qed.

Lemma map_fst_renamed (p : renaming)
    (handlers : list (continuation * continuation_handler)) :
  map fst
    (map (fun '(k, h) => (rename_continuation p k, rename_handler p h))
       handlers)
  = map (rename_continuation p) (map fst handlers).
Proof.
  rewrite !map_map. apply map_ext. intros [k h]. reflexivity.
Qed.

Section Carried.

(* The renaming of the term. *)
Variable p : renaming.

(* The new names are renamed as variables are. One that a continuation
   takes must be renamed in the same way as a continuation. *)
Definition agree (news : list positive) : Prop :=
  ∀ a : positive, In a news -> rename_variable p a = rename_continuation p a.

Lemma tl_news (news : list positive) :
  tl (map (rename_variable p) news) = map (rename_variable p) (tl news).
Proof. apply tl_map. Qed.

Lemma agree_tl (news : list positive) : agree news -> agree (tl news).
Proof. intros Hagree a Ha. apply Hagree, in_tl, Ha. Qed.

Lemma agree_skipn (n : nat) (news : list positive) :
  agree news -> agree (skipn n news).
Proof. intros Hagree a Ha. apply Hagree, (in_skipn a n), Ha. Qed.

(* The renaming of the binders in scope in the renamed term is the one
   in the term, carried over by [p]. *)
Definition carries (r r' : renaming) : Prop :=
  (∀ x : variable,
    rename_variable p (rename_variable r x)
    = rename_variable r' (rename_variable p x))
  ∧ (∀ k : continuation,
      rename_continuation p (rename_continuation r k)
      = rename_continuation r' (rename_continuation p k)).

Lemma carries_identity : carries identity_renaming identity_renaming.
Proof. split; intros name; reflexivity. Qed.

Lemma carries_bind_variable (r r' : renaming) (x x' : variable) :
  carries r r' ->
  carries (bind_variable r x x')
    (bind_variable r' (rename_variable p x) (rename_variable p x')).
Proof.
  intros [Hv Hk]. split.
  - intros y.
    rewrite !rename_variable_bind_variable, rename_variable_swap, !Hv.
    reflexivity.
  - exact Hk.
Qed.

Lemma carries_bind_continuation (r r' : renaming) (k k' : continuation) :
  carries r r' ->
  carries (bind_continuation r k k')
    (bind_continuation r' (rename_continuation p k)
       (rename_continuation p k')).
Proof.
  intros [Hv Hk]. split.
  - exact Hv.
  - intros j.
    rewrite !rename_continuation_bind_continuation,
      rename_continuation_swap, !Hk.
    reflexivity.
Qed.

Lemma new_variable_renamed (r r' : renaming) (news : list positive)
    (x : variable) :
  carries r r' ->
  new_variable r' (map (rename_variable p) news) (rename_variable p x)
  = rename_variable p (new_variable r news x).
Proof.
  intros [Hv _]. unfold new_variable. rewrite <- Hv. apply hd_map.
Qed.

Lemma new_continuation_renamed (r r' : renaming) (news : list positive)
    (k : continuation) :
  carries r r' ->
  agree news ->
  new_continuation r' (map (rename_variable p) news)
    (rename_continuation p k)
  = rename_continuation p (new_continuation r news k).
Proof.
  intros [_ Hk] Hagree. unfold new_continuation.
  destruct news as [ | a news]; cbn [map hd];
    [ symmetry; apply Hk | apply Hagree; left; reflexivity ].
Qed.

Lemma bind_variables_renamed (xs : list variable) :
  ∀ (r r' : renaming) (news : list positive),
    carries r r' ->
    fst
      (bind_variables r' (map (rename_variable p) news)
         (map (rename_variable p) xs))
    = (map (rename_variable p) (fst (fst (bind_variables r news xs))),
       map (rename_variable p) (snd (fst (bind_variables r news xs))))
    ∧ carries (snd (bind_variables r news xs))
        (snd
           (bind_variables r' (map (rename_variable p) news)
              (map (rename_variable p) xs))).
Proof.
  induction xs as [ | x xs IH]; intros r r' news Hcarries;
    cbn [bind_variables map]; [ split; [ reflexivity | exact Hcarries ] | ].
  rewrite (new_variable_renamed r r' news x Hcarries), tl_news.
  pose proof
    (IH _ _ (tl news)
       (carries_bind_variable r r' x (new_variable r news x) Hcarries))
    as [Hbound Hcarries'].
  destruct
    (bind_variables (bind_variable r x (new_variable r news x)) (tl news) xs)
    as [[rest xs'] r1].
  destruct
    (bind_variables
       (bind_variable r' (rename_variable p x)
          (rename_variable p (new_variable r news x)))
       (map (rename_variable p) (tl news)) (map (rename_variable p) xs))
    as [[rest2 xs2] r2].
  cbn [fst snd] in Hbound, Hcarries' |- *.
  injection Hbound as -> ->. split; [ reflexivity | exact Hcarries' ].
Qed.

Lemma bind_continuations_renamed (ks : list continuation) :
  ∀ (r r' : renaming) (news : list positive),
    carries r r' ->
    agree news ->
    fst
      (bind_continuations r' (map (rename_variable p) news)
         (map (rename_continuation p) ks))
    = (map (rename_variable p) (fst (fst (bind_continuations r news ks))),
       map (rename_continuation p)
         (snd (fst (bind_continuations r news ks))))
    ∧ carries (snd (bind_continuations r news ks))
        (snd
           (bind_continuations r' (map (rename_variable p) news)
              (map (rename_continuation p) ks))).
Proof.
  induction ks as [ | k ks IH]; intros r r' news Hcarries Hagree;
    cbn [bind_continuations map];
    [ split; [ reflexivity | exact Hcarries ] | ].
  rewrite (new_continuation_renamed r r' news k Hcarries Hagree), tl_news.
  pose proof
    (IH _ _ (tl news)
       (carries_bind_continuation r r' k (new_continuation r news k)
          Hcarries)
       (agree_tl news Hagree))
    as [Hbound Hcarries'].
  destruct
    (bind_continuations (bind_continuation r k (new_continuation r news k))
       (tl news) ks)
    as [[rest ks'] r1].
  destruct
    (bind_continuations
       (bind_continuation r' (rename_continuation p k)
          (rename_continuation p (new_continuation r news k)))
       (map (rename_variable p) (tl news)) (map (rename_continuation p) ks))
    as [[rest2 ks2] r2].
  cbn [fst snd] in Hbound, Hcarries' |- *.
  injection Hbound as -> ->. split; [ reflexivity | exact Hcarries' ].
Qed.

(* What has no binder is renamed by the renaming in scope, before [p]
   or after. *)
Section Leaves.

Variables r r' : renaming.

Hypothesis carried : carries r r'.

Lemma carried_variables (x : variable) :
  rename_variable (compose_renaming r' p) x
  = rename_variable (compose_renaming p r) x.
Proof. rewrite !rename_variable_compose. symmetry. apply (proj1 carried). Qed.

Lemma carried_continuations (k : continuation) :
  rename_continuation (compose_renaming r' p) k
  = rename_continuation (compose_renaming p r) k.
Proof.
  rewrite !rename_continuation_compose. symmetry. apply (proj2 carried).
Qed.

Lemma rename_named_carried (d : named) :
  rename_named r' (rename_named p d) = rename_named p (rename_named r d).
Proof.
  rewrite <- !rename_named_compose.
  apply rename_named_ext;
    first [ exact carried_variables | exact carried_continuations ].
Qed.

Lemma rename_static_const_carried (const : static_const) :
  rename_static_const r' (rename_static_const p const)
  = rename_static_const p (rename_static_const r const).
Proof.
  rewrite <- !rename_static_const_compose.
  apply rename_static_const_ext;
    first [ exact carried_variables | exact carried_continuations ].
Qed.

Lemma rename_expr_carried (e : expr) :
  rename_expr r' (rename_expr p e) = rename_expr p (rename_expr r e).
Proof.
  rewrite <- (rename_expr_compose r' p e), <- (rename_expr_compose p r e).
  apply rename_expr_ext;
    first [ exact carried_variables | exact carried_continuations ].
Qed.

End Leaves.

(* The binders of the renamed term take the new names renamed, and what
   comes of it is what comes of the term, renamed. *)
Fixpoint apart_renamed (e : expr) :
  ∀ (r r' : renaming) (news : list positive),
    carries r r' ->
    agree news ->
    apart r' (map (rename_variable p) news) (rename_expr p e)
    = (map (rename_variable p) (fst (apart r news e)),
       rename_expr p (snd (apart r news e)))
with apart_handler_renamed (h : continuation_handler) :
  ∀ (r r' : renaming) (news : list positive),
    carries r r' ->
    agree news ->
    apart_handler r' (map (rename_variable p) news) (rename_handler p h)
    = (map (rename_variable p) (fst (apart_handler r news h)),
       rename_handler p (snd (apart_handler r news h)))
with apart_binding_renamed (binding : static_binding) :
  ∀ (r r' : renaming) (news : list positive),
    carries r r' ->
    agree news ->
    apart_binding r' (map (rename_variable p) news)
      (rename_static_binding p binding)
    = (map (rename_variable p) (fst (apart_binding r news binding)),
       rename_static_binding p (snd (apart_binding r news binding)))
with apart_code_renamed (c : code) :
  ∀ (r r' : renaming) (news : list positive),
    carries r r' ->
    agree news ->
    apart_code r' (map (rename_variable p) news) (rename_code p c)
    = (map (rename_variable p) (fst (apart_code r news c)),
       rename_code p (snd (apart_code r news c))).
Proof.
  - destruct e
      as [x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros r r' news Hcarries Hagree;
      cbn [rename_expr rename_handler rename_static_binding rename_code
           apart apart_handler apart_binding apart_code].
    + rewrite (new_variable_renamed r r' news x Hcarries), tl_news.
      rewrite
        (apart_renamed body _ _ (tl news)
           (carries_bind_variable r r' x (new_variable r news x) Hcarries)
           (agree_tl news Hagree)).
      destruct
        (apart (bind_variable r x (new_variable r news x)) (tl news) body)
        as [rest body'].
      cbn [fst snd rename_expr].
      rewrite (rename_named_carried r r' Hcarries). reflexivity.
    + assert (Hthread :
        ∀ news' : list positive,
          agree news' ->
          thread (apart_binding r') (map (rename_variable p) news')
            (map (rename_static_binding p) bindings)
          = (map (rename_variable p)
               (fst (thread (apart_binding r) news' bindings)),
             map (rename_static_binding p)
               (snd (thread (apart_binding r) news' bindings)))).
      { induction bindings as [ | binding bindings IH];
          intros news' Hagree'; cbn [thread map]; [ reflexivity | ].
        rewrite (apart_binding_renamed binding r r' news' Hcarries Hagree').
        pose proof (apart_binding_rest binding r news') as Hrest.
        destruct (apart_binding r news' binding) as [rest binding'].
        cbn [fst snd] in Hrest |- *.
        rewrite (IH rest)
          by (rewrite Hrest; apply agree_skipn; exact Hagree').
        destruct (thread (apart_binding r) rest bindings)
          as [rest' bindings'].
        reflexivity. }
      rewrite (Hthread news Hagree).
      pose proof (thread_bindings_rest r bindings news) as Hrest.
      destruct (thread (apart_binding r) news bindings) as [rest bindings'].
      cbn [fst snd] in Hrest |- *.
      rewrite (apart_renamed body r r' rest Hcarries)
        by (rewrite Hrest; apply agree_skipn; exact Hagree).
      destruct (apart r rest body) as [rest' body']. reflexivity.
    + rewrite (apart_handler_renamed h r r' news Hcarries Hagree).
      pose proof (apart_handler_rest h r news) as Hrest.
      destruct (apart_handler r news h) as [rest h'].
      cbn [fst snd] in Hrest |- *.
      assert (Hagree' : agree rest)
        by (rewrite Hrest; apply agree_skipn; exact Hagree).
      rewrite (new_continuation_renamed r r' rest k Hcarries Hagree'),
        tl_news.
      rewrite
        (apart_renamed body _ _ (tl rest)
           (carries_bind_continuation r r' k (new_continuation r rest k)
              Hcarries)
           (agree_tl rest Hagree')).
      destruct
        (apart (bind_continuation r k (new_continuation r rest k)) (tl rest)
           body)
        as [rest' body'].
      reflexivity.
    + rewrite map_fst_renamed.
      pose proof
        (bind_continuations_renamed (map fst handlers) r r' news Hcarries
           Hagree)
        as [Hbound Hcarries'].
      pose proof (bind_continuations_rest (map fst handlers) r news)
        as Hrest.
      destruct (bind_continuations r news (map fst handlers))
        as [[rest ks'] r1].
      destruct
        (bind_continuations r' (map (rename_variable p) news)
           (map (rename_continuation p) (map fst handlers)))
        as [[rest2 ks2] r2].
      cbn [fst snd] in Hbound, Hcarries', Hrest |- *.
      injection Hbound as -> ->.
      assert (Hagree' : agree rest)
        by (rewrite Hrest; apply agree_skipn; exact Hagree).
      clear Hrest.
      assert (Hthread :
        ∀ news' : list positive,
          agree news' ->
          thread (fun news kh => apart_handler r2 news (snd kh))
            (map (rename_variable p) news')
            (map
               (fun '(k, h) => (rename_continuation p k, rename_handler p h))
               handlers)
          = (map (rename_variable p)
               (fst
                  (thread (fun news kh => apart_handler r1 news (snd kh))
                     news' handlers)),
             map (rename_handler p)
               (snd
                  (thread (fun news kh => apart_handler r1 news (snd kh))
                     news' handlers)))).
      { induction handlers as [ | [k h] handlers IH];
          intros news' Hagree''; cbn [thread map snd]; [ reflexivity | ].
        rewrite (apart_handler_renamed h r1 r2 news' Hcarries' Hagree'').
        pose proof (apart_handler_rest h r1 news') as Hrest.
        destruct (apart_handler r1 news' h) as [rest0 h'].
        cbn [fst snd] in Hrest |- *.
        rewrite (IH rest0)
          by (rewrite Hrest; apply agree_skipn; exact Hagree'').
        destruct
          (thread (fun news kh => apart_handler r1 news (snd kh)) rest0
             handlers)
          as [rest0' hs'].
        reflexivity. }
      rewrite (Hthread rest Hagree').
      pose proof (thread_handlers_rest r1 handlers rest) as Hrest.
      destruct
        (thread (fun news kh => apart_handler r1 news (snd kh)) rest
           handlers)
        as [rest' hs'].
      cbn [fst snd] in Hrest |- *.
      rewrite (apart_renamed body r1 r2 rest' Hcarries')
        by (rewrite Hrest; apply agree_skipn; exact Hagree').
      destruct (apart r1 rest' body) as [rest'' body'].
      cbn [fst snd rename_expr].
      rewrite (map_combine (rename_continuation p) (rename_handler p)).
      reflexivity.
    + exact
        (f_equal (pair _) (rename_expr_carried r r' Hcarries (Apply ap))).
    + exact
        (f_equal (pair _)
           (rename_expr_carried r r' Hcarries (Apply_cont ac))).
    + exact
        (f_equal (pair _)
           (rename_expr_carried r r' Hcarries (Switch scrutinee arms))).
    + reflexivity.
  - destruct h as [params metadata handler].
    intros r r' news Hcarries Hagree.
    cbn [rename_expr rename_handler rename_static_binding rename_code
         apart apart_handler apart_binding apart_code].
    pose proof (bind_variables_renamed params r r' news Hcarries)
      as [Hbound Hcarries'].
    pose proof (bind_variables_rest params r news) as Hrest.
    destruct (bind_variables r news params) as [[rest params'] r1].
    destruct
      (bind_variables r' (map (rename_variable p) news)
         (map (rename_variable p) params))
      as [[rest2 params2] r2].
    cbn [fst snd] in Hbound, Hcarries', Hrest |- *.
    injection Hbound as -> ->.
    rewrite (apart_renamed handler r1 r2 rest Hcarries')
      by (rewrite Hrest; apply agree_skipn; exact Hagree).
    destruct (apart r1 rest handler) as [rest' handler']. reflexivity.
  - destruct binding as [id c | id | sym const];
      intros r r' news Hcarries Hagree;
      cbn [rename_expr rename_handler rename_static_binding rename_code
           apart apart_handler apart_binding apart_code].
    + rewrite (apart_code_renamed c r r' news Hcarries Hagree).
      destruct (apart_code r news c) as [rest c']. reflexivity.
    + reflexivity.
    + cbn [fst snd]. rewrite (rename_static_const_carried r r' Hcarries).
      reflexivity.
  - destruct c as [params my_closure ret exn metadata body].
    intros r r' news Hcarries Hagree.
    cbn [rename_expr rename_handler rename_static_binding rename_code
         apart apart_handler apart_binding apart_code].
    rewrite (new_variable_renamed r r' news my_closure Hcarries), tl_news.
    pose proof
      (bind_variables_renamed params _ _ (tl news)
         (carries_bind_variable r r' my_closure
            (new_variable r news my_closure) Hcarries))
      as [Hbound Hcarries1].
    pose proof
      (bind_variables_rest params
         (bind_variable r my_closure (new_variable r news my_closure))
         (tl news))
      as Hrest.
    destruct
      (bind_variables
         (bind_variable r my_closure (new_variable r news my_closure))
         (tl news) params)
      as [[rest params'] r1].
    destruct
      (bind_variables
         (bind_variable r' (rename_variable p my_closure)
            (rename_variable p (new_variable r news my_closure)))
         (map (rename_variable p) (tl news))
         (map (rename_variable p) params))
      as [[rest2 params2] r2].
    cbn [fst snd] in Hbound, Hcarries1, Hrest |- *.
    injection Hbound as -> ->.
    assert (Hagree1 : agree rest)
      by (rewrite Hrest; apply agree_skipn, agree_tl; exact Hagree).
    rewrite (new_continuation_renamed r1 r2 rest ret Hcarries1 Hagree1),
      !tl_news.
    pose proof
      (carries_bind_continuation r1 r2 ret (new_continuation r1 rest ret)
         Hcarries1)
      as Hcarries2.
    rewrite
      (new_continuation_renamed _ _ (tl rest) exn Hcarries2
         (agree_tl rest Hagree1)).
    pose proof
      (carries_bind_continuation _ _ exn
         (new_continuation
            (bind_continuation r1 ret (new_continuation r1 rest ret))
            (tl rest) exn)
         Hcarries2)
      as Hcarries3.
    rewrite
      (apart_renamed body _ _ (tl (tl rest)) Hcarries3
         (agree_tl _ (agree_tl _ Hagree1))).
    match goal with
    | |- context [apart ?r4 ?l body] =>
      destruct (apart r4 l body) as [rest' body']
    end.
    reflexivity.
Qed.

End Carried.

(* ======================================================================== *)
(* The new names of the term and those of the renamed term                  *)
(* ======================================================================== *)

(* Two lists of different names, as long as each other and outside a
   set: a permutation that fixes the set sends the first to the second.
   It goes by a third list that has no name of either. *)
Lemma exchange (fixed names names' : list positive) :
  length names = length names' ->
  distinct names = true ->
  distinct names' = true ->
  (∀ name : positive, among name names = true -> among name fixed = false)
  ->
  (∀ name : positive, among name names' = true -> among name fixed = false)
  ->
  ∃ swaps : permutation,
    fixes swaps fixed ∧ map (permute swaps) names = names'.
Proof.
  intros Hlength Hdistinct Hdistinct' Houtside Houtside'.
  pose proof (fresh_names_length (length names) (names ++ names' ++ fixed))
    as Hlength_between.
  destruct (fresh_names_outside (length names) (names ++ names' ++ fixed))
    as [Hdistinct_between Hbetween].
  set (between := fresh_names (length names) (names ++ names' ++ fixed))
    in Hlength_between, Hdistinct_between, Hbetween.
  assert (Hbetween_names :
    ∀ name : positive,
      among name between = true -> among name names = false)
    by (intros name Hname; apply Hbetween, among_app_false in Hname;
        exact (proj1 Hname)).
  assert (Hbetween_names' :
    ∀ name : positive,
      among name between = true -> among name names' = false).
  { intros name Hname. apply Hbetween, among_app_false in Hname.
    destruct Hname as [_ Hname]. apply among_app_false in Hname.
    exact (proj1 Hname). }
  assert (Hbetween_fixed :
    ∀ name : positive,
      among name between = true -> among name fixed = false).
  { intros name Hname. apply Hbetween, among_app_false in Hname.
    destruct Hname as [_ Hname]. apply among_app_false in Hname.
    exact (proj2 Hname). }
  exists (combine between names' ++ combine names between). split.
  - intros name Hname. rewrite permute_app.
    rewrite (fixes_combine fixed names between Houtside Hbetween_fixed name
               Hname).
    exact (fixes_combine fixed between names' Hbetween_fixed Houtside' name
             Hname).
  - rewrite (map_ext _ _ (permute_app _ _)),
      <- (map_map (permute (combine names between))
            (permute (combine between names'))).
    rewrite (permute_combine names between);
      [ | symmetry; exact Hlength_between | exact Hdistinct
      | split; [ exact Hdistinct_between | exact Hbetween_names ] ].
    apply permute_combine;
      [ rewrite Hlength_between; exact Hlength | exact Hdistinct_between | ].
    split; [ exact Hdistinct' | ]. intros name Hname.
    destruct (among name between) eqn:Hamong; [ | reflexivity ].
    apply Hbetween_names' in Hamong. congruence.
Qed.

Lemma map_same {A B : Type} (f g : A -> B) (l : list A) :
  map f l = map g l -> ∀ a : A, In a l -> f a = g a.
Proof.
  induction l as [ | b l IH]; simpl; intros Hsame a Ha; [ contradiction | ].
  injection Hsame as Hhead Htail.
  destruct Ha as [-> | Ha]; [ exact Hhead | exact (IH Htail a Ha) ].
Qed.

(* Of the first candidate. For a renaming of the term there is one that
   does to every name of the term, and to every continuation to avoid,
   what it does, with which the first candidate of the renamed term is
   the renamed first candidate. The candidate that
   [rename_apart_avoiding] takes now has not this
   (FlambdaRenameApartCanonicalSpec
   .canonical_avoiding_not_equivariant_up_to_names_statement); what it
   has is in FlambdaRenameApartCanonicalProofs.v. *)
Lemma renamed_apart_avoiding_equivariant_up_to_names (r : renaming)
    (avoid : list continuation) (e : expr) :
  ∃ r' : renaming,
    (∀ x : variable,
      among x (variables_of e) = true ->
      rename_variable r' x = rename_variable r x)
    ∧ (∀ k : continuation,
        among k (continuations_of e) = true ->
        rename_continuation r' k = rename_continuation r k)
    ∧ (∀ k : continuation,
        among k avoid = true ->
        rename_continuation r' k = rename_continuation r k)
    ∧ renamed_apart_avoiding (map (rename_continuation r) avoid)
        (rename_expr r e)
      = rename_expr r' (renamed_apart_avoiding avoid e).
Proof.
  pose proof (fresh_names_length (binders e) (avoid ++ names_of e))
    as Hlength.
  destruct (fresh_names_outside (binders e) (avoid ++ names_of e))
    as [Hdistinct Houtside].
  pose proof
    (fresh_names_length (binders (rename_expr r e))
       (map (rename_continuation r) avoid ++ names_of (rename_expr r e)))
    as Hlength'.
  destruct
    (fresh_names_outside (binders (rename_expr r e))
       (map (rename_continuation r) avoid ++ names_of (rename_expr r e)))
    as [Hdistinct' Houtside'].
  assert (Hcandidate :
    renamed_apart_avoiding avoid e
    = snd
        (apart identity_renaming
           (fresh_names (binders e) (avoid ++ names_of e)) e))
    by reflexivity.
  assert (Hcandidate' :
    renamed_apart_avoiding (map (rename_continuation r) avoid)
      (rename_expr r e)
    = snd
        (apart identity_renaming
           (fresh_names (binders (rename_expr r e))
              (map (rename_continuation r) avoid
               ++ names_of (rename_expr r e)))
           (rename_expr r e)))
    by reflexivity.
  revert Hlength Hdistinct Houtside Hlength' Hdistinct' Houtside' Hcandidate
    Hcandidate'.
  generalize (fresh_names (binders e) (avoid ++ names_of e)) as news.
  generalize
    (fresh_names (binders (rename_expr r e))
       (map (rename_continuation r) avoid ++ names_of (rename_expr r e)))
    as news'.
  intros news' news Hlength Hdistinct Houtside Hlength' Hdistinct' Houtside'
    Hcandidate Hcandidate'.
  rewrite binders_renamed in Hlength'.
  unfold names_of in Houtside, Houtside'.
  rewrite variables_of_renamed, continuations_of_renamed in Houtside'.
  (* The variables. *)
  destruct
    (exchange (map (rename_variable r) (variables_of e))
       (map (rename_variable r) news) news')
    as (swaps_v & Hfixes_v & Hmoves_v).
  { rewrite length_map.
    transitivity (binders e); [ exact Hlength | symmetry; exact Hlength' ]. }
  { rewrite (distinct_map_injective _ (rename_variable_eqb r)).
    exact Hdistinct. }
  { exact Hdistinct'. }
  { intros name Hname. apply among_In, in_map_iff in Hname.
    destruct Hname as (a & <- & Ha).
    rewrite (among_map_injective _ (rename_variable_eqb r)).
    apply among_In, Houtside, among_app_false in Ha.
    destruct Ha as [_ Ha]. apply among_app_false in Ha. exact (proj1 Ha). }
  { intros name Hname. apply Houtside', among_app_false in Hname.
    destruct Hname as [_ Hname]. apply among_app_false in Hname.
    exact (proj1 Hname). }
  (* The continuations, those to avoid among them. *)
  destruct
    (exchange (map (rename_continuation r) (avoid ++ continuations_of e))
       (map (rename_continuation r) news) news')
    as (swaps_k & Hfixes_k & Hmoves_k).
  { rewrite length_map.
    transitivity (binders e); [ exact Hlength | symmetry; exact Hlength' ]. }
  { rewrite (distinct_map_injective _ (rename_continuation_eqb r)).
    exact Hdistinct. }
  { exact Hdistinct'. }
  { intros name Hname. apply among_In, in_map_iff in Hname.
    destruct Hname as (a & <- & Ha).
    rewrite (among_map_injective _ (rename_continuation_eqb r)).
    apply among_In, Houtside, among_app_false in Ha.
    destruct Ha as [Havoid Ha]. apply among_app_false in Ha.
    apply among_app_false. split; [ exact Havoid | exact (proj2 Ha) ]. }
  { intros name Hname. apply Houtside', among_app_false in Hname.
    destruct Hname as [Havoid Hname]. apply among_app_false in Hname.
    rewrite map_app. apply among_app_false.
    split; [ exact Havoid | exact (proj2 Hname) ]. }
  (* The renaming: [r], then the swaps. *)
  set (s :=
    Renaming (swaps_v ++ variable_permutation r)
      (swaps_k ++ continuation_permutation r)).
  assert (Hs_v :
    ∀ x : variable,
      rename_variable s x = permute swaps_v (rename_variable r x))
    by (intros x; apply permute_app).
  assert (Hs_k :
    ∀ k : continuation,
      rename_continuation s k = permute swaps_k (rename_continuation r k))
    by (intros k; apply permute_app).
  assert (Hsame_v :
    ∀ x : variable,
      In x (variables_of e) -> rename_variable r x = rename_variable s x).
  { intros x Hx. rewrite Hs_v. symmetry. apply Hfixes_v, among_In, in_map, Hx.
  }
  assert (Hsame_fixed :
    ∀ k : continuation,
      In k (avoid ++ continuations_of e) ->
      rename_continuation r k = rename_continuation s k).
  { intros k Hk. rewrite Hs_k. symmetry.
    apply Hfixes_k, among_In, in_map, Hk. }
  assert (Hsame_k :
    ∀ k : continuation,
      In k (continuations_of e) ->
      rename_continuation r k = rename_continuation s k)
    by (intros k Hk; apply Hsame_fixed, in_or_app; right; exact Hk).
  assert (Hnews_v : map (rename_variable s) news = news').
  { rewrite (map_ext _ _ Hs_v),
      <- (map_map (rename_variable r) (permute swaps_v)).
    exact Hmoves_v. }
  assert (Hnews_k : map (rename_continuation s) news = news').
  { rewrite (map_ext _ _ Hs_k),
      <- (map_map (rename_continuation r) (permute swaps_k)).
    exact Hmoves_k. }
  assert (Hagree : agree s news).
  { intros a Ha.
    exact
      (map_same (rename_variable s) (rename_continuation s) news
         (eq_trans Hnews_v (eq_sym Hnews_k)) a Ha). }
  pose proof (rename_expr_agree r s e Hsame_v Hsame_k) as Hsame.
  assert (Hrenamed :
    renamed_apart_avoiding (map (rename_continuation r) avoid)
      (rename_expr r e)
    = rename_expr s (renamed_apart_avoiding avoid e)).
  { rewrite Hcandidate', Hcandidate, <- Hnews_v, Hsame.
    rewrite
      (apart_renamed s e identity_renaming identity_renaming news
         (carries_identity s) Hagree).
    reflexivity. }
  exists s. split; [ | split; [ | split ] ].
  - intros x Hx. symmetry. apply Hsame_v, among_In, Hx.
  - intros k Hk. symmetry. apply Hsame_k, among_In, Hk.
  - intros k Hk. symmetry. apply Hsame_fixed, in_or_app. left.
    apply among_In, Hk.
  - exact Hrenamed.
Qed.

(* ======================================================================== *)
(* The free names of a term are names of it                                 *)
(* ======================================================================== *)

Lemma var_free_in_simple_variables (x : variable) (s : simple) :
  var_free_in_simple x s -> In x (variables_in_simple s).
Proof.
  destruct s as [y | sym | c]; simpl;
    [ intros ->; left; reflexivity | tauto | tauto ].
Qed.

Lemma var_free_in_simples_variables (x : variable) (args : list simple) :
  var_free_in_simples x args -> In x (variables_in_simples args).
Proof.
  unfold var_free_in_simples, variables_in_simples. intros Hx.
  apply Exists_exists in Hx. destruct Hx as (s & Hs & Hx).
  apply in_flat_map. exists s.
  split; [ exact Hs | apply var_free_in_simple_variables, Hx ].
Qed.

Lemma var_free_in_slots_variables (x : variable)
    (slots : list (value_slot * simple)) :
  Exists (fun slot => var_free_in_simple x (snd slot)) slots ->
  In x (variables_in_slots slots).
Proof.
  unfold variables_in_slots. intros Hx.
  apply Exists_exists in Hx. destruct Hx as (slot & Hslot & Hx).
  apply in_flat_map. exists slot.
  split; [ exact Hslot | apply var_free_in_simple_variables, Hx ].
Qed.

Lemma var_free_in_named_variables (x : variable) (d : named) :
  var_free_in_named x d -> In x (variables_in_named d).
Proof.
  destruct d; simpl;
    first
      [ apply var_free_in_simple_variables
      | apply var_free_in_simples_variables
      | apply var_free_in_slots_variables ].
Qed.

Lemma var_free_in_static_const_variables (x : variable)
    (const : static_const) :
  var_free_in_static_const x const -> In x (variables_in_static_const const).
Proof.
  destruct const; simpl;
    first
      [ apply var_free_in_simple_variables
      | apply var_free_in_simples_variables
      | apply var_free_in_slots_variables | tauto ].
Qed.

Lemma var_free_in_binding_variables (x : variable)
    (binding : static_binding) :
  var_free_in_static_binding x binding -> In x (variables_in_binding binding).
Proof.
  destruct binding as [id c | id | sym const]; simpl;
    [ tauto | tauto | apply var_free_in_static_const_variables ].
Qed.

Fixpoint var_free_in_variables_of (x : variable) (e : expr) :
  var_free_in_expr x e -> In x (variables_of e)
with var_free_in_variables_in_handler (x : variable)
    (h : continuation_handler) :
  var_free_in_handler x h -> In x (variables_in_handler h).
Proof.
  - destruct e
      as [y kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [var_free_in_expr var_free_in_handler variables_of
           variables_in_handler variables_in_binding variables_in_code];
      intros Hx.
    + right. apply in_or_app.
      destruct Hx as [Hx | [_ Hx]];
        [ left; apply var_free_in_named_variables, Hx
        | right; exact (var_free_in_variables_of x body Hx) ].
    + apply in_or_app.
      destruct Hx as [Hx | Hx];
        [ left | right; exact (var_free_in_variables_of x body Hx) ].
      apply Exists_exists in Hx. destruct Hx as (binding & Hin & Hx).
      apply in_flat_map. exists binding.
      split; [ exact Hin | apply var_free_in_binding_variables, Hx ].
    + apply in_or_app.
      destruct Hx as [Hx | Hx];
        [ left; exact (var_free_in_variables_in_handler x h Hx)
        | right; exact (var_free_in_variables_of x body Hx) ].
    + apply in_or_app.
      destruct Hx as [Hx | Hx];
        [ left | right; exact (var_free_in_variables_of x body Hx) ].
      induction handlers as [ | [k h] handlers IH];
        cbn [any_handler flat_map snd] in Hx |- *; [ contradiction | ].
      apply in_or_app.
      destruct Hx as [Hx | Hx];
        [ left; exact (var_free_in_variables_in_handler x h Hx)
        | right; exact (IH Hx) ].
    + unfold variables_in_apply. apply in_or_app.
      destruct Hx as [Hx | [Hx | Hx]];
        [ left; apply var_free_in_simple_variables, Hx
        | right; apply in_or_app; left;
          apply var_free_in_simples_variables, Hx
        | right; apply in_or_app; right;
          apply var_free_in_simples_variables, Hx ].
    + apply var_free_in_simples_variables, Hx.
    + apply in_or_app.
      destruct Hx as [Hx | Hx];
        [ left; apply var_free_in_simple_variables, Hx | right ].
      apply Exists_exists in Hx. destruct Hx as ([i ac] & Hin & Hx).
      apply in_flat_map. exists (i, ac).
      split; [ exact Hin | apply var_free_in_simples_variables, Hx ].
    + contradiction.
  - destruct h as [params metadata handler].
    cbn [var_free_in_expr var_free_in_handler variables_of
         variables_in_handler variables_in_binding variables_in_code].
    intros [_ Hx]. apply in_or_app. right.
    exact (var_free_in_variables_of x handler Hx).
Qed.

Fixpoint cont_free_in_continuations_of (k : continuation) (e : expr) :
  cont_free_in_expr k e -> In k (continuations_of e)
with cont_free_in_continuations_in_handler (k : continuation)
    (h : continuation_handler) :
  cont_free_in_handler k h -> In k (continuations_in_handler h).
Proof.
  - destruct e
      as [y kind d body | bindings body | bound h body | handlers body | ap
         | ac | scrutinee arms | ];
      cbn [cont_free_in_expr cont_free_in_handler continuations_of
           continuations_in_handler continuations_in_binding
           continuations_in_code];
      intros Hk.
    + exact (cont_free_in_continuations_of k body Hk).
    + apply in_or_app. right.
      exact (cont_free_in_continuations_of k body Hk).
    + right. apply in_or_app.
      destruct Hk as [Hk | [_ Hk]];
        [ left; exact (cont_free_in_continuations_in_handler k h Hk)
        | right; exact (cont_free_in_continuations_of k body Hk) ].
    + destruct Hk as [_ Hk]. apply in_or_app.
      destruct Hk as [Hk | Hk];
        [ left | right; exact (cont_free_in_continuations_of k body Hk) ].
      induction handlers as [ | [bound h] handlers IH];
        cbn [any_handler flat_map fst snd] in Hk |- *; [ contradiction | ].
      right. apply in_or_app.
      destruct Hk as [Hk | Hk];
        [ left; exact (cont_free_in_continuations_in_handler k h Hk)
        | right; exact (IH Hk) ].
    + unfold continuations_in_apply.
      destruct Hk as [-> | ->];
        [ left; reflexivity | right; left; reflexivity ].
    + exact Hk.
    + apply Exists_exists in Hk. destruct Hk as ([i ac] & Hin & Hk).
      apply in_flat_map. exists (i, ac). split; [ exact Hin | exact Hk ].
    + contradiction.
  - destruct h as [params metadata handler].
    cbn [cont_free_in_expr cont_free_in_handler continuations_of
         continuations_in_handler continuations_in_binding
         continuations_in_code].
    intros Hk. exact (cont_free_in_continuations_of k handler Hk).
Qed.

(* The free variables and continuations of a term, as
   FlambdaBindersSpec.v lists them, in its code too, are among the
   variables and continuations of the term. *)
Lemma free_variables_of_named_variables (d : named) :
  free_variables_of_named d = variables_in_named d.
Proof. destruct d; reflexivity. Qed.

Fixpoint free_variables_in_variables_of (e : expr) :
  incl (free_variables_of_expr e) (variables_of e)
with free_variables_in_variables_in_handler (h : continuation_handler) :
  incl (free_variables_of_handler h) (variables_in_handler h)
with free_variables_in_variables_in_binding (binding : static_binding) :
  incl (free_variables_of_binding binding) (variables_in_binding binding)
with free_variables_in_variables_in_code (c : code) :
  incl (free_variables_of_code c) (variables_in_code c).
Proof.
  - destruct e
      as [y kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [free_variables_of_expr free_variables_of_handler
           free_variables_of_binding free_variables_of_code variables_of
           variables_in_handler variables_in_binding variables_in_code].
    + intros x Hx. right. apply in_or_app. apply in_app_or in Hx.
      destruct Hx as [Hx | Hx];
        [ left; rewrite <- free_variables_of_named_variables; exact Hx
        | right ].
      unfold without in Hx. apply filter_In in Hx.
      exact (free_variables_in_variables_of body x (proj1 Hx)).
    + apply incl_app_app; [ | apply free_variables_in_variables_of ].
      induction bindings as [ | binding bindings IH]; cbn [flat_map];
        [ apply incl_refl | ].
      apply incl_app_app;
        [ apply free_variables_in_variables_in_binding | exact IH ].
    + apply incl_app_app;
        [ apply free_variables_in_variables_in_handler
        | apply free_variables_in_variables_of ].
    + apply incl_app_app; [ | apply free_variables_in_variables_of ].
      induction handlers as [ | [k h] handlers IH]; cbn [flat_map snd];
        [ apply incl_refl | ].
      apply incl_app_app;
        [ apply free_variables_in_variables_in_handler | exact IH ].
    + apply incl_refl.
    + apply incl_refl.
    + apply incl_refl.
    + apply incl_refl.
  - destruct h as [params metadata handler].
    cbn [free_variables_of_expr free_variables_of_handler
         free_variables_of_binding free_variables_of_code variables_of
         variables_in_handler variables_in_binding variables_in_code].
    intros x Hx. unfold without in Hx. apply filter_In in Hx.
    apply in_or_app. right.
    exact (free_variables_in_variables_of handler x (proj1 Hx)).
  - destruct binding as [id c | id | sym const];
      cbn [free_variables_of_expr free_variables_of_handler
           free_variables_of_binding free_variables_of_code variables_of
           variables_in_handler variables_in_binding variables_in_code].
    + apply free_variables_in_variables_in_code.
    + apply incl_refl.
    + destruct const; apply incl_refl.
  - destruct c as [params my_closure ret exn metadata body].
    cbn [free_variables_of_expr free_variables_of_handler
         free_variables_of_binding free_variables_of_code variables_of
         variables_in_handler variables_in_binding variables_in_code].
    intros x Hx. unfold without in Hx. apply filter_In in Hx.
    right. apply in_or_app. right.
    exact (free_variables_in_variables_of body x (proj1 Hx)).
Qed.

Fixpoint free_continuations_in_continuations_of (e : expr) :
  incl (free_continuations_of_expr e) (continuations_of e)
with free_continuations_in_continuations_in_handler
    (h : continuation_handler) :
  incl (free_continuations_of_handler h) (continuations_in_handler h)
with free_continuations_in_continuations_in_binding
    (binding : static_binding) :
  incl (free_continuations_of_binding binding)
    (continuations_in_binding binding)
with free_continuations_in_continuations_in_code (c : code) :
  incl (free_continuations_of_code c) (continuations_in_code c).
Proof.
  - destruct e
      as [y kind d body | bindings body | bound h body | handlers body | ap
         | ac | scrutinee arms | ];
      cbn [free_continuations_of_expr free_continuations_of_handler
           free_continuations_of_binding free_continuations_of_code
           continuations_of continuations_in_handler
           continuations_in_binding continuations_in_code].
    + apply free_continuations_in_continuations_of.
    + apply incl_app_app; [ | apply free_continuations_in_continuations_of ].
      induction bindings as [ | binding bindings IH]; cbn [flat_map];
        [ apply incl_refl | ].
      apply incl_app_app;
        [ apply free_continuations_in_continuations_in_binding | exact IH ].
    + intros j Hj. right. apply in_or_app. apply in_app_or in Hj.
      destruct Hj as [Hj | Hj];
        [ left;
          exact (free_continuations_in_continuations_in_handler h j Hj)
        | right ].
      unfold without in Hj. apply filter_In in Hj.
      exact (free_continuations_in_continuations_of body j (proj1 Hj)).
    + intros j Hj. unfold without in Hj. apply filter_In in Hj.
      destruct Hj as [Hj _]. apply in_app_or in Hj. apply in_or_app.
      destruct Hj as [Hj | Hj];
        [ left
        | right; exact (free_continuations_in_continuations_of body j Hj) ].
      revert Hj.
      induction handlers as [ | [bound h] handlers IH];
        cbn [flat_map fst snd]; intros Hj; [ destruct Hj | ].
      apply in_or_app. apply in_app_or in Hj.
      destruct Hj as [Hj | Hj];
        [ left; right;
          exact (free_continuations_in_continuations_in_handler h j Hj)
        | right; exact (IH Hj) ].
    + apply incl_refl.
    + apply incl_refl.
    + apply incl_refl.
    + apply incl_refl.
  - destruct h as [params metadata handler].
    cbn [free_continuations_of_expr free_continuations_of_handler
         free_continuations_of_binding free_continuations_of_code
         continuations_of continuations_in_handler continuations_in_binding
         continuations_in_code].
    apply free_continuations_in_continuations_of.
  - destruct binding as [id c | id | sym const];
      cbn [free_continuations_of_expr free_continuations_of_handler
           free_continuations_of_binding free_continuations_of_code
           continuations_of continuations_in_handler
           continuations_in_binding continuations_in_code].
    + apply free_continuations_in_continuations_in_code.
    + apply incl_refl.
    + apply incl_refl.
  - destruct c as [params my_closure ret exn metadata body].
    cbn [free_continuations_of_expr free_continuations_of_handler
         free_continuations_of_binding free_continuations_of_code
         continuations_of continuations_in_handler continuations_in_binding
         continuations_in_code].
    intros j Hj. unfold without in Hj. apply filter_In in Hj.
    right. right.
    exact (free_continuations_in_continuations_of body j (proj1 Hj)).
Qed.
