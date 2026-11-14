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

(* Lemmas for the proofs of the statements of FlambdaRegionsCheckedSpec.v:
   the parameters of a handler that is entered, one by one: what
   [bind_params], [bind_entry_params], [region_args] and [value_args] say
   of the parameter at a place. Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool Lia PeanoNat.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaHeapProofs
  FlambdaSemanticsSpec FlambdaClosuresProofs FlambdaRegionsSpec
  FlambdaRegionsProofs FlambdaRegionsReadsProofs FlambdaRegionsInvariantSpec
  FlambdaRegionsInvariantProofs FlambdaRegionsErasureSpec
  FlambdaRegionsCallsSpec FlambdaRegionsCheckSpec FlambdaRegionsCheckedSpec.
From OxCamlFlambda2 Require Import FlambdaRegionsCheckedBasicsProofs
  FlambdaRegionsCheckedStableProofs FlambdaRegionsCheckedMarksProofs
  FlambdaRegionsCheckedOutlivesProofs.
Import ListNotations.

Lemma nth_error_same_length {A B : Type} (l : list A) (l' : list B) :
  length l = length l' ->
  ∀ (i : nat) (a : A),
    nth_error l i = Some a -> ∃ b : B, nth_error l' i = Some b.
Proof.
  intros Hlength i a Hi.
  destruct (nth_error l' i) as [b | ] eqn:Hi'; [ exists b; reflexivity | ].
  apply nth_error_None in Hi'.
  pose proof (proj1 (nth_error_Some l i) ltac:(rewrite Hi; discriminate)).
  lia.
Qed.

Lemma Forall2_nth_error {A B : Type} (R : A -> B -> Prop) (l : list A)
    (l' : list B) :
  Forall2 R l l' ->
  ∀ (i : nat) (a : A) (b : B),
    nth_error l i = Some a -> nth_error l' i = Some b -> R a b.
Proof.
  induction 1 as [ | a0 b0 l l' Hab _ IH]; intros [ | i] a b; cbn [nth_error];
    try discriminate.
  - intros [= <-] [= <-]. exact Hab.
  - exact (IH i a b).
Qed.

(* ======================================================================== *)
(* The variables after the parameters are bound                             *)
(* ======================================================================== *)

Lemma bind_params_nth (params : list variable) :
  ∀ (vs : list value) (vars vars' : env),
    distinct params = true ->
    bind_params params vs vars = Some vars' ->
    length vs = length params
    ∧ ∀ (i : nat) (x : variable) (v : value),
        nth_error params i = Some x ->
        nth_error vs i = Some v ->
        lookup x vars' = Some v.
Proof.
  induction params as [ | p params IH]; intros [ | v vs] vars vars';
    cbn [bind_params distinct]; try discriminate.
  - intros _ _. split; [ reflexivity | ]. intros [ | i]; discriminate.
  - intros Hdistinct Hbind. apply andb_true_iff in Hdistinct as [Hp Hrest].
    apply negb_true_iff in Hp.
    destruct (IH vs _ vars' Hrest Hbind) as [Hlength Hnth].
    split; [ cbn [length]; rewrite Hlength; reflexivity | ].
    intros [ | i] x w; cbn [nth_error].
    + intros [= <-] [= <-].
      rewrite (bind_params_lookup_other p params vs _ vars' Hp Hbind).
      apply lookup_head.
    + exact (Hnth i x w).
Qed.

(* ======================================================================== *)
(* The lifetimes of the parameters                                          *)
(* ======================================================================== *)

(* The lifetime that an entry gives a parameter of a kind. *)
Definition param_lifetime (he_open : list variable) (kind : flambda_kind)
    (w : lifetime) : lifetime :=
  if kind_eqb kind Region then Gone
  else
    match w with
    | Of_region r => if among r he_open then w else Gone
    | Forever | Of_caller | Gone => w
    end.

Lemma bind_entry_params_other (he_open : list variable) (x : variable)
    (params : list variable) :
  ∀ (kinds : list flambda_kind) (wanted : list lifetime)
    (env : list (variable * lifetime)),
    among x params = false ->
    lookup x (bind_entry_params he_open params kinds wanted env)
    = lookup x env.
Proof.
  induction params as [ | p params IH]; intros kinds wanted env;
    [ reflexivity | ].
  rewrite among_cons. intros H. apply orb_false_iff in H as [Hxp Hrest].
  cbn [bind_entry_params].
  destruct kinds as [ | kind kinds]; [ reflexivity | ].
  destruct wanted as [ | w wanted]; [ reflexivity | ].
  rewrite lookup_cons, Hxp. exact (IH kinds wanted env Hrest).
Qed.

Lemma bind_entry_params_nth (he_open : list variable)
    (params : list variable) :
  ∀ (kinds : list flambda_kind) (wanted : list lifetime)
    (env : list (variable * lifetime)) (i : nat) (x : variable)
    (kind : flambda_kind) (w : lifetime),
    distinct params = true ->
    nth_error params i = Some x ->
    nth_error kinds i = Some kind ->
    nth_error wanted i = Some w ->
    lookup x (bind_entry_params he_open params kinds wanted env)
    = Some (param_lifetime he_open kind w).
Proof.
  induction params as [ | p params IH];
    intros kinds wanted env i x kind w Hdistinct;
    [ destruct i; discriminate | ].
  cbn [distinct] in Hdistinct. apply andb_true_iff in Hdistinct as [Hp Hrest].
  apply negb_true_iff in Hp.
  destruct kinds as [ | kind0 kinds]; [ destruct i; discriminate | ].
  destruct wanted as [ | w0 wanted];
    [ destruct i; intros _ _ H; discriminate H | ].
  cbn [bind_entry_params]. destruct i as [ | i]; cbn [nth_error].
  - intros [= <-] [= <-] [= <-]. apply lookup_head.
  - intros Hx Hkind Hw. rewrite lookup_cons.
    assert (Hne : Pos.eqb x p = false).
    { destruct (Pos.eqb x p) eqn:Heq; [ | reflexivity ].
      apply Pos.eqb_eq in Heq. subst x.
      apply nth_error_In in Hx. apply among_In in Hx. congruence. }
    rewrite Hne. exact (IH kinds wanted env i x kind w Hrest Hx Hkind Hw).
Qed.

(* ======================================================================== *)
(* The parameters that are regions                                          *)
(* ======================================================================== *)

Lemma region_params_of_in (params : list variable) :
  ∀ (kinds : list flambda_kind) (r : variable),
    In r (region_params_of params kinds) ->
    ∃ (i : nat) (kind : flambda_kind),
      nth_error params i = Some r
      ∧ nth_error kinds i = Some kind
      ∧ kind_eqb kind Region = true.
Proof.
  induction params as [ | p params IH]; intros kinds r; [ contradiction | ].
  destruct kinds as [ | kind kinds]; [ contradiction | ].
  cbn [region_params_of]. destruct (kind_eqb kind Region) eqn:Hkind.
  - intros [<- | Hin].
    + exists 0, kind.
      split; [ reflexivity | split; [ reflexivity | exact Hkind ] ].
    + destruct (IH kinds r Hin) as (i & kind' & H1 & H2 & H3).
      exists (S i), kind'. repeat split; assumption.
  - intros Hin. destruct (IH kinds r Hin) as (i & kind' & H1 & H2 & H3).
    exists (S i), kind'. repeat split; assumption.
Qed.

Lemma region_params_of_among (params : list variable) :
  ∀ (kinds : list flambda_kind) (r : variable),
    In r (region_params_of params kinds) -> In r params.
Proof.
  intros kinds r Hin.
  destruct (region_params_of_in params kinds r Hin) as (i & _ & Hi & _).
  exact (nth_error_In params i Hi).
Qed.

Lemma region_params_of_NoDup (params : list variable) :
  ∀ kinds : list flambda_kind,
    NoDup params -> NoDup (region_params_of params kinds).
Proof.
  induction params as [ | p params IH]; intros kinds Hnodup;
    [ constructor | ].
  inversion Hnodup as [ | p0 rest Hnotin Hrest]; subst.
  destruct kinds as [ | kind kinds]; [ constructor | ].
  cbn [region_params_of]. destruct (kind_eqb kind Region);
    [ | exact (IH kinds Hrest) ].
  constructor; [ | exact (IH kinds Hrest) ].
  intros Hin. exact (Hnotin (region_params_of_among params kinds p Hin)).
Qed.

(* The regions given: one for each parameter that is a region, the
   variable of an open region that the argument at its place is. *)
Lemma region_args_spec (cs : check_state) (params : list variable) :
  ∀ (kinds : list flambda_kind) (args : list (option simple * lifetime))
    (given : list (variable * variable)),
    region_args cs params kinds args = Some given ->
    length kinds = length params
    ∧ length args = length params
    ∧ map fst given = region_params_of params kinds
    ∧ ∀ (i : nat) (p : variable) (kind : flambda_kind) (s : option simple)
        (l : lifetime),
        nth_error params i = Some p ->
        nth_error kinds i = Some kind ->
        nth_error args i = Some (s, l) ->
        kind_eqb kind Region = true ->
        ∃ r : variable,
          s = Some (Var r)
          ∧ among r (cs_open cs) = true
          ∧ In (p, r) given.
Proof.
  induction params as [ | p params IH]; intros kinds args given.
  - destruct kinds; [ | discriminate ]. destruct args; [ | discriminate ].
    intros [= <-]. repeat split; try reflexivity.
    intros [ | i]; discriminate.
  - destruct kinds as [ | kind kinds]; [ discriminate | ].
    destruct args as [ | [s l] args]; [ discriminate | ].
    cbn [region_args].
    destruct (region_args cs params kinds args) as [rest | ] eqn:Hrest;
      [ | discriminate ].
    destruct (IH kinds args rest Hrest) as (Hk & Ha & Hkeys & Hnth).
    cbn [region_params_of length]. simpl.
    destruct (kind_eqb kind Region) eqn:Hkind.
    + destruct s as [[r | sym | c] | ]; try discriminate.
      destruct (among r (cs_open cs)) eqn:Hr; [ | discriminate ].
      intros [= <-]. split; [ congruence | ]. split; [ congruence | ].
      split; [ cbn [map fst]; rewrite Hkeys; reflexivity | ].
      intros [ | i] p0 kind0 s0 l0; cbn [nth_error].
      * intros [= <-] [= <-] [= <- <-] _. exists r.
        split; [ reflexivity | ]. split; [ exact Hr | left; reflexivity ].
      * intros H1 H2 H3 H4.
        destruct (Hnth i p0 kind0 s0 l0 H1 H2 H3 H4) as (r0 & -> & Hr0 & Hin).
        exists r0. split; [ reflexivity | ].
        split; [ exact Hr0 | right; exact Hin ].
    + intros [= <-]. split; [ congruence | ]. split; [ congruence | ].
      split; [ exact Hkeys | ].
      intros [ | i] p0 kind0 s0 l0; cbn [nth_error].
      * intros _ [= <-] _ H. congruence.
      * exact (Hnth i p0 kind0 s0 l0).
Qed.

(* The arguments at the parameters that are not regions. *)
Lemma value_args_nth (cs : check_state) (given : list (variable * variable))
    (named : list variable) (kinds : list flambda_kind) :
  ∀ (args : list (option simple * lifetime)) (wanted : list lifetime)
    (i : nat) (kind : flambda_kind) (s : option simple) (l w : lifetime),
    value_args cs given named kinds args wanted = true ->
    nth_error kinds i = Some kind ->
    nth_error args i = Some (s, l) ->
    nth_error wanted i = Some w ->
    kind_eqb kind Region = true
    ∨ (lifetime_usable cs l = true
       ∧ outlives named (cs_open cs) l (lifetime_at_jump given w) = true).
Proof.
  induction kinds as [ | kind0 kinds IH]; intros args wanted i kind s l w;
    [ destruct i; discriminate | ].
  destruct args as [ | [s0 l0] args]; [ discriminate | ].
  destruct wanted as [ | w0 wanted]; [ discriminate | ].
  cbn [value_args]. intros H. apply andb_true_iff in H as [Hhead Hrest].
  destruct i as [ | i]; cbn [nth_error].
  - intros [= <-] [= <- <-] [= <-].
    apply orb_true_iff in Hhead as [H | H]; [ left; exact H | right ].
    apply andb_true_iff in H. exact H.
  - exact (IH args wanted i kind s l w Hrest).
Qed.

Lemma value_args_length (cs : check_state) (given : list (variable * variable))
    (named : list variable) (kinds : list flambda_kind) :
  ∀ (args : list (option simple * lifetime)) (wanted : list lifetime),
    value_args cs given named kinds args wanted = true ->
    length wanted = length kinds.
Proof.
  induction kinds as [ | kind kinds IH]; intros args wanted.
  - destruct args; [ | discriminate ]. destruct wanted; [ | discriminate ].
    reflexivity.
  - destruct args as [ | [s l] args]; [ discriminate | ].
    destruct wanted as [ | w wanted]; [ discriminate | ].
    cbn [value_args length]. intros H. apply andb_true_iff in H as [_ H].
    rewrite (IH args wanted H). reflexivity.
Qed.
