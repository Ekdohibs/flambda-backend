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


(* Proofs of the statements of FlambdaTwinParamsSpec.v. Nothing here
   needs review.

   The term and its result run side by side under a relation of
   states ([state_rel]): the same expression, where neither run is in
   a term that the pass rewrote; or an expression that the test
   accepts and the same rewritten, under continuations that are
   closures of a handler and of that handler rewritten. The relation
   has a number, the [Let]s at the head of a handler that the term
   has still to run and the result has not: the term runs each alone,
   and the two runs are in step everywhere else, so that the result
   refines the term by FlambdaLaggingProofs.counted_lagging_refines.

   The variables ([vars_rel]): while such [Let]s are still to come,
   the two runs agree on every variable but those that the [Let]s
   bind, which are parameters in the result and hold there already
   what the parameters that the [Let]s read hold; once a [Let] has
   run they agree on its variable too. A variable that has a twin
   holds in the result what its twin holds, so an argument and the
   untagging of the one are those of the other. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaPrimitiveModel FlambdaValueSpec
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaLaggingProofs
  FlambdaNormaliseSpec FlambdaNormaliseProofs FlambdaUnusedParamsProofs
  FlambdaUntagParamsSpec FlambdaUntagParamsProofs FlambdaTwinParamsSpec.
Import ListNotations.

(* ======================================================================== *)
(* Lists                                                                    *)
(* ======================================================================== *)

Lemma combine_append {A B : Type} (l : list A) :
  ∀ (l' : list A) (m m' : list B),
    length l = length m ->
    combine (l ++ l') (m ++ m') = combine l m ++ combine l' m'.
Proof.
  induction l as [ | a l IH]; intros l' [ | b m] m' Hlength;
    try discriminate Hlength; [ reflexivity | ].
  injection Hlength as Hlength. cbn [app combine].
  rewrite (IH l' m m' Hlength). reflexivity.
Qed.

Lemma Forall2_impl_among (P Q : variable -> variable -> Prop)
    (l l' : list variable) :
  (∀ a b : variable,
    among a l = true -> among b l' = true -> P a b -> Q a b) ->
  Forall2 P l l' ->
  Forall2 Q l l'.
Proof.
  intros Himpl Hall. induction Hall as [ | a b l l' Hab Hall IH];
    constructor.
  - apply Himpl; [ | | exact Hab ]; rewrite among_cons, Pos.eqb_refl;
      reflexivity.
  - apply IH. intros a' b' Ha' Hb'.
    apply Himpl; rewrite among_cons;
      [ rewrite Ha' | rewrite Hb' ]; apply orb_true_r.
Qed.

(* The parts of a list without repetition, and a list without one of
   its elements. *)
Lemma distinct_append (l : list variable) :
  ∀ m : list variable,
    distinct (l ++ m) = true ->
    distinct l = true
    ∧ distinct m = true
    ∧ ∀ z : variable, among z m = true -> among z l = false.
Proof.
  induction l as [ | a l IH]; intros m; cbn [app].
  - intros Hm. split; [ reflexivity | split; [ exact Hm | ] ].
    intros z _. reflexivity.
  - cbn [distinct]. intros H. apply andb_true_iff in H as [Ha H].
    apply negb_true_iff in Ha. rewrite among_app in Ha.
    apply orb_false_iff in Ha as [Hal Ham].
    destruct (IH m H) as (Hl & Hm & Hapart).
    split; [ rewrite Hal; exact Hl | split; [ exact Hm | ] ].
    intros z Hz. rewrite among_cons, (Hapart z Hz), orb_false_r.
    destruct (Pos.eqb_spec z a) as [-> | _]; [ congruence | reflexivity ].
Qed.

Lemma distinct_middle (l : list variable) (a : variable)
    (m : list variable) :
  distinct (l ++ a :: m) = true ->
  among a (l ++ m) = false ∧ distinct (l ++ m) = true.
Proof.
  intros H. apply distinct_NoDup in H.
  apply NoDup_remove in H as [Hnodup Hnotin].
  split; [ | apply distinct_NoDup; exact Hnodup ].
  destruct (among a (l ++ m)) eqn:Ha; [ | reflexivity ].
  apply among_In in Ha. contradiction.
Qed.

(* ======================================================================== *)
(* The parameters of a handler that has twins                               *)
(* ======================================================================== *)

Lemma renamed_from_length (ts : list nat) (params : list variable) :
  ∀ (i : nat) (names : list variable),
    length (renamed_from ts i params names) = length params.
Proof.
  induction params as [ | p params IH]; intros i names; cbn [renamed_from];
    [ reflexivity | ].
  destruct (at_position ts i); [ destruct names as [ | x names] | ];
    cbn [length]; rewrite IH; reflexivity.
Qed.

(* The names are among the parameters renamed. *)
Lemma among_names_renamed (ts : list nat) (x : variable)
    (params : list variable) :
  ∀ (i : nat) (names : list variable),
    length names = length (moved_from ts i params) ->
    among x names = true ->
    among x (renamed_from ts i params names) = true.
Proof.
  induction params as [ | p params IH]; intros i names;
    cbn [moved_from renamed_from].
  - destruct names; simpl; [ intros _ H; exact H | discriminate ].
  - destruct (at_position ts i).
    + destruct names as [ | y names]; cbn [length]; [ discriminate | ].
      intros [= Hlength]. rewrite !among_cons.
      destruct (Pos.eqb x y); [ reflexivity | exact (IH (S i) names Hlength) ].
    + intros Hlength Hx. rewrite among_cons, (IH (S i) names Hlength Hx).
      apply orb_true_r.
Qed.

(* A parameter is among the parameters renamed or among those moved. *)
Lemma among_twin_params (ts : list nat) (z : variable)
    (params : list variable) :
  ∀ (i : nat) (names : list variable),
    among z params = true ->
    among z (renamed_from ts i params names ++ moved_from ts i params)
    = true.
Proof.
  induction params as [ | p params IH]; intros i names;
    [ simpl; discriminate | ].
  rewrite among_cons. cbn [renamed_from moved_from].
  destruct (Pos.eqb_spec z p) as [-> | Hne]; cbn [orb]; intros Hz.
  - destruct (at_position ts i).
    + rewrite among_app, (among_cons p p), Pos.eqb_refl. cbn [orb].
      apply orb_true_r.
    + cbn [app]. rewrite among_cons, Pos.eqb_refl. reflexivity.
  - destruct (at_position ts i).
    + pose proof (IH (S i) (tl names) Hz) as H. rewrite among_app in H.
      rewrite among_app, (among_cons z p).
      destruct names as [ | x names]; rewrite among_cons; cbn [tl] in H;
        apply orb_true_iff in H as [H | H]; rewrite H; rewrite ?orb_true_r;
        reflexivity.
    + cbn [app]. rewrite among_cons, (IH (S i) names Hz). apply orb_true_r.
Qed.

(* The parameters of the handler were distinct, if the new ones are. *)
Lemma distinct_twin_params (ts : list nat) (params : list variable) :
  ∀ (i : nat) (names : list variable),
    distinct (renamed_from ts i params names ++ moved_from ts i params)
    = true ->
    distinct params = true.
Proof.
  induction params as [ | p params IH]; intros i names;
    [ intros _; reflexivity | ].
  cbn [renamed_from moved_from distinct].
  destruct (at_position ts i).
  - intros H.
    assert (H' :
      distinct
        (renamed_from ts (S i) params (tl names)
         ++ p :: moved_from ts (S i) params)
      = true).
    { destruct names as [ | x names]; cbn [app distinct tl] in H |- *;
        apply andb_true_iff in H as [_ H]; exact H. }
    destruct (distinct_middle _ _ _ H') as [Hp Hrest].
    rewrite (IH (S i) (tl names) Hrest), andb_true_r.
    apply negb_true_iff. destruct (among p params) eqn:Hpp; [ | reflexivity ].
    rewrite (among_twin_params ts p params (S i) (tl names) Hpp) in Hp.
    discriminate Hp.
  - cbn [app distinct]. intros H. apply andb_true_iff in H as [Hp H].
    apply negb_true_iff in Hp.
    rewrite (IH (S i) names H), andb_true_r.
    apply negb_true_iff. destruct (among p params) eqn:Hpp; [ | reflexivity ].
    rewrite (among_twin_params ts p params (S i) names Hpp) in Hp.
    discriminate Hp.
Qed.

(* A variable that is not one of the names, nor one of the parameters
   moved, is bound by the parameters renamed as it was. *)
Lemma lookup_renamed_other (ts : list nat) (z : variable)
    (params : list variable) :
  ∀ (vs : list value) (i : nat) (names : list variable),
    length params = length vs ->
    among z names = false ->
    among z (moved_from ts i params) = false ->
    lookup z (combine (renamed_from ts i params names) vs)
    = lookup z (combine params vs).
Proof.
  induction params as [ | p params IH]; intros [ | v vs] i names Hlength;
    try discriminate Hlength; [ reflexivity | ].
  injection Hlength as Hlength. cbn [renamed_from moved_from].
  destruct (at_position ts i).
  - intros Hnames Hmoved. rewrite among_cons in Hmoved.
    apply orb_false_iff in Hmoved as [Hzp Hmoved].
    destruct names as [ | x names].
    + cbn [combine]. rewrite !lookup_cons, Hzp.
      exact (IH vs (S i) [] Hlength Hnames Hmoved).
    + rewrite among_cons in Hnames.
      apply orb_false_iff in Hnames as [Hzx Hnames].
      cbn [combine]. rewrite !lookup_cons, Hzx, Hzp.
      exact (IH vs (S i) names Hlength Hnames Hmoved).
  - intros Hnames Hmoved. cbn [combine]. rewrite !lookup_cons.
    destruct (Pos.eqb z p); [ reflexivity | ].
    exact (IH vs (S i) names Hlength Hnames Hmoved).
Qed.

(* A parameter moved is bound among those moved as it was among
   all. *)
Lemma lookup_moved_twin (ts : list nat) (z : variable)
    (params : list variable) :
  ∀ (vs : list value) (i : nat),
    distinct params = true ->
    length params = length vs ->
    among z (moved_from ts i params) = true ->
    lookup z (combine (moved_from ts i params) (moved_from ts i vs))
    = lookup z (combine params vs).
Proof.
  induction params as [ | p params IH]; intros [ | v vs] i Hdistinct Hlength;
    try discriminate Hlength; [ simpl; discriminate | ].
  cbn [distinct] in Hdistinct.
  apply andb_true_iff in Hdistinct as [Hp Hdistinct].
  apply negb_true_iff in Hp. injection Hlength as Hlength.
  cbn [moved_from]. destruct (at_position ts i).
  - cbn [combine]. rewrite among_cons, !lookup_cons.
    destruct (Pos.eqb z p); [ intros _; reflexivity | ].
    exact (IH vs (S i) Hdistinct Hlength).
  - intros Hmoved. cbn [combine]. rewrite lookup_cons.
    destruct (Pos.eqb_spec z p) as [-> | _];
      [ | exact (IH vs (S i) Hdistinct Hlength Hmoved) ].
    apply among_moved_from in Hmoved. congruence.
Qed.

(* The names are bound to what the parameters moved were. *)
Lemma lookup_renamed_names (ts : list nat) (params : list variable) :
  ∀ (vs : list value) (i : nat) (names : list variable),
    distinct params = true ->
    distinct (renamed_from ts i params names) = true ->
    length params = length vs ->
    length names = length (moved_from ts i params) ->
    Forall2
      (fun x p =>
         lookup x (combine (renamed_from ts i params names) vs)
         = lookup p (combine params vs))
      names (moved_from ts i params).
Proof.
  induction params as [ | p params IH];
    intros [ | v vs] i names Hdistinct Hrenamed Hlength;
    try discriminate Hlength; revert Hrenamed;
    cbn [moved_from renamed_from].
  - intros _. destruct names; simpl; [ intros _; constructor | discriminate ].
  - cbn [distinct] in Hdistinct.
    apply andb_true_iff in Hdistinct as [Hp Hdistinct].
    apply negb_true_iff in Hp. injection Hlength as Hlength.
    destruct (at_position ts i).
    + destruct names as [ | x names]; simpl length; [ discriminate | ].
      cbn [distinct]. intros Hrenamed [= Hnames].
      apply andb_true_iff in Hrenamed as [Hx Hrenamed].
      apply negb_true_iff in Hx. cbn [combine]. constructor.
      * rewrite !lookup_cons, !Pos.eqb_refl. reflexivity.
      * generalize (IH vs (S i) names Hdistinct Hrenamed Hlength Hnames).
        apply Forall2_impl_among. intros x' p' Hx' Hp' Heq.
        rewrite !lookup_cons.
        destruct (Pos.eqb_spec x' x) as [-> | _].
        { rewrite (among_names_renamed ts x params (S i) names Hnames Hx')
            in Hx.
          discriminate Hx. }
        destruct (Pos.eqb_spec p' p) as [-> | _]; [ | exact Heq ].
        rewrite (among_moved_from ts p params (S i) Hp') in Hp.
        discriminate Hp.
    + cbn [distinct]. intros Hrenamed Hnames.
      apply andb_true_iff in Hrenamed as [Hpr Hrenamed].
      apply negb_true_iff in Hpr. cbn [combine].
      generalize (IH vs (S i) names Hdistinct Hrenamed Hlength Hnames).
      apply Forall2_impl_among. intros x' p' Hx' Hp' Heq.
      rewrite !lookup_cons.
      destruct (Pos.eqb_spec x' p) as [-> | _].
      { rewrite (among_names_renamed ts p params (S i) names Hnames Hx')
          in Hpr.
        discriminate Hpr. }
      destruct (Pos.eqb_spec p' p) as [-> | _]; [ | exact Heq ].
      rewrite (among_moved_from ts p params (S i) Hp') in Hp.
      discriminate Hp.
Qed.

(* ======================================================================== *)
(* The [Let]s at the head of a handler                                      *)
(* ======================================================================== *)

Lemma alias_of_some (d : named) (q : variable) :
  alias_of d = Some q -> d = Simple (Var q).
Proof.
  destruct d as [[x | sym | c] | | | | | | ]; simpl; try discriminate.
  intros [= ->]. reflexivity.
Qed.

Lemma twin_names_nil (e : expr) : twin_names [] e = [].
Proof. destruct e; reflexivity. Qed.

Lemma head_ok_nil (avoid : list variable) (e : expr) :
  head_ok avoid [] e = true.
Proof. destruct e; reflexivity. Qed.

(* A head that the test accepts has a [Let] for each parameter. *)
Lemma head_ok_names_length (avoid : list variable) (e : expr) :
  ∀ pending : list variable,
    head_ok avoid pending e = true ->
    length (twin_names pending e) = length pending.
Proof.
  induction e
    as [x kind d body IH | bindings body IH | k h body IH | handlers body IH
       | ap | ac | scrutinee arms | ];
    intros [ | p rest]; cbn [head_ok twin_names];
    try (intros _; reflexivity); try discriminate.
  destruct (alias_of d) as [q | ]; [ | discriminate ].
  destruct (Pos.eqb q p).
  - intros Hhead. cbn [length]. rewrite (IH rest Hhead). reflexivity.
  - intros Hhead. apply andb_true_iff in Hhead as [_ Hhead].
    exact (IH (p :: rest) Hhead).
Qed.

(* The test with fewer variables to avoid. *)
Lemma head_ok_weaken (avoid avoid' : list variable) (e : expr) :
  (∀ z : variable, among z avoid' = true -> among z avoid = true) ->
  ∀ pending : list variable,
    head_ok avoid pending e = true -> head_ok avoid' pending e = true.
Proof.
  intros Hsub.
  assert (Hnot :
    ∀ z : variable, among z avoid = false -> among z avoid' = false).
  { intros z Hz. destruct (among z avoid') eqn:Hz'; [ | reflexivity ].
    rewrite (Hsub z Hz') in Hz. discriminate Hz. }
  induction e
    as [x kind d body IH | bindings body IH | k h body IH | handlers body IH
       | ap | ac | scrutinee arms | ];
    intros [ | p rest]; cbn [head_ok];
    try (intros _; reflexivity); try discriminate.
  destruct (alias_of d) as [q | ]; [ | discriminate ].
  destruct (Pos.eqb q p); [ apply IH | ].
  intros Hhead. apply andb_true_iff in Hhead as [Hhead Hbody].
  apply andb_true_iff in Hhead as [Hx Hq].
  apply negb_true_iff in Hx. apply negb_true_iff in Hq.
  rewrite (IH (p :: rest) Hbody), (Hnot x Hx), (Hnot q Hq). reflexivity.
Qed.

Lemma among_remaining (z x p : variable) (names rest : list variable) :
  among z (names ++ rest) = true ->
  among z ((x :: names) ++ p :: rest) = true.
Proof.
  cbn [app]. rewrite among_cons, !among_app, among_cons. intros H.
  apply orb_true_iff in H as [H | H]; rewrite H; rewrite ?orb_true_r;
    reflexivity.
Qed.

(* ======================================================================== *)
(* The variables of the two runs                                            *)
(* ======================================================================== *)

(* The values that a continuation with twins is given: the values,
   and those at the positions again. *)
Definition twin_values (ts : list nat) (vs : list value) : list value :=
  vs ++ moved_from ts 0 vs.

Lemma twin_values_nil (vs : list value) : twin_values [] vs = vs.
Proof. unfold twin_values. rewrite moved_from_nil. apply app_nil_r. Qed.

(* The variables of the two runs, where the [Let]s at the head of a
   handler that bind [names] to the parameters [pending] are still to
   come: none of the parameters is one of the names; the runs agree
   on every variable that is not one of the names; each name holds in
   the second run what its parameter holds; and a variable that has a
   twin holds in the second run what the twin holds, if neither is
   one of the names. *)
Definition vars_rel (names pending : list variable) (pairs : twinned)
    (vars vars' : env) : Prop :=
  (∀ p : variable, among p pending = true -> among p names = false)
  ∧ (∀ z : variable, among z names = false -> lookup z vars = lookup z vars')
  ∧ Forall2 (fun x p => lookup x vars' = lookup p vars') names pending
  ∧ ∀ a b : variable,
      In (a, b) pairs ->
      among a names = false ->
      among b names = false ->
      lookup a vars' = lookup b vars'.

Lemma vars_rel_nil (vars : env) : vars_rel [] [] [] vars vars.
Proof.
  split; [ intros p Hp; simpl in Hp; discriminate Hp | ].
  split; [ intros z _; reflexivity | ].
  split; [ constructor | intros a b [] ].
Qed.

Lemma vars_rel_lookup (names pending : list variable) (pairs : twinned)
    (vars vars' : env) (z : variable) :
  vars_rel names pending pairs vars vars' ->
  among z names = false ->
  lookup z vars = lookup z vars'.
Proof. intros (_ & Hagree & _) Hz. exact (Hagree z Hz). Qed.

Lemma vars_rel_agree (pairs : twinned) (vars vars' : env) :
  vars_rel [] [] pairs vars vars' ->
  ∀ z : variable, lookup z vars = lookup z vars'.
Proof. intros (_ & Hagree & _) z. exact (Hagree z eq_refl). Qed.

Lemma vars_rel_pairs (pairs : twinned) (vars vars' : env) :
  vars_rel [] [] pairs vars vars' ->
  ∀ a b : variable, In (a, b) pairs -> lookup a vars' = lookup b vars'.
Proof.
  intros (_ & _ & _ & Hpairs) a b Hin. exact (Hpairs a b Hin eq_refl eq_refl).
Qed.

(* The [Let] of the first parameter is run by the first run: its
   variable holds there what it held in the second, and has the
   parameter for its twin. *)
Lemma vars_rel_strip (x p : variable) (names rest : list variable)
    (pairs : twinned) (vars vars' : env) (v : value) :
  vars_rel (x :: names) (p :: rest) pairs vars vars' ->
  lookup p vars = Some v ->
  vars_rel names rest ((x, p) :: tags_without [x] pairs) ((x, v) :: vars)
    vars'.
Proof.
  intros (Hapart & Hagree & Hcoming & Hpairs) Hp.
  inversion Hcoming as [ | x0 p0 names0 rest0 Hxp Hrest]; subst.
  assert (Hpn : among p (x :: names) = false).
  { apply Hapart. rewrite among_cons, Pos.eqb_refl. reflexivity. }
  assert (Hxv : lookup x vars' = Some v).
  { rewrite Hxp, <- (Hagree p Hpn). exact Hp. }
  split; [ | split; [ | split ] ].
  - intros p' Hp'.
    assert (H : among p' (x :: names) = false).
    { apply Hapart. rewrite among_cons, Hp'. apply orb_true_r. }
    rewrite among_cons in H. apply orb_false_iff in H as [_ H]. exact H.
  - intros z Hz. rewrite lookup_cons.
    destruct (Pos.eqb_spec z x) as [-> | Hne]; [ symmetry; exact Hxv | ].
    apply Hagree. rewrite among_cons, Hz. apply Pos.eqb_neq in Hne.
    rewrite Hne. reflexivity.
  - exact Hrest.
  - intros a b [[= <- <-] | Hin] Ha Hb; [ exact Hxp | ].
    apply in_tags_without in Hin as (Hin & Hax & Hbx).
    rewrite among_cons in Hax, Hbx. cbn [among existsb] in Hax, Hbx.
    rewrite orb_false_r in Hax, Hbx.
    apply Hpairs; [ exact Hin | | ]; rewrite among_cons;
      [ rewrite Hax, Ha | rewrite Hbx, Hb ]; reflexivity.
Qed.

(* A [Let] that both runs run, of a variable that is none of the
   names and none of the parameters. *)
Lemma vars_rel_kept (y : variable) (names pending : list variable)
    (pairs : twinned) (vars vars' : env) (v : value) :
  vars_rel names pending pairs vars vars' ->
  among y (names ++ pending) = false ->
  vars_rel names pending (tags_without [y] pairs) ((y, v) :: vars)
    ((y, v) :: vars').
Proof.
  intros (Hapart & Hagree & Hcoming & Hpairs) Hy.
  rewrite among_app in Hy. apply orb_false_iff in Hy as [Hyn Hyp].
  split; [ exact Hapart | split; [ | split ] ].
  - intros z Hz. rewrite !lookup_cons, (Hagree z Hz). reflexivity.
  - revert Hcoming. apply Forall2_impl_among. intros x p Hx Hp Heq.
    rewrite !lookup_cons.
    destruct (Pos.eqb_spec x y) as [-> | _]; [ congruence | ].
    destruct (Pos.eqb_spec p y) as [-> | _]; [ congruence | exact Heq ].
  - intros a b Hin Ha Hb. apply in_tags_without in Hin as (Hin & Hay & Hby).
    rewrite among_cons in Hay, Hby. cbn [among existsb] in Hay, Hby.
    rewrite orb_false_r in Hay, Hby.
    rewrite !lookup_cons, Hay, Hby. exact (Hpairs a b Hin Ha Hb).
Qed.

(* ======================================================================== *)
(* What is evaluated                                                        *)
(* ======================================================================== *)

(* A defining expression runs as it did: the untagging of the twin is
   the same primitive, of an argument that is worth the same. *)
Lemma exec_twin_named (statics : static_env) (hp : heap) (rd : read_record)
    (pairs : twinned) (vars vars' : env) (d : named) :
  vars_rel [] [] pairs vars vars' ->
  exec_named statics hp rd vars' (twin_named pairs d)
  = exec_named statics hp rd vars d.
Proof.
  intros Hvars.
  assert
    (Hsame :
      exec_named statics hp rd vars' d = exec_named statics hp rd vars d).
  { apply exec_named_agree. intros z _.
    exact (vars_rel_agree pairs vars vars' Hvars z). }
  unfold twin_named.
  destruct (untag_of d) as [x | ] eqn:Huntag; [ | exact Hsame ].
  destruct (lookup x pairs) as [p | ] eqn:Hp; [ | exact Hsame ].
  apply untag_of_some in Huntag. subst d.
  apply exec_named_args. cbn [eval_simples eval_simple].
  rewrite
    <- (vars_rel_pairs pairs vars vars' Hvars x p (lookup_in x p pairs Hp)),
    <- (vars_rel_agree pairs vars vars' Hvars x).
  reflexivity.
Qed.

Lemma eval_twin_simple (pairs : twinned) (vars vars' : env) (s : simple) :
  vars_rel [] [] pairs vars vars' ->
  eval_simple vars' (twin_simple pairs s) = eval_simple vars' s.
Proof.
  intros Hvars. destruct s as [x | sym | c]; try reflexivity.
  cbn [twin_simple]. destruct (lookup x pairs) as [p | ] eqn:Hp;
    [ | reflexivity ].
  cbn [eval_simple]. symmetry.
  exact (vars_rel_pairs pairs vars vars' Hvars x p (lookup_in x p pairs Hp)).
Qed.

Lemma eval_twin_simples (pairs : twinned) (vars vars' : env)
    (args : list simple) :
  vars_rel [] [] pairs vars vars' ->
  eval_simples vars' (map (twin_simple pairs) args) = eval_simples vars' args.
Proof.
  intros Hvars. induction args as [ | s args IH]; [ reflexivity | ].
  cbn [map eval_simples].
  rewrite (eval_twin_simple pairs vars vars' s Hvars), IH. reflexivity.
Qed.

(* The arguments of an application: the values, and those at the
   positions again. *)
Lemma eval_twin_args (ts : list nat) (pairs : twinned) (vars vars' : env)
    (args : list simple) (vs : list value) :
  vars_rel [] [] pairs vars vars' ->
  eval_simples vars args = Some vs ->
  eval_simples vars' (twin_args ts pairs args) = Some (twin_values ts vs).
Proof.
  intros Hvars Hvs.
  assert (Hvs' : eval_simples vars' args = Some vs).
  { rewrite
      (eval_simples_agree vars vars' args
         (fun z _ => vars_rel_agree pairs vars vars' Hvars z)).
    exact Hvs. }
  destruct (eval_simples_split ts vars' args 0 vs Hvs') as [_ Hmoved].
  unfold twin_args, twin_values.
  rewrite eval_simples_app, Hvs', (eval_twin_simples pairs vars vars' _ Hvars),
    Hmoved.
  reflexivity.
Qed.

(* The kinds of the values that a handler with twins is given are
   those of its metadata. *)
Lemma kinds_twin (ts : list nat) (vs : list value)
    (metadata : handler_metadata) :
  kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata) = true ->
  kinds_eqb (map kind_of_value (twin_values ts vs))
    (hm_params_kinds (twin_metadata ts metadata))
  = true.
Proof.
  intros Hkinds. apply kinds_eqb_eq in Hkinds.
  unfold twin_values, twin_metadata. cbn [hm_params_kinds].
  rewrite <- Hkinds, map_app, moved_from_map. apply kinds_eqb_refl.
Qed.

(* The handler entered: the new parameters are bound, and the
   variables of the two runs are related with the [Let]s of the
   parameters that have twins still to come. *)
Lemma bind_twin_params (ts : list nat) (pairs : twinned)
    (params names : list variable) (vs : list value)
    (vars vars' bound : env) :
  distinct (renamed_from ts 0 params names ++ moved_from ts 0 params) = true ->
  length names = length (moved_from ts 0 params) ->
  vars_rel [] [] pairs vars vars' ->
  bind_params params vs vars = Some bound ->
  ∃ bound' : env,
    bind_params (renamed_from ts 0 params names ++ moved_from ts 0 params)
      (twin_values ts vs) vars'
    = Some bound'
    ∧ vars_rel names (moved_from ts 0 params) (tags_without params pairs)
        bound bound'.
Proof.
  intros Hdistinct' Hnames Hvars Hbind.
  pose proof (bind_params_same_length params vs vars bound Hbind) as Hlength.
  pose proof (distinct_twin_params ts params 0 names Hdistinct') as Hdistinct.
  destruct (distinct_append _ _ Hdistinct') as (Hrenamed & _ & Hapart).
  pose proof (vars_rel_agree pairs vars vars' Hvars) as Hagree.
  destruct
    (bind_params_of_length
       (renamed_from ts 0 params names ++ moved_from ts 0 params)
       (twin_values ts vs) vars')
    as [bound' Hbind'].
  { unfold twin_values.
    rewrite !length_app, renamed_from_length, Hlength,
      (moved_from_length ts params vs 0 Hlength).
    reflexivity. }
  exists bound'. split; [ exact Hbind' | ].
  assert (Hbound :
    ∀ z : variable,
      lookup z bound
      = match lookup z (combine params vs) with
        | Some v => Some v
        | None => lookup z vars
        end).
  { intros z.
    exact (lookup_bind_params params vs vars bound z Hdistinct Hbind). }
  assert (Hbound' :
    ∀ z : variable,
      lookup z bound'
      = match lookup z (combine (renamed_from ts 0 params names) vs) with
        | Some v => Some v
        | None =>
          match
            lookup z
              (combine (moved_from ts 0 params) (moved_from ts 0 vs))
          with
          | Some v => Some v
          | None => lookup z vars'
          end
        end).
  { intros z.
    rewrite (lookup_bind_params _ _ vars' bound' z Hdistinct' Hbind').
    unfold twin_values.
    rewrite combine_append by (rewrite renamed_from_length; exact Hlength).
    rewrite lookup_app.
    destruct (lookup z (combine (renamed_from ts 0 params names) vs));
      reflexivity. }
  assert (Hmoved :
    ∀ z : variable,
      among z (moved_from ts 0 params) = true ->
      ∃ v : value,
        lookup z (combine params vs) = Some v ∧ lookup z bound' = Some v).
  { intros z Hz.
    destruct
      (lookup_combine_some z params vs Hlength
         (among_moved_from ts z params 0 Hz))
      as [v Hv].
    exists v. split; [ exact Hv | ].
    rewrite Hbound', (lookup_combine_none z _ vs (Hapart z Hz)),
      (lookup_moved_twin ts z params vs 0 Hdistinct Hlength Hz), Hv.
    reflexivity. }
  assert (Hother :
    ∀ z : variable,
      among z names = false ->
      among z (moved_from ts 0 params) = false ->
      lookup z bound'
      = match lookup z (combine params vs) with
        | Some v => Some v
        | None => lookup z vars'
        end).
  { intros z Hzn Hzm.
    rewrite Hbound',
      (lookup_renamed_other ts z params vs 0 names Hlength Hzn Hzm),
      (lookup_combine_none z _ (moved_from ts 0 vs) Hzm).
    reflexivity. }
  split; [ | split; [ | split ] ].
  - intros p Hp. destruct (among p names) eqn:Hpn; [ | reflexivity ].
    pose proof (Hapart p Hp) as H.
    rewrite (among_names_renamed ts p params 0 names Hnames Hpn) in H.
    discriminate H.
  - intros z Hz. rewrite Hbound.
    destruct (among z (moved_from ts 0 params)) eqn:Hzm.
    + destruct (Hmoved z Hzm) as (v & Hv & Hv'). rewrite Hv, Hv'.
      reflexivity.
    + rewrite (Hother z Hz Hzm).
      destruct (lookup z (combine params vs)); [ reflexivity | apply Hagree ].
  - generalize
      (lookup_renamed_names ts params vs 0 names Hdistinct Hrenamed Hlength
         Hnames).
    apply Forall2_impl_among. intros x p Hx Hp Heq.
    destruct (Hmoved p Hp) as (v & Hv & Hv').
    rewrite Hv', Hbound', Heq, Hv. reflexivity.
  - intros a b Hin Ha Hb. apply in_tags_without in Hin as (Hin & Hap & Hbp).
    assert (Houter :
      ∀ z : variable,
        among z names = false ->
        among z params = false ->
        lookup z bound' = lookup z vars').
    { intros z Hzn Hzp.
      assert (Hzm : among z (moved_from ts 0 params) = false).
      { destruct (among z (moved_from ts 0 params)) eqn:Hzm; [ | reflexivity ].
        rewrite (among_moved_from ts z params 0 Hzm) in Hzp.
        discriminate Hzp. }
      rewrite (Hother z Hzn Hzm), (lookup_combine_none z params vs Hzp).
      reflexivity. }
    rewrite (Houter a Ha Hap), (Houter b Hb Hbp).
    exact (vars_rel_pairs pairs vars vars' Hvars a b Hin).
Qed.

(* ======================================================================== *)
(* What the rewrite keeps of the syntax                                     *)
(* ======================================================================== *)

Fixpoint twin_expr_binds_statics (sites active : positions) (pairs : twinned)
    (pending : list variable) (e : expr) {struct e} :
  binds_statics (twin_expr sites active pairs pending e) = binds_statics e
with twin_handler_binds_statics (sites active : positions) (pairs : twinned)
    (ts : list nat) (h : continuation_handler) {struct h} :
  handler_binds_statics (twin_handler sites active pairs ts h)
  = handler_binds_statics h.
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [twin_expr binds_statics]; try reflexivity.
    + destruct pending as [ | p rest];
        [ cbn [binds_statics]; apply twin_expr_binds_statics | ].
      destruct (alias_of d) as [q | ];
        [ | cbn [binds_statics]; apply twin_expr_binds_statics ].
      destruct (Pos.eqb q p); [ | cbn [binds_statics] ];
        apply twin_expr_binds_statics.
    + rewrite twin_expr_binds_statics. reflexivity.
    + rewrite twin_handler_binds_statics, twin_expr_binds_statics.
      reflexivity.
    + rewrite twin_expr_binds_statics. f_equal.
      generalize (active_under sites handlers active). intros active'.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map existsb]. rewrite twin_handler_binds_statics, IH.
      reflexivity.
  - destruct h as [params metadata handler].
    cbn [twin_handler handler_binds_statics].
    apply twin_expr_binds_statics.
Qed.

Definition twin_handlers (sites active : positions) (pairs : twinned)
    (handlers : list (continuation * continuation_handler))
    : list (continuation * continuation_handler) :=
  map
    (fun '(k, h) =>
       (k, twin_handler sites active pairs (positions_of sites k) h))
    handlers.

Lemma twin_handlers_bind_statics (sites active : positions) (pairs : twinned)
    (handlers : list (continuation * continuation_handler)) :
  handlers_bind_statics (twin_handlers sites active pairs handlers)
  = handlers_bind_statics handlers.
Proof.
  unfold handlers_bind_statics, twin_handlers.
  induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
  cbn [map existsb]. rewrite twin_handler_binds_statics, IH. reflexivity.
Qed.

Lemma lookup_twin_handlers (sites active : positions) (pairs : twinned)
    (k1 : continuation)
    (handlers : list (continuation * continuation_handler)) :
  lookup k1 (twin_handlers sites active pairs handlers)
  = option_map (twin_handler sites active pairs (positions_of sites k1))
      (lookup k1 handlers).
Proof.
  unfold twin_handlers.
  induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
  cbn [map]. rewrite !lookup_cons, IH.
  destruct (Pos.eqb_spec k1 k') as [-> | Hne]; reflexivity.
Qed.

Lemma find_arm_twin (active : positions) (pairs : twinned) (j : Z)
    (arms : list (Z * apply_cont)) :
  find_arm j
    (map (fun '(n, ac) => (n, twin_apply_cont active pairs ac)) arms)
  = option_map (twin_apply_cont active pairs) (find_arm j arms).
Proof.
  induction arms as [ | [n ac] arms IH]; [ reflexivity | ].
  cbn [map find_arm]. destruct (Z.eqb j n); [ reflexivity | exact IH ].
Qed.

(* ======================================================================== *)
(* The two runs                                                             *)
(* ======================================================================== *)

Section Runs.

Variable sites : positions.

Definition handlers_ok (active : positions)
    (handlers : list (continuation * continuation_handler)) : bool :=
  forallb
    (fun '(k, h) => twin_handler_ok sites active (positions_of sites k) h)
    handlers.

(* What a continuation name stands for in the two runs, and which of
   the values that it is given are given again in the second: the same
   thing; or two closures of one handler under the same variables and
   continuations so related; or a closure of a handler that the test
   accepts and a closure of that handler rewritten, under variables and
   continuations that are related; or two so related with the same
   extra arguments; or what is left of the same over-application. *)
Inductive cont_rel : list nat -> cont_value -> cont_value -> Prop :=
  | Rel_same (c : cont_value) : cont_rel [] c c
  | Rel_plain (hh : continuation_handler) (dv : env) (dc dc' : cont_env)
      (recursive : list (continuation * continuation_handler)) :
    (∀ k1 : continuation, cont_rel [] (resolve k1 dc) (resolve k1 dc')) ->
    cont_rel [] (Closure hh dv dc recursive) (Closure hh dv dc' recursive)
  | Rel_inside (ts : list nat) (hh : continuation_handler) (dv dv' : env)
      (dc dc' : cont_env)
      (recursive : list (continuation * continuation_handler))
      (active : positions) (pairs : twinned) :
    twin_handler_ok sites (active_under sites recursive active) ts hh = true ->
    handlers_ok (active_under sites recursive active) recursive = true ->
    vars_rel [] [] pairs dv dv' ->
    (∀ k1 : continuation,
      cont_rel (positions_of active k1) (resolve k1 dc) (resolve k1 dc')) ->
    cont_rel ts (Closure hh dv dc recursive)
      (Closure
         (twin_handler sites (active_under sites recursive active) pairs ts
            hh)
         dv' dc'
         (twin_handlers sites (active_under sites recursive active) pairs
            recursive))
  | Rel_extra (c c' : cont_value) (extra : list value) :
    cont_rel [] c c' -> cont_rel [] (Extra_args c extra) (Extra_args c' extra)
  | Rel_over (remaining : list value) (remaining_arity return_arity : arity)
      (return_to return_to' exn_to exn_to' : cont_value) :
    cont_rel [] return_to return_to' ->
    cont_rel [] exn_to exn_to' ->
    cont_rel []
      (Over_application remaining remaining_arity return_arity return_to
         exn_to)
      (Over_application remaining remaining_arity return_arity return_to'
         exn_to').

Definition conts_rel (active : positions) (conts conts' : cont_env) : Prop :=
  ∀ k1 : continuation,
    cont_rel (positions_of active k1) (resolve k1 conts) (resolve k1 conts').

(* Two states, and the [Let]s that the first has still to run alone:
   the same expression, under the same variables; or an expression
   that the test accepts and the same rewritten, the [Let]s of the
   parameters [pending] being still to come. *)
Inductive state_rel : nat -> state -> state -> Prop :=
  | Rel_state_plain (e : expr) (vars : env) (conts conts' : cont_env)
      (statics : static_env) (hp : heap) (rd : read_record) :
    conts_rel [] conts conts' ->
    state_rel 0 (State e vars conts statics hp rd)
      (State e vars conts' statics hp rd)
  | Rel_state_inside (e : expr) (vars vars' : env) (conts conts' : cont_env)
      (statics : static_env) (hp : heap) (rd : read_record) (active : positions)
      (pairs : twinned) (pending : list variable) :
    twin_ok sites active e = true ->
    head_ok (twin_names pending e ++ pending) pending e = true ->
    vars_rel (twin_names pending e) pending pairs vars vars' ->
    conts_rel active conts conts' ->
    state_rel (length pending) (State e vars conts statics hp rd)
      (State (twin_expr sites active pairs pending e) vars' conts' statics
         hp rd).

Definition related (s s' : state) : Prop := ∃ n : nat, state_rel n s s'.

Lemma related_refl (s : state) : related s s.
Proof.
  destruct s as [e vars conts statics hp rd]. exists 0.
  apply Rel_state_plain. intros k1. apply Rel_same.
Qed.

(* An expression that the test accepts and the same rewritten, with
   no [Let] still to come. *)
Lemma state_rel_settled (e : expr) (vars vars' : env)
    (conts conts' : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record)
    (active : positions) (pairs : twinned) :
  twin_ok sites active e = true ->
  vars_rel [] [] pairs vars vars' ->
  conts_rel active conts conts' ->
  state_rel 0 (State e vars conts statics hp rd)
    (State (twin_expr sites active pairs [] e) vars' conts' statics hp rd).
Proof.
  intros Hok Hvars Hconts.
  apply
    (Rel_state_inside e vars vars' conts conts' statics hp rd_ active pairs []);
    [ exact Hok | apply head_ok_nil | rewrite twin_names_nil; exact Hvars
    | exact Hconts ].
Qed.

Lemma simulated_of_related (r r' : step_result) :
  result_related related r r' -> result_simulated related r r'.
Proof.
  destruct r as [t | k1 vs hp rd | ], r' as [t' | k1' vs' hp' rd' | ]; simpl;
    try contradiction; auto.
Qed.

Lemma with_extra_args_rel (c c' : cont_value) (extra : list value) :
  cont_rel [] c c' ->
  cont_rel [] (with_extra_args c extra) (with_extra_args c' extra).
Proof.
  intros Hc. destruct extra; simpl; [ exact Hc | apply Rel_extra, Hc ].
Qed.

(* The closures of the handlers of a recursive [Let_cont]. *)
Lemma rec_conts_plain
    (handlers : list (continuation * continuation_handler)) (vars : env)
    (conts conts' : cont_env) :
  conts_rel [] conts conts' ->
  conts_rel [] (rec_closures handlers vars conts ++ conts)
    (rec_closures handlers vars conts' ++ conts').
Proof.
  intros Hconts k1. rewrite !resolve_rec_closures.
  destruct (lookup k1 handlers) as [hh | ]; [ | apply Hconts ].
  apply Rel_plain. exact Hconts.
Qed.

Lemma rec_conts_inside
    (handlers : list (continuation * continuation_handler))
    (vars vars' : env) (conts conts' : cont_env) (active : positions)
    (pairs : twinned) :
  handlers_ok (active_under sites handlers active) handlers = true ->
  vars_rel [] [] pairs vars vars' ->
  conts_rel active conts conts' ->
  conts_rel (active_under sites handlers active)
    (rec_closures handlers vars conts ++ conts)
    (rec_closures
       (twin_handlers sites (active_under sites handlers active) pairs
          handlers)
       vars' conts'
     ++ conts').
Proof.
  intros Hok Hvars Hconts k1.
  rewrite !resolve_rec_closures, lookup_twin_handlers,
    positions_of_active_under.
  destruct (lookup k1 handlers) as [hh | ] eqn:Hlookup; cbn [option_map];
    [ | exact (Hconts k1) ].
  apply Rel_inside; try assumption.
  unfold handlers_ok in Hok. rewrite forallb_forall in Hok.
  exact (Hok (k1, hh) (lookup_in k1 hh handlers Hlookup)).
Qed.

(* The same full call, returning and raising to related continuations. *)
Lemma call_code_rel (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  cont_rel [] return_to return_to' ->
  cont_rel [] exn_to exn_to' ->
  result_related related
    (call_code closure c vs args_arity return_arity return_to exn_to statics hp
       rd)
    (call_code closure c vs args_arity return_arity return_to' exn_to'
       statics hp rd).
Proof.
  intros Hreturn Hraise.
  destruct (call_code_cases closure c vs args_arity return_arity statics hp rd)
    as [Hundefined | (callee_vars & Hcall)].
  - rewrite !Hundefined. exact I.
  - rewrite !Hcall. simpl. exists 0.
    apply Rel_state_plain.
    intros k'. cbn [positions_of lookup].
    destruct (Pos.eq_dec k' (code_return_continuation c)) as [-> | Hne];
      [ rewrite !resolve_head; exact Hreturn | ].
    rewrite !(resolve_other k' (code_return_continuation c)) by exact Hne.
    destruct (Pos.eq_dec k' (code_exn_continuation c)) as [-> | Hne'];
      [ rewrite !resolve_head; exact Hraise | ].
    rewrite !(resolve_other k' (code_exn_continuation c)) by exact Hne'.
    apply Rel_same.
Qed.

Lemma run_call_rel (shape : option call_shape) (return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  cont_rel [] return_to return_to' ->
  cont_rel [] exn_to exn_to' ->
  (∀ partial : value,
    result_simulated related
      (apply_cont_value return_to [partial] statics hp rd)
      (apply_cont_value return_to' [partial] statics hp rd)) ->
  result_simulated related
    (run_call
       (fun partial => apply_cont_value return_to [partial] statics hp rd)
       shape return_arity return_to exn_to statics hp rd)
    (run_call
       (fun partial => apply_cont_value return_to' [partial] statics hp rd)
       shape return_arity return_to' exn_to' statics hp rd).
Proof.
  intros Hreturn Hraise Hpartial.
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ];
    simpl.
  - apply simulated_of_related, call_code_rel; assumption.
  - apply simulated_of_related, call_code_rel;
      [ apply Rel_over; assumption | exact Hraise ].
  - apply Hpartial.
  - exact I.
Qed.

(* Related continuations, applied to values and to the same with
   those at the positions given again. *)
Lemma apply_cont_value_rel (ts : list nat) (c c' : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  cont_rel ts c c' ->
  ∀ vs : list value,
    result_simulated related (apply_cont_value c vs statics hp rd)
      (apply_cont_value c' (twin_values ts vs) statics hp rd).
Proof.
  intros Hc.
  induction Hc
    as [c
       | [ps m he] dv dc dc' recursive Hdc _
       | ts [ps m he] dv dv' dc dc' recursive active pairs Hok Hrec Hvars Hdc
           _
       | c c' extra Hc IHc
       | remaining remaining_arity return_arity return_to return_to' exn_to
           exn_to' Hreturn IHreturn Hraise _];
    intros vs.
  - rewrite twin_values_nil.
    apply simulated_of_related, result_related_refl, related_refl.
  - rewrite twin_values_nil. simpl.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds m));
      [ | exact I ].
    destruct (bind_params ps vs dv) as [bound | ]; [ | exact I ].
    exists 0. apply Rel_state_plain, rec_conts_plain. exact Hdc.
  - cbn [twin_handler apply_cont_value].
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds m))
      eqn:Hkinds; [ | exact I ].
    destruct (bind_params ps vs dv) as [bound | ] eqn:Hbound; [ | exact I ].
    cbn [twin_handler_ok] in Hok. apply andb_true_iff in Hok as [Hfits Hok].
    unfold handler_twins_fit in Hfits.
    apply andb_true_iff in Hfits as [Hdistinct Hhead].
    rewrite (kinds_twin ts vs m Hkinds).
    destruct
      (bind_twin_params ts pairs ps
         (twin_names (moved_from ts 0 ps) he) vs dv dv' bound Hdistinct
         (head_ok_names_length _ he _ Hhead) Hvars Hbound)
      as (bound' & Hbound' & Hrel).
    unfold twin_params_list. rewrite Hbound'.
    exists (length (moved_from ts 0 ps)).
    apply Rel_state_inside; [ exact Hok | exact Hhead | exact Hrel | ].
    apply rec_conts_inside; assumption.
  - rewrite twin_values_nil. cbn [apply_cont_value].
    specialize (IHc (vs ++ extra)). rewrite twin_values_nil in IHc.
    exact IHc.
  - rewrite twin_values_nil, !apply_cont_value_over_application.
    destruct vs as [ | v [ | v' vs]]; try exact I.
    apply run_call_rel; [ exact Hreturn | exact Hraise | ].
    intros partial. specialize (IHreturn [partial]).
    rewrite twin_values_nil in IHreturn. exact IHreturn.
Qed.

Lemma apply_cont_value_plain (c c' : cont_value) (vs : list value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  cont_rel [] c c' ->
  result_simulated related (apply_cont_value c vs statics hp rd)
    (apply_cont_value c' vs statics hp rd).
Proof.
  intros Hc.
  pose proof (apply_cont_value_rel [] c c' statics hp rd Hc vs) as Happly.
  rewrite twin_values_nil in Happly. exact Happly.
Qed.

Lemma call_function_rel (callee : value) (call_kind : call_kind)
    (vs : list value) (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  cont_rel [] return_to return_to' ->
  cont_rel [] exn_to exn_to' ->
  result_simulated related
    (call_function callee call_kind vs args_arity return_arity return_to
       exn_to statics hp rd)
    (call_function callee call_kind vs args_arity return_arity return_to'
       exn_to' statics hp rd).
Proof.
  intros Hreturn Hraise. unfold call_function.
  apply run_call_rel; [ exact Hreturn | exact Hraise | ].
  intros partial. apply apply_cont_value_plain, Hreturn.
Qed.

(* What a step of the first run is answered by: a step of the second,
   to related states; or nothing, the states being then related with
   fewer [Let]s to come; the same exit. *)
Definition answer (n : nat) (r : step_result) (s' : state) : Prop :=
  match r with
  | Step t =>
    (∃ (m : nat) (t' : state), step s' = Step t' ∧ state_rel m t t')
    ∨ ∃ m : nat, m < n ∧ state_rel m t s'
  | Exit k vs hp rd => step s' = Exit k vs hp rd
  | Undefined => True
  end.

Lemma answer_together (n : nat) (r : step_result) (s' : state) :
  result_simulated related r (step s') -> answer n r s'.
Proof.
  unfold answer.
  destruct r as [t | k vs hp rd | ], (step s') as [t' | k' vs' hp' rd' | ];
    simpl;
    try contradiction; auto.
  - intros [m Hrel]. left. exists m, t'. split; [ reflexivity | exact Hrel ].
  - intros (-> & -> & -> & <-). reflexivity.
Qed.

(* The same expression under related continuations: the two states
   step together. *)
Lemma plain_step (e : expr) (vars : env) (conts conts' : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  conts_rel [] conts conts' ->
  result_simulated related (step (State e vars conts statics hp rd))
    (step (State e vars conts' statics hp rd)).
Proof.
  intros Hconts.
  destruct e as [y kind1 d body | bindings body | k1 hh body
                | handlers body | ap | ac | scrutinee arms | ].
  - rewrite !step_let_exec.
    destruct (exec_named _ _ rd_ _ d) as [[[v hp'] rd'] | ]; [ | exact I ].
    destruct (kind_eqb (kind_of_value v) kind1); [ | exact I ].
    exists 0. apply Rel_state_plain, Hconts.
  - rewrite !step_let_static.
    destruct (eval_static_bindings vars bindings statics); [ | exact I ].
    exists 0. apply Rel_state_plain, Hconts.
  - rewrite !step_let_cont_non_recursive. exists 0.
    apply Rel_state_plain.
    intros k2. destruct (Pos.eq_dec k2 k1) as [-> | Hne].
    + rewrite !resolve_head. apply Rel_plain. exact Hconts.
    + rewrite !(resolve_other k2 k1) by exact Hne. apply Hconts.
  - rewrite !step_let_cont_recursive.
    destruct (handlers_bind_statics handlers); [ exact I | ].
    exists 0. apply Rel_state_plain, rec_conts_plain, Hconts.
  - rewrite !step_apply.
    destruct (eval_simple vars (ap_callee ap)) as [callee | ]; [ | exact I ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | exact I ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
      [ | exact I ].
    apply call_function_rel;
      [ apply Hconts | apply with_extra_args_rel, Hconts ].
  - rewrite !step_apply_cont.
    destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | exact I ].
    unfold apply_continuation. apply apply_cont_value_plain, Hconts.
  - rewrite !step_switch.
    destruct (eval_simple vars scrutinee) as [v | ]; [ | exact I ].
    destruct (naked_immediate_of_value v) as [j | ]; [ | exact I ].
    destruct (find_arm j arms) as [ac | ]; [ | exact I ].
    exists 0. apply Rel_state_plain, Hconts.
  - rewrite !step_invalid. exact I.
Qed.

(* An expression that the test accepts and the same rewritten: the
   first run takes a [Let] that went alone, and the two step together
   otherwise. *)
Lemma inside_step (e : expr) (vars vars' : env) (conts conts' : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) (active : positions)
    (pairs : twinned) (pending : list variable) :
  twin_ok sites active e = true ->
  head_ok (twin_names pending e ++ pending) pending e = true ->
  vars_rel (twin_names pending e) pending pairs vars vars' ->
  conts_rel active conts conts' ->
  answer (length pending) (step (State e vars conts statics hp rd))
    (State (twin_expr sites active pairs pending e) vars' conts' statics hp rd).
Proof.
  intros Hok Hhead Hvars Hconts.
  destruct e as [y kind1 d body | bindings body | k1 hh body
                | handlers body | ap | ac | scrutinee arms | ];
    [ | destruct pending as [ | p rest];
        [ | cbn [head_ok] in Hhead; discriminate Hhead ] .. ].
  - (* a [Let] *)
    destruct pending as [ | p rest].
    + cbn [twin_ok twin_expr twin_names] in Hok, Hvars |- *.
      apply answer_together.
      rewrite !step_let_exec,
        (exec_twin_named statics hp rd pairs vars vars' d Hvars).
      destruct (exec_named statics hp rd vars d) as [[[v hp'] rd'] | ];
        [ | exact I ].
      destruct (kind_eqb (kind_of_value v) kind1); [ | exact I ].
      exists 0. apply state_rel_settled; [ exact Hok | | exact Hconts ].
      exact (vars_rel_kept y [] [] pairs vars vars' v Hvars eq_refl).
    + revert Hhead Hvars.
      cbn [twin_ok twin_expr twin_names head_ok] in Hok |- *.
      destruct (alias_of d) as [q | ] eqn:Halias; [ | discriminate ].
      apply alias_of_some in Halias. subst d.
      destruct (Pos.eqb_spec q p) as [-> | Hne]; intros Hhead Hvars.
      * (* the [Let] of the first parameter to come: the first run
           takes it alone *)
        rewrite step_let_simple. cbn [eval_simple].
        destruct (lookup p vars) as [v | ] eqn:Hp; [ | exact I ].
        destruct (kind_eqb (kind_of_value v) kind1); [ | exact I ].
        right. exists (length rest). split; [ cbn [length]; lia | ].
        apply Rel_state_inside; [ exact Hok | | | exact Hconts ].
        -- revert Hhead. apply head_ok_weaken. intros z.
           apply among_remaining.
        -- exact (vars_rel_strip y p _ rest pairs vars vars' v Hvars Hp).
      * (* another [Let] of a variable to a variable: both runs take
           it *)
        apply andb_true_iff in Hhead as [Hhead Hbody].
        apply andb_true_iff in Hhead as [Hy Hq].
        apply negb_true_iff in Hy. apply negb_true_iff in Hq.
        rewrite among_app in Hq. apply orb_false_iff in Hq as [Hq _].
        apply answer_together. rewrite !step_let_simple.
        cbn [eval_simple].
        rewrite <- (vars_rel_lookup _ _ pairs vars vars' q Hvars Hq).
        destruct (lookup q vars) as [v | ]; [ | exact I ].
        destruct (kind_eqb (kind_of_value v) kind1); [ | exact I ].
        exists (length (p :: rest)).
        apply Rel_state_inside; [ exact Hok | exact Hbody | | exact Hconts ].
        exact (vars_rel_kept y _ _ pairs vars vars' v Hvars Hy).
  - (* [Let_static] *)
    cbn [twin_ok twin_expr twin_names] in Hok, Hvars |- *.
    apply answer_together.
    pose proof (vars_rel_agree pairs vars vars' Hvars) as Hagree.
    rewrite !step_let_static,
      (eval_static_bindings_agree vars vars' bindings (fun x _ => Hagree x)).
    destruct (eval_static_bindings vars bindings statics); [ | exact I ].
    exists 0. apply state_rel_settled; assumption.
  - (* [Let_cont_non_recursive] *)
    cbn [twin_ok twin_expr twin_names] in Hok, Hvars |- *.
    apply andb_true_iff in Hok as [Hhh Hbody]. apply answer_together.
    rewrite !step_let_cont_non_recursive. exists 0.
    apply state_rel_settled; [ exact Hbody | exact Hvars | ].
    intros k2. rewrite positions_of_cons.
    destruct (Pos.eqb_spec k2 k1) as [-> | Hne].
    + rewrite !resolve_head.
      exact
        (Rel_inside (positions_of sites k1) hh vars vars' conts conts' []
           active pairs Hhh eq_refl Hvars Hconts).
    + rewrite !(resolve_other k2 k1) by exact Hne. exact (Hconts k2).
  - (* [Let_cont_recursive] *)
    cbn [twin_ok twin_expr twin_names] in Hok, Hvars |- *.
    apply andb_true_iff in Hok as [Hhandlers Hbody]. apply answer_together.
    rewrite !step_let_cont_recursive.
    fold
      (twin_handlers sites (active_under sites handlers active) pairs
         handlers).
    rewrite twin_handlers_bind_statics.
    destruct (handlers_bind_statics handlers); [ exact I | ].
    exists 0. apply state_rel_settled; [ exact Hbody | exact Hvars | ].
    apply rec_conts_inside; assumption.
  - (* [Apply] *)
    cbn [twin_ok twin_expr twin_names] in Hok, Hvars |- *.
    apply andb_true_iff in Hok as [Hreturn Hraise]. apply answer_together.
    pose proof (vars_rel_agree pairs vars vars' Hvars) as Hagree.
    rewrite !step_apply,
      (eval_simple_agree vars vars' _ (fun x _ => Hagree x)),
      (eval_simples_agree vars vars' (ap_args ap) (fun x _ => Hagree x)),
      (eval_simples_agree vars vars' (exn_extra_simples ap)
         (fun x _ => Hagree x)).
    destruct (eval_simple vars (ap_callee ap)) as [callee | ]; [ | exact I ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | exact I ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
      [ | exact I ].
    pose proof (Hconts (ap_continuation ap)) as Hto.
    pose proof (Hconts (ap_exn_continuation ap)) as Hexn.
    unfold no_positions in Hreturn, Hraise.
    destruct (positions_of active (ap_continuation ap));
      [ | discriminate Hreturn ].
    destruct (positions_of active (ap_exn_continuation ap));
      [ | discriminate Hraise ].
    apply call_function_rel;
      [ exact Hto | apply with_extra_args_rel, Hexn ].
  - (* [Apply_cont] *)
    cbn [twin_expr twin_names] in Hvars |- *. apply answer_together.
    rewrite !step_apply_cont. unfold twin_apply_cont.
    cbn [ac_args ac_continuation].
    destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hvs;
      [ | exact I ].
    rewrite (eval_twin_args _ pairs vars vars' (ac_args ac) vs Hvars Hvs).
    unfold apply_continuation.
    exact
      (apply_cont_value_rel _ _ _ statics hp rd
         (Hconts (ac_continuation ac)) vs).
  - (* [Switch] *)
    cbn [twin_expr twin_names] in Hvars |- *. apply answer_together.
    pose proof (vars_rel_agree pairs vars vars' Hvars) as Hagree.
    rewrite !step_switch,
      (eval_simple_agree vars vars' scrutinee (fun x _ => Hagree x)).
    destruct (eval_simple vars scrutinee) as [v | ]; [ | exact I ].
    destruct (naked_immediate_of_value v) as [j | ]; [ | exact I ].
    rewrite find_arm_twin.
    destruct (find_arm j arms) as [ac | ]; [ | exact I ].
    cbn [option_map]. exists 0.
    exact
      (state_rel_settled (Apply_cont ac) vars vars' conts conts' statics hp
         rd active pairs eq_refl Hvars Hconts).
  - (* [Invalid] *)
    apply answer_together. cbn [twin_expr]. rewrite !step_invalid. exact I.
Qed.

Lemma twin_step (n : nat) (s s' : state) :
  state_rel n s s' -> answer n (step s) s'.
Proof.
  intros
    [e vars conts conts' statics hp rd Hconts
    | e vars vars' conts conts' statics hp active pairs pending Hok Hhead
        Hvars Hconts].
  - apply answer_together, plain_step, Hconts.
  - apply inside_step; assumption.
Qed.

Lemma related_refines (n : nat) (s s' : state) :
  state_rel n s s' -> refines s s'.
Proof. exact (counted_lagging_refines state_rel twin_step n s s'). Qed.

End Runs.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Theorem twin_at_refines : twin_at_refines_statement.
Proof.
  intros sites e vars conts statics hp rd. unfold twin_at.
  destruct (twin_ok sites [] e) eqn:Hok; [ | apply refines_refl ].
  apply (related_refines sites 0).
  apply state_rel_settled; [ exact Hok | apply vars_rel_nil | ].
  intros k1. apply Rel_same.
Qed.

(* The twins untagged: the rewrite, then
   FlambdaUntagParamsSpec.untag_expr at the positions of the twins. *)
Theorem twin_untag_at_refines : twin_untag_at_refines_statement.
Proof.
  intros sites e. unfold twin_untag_at.
  destruct (twin_ok sites [] e) eqn:Hok; cbn [andb];
    [ | apply expr_refines_refl ].
  destruct
    (untag_ok (twin_sites sites e) [] [] [] (twin_expr sites [] [] [] e))
    eqn:Huntag;
    [ | apply expr_refines_refl ].
  apply (expr_refines_trans e (twin_expr sites [] [] [] e)).
  - pose proof (twin_at_refines sites e) as H. unfold twin_at in H.
    rewrite Hok in H. exact H.
  - pose proof
      (untag_at_refines (twin_sites sites e) (twin_expr sites [] [] [] e))
      as H.
    unfold untag_at in H. rewrite Huntag in H. exact H.
Qed.

(* The pass of the validation is the rewrite at the positions that it
   computes, or the term. *)
Lemma twin_params_refines (e : expr) : expr_refines e (twin_params e).
Proof.
  unfold twin_params.
  destruct (count_positions (twin_candidates e));
    [ apply expr_refines_refl | apply twin_untag_at_refines ].
Qed.
