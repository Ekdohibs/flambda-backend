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


(* Proofs of FlambdaStaticsInScopeSpec.v. Nothing here needs review.

   The run of a unit keeps two invariants. On terms ([terms_inv]): the
   expression of the state passes the check under what the static
   environment binds, and each continuation that it can use leads to a
   state that does, once what the expression is still to bind at
   toplevel is bound ([cont_good]). On values ([state_statics_bound]):
   everything that the state holds mentions only what is bound. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaMapProofs.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaPrimitiveSpec FlambdaPrimitiveProofs FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaNominalSpec FlambdaNominalProofs
  FlambdaWellFormedSpec FlambdaWellFormedProofs FlambdaStaticsInScopeSpec.
Import ListNotations.

(* [all_among] is the test, here. FlambdaMapProofs.v proves its statement
   under the same name. *)
Lemma all_among_spec (names allowed : list positive) :
  all_among names allowed = true
  <-> (∀ name : positive,
        among name names = true -> among name allowed = true).
Proof. exact (FlambdaMapProofs.all_among names allowed). Qed.

(* ======================================================================== *)
(* One scope inside another                                                 *)
(* ======================================================================== *)

Definition sub (names names' : list positive) : Prop :=
  ∀ name : positive, among name names = true -> among name names' = true.

Lemma sub_refl (names : list positive) : sub names names.
Proof. intros name H. exact H. Qed.

Lemma sub_trans (names names' names'' : list positive) :
  sub names names' -> sub names' names'' -> sub names names''.
Proof. intros H H' name Hn. exact (H' name (H name Hn)). Qed.

Lemma sub_app_l (names names' : list positive) : sub names (names ++ names').
Proof. intros name H. rewrite among_app, H. reflexivity. Qed.

Lemma sub_app_r (names names' : list positive) : sub names' (names ++ names').
Proof. intros name H. rewrite among_app, H. apply orb_true_r. Qed.

Lemma sub_app (names names' all : list positive) :
  sub names all -> sub names' all -> sub (names ++ names') all.
Proof.
  intros H H' name Hn. rewrite among_app in Hn.
  apply orb_true_iff in Hn as [Hn | Hn]; [ exact (H name Hn) | ].
  exact (H' name Hn).
Qed.

Lemma sub_app_mono (first names names' : list positive) :
  sub names names' -> sub (first ++ names) (first ++ names').
Proof.
  intros H. apply sub_app; [ apply sub_app_l | ].
  exact (sub_trans _ _ _ H (sub_app_r first names')).
Qed.

Lemma sub_cons_mono (first : positive) (names names' : list positive) :
  sub names names' -> sub (first :: names) (first :: names').
Proof. exact (sub_app_mono [first] names names'). Qed.

Lemma sub_nil (names : list positive) : sub [] names.
Proof. intros name H. discriminate H. Qed.

Lemma sub_in (names names' : list positive) :
  sub names names' <-> (∀ name : positive, name ∈ names -> name ∈ names').
Proof.
  unfold sub. split; intros H name Hn; apply among_In, H, among_In, Hn.
Qed.

(* One scope is inside another, given that others are: by the names. *)
Ltac sub_tac :=
  let name := fresh "name" in
  let Hn := fresh "Hn" in
  apply sub_in; intros name Hn;
  repeat match goal with
         | H : sub _ _ |- _ =>
           let H' := fresh in
           pose proof (proj1 (sub_in _ _) H name) as H'; clear H
         end;
  repeat match goal with
         | H : _ ∈ _ ++ _ |- _ => apply in_app_or in H; destruct H
         | H : ?P -> _, H' : ?P |- _ => specialize (H H')
         end;
  rewrite ?in_app_iff; tauto.

Lemma among_sub (name : positive) (names names' : list positive) :
  sub names names' -> among name names = true -> among name names' = true.
Proof. intros H. apply H. Qed.

Lemma all_among_sub (names allowed allowed' : list positive) :
  sub allowed allowed' ->
  all_among names allowed = true ->
  all_among names allowed' = true.
Proof.
  intros H Hall. apply all_among_spec. intros name Hn.
  exact (H name (proj1 (all_among_spec names allowed) Hall name Hn)).
Qed.

Lemma all_among_is_sub (names allowed : list positive) :
  all_among names allowed = true <-> sub names allowed.
Proof. apply all_among_spec. Qed.

Lemma all_among_app (names names' allowed : list positive) :
  all_among (names ++ names') allowed
  = all_among names allowed && all_among names' allowed.
Proof. unfold all_among. apply forallb_app. Qed.

Lemma among_false_app (name : positive) (names names' : list positive) :
  among name (names ++ names') = false ->
  among name names = false ∧ among name names' = false.
Proof. rewrite among_app. apply orb_false_iff. Qed.

Lemma among_flat_map_false {A : Type} (f : A -> list positive)
    (name : positive) (l : list A) (a : A) :
  among name (flat_map f l) = false -> a ∈ l -> among name (f a) = false.
Proof.
  intros Hnone Hin. destruct (among name (f a)) eqn:Hname; [ | reflexivity ].
  rewrite <- Hnone. symmetry. apply among_In, in_flat_map. exists a.
  split; [ exact Hin | apply among_In, Hname ].
Qed.

Lemma among_false_cons (name first : positive) (names : list positive) :
  among name (first :: names) = false ->
  name <> first ∧ among name names = false.
Proof.
  rewrite among_cons. intros H. apply orb_false_iff in H as [Hne H].
  split; [ | exact H ]. intros ->. rewrite Pos.eqb_refl in Hne.
  discriminate Hne.
Qed.

(* ======================================================================== *)
(* The ways out of an expression                                            *)
(* ======================================================================== *)

Lemma sub_inter_l (names names' : list positive) :
  sub (inter names names') names.
Proof.
  intros name H. rewrite among_inter in H.
  apply andb_true_iff in H as [H _]. exact H.
Qed.

Lemma sub_inter_r (names names' : list positive) :
  sub (inter names names') names'.
Proof.
  intros name H. rewrite among_inter in H.
  apply andb_true_iff in H as [_ H]. exact H.
Qed.

Lemma lookup_with_bound (names : list positive) (k : continuation)
    (l : exit_names) :
  lookup k (with_bound names l)
  = option_map (fun entry => names ++ entry) (lookup k l).
Proof. apply lookup_map_values. Qed.

Lemma lookup_add_exit (k0 k : continuation) (names : list positive)
    (l : exit_names) :
  lookup k0 (add_exit k names l)
  = if Pos.eqb k0 k
    then
      Some
        match lookup k l with
        | Some names' => inter names names'
        | None => names
        end
    else lookup k0 l.
Proof.
  induction l as [ | [k' names'] l IH]; cbn [add_exit].
  - rewrite lookup_cons. reflexivity.
  - destruct (Pos.eqb_spec k k') as [-> | Hne].
    + rewrite !lookup_cons, Pos.eqb_refl.
      destruct (Pos.eqb k0 k'); reflexivity.
    + rewrite !lookup_cons, IH.
      destruct (Pos.eqb_spec k0 k') as [-> | Hne'].
      * destruct (Pos.eqb_spec k' k) as [-> | _];
          [ contradiction | reflexivity ].
      * destruct (Pos.eqb_spec k k') as [-> | _];
          [ contradiction | reflexivity ].
Qed.

Lemma lookup_remove_keys (bound : list continuation) (k : continuation)
    (l : exit_names) :
  lookup k (remove_keys bound l)
  = if among k bound then None else lookup k l.
Proof.
  induction l as [ | [k' names'] l IH]; cbn [remove_keys filter fst].
  - destruct (among k bound); reflexivity.
  - fold (remove_keys bound l).
    destruct (among k' bound) eqn:Hbound; cbn [negb];
      rewrite ?lookup_cons, IH;
      destruct (Pos.eqb_spec k k') as [-> | Hne]; try reflexivity.
    + rewrite Hbound. reflexivity.
    + rewrite Hbound. reflexivity.
Qed.

Lemma meet_exits_cons (k : continuation) (names : list positive)
    (l l' : exit_names) :
  meet_exits ((k, names) :: l) l' = add_exit k names (meet_exits l l').
Proof. reflexivity. Qed.

(* What the meet has for a continuation is within what each has. *)
Lemma lookup_meet_exits_r (k : continuation) (a : list positive)
    (l l' : exit_names) :
  lookup k l' = Some a ->
  ∃ m : list positive, lookup k (meet_exits l l') = Some m ∧ sub m a.
Proof.
  intros H. induction l as [ | [k1 names1] l IH].
  - exists a. split; [ exact H | apply sub_refl ].
  - destruct IH as (m & Hm & Hsub).
    rewrite meet_exits_cons, lookup_add_exit.
    destruct (Pos.eqb_spec k k1) as [-> | Hne].
    + rewrite Hm. exists (inter names1 m). split; [ reflexivity | ].
      exact (sub_trans _ _ _ (sub_inter_r names1 m) Hsub).
    + exists m. split; assumption.
Qed.

Lemma lookup_meet_exits_l (k : continuation) (a : list positive)
    (l l' : exit_names) :
  (k, a) ∈ l ->
  ∃ m : list positive, lookup k (meet_exits l l') = Some m ∧ sub m a.
Proof.
  induction l as [ | [k1 names1] l IH]; intros H; [ destruct H | ].
  rewrite meet_exits_cons, lookup_add_exit.
  destruct H as [[= -> ->] | H].
  - rewrite Pos.eqb_refl. eexists. split; [ reflexivity | ].
    destruct (lookup k (meet_exits l l'));
      [ apply sub_inter_l | apply sub_refl ].
  - destruct (IH H) as (m & Hm & Hsub).
    destruct (Pos.eqb_spec k k1) as [-> | Hne].
    + rewrite Hm. exists (inter names1 m). split; [ reflexivity | ].
      exact (sub_trans _ _ _ (sub_inter_r names1 m) Hsub).
    + exists m. split; assumption.
Qed.

(* A way of the meet is a way of one of the two. *)
Lemma lookup_meet_exits_inv (k : continuation) (m : list positive)
    (l l' : exit_names) :
  lookup k (meet_exits l l') = Some m ->
  (∃ a : list positive, (k, a) ∈ l)
  ∨ (∃ a : list positive, lookup k l' = Some a).
Proof.
  induction l as [ | [k1 names1] l IH]; intros H.
  - right. exists m. exact H.
  - rewrite meet_exits_cons, lookup_add_exit in H.
    destruct (Pos.eqb_spec k k1) as [-> | Hne].
    + left. exists names1. left. reflexivity.
    + destruct (IH H) as [(a & Ha) | Hr]; [ left | right; exact Hr ].
      exists a. right. exact Ha.
Qed.

(* A name that every way to a continuation has, the meet has. *)
Lemma lookup_meet_exits_all (k : continuation) (m : list positive)
    (name : positive) (l l' : exit_names) :
  lookup k (meet_exits l l') = Some m ->
  (∀ a : list positive, (k, a) ∈ l -> among name a = true) ->
  (∀ a : list positive, lookup k l' = Some a -> among name a = true) ->
  among name m = true.
Proof.
  revert m. induction l as [ | [k1 names1] l IH]; intros m H Hl Hl'.
  - exact (Hl' m H).
  - rewrite meet_exits_cons, lookup_add_exit in H.
    assert (Hrest :
      ∀ a : list positive, (k, a) ∈ l -> among name a = true).
    { intros a Ha. apply Hl. right. exact Ha. }
    destruct (Pos.eqb_spec k k1) as [-> | Hne].
    + injection H as <-.
      assert (H1 : among name names1 = true).
      { apply Hl. left. reflexivity. }
      destruct (lookup k1 (meet_exits l l')) as [m' | ] eqn:Hm';
        [ | exact H1 ].
      rewrite among_inter, H1. exact (IH m' eq_refl Hrest Hl').
    + exact (IH m H Hrest Hl').
Qed.

Lemma group_entry_lower (group : list continuation) (l : exit_names) :
  ∀ entry : list positive,
    group_entry group l = Some entry ->
    ∀ (k : continuation) (a : list positive),
      k ∈ group -> lookup k l = Some a -> sub entry a.
Proof.
  induction group as [ | k1 group IH]; cbn [group_entry]; [ discriminate | ].
  intros entry Hentry k a [-> | Hin] Hk.
  - rewrite Hk in Hentry.
    destruct (group_entry group l); injection Hentry as <-;
      [ apply sub_inter_l | apply sub_refl ].
  - destruct (lookup k1 l) as [names1 | ].
    + destruct (group_entry group l) as [entry' | ] eqn:Hgroup.
      * injection Hentry as <-.
        exact (sub_trans _ _ _ (sub_inter_r names1 entry')
                 (IH entry' eq_refl k a Hin Hk)).
      * exfalso. clear Hentry. revert Hgroup.
        clear IH. induction group as [ | k2 group IH]; [ destruct Hin | ].
        cbn [group_entry]. destruct Hin as [-> | Hin].
        -- rewrite Hk. destruct (group_entry group l); discriminate.
        -- destruct (lookup k2 l); [ destruct (group_entry group l) | ];
             try discriminate.
           apply IH, Hin.
    + exact (IH entry Hentry k a Hin Hk).
Qed.

Lemma group_entry_some (group : list continuation) (l : exit_names)
    (k : continuation) (a : list positive) :
  k ∈ group -> lookup k l = Some a ->
  ∃ entry : list positive, group_entry group l = Some entry.
Proof.
  intros Hin Hk. destruct (group_entry group l) as [entry | ] eqn:Hgroup;
    [ exists entry; reflexivity | exfalso ].
  revert Hgroup. induction group as [ | k2 group IH]; [ destruct Hin | ].
  cbn [group_entry]. destruct Hin as [-> | Hin].
  - rewrite Hk. destruct (group_entry group l); discriminate.
  - destruct (lookup k2 l); [ destruct (group_entry group l) | ];
      try discriminate.
    apply IH, Hin.
Qed.

(* A name that every way to the group has, the entry has. *)
Lemma group_entry_all (group : list continuation) (l : exit_names)
    (name : positive) :
  ∀ entry : list positive,
    group_entry group l = Some entry ->
    (∀ (k : continuation) (a : list positive),
      k ∈ group -> lookup k l = Some a -> among name a = true) ->
    among name entry = true.
Proof.
  induction group as [ | k1 group IH]; cbn [group_entry]; [ discriminate | ].
  intros entry Hentry Hall.
  assert (Hrest :
    ∀ (k : continuation) (a : list positive),
      k ∈ group -> lookup k l = Some a -> among name a = true).
  { intros k a Hin. apply Hall. right. exact Hin. }
  destruct (lookup k1 l) as [names1 | ] eqn:Hk1.
  - assert (H1 : among name names1 = true).
    { exact (Hall k1 names1 (or_introl eq_refl) Hk1). }
    destruct (group_entry group l) as [entry' | ]; injection Hentry as <-;
      [ | exact H1 ].
    rewrite among_inter, H1. exact (IH entry' eq_refl Hrest).
  - exact (IH entry Hentry Hrest).
Qed.

(* The ways of the handlers of a recursive group, together. *)
Definition handlers_exits (names_of : static_binding -> list positive)
    (handlers : list (continuation * continuation_handler)) : exit_names :=
  fold_right (fun '(_, h) l => meet_exits (handler_exits names_of h) l) []
    handlers.

Lemma handlers_exits_cons (names_of : static_binding -> list positive)
    (k : continuation) (h : continuation_handler)
    (handlers : list (continuation * continuation_handler)) :
  handlers_exits names_of ((k, h) :: handlers)
  = meet_exits (handler_exits names_of h) (handlers_exits names_of handlers).
Proof. reflexivity. Qed.

Lemma lookup_handlers_exits (names_of : static_binding -> list positive)
    (handlers : list (continuation * continuation_handler))
    (k0 : continuation) (h : continuation_handler) (k : continuation)
    (a : list positive) :
  (k0, h) ∈ handlers ->
  lookup k (handler_exits names_of h) = Some a ->
  ∃ m : list positive,
    lookup k (handlers_exits names_of handlers) = Some m ∧ sub m a.
Proof.
  intros Hin Hk.
  induction handlers as [ | [k1 h1] handlers IH]; [ destruct Hin | ].
  cbn [handlers_exits fold_right]. fold (handlers_exits names_of handlers).
  destruct Hin as [[= -> ->] | Hin].
  - exact (lookup_meet_exits_l k a _ _ (lookup_in _ _ _ Hk)).
  - destruct (IH Hin) as (m & Hm & Hsub).
    destruct (lookup_meet_exits_r k m (handler_exits names_of h1) _ Hm)
      as (m' & Hm' & Hsub').
    exists m'. split; [ exact Hm' | exact (sub_trans _ _ _ Hsub' Hsub) ].
Qed.

(* The ways out of a [Switch]: its arms, with nothing bound. *)
Definition arms_exits (arms : list (Z * apply_cont)) : exit_names :=
  fold_right (fun arm l => add_exit (ac_continuation (snd arm)) [] l) []
    arms.

Lemma lookup_arms_exits (arms : list (Z * apply_cont)) (k : continuation) :
  lookup k (arms_exits arms)
  = if existsb (fun arm => Pos.eqb k (ac_continuation (snd arm))) arms
    then Some []
    else None.
Proof.
  induction arms as [ | arm arms IH]; [ reflexivity | ].
  cbn [arms_exits fold_right existsb]. fold (arms_exits arms).
  rewrite lookup_add_exit.
  destruct (Pos.eqb k (ac_continuation (snd arm))); [ | exact IH ].
  destruct (lookup (ac_continuation (snd arm)) (arms_exits arms));
    reflexivity.
Qed.

Lemma lookup_arm_exit (arms : list (Z * apply_cont)) (arm : Z * apply_cont) :
  arm ∈ arms ->
  lookup (ac_continuation (snd arm)) (arms_exits arms) = Some [].
Proof.
  intros Hin. rewrite lookup_arms_exits.
  replace
    (existsb
       (fun arm' =>
          Pos.eqb (ac_continuation (snd arm)) (ac_continuation (snd arm')))
       arms)
    with true; [ reflexivity | ].
  symmetry. apply existsb_exists. exists arm.
  split; [ exact Hin | apply Pos.eqb_refl ].
Qed.

(* The ways out of a call. *)
Lemma lookup_apply_exits (ap : apply_expr) (k : continuation) :
  lookup k (add_exit (ap_continuation ap) [] [(ap_exn_continuation ap, [])])
  = if Pos.eqb k (ap_continuation ap) || Pos.eqb k (ap_exn_continuation ap)
    then Some []
    else None.
Proof.
  rewrite lookup_add_exit, !lookup_cons.
  destruct (Pos.eqb k (ap_continuation ap));
    [ destruct (Pos.eqb (ap_continuation ap) (ap_exn_continuation ap));
        reflexivity | ].
  destruct (Pos.eqb k (ap_exn_continuation ap)); reflexivity.
Qed.

(* The ways out of a [Let_cont], from those of its body and of its
   handlers. *)
Lemma exits_let_cont_other (names_of : static_binding -> list positive)
    (k : continuation) (h : continuation_handler) (body : expr)
    (k' : continuation) (a : list positive) :
  k' <> k ->
  lookup k' (exits names_of body) = Some a ->
  ∃ m : list positive,
    lookup k' (exits names_of (Let_cont_non_recursive k h body)) = Some m
    ∧ sub m a.
Proof.
  intros Hne Hk'. cbn [exits].
  destruct (lookup k (exits names_of body)) as [entry | ];
    [ | exists a; split; [ exact Hk' | apply sub_refl ] ].
  apply lookup_meet_exits_l. unfold remove_keys. apply filter_In.
  split; [ exact (lookup_in _ _ _ Hk') | ].
  cbn [fst]. unfold among. cbn [existsb].
  rewrite (proj2 (Pos.eqb_neq k' k) Hne). reflexivity.
Qed.

Lemma exits_let_cont_handler (names_of : static_binding -> list positive)
    (k : continuation) (h : continuation_handler) (body : expr)
    (entry : list positive) (k' : continuation) (a : list positive) :
  lookup k (exits names_of body) = Some entry ->
  lookup k' (handler_exits names_of h) = Some a ->
  ∃ m : list positive,
    lookup k' (exits names_of (Let_cont_non_recursive k h body)) = Some m
    ∧ sub m (entry ++ a).
Proof.
  intros Hentry Hk'. cbn [exits]. rewrite Hentry.
  apply lookup_meet_exits_r. rewrite lookup_with_bound, Hk'. reflexivity.
Qed.

Lemma exits_let_cont_rec_other (names_of : static_binding -> list positive)
    (handlers : list (continuation * continuation_handler)) (body : expr)
    (k' : continuation) (a : list positive) :
  among k' (map fst handlers) = false ->
  lookup k' (exits names_of body) = Some a ->
  ∃ m : list positive,
    lookup k' (exits names_of (Let_cont_recursive handlers body)) = Some m
    ∧ sub m a.
Proof.
  intros Hnot Hk'. cbn [exits].
  destruct (group_entry (map fst handlers) (exits names_of body))
    as [entry | ];
    [ | exists a; split; [ exact Hk' | apply sub_refl ] ].
  apply lookup_meet_exits_l. unfold remove_keys. apply filter_In.
  split; [ exact (lookup_in _ _ _ Hk') | ].
  cbn [fst]. rewrite Hnot. reflexivity.
Qed.

Lemma exits_let_cont_rec_handler
    (names_of : static_binding -> list positive)
    (handlers : list (continuation * continuation_handler)) (body : expr)
    (entry : list positive) (k0 : continuation) (h : continuation_handler)
    (k' : continuation) (a : list positive) :
  group_entry (map fst handlers) (exits names_of body) = Some entry ->
  (k0, h) ∈ handlers ->
  lookup k' (handler_exits names_of h) = Some a ->
  among k' (map fst handlers) = false ->
  ∃ m : list positive,
    lookup k' (exits names_of (Let_cont_recursive handlers body)) = Some m
    ∧ sub m (entry ++ a).
Proof.
  intros Hentry Hin Hk' Hnot. cbn [exits]. rewrite Hentry.
  destruct (lookup_handlers_exits names_of handlers k0 h k' a Hin Hk')
    as (m0 & Hm0 & Hsub0).
  destruct
    (lookup_meet_exits_r k' (entry ++ m0)
       (remove_keys (map fst handlers) (exits names_of body))
       (with_bound entry
          (remove_keys (map fst handlers) (handlers_exits names_of handlers))))
    as (m & Hm & Hsub).
  { rewrite lookup_with_bound, lookup_remove_keys, Hnot, Hm0. reflexivity. }
  exists m. split; [ exact Hm | ].
  exact (sub_trans _ _ _ Hsub (sub_app_mono entry _ _ Hsub0)).
Qed.

(* What a rewrite that keeps the ways out and the static names of an
   expression keeps. *)
Lemma handler_scope_ext (names_of : static_binding -> list positive)
    (k : continuation) (body body' : expr) :
  exits names_of body' = exits names_of body ->
  static_names names_of body' = static_names names_of body ->
  handler_scope names_of k body' = handler_scope names_of k body.
Proof. unfold handler_scope. intros -> ->. reflexivity. Qed.

Lemma recursive_scope_ext (names_of : static_binding -> list positive)
    (group : list continuation) (body body' : expr) :
  exits names_of body' = exits names_of body ->
  static_names names_of body' = static_names names_of body ->
  recursive_scope names_of group body' = recursive_scope names_of group body.
Proof. unfold recursive_scope. intros -> ->. reflexivity. Qed.

Lemma exits_let_cont_ext (names_of : static_binding -> list positive)
    (k : continuation) (h h' : continuation_handler) (body body' : expr) :
  handler_exits names_of h' = handler_exits names_of h ->
  exits names_of body' = exits names_of body ->
  exits names_of (Let_cont_non_recursive k h' body')
  = exits names_of (Let_cont_non_recursive k h body).
Proof. intros Hh Hb. cbn [exits]. rewrite Hh, Hb. reflexivity. Qed.

Lemma exits_let_cont_rec_ext (names_of : static_binding -> list positive)
    (handlers handlers' : list (continuation * continuation_handler))
    (body body' : expr) :
  map fst handlers' = map fst handlers ->
  handlers_exits names_of handlers' = handlers_exits names_of handlers ->
  exits names_of body' = exits names_of body ->
  exits names_of (Let_cont_recursive handlers' body')
  = exits names_of (Let_cont_recursive handlers body).
Proof.
  intros Hk Hh Hb. unfold handlers_exits in Hh. cbn [exits].
  rewrite Hk, Hh, Hb. reflexivity.
Qed.

Lemma map_fst_map_handlers
    (f : continuation_handler -> continuation_handler)
    (handlers : list (continuation * continuation_handler)) :
  map fst (map (fun '(k, h) => (k, f h)) handlers) = map fst handlers.
Proof.
  induction handlers as [ | [k h] handlers IH]; cbn [map fst];
    [ reflexivity | ].
  rewrite IH. reflexivity.
Qed.

Lemma handlers_exits_map (names_of : static_binding -> list positive)
    (f : continuation_handler -> continuation_handler)
    (handlers : list (continuation * continuation_handler)) :
  (∀ (k : continuation) (h : continuation_handler),
    (k, h) ∈ handlers ->
    handler_exits names_of (f h) = handler_exits names_of h) ->
  handlers_exits names_of (map (fun '(k, h) => (k, f h)) handlers)
  = handlers_exits names_of handlers.
Proof.
  induction handlers as [ | [k h] handlers IH]; intros Hf; [ reflexivity | ].
  cbn [map handlers_exits fold_right].
  fold (handlers_exits names_of (map (fun '(k, h) => (k, f h)) handlers)).
  fold (handlers_exits names_of handlers).
  rewrite (Hf k h (or_introl eq_refl)), IH; [ reflexivity | ].
  intros k' h' Hin. apply (Hf k' h'). right. exact Hin.
Qed.

(* What each way of a summary is made of. *)
Lemma in_with_bound (names : list positive) (k : continuation)
    (n : list positive) (l : exit_names) :
  (k, n) ∈ with_bound names l ->
  ∃ a : list positive, (k, a) ∈ l ∧ n = names ++ a.
Proof.
  unfold with_bound, map_values. intros H.
  apply in_map_iff in H as ([k' a] & [= -> <-] & Hin).
  exists a. split; [ exact Hin | reflexivity ].
Qed.

Lemma in_remove_keys (bound : list continuation) (k : continuation)
    (n : list positive) (l : exit_names) :
  (k, n) ∈ remove_keys bound l -> (k, n) ∈ l.
Proof.
  unfold remove_keys. intros H. apply filter_In in H as [H _]. exact H.
Qed.

Lemma in_add_exit (k0 k : continuation) (n0 names : list positive)
    (l : exit_names) :
  (k0, n0) ∈ add_exit k names l ->
  (k0 = k ∧ sub n0 names) ∨ (∃ a : list positive, (k0, a) ∈ l ∧ sub n0 a).
Proof.
  induction l as [ | [k' names'] l IH]; cbn [add_exit].
  - intros [[= <- <-] | []]. left. split; [ reflexivity | apply sub_refl ].
  - destruct (Pos.eqb_spec k k') as [-> | Hne].
    + intros [[= <- <-] | Hin].
      * left. split; [ reflexivity | apply sub_inter_l ].
      * right. exists n0. split; [ right; exact Hin | apply sub_refl ].
    + intros [[= <- <-] | Hin].
      * right. exists names'. split; [ left; reflexivity | apply sub_refl ].
      * destruct (IH Hin) as [Hl | (a & Ha & Hsub)]; [ left; exact Hl | ].
        right. exists a. split; [ right; exact Ha | exact Hsub ].
Qed.

Lemma in_meet_exits (k0 : continuation) (n0 : list positive)
    (l l' : exit_names) :
  (k0, n0) ∈ meet_exits l l' ->
  ∃ a : list positive, ((k0, a) ∈ l ∨ (k0, a) ∈ l') ∧ sub n0 a.
Proof.
  revert n0. induction l as [ | [k1 names1] l IH]; intros n0 H.
  - exists n0. split; [ right; exact H | apply sub_refl ].
  - rewrite meet_exits_cons in H.
    destruct (in_add_exit _ _ _ _ _ H) as [[-> Hsub] | (a & Ha & Hsub)].
    + exists names1. split; [ left; left; reflexivity | exact Hsub ].
    + destruct (IH a Ha) as (a' & Hin & Hsub').
      exists a'. split; [ | exact (sub_trans _ _ _ Hsub Hsub') ].
      destruct Hin as [Hin | Hin];
        [ left; right; exact Hin | right; exact Hin ].
Qed.

Lemma group_entry_within (group : list continuation) (l : exit_names) :
  ∀ entry : list positive,
    group_entry group l = Some entry ->
    ∃ (k : continuation) (a : list positive),
      k ∈ group ∧ lookup k l = Some a ∧ sub entry a.
Proof.
  induction group as [ | k1 group IH]; cbn [group_entry]; [ discriminate | ].
  intros entry Hentry.
  destruct (lookup k1 l) as [names1 | ] eqn:Hk1.
  - exists k1, names1. split; [ left; reflexivity | ].
    split; [ exact Hk1 | ].
    destruct (group_entry group l); injection Hentry as <-;
      [ apply sub_inter_l | apply sub_refl ].
  - destruct (IH entry Hentry) as (k & a & Hin & Hk & Hsub).
    exists k, a. split; [ right; exact Hin | split; assumption ].
Qed.

(* A way out has only names that the expression binds. *)
Fixpoint exits_in_within (names_of : static_binding -> list positive)
    (e : expr) {struct e} :
  ∀ (k : continuation) (names : list positive),
    (k, names) ∈ exits names_of e -> sub names (static_names names_of e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k0 [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros k names;
    cbn [exits handler_exits static_names handler_static_names].
  - apply exits_in_within.
  - intros H. apply in_with_bound in H as (a & Ha & ->).
    apply sub_app_mono. exact (exits_in_within names_of body k a Ha).
  - destruct (lookup k0 (exits names_of body)) as [entry | ] eqn:Hentry.
    + intros H. apply in_meet_exits in H as (a & [Ha | Ha] & Hsub);
        refine (sub_trans _ _ _ Hsub _).
      * apply in_remove_keys in Ha.
        exact (sub_trans _ _ _ (exits_in_within names_of body k a Ha)
                 (sub_app_r _ _)).
      * apply in_with_bound in Ha as (a' & Ha' & ->).
        apply sub_app.
        -- exact (sub_trans _ _ _
                    (exits_in_within names_of body k0 entry
                       (lookup_in _ _ _ Hentry))
                    (sub_app_r _ _)).
        -- exact (sub_trans _ _ _ (exits_in_within names_of handler k a' Ha')
                    (sub_app_l _ _)).
    + intros H.
      exact (sub_trans _ _ _ (exits_in_within names_of body k names H)
               (sub_app_r _ _)).
  - destruct (group_entry (map fst handlers) (exits names_of body))
      as [entry | ] eqn:Hentry.
    + intros H. apply in_meet_exits in H as (a & [Ha | Ha] & Hsub);
        refine (sub_trans _ _ _ Hsub _).
      * apply in_remove_keys in Ha.
        exact (sub_trans _ _ _ (exits_in_within names_of body k a Ha)
                 (sub_app_r _ _)).
      * apply in_with_bound in Ha as (a' & Ha' & ->).
        apply in_remove_keys in Ha'.
        apply sub_app.
        -- destruct (group_entry_within _ _ entry Hentry)
             as (k1 & a1 & _ & Hk1 & Hsub1).
           refine (sub_trans _ _ _ Hsub1 _).
           exact (sub_trans _ _ _
                    (exits_in_within names_of body k1 a1
                       (lookup_in _ _ _ Hk1))
                    (sub_app_r _ _)).
        -- refine (sub_trans _ _ _ _ (sub_app_l _ _)).
           clear Hentry Hsub. revert a' Ha'.
           induction handlers
             as [ | [k1 [params metadata handler]] handlers IH];
             cbn [fold_right flat_map handler_exits handler_static_names];
             intros a' Ha'; [ destruct Ha' | ].
           apply in_meet_exits in Ha' as (a'' & [Ha'' | Ha''] & Hsub'');
             refine (sub_trans _ _ _ Hsub'' _).
           ++ exact (sub_trans _ _ _
                       (exits_in_within names_of handler k a'' Ha'')
                       (sub_app_l _ _)).
           ++ exact (sub_trans _ _ _ (IH a'' Ha'') (sub_app_r _ _)).
    + intros H.
      exact (sub_trans _ _ _ (exits_in_within names_of body k names H)
               (sub_app_r _ _)).
  - intros H.
    apply in_add_exit in H as [[_ Hsub] | (a & [[= _ <-] | []] & Hsub)];
      exact Hsub.
  - intros [[= _ <-] | []]. apply sub_nil.
  - intros H. assert (Hnil : ∀ a : list positive,
        (k, a) ∈ arms_exits arms -> sub a []).
    { clear H. induction arms as [ | arm arms IH]; intros a Ha;
        [ destruct Ha | ].
      cbn [arms_exits fold_right] in Ha. fold (arms_exits arms) in Ha.
      apply in_add_exit in Ha as [[_ Hsub] | (a' & Ha' & Hsub)];
        [ exact Hsub | exact (sub_trans _ _ _ Hsub (IH a' Ha')) ]. }
    exact (Hnil names H).
  - intros [].
Qed.

Theorem exits_within_static_names : exits_within_static_names_statement.
Proof.
  intros names_of e k names H. apply all_among_is_sub.
  exact (exits_in_within names_of e k names (lookup_in _ _ _ H)).
Qed.

(* The scope that a handler is given is within what the body binds. *)
Lemma handler_scope_within (names_of : static_binding -> list positive)
    (k : continuation) (body : expr) :
  sub (handler_scope names_of k body) (static_names names_of body).
Proof.
  unfold handler_scope.
  destruct (lookup k (exits names_of body)) as [names | ] eqn:Hk;
    [ | apply sub_refl ].
  exact (exits_in_within names_of body k names (lookup_in _ _ _ Hk)).
Qed.

Lemma recursive_scope_within (names_of : static_binding -> list positive)
    (group : list continuation) (body : expr) :
  sub (recursive_scope names_of group body) (static_names names_of body).
Proof.
  unfold recursive_scope.
  destruct (group_entry group (exits names_of body)) as [entry | ] eqn:Hentry;
    [ | apply sub_refl ].
  destruct (group_entry_within _ _ entry Hentry) as (k & a & _ & Hk & Hsub).
  refine (sub_trans _ _ _ Hsub _).
  exact (exits_in_within names_of body k a (lookup_in _ _ _ Hk)).
Qed.

(* Where the body binds nothing static, its handlers are given
   nothing. *)
Lemma handler_scope_nothing (names_of : static_binding -> list positive)
    (k : continuation) (body : expr) :
  static_names names_of body = [] -> handler_scope names_of k body = [].
Proof.
  intros Hnone. pose proof (handler_scope_within names_of k body) as Hsub.
  rewrite Hnone in Hsub.
  destruct (handler_scope names_of k body) as [ | name names];
    [ reflexivity | ].
  specialize (Hsub name). rewrite among_cons, Pos.eqb_refl in Hsub.
  discriminate (Hsub eq_refl).
Qed.

Lemma recursive_scope_nothing (names_of : static_binding -> list positive)
    (group : list continuation) (body : expr) :
  static_names names_of body = [] -> recursive_scope names_of group body = [].
Proof.
  intros Hnone. pose proof (recursive_scope_within names_of group body) as Hsub.
  rewrite Hnone in Hsub.
  destruct (recursive_scope names_of group body) as [ | name names];
    [ reflexivity | ].
  specialize (Hsub name). rewrite among_cons, Pos.eqb_refl in Hsub.
  discriminate (Hsub eq_refl).
Qed.

Theorem exits_rebound : exits_rebound_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem exits_never_applied : exits_never_applied_statement.
Proof. split; vm_compute; reflexivity. Qed.

Theorem exits_group : exits_group_statement.
Proof. repeat split; vm_compute; reflexivity. Qed.

Theorem exits_call : exits_call_statement.
Proof. repeat split; vm_compute; reflexivity. Qed.

(* ======================================================================== *)
(* The ways out of a renamed expression                                     *)
(* ======================================================================== *)

(* A renaming renames continuations, and no static name: the ways out
   of the renamed expression are those of the expression, to the
   renamed continuations. *)
Definition rename_exits (r : renaming) (l : exit_names) : exit_names :=
  map (fun exit => (rename_continuation r (fst exit), snd exit)) l.

Lemma rename_exits_cons (r : renaming) (k : continuation)
    (names : list positive) (l : exit_names) :
  rename_exits r ((k, names) :: l)
  = (rename_continuation r k, names) :: rename_exits r l.
Proof. reflexivity. Qed.

Lemma lookup_rename_exits (r : renaming) (k : continuation)
    (l : exit_names) :
  lookup (rename_continuation r k) (rename_exits r l) = lookup k l.
Proof.
  induction l as [ | [k' names'] l IH]; [ reflexivity | ].
  rewrite rename_exits_cons, !lookup_cons, rename_continuation_eqb, IH.
  reflexivity.
Qed.

Lemma add_exit_renamed (r : renaming) (k : continuation)
    (names : list positive) (l : exit_names) :
  add_exit (rename_continuation r k) names (rename_exits r l)
  = rename_exits r (add_exit k names l).
Proof.
  induction l as [ | [k' names'] l IH]; [ reflexivity | ].
  rewrite rename_exits_cons. cbn [add_exit].
  rewrite rename_continuation_eqb.
  destruct (Pos.eqb k k'); rewrite rename_exits_cons; [ reflexivity | ].
  rewrite IH. reflexivity.
Qed.

Lemma meet_exits_renamed (r : renaming) (l l' : exit_names) :
  meet_exits (rename_exits r l) (rename_exits r l')
  = rename_exits r (meet_exits l l').
Proof.
  induction l as [ | [k names] l IH]; [ reflexivity | ].
  rewrite rename_exits_cons, !meet_exits_cons, IH, add_exit_renamed.
  reflexivity.
Qed.

Lemma with_bound_renamed (r : renaming) (names : list positive)
    (l : exit_names) :
  with_bound names (rename_exits r l) = rename_exits r (with_bound names l).
Proof.
  unfold with_bound, map_values, rename_exits. rewrite !map_map.
  reflexivity.
Qed.

Lemma among_renamed (r : renaming) (k : continuation)
    (bound : list continuation) :
  among (rename_continuation r k) (map (rename_continuation r) bound)
  = among k bound.
Proof.
  induction bound as [ | k' bound IH]; [ reflexivity | ].
  cbn [map]. rewrite !among_cons, rename_continuation_eqb, IH.
  reflexivity.
Qed.

Lemma remove_keys_renamed (r : renaming) (bound : list continuation)
    (l : exit_names) :
  remove_keys (map (rename_continuation r) bound) (rename_exits r l)
  = rename_exits r (remove_keys bound l).
Proof.
  induction l as [ | [k names] l IH]; [ reflexivity | ].
  rewrite rename_exits_cons. cbn [remove_keys filter fst].
  fold (remove_keys (map (rename_continuation r) bound) (rename_exits r l)).
  fold (remove_keys bound l).
  rewrite among_renamed, IH.
  destruct (among k bound); reflexivity.
Qed.

Lemma group_entry_renamed (r : renaming) (group : list continuation)
    (l : exit_names) :
  group_entry (map (rename_continuation r) group) (rename_exits r l)
  = group_entry group l.
Proof.
  induction group as [ | k group IH]; [ reflexivity | ].
  cbn [map group_entry]. rewrite lookup_rename_exits, IH. reflexivity.
Qed.

Section Renamed.

Variable r : renaming.
Variable names_of : static_binding -> list positive.
Hypothesis names_of_renamed :
  ∀ binding : static_binding,
    names_of (rename_static_binding r binding) = names_of binding.

Lemma bindings_names_renamed (bindings : list static_binding) :
  flat_map names_of (map (rename_static_binding r) bindings)
  = flat_map names_of bindings.
Proof.
  induction bindings as [ | binding bindings IH]; [ reflexivity | ].
  cbn [map flat_map]. rewrite names_of_renamed, IH. reflexivity.
Qed.

Fixpoint static_names_renamed (e : expr) {struct e} :
  static_names names_of (rename_expr r e) = static_names names_of e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [rename_expr rename_handler static_names handler_static_names];
    try reflexivity.
  - apply static_names_renamed.
  - rewrite bindings_names_renamed, static_names_renamed. reflexivity.
  - rewrite !static_names_renamed. reflexivity.
  - rewrite static_names_renamed. f_equal.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      [ reflexivity | ].
    cbn [map flat_map rename_handler handler_static_names].
    rewrite static_names_renamed, IH. reflexivity.
Qed.

Fixpoint exits_renamed (e : expr) {struct e} :
  exits names_of (rename_expr r e) = rename_exits r (exits names_of e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [rename_expr rename_handler exits handler_exits].
  - apply exits_renamed.
  - rewrite bindings_names_renamed, exits_renamed.
    apply with_bound_renamed.
  - rewrite (exits_renamed body), lookup_rename_exits.
    destruct (lookup k (exits names_of body)) as [entry | ];
      [ | reflexivity ].
    rewrite (exits_renamed handler), with_bound_renamed.
    change [rename_continuation r k]
      with (map (rename_continuation r) [k]).
    rewrite remove_keys_renamed. apply meet_exits_renamed.
  - fold (rename_handlers r handlers).
    rewrite map_fst_rename_handlers, (exits_renamed body), group_entry_renamed.
    destruct (group_entry (map fst handlers) (exits names_of body))
      as [entry | ]; [ | reflexivity ].
    assert (Hhandlers :
      fold_right
        (fun '(_, h) l => meet_exits (handler_exits names_of h) l) []
        (rename_handlers r handlers)
      = rename_exits r
          (fold_right
             (fun '(_, h) l => meet_exits (handler_exits names_of h) l) []
             handlers)).
    { induction handlers as [ | [k [params metadata handler]] handlers IH];
        [ reflexivity | ].
      cbn [rename_handlers map fold_right rename_handler handler_exits].
      fold (rename_handlers r handlers).
      rewrite (exits_renamed handler), IH. apply meet_exits_renamed. }
    rewrite Hhandlers, !remove_keys_renamed, with_bound_renamed.
    apply meet_exits_renamed.
  - cbn [rename_apply ap_continuation ap_exn_continuation].
    exact (add_exit_renamed r (ap_continuation ap) []
             [(ap_exn_continuation ap, [])]).
  - reflexivity.
  - induction arms as [ | arm arms IH]; [ reflexivity | ].
    cbn [map fold_right snd rename_apply_cont ac_continuation].
    rewrite IH. apply add_exit_renamed.
  - reflexivity.
Qed.

Lemma handler_scope_renamed (k : continuation) (body : expr) :
  handler_scope names_of (rename_continuation r k) (rename_expr r body)
  = handler_scope names_of k body.
Proof.
  unfold handler_scope.
  rewrite exits_renamed, lookup_rename_exits, static_names_renamed.
  reflexivity.
Qed.

Lemma recursive_scope_renamed (group : list continuation) (body : expr) :
  recursive_scope names_of (map (rename_continuation r) group)
    (rename_expr r body)
  = recursive_scope names_of group body.
Proof.
  unfold recursive_scope.
  rewrite exits_renamed, group_entry_renamed, static_names_renamed.
  reflexivity.
Qed.

End Renamed.

Lemma symbols_of_binding_renamed (r : renaming) (binding : static_binding) :
  symbols_of_binding (rename_static_binding r binding)
  = symbols_of_binding binding.
Proof. destruct binding; reflexivity. Qed.

Lemma declared_code_ids_of_binding_renamed (r : renaming)
    (binding : static_binding) :
  declared_code_ids_of_binding (rename_static_binding r binding)
  = declared_code_ids_of_binding binding.
Proof. destruct binding; reflexivity. Qed.

(* ======================================================================== *)
(* The check under a larger scope                                           *)
(* ======================================================================== *)

Lemma simple_scope_sub (syms syms' : list symbol) (s : simple) :
  sub syms syms' ->
  simple_statics_in_scope syms s = true ->
  simple_statics_in_scope syms' s = true.
Proof. intros H. destruct s; simpl; auto. Qed.

Lemma simples_scope_sub (syms syms' : list symbol) (args : list simple) :
  sub syms syms' ->
  simples_statics_in_scope syms args = true ->
  simples_statics_in_scope syms' args = true.
Proof.
  intros H. unfold simples_statics_in_scope. rewrite !forallb_forall.
  intros Hall s Hs. exact (simple_scope_sub _ _ s H (Hall s Hs)).
Qed.

Lemma value_slots_scope_sub (syms syms' : list symbol)
    (value_slots : list (value_slot * simple)) :
  sub syms syms' ->
  value_slots_statics_in_scope syms value_slots = true ->
  value_slots_statics_in_scope syms' value_slots = true.
Proof.
  intros H. unfold value_slots_statics_in_scope. rewrite !forallb_forall.
  intros Hall slot Hs. exact (simple_scope_sub _ _ _ H (Hall slot Hs)).
Qed.

Lemma named_scope_sub (syms syms' : list symbol) (ids ids' : list code_id)
    (d : named) :
  sub syms syms' ->
  sub ids ids' ->
  named_statics_in_scope syms ids d = true ->
  named_statics_in_scope syms' ids' d = true.
Proof.
  intros Hs Hi.
  destruct d as [s | p args | code slots | slot decls slots | slot closure
                | from slot closure | from to closure];
    simpl; rewrite ?andb_true_iff.
  - apply simple_scope_sub, Hs.
  - apply simples_scope_sub, Hs.
  - intros [Hc Hslots]. split; [ exact (Hi code Hc) | ].
    exact (value_slots_scope_sub _ _ _ Hs Hslots).
  - intros [Hc Hslots]. split; [ exact (all_among_sub _ _ _ Hi Hc) | ].
    exact (value_slots_scope_sub _ _ _ Hs Hslots).
  - apply simple_scope_sub, Hs.
  - apply simple_scope_sub, Hs.
  - apply simple_scope_sub, Hs.
Qed.

Lemma static_const_scope_sub (syms syms' : list symbol)
    (ids ids' : list code_id) (const : static_const) :
  sub syms syms' ->
  sub ids ids' ->
  static_const_statics_in_scope syms ids const = true ->
  static_const_statics_in_scope syms' ids' const = true.
Proof.
  intros Hs Hi.
  destruct const as [code slots | slot symbols decls slots | tag fields
                    | contents | bytes | elements];
    simpl; rewrite ?andb_true_iff.
  - intros [Hc Hslots]. split; [ exact (Hi code Hc) | ].
    exact (value_slots_scope_sub _ _ _ Hs Hslots).
  - intros [[Hsyms Hc] Hslots].
    split; [ split | exact (value_slots_scope_sub _ _ _ Hs Hslots) ].
    + exact (all_among_sub _ _ _ Hs Hsyms).
    + exact (all_among_sub _ _ _ Hi Hc).
  - apply simples_scope_sub, Hs.
  - apply simple_scope_sub, Hs.
  - auto.
  - apply simples_scope_sub, Hs.
Qed.

Lemma apply_scope_sub (syms syms' : list symbol) (ids ids' : list code_id)
    (ap : apply_expr) :
  sub syms syms' ->
  sub ids ids' ->
  apply_statics_in_scope syms ids ap = true ->
  apply_statics_in_scope syms' ids' ap = true.
Proof.
  intros Hs Hi. unfold apply_statics_in_scope. rewrite !andb_true_iff.
  intros [[[Hcallee Hkind] Hargs] Hextra].
  repeat split.
  - exact (simple_scope_sub _ _ _ Hs Hcallee).
  - destruct (ap_call_kind ap) as [code | | [code_ids | ]];
      [ exact (Hi code Hkind) | auto | | auto ].
    apply forallb_forall. intros id Hin. apply Hi.
    exact (proj1 (forallb_forall _ _) Hkind id Hin).
  - exact (simples_scope_sub _ _ _ Hs Hargs).
  - exact (simples_scope_sub _ _ _ Hs Hextra).
Qed.

Lemma group_consts_scope_sub (bindings : list static_binding) :
  ∀ (before before' all all' : list symbol) (ids ids' : list code_id),
    sub before before' ->
    sub all all' ->
    sub ids ids' ->
    group_consts_in_scope before all ids bindings = true ->
    group_consts_in_scope before' all' ids' bindings = true.
Proof.
  induction bindings as [ | [id c | id | s const] bindings IH];
    intros before before' all all' ids ids' Hb Ha Hi; simpl; auto.
  - apply IH; assumption.
  - apply IH; assumption.
  - destruct (is_closure_const const); rewrite !andb_true_iff;
      intros [Hconst Hrest]; split.
    + exact (static_const_scope_sub _ _ _ _ _ Ha Hi Hconst).
    + exact (IH _ _ _ _ _ _ Hb Ha Hi Hrest).
    + exact (static_const_scope_sub _ _ _ _ _ Hb Hi Hconst).
    + exact (IH _ _ _ _ _ _ (sub_cons_mono s _ _ Hb) Ha Hi Hrest).
Qed.

(* The check passes under a larger scope. *)
Fixpoint expr_scope_sub (e : expr) {struct e} :
  ∀ (syms syms' : list symbol) (ids ids' : list code_id),
    sub syms syms' ->
    sub ids ids' ->
    expr_statics_in_scope syms ids e = true ->
    expr_statics_in_scope syms' ids' e = true.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    intros syms syms' ids ids' Hs Hi; simpl.
  - rewrite !andb_true_iff. intros [Hd Hbody].
    split; [ exact (named_scope_sub _ _ _ _ _ Hs Hi Hd) | ].
    exact (expr_scope_sub body _ _ _ _ Hs Hi Hbody).
  - rewrite !andb_true_iff. intros [[Hconsts Hcode] Hbody].
    pose proof (sub_app_mono (flat_map symbols_of_binding bindings) _ _ Hs)
      as Hall.
    pose proof
      (sub_app_mono (flat_map declared_code_ids_of_binding bindings) _ _ Hi)
      as Hids.
    split; [ split | ].
    + refine (group_consts_scope_sub bindings _ _ _ _ _ _ _ Hall Hids
                Hconsts).
      apply sub_app_mono, Hs.
    + clear Hconsts Hbody.
      assert (Hlist :
        ∀ (all all' : list symbol) (cids cids' : list code_id),
          sub all all' ->
          sub cids cids' ->
          forallb (binding_statics_in_scope all cids) bindings = true ->
          forallb (binding_statics_in_scope all' cids') bindings = true).
      { intros all all' cids cids' Ha Hc. clear Hcode Hall Hids.
        induction bindings as [ | binding bindings IH]; simpl; [ auto | ].
        rewrite !andb_true_iff. intros [Hb Hrest].
        split; [ | exact (IH Hrest) ].
        destruct binding as [id [params my_closure ret exn metadata cbody]
                            | id | s const];
          simpl in Hb |- *; auto.
        exact (expr_scope_sub cbody _ _ _ _ Ha Hc Hb). }
      exact (Hlist _ _ _ _ Hall Hids Hcode).
    + exact (expr_scope_sub body _ _ _ _ Hall Hids Hbody).
  - rewrite !andb_true_iff. intros [Hh Hbody].
    split; [ | exact (expr_scope_sub body _ _ _ _ Hs Hi Hbody) ].
    destruct h as [params metadata handler]. simpl in Hh |- *.
    refine (expr_scope_sub handler _ _ _ _ _ _ Hh);
      apply sub_app_mono; assumption.
  - rewrite !andb_true_iff. intros [Hhs Hbody].
    split; [ | exact (expr_scope_sub body _ _ _ _ Hs Hi Hbody) ].
    clear Hbody. revert Hhs.
    generalize (recursive_scope symbols_of_binding (map fst handlers) body),
      (recursive_scope declared_code_ids_of_binding (map fst handlers) body).
    intros group_syms group_ids Hhs.
    induction handlers as [ | [k' [params metadata handler]] handlers IH];
      simpl in Hhs |- *; [ reflexivity | ].
    apply andb_true_iff in Hhs as [Hh Hrest].
    apply andb_true_iff. split; [ | exact (IH Hrest) ].
    refine (expr_scope_sub handler _ _ _ _ _ _ Hh);
      apply sub_app_mono; assumption.
  - apply apply_scope_sub; assumption.
  - apply simples_scope_sub, Hs.
  - rewrite !andb_true_iff. intros [Hscrutinee Harms].
    split; [ exact (simple_scope_sub _ _ _ Hs Hscrutinee) | ].
    revert Harms. rewrite !forallb_forall. intros Harms arm Hin.
    exact (simples_scope_sub _ _ _ Hs (Harms arm Hin)).
  - auto.
Qed.

(* ======================================================================== *)
(* What a group binds                                                       *)
(* ======================================================================== *)

(* After a group, the symbols bound are those of the group and those
   bound before; likewise the code ids. *)
Lemma eval_static_bindings_bound (vars : env)
    (bindings : list static_binding) :
  ∀ (statics statics' : static_env) (name : positive),
    eval_static_bindings vars bindings statics = Some statics' ->
    among name (bound_symbols statics')
    = among name (flat_map symbols_of_binding bindings)
      || among name (bound_symbols statics)
    ∧ among name (bound_code_ids statics')
      = among name (flat_map code_ids_of_binding bindings)
        || among name (bound_code_ids statics).
Proof.
  induction bindings as [ | [id c | id | s const] bindings IH];
    intros statics statics' name; simpl.
  - intros [= <-]. auto.
  - destruct (lookup id (static_code statics)); [ discriminate | ].
    intros Heval. destruct (IH _ _ name Heval) as [Hs Hc].
    rewrite Hs, Hc. unfold bound_symbols, bound_code_ids, among. simpl.
    split; [ reflexivity | ].
    destruct (Pos.eqb name id); simpl;
      [ apply orb_true_r | reflexivity ].
  - apply IH.
  - destruct (lookup s (static_values statics)); [ discriminate | ].
    destruct (eval_static_const vars const) as [allocated | ]; simpl;
      [ | discriminate ].
    intros Heval. destruct (IH _ _ name Heval) as [Hs Hc].
    rewrite Hs, Hc. unfold bound_symbols, bound_code_ids, among. simpl.
    split; [ | reflexivity ].
    destruct (Pos.eqb name s); simpl;
      [ apply orb_true_r | reflexivity ].
Qed.

(* The code that is bound after a group was bound before or is bound by
   the group. *)
Lemma eval_static_bindings_code (vars : env)
    (bindings : list static_binding) :
  ∀ (statics statics' : static_env) (id : code_id) (c : code),
    eval_static_bindings vars bindings statics = Some statics' ->
    (id, c) ∈ static_code statics' ->
    (id, c) ∈ static_code statics ∨ Bind_code id c ∈ bindings.
Proof.
  induction bindings as [ | [id' c' | id' | s const] bindings IH];
    intros statics statics' id c; simpl.
  - intros [= <-]. auto.
  - destruct (lookup id' (static_code statics)); [ discriminate | ].
    intros Heval Hin. destruct (IH _ _ id c Heval Hin) as [Hold | Hnew].
    + simpl in Hold. destruct Hold as [[= <- <-] | Hold]; auto.
    + auto.
  - intros Heval Hin. destruct (IH _ _ id c Heval Hin); auto.
  - destruct (lookup s (static_values statics)); [ discriminate | ].
    destruct (eval_static_const vars const) as [allocated | ]; simpl;
      [ | discriminate ].
    intros Heval Hin. destruct (IH _ _ id c Heval Hin); auto.
Qed.

(* A code id that a group brings into scope is bound by it or declared
   deleted by it. *)
Lemma declared_code_ids_split (bindings : list static_binding)
    (name : positive) :
  among name (flat_map declared_code_ids_of_binding bindings)
  = among name (flat_map code_ids_of_binding bindings)
    || among name (flat_map deleted_code_ids_of_binding bindings).
Proof.
  unfold among in *.
  induction bindings as [ | [id c | id | s const] bindings IH]; simpl.
  - reflexivity.
  - rewrite IH. apply orb_assoc.
  - rewrite IH.
    destruct (Pos.eqb name id); simpl;
      [ symmetry; apply orb_true_r | reflexivity ].
  - exact IH.
Qed.

(* ======================================================================== *)
(* The invariant of a run, on terms                                         *)
(* ======================================================================== *)

Section Terms.

(* The code ids that the unit declares deleted. *)
Variable deleted : list code_id.

(* The code ids in scope in a state: those bound, and those declared
   deleted. *)
Definition ids_of (statics : static_env) : list code_id :=
  bound_code_ids statics ++ deleted.

(* What is asked of a handler that a continuation keeps: its static
   bindings are outside loops and code, the handler itself taken to be
   outside; what it mentions is in scope; and the code ids that it
   declares deleted are those of the unit. *)
Definition handler_fine (syms : list symbol) (ids : list code_id)
    (h : continuation_handler) : Prop :=
  handler_statics_outside_loops_and_code true h = true
  ∧ handler_statics_in_scope syms ids h = true
  ∧ all_among (handler_static_names deleted_code_ids_of_binding h) deleted
    = true.

Lemma handler_fine_sub (syms syms' : list symbol) (ids ids' : list code_id)
    (h : continuation_handler) :
  sub syms syms' ->
  sub ids ids' ->
  handler_fine syms ids h ->
  handler_fine syms' ids' h.
Proof.
  intros Hs Hi (P & S & D). refine (conj P (conj _ D)).
  destruct h as [params metadata handler]. simpl in S |- *.
  exact (expr_scope_sub handler _ _ _ _ Hs Hi S).
Qed.

(* A continuation is good under a scope: when it is applied in a state
   whose static environment binds that scope, the state that it leads
   to has the invariant. A handler may bind static names before it
   applies a continuation, and that continuation may mention them; so a
   continuation that a closure keeps need be good only under the scope
   with, added, what the handler of the closure binds on every way to
   it ([exits]). Only the continuations that the handlers of the
   closure have a way to are asked for. *)
Inductive cont_good : list symbol -> list code_id -> cont_value -> Prop :=
  | Good_outer (syms : list symbol) (ids : list code_id) (k : continuation) :
    cont_good syms ids (Outer k)
  | Good_extra (syms : list symbol) (ids : list code_id) (c : cont_value)
      (extra : list value) :
    cont_good syms ids c -> cont_good syms ids (Extra_args c extra)
  | Good_over (syms : list symbol) (ids : list code_id)
      (remaining : list value) (remaining_arity return_arity : arity)
      (return_to exn_to : cont_value) :
    cont_good syms ids return_to ->
    cont_good syms ids exn_to ->
    cont_good syms ids
      (Over_application remaining remaining_arity return_arity return_to
         exn_to)
  | Good_closure (syms : list symbol) (ids : list code_id)
      (h : continuation_handler) (vars : env) (conts : cont_env)
      (recursive : list (continuation * continuation_handler)) :
    handler_fine syms ids h ->
    Forall (fun kh => handler_fine syms ids (snd kh)) recursive ->
    (∀ (h' : continuation_handler) (k : continuation) (s : list symbol)
       (i : list code_id),
      h' ∈ h :: map snd recursive ->
      lookup k (handler_exits symbols_of_binding h') = Some s ->
      lookup k (handler_exits declared_code_ids_of_binding h') = Some i ->
      among k (map fst recursive) = false ->
      cont_good (s ++ syms) (i ++ ids) (resolve k conts)) ->
    cont_good syms ids (Closure h vars conts recursive).

Lemma cont_good_sub (syms : list symbol) (ids : list code_id)
    (c : cont_value) :
  cont_good syms ids c ->
  ∀ (syms' : list symbol) (ids' : list code_id),
    sub syms syms' -> sub ids ids' -> cont_good syms' ids' c.
Proof.
  induction 1
    as [ syms ids k | syms ids c extra _ IH
       | syms ids remaining remaining_arity return_arity return_to exn_to
           _ IHreturn _ IHexn
       | syms ids h vars conts recursive Hh Hrec _ IH];
    intros syms' ids' Hs Hi.
  - apply Good_outer.
  - apply Good_extra, IH; assumption.
  - apply Good_over; [ apply IHreturn | apply IHexn ]; assumption.
  - apply Good_closure.
    + exact (handler_fine_sub _ _ _ _ _ Hs Hi Hh).
    + revert Hrec. apply Forall_impl. intros kh.
      apply handler_fine_sub; assumption.
    + intros h' k s i Hin Hks Hki Hnot.
      apply (IH h' k s i Hin Hks Hki Hnot); apply sub_app_mono; assumption.
Qed.

Lemma with_extra_args_good (syms : list symbol) (ids : list code_id)
    (c : cont_value) (extra : list value) :
  cont_good syms ids c -> cont_good syms ids (with_extra_args c extra).
Proof.
  intros H. destruct extra; simpl; [ exact H | apply Good_extra, H ].
Qed.

(* What is asked of the code that the static environment binds. *)
Definition code_fine (syms : list symbol) (ids : list code_id) (c : code)
    : Prop :=
  code_statics_outside_loops_and_code c = true
  ∧ code_statics_in_scope syms ids c = true.

Lemma code_fine_sub (syms syms' : list symbol) (ids ids' : list code_id)
    (c : code) :
  sub syms syms' -> sub ids ids' -> code_fine syms ids c ->
  code_fine syms' ids' c.
Proof.
  intros Hs Hi (P & S). refine (conj P _).
  destruct c as [params my_closure ret exn metadata body].
  simpl in S |- *.
  exact (expr_scope_sub body _ _ _ _ Hs Hi S).
Qed.

Definition code_all_fine (statics : static_env) : Prop :=
  Forall
    (fun binding =>
       code_fine (bound_symbols statics) (ids_of statics) (snd binding))
    (static_code statics).

(* The invariant: the expression of the state has its static bindings
   outside loops and code, the expression itself taken to be outside,
   and mentions only what is in scope, which is what the static
   environment binds, with the code ids declared deleted; each
   continuation that it has a way to is good under that scope with,
   added, what the expression binds on every way to it; and the code
   bound is fine. *)
Definition terms_inv (s : state) : Prop :=
  statics_outside_loops_and_code true (state_expr s) = true
  ∧ expr_statics_in_scope (bound_symbols (state_statics s))
      (ids_of (state_statics s)) (state_expr s)
    = true
  ∧ all_among (static_names deleted_code_ids_of_binding (state_expr s))
      deleted
    = true
  ∧ (∀ (k : continuation) (syms : list symbol) (ids : list code_id),
      lookup k (exits symbols_of_binding (state_expr s)) = Some syms ->
      lookup k (exits declared_code_ids_of_binding (state_expr s))
      = Some ids ->
      cont_good (syms ++ bound_symbols (state_statics s))
        (ids ++ ids_of (state_statics s))
        (resolve k (state_conts s)))
  ∧ code_all_fine (state_statics s).

(* A call of code. *)
Lemma call_code_terms (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity) (return_to exn_to : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) (s' : state) :
  code_all_fine statics ->
  code_fine (bound_symbols statics) (ids_of statics) c ->
  cont_good (bound_symbols statics) (ids_of statics) return_to ->
  cont_good (bound_symbols statics) (ids_of statics) exn_to ->
  call_code closure c vs args_arity return_arity return_to exn_to statics hp rd
  = Step s' ->
  terms_inv s'.
Proof.
  intros Hcode (P & S) Hreturn Hexn Hstep.
  apply call_code_step in Hstep as (callee_vars & _ & _ & _ & _ & _ & ->).
  destruct c as [params my_closure ret exn metadata body].
  simpl in P, S |- *.
  pose proof (outside_nowhere deleted_code_ids_of_binding body P)
    as Ndeleted.
  unfold terms_inv. simpl_state. rewrite Ndeleted.
  refine
    (conj (outside_weaken body P) (conj S (conj eq_refl (conj _ Hcode)))).
  intros k syms ids _ _. unfold resolve. rewrite !lookup_cons.
  destruct (Pos.eqb k ret);
    [ exact (cont_good_sub _ _ _ Hreturn _ _ (sub_app_r _ _) (sub_app_r _ _))
    | ].
  destruct (Pos.eqb k exn); [ | apply Good_outer ].
  exact (cont_good_sub _ _ _ Hexn _ _ (sub_app_r _ _) (sub_app_r _ _)).
Qed.

Lemma run_call_terms (return_partial : value -> step_result)
    (kind : call_kind) (shape : option call_shape) (return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
      (rd : read_record)
      (s' : state) :
  code_all_fine statics ->
  shape_calls statics kind shape ->
  cont_good (bound_symbols statics) (ids_of statics) return_to ->
  cont_good (bound_symbols statics) (ids_of statics) exn_to ->
  (∀ partial : value, return_partial partial = Step s' -> terms_inv s') ->
  run_call return_partial shape return_arity return_to exn_to statics hp rd
  = Step s' ->
  terms_inv s'.
Proof.
  intros Hcode Hcalls Hreturn Hexn Hpartial.
  assert (Hfine :
    ∀ (called : code_id) (c : code),
      lookup called (static_code statics) = Some c ->
      code_fine (bound_symbols statics) (ids_of statics) c).
  { intros called c Hlookup. apply lookup_in in Hlookup.
    unfold code_all_fine in Hcode. rewrite Forall_forall in Hcode.
    exact (Hcode _ Hlookup). }
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ];
    simpl in Hcalls |- *.
  - destruct Hcalls as (called & slots & _ & Hlookup & _).
    exact (call_code_terms closure c vs _ _ _ _ statics _ rd_ s' Hcode
             (Hfine called c Hlookup) Hreturn Hexn).
  - destruct Hcalls as (called & slots & _ & Hlookup & _).
    apply (call_code_terms closure c vs _ _ _ _ statics _ rd_ s' Hcode
             (Hfine called c Hlookup));
      [ apply Good_over; assumption | exact Hexn ].
  - apply Hpartial.
  - discriminate.
Qed.

(* A sibling of a recursive handler: its closure is good where the
   closure of the handler is. *)
Lemma sibling_good (syms : list symbol) (ids : list code_id)
    (h : continuation_handler) (vars : env) (conts : cont_env)
    (recursive : list (continuation * continuation_handler))
    (k : continuation) (h' : continuation_handler) :
  cont_good syms ids (Closure h vars conts recursive) ->
  lookup k recursive = Some h' ->
  cont_good syms ids (Closure h' vars conts recursive).
Proof.
  intros Hgood Hlookup.
  inversion Hgood
    as [ | | | syms0 ids0 h0 vars0 conts0 recursive0 Hh Hrec Hconts];
    subst.
  assert (Hin : (k, h') ∈ recursive) by exact (lookup_in _ _ _ Hlookup).
  pose proof (proj1 (Forall_forall _ _) Hrec _ Hin) as Hh'. simpl in Hh'.
  apply Good_closure; [ exact Hh' | exact Hrec | ].
  intros h'' k' s i [<- | Hin''] Hs Hi Hnot.
  - apply (Hconts h' k' s i); try assumption.
    right. apply in_map_iff. exists (k, h'). auto.
  - apply (Hconts h'' k' s i); try assumption. right. exact Hin''.
Qed.

(* The application of a good continuation. *)
Lemma apply_cont_value_terms (c : cont_value) :
  ∀ (vs : list value) (statics : static_env) (hp : heap) (rd : read_record)
    (s' : state),
    code_all_fine statics ->
    cont_good (bound_symbols statics) (ids_of statics) c ->
    apply_cont_value c vs statics hp rd = Step s' ->
    terms_inv s'.
Proof.
  induction c
    as [[params metadata handler] def_vars def_conts recursive | k
       | c IHc extra
       | remaining remaining_arity return_arity return_to exn_to IHreturn _]
    using cont_value_over_ind;
    intros vs statics hp rd s' Hcode Hgood Hstep.
  - simpl in Hstep.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | discriminate ].
    destruct (bind_params params vs def_vars); [ | discriminate ].
    injection Hstep as <-.
    inversion Hgood
      as [ | | | syms0 ids0 h0 vars0 conts0 recursive0 Hh Hrec Hconts];
      subst.
    destruct Hh as (P & S & D). simpl in P, S, D.
    unfold terms_inv. simpl_state.
    refine (conj P (conj S (conj D (conj _ Hcode)))).
    intros k syms ids Hks Hki. rewrite resolve_rec_closures.
    destruct (lookup k recursive) as [h' | ] eqn:Hlookup.
    + refine (cont_good_sub _ _ _
                (sibling_good _ _ _ _ _ _ k h' Hgood Hlookup) _ _ _ _);
        apply sub_app_r.
    + apply (Hconts (Handler params metadata handler) k syms ids);
        [ left; reflexivity | exact Hks | exact Hki | ].
      destruct (among k (map fst recursive)) eqn:Hamong; [ | reflexivity ].
      apply among_In in Hamong.
      exfalso. exact (lookup_none_not_in _ _ Hlookup Hamong).
  - discriminate.
  - inversion Hgood; subst.
    exact (IHc (vs ++ extra) statics hp rd_ s' Hcode ltac:(assumption) Hstep).
  - rewrite apply_cont_value_over_application in Hstep.
    destruct vs as [ | v [ | v' vs]]; [ discriminate | | discriminate ].
    inversion Hgood as [ | | syms0 ids0 r ra rta rt et Hreturn Hexn | ]; subst.
    unfold call_function in Hstep.
    apply (run_call_terms _ Indirect_unknown_arity _ _ _ _ statics _ rd_ s'
             Hcode (shape_of_call_calls _ _ _ _ _ _) Hreturn Hexn)
      in Hstep; [ exact Hstep | ].
    intros partial Hpartial.
    exact (IHreturn [partial] statics hp rd_ s' Hcode Hreturn Hpartial).
Qed.

Lemma call_function_terms (callee : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
      (rd : read_record)
      (s' : state) :
  code_all_fine statics ->
  cont_good (bound_symbols statics) (ids_of statics) return_to ->
  cont_good (bound_symbols statics) (ids_of statics) exn_to ->
  call_function callee kind vs args_arity return_arity return_to exn_to
    statics hp rd
  = Step s' ->
  terms_inv s'.
Proof.
  intros Hcode Hreturn Hexn. unfold call_function.
  apply (run_call_terms _ kind _ _ _ _ statics _ rd_ s' Hcode
           (shape_of_call_calls _ _ _ _ _ _) Hreturn Hexn).
  intros partial.
  exact (apply_cont_value_terms return_to _ statics _ rd_ s' Hcode Hreturn).
Qed.

(* The unfoldings of the checks at one constructor, and nothing else. *)
Local Ltac at_constructor H :=
  cbn [statics_outside_loops_and_code handler_statics_outside_loops_and_code
       binding_statics_outside_loops_and_code
       code_statics_outside_loops_and_code
       expr_statics_in_scope handler_statics_in_scope
       binding_statics_in_scope code_statics_in_scope
       static_names handler_static_names
       andb app] in H.

(* A step keeps the invariant. *)
Lemma terms_inv_step (s s' : state) :
  terms_inv s -> step s = Step s' -> terms_inv s'.
Proof.
  intros (P & S & D & K & C) Hstep. rewrite (state_eta s) in Hstep.
  set (vars := state_vars s) in *. set (conts := state_conts s) in *.
  set (statics := state_statics s) in *.
  destruct (state_expr s)
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    autorewrite with step in Hstep.
  - (* Let *)
    destruct (exec_named _ _ rd_ _ d) as [[[v hp'] rd'] | ]; [ | discriminate ].
    destruct (kind_eqb (kind_of_value v) kind); [ | discriminate ].
    injection Hstep as <-. unfold terms_inv. simpl_state.
    at_constructor P. at_constructor S. at_constructor D.
    apply andb_true_iff in S as [_ S].
    exact (conj P (conj S (conj D (conj K C)))).
  - (* Let_static *)
    destruct (eval_static_bindings vars bindings statics) as [statics' | ]
      eqn:Heval; [ | discriminate ].
    injection Hstep as <-. unfold terms_inv. simpl_state.
    at_constructor P. at_constructor S. at_constructor D.
    apply andb_true_iff in P as [Pbindings Pbody].
    apply andb_true_iff in S as [S Sbody].
    apply andb_true_iff in S as [_ Scode].
    rewrite all_among_app in D. apply andb_true_iff in D as [Dgroup Dbody].
    assert (Hsyms :
      sub (flat_map symbols_of_binding bindings ++ bound_symbols statics)
        (bound_symbols statics')).
    { intros name Hn.
      rewrite
        (proj1 (eval_static_bindings_bound vars bindings statics statics'
                  name Heval)).
      rewrite among_app in Hn. exact Hn. }
    assert (Hids :
      sub (flat_map declared_code_ids_of_binding bindings ++ ids_of statics)
        (ids_of statics')).
    { intros name Hn. unfold ids_of in Hn |- *.
      rewrite !among_app, declared_code_ids_split in Hn.
      rewrite among_app,
        (proj2 (eval_static_bindings_bound vars bindings statics statics'
                  name Heval)).
      apply orb_true_iff in Hn as [Hn | Hn];
        apply orb_true_iff in Hn as [Hn | Hn].
      - rewrite Hn. reflexivity.
      - rewrite (proj1 (all_among_spec _ _) Dgroup name Hn).
        apply orb_true_r.
      - rewrite Hn, orb_true_r. reflexivity.
      - rewrite Hn. apply orb_true_r. }
    refine (conj Pbody (conj _ (conj Dbody (conj _ _)))).
    + exact (expr_scope_sub body _ _ _ _ Hsyms Hids Sbody).
    + intros k syms ids Hks Hki.
      refine (cont_good_sub _ _ _
                (K k (flat_map symbols_of_binding bindings ++ syms)
                   (flat_map declared_code_ids_of_binding bindings ++ ids)
                   _ _)
                _ _ _ _).
      * cbn [exits]. rewrite lookup_with_bound, Hks. reflexivity.
      * cbn [exits]. rewrite lookup_with_bound, Hki. reflexivity.
      * apply sub_app; [ apply sub_app | ].
        -- exact (sub_trans _ _ _ (sub_trans _ _ _ (sub_app_l _ _) Hsyms)
                    (sub_app_r _ _)).
        -- apply sub_app_l.
        -- exact (sub_trans _ _ _ (sub_trans _ _ _ (sub_app_r _ _) Hsyms)
                    (sub_app_r _ _)).
      * apply sub_app; [ apply sub_app | ].
        -- exact (sub_trans _ _ _ (sub_trans _ _ _ (sub_app_l _ _) Hids)
                    (sub_app_r _ _)).
        -- apply sub_app_l.
        -- exact (sub_trans _ _ _ (sub_trans _ _ _ (sub_app_r _ _) Hids)
                    (sub_app_r _ _)).
    + unfold code_all_fine. apply Forall_forall. intros [id c] Hin.
      cbn [snd].
      destruct (eval_static_bindings_code vars bindings statics statics' id c
                  Heval Hin) as [Hold | Hnew].
      * pose proof (proj1 (Forall_forall _ _) C _ Hold) as Hfine.
        cbn [snd] in Hfine.
        refine (code_fine_sub _ _ _ _ _ _ _ Hfine).
        -- exact (sub_trans _ _ _ (sub_app_r _ _) Hsyms).
        -- exact (sub_trans _ _ _ (sub_app_r _ _) Hids).
      * pose proof (proj1 (forallb_forall _ _) Pbindings _ Hnew) as Pc.
        pose proof (proj1 (forallb_forall _ _) Scode _ Hnew) as Sc.
        cbn [binding_statics_outside_loops_and_code
             binding_statics_in_scope] in Pc, Sc.
        exact (code_fine_sub _ _ _ _ _ Hsyms Hids (conj Pc Sc)).
  - (* Let_cont_non_recursive *)
    injection Hstep as <-. unfold terms_inv. simpl_state.
    at_constructor P. at_constructor S. at_constructor D.
    apply andb_true_iff in P as [Ph Pbody].
    apply andb_true_iff in S as [Sh Sbody].
    rewrite all_among_app in D. apply andb_true_iff in D as [Dh Dbody].
    refine (conj Pbody (conj Sbody (conj Dbody (conj _ C)))).
    intros k' syms ids Hks Hki.
    destruct (Pos.eqb_spec k' k) as [-> | Hne].
    + rewrite resolve_head. unfold handler_scope in Sh.
      rewrite Hks, Hki in Sh.
      apply Good_closure.
      * exact (conj Ph (conj Sh Dh)).
      * apply Forall_nil.
      * intros h' k'' xs xi [<- | []] Hs Hi _.
        destruct (exits_let_cont_handler symbols_of_binding k h body syms
                    k'' xs Hks Hs) as (ms & Hms & Hsubs).
        destruct (exits_let_cont_handler declared_code_ids_of_binding k h
                    body ids k'' xi Hki Hi) as (mi & Hmi & Hsubi).
        refine (cont_good_sub _ _ _ (K k'' ms mi Hms Hmi) _ _ _ _);
          sub_tac.
    + rewrite (resolve_other _ _ _ _ Hne).
      destruct (exits_let_cont_other symbols_of_binding k h body k' syms Hne
                  Hks) as (ms & Hms & Hsubs).
      destruct (exits_let_cont_other declared_code_ids_of_binding k h body k'
                  ids Hne Hki) as (mi & Hmi & Hsubi).
      refine (cont_good_sub _ _ _ (K k' ms mi Hms Hmi) _ _ _ _); sub_tac.
  - (* Let_cont_recursive *)
    destruct (handlers_bind_statics handlers); [ discriminate | ].
    injection Hstep as <-. unfold terms_inv. simpl_state.
    at_constructor P. at_constructor S. at_constructor D.
    apply andb_true_iff in P as [Phandlers Pbody].
    apply andb_true_iff in S as [Shandlers Sbody].
    rewrite all_among_app in D. apply andb_true_iff in D as [_ Dbody].
    refine (conj Pbody (conj Sbody (conj Dbody (conj _ C)))).
    intros k syms ids Hks Hki. rewrite resolve_rec_closures.
    destruct (lookup k handlers) as [h' | ] eqn:Hlookup.
    + assert (Hgroup : k ∈ map fst handlers).
      { apply in_map_iff. exists (k, h').
        split; [ reflexivity | exact (lookup_in _ _ _ Hlookup) ]. }
      destruct (group_entry_some _ _ k syms Hgroup Hks) as (es & Hes).
      destruct (group_entry_some _ _ k ids Hgroup Hki) as (ei & Hei).
      pose proof (group_entry_lower _ _ es Hes k syms Hgroup Hks) as Hes_sub.
      pose proof (group_entry_lower _ _ ei Hei k ids Hgroup Hki) as Hei_sub.
      unfold recursive_scope in Shandlers. rewrite Hes, Hei in Shandlers.
      assert (Hfine :
        Forall
          (fun kh =>
             handler_fine (syms ++ bound_symbols statics)
               (ids ++ ids_of statics) (snd kh))
          handlers).
      { apply Forall_forall. intros [k0 h0] Hin. cbn [snd].
        pose proof (proj1 (forallb_forall _ _) Phandlers _ Hin) as Ph0.
        pose proof (proj1 (forallb_forall _ _) Shandlers _ Hin) as Sh0.
        cbn beta iota in Ph0, Sh0.
        refine (conj _ (conj _ _)).
        - destruct h0 as [params0 metadata0 handler0].
          exact (outside_weaken handler0 Ph0).
        - destruct h0 as [params0 metadata0 handler0].
          cbn [handler_statics_in_scope] in Sh0 |- *.
          refine (expr_scope_sub handler0 _ _ _ _ _ _ Sh0); sub_tac.
        - destruct h0 as [params0 metadata0 handler0].
          cbn [handler_static_names].
          rewrite (outside_nowhere deleted_code_ids_of_binding handler0 Ph0).
          reflexivity. }
      apply Good_closure.
      * exact (proj1 (Forall_forall _ _) Hfine _ (lookup_in _ _ _ Hlookup)).
      * exact Hfine.
      * intros h'' k'' xs xi Hin'' Hs Hi Hnot.
        assert (Hin0 : ∃ k0 : continuation, (k0, h'') ∈ handlers).
        { destruct Hin'' as [<- | Hin''];
            [ exists k; exact (lookup_in _ _ _ Hlookup) | ].
          apply in_map_iff in Hin'' as ([k0 h0] & Heq & Hin'').
          cbn [snd] in Heq. subst h0. exists k0. exact Hin''. }
        destruct Hin0 as (k0 & Hin0).
        destruct (exits_let_cont_rec_handler symbols_of_binding handlers body
                    es k0 h'' k'' xs Hes Hin0 Hs Hnot) as (ms & Hms & Hsubs).
        destruct (exits_let_cont_rec_handler declared_code_ids_of_binding
                    handlers body ei k0 h'' k'' xi Hei Hin0 Hi Hnot)
          as (mi & Hmi & Hsubi).
        refine (cont_good_sub _ _ _ (K k'' ms mi Hms Hmi) _ _ _ _);
          sub_tac.
    + assert (Hnot : among k (map fst handlers) = false).
      { destruct (among k (map fst handlers)) eqn:Hamong; [ | reflexivity ].
        apply among_In in Hamong.
        exfalso. exact (lookup_none_not_in _ _ Hlookup Hamong). }
      destruct (exits_let_cont_rec_other symbols_of_binding handlers body k
                  syms Hnot Hks) as (ms & Hms & Hsubs).
      destruct (exits_let_cont_rec_other declared_code_ids_of_binding handlers
                  body k ids Hnot Hki) as (mi & Hmi & Hsubi).
      refine (cont_good_sub _ _ _ (K k ms mi Hms Hmi) _ _ _ _); sub_tac.
  - (* Apply *)
    destruct (eval_simple vars (ap_callee ap)) as [callee | ];
      [ | discriminate ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | discriminate ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
      [ | discriminate ].
    cbn [exits] in K.
    apply (call_function_terms _ _ _ _ _ _ _ statics _ rd_ s' C) in Hstep;
      [ exact Hstep | | ].
    + apply (K (ap_continuation ap) [] []);
        rewrite lookup_apply_exits, Pos.eqb_refl; reflexivity.
    + apply with_extra_args_good.
      apply (K (ap_exn_continuation ap) [] []);
        rewrite lookup_apply_exits, Pos.eqb_refl, orb_true_r; reflexivity.
  - (* Apply_cont *)
    destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | discriminate ].
    unfold apply_continuation in Hstep. cbn [exits] in K.
    refine (apply_cont_value_terms _ vs statics _ rd_ s' C
              (K (ac_continuation ac) [] [] _ _) Hstep);
      apply lookup_head.
  - (* Switch *)
    destruct (eval_simple vars scrutinee) as [v | ]; [ | discriminate ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | discriminate ].
    destruct (find_arm i arms) as [ac | ] eqn:Harm; [ | discriminate ].
    injection Hstep as <-. unfold terms_inv. simpl_state.
    apply find_arm_in in Harm.
    at_constructor S. cbn [exits] in K.
    apply andb_true_iff in S as [_ Sarms].
    refine (conj eq_refl (conj _ (conj eq_refl (conj _ C)))).
    + exact (proj1 (forallb_forall _ _) Sarms _ Harm).
    + intros k syms ids Hks Hki. cbn [exits] in Hks, Hki.
      rewrite lookup_cons in Hks, Hki.
      destruct (Pos.eqb_spec k (ac_continuation ac)) as [-> | Hne];
        [ | discriminate ].
      injection Hks as <-. injection Hki as <-.
      apply (K (ac_continuation ac) [] []);
        exact (lookup_arm_exit arms (i, ac) Harm).
  - discriminate.
Qed.

Lemma terms_inv_run (n : nat) :
  ∀ s s' : state, terms_inv s -> run n s = Step s' -> terms_inv s'.
Proof.
  induction n as [ | n IH]; intros s s' Hinv; simpl.
  - intros [= <-]. exact Hinv.
  - destruct (step s) as [next | k vs | ] eqn:Hstep; try discriminate.
    exact (IH next s' (terms_inv_step s next Hinv Hstep)).
Qed.

End Terms.

(* ======================================================================== *)
(* Values under a larger scope                                              *)
(* ======================================================================== *)

Fixpoint value_among_sub (syms syms' : list symbol) (ids ids' : list code_id)
    (Hs : sub syms syms') (Hi : sub ids ids') (v : value) {struct v} :
  value_statics_among syms ids v = true ->
  value_statics_among syms' ids' v = true.
Proof.
  destruct v
    as [c | code slots | slot decls slots | s | tag fields | elements | c
       | l | callee args arity];
    simpl; auto.
  - rewrite !andb_true_iff. intros [Hc Hslots].
    split; [ exact (Hi code Hc) | ].
    induction slots as [ | [slot v] slots IH]; simpl in Hslots |- *;
      [ reflexivity | ].
    apply andb_true_iff in Hslots as [Hv Hrest]. apply andb_true_iff.
    split; [ exact (value_among_sub _ _ _ _ Hs Hi v Hv) | exact (IH Hrest) ].
  - rewrite !andb_true_iff. intros [Hc Hslots].
    split; [ exact (all_among_sub _ _ _ Hi Hc) | ].
    induction slots as [ | [slot' v] slots IH]; simpl in Hslots |- *;
      [ reflexivity | ].
    apply andb_true_iff in Hslots as [Hv Hrest]. apply andb_true_iff.
    split; [ exact (value_among_sub _ _ _ _ Hs Hi v Hv) | exact (IH Hrest) ].
  - intros Hfields.
    induction fields as [ | v fields IH]; simpl in Hfields |- *;
      [ reflexivity | ].
    apply andb_true_iff in Hfields as [Hv Hrest]. apply andb_true_iff.
    split; [ exact (value_among_sub _ _ _ _ Hs Hi v Hv) | exact (IH Hrest) ].
  - intros Helements.
    induction elements as [ | v elements IH]; simpl in Helements |- *;
      [ reflexivity | ].
    apply andb_true_iff in Helements as [Hv Hrest]. apply andb_true_iff.
    split; [ exact (value_among_sub _ _ _ _ Hs Hi v Hv) | exact (IH Hrest) ].
  - rewrite !andb_true_iff. intros [Hcallee Hargs].
    split; [ exact (value_among_sub _ _ _ _ Hs Hi callee Hcallee) | ].
    induction args as [ | v args IH]; simpl in Hargs |- *; [ reflexivity | ].
    apply andb_true_iff in Hargs as [Hv Hrest]. apply andb_true_iff.
    split; [ exact (value_among_sub _ _ _ _ Hs Hi v Hv) | exact (IH Hrest) ].
Qed.

Lemma slots_among_sub (syms syms' : list symbol) (ids ids' : list code_id)
    (slots : list (value_slot * value)) :
  sub syms syms' ->
  sub ids ids' ->
  forallb (fun '(_, v) => value_statics_among syms ids v) slots = true ->
  forallb (fun '(_, v) => value_statics_among syms' ids' v) slots = true.
Proof.
  intros Hs Hi. rewrite !forallb_forall. intros H [slot v] Hin.
  exact (value_among_sub _ _ _ _ Hs Hi v (H (slot, v) Hin)).
Qed.

Lemma values_among_sub (syms syms' : list symbol) (ids ids' : list code_id)
    (vs : list value) :
  sub syms syms' ->
  sub ids ids' ->
  forallb (value_statics_among syms ids) vs = true ->
  forallb (value_statics_among syms' ids') vs = true.
Proof.
  intros Hs Hi. rewrite !forallb_forall. intros H v Hin.
  exact (value_among_sub _ _ _ _ Hs Hi v (H v Hin)).
Qed.

Lemma allocated_among_sub (syms syms' : list symbol) (ids ids' : list code_id)
    (allocated : statically_allocated_value) :
  sub syms syms' ->
  sub ids ids' ->
  allocated_statics_among syms ids allocated = true ->
  allocated_statics_among syms' ids' allocated = true.
Proof.
  intros Hs Hi.
  destruct allocated
    as [code slots | slot symbols decls slots | tag fields | contents
       | bytes | elements | c];
    simpl; rewrite ?andb_true_iff.
  - intros [Hc Hslots]. split; [ exact (Hi code Hc) | ].
    exact (slots_among_sub _ _ _ _ _ Hs Hi Hslots).
  - intros [[Hsyms Hc] Hslots].
    split; [ split | exact (slots_among_sub _ _ _ _ _ Hs Hi Hslots) ].
    + exact (all_among_sub _ _ _ Hs Hsyms).
    + exact (all_among_sub _ _ _ Hi Hc).
  - apply values_among_sub; assumption.
  - apply value_among_sub; assumption.
  - auto.
  - apply values_among_sub; assumption.
  - auto.
Qed.

(* ======================================================================== *)
(* What evaluation makes                                                    *)
(* ======================================================================== *)

Section Evaluation.

Variable syms : list symbol.
Variable ids : list code_id.

Let P (v : value) : Prop := value_statics_among syms ids v = true.

Lemma eval_simple_among (vars : env) (s : simple) (v : value) :
  env_values_all P vars ->
  simple_statics_in_scope syms s = true ->
  eval_simple vars s = Some v ->
  P v.
Proof.
  intros Hvars. destruct s as [x | sym | c]; simpl.
  - intros _ Hlookup. exact (env_values_all_lookup P vars x v Hvars Hlookup).
  - intros Hsym [= <-]. exact Hsym.
  - intros _ [= <-]. reflexivity.
Qed.

Lemma eval_simples_among (vars : env) (args : list simple) :
  env_values_all P vars ->
  ∀ vs : list value,
    simples_statics_in_scope syms args = true ->
    eval_simples vars args = Some vs ->
    forallb (value_statics_among syms ids) vs = true.
Proof.
  intros Hvars. unfold simples_statics_in_scope.
  induction args as [ | s args IH]; intros vs; simpl.
  - intros _ [= <-]. reflexivity.
  - intros Hav. apply andb_true_iff in Hav as [Hs Hargs].
    destruct (eval_simple vars s) as [v | ] eqn:Heval; [ | discriminate ].
    destruct (eval_simples vars args) as [rest | ]; [ | discriminate ].
    simpl. intros [= <-]. simpl.
    rewrite (eval_simple_among vars s v Hvars Hs Heval), (IH rest Hargs);
      reflexivity.
Qed.

Lemma eval_value_slots_among (vars : env)
    (slots : list (value_slot * simple)) :
  env_values_all P vars ->
  ∀ evaluated : list (value_slot * value),
    value_slots_statics_in_scope syms slots = true ->
    eval_value_slots vars slots = Some evaluated ->
    forallb (fun '(_, v) => value_statics_among syms ids v) evaluated = true.
Proof.
  intros Hvars. unfold value_slots_statics_in_scope.
  induction slots as [ | [slot s] slots IH]; intros evaluated; simpl.
  - intros _ [= <-]. reflexivity.
  - intros Hav. apply andb_true_iff in Hav as [Hs Hslots].
    destruct (eval_simple vars s) as [v | ] eqn:Heval; [ | discriminate ].
    destruct (eval_value_slots vars slots) as [rest | ]; [ | discriminate ].
    simpl. intros [= <-]. simpl.
    rewrite (eval_simple_among vars s v Hvars Hs Heval), (IH rest Hslots);
      reflexivity.
Qed.

(* What the static environment binds its symbols to mentions only what
   is in scope. *)
Definition statics_among (statics : static_env) : Prop :=
  Forall
    (fun binding => allocated_statics_among syms ids (snd binding) = true)
    (static_values statics).

Lemma statics_among_lookup (statics : static_env) (s : symbol)
    (allocated : statically_allocated_value) :
  statics_among statics ->
  lookup s (static_values statics) = Some allocated ->
  allocated_statics_among syms ids allocated = true.
Proof.
  intros Hstatics Hlookup. apply lookup_in in Hlookup.
  exact (proj1 (Forall_forall _ _) Hstatics _ Hlookup).
Qed.

Lemma lookup_decl_among (slot : function_slot)
    (decls : list (function_slot * code_id)) (code : code_id) :
  all_among (map snd decls) ids = true ->
  lookup slot decls = Some code ->
  among code ids = true.
Proof.
  intros Hall Hlookup. apply (proj1 (all_among_spec _ _) Hall).
  apply among_In, in_map_iff. exists (slot, code).
  split; [ reflexivity | exact (lookup_in _ _ _ Hlookup) ].
Qed.

Lemma closure_in_set_among (statics : static_env) (v : value)
    (slot : function_slot) (decls : list (function_slot * code_id))
    (slots : list (value_slot * value)) :
  statics_among statics ->
  P v ->
  closure_in_set statics v = Some (slot, decls, slots) ->
  all_among (map snd decls) ids = true
  ∧ forallb (fun '(_, v) => value_statics_among syms ids v) slots = true.
Proof.
  intros Hstatics Hv. unfold P in Hv.
  destruct v
    as [c | code vslots | vslot vdecls vslots | s | tag fields
       | elements | c | l | callee args arity];
    simpl in Hv |- *; try discriminate.
  - intros [= <- <- <-]. apply andb_true_iff in Hv as [Hc Hslots].
    split; [ | exact Hslots ]. simpl. rewrite Hc. reflexivity.
  - destruct (lookup vslot vdecls); [ | discriminate ].
    simpl. intros [= <- <- <-]. apply andb_true_iff in Hv. exact Hv.
  - destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hlookup;
      [ | discriminate ].
    pose proof (statics_among_lookup statics s allocated Hstatics Hlookup)
      as Hallocated.
    destruct allocated
      as [code aslots | aslot symbols adecls aslots | tag fields | contents
         | bytes | elements | ];
      simpl in Hallocated; try discriminate.
    + intros [= <- <- <-]. apply andb_true_iff in Hallocated as [Hc Hslots].
      split; [ | exact Hslots ]. simpl. rewrite Hc. reflexivity.
    + destruct (lookup aslot adecls); [ | discriminate ].
      simpl. intros [= <- <- <-].
      apply andb_true_iff in Hallocated as [Hrest Hslots].
      apply andb_true_iff in Hrest as [_ Hc]. auto.
Qed.

Lemma closure_of_among (statics : static_env) (v : value) (code : code_id)
    (slots : list (value_slot * value)) :
  statics_among statics ->
  P v ->
  closure_of statics v = Some (code, slots) ->
  among code ids = true
  ∧ forallb (fun '(_, v) => value_statics_among syms ids v) slots = true.
Proof.
  intros Hstatics Hv Hclosure.
  apply (proj1 (closure_in_set_of statics v code slots)) in Hclosure
    as (slot & decls & Hset & Hlookup).
  destruct (closure_in_set_among statics v slot decls slots Hstatics Hv Hset)
    as [Hdecls Hslots].
  split; [ exact (lookup_decl_among slot decls code Hdecls Hlookup) | ].
  exact Hslots.
Qed.

Lemma lookup_slot_among (slot : value_slot)
    (slots : list (value_slot * value)) (v : value) :
  forallb (fun '(_, v) => value_statics_among syms ids v) slots = true ->
  lookup slot slots = Some v ->
  P v.
Proof.
  intros Hslots Hlookup. apply lookup_in in Hlookup.
  exact (proj1 (forallb_forall _ _) Hslots (slot, v) Hlookup).
Qed.

Lemma project_function_slot_among (statics : static_env)
    (move_from move_to : function_slot) (v w : value) :
  statics_among statics ->
  P v ->
  project_function_slot statics move_from move_to v = Some w ->
  P w.
Proof.
  intros Hstatics Hv. unfold P in Hv |- *. unfold project_function_slot.
  assert (Hunnamed :
    (if Pos.eqb move_from unnamed_function_slot
     then if Pos.eqb move_to unnamed_function_slot then Some v else None
     else None)
    = Some w ->
    value_statics_among syms ids w = true).
  { destruct (Pos.eqb move_from unnamed_function_slot); [ | discriminate ].
    destruct (Pos.eqb move_to unnamed_function_slot); [ | discriminate ].
    intros [= <-]. exact Hv. }
  destruct v
    as [c | code vslots | vslot vdecls vslots | s | tag fields
       | elements | c | l | callee args arity];
    try discriminate.
  - exact Hunnamed.
  - destruct (Pos.eqb vslot move_from); [ | discriminate ].
    destruct (lookup vslot vdecls); [ | discriminate ].
    destruct (lookup move_to vdecls); [ | discriminate ].
    simpl. intros [= <-]. exact Hv.
  - destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hlookup;
      [ | discriminate ].
    pose proof (statics_among_lookup statics s allocated Hstatics Hlookup)
      as Hallocated.
    destruct allocated
      as [code aslots | aslot symbols adecls aslots | tag fields | contents
         | bytes | elements | ];
      try discriminate.
    + exact Hunnamed.
    + destruct (Pos.eqb aslot move_from); [ | discriminate ].
      destruct (lookup aslot adecls); [ | discriminate ].
      destruct (lookup move_to adecls); [ | discriminate ].
      destruct (lookup move_to symbols) as [s' | ] eqn:Hsym;
        [ | discriminate ].
      simpl. intros [= <-]. simpl. simpl in Hallocated.
      apply andb_true_iff in Hallocated as [Hrest _].
      apply andb_true_iff in Hrest as [Hsymbols _].
      apply (proj1 (all_among_spec _ _) Hsymbols).
      apply among_In, in_map_iff. exists (move_to, s').
      split; [ reflexivity | exact (lookup_in _ _ _ Hsym) ].
Qed.

Lemma eval_prim_among (statics : static_env) (p : prim) (vs : list value)
    (v : value) :
  statics_among statics ->
  forallb (value_statics_among syms ids) vs = true ->
  eval_prim statics p vs = Some v ->
  P v.
Proof.
  intros Hstatics Hvs.
  apply (eval_prim_preserves P).
  - intros c. split; reflexivity.
  - intros tag fields. unfold P. simpl. apply forallb_forall.
  - intros elements. unfold P. simpl. apply forallb_forall.
  - intros s tag fields _ Hlookup.
    pose proof (statics_among_lookup statics s _ Hstatics Hlookup)
      as Hfields.
    simpl in Hfields. apply forallb_forall. exact Hfields.
  - intros s elements _ Hlookup.
    pose proof (statics_among_lookup statics s _ Hstatics Hlookup)
      as Helements.
    simpl in Helements. apply forallb_forall. exact Helements.
  - apply forallb_forall. exact Hvs.
Qed.

Lemma eval_named_among (statics : static_env) (vars : env) (d : named)
    (v : value) :
  statics_among statics ->
  env_values_all P vars ->
  named_statics_in_scope syms ids d = true ->
  eval_named statics vars d = Some v ->
  P v.
Proof.
  intros Hstatics Hvars.
  destruct d as [s | p args | code slots | slot decls slots | slot closure
                | from slot closure | from to closure];
    simpl.
  - apply eval_simple_among, Hvars.
  - intros Hargs.
    destruct (eval_simples vars args) as [vs | ] eqn:Heval; [ | discriminate ].
    simpl.
    exact (eval_prim_among statics p vs v Hstatics
             (eval_simples_among vars args Hvars vs Hargs Heval)).
  - intros H. apply andb_true_iff in H as [Hc Hslots].
    destruct (eval_value_slots vars slots) as [evaluated | ] eqn:Heval;
      [ | discriminate ].
    simpl. intros [= <-]. unfold P. simpl. rewrite Hc.
    exact (eval_value_slots_among vars slots Hvars evaluated Hslots Heval).
  - intros H. apply andb_true_iff in H as [Hc Hslots].
    destruct (lookup slot decls); [ | discriminate ]. simpl.
    destruct (eval_value_slots vars slots) as [evaluated | ] eqn:Heval;
      [ | discriminate ].
    simpl. intros [= <-]. unfold P. simpl. rewrite Hc.
    exact (eval_value_slots_among vars slots Hvars evaluated Hslots Heval).
  - intros Hclosure.
    destruct (eval_simple vars closure) as [w | ] eqn:Heval;
      [ | discriminate ].
    simpl.
    destruct (closure_of statics w) as [[code wslots] | ] eqn:Hof;
      [ | discriminate ].
    simpl.
    destruct (closure_of_among statics w code wslots Hstatics
                (eval_simple_among vars closure w Hvars Hclosure Heval) Hof)
      as [_ Hslots].
    exact (lookup_slot_among slot wslots v Hslots).
  - intros Hclosure.
    destruct (eval_simple vars closure) as [w | ] eqn:Heval;
      [ | discriminate ].
    simpl. unfold project_value_slot.
    destruct (closure_in_set statics w) as [[[at_slot decls] wslots] | ]
      eqn:Hset; [ | discriminate ].
    simpl.
    destruct (Pos.eqb at_slot from); [ | discriminate ].
    destruct (closure_in_set_among statics w at_slot decls wslots Hstatics
                (eval_simple_among vars closure w Hvars Hclosure Heval) Hset)
      as [_ Hslots].
    exact (lookup_slot_among slot wslots v Hslots).
  - intros Hclosure.
    destruct (eval_simple vars closure) as [w | ] eqn:Heval;
      [ | discriminate ].
    simpl.
    exact (project_function_slot_among statics from to w v Hstatics
             (eval_simple_among vars closure w Hvars Hclosure Heval)).
Qed.

(* The same of [exec_named], with the values that the heap holds. *)
Lemma exec_named_among (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (vars : env)
    (d : named) (v : value) :
  statics_among statics ->
  env_values_all P vars ->
  heap_values_all P hp ->
  named_statics_in_scope syms ids d = true ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  P v ∧ heap_values_all P hp'.
Proof.
  intros Hstatics Hvars Hheap Hd.
  apply (exec_named_values_all P statics hp hp' rd rd' vars d v).
  - intros c. split; reflexivity.
  - intros tag fields. unfold P. simpl. apply forallb_forall.
  - intros elements. unfold P. simpl. apply forallb_forall.
  - intros s tag fields _ Hlookup.
    pose proof (statics_among_lookup statics s _ Hstatics Hlookup)
      as Hfields.
    simpl in Hfields. apply forallb_forall. exact Hfields.
  - intros s elements _ Hlookup.
    pose proof (statics_among_lookup statics s _ Hstatics Hlookup)
      as Helements.
    simpl in Helements. apply forallb_forall. exact Helements.
  - intros l. reflexivity.
  - exact Hheap.
  - intros p args vs -> Hvs. apply forallb_forall.
    exact (eval_simples_among vars args Hvars vs Hd Hvs).
  - intros _ w. exact (eval_named_among statics vars d w Hstatics Hvars Hd).
Qed.

Lemma eval_static_const_among (vars : env) (const : static_const)
    (allocated : statically_allocated_value) :
  env_values_all P vars ->
  static_const_statics_in_scope syms ids const = true ->
  eval_static_const vars const = Some allocated ->
  allocated_statics_among syms ids allocated = true.
Proof.
  intros Hvars.
  destruct const as [code slots | slot symbols decls slots | tag fields
                    | contents | bytes | elements];
    simpl.
  - intros H. apply andb_true_iff in H as [Hc Hslots].
    destruct (eval_value_slots vars slots) as [evaluated | ] eqn:Heval;
      [ | discriminate ].
    simpl. intros [= <-]. simpl. rewrite Hc.
    exact (eval_value_slots_among vars slots Hvars evaluated Hslots Heval).
  - intros H. apply andb_true_iff in H as [H Hslots].
    destruct (lookup slot decls); [ | discriminate ]. simpl.
    destruct (eval_value_slots vars slots) as [evaluated | ] eqn:Heval;
      [ | discriminate ].
    simpl. intros [= <-]. simpl. rewrite H.
    exact (eval_value_slots_among vars slots Hvars evaluated Hslots Heval).
  - intros Hfields.
    destruct (eval_simples vars fields) as [vs | ] eqn:Heval;
      [ | discriminate ].
    simpl. intros [= <-]. simpl.
    exact (eval_simples_among vars fields Hvars vs Hfields Heval).
  - intros Hcontents.
    destruct (eval_simple vars contents) as [v | ] eqn:Heval;
      [ | discriminate ].
    simpl. intros [= <-]. simpl.
    exact (eval_simple_among vars contents v Hvars Hcontents Heval).
  - intros _ [= <-]. reflexivity.
  - intros Helements.
    destruct (eval_simples vars elements) as [vs | ] eqn:Heval;
      [ | discriminate ].
    simpl. intros [= <-]. simpl.
    exact (eval_simples_among vars elements Hvars vs Helements Heval).
Qed.

End Evaluation.

(* ======================================================================== *)
(* What a group binds its symbols to                                        *)
(* ======================================================================== *)

(* What a symbol is bound to after a group: what it was bound to before,
   or the value of a constant of the group. *)
Lemma eval_static_bindings_values (vars : env)
    (bindings : list static_binding) :
  ∀ (statics statics' : static_env) (s : symbol)
    (allocated : statically_allocated_value),
    eval_static_bindings vars bindings statics = Some statics' ->
    (s, allocated) ∈ static_values statics' ->
    (s, allocated) ∈ static_values statics
    ∨ ∃ const : static_const,
        Bind_symbol s const ∈ bindings
        ∧ eval_static_const vars const = Some allocated.
Proof.
  induction bindings as [ | [id' c' | id' | s' const'] bindings IH];
    intros statics statics' s allocated; simpl.
  - intros [= <-]. auto.
  - destruct (lookup id' (static_code statics)); [ discriminate | ].
    intros Heval Hin.
    destruct (IH _ _ s allocated Heval Hin) as [Hold | (const & Hc & He)];
      [ auto | ].
    right. exists const. auto.
  - intros Heval Hin.
    destruct (IH _ _ s allocated Heval Hin) as [Hold | (const & Hc & He)];
      [ auto | ].
    right. exists const. auto.
  - destruct (lookup s' (static_values statics)); [ discriminate | ].
    destruct (eval_static_const vars const') as [allocated' | ] eqn:Hconst;
      simpl; [ | discriminate ].
    intros Heval Hin.
    destruct (IH _ _ s allocated Heval Hin) as [Hold | (const & Hc & He)].
    + simpl in Hold. destruct Hold as [[= <- <-] | Hold]; [ | auto ].
      right. exists const'. auto.
    + right. exists const. auto.
Qed.

(* Every constant of a group mentions only what is in scope after the
   group. *)
Lemma group_consts_all (all : list symbol) (ids : list code_id)
    (bindings : list static_binding) :
  ∀ before : list symbol,
    sub before all ->
    sub (flat_map symbols_of_binding bindings) all ->
    group_consts_in_scope before all ids bindings = true ->
    ∀ (s : symbol) (const : static_const),
      Bind_symbol s const ∈ bindings ->
      static_const_statics_in_scope all ids const = true.
Proof.
  induction bindings as [ | [id' c' | id' | s' const'] bindings IH];
    intros before Hbefore Hgroup; simpl; intros Hscope s const Hin.
  - destruct Hin.
  - destruct Hin as [[=] | Hin].
    exact (IH before Hbefore Hgroup Hscope _ _ Hin).
  - destruct Hin as [[=] | Hin].
    exact (IH before Hbefore Hgroup Hscope _ _ Hin).
  - simpl in Hgroup.
    assert (Hs' : among s' all = true).
    { apply Hgroup. unfold among. simpl. rewrite Pos.eqb_refl. reflexivity. }
    assert (Hrest : sub (flat_map symbols_of_binding bindings) all).
    { intros name Hn. apply Hgroup. unfold among in Hn |- *. simpl.
      rewrite Hn. apply orb_true_r. }
    destruct (is_closure_const const');
      apply andb_true_iff in Hscope as [Hconst Hscope].
    + destruct Hin as [[= <- <-] | Hin]; [ exact Hconst | ].
      exact (IH before Hbefore Hrest Hscope _ _ Hin).
    + destruct Hin as [[= <- <-] | Hin].
      * exact (static_const_scope_sub _ _ _ _ _ Hbefore (sub_refl _) Hconst).
      * refine (IH (s' :: before) _ Hrest Hscope _ _ Hin).
        intros name Hn. unfold among in Hn. simpl in Hn.
        apply orb_true_iff in Hn as [Hn | Hn]; [ | exact (Hbefore name Hn) ].
        apply Pos.eqb_eq in Hn. subst name. exact Hs'.
Qed.

(* The symbols that a group binds to closures are among those that it
   binds. *)
Lemma closure_symbols_sub (bindings : list static_binding) :
  sub (flat_map closure_symbols_of_binding bindings)
    (flat_map symbols_of_binding bindings).
Proof.
  intros name. unfold among.
  induction bindings as [ | [id c | id | s const] bindings IH]; simpl; auto.
  destruct (is_closure_const const); simpl.
  - intros Hn. apply orb_true_iff in Hn as [Hn | Hn];
      [ rewrite Hn; reflexivity | rewrite (IH Hn); apply orb_true_r ].
  - intros Hn. rewrite (IH Hn). apply orb_true_r.
Qed.

(* ======================================================================== *)
(* The invariant of a run, on values                                        *)
(* ======================================================================== *)

Lemma env_values_all_impl (P P' : value -> Prop) (vars : env) :
  (∀ v : value, P v -> P' v) ->
  env_values_all P vars -> env_values_all P' vars.
Proof.
  intros Himpl. unfold env_values_all. apply Forall_impl.
  intros binding. apply Himpl.
Qed.

Lemma forallb_Forall_among (syms : list symbol) (ids : list code_id)
    (vs : list value) :
  forallb (value_statics_among syms ids) vs = true
  <-> Forall (fun v => value_statics_among syms ids v = true) vs.
Proof. rewrite forallb_forall, Forall_forall. reflexivity. Qed.

Section Values.

Variable deleted : list code_id.

(* A step keeps what the state holds bound, given the invariant on
   terms. *)
Lemma values_step (s : state) :
  terms_inv deleted s ->
  state_statics_bound deleted s ->
  match step s with
  | Step s' => state_statics_bound deleted s'
  | Exit _ vs _ _ => Forall (value_statics_bound deleted (state_statics s)) vs
  | Undefined => True
  end.
Proof.
  intros (P & S & D & K & C) Hvalues.
  pose proof Hvalues as (Hvars & Hconts & Hstatics & Hheap).
  apply (step_values_all (value_statics_bound deleted)
           (allocated_statics_bound deleted) s);
    [ | exact Hvalues | | | | | ].
  - intros statics callee args arity. unfold value_statics_bound. simpl.
    rewrite andb_true_iff, forallb_Forall_among. reflexivity.
  - (* the fields of a tuple: of a block, or of a symbol bound to one *)
    intros size tuple fields Htuple Hfields.
    apply (tuple_fields_of_blocks _ (state_statics s) size tuple fields);
      [ | exact Hfields ].
    intros tag block Hblock. apply forallb_Forall_among.
    destruct (block_of_cases _ _ _ _ Hblock) as [-> | (sym & -> & Hs)].
    + exact Htuple.
    + exact (statics_among_lookup _ _ (state_statics s) sym _ Hstatics Hs).
  - intros x kind d body v hp' rd' He Hexec. rewrite He in S. simpl in S.
    apply andb_true_iff in S as [S _].
    exact (exec_named_among _ _ (state_statics s) (state_heap s) hp' _ _
             (state_vars s) d v Hstatics Hvars Hheap S Hexec).
  - intros bindings body statics' He Heval. rewrite He in S, D.
    cbn [expr_statics_in_scope static_names] in S, D.
    apply andb_true_iff in S as [S _].
    apply andb_true_iff in S as [Sgroup _].
    rewrite all_among_app in D. apply andb_true_iff in D as [Dgroup _].
    set (statics := state_statics s) in *. set (vars := state_vars s) in *.
    assert (Hsyms :
      sub (flat_map symbols_of_binding bindings ++ bound_symbols statics)
        (bound_symbols statics')).
    { intros name Hn.
      rewrite
        (proj1 (eval_static_bindings_bound vars bindings statics statics'
                  name Heval)).
      rewrite among_app in Hn. exact Hn. }
    assert (Hids :
      sub
        (flat_map declared_code_ids_of_binding bindings
         ++ ids_of deleted statics)
        (ids_of deleted statics')).
    { intros name Hn. unfold ids_of in Hn |- *.
      rewrite !among_app, declared_code_ids_split in Hn.
      rewrite among_app,
        (proj2 (eval_static_bindings_bound vars bindings statics statics'
                  name Heval)).
      apply orb_true_iff in Hn as [Hn | Hn];
        apply orb_true_iff in Hn as [Hn | Hn].
      - rewrite Hn. reflexivity.
      - rewrite (proj1 (all_among_spec _ _) Dgroup name Hn).
        apply orb_true_r.
      - rewrite Hn, orb_true_r. reflexivity.
      - rewrite Hn. apply orb_true_r. }
    assert (Hsyms_old : sub (bound_symbols statics) (bound_symbols statics'))
      by exact (sub_trans _ _ _ (sub_app_r _ _) Hsyms).
    assert (Hids_old : sub (ids_of deleted statics) (ids_of deleted statics'))
      by exact (sub_trans _ _ _ (sub_app_r _ _) Hids).
    split.
    + intros v. apply value_among_sub; assumption.
    + apply Forall_forall. intros [sym allocated] Hin. cbn [snd].
      unfold allocated_statics_bound.
      destruct (eval_static_bindings_values vars bindings statics statics'
                  sym allocated Heval Hin) as [Hold | (const & Hc & He')].
      * pose proof (proj1 (Forall_forall _ _) Hstatics _ Hold) as Hall.
        cbn [snd] in Hall.
        exact (allocated_among_sub _ _ _ _ _ Hsyms_old Hids_old Hall).
      * refine (allocated_among_sub _ _ _ _ _ Hsyms Hids _).
        refine (eval_static_const_among _ _ vars const allocated _ _ He').
        -- revert Hvars. apply env_values_all_impl. intros v.
           apply value_among_sub; [ apply sub_app_r | apply sub_app_r ].
        -- refine (group_consts_all _ _ bindings _ _ _ Sgroup sym const Hc).
           ++ apply sub_app; [ | apply sub_app_r ].
              exact (sub_trans _ _ _ (closure_symbols_sub bindings)
                       (sub_app_l _ _)).
           ++ apply sub_app_l.
  - intros ap callee vs extra He Hcallee Hvs Hextra. rewrite He in S.
    cbn [expr_statics_in_scope] in S. unfold apply_statics_in_scope in S.
    apply andb_true_iff in S as [S Sextra].
    apply andb_true_iff in S as [S Sargs].
    apply andb_true_iff in S as [Scallee _].
    refine (conj _ (conj _ _)).
    + exact (eval_simple_among _ _ _ _ _ Hvars Scallee Hcallee).
    + apply forallb_Forall_among.
      exact (eval_simples_among _ _ _ _ Hvars vs Sargs Hvs).
    + apply forallb_Forall_among.
      exact (eval_simples_among _ _ _ _ Hvars extra Sextra Hextra).
  - intros ac vs He Hvs. rewrite He in S. cbn [expr_statics_in_scope] in S.
    apply forallb_Forall_among.
    exact (eval_simples_among _ _ _ _ Hvars vs S Hvs).
Qed.

End Values.

(* ======================================================================== *)
(* No step looks up what is not bound                                       *)
(* ======================================================================== *)

Lemma bound_not_unbound (deleted : list code_id) (statics : static_env)
    (v : value) :
  value_statics_bound deleted statics v -> unbound_symbol statics v = false.
Proof.
  unfold value_statics_bound. destruct v; simpl; intros H; try reflexivity.
  unfold symbol_bound. unfold bound_symbols in H.
  rewrite <- is_bound_among in H. unfold is_bound in H.
  destruct (lookup s (static_values statics));
    [ reflexivity | discriminate H ].
Qed.

Lemma closure_called_bound (deleted : list code_id) (statics : static_env)
    (callee : value) :
  value_statics_bound deleted statics callee ->
  value_statics_bound deleted statics (closure_called callee).
Proof.
  unfold value_statics_bound, closure_called. destruct callee; simpl; auto.
  intros H. apply andb_true_iff in H as [H _]. exact H.
Qed.

Lemma calls_bound (deleted : list code_id) (statics : static_env)
    (callee : value) :
  Forall
    (fun binding => allocated_statics_bound deleted statics (snd binding))
    (static_values statics) ->
  value_statics_bound deleted statics callee ->
  calls_unbound deleted statics callee = false.
Proof.
  intros Hstatics Hcallee. apply closure_called_bound in Hcallee.
  unfold calls_unbound. rewrite (bound_not_unbound _ _ _ Hcallee). simpl.
  destruct (closure_of statics (closure_called callee)) as [[called slots] | ]
    eqn:Hof; [ | reflexivity ].
  destruct (closure_of_among _ _ statics _ called slots Hstatics Hcallee Hof)
    as [Hc _].
  rewrite among_app in Hc.
  replace (code_id_bound statics called)
    with (among called (bound_code_ids statics)).
  - rewrite Hc. reflexivity.
  - unfold bound_code_ids. rewrite <- is_bound_among. reflexivity.
Qed.

Lemma called_by_cont_bound (P : value -> Prop) (c : cont_value) :
  ∀ (vs : list value) (v : value) (remaining : list value),
    cont_values_all P c -> Forall P vs ->
    called_by_cont c vs = Some (v, remaining) ->
    P v ∧ Forall P remaining.
Proof.
  induction c
    as [ h vars conts recursive | k | c IH extra
       | kept remaining_arity return_arity return_to _ exn_to _];
    intros vs v remaining; simpl; try discriminate.
  - intros [Hextra Hc] Hvs. apply IH; [ exact Hc | ].
    apply Forall_app. auto.
  - intros (Hkept & _) Hvs.
    destruct vs as [ | v' [ | v'' vs]]; try discriminate.
    intros [= <- <-]. split; [ exact (Forall_inv Hvs) | exact Hkept ].
Qed.

(* The tuple that a call reads is one of the values that the call has in
   hand: an argument, or an argument that a partial application kept. *)
Lemma reads_bound_tuple (deleted : list code_id) (statics : static_env)
    (callee : value) (vs : list value) :
  value_statics_bound deleted statics callee ->
  Forall (value_statics_bound deleted statics) vs ->
  reads_unbound_tuple statics callee vs = false.
Proof.
  intros Hcallee Hvs. unfold reads_unbound_tuple, tuple_read.
  destruct (closure_of statics (closure_called callee)) as [[called slots] | ];
    [ | reflexivity ].
  destruct (lookup called (static_code statics)) as [c | ]; [ | reflexivity ].
  destruct (code_is_tupled c); [ | reflexivity ].
  assert (Hall :
    Forall (value_statics_bound deleted statics) (arguments_called callee vs)).
  { unfold arguments_called. destruct callee; simpl; try exact Hvs.
    apply Forall_app. split; [ | exact Hvs ].
    unfold value_statics_bound in Hcallee. simpl in Hcallee.
    apply andb_true_iff in Hcallee as [_ Hkept].
    apply forallb_Forall_among. exact Hkept. }
  destruct (arguments_called callee vs) as [ | tuple rest]; [ reflexivity | ].
  simpl. exact (bound_not_unbound deleted statics tuple (Forall_inv Hall)).
Qed.

Lemma existsb_unbound_false (deleted : list code_id) (statics : static_env)
    (vs : list value) :
  Forall (value_statics_bound deleted statics) vs ->
  existsb (unbound_symbol statics) vs = false.
Proof.
  induction 1 as [ | v vs Hv _ IH]; simpl; [ reflexivity | ].
  rewrite (bound_not_unbound _ _ _ Hv), IH. reflexivity.
Qed.

Section Fault.

Variable deleted : list code_id.

Lemma no_scope_fault (s : state) :
  terms_inv deleted s ->
  state_statics_bound deleted s ->
  statics_scope_fault deleted s = false.
Proof.
  intros (_ & S & _) (Hvars & Hconts & Hstatics & _).
  unfold statics_scope_fault.
  destruct (state_expr s)
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    try reflexivity.
  - cbn [expr_statics_in_scope] in S. apply andb_true_iff in S as [S _].
    destruct d as [simple | p args | code slots | slot decls slots
                  | slot closure | from slot closure | from to closure];
      try reflexivity; simpl in S.
    + destruct (eval_simples (state_vars s) args) as [vs | ] eqn:Heval;
        [ | reflexivity ].
      apply (existsb_unbound_false deleted), forallb_Forall_among.
      exact (eval_simples_among _ _ _ _ Hvars vs S Heval).
    + destruct (eval_simple (state_vars s) closure) as [v | ] eqn:Heval;
        [ | reflexivity ].
      apply (bound_not_unbound deleted).
      exact (eval_simple_among _ _ _ _ _ Hvars S Heval).
    + destruct (eval_simple (state_vars s) closure) as [v | ] eqn:Heval;
        [ | reflexivity ].
      apply (bound_not_unbound deleted).
      exact (eval_simple_among _ _ _ _ _ Hvars S Heval).
    + destruct (eval_simple (state_vars s) closure) as [v | ] eqn:Heval;
        [ | reflexivity ].
      apply (bound_not_unbound deleted).
      exact (eval_simple_among _ _ _ _ _ Hvars S Heval).
  - cbn [expr_statics_in_scope] in S. unfold apply_statics_in_scope in S.
    apply andb_true_iff in S as [S _].
    apply andb_true_iff in S as [S Sargs].
    apply andb_true_iff in S as [Scallee _].
    destruct (eval_simple (state_vars s) (ap_callee ap)) as [callee | ]
      eqn:Heval; [ | reflexivity ].
    pose proof (eval_simple_among _ _ _ _ _ Hvars Scallee Heval) as Hcallee.
    rewrite (calls_bound deleted _ callee Hstatics Hcallee). simpl.
    destruct (eval_simples (state_vars s) (ap_args ap)) as [vs | ] eqn:Hargs;
      [ | reflexivity ].
    apply (reads_bound_tuple deleted); [ exact Hcallee | ].
    apply forallb_Forall_among.
    exact (eval_simples_among _ _ _ _ Hvars vs Sargs Hargs).
  - cbn [expr_statics_in_scope] in S.
    destruct (eval_simples (state_vars s) (ac_args ac)) as [vs | ] eqn:Heval;
      [ | reflexivity ].
    destruct (called_by_cont (resolve (ac_continuation ac) (state_conts s)) vs)
      as [[v remaining] | ] eqn:Hcalled; [ | reflexivity ].
    destruct (called_by_cont_bound _ _ vs v remaining
                (resolve_values_all _ _ _ Hconts)
                (proj1 (forallb_Forall_among _ _ _)
                   (eval_simples_among _ _ _ _ Hvars vs S Heval))
                Hcalled)
      as [Hv Hremaining].
    rewrite (calls_bound deleted _ v Hstatics Hv). simpl.
    exact (reads_bound_tuple deleted _ v remaining Hv Hremaining).
Qed.

End Fault.

(* ======================================================================== *)
(* The run of a unit                                                        *)
(* ======================================================================== *)

Lemma all_among_refl (names : list positive) : all_among names names = true.
Proof. apply all_among_spec. intros name H. exact H. Qed.

(* The state that a unit is run from has both invariants. *)
Lemma initial_state_inv (imports : list symbol) (u : unit_interface)
    (e : expr) (statics : static_env) (hp : heap) (rd : read_record) :
  well_formed_unit u e = true ->
  statics_in_scope imports e = true ->
  imports_bound imports statics ->
  heap_values_all (fun v => value_statics_among imports [] v = true) hp ->
  terms_inv (deleted_code_ids e) (State e [] [] statics hp rd)
  ∧ state_statics_bound (deleted_code_ids e) (State e [] [] statics hp rd).
Proof.
  intros Hwf Hscope (Hcode & Hbound & Hvalues) Hheap.
  assert (Hsyms : sub imports (bound_symbols statics)).
  { intros name Hn. unfold bound_symbols. rewrite <- is_bound_among.
    rewrite <- Hn, <- Hbound. reflexivity. }
  split.
  - unfold terms_inv. simpl_state.
    refine (conj _ (conj _ (conj _ (conj _ _)))).
    + exact (well_formed_unit_statics_outside_loops_and_code u e Hwf).
    + exact (expr_scope_sub e _ _ _ _ Hsyms (sub_nil _) Hscope).
    + apply all_among_refl.
    + intros k syms ids _ _. apply Good_outer.
    + unfold code_all_fine. rewrite Hcode. apply Forall_nil.
  - unfold state_statics_bound, state_values_all. simpl_state.
    refine (conj (Forall_nil _) (conj I (conj _ _))).
    + revert Hvalues. apply Forall_impl. intros binding.
      apply allocated_among_sub; [ exact Hsyms | apply sub_nil ].
    + intros l c v Hin Hv.
      exact (value_among_sub _ _ _ _ Hsyms (sub_nil _) v (Hheap l c v Hin Hv)).
Qed.

Lemma inv_run (deleted : list code_id) (n : nat) :
  ∀ s s' : state,
    terms_inv deleted s ->
    state_statics_bound deleted s ->
    run n s = Step s' ->
    terms_inv deleted s' ∧ state_statics_bound deleted s'.
Proof.
  induction n as [ | n IH]; intros s s' Hterms Hvalues; simpl.
  - intros [= <-]. auto.
  - pose proof (values_step deleted s Hterms Hvalues) as Hstep.
    destruct (step s) as [next | k vs | ] eqn:Hs; try discriminate.
    exact (IH next s' (terms_inv_step deleted s next Hterms Hs)
             Hstep).
Qed.

Theorem statics_bound_in_run : statics_bound_in_run_statement.
Proof.
  intros imports u e statics hp rd n s Hwf Hscope Himports Hheap Hrun.
  destruct
    (initial_state_inv imports u e statics hp rd Hwf Hscope Himports Hheap)
    as [Hterms Hvalues].
  destruct (inv_run (deleted_code_ids e) n _ s Hterms Hvalues
              Hrun) as [Hterms' Hvalues'].
  refine (conj Hvalues' (conj _ _)).
  - exact (no_scope_fault (deleted_code_ids e) s Hterms'
             Hvalues').
  - pose proof (values_step (deleted_code_ids e) s Hterms'
                  Hvalues') as Hstep.
    destruct (step s); auto.
Qed.

(* ======================================================================== *)
(* A code id that is declared deleted is not bound                          *)
(* ======================================================================== *)

(* Of three lists of names, the bound, the deleted and the declared:
   the first two are among the third, and where the third has no name
   twice, no name is in both of the first two. *)
Definition apart (bound deleted declared : list positive) : Prop :=
  incl bound declared
  ∧ incl deleted declared
  ∧ (NoDup declared -> ∀ name : positive, In name deleted -> ~ In name bound).

Lemma apart_nil : apart [] [] [].
Proof.
  refine (conj (incl_refl _) (conj (incl_refl _) _)). intros _ name [].
Qed.

Lemma apart_app (bound deleted declared bound' deleted' declared'
    : list positive) :
  apart bound deleted declared ->
  apart bound' deleted' declared' ->
  apart (bound ++ bound') (deleted ++ deleted') (declared ++ declared').
Proof.
  intros (Hb & Hd & Ha) (Hb' & Hd' & Ha').
  refine (conj (incl_app_app Hb Hb') (conj (incl_app_app Hd Hd') _)).
  intros Hnodup name Hdeleted Hbound.
  pose proof (NoDup_app_remove_r _ _ Hnodup) as Hleft.
  pose proof (NoDup_app_remove_l _ _ Hnodup) as Hright.
  assert (Hdisjoint : ∀ n : positive, In n declared -> ~ In n declared').
  { intros n Hn Hn'. revert Hnodup Hn Hn'. clear.
    induction declared as [ | first declared IH]; simpl;
      [ intros _ [] | ].
    intros Hnodup [-> | Hn] Hn'.
    - apply NoDup_cons_iff in Hnodup as [Hnot _]. apply Hnot.
      apply in_or_app. right. exact Hn'.
    - apply NoDup_cons_iff in Hnodup as [_ Hnodup].
      exact (IH Hnodup Hn Hn'). }
  apply in_app_or in Hdeleted as [Hdeleted | Hdeleted];
    apply in_app_or in Hbound as [Hbound | Hbound].
  - exact (Ha Hleft name Hdeleted Hbound).
  - exact (Hdisjoint name (Hd name Hdeleted) (Hb' name Hbound)).
  - exact (Hdisjoint name (Hb name Hbound) (Hd' name Hdeleted)).
  - exact (Ha' Hright name Hdeleted Hbound).
Qed.

Lemma binding_apart (binding : static_binding) :
  apart (code_ids_of_binding binding) (deleted_code_ids_of_binding binding)
    (declared_code_ids_of_binding binding).
Proof.
  destruct binding as [id c | id | s const]; simpl.
  - refine (conj (incl_refl _) (conj (incl_nil_l _) _)). intros _ name [].
  - refine (conj (incl_nil_l _) (conj (incl_refl _) _)). intros _ name _ [].
  - apply apart_nil.
Qed.

Lemma bindings_apart (bindings : list static_binding) :
  apart (flat_map code_ids_of_binding bindings)
    (flat_map deleted_code_ids_of_binding bindings)
    (flat_map declared_code_ids_of_binding bindings).
Proof.
  induction bindings as [ | binding bindings IH]; simpl;
    [ apply apart_nil | ].
  exact (apart_app _ _ _ _ _ _ (binding_apart binding) IH).
Qed.

Fixpoint static_names_apart_lists (e : expr) {struct e} :
  apart (static_names code_ids_of_binding e)
    (static_names deleted_code_ids_of_binding e)
    (static_names declared_code_ids_of_binding e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [static_names handler_static_names]; try apply apart_nil.
  - apply static_names_apart_lists.
  - exact (apart_app _ _ _ _ _ _ (bindings_apart bindings)
             (static_names_apart_lists body)).
  - exact (apart_app _ _ _ _ _ _ (static_names_apart_lists handler)
             (static_names_apart_lists body)).
  - refine (apart_app _ _ _ _ _ _ _ (static_names_apart_lists body)).
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      cbn [flat_map handler_static_names]; [ apply apart_nil | ].
    exact (apart_app _ _ _ _ _ _ (static_names_apart_lists handler) IH).
Qed.

(* A unit whose static names are apart binds no code id that it declares
   deleted. *)
Lemma deleted_not_bound (imports : list symbol) (e : expr) (id : code_id) :
  static_names_apart imports e = true ->
  among id (deleted_code_ids e) = true ->
  among id (static_names code_ids_of_binding e) = false.
Proof.
  unfold static_names_apart, deleted_code_ids. intros Hapart Hdeleted.
  apply andb_true_iff in Hapart as [_ Hdistinct].
  apply distinct_NoDup in Hdistinct. apply among_In in Hdeleted.
  destruct (static_names_apart_lists e) as (_ & _ & Hlists).
  destruct (among id (static_names code_ids_of_binding e)) eqn:Hbound;
    [ | reflexivity ].
  apply among_In in Hbound.
  destruct (Hlists Hdistinct id Hdeleted Hbound).
Qed.

(* ======================================================================== *)
(* A test inside a test                                                     *)
(* ======================================================================== *)

(* The shape that Simplify gives a match of three cases whose arms are
   lifted: the handler of the inner test is defined above the group that
   binds the symbols of the arms, and mentions two of them. It is not at
   toplevel, since its body also applies the continuation of the whole
   match; the symbols are in scope in it because the body of its
   [Let_cont] binds them. The arms here are blocks, where Simplify has
   closures. The
   continuations are [1] and [2] for the unit, [3] for the match and [4]
   for the inner test; the symbol [1] is what the unit returns, and
   [4], [2] and [3] are those of the arms. *)
Definition nested_test_interface : unit_interface :=
  Unit_interface 1%positive 2%positive.

Definition nested_test_arm (i : Z) : static_const :=
  Static_const_block 0 [Const (Tagged_immediate i)].

Definition nested_test_unit (scrutinee : Z) : expr :=
  Let_cont_non_recursive 3%positive
    (Handler [1%positive] (make_handler_metadata [Value])
       (Let_static
          [Bind_symbol 1%positive (Static_const_block 0 [Var 1%positive])]
          (Apply_cont (Apply_cont_expr 1%positive [Symbol 1%positive] None))))
    (Let 2%positive (Naked_number Kind_naked_immediate)
       (Simple (Const (Naked_immediate scrutinee)))
       (Let_cont_non_recursive 4%positive
          (Handler [] (make_handler_metadata [])
             (Switch (Var 2%positive)
                [(1%Z, Apply_cont_expr 3%positive [Symbol 2%positive] None);
                 (2%Z, Apply_cont_expr 3%positive [Symbol 3%positive] None)]))
          (Let_static
             [Bind_symbol 4%positive (nested_test_arm 0);
              Bind_symbol 2%positive (nested_test_arm 1);
              Bind_symbol 3%positive (nested_test_arm 2)]
             (Switch (Var 2%positive)
                [(0%Z, Apply_cont_expr 3%positive [Symbol 4%positive] None);
                 (1%Z, Apply_cont_expr 4%positive [] None);
                 (2%Z, Apply_cont_expr 4%positive [] None)])))).

Example nested_test_passes :
  well_formed_unit nested_test_interface (nested_test_unit 1) = true
  ∧ statics_in_scope [] (nested_test_unit 1) = true
  ∧ static_names_apart [] (nested_test_unit 1) = true
  ∧ interface_conts_not_rebound nested_test_interface (nested_test_unit 1)
    = true.
Proof. repeat split; vm_compute; reflexivity. Qed.

(* It goes through the inner test and returns. *)
Example nested_test_returns :
  run 12
    (State (nested_test_unit 1) [] [] (Static_env [] []) empty_heap no_reads)
  = Exit 1%positive [Value_symbol 1%positive] empty_heap no_reads.
Proof. vm_compute. reflexivity. Qed.

(* ======================================================================== *)
(* Handlers that mention what the body binds                                *)
(* ======================================================================== *)

Definition handlers_interface : unit_interface :=
  Unit_interface 1%positive 2%positive.

(* A recursive handler that mentions a symbol which the body of its
   [Let_cont] binds, and returns it. *)
Definition recursive_handler_unit : expr :=
  Let_cont_recursive
    [(3%positive,
      Handler [] (make_handler_metadata [])
        (Apply_cont (Apply_cont_expr 1%positive [Symbol 1%positive] None)))]
    (Let_static [Bind_symbol 1%positive (nested_test_arm 0)]
       (Apply_cont (Apply_cont_expr 3%positive [] None))).

Example recursive_handler_passes :
  well_formed_unit handlers_interface recursive_handler_unit = true
  ∧ statics_in_scope [] recursive_handler_unit = true
  ∧ static_names_apart [] recursive_handler_unit = true
  ∧ interface_conts_not_rebound handlers_interface recursive_handler_unit
    = true.
Proof. repeat split; vm_compute; reflexivity. Qed.

Example recursive_handler_returns :
  run 4
    (State recursive_handler_unit [] [] (Static_env [] []) empty_heap no_reads)
  = Exit 1%positive [Value_symbol 1%positive] empty_heap no_reads.
Proof. vm_compute. reflexivity. Qed.

(* A handler, of the continuation [3], that mentions a symbol which the
   body of its [Let_cont] binds in the handler of another continuation,
   [4], and not on its spine. The handler of [3] is not at toplevel: the
   body of its [Let_cont] can also return. The handler of [4] is, so it
   may bind the symbol, and it always runs before [3] is applied: this
   is what row 1 gives, and what the check relies on. *)
Definition sibling_handler_unit (scrutinee : Z) : expr :=
  Let 1%positive (Naked_number Kind_naked_immediate)
    (Simple (Const (Naked_immediate scrutinee)))
    (Let_cont_non_recursive 3%positive
       (Handler [] (make_handler_metadata [])
          (Apply_cont (Apply_cont_expr 1%positive [Symbol 1%positive] None)))
       (Let_cont_non_recursive 4%positive
          (Handler [] (make_handler_metadata [])
             (Let_static [Bind_symbol 1%positive (nested_test_arm 0)]
                (Switch (Var 1%positive)
                   [(0%Z, Apply_cont_expr 3%positive [] None);
                    (1%Z,
                     Apply_cont_expr 1%positive [Const (Tagged_immediate 7)]
                       None)])))
          (Apply_cont (Apply_cont_expr 4%positive [] None)))).

Example sibling_handler_passes :
  well_formed_unit handlers_interface (sibling_handler_unit 0) = true
  ∧ statics_in_scope [] (sibling_handler_unit 0) = true
  ∧ static_names_apart [] (sibling_handler_unit 0) = true
  ∧ interface_conts_not_rebound handlers_interface (sibling_handler_unit 0)
    = true.
Proof. repeat split; vm_compute; reflexivity. Qed.

Example sibling_handler_returns :
  run 8
    (State (sibling_handler_unit 0) [] [] (Static_env [] []) empty_heap
       no_reads)
  = Exit 1%positive [Value_symbol 1%positive] empty_heap no_reads.
Proof. vm_compute. reflexivity. Qed.

(* A unit that leaves by its exception continuation before the handler
   that binds a symbol has run. The handler of [3] mentions the symbol;
   the handler of [4] binds it and applies [3]; the body raises. Row 1
   allows the raise, and the unit passes the checks: no handler is
   entered, and it exits without the symbol bound. *)
Definition raise_past_binding_unit : expr :=
  Let_cont_non_recursive 3%positive
    (Handler [] (make_handler_metadata [])
       (Apply_cont (Apply_cont_expr 1%positive [Symbol 1%positive] None)))
    (Let_cont_non_recursive 4%positive
       (Handler [] (make_handler_metadata [])
          (Let_static [Bind_symbol 1%positive (nested_test_arm 0)]
             (Apply_cont (Apply_cont_expr 3%positive [] None))))
       (Apply_cont
          (Apply_cont_expr 2%positive [Const (Tagged_immediate 0)]
             (Some (Pop 2%positive None))))).

Example raise_past_binding_passes :
  well_formed_unit handlers_interface raise_past_binding_unit = true
  ∧ statics_in_scope [] raise_past_binding_unit = true
  ∧ static_names_apart [] raise_past_binding_unit = true
  ∧ interface_conts_not_rebound handlers_interface raise_past_binding_unit
    = true.
Proof. repeat split; vm_compute; reflexivity. Qed.

Example raise_past_binding_exits :
  run 3
    (State raise_past_binding_unit [] [] (Static_env [] []) empty_heap no_reads)
  = Exit 2%positive [Value_const (Tagged_immediate 0)] empty_heap no_reads.
Proof. vm_compute. reflexivity. Qed.

(* ======================================================================== *)
(* The check that gave a handler all that the body binds                    *)
(* ======================================================================== *)

Lemma in_remove_keys_not (bound : list continuation) (k : continuation)
    (n : list positive) (l : exit_names) :
  (k, n) ∈ remove_keys bound l -> among k bound = false.
Proof.
  unfold remove_keys. intros H. apply filter_In in H as [_ H].
  apply negb_true_iff in H. exact H.
Qed.

Lemma in_lookup_some (k : continuation) (n : list positive) (l : exit_names) :
  (k, n) ∈ l -> lookup k l <> None.
Proof.
  intros Hin Hnone. apply (lookup_none_not_in _ _ Hnone).
  apply in_map_iff. exists (k, n). split; [ reflexivity | exact Hin ].
Qed.

Lemma group_entry_none (group : list continuation) (l : exit_names) :
  group_entry group l = None ->
  ∀ k : continuation, k ∈ group -> lookup k l = None.
Proof.
  induction group as [ | k1 group IH]; cbn [group_entry]; intros H k Hin;
    [ destruct Hin | ].
  destruct (lookup k1 l) as [names1 | ] eqn:Hk1;
    [ destruct (group_entry group l); discriminate | ].
  destruct Hin as [<- | Hin]; [ exact Hk1 | exact (IH H k Hin) ].
Qed.

(* A way of the handlers of a group is a way of one of them. *)
Lemma in_handlers_exits (names_of : static_binding -> list positive)
    (handlers : list (continuation * continuation_handler))
    (k : continuation) :
  ∀ a : list positive,
    (k, a) ∈ handlers_exits names_of handlers ->
    ∃ (k0 : continuation) (h : continuation_handler) (a' : list positive),
      (k0, h) ∈ handlers ∧ (k, a') ∈ handler_exits names_of h ∧ sub a a'.
Proof.
  induction handlers as [ | [k1 h1] handlers IH]; intros a Ha;
    [ destruct Ha | ].
  rewrite handlers_exits_cons in Ha.
  apply in_meet_exits in Ha as (a1 & [Ha1 | Ha1] & Hsub).
  - exists k1, h1, a1. split; [ left; reflexivity | split; assumption ].
  - destruct (IH a1 Ha1) as (k0 & h & a' & Hin & Ha' & Hsub').
    exists k0, h, a'. split; [ right; exact Hin | ].
    split; [ exact Ha' | exact (sub_trans _ _ _ Hsub Hsub') ].
Qed.

(* A name that every way to a continuation has, each way of the meet to
   it has. *)
Lemma in_meet_exits_all (k : continuation) (name : positive)
    (l l' : exit_names) :
  (∀ a : list positive, (k, a) ∈ l -> among name a = true) ->
  (∀ a : list positive, (k, a) ∈ l' -> among name a = true) ->
  ∀ m : list positive, (k, m) ∈ meet_exits l l' -> among name m = true.
Proof.
  intros Hl Hl'. induction l as [ | [k1 names1] l IH]; intros m Hm;
    [ exact (Hl' m Hm) | ].
  assert (Hrest :
    ∀ a : list positive, (k, a) ∈ l -> among name a = true).
  { intros a Ha. apply Hl. right. exact Ha. }
  rewrite meet_exits_cons in Hm.
  revert m Hm. generalize (IH Hrest). generalize (meet_exits l l').
  intros rest Hrest_all. clear IH Hrest Hl'.
  induction rest as [ | [k' names'] rest IHrest]; cbn [add_exit];
    intros m Hm.
  - destruct Hm as [[= <- <-] | []]. apply Hl. left. reflexivity.
  - destruct (Pos.eqb_spec k1 k') as [-> | Hne].
    + destruct Hm as [[= <- <-] | Hm].
      * rewrite among_inter. apply andb_true_iff. split.
        -- apply Hl. left. reflexivity.
        -- apply Hrest_all. left. reflexivity.
      * apply Hrest_all. right. exact Hm.
    + destruct Hm as [[= <- <-] | Hm].
      * apply Hrest_all. left. reflexivity.
      * apply IHrest; [ | exact Hm ].
        intros a Ha. apply Hrest_all. right. exact Ha.
Qed.

(* A continuation that an expression has a way to is free in it. *)
Fixpoint exits_keys_free (names_of : static_binding -> list positive)
    (e : expr) {struct e} :
  ∀ (k : continuation) (names : list positive),
    (k, names) ∈ exits names_of e -> k ∈ free_continuations e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k0 [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros k names;
    cbn [exits handler_exits free_continuations handler_free_continuations].
  - apply exits_keys_free.
  - intros H. apply in_with_bound in H as (a & Ha & _).
    exact (exits_keys_free names_of body k a Ha).
  - assert (Hbody :
      ∀ a : list positive,
        (k, a) ∈ exits names_of body -> k <> k0 ->
        k ∈ free_continuations handler
            ++ without [k0] (free_continuations body)).
    { intros a Ha Hne. apply in_or_app. right. apply among_In.
      rewrite FlambdaMapProofs.among_without. apply andb_true_iff.
      split; [ apply among_In, (exits_keys_free names_of body k a Ha) | ].
      unfold among. cbn [existsb].
      rewrite (proj2 (Pos.eqb_neq k k0) Hne). reflexivity. }
    destruct (lookup k0 (exits names_of body)) as [entry | ] eqn:Hentry.
    + intros H. apply in_meet_exits in H as (a & [Ha | Ha] & _).
      * pose proof (in_remove_keys_not _ _ _ _ Ha) as Hnot.
        apply in_remove_keys in Ha. apply (Hbody a Ha).
        intros ->. unfold among in Hnot. cbn [existsb] in Hnot.
        rewrite Pos.eqb_refl in Hnot. discriminate Hnot.
      * apply in_with_bound in Ha as (a' & Ha' & _).
        apply in_or_app. left.
        exact (exits_keys_free names_of handler k a' Ha').
    + intros H. apply (Hbody names H). intros ->.
      exact (in_lookup_some _ _ _ H Hentry).
  - assert (Hbody :
      ∀ a : list positive,
        (k, a) ∈ exits names_of body ->
        among k (map fst handlers) = false ->
        k ∈ without (map fst handlers)
              (flat_map (fun '(_, h) => handler_free_continuations h)
                 handlers
               ++ free_continuations body)).
    { intros a Ha Hnot. apply among_In.
      rewrite FlambdaMapProofs.among_without, Hnot, andb_true_r.
      apply among_In, in_or_app. right.
      exact (exits_keys_free names_of body k a Ha). }
    destruct (group_entry (map fst handlers) (exits names_of body))
      as [entry | ] eqn:Hentry.
    + intros H. apply in_meet_exits in H as (a & [Ha | Ha] & _).
      * pose proof (in_remove_keys_not _ _ _ _ Ha) as Hnot.
        apply in_remove_keys in Ha. exact (Hbody a Ha Hnot).
      * apply in_with_bound in Ha as (a' & Ha' & _).
        pose proof (in_remove_keys_not _ _ _ _ Ha') as Hnot.
        apply in_remove_keys in Ha'.
        apply among_In.
        rewrite FlambdaMapProofs.among_without, Hnot, andb_true_r.
        apply among_In, in_or_app. left.
        clear Hentry Hbody Hnot. revert a' Ha'.
        induction handlers
          as [ | [k1 [params metadata handler]] handlers IH];
          intros a' Ha'; [ destruct Ha' | ].
        cbn [fold_right flat_map handler_exits
             handler_free_continuations] in Ha' |- *.
        apply in_meet_exits in Ha' as (a'' & [Ha'' | Ha''] & _).
        -- apply in_or_app. left.
           exact (exits_keys_free names_of handler k a'' Ha'').
        -- apply in_or_app. right. exact (IH a'' Ha'').
    + intros H. apply (Hbody names H).
      destruct (among k (map fst handlers)) eqn:Hamong; [ | reflexivity ].
      apply among_In in Hamong.
      exfalso.
      exact (in_lookup_some _ _ _ H (group_entry_none _ _ Hentry k Hamong)).
  - intros H.
    apply in_add_exit in H as [[-> _] | (a & [[= -> _] | []] & _)];
      [ left; reflexivity | right; left; reflexivity ].
  - intros [[= <- _] | []]. unfold apply_cont_continuations. left.
    reflexivity.
  - intros H. change (fold_right
        (fun arm l => add_exit (ac_continuation (snd arm)) [] l) [] arms)
      with (arms_exits arms) in H.
    revert names H.
    induction arms as [ | arm arms IH]; intros names H; [ destruct H | ].
    cbn [arms_exits fold_right] in H. fold (arms_exits arms) in H.
    cbn [flat_map]. apply in_or_app.
    apply in_add_exit in H as [[-> _] | (a & Ha & _)].
    + left. unfold apply_cont_continuations. left. reflexivity.
    + right. exact (IH a Ha).
  - intros [].
Qed.

Section Whole_bodies.

Variable unit_exn : continuation.
Variable names_of : static_binding -> list positive.

(* Under Simplify's rule of place, where no [Let_cont] binds the name
   of the exception continuation of the unit, every way out of an
   expression, but for those by that name, has all that the expression
   binds. *)
Fixpoint at_toplevel_exits_whole (e : expr) {struct e} :
  ∀ toplevel : bool,
    statics_at_unit_toplevel unit_exn toplevel e = true ->
    among unit_exn (bound_continuations e) = false ->
    ∀ (k : continuation) (names : list positive),
      (k, names) ∈ exits names_of e -> k <> unit_exn ->
      sub (static_names names_of e) names.
Proof.
  destruct e
    as [ x kind d body | bindings body | k0 [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros toplevel;
    cbn [statics_at_unit_toplevel handler_statics_at_unit_toplevel
         bound_continuations handler_bound_continuations
         exits handler_exits static_names handler_static_names];
    try (intros _ _ k names _ _; apply sub_nil).
  - apply at_toplevel_exits_whole.
  - rewrite !andb_true_iff. intros [[_ _] Pbody] U k names H Hk.
    apply among_false_app in U as [_ Ubody].
    apply in_with_bound in H as (a & Ha & ->).
    apply sub_app_mono.
    exact (at_toplevel_exits_whole body _ Pbody Ubody k a Ha Hk).
  - rewrite andb_true_iff. intros [Ph Pbody] U k names H Hk.
    apply among_false_cons in U as [Hk0 U].
    apply among_false_app in U as [Uh Ubody].
    pose proof (at_toplevel_exits_whole body _ Pbody Ubody) as IHbody.
    pose proof (at_toplevel_exits_whole handler _ Ph Uh) as IHh.
    (* The ways of the body, where the handler is at toplevel, are to
       the continuation or leave the unit. *)
    assert (Hways :
      static_names names_of handler <> [] ->
      ∀ (k' : continuation) (a : list positive),
        (k', a) ∈ exits names_of body -> k' <> unit_exn -> k' = k0).
    { intros Hsome k' a Ha Hk'.
      destruct (toplevel
                && handler_at_unit_toplevel unit_exn k0
                     (Handler params metadata handler) body) eqn:Hat.
      - apply andb_true_iff in Hat as [_ Hat].
        unfold handler_at_unit_toplevel in Hat.
        apply andb_true_iff in Hat as [_ Hat].
        pose proof
          (proj1 (all_among_spec _ _) Hat k'
             (proj2 (among_In _ _) (exits_keys_free names_of body k' a Ha)))
          as Hk2.
        unfold among in Hk2. cbn [existsb] in Hk2.
        rewrite (proj2 (Pos.eqb_neq k' unit_exn) Hk'), !orb_false_r in Hk2.
        apply Pos.eqb_eq in Hk2. exact Hk2.
      - exfalso. apply Hsome.
        exact (outside_nowhere names_of handler
                 (at_unit_toplevel_outside unit_exn handler false Ph)). }
    assert (Hk0' : k0 <> unit_exn) by (intros ->; apply Hk0; reflexivity).
    apply sub_in. intros name Hname. apply among_In.
    apply in_app_or in Hname.
    destruct (lookup k0 (exits names_of body)) as [entry | ] eqn:Hentry.
    + refine (in_meet_exits_all k name _ _ _ _ names H).
      * intros a Ha. pose proof (in_remove_keys_not _ _ _ _ Ha) as Hnot.
        apply in_remove_keys in Ha.
        destruct Hname as [Hname | Hname].
        -- exfalso.
           assert (Hsome : static_names names_of handler <> []).
           { intros Hnil. rewrite Hnil in Hname. destruct Hname. }
           rewrite (Hways Hsome k a Ha Hk) in Hnot.
           unfold among in Hnot. cbn [existsb] in Hnot.
           rewrite Pos.eqb_refl in Hnot. discriminate Hnot.
        -- exact (IHbody k a Ha Hk name (proj2 (among_In _ _) Hname)).
      * intros a Ha. apply in_with_bound in Ha as (a' & Ha' & ->).
        rewrite among_app. apply orb_true_iff.
        destruct Hname as [Hname | Hname].
        -- right.
           exact (IHh k a' Ha' Hk name (proj2 (among_In _ _) Hname)).
        -- left.
           exact (IHbody k0 entry (lookup_in _ _ _ Hentry) Hk0' name
                    (proj2 (among_In _ _) Hname)).
    + destruct Hname as [Hname | Hname].
      * exfalso.
        assert (Hsome : static_names names_of handler <> []).
        { intros Hnil. rewrite Hnil in Hname. destruct Hname. }
        pose proof (Hways Hsome k names H Hk) as ->.
        exact (in_lookup_some _ _ _ H Hentry).
      * exact (IHbody k names H Hk name (proj2 (among_In _ _) Hname)).
  - rewrite andb_true_iff. intros [Phandlers Pbody] U k names H Hk.
    apply among_false_app in U as [Unames U].
    apply among_false_app in U as [_ Ubody].
    pose proof (at_toplevel_exits_whole body _ Pbody Ubody) as IHbody.
    assert (Hnone :
      flat_map (fun '(_, h) => handler_static_names names_of h) handlers
      = []).
    { revert Phandlers. clear.
      induction handlers as [ | [k1 [params metadata handler]] handlers IH];
        [ reflexivity | ].
      cbn [forallb handler_statics_at_unit_toplevel flat_map
           handler_static_names].
      intros Phandlers. apply andb_true_iff in Phandlers as [Ph Prest].
      rewrite
        (outside_nowhere names_of handler
           (at_unit_toplevel_outside unit_exn handler false Ph)),
        (IH Prest).
      reflexivity. }
    rewrite Hnone, app_nil_l.
    destruct (group_entry (map fst handlers) (exits names_of body))
      as [entry | ] eqn:Hentry; [ | exact (IHbody k names H Hk) ].
    apply sub_in. intros name Hname. apply among_In.
    apply among_In in Hname.
    refine (in_meet_exits_all k name _ _ _ _ names H).
    + intros a Ha. apply in_remove_keys in Ha.
      exact (IHbody k a Ha Hk name Hname).
    + intros a Ha. apply in_with_bound in Ha as (a' & _ & ->).
      rewrite among_app. apply orb_true_iff. left.
      apply (group_entry_all _ _ name entry Hentry).
      intros k1 a1 Hin1 Hk1.
      refine (IHbody k1 a1 (lookup_in _ _ _ Hk1) _ name Hname).
      intros ->. apply among_In in Hin1. rewrite Hin1 in Unames.
      discriminate Unames.
Qed.

(* So the handler of a [Let_cont] has, by the ways to it, all that the
   body binds. *)
Lemma at_toplevel_handler_scope_whole (k : continuation) (body : expr)
    (toplevel : bool) :
  statics_at_unit_toplevel unit_exn toplevel body = true ->
  among unit_exn (bound_continuations body) = false ->
  k <> unit_exn ->
  sub (static_names names_of body) (handler_scope names_of k body).
Proof.
  intros P U Hk. unfold handler_scope.
  destruct (lookup k (exits names_of body)) as [names | ] eqn:Hlookup;
    [ | apply sub_refl ].
  exact (at_toplevel_exits_whole body toplevel P U k names
           (lookup_in _ _ _ Hlookup) Hk).
Qed.

Lemma at_toplevel_recursive_scope_whole (group : list continuation)
    (body : expr) (toplevel : bool) :
  statics_at_unit_toplevel unit_exn toplevel body = true ->
  among unit_exn (bound_continuations body) = false ->
  among unit_exn group = false ->
  sub (static_names names_of body) (recursive_scope names_of group body).
Proof.
  intros P U Hgroup. unfold recursive_scope.
  destruct (group_entry group (exits names_of body)) as [entry | ] eqn:Hentry;
    [ | apply sub_refl ].
  intros name Hname. apply (group_entry_all _ _ name entry Hentry).
  intros k a Hin Hk.
  refine (at_toplevel_exits_whole body toplevel P U k a
            (lookup_in _ _ _ Hk) _ name Hname).
  intros ->. apply among_In in Hin. rewrite Hin in Hgroup.
  discriminate Hgroup.
Qed.

End Whole_bodies.

Lemma sub_app_mono_l (names names' rest : list positive) :
  sub names names' -> sub (names ++ rest) (names' ++ rest).
Proof.
  intros H. apply sub_app; [ | apply sub_app_r ].
  exact (sub_trans _ _ _ H (sub_app_l names' rest)).
Qed.

(* The check that gave a handler all that the body binds, passed under
   Simplify's rule of place by an expression where no [Let_cont] binds
   the name of the exception continuation of the unit, gives the check
   by the ways to the handler. *)
Fixpoint expr_whole_bodies_in_scope (unit_exn : continuation) (e : expr)
    {struct e} :
  ∀ (toplevel : bool) (syms : list symbol) (ids : list code_id),
    statics_at_unit_toplevel unit_exn toplevel e = true ->
    among unit_exn (bound_continuations e) = false ->
    expr_statics_in_scope_of_whole_bodies syms ids e = true ->
    expr_statics_in_scope syms ids e = true.
Proof.
  destruct e
    as [ x kind d body | bindings body | k0 [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros toplevel syms ids;
    cbn [statics_at_unit_toplevel handler_statics_at_unit_toplevel
         bound_continuations handler_bound_continuations
         expr_statics_in_scope_of_whole_bodies
         handler_statics_in_scope_of_whole_bodies
         expr_statics_in_scope handler_statics_in_scope];
    try (intros _ _ S; exact S).
  - intros P U. rewrite !andb_true_iff. intros [Sd Sbody].
    split; [ exact Sd | ].
    exact (expr_whole_bodies_in_scope unit_exn body _ _ _ P U Sbody).
  - rewrite !andb_true_iff. intros [[_ Pbindings] Pbody] U.
    apply among_false_app in U as [Ubindings Ubody].
    intros [[Sconsts Scode] Sbody].
    split; [ split; [ exact Sconsts | ] | ].
    + revert Pbindings Ubindings Scode.
      generalize (flat_map symbols_of_binding bindings ++ syms),
        (flat_map declared_code_ids_of_binding bindings ++ ids).
      intros all ids'. clear Sconsts Sbody Pbody Ubody.
      induction bindings
        as [ | [id [cparams my_closure ret exn cmetadata cbody] | id
               | s const] bindings IH];
        cbn [forallb flat_map binding_statics_at_unit_toplevel
             code_statics_at_unit_toplevel binding_bound_continuations
             code_bound_continuations
             binding_statics_in_scope_of_whole_bodies
             code_statics_in_scope_of_whole_bodies
             binding_statics_in_scope code_statics_in_scope];
        [ reflexivity | | exact IH | exact IH ].
      rewrite !andb_true_iff. intros [Pcode Prest] Ucode [Sc Srest].
      apply among_false_app in Ucode as [Ucode Urest].
      apply among_false_cons in Ucode as [_ Ucode].
      apply among_false_cons in Ucode as [_ Ucode].
      split; [ | exact (IH Prest Urest Srest) ].
      exact (expr_whole_bodies_in_scope unit_exn cbody _ _ _ Pcode Ucode
               Sc).
    + exact (expr_whole_bodies_in_scope unit_exn body _ _ _ Pbody Ubody
               Sbody).
  - rewrite !andb_true_iff. intros [Ph Pbody] U [Sh Sbody].
    apply among_false_cons in U as [Hk0 U].
    apply among_false_app in U as [Uh Ubody].
    assert (Hk0' : k0 <> unit_exn) by (intros ->; apply Hk0; reflexivity).
    split;
      [ | exact (expr_whole_bodies_in_scope unit_exn body _ _ _ Pbody Ubody
                   Sbody) ].
    refine (expr_scope_sub handler _ _ _ _ _ _
              (expr_whole_bodies_in_scope unit_exn handler _ _ _ Ph Uh Sh));
      apply sub_app_mono_l;
      exact (at_toplevel_handler_scope_whole unit_exn _ k0 body toplevel
               Pbody Ubody Hk0').
  - rewrite !andb_true_iff. intros [Phandlers Pbody] U [Shandlers Sbody].
    apply among_false_app in U as [Unames U].
    apply among_false_app in U as [Uhandlers Ubody].
    split;
      [ | exact (expr_whole_bodies_in_scope unit_exn body _ _ _ Pbody Ubody
                   Sbody) ].
    pose proof
      (at_toplevel_recursive_scope_whole unit_exn symbols_of_binding
         (map fst handlers) body toplevel Pbody Ubody Unames) as Hsyms.
    pose proof
      (at_toplevel_recursive_scope_whole unit_exn declared_code_ids_of_binding
         (map fst handlers) body toplevel Pbody Ubody Unames) as Hids.
    revert Hsyms Hids Phandlers Uhandlers Shandlers.
    generalize (recursive_scope symbols_of_binding (map fst handlers) body),
      (recursive_scope declared_code_ids_of_binding (map fst handlers) body),
      (static_names symbols_of_binding body),
      (static_names declared_code_ids_of_binding body).
    intros group_syms group_ids body_syms body_ids Hsyms Hids.
    clear Unames Pbody Ubody Sbody.
    induction handlers as [ | [k1 [params metadata handler]] handlers IH];
      cbn [forallb flat_map handler_statics_at_unit_toplevel
           handler_bound_continuations
           handler_statics_in_scope_of_whole_bodies
           handler_statics_in_scope];
      [ reflexivity | ].
    rewrite !andb_true_iff. intros [Ph Prest] Uh [Sh Srest].
    apply among_false_app in Uh as [Uh Urest].
    split; [ | exact (IH Prest Urest Srest) ].
    refine (expr_scope_sub handler _ _ _ _ _ _
              (expr_whole_bodies_in_scope unit_exn handler _ _ _ Ph Uh Sh));
      apply sub_app_mono_l; assumption.
Qed.

Theorem whole_bodies_in_scope : whole_bodies_in_scope_statement.
Proof.
  intros imports u e P U S. unfold interface_conts_not_rebound in U.
  apply andb_true_iff in U as [_ U]. apply negb_true_iff in U.
  exact (expr_whole_bodies_in_scope (unit_exn u) e true imports [] P U S).
Qed.

(* ======================================================================== *)
(* Two units that the check refuses                                         *)
(* ======================================================================== *)

(* The state of the run of the first that reads the symbol. *)
Definition round_state : state :=
  match run 5 (State round_unit [] [] (Static_env [] []) empty_heap no_reads)
    with
  | Step s => s
  | Exit _ _ _ _ | Undefined => State Invalid [] [] (Static_env [] [])
    empty_heap no_reads
  end.

Theorem round_unit_refused : round_unit_refused_statement.
Proof.
  refine (conj _ (conj _ (conj _ (conj _ (conj _ _)))));
    [ vm_compute; reflexivity .. | ].
  exists 5%nat, round_state. split; vm_compute; reflexivity.
Qed.

(* The state of the run of the second that reads the symbol. *)
Definition rebound_state : state :=
  match run 5 (State rebound_unit [] [] (Static_env [] []) empty_heap no_reads)
    with
  | Step s => s
  | Exit _ _ _ _ | Undefined => State Invalid [] [] (Static_env [] [])
    empty_heap no_reads
  end.

Theorem rebound_unit_refused : rebound_unit_refused_statement.
Proof.
  refine (conj _ (conj _ (conj _ (conj _ _))));
    [ vm_compute; reflexivity .. | ].
  exists 5%nat, rebound_state. split; vm_compute; reflexivity.
Qed.
