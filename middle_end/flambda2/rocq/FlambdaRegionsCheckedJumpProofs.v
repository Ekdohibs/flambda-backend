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
   entering a handler. The state that the handler runs in is as the state
   of the check that the handler was checked in says. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List Bool Lia PeanoNat.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaBindersProofs FlambdaValueSpec
  FlambdaHeapProofs FlambdaSemanticsSpec FlambdaClosuresProofs
  FlambdaRegionsSpec FlambdaRegionsProofs FlambdaRegionsReadsProofs
  FlambdaRegionsInvariantSpec FlambdaRegionsInvariantProofs
  FlambdaRegionsErasureSpec FlambdaRegionsCallsSpec FlambdaRegionsCheckSpec
  FlambdaRegionsCheckedSpec.
From OxCamlFlambda2 Require Import FlambdaRegionsCheckedBasicsProofs
  FlambdaRegionsCheckedHeadProofs FlambdaRegionsCheckedStableProofs
  FlambdaRegionsCheckedMarksProofs FlambdaRegionsCheckedOutlivesProofs
  FlambdaRegionsCheckedRankProofs FlambdaRegionsCheckedParamsProofs.
Import ListNotations.

(* ======================================================================== *)
(* Lists                                                                    *)
(* ======================================================================== *)

Lemma lookup_map_entry {A B : Type} (f : positive -> A -> B) (k : positive)
    (l : list (positive * A)) :
  lookup k (map (fun '(k0, a) => (k0, f k0 a)) l)
  = option_map (f k) (lookup k l).
Proof.
  induction l as [ | [k0 a] l IH]; [ reflexivity | ].
  cbn [map]. rewrite !lookup_cons.
  destruct (Pos.eqb k k0) eqn:Heq; [ | exact IH ].
  apply Pos.eqb_eq in Heq. subst k0. reflexivity.
Qed.

Lemma Forall2_map_left {A B C : Type} (R : B -> C -> Prop) (f : A -> B)
    (l : list A) :
  ∀ l' : list C,
    Forall2 R (map f l) l' -> Forall2 (fun a c => R (f a) c) l l'.
Proof.
  induction l as [ | a l IH]; intros l' H; inversion H; subst; constructor;
    [ assumption | apply IH; assumption ].
Qed.

Lemma Forall2_impl_in {A B : Type} (R R' : A -> B -> Prop) (l : list A) :
  ∀ l' : list B,
    (∀ (a : A) (b : B), In a l -> R a b -> R' a b) ->
    Forall2 R l l' -> Forall2 R' l l'.
Proof.
  induction l as [ | a l IH]; intros l' Himpl H; inversion H; subst;
    constructor.
  - apply Himpl; [ left; reflexivity | assumption ].
  - apply IH; [ | assumption ]. intros a0 b0 Hin. apply Himpl. right.
    exact Hin.
Qed.

(* ======================================================================== *)
(* Marks                                                                    *)
(* ======================================================================== *)

(* Regions bound to marks that decrease are distinct. *)
Lemma marked_NoDup (vars : env) (l : list variable) :
  ∀ ms : list mark,
    Forall2 (marked vars) l ms -> marks_decrease ms = true -> NoDup l.
Proof.
  induction l as [ | x l IH]; intros ms Hmarked Hdecrease; [ constructor | ].
  inversion Hmarked as [ | x0 m l0 ms0 Hx Hrest]; subst.
  apply marks_decrease_cons in Hdecrease as [Hlt Htail].
  constructor; [ | exact (IH ms0 Hrest Htail) ].
  intros Hin. destruct (Forall2_in_left _ _ _ x Hrest Hin) as (m' & Hin' & Hx').
  rewrite (marked_fun vars x m' m Hx' Hx) in Hin'.
  pose proof (Hlt m Hin'). lia.
Qed.

Lemma region_place_marked (vars : env) (opens : list variable) :
  ∀ (ms : list mark) (above : option symbol) (r : variable) (m : mark)
    (stop : option symbol),
    Forall2 (marked vars) opens ms ->
    region_place opens ms above r = Some (m, stop) ->
    marked vars r m ∧ In r opens.
Proof.
  induction opens as [ | x opens IH]; intros ms above r m stop Hmarked;
    [ discriminate | ].
  inversion Hmarked as [ | x0 m0 opens0 ms0 Hx Hrest]; subst.
  cbn [region_place]. destruct (Pos.eqb x r) eqn:Heq.
  - apply Pos.eqb_eq in Heq. subst x. intros [= <- _].
    split; [ exact Hx | left; reflexivity ].
  - intros H. destruct (IH ms0 (Some m0) r m stop Hrest H) as [Hr Hin].
    split; [ exact Hr | right; exact Hin ].
Qed.

(* The place of a region under a map of the names that keeps the open
   regions apart. *)
Lemma region_place_map (f : variable -> variable) (opens : list variable) :
  ∀ (ms : list mark) (above : option symbol) (r : variable),
    NoDup (map f opens) ->
    In r opens ->
    region_place (map f opens) ms above (f r) = region_place opens ms above r.
Proof.
  induction opens as [ | x opens IH]; intros ms above r Hnodup Hin;
    [ contradiction | ].
  cbn [map] in *. inversion Hnodup as [ | y rest Hnotin Hrest]; subst.
  cbn [region_place]. destruct ms as [ | m ms]; [ reflexivity | ].
  destruct (Pos.eqb x r) eqn:Heq.
  - apply Pos.eqb_eq in Heq. subst x. rewrite Pos.eqb_refl. reflexivity.
  - destruct Hin as [-> | Hin]; [ rewrite Pos.eqb_refl in Heq; discriminate | ].
    assert (Hne : Pos.eqb (f x) (f r) = false).
    { destruct (Pos.eqb (f x) (f r)) eqn:Hf; [ | reflexivity ].
      apply Pos.eqb_eq in Hf. exfalso. apply Hnotin. rewrite Hf.
      exact (in_map f opens r Hin). }
    rewrite Hne. exact (IH ms (Some m) r Hrest Hin).
Qed.

(* A value as a lifetime says in the names of the jump is so in the
   names of the handler. *)
Lemma holds_at_jump (rs : region_state) (vars : env)
    (given : list (variable * variable)) (opens : list variable)
    (ms : list mark) (w : lifetime) (v : value) :
  Forall2 (marked vars) opens ms ->
  NoDup (map (region_at_jump given) opens) ->
  lifetime_holds rs (map (region_at_jump given) opens) ms None
    (lifetime_at_jump given w) v ->
  lifetime_holds rs opens ms None w v.
Proof.
  intros Hmarked Hnodup. destruct w as [ | | r | ]; try (intros H; exact H).
  cbn [lifetime_at_jump lifetime_holds].
  destruct (region_place opens ms None r) as [[m stop] | ] eqn:Hplace;
    [ | intros _; exact I ].
  destruct (region_place_marked vars opens ms None r m stop Hmarked Hplace)
    as [_ Hin].
  rewrite (region_place_map (region_at_jump given) opens ms None r Hnodup Hin),
    Hplace.
  intros H. exact H.
Qed.

(* ======================================================================== *)
(* The state that runs, by its parts                                        *)
(* ======================================================================== *)

Definition running_at (rs : region_state) (vars : env) (cs : check_state)
    (ms : list mark) : Prop :=
  open_marks vars (cs_open cs) = Some ms
  ∧ marks_placed (rs_frames rs) ms = true
  ∧ check_state_scoped cs = true
  ∧ ∀ (x : variable) (l : lifetime) (v : value),
      lookup x (cs_env cs) = Some l ->
      lookup x vars = Some v ->
      lifetime_holds rs (cs_open cs) ms None l v.

(* What a variable that a closure holds was is what it is in the
   handler: the bound that it was within is at most the bound that the
   handler asks. *)
Lemma captured_holds (rs : region_state) (vars_c vars' : env)
    (opens_c opens' : list variable) (ms_c ms' : list mark) (next : symbol)
    (l : lifetime) (v : value) :
  Forall2 (marked vars_c) opens_c ms_c ->
  marks_decrease ms_c = true ->
  (∀ m : mark, In m ms_c -> (m < next)%positive) ->
  Forall2 (marked vars') opens' ms' ->
  marks_decrease ms' = true ->
  (∀ m : mark, In m ms' -> In m ms_c ∨ (next <= m)%positive) ->
  (∀ r : variable,
     l = Of_region r -> In r opens' ->
     In r opens_c ∧ lookup r vars' = lookup r vars_c) ->
  lifetime_holds rs opens_c ms_c (Some next) l v ->
  lifetime_holds rs opens' ms' None l v.
Proof.
  intros Hmarked_c Hdecrease_c Hbelow Hmarked' Hdecrease' Hbefore Hregion.
  destruct l as [ | | r | ]; cbn [lifetime_holds].
  - intros H. exact H.
  - intros H. apply (within_later rs _ _ v) with (2 := H).
    intros b' Hb'.
    destruct (caller_bound_least ms_c next Hbelow Hdecrease_c)
      as (b & -> & Hle & Hleast).
    exists b. split; [ reflexivity | ].
    unfold caller_bound in Hb'.
    destruct (lowest ms') as [b0 | ] eqn:Hlowest; [ | discriminate ].
    injection Hb' as ->.
    destruct (Hbefore b' (proj1 (lowest_least ms' b' Hlowest Hdecrease')))
      as [Hin | Hge]; [ exact (Hleast b' Hin) | lia ].
  - destruct (region_place opens' ms' None r) as [[m stop'] | ] eqn:Hplace';
      [ | intros _; exact I ].
    destruct (region_place_least vars' opens' ms' r m stop' Hmarked'
                Hdecrease' Hplace')
      as (Hr' & Hm' & Hstop').
    destruct (region_place_marked vars' opens' ms' None r m stop' Hmarked'
                Hplace')
      as [_ Hin'].
    destruct (Hregion r eq_refl Hin') as [Hin_c Hsame].
    destruct (region_place_open vars_c opens_c ms_c (Some next) r
                (proj2 (open_marks_marked vars_c opens_c ms_c) Hmarked_c)
                (proj2 (among_In r opens_c) Hin_c))
      as (m_c & stop_c & Hplace_c & _).
    rewrite Hplace_c.
    destruct (region_place_marked vars_c opens_c ms_c (Some next) r m_c stop_c
                Hmarked_c Hplace_c)
      as [Hr_c _].
    assert (Hm : m_c = m).
    { unfold marked in Hr', Hr_c. rewrite Hsame, Hr_c in Hr'. congruence. }
    subst m_c.
    destruct (region_place_bound opens_c ms_c next r m stop_c Hplace_c Hbelow
                Hdecrease_c)
      as (b & -> & Hle & _ & Hleast).
    intros H Hstack. apply (within_later rs _ _ v) with (2 := H Hstack).
    intros b' ->. exists b. split; [ reflexivity | ].
    destruct Hstop' as (Hb' & Hlt & _).
    destruct (Hbefore b' Hb') as [Hin | Hge];
      [ exact (Hleast b' Hin Hlt) | lia ].
  - intros _. exact I.
Qed.

Lemma bind_entry_params_scoped (he_open regions : list variable)
    (params : list variable) :
  ∀ (kinds : list flambda_kind) (wanted : list lifetime)
    (env : list (variable * lifetime)),
    (∀ r : variable, among r he_open = true -> among r regions = true) ->
    forallb
      (fun '(_, l) =>
         match l with
         | Of_region r => among r regions
         | Forever | Of_caller | Gone => true
         end)
      env
    = true ->
    forallb
      (fun '(_, l) =>
         match l with
         | Of_region r => among r regions
         | Forever | Of_caller | Gone => true
         end)
      (bind_entry_params he_open params kinds wanted env)
    = true.
Proof.
  induction params as [ | p params IH]; intros kinds wanted env Hopen Henv;
    [ exact Henv | ].
  cbn [bind_entry_params].
  destruct kinds as [ | kind kinds]; [ exact Henv | ].
  destruct wanted as [ | w wanted]; [ exact Henv | ].
  cbn [forallb]. rewrite (IH kinds wanted env Hopen Henv), andb_true_r.
  destruct (kind_eqb kind Region); [ reflexivity | ].
  destruct w as [ | | r | ]; try reflexivity.
  destruct (among r he_open) eqn:Hr; [ exact (Hopen r Hr) | reflexivity ].
Qed.

Lemma lookup_entries (cert : certificate) (k : continuation)
    (recursive : list (continuation * continuation_handler)) :
  lookup k (map (fun '(k, h) => (k, cont_entry_of cert k h)) recursive)
  = option_map (cont_entry_of cert k) (lookup k recursive).
Proof.
  exact (lookup_map_entry (fun k h => cont_entry_of cert k h) k recursive).
Qed.

Lemma lookup_rec_closures (k : continuation)
    (recursive : list (continuation * continuation_handler)) (vars : env)
    (conts : cont_env) :
  lookup k (rec_closures recursive vars conts)
  = option_map (fun h => Closure h vars conts recursive) (lookup k recursive).
Proof.
  exact
    (lookup_map_entry (fun _ h => Closure h vars conts recursive) k recursive).
Qed.

(* ======================================================================== *)
(* Entering a handler                                                       *)
(* ======================================================================== *)

Section Enter.

Variable cert : certificate.

Variable codes : list (code_id * code).

Variable rs : region_state.

(* Who jumps: a state of the check and a user. The open regions of
   the state are bound, for the user, to its marks, which are on the
   stack; its counter is not after the machine's. *)
Record frame_ok (cs : check_state) (u : user) : Prop := Frame_ok {
  frame_marks : Forall2 (marked (u_vars u)) (cs_open cs) (u_marks u);
  frame_placed : marks_placed (rs_frames rs) (u_marks u) = true;
  frame_limit : (u_limit u <= rs_next rs)%positive
}.

(* An argument of a jump and its value: the value is as the lifetime
   says, and is that of the simple where the simple is a variable. *)
Definition arg_holds (cs : check_state) (u : user)
    (arg : option simple * lifetime) (v : value) : Prop :=
  lifetime_holds rs (cs_open cs) (u_marks u) None (snd arg) v
  ∧ ∀ r : variable, fst arg = Some (Var r) -> lookup r (u_vars u) = Some v.

Lemma ok_handler_inv (u : user) (params : list variable)
    (kinds : list flambda_kind) (entry : handler_entry) (c : cont_value) :
  cont_ok cert codes rs u (To_handler params kinds entry) c ->
  ∃ (metadata : handler_metadata) (handler : expr) (vars : env)
    (conts : cont_env)
    (recursive : list (continuation * continuation_handler))
    (cs : check_state) (base : list (continuation * cont_entry))
    (ms : list mark) (next : symbol),
    c = Closure (Handler params metadata handler) vars conts recursive
    ∧ kinds = hm_params_kinds metadata
    ∧ made_at cert codes rs u vars recursive cs base ms next
    ∧ entry_fits cs params kinds entry = true
    ∧ check_expr cert codes (handler_state cs params kinds entry) handler
      = true
    ∧ incl (code_of_expr handler) codes
    ∧ (∀ (k : continuation) (ce : cont_entry),
         lookup k base = Some ce ->
         cont_ok cert codes rs (User vars (cs_regions cs) ms next) ce
           (resolve k conts))
    ∧ (∀ k : continuation, lookup k base = None -> lookup k conts = None).
Proof.
  intros H.
  inversion H
    as [u0 params0 metadata handler entry0 vars conts recursive cs base ms
          next Hmade Hfits Hcheck Hincl Hconts Hnone | | | ];
    subst.
  exists metadata, handler, vars, conts, recursive, cs, base, ms, next.
  exact
    (conj eq_refl
       (conj eq_refl
          (conj Hmade
             (conj Hfits (conj Hcheck (conj Hincl (conj Hconts Hnone))))))).
Qed.

Lemma enter_handler (cs_F : check_state) (u : user) (params : list variable)
    (metadata : handler_metadata) (handler : expr) (entry : handler_entry)
    (vars_c : env) (conts_c : cont_env)
    (recursive : list (continuation * continuation_handler))
    (args : list (option simple * lifetime)) (vs : list value) (vars' : env)
    (given : list (variable * variable)) :
  frame_ok cs_F u ->
  cont_ok cert codes rs u (To_handler params (hm_params_kinds metadata) entry)
    (Closure (Handler params metadata handler) vars_c conts_c recursive) ->
  region_args cs_F params (hm_params_kinds metadata) args = Some given ->
  in_order (map (region_at_jump given) (he_open entry)) (cs_open cs_F) = true ->
  value_args cs_F given (map (region_at_jump given) (he_open entry))
    (hm_params_kinds metadata) args (he_params entry)
  = true ->
  Forall2 (arg_holds cs_F u) args vs ->
  bind_params params vs vars_c = Some vars' ->
  ∃ (cs' : check_state) (ms' : list mark),
    check_expr cert codes cs' handler = true
    ∧ incl (code_of_expr handler) codes
    ∧ running_at rs vars' cs' ms'
    ∧ conts_checked cert codes rs
        (User vars' (cs_regions cs') ms' (rs_next rs)) (cs_conts cs')
        (rec_closures recursive vars_c conts_c ++ conts_c).
Proof.
  intros [Hmarks Hplaced Hlimit] Hok Hregion Horder Hvalues Hargs Hbind.
  destruct (ok_handler_inv _ _ _ _ _ Hok)
    as (metadata0 & handler0 & vars0 & conts0 & recursive0 & cs & base & ms
        & next & Hc & _ & Hmade & Hfits & Hcheck & Hincl & Hconts & Hnone).
  injection Hc as Hc1 Hc2 Hc3 Hc4 Hc5. subst metadata0 handler0 vars0 conts0
    recursive0.
  destruct Hmade
    as (Hconts_eq & Hrec_check & Hrec_incl & Hscoped & Hopen_c & Hdecrease_c
        & Hbelow_c & Hnext & Hheld & Hbefore & Hregions).
  pose proof (proj1 (open_marks_marked _ _ _) Hopen_c) as Hmarked_c.
  apply Pos.leb_le in Hnext.
  unfold entry_fits in Hfits.
  apply andb_true_iff in Hfits as [Hfits Hentry_open].
  apply andb_true_iff in Hfits as [Hfits Hopen_distinct].
  apply andb_true_iff in Hfits as [Hfits Hnot_region].
  apply andb_true_iff in Hfits as [Hfits Hdistinct].
  apply andb_true_iff in Hfits as [Hlen_kinds Hlen_wanted].
  apply Nat.eqb_eq in Hlen_kinds, Hlen_wanted.
  pose proof Hscoped as Hscoped'.
  apply andb_true_iff in Hscoped' as [Hopen_regions Henv_scoped].
  destruct (region_args_spec cs_F params _ args given Hregion)
    as (_ & Hlen_args & Hkeys & Hregion_nth).
  destruct (bind_params_nth params vs vars_c vars' Hdistinct Hbind)
    as [Hlen_vs Hvs_nth].
  set (named := map (region_at_jump given) (he_open entry)) in *.
  destruct (in_order_marks (u_vars u) (cs_open cs_F) named (u_marks u) Horder
              Hmarks)
    as (ms' & Hnamed & Hsel).
  assert (Hdecrease_F : marks_decrease (u_marks u) = true).
  { unfold marks_placed in Hplaced. apply andb_true_iff in Hplaced as [_ H].
    exact H. }
  pose proof (selection_decrease ms' (u_marks u) Hsel Hdecrease_F)
    as Hdecrease'.
  pose proof (marked_NoDup (u_vars u) named ms' Hnamed Hdecrease') as Hnodup.
  assert (Hparam_not_region :
    ∀ x : variable,
      among x params = true -> among x (cs_regions cs) = false).
  { intros x Hx. apply among_In in Hx.
    pose proof (forallb_in _ _ x Hnot_region Hx) as H.
    unfold not_a_region in H. apply negb_true_iff in H. exact H. }
  assert (Hregion_not_param :
    ∀ r : variable,
      among r (cs_regions cs) = true -> among r params = false).
  { intros r Hr. destruct (among r params) eqn:H; [ | reflexivity ].
    rewrite (Hparam_not_region r H) in Hr. discriminate. }
  assert (Hgiven_nodup : NoDup (map fst given)).
  { rewrite Hkeys. apply region_params_of_NoDup. apply distinct_NoDup.
    exact Hdistinct. }
  assert (Hmark_of :
    ∀ (r : variable) (m : mark),
      In r (he_open entry) ->
      marked (u_vars u) (region_at_jump given r) m ->
      marked vars' r m).
  { intros r m Hin Hm.
    pose proof (forallb_in _ _ r Hentry_open Hin) as Hr.
    apply orb_true_iff in Hr as [Hr | Hr].
    - apply among_In in Hr.
      destruct (region_params_of_in params _ r Hr)
        as (i & kind & Hi & Hkind_i & Hkind).
      destruct (nth_error_same_length params args (eq_sym Hlen_args) i r Hi)
        as ([s l] & Harg).
      destruct (Hregion_nth i r kind s l Hi Hkind_i Harg Hkind)
        as (r_F & -> & Hopen_F & Hin_given).
      destruct (nth_error_same_length params vs (eq_sym Hlen_vs) i r Hi)
        as (v & Hv).
      pose proof (Forall2_nth_error _ _ _ Hargs i _ _ Harg Hv) as [_ Hvalue].
      specialize (Hvalue r_F eq_refl).
      unfold region_at_jump in Hm.
      rewrite (lookup_NoDup_in given r r_F Hgiven_nodup Hin_given) in Hm.
      unfold marked in *. rewrite (Hvs_nth i r v Hi Hv), <- Hvalue. exact Hm.
    - pose proof (all_among_among _ _ r Hopen_regions Hr) as Hregion_r.
      assert (Hnot_given : lookup r given = None).
      { destruct (lookup r given) as [r' | ] eqn:H; [ | reflexivity ].
        apply lookup_in in H. apply (in_map fst) in H.
        change (In r (map fst given)) in H. rewrite Hkeys in H.
        apply region_params_of_among in H.
        apply among_In in H. rewrite (Hregion_not_param r Hregion_r) in H.
        discriminate. }
      unfold region_at_jump in Hm. rewrite Hnot_given in Hm. unfold marked in *.
      rewrite (bind_params_lookup_other r params vs vars_c vars'
                 (Hregion_not_param r Hregion_r) Hbind).
      rewrite <- (proj1 (Hregions r Hregion_r)). exact Hm. }
  assert (Hmarked' : Forall2 (marked vars') (he_open entry) ms').
  { apply (Forall2_impl_in
             (fun r m => marked (u_vars u) (region_at_jump given r) m));
      [ exact Hmark_of | ].
    apply Forall2_map_left. exact Hnamed. }
  assert (Hbefore' :
    ∀ m : mark, In m ms' -> In m ms ∨ (next <= m)%positive).
  { intros m Hin. pose proof (selection_in _ _ m Hsel Hin) as Hin_F.
    destruct (Pos.ltb m next) eqn:Hlt; [ left; exact (Hbefore m Hin_F Hlt) | ].
    right. apply Pos.ltb_ge in Hlt. exact Hlt. }
  assert (Hagree :
    ∀ r : variable,
      among r (cs_regions cs) = true ->
      lookup r vars' = lookup r vars_c
      ∧ among r
          (region_params_of params (hm_params_kinds metadata)
           ++ cs_regions cs)
        = true).
  { intros r Hr. split.
    - exact (bind_params_lookup_other r params vs vars_c vars'
               (Hregion_not_param r Hr) Hbind).
    - rewrite among_app, Hr. apply orb_true_r. }
  exists (handler_state cs params (hm_params_kinds metadata) entry), ms'.
  split; [ exact Hcheck | ]. split; [ exact Hincl | ]. split.
  - split;
      [ exact (proj2 (open_marks_marked vars' (he_open entry) ms') Hmarked')
      | ].
    split;
      [ exact (selection_placed (rs_frames rs) ms' (u_marks u) Hsel Hplaced)
      | ].
    split.
    + unfold check_state_scoped, handler_state.
      cbn [cs_open cs_regions cs_env].
      assert (Hopen_in :
        ∀ r : variable,
          among r (he_open entry) = true ->
          among r
            (region_params_of params (hm_params_kinds metadata)
             ++ cs_regions cs)
          = true).
      { intros r Hr. apply among_In in Hr.
        pose proof (forallb_in _ _ r Hentry_open Hr) as H.
        rewrite among_app. apply orb_true_iff in H as [H | H];
          [ rewrite H; reflexivity | ].
        rewrite (all_among_among _ _ r Hopen_regions H). apply orb_true_r. }
      apply andb_true_iff. split; [ apply all_among_iff; exact Hopen_in | ].
      apply bind_entry_params_scoped; [ exact Hopen_in | ].
      revert Henv_scoped. apply forallb_impl.
      intros [x [ | | r | ]] _; try (intros _; reflexivity).
      intros H. rewrite among_app, H. apply orb_true_r.
    + intros x l v Hl Hv. cbn [handler_state cs_env cs_open] in *.
      destruct (among x params) eqn:Hx.
      * apply among_In in Hx. destruct (In_nth_error params x Hx) as [i Hi].
        destruct (nth_error_same_length params _ (eq_sym Hlen_kinds) i x Hi)
          as (kind & Hkind_i).
        destruct (nth_error_same_length params (he_params entry)
                    (eq_sym Hlen_wanted) i x Hi)
          as (w & Hw).
        destruct (nth_error_same_length params args (eq_sym Hlen_args) i x Hi)
          as ([s l_i] & Harg).
        destruct (nth_error_same_length params vs (eq_sym Hlen_vs) i x Hi)
          as (v_i & Hv_i).
        rewrite (bind_entry_params_nth (he_open entry) params _ _ (cs_env cs)
                   i x kind w Hdistinct Hi Hkind_i Hw)
          in Hl.
        injection Hl as <-.
        rewrite (Hvs_nth i x v_i Hi Hv_i) in Hv. injection Hv as <-.
        unfold param_lifetime.
        destruct (kind_eqb kind Region) eqn:Hkind; [ exact I | ].
        destruct (value_args_nth cs_F given named _ args (he_params entry) i
                    kind s l_i w Hvalues Hkind_i Harg Hw)
          as [H | [_ Houtlives]]; [ congruence | ].
        pose proof (Forall2_nth_error _ _ _ Hargs i _ _ Harg Hv_i)
          as [Hholds _].
        cbn [snd] in Hholds.
        pose proof (outlives_holds rs (u_vars u) (cs_open cs_F) named
                      (u_marks u) ms' Hmarks Hplaced Hnamed Hsel Horder l_i
                      (lifetime_at_jump given w) v_i Houtlives Hholds)
          as Hnamed_holds.
        pose proof (holds_at_jump rs vars' given (he_open entry) ms' w v_i
                      Hmarked' Hnodup Hnamed_holds)
          as Hw_holds.
        destruct w as [ | | r | ]; try exact Hw_holds.
        destruct (among r (he_open entry)); [ exact Hw_holds | exact I ].
      * rewrite (bind_entry_params_other (he_open entry) x params _ _
                   (cs_env cs) Hx)
          in Hl.
        rewrite (bind_params_lookup_other x params vs vars_c vars' Hx Hbind)
          in Hv.
        apply (captured_holds rs vars_c vars' (cs_open cs) (he_open entry) ms
                 ms' next l v Hmarked_c Hdecrease_c
                 (fun m => forallb_ltb_in next ms m Hbelow_c) Hmarked'
                 Hdecrease' Hbefore');
          [ | exact (Hheld x l v Hl Hv) ].
        intros r -> Hin.
        pose proof
          (lookup_forallb
             (fun l =>
                match l with
                | Of_region r => among r (cs_regions cs)
                | Forever | Of_caller | Gone => true
                end)
             x (Of_region r) (cs_env cs) Henv_scoped Hl)
          as Hregion_r.
        cbv beta iota in Hregion_r.
        pose proof (forallb_in _ _ r Hentry_open Hin) as Hr.
        apply orb_true_iff in Hr as [Hr | Hr].
        -- apply among_In in Hr. apply region_params_of_among in Hr.
           apply among_In in Hr. rewrite (Hparam_not_region r Hr) in Hregion_r.
           discriminate.
        -- split; [ apply among_In; exact Hr | ].
           exact (proj1 (Hagree r Hregion_r)).
  - cbn [handler_state cs_conts cs_regions].
    set (u' :=
      User vars'
        (region_params_of params (hm_params_kinds metadata) ++ cs_regions cs)
        ms' (rs_next rs)).
    assert (Hmade' : made_at cert codes rs u' vars_c recursive cs base ms next).
    { split; [ exact Hconts_eq | ]. split; [ exact Hrec_check | ].
      split; [ exact Hrec_incl | ]. split; [ exact Hscoped | ].
      split; [ exact Hopen_c | ]. split; [ exact Hdecrease_c | ].
      split; [ exact Hbelow_c | ].
      split; [ apply Pos.leb_le; cbn [u_limit u']; lia | ].
      split; [ exact Hheld | ]. split.
      - intros m Hin Hlt. cbn [u_marks u'] in Hin. apply Pos.ltb_lt in Hlt.
        destruct (Hbefore' m Hin) as [H | H]; [ exact H | lia ].
      - exact Hagree. }
    split.
    + intros k ce.
      rewrite Hconts_eq, lookup_app, lookup_entries.
      unfold resolve. rewrite lookup_app, lookup_rec_closures.
      destruct (lookup k recursive) as [[params_k metadata_k handler_k] | ]
        eqn:Hk; cbn [option_map].
      * intros [= <-]. unfold cont_entry_of.
        destruct (lookup k (cert_handlers cert)) as [entry_k | ] eqn:Hentry;
          [ | apply Ok_unreachable ].
        pose proof (forallb_in _ _ _ Hrec_check (lookup_in _ _ _ Hk))
          as Hhandler.
        cbn [check_handler] in Hhandler. rewrite Hentry in Hhandler.
        apply andb_true_iff in Hhandler as [Hfits_k Hcheck_k].
        apply (Ok_handler cert codes rs u' params_k metadata_k handler_k
                 entry_k vars_c conts_c recursive cs base ms next Hmade'
                 Hfits_k Hcheck_k);
          [ | exact Hconts | exact Hnone ].
        exact (Hrec_incl k _ (lookup_in _ _ _ Hk)).
      * intros Hbase.
        apply (user_conts cert codes rs (User vars_c (cs_regions cs) ms next)
                 u');
          [ | exact (Hconts k ce Hbase) ].
        split; [ exact Hagree | ]. split; [ exact Hbefore' | ].
        cbn [u_limit u']. lia.
    + intros k.
      rewrite Hconts_eq, !lookup_app, lookup_entries, lookup_rec_closures.
      destruct (lookup k recursive); cbn [option_map]; [ discriminate | ].
      exact (Hnone k).
Qed.

End Enter.
