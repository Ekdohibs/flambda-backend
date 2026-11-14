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
   what the steps that enter a handler or the body of code come to
   ([entered]); the values that are good (as [values_plain] and [paps_fit]
   ask) and those that may be evaluated; a full call. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List Bool Lia PeanoNat.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaBindersProofs FlambdaValueSpec
  FlambdaHeapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaClosuresProofs FlambdaRegionsSpec FlambdaRegionsProofs
  FlambdaRegionsReadsProofs FlambdaRegionsFreshProofs
  FlambdaRegionsInvariantSpec FlambdaRegionsInvariantProofs
  FlambdaRegionsErasureSpec FlambdaRegionsCallsSpec FlambdaRegionsCheckSpec
  FlambdaRegionsCheckedSpec.
From OxCamlFlambda2 Require Import FlambdaRegionsCheckedBasicsProofs
  FlambdaRegionsCheckedHeadProofs FlambdaRegionsCheckedStableProofs
  FlambdaRegionsCheckedMarksProofs FlambdaRegionsCheckedOutlivesProofs
  FlambdaRegionsCheckedRankProofs FlambdaRegionsCheckedParamsProofs
  FlambdaRegionsCheckedJumpProofs.
Import ListNotations.

(* ======================================================================== *)
(* Code that binds nothing static binds no code                             *)
(* ======================================================================== *)

Fixpoint no_statics_no_code (e : expr) {struct e} :
  binds_statics e = false -> code_of_expr e = [].
Proof.
  destruct e
    as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    try (intros _; reflexivity).
  - exact (no_statics_no_code body).
  - cbn [binds_statics code_of_expr]. intros H.
    apply orb_false_iff in H as [Hbindings Hbody].
    rewrite (no_statics_no_code body Hbody), app_nil_r.
    induction bindings as [ | binding bindings IH]; [ reflexivity | ].
    cbn [existsb] in Hbindings.
    apply orb_false_iff in Hbindings as [Hbinding Hrest].
    cbn [flat_map]. rewrite (IH Hrest), app_nil_r.
    destruct binding; [ discriminate | reflexivity | discriminate ].
  - destruct h as [params metadata handler].
    cbn [binds_statics handler_binds_statics code_of_expr code_of_handler].
    intros H. apply orb_false_iff in H as [Hhandler Hbody].
    rewrite (no_statics_no_code handler Hhandler),
      (no_statics_no_code body Hbody).
    reflexivity.
  - cbn [binds_statics code_of_expr]. intros H.
    apply orb_false_iff in H as [Hhandlers Hbody].
    rewrite (no_statics_no_code body Hbody), app_nil_r.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      [ reflexivity | ].
    cbn [existsb handler_binds_statics] in Hhandlers.
    apply orb_false_iff in Hhandlers as [Hhandler Hrest].
    cbn [flat_map code_of_handler].
    rewrite (no_statics_no_code handler Hhandler), (IH Hrest). reflexivity.
Qed.

(* ======================================================================== *)
(* Values                                                                   *)
(* ======================================================================== *)

(* A value that reaches symbols of [ok] only is plain. *)
Lemma all_plain (ok : symbol -> bool) (v : value) :
  value_all ok v = true -> value_plain ok v = true.
Proof.
  induction v
    as [c | code slots IH | slot decls slots IH | s | tag fields IH
       | elements IH | c | l | callee args args_arity IHcallee IHargs]
    using value_induction;
    cbn [value_plain value_all]; auto.
  intros H. apply andb_true_iff in H as [Hcallee Hargs].
  rewrite (IHcallee Hcallee). cbn [andb].
  apply forallb_forall. intros arg Hin. rewrite Forall_forall in IHargs.
  exact (IHargs _ Hin (forallb_in _ _ arg Hargs Hin)).
Qed.

Lemma lookup_combine_nth {A : Type} (params : list variable) :
  ∀ (l : list A) (i : nat) (x : variable) (a : A),
    distinct params = true ->
    nth_error params i = Some x ->
    nth_error l i = Some a ->
    lookup x (combine params l) = Some a.
Proof.
  induction params as [ | p params IH]; intros l i x a Hdistinct;
    [ destruct i; discriminate | ].
  cbn [distinct] in Hdistinct. apply andb_true_iff in Hdistinct as [Hp Hrest].
  apply negb_true_iff in Hp.
  destruct l as [ | a0 l]; [ destruct i; intros _ H; discriminate H | ].
  cbn [combine]. destruct i as [ | i]; cbn [nth_error].
  - intros [= <-] [= <-]. apply lookup_head.
  - intros Hx Ha. rewrite lookup_cons.
    assert (Hne : Pos.eqb x p = false).
    { destruct (Pos.eqb x p) eqn:Heq; [ | reflexivity ].
      apply Pos.eqb_eq in Heq. subst x.
      apply nth_error_In in Hx. apply among_In in Hx. congruence. }
    rewrite Hne. exact (IH l i x a Hrest Hx Ha).
Qed.

Lemma lookup_combine_none {A : Type} (params : list variable) :
  ∀ (l : list A) (x : variable),
    among x params = false -> lookup x (combine params l) = None.
Proof.
  induction params as [ | p params IH]; intros l x; [ reflexivity | ].
  rewrite among_cons. intros H. apply orb_false_iff in H as [Hxp Hrest].
  destruct l as [ | a l]; [ reflexivity | ].
  cbn [combine]. rewrite lookup_cons, Hxp. exact (IH l x Hrest).
Qed.

Lemma values_fit_nth (rs : region_state) (flags : list bool) :
  ∀ (vs : list value) (i : nat) (v : value) (flag : bool),
    values_fit rs flags vs = true ->
    nth_error vs i = Some v ->
    nth_error flags i = Some flag ->
    value_all (heap_symbol rs) v = true ∨ flag = true.
Proof.
  induction flags as [ | flag0 flags IH]; intros vs i v flag.
  - destruct vs; [ destruct i; discriminate | discriminate ].
  - destruct vs as [ | v0 vs]; [ destruct i; discriminate | ].
    cbn [values_fit]. intros H. apply andb_true_iff in H as [Hhead Hrest].
    destruct i as [ | i]; cbn [nth_error].
    + intros [= <-] [= <-]. apply orb_true_iff in Hhead. exact Hhead.
    + exact (IH vs i v flag Hrest).
Qed.

Lemma values_fit_length (rs : region_state) (flags : list bool) :
  ∀ vs : list value,
    values_fit rs flags vs = true -> length vs <= length flags.
Proof.
  induction flags as [ | flag flags IH]; intros [ | v vs]; cbn [values_fit];
    try discriminate; cbn [length]; try lia.
  intros H. apply andb_true_iff in H as [_ H]. pose proof (IH vs H). lia.
Qed.

Section State.

Variable cert : certificate.

Variable codes : list (code_id * code).

Variable rs : region_state.

(* As [values_plain] and [paps_fit] ask of a value. *)
Definition good (v : value) : Prop :=
  value_plain (heap_symbol rs) v = true
  ∧ value_paps (pap_fits cert codes rs) v = true.

(* Every symbol that the value reaches may be evaluated. *)
Definition live (v : value) : Prop := value_within rs None v = true.

(* What a step that enters a handler or the body of code comes to, by
   the parts of the state: the frames, the counter and the static
   environment are as they were. *)
Definition entered (s' : state) : Prop :=
  ∃ (cs' : check_state) (ms' : list mark),
    check_expr cert codes cs' (state_expr s') = true
    ∧ incl (code_of_expr (state_expr s')) codes
    ∧ running_at rs (state_vars s') cs' ms'
    ∧ conts_checked cert codes rs
        (User (state_vars s') (cs_regions cs') ms' (rs_next rs))
        (cs_conts cs') (state_conts s')
    ∧ env_values_all good (state_vars s')
    ∧ conts_values_all good (state_conts s').

(* A caller that a callee can be given: the facts of [called_from]
   that are of the caller alone, and its counter is not after the
   machine's. *)
Record caller_ok (a : caller) : Prop := Caller_ok {
  caller_open :
    open_marks (caller_vars a) (cs_open (caller_check a))
    = Some (caller_marks a);
  caller_placed : marks_placed (rs_frames rs) (caller_marks a) = true;
  caller_among :
    FlambdaMapSpec.all_among (cs_open (caller_check a))
      (cs_regions (caller_check a))
    = true;
  caller_below :
    forallb (fun m => Pos.ltb m (caller_next a)) (caller_marks a) = true;
  caller_limit : (caller_next a <= rs_next rs)%positive
}.

Lemma called_from_caller_ok (u : user) (a : caller) :
  called_from rs u a ->
  (u_limit u <= rs_next rs)%positive ->
  caller_ok a.
Proof.
  intros (H1 & H2 & H3 & H4 & H5 & _) Hlimit. apply Pos.leb_le in H5.
  constructor; try assumption. lia.
Qed.

Lemma good_partial (callee : value) (args : list value) (args_arity : arity) :
  good (Value_partial_application callee args args_arity) ->
  good callee ∧ Forall good args.
Proof.
  unfold good. cbn [value_plain value_paps]. intros [Hplain Hfits].
  apply andb_true_iff in Hplain as [Hplain_callee Hplain_args].
  apply andb_true_iff in Hfits as [Hfits Hfits_args].
  apply andb_true_iff in Hfits as [_ Hfits_callee].
  split; [ split; assumption | ].
  apply Forall_forall. intros arg Hin.
  split; [ exact (forallb_in _ _ arg Hplain_args Hin) | ].
  exact (forallb_in _ _ arg Hfits_args Hin).
Qed.

Lemma live_partial (callee : value) (args : list value) (args_arity : arity) :
  live (Value_partial_application callee args args_arity)
  <-> live callee ∧ Forall live args.
Proof.
  unfold live, value_within. cbn [value_all]. rewrite andb_true_iff.
  rewrite forallb_Forall. reflexivity.
Qed.

(* A value as a lifetime that is not dead says is live. *)
Lemma holds_live (vars : env) (opens : list variable) (ms : list mark)
    (l : lifetime) (v : value) :
  Forall2 (marked vars) opens ms ->
  marks_placed (rs_frames rs) ms = true ->
  0 < rank opens l ->
  lifetime_holds rs opens ms None l v ->
  live v.
Proof.
  intros Hmarked Hplaced Hrank.
  assert (Hweaken :
    ∀ stop : option symbol, value_within rs stop v = true -> live v).
  { intros stop. apply within_later. intros b' H. discriminate H. }
  destruct l as [ | | r | ]; cbn [lifetime_holds rank] in *.
  - exact (heap_within rs None v).
  - exact (Hweaken _).
  - destruct (position r opens) as [i | ] eqn:Hr; [ | lia ].
    destruct (region_place_open vars opens ms None r
                (proj2 (open_marks_marked vars opens ms) Hmarked)
                (proj2 (position_among r opens) (ex_intro _ i Hr)))
      as (m & stop & Hplace & Hin).
    rewrite Hplace. intros H.
    exact (Hweaken _ (H (marks_placed_in _ _ m Hplaced Hin))).
  - lia.
Qed.

(* A full call: the body of the code runs in a state that is as the
   state in which the check takes the body says. *)
Lemma enter_code (a : caller) (closure : value) (id : code_id) (c : code)
    (entry : code_entry) (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (hp : heap) (rd : read_record)
      (s' : state) :
  caller_ok a ->
  check_code cert codes id c = true ->
  lookup id (cert_code cert) = Some entry ->
  good closure ->
  live closure ->
  (ce_closure_local entry = false ->
   value_all (heap_symbol rs) closure = true) ->
  Forall good vs ->
  Forall live vs ->
  values_fit rs (ce_params_local entry) vs = true ->
  ret_ok cert codes rs a
    (if ce_returns_local entry then newest_lifetime (caller_check a)
     else Forever)
    return_to ->
  raise_ok cert codes rs a exn_to ->
  cont_values_all good return_to ->
  cont_values_all good exn_to ->
  call_code closure c vs args_arity return_arity return_to exn_to
    (live_statics rs) hp rd
  = Step s' ->
  entered s'.
Proof.
  intros Hcaller Hcheck Hentry Hgood_closure Hlive_closure Hheap_closure
    Hgood_vs Hlive_vs Hfit Hret Hraise Hgood_ret Hgood_exn Hcall.
  destruct (call_code_step _ _ _ _ _ _ _ _ _ rd_ _ Hcall)
    as (callee_vars & Hno_statics & _ & _ & _ & Hbind & ->).
  destruct c as [params my_closure ret exn metadata body].
  cbn [code_params code_my_closure code_return_continuation
         code_exn_continuation code_body] in *.
  cbn [check_code] in Hcheck. rewrite Hentry in Hcheck.
  apply andb_true_iff in Hcheck as [Hcheck Hbody].
  apply andb_true_iff in Hcheck as [Hcheck _].
  apply andb_true_iff in Hcheck as [Hcheck _].
  apply andb_true_iff in Hcheck as [Hcheck _].
  apply andb_true_iff in Hcheck as [Hcheck Hconts_differ].
  apply andb_true_iff in Hcheck as [Hcheck Hdistinct].
  apply andb_true_iff in Hcheck as [Hlen_flags _].
  apply Nat.eqb_eq in Hlen_flags. apply negb_true_iff in Hconts_differ.
  cbn [distinct] in Hdistinct.
  apply andb_true_iff in Hdistinct as [Hclosure_fresh Hdistinct].
  apply negb_true_iff in Hclosure_fresh.
  destruct (bind_params_nth params vs _ callee_vars Hdistinct Hbind)
    as [Hlen_vs Hvs_nth].
  exists (code_state entry params my_closure ret exn), [].
  cbn [state_expr state_vars state_conts].
  split; [ exact Hbody | ].
  split; [ rewrite (no_statics_no_code body Hno_statics); intros x [] | ].
  split; [ | split; [ | split ] ].
  - split; [ reflexivity | ]. split; [ reflexivity | ]. split.
    + unfold check_state_scoped, code_state. cbn [cs_open cs_regions cs_env].
      cbn [FlambdaMapSpec.all_among forallb andb].
      destruct (ce_closure_local entry); cbn [andb];
        apply forallb_forall; intros [x l] Hin;
        apply in_combine_r in Hin; apply in_map_iff in Hin as ([ | ] & <- & _);
        reflexivity.
    + intros x l v. unfold code_state. cbn [cs_env cs_open].
      rewrite lookup_cons. destruct (Pos.eqb x my_closure) eqn:Hx.
      * apply Pos.eqb_eq in Hx. subst x. intros [= <-].
        rewrite (bind_params_lookup_other my_closure params vs
                   [(my_closure, closure)] callee_vars Hclosure_fresh Hbind),
          lookup_head.
        intros [= <-].
        destruct (ce_closure_local entry) eqn:Hlocal;
          [ exact Hlive_closure | exact (Hheap_closure eq_refl) ].
      * destruct (among x params) eqn:Hparam;
          [ | rewrite (lookup_combine_none params _ x Hparam); discriminate ].
        apply among_In in Hparam.
        destruct (In_nth_error params x Hparam) as [i Hi].
        destruct (nth_error_same_length params (ce_params_local entry)
                    (eq_sym Hlen_flags) i x Hi)
          as (flag & Hflag).
        destruct (nth_error_same_length params vs (eq_sym Hlen_vs) i x Hi)
          as (v_i & Hv_i).
        rewrite (lookup_combine_nth params _ i x _ Hdistinct Hi
                   (map_nth_error _ i (ce_params_local entry) Hflag)),
          (Hvs_nth i x v_i Hi Hv_i).
        intros [= <-] [= <-].
        destruct (values_fit_nth rs _ vs i v_i flag Hfit Hv_i Hflag)
          as [Hheap | ->].
        -- destruct flag;
             [ exact (heap_within rs None v_i Hheap) | exact Hheap ].
        -- exact (proj1 (Forall_forall live vs) Hlive_vs v_i
                    (nth_error_In vs i Hv_i)).
  - unfold code_state. cbn [cs_conts cs_regions]. split.
    + intros k ce. unfold resolve. rewrite !lookup_cons.
      destruct (Pos.eqb k ret) eqn:Hk.
      * intros [= <-].
        apply (Ok_return cert codes rs _ (ce_returns_local entry) a return_to);
          [ | exact Hret ].
        destruct Hcaller as [H1 H2 H3 H4 H5].
        split; [ exact H1 | ]. split; [ exact H2 | ]. split; [ exact H3 | ].
        split; [ exact H4 | ].
        split; [ apply Pos.leb_le; exact H5 | reflexivity ].
      * destruct (Pos.eqb k exn) eqn:Hk'; [ | discriminate ].
        intros [= <-]. apply (Ok_raise cert codes rs _ a exn_to);
          [ | exact Hraise ].
        destruct Hcaller as [H1 H2 H3 H4 H5].
        split; [ exact H1 | ]. split; [ exact H2 | ]. split; [ exact H3 | ].
        split; [ exact H4 | ].
        split; [ apply Pos.leb_le; exact H5 | reflexivity ].
    + intros k. rewrite !lookup_cons.
      destruct (Pos.eqb k ret); [ discriminate | ].
      destruct (Pos.eqb k exn); [ discriminate | reflexivity ].
  - apply (bind_params_values_all good params vs [(my_closure, closure)]
             callee_vars Hgood_vs);
      [ | exact Hbind ].
    apply Forall_cons; [ exact Hgood_closure | apply Forall_nil ].
  - split; [ exact Hgood_ret | ]. split; [ exact Hgood_exn | exact I ].
Qed.

End State.
