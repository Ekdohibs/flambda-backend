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
   what the property gives of the head of a state. The simples that a step
   evaluates pass the test of the machine, and a continuation that takes
   what a call returns, or its exception, takes no region. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaHeapProofs
  FlambdaSemanticsSpec FlambdaRegionsSpec FlambdaRegionsProofs
  FlambdaRegionsReadsProofs FlambdaRegionsStepsProofs
  FlambdaRegionsFreshProofs FlambdaRegionsInvariantSpec
  FlambdaRegionsInvariantProofs FlambdaRegionsErasureSpec
  FlambdaRegionsCallsSpec FlambdaRegionsCheckSpec FlambdaRegionsCheckedSpec.
From OxCamlFlambda2 Require Import FlambdaRegionsCheckedBasicsProofs.
Import ListNotations.

Scheme cont_ok_mut := Induction for cont_ok Sort Prop
  with ret_ok_mut := Induction for ret_ok Sort Prop
  with raise_ok_mut := Induction for raise_ok Sort Prop.
Combined Scheme conts_ok_mutind from cont_ok_mut, ret_ok_mut, raise_ok_mut.

(* ======================================================================== *)
(* The simples of the head                                                  *)
(* ======================================================================== *)

(* Of the simples that a step evaluates: one at the place of a region
   has for its value a mark that is on the stack; the value of any
   other is within no bound. *)
Fixpoint simples_ok (rs : region_state) (simples : list simple)
    (places : list bool) : Prop :=
  match simples with
  | [] => True
  | s :: simples =>
    (if hd false places then simple_open_mark rs s = true
     else
       ∀ v : value,
         eval_simple (state_vars (rs_state rs)) s = Some v ->
         value_within rs None v = true)
    ∧ simples_ok rs simples (tl places)
  end.

(* What the other half is given. *)
Lemma simples_ok_live (rs : region_state) (simples : list simple) :
  ∀ places : list bool,
    simples_ok rs simples places -> simples_live rs simples places = true.
Proof.
  induction simples as [ | s simples IH]; intros places; [ reflexivity | ].
  cbn [simples_ok simples_live]. intros [Hs Hrest].
  apply andb_true_iff. split; [ | exact (IH _ Hrest) ].
  destruct (hd false places); [ exact Hs | ].
  destruct (eval_simple (state_vars (rs_state rs)) s) as [v | ];
    [ | reflexivity ].
  exact (value_within_live rs None v (Hs v eq_refl)).
Qed.

(* A mark that is on the stack may not be evaluated. *)
Lemma open_mark_unusable (rs : region_state) (s : simple) :
  simple_open_mark rs s = true ->
  FlambdaRegionsSpec.simple_usable rs s = false.
Proof.
  unfold simple_open_mark, FlambdaRegionsSpec.simple_usable.
  destruct (eval_simple (state_vars (rs_state rs)) s) as [v | ];
    [ | discriminate ].
  unfold value_open_mark. destruct v; try discriminate.
  intros H. apply andb_true_iff in H as [H _].
  destruct (value_usable rs (Value_symbol s0)); [ discriminate | reflexivity ].
Qed.

(* No fault of the simples. *)
Lemma simples_ok_no_fault (rs : region_state) (simples : list simple) :
  ∀ places : list bool,
    simples_ok rs simples places -> simples_fault rs simples places = None.
Proof.
  induction simples as [ | s simples IH]; intros places; [ reflexivity | ].
  cbn [simples_ok simples_fault]. intros [Hs Hrest].
  destruct (hd false places).
  - rewrite (open_mark_unusable rs s Hs), Hs. exact (IH _ Hrest).
  - assert (Husable : FlambdaRegionsSpec.simple_usable rs s = true).
    { unfold FlambdaRegionsSpec.simple_usable.
      destruct (eval_simple (state_vars (rs_state rs)) s) as [v | ];
        [ | reflexivity ].
      exact (within_usable rs None v (Hs v eq_refl)). }
    rewrite Husable. exact (IH _ Hrest).
Qed.

(* With no place of a region: every simple that the check lets be
   evaluated. *)
Lemma usable_simples_ok (rs : region_state) (cs : check_state)
    (ms : list mark) (simples : list simple) :
  running_checked rs cs ms ->
  (∀ s : simple, In s simples -> simple_below (rs_base rs) s = true) ->
  forallb (check_usable cs) simples = true ->
  simples_ok rs simples [].
Proof.
  intros Hrun.
  induction simples as [ | s simples IH]; [ exact (fun _ _ => I) | ].
  intros Hbelow H. cbn [forallb] in H. apply andb_true_iff in H as [Hs Hrest].
  cbn [simples_ok hd tl]. split.
  - intros v Hv.
    exact
      (running_usable_within rs cs ms s v Hrun (Hbelow s (or_introl eq_refl))
         Hs Hv).
  - apply IH; [ | exact Hrest ]. intros s' Hin.
    exact (Hbelow s' (or_intror Hin)).
Qed.

Lemma forever_usable (cs : check_state) (l : lifetime) :
  is_forever l = true -> lifetime_usable cs l = true.
Proof. destruct l; cbn; first [ reflexivity | discriminate ]. Qed.

(* The arguments of a jump to a handler: one at a parameter that is a
   region is the variable of an open region; any other may be
   evaluated. *)
Lemma handler_simples_ok (rs : region_state) (cs : check_state)
    (ms : list mark) (given0 : list (variable * variable))
    (named : list variable) :
  symbols_fresh rs = true ->
  running_checked rs cs ms ->
  ∀ (simples : list simple) (params : list variable)
    (kinds : list flambda_kind) (wanted : list lifetime)
    (given : list (variable * variable)),
    (∀ s : simple, In s simples -> simple_below (rs_base rs) s = true) ->
    region_args cs params kinds
      (map (fun s => (Some s, lifetime_of cs s)) simples)
    = Some given ->
    value_args cs given0 named kinds
      (map (fun s => (Some s, lifetime_of cs s)) simples) wanted
    = true ->
    simples_ok rs simples (map (fun kind => kind_eqb kind Region) kinds).
Proof.
  intros Hfresh Hrun.
  induction simples as [ | s simples IH];
    intros params kinds wanted given Hbelow; [ exact (fun _ _ => I) | ].
  cbn [map].
  destruct params as [ | p params];
    [ destruct kinds; cbn [region_args]; discriminate | ].
  destruct kinds as [ | kind kinds]; [ cbn [region_args]; discriminate | ].
  cbn [region_args value_args].
  destruct (region_args cs params kinds
              (map (fun s => (Some s, lifetime_of cs s)) simples))
    as [rest | ] eqn:Hrest; [ | simpl; discriminate ].
  simpl. intros Hgiven.
  destruct wanted as [ | w wanted]; [ discriminate | ].
  intros Hvalues. apply andb_true_iff in Hvalues as [Hvalue Hvalues].
  cbn [simples_ok map hd tl]. split.
  - destruct (kind_eqb kind Region) eqn:Hkind.
    + destruct s as [r | sym | c]; try discriminate.
      destruct (among r (cs_open cs)) eqn:Hopen; [ | discriminate ].
      destruct (running_region_open rs cs ms r Hrun Hopen) as (m & Hr & Hon).
      unfold simple_open_mark. cbn [eval_simple]. rewrite Hr.
      exact (mark_open rs m Hfresh Hon).
    + cbn [orb] in Hvalue. apply andb_true_iff in Hvalue as [Husable _].
      intros v Hv.
      exact
        (running_usable_within rs cs ms s v Hrun
           (Hbelow s (or_introl eq_refl)) Husable Hv).
  - apply (IH params kinds wanted rest); [ | exact Hrest | exact Hvalues ].
    intros s' Hin. exact (Hbelow s' (or_intror Hin)).
Qed.

(* ======================================================================== *)
(* What a caller gave takes no region                                       *)
(* ======================================================================== *)

(* With no simple among the arguments, no parameter is a region. *)
Lemma region_args_no_region (cs : check_state) :
  ∀ (params : list variable) (kinds : list flambda_kind)
    (args : list (option simple * lifetime))
    (given : list (variable * variable)),
    (∀ (s : option simple) (l : lifetime), In (s, l) args -> s = None) ->
    region_args cs params kinds args = Some given ->
    forallb negb (map (fun kind => kind_eqb kind Region) kinds) = true.
Proof.
  induction params as [ | p params IH]; intros kinds args given Hnone.
  - destruct kinds; [ reflexivity | ]. cbn [region_args]. discriminate.
  - destruct kinds as [ | kind kinds]; [ cbn [region_args]; discriminate | ].
    destruct args as [ | [s l] args]; [ cbn [region_args]; discriminate | ].
    cbn [region_args].
    destruct (region_args cs params kinds args) as [rest | ] eqn:Hrest;
      [ | simpl; discriminate ].
    simpl. intros Hgiven. apply andb_true_iff. split.
    + destruct (kind_eqb kind Region); [ | reflexivity ].
      rewrite (Hnone s l (or_introl eq_refl)) in Hgiven. discriminate.
    + apply (IH kinds args rest); [ | exact Hrest ].
      intros s' l' Hin. exact (Hnone s' l' (or_intror Hin)).
Qed.

Definition takes_no_region (c : cont_value) : Prop :=
  match c with
  | Closure h _ _ _ => forallb negb (region_params h) = true
  | _ => True
  end.

Section Callers.

Variable cert : certificate.
Variable codes : list (code_id * code).
Variable rs : region_state.

(* A handler that a jump with no simple among its arguments goes to
   takes no region. *)
Lemma jump_none_no_region (cs : check_state) (u : user) (k : continuation)
    (args : list (option simple * lifetime)) (c : cont_value) :
  (∀ (s : option simple) (l : lifetime), In (s, l) args -> s = None) ->
  jump cs k args = true ->
  (∀ ce : cont_entry,
     lookup k (cs_conts cs) = Some ce ->
     cont_ok cert codes rs u ce c
     ∧ match ce with
       | To_return _ | To_raise => takes_no_region c
       | _ => True
       end) ->
  (lookup k (cs_conts cs) = None -> c = Outer k) ->
  takes_no_region c.
Proof.
  intros Hnone Hjump Hsome Houter. unfold jump in Hjump.
  destruct (lookup k (cs_conts cs)) as [ce | ] eqn:Hk;
    [ | rewrite (Houter eq_refl); exact I ].
  destruct (Hsome ce eq_refl) as [Hok Hce].
  destruct ce as [params kinds entry | | local | ];
    [ | discriminate | exact Hce | exact Hce ].
  inversion Hok; subst.
  destruct (region_args cs params (hm_params_kinds metadata) args)
    as [given | ] eqn:Hgiven; [ | discriminate ].
  unfold takes_no_region, region_params. cbn [handler_kinds].
  exact (region_args_no_region cs _ _ _ _ Hnone Hgiven).
Qed.

Lemma callers_take_no_region :
  (∀ (u : user) (ce : cont_entry) (c : cont_value),
     cont_ok cert codes rs u ce c ->
     match ce with
     | To_return _ | To_raise => takes_no_region c
     | _ => True
     end)
  ∧ (∀ (a : caller) (l : lifetime) (c : cont_value),
       ret_ok cert codes rs a l c -> takes_no_region c)
  ∧ (∀ (a : caller) (c : cont_value),
       raise_ok cert codes rs a c -> takes_no_region c).
Proof.
  apply (conts_ok_mutind cert codes rs
           (fun u ce c _ =>
              match ce with
              | To_return _ | To_raise => takes_no_region c
              | _ => True
              end)
           (fun a l c _ => takes_no_region c)
           (fun a c _ => takes_no_region c)).
  - intros. exact I.
  - intros. exact I.
  - intros u local a c Hcalled Hret IH. exact IH.
  - intros u a c Hcalled Hraise IH. exact IH.
  - intros a l k c Hjump Hsome IH Houter.
    unfold jump_result in Hjump.
    assert (Hargs :
      ∃ args : list (option simple * lifetime),
        (∀ (s : option simple) (l0 : lifetime), In (s, l0) args -> s = None)
        ∧ jump (caller_check a) k args = true).
    { destruct (lookup k (cs_conts (caller_check a)))
        as [[params kinds entry | | local | ] | ];
        eexists; (split; [ | exact Hjump ]); intros s l0 Hin;
        try (apply in_map_iff in Hin as (p & [= <- _] & _); reflexivity);
        destruct Hin as [[= <- _] | []]; reflexivity. }
    destruct Hargs as (args & Hnone & Hjump').
    apply (jump_none_no_region (caller_check a) (caller_user a) k args c Hnone
             Hjump'); [ | exact Houter ].
    intros ce Hk. exact (conj (Hsome ce Hk) (IH ce Hk)).
  - intros. exact I.
  - intros a l l' c Hlasts Hret IH. exact IH.
  - intros a k simples extra c Heval Hholds Hjump Hsome IH Houter.
    destruct extra as [ | v extra]; [ | exact I ].
    cbn [with_extra_args]. inversion Hholds; subst.
    cbn [map] in Hjump.
    apply (jump_none_no_region (caller_check a) (caller_user a) k
             [(None, Forever)] c); [ | exact Hjump | | exact Houter ].
    + intros s l [[= <- _] | []]. reflexivity.
    + intros ce Hk. exact (conj (Hsome ce Hk) (IH ce Hk)).
Qed.

End Callers.

(* ======================================================================== *)
(* The simples of the head of a checked state                               *)
(* ======================================================================== *)

Lemma simples_ok_no_region (rs : region_state) (simples : list simple) :
  ∀ places : list bool,
    forallb negb places = true ->
    simples_ok rs simples [] ->
    simples_ok rs simples places.
Proof.
  induction simples as [ | s simples IH]; intros places Hplaces;
    [ exact (fun H => H) | ].
  cbn [simples_ok hd tl]. intros [Hs Hrest].
  destruct places as [ | place places]; [ exact (conj Hs Hrest) | ].
  cbn [forallb] in Hplaces. apply andb_true_iff in Hplaces as [Hplace Hplaces].
  destruct place; [ discriminate | ]. cbn [hd tl].
  split; [ exact Hs | exact (IH places Hplaces Hrest) ].
Qed.

(* The simples of a [Let] that is not of a primitive of regions, and
   those of a [Let_static], an [Apply] and a [Switch], are simples
   that the check lets be evaluated. *)
Lemma check_usable_simples (cert : certificate)
    (codes : list (code_id * code)) (cs : check_state) (e : expr) :
  check_expr cert codes cs e = true ->
  match e with
  | Apply_cont _ => True
  | _ => forallb (check_usable cs) (evaluated_simples e) = true
  end.
Proof.
  destruct e
    as [x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    try reflexivity; try exact (fun _ => I).
  - destruct d as [s | [u | b | v | hp | rp] args | | | | | ];
      try reflexivity;
      cbn [check_expr evaluated_simples]; intros H;
      apply andb_true_iff in H as [H _]; apply andb_true_iff in H as [_ H];
      exact H.
  - cbn [check_expr evaluated_simples]. intros H.
    apply andb_true_iff in H as [H _]. apply andb_true_iff in H as [H _].
    apply forallb_forall. intros s Hin.
    exact (forever_usable cs _ (forallb_in _ _ s H Hin)).
  - cbn [check_expr evaluated_simples]. intros H.
    apply andb_true_iff in H as [H _]. apply andb_true_iff in H as [H Hextra].
    apply andb_true_iff in H as [Hcallee Hargs].
    cbn [forallb]. rewrite Hcallee, forallb_app, Hargs, Hextra. reflexivity.
  - cbn [check_expr evaluated_simples forallb]. intros H.
    apply andb_true_iff in H as [H _]. rewrite H. reflexivity.
Qed.

Section Head.

Variable cert : certificate.
Variable codes : list (code_id * code).

Lemma head_simples_ok (rs : region_state) (cs : check_state)
    (ms : list mark) :
  symbols_fresh rs = true ->
  check_expr cert codes cs (state_expr (rs_state rs)) = true ->
  running_checked rs cs ms ->
  conts_checked cert codes rs (running_user rs cs ms) (cs_conts cs)
    (state_conts (rs_state rs)) ->
  simples_ok rs (evaluated_simples (state_expr (rs_state rs)))
    (region_places rs).
Proof.
  intros Hfresh Hcheck Hrun [Hconts Houter].
  assert (Hbelow :
    ∀ s : simple,
      In s (evaluated_simples (state_expr (rs_state rs))) ->
      simple_below (rs_base rs) s = true)
    by (intros s Hin; exact (head_simples_below rs s Hfresh Hin)).
  pose proof (check_usable_simples cert codes cs _ Hcheck) as Husable.
  unfold region_places.
  destruct (state_expr (rs_state rs))
    as [x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ] eqn:He;
    try exact (usable_simples_ok rs cs ms _ Hrun Hbelow Husable).
  clear Husable. cbn [check_expr] in Hcheck. cbn [evaluated_simples] in *.
  set (k := ac_continuation ac) in *.
  unfold jump, apply_cont_args in Hcheck.
  assert (Hplain :
    forallb (fun '(_, l) => lifetime_usable cs l)
      (map (fun s => (Some s, lifetime_of cs s)) (ac_args ac))
    = true ->
    simples_ok rs (ac_args ac) []).
  { intros H. apply (usable_simples_ok rs cs ms _ Hrun Hbelow).
    apply forallb_forall. intros s Hin.
    exact
      (forallb_in _ _ (Some s, lifetime_of cs s) H
         (in_map (fun s => (Some s, lifetime_of cs s)) _ s Hin)). }
  destruct (lookup k (cs_conts cs)) as [ce | ] eqn:Hk.
  - specialize (Hconts k ce Hk).
    destruct ce as [params kinds entry | | local | ].
    + inversion Hconts; subst.
      destruct (region_args cs params (hm_params_kinds metadata) _)
        as [given | ] eqn:Hgiven; [ | discriminate ].
      apply andb_true_iff in Hcheck as [_ Hvalues].
      unfold region_params. cbn [handler_kinds].
      exact
        (handler_simples_ok rs cs ms _ _ Hfresh Hrun _ _ _ _ _ Hbelow Hgiven
           Hvalues).
    + discriminate.
    + pose proof (proj1 (callers_take_no_region cert codes rs) _ _ _ Hconts)
        as Hno.
      assert (Hok : simples_ok rs (ac_args ac) []).
      { apply Hplain. apply forallb_forall. intros [s l] Hin.
        pose proof (forallb_in _ _ (s, l) Hcheck Hin) as H. cbv beta iota in H.
        apply andb_true_iff in H as [H _]. exact H. }
      destruct (resolve k (state_conts (rs_state rs))); try exact Hok.
      exact (simples_ok_no_region rs _ _ Hno Hok).
    + pose proof (proj1 (callers_take_no_region cert codes rs) _ _ _ Hconts)
        as Hno.
      assert (Hok : simples_ok rs (ac_args ac) []).
      { apply Hplain. apply forallb_forall. intros [s l] Hin.
        pose proof (forallb_in _ _ (s, l) Hcheck Hin) as H. cbv beta iota in H.
        exact (forever_usable cs l H). }
      destruct (resolve k (state_conts (rs_state rs))); try exact Hok.
      exact (simples_ok_no_region rs _ _ Hno Hok).
  - unfold resolve. rewrite (Houter k Hk). exact (Hplain Hcheck).
Qed.

End Head.
