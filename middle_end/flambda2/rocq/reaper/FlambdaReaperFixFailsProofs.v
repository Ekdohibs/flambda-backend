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



(* The proofs of FlambdaReaperFixFailsSpec.v. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaWellFormedSpec FlambdaSemanticsSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperRebuild FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaReaperLeast FlambdaReaperFlowSpec FlambdaReaperLeastSpec
  FlambdaReaperWrapperSpec FlambdaReaperWrapperDerivedSpec
  FlambdaReaperFixFailsSpec.
From OxCamlFlambda2 Require FlambdaReaperWrapperProofs
  FlambdaReaperWrapperDerivedProofs FlambdaReaperFlowProofs FlambdaMapProofs.
From OxCamlFlambda2 Require Import FlambdaBindersSpec.
Import ListNotations.

(* ======================================================================== *)
(* The arm                                                                  *)
(* ======================================================================== *)

(* [wrapper_fails_u] holds where a position fails. *)
Lemma wrapper_fails_at (ads : list decision) :
  ∀ (func : list decision) (i : nat) (fields : unboxed_fields)
    (fd : decision),
    nth_error ads i = Some (Unbox fields) -> nth_error func i = Some fd ->
    arm_pair (Unbox fields) fd = true -> wrapper_fails_u ads func = true.
Proof.
  induction ads as [ | ad ads IH]; intros [ | fd' func] [ | i] fields fd Ha Hf
    Harm; cbn in Ha, Hf; try discriminate.
  - injection Ha as ->. injection Hf as ->.
    destruct fd; cbn in Harm |- *; first [ reflexivity | discriminate ].
  - cbn [wrapper_fails_u].
    destruct ad as [d | fs | k], fd' as [d' | fs' | k'];
      first [ reflexivity | exact (IH func i fields fd Ha Hf Harm) ].
Qed.

(* The decisions of a continuation with no more parameters than the call
   has results fail at the callee's as at their pairs with the kinds. *)
Lemma wrapper_fails_combine (ads rd : list decision)
    (kinds : list flambda_kind) :
  length ads <= length kinds -> wrapper_fails_u ads rd = true ->
  wrapper_fails_u ads (map fst (combine rd kinds)) = true.
Proof.
  intros Hlen H.
  destruct (FlambdaReaperWrapperDerivedProofs.wrapper_fails_pos ads rd H)
    as (i & fields & fd & Ha & Hf & Harm).
  assert (Hi : i < length ads) by (apply nth_error_Some; congruence).
  destruct (nth_error kinds i) as [k | ] eqn:Hk;
    [ | apply nth_error_None in Hk; lia ].
  apply (wrapper_fails_at ads _ i fields fd Ha); [ | exact Harm ].
  rewrite nth_error_map,
    (FlambdaReaperWrapperDerivedProofs.CheckNeverReturns.combine_nth_some
       _ _ i fd k Hf Hk).
  reflexivity.
Qed.

(* ======================================================================== *)
(* The walk                                                                 *)
(* ======================================================================== *)

Lemma no_statics_codes :
  (∀ e : expr, binds_statics e = false -> unit_codes e = [])
  ∧ (∀ h : continuation_handler,
       handler_binds_statics h = false -> handler_unit_codes h = []).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind;
    cbn [binds_statics handler_binds_statics unit_codes handler_unit_codes].
  - intros x kind d body IH H. exact (IH H).
  - intros bindings body _ IH H. apply orb_false_iff in H as [H1 H2].
    rewrite (IH H2), app_nil_r. clear -H1.
    induction bindings as [ | b bs IHb]; [ reflexivity | ].
    cbn [existsb] in H1. apply orb_false_iff in H1 as [Hb Hbs].
    destruct b; cbn [binding_binds] in Hb; try discriminate.
    exact (IHb Hbs).
  - intros k h body IHh IH H. apply orb_false_iff in H as [H1 H2].
    rewrite (IHh H1), (IH H2). reflexivity.
  - intros handlers body IHh IH H. apply orb_false_iff in H as [H1 H2].
    rewrite (IH H2), app_nil_r.
    induction handlers as [ | [k h] hs IHl]; [ reflexivity | ].
    cbn [existsb] in H1. apply orb_false_iff in H1 as [Hh Hhs].
    cbn [flat_map]. rewrite (IHh k h (or_introl eq_refl) Hh).
    exact (IHl (fun k' h' Hin => IHh k' h' (or_intror Hin)) Hhs).
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - intros params md handler IH H. exact (IH H).
Qed.

Section Walk.

Variable F : facts.
Variable sol : solution.
Variable cc : node -> bool.
Variable cds : finite_map code_dep.
Variable pdc : bool.
Variable uci : list code_id.
Variable codes : finite_map code_metadata.
Variable unboxed : list (node * unboxed_fields).
Variable ff : positive.
Variable changed : list (node * closure_repr).

Abbreviation fails :=
  (rebuild_fails_u true F sol cc cds pdc uci codes unboxed ff changed).
Abbreviation hfails :=
  (handler_fails_u true F sol cc cds pdc uci codes unboxed ff changed).
Abbreviation cfails :=
  (code_fails_u true F sol cc cds pdc uci codes unboxed ff changed).
Abbreviation rcode :=
  (rebuild_code_u true F sol cc cds pdc uci codes unboxed ff changed).
Abbreviation calls := (rebuilt_calls sol cc cds uci unboxed).
Abbreviation hcalls := (handler_rebuilt_calls sol cc cds uci unboxed).

Definition call_decisions (ds : finite_map (list decision))
    (ap : apply_expr) : list decision :=
  match lookup (ap_continuation ap) ds with
  | Some decisions => decisions
  | None => []
  end.

(* What [apply_fails_u] gives [wrapper_fails_u] for the callee. *)
Definition callee_returns (ap : apply_expr) : list decision :=
  match apply_change_u F sol cc cds uci codes unboxed ap with
  | Not_changing_u =>
    map (fun kind => Decided (Keep kind)) (unarize (ap_return_arity ap))
  | Changing_u _ return_decisions => return_decisions
  end.

(* A call at which the fix stops: an unboxed callee of code that keeps
   its calling convention, or the arm of a call that may return. *)
Definition fails_at (ads : list decision) (ap : apply_expr) : Prop :=
  (apply_change_u F sol cc cds uci codes unboxed ap = Not_changing_u
   ∧ ∃ fields, unboxed_of unboxed (ap_callee ap) = Some fields)
  ∨ (apply_never_returns F sol cds codes ap = false
     ∧ wrapper_fails_u ads (callee_returns ap) = true).

Lemma apply_fails_split (en : unbox_env) (ap : apply_expr) :
  apply_fails_u F sol cc cds uci codes unboxed en ap = true ->
  fails_at (call_decisions (ue_decisions en) ap) ap.
Proof.
  unfold apply_fails_u, fails_at, callee_returns, call_decisions. cbv zeta.
  destruct (_ || _); [ discriminate | ].
  destruct (rebuild_exn_extra_args_u _ _ _ _); [ | discriminate ].
  destruct (apply_change_u _ _ _ _ _ _ _ _) as [ | pd rd].
  - destruct (unboxed_of _ _) as [fields | ].
    + intros _. left. split; [ reflexivity | exists fields; reflexivity ].
    + intros H. apply andb_true_iff in H as [H1 H2].
      apply negb_true_iff in H1. right. split; assumption.
  - repeat match goal with
    | |- context [match ?x with _ => _ end] => destruct x
    end; intros H; try discriminate H;
    apply andb_true_iff in H as [H1 H2]; apply negb_true_iff in H1;
    right; split; assumption.
Qed.

Abbreviation bfails :=
  (binding_fails_u true F sol cc cds pdc uci codes unboxed ff changed).

Lemma fails_static (en : unbox_env) (bindings : list static_binding)
    (body : expr) :
  fails en (Let_static bindings body)
  = existsb bfails bindings || fails (ubind_variables en []) body.
Proof. reflexivity. Qed.

Lemma bfails_code (id : code_id) (c : code) :
  bfails (Bind_code id c) = cfails id c.
Proof. reflexivity. Qed.

Lemma cfails_code (id : code_id) (params : list variable)
    (mc : variable) (ret exn : continuation) (md : code_metadata)
    (body : expr) :
  let bound :=
    (mc, Value) :: with_kinds params (unarize (cm_params_arity md))
  in
  cfails id (Code params mc ret exn md body)
  = negb (existsb (fun '(x, kind) => is_dead_var sol x kind) bound)
    && fails
         (ubind_variables
            (ubind_continuations empty_unbox_env
               (code_continuation_decisions sol cds unboxed id md
                  (code_change_u sol cc cds uci unboxed id md) ret exn))
            bound)
         body
    || code_never_returns sol cds id
       && among ret
            (free_continuations_of_expr
               (code_body (rcode id (Code params mc ret exn md body)))).
Proof. reflexivity. Qed.

Lemma fails_rec (en : unbox_env)
    (handlers : list (continuation * continuation_handler)) (body : expr) :
  let en' :=
    ubind_continuations en
      (map (fun '(k, h) => (k, handler_decisions true sol unboxed false h))
         handlers)
  in
  fails en (Let_cont_recursive handlers body)
  = existsb (fun '(_, h) => hfails en' false h) handlers || fails en' body.
Proof. reflexivity. Qed.

(* Code whose [my_closure] is dead stops nothing: its body is
   [Invalid]. *)
Lemma code_fails_dead (id : code_id) (c : code) :
  is_dead_var sol (code_my_closure c) Value = true -> cfails id c = false.
Proof.
  destruct c as [params mc ret exn md body]. cbn [code_my_closure].
  intros Hd. rewrite cfails_code. cbv zeta. cbn [existsb]. rewrite Hd.
  cbn [orb negb andb].
  assert (E : code_body (rcode id (Code params mc ret exn md body))
              = Invalid).
  { cbn [rebuild_code_u existsb]. rewrite Hd. cbn [orb].
    destruct (code_change_u _ _ _ _ _ _ _); reflexivity. }
  rewrite E. apply andb_false_r.
Qed.

(* What the walk finds: a call of [cl] at which the fix stops, or code of
   [cs] that is used and fails [check_never_returns]. *)
Definition found (cl : list (list decision * apply_expr))
    (cs : list (code_id * code)) : Prop :=
  (∃ ads ap, (ads, ap) ∈ cl ∧ fails_at ads ap)
  ∨ (∃ id c,
       (id, c) ∈ cs ∧ is_code_id_used sol id = true
       ∧ code_never_returns sol cds id
         && among (code_return_continuation c)
              (free_continuations_of_expr (code_body (rcode id c)))
         = true).

Lemma found_mono (cl cl' : list (list decision * apply_expr))
    (cs cs' : list (code_id * code)) :
  (∀ x, x ∈ cl -> x ∈ cl') -> (∀ x, x ∈ cs -> x ∈ cs') ->
  found cl cs -> found cl' cs'.
Proof.
  intros Hl Hs [(ads & ap & Hin & H) | (id & c & Hin & H)];
    [ left; exists ads, ap | right; exists id, c ]; split; auto.
Qed.

(* The code of the walk: code that is not used has a dead [my_closure],
   and the body of code binds nothing static. *)
Definition codes_ok (cs : list (code_id * code)) : Prop :=
  ∀ id c, (id, c) ∈ cs ->
    (is_code_id_used sol id = false ->
     is_dead_var sol (code_my_closure c) Value = true)
    ∧ binds_statics (code_body c) = false.

Lemma codes_ok_sub (cs cs' : list (code_id * code)) :
  (∀ x, x ∈ cs' -> x ∈ cs) -> codes_ok cs -> codes_ok cs'.
Proof. intros Hs H id c Hin. exact (H id c (Hs _ Hin)). Qed.

Lemma codes_ok_l (cs cs' : list (code_id * code)) :
  codes_ok (cs ++ cs') -> codes_ok cs.
Proof.
  apply codes_ok_sub. intros x Hx. apply in_or_app. left. exact Hx.
Qed.

Lemma codes_ok_r (cs cs' : list (code_id * code)) :
  codes_ok (cs ++ cs') -> codes_ok cs'.
Proof.
  apply codes_ok_sub. intros x Hx. apply in_or_app. right. exact Hx.
Qed.

Lemma found_l (cl cl' : list (list decision * apply_expr))
    (cs cs' : list (code_id * code)) :
  found cl cs -> found (cl ++ cl') (cs ++ cs').
Proof.
  apply found_mono; intros x Hx; apply in_or_app; left; exact Hx.
Qed.

Lemma found_r (cl cl' : list (list decision * apply_expr))
    (cs cs' : list (code_id * code)) :
  found cl' cs' -> found (cl ++ cl') (cs ++ cs').
Proof.
  apply found_mono; intros x Hx; apply in_or_app; right; exact Hx.
Qed.

Lemma walk :
  (∀ e : expr, ∀ en,
     codes_ok (unit_codes e) -> fails en e = true ->
     found (calls (ue_decisions en) e) (unit_codes e))
  ∧ (∀ h : continuation_handler, ∀ en dc,
       codes_ok (handler_unit_codes h) -> hfails en dc h = true ->
       found (hcalls (ue_decisions en) h) (handler_unit_codes h)).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH en Hok H.
    cbn [rebuild_fails_u handler_fails_u binding_fails_u
         code_fails_u] in H.
    cbn [rebuilt_calls handler_rebuilt_calls binding_rebuilt_calls
         code_rebuilt_calls unit_codes handler_unit_codes] in Hok |- *.
    pose proof (IH _ Hok H) as Hf.
    match type of Hf with
    | found (calls (ue_decisions ?en') _) _ =>
      replace (ue_decisions en') with (ue_decisions en) in Hf;
        [ exact Hf | ]
    end.
    repeat match goal with
    | |- context [if ?b then _ else _] => destruct b
    | |- context [match ?d with _ => _ end] => destruct d
    end; reflexivity.
  - intros bindings body IHc IH en Hok H.
    rewrite fails_static in H. apply orb_true_iff in H as [H | H].
    + apply existsb_exists in H as (b & Hb & H).
      destruct b as [id c | id | sym const];
        [ rewrite bfails_code in H | discriminate | discriminate ].
      assert (Hc : (id, c) ∈ unit_codes (Let_static bindings body)).
      { cbn [unit_codes]. apply in_or_app. left. apply in_flat_map.
        exists (Bind_code id c). split; [ exact Hb | left; reflexivity ]. }
      destruct (Hok id c Hc) as [Hun Hbs].
      destruct (is_code_id_used sol id) eqn:Hused.
      2:{ rewrite (code_fails_dead id c (Hun eq_refl)) in H. discriminate. }
      pose proof (IHc id c Hb) as IHb.
      destruct c as [params mc ret exn md cbody].
      cbn [code_body] in IHb, Hbs.
      rewrite cfails_code in H. cbv zeta in H.
      apply orb_true_iff in H as [H | H].
      * apply andb_true_iff in H as [_ H].
        assert (Hok' : codes_ok (unit_codes cbody))
          by (rewrite (proj1 no_statics_codes cbody Hbs);
              intros ? ? []).
        pose proof (IHb _ Hok' H) as Hf.
        cbn [ue_decisions ubind_variables ubind_variables_then ubind_set
             ubind_continuations empty_unbox_env] in Hf.
        rewrite app_nil_r, (proj1 no_statics_codes cbody Hbs) in Hf.
        destruct Hf as [(ads & ap & Hin & Hat) | (? & ? & [] & _)].
        left. exists ads, ap. split; [ | exact Hat ].
        cbn [rebuilt_calls handler_rebuilt_calls binding_rebuilt_calls
         code_rebuilt_calls]. apply in_or_app. left. apply in_flat_map.
        exists (Bind_code id (Code params mc ret exn md cbody)).
        split; [ exact Hb | ].
        cbn [rebuilt_calls handler_rebuilt_calls binding_rebuilt_calls
         code_rebuilt_calls]. rewrite Hused.
        exact Hin.
      * right. exists id, (Code params mc ret exn md cbody).
        split; [ exact Hc | split; [ exact Hused | exact H ] ].
    + cbn [unit_codes] in Hok.
      pose proof (IH _ (codes_ok_r _ _ Hok) H) as Hf.
      cbn [ue_decisions ubind_variables ubind_variables_then ubind_set]
        in Hf.
      cbn [rebuilt_calls handler_rebuilt_calls binding_rebuilt_calls
         code_rebuilt_calls unit_codes handler_unit_codes].
      exact (found_r _ _ _ _ Hf).
  - intros k h body IHh IH en Hok H.
    cbn [rebuild_fails_u handler_fails_u binding_fails_u
         code_fails_u] in H. cbv zeta in H.
    cbn [unit_codes] in Hok.
    cbn [rebuilt_calls handler_rebuilt_calls binding_rebuilt_calls
         code_rebuilt_calls unit_codes handler_unit_codes].
    apply orb_true_iff in H as [H | H].
    + pose proof (IH _ (codes_ok_r _ _ Hok) H) as Hf.
      cbn [ue_decisions ubind_continuations app] in Hf.
      exact (found_r _ _ _ _ Hf).
    + apply andb_true_iff in H as [_ H].
      pose proof (IHh _ _ (codes_ok_l _ _ Hok) H) as Hf.
      exact (found_l _ _ _ _ Hf).
  - intros handlers body IHh IH en Hok H.
    rewrite fails_rec in H. cbv zeta in H.
    cbn [unit_codes] in Hok.
    cbn [rebuilt_calls handler_rebuilt_calls binding_rebuilt_calls
         code_rebuilt_calls unit_codes handler_unit_codes]. cbv zeta.
    apply orb_true_iff in H as [H | H].
    + apply existsb_exists in H as ([k h] & Hkh & H).
      assert (Hs : ∀ x, x ∈ handler_unit_codes h ->
                     x ∈ flat_map (fun '(_, h) => handler_unit_codes h)
                           handlers
                         ++ unit_codes body)
        by (intros x Hx; apply in_or_app; left; apply in_flat_map;
            exists (k, h); split; [ exact Hkh | exact Hx ]).
      pose proof (IHh k h Hkh _ _ (codes_ok_sub _ _ Hs Hok) H) as Hf.
      cbn [ue_decisions ubind_continuations] in Hf.
      refine (found_mono _ _ _ _ _ Hs Hf).
      intros x Hx. apply in_or_app. left. apply in_flat_map.
      exists (k, h). split; [ exact Hkh | exact Hx ].
    + pose proof (IH _ (codes_ok_r _ _ Hok) H) as Hf.
      cbn [ue_decisions ubind_continuations] in Hf.
      exact (found_r _ _ _ _ Hf).
  - intros ap en Hok H. cbn [rebuild_fails_u handler_fails_u binding_fails_u
         code_fails_u] in H.
    left. exists (call_decisions (ue_decisions en) ap), ap.
    split; [ left; reflexivity | exact (apply_fails_split en ap H) ].
  - intros ac en _ H. discriminate H.
  - intros scrutinee arms en _ H. discriminate H.
  - intros en _ H. discriminate H.
  - intros params md handler IH en dc Hok H.
    cbn [rebuild_fails_u handler_fails_u binding_fails_u
         code_fails_u] in H. apply andb_true_iff in H as [_ H].
    exact (IH _ Hok H).
Qed.

End Walk.

(* ======================================================================== *)
(* The decisions of a call                                                  *)
(* ======================================================================== *)

Lemma lookup_two {A : Type} (k a b : positive) (x y z : A) :
  lookup k [(a, x); (b, y)] = Some z -> (k = a ∧ z = x) ∨ (k = b ∧ z = y).
Proof.
  cbn [lookup].
  destruct (Pos.eqb k a) eqn:E1;
    [ apply Pos.eqb_eq in E1; intros [= <-]; left; auto | ].
  destruct (Pos.eqb k b) eqn:E2;
    [ apply Pos.eqb_eq in E2; intros [= <-]; right; auto | ].
  discriminate.
Qed.

Section Lengths.

Variable sol : solution.
Variable cc : node -> bool.
Variable cds : finite_map code_dep.
Variable uci : list code_id.
Variable unboxed : list (node * unboxed_fields).

Abbreviation calls := (rebuilt_calls sol cc cds uci unboxed).
Abbreviation hcalls := (handler_rebuilt_calls sol cc cds uci unboxed).

(* The decisions of each continuation of [ds] are no more than its count
   in [ks]. *)
Definition bounded (ds : finite_map (list decision)) (ks : finite_map nat)
    : Prop :=
  ∀ k ads, lookup k ds = Some ads ->
    ∃ m, lookup k ks = Some m ∧ length ads <= m.

Lemma bounded_nil (ks : finite_map nat) : bounded [] ks.
Proof. intros k a H. discriminate H. Qed.

Lemma bounded_cons (ds : finite_map (list decision)) (ks : finite_map nat)
    (k : continuation) (ads : list decision) (n : nat) :
  length ads <= n -> bounded ds ks -> bounded ((k, ads) :: ds) ((k, n) :: ks).
Proof.
  intros Hn Hb k' a H. cbn [lookup] in H |- *.
  destruct (Pos.eqb k' k).
  - injection H as <-. exists n. split; [ reflexivity | exact Hn ].
  - exact (Hb k' a H).
Qed.

Lemma bounded_app (hs : list (continuation * continuation_handler))
    (f : continuation_handler -> list decision)
    (g : continuation_handler -> nat) (ds : finite_map (list decision))
    (ks : finite_map nat) :
  (∀ h, length (f h) <= g h) -> bounded ds ks ->
  bounded (map (fun '(k, h) => (k, f h)) hs ++ ds)
    (map (fun '(k, h) => (k, g h)) hs ++ ks).
Proof.
  intros Hfg Hb. induction hs as [ | [k h] hs IH]; [ exact Hb | ].
  exact (bounded_cons _ _ k _ _ (Hfg h) IH).
Qed.

Lemma handler_decisions_length (b : bool) (h : continuation_handler) :
  length (handler_decisions true sol unboxed b h)
  <= length (FlambdaReaperTraversal.handler_params h).
Proof.
  destruct h as [params md body].
  cbn [handler_decisions FlambdaReaperTraversal.handler_params].
  rewrite !length_map, length_combine,
    FlambdaReaperWrapperProofs.with_kinds_length.
  lia.
Qed.

Lemma code_decisions_bounded (id : code_id) (md : code_metadata)
    (ret exn : continuation) :
  ret <> exn ->
  bounded
    (code_continuation_decisions sol cds unboxed id md
       (code_change_u sol cc cds uci unboxed id md) ret exn)
    [(exn, 1); (ret, length (unarize (cm_result_arity md)))].
Proof.
  intros Hne k ads Hk.
  set (n := length (unarize (cm_result_arity md))).
  assert (Hr : lookup ret [(exn, 1); (ret, n)] = Some n)
    by (cbn [lookup]; destruct (Pos.eqb ret exn) eqn:E;
        [ apply Pos.eqb_eq in E; contradiction
        | rewrite Pos.eqb_refl; reflexivity ]).
  assert (He : lookup exn [(exn, 1); (ret, n)] = Some 1)
    by (cbn [lookup]; rewrite Pos.eqb_refl; reflexivity).
  unfold code_continuation_decisions, code_change_u in Hk.
  destruct (cannot_change_calling_convention _ _ _);
    destruct (lookup id cds) as [cd | ]; cbv beta iota zeta in Hk;
    try discriminate Hk; cbn [app] in Hk;
    destruct (lookup_two _ _ _ _ _ _ Hk) as [[-> ->] | [-> ->]];
    first [ exists n; split; [ exact Hr | ]
          | exists 1; split; [ exact He | ] ];
    unfold fixed_arity_decisions, decisions_u;
    rewrite length_map, length_combine; cbn [length]; lia.
Qed.

(* The code of the unit: its two continuations distinct, its body
   counted, and binding nothing static. *)
Definition code_counted (c : code) : Prop :=
  match c with
  | Code _ _ ret exn md body =>
    ret <> exn
    ∧ conts_counted [(exn, 1); (ret, length (unarize (cm_result_arity md)))]
        body
      = true
    ∧ binds_statics body = false
  end.

Lemma calls_bounded :
  (∀ e : expr, ∀ ds ks,
     bounded ds ks -> conts_counted ks e = true ->
     (∀ id c, (id, c) ∈ unit_codes e -> code_counted c) ->
     ∀ ads ap, (ads, ap) ∈ calls ds e ->
       length ads <= length (unarize (ap_return_arity ap)))
  ∧ (∀ h : continuation_handler, ∀ ds ks,
       bounded ds ks -> handler_conts_counted ks h = true ->
       (∀ id c, (id, c) ∈ handler_unit_codes h -> code_counted c) ->
       ∀ ads ap, (ads, ap) ∈ hcalls ds h ->
         length ads <= length (unarize (ap_return_arity ap))).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH ds ks Hb Hc Hcodes ads ap Hin.
    exact (IH ds ks Hb Hc Hcodes ads ap Hin).
  - intros bindings body IHc IH ds ks Hb Hc Hcodes ads ap Hin.
    cbn [unit_codes] in Hcodes.
    cbn [rebuilt_calls handler_rebuilt_calls binding_rebuilt_calls
         code_rebuilt_calls] in Hin.
    apply in_app_or in Hin as [Hin | Hin].
    2:{ refine (IH ds ks Hb Hc _ ads ap Hin).
        intros id c H. apply (Hcodes id c). apply in_or_app. right. exact H. }
    apply in_flat_map in Hin as (b & Hbin & Hin).
    destruct b as [id c | id | sym const]; cbn in Hin;
      [ | destruct Hin | destruct Hin ].
    destruct (is_code_id_used sol id); [ | destruct Hin ].
    assert (Hcu : (id, c) ∈ flat_map
                    (fun binding =>
                       match binding with
                       | Bind_code id c => [(id, c)]
                       | Bind_deleted_code _ | Bind_symbol _ _ => []
                       end)
                    bindings ++ unit_codes body)
      by (apply in_or_app; left; apply in_flat_map;
          exists (Bind_code id c); split; [ exact Hbin | left; reflexivity ]).
    pose proof (Hcodes id c Hcu) as Hcc.
    pose proof (IHc id c Hbin) as IHb.
    destruct c as [params mc ret exn md cbody].
    destruct Hcc as (Hne & Hcnt & Hbs). cbn [code_body] in IHb.
    refine (IHb _ _ (code_decisions_bounded id md ret exn Hne) Hcnt _ ads ap
              Hin).
    rewrite (proj1 no_statics_codes cbody Hbs). intros ? ? [].
  - intros k h body IHh IH ds ks Hb Hc Hcodes ads ap Hin.
    cbn [conts_counted] in Hc. apply andb_true_iff in Hc as [Hch Hc].
    cbn [unit_codes] in Hcodes.
    cbn [rebuilt_calls handler_rebuilt_calls] in Hin.
    apply in_app_or in Hin as [Hin | Hin].
    + refine (IHh ds ks Hb Hch _ ads ap Hin).
      intros id c H. apply (Hcodes id c). apply in_or_app. left. exact H.
    + refine (IH _ _ (bounded_cons _ _ k _ _ (handler_decisions_length _ h)
                        Hb)
                Hc _ ads ap Hin).
      intros id c H. apply (Hcodes id c). apply in_or_app. right. exact H.
  - intros handlers body IHh IH ds ks Hb Hc Hcodes ads ap Hin.
    cbn [conts_counted] in Hc. apply andb_true_iff in Hc as [Hch Hc].
    cbn [unit_codes] in Hcodes.
    cbn [rebuilt_calls handler_rebuilt_calls] in Hin. cbv zeta in Hin.
    pose proof (bounded_app handlers (handler_decisions true sol unboxed false)
                  (fun h => length (FlambdaReaperTraversal.handler_params h))
                  ds ks (handler_decisions_length false) Hb) as Hb'.
    apply in_app_or in Hin as [Hin | Hin].
    + apply in_flat_map in Hin as ([k h] & Hkh & Hin).
      rewrite forallb_forall in Hch. pose proof (Hch (k, h) Hkh) as Hch1.
      refine (IHh k h Hkh _ _ Hb' Hch1 _ ads ap Hin).
      intros id c H. apply (Hcodes id c). apply in_or_app. left.
      apply in_flat_map. exists (k, h). split; [ exact Hkh | exact H ].
    + refine (IH _ _ Hb' Hc _ ads ap Hin).
      intros id c H. apply (Hcodes id c). apply in_or_app. right. exact H.
  - intros ap0 ds ks Hb Hc _ ads ap Hin.
    cbn [rebuilt_calls] in Hin. destruct Hin as [[= <- <-] | []].
    cbn [conts_counted] in Hc. apply andb_true_iff in Hc as [Hc _].
    unfold count_ok in Hc.
    destruct (lookup (ap_continuation ap0) ks) as [m | ] eqn:Ek;
      [ | discriminate ].
    apply Nat.eqb_eq in Hc.
    destruct (lookup (ap_continuation ap0) ds) as [d | ] eqn:Ed;
      [ | cbn [length]; lia ].
    destruct (Hb _ _ Ed) as (m' & Hm & Hle). rewrite Ek in Hm.
    injection Hm as <-. lia.
  - intros ac ds ks _ _ _ ads ap [].
  - intros scrutinee arms ds ks _ _ _ ads ap [].
  - intros ds ks _ _ _ ads ap [].
  - intros params md handler IH ds ks Hb Hc Hcodes ads ap Hin.
    exact (IH ds ks Hb Hc Hcodes ads ap Hin).
Qed.

(* The calls of the rebuild are calls of the term. *)
Lemma rebuilt_calls_of :
  (∀ e : expr, ∀ ds ads ap, (ads, ap) ∈ calls ds e -> ap ∈ calls_of e)
  ∧ (∀ h : continuation_handler, ∀ ds ads ap,
       (ads, ap) ∈ hcalls ds h -> ap ∈ handler_calls_of h).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH ds ads ap Hin. exact (IH ds ads ap Hin).
  - intros bindings body IHc IH ds ads ap Hin.
    cbn [rebuilt_calls handler_rebuilt_calls binding_rebuilt_calls
         code_rebuilt_calls] in Hin.
    cbn [calls_of handler_calls_of binding_calls_of code_calls_of].
    apply in_app_or in Hin as [Hin | Hin];
      [ | apply in_or_app; right; exact (IH ds ads ap Hin) ].
    apply in_flat_map in Hin as (b & Hbin & Hin).
    destruct b as [id c | id | sym const]; cbn in Hin;
      [ | destruct Hin | destruct Hin ].
    destruct (is_code_id_used sol id); [ | destruct Hin ].
    apply in_or_app. left. apply in_flat_map.
    exists (Bind_code id c). split; [ exact Hbin | ].
    pose proof (IHc id c Hbin) as IHb.
    destruct c as [params mc ret exn md cbody]. cbn [code_body] in IHb.
    exact (IHb _ ads ap Hin).
  - intros k h body IHh IH ds ads ap Hin.
    cbn [rebuilt_calls handler_rebuilt_calls] in Hin.
    cbn [calls_of handler_calls_of].
    apply in_app_or in Hin as [Hin | Hin]; apply in_or_app;
      [ left; exact (IHh ds ads ap Hin) | right; exact (IH _ ads ap Hin) ].
  - intros handlers body IHh IH ds ads ap Hin.
    cbn [rebuilt_calls handler_rebuilt_calls] in Hin. cbv zeta in Hin.
    cbn [calls_of handler_calls_of].
    apply in_app_or in Hin as [Hin | Hin]; apply in_or_app;
      [ left | right; exact (IH _ ads ap Hin) ].
    apply in_flat_map in Hin as ([k h] & Hkh & Hin).
    apply in_flat_map. exists (k, h).
    split; [ exact Hkh | exact (IHh k h Hkh _ ads ap Hin) ].
  - intros ap0 ds ads ap Hin. cbn [rebuilt_calls] in Hin.
    destruct Hin as [[= <- <-] | []]. left. reflexivity.
  - intros ac ds ads ap [].
  - intros scrutinee arms ds ads ap [].
  - intros ds ads ap [].
  - intros params md handler IH ds ads ap Hin. exact (IH ds ads ap Hin).
Qed.

End Lengths.

(* ======================================================================== *)
(* The statement                                                            *)
(* ======================================================================== *)

Theorem fix_fails_only_by_unboxed_callee :
  fix_fails_only_by_unboxed_callee_statement.
Proof.
  intros is_local u e T max_unbox_size lambda_lifting pdc tcr F S code_deps
    cannot_change dominated to_unbox unboxed Hloc Hslots Hfrag Hleast Hcount
    Hkinds Hwf Hone Hres Hdead Hfails.
  destruct (FlambdaReaperWrapperProofs.unit_facts Auto u e
              (∀ f : field, is_local f = false) Hfrag Hslots)
    as (noted & deps & Hedges & Hconsts & _ & Hnodup & Hids).
  pose proof (fun id c Hc =>
                proj1 (FlambdaReaperWrapperProofs.unit_codes_edges
                         _ _ _ _ _ _ _ _ _)
                  e None (unit_conts u) Hedges id c Hc) as Hcode_edges.
  (* The calls and the code of the unit. *)
  assert (Hcalls_of :
            ∀ ads ap,
              (ads, ap) ∈ unit_rebuilt_calls T cannot_change code_deps unboxed e
              -> ap ∈ calls_of e)
    by (intros ads ap H;
        exact (proj1 (rebuilt_calls_of _ _ _ _ _) e [] ads ap H)).
  assert (Hgroup :
            ∀ ap, ap ∈ calls_of e -> ap_call_kind ap = Indirect_unknown_arity ->
              length (ap_args_arity ap) = 1).
  { intros ap Hap Hk. unfold one_group_unknown_calls in Hone.
    rewrite forallb_forall in Hone. pose proof (Hone ap Hap) as H.
    rewrite Hk in H. apply Nat.eqb_eq in H. exact H. }
  assert (Har :
            ∀ ap, ap ∈ calls_of e ->
              ∀ id cd, code_id_actually_called F S ap = Some id ->
                (id, cd) ∈ code_deps ->
                length (unarize (ap_return_arity ap)) <= length (cd_return cd)).
  { intros ap Hap id cd Hid Hcd.
    assert (Hk : ap_call_kind ap <> Indirect_unknown_arity).
    { intros Hk. unfold code_id_actually_called in Hid. rewrite Hk in Hid.
      discriminate. }
    unfold results_within_codes in Hres. rewrite forallb_forall in Hres.
    pose proof (Hres ap Hap) as H.
    assert (Hfa :
              forallb
                (fun '(_, c) =>
                   Nat.leb (length (unarize (ap_return_arity ap)))
                     (length (code_result_kinds c)))
                (unit_codes e)
              = true)
      by (revert H; destruct (ap_call_kind ap) eqn:Ek; intros H;
          first [ exact H | exfalso; exact (Hk eq_refl) ]).
    rewrite forallb_forall in Hfa.
    assert (Hidin : id ∈ static_names code_ids_of_binding e)
      by (apply Hids; apply in_map_iff; exists (id, cd); auto).
    destruct (proj1 FlambdaReaperWrapperProofs.unit_codes_names e id Hidin)
      as [(c & Hc) _].
    pose proof (Hfa (id, c) Hc) as Hle. cbv beta iota in Hle.
    apply Nat.leb_le in Hle.
    pose proof (Hcode_edges id c Hc) as Hce.
    destruct c as [params mc ret exn md body].
    cbn [FlambdaReaperFlowProofs.code_edges] in Hce.
    destruct Hce as (cd' & Hcd' & _ & _ & _ & _ & Hlen & _).
    pose proof (FlambdaReaperFlowProofs.lookup_nodup_in _ _ _ Hnodup Hcd)
      as E1.
    rewrite (FlambdaReaperFlowProofs.lookup_nodup_in _ _ _ Hnodup Hcd') in E1.
    injection E1 as <-.
    cbn [code_result_kinds code_result_arity] in Hle. rewrite Hlen.
    exact Hle. }
  assert (Hcalls :
            ∀ ads ap,
              (ads, ap) ∈ unit_rebuilt_calls T cannot_change code_deps unboxed e
              -> (ap_call_kind ap = Indirect_unknown_arity ->
                  ap_args_arity ap <> [])
                 ∧ ∀ id cd, code_id_actually_called F S ap = Some id ->
                     (id, cd) ∈ code_deps ->
                     length (unarize (ap_return_arity ap))
                     <= length (cd_return cd)).
  { intros ads ap H. pose proof (Hcalls_of ads ap H) as Hap. split.
    - intros Hk E. pose proof (Hgroup ap Hap Hk) as H1. rewrite E in H1.
      discriminate.
    - exact (Har ap Hap). }
  assert (Hcounted : ∀ id c, (id, c) ∈ unit_codes e -> code_counted c).
  { intros id c Hc. pose proof (Hcode_edges id c Hc) as Hce.
    destruct c as [params mc ret exn md body].
    cbn [FlambdaReaperFlowProofs.code_edges] in Hce.
    destruct Hce as (cd & _ & _ & _ & _ & _ & _ & _ & _ & Hne & Hbs & Hcnt
                     & _).
    split; [ exact Hne | split; [ exact Hcnt | exact Hbs ] ]. }
  assert (Hok : codes_ok S (unit_codes e)).
  { intros id c Hc. split.
    - intros Hun. unfold unused_code_dead in Hdead.
      rewrite forallb_forall in Hdead. pose proof (Hdead (id, c) Hc) as H.
      cbv beta iota in H. rewrite Hun in H. exact H.
    - pose proof (Hcounted id c Hc) as H. destruct c.
      exact (proj2 (proj2 H)). }
  (* The walk. *)
  unfold rebuild_unit_unbox_repr_fixed_fails in Hfails.
  assert (Eu : unit_unboxed_fields F T dominated to_unbox e
               = (unboxed, snd (unit_unboxed_fields F T dominated to_unbox e)))
    by apply surjective_pairing.
  rewrite Eu in Hfails. cbv beta iota in Hfails.
  set (next := snd (unit_unboxed_fields F T dominated to_unbox e)) in Hfails.
  set (changed :=
         unit_changed_representations is_local F T dominated to_unbox tcr e)
    in Hfails.
  pose proof (proj1 (walk F S (fun n => mem_node n cannot_change) code_deps
                       pdc (static_names code_ids_bound e)
                       (bound_code_metadata e) unboxed next changed)
                e empty_unbox_env Hok Hfails)
    as Hfound.
  destruct Hfound as [(ads & ap & Hin & Hat) | (id & c & Hc & Hused & Hnr)].
  - change ((ads, ap) ∈ unit_rebuilt_calls T cannot_change code_deps unboxed e)
      in Hin.
    destruct Hat as [(Hnc & fields & Hf) | (Hnr & Hw)].
    + exists ads, ap, fields. split; [ exact Hin | split; assumption ].
    + exfalso.
      pose proof (proj1 (calls_bounded _ _ _ _ _) e [] (unit_counts u)
                    (bounded_nil _) Hcount Hcounted ads ap Hin) as Hlen.
      assert (Hw' :
                wrapper_fails_u ads
                  (map fst
                     (unit_callee_decisions F T cannot_change code_deps
                        unboxed e ap))
                = true).
      { unfold unit_callee_decisions, callee_decisions.
        unfold callee_returns in Hw. change (solution_of_tables T) with S.
        destruct (apply_change_u _ _ _ _ _ _ _ ap) as [ | pd rd].
        - rewrite map_map. exact Hw.
        - exact (wrapper_fails_combine ads rd _ Hlen Hw). }
      pose proof (Hcalls_of ads ap Hin) as Hap.
      assert (Hkind :
                ap_call_kind ap <> Indirect_unknown_arity
                ∨ length (ap_args_arity ap) = 1)
        by (destruct (ap_call_kind ap) eqn:Ek;
            first [ right; exact (Hgroup ap Hap Ek) | left; discriminate ]).
      pose proof (FlambdaReaperWrapperDerivedProofs
                    .fix_arm_never_returns_auto_derived is_local u e T
                    max_unbox_size lambda_lifting Hloc Hslots Hfrag Hleast
                    Hcount Hkinds ads ap Hin Hkind (Har ap Hap) Hw')
        as Hnr'.
      discriminate (eq_trans (eq_sym Hnr') Hnr).
  - exfalso.
    pose proof (FlambdaReaperWrapperDerivedProofs.check_never_returns_auto
                  is_local u e T cannot_change unboxed next changed pdc
                  Hloc Hslots
                  Hfrag Hleast Hwf Hcalls id c Hc Hused) as Hcnr.
    discriminate (eq_trans (eq_sym Hnr) Hcnr).
Qed.

Theorem fix_fails_only_by_unboxed_callee_any_groups :
  fix_fails_only_by_unboxed_callee_any_groups_statement.
Proof.
  intros is_local u e T max_unbox_size lambda_lifting pdc tcr F S code_deps
    cannot_change dominated to_unbox unboxed Hloc Hslots Hfrag Hleast Hcount
    Hkinds Hwf Hhas Hres Hdead Hfails.
  destruct (FlambdaReaperWrapperProofs.unit_facts Auto u e
              (∀ f : field, is_local f = false) Hfrag Hslots)
    as (noted & deps & Hedges & Hconsts & _ & Hnodup & Hids).
  pose proof (fun id c Hc =>
                proj1 (FlambdaReaperWrapperProofs.unit_codes_edges
                         _ _ _ _ _ _ _ _ _)
                  e None (unit_conts u) Hedges id c Hc) as Hcode_edges.
  (* The calls and the code of the unit. *)
  assert (Hcalls_of :
            ∀ ads ap,
              (ads, ap) ∈ unit_rebuilt_calls T cannot_change code_deps unboxed e
              -> ap ∈ calls_of e)
    by (intros ads ap H;
        exact (proj1 (rebuilt_calls_of _ _ _ _ _) e [] ads ap H)).
  assert (Hnil :
            ∀ ap, ap ∈ calls_of e -> ap_call_kind ap = Indirect_unknown_arity ->
              ap_args_arity ap <> []).
  { intros ap Hap Hk E. unfold unknown_calls_have_args in Hhas.
    rewrite forallb_forall in Hhas. pose proof (Hhas ap Hap) as H.
    rewrite Hk, E in H. discriminate. }
  assert (Har :
            ∀ ap, ap ∈ calls_of e ->
              ∀ id cd, code_id_actually_called F S ap = Some id ->
                (id, cd) ∈ code_deps ->
                length (unarize (ap_return_arity ap)) <= length (cd_return cd)).
  { intros ap Hap id cd Hid Hcd.
    assert (Hk : ap_call_kind ap <> Indirect_unknown_arity).
    { intros Hk. unfold code_id_actually_called in Hid. rewrite Hk in Hid.
      discriminate. }
    unfold results_within_codes in Hres. rewrite forallb_forall in Hres.
    pose proof (Hres ap Hap) as H.
    assert (Hfa :
              forallb
                (fun '(_, c) =>
                   Nat.leb (length (unarize (ap_return_arity ap)))
                     (length (code_result_kinds c)))
                (unit_codes e)
              = true)
      by (revert H; destruct (ap_call_kind ap) eqn:Ek; intros H;
          first [ exact H | exfalso; exact (Hk eq_refl) ]).
    rewrite forallb_forall in Hfa.
    assert (Hidin : id ∈ static_names code_ids_of_binding e)
      by (apply Hids; apply in_map_iff; exists (id, cd); auto).
    destruct (proj1 FlambdaReaperWrapperProofs.unit_codes_names e id Hidin)
      as [(c & Hc) _].
    pose proof (Hfa (id, c) Hc) as Hle. cbv beta iota in Hle.
    apply Nat.leb_le in Hle.
    pose proof (Hcode_edges id c Hc) as Hce.
    destruct c as [params mc ret exn md body].
    cbn [FlambdaReaperFlowProofs.code_edges] in Hce.
    destruct Hce as (cd' & Hcd' & _ & _ & _ & _ & Hlen & _).
    pose proof (FlambdaReaperFlowProofs.lookup_nodup_in _ _ _ Hnodup Hcd)
      as E1.
    rewrite (FlambdaReaperFlowProofs.lookup_nodup_in _ _ _ Hnodup Hcd') in E1.
    injection E1 as <-.
    cbn [code_result_kinds code_result_arity] in Hle. rewrite Hlen.
    exact Hle. }
  assert (Hcalls :
            ∀ ads ap,
              (ads, ap) ∈ unit_rebuilt_calls T cannot_change code_deps unboxed e
              -> (ap_call_kind ap = Indirect_unknown_arity ->
                  ap_args_arity ap <> [])
                 ∧ ∀ id cd, code_id_actually_called F S ap = Some id ->
                     (id, cd) ∈ code_deps ->
                     length (unarize (ap_return_arity ap))
                     <= length (cd_return cd)).
  { intros ads ap H. pose proof (Hcalls_of ads ap H) as Hap. split.
    - exact (Hnil ap Hap).
    - exact (Har ap Hap). }
  assert (Hcounted : ∀ id c, (id, c) ∈ unit_codes e -> code_counted c).
  { intros id c Hc. pose proof (Hcode_edges id c Hc) as Hce.
    destruct c as [params mc ret exn md body].
    cbn [FlambdaReaperFlowProofs.code_edges] in Hce.
    destruct Hce as (cd & _ & _ & _ & _ & _ & _ & _ & _ & Hne & Hbs & Hcnt
                     & _).
    split; [ exact Hne | split; [ exact Hcnt | exact Hbs ] ]. }
  assert (Hok : codes_ok S (unit_codes e)).
  { intros id c Hc. split.
    - intros Hun. unfold unused_code_dead in Hdead.
      rewrite forallb_forall in Hdead. pose proof (Hdead (id, c) Hc) as H.
      cbv beta iota in H. rewrite Hun in H. exact H.
    - pose proof (Hcounted id c Hc) as H. destruct c.
      exact (proj2 (proj2 H)). }
  (* The walk. *)
  unfold rebuild_unit_unbox_repr_fixed_fails in Hfails.
  assert (Eu : unit_unboxed_fields F T dominated to_unbox e
               = (unboxed, snd (unit_unboxed_fields F T dominated to_unbox e)))
    by apply surjective_pairing.
  rewrite Eu in Hfails. cbv beta iota in Hfails.
  set (next := snd (unit_unboxed_fields F T dominated to_unbox e)) in Hfails.
  set (changed :=
         unit_changed_representations is_local F T dominated to_unbox tcr e)
    in Hfails.
  pose proof (proj1 (walk F S (fun n => mem_node n cannot_change) code_deps
                       pdc (static_names code_ids_bound e)
                       (bound_code_metadata e) unboxed next changed)
                e empty_unbox_env Hok Hfails)
    as Hfound.
  destruct Hfound as [(ads & ap & Hin & Hat) | (id & c & Hc & Hused & Hnr)].
  - change ((ads, ap) ∈ unit_rebuilt_calls T cannot_change code_deps unboxed e)
      in Hin.
    destruct Hat as [(Hnc & fields & Hf) | (Hnr & Hw)].
    + exists ads, ap, fields. split; [ exact Hin | split; assumption ].
    + exfalso.
      pose proof (proj1 (calls_bounded _ _ _ _ _) e [] (unit_counts u)
                    (bounded_nil _) Hcount Hcounted ads ap Hin) as Hlen.
      assert (Hw' :
                wrapper_fails_u ads
                  (map fst
                     (unit_callee_decisions F T cannot_change code_deps
                        unboxed e ap))
                = true).
      { unfold unit_callee_decisions, callee_decisions.
        unfold callee_returns in Hw. change (solution_of_tables T) with S.
        destruct (apply_change_u _ _ _ _ _ _ _ ap) as [ | pd rd].
        - rewrite map_map. exact Hw.
        - exact (wrapper_fails_combine ads rd _ Hlen Hw). }
      pose proof (Hcalls_of ads ap Hin) as Hap.
      pose proof (FlambdaReaperWrapperDerivedProofs
                    .fix_arm_never_returns_auto_derived_full is_local u e T
                    max_unbox_size lambda_lifting Hloc Hslots Hfrag Hleast
                    Hcount Hkinds ads ap Hin (Hnil ap Hap) (Har ap Hap) Hw')
        as Hnr'.
      discriminate (eq_trans (eq_sym Hnr') Hnr).
  - exfalso.
    pose proof (FlambdaReaperWrapperDerivedProofs.check_never_returns_auto
                  is_local u e T cannot_change unboxed next changed pdc
                  Hloc Hslots
                  Hfrag Hleast Hwf Hcalls id c Hc Hused) as Hcnr.
    discriminate (eq_trans (eq_sym Hnr) Hcnr).
Qed.

Theorem one_group_unknown_calls_have_args :
  one_group_unknown_calls_have_args_statement.
Proof.
  intros e H. unfold one_group_unknown_calls in H.
  unfold unknown_calls_have_args. rewrite forallb_forall in H |- *.
  intros ap Hap. specialize (H ap Hap). cbv beta in H |- *.
  destruct (ap_call_kind ap), (ap_args_arity ap); try reflexivity.
  discriminate H.
Qed.

Theorem unboxed_callee_meets_premises :
  unboxed_callee_meets_premises_statement.
Proof. hnf. repeat split; vm_compute; reflexivity. Qed.
