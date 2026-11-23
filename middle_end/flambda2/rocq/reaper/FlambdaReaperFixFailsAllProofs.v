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

(* The proofs of FlambdaReaperFixFailsAllSpec.v. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaWellFormedSpec FlambdaReaperGraphSpec
  FlambdaReaperTraversal FlambdaReaperSolution FlambdaReaperRebuild
  FlambdaReaperCallingConventions FlambdaReaperUnboxing FlambdaReaperLeast
  FlambdaReaperFlowSpec FlambdaReaperLeastSpec FlambdaReaperWrapperSpec
  FlambdaReaperWrapperDerivedSpec FlambdaReaperFixFailsSpec
  FlambdaReaperFixFailsAllSpec FlambdaSemanticsSpec.
From OxCamlFlambda2 Require FlambdaReaperWrapperProofs
  FlambdaReaperFlowProofs FlambdaReaperFixFailsProofs
  FlambdaReaperLeastProofs.
From OxCamlFlambda2 Require Import FlambdaBindersSpec.
Import ListNotations.

(* ======================================================================== *)
(* The uses fit                                                             *)
(* ======================================================================== *)

Section Fit.

Variable sol : solution.
Variable cc : node -> bool.
Variable cds : finite_map code_dep.
Variable uci : list code_id.
Variable unboxed : list (node * unboxed_fields).
Variable counts : finite_map nat.

Abbreviation uses := (cont_uses sol cc cds uci unboxed).
Abbreviation huses := (handler_cont_uses sol cc cds uci unboxed).

(* Each continuation counted in [ks] has as many decisions in [ds], or
   none, and that count in [counts]. *)
Definition fit_env (ds : finite_map (list decision)) (ks : finite_map nat)
    : Prop :=
  ∀ k n, lookup k ks = Some n ->
    (∃ d, lookup k ds = Some d ∧ length d = n)
    ∨ (lookup k ds = None ∧ lookup k counts = Some n).

Lemma fit_env_top : fit_env [] counts.
Proof. intros k n H. right. split; [ reflexivity | exact H ]. Qed.

Lemma fit_env_cons (ds : finite_map (list decision)) (ks : finite_map nat)
    (k : continuation) (d : list decision) :
  fit_env ds ks -> fit_env ((k, d) :: ds) ((k, length d) :: ks).
Proof.
  intros Hf k' n H. cbn [lookup] in H |- *.
  destruct (Pos.eqb k' k).
  - injection H as <-. left. exists d. split; reflexivity.
  - exact (Hf k' n H).
Qed.

Lemma fit_env_app (hs : list (continuation * continuation_handler))
    (f : continuation_handler -> list decision)
    (g : continuation_handler -> nat) (ds : finite_map (list decision))
    (ks : finite_map nat) :
  (∀ h, length (f h) = g h) -> fit_env ds ks ->
  fit_env (map (fun '(k, h) => (k, f h)) hs ++ ds)
    (map (fun '(k, h) => (k, g h)) hs ++ ks).
Proof.
  intros Hfg Hf. induction hs as [ | [k h] hs IH]; [ exact Hf | ].
  cbn [map app]. rewrite <- (Hfg h). exact (fit_env_cons _ _ k _ IH).
Qed.

Lemma handler_decisions_eq_length (b : bool) (h : continuation_handler) :
  length (handler_decisions true sol unboxed b h)
  = length (FlambdaReaperTraversal.handler_params h).
Proof.
  destruct h as [params md body].
  cbn [handler_decisions FlambdaReaperTraversal.handler_params].
  unfold params_to_keep.
  rewrite !length_map, length_combine, length_map,
    FlambdaReaperWrapperProofs.with_kinds_length.
  apply Nat.min_id.
Qed.

Lemma use_fit_of_count (ds : finite_map (list decision))
    (ks : finite_map nat) (k : continuation) (n : nat) :
  fit_env ds ks -> count_ok ks k n = true ->
  use_fits counts (k, lookup k ds, n) = true.
Proof.
  intros Hf Hc. unfold count_ok in Hc.
  destruct (lookup k ks) as [m | ] eqn:Ek; [ | discriminate ].
  apply Nat.eqb_eq in Hc. subst m.
  destruct (Hf k n Ek) as [(d & Ed & Hl) | (Ed & Ec)]; rewrite Ed;
    cbn [use_fits].
  - apply Nat.eqb_eq. exact Hl.
  - unfold count_ok. rewrite Ec. apply Nat.eqb_refl.
Qed.

(* Code of the unit: its continuations distinct, its body counted and
   binding nothing static, and a code dep with as many results. *)
Definition code_fits (id : code_id) (c : code) : Prop :=
  match c with
  | Code _ _ ret exn md body =>
    ret <> exn
    ∧ conts_counted [(exn, 1); (ret, length (unarize (cm_result_arity md)))]
        body
      = true
    ∧ binds_statics body = false
    ∧ ∃ cd, lookup id cds = Some cd
            ∧ length (cd_return cd) = length (unarize (cm_result_arity md))
  end.

Lemma code_decisions_fit (id : code_id) (md : code_metadata)
    (ret exn : continuation) (cd : code_dep) :
  ret <> exn -> lookup id cds = Some cd ->
  length (cd_return cd) = length (unarize (cm_result_arity md)) ->
  fit_env
    (code_continuation_decisions sol cds unboxed id md
       (code_change_u sol cc cds uci unboxed id md) ret exn)
    [(exn, 1); (ret, length (unarize (cm_result_arity md)))].
Proof.
  intros Hne Hcd Hlen k n Hk. left.
  assert (E : Pos.eqb exn ret = false)
    by (apply Pos.eqb_neq; intros ->; exact (Hne eq_refl)).
  unfold code_continuation_decisions, code_change_u.
  rewrite Hcd.
  destruct (FlambdaReaperFixFailsProofs.lookup_two _ _ _ _ _ _ Hk)
    as [[-> ->] | [-> ->]];
    destruct (cannot_change_calling_convention _ _ _);
    cbn [app lookup]; rewrite ?Pos.eqb_refl, ?E;
    eexists; (split; [ reflexivity | ]);
    unfold fixed_arity_decisions, decisions_u;
    rewrite length_map, length_combine; cbn [length]; lia.
Qed.

Lemma uses_fit_app (l l' : list (continuation * option (list decision) * nat))
    : uses_fit counts (l ++ l') = uses_fit counts l && uses_fit counts l'.
Proof. apply forallb_app. Qed.

Lemma uses_fit_walk :
  (∀ e : expr, ∀ ds ks,
     fit_env ds ks -> conts_counted ks e = true ->
     (∀ id c, (id, c) ∈ unit_codes e -> code_fits id c) ->
     uses_fit counts (uses ds e) = true)
  ∧ (∀ h : continuation_handler, ∀ ds ks,
       fit_env ds ks -> handler_conts_counted ks h = true ->
       (∀ id c, (id, c) ∈ handler_unit_codes h -> code_fits id c) ->
       uses_fit counts (huses ds h) = true).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH ds ks Hf Hc Hcodes.
    exact (IH ds ks Hf Hc Hcodes).
  - intros bindings body IHc IH ds ks Hf Hc Hcodes.
    cbn [unit_codes] in Hcodes. cbn [cont_uses].
    rewrite uses_fit_app. apply andb_true_iff. split.
    2:{ refine (IH ds ks Hf Hc _).
        intros id c H. apply (Hcodes id c). apply in_or_app. right.
        exact H. }
    unfold uses_fit. rewrite forallb_forall. intros use Hu.
    apply in_flat_map in Hu as (b & Hb & Hu).
    destruct b as [id c | id | sym const]; cbn [binding_cont_uses] in Hu;
      [ | destruct Hu | destruct Hu ].
    assert (Hcu : (id, c) ∈ flat_map
                    (fun binding =>
                       match binding with
                       | Bind_code id c => [(id, c)]
                       | Bind_deleted_code _ | Bind_symbol _ _ => []
                       end)
                    bindings ++ unit_codes body)
      by (apply in_or_app; left; apply in_flat_map;
          exists (Bind_code id c); split; [ exact Hb | left; reflexivity ]).
    pose proof (Hcodes id c Hcu) as Hcf.
    pose proof (IHc id c Hb) as IHb.
    destruct c as [params mc ret exn md cbody].
    cbn [code_body] in IHb. cbn [code_cont_uses] in Hu.
    destruct Hcf as (Hne & Hcnt & Hbs & cd & Hcd & Hlen).
    assert (H : uses_fit counts
                  (uses (code_continuation_decisions sol cds unboxed id md
                           (code_change_u sol cc cds uci unboxed id md)
                           ret exn)
                     cbody)
                = true).
    { refine (IHb _ _ (code_decisions_fit id md ret exn cd Hne Hcd Hlen)
                Hcnt _).
      rewrite (proj1 FlambdaReaperFixFailsProofs.no_statics_codes cbody Hbs).
      intros ? ? []. }
    unfold uses_fit in H. rewrite forallb_forall in H. exact (H use Hu).
  - intros k h body IHh IH ds ks Hf Hc Hcodes.
    cbn [conts_counted] in Hc. apply andb_true_iff in Hc as [Hch Hc].
    cbn [unit_codes] in Hcodes. cbn [cont_uses].
    rewrite uses_fit_app. apply andb_true_iff. split.
    + refine (IHh ds ks Hf Hch _).
      intros id c H. apply (Hcodes id c). apply in_or_app. left. exact H.
    + refine (IH _ _ _ Hc _).
      * rewrite <- (handler_decisions_eq_length (handler_is_exn_handler h) h).
        exact (fit_env_cons _ _ k _ Hf).
      * intros id c H. apply (Hcodes id c). apply in_or_app. right.
        exact H.
  - intros handlers body IHh IH ds ks Hf Hc Hcodes.
    cbn [conts_counted] in Hc. apply andb_true_iff in Hc as [Hch Hc].
    cbn [unit_codes] in Hcodes. cbn [cont_uses]. cbv zeta.
    pose proof (fit_env_app handlers
                  (handler_decisions true sol unboxed false)
                  (fun h => length (FlambdaReaperTraversal.handler_params h))
                  ds ks (handler_decisions_eq_length false) Hf) as Hf'.
    rewrite uses_fit_app. apply andb_true_iff. split.
    + unfold uses_fit. rewrite forallb_forall. intros use Hu.
      apply in_flat_map in Hu as ([k h] & Hkh & Hu).
      rewrite forallb_forall in Hch. pose proof (Hch (k, h) Hkh) as Hch1.
      assert (H : uses_fit counts
                    (huses
                       (map (fun '(k, h) =>
                               (k, handler_decisions true sol unboxed false
                                     h))
                          handlers ++ ds)
                       h)
                  = true).
      { refine (IHh k h Hkh _ _ Hf' Hch1 _).
        intros id c H. apply (Hcodes id c). apply in_or_app. left.
        apply in_flat_map. exists (k, h). split; [ exact Hkh | exact H ]. }
      unfold uses_fit in H. rewrite forallb_forall in H. exact (H use Hu).
    + refine (IH _ _ Hf' Hc _).
      intros id c H. apply (Hcodes id c). apply in_or_app. right. exact H.
  - intros ap ds ks Hf Hc _.
    cbn [conts_counted] in Hc. apply andb_true_iff in Hc as [Hr Hx].
    cbn [cont_uses]. unfold uses_fit. cbn [forallb].
    rewrite (use_fit_of_count _ _ _ _ Hf Hr),
      (use_fit_of_count _ _ _ _ Hf Hx).
    reflexivity.
  - intros ac ds ks Hf Hc _.
    cbn [conts_counted] in Hc. cbn [cont_uses]. unfold uses_fit.
    cbn [forallb]. rewrite (use_fit_of_count _ _ _ _ Hf Hc). reflexivity.
  - intros scrutinee arms ds ks Hf Hc _.
    cbn [conts_counted] in Hc. cbn [cont_uses]. unfold uses_fit.
    rewrite forallb_forall in Hc |- *. intros use Hu.
    apply in_map_iff in Hu as ([z ac] & <- & Hin).
    exact (use_fit_of_count _ _ _ _ Hf (Hc (z, ac) Hin)).
  - intros ds ks _ _ _. reflexivity.
  - intros params md handler IH ds ks Hf Hc Hcodes.
    exact (IH ds ks Hf Hc Hcodes).
Qed.

End Fit.

(* ======================================================================== *)
(* The statement                                                            *)
(* ======================================================================== *)

Theorem fix_never_fails_all : fix_never_fails_all_statement.
Proof.
  intros is_local u e T max_unbox_size lambda_lifting pdc tcr F code_deps
    cannot_change dominated to_unbox unboxed Hloc Hslots Hfrag Hleast Hcount
    Hkinds Hwf Hhas Hres Hbound.
  split.
  { exact (FlambdaReaperFixFailsProofs.fix_never_fails_full is_local u e T
             max_unbox_size lambda_lifting pdc tcr Hloc Hslots Hfrag Hleast
             Hcount Hkinds Hwf Hhas Hres Hbound). }
  destruct (FlambdaReaperWrapperProofs.unit_facts Auto u e
              (∀ f : field, is_local f = false) Hfrag Hslots)
    as (noted & deps & Hedges & _ & _ & Hnodup & _).
  assert (Hcodes :
            ∀ id c, (id, c) ∈ unit_codes e -> code_fits code_deps id c).
  { intros id c Hc.
    pose proof (proj1 (FlambdaReaperWrapperProofs.unit_codes_edges
                         _ _ _ _ _ _ _ _ _)
                  e None (unit_conts u) Hedges id c Hc) as Hce.
    destruct c as [params mc ret exn md body].
    cbn [FlambdaReaperFlowProofs.code_edges] in Hce.
    destruct Hce as (cd & Hcd & _ & _ & _ & _ & Hlen & _ & _ & Hne & Hbs
                     & Hcnt & _).
    split; [ exact Hne | split; [ exact Hcnt | split; [ exact Hbs | ] ] ].
    exists cd. split; [ | exact Hlen ].
    exact (FlambdaReaperFlowProofs.lookup_nodup_in _ _ _ Hnodup Hcd). }
  exact (proj1 (uses_fit_walk _ _ _ _ _ (unit_counts u)) e []
           (unit_counts u) (fit_env_top _) Hcount Hcodes).
Qed.

(* ======================================================================== *)
(* The two units                                                            *)
(* ======================================================================== *)

Theorem more_results : more_results_statement.
Proof.
  hnf. split.
  { apply (FlambdaReaperLeastProofs.compute_least_sound _ _ 40).
    vm_compute. reflexivity. }
  repeat match goal with |- _ ∧ _ => split end; vm_compute; reflexivity.
Qed.

Theorem empty_callee_codes : empty_callee_codes_statement.
Proof.
  hnf. split.
  { apply (FlambdaReaperLeastProofs.compute_least_sound _ _ 40).
    vm_compute. reflexivity. }
  repeat match goal with |- _ ∧ _ => split end; vm_compute; reflexivity.
Qed.

Theorem no_parameter_code : no_parameter_code_statement.
Proof.
  hnf. split.
  { apply (FlambdaReaperLeastProofs.compute_least_sound _ _ 40).
    vm_compute. reflexivity. }
  repeat match goal with |- _ ∧ _ => split end; vm_compute; reflexivity.
Qed.
