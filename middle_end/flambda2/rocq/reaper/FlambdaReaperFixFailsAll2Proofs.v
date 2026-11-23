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

(* The proofs of FlambdaReaperFixFailsAll2Spec.v. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaWellFormedSpec FlambdaReaperGraphSpec
  FlambdaReaperTraversal FlambdaReaperSolution FlambdaReaperRebuild
  FlambdaReaperCallingConventions FlambdaReaperUnboxing FlambdaReaperLeast
  FlambdaReaperFlowSpec FlambdaReaperLeastSpec FlambdaReaperWrapperSpec
  FlambdaReaperWrapperDerivedSpec FlambdaReaperFixFailsSpec
  FlambdaReaperFixFailsAllSpec FlambdaReaperFixFailsAll2Spec
  FlambdaSemanticsSpec.
From OxCamlFlambda2 Require FlambdaReaperWrapperProofs
  FlambdaReaperFlowProofs FlambdaReaperFixFailsProofs
  FlambdaReaperFixFailsAllProofs FlambdaMapProofs.
From OxCamlFlambda2 Require Import FlambdaBindersSpec.
Import ListNotations.

Theorem results_match_within : results_match_within_statement.
Proof.
  intros e H. unfold results_match_callees in H.
  unfold results_within_callees. rewrite forallb_forall in H |- *.
  intros ap Hap. pose proof (H ap Hap) as Ha.
  rewrite forallb_forall in Ha |- *. intros [id c] Hc.
  pose proof (Ha (id, c) Hc) as Hac. cbv beta iota in Hac |- *.
  destruct (call_promises ap id c); cbn [negb orb] in Hac |- *;
    [ | reflexivity ].
  apply Nat.eqb_eq in Hac. apply Nat.leb_le. lia.
Qed.

(* Every call of a term of the fragment is of the fragment. *)
Lemma fragment_calls (heap : bool) (codes : list (code_id * code))
    (symbols : list symbol) :
  (∀ e : expr,
     flow_fragment heap codes symbols e = true ->
     ∀ ap, ap ∈ calls_of e -> apply_in_flow_fragment codes ap = true)
  ∧ (∀ h : continuation_handler,
       handler_flow_fragment heap codes symbols h = true ->
       ∀ ap, ap ∈ handler_calls_of h ->
         apply_in_flow_fragment codes ap = true).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind;
    cbn [flow_fragment handler_flow_fragment calls_of handler_calls_of].
  - intros x kind d body IH H ap Hap.
    apply andb_true_iff in H as [_ H]. exact (IH H ap Hap).
  - intros bindings body IHc IH H ap Hap.
    apply andb_true_iff in H as [Hb H].
    apply in_app_or in Hap as [Hap | Hap]; [ | exact (IH H ap Hap) ].
    apply in_flat_map in Hap as (b & Hbin & Hap).
    rewrite forallb_forall in Hb. pose proof (Hb b Hbin) as Hbf.
    destruct b as [id c | id | sym const];
      cbn [binding_calls_of] in Hap; [ | destruct Hap | destruct Hap ].
    pose proof (IHc id c Hbin) as IHb.
    destruct c as [params mc ret exn md cbody].
    cbn [binding_flow_fragment code_flow_fragment code_calls_of code_body]
      in Hbf, Hap, IHb.
    apply andb_true_iff in Hbf as [_ Hbf]. exact (IHb Hbf ap Hap).
  - intros k h body IHh IH H ap Hap.
    apply andb_true_iff in H as [Hh H].
    apply in_app_or in Hap as [Hap | Hap];
      [ exact (IHh Hh ap Hap) | exact (IH H ap Hap) ].
  - intros handlers body IHh IH H ap Hap.
    apply andb_true_iff in H as [Hh H].
    apply in_app_or in Hap as [Hap | Hap]; [ | exact (IH H ap Hap) ].
    apply in_flat_map in Hap as ([k h] & Hkh & Hap).
    rewrite forallb_forall in Hh.
    exact (IHh k h Hkh (Hh (k, h) Hkh) ap Hap).
  - intros ap0 H ap [<- | []]. exact H.
  - intros ac _ ap [].
  - intros scrutinee arms _ ap [].
  - intros _ ap [].
  - intros params md handler IH H ap Hap. exact (IH H ap Hap).
Qed.

(* The metadata that [apply_change_u] finds is that of code of the
   unit. *)
Lemma lookup_codes (e : expr) (id : code_id) (md : code_metadata) :
  lookup id (bound_code_metadata e) = Some md ->
  ∃ c, (id, c) ∈ unit_codes e ∧ metadata_of_code c = md.
Proof.
  intros H. apply FlambdaMapProofs.lookup_in in H.
  rewrite (proj1 FlambdaReaperWrapperProofs.bound_code_metadata_codes e)
    in H.
  apply in_map_iff in H as ([id' c] & E & Hin). cbn in E.
  injection E as E1 E2. subst id'. exists c. split; [ exact Hin | exact E2 ].
Qed.

Lemma group_by_parameter_nonempty {A : Type} (a : arity) (l : list A) :
  a <> [] -> length (group_by_parameter a l) <> 0.
Proof.
  intros Ha. destruct a as [ | c a]; [ contradiction | ].
  unfold group_by_parameter, unarize_per_parameter. cbn. discriminate.
Qed.

Theorem fix_never_fails_all2 : fix_never_fails_all2_statement.
Proof.
  intros is_local u e T max_unbox_size lambda_lifting pdc tcr F code_deps
    cannot_change dominated to_unbox unboxed Hloc Hslots Hfrag Hleast Hcount
    Hkinds Hwf Hhas Hmatch Hprom Hbound Hparams Hcode.
  pose proof (results_match_within e Hmatch) as Hres.
  destruct (FlambdaReaperWrapperProofs.unit_facts Auto u e
              (∀ f : field, is_local f = false) Hfrag Hslots)
    as (noted & deps & Hedges & _ & _ & Hnodup & _).
  pose proof (fun id c Hc =>
                proj1 (FlambdaReaperWrapperProofs.unit_codes_edges
                         _ _ _ _ _ _ _ _ _)
                  e None (unit_conts u) Hedges id c Hc) as Hcode_edges.
  (* The code dep of code of the unit. *)
  assert (Hdep :
            ∀ id c, (id, c) ∈ unit_codes e ->
              ∃ cd, lookup id code_deps = Some cd
                    ∧ length (cd_return cd) = length (code_result_kinds c)).
  { intros id c Hc. pose proof (Hcode_edges id c Hc) as Hce.
    destruct c as [params mc ret exn md body].
    cbn [FlambdaReaperFlowProofs.code_edges] in Hce.
    destruct Hce as (cd & Hcd & _ & _ & _ & _ & Hlen & _).
    exists cd. split; [ | exact Hlen ].
    exact (FlambdaReaperFlowProofs.lookup_nodup_in _ _ _ Hnodup Hcd). }
  split; [ | split; [ | split ] ].
  - exact (FlambdaReaperFixFailsProofs.fix_never_fails_callees is_local u e
             T max_unbox_size lambda_lifting pdc tcr Hloc Hslots Hfrag Hleast
             Hcount Hkinds Hwf Hhas Hres Hprom Hbound).
  - assert (Hcodes :
              ∀ id c, (id, c) ∈ unit_codes e ->
                FlambdaReaperFixFailsAllProofs.code_fits code_deps id c).
    { intros id c Hc.
      pose proof (Hcode_edges id c Hc) as Hce.
      destruct c as [params mc ret exn md body].
      cbn [FlambdaReaperFlowProofs.code_edges] in Hce.
      destruct Hce as (cd & Hcd & _ & _ & _ & _ & Hlen & _ & _ & Hne & Hbs
                       & Hcnt & _).
      split; [ exact Hne | split; [ exact Hcnt | split; [ exact Hbs | ] ] ].
      exists cd. split; [ | exact Hlen ].
      exact (FlambdaReaperFlowProofs.lookup_nodup_in _ _ _ Hnodup Hcd). }
    exact (proj1 (FlambdaReaperFixFailsAllProofs.uses_fit_walk
                    _ _ _ _ _ (unit_counts u))
             e [] (unit_counts u)
             (FlambdaReaperFixFailsAllProofs.fit_env_top _) Hcount Hcodes).
  - unfold calls_fit. rewrite forallb_forall. intros [ads ap] Hin.
    cbv beta iota.
    pose proof (proj1 (FlambdaReaperFixFailsProofs.rebuilt_calls_of
                         _ _ _ _ _)
                  e [] ads ap Hin) as Hap.
    unfold call_fits. apply andb_true_iff. split.
    2:{ unfold known_callees_have_code in Hcode.
        rewrite forallb_forall in Hcode. exact (Hcode ap Hap). }
    destruct (apply_change_u _ _ _ _ _ _ _ ap) as [ | pd rd] eqn:Ech;
      [ reflexivity | ].
    unfold apply_change_u in Ech.
    destruct (code_id_actually_called _ _ ap) as [id | ] eqn:Eid;
      [ | discriminate ].
    destruct (lookup id (bound_code_metadata e)) as [md | ] eqn:Emd;
      [ | discriminate ].
    unfold code_change_u in Ech.
    destruct (cannot_change_calling_convention _ _ _); [ discriminate | ].
    destruct (lookup id code_deps) as [cd | ] eqn:Ecd; [ | discriminate ].
    injection Ech as <- <-.
    destruct (lookup_codes e id md Emd) as (c & Hc & Hmd).
    destruct (Hdep id c Hc) as (cd' & Ecd' & Hlen).
    rewrite Ecd in Ecd'. injection Ecd' as <-.
    (* The call promises to call the code. *)
    assert (Hpc : call_promises ap id c = true).
    { unfold callees_promised in Hprom. rewrite forallb_forall in Hprom.
      pose proof (Hprom ap Hap) as Hp. cbv beta in Hp.
      change (solution_of_tables T) with (solution_of_tables T) in Hp.
      rewrite Eid in Hp. rewrite forallb_forall in Hp.
      pose proof (Hp (id, c) Hc) as H. cbv beta iota in H.
      rewrite Pos.eqb_refl in H. exact H. }
    assert (Hres_eq :
              length (unarize (ap_return_arity ap))
              = length (code_result_kinds c)).
    { unfold results_match_callees in Hmatch.
      rewrite forallb_forall in Hmatch. pose proof (Hmatch ap Hap) as Hm.
      rewrite forallb_forall in Hm. pose proof (Hm (id, c) Hc) as H.
      cbv beta iota in H. rewrite Hpc in H. cbn [negb orb] in H.
      apply Nat.eqb_eq. exact H. }
    apply andb_true_iff. split.
    + apply Nat.eqb_eq. unfold decisions_u.
      rewrite length_map, length_combine, Hres_eq, <- Hlen, <- Hmd.
      destruct c as [params mc ret exn md' body].
      cbn [metadata_of_code code_result_kinds code_result_arity].
      cbn [code_result_kinds code_result_arity] in Hlen.
      rewrite Hlen. apply Nat.min_id.
    + apply negb_true_iff. apply Nat.eqb_neq.
      unfold call_promises in Hpc.
      destruct (ap_call_kind ap) as [named | | ids] eqn:Ek.
      * destruct (Nat.eqb_spec (length (ap_args_arity ap)) 0) as [E0 | ];
          [ exfalso | assumption ].
        unfold in_flow_fragment in Hfrag.
        apply andb_true_iff in Hfrag as [Hff _].
        pose proof (proj1 (fragment_calls _ _ _) e Hff ap Hap) as Hapf.
        unfold apply_in_flow_fragment in Hapf. rewrite Ek in Hapf.
        apply andb_true_iff in Hapf as [_ H0]. rewrite E0 in H0.
        discriminate.
      * unfold code_id_actually_called in Eid. rewrite Ek in Eid.
        discriminate.
      * destruct (ap_args_arity ap); [ discriminate | cbn; discriminate ].
  - unfold codes_grouped. rewrite forallb_forall. intros [id c] Hc.
    cbv beta iota.
    destruct (Hdep id c Hc) as (cd & Ecd & _). rewrite Ecd.
    apply negb_true_iff. apply Nat.eqb_neq.
    apply group_by_parameter_nonempty.
    unfold unit_codes_have_params in Hparams.
    rewrite forallb_forall in Hparams. pose proof (Hparams (id, c) Hc) as H.
    cbv beta iota in H. intros E. rewrite E in H. discriminate.
Qed.

Theorem premises_exclude_units : premises_exclude_units_statement.
Proof.
  hnf. repeat match goal with |- _ ∧ _ => split end; vm_compute; reflexivity.
Qed.

Lemma group_by_sizes_concat {A : Type} (sizes : list nat) :
  ∀ l : list A,
    fold_right plus 0 sizes = length l ->
    concat (group_by_sizes sizes l) = l.
Proof.
  induction sizes as [ | n sizes IH]; intros l H.
  - cbn in H. destruct l; [ reflexivity | discriminate ].
  - cbn [group_by_sizes concat fold_right] in *.
    rewrite IH; [ apply firstn_skipn | ].
    rewrite length_skipn. lia.
Qed.

Lemma unarize_sizes (a : arity) :
  fold_right plus 0
    (map (@length flambda_kind) (unarize_per_parameter a))
  = length (unarize a).
Proof.
  unfold unarize_per_parameter, unarize.
  induction a as [ | c a IH]; [ reflexivity | ].
  cbn [map fold_right flat_map]. rewrite length_app, IH. reflexivity.
Qed.

Theorem call_args_grouped : call_args_grouped_statement.
Proof.
  intros e H ap Hap. unfold call_args_fit in H.
  rewrite forallb_forall in H. pose proof (H ap Hap) as Hl.
  apply Nat.eqb_eq in Hl. unfold group_by_parameter.
  apply group_by_sizes_concat. rewrite unarize_sizes. lia.
Qed.
