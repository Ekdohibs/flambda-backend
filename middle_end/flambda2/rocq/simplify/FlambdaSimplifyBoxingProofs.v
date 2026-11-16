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


(* Proofs of the statements of FlambdaSimplifyBoxingSpec.v. Nothing
   here needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaPrimitiveModel
  FlambdaPrimitiveSpec FlambdaPrimitiveProofs FlambdaSemanticsSpec
  FlambdaSemanticsProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaTypeWellFormedSpec FlambdaTypeWellFormedProofs FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaEnvAliasesProofs
  FlambdaTypingEnv FlambdaTypingEnvSpec FlambdaTypingEnvProofs
  FlambdaTypingEnvInvariantSpec FlambdaTypingEnvInvariantProofs
  FlambdaTypingEnvExamplesSpec FlambdaTypingEnvRefinementSpec
  FlambdaTypingEnvRefinementProofs FlambdaProvers
  FlambdaProversSpec FlambdaProversProofs FlambdaSimplifyPrimitivesSpec
  FlambdaSimplifyPrimitivesProofs FlambdaSimplifyProjectionsSpec
  FlambdaSimplifyProjectionsProofs FlambdaSimplifyBoxingSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* A number that is tagged                                                  *)
(* ======================================================================== *)

Theorem tag_immediate_typed : tag_immediate_typed_statement.
Proof.
  intros tenv statics vars arg ty v Hsat Hty Hnamed.
  destruct (unary_prim_value statics vars _ arg v Hnamed) as [w [Hw Hprim]].
  rewrite eval_prim_unary in Hprim. cbn [eval_unary_primitive] in Hprim.
  destruct (int_of_value Standard_naked_immediate w) as [i | ] eqn:Hi;
    [ | discriminate ].
  injection Hprim as <-.
  destruct (int_of_value_const _ _ _ Hi) as [-> _].
  apply tag_immediate_satisfied. exists i. split; [ reflexivity | ].
  exact (argument_type_in_machine tenv statics vars arg ty _ Hsat Hty Hw).
Qed.

(* The variables of the type of an argument are in scope, in the form
   that [let_typed_refines_under] asks for. *)
Lemma tag_immediate_free_variables (ty : flambda_type) (y : variable) :
  y ∈ free_variables (tag_immediate ty) -> y ∈ free_variables ty.
Proof.
  unfold tag_immediate. cbn. rewrite !app_nil_r. intros Hin. exact Hin.
Qed.

Theorem let_tag_immediate : let_tag_immediate_statement.
Proof.
  intros tenv x declared arg ty body body' Hwf Hfresh Hty tagged Hbody.
  apply (let_typed_refines_under_pure tenv x declared tagged body body'
           (tag_immediate ty) eq_refl Hwf Hfresh).
  - intros y Hy. apply tag_immediate_free_variables in Hy.
    exact (argument_type_in_scope tenv arg ty y Hty Hy).
  - intros statics vars v Hsat Hnamed.
    exact (tag_immediate_typed tenv statics vars arg ty v
             (env_satisfies_strictly_env_satisfies tenv statics vars Hsat)
             Hty Hnamed).
  - exact Hbody.
Qed.

(* ======================================================================== *)
(* A number that is boxed                                                   *)
(* ======================================================================== *)

Lemma is_number_of_kind_boxable (kind : boxable_number) (c : const) :
  is_number_of_kind kind c = true -> boxable_contents c = Some c.
Proof. destruct kind, c; simpl; try discriminate; reflexivity. Qed.

Theorem box_number_typed : box_number_typed_statement.
Proof.
  intros tenv statics vars kind arg ty v Hsat Hty Hnamed.
  destruct (unary_prim_value statics vars _ arg v Hnamed) as [w [Hw Hprim]].
  rewrite eval_prim_unary in Hprim. cbn [eval_unary_primitive] in Hprim.
  destruct w; try discriminate.
  match type of Hprim with
  | context [is_number_of_kind kind ?c] =>
    destruct (is_number_of_kind kind c) eqn:Hkind; [ | discriminate ]
  end.
  injection Hprim as <-.
  apply box_number_satisfied. eexists.
  split; [ exact (is_number_of_kind_boxable kind _ Hkind) | ].
  split; [ exact (is_number_of_kind_kind kind _ Hkind) | ].
  exact (argument_type_in_machine tenv statics vars arg ty _ Hsat Hty Hw).
Qed.

Lemma box_number_free_variables (kind : boxable_number) (ty : flambda_type)
    (alloc_mode : Alloc_mode.t) (y : variable) :
  y ∈ free_variables (box_number kind ty alloc_mode) ->
  y ∈ free_variables ty.
Proof.
  unfold box_number. cbn. rewrite ?app_nil_r. intros Hin. exact Hin.
Qed.

Theorem let_box_number : let_box_number_statement.
Proof.
  intros tenv x declared kind arg ty body body' Hwf Hfresh Hty boxed Hbody.
  apply (let_typed_refines_under_pure tenv x declared boxed body body'
           (box_number kind ty Alloc_mode.Heap) eq_refl Hwf Hfresh).
  - intros y Hy. apply box_number_free_variables in Hy.
    exact (argument_type_in_scope tenv arg ty y Hty Hy).
  - intros statics vars v Hsat Hnamed.
    exact (box_number_typed tenv statics vars kind arg ty v
             (env_satisfies_strictly_env_satisfies tenv statics vars Hsat)
             Hty Hnamed).
  - exact Hbody.
Qed.

(* ======================================================================== *)
(* The number inside                                                        *)
(* ======================================================================== *)

Lemma constant_in_terms (tenv : Typing_env.t) (c : const) :
  simple_in_terms tenv (Const c).
Proof. intros x [=]. Qed.

Theorem unboxing_provers_answer_in_terms
  : unboxing_provers_answer_in_terms_statement.
Proof.
  split.
  - intros tenv t s Hanswer.
    destruct (gen_value_to_meet_known _ tenv t s Hanswer) as [head Hhead].
    revert Hhead.
    unfold Provers.inspect_tagging_of_simple_value,
      Provers.inspect_tagging_immediates.
    answer_cases; intros [= <-];
      first
        [ eapply canonical_simple_of_alias_in_terms; eassumption
        | apply constant_in_terms ].
  - intros kind tenv t s.
    unfold Provers.meet_boxed_number_containing_simple.
    answer_cases. intros [= <-].
    eapply canonical_simple_of_alias_in_terms. eassumption.
Qed.

(* The value of an argument that may occur in terms has the alias type
   of the argument at the kind value, if it is of that kind, under a
   valuation that satisfies the environment and agrees with the
   machine. *)
Lemma alias_type_of_argument (tenv : Typing_env.t) (statics : static_env)
    (vars : env) (arg : simple) (v : value) :
  env_satisfies_strictly tenv statics vars ->
  simple_in_terms tenv arg ->
  eval_simple vars arg = Some v ->
  kind_of_value v = Value ->
  ∃ valuation : env,
    typing_env_satisfied_strictly statics valuation tenv
    ∧ agrees_in_terms tenv valuation vars
    ∧ type_satisfied statics valuation v (alias_type_of Value arg).
Proof.
  intros [_ [valuation [Hsatisfied Hagrees]]] Hterms Hv Hkind.
  exists valuation. split; [ exact Hsatisfied | ].
  split; [ exact Hagrees | ].
  apply alias_satisfied. split; [ | exact Hkind ].
  rewrite <- (simple_in_terms_agrees tenv valuation vars arg Hagrees Hterms).
  exact Hv.
Qed.

Theorem let_untag_immediate : let_untag_immediate_statement.
Proof.
  intros tenv x declared arg body Hterms Hwf untagged.
  assert (Hsound :
    ∀ (statics : static_env) (vars : env) (r : value),
      env_satisfies_strictly tenv statics vars ->
      eval_named statics vars untagged = Some r ->
      match
        Provers.meet_tagging_of_simple tenv Name_mode.normal
          (alias_type_of Value arg)
      with
      | Some (Meet_shortcut.Known_result s) => eval_simple vars s = Some r
      | Some Meet_shortcut.Invalid => False
      | Some Meet_shortcut.Need_meet | None => True
      end).
  { intros statics vars r Hsat Hnamed.
    destruct (unary_prim_value statics vars _ arg r Hnamed) as [v [Hv Hprim]].
    destruct (untag_immediate_defined statics v r Hprim) as [i [-> ->]].
    destruct (alias_type_of_argument tenv statics vars arg _ Hsat Hterms Hv
                eq_refl) as [valuation [Hsatisfied [Hagrees Htyped]]].
    pose proof (meet_tagging_of_simple_sound Name_mode.normal statics
                  valuation tenv (alias_type_of Value arg) _ Hsatisfied
                  (conj Hwf (alias_type_of_well_formed Value arg)) Htyped
                  (ex_intro _ i eq_refl)) as Hanswer.
    pose proof (proj1 unboxing_provers_answer_in_terms tenv
                  (alias_type_of Value arg)) as Hs.
    cbv beta in Hanswer, Hs.
    destruct (Provers.meet_tagging_of_simple tenv Name_mode.normal
                (alias_type_of Value arg)) as [[s | | ] | ];
      try exact I; [ | exact Hanswer ].
    destruct Hanswer as [i' [[= <-] Hnumber]].
    rewrite (simple_in_terms_agrees tenv valuation vars s Hagrees
               (Hs s eq_refl)).
    exact Hnumber. }
  destruct (Provers.meet_tagging_of_simple tenv Name_mode.normal
              (alias_type_of Value arg)) as [[s | | ] | ];
    try exact I.
  - apply let_simple_pure; [ reflexivity | exact Hsound ].
  - apply let_no_value_pure; [ reflexivity | exact Hsound ].
Qed.

Theorem let_unbox_number : let_unbox_number_statement.
Proof.
  intros tenv x declared kind arg body Hterms unboxed.
  assert (Hsound :
    ∀ (statics : static_env) (vars : env) (r : value),
      env_satisfies_strictly tenv statics vars ->
      eval_named statics vars unboxed = Some r ->
      match
        Provers.meet_boxed_number_containing_simple kind tenv
          Name_mode.normal (alias_type_of Value arg)
      with
      | Some (Meet_shortcut.Known_result s) => eval_simple vars s = Some r
      | Some Meet_shortcut.Invalid => False
      | Some Meet_shortcut.Need_meet | None => True
      end).
  { intros statics vars r Hsat Hnamed.
    destruct (unary_prim_value statics vars _ arg r Hnamed) as [v [Hv Hprim]].
    destruct (unbox_number_defined statics kind v r Hprim)
      as [Hdomain [c [Hboxed ->]]].
    destruct (alias_type_of_argument tenv statics vars arg _ Hsat Hterms Hv
                (boxed_number_of_kind statics v c Hboxed))
      as [valuation [Hsatisfied [Hagrees Htyped]]].
    pose proof (meet_boxed_number_containing_simple_sound kind
                  Name_mode.normal statics valuation tenv
                  (alias_type_of Value arg) v Hsatisfied I Htyped Hdomain)
      as Hanswer.
    pose proof (proj2 unboxing_provers_answer_in_terms kind tenv
                  (alias_type_of Value arg)) as Hs.
    cbv beta in Hanswer, Hs.
    destruct (Provers.meet_boxed_number_containing_simple kind tenv
                Name_mode.normal (alias_type_of Value arg)) as [[s | | ] | ];
      try exact I; [ | exact Hanswer ].
    destruct Hanswer as [c' [Hboxed' Hcontents]].
    rewrite Hboxed in Hboxed'. injection Hboxed' as <-.
    rewrite (simple_in_terms_agrees tenv valuation vars s Hagrees
               (Hs s eq_refl)).
    exact Hcontents. }
  destruct (Provers.meet_boxed_number_containing_simple kind tenv
              Name_mode.normal (alias_type_of Value arg)) as [[s | | ] | ];
    try exact I.
  - apply let_simple_pure; [ reflexivity | exact Hsound ].
  - apply let_no_value_pure; [ reflexivity | exact Hsound ].
Qed.

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

(* The end of the two examples: [q] is made equal to [x], and replaced
   by it in the body. *)
Ltac example_number_inside kind :=
  eapply (let_simple_refines_under _ example_q kind (Var example_x) _ _
            kind);
  [ vm_compute; reflexivity | vm_compute; reflexivity
  | intros ? [= <-]; vm_compute; reflexivity
  | eexists; vm_compute; reflexivity | vm_compute; reflexivity
  | apply expr_equivalent_under_refines; apply apply_cont_canonical;
    apply Forall2_cons; [ canonical_computed | apply Forall2_nil ] ].

Theorem example_untag_of_tag : example_untag_of_tag_statement.
Proof.
  intros k tagged.
  apply (let_tag_immediate (example_number_env naked_immediate_kind)
           example_p Value (Var example_x)
           (alias_type_of naked_immediate_kind (Var example_x)));
    [ vm_compute; reflexivity | vm_compute; reflexivity | argument_computed
    | ].
  eapply expr_refines_under_trans.
  - match goal with
    | |- expr_refines_under ?tenv (Let _ _ _ ?body) _ =>
      pose proof (let_untag_immediate tenv example_q
                    naked_immediate_kind (Var example_p) body) as Huntag
    end.
    cbv zeta in Huntag.
    match type of Huntag with
    | context [Provers.meet_tagging_of_simple ?tenv ?mode ?ty] =>
      let answer :=
        eval vm_compute in (Provers.meet_tagging_of_simple tenv mode ty)
      in
      replace (Provers.meet_tagging_of_simple tenv mode ty)
        with answer in Huntag by (vm_compute; reflexivity)
    end.
    cbv beta iota in Huntag. apply Huntag.
    + intros ? [= <-]. vm_compute. reflexivity.
    + apply (proj1 (types_well_formed_b_iff _)). vm_compute. reflexivity.
  - example_number_inside naked_immediate_kind.
Qed.

Theorem example_unbox_of_box : example_unbox_of_box_statement.
Proof.
  intros k float boxed.
  apply (let_box_number (example_number_env float) example_p Value
           Boxable_naked_float (Var example_x)
           (alias_type_of float (Var example_x)));
    [ vm_compute; reflexivity | vm_compute; reflexivity | argument_computed
    | ].
  eapply expr_refines_under_trans.
  - match goal with
    | |- expr_refines_under ?tenv (Let _ _ _ ?body) _ =>
      pose proof (let_unbox_number tenv example_q float
                    Boxable_naked_float (Var example_p) body) as Hunbox
    end.
    cbv zeta in Hunbox.
    match type of Hunbox with
    | context [Provers.meet_boxed_number_containing_simple ?kind ?tenv ?mode
                 ?ty] =>
      let answer :=
        eval vm_compute in
          (Provers.meet_boxed_number_containing_simple kind tenv mode ty)
      in
      replace (Provers.meet_boxed_number_containing_simple kind tenv mode ty)
        with answer in Hunbox by (vm_compute; reflexivity)
    end.
    cbv beta iota in Hunbox. apply Hunbox.
    intros ? [= <-]. vm_compute. reflexivity.
  - example_number_inside float.
Qed.
