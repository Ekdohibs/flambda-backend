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


(* Proofs of the statements of FlambdaSimplifyArithSpec.v. Nothing
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
  FlambdaSimplifyProjectionsProofs FlambdaSimplifyArithSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* The numbers that an argument may be                                      *)
(* ======================================================================== *)

Theorem argument_possible : argument_possible_statement.
Proof.
  intros tenv statics vars arg ty i Hsat Hty Hwf Hv Hrange.
  split; [ exact Hrange | ].
  destruct (argument_type_satisfied tenv statics vars arg ty _ Hsat Hty Hv)
    as [valuation [Hsatisfied [_ Htyped]]].
  pose proof (meet_equals_tagged_immediates_sound statics valuation
                tenv ty _ Hsatisfied Hwf Htyped (ex_intro _ i eq_refl))
    as Hanswer.
  destruct (Provers.meet_equals_tagged_immediates tenv ty)
    as [[imms | | ] | ]; try exact I; [ | exact Hanswer ].
  destruct Hanswer as [j [[= <-] Hin]]. exact Hin.
Qed.

(* The arithmetic of tagged immediates that has a value was applied to
   two tagged immediates in range, and its value is the tagged
   immediate of the result. *)
Lemma int_arith_tagged_defined (statics : static_env)
    (op : binary_int_arith_op) (v1 v2 r : value) :
  eval_prim statics (Binary (Int_arith Standard_tagged_immediate op)) [v1; v2]
  = Some r ->
  ∃ i1 i2 result : Z,
    v1 = Value_const (Tagged_immediate i1)
    ∧ v2 = Value_const (Tagged_immediate i2)
    ∧ in_range tagged_width i1 = true
    ∧ in_range tagged_width i2 = true
    ∧ eval_int_arith tagged_width op i1 i2 = Some result
    ∧ r = Value_const (Tagged_immediate result).
Proof.
  cbn [eval_prim eval_binary_primitive].
  destruct (int_of_value Standard_tagged_immediate v1) as [a | ] eqn:Ha;
    [ | discriminate ].
  destruct (int_of_value Standard_tagged_immediate v2) as [b | ] eqn:Hb;
    [ | discriminate ].
  destruct (eval_int_arith (int_width Standard_tagged_immediate) op a b)
    as [result | ] eqn:Hresult; [ | discriminate ].
  intros [= <-].
  destruct (int_of_value_const _ _ _ Ha) as [-> Hrange_a].
  destruct (int_of_value_const _ _ _ Hb) as [-> Hrange_b].
  exists a, b, result. repeat split; assumption.
Qed.

(* So, in a state that satisfies the environment, to two of the numbers
   that the arguments may be. *)
Lemma int_arith_tagged_possible (tenv : Typing_env.t) (statics : static_env)
    (vars : env) (op : binary_int_arith_op) (arg1 arg2 : simple)
    (ty1 ty2 : flambda_type) (r : value) :
  env_satisfies_strictly tenv statics vars ->
  argument_type tenv arg1 ty1 ->
  argument_type tenv arg2 ty2 ->
  well_formed_input tenv ty1 ->
  well_formed_input tenv ty2 ->
  eval_named statics vars
    (Prim (Binary (Int_arith Standard_tagged_immediate op)) [arg1; arg2])
  = Some r ->
  ∃ i1 i2 result : Z,
    possible_tagged_immediate tenv ty1 i1
    ∧ possible_tagged_immediate tenv ty2 i2
    ∧ eval_simple vars arg1 = Some (Value_const (Tagged_immediate i1))
    ∧ eval_simple vars arg2 = Some (Value_const (Tagged_immediate i2))
    ∧ eval_int_arith tagged_width op i1 i2 = Some result
    ∧ r = Value_const (Tagged_immediate result).
Proof.
  intros Hsat Hty1 Hty2 Hwf1 Hwf2 Hnamed. simpl in Hnamed.
  destruct (eval_simple vars arg1) as [v1 | ] eqn:Hv1; [ | discriminate ].
  destruct (eval_simple vars arg2) as [v2 | ] eqn:Hv2; [ | discriminate ].
  change (eval_prim statics (Binary (Int_arith Standard_tagged_immediate op))
            [v1; v2]
          = Some r) in Hnamed.
  destruct (int_arith_tagged_defined statics op v1 v2 r Hnamed)
    as [i1 [i2 [result [-> [-> [Hrange1 [Hrange2 [Hresult ->]]]]]]]].
  exists i1, i2, result.
  split;
    [ exact (argument_possible tenv statics vars arg1 ty1 i1 Hsat
               Hty1 Hwf1 Hv1 Hrange1) | ].
  split;
    [ exact (argument_possible tenv statics vars arg2 ty2 i2 Hsat
               Hty2 Hwf2 Hv2 Hrange2) | ].
  repeat split; assumption || reflexivity.
Qed.

(* ======================================================================== *)
(* The four conclusions                                                     *)
(* ======================================================================== *)

Theorem let_int_arith_tagged : let_int_arith_tagged_statement.
Proof.
  intros tenv x declared op arg1 arg2 ty1 ty2 body Hty1 Hty2 Hwf1
    Hwf2 arith results.
  assert (Hsound :
    ∀ (statics : static_env) (vars : env) (r : value),
      env_satisfies_strictly tenv statics vars ->
      eval_named statics vars arith = Some r ->
      ∃ i1 i2 result : Z,
        possible_tagged_immediate tenv ty1 i1
        ∧ possible_tagged_immediate tenv ty2 i2
        ∧ eval_simple vars arg1 = Some (Value_const (Tagged_immediate i1))
        ∧ eval_simple vars arg2 = Some (Value_const (Tagged_immediate i2))
        ∧ eval_int_arith tagged_width op i1 i2 = Some result
        ∧ r = Value_const (Tagged_immediate result)).
  { intros statics vars r Hsat Hnamed.
    exact (int_arith_tagged_possible tenv statics vars op arg1 arg2 ty1 ty2
             r Hsat Hty1 Hty2 Hwf1 Hwf2 Hnamed). }
  split; [ | split; [ | split ] ].
  - intros result Hresults. apply let_constant_pure; [ reflexivity | ].
    intros statics vars r Hsat Hnamed.
    destruct (Hsound statics vars r Hsat Hnamed)
      as [i1 [i2 [result' [Hi1 [Hi2 [_ [_ [Heval ->]]]]]]]].
    rewrite (Hresults i1 i2 Hi1 Hi2 result' Heval). reflexivity.
  - intros Hresults. apply let_simple_pure; [ reflexivity | ].
    intros statics vars r Hsat Hnamed.
    destruct (Hsound statics vars r Hsat Hnamed)
      as [i1 [i2 [result' [Hi1 [Hi2 [_ [Hv2 [Heval ->]]]]]]]].
    rewrite (Hresults i1 i2 Hi1 Hi2 result' Heval). exact Hv2.
  - intros Hresults. apply let_simple_pure; [ reflexivity | ].
    intros statics vars r Hsat Hnamed.
    destruct (Hsound statics vars r Hsat Hnamed)
      as [i1 [i2 [result' [Hi1 [Hi2 [Hv1 [_ [Heval ->]]]]]]]].
    rewrite (Hresults i1 i2 Hi1 Hi2 result' Heval). exact Hv1.
  - intros Hresults. apply let_no_value_pure; [ reflexivity | ].
    intros statics vars r Hsat Hnamed.
    destruct (Hsound statics vars r Hsat Hnamed)
      as [i1 [i2 [result' [Hi1 [Hi2 [_ [_ [Heval _]]]]]]]].
    rewrite (Hresults i1 i2 Hi1 Hi2) in Heval. discriminate.
Qed.

(* ======================================================================== *)
(* The identities                                                           *)
(* ======================================================================== *)

(* A number in range wraps to itself. *)
Lemma wrap_small (width z : Z) :
  1 <= width -> in_range width z = true -> wrap width z = z.
Proof.
  intros Hwidth Hrange. unfold in_range in Hrange.
  apply andb_true_iff in Hrange. destruct Hrange as [Hlow Hhigh].
  apply Z.leb_le in Hlow. apply Z.ltb_lt in Hhigh. unfold wrap.
  assert (Hhalf : 0 < 2 ^ (width - 1)) by (apply Z.pow_pos_nonneg; lia).
  assert (Hdouble : 2 ^ width = 2 * 2 ^ (width - 1)).
  { rewrite <- Z.pow_succ_r by lia. f_equal. lia. }
  rewrite Z.mod_small by lia. lia.
Qed.

(* The unsigned reading of a number in range wraps to the number. *)
Lemma wrap_unsigned (width z : Z) :
  1 <= width -> in_range width z = true -> wrap width (unsigned width z) = z.
Proof.
  intros Hwidth Hrange. unfold unsigned.
  destruct (z <? 0) eqn:Hneg; [ | exact (wrap_small width z Hwidth Hrange) ].
  rewrite <- (wrap_small width z Hwidth Hrange) at 2. unfold wrap.
  replace (z + 2 ^ width + 2 ^ (width - 1))
    with (z + 2 ^ (width - 1) + 1 * 2 ^ width) by lia.
  rewrite Z.mod_add; [ reflexivity | ].
  apply Z.pow_nonzero; lia.
Qed.

Lemma in_range_small (width : Z) (z : Z) :
  1 <= width -> -1 <= z <= 0 -> in_range width z = true.
Proof.
  intros Hwidth Hz. unfold in_range.
  assert (Hhalf : 0 < 2 ^ (width - 1)) by (apply Z.pow_pos_nonneg; lia).
  apply andb_true_iff. split; [ apply Z.leb_le | apply Z.ltb_lt ]; lia.
Qed.

Theorem int_arith_identities : int_arith_identities_statement.
Proof.
  intros kind j width Hrange.
  pose proof (int_width_positive kind) as Hwidth. fold width in Hwidth.
  pose proof (wrap_small width j Hwidth Hrange) as Hj.
  pose proof (wrap_small width 0 Hwidth (in_range_small width 0 Hwidth
                                           ltac:(lia))) as H0.
  pose proof (wrap_small width (-1) Hwidth (in_range_small width (-1) Hwidth
                                              ltac:(lia))) as Hm1.
  cbn [eval_int_arith].
  rewrite Z.add_0_l, Z.add_0_r, Z.sub_0_r, Z.mul_0_l, Z.mul_0_r, Z.mul_1_l,
    Z.mul_1_r, Z.land_m1_l, Z.land_m1_r, Z.land_0_l, Z.land_0_r, Z.lor_m1_l,
    Z.lor_m1_r, Z.lor_0_l, Z.lor_0_r, Z.lxor_0_l, Z.lxor_0_r.
  change (Z.opp 1) with (-1). rewrite Hj, H0, Hm1.
  repeat (split; [ reflexivity | ]).
  split.
  - intros s. split; [ reflexivity | ]. split; [ reflexivity | ].
    replace (1 =? 0) with false by reflexivity.
    assert (Hone : reading width s 1 = 1) by (destruct s; reflexivity).
    rewrite Hone, Z.quot_1_r, Z.rem_1_r, H0.
    split; [ | reflexivity ].
    destruct s; simpl; [ rewrite Hj; reflexivity | ].
    rewrite (wrap_unsigned width j Hwidth Hrange). reflexivity.
  - replace (-1 =? 0) with false by reflexivity. simpl.
    assert (Hrem : Z.rem j (-1) = 0).
    { change (-1) with (Z.opp 1). rewrite Z.rem_opp_r'. apply Z.rem_1_r. }
    rewrite Hrem, H0. reflexivity.
Qed.

(* ======================================================================== *)
(* Where the primitive stays                                                *)
(* ======================================================================== *)

Lemma these_tagged_immediates_kind (imms : list Z) :
  FlambdaTypeGrammar.kind (these_tagged_immediates imms) = Value.
Proof. destruct imms as [ | i [ | i' imms]]; reflexivity. Qed.

Lemma these_tagged_immediates_closed (imms : list Z) (y : variable) :
  ~ y ∈ free_variables (these_tagged_immediates imms).
Proof.
  destruct imms as [ | i [ | i' imms]]; cbn; rewrite ?app_nil_r;
    intros [].
Qed.

Theorem let_int_arith_tagged_these : let_int_arith_tagged_these_statement.
Proof.
  intros tenv x declared op arg1 arg2 ty1 ty2 results body body' Hwf
    Hfresh Hty1 Hty2 Hwf1 Hwf2 Hresults arith Hbody.
  apply (let_typed_refines_under_pure tenv x declared arith body body'
           (these_tagged_immediates results) eq_refl Hwf Hfresh).
  - intros y Hy. exfalso.
    exact (these_tagged_immediates_closed results y Hy).
  - intros statics vars v Hsat Hnamed.
    destruct (int_arith_tagged_possible tenv statics vars op arg1 arg2 ty1
                ty2 v Hsat Hty1 Hty2 Hwf1 Hwf2 Hnamed)
      as [i1 [i2 [result [Hi1 [Hi2 [_ [_ [Heval ->]]]]]]]].
    apply these_tagged_immediates_satisfied. exists result.
    split; [ reflexivity | exact (Hresults i1 i2 result Hi1 Hi2 Heval) ].
  - rewrite these_tagged_immediates_kind. exact Hbody.
Qed.

Theorem let_int_arith_tagged_unknown
  : let_int_arith_tagged_unknown_statement.
Proof.
  intros tenv x declared op arg1 arg2 body body' Hwf Hfresh arith Hbody.
  apply (let_typed_refines_under_pure tenv x declared arith body body'
           any_tagged_immediate eq_refl Hwf Hfresh).
  - intros y Hy. cbn in Hy. rewrite ?app_nil_r in Hy. destruct Hy.
  - intros statics vars v Hsat Hnamed. unfold arith in Hnamed.
    simpl in Hnamed.
    destruct (eval_simple vars arg1) as [v1 | ]; [ | discriminate ].
    destruct (eval_simple vars arg2) as [v2 | ]; [ | discriminate ].
    change (eval_prim statics
              (Binary (Int_arith Standard_tagged_immediate op)) [v1; v2]
            = Some v) in Hnamed.
    destruct (int_arith_tagged_defined statics op v1 v2 v Hnamed)
      as [i1 [i2 [result [_ [_ [_ [_ [_ ->]]]]]]]].
    apply any_tagged_immediate_satisfied. exists result. reflexivity.
  - exact Hbody.
Qed.

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

(* The answers of the prover in the hypotheses are computed. *)
Ltac possible_computed :=
  unfold possible_tagged_immediate in *;
  repeat match goal with
         | H : context [Provers.meet_equals_tagged_immediates ?tenv ?ty]
           |- _ =>
           let answer :=
             eval vm_compute in
               (Provers.meet_equals_tagged_immediates tenv ty)
           in
           replace (Provers.meet_equals_tagged_immediates tenv ty)
             with answer in H by (vm_compute; reflexivity);
           cbv beta iota in H
         end.

Theorem example_add_constants : example_add_constants_statement.
Proof.
  intros k. apply expr_refines_under_create.
  eapply (let_simple_refines_under Typing_env.create example_x Value
            (Const (Tagged_immediate 3)) _ _ Value);
    [ reflexivity | reflexivity | intros ? [=] | reflexivity
    | vm_compute; reflexivity | ].
  eapply expr_refines_under_trans.
  - match goal with
    | |- expr_refines_under ?tenv (Let _ _ _ ?body) _ =>
      pose proof (let_int_arith_tagged tenv example_y Value Add
                    (Var example_x) (Const (Tagged_immediate 4))
                    (alias_type_of Value (Const (Tagged_immediate 3)))
                    (alias_type_of Value (Const (Tagged_immediate 4))) body)
        as Harith
    end.
    cbv zeta in Harith.
    destruct Harith as [Hconstant _];
      [ argument_computed | argument_computed | well_formed_input_computed
      | well_formed_input_computed | ].
    apply (Hconstant 7). intros i1 i2 [_ H1] [_ H2]. possible_computed.
    destruct H1 as [<- | []]. destruct H2 as [<- | []].
    vm_compute. intros r' [= <-]. reflexivity.
  - eapply (let_simple_refines_under _ example_y Value
              (Const (Tagged_immediate 7)) _ _ Value);
      [ vm_compute; reflexivity | vm_compute; reflexivity | intros ? [=]
      | reflexivity | vm_compute; reflexivity
      | apply expr_equivalent_under_refines; apply apply_cont_canonical;
        apply Forall2_cons; [ canonical_computed | apply Forall2_nil ] ].
Qed.

Theorem example_add_zero : example_add_zero_statement.
Proof.
  intros k. eapply expr_refines_under_trans.
  - match goal with
    | |- expr_refines_under ?tenv (Let _ _ _ ?body) _ =>
      pose proof (let_int_arith_tagged tenv example_y Value Add
                    (Var example_x) (Const (Tagged_immediate 0))
                    (alias_type_of Value (Var example_x))
                    (alias_type_of Value (Const (Tagged_immediate 0))) body)
        as Harith
    end.
    cbv zeta in Harith.
    destruct Harith as [_ [_ [Hfirst _]]];
      [ argument_computed | argument_computed | well_formed_input_computed
      | well_formed_input_computed | ].
    apply Hfirst. intros i1 i2 [Hrange1 _] [_ H2]. possible_computed.
    destruct H2 as [<- | []].
    destruct (int_arith_identities Standard_tagged_immediate i1 Hrange1)
      as [_ [Hadd _]].
    unfold tagged_width. rewrite Hadd. intros r' [= <-]. reflexivity.
  - eapply (let_simple_refines_under _ example_y Value (Var example_x) _ _
              Value);
      [ vm_compute; reflexivity | vm_compute; reflexivity
      | intros ? [= <-]; vm_compute; reflexivity
      | eexists; vm_compute; reflexivity | vm_compute; reflexivity
      | apply expr_equivalent_under_refines; apply apply_cont_canonical;
        apply Forall2_cons; [ canonical_computed | apply Forall2_nil ] ].
Qed.
