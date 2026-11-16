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

(* Proofs of the statements of FlambdaSimplifyValueSlotsSpec.v. Nothing
   here needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaNominalSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaTypingEnv
  FlambdaTypingEnvSpec FlambdaTypingEnvRefinementSpec FlambdaTypeRenameSpec
  FlambdaEnvRenameSpec FlambdaProvers FlambdaProversSpec FlambdaProversProofs
  FlambdaProversRenameSpec FlambdaProversRenameProofs
  FlambdaSimplifyPrimitivesSpec FlambdaSimplifyPrimitivesProofs
  FlambdaSimplifyProjectionsSpec FlambdaSimplifyProjectionsProofs
  FlambdaSimplifyValueSlotsSpec.
Import ListNotations.

Lemma simplify_project_value_slot_known (tenv : Typing_env.t)
    (value_slot_kind : flambda_kind) (slot : value_slot) (ty : flambda_type)
    (s : simple) :
  simplify_project_value_slot tenv value_slot_kind slot ty = Some s ->
  Provers.meet_project_value_slot_simple tenv Name_mode.normal ty slot
    value_slot_kind
  = Some (Meet_shortcut.Known_result s).
Proof.
  unfold simplify_project_value_slot.
  destruct (Provers.meet_project_value_slot_simple tenv Name_mode.normal ty
              slot value_slot_kind) as [[s' | | ] | ];
    try discriminate.
  intros [= <-]. reflexivity.
Qed.

(* In a state that satisfies the environment, the [Simple] answered has
   the value that the slot has, whatever function slot the projection
   is from. *)
Lemma answer_has_contents (tenv : Typing_env.t)
    (value_slot_kind : flambda_kind) (slot : value_slot) (closure : simple)
    (ty : flambda_type) (s : simple) (statics : static_env) (vars : env)
    (v contents : value) (project_from : function_slot) :
  argument_type tenv closure ty ->
  simplify_project_value_slot tenv value_slot_kind slot ty = Some s ->
  env_satisfies_strictly tenv statics vars ->
  eval_simple vars closure = Some v ->
  project_value_slot statics project_from slot v = Some contents ->
  eval_simple vars s = Some contents.
Proof.
  intros Hty Hanswer Hsat Hv Hproject.
  apply simplify_project_value_slot_known in Hanswer.
  destruct (argument_type_satisfied tenv statics vars closure ty v Hsat Hty
              Hv) as [valuation [Hsatisfied [Hagrees Htyped]]].
  rewrite (simple_in_terms_agrees tenv valuation vars s Hagrees
             (proj2 (proj2 provers_answer_in_terms) slot value_slot_kind
                tenv ty s Hanswer)).
  exact (meet_project_value_slot_simple_known statics valuation tenv
    Name_mode.normal ty slot value_slot_kind s v Hsatisfied Htyped Hanswer
    project_from contents Hproject).
Qed.

Theorem let_project_value_slot_from_known :
  let_project_value_slot_from_known_statement.
Proof.
  intros tenv x declared project_from slot value_slot_kind closure ty s body
    Hty Hanswer.
  apply let_simple_pure; [ reflexivity | ].
  intros statics vars contents Hsat Hnamed.
  cbn [eval_named] in Hnamed.
  destruct (eval_simple vars closure) as [v | ] eqn:Hv; [ | discriminate ].
  simpl in Hnamed.
  exact (answer_has_contents tenv value_slot_kind slot closure ty s statics
    vars v contents project_from Hty Hanswer Hsat Hv Hnamed).
Qed.

Theorem let_project_value_slot_known : let_project_value_slot_known_statement.
Proof.
  intros tenv x declared slot value_slot_kind closure ty s body Hty Hanswer.
  apply let_simple_pure; [ reflexivity | ].
  intros statics vars contents Hsat Hnamed.
  cbn [eval_named] in Hnamed.
  destruct (eval_simple vars closure) as [v | ] eqn:Hv; [ | discriminate ].
  simpl in Hnamed.
  destruct (closure_of statics v) as [[code slot_values] | ] eqn:Hclosure;
    [ | discriminate ].
  simpl in Hnamed.
  destruct (proj1 (closure_in_set_of statics v code slot_values) Hclosure)
    as (at_slot & decls & Hin_set & _).
  apply
    (answer_has_contents tenv value_slot_kind slot closure ty s statics vars v
       contents at_slot Hty Hanswer Hsat Hv).
  unfold project_value_slot. rewrite Hin_set. simpl.
  rewrite Pos.eqb_refl. exact Hnamed.
Qed.

Theorem simplify_project_value_slot_rename :
  simplify_project_value_slot_rename_statement.
Proof.
  intros r tenv value_slot_kind slot ty. unfold simplify_project_value_slot.
  rewrite
    (meet_project_value_slot_simple_rename Name_mode.normal slot
       value_slot_kind r tenv ty).
  destruct (Provers.meet_project_value_slot_simple tenv Name_mode.normal ty
              slot value_slot_kind) as [[s | | ] | ];
    reflexivity.
Qed.
