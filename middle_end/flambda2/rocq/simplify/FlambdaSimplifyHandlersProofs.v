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


(* Proofs of the statements of FlambdaSimplifyHandlersSpec.v. Nothing
   here needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaPrimitiveModel
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaWellFormedSpec
  FlambdaWellFormedProofs FlambdaContextProofs FlambdaNominalSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaTypingEnv
  FlambdaTypingEnvSpec FlambdaTypingEnvProofs FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvOperationsSpec FlambdaTypingEnvOperationsProofs
  FlambdaTypingEnvExamplesSpec FlambdaTypingEnvRefinementSpec
  FlambdaTypingEnvRefinementProofs FlambdaTypeRenameSpec
  FlambdaEnvRenameSpec FlambdaEnvRenameProofs FlambdaSimplifyCodeEnvSpec
  FlambdaSimplifyCodeEnvProofs FlambdaSimplifyHandlersSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* The environment of a handler                                             *)
(* ======================================================================== *)

Theorem handler_env_satisfied : handler_env_satisfied_statement.
Proof.
  intros tenv params metadata tenv_handler statics vars vs handler_vars
    Henv Hsat Hkinds Hbind.
  unfold handler_env in Henv.
  destruct (Nat.eqb (length (hm_params_kinds metadata)) (length params))
    eqn:Hlength;
    [ | discriminate ].
  apply Nat.eqb_eq in Hlength.
  apply (define_variables_checked_satisfied
           (combine params (hm_params_kinds metadata)) tenv tenv_handler
           statics vars handler_vars vs Henv).
  - rewrite <- Hkinds. apply kinds_of_values.
    exact (bind_params_length params vs vars handler_vars Hbind).
  - rewrite (map_fst_combine params _ Hlength). exact Hbind.
  - exact Hsat.
Qed.

(* ======================================================================== *)
(* A handler rewritten                                                      *)
(* ======================================================================== *)

(* A handler that refines under its environment refines where it
   runs. *)
Lemma handler_typed_where (tenv : Typing_env.t) (statics : static_env)
    (vars : env) (params : list variable) (metadata : handler_metadata)
    (handler handler' : expr) (tenv_handler : Typing_env.t) :
  handler_env tenv params metadata = Some tenv_handler ->
  expr_refines_under tenv_handler handler handler' ->
  env_satisfies_strictly tenv statics vars ->
  handler_refines_where (statics_extend statics) params
    (hm_params_kinds metadata) vars handler handler'.
Proof.
  intros Henv Hhandler Hsat vs handler_vars conts' statics' hp rd Hextend Hkinds
    Hbind.
  apply Hhandler.
  exact (handler_env_satisfied tenv params metadata tenv_handler statics'
           vars vs handler_vars Henv
           (env_satisfies_strictly_statics_extend tenv statics statics' vars
              Hextend Hsat)
           Hkinds Hbind).
Qed.

Theorem let_cont_handler_typed : let_cont_handler_typed_statement.
Proof.
  intros tenv k params metadata handler handler' body tenv_handler Henv
    Hhandler vars conts statics hp rd Hsat.
  apply (let_cont_handler_refines_where (statics_extend statics)
           (fun s s' Hs Hs' => statics_extend_trans statics s s' Hs Hs'));
    [ apply statics_extend_refl | ].
  exact (handler_typed_where tenv statics vars params metadata handler
           handler' tenv_handler Henv Hhandler Hsat).
Qed.

Theorem let_cont_recursive_handler_typed
    : let_cont_recursive_handler_typed_statement.
Proof.
  intros tenv before k params metadata handler handler' after body
    tenv_handler Henv Hbinds Hhandler vars conts statics hp rd Hsat.
  pose proof (handler_typed_where tenv statics vars params metadata handler
                handler' tenv_handler Henv Hhandler Hsat)
    as Hwhere.
  apply (let_cont_recursive_handlers_refine_where (statics_extend statics)
           (fun s s' Hs Hs' => statics_extend_trans statics s s' Hs Hs'));
    [ apply statics_extend_refl
    | unfold handlers_bind_statics; rewrite !existsb_app; simpl;
      rewrite Hbinds; reflexivity
    | | ];
    (apply Forall2_app; [ apply handlers_related_refl | ]);
    (apply Forall2_cons; [ | apply handlers_related_refl ]);
    (split; [ reflexivity | ]); apply Handler_related; exact Hwhere.
Qed.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

Theorem handler_env_rename : handler_env_rename_statement.
Proof.
  intros r tenv params metadata. unfold handler_env. rewrite length_map.
  destruct (Nat.eqb (length (hm_params_kinds metadata)) (length params));
    [ | reflexivity ].
  rewrite combine_map_fst.
  exact (define_variables_checked_renamed r
           (combine params (hm_params_kinds metadata)) tenv).
Qed.
