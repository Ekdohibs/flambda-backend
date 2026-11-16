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

(* Simplifying the projection of a value slot from the type of the
   closure: the rewrite to a [Simple], with nothing assumed of kinds.

   This file is to be reviewed: the definition and the statements,
   against middle_end/flambda2/simplify/simplify_unary_primitive.ml
   ([simplify_project_value_slot], the case [Known_result]). It follows
   FlambdaSimplifyProjectionsSpec.v, whose statement about this
   projection ([let_project_value_slot_statement]) has both answers of
   the prover, [Known_result] and [Invalid], and the premise that the
   projection, where it has a value, has one of the kind of the value
   slot. A value slot of the model has no kind, so nothing gives that
   premise, and the function that puts the rewrites together
   (FlambdaSimplifyPassSpec.v) does not use that statement.

   The premise is needed for the answer [Invalid] alone
   (FlambdaProversSpec.v, [meet_project_value_slot_simple_known_statement];
   discrepancy T6-4). Here are the two statements for the answer
   [Known_result], with no premise but the type of the closure: where
   [Provers.meet_project_value_slot_simple], asked at any kind, answers
   a [Simple], the [Let] of the projection refines the [Let] of that
   [Simple], under the typing environment.
   - [Project_value_slot_from], which names the function slot that the
     closure is projected from, as the compiler's primitive does
     ([let_project_value_slot_from_known_statement]). Simplify does not
     give that function slot to the prover, and neither does this: the
     prover answers what the value slot of the closures of the type
     holds, at whatever function slot they are. In the model the
     projection from a closure at another function slot than the one
     named has no value (FlambdaValueSpec.v, [project_value_slot];
     decision D3), and the [Let] of the [Simple] then refines a term
     that is undefined. So the statement is a refinement and not an
     equivalence.
   - [Project_value_slot], the projection of the model that names no
     function slot ([let_project_value_slot_known_statement]).
   [simplify_project_value_slot] is the answer as a function, for the
   function that puts the rewrites together: the [Simple], if the prover
   answers one. The kind to ask at is its caller's to give: in the
   compiler it is the kind of the value slot, which is the kind of the
   variable of the [Let]. It commutes with renaming
   ([simplify_project_value_slot_rename_statement]).

   Not here: the answer [Invalid] (the statement of
   FlambdaSimplifyProjectionsSpec.v, with its premise), and the meet
   with a closure that has at least this value slot, where the prover
   answers neither ([Simplify_common.simplify_projection]).

   The proofs are in FlambdaSimplifyValueSlotsProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec FlambdaNominalSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaTypingEnv
  FlambdaTypingEnvSpec FlambdaTypingEnvRefinementSpec FlambdaTypeRenameSpec
  FlambdaEnvRenameSpec FlambdaProvers FlambdaProversSpec
  FlambdaSimplifyPrimitivesSpec.
Import ListNotations.

(* ======================================================================== *)
(* The rewrite                                                              *)
(* ======================================================================== *)

(* [Simplify_unary_primitive.simplify_project_value_slot], the case
   [Known_result]: the [Simple] that the prover answers for the value
   slot [slot] of a closure of the type [ty], asked at the kind
   [value_slot_kind] and at the normal name mode. [None] where the
   prover answers anything else. *)
Definition simplify_project_value_slot (tenv : Typing_env.t)
    (value_slot_kind : flambda_kind) (slot : value_slot) (ty : flambda_type)
    : option simple :=
  match
    Provers.meet_project_value_slot_simple tenv Name_mode.normal ty slot
      value_slot_kind
  with
  | Some (Meet_shortcut.Known_result s) => Some s
  | Some Meet_shortcut.Invalid | Some Meet_shortcut.Need_meet | None => None
  end.

(* The projection that names the function slot of the closure. *)
Definition let_project_value_slot_from_known_statement : Prop :=
  ∀ (tenv : Typing_env.t) (x : variable) (declared : flambda_kind)
    (project_from : function_slot) (slot : value_slot)
    (value_slot_kind : flambda_kind) (closure : simple) (ty : flambda_type)
    (s : simple) (body : expr),
    argument_type tenv closure ty ->
    simplify_project_value_slot tenv value_slot_kind slot ty = Some s ->
    expr_refines_under tenv
      (Let x declared (Project_value_slot_from project_from slot closure)
         body)
      (Let x declared (Simple s) body).

(* The projection that names no function slot. *)
Definition let_project_value_slot_known_statement : Prop :=
  ∀ (tenv : Typing_env.t) (x : variable) (declared : flambda_kind)
    (slot : value_slot) (value_slot_kind : flambda_kind) (closure : simple)
    (ty : flambda_type) (s : simple) (body : expr),
    argument_type tenv closure ty ->
    simplify_project_value_slot tenv value_slot_kind slot ty = Some s ->
    expr_refines_under tenv
      (Let x declared (Project_value_slot slot closure) body)
      (Let x declared (Simple s) body).

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

(* The answer for the renamed environment and the renamed type is the
   answer renamed. A kind and a value slot contain no variable and no
   continuation, and are as they were. *)
Definition simplify_project_value_slot_rename_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (value_slot_kind : flambda_kind)
    (slot : value_slot) (ty : flambda_type),
    simplify_project_value_slot (rename_typing_env r tenv) value_slot_kind
      slot (rename_type r ty)
    = option_map (rename_simple r)
        (simplify_project_value_slot tenv value_slot_kind slot ty).
