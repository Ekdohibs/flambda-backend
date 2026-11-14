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


(* The arms of a [Switch], each rewritten under what its arm knows.

   This file is to be reviewed: the two definitions and the
   statements. Nothing runs the function yet: the function that
   rewrites a term from the types (FlambdaSimplifyPassSpec.simplify_expr)
   still calls [simplify_switch] of that file, which leaves the arms
   as they are written.

   What Simplify does (middle_end/flambda2/simplify/
   simplify_switch_expr.ml, [simplify_arm]): for each arm it makes an
   environment in which the scrutinee is the number of the arm, and
   replaces the arguments of the [Apply_cont] of the arm by their
   canonical elements there. So [switch x with 0 -> k x | 1 -> k' x]
   becomes [switch x with 0 -> k 0 | 1 -> k' 1]. Simplify makes that
   environment by a meet, and drops the arm where the meet is bottom;
   FlambdaSimplifySwitchArmsSpec.v has that function, with the meet as
   a parameter and with what the contract of the meet asks: a static
   environment that is valid and an environment in which a meet may
   be made.

   What is here is the same rewrite with the environment of an arm
   that FlambdaSimplifyPassSpec.v already makes for the handler of a
   continuation that one arm applies ([arm_env]): the variable of the
   scrutinee is made an alias of the number, with no meet. No arm is
   dropped. So the statements have no premise but that the state
   satisfies the environment, as those of [simplify_expr] have, and
   the function can take the place of [simplify_switch] in it.

   - [simplify_arm_args] rewrites the arguments of one arm under
     [arm_env]; what it answers is equivalent to the arm in every
     state that satisfies the environment of the [Switch] and in
     which the scrutinee is the number of the arm
     ([simplify_arm_args_equivalent_statement]).
   - Of the machine: the arms of a [Switch] may each be replaced by
     an arm for the same number that refines it in the states in
     which it is taken ([switch_arms_refined_under_statement]). It
     has the form of [switch_arms_refined_statement] of
     FlambdaSimplifySwitchArmsSpec.v, with the valid static
     environment taken out of both sides: an arm is asked to refine
     in every state that satisfies the environment and takes it,
     whatever its static environment, which is more than is asked
     there; and the [Switch] is then refined in every state that
     satisfies the environment, which is more than is concluded
     there. Neither statement is an instance of the other. The proof
     there does not use that the static environment is valid.
   - [simplify_switch_and_arms] is [simplify_switch] with the arms
     that stay rewritten so. What it answers refines the [Switch]
     under the environment
     ([simplify_switch_and_arms_refines_statement]). Refinement is
     what is stated, as of [simplify_switch], and what
     [simplify_expr] needs; whether the two terms are equivalent is
     not stated.
   - Both functions commute with a renaming of variables and
     continuations ([simplify_arm_args_equivariant_statement],
     [simplify_switch_and_arms_equivariant_statement]).
   - Two terms on which the function is computed
     ([example_scrutinee_under_arms_statement],
     [example_tagging_under_arms_statement]).

   What this does not do. It replaces a use of the scrutinee itself.
   A value of which the scrutinee is the untagging keeps its variable
   in the arm ([example_tagging_under_arms_statement]): the
   environment of the arm says nothing of it. Simplify keeps the
   variable there too: it knows the type of the tagged value in the
   arm, and the canonical element of that value is still the
   variable.

   The proofs are in FlambdaSimplifyArmsProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaValueSpec FlambdaPrimitiveModel FlambdaPrimitiveSpec
  FlambdaSemanticsSpec FlambdaNominalSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvExamplesSpec FlambdaTypingEnvRefinementSpec
  FlambdaEnvRenameSpec FlambdaSimplifyPrimitivesSpec FlambdaSimplifyPassSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* The function                                                             *)
(* ======================================================================== *)

(* One arm: the arguments of its [Apply_cont] are replaced by their
   canonical elements in the environment of the arm. *)
Definition simplify_arm_args (tenv : Typing_env.t) (scrutinee : simple)
    (arm : Z * apply_cont) : Z * apply_cont :=
  (fst arm, simplify_apply_cont (arm_env tenv scrutinee (fst arm)) (snd arm)).

(* A [Switch] whose scrutinee has a constant for canonical element is
   the [Apply_cont] of its arm, as [simplify_switch] makes it; any
   other keeps its scrutinee and has each arm rewritten. *)
Definition simplify_switch_and_arms (tenv : Typing_env.t) (scrutinee : simple)
    (arms : list (Z * apply_cont)) : expr :=
  match simplify_switch tenv scrutinee arms with
  | Switch scrutinee' arms' =>
    Switch scrutinee' (map (simplify_arm_args tenv scrutinee') arms')
  | (Let _ _ _ _ | Let_static _ _ | Let_cont_non_recursive _ _ _
    | Let_cont_recursive _ _ | Apply _ | Apply_cont _ | Invalid) as e =>
    e
  end.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* The arm rewritten is equivalent to the arm where the arm is
   taken. *)
Definition simplify_arm_args_equivalent_statement : Prop :=
  ∀ (tenv : Typing_env.t) (scrutinee : simple) (arm : Z * apply_cont)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record),
    env_satisfies_strictly tenv statics vars ->
    eval_simple vars scrutinee
    = Some (Value_const (Naked_immediate (fst arm))) ->
    equivalent (State (Apply_cont (snd arm)) vars conts statics hp rd)
      (State (Apply_cont (snd (simplify_arm_args tenv scrutinee arm))) vars
         conts statics hp rd).

(* An arm refines another in the states that satisfy the environment
   and in which the scrutinee is its number. *)
Definition arm_refines_where_taken_under (tenv : Typing_env.t)
    (scrutinee : simple) (i : Z) (ac ac' : apply_cont) : Prop :=
  ∀ (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
    (rd : read_record),
    env_satisfies_strictly tenv statics vars ->
    eval_simple vars scrutinee = Some (Value_const (Naked_immediate i)) ->
    refines (State (Apply_cont ac) vars conts statics hp rd)
      (State (Apply_cont ac') vars conts statics hp rd).

(* The arms of a [Switch] may each be replaced by an arm for the same
   number that refines it where it is taken. *)
Definition switch_arms_refined_under_statement : Prop :=
  ∀ (tenv : Typing_env.t) (scrutinee : simple)
    (arms arms' : list (Z * apply_cont)),
    Forall2
      (fun arm arm' =>
         fst arm' = fst arm
         ∧ arm_refines_where_taken_under tenv scrutinee (fst arm) (snd arm)
             (snd arm'))
      arms arms' ->
    expr_refines_under tenv (Switch scrutinee arms) (Switch scrutinee arms').

(* What the function answers refines the [Switch], in every state
   that satisfies the environment. *)
Definition simplify_switch_and_arms_refines_statement : Prop :=
  ∀ (tenv : Typing_env.t) (scrutinee : simple) (arms : list (Z * apply_cont)),
    expr_refines_under tenv (Switch scrutinee arms)
      (simplify_switch_and_arms tenv scrutinee arms).

(* Renaming. The number of an arm holds no name. *)
Definition simplify_arm_args_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (scrutinee : simple)
    (arm : Z * apply_cont),
    simplify_arm_args (rename_typing_env r tenv) (rename_simple r scrutinee)
      (fst arm, rename_apply_cont r (snd arm))
    = (fst (simplify_arm_args tenv scrutinee arm),
       rename_apply_cont r (snd (simplify_arm_args tenv scrutinee arm))).

(* The arms are renamed as [rename_expr] renames those of a
   [Switch]. *)
Definition simplify_switch_and_arms_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (scrutinee : simple)
    (arms : list (Z * apply_cont)),
    simplify_switch_and_arms (rename_typing_env r tenv)
      (rename_simple r scrutinee)
      (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms)
    = rename_expr r (simplify_switch_and_arms tenv scrutinee arms).

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

(* [switch x | 0 -> k (x, 10) | 1 -> k (x, 20)], where nothing is
   known of the naked immediate [x] (it is bound to [Opaque_identity]
   of a constant): the arms become [k (0, 10)] and [k (1, 20)]. *)
Definition example_scrutinee_under_arms_statement : Prop :=
  ∀ k : continuation,
    match
      let_env Typing_env.create example_x naked_immediate_kind
        (example_opaque naked_immediate_kind (Naked_immediate 0))
    with
    | Some tenv =>
      Some
        (simplify_switch_and_arms tenv (Var example_x)
           [(0,
             Apply_cont_expr k [Var example_x; Const (Tagged_immediate 10)]
               None);
            (1,
             Apply_cont_expr k [Var example_x; Const (Tagged_immediate 20)]
               None)])
    | None => None
    end
    = Some
        (Switch (Var example_x)
           [(0,
             Apply_cont_expr k
               [Const (Naked_immediate 0); Const (Tagged_immediate 10)] None);
            (1,
             Apply_cont_expr k
               [Const (Naked_immediate 1); Const (Tagged_immediate 20)] None)]).

(* [let y = opaque 1 in let x = Untag_immediate y in
   switch x | 0 -> k y | 1 -> k y]: the arms give the tagged value
   [y], of which [x] is the untagging, and keep it. *)
Definition example_tagging_under_arms_statement : Prop :=
  ∀ k : continuation,
    match
      let_env Typing_env.create example_y Value
        (example_opaque Value (Tagged_immediate 1))
    with
    | Some tenv_y =>
      match
        let_env tenv_y example_x naked_immediate_kind
          (Prim (Unary Untag_immediate) [Var example_y])
      with
      | Some tenv =>
        Some
          (simplify_switch_and_arms tenv (Var example_x)
             [(0, Apply_cont_expr k [Var example_y] None);
              (1, Apply_cont_expr k [Var example_y] None)])
      | None => None
      end
    | None => None
    end
    = Some
        (Switch (Var example_x)
           [(0, Apply_cont_expr k [Var example_y] None);
            (1, Apply_cont_expr k [Var example_y] None)]).
