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


(* The function that rewrites primitives from the types commutes with
   renaming.

   This file is to be reviewed: the action of a renaming on what the
   function decides of a defining expression and of a [Let]
   ([rename_outcome], [rename_let_step]), and the statements.

   [simplify_expr] (FlambdaSimplifyPassSpec.v) takes a typing
   environment and a term. A renaming (FlambdaNominalSpec.renaming)
   permutes variables and continuations; it acts on terms
   ([rename_expr]), on types ([rename_type], FlambdaTypeRenameSpec.v)
   and on typing environments ([rename_typing_env],
   FlambdaEnvRenameSpec.v). The statement
   ([simplify_expr_equivariant_statement]): the function, given the
   renamed environment and the renamed term, answers the renamed
   answer. From the environment that knows nothing, which a renaming
   leaves as it is, the function is a rewrite that commutes with
   renaming in the form of the other rewrites of the validation
   ([simplify_by_types_equivariant_statement], with
   FlambdaValidatorEquivarianceSpec.rewrite_equivariant).

   The statements are equations and have no premise: nothing is asked
   of the environment or of the term. The function makes no name; the
   variables that it puts into a term are those that the environment
   answers for canonical elements and those that the provers answer,
   and both are renamed with the environment
   (FlambdaEnvRenameSpec.v, FlambdaProversRenameSpec.v). Every test
   that it makes answers the same of the renamed environment and
   term: the tests of the invariant and of the well-formed types,
   that a variable is new, that a type speaks of variables in scope,
   and those of the environment of the body of code and of the
   environment of a handler
   (FlambdaSimplifyCodeEnvSpec.code_env_in_rename_statement,
   FlambdaSimplifyHandlersSpec.handler_env_rename_statement).

   The statements for the parts of the function are here too, since
   each is a function that reads names and owes the same equation
   (those for [simplify_simple] and [simplify_simples] are in
   FlambdaSimplifyPrimitivesRenamingSpec.v):
   - [simplify_named]: the outcome is renamed ([rename_outcome]: the
     [Simple] or the type that it holds);
   - [simplify_let]: the step is renamed ([rename_let_step]: the
     defining expression and the environment that it holds);
   - [resolve_value_slot], [declare_variable], [simplify_apply_cont]
     and [simplify_switch];
   - the functions of the section "Where a continuation is applied"
     of FlambdaSimplifyPassSpec.v: [arm_env], [use_env], [leaf_uses]
     (each application is renamed with its environment:
     [rename_application]), [let_env], [find_use] (the environment
     and the binders passed are renamed: [rename_use]),
     [binders_not_free], [gather_uses] (the applications and the
     binders passed are renamed: [rename_gathered]) and [find_one_use]
     (as [find_use]); and, after the function,
     [rewrite_handler_at_use]. The two tests of free names that
     [find_use] and [binders_not_free] make are the validator's, and
     that they commute with renaming is proved in the validator's
     proof file (FlambdaValidatorEquivarianceProofs.v),
     which the proof file of this one imports. Those lemmas are to
     move to a proof file below both before the validation runs the
     function: the validator's proof file will then import this
     one's.

   Nothing is trusted here. The proofs are in
   FlambdaSimplifyPassRenamingProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaSemanticsSpec FlambdaNominalSpec
  FlambdaValidatorEquivarianceSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar FlambdaTypingEnv
  FlambdaTypeRenameSpec FlambdaEnvRenameSpec FlambdaSimplifyPrimitivesSpec
  FlambdaSimplifyPrimitivesRenamingSpec FlambdaSimplifyPassSpec.
Import ListNotations.

(* ======================================================================== *)
(* What the function decides, renamed                                       *)
(* ======================================================================== *)

(* An outcome: the [Simple] or the type that it holds is renamed. *)
Definition rename_outcome (r : renaming) (o : outcome) : outcome :=
  match o with
  | Becomes_invalid => Becomes_invalid
  | Becomes_simple s => Becomes_simple (rename_simple r s)
  | Stays_with_type ty => Stays_with_type (rename_type r ty)
  | Stays_with_kind kind => Stays_with_kind kind
  | Not_known => Not_known
  end.

(* What becomes of a [Let]: the defining expression and the
   environment are renamed. *)
Definition rename_let_step (r : renaming) (step : let_step) : let_step :=
  match step with
  | Step_invalid => Step_invalid
  | Step_let defining_expr tenv =>
    Step_let (rename_named r defining_expr) (rename_typing_env r tenv)
  | Step_stop => Step_stop
  end.

(* ======================================================================== *)
(* The parts of the function                                                *)
(* ======================================================================== *)

Definition simplify_named_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (defining_expr : named),
    simplify_named (rename_typing_env r tenv) (rename_named r defining_expr)
    = rename_outcome r (simplify_named tenv defining_expr).

Definition simplify_let_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (x : variable)
    (defining_expr : named),
    simplify_let (rename_typing_env r tenv) (rename_variable r x)
      (rename_named r defining_expr)
    = rename_let_step r (simplify_let tenv x defining_expr).

(* The kind that a [Let] declares holds no name. *)
Definition resolve_value_slot_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (declared : flambda_kind)
    (defining_expr : named),
    resolve_value_slot (rename_typing_env r tenv) declared
      (rename_named r defining_expr)
    = rename_named r (resolve_value_slot tenv declared defining_expr).

Definition declare_variable_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (x : variable)
    (declared : flambda_kind),
    declare_variable (rename_typing_env r tenv) (rename_variable r x)
      declared
    = option_map (rename_typing_env r) (declare_variable tenv x declared).

Definition simplify_apply_cont_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (ac : apply_cont),
    simplify_apply_cont (rename_typing_env r tenv) (rename_apply_cont r ac)
    = rename_apply_cont r (simplify_apply_cont tenv ac).

(* The arms are renamed as [rename_expr] renames those of a
   [Switch]. *)
Definition simplify_switch_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (scrutinee : simple)
    (arms : list (Z * apply_cont)),
    simplify_switch (rename_typing_env r tenv) (rename_simple r scrutinee)
      (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms)
    = rename_expr r (simplify_switch tenv scrutinee arms).

(* ======================================================================== *)
(* Where a continuation is applied                                          *)
(* ======================================================================== *)

(* The discriminant of an arm holds no name. *)
Definition arm_env_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (scrutinee : simple) (d : Z),
    arm_env (rename_typing_env r tenv) (rename_simple r scrutinee) d
    = rename_typing_env r (arm_env tenv scrutinee d).

(* The metadata of a handler holds no name. *)
Definition use_env_equivariant_statement : Prop :=
  ∀ (r : renaming) (scope tenv : Typing_env.t) (params : list variable)
    (metadata : handler_metadata) (args : list simple),
    use_env (rename_typing_env r scope) (rename_typing_env r tenv)
      (map (rename_variable r) params) metadata (map (rename_simple r) args)
    = option_map (rename_typing_env r)
        (use_env scope tenv params metadata args).

(* An application with its environment: both are renamed. *)
Definition rename_application (r : renaming)
    (application : Typing_env.t * list simple) : Typing_env.t * list simple :=
  (rename_typing_env r (fst application),
   map (rename_simple r) (snd application)).

Definition leaf_uses_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (k : continuation) (e : expr),
    leaf_uses (rename_typing_env r tenv) (rename_continuation r k)
      (rename_expr r e)
    = option_map (map (rename_application r)) (leaf_uses tenv k e).

Definition let_env_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (x : variable)
    (declared : flambda_kind) (defining_expr : named),
    let_env (rename_typing_env r tenv) (rename_variable r x) declared
      (rename_named r defining_expr)
    = option_map (rename_typing_env r)
        (let_env tenv x declared defining_expr).

(* What [find_use] answers: the environment and the binders passed on
   the way are renamed. *)
Definition rename_use (r : renaming)
    (use : Typing_env.t * list variable * list continuation)
    : Typing_env.t * list variable * list continuation :=
  (rename_typing_env r (fst (fst use)),
   map (rename_variable r) (snd (fst use)),
   map (rename_continuation r) (snd use)).

Definition find_use_equivariant_statement : Prop :=
  ∀ (r : renaming) (scope tenv : Typing_env.t) (k : continuation)
    (h : continuation_handler) (body : expr),
    find_use (rename_typing_env r scope) (rename_typing_env r tenv)
      (rename_continuation r k) (rename_handler r h) (rename_expr r body)
    = option_map (rename_use r) (find_use scope tenv k h body).

Definition binders_not_free_equivariant_statement : Prop :=
  ∀ (r : renaming) (xs : list variable) (ks : list continuation)
    (h : continuation_handler),
    binders_not_free (map (rename_variable r) xs)
      (map (rename_continuation r) ks) (rename_handler r h)
    = binders_not_free xs ks h.

(* What [gather_uses] answers: each application is renamed with its
   environment, and the binders passed are renamed. *)
Definition rename_gathered (r : renaming)
    (gathered :
       list (Typing_env.t * list simple) * list variable * list continuation)
    : list (Typing_env.t * list simple) * list variable * list continuation :=
  (map (rename_application r) (fst (fst gathered)),
   map (rename_variable r) (snd (fst gathered)),
   map (rename_continuation r) (snd gathered)).

Definition gather_uses_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (k : continuation)
    (h : continuation_handler) (body : expr),
    gather_uses (rename_typing_env r tenv) (rename_continuation r k)
      (rename_handler r h) (rename_expr r body)
    = option_map (rename_gathered r) (gather_uses tenv k h body).

Definition find_one_use_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (k : continuation)
    (h : continuation_handler) (body : expr),
    find_one_use (rename_typing_env r tenv) (rename_continuation r k)
      (rename_handler r h) (rename_expr r body)
    = option_map (rename_use r) (find_one_use tenv k h body).

(* ======================================================================== *)
(* The function                                                             *)
(* ======================================================================== *)

Definition simplify_expr_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (e : expr),
    simplify_expr (rename_typing_env r tenv) (rename_expr r e)
    = rename_expr r (simplify_expr tenv e).

(* From the environment that knows nothing it is a rewrite of terms
   that commutes with renaming. *)
Definition simplify_by_types_equivariant_statement : Prop :=
  rewrite_equivariant simplify_by_types.

(* The handler rewritten under the environment of its one
   application: that of the renamed handler and body, under the
   renamed environment, is the renamed handler. *)
Definition rewrite_handler_at_use_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (k : continuation)
    (h : continuation_handler) (body : expr),
    rewrite_handler_at_use (rename_typing_env r tenv)
      (rename_continuation r k) (rename_handler r h) (rename_expr r body)
    = option_map (rename_handler r) (rewrite_handler_at_use tenv k h body).
