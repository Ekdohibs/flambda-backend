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

(* The terms of Flambda 2 as things that contain names, and their binders
   as name abstractions: instances of the interface of
   FlambdaNameAbstractionSpec.v.

   This file is to be reviewed: the instances (which action, which free
   names and which equality each has), and the statements.

   FlambdaNameAbstractionSpec.v has the interface of the compiler's
   nominal/ directory with its laws, and mentions no term. Here it meets
   the model. An instance is for one sort of name: the variables or the
   continuations. A renaming of one sort is a renaming of
   FlambdaNominalSpec.v whose other permutation is empty.

   What has no binder is an instance with [=] for its equality: a
   [simple], a [named], a static constant, and an environment of
   variables, whose names are the variables it binds. A value holds no
   variable and no continuation: it is [Nominal.no_names].

   Expressions. The equality of expressions is alpha-equivalence
   (FlambdaAlphaSpec.alpha_equivalent), and their free names are those
   of FlambdaBindersSpec.v. The laws hold of the expressions whose code
   is closed, and not of the others: a binder around code that is not
   closed counts as binding what the code leaves free, but the code
   does not see it when it runs, so a renaming that fixes the free
   names of such a term can change it. Code is closed in every
   well-formed term (decision D31). So the instance is on
   [closed_expr], the expressions whose code is closed. A renaming
   keeps code closed (FlambdaAlphaSpec.code_closed_equivariant_statement),
   so the action is [rename_expr]; the definition tests the result,
   since a definition here cannot use a proof, and
   [rename_closed_expr_term_statement] says that the test never fails.

   The binders. Each binder of the model is a name abstraction of the
   interface over closed expressions, and so has all its laws
   (FlambdaNameAbstractionSpec.abstraction_nominal_statement,
   [pattern_match_pair_statement]) with no further proof. What is
   proved here is that the term built from an abstraction depends on
   the abstraction only up to its equality: two abstractions that are
   equal in the sense of the interface give alpha-equivalent terms.
   This is the sense in which a binder of the model is a
   [Name_abstraction.t]:
   - the variable of a [Let] ([let_abstraction_statement]);
   - the continuation of a non-recursive [Let_cont]
     ([let_cont_abstraction_statement]);
   - the parameters of a handler ([handler_abstraction_statement]);
   - the parameters and [my_closure] of code, and its return and
     exception continuations ([code_variables_abstraction_statement],
     [code_continuations_abstraction_statement]).
   The continuations of a recursive [Let_cont] bind in its handlers and
   in its body together; that scope is not one expression, and it is
   not stated here as an abstraction
   (FlambdaAlphaSpec.alpha_let_cont_recursive_canonical_statement says
   that a renaming of them gives an alpha-equivalent term).

   A state is not made an instance: the action on states and the
   equivariance of the machine are in FlambdaEquivarianceSpec.v, stated
   directly.

   Nothing is trusted here. The proofs are in
   FlambdaNominalInstancesProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaBindersSpec FlambdaSemanticsSpec FlambdaNominalSpec
  FlambdaAlphaSpec FlambdaNameAbstractionSpec.
Import ListNotations.
Import Nominal.

(* ======================================================================== *)
(* Renamings of one sort                                                    *)
(* ======================================================================== *)

Definition variables_renaming (p : permutation) : renaming := Renaming p [].

Definition continuations_renaming (p : permutation) : renaming :=
  Renaming [] p.

(* ======================================================================== *)
(* What has no binder                                                       *)
(* ======================================================================== *)

Definition simple_names : contains_names simple :=
  Contains_names (fun p s => rename_simple (variables_renaming p) s)
    free_variables_of_simple eq.

Definition named_names : contains_names named :=
  Contains_names (fun p d => rename_named (variables_renaming p) d)
    free_variables_of_named eq.

Definition static_const_names : contains_names static_const :=
  Contains_names
    (fun p const => rename_static_const (variables_renaming p) const)
    free_variables_of_static_const eq.

(* An environment: the names it binds, shadowed ones included. *)
Definition env_names : contains_names env :=
  Contains_names (fun p vars => rename_env (variables_renaming p) vars)
    (fun vars => map fst vars) eq.

Definition simple_nominal_statement : Prop := nominal simple_names.

Definition named_nominal_statement : Prop := nominal named_names.

Definition static_const_nominal_statement : Prop :=
  nominal static_const_names.

Definition env_nominal_statement : Prop := nominal env_names.

(* ======================================================================== *)
(* Expressions whose code is closed                                         *)
(* ======================================================================== *)

Definition closed_expr : Type := { e : expr | code_closed e = true }.

(* The action is [rename_expr]. The test is of what a renaming always
   keeps ([rename_closed_expr_term_statement]). *)
Definition rename_closed_expr (r : renaming) (e : closed_expr)
    : closed_expr :=
  match bool_dec (code_closed (rename_expr r (proj1_sig e))) true with
  | left closed => exist _ (rename_expr r (proj1_sig e)) closed
  | right _ => e
  end.

Definition rename_closed_expr_term_statement : Prop :=
  ∀ (r : renaming) (e : closed_expr),
    proj1_sig (rename_closed_expr r e) = rename_expr r (proj1_sig e).

(* Two closed expressions that are the same expression are equal. *)
Definition closed_expr_equal_statement : Prop :=
  ∀ e e' : closed_expr, proj1_sig e = proj1_sig e' -> e = e'.

Definition alpha_equivalent_closed (e e' : closed_expr) : Prop :=
  alpha_equivalent (proj1_sig e) (proj1_sig e').

Definition closed_expr_variables : contains_names closed_expr :=
  Contains_names (fun p e => rename_closed_expr (variables_renaming p) e)
    (fun e => free_variables_of_expr (proj1_sig e)) alpha_equivalent_closed.

Definition closed_expr_continuations : contains_names closed_expr :=
  Contains_names (fun p e => rename_closed_expr (continuations_renaming p) e)
    (fun e => free_continuations_of_expr (proj1_sig e))
    alpha_equivalent_closed.

Definition closed_expr_variables_nominal_statement : Prop :=
  nominal closed_expr_variables.

Definition closed_expr_continuations_nominal_statement : Prop :=
  nominal closed_expr_continuations.

(* ======================================================================== *)
(* The binders are name abstractions                                        *)
(* ======================================================================== *)

(* The variable of a [Let] and its body. The kind and the defining
   expression are the same on both sides. *)
Definition let_abstraction_statement : Prop :=
  ∀ (kind : flambda_kind) (defining_expr : named)
    (a a' : name_abstraction variable closed_expr),
    equal_abstraction single_name closed_expr_variables a a' ->
    alpha_equivalent
      (Let (bound a) kind defining_expr (proj1_sig (scope a)))
      (Let (bound a') kind defining_expr (proj1_sig (scope a'))).

(* The continuation of a non-recursive [Let_cont] and its body. *)
Definition let_cont_abstraction_statement : Prop :=
  ∀ (handler : continuation_handler)
    (a a' : name_abstraction continuation closed_expr),
    equal_abstraction single_name closed_expr_continuations a a' ->
    alpha_equivalent
      (Let_cont_non_recursive (bound a) handler (proj1_sig (scope a)))
      (Let_cont_non_recursive (bound a') handler (proj1_sig (scope a'))).

(* The parameters of a handler and the handler. The metadata, which has
   no names in it, is the same on both sides. *)
Definition handler_abstraction_statement : Prop :=
  ∀ (k : continuation) (metadata : handler_metadata) (body : expr)
    (a a' : name_abstraction (list variable) closed_expr),
    equal_abstraction list_of_names closed_expr_variables a a' ->
    alpha_equivalent
      (Let_cont_non_recursive k
         (Handler (bound a) metadata (proj1_sig (scope a))) body)
      (Let_cont_non_recursive k
         (Handler (bound a') metadata (proj1_sig (scope a'))) body).

(* Code, as an abstraction of [my_closure] and the parameters over the
   body, the two continuations and the metadata being the same on both
   sides. The codes are closed. *)
Definition code_of_variables (return_continuation exn_continuation
    : continuation) (metadata : code_metadata)
    (a : name_abstraction (list variable) closed_expr) : option code :=
  match bound a with
  | my_closure :: params =>
    Some
      (Code params my_closure return_continuation exn_continuation metadata
         (proj1_sig (scope a)))
  | [] => None
  end.

Definition code_variables_abstraction_statement : Prop :=
  ∀ (before : list static_binding) (id : code_id)
    (return_continuation exn_continuation : continuation)
    (metadata : code_metadata)
    (a a' : name_abstraction (list variable) closed_expr) (c c' : code)
    (after : list static_binding) (body : expr),
    code_of_variables return_continuation exn_continuation metadata a
    = Some c ->
    code_of_variables return_continuation exn_continuation metadata a'
    = Some c' ->
    code_is_closed c = true ->
    code_is_closed c' = true ->
    equal_abstraction list_of_names closed_expr_variables a a' ->
    alpha_equivalent (Let_static (before ++ Bind_code id c :: after) body)
      (Let_static (before ++ Bind_code id c' :: after) body).

(* Code, as an abstraction of its return and exception continuations
   over the body, the variables and the metadata being the same on both
   sides. *)
Definition code_of_continuations (params : list variable)
    (my_closure : variable) (metadata : code_metadata)
    (a : name_abstraction (list continuation) closed_expr) : option code :=
  match bound a with
  | [return_continuation; exn_continuation] =>
    Some
      (Code params my_closure return_continuation exn_continuation metadata
         (proj1_sig (scope a)))
  | _ => None
  end.

Definition code_continuations_abstraction_statement : Prop :=
  ∀ (before : list static_binding) (id : code_id) (params : list variable)
    (my_closure : variable) (metadata : code_metadata)
    (a a' : name_abstraction (list continuation) closed_expr) (c c' : code)
    (after : list static_binding) (body : expr),
    code_of_continuations params my_closure metadata a = Some c ->
    code_of_continuations params my_closure metadata a' = Some c' ->
    code_is_closed c = true ->
    code_is_closed c' = true ->
    equal_abstraction list_of_names closed_expr_continuations a a' ->
    alpha_equivalent (Let_static (before ++ Bind_code id c :: after) body)
      (Let_static (before ++ Bind_code id c' :: after) body).
