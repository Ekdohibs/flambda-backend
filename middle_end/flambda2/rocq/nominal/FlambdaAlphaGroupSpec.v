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

(* The continuations of a recursive [Let_cont]: a name abstraction.

   This file is to be reviewed: [alpha_group], [list_names],
   [handler_continuations], [group_handler_continuations],
   [group_continuations], [handler_of], [group_term] and the
   statements; and against the compiler's [Recursive_let_cont_handlers]
   (terms/flambda.ml), a name abstraction of the continuations of the
   group over its handlers and its body, which compare/compare.ml opens
   in two terms at the same fresh continuations
   ([Recursive_let_cont_handlers.pattern_match_pair]).

   What is bound where. [Let_cont_recursive handlers body] binds the
   continuations of [handlers], and they are in scope in [body] and in
   every handler. Each handler binds its parameters, and they are in
   scope in that handler only.

   This is the one binder of the terms that FlambdaAlphaFreshSpec.v and
   FlambdaNominalInstancesSpec.v leave out, because its scope is not
   one expression. Two things are said of it here.

   - The rule by which two groups are compared, read both ways
     ([alpha_let_cont_recursive_iff_statement]): two recursive
     [Let_cont]s are alpha-equivalent exactly when they have as many
     handlers and, once the continuations of both groups are exchanged
     for the same fresh continuations in the handlers and in the body
     together, the handlers agree one by one, in order, as name
     abstractions of their parameters
     (FlambdaAlphaFreshSpec.alpha_handlers), and the bodies are
     alpha-equivalent. It is for terms whose code is closed and whose
     binders repeat no name, as the statements of
     FlambdaAlphaFreshSpec.v are, and its two directions use the two
     conditions as they do there: from right to left, closed code and
     that the continuations of each group and the parameters of each
     handler repeat no name; from left to right, both conditions on
     both terms, since the proof goes through the comparison.

   - The group is a name abstraction in the sense of
     FlambdaNameAbstractionSpec.v
     ([group_abstraction_statement]): two abstractions that are equal
     in the sense of the interface give alpha-equivalent terms. The
     bound names are the list of the continuations of the group. The
     scope is the handlers, in order, and the body. A handler is, as
     in FlambdaNominalInstancesSpec.handler_abstraction_statement, an
     abstraction of its parameters over an expression whose code is
     closed, with its metadata, which has no name in it; two handlers
     are equal when they have the same metadata and are equal as such
     abstractions, so that the parameters of each may be renamed.
     With this the laws of abstractions, and that two equal
     abstractions opened at the same names have equal scopes
     (FlambdaNameAbstractionSpec.pattern_match_pair_statement), hold
     of the group with no further proof, the scope being a thing that
     contains names with all its laws
     ([group_continuations_nominal_statement]). No name is asked to
     be distinct: the equality of the interface is by permutations,
     and a permutation keeps apart what a list repeats and what it
     does not. For groups whose binders repeat no name the converse
     holds too ([group_abstraction_iff_statement]): the abstractions
     are equal exactly when the terms are alpha-equivalent, that is,
     exactly when the groups agree at fresh continuations.

   The interface has pairs and no lists. A list of things that contain
   names is given here ([list_names]), for any such things, with its
   laws ([list_nominal_statement]).

   Nothing is trusted here. The proofs are in FlambdaAlphaGroupProofs.v,
   which needs no review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaBindersSpec
  FlambdaNominalSpec FlambdaAlphaSpec FlambdaNameAbstractionSpec
  FlambdaNominalInstancesSpec FlambdaFreshQuantifierSpec
  FlambdaFreshBindersSpec FlambdaAlphaFreshSpec.
Import ListNotations.
Import Nominal.

(* ======================================================================== *)
(* Two groups compared at fresh continuations                               *)
(* ======================================================================== *)

(* Two groups and their bodies agree: they have as many handlers, and
   once the continuations of both groups are exchanged, in the handlers
   and in the bodies, for the same fresh continuations, the handlers
   agree one by one in the order of the groups and the bodies are
   alpha-equivalent. *)
Definition alpha_group
    (handlers handlers' : list (continuation * continuation_handler))
    (body body' : expr) : Prop :=
  length handlers = length handlers'
  ∧ И[length handlers] fresh,
      Forall2 alpha_handlers
        (map snd
           (rename_handlers (swap_continuations (map fst handlers) fresh)
              handlers))
        (map snd
           (rename_handlers (swap_continuations (map fst handlers') fresh)
              handlers'))
      ∧ alpha_equivalent
          (rename_expr (swap_continuations (map fst handlers) fresh) body)
          (rename_expr (swap_continuations (map fst handlers') fresh) body').

Definition alpha_let_cont_recursive_iff_statement : Prop :=
  ∀ (handlers handlers' : list (continuation * continuation_handler))
    (body body' : expr),
    code_closed (Let_cont_recursive handlers body) = true ->
    code_closed (Let_cont_recursive handlers' body') = true ->
    binders_distinct (Let_cont_recursive handlers body) = true ->
    binders_distinct (Let_cont_recursive handlers' body') = true ->
    (alpha_equivalent (Let_cont_recursive handlers body)
       (Let_cont_recursive handlers' body')
     <-> alpha_group handlers handlers' body body').

(* ======================================================================== *)
(* Lists of things that contain names                                       *)
(* ======================================================================== *)

(* The action is on each element, the free names are those of the
   elements, and two lists are equal when they have as many elements
   and those are equal in order. *)
Definition list_names {T : Type} (N : contains_names T)
    : contains_names (list T) :=
  Contains_names (fun p ts => map (apply_renaming N p) ts)
    (fun ts => flat_map (free_names N) ts) (Forall2 (equal N)).

Definition list_nominal_statement : Prop :=
  ∀ (T : Type) (N : contains_names T), nominal N -> nominal (list_names N).

(* ======================================================================== *)
(* The group as a name abstraction                                          *)
(* ======================================================================== *)

(* A handler without its metadata: its parameters and its expression,
   whose code is closed. *)
Definition abstract_handler : Type :=
  name_abstraction (list variable) closed_expr.

(* It is a thing that contains continuations. A renaming of
   continuations acts on its expression, its free continuations are
   those of its expression, and two are equal when they are equal as
   abstractions of their parameters. *)
Definition handler_continuations : contains_names abstract_handler :=
  Contains_names
    (fun p h =>
       create (bound h) (apply_renaming closed_expr_continuations p (scope h)))
    (fun h => free_names closed_expr_continuations (scope h))
    (equal_abstraction list_of_names closed_expr_variables).

Definition handler_continuations_nominal_statement : Prop :=
  nominal handler_continuations.

(* A handler of a group: its metadata, which has no name in it, and the
   rest. Two are equal when they have the same metadata and the rest
   is equal. *)
Definition group_handler : Type := handler_metadata * abstract_handler.

Definition handler_of (h : group_handler) : continuation_handler :=
  Handler (bound (snd h)) (fst h) (proj1_sig (scope (snd h))).

Definition group_handler_continuations : contains_names group_handler :=
  pair_names (no_names handler_metadata) handler_continuations.

(* The scope of the continuations of a group: the handlers, in order,
   and the body. *)
Definition group_scope : Type := list group_handler * closed_expr.

Definition group_continuations : contains_names group_scope :=
  pair_names (list_names group_handler_continuations)
    closed_expr_continuations.

Definition group_continuations_nominal_statement : Prop :=
  nominal group_continuations.

(* The term: each bound continuation with the handler at its
   position. *)
Definition group_term
    (a : name_abstraction (list continuation) group_scope) : expr :=
  Let_cont_recursive
    (combine (bound a) (map handler_of (fst (scope a))))
    (proj1_sig (snd (scope a))).

(* The continuations of a group are bound in every handler and in the
   body; the parameters of a handler are bound in that handler only,
   which the equality of handlers says. The first abstraction has as
   many bound continuations as handlers, and then so has the second.
   Neither the continuations nor the parameters are asked to be
   distinct. *)
Definition group_abstraction_statement : Prop :=
  ∀ a a' : name_abstraction (list continuation) group_scope,
    length (bound a) = length (fst (scope a)) ->
    equal_abstraction list_of_names group_continuations a a' ->
    alpha_equivalent (group_term a) (group_term a').

(* For groups whose binders repeat no name the two are one relation:
   the terms are alpha-equivalent exactly when the abstractions are
   equal. So, by [alpha_let_cont_recursive_iff_statement], two groups
   are equal as name abstractions exactly when they agree at fresh
   continuations ([alpha_group]). Each abstraction is asked to have
   as many bound continuations as handlers, since the term keeps
   neither a continuation without a handler nor a handler without a
   continuation, and the equality of abstractions sees both. *)
Definition group_abstraction_iff_statement : Prop :=
  ∀ a a' : name_abstraction (list continuation) group_scope,
    length (bound a) = length (fst (scope a)) ->
    length (bound a') = length (fst (scope a')) ->
    binders_distinct (group_term a) = true ->
    binders_distinct (group_term a') = true ->
    (equal_abstraction list_of_names group_continuations a a'
     <-> alpha_equivalent (group_term a) (group_term a')).
