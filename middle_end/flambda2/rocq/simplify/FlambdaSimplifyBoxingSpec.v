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


(* Numbers that are tagged and boxed: the types that Simplify gives
   them, and the [Simple] that a prover answers for the number inside.

   This file is to be reviewed: the definitions and the statements,
   against middle_end/flambda2/simplify/simplify_unary_primitive.ml
   ([simplify_tag_immediate], [simplify_box_number]) and
   middle_end/flambda2/simplify/unboxing/unboxing_epa.ml ([unbox_arg])
   with unboxers.ml. It follows FlambdaSimplifyPrimitivesSpec.v and
   FlambdaSimplifyProjectionsSpec.v, which have the type of an
   argument ([argument_type]), a [Simple] that may occur in terms
   ([simple_in_terms]), the conclusions, and what is assumed: strict
   satisfaction of the environment.

   What Simplify does, and what is here.
   - [Tag_immediate] ([simplify_tag_immediate]) and [Box_number]
     ([simplify_box_number]): the primitive stays, and its variable
     is given the type of the tagged immediates, or of the boxed
     numbers of the kind, whose number has the type of the argument,
     which is an alias. The value of the primitive has that type
     ([tag_immediate_typed_statement], [box_number_typed_statement]),
     so the body of the [Let] is rewritten under the environment with
     that equation ([let_tag_immediate_statement],
     [let_box_number_statement]). The model's [Box_number] allocates
     on the heap.
   - [Untag_immediate] and [Unbox_number]: where Simplify comes to
     the primitive ([simplify_untag_immediate],
     [simplify_unbox_number]) it does not ask a prover for the number
     inside. It meets the type of the argument with the type of the
     tagged immediates, or of the boxed numbers, whose number is the
     variable of the [Let] ([Simplify_common.simplify_projection]),
     and reifies the type that the variable then has: that path is
     for the statements about the meet. The provers that answer a
     [Simple] for the
     number inside, [Provers.meet_tagging_of_simple] and
     [Provers.meet_boxed_number_containing_simple], are asked by the
     unboxing of the parameters of a continuation
     ([Unboxing_epa.unbox_arg], with the unboxers of [Unboxers]). At
     a use of the continuation it asks the prover about the alias
     type, at the kind value, of the argument that is unboxed, at the
     normal name mode. Where the prover answers a [Simple], that is
     the extra argument; where it answers neither, the extra argument
     is a new variable bound by a [Let] to the unboxing primitive on
     the argument; where it answers [Invalid], the use is invalid.
     The statements here are that the first and the third are right
     against the second: the [Let] of the unboxing primitive refines
     the [Let] of the [Simple] answered, and is equivalent to
     [Invalid] where the prover answers [Invalid]
     ([let_untag_immediate_statement], [let_unbox_number_statement]).
     The [Simple] answered may occur in the term
     ([unboxing_provers_answer_in_terms_statement]). The prover of
     tagged immediates reads the numbers of a variant, so the types
     of the environment are to be well formed
     ([types_well_formed], FlambdaProversSpec.v).
   - Two examples: the number of a tagged immediate that was just
     tagged, and the number in a boxed float that was just boxed, are
     the number that was tagged or boxed
     ([example_untag_of_tag_statement],
     [example_unbox_of_box_statement]).

   Not here: the unboxing of parameters itself (its decisions, the
   extra parameters and arguments, poison); the equations of common
   subexpressions that Simplify records at an unboxing primitive.

   The proofs are in FlambdaSimplifyBoxingProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaValueSpec FlambdaPrimitiveModel FlambdaPrimitiveSpec
  FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaEnvAliasesSpec FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvInvariantSpec FlambdaTypingEnvExamplesSpec
  FlambdaTypingEnvRefinementSpec FlambdaProvers
  FlambdaProversSpec FlambdaSimplifyPrimitivesSpec
  FlambdaSimplifyProjectionsSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* A number that is tagged                                                  *)
(* ======================================================================== *)

(* [Simplify_unary_primitive.simplify_tag_immediate]: the value of
   [Tag_immediate] has the type of the tagged immediates whose number
   has the type of the argument. *)
Definition tag_immediate_typed_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars : env) (arg : simple)
    (ty : flambda_type) (v : value),
    env_satisfies tenv statics vars ->
    argument_type tenv arg ty ->
    eval_named statics vars (Prim (Unary Tag_immediate) [arg]) = Some v ->
    type_satisfied statics vars v (tag_immediate ty).

(* So the body of the [Let] is rewritten under the environment in
   which the variable is defined with that type. *)
Definition let_tag_immediate_statement : Prop :=
  ∀ (tenv : Typing_env.t) (x : variable) (declared : flambda_kind)
    (arg : simple) (ty : flambda_type) (body body' : expr),
    typing_env_well_formed tenv = true ->
    Typing_env.mem None tenv (Name_var x) = false ->
    argument_type tenv arg ty ->
    let tagged := Prim (Unary Tag_immediate) [arg] in
    expr_refines_under
      (Typing_env.replace_equation
         (Typing_env.add_variable_definition tenv x Value Name_mode.normal)
         (Name_var x) (tag_immediate ty))
      body body' ->
    expr_refines_under tenv (Let x declared tagged body)
      (Let x declared tagged body').

(* ======================================================================== *)
(* A number that is boxed                                                   *)
(* ======================================================================== *)

(* [Simplify_unary_primitive.simplify_box_number]: the value of
   [Box_number] has the type of the boxed numbers of the kind whose
   contents have the type of the argument. *)
Definition box_number_typed_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars : env)
    (kind : boxable_number) (arg : simple) (ty : flambda_type) (v : value),
    env_satisfies tenv statics vars ->
    argument_type tenv arg ty ->
    eval_named statics vars (Prim (Unary (Box_number kind)) [arg]) = Some v ->
    type_satisfied statics vars v (box_number kind ty Alloc_mode.Heap).

Definition let_box_number_statement : Prop :=
  ∀ (tenv : Typing_env.t) (x : variable) (declared : flambda_kind)
    (kind : boxable_number) (arg : simple) (ty : flambda_type)
    (body body' : expr),
    typing_env_well_formed tenv = true ->
    Typing_env.mem None tenv (Name_var x) = false ->
    argument_type tenv arg ty ->
    let boxed := Prim (Unary (Box_number kind)) [arg] in
    expr_refines_under
      (Typing_env.replace_equation
         (Typing_env.add_variable_definition tenv x Value Name_mode.normal)
         (Name_var x) (box_number kind ty Alloc_mode.Heap))
      body body' ->
    expr_refines_under tenv (Let x declared boxed body)
      (Let x declared boxed body').

(* ======================================================================== *)
(* The number inside                                                        *)
(* ======================================================================== *)

(* The [Simple] that the two provers answer at the normal name mode
   may occur in terms. *)
Definition unboxing_provers_answer_in_terms_statement : Prop :=
  answer_in_terms
    (fun tenv t => Provers.meet_tagging_of_simple tenv Name_mode.normal t)
  ∧ ∀ kind : boxable_number,
      answer_in_terms
        (fun tenv t =>
           Provers.meet_boxed_number_containing_simple kind tenv
             Name_mode.normal t).

(* [Unboxing_epa.unbox_arg] with [Unboxers.Immediate.unboxer]: where
   [Provers.meet_tagging_of_simple], asked about the alias type of the
   argument, answers a [Simple], the [Let] of [Untag_immediate] on the
   argument refines the [Let] of that [Simple]; where it answers that
   no value of the type is a tagged immediate, the [Let] is
   [Invalid]. *)
Definition let_untag_immediate_statement : Prop :=
  ∀ (tenv : Typing_env.t) (x : variable) (declared : flambda_kind)
    (arg : simple) (body : expr),
    simple_in_terms tenv arg ->
    types_well_formed tenv ->
    let untagged := Prim (Unary Untag_immediate) [arg] in
    match
      Provers.meet_tagging_of_simple tenv Name_mode.normal
        (alias_type_of Value arg)
    with
    | Some (Meet_shortcut.Known_result s) =>
      expr_refines_under tenv (Let x declared untagged body)
        (Let x declared (Simple s) body)
    | Some Meet_shortcut.Invalid =>
      expr_equivalent_under tenv (Let x declared untagged body) Invalid
    | Some Meet_shortcut.Need_meet | None => True
    end.

(* The same with the unboxers of the nine boxed numbers:
   [Provers.meet_boxed_number_containing_simple] and [Unbox_number]
   of the kind. *)
Definition let_unbox_number_statement : Prop :=
  ∀ (tenv : Typing_env.t) (x : variable) (declared : flambda_kind)
    (kind : boxable_number) (arg : simple) (body : expr),
    simple_in_terms tenv arg ->
    let unboxed := Prim (Unary (Unbox_number kind)) [arg] in
    match
      Provers.meet_boxed_number_containing_simple kind tenv Name_mode.normal
        (alias_type_of Value arg)
    with
    | Some (Meet_shortcut.Known_result s) =>
      expr_refines_under tenv (Let x declared unboxed body)
        (Let x declared (Simple s) body)
    | Some Meet_shortcut.Invalid =>
      expr_equivalent_under tenv (Let x declared unboxed body) Invalid
    | Some Meet_shortcut.Need_meet | None => True
    end.

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

(* The variables are those of FlambdaTypingEnvExamplesSpec.v. *)

(* The environment in which [x] is defined at a kind and nothing more
   is known of it, as for a parameter. *)
Definition example_number_env (kind : flambda_kind) : Typing_env.t :=
  Typing_env.add_definitions_of_params Typing_env.create [(example_x, kind)].

(* [let p = Tag_immediate x in let q = Untag_immediate p in
   apply_cont k q], for a naked immediate [x]. [p] is given the type
   of the tagged immediates whose number is [x]; of the type of [p]
   the prover answers that the number is [x]; [q] is made equal to
   [x], and replaced by it in the body. *)
Definition example_untag_of_tag_statement : Prop :=
  ∀ k : continuation,
    let tagged := Prim (Unary Tag_immediate) [Var example_x] in
    expr_refines_under (example_number_env naked_immediate_kind)
      (Let example_p Value tagged
         (Let example_q naked_immediate_kind
            (Prim (Unary Untag_immediate) [Var example_p])
            (Apply_cont (Apply_cont_expr k [Var example_q] None))))
      (Let example_p Value tagged
         (Let example_q naked_immediate_kind (Simple (Var example_x))
            (Apply_cont (Apply_cont_expr k [Var example_x] None)))).

(* [let p = Box_number x in let q = Unbox_number p in apply_cont k q],
   for a naked float [x]. *)
Definition example_unbox_of_box_statement : Prop :=
  ∀ k : continuation,
    let float := Naked_number Kind_naked_float in
    let boxed := Prim (Unary (Box_number Boxable_naked_float)) [Var example_x]
    in
    expr_refines_under (example_number_env float)
      (Let example_p Value boxed
         (Let example_q float
            (Prim (Unary (Unbox_number Boxable_naked_float)) [Var example_p])
            (Apply_cont (Apply_cont_expr k [Var example_q] None))))
      (Let example_p Value boxed
         (Let example_q float (Simple (Var example_x))
            (Apply_cont (Apply_cont_expr k [Var example_x] None)))).
