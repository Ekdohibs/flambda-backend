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


(* Uses made from the source, and the premises on the inputs of the
   join.

   This file is to be reviewed: the definitions and the statements.
   The proofs are in FlambdaJoinUsesProofs.v, which needs no review.

   What it is for. The statements of FlambdaJoinInTermsSpec.v about
   the join that Simplify calls have premises about the inputs:
   those of the five parts
   ([model_join_with_guarantees_on_inputs_statement]:
   [join_inputs_well_formed], [uses_kinded],
   [uses_agree_with_source_on_kinds], [uses_fit_otherwise], and that
   a meet may be made in the source), the shape that the sixth part
   supposes ([uses_know_the_source]), and what the statement in the
   machine supposes ([terms_in_scope_at_uses], FlambdaJoinSpec.v).
   This file gives a way of making uses of which they all hold, with
   side conditions on the environment before each step and on what
   the step is given.

   The uses. A use is made from the source ([made_from]) by going one
   scope further ([Typing_env.increment_scope]) and then, any number
   of times and in any order, by one of two steps.
   - A variable is made equal to a [Simple] ([alias_with_equation]):
     [Typing_env.add_alias] of the two, and then the equation of the
     name that is no longer canonical, an alias to the one that is.
     The meet's environment adds that equation after such an alias
     ([Meet_env.record_demotion], FlambdaMeetEnv.v), and the uses of
     the examples of FlambdaJoinInTermsSpec.v are made so
     ([Arms.alias] with the equation, which is this function:
     [alias_with_equation_is_arms_alias_statement]); without it the
     join says nothing of the two
     ([arms_without_equations_statement], same file). The two are to
     be canonical, different and of one kind. Either may be the one
     that is no longer canonical afterwards.
   - A variable is given a type that is not an alias
     ([Typing_env.replace_equation]). The type is to fit the
     environment ([type_fits], FlambdaMeetSpec.v: it is well formed,
     the environment knows its variables, and it is kinded there), to
     be of the kind of the variable, and to be neither obviously
     unknown nor obviously bottom.
   No step defines a variable. So a type that is given speaks of
   variables that the source knows: a caller whose types mention
   variables defined between the source and the application has them
   removed first ([project_variables_out], FlambdaTypeProject.v,
   whose statements are in FlambdaJoinFitsSpec.v and
   FlambdaJoinDefineSpec.v, is one way).

   The statement ([uses_made_from_source_statement]): if a meet may
   be made in the source ([typing_env_fits], FlambdaMeetSpec.v), the
   variables to avoid include those that the source knows, and each
   use is made from the source, then the inputs with these uses, cut
   after the scope of the source, have the six premises; and a use is
   bottom exactly if the source is.

   The examples ([arms_made_from_source_statement]): a meet may be
   made in the source of the examples of FlambdaJoinInTermsSpec.v,
   and each of their uses with equations is made from it. So the
   side conditions hold of some uses; and by the statement above the
   six premises hold of inputs made of these uses, with variables to
   avoid that include those of that source.

   What this does not give. The statement is about what the uses
   know, their kinds and their scopes. It does not say that a
   valuation satisfies a use, nor that there is one. The other
   premises of the statements of FlambdaJoinInTermsSpec.v stay with
   the caller: the contracts of the meet that is given; that the join
   answers; the use at its index (that it is not bottom is had here
   if the source is not); the test on the answer ([kept_defined_at]);
   that the valuation satisfies the use and the source strictly; the
   conditions on the run ([run_with_values]); and, for the statement
   in the machine, that the machine binds the variables that may
   occur in terms at the use and that the valuation agrees with it on
   them.

   Nothing is assumed beyond the premises of each statement. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaEnvAliasesSpec FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvInvariantSpec FlambdaTypingEnvKindedSpec FlambdaJoinEnv
  FlambdaJoinSpec FlambdaMeetSpec FlambdaJoinMeetFitsSpec
  FlambdaJoinInTermsSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* The uses                                                                 *)
(* ======================================================================== *)

(* A variable is made equal to a [Simple]: the two are made aliases,
   and the name that is no longer canonical is given the equation
   that says so. [None] where [Typing_env.add_alias] answers nothing,
   or bottom (two different constants). *)
Definition alias_with_equation (kind : flambda_kind) (tenv : Typing_env.t)
    (x : variable) (simple : simple) : option Typing_env.t :=
  let* result := Typing_env.add_alias tenv (Var x) simple in
  match result with
  | Ok result =>
    Some
      (Typing_env.replace_equation (Typing_env.add_alias_result_t result)
         (Typing_env.demoted_name result)
         (alias_type_of kind (Typing_env.canonical_element result)))
  | Unknown | Bottom => None
  end.

(* It is the function of the examples of FlambdaJoinInTermsSpec.v. *)
Definition alias_with_equation_is_arms_alias_statement : Prop :=
  ∀ (kind : flambda_kind) (tenv : Typing_env.t) (x : variable)
    (simple : simple),
    Arms.alias true kind tenv x simple
    = alias_with_equation kind tenv x simple.

(* A use made from the source: the source one scope further, and then
   variables made equal to [Simple]s and variables given types. *)
Inductive made_from (source : Typing_env.t) : Typing_env.t -> Prop :=
  | Made_one_scope_further :
      made_from source (Typing_env.increment_scope source)
  | Made_equal (tenv tenv' : Typing_env.t) (kind : flambda_kind)
      (x : variable) (simple : simple) :
      made_from source tenv ->
      is_canonical (Typing_env.aliases tenv) (Var x) ->
      is_canonical (Typing_env.aliases tenv) simple ->
      Var x <> simple ->
      simple_of_kind tenv (Var x) kind ->
      simple_of_kind tenv simple kind ->
      alias_with_equation kind tenv x simple = Some tenv' ->
      made_from source tenv'
  | Made_typed (tenv : Typing_env.t) (x : variable) (ty : flambda_type) :
      made_from source tenv ->
      simple_of_kind tenv (Var x) (kind ty) ->
      type_fits tenv ty ->
      get_alias_opt ty = None ->
      is_obviously_unknown ty = false ->
      is_obviously_bottom ty = false ->
      made_from source
        (Typing_env.replace_equation tenv (Name_var x) ty).

(* ======================================================================== *)
(* The premises on the inputs                                               *)
(* ======================================================================== *)

(* The inputs whose uses are made from the source, cut after the
   scope of the source, have the premises about inputs of the
   statements of FlambdaJoinInTermsSpec.v; and a use is bottom
   exactly if the source is. *)
Definition uses_made_from_source_statement : Prop :=
  ∀ (A : Type) (avoid : list variable) (source : Typing_env.t)
    (uses : list (A * Typing_env.t)),
    typing_env_fits source ->
    (∀ x : variable, knows source x -> among x avoid = true) ->
    (∀ (id : A) (use : Typing_env.t),
       (id, use) ∈ uses -> made_from source use) ->
    let inputs :=
      {| source := source;
         cut_after := Typing_env.current_scope source;
         uses := uses |}
    in
    join_inputs_well_formed typing_env_invariant avoid inputs
    ∧ uses_kinded inputs
    ∧ uses_agree_with_source_on_kinds inputs
    ∧ uses_fit_otherwise inputs
    ∧ uses_know_the_source inputs
    ∧ terms_in_scope_at_uses inputs
    ∧ ∀ (id : A) (use : Typing_env.t),
        (id, use) ∈ uses ->
        Typing_env.is_bottom use = Typing_env.is_bottom source.

(* ======================================================================== *)
(* The examples                                                             *)
(* ======================================================================== *)

(* The uses with equations of the examples of
   FlambdaJoinInTermsSpec.v. *)
Definition arms_uses : list (option Typing_env.t) :=
  [Arms.use true 0 (Var Arms.y1) (Var Arms.z1);
   Arms.use true 1 (Const (Tagged_immediate 0)) (Var Arms.z1);
   Arms.use true 1 (Var Arms.y2) (Var Arms.z2);
   Arms.use true 2 (Const (Tagged_immediate 0))
     (Const (Tagged_immediate 0))].

(* A meet may be made in their source, and each is made from it. *)
Definition arms_made_from_source_statement : Prop :=
  typing_env_fits Arms.source_env
  ∧ ∀ use : option Typing_env.t,
      use ∈ arms_uses ->
      ∃ tenv : Typing_env.t,
        use = Some tenv ∧ made_from Arms.source_env tenv.
