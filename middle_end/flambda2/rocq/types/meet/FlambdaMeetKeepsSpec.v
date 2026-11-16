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


(* What a meet keeps, for a join that has the half of its contract
   that is not about values.

   This file is to be reviewed: two definitions and seven statements.
   The proofs are in FlambdaMeetKeepsProofs.v, which needs no review.

   The contract of the meet (FlambdaMeetSpec.v) has two halves. One
   speaks of values: no value that has both types is lost
   ([meet_expanded_head_sound] for the meet of expanded types). The
   other speaks of environments and types only: the environment
   answered is the one given, extended; it fits; the type answered
   fits it and has the kind of the inputs; the variables defined are
   new and may not occur in terms ([meet_expanded_head_keeps]). The
   contract of the join that the meet is given
   ([join_scoped_envs_sound]) has the same two halves, and the main
   theorem (FlambdaMeetClosuresSpec.v) supposes both of the join to
   give both of the meet.

   Here the second half is given alone. [join_scoped_envs_keeps] is
   what the contract of the join says that is not about values: its
   premises but the one about the values that the meet of expanded
   types keeps, and its conclusions but the first. The statement
   ([meet_expanded_head_fuel_keeps_statement]): with a join that has
   that, the meet of expanded types has [meet_expanded_head_keeps] at
   every fuel, for choices as the main theorem has them (the type of
   the [Get_tag] step that is true, no variable defined again, the
   variables of a single case lifted; [repaired_choices] has the
   three: [meet_expanded_head_repaired_keeps_statement]).

   What it is for. A join of which only the half of the contract
   that is not about values is known: the main theorem cannot be
   used of the meet that is given it, and this one can. The file
   gives an interface ([join_scoped_envs_keeps]) and a conditional
   theorem. It was asked for the join of the model; the instance for
   that join is neither proved nor used here: it belongs to the
   separate work on the join, which is under review.

   Why it holds, and how it is proved. No proof of the nine files of
   the contract looks into what it is for a static environment to be
   valid: they hand that supposition on, and to the facts about
   closures that need it. FlambdaMeetValidity.v makes them proofs
   for any predicate on static environments that implies validity.
   At validity itself they are the signed theorems. At the predicate
   that no static environment has, every clause about values is
   vacuous: the join's contract is then [join_scoped_envs_keeps],
   and what the chain proves of the meet is
   [meet_expanded_head_keeps].

   The meet of types ([Meet.meet]) made of that meet of expanded
   types has the part of its contract ([meet_contract],
   FlambdaMeetSpec.v) that is about what is kept
   ([meet_keeps_well_formed]: what it answers may be met again), by
   the same premise ([meet_of_types_keeps_statement]). The two other
   parts that are not about values, the refusal of two kinds and the
   bottom environment, need no premise at all
   ([meet_of_types_kinds_statement]).

   The functions that add to a typing environment
   ([Flambda2_types.add_equation], [add_env_extension],
   [add_equations_on_params], FlambdaMeet.v). Their contracts in
   FlambdaMeetSpec.v ([add_equation_sound] and the two others) are
   about values throughout: they say that the environment answered
   is satisfied, and not that it fits, which they suppose of the
   environment given. So they alone do not give what is needed to
   use them again of an environment answered; this is about what
   can be proved from the contracts, not about whether a second
   call answers. The half that is not about values is
   [typing_env_functions_keep_statement]: the
   environment answered fits, and what fitted the first, was of a
   kind in it or was known to it is so in the second
   ([typing_env_kept]). Its premise is that the meet of expanded
   types keeps, which [meet_expanded_head_fuel_keeps_statement]
   gives for a join that has [join_scoped_envs_keeps] and choices as
   the main theorem has them.

   What is not stated. Nothing about values, by construction.
   Nothing that is kept for the choices of the compiler
   ([compiler_choices]), which the main theorem does not cover
   either: of the meet with those choices only
   [meet_of_types_kinds_statement], which is for any choices, says
   anything. Nothing of the functions that add a relation
   ([Flambda2_types.add_relation] and the three made of it). A join
   that has [join_scoped_envs_sound] is not thereby shown to have
   [join_scoped_envs_keeps]: the first gives its conclusions only for
   a meet that loses no value, the second for any meet that keeps;
   neither definition implies the other. The join that forgets has
   both ([forgetful_join_keeps_statement]), which also shows that
   the premise can be had. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaBindingTime FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvKindedSpec FlambdaMeetGrammar FlambdaExpandHead
  FlambdaMeetEnv FlambdaMeet FlambdaMeetSpec FlambdaMeetEnvSpec
  FlambdaMeetExamplesSpec.
Import ListNotations.

(* The conclusions of the contract that the meet has for its join
   ([join_scoped_envs_sound], FlambdaMeetSpec.v) that are not about
   values, under its premises but the one about the values that the
   meet of expanded types keeps ([meet_expanded_head_sound]), which
   they do not need. *)
Definition join_scoped_envs_keeps (join : join_scoped_envs) : Prop :=
  ∀ (meet_head : meet_expanded_head) (initial_env result : Meet_env.t)
    (scoped_envs : list Typing_env.t),
    let initial_tenv := Meet_env.typing_env initial_env in
    meet_expanded_head_keeps meet_head ->
    typing_env_fits initial_tenv ->
    (∀ scoped_env : Typing_env.t,
      scoped_env ∈ scoped_envs -> scoped_env_of initial_tenv scoped_env) ->
    scoped_envs_agree scoped_envs ->
    join meet_head (Typing_env.current_scope initial_tenv) initial_env
      initial_tenv scoped_envs
    = Some result ->
    meet_env_extends initial_env result
    ∧ typing_env_fits (Meet_env.typing_env result)
    ∧ (∀ x : variable,
        Typing_env.mem None (Meet_env.typing_env result) (Name_var x)
        = true ->
        Typing_env.mem None initial_tenv (Name_var x) = false ->
        (∀ scoped_env : Typing_env.t,
          scoped_env ∈ scoped_envs ->
          Typing_env.mem None scoped_env (Name_var x) = false) ->
        among x (Meet_env.names_to_avoid initial_env) = false
        ∧ among x (Meet_env.names_to_avoid result) = true)
    ∧ (∀ x : variable,
        Typing_env.mem None (Meet_env.typing_env result) (Name_var x)
        = true ->
        Typing_env.mem None initial_tenv (Name_var x) = false ->
        Typing_env.mem (Some Name_mode.normal) (Meet_env.typing_env result)
          (Name_var x)
        = false)
    ∧ ∀ (scoped_env : Typing_env.t) (s : simple) (kind : flambda_kind),
        scoped_env ∈ scoped_envs ->
        Typing_env.mem_simple None (Meet_env.typing_env result) s = true ->
        simple_of_kind scoped_env s kind ->
        simple_of_kind (Meet_env.typing_env result) s kind.

(* ======================================================================== *)
(* Statements                                                               *)
(* ======================================================================== *)

(* With a join that has that, the meet of expanded types keeps what it
   keeps, whatever the fuel. *)
Definition meet_expanded_head_fuel_keeps_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (fuel : nat),
    join_scoped_envs_keeps join ->
    type_of_get_tag_scrutinee choices = immediate_or_blocks_with_these_tags ->
    define_again choices = false ->
    lift_variables_of_single_case choices = true ->
    meet_expanded_head_keeps (Meet.meet_expanded_head join choices fuel).

(* The choices that the contract is proved of have the three. *)
Definition meet_expanded_head_repaired_keeps_statement : Prop :=
  ∀ (join : join_scoped_envs) (fuel : nat),
    join_scoped_envs_keeps join ->
    meet_expanded_head_keeps
      (Meet.meet_expanded_head join repaired_choices fuel).

(* The join that forgets has the half of the contract that is not
   about values; so the meet with it keeps, with no premise. *)
Definition forgetful_join_keeps_statement : Prop :=
  join_scoped_envs_keeps forgetful_join.

Definition meet_with_forgetful_join_keeps_statement : Prop :=
  ∀ fuel : nat,
    meet_expanded_head_keeps
      (Meet.meet_expanded_head forgetful_join repaired_choices fuel).

(* The meet of types made of such a meet of expanded types: what it
   answers may be met again ([meet_keeps_well_formed],
   FlambdaMeetSpec.v). *)
Definition meet_of_types_keeps_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (fuel : nat),
    join_scoped_envs_keeps join ->
    type_of_get_tag_scrutinee choices = immediate_or_blocks_with_these_tags ->
    define_again choices = false ->
    lift_variables_of_single_case choices = true ->
    meet_keeps_well_formed (Meet.meet join choices fuel).

(* The two other parts of [meet_contract] that are not about values
   hold of the meet of types whatever the join, the choices and the
   fuel. *)
Definition meet_of_types_kinds_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (fuel : nat),
    meet_kinds (Meet.meet join choices fuel)
    ∧ meet_bottom_env (Meet.meet join choices fuel).

(* ======================================================================== *)
(* Adding to a typing environment                                           *)
(* ======================================================================== *)

(* What the environment answered keeps of the environment given: it
   fits, and what fitted the first, was of a kind in it or was known
   to it is so in the second. So the premises about fitting, kinds
   and names of a further addition, if they held of the first
   environment, hold of the second. Nothing is said of whether that
   further addition answers. *)
Definition typing_env_kept (tenv tenv' : Typing_env.t) : Prop :=
  typing_env_fits tenv'
  ∧ (∀ ty : flambda_type, type_fits tenv ty -> type_fits tenv' ty)
  ∧ (∀ (s : simple) (kind : flambda_kind),
      simple_of_kind tenv s kind -> simple_of_kind tenv' s kind)
  ∧ ∀ n : name,
      Typing_env.mem None tenv n = true ->
      Typing_env.mem None tenv' n = true.

(* The three functions that add to a typing environment, made of a
   meet of expanded types that keeps. The premises are those of the
   functions of [Meet_env] that they are made of
   (FlambdaMeetEnvSpec.v): what is added fits, and the name of an
   equation is of the kind of its type ([simple_of_kind],
   [env_extension_kinded]), which the contracts about values have
   from the value that the name holds. *)
Definition typing_env_functions_keep_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (fuel : nat),
    meet_expanded_head_keeps (Meet.meet_expanded_head join choices fuel) ->
    (∀ (tenv tenv' : Typing_env.t) (n : name) (ty : flambda_type),
       typing_env_fits tenv ->
       type_fits tenv ty ->
       simple_of_kind tenv (simple_of_name n) (kind ty) ->
       Flambda2_types.add_equation join choices fuel tenv n ty = Some tenv' ->
       typing_env_kept tenv tenv')
    ∧ (∀ (tenv tenv' : Typing_env.t) (extension : env_extension),
         typing_env_fits tenv ->
         env_extension_fits tenv extension ->
         env_extension_kinded tenv extension ->
         Flambda2_types.add_env_extension join choices fuel tenv extension
         = Some tenv' ->
         typing_env_kept tenv tenv')
    ∧ (∀ (tenv tenv' : Typing_env.t)
         (params : list (variable * flambda_kind))
         (param_types : list flambda_type),
         typing_env_fits tenv ->
         Forall2
           (fun (param : variable * flambda_kind) (ty : flambda_type) =>
              type_fits tenv ty
              ∧ simple_of_kind tenv (Var (fst param)) (kind ty))
           params param_types ->
         Flambda2_types.add_equations_on_params join choices fuel tenv
           params param_types
         = Some tenv' ->
         typing_env_kept tenv tenv').
