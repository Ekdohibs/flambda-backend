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

(* The model's join with its one source of new names, on renamed
   arguments: an example in which it does not answer the renamed
   answer.

   This file is to be reviewed: the inputs and the three statements.

   FlambdaEnvRenameJoinSpec.v says that no source of new names is
   related to itself by every renaming, which is a fact about the
   source alone; FlambdaJoinMeetRenameSpec.v says that the model's
   join with the source renamed, of renamed arguments, answers the
   renamed answer. Neither says that the model's join as it is, with
   the source that the model uses
   (FlambdaNameAbstractionSpec.fresh_name), fails to commute with a
   renaming. This file gives a join that does.

   The example. The source environment defines three variables; one of
   them, [example_p], is a tagged immediate at one use and a block with
   one field at the other. The join makes a variable
   ([example_join_makes_variable_statement]): its analysis defines one
   variable in the joined environments, and that variable,
   [example_made], is not one of the inputs. The renaming
   [example_exchange] exchanges that variable with another,
   [example_elsewhere], that is nowhere in the inputs, so it leaves
   the inputs and the names to avoid as they are. The join of the
   renamed inputs is then the join of the inputs, whose answer has
   [example_made]; the renamed answer has [example_elsewhere] in its
   place. They differ ([example_join_one_source_statement]). With the
   source renamed the join answers the renamed answer, as
   FlambdaJoinMeetRenameSpec.model_join_related_statement says of
   every join ([example_join_renamed_source_statement]): here it is a
   computation, and it shows that the join of the example does not
   fail.

   What this is and is not. It is one join of the model: the join that
   Simplify calls, with the compiler's choices, at the depth and with
   the fuels of the examples of FlambdaJoinMeetExamplesSpec.v. It
   shows that the equation with one source on both of its sides is
   false of the model's join. It is not a statement about every source
   of new names: the same construction, a renaming that exchanges the
   variable made with one that is nowhere in the inputs, would be made
   for another source at a join that lets the variable made show in
   its answer, and that is not proved here. It is not a discrepancy
   with the compiler, whose join is never given renamed environments
   and whose new variables differ from every other by construction.
   What is owed in place of the false equation is the statement up to
   the variables made (NAMES.md, section 7).

   The proofs are in FlambdaJoinMeetRenameExampleProofs.v, which needs
   no review: each is a computation. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaNominalSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar FlambdaTypingEnv
  FlambdaEnvRenameSpec FlambdaJoinEnv FlambdaJoinSpec
  FlambdaJoinTypesExamplesSpec FlambdaMeetGrammar FlambdaMeet FlambdaJoinMeet
  FlambdaJoinMeetExamplesSpec FlambdaEnvRenameJoinSpec
  FlambdaJoinMeetRenameSpec.
Import ListNotations.

(* ======================================================================== *)
(* The inputs                                                               *)
(* ======================================================================== *)

(* Two uses of the source environment of the examples: [example_p] is
   the tagged immediate 0 at the first, and at the second a block of
   tag 0 whose one field is [example_y]. *)
Definition example_renamed_uses : option (list Typing_env.t) :=
  example_all
    [example_use [] (this_tagged_immediate 0%Z);
     example_use []
       (example_join_block 0%Z [alias_type_of Value (Var example_y)])].

Definition example_renamed_inputs (envs : list Typing_env.t)
    : join_inputs unit :=
  {| source := example_source_env;
     cut_after := Typing_env.current_scope example_source_env;
     uses := map (fun env => (tt, env)) envs |}.

(* The join that Simplify calls, with the parameters given, of those
   inputs and of the names to avoid, all renamed. *)
Definition example_renamed_join (r : renaming) (p : Join_env.parameters)
    : option (Typing_env.t * Analysis.t unit) :=
  let* envs := example_renamed_uses in
  join_with_parameters p Join_env.compiler_choices compiler_choices
    example_fuel
    (map (rename_variable r) [example_y; example_z; example_p; example_a])
    (rename_join_inputs r (example_renamed_inputs envs)).

(* The variable that the join makes, and a variable that is nowhere in
   the inputs; the renaming that exchanges the two. *)
Definition example_made : variable := 5%positive.

Definition example_elsewhere : variable := 9%positive.

Definition example_exchange : renaming :=
  Renaming [(example_made, example_elsewhere)] [].

(* The answer of a join, renamed. *)
Definition rename_join_answer (r : renaming)
    (answer : Typing_env.t * Analysis.t unit)
    : Typing_env.t * Analysis.t unit :=
  (rename_typing_env r (fst answer), rename_analysis r (snd answer)).

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* The model's join of the inputs answers, and its analysis defines one
   variable in the joined environments, the variable made. *)
Definition example_join_makes_variable_statement : Prop :=
  option_map
    (fun answer : Typing_env.t * Analysis.t unit =>
       map fst (Analysis.definitions_in_joined_envs (snd answer)))
    (example_renamed_join identity_renaming (join_parameters 5))
  = Some [example_made].

(* With the model's source of new names, the join of the renamed
   inputs is not the renamed answer of the join of the inputs. *)
Definition example_join_one_source_statement : Prop :=
  ¬ (example_renamed_join example_exchange (join_parameters 5)
     = option_map (rename_join_answer example_exchange)
         (example_renamed_join identity_renaming (join_parameters 5))).

(* With the source renamed, it is. *)
Definition example_join_renamed_source_statement : Prop :=
  example_renamed_join example_exchange
    (renamed_join_parameters example_exchange 5)
  = option_map (rename_join_answer example_exchange)
      (example_renamed_join identity_renaming (join_parameters 5)).
