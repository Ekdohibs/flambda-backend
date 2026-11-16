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


(* The slices of the [Switch] as rewrites that the function which
   simplifies a whole term may be given.

   This file is to be reviewed: the statements. The functions are
   those of the second and the third slice
   (FlambdaSimplifySwitchArmsSpec.v, FlambdaSimplifySwitchMergeSpec.v);
   what is asked of a rewrite of a [Switch] is
   FlambdaSimplifyPruningSpec.v ([switch_rewrite],
   [switch_rewrite_sound]).

   The function of FlambdaSimplifyPruningSpec.v simplifies a whole
   term by types and takes, as a parameter, a function that may
   rewrite a [Switch] from the typing environment where the [Switch]
   is. Its theorem asks of that function that the sets of closures of
   what it answers are well formed, and that what it answers refines
   the [Switch] in the states that satisfy the environment and whose
   static environment is valid ([switch_rewrite_sound]). That file
   states it of the first slice. Here it is stated of the second
   slice, [simplify_switch_arms meet rewrite], and of the third with
   nothing known of the primitives,
   [simplify_switch_merging meet rewrite [] fresh].

   What is stated.
   - [simplify_switch_arms_rewrite_sound_statement] and
     [simplify_switch_merging_rewrite_sound_statement]: each is a
     rewrite of a [Switch] that has what is asked, for a meet and a
     rewrite of [Apply_cont]s that have what the slices ask.
   - The same at the transcribed meet that the contract is proved of,
     with a join that has the contract of a join
     ([switch_arms_of_meet_to_prove_rewrite_sound_statement],
     [switch_merging_of_meet_to_prove_rewrite_sound_statement]).
   - [simplify_switch_arms_rewrite_equivariant_statement]: the second
     slice commutes with renaming in the form that
     FlambdaSimplifyPruningRenamingSpec.v asks of a rewrite of a
     [Switch] ([switch_rewrite_equivariant]), for a meet and a rewrite
     of [Apply_cont]s that do; at the transcribed meet, for a join
     that does
     ([switch_arms_of_meet_to_prove_rewrite_equivariant_statement]).
     So the function that simplifies a whole term by types, given the
     second slice, is a rewrite of terms that commutes with renaming
     ([simplify_by_types_with_switch_arms_equivariant_statement]).
   - [simplify_switch_merging_fixed_names_rename_statement]: the third
     slice with given names commutes with a renaming that fixes those
     names. That is less than to commute with every renaming, which
     it does not (see "Renaming").

   What is assumed. Of the meet, [meet_sound] and [meet_keeps_terms],
   as in the second slice, where the first slice asks
   [meet_bottom_sound]; the transcribed meet has both, with a join
   that has the contract of a join. Of the rewrite of an [Apply_cont],
   that it is right under every environment
   ([apply_cont_rewrite_sound]); the rewrite that changes nothing is,
   and so is that of FlambdaSimplifyPassSpec.v
   ([simplify_apply_cont_sound_statement],
   FlambdaSimplifySwitchArmsSpec.v).

   Nothing known. A rewrite of a [Switch] is given the typing
   environment and no more, so the third slice is taken with nothing
   known of the primitives that have been computed: it then binds a
   primitive the first time an argument asks for it. With something
   known, the refinement is in the states in which it holds
   ([expr_refines_under_valid_where],
   FlambdaSimplifySwitchMergeSpec.v), which a function over terms
   would have to keep as it goes; that is not here.

   Renaming. The second slice commutes with a renaming for a meet and
   a rewrite that do: that is [simplify_switch_arms_rename_statement]
   of its file, which has the form that is asked of a rewrite of a
   [Switch] ([switch_rewrite_equivariant],
   FlambdaSimplifyPruningRenamingSpec.v), and is stated here in that
   form. The third slice takes the names for the variables that
   it binds, and commutes with a renaming when they are renamed with
   the term ([simplify_switch_merging_rename_statement]). As a rewrite
   of a [Switch] its names are fixed, and then it commutes with the
   renamings that fix them, and not in general with a renaming that
   moves one: where such a name is bound in what is answered, the
   renaming moves it in the term and not in the names that are given.
   So the statement here does not give what a function over a whole
   term asks of a rewrite of a [Switch] when it asks that the rewrite
   commute with every renaming. For that, the names have to be made
   in a way that itself commutes with renaming, or the function has
   to ask less of the rewrite; neither is here.

   The proofs are in FlambdaSimplifySwitchRewriteProofs.v, which needs
   no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaNominalSpec FlambdaCseSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeRenameSpec FlambdaTypingEnv FlambdaEnvRenameSpec FlambdaMeet
  FlambdaMeetSpec FlambdaMeetTermsSpec FlambdaMeetRenameSpec
  FlambdaSimplifySwitchSpec FlambdaSimplifySwitchArmsSpec
  FlambdaSimplifySwitchMergeSpec FlambdaSimplifyPruningSpec
  FlambdaSimplifyPruningRenamingSpec.
From OxCamlFlambda2 Require FlambdaValidatorEquivarianceSpec.
Import ListNotations.

(* ======================================================================== *)
(* The slices have what is asked of a rewrite of a [Switch]                 *)
(* ======================================================================== *)

Definition simplify_switch_arms_rewrite_sound_statement : Prop :=
  ∀ (meet : meet_function) (rewrite : apply_cont_rewrite),
    meet_sound meet ->
    meet_keeps_terms meet ->
    apply_cont_rewrite_sound rewrite ->
    switch_rewrite_sound (simplify_switch_arms meet rewrite).

Definition simplify_switch_merging_rewrite_sound_statement : Prop :=
  ∀ (meet : meet_function) (rewrite : apply_cont_rewrite)
    (fresh : list variable),
    meet_sound meet ->
    meet_keeps_terms meet ->
    apply_cont_rewrite_sound rewrite ->
    switch_rewrite_sound (simplify_switch_merging meet rewrite [] fresh).

(* At the transcribed meet that the contract is proved of. *)
Definition switch_arms_of_meet_to_prove_rewrite_sound_statement : Prop :=
  ∀ (join : join_scoped_envs) (fuel : nat) (rewrite : apply_cont_rewrite),
    join_scoped_envs_sound join ->
    apply_cont_rewrite_sound rewrite ->
    switch_rewrite_sound
      (simplify_switch_arms (meet_to_prove join fuel) rewrite).

Definition switch_merging_of_meet_to_prove_rewrite_sound_statement : Prop :=
  ∀ (join : join_scoped_envs) (fuel : nat) (rewrite : apply_cont_rewrite)
    (fresh : list variable),
    join_scoped_envs_sound join ->
    apply_cont_rewrite_sound rewrite ->
    switch_rewrite_sound
      (simplify_switch_merging (meet_to_prove join fuel) rewrite [] fresh).

(* ======================================================================== *)
(* Renaming: the second slice                                               *)
(* ======================================================================== *)

(* The second slice commutes with renaming in the form that is asked
   of a rewrite of a [Switch], for a meet and a rewrite of
   [Apply_cont]s that do. *)
Definition simplify_switch_arms_rewrite_equivariant_statement : Prop :=
  ∀ (meet : meet_function) (rewrite : apply_cont_rewrite),
    meet_function_equivariant meet ->
    apply_cont_rewrite_equivariant rewrite ->
    switch_rewrite_equivariant (simplify_switch_arms meet rewrite).

(* At the transcribed meet that the contract is proved of, with a join
   that commutes with renaming. *)
Definition switch_arms_of_meet_to_prove_rewrite_equivariant_statement : Prop :=
  ∀ (join : join_scoped_envs) (fuel : nat) (rewrite : apply_cont_rewrite),
    join_scoped_envs_equivariant join ->
    apply_cont_rewrite_equivariant rewrite ->
    switch_rewrite_equivariant
      (simplify_switch_arms (meet_to_prove join fuel) rewrite).

(* So the function that simplifies a whole term by types, from the
   environment that knows nothing, given the second slice, is a
   rewrite of terms that commutes with renaming. *)
Definition simplify_by_types_with_switch_arms_equivariant_statement : Prop :=
  ∀ (meet : meet_function) (rewrite : apply_cont_rewrite),
    meet_function_equivariant meet ->
    apply_cont_rewrite_equivariant rewrite ->
    FlambdaValidatorEquivarianceSpec.rewrite_equivariant
      (simplify_by_types_pruning (simplify_switch_arms meet rewrite)).

(* ======================================================================== *)
(* Renaming, with names that are fixed                                      *)
(* ======================================================================== *)

(* The third slice with given names and nothing known commutes with a
   renaming that fixes the names. *)
Definition simplify_switch_merging_fixed_names_rename_statement : Prop :=
  ∀ (meet : meet_function) (rewrite : apply_cont_rewrite) (r : renaming)
    (fresh : list variable) (tenv : Typing_env.t) (scrutinee : simple)
    (arms : list (Z * apply_cont)),
    meet_function_equivariant meet ->
    apply_cont_rewrite_equivariant rewrite ->
    (∀ x : variable, x ∈ fresh -> rename_variable r x = x) ->
    simplify_switch_merging meet rewrite [] fresh (rename_typing_env r tenv)
      (rename_simple r scrutinee) (rename_arms r arms)
    = option_map (rename_expr r)
        (simplify_switch_merging meet rewrite [] fresh tenv scrutinee arms).
