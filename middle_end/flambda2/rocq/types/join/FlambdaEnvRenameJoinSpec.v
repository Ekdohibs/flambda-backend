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


(* The join of typing environments under a renaming of variables: the
   source of new names, and the action of a renaming on what the join
   holds.

   This file is to be reviewed: the definitions and the statements.
   The join is FlambdaJoinEnv.v, put together with the join of types
   and the meet in FlambdaJoinMeet.v; the renaming of types is
   FlambdaTypeRenameSpec.v and that of typing environments
   FlambdaEnvRenameSpec.v. The contract of the join, FlambdaJoinSpec.v
   and the files beside it, does not speak of renaming.

   The compiler has no such statement: its join is not given renamed
   environments, and a variable that it makes ([Variable.create])
   differs from every other by construction. It is here because every
   function of the model that reads names is to commute with renaming
   (NAMES.md, section 7), and the meet of types, which is stated to
   commute with renaming for a join that does, calls this join.

   The join makes variables. The model's is given a source of new
   names, [Join_env.fresh], a function from the names to avoid to a
   name that is not one of them, and calls it in one place
   ([Bindings_in_target_env.existential_for_these_simples]). The
   variables that it makes are defined by the environment that it
   answers, so they show in its answer.

   No function from lists of variables to variables commutes with
   every renaming ([no_source_commutes_statement]): the empty list is
   left as it is by every renaming, and no variable is. This is a fact
   about the source alone. What follows from it is that no source is
   related to itself by every renaming, so the equation below cannot
   be had with one source on both of its sides in the way it is proved
   here, function by function down to the call of the source. Nothing
   here proves that the join as a whole, with one source, fails to
   commute with some renaming: a join that does not call the source,
   or that fails, does not show the source in its answer, and of one
   that calls it an example is needed. There is one, of the model's
   join with the source that the model uses, in
   FlambdaJoinMeetRenameExampleSpec.v. It is not a defect of that
   source (FlambdaNameAbstractionSpec.fresh_name): no other is related
   to itself by every renaming either.

   What is stated instead: a renaming acts on the source of new names
   too, and the join with the renamed source, of the renamed
   arguments, answers the renamed answer. The renamed source
   ([rename_fresh]) undoes the renaming on the list that it is given,
   asks the source, and renames the name that the source answers; this
   is the action of a renaming on a function. It is said here as a
   relation between two sources, [fresh_related r fresh fresh']: on a
   renamed list, [fresh'] answers the renamed answer of [fresh]. The
   two say the same ([rename_fresh_related_statement],
   [fresh_related_unique_statement]); the relation needs no inverse
   and no equality of functions. A source related to one that avoids
   the names it is given avoids them too
   ([fresh_related_avoids_statement]).

   The other things that the join is given are related in the same
   way. Its parameters ([parameters_related]): the two tests of
   equality answer the same of renamed types and of renamed
   extensions, the type of a constant is renamed, the depth is the
   same. The model's parameters, with the source renamed, are related
   to the model's parameters ([join_parameters_related_statement]):
   its tests of equality do not see a renaming and the type of a
   constant holds no variable.

   The action of a renaming on what the join holds, which is to be
   reviewed: a vector of [Simple]s across the joined environments
   ([rename_simples_in_joined_envs]: the indices are kept); the
   candidate for the canonical [Simple] of a vector
   ([rename_candidate]); the names of the target
   ([rename_bindings_in_target_env]: the source environment, the
   variable made for each vector, with the vector, the variables
   imported and made, with their kinds, and the names to avoid); the
   joined environments ([rename_joined_envs]: each environment and the
   equations of the level cut from it, on both sides); the aliases of
   the variables made ([rename_aliases_of_existentials]); and the
   state of the join ([rename_join_env]: every variable in it, where
   it is a key as where it is in a type). Indices, kinds and symbols
   are kept.

   Statements, beside those of the source: the one function that
   calls the source, and the function of the state that calls it,
   answer the renamed variable and the renamed state when the sources
   are related ([existential_for_these_simples_renamed_statement],
   [existential_for_these_canonicals_renamed_statement]). These are
   where a variable is made. The other functions of the join make no
   variable and compare variables by equality and by binding time
   only; that each gives the renamed answer on renamed arguments is in
   the Proofs file as lemmas, one for a function, named after it.

   The join of environments is handed two more functions of its
   state: the join of types and the choices of the model. What it is
   for a renaming to relate two of each is defined
   ([n_way_join_type_related], [choices_related]), and the choices
   that the model has, the compiler's and the repaired ones, are
   related to themselves ([compiler_choices_related_statement],
   [repaired_choices_related_statement]). The join of types that the
   model gives the join of environments is related to itself where
   the parameters are related
   ([n_way_join_of_types_related_statement]): the traversal of types
   (FlambdaJoinTypes.v) makes no variable, and what it calls of the
   state does as the statements above say.

   The equation of the join as a whole is the last statement
   ([cut_and_n_way_join_related_statement]): the join that the meet
   calls ([Join_env.cut_and_n_way_join]), where everything that it is
   given is related, answers of the renamed environments the renamed
   answer. That join is written for any functions of the environment
   of the meet ([Join_env.meet_env_functions]) and any type of meets
   of expanded types, so the statement is for any action of the
   renaming on the environments of the meet and any relation between
   two meets of expanded types under which the functions are related
   ([meet_env_functions_related], which is to be reviewed). That the
   functions of the model's meet are so related, at the action and the
   relation of FlambdaMeetRenameSpec.v, is not stated here.

   The join that Simplify calls answers an analysis as well: the
   definitions of the variables that the join made, and the
   identifiers that the caller gave the environments. A renaming acts
   on the definitions and keeps the identifiers ([rename_analysis],
   which is to be reviewed). The same equation is stated of that join
   ([cut_and_n_way_join_with_analysis_related_statement]) and of the
   function of types/join_levels.ml that Simplify calls
   ([join_levels_cut_and_n_way_join_related_statement]), and what
   Simplify reads of an analysis is read renamed of the renamed
   analysis ([analysis_read_renamed_statement]).

   What this does not say: that the join with one source of new names
   answers, of renamed arguments, the answer renamed up to the
   variables that it made. That follows from the equation with the
   renamed source and one more fact, that two sources of new names
   give answers that differ by a renaming of the variables made; that
   fact is not stated.

   The proofs are in FlambdaEnvRenameJoinProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaNominalSpec FlambdaAlphaSpec FlambdaNameAbstractionSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeRenameSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaTypingEnv FlambdaEnvRenameSpec FlambdaJoinEnv FlambdaJoinEnvSpec
  FlambdaJoinTypesWithEnvSpec FlambdaMeet FlambdaJoinMeet.
Import ListNotations.

(* ======================================================================== *)
(* The source of new names                                                  *)
(* ======================================================================== *)

(* A source of new names is, to renamed lists, what another is to the
   lists: on a renamed list the second answers the renamed answer of
   the first. *)
Definition fresh_related (r : renaming)
    (fresh fresh' : list variable -> variable) : Prop :=
  ∀ names : list variable,
    fresh' (map (rename_variable r) names) = rename_variable r (fresh names).

(* No function from lists of variables to variables is so related to
   itself by every renaming. *)
Definition no_source_commutes_statement : Prop :=
  ∀ fresh : list variable -> variable,
    ¬ (∀ r : renaming, fresh_related r fresh fresh).

(* The action of a renaming on a source of new names: the list is
   renamed back, the source asked, and its answer renamed. *)
Definition rename_fresh (r : renaming) (fresh : list variable -> variable)
    : list variable -> variable :=
  fun names =>
    rename_variable r
      (fresh (map (rename_variable (inverse_renaming r)) names)).

(* The renamed source is related to the source, and is the only one
   that is. *)
Definition rename_fresh_related_statement : Prop :=
  ∀ (r : renaming) (fresh : list variable -> variable),
    fresh_related r fresh (rename_fresh r fresh).

Definition fresh_related_unique_statement : Prop :=
  ∀ (r : renaming) (fresh fresh' : list variable -> variable),
    fresh_related r fresh fresh' ->
    ∀ names : list variable, fresh' names = rename_fresh r fresh names.

(* A source related to one that avoids the names it is given avoids
   them too. *)
Definition fresh_related_avoids_statement : Prop :=
  ∀ (r : renaming) (fresh fresh' : list variable -> variable),
    fresh_related r fresh fresh' ->
    (∀ names : list variable, ¬ fresh names ∈ names) ->
    ∀ names : list variable, ¬ fresh' names ∈ names.

(* ======================================================================== *)
(* The parameters of the join                                               *)
(* ======================================================================== *)

(* The parameters of a join are, to renamed arguments, what others are
   to the arguments. *)
Definition parameters_related (r : renaming) (p p' : Join_env.parameters)
    : Prop :=
  fresh_related r (Join_env.fresh p) (Join_env.fresh p')
  ∧ (∀ t t' : flambda_type,
      Join_env.type_eqb p' (rename_type r t) (rename_type r t')
      = Join_env.type_eqb p t t')
  ∧ (∀ extension extension' : env_extension,
      Join_env.env_extension_eqb p' (rename_env_extension r extension)
        (rename_env_extension r extension')
      = Join_env.env_extension_eqb p extension extension')
  ∧ (∀ c : const,
      Join_env.expand_const p' c = rename_type r (Join_env.expand_const p c))
  ∧ Join_env.join_depth p' = Join_env.join_depth p.

(* Parameters with another source of new names. *)
Definition with_fresh (fresh : list variable -> variable)
    (p : Join_env.parameters) : Join_env.parameters :=
  {| Join_env.fresh := fresh;
     Join_env.type_eqb := Join_env.type_eqb p;
     Join_env.env_extension_eqb := Join_env.env_extension_eqb p;
     Join_env.expand_const := Join_env.expand_const p;
     Join_env.join_depth := Join_env.join_depth p |}.

(* The parameters of the model, with the source renamed, are related
   to the parameters of the model. *)
Definition join_parameters_related_statement : Prop :=
  ∀ (r : renaming) (join_depth : nat),
    parameters_related r (join_parameters join_depth)
      (with_fresh (rename_fresh r fresh_name) (join_parameters join_depth)).

(* Related parameters: if the source of the first avoids the names it
   is given, so does that of the second. *)
Definition parameters_related_avoids_statement : Prop :=
  ∀ (r : renaming) (p p' : Join_env.parameters),
    parameters_related r p p' -> fresh_avoids p -> fresh_avoids p'.

(* ======================================================================== *)
(* The action of a renaming on what the join holds                          *)
(* ======================================================================== *)

(* A vector of [Simple]s across the joined environments: each is
   renamed, at its index. *)
Definition rename_simples_in_joined_envs (r : renaming)
    (simples : Simples_in_joined_envs.t) : Simples_in_joined_envs.t :=
  Keyed.map_values (rename_simple r) simples.

(* The candidate for the canonical [Simple] of a vector. *)
Definition rename_candidate (r : renaming) (candidate : Source_env.candidate)
    : Source_env.candidate :=
  match candidate with
  | Source_env.No_simples_in_joined_envs =>
    Source_env.No_simples_in_joined_envs
  | Source_env.No_canonical_in_source_env =>
    Source_env.No_canonical_in_source_env
  | Source_env.Canonical_in_all_joined_envs simple =>
    Source_env.Canonical_in_all_joined_envs (rename_simple r simple)
  | Source_env.Latest_bound_source_var var =>
    Source_env.Latest_bound_source_var (rename_variable r var)
  end.

(* A map from variables to things that hold no variable: the keys are
   renamed. *)
Definition rename_keys {A : Type} (r : renaming) (bindings : finite_map A)
    : finite_map A :=
  Keyed.map_bindings (rename_variable r) (fun a => a) bindings.

(* The names of the target: the source environment; the variable made
   for each vector, with the vector; the variables imported and those
   made, which keep their kinds; the names to avoid. *)
Definition rename_bindings_in_target_env (r : renaming)
    (bindings : Bindings_in_target_env.t) : Bindings_in_target_env.t :=
  {| Bindings_in_target_env.source_env :=
       rename_typing_env r (Bindings_in_target_env.source_env bindings);
     Bindings_in_target_env.existential_for_these_simples_map :=
       Keyed.map_bindings (rename_simples_in_joined_envs r)
         (rename_variable r)
         (Bindings_in_target_env.existential_for_these_simples_map bindings);
     Bindings_in_target_env.imported_variables :=
       rename_keys r (Bindings_in_target_env.imported_variables bindings);
     Bindings_in_target_env.existential_variables :=
       rename_keys r (Bindings_in_target_env.existential_variables bindings);
     Bindings_in_target_env.names_to_avoid :=
       map (rename_variable r)
         (Bindings_in_target_env.names_to_avoid bindings) |}.

(* The joined environments: each environment, and the equations of the
   level cut from it, on both sides. *)
Definition rename_joined_envs (r : renaming) (envs : Joined_envs.t)
    : Joined_envs.t :=
  Keyed.map_values
    (fun joined : Typing_env.t * Keyed.finite_map name flambda_type =>
       (rename_typing_env r (fst joined),
        Keyed.map_bindings (rename_name r) (rename_type r) (snd joined)))
    envs.

(* The aliases of the variables made: the variable, and each of its
   aliases. *)
Definition rename_aliases_of_existentials (r : renaming)
    (aliases : Aliases_of_existentials.t) : Aliases_of_existentials.t :=
  Keyed.map_bindings (rename_variable r) (map (rename_variable r)) aliases.

(* The state of a join. A variable is renamed where it is a key as
   where it is in a type or a vector; the indices of the joined
   environments and the kinds are kept. *)
Definition rename_join_env (r : renaming) (env : Join_env.t) : Join_env.t :=
  {| Join_env.joined_envs := rename_joined_envs r (Join_env.joined_envs env);
     Join_env.types_in_target_env :=
       Keyed.map_bindings (rename_variable r) (rename_type r)
         (Join_env.types_in_target_env env);
     Join_env.types_in_joined_envs :=
       Keyed.map_bindings (rename_variable r)
         (Keyed.map_values (rename_type r))
         (Join_env.types_in_joined_envs env);
     Join_env.aliases_of_existentials :=
       rename_aliases_of_existentials r (Join_env.aliases_of_existentials env);
     Join_env.imported_vars :=
       map (rename_variable r) (Join_env.imported_vars env);
     Join_env.definitions_of_existentials :=
       Keyed.map_bindings (rename_variable r)
         (fun definition : Simples_in_joined_envs.t * flambda_kind =>
            (rename_simples_in_joined_envs r (fst definition),
             snd definition))
         (Join_env.definitions_of_existentials env);
     Join_env.bindings :=
       rename_bindings_in_target_env r (Join_env.bindings env) |}.

(* ======================================================================== *)
(* Where a variable is made                                                 *)
(* ======================================================================== *)

(* The variable that stands for a vector: the one made for it before,
   or a new one. With related sources, the renamed names of the target
   and the renamed vector give the renamed variable and the renamed
   names. *)
Definition existential_for_these_simples_renamed_statement : Prop :=
  ∀ (r : renaming) (fresh fresh' : list variable -> variable)
    (bindings : Bindings_in_target_env.t) (simples : Simples_in_joined_envs.t)
    (kind : flambda_kind),
    fresh_related r fresh fresh' ->
    Bindings_in_target_env.existential_for_these_simples fresh'
      (rename_bindings_in_target_env r bindings)
      (rename_simples_in_joined_envs r simples) kind
    = (rename_variable r
         (fst
            (Bindings_in_target_env.existential_for_these_simples fresh
               bindings simples kind)),
       rename_bindings_in_target_env r
         (snd
            (Bindings_in_target_env.existential_for_these_simples fresh
               bindings simples kind))).

(* The same of the function of the state that calls it. *)
Definition existential_for_these_canonicals_renamed_statement : Prop :=
  ∀ (r : renaming) (p p' : Join_env.parameters) (env : Join_env.t)
    (canonicals : Simples_in_joined_envs.t) (kind : flambda_kind),
    parameters_related r p p' ->
    Join_env.existential_for_these_canonicals p' (rename_join_env r env)
      (rename_simples_in_joined_envs r canonicals) kind
    = (rename_variable r
         (fst
            (Join_env.existential_for_these_canonicals p env canonicals kind)),
       rename_join_env r
         (snd
            (Join_env.existential_for_these_canonicals p env canonicals kind))).

(* ======================================================================== *)
(* The join of types and the choices that the join is handed                *)
(* ======================================================================== *)

(* Two joins of types: on the renamed state and the renamed types, the
   second answers the renamed type and the renamed state of the
   first. *)
Definition n_way_join_type_related (r : renaming)
    (join_type join_type' : Join_env.n_way_join_type) : Prop :=
  ∀ (env : Join_env.t) (types : list (Index.t * flambda_type)),
    join_type' (rename_join_env r env)
      (Keyed.map_values (rename_type r) types)
    = option_map
        (fun result : Or_unknown.t flambda_type * Join_env.t =>
           (or_unknown_map (rename_type r) (fst result),
            rename_join_env r (snd result)))
        (join_type env types).

(* Two choices: the second decides of the renamed state, variable and
   types as the first does of the state, the variable and the types. *)
Definition choices_related (r : renaming) (choices choices' : Join_env.choices)
    : Prop :=
  ∀ (env : Join_env.t) (var : variable)
    (types : Keyed.finite_map Index.t flambda_type),
    Join_env.recover_relations_on choices' (rename_join_env r env)
      (rename_variable r var) (Keyed.map_values (rename_type r) types)
    = Join_env.recover_relations_on choices env var types.

(* The choices of the compiler, and the repaired ones, are related to
   themselves by every renaming. *)
Definition compiler_choices_related_statement : Prop :=
  ∀ r : renaming,
    choices_related r Join_env.compiler_choices Join_env.compiler_choices.

Definition repaired_choices_related_statement : Prop :=
  ∀ r : renaming,
    choices_related r Join_env.repaired_choices Join_env.repaired_choices.

(* The join of types that the model gives the join of environments
   (FlambdaJoinTypesWithEnvSpec.n_way_join_of_types, with the
   expansion of heads and the reduction of the heads of naked
   immediates that FlambdaJoinMeet.v gives it): with related
   parameters, a renaming relates the two. The traversal of the types
   makes no variable and compares variables by equality only; the
   variables are made by the functions of the state that it calls,
   with the source of the parameters. *)
Definition n_way_join_of_types_related_statement : Prop :=
  ∀ (r : renaming) (p p' : Join_env.parameters) (fuel_of_env fuel : nat),
    parameters_related r p p' ->
    n_way_join_type_related r
      (n_way_join_of_types p fuel_of_env expand_head_for_join
         reduce_head_of_kind_naked_immediate fuel)
      (n_way_join_of_types p' fuel_of_env expand_head_for_join
         reduce_head_of_kind_naked_immediate fuel).

(* ======================================================================== *)
(* The join that the meet calls                                             *)
(* ======================================================================== *)

(* The join is given the functions of the environment of the meet that
   it uses ([Join_env.meet_env_functions]), and the meet of expanded
   types, which it hands to one of them. Two sets of such functions
   are related by a renaming, given an action [rename_env] of the
   renaming on the environments of the meet and a relation
   [heads_related] between two meets of expanded types: each function
   of the second, on renamed arguments, answers the renamed answer of
   the first; the one that takes a meet of expanded types does so for
   two that are related. *)
Definition meet_env_functions_related {meet_env meet_expanded_head : Type}
    (r : renaming) (rename_env : meet_env -> meet_env)
    (heads_related : meet_expanded_head -> meet_expanded_head -> Prop)
    (ops ops' : Join_env.meet_env_functions meet_env meet_expanded_head)
    : Prop :=
  (∀ tenv : Typing_env.t,
    Join_env.create_meet_env ops' (rename_typing_env r tenv)
    = rename_env (Join_env.create_meet_env ops tenv))
  ∧ (∀ env : meet_env,
      Join_env.current_typing_env ops' (rename_env env)
      = rename_typing_env r (Join_env.current_typing_env ops env))
  ∧ (∀ (env : meet_env) (var : variable) (kind : flambda_kind)
       (mode : Name_mode.t),
      Join_env.add_variable_definition ops' (rename_env env)
        (rename_variable r var) kind mode
      = rename_env (Join_env.add_variable_definition ops env var kind mode))
  ∧ (∀ (env : meet_env) (extension : env_extension)
       (m m' : meet_expanded_head),
      heads_related m m' ->
      Join_env.add_env_extension ops' (rename_env env)
        (rename_env_extension r extension) m'
      = option_map rename_env
          (Join_env.add_env_extension ops env extension m))
  ∧ (∀ env : meet_env,
      Join_env.make_bottom ops' (rename_env env)
      = rename_env (Join_env.make_bottom ops env))
  ∧ (∀ env : meet_env,
      Join_env.final_typing_env ops' (rename_env env)
      = rename_typing_env r (Join_env.final_typing_env ops env))
  ∧ (∀ env : meet_env,
      Join_env.names_to_avoid ops' (rename_env env)
      = map (rename_variable r) (Join_env.names_to_avoid ops env))
  ∧ ∀ (env : meet_env) (names : list variable),
      Join_env.with_names_to_avoid ops' (rename_env env)
        (map (rename_variable r) names)
      = rename_env (Join_env.with_names_to_avoid ops env names).

(* The join that the meet calls: where everything that it is given is
   related (the functions of the environment of the meet, the
   parameters, the choices, the join of types and the meet of expanded
   types), the second join answers, of the renamed environment of the
   meet and the renamed typing environments, the renamed answer of the
   first. The scope is not renamed. *)
Definition cut_and_n_way_join_related_statement : Prop :=
  ∀ (meet_env meet_expanded_head : Type) (r : renaming)
    (rename_env : meet_env -> meet_env)
    (heads_related : meet_expanded_head -> meet_expanded_head -> Prop)
    (ops ops' : Join_env.meet_env_functions meet_env meet_expanded_head)
    (p p' : Join_env.parameters) (fuel : nat)
    (choices choices' : Join_env.choices)
    (join_type join_type' : Join_env.n_way_join_type)
    (m m' : meet_expanded_head) (cut_after : Scope.t) (source_env : meet_env)
    (source_tenv : Typing_env.t) (joined_envs : list Typing_env.t),
    meet_env_functions_related r rename_env heads_related ops ops' ->
    parameters_related r p p' ->
    choices_related r choices choices' ->
    n_way_join_type_related r join_type join_type' ->
    heads_related m m' ->
    Join_env.cut_and_n_way_join ops' p' fuel choices' join_type' m'
      cut_after (rename_env source_env) (rename_typing_env r source_tenv)
      (map (rename_typing_env r) joined_envs)
    = option_map rename_env
        (Join_env.cut_and_n_way_join ops p fuel choices join_type m cut_after
           source_env source_tenv joined_envs).

(* ======================================================================== *)
(* The join that Simplify calls                                             *)
(* ======================================================================== *)

(* The definitions of the variables that a join made: each variable,
   and the vector that it stands for; the kinds are kept. *)
Definition rename_definitions (r : renaming)
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind))
    : finite_map (Simples_in_joined_envs.t * flambda_kind) :=
  Keyed.map_bindings (rename_variable r)
    (fun definition : Simples_in_joined_envs.t * flambda_kind =>
       (rename_simples_in_joined_envs r (fst definition), snd definition))
    definitions.

(* The analysis of a join: its two maps of definitions. The
   identifiers that the caller gave the environments joined are
   kept. *)
Definition rename_analysis {A : Type} (r : renaming)
    (analysis : Analysis.t A) : Analysis.t A :=
  {| Analysis.definitions_in_joined_envs :=
       rename_definitions r (Analysis.definitions_in_joined_envs analysis);
     Analysis.canonical_definitions_at_normal_mode :=
       rename_definitions r
         (Analysis.canonical_definitions_at_normal_mode analysis);
     Analysis.external_ids := Analysis.external_ids analysis |}.

(* What the join did to a [Simple]. *)
Definition rename_simple_refined_at_join (r : renaming)
    (refined : Analysis.simple_refined_at_join)
    : Analysis.simple_refined_at_join :=
  match refined with
  | Analysis.Not_refined_at_join => Analysis.Not_refined_at_join
  | Analysis.Invariant_in_all_uses simple =>
    Analysis.Invariant_in_all_uses (rename_simple r simple)
  | Analysis.Variable_refined_at_these_uses canonicals kind =>
    Analysis.Variable_refined_at_these_uses
      (rename_simples_in_joined_envs r canonicals) kind
  end.

(* The join that Simplify calls, which answers the joined environment
   and an analysis: where everything that it is given is related, the
   second join answers, of the renamed names to avoid and the renamed
   environments, the renamed environment and the renamed analysis. The
   scope and the identifiers of the environments are not renamed. *)
Definition cut_and_n_way_join_with_analysis_related_statement : Prop :=
  ∀ (A meet_env meet_expanded_head : Type) (r : renaming)
    (rename_env : meet_env -> meet_env)
    (heads_related : meet_expanded_head -> meet_expanded_head -> Prop)
    (ops ops' : Join_env.meet_env_functions meet_env meet_expanded_head)
    (p p' : Join_env.parameters) (fuel : nat)
    (choices choices' : Join_env.choices)
    (join_type join_type' : Join_env.n_way_join_type)
    (m m' : meet_expanded_head) (avoid : list variable) (cut_after : Scope.t)
    (source_tenv : Typing_env.t) (joined_envs : list (A * Typing_env.t)),
    meet_env_functions_related r rename_env heads_related ops ops' ->
    parameters_related r p p' ->
    choices_related r choices choices' ->
    n_way_join_type_related r join_type join_type' ->
    heads_related m m' ->
    cut_and_n_way_join_with_analysis ops' p' fuel choices' join_type' m'
      (map (rename_variable r) avoid) cut_after
      (rename_typing_env r source_tenv)
      (map
         (fun use : A * Typing_env.t =>
            (fst use, rename_typing_env r (snd use)))
         joined_envs)
    = option_map
        (fun result : Typing_env.t * Analysis.t A =>
           (rename_typing_env r (fst result), rename_analysis r (snd result)))
        (cut_and_n_way_join_with_analysis ops p fuel choices join_type m
           avoid cut_after source_tenv joined_envs).

(* The same of the function of types/join_levels.ml, which checks
   first that the parameters of the continuation and the symbols
   lifted are defined at the fork. The parameters are renamed; the
   symbols are not. *)
Definition join_levels_cut_and_n_way_join_related_statement : Prop :=
  ∀ (A meet_env meet_expanded_head : Type) (r : renaming)
    (rename_env : meet_env -> meet_env)
    (heads_related : meet_expanded_head -> meet_expanded_head -> Prop)
    (ops ops' : Join_env.meet_env_functions meet_env meet_expanded_head)
    (p p' : Join_env.parameters) (fuel : nat)
    (choices choices' : Join_env.choices)
    (join_type join_type' : Join_env.n_way_join_type)
    (m m' : meet_expanded_head) (avoid : list variable)
    (definition_typing_env : Typing_env.t)
    (ts_and_use_ids : list (Typing_env.t * A)) (params : list variable)
    (cut_after : Scope.t) (extra_lifted_consts_in_use_envs : list symbol),
    meet_env_functions_related r rename_env heads_related ops ops' ->
    parameters_related r p p' ->
    choices_related r choices choices' ->
    n_way_join_type_related r join_type join_type' ->
    heads_related m m' ->
    join_levels_cut_and_n_way_join ops' p' fuel choices' join_type' m'
      (map (rename_variable r) avoid)
      (rename_typing_env r definition_typing_env)
      (map
         (fun use : Typing_env.t * A =>
            (rename_typing_env r (fst use), snd use))
         ts_and_use_ids)
      (map (rename_variable r) params) cut_after
      extra_lifted_consts_in_use_envs
    = option_map
        (fun result : Typing_env.t * Analysis.t A =>
           (rename_typing_env r (fst result), rename_analysis r (snd result)))
        (join_levels_cut_and_n_way_join ops p fuel choices join_type m avoid
           definition_typing_env ts_and_use_ids params cut_after
           extra_lifted_consts_in_use_envs).

(* What Simplify reads of an analysis: of the renamed analysis it reads
   the renamed answer. A value at a use is a constant or unknown, and
   is the same. *)
Definition analysis_read_renamed_statement : Prop :=
  ∀ (A : Type) (r : renaming),
    (∀ (analysis : Analysis.t A) (env : Typing_env.t) (simple : simple),
      Analysis.simple_refined_at_join_of (rename_analysis r analysis)
        (rename_typing_env r env) (rename_simple r simple)
      = rename_simple_refined_at_join r
          (Analysis.simple_refined_at_join_of analysis env simple))
    ∧ (∀ (external_ids : Keyed.finite_map Index.t A)
         (canonicals : Simples_in_joined_envs.t),
        Analysis.values_at_uses external_ids
          (rename_simples_in_joined_envs r canonicals)
        = Analysis.values_at_uses external_ids canonicals)
    ∧ (∀ (external_ids : Keyed.finite_map Index.t A)
         (canonicals : Simples_in_joined_envs.t),
        Analysis.definitions_at_uses external_ids
          (rename_simples_in_joined_envs r canonicals)
        = option_map
            (map
               (fun definition : A * simple =>
                  (fst definition, rename_simple r (snd definition))))
            (Analysis.definitions_at_uses external_ids canonicals))
    ∧ ∀ analysis : Analysis.t A,
        Analysis.variables_created_at_join (rename_analysis r analysis)
        = map
            (fun created : variable * Simples_in_joined_envs.t * flambda_kind
             =>
               (rename_variable r (fst (fst created)),
                rename_simples_in_joined_envs r (snd (fst created)),
                snd created))
            (Analysis.variables_created_at_join analysis).
