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


(* The environment side of the n-way join of typing environments.

   This file is to be reviewed line by line against
   middle_end/flambda2/types/env/join_env.ml, as FlambdaTypingEnv.v is
   against typing_env.ml, and its last function against the n-way
   entry of types/join_levels.ml. It holds definitions only, in modules
   that have the compiler's names. What the join guarantees is
   FlambdaJoinSpec.v. [Type_grammar.project_variables_out], which the
   join calls, is FlambdaTypeProject.v.

   What is here, against join_env.ml, in the order of that file:
   - [Index]: the environments joined are numbered from 0 in the order
     of the list given ([Index.fold_list]). An index is a position, not
     a name: renaming leaves it alone. [Index.choose] is
     [Index.Map.choose], as the binding of the least index.
   - [Simples_in_joined_envs.t]: a [Simple] for each of some of the
     indices, as a map read with [Index.equal]. In the compiler it is
     also the key of a map ([Simples_in_joined_envs.Map]); two vectors
     are the same key when they bind the same indices to the same
     [Simple]s ([Simples_in_joined_envs.equal]).
     [choose_a_suitable_name] is left out: the names of the model are
     numbers.
   - [Source_env]: [exists_in_source_env], [exists_at_name_mode] and
     [candidate_canonical_in_source_env]. The compiler's type of that
     name is [candidate] here.
   - [Bindings_in_target_env]: the record, [from_source_env],
     [exists_in_target_env], [has_existential_for_these_simples],
     [existential_for_these_simples], [import_from_all_envs] and the
     two folds, as lists.
   - [Joined_envs]: [get_nth_joined_env], [exists_in_all_joined_envs],
     [equal_in_all_joined_envs],
     [get_alias_then_canonical_simple_ignoring_name_mode_exn],
     [find_simples_in_joined_envs], [find_imported_var],
     [get_canonical_simples_ignoring_name_mode].
   - [Aliases_of_existentials].
   - [Join_env]: the state [t] and [create];
     [new_definitions_of_existentials], [new_equations_in_joined_envs];
     [existential_for_these_canonicals], [add_alias_of_existential],
     [add_equation_in_target_env], [add_equations_in_joined_envs];
     [n_way_join_canonical_simples]; the five functions that import
     ([import_or_delay_n_way_join_type],
     [import_or_delay_n_way_join_of_concrete_type],
     [add_equals_in_joined_envs], [import_type_transitive],
     [import_var_transitive]) and [import_name_transitive];
     [join_aliases_into_bindings];
     [add_inverse_relation_to_env_extension] and
     [recover_inverse_relations]; [n_way_join_round], and the two loops
     of [cut_and_n_way_join0] ([n_way_join_delayed_equations],
     [n_way_join_loop]); what the join of types calls: [joined_env],
     [import_type], [n_way_join_simples], [n_way_join_env_extension];
     [move_definition], [move_equation], [move_inverse_relation] and
     [define_or_eliminate_variables_and_add_equations]; [cut_for_join],
     [cut_and_n_way_join0] and [cut_and_n_way_join], which is the join
     that the meet calls.
   - [Analysis.t]: the three fields of the compiler's record;
     [Analysis.simple_refined_at_join], with
     [Variable_refined_at_join.fold_values_at_uses] as the list
     [values_at_uses], and [fold_variables_created_at_join] with
     [Simples_at_join.fold_definitions_at_uses] as the list
     [variables_created_at_join]: what Simplify reads
     (simplify/join_points.ml, simplify_switch_expr.ml,
     specialization_cost.ml); and [Analysis.create].
   - [cut_and_n_way_join_with_analysis], after the analysis here, and
     [Join_levels.cut_and_n_way_join] for the n-way algorithm.

   The differences:
   - A [Simple] of the model has no coercion (the terms have no
     [Rec_info]), so the wrappers that keep a coercion beside a name
     are plain [simple]s, and [Import_from_all_joined_envs] and
     [Latest_bound_source_var] carry a variable only.
   - The wrappers that tell the environment a name belongs to
     ([Variable_in_target_env], [Simple_in_one_joined_env], ...) are
     the plain types: they are a discipline of the compiler's types and
     hold no data.
   - New names. A variable that the compiler makes ([Variable.create])
     differs from every other. Here [existential_for_these_simples]
     is given a function [fresh] and the bindings carry the names to
     avoid; the variable made is [fresh] of those names and is added to
     them. Which variable is made is not specified
     (FlambdaJoinSpec.v). The join that Simplify calls is given the
     names to avoid; the one that the meet calls takes them from the
     environment of the meet and adds the variables that the
     environments it is given know. Either way the last phase, which
     adds equations to the target through the meet, is given the
     environment of the meet with the names of the join for names to
     avoid: those that it was given and the variables that it made.
     So a join that one of those meets calls makes no variable that
     the outer join made or that an environment joined knows; and the
     target answered has for names to avoid those of the join and
     those that the meets added, so that the joins of two branches of
     a meet do not make one variable twice.
   - The environment of the meet. Its functions are given to the join
     in a record ([meet_env_functions]), with the meet of expanded
     types that they take; the model's are in the files of the meet.
   - The variable that stands for an existential variable that has
     aliases among the variables of the target. The compiler takes the
     least of them in the order of variables; the model, the one that
     the target binds first ([choose_alias]). The others become its
     aliases in either case, so the environments differ in which
     variable holds the type, the definition of the analysis and the
     inverse relations before the equations are added.
   - Kinds of variables. A variable of the compiler carries its kind
     ([Variable.kind]), and [import_var_transitive] reads it. A
     variable of the model is a number. The kind of a variable that is
     imported is that of the type that the first joined environment
     that knows it has for it ([Joined_envs.kind_of_imported_var]); the
     other environments are then asked for its type at that kind, as
     in the compiler, and [None] is answered if one has it at another.
   - Physical equality of types is not in the model. In
     [import_or_delay_n_way_join_of_concrete_type] and
     [n_way_join_env_extension] the compiler tests it; the functions
     here are given an equality of types and of environment extensions
     ([parameters]) and test that, which is what the compiler does
     under -flambda2-join-sharing structural.
   - Recursion. The five functions that import are one recursion in
     the compiler, which ends because a variable is imported once.
     Here each takes the function that imports a variable, and
     [import_var_transitive] is defined last by recursion on fuel. The
     loop of [n_way_join_delayed_equations] takes fuel too, and so
     does [expand_variable], the function that gives the type to put
     for a variable that is removed (in the compiler it ends because
     such a variable is mentioned once). The loop of
     [add_inverse_relation_to_env_extension] follows aliases until it
     has seen a name twice; here it has also a bound on the number of
     steps.
   - The names that every level has an equation for. The compiler
     finds them with iterators over the maps of the levels
     ([fold_incremental_join], in the order of names); here they are
     those of the first level that the others have too
     ([names_with_equations_everywhere]). The compiler's inputs are
     incremental (the fields [previous], [diff] and [current] of a
     level's equations), for a join of environment extensions that it
     does not do: where they are made ([cut_for_join]), [previous] is
     empty and [diff] is [current], and then the fold visits the names
     that every level has an equation for, or none if a level has no
     equation. The model has the equations only.
   - [Map.diff_shared] in [new_definitions_of_existentials] and
     [new_equations_in_joined_envs] is "the bindings whose key the
     older map has not": each key is added once.
   - Order. A fold is the list of what it visits, in the order of the
     map that the model has (the order in which the bindings were
     added, or its reverse where the header of a function says so),
     which is not the compiler's (the order of the keys). It visits the
     bindings that count ([compact]): a map of the model is a list of
     which the first binding of a key counts. (One exception: the fold
     over the equations of an environment extension in
     [n_way_join_env_extension] visits every equation; see there.) The
     canonical element
     that is chosen for a vector does not depend on the order
     ([candidate_canonical_in_source_env] answers the same for every
     order of the indices); which variable is imported first, and so
     which names are made, does.
     [define_or_eliminate_variables_and_add_equations] adds the
     equations in the order of the list that it ends with: first the
     equations that it moves to, or makes for, the variables of the
     target that replace an existential variable ([move_equation] and
     the fold over the demoted aliases both put in front), then what is
     left of [types_in_target_env], which [add_equation_in_target_env]
     builds by putting each equation in front. Where one environment is
     joined, the variables of the target that its level has an equation
     for are met in the order of the level
     ([names_with_equations_everywhere]) and no join is delayed, so
     their equations stand in the reverse of the order of the level;
     the equation of a variable that is imported on the way stands
     where the variable was imported, whatever its place in the level.
     The meet that called the join may then add them again:
     [direct_return] of [meet_disjunction] cuts the level of its scope,
     which holds what the join added, and goes through it in the
     level's order, the equation added last first (FlambdaMeet.v, "The
     order of what is gone through"). That is the reverse of the order
     in which the join added them, where the compiler goes through a
     map of names both times; the two orders can decide whether the
     meet answers bottom (T4-21).
   - Two variables of the source with one binding time are ordered by
     the compiler by their names ([Typing_env.stable_compare_simples]);
     [Source_env.latest_bound] answers [None] for them. A well-formed
     environment has no two such variables.
   - A fatal error of the compiler, and a failed [assert], is [None]:
     an index with no environment or no external identifier, a
     variable with two equations in the target, a join of no type, a
     name that has not a type of naked immediates where an inverse
     relation is to be added. The checks that the compiler makes only
     with [Flambda_features.check_light_invariants] are not made.
   - The test for recovering relations. In [n_way_join_round] the
     compiler recovers the relations that the type joined for a
     variable names if every joined environment gave a type
     ([Joined_envs.exists_in_all_joined_envs]). Here the test is an
     argument of the rounds and of the joins made with them
     ([choices]). [compiler_choices] is the compiler's test, and with
     it the functions are the compiler's. [repaired_choices] is not in
     the compiler: it asks also that the variable stands, in every
     joined environment, for something that has a value there
     (discrepancy T5-10).
   - [machine_width], [Analysis.print] and the symbol projections
     ([n_way_join_symbol_projections], and the second component of
     [cut_for_join]) are left out: the model's environment has no
     symbol projections (FlambdaTypingEnv.v). The handler that prints
     the levels when the join stops with a fatal error is left out. *)

From Stdlib Require Import ZArith List Bool PeanoNat.
From OxCamlStdlib Require Import OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar FlambdaTypeProject
  FlambdaBindingTime FlambdaEnvAliases FlambdaTypingEnv.
Import ListNotations.

(* ======================================================================== *)
(* The environments joined, and vectors of [Simple]s across them            *)
(* ======================================================================== *)

(* [Index.t]: the number of one of the environments joined. *)
Module Index.

  Definition t : Type := nat.

  Definition equal (index index' : t) : bool := Nat.eqb index index'.

  (* [Index.fold_list], as the list of the elements with their indices:
     the first has index 0. *)
  Definition number {A : Type} (xs : list A) : list (t * A) :=
    combine (seq 0 (length xs)) xs.

  (* [Index.Map.bindings]: the bindings that count, by increasing
     index. (An order on indices is an order on the positions of the
     inputs, not on names.) *)
  Fixpoint insert {A : Type} (binding : t * A) (sorted : list (t * A))
      : list (t * A) :=
    match sorted with
    | [] => [binding]
    | binding' :: rest =>
      if Nat.leb (fst binding) (fst binding')
      then binding :: sorted
      else binding' :: insert binding rest
    end.

  Definition bindings {A : Type} (map : Keyed.finite_map t A)
      : list (t * A) :=
    List.fold_right insert [] (Keyed.compact equal map).

  (* [Index.Map.choose]: the binding of the least index. *)
  Definition choose {A : Type} (map : Keyed.finite_map t A) : option (t * A) :=
    match bindings map with
    | [] => None
    | binding :: _ => Some binding
    end.

End Index.

(* [Simples_in_joined_envs.t]: a [Simple] in each of some of the
   environments joined. *)
Module Simples_in_joined_envs.

  Definition t : Type := Keyed.finite_map Index.t simple.

  Definition find (index : Index.t) (simples : t) : option simple :=
    Keyed.lookup Index.equal index simples.

  (* [Simples_in_joined_envs.equal]: the same indices have the same
     [Simple]s. This is the test with which a vector is the key of a
     map. *)
  Definition equal (simples simples' : t) : bool :=
    Keyed.for_all_bindings_b Index.equal
      (fun index simple =>
         match find index simples' with
         | Some simple' => simple_equal simple simple'
         | None => false
         end)
      simples
    && Keyed.for_all_bindings_b Index.equal
         (fun index _ => Keyed.is_bound Index.equal index simples)
         simples'.

End Simples_in_joined_envs.

(* ======================================================================== *)
(* The source environment                                                   *)
(* ======================================================================== *)

(* [Source_env]: the environment that the join extends, and the search
   in it for a [Simple] that stands for a vector. *)
Module Source_env.

  Definition t : Type := Typing_env.t.

  Definition exists_in_source_env (source_env : t) (var : variable) : bool :=
    Typing_env.mem None source_env (Name_var var).

  Definition exists_at_name_mode (min_name_mode : Name_mode.t)
      (source_env : t) (var : variable) : bool :=
    Typing_env.mem (Some min_name_mode) source_env (Name_var var).

  (* Of two variables of the source, the one bound later:
     [total_compare_binding_times] and the test on its result in
     [latest_source_var_with]. [None] is a fatal error of the compiler
     (a variable that is not bound), or two variables with one binding
     time, which the compiler orders by their names. *)
  Definition latest_bound (source_env : t) (var var0 : variable)
      : option variable :=
    if Pos.eqb var var0
    then Some var
    else
      let* time :=
        Typing_env.binding_time_and_mode source_env (Name_var var)
      in
      let* time0 :=
        Typing_env.binding_time_and_mode source_env (Name_var var0)
      in
      let binding_time := Binding_time.With_name_mode.binding_time time in
      let binding_time0 := Binding_time.With_name_mode.binding_time time0 in
      if Binding_time.strictly_earlier binding_time binding_time0
      then Some var0
      else if Binding_time.strictly_earlier binding_time0 binding_time
      then Some var
      else None.

  (* The compiler's type [candidate_canonical_in_source_env]; the
     function has that name here. *)
  Inductive candidate : Type :=
    | No_simples_in_joined_envs
    | No_canonical_in_source_env
    | Canonical_in_all_joined_envs (simple : simple)
    | Latest_bound_source_var (var : variable).

  (* [pattern_match_local_simple]: a [Simple] of a joined environment
     is a variable of the source, or is local. *)
  Definition source_var_of (source_env : t) (simple : simple)
      : option variable :=
    match simple with
    | Var var => if exists_in_source_env source_env var then Some var else None
    | Symbol _ | Const _ => None
    end.

  (* One step of the fold of [candidate_canonical_in_source_env]. *)
  Definition candidate_step (source_env : t) (possible : candidate)
      (canonical : simple) : option candidate :=
    let maybe_this_source_var :=
      match source_var_of source_env canonical with
      | None => No_canonical_in_source_env
      | Some var => Latest_bound_source_var var
      end
    in
    let latest_source_var_with (var : variable) :=
      match source_var_of source_env canonical with
      | None => Some (Latest_bound_source_var var)
      | Some var0 =>
        option_map Latest_bound_source_var (latest_bound source_env var var0)
      end
    in
    match possible with
    | No_simples_in_joined_envs =>
      Some (Canonical_in_all_joined_envs canonical)
    | No_canonical_in_source_env => Some maybe_this_source_var
    | Latest_bound_source_var var => latest_source_var_with var
    | Canonical_in_all_joined_envs shared_simple =>
      if simple_equal canonical shared_simple
      then Some possible
      else
        match source_var_of source_env shared_simple with
        | None => Some maybe_this_source_var
        | Some var => latest_source_var_with var
        end
    end.

  Definition candidate_canonical_in_source_env (source_env : t)
      (canonicals_in_joined_envs : Simples_in_joined_envs.t)
      : option candidate :=
    List.fold_left
      (fun possible binding =>
         let* possible := possible in
         candidate_step source_env possible (snd binding))
      (Keyed.compact Index.equal canonicals_in_joined_envs)
      (Some No_simples_in_joined_envs).

End Source_env.

(* ======================================================================== *)
(* The names of the target environment                                      *)
(* ======================================================================== *)

(* [Bindings_in_target_env]: the variables that the target has and the
   source has not. An imported variable is one of the joined
   environments, under its own name; an existential variable is new,
   and stands for a vector of [Simple]s. *)
Module Bindings_in_target_env.

  Record t : Type := {
    source_env : Source_env.t;
    existential_for_these_simples_map :
      Keyed.finite_map Simples_in_joined_envs.t variable;
    imported_variables : finite_map flambda_kind;
    existential_variables : finite_map flambda_kind;
    names_to_avoid : list variable
  }.

  Definition from_source_env (names_to_avoid : list variable)
      (source_env : Source_env.t) : t :=
    {| source_env := source_env;
       existential_for_these_simples_map := [];
       imported_variables := [];
       existential_variables := [];
       names_to_avoid := names_to_avoid |}.

  Definition exists_in_target_env (bindings : t) (var : variable) : bool :=
    Source_env.exists_in_source_env (source_env bindings) var
    || is_bound var (existential_variables bindings)
    || is_bound var (imported_variables bindings).

  Definition has_existential_for_these_simples (bindings : t)
      (simples : Simples_in_joined_envs.t) : option variable :=
    Keyed.lookup Simples_in_joined_envs.equal simples
      (existential_for_these_simples_map bindings).

  (* The variable that stands for a vector: the one made for it before,
     or a new one, which is then one of the names to avoid. *)
  Definition existential_for_these_simples
      (fresh : list variable -> variable) (bindings : t)
      (simples : Simples_in_joined_envs.t) (kind : flambda_kind)
      : variable * t :=
    match has_existential_for_these_simples bindings simples with
    | Some existing_canonical => (existing_canonical, bindings)
    | None =>
      let var := fresh (names_to_avoid bindings) in
      (var,
       {| source_env := source_env bindings;
          existential_for_these_simples_map :=
            (simples, var) :: existential_for_these_simples_map bindings;
          imported_variables := imported_variables bindings;
          existential_variables :=
            (var, kind) :: existential_variables bindings;
          names_to_avoid := var :: names_to_avoid bindings |})
    end.

  Definition import_from_all_envs (bindings : t) (imported_var : variable)
      (kind : flambda_kind) : t :=
    if is_bound imported_var (imported_variables bindings)
    then bindings
    else
      {| source_env := source_env bindings;
         existential_for_these_simples_map :=
           existential_for_these_simples_map bindings;
         imported_variables :=
           (imported_var, kind) :: imported_variables bindings;
         existential_variables := existential_variables bindings;
         names_to_avoid := names_to_avoid bindings |}.

  (* [fold_imported_variables] and [fold_existential_variables], as the
     lists of what they visit: the variables in the order in which they
     were imported or made. *)
  Definition imported_variables_in_order (bindings : t)
      : list (variable * flambda_kind) :=
    rev (compact (imported_variables bindings)).

  Definition existential_variables_in_order (bindings : t)
      : list (variable * flambda_kind) :=
    rev (compact (existential_variables bindings)).

End Bindings_in_target_env.

(* ======================================================================== *)
(* The joined environments                                                  *)
(* ======================================================================== *)

(* [Joined_envs]: the environments joined that are not bottom, each
   with the equations of the level cut from it. *)
Module Joined_envs.

  Definition t : Type :=
    Keyed.finite_map Index.t
      (Typing_env.t * Keyed.finite_map name flambda_type).

  Definition get_nth_joined_env (envs : t) (index : Index.t)
      : option Typing_env.t :=
    option_map fst (Keyed.lookup Index.equal index envs).

  Definition keys (envs : t) : list Index.t := Keyed.domain Index.equal envs.

  Definition exists_in_all_joined_envs {A : Type} (envs : t)
      (map : Keyed.finite_map Index.t A) : bool :=
    forallb (fun index => Keyed.is_bound Index.equal index map) (keys envs).

  (* [equal_in_all_joined_envs]: a [Simple] has, in each environment of
     a vector, the canonical [Simple] that the vector has there. *)
  Fixpoint equal_in_bindings (envs : t) (simple : simple)
      (bindings : list (Index.t * FlambdaModel.simple)) : option bool :=
    match bindings with
    | [] => Some true
    | (index, canonical) :: bindings =>
      if simple_equal canonical simple
      then equal_in_bindings envs simple bindings
      else
        let* env := get_nth_joined_env envs index in
        if simple_equal canonical
             (Typing_env.get_canonical_simple_ignoring_name_mode env simple)
        then equal_in_bindings envs simple bindings
        else Some false
    end.

  Definition equal_in_all_joined_envs (envs : t) (simple : simple)
      (simples_in_joined_envs : Simples_in_joined_envs.t) : option bool :=
    equal_in_bindings envs simple
      (Keyed.compact Index.equal simples_in_joined_envs).

  (* [get_alias_then_canonical_simple_ignoring_name_mode_exn]: the
     canonical [Simple]s of types that are all aliases. The inner
     [None] is the compiler's [Not_found]: a type is not an alias. *)
  Fixpoint canonicals_of_alias_types (envs : t)
      (types : list (Index.t * flambda_type))
      : option (option Simples_in_joined_envs.t) :=
    match types with
    | [] => Some (Some [])
    | (index, ty) :: types =>
      let* env := get_nth_joined_env envs index in
      let* canonicals := canonicals_of_alias_types envs types in
      Some
        (match get_alias_opt ty, canonicals with
         | Some alias, Some canonicals =>
           Some
             ((index,
               Typing_env.get_canonical_simple_ignoring_name_mode env alias)
              :: canonicals)
         | _, _ => None
         end)
    end.

  Definition get_alias_then_canonical_simple_ignoring_name_mode_exn
      (envs : t) (types : Keyed.finite_map Index.t flambda_type)
      : option (option Simples_in_joined_envs.t) :=
    canonicals_of_alias_types envs (Keyed.compact Index.equal types).

  (* [find_simples_in_joined_envs]: the type that each environment of
     a vector has for the [Simple] of the vector there. *)
  Fixpoint types_of_simples (expand_const : const -> flambda_type)
      (envs : t) (kind : flambda_kind) (simples : list (Index.t * simple))
      : option (Keyed.finite_map Index.t flambda_type) :=
    match simples with
    | [] => Some []
    | (index, simple) :: simples =>
      let* ty :=
        match simple with
        | Const const => Some (expand_const const)
        | Var var =>
          let* env := get_nth_joined_env envs index in
          Typing_env.find env (Name_var var) (Some kind)
        | Symbol symbol =>
          let* env := get_nth_joined_env envs index in
          Typing_env.find env (Name_symbol symbol) (Some kind)
        end
      in
      let* types := types_of_simples expand_const envs kind simples in
      Some ((index, ty) :: types)
    end.

  Definition find_simples_in_joined_envs
      (expand_const : const -> flambda_type) (envs : t)
      (simples : Simples_in_joined_envs.t) (kind : flambda_kind)
      : option (Keyed.finite_map Index.t flambda_type) :=
    types_of_simples expand_const envs kind
      (Keyed.compact Index.equal simples).

  (* [find_imported_var]: the type of a variable in each environment
     that has it. *)
  Fixpoint types_of_imported_var
      (envs :
         list
           (Index.t * (Typing_env.t * Keyed.finite_map name flambda_type)))
      (var : variable) (kind : flambda_kind)
      : option (Keyed.finite_map Index.t flambda_type) :=
    match envs with
    | [] => Some []
    | (index, (env, _)) :: envs =>
      let* types := types_of_imported_var envs var kind in
      if Typing_env.mem None env (Name_var var)
      then
        let* ty := Typing_env.find env (Name_var var) (Some kind) in
        Some ((index, ty) :: types)
      else Some types
    end.

  Definition find_imported_var (envs : t) (var : variable)
      (kind : flambda_kind) : option (Keyed.finite_map Index.t flambda_type) :=
    types_of_imported_var (Keyed.compact Index.equal envs) var kind.

  (* The kind of a variable of the joined environments: that of its
     type in the first of them that has it. (A variable of the compiler
     carries its kind.) *)
  Fixpoint kind_of_var_in
      (envs :
         list
           (Index.t * (Typing_env.t * Keyed.finite_map name flambda_type)))
      (var : variable) : option flambda_kind :=
    match envs with
    | [] => None
    | (_, (env, _)) :: envs =>
      match Typing_env.find env (Name_var var) None with
      | Some ty =>
        if Typing_env.mem None env (Name_var var)
        then Some (kind ty)
        else kind_of_var_in envs var
      | None => kind_of_var_in envs var
      end
    end.

  Definition kind_of_imported_var (envs : t) (var : variable)
      : option flambda_kind :=
    kind_of_var_in (Keyed.compact Index.equal envs) var.

  (* [get_canonical_simples_ignoring_name_mode]: the vector of the
     canonical [Simple]s of a list of [Simple]s, each in its
     environment. *)
  Definition get_canonical_simples_ignoring_name_mode (envs : t)
      (simples : list (Index.t * simple))
      : option Simples_in_joined_envs.t :=
    List.fold_left
      (fun acc binding =>
         let* acc := acc in
         let* env := get_nth_joined_env envs (fst binding) in
         Some
           ((fst binding,
             Typing_env.get_canonical_simple_ignoring_name_mode env
               (snd binding))
            :: acc))
      simples (Some []).

End Joined_envs.

(* [Aliases_of_existentials]: for an existential variable, the
   variables of the target that are to be its aliases. *)
Module Aliases_of_existentials.

  Definition t : Type := finite_map (list variable).

  Definition empty : t := [].

  Definition aliases_of_existential_var (aliases : t) (var : variable)
      : list variable :=
    match lookup var aliases with
    | Some vars => vars
    | None => []
    end.

  Definition add (aliases : t) (demoted_var existential_var : variable) : t :=
    let aliases_of_canonical_element :=
      aliases_of_existential_var aliases existential_var
    in
    if among demoted_var aliases_of_canonical_element
    then aliases
    else
      (existential_var, demoted_var :: aliases_of_canonical_element)
      :: aliases.

End Aliases_of_existentials.

(* ======================================================================== *)
(* The state of a join, and the join of aliases                             *)
(* ======================================================================== *)

Module Join_env.

  (* What the model's functions are given and the compiler's are not:
     the source of new names ([fresh l] is not in [l]); the equality of
     types and of environment extensions that stands where the compiler
     tests physical equality (-flambda2-join-sharing structural); the
     type of a constant that is not an alias
     ([Expand_head.Expanded_type.create_const]); and the depth at which
     the join stops looking at the variables it makes
     ([Flambda_features.join_depth]). *)
  Record parameters : Type := {
    fresh : list variable -> variable;
    type_eqb : flambda_type -> flambda_type -> bool;
    env_extension_eqb : env_extension -> env_extension -> bool;
    expand_const : const -> flambda_type;
    join_depth : nat
  }.

  (* [Join_env.t]. [types_in_target_env]: the equations of the target
     made so far, which are on variables. [types_in_joined_envs]: for a
     variable of the target, the types that the joined environments
     have for what it stands for, still to be joined.
     [definitions_of_existentials]: the vector that each existential
     variable stands for, with its kind. *)
  Record t : Type := {
    joined_envs : Joined_envs.t;
    types_in_target_env : finite_map flambda_type;
    types_in_joined_envs :
      finite_map (Keyed.finite_map Index.t flambda_type);
    aliases_of_existentials : Aliases_of_existentials.t;
    imported_vars : list variable;
    definitions_of_existentials :
      finite_map (Simples_in_joined_envs.t * flambda_kind);
    bindings : Bindings_in_target_env.t
  }.

  Definition create (joined_envs : Joined_envs.t)
      (bindings : Bindings_in_target_env.t) : t :=
    {| joined_envs := joined_envs;
       types_in_target_env := [];
       types_in_joined_envs := [];
       aliases_of_existentials := Aliases_of_existentials.empty;
       imported_vars := [];
       definitions_of_existentials := [];
       bindings := bindings |}.

  Definition source_env (env : t) : Source_env.t :=
    Bindings_in_target_env.source_env (bindings env).

  (* The bindings of the newer state whose key the older has not. (The
     compiler takes a difference of maps that share structure; each key
     is added once, so that is what it computes.) *)
  Definition new_definitions_of_existentials (env since : t)
      : finite_map (Simples_in_joined_envs.t * flambda_kind) :=
    filter_bindings
      (fun var _ => negb (is_bound var (definitions_of_existentials since)))
      (definitions_of_existentials env).

  Definition new_equations_in_joined_envs (env since : t)
      : finite_map (Keyed.finite_map Index.t flambda_type) :=
    filter_bindings
      (fun var _ => negb (is_bound var (types_in_joined_envs since)))
      (types_in_joined_envs env).

  Definition with_bindings (env : t) (bindings : Bindings_in_target_env.t)
      : t :=
    {| joined_envs := joined_envs env;
       types_in_target_env := types_in_target_env env;
       types_in_joined_envs := types_in_joined_envs env;
       aliases_of_existentials := aliases_of_existentials env;
       imported_vars := imported_vars env;
       definitions_of_existentials := definitions_of_existentials env;
       bindings := bindings |}.

  Definition existential_for_these_canonicals (p : parameters) (env : t)
      (canonicals : Simples_in_joined_envs.t) (kind : flambda_kind)
      : variable * t :=
    let '(existential_var, bindings) :=
      Bindings_in_target_env.existential_for_these_simples (fresh p)
        (bindings env) canonicals kind
    in
    let definitions_of_existentials :=
      if is_bound existential_var (definitions_of_existentials env)
      then definitions_of_existentials env
      else (existential_var, (canonicals, kind))
           :: definitions_of_existentials env
    in
    (existential_var,
     {| joined_envs := joined_envs env;
        types_in_target_env := types_in_target_env env;
        types_in_joined_envs := types_in_joined_envs env;
        aliases_of_existentials := aliases_of_existentials env;
        imported_vars := imported_vars env;
        definitions_of_existentials := definitions_of_existentials;
        bindings := bindings |}).

  (* The alias equation is not recorded yet: another representative may
     be chosen when the variables are defined. *)
  Definition add_alias_of_existential (p : parameters) (env : t)
      (kind : flambda_kind) (var : variable)
      (canonicals : Simples_in_joined_envs.t) : t :=
    let '(existential_var, env) :=
      existential_for_these_canonicals p env canonicals kind
    in
    {| joined_envs := joined_envs env;
       types_in_target_env := types_in_target_env env;
       types_in_joined_envs := types_in_joined_envs env;
       aliases_of_existentials :=
         Aliases_of_existentials.add (aliases_of_existentials env) var
           existential_var;
       imported_vars := imported_vars env;
       definitions_of_existentials := definitions_of_existentials env;
       bindings := bindings env |}.

  (* [None] is the compiler's [assert]: the variable has an equation in
     the target already. *)
  Definition add_equation_in_target_env (env : t) (var : variable)
      (ty : flambda_type) : option t :=
    if is_bound var (types_in_target_env env)
    then None
    else
      Some
        {| joined_envs := joined_envs env;
           types_in_target_env := (var, ty) :: types_in_target_env env;
           types_in_joined_envs := types_in_joined_envs env;
           aliases_of_existentials := aliases_of_existentials env;
           imported_vars := imported_vars env;
           definitions_of_existentials := definitions_of_existentials env;
           bindings := bindings env |}.

  Definition add_equations_in_joined_envs (env : t) (var : variable)
      (types : Keyed.finite_map Index.t flambda_type) : t :=
    {| joined_envs := joined_envs env;
       types_in_target_env := types_in_target_env env;
       types_in_joined_envs := (var, types) :: types_in_joined_envs env;
       aliases_of_existentials := aliases_of_existentials env;
       imported_vars := imported_vars env;
       definitions_of_existentials := definitions_of_existentials env;
       bindings := bindings env |}.

  Inductive canonical_in_target_env : Type :=
    | Canonical_in_source_env (simple : simple)
    | Import_from_all_joined_envs (var : variable)
    | Existential_for_these_canonicals
        (canonicals : Simples_in_joined_envs.t).

  (* What stands in the target for a vector of canonical [Simple]s: a
     [Simple] of the source that is that vector, a variable that is the
     vector in every environment of it and is to be imported, or an
     existential variable. *)
  Definition n_way_join_canonical_simples (env : t)
      (canonicals : Simples_in_joined_envs.t)
      : option canonical_in_target_env :=
    let* candidate :=
      Source_env.candidate_canonical_in_source_env (source_env env) canonicals
    in
    match candidate with
    | Source_env.No_simples_in_joined_envs
    | Source_env.No_canonical_in_source_env =>
      Some (Existential_for_these_canonicals canonicals)
    | Source_env.Canonical_in_all_joined_envs simple =>
      match simple with
      | Const _ | Symbol _ => Some (Canonical_in_source_env simple)
      | Var var =>
        if Source_env.exists_in_source_env (source_env env) var
        then Some (Canonical_in_source_env simple)
        else Some (Import_from_all_joined_envs var)
      end
    | Source_env.Latest_bound_source_var var =>
      let* equal :=
        Joined_envs.equal_in_all_joined_envs (joined_envs env) (Var var)
          canonicals
      in
      Some
        (if equal
         then Canonical_in_source_env (Var var)
         else Existential_for_these_canonicals canonicals)
    end.

  (* The five functions below are one recursion in the compiler
     ([import_or_delay_n_way_join_type] and those defined with it). It
     ends because a variable is imported once. Here each takes the
     function that imports a variable, and [import_var_transitive] is
     defined last, by recursion on fuel; [None] is also fuel run
     out. *)
  Definition import_type_transitive_with
      (import_var : t -> variable -> option t) (env : t) (ty : flambda_type)
      : option t :=
    List.fold_left
      (fun env var =>
         let* env := env in
         import_var env var)
      (free_variables ty) (Some env).

  Definition add_equals_in_joined_envs_with (p : parameters)
      (import_var : t -> variable -> option t) (env : t)
      (kind : flambda_kind) (var : variable)
      (simples : Simples_in_joined_envs.t) : option t :=
    let* canonical := n_way_join_canonical_simples env simples in
    match canonical with
    | Canonical_in_source_env canonical =>
      add_equation_in_target_env env var (alias_type_of kind canonical)
    | Import_from_all_joined_envs imported =>
      let* env := import_var env imported in
      add_equation_in_target_env env var (alias_type_of kind (Var imported))
    | Existential_for_these_canonicals canonicals =>
      Some (add_alias_of_existential p env kind var canonicals)
    end.

  (* Where the types are the same in every environment, that type is
     the equation of the target and its variables are imported;
     otherwise the types are kept to be joined. "The same" is the
     equality of types, between the type of the least index and each of
     the others. *)
  Definition import_or_delay_n_way_join_of_concrete_type_with (p : parameters)
      (import_var : t -> variable -> option t) (env : t) (var : variable)
      (types : Keyed.finite_map Index.t flambda_type) : option t :=
    let* '(index, any_ty) := Index.choose types in
    if forallb
         (fun binding =>
            Index.equal (fst binding) index
            || type_eqb p any_ty (snd binding))
         (Keyed.compact Index.equal types)
    then
      let* env := add_equation_in_target_env env var any_ty in
      import_type_transitive_with import_var env any_ty
    else Some (add_equations_in_joined_envs env var types).

  Definition import_or_delay_n_way_join_type_with (p : parameters)
      (import_var : t -> variable -> option t) (env : t) (var : variable)
      (types : Keyed.finite_map Index.t flambda_type) : option t :=
    let* '(_, any_ty) := Index.choose types in
    let* canonical_simples :=
      Joined_envs.get_alias_then_canonical_simple_ignoring_name_mode_exn
        (joined_envs env) types
    in
    match canonical_simples with
    | Some canonical_simples =>
      add_equals_in_joined_envs_with p import_var env (kind any_ty) var
        canonical_simples
    | None =>
      import_or_delay_n_way_join_of_concrete_type_with p import_var env var
        types
    end.

  (* A variable of the joined environments that the source has not
     becomes a variable of the target, with the types that the
     environments that have it have for it. If one of those types is an
     alias, the variable is made an alias of what stands for the vector
     of its canonical [Simple]s (itself, where its type is not an
     alias). *)
  Definition import_var_transitive_body (p : parameters)
      (import_var : t -> variable -> option t) (env : t) (var : variable)
      : option t :=
    if Source_env.exists_in_source_env (source_env env) var
    then Some env
    else if among var (imported_vars env)
    then Some env
    else
      let* kind := Joined_envs.kind_of_imported_var (joined_envs env) var in
      let env :=
        {| joined_envs := joined_envs env;
           types_in_target_env := types_in_target_env env;
           types_in_joined_envs := types_in_joined_envs env;
           aliases_of_existentials := aliases_of_existentials env;
           imported_vars := var :: imported_vars env;
           definitions_of_existentials := definitions_of_existentials env;
           bindings :=
             Bindings_in_target_env.import_from_all_envs (bindings env) var
               kind |}
      in
      let* types :=
        Joined_envs.find_imported_var (joined_envs env) var kind
      in
      let has_alias_type :=
        existsb (fun binding => is_some (get_alias_opt (snd binding)))
          (Keyed.compact Index.equal types)
      in
      if has_alias_type
      then
        let alias_types :=
          Keyed.map_values
            (fun ty =>
               match get_alias_opt ty with
               | None => alias_type_of kind (Var var)
               | Some _ => ty
               end)
            types
        in
        let* canonical_simples :=
          Joined_envs.get_alias_then_canonical_simple_ignoring_name_mode_exn
            (joined_envs env) alias_types
        in
        match canonical_simples with
        | None => None
        | Some canonical_simples =>
          add_equals_in_joined_envs_with p import_var env kind var
            canonical_simples
        end
      else
        import_or_delay_n_way_join_of_concrete_type_with p import_var env var
          types.

  Fixpoint import_var_transitive (p : parameters) (fuel : nat) (env : t)
      (var : variable) : option t :=
    match fuel with
    | O => None
    | S fuel =>
      import_var_transitive_body p (import_var_transitive p fuel) env var
    end.

  Definition import_type_transitive (p : parameters) (fuel : nat) (env : t)
      (ty : flambda_type) : option t :=
    import_type_transitive_with (import_var_transitive p fuel) env ty.

  Definition import_or_delay_n_way_join_type (p : parameters) (fuel : nat)
      (env : t) (var : variable)
      (types : Keyed.finite_map Index.t flambda_type) : option t :=
    import_or_delay_n_way_join_type_with p (import_var_transitive p fuel) env
      var types.

  Definition import_name_transitive (p : parameters) (fuel : nat) (env : t)
      (name : name) : option t :=
    match name with
    | Name_symbol _ => Some env
    | Name_var var => import_var_transitive p fuel env var
    end.

  (* The names that have an equation in the level of every joined
     environment: what [fold_incremental_join] visits. (The compiler
     finds them with iterators over the maps, in the order of names;
     here they are those of the first environment that the others have
     too, in the order of its level.) *)
  Definition names_with_equations_everywhere
      (equations_to_join : Joined_envs.t) : list name :=
    match Keyed.compact Index.equal equations_to_join with
    | [] => []
    | (_, (_, equations)) :: others =>
      filter
        (fun name =>
           forallb
             (fun other =>
                Keyed.is_bound name_eqb name (snd (snd other)))
             others)
        (Keyed.domain name_eqb equations)
    end.

  (* The types that the levels have for a name. *)
  Definition types_of_name_in_levels (equations_to_join : Joined_envs.t)
      (name : name) : Keyed.finite_map Index.t flambda_type :=
    flat_map
      (fun binding =>
         match Keyed.lookup name_eqb name (snd (snd binding)) with
         | Some ty => [(fst binding, ty)]
         | None => []
         end)
      (Keyed.compact Index.equal equations_to_join).

  (* [join_aliases_into_bindings]: each variable of the target that has
     a new equation in every joined environment is given its alias in
     the target, or its types are kept to be joined. Symbols are passed
     over: their types are not joined. The variables of the target are
     those that it has when the function is called (the compiler takes
     the test from the state that it is given, before the fold): a
     variable that is imported on the way is not looked at here, though
     every level has an equation for it. *)
  Definition join_aliases_into_bindings (p : parameters) (fuel : nat)
      (env : t) (equations_to_join : Joined_envs.t) : option t :=
    let exists_in_target_env :=
      Bindings_in_target_env.exists_in_target_env (bindings env)
    in
    List.fold_left
      (fun env name =>
         let* env := env in
         match name with
         | Name_symbol _ => Some env
         | Name_var var =>
           if exists_in_target_env var
           then
             import_or_delay_n_way_join_type p fuel env var
               (types_of_name_in_levels equations_to_join name)
           else Some env
         end)
      (names_with_equations_everywhere equations_to_join) (Some env).

  (* ---------------------------------------------------------------- *)
  (* Inverse relations                                                 *)
  (* ---------------------------------------------------------------- *)

  (* For a variable that holds the result of a relation, the relations
     and the names that they are of. *)
  Definition inverse_relations : Type :=
    finite_map (list (Relation.t * name)).

  Definition add_to_inverse_relations (relations : inverse_relations)
      (var : variable) (relation : Relation.t) (scrutinee : name)
      : inverse_relations :=
    (var,
     (relation, scrutinee)
     :: match lookup var relations with
        | Some existing => existing
        | None => []
        end)
    :: relations.

  (* The type of naked immediates that is a head with one more inverse
     relation. *)
  Definition with_inverse_relation (head : head_of_kind_naked_immediate)
      (relation : Relation.t) (scrutinee : name) : flambda_type :=
    Type_naked_immediate
      (Ok
         (No_alias
            (Head_of_kind_naked_immediate (naked_immediates head)
               ((relation, scrutinee) :: FlambdaTypeGrammar.inverse_relations
                                           head)))).

  Definition empty_naked_immediate_head : head_of_kind_naked_immediate :=
    Head_of_kind_naked_immediate Or_unknown.Unknown [].

  (* [TEE.add_or_replace_equation], which is [Name.Map.add]: the
     equation that the name had, if any, is replaced, so that a name
     has one equation at most (the well-formedness of types asks it of
     an extension). *)
  Definition add_or_replace_equation (extension : env_extension) (name : name)
      (ty : flambda_type) : env_extension :=
    match extension with
    | Env_extension equations =>
      Env_extension ((name, ty) :: Keyed.unbind name_eqb name equations)
    end.

  Definition equations_of (extension : env_extension)
      : Keyed.finite_map name flambda_type :=
    match extension with
    | Env_extension equations => equations
    end.

  (* [add_inverse_relation_to_env_extension]: the name, or the name
     that the extension makes it an alias of, gets the inverse relation
     in the extension. The compiler follows aliases until it has seen a
     name twice; here the fuel is the number of names to see. [None] is
     a fatal error of the compiler: the name has in the extension a
     type that is not of naked immediates, or a head with an empty set
     of numbers ([from_descr_non_empty]). *)
  Fixpoint add_inverse_relation_to_env_extension (fuel : nat)
      (seen : list name) (extension : env_extension) (name : name)
      (relation : Relation.t) (scrutinee : FlambdaTypeGrammar.name)
      : option env_extension :=
    match Keyed.lookup name_eqb name (equations_of extension) with
    | None =>
      Some
        (add_or_replace_equation extension name
           (with_inverse_relation empty_naked_immediate_head relation
              scrutinee))
    | Some existing_ty =>
      match existing_ty with
      | Type_naked_immediate Bottom => Some extension
      | Type_naked_immediate Unknown =>
        Some
          (add_or_replace_equation extension name
             (with_inverse_relation empty_naked_immediate_head relation
                scrutinee))
      | Type_naked_immediate (Ok (No_alias head)) =>
        match naked_immediates head with
        | Or_unknown.Known [] => None
        | Or_unknown.Known (_ :: _) | Or_unknown.Unknown =>
          Some
            (add_or_replace_equation extension name
               (with_inverse_relation head relation scrutinee))
        end
      | Type_naked_immediate (Ok (Equals simple)) =>
        match name_of_simple simple with
        | None => Some extension
        | Some name' =>
          match fuel with
          | O =>
            Some
              (add_or_replace_equation extension name
                 (with_inverse_relation empty_naked_immediate_head relation
                    scrutinee))
          | S fuel =>
            if Keyed.among name_eqb name' seen
            then
              Some
                (add_or_replace_equation extension name
                   (with_inverse_relation empty_naked_immediate_head
                      relation scrutinee))
            else
              add_inverse_relation_to_env_extension fuel (name :: seen)
                extension name' relation scrutinee
          end
        end
      | Type_value _ | Type_naked_number _ _ | Type_rec_info _
      | Type_region _ => None
      end
    end.

  (* [recover_inverse_relations]: from the type joined for a name, the
     inverse relations that its variables are to have. The relation
     [Get_tag] is recorded at the top only if the type has no
     immediates; otherwise it goes into the extension for blocks. Of a
     type that may be null only [Is_null] is recorded. The caller gives
     only types that exist in every joined environment. [None] is as
     for [add_inverse_relation_to_env_extension]. *)
  Definition recover_inverse_relations (relations : inverse_relations)
      (name : name) (ty : flambda_type)
      : option (flambda_type * inverse_relations) :=
    match ty with
    | Type_value (Ok (No_alias (Head_of_kind_value (Ok head) Not_null))) =>
      match head with
      | Variant is_int immediates get_tag blocks extensions is_unique =>
        let no_immediates :=
          match immediates with
          | Or_unknown.Known imm_ty => is_obviously_bottom imm_ty
          | Or_unknown.Unknown => false
          end
        in
        match no_immediates, get_tag with
        | true, Some get_tag_var =>
          Some
            (ty,
             add_to_inverse_relations relations get_tag_var Relation.Get_tag
               name)
        | _, _ =>
          let relations :=
            match is_int with
            | None => relations
            | Some is_int_var =>
              add_to_inverse_relations relations is_int_var Relation.Is_int
                name
            end
          in
          match get_tag with
          | None => Some (ty, relations)
          | Some get_tag_var =>
            let '(when_immediate, when_block) :=
              match extensions with
              | No_extensions => (empty_env_extension, empty_env_extension)
              | Ext when_immediate when_block => (when_immediate, when_block)
              end
            in
            let* when_block :=
              add_inverse_relation_to_env_extension
                (length (equations_of when_block)) [] when_block
                (Name_var get_tag_var) Relation.Get_tag name
            in
            Some
              (Type_value
                 (Ok
                    (No_alias
                       (Head_of_kind_value
                          (Ok
                             (Variant is_int immediates get_tag blocks
                                (Ext when_immediate when_block) is_unique))
                          Not_null))),
               relations)
          end
        end
      | Mutable_block _ | Boxed_number _ _ _ | Closures _ _ | String _ =>
        Some (ty, relations)
      end
    | Type_value
        (Ok (No_alias (Head_of_kind_value _ (Maybe_null is_null)))) =>
      Some
        (ty,
         match is_null with
         | None => relations
         | Some is_null_var =>
           add_to_inverse_relations relations is_null_var Relation.Is_null
             name
         end)
    | Type_value _ | Type_naked_immediate _ | Type_naked_number _ _
    | Type_rec_info _ | Type_region _ => Some (ty, relations)
    end.

  (* ---------------------------------------------------------------- *)
  (* The rounds                                                        *)
  (* ---------------------------------------------------------------- *)

  (* The join of types, which the join of environments is given:
     [n_way_join_type] in the compiler. *)
  Definition n_way_join_type : Type :=
    t -> list (Index.t * flambda_type)
    -> option (Or_unknown.t flambda_type * t).

  (* What a variant of the join decides, where it may decide otherwise
     than the compiler. There is one such decision: whether the
     relations that the type joined for a variable names are
     recovered. The round asks it with the state, the variable and the
     types that were joined. *)
  Record choices : Type := {
    recover_relations_on :
      t -> variable -> Keyed.finite_map Index.t flambda_type -> bool
  }.

  (* The compiler's: every joined environment gave a type. *)
  Definition compiler_choices : choices :=
    {| recover_relations_on :=
         fun env _ types =>
           Joined_envs.exists_in_all_joined_envs (joined_envs env) types |}.

  (* What a variable of the target stands for in a joined environment:
     the [Simple] that its definition has there, if the join made the
     variable; the variable, otherwise. *)
  Definition stands_for (env : t) (index : Index.t) (var : variable)
      : option simple :=
    match lookup var (definitions_of_existentials env) with
    | Some (simples, _) => Simples_in_joined_envs.find index simples
    | None => Some (Var var)
    end.

  (* A [Simple] has, in an environment, an alias that may occur in
     terms: a canonical element at the normal name mode, which is what
     [Analysis.create] asks of the definitions that it keeps for
     Simplify. That the element answered is bound at that mode is
     asked again. In an environment that has its invariant it is, if
     the [Simple] asked about is bound there (a symbol that is not
     defined is answered for itself, and is not bound); with the test
     nothing need be supposed of the environment or of the [Simple]. *)
  Definition has_alias_in_terms (tenv : Typing_env.t) (simple : simple)
      : bool :=
    match
      Typing_env.get_canonical_simple_exn tenv (Some Name_mode.normal) None
        simple
    with
    | Some (Some canonical) =>
      Typing_env.mem_simple (Some Name_mode.normal) tenv canonical
    | Some None | None => false
    end.

  (* In every joined environment the variable stands for a [Simple]
     that has an alias that may occur in terms. *)
  Definition in_terms_in_all_joined_envs (env : t) (var : variable) : bool :=
    forallb
      (fun index =>
         match
           Joined_envs.get_nth_joined_env (joined_envs env) index,
           stands_for env index var
         with
         | Some tenv, Some simple => has_alias_in_terms tenv simple
         | _, _ => false
         end)
      (Joined_envs.keys (joined_envs env)).

  (* Not the compiler's: the compiler's test, and the variable has a
     value at every use. A relation recovered for a variable [x] says
     of another, [r], that it holds the result of a primitive on [x];
     that is true of the value of [r] only where [x] has a value, and
     a variable that exists only in types may have none (discrepancy
     T5-10). A variable that may occur in terms has a value, and so
     has each of its aliases. *)
  Definition repaired_choices : choices :=
    {| recover_relations_on :=
         fun env var types =>
           Joined_envs.exists_in_all_joined_envs (joined_envs env) types
           && in_terms_in_all_joined_envs env var |}.

  (* [n_way_join_round]: the types kept for each variable are joined,
     and what is answered is the variable's equation in the target. *)
  Definition n_way_join_round (choices : choices)
      (n_way_join_type : n_way_join_type) (env : t)
      (equations_to_join :
         finite_map (Keyed.finite_map Index.t flambda_type))
      (relations : inverse_relations) : option (inverse_relations * t) :=
    List.fold_left
      (fun acc binding =>
         let* '(relations, env) := acc in
         let var := fst binding in
         let types := snd binding in
         let* '(joined, env) := n_way_join_type env (Index.bindings types) in
         match joined with
         | Or_unknown.Unknown => Some (relations, env)
         | Or_unknown.Known ty =>
           let* '(ty, relations) :=
             if recover_relations_on choices env var types
             then recover_inverse_relations relations (Name_var var) ty
             else Some (ty, relations)
           in
           let* env := add_equation_in_target_env env var ty in
           Some (relations, env)
         end)
      (compact equations_to_join) (Some (relations, env)).

  (* [n_way_join_delayed_equations]: rounds until a round keeps no new
     types. In the compiler it ends because a variable is imported
     once; here on fuel. *)
  Fixpoint n_way_join_delayed_equations (fuel : nat) (choices : choices)
      (n_way_join_type : n_way_join_type) (env_this_round : t)
      (relations : inverse_relations)
      (equations_this_round :
         finite_map (Keyed.finite_map Index.t flambda_type))
      : option (inverse_relations * t) :=
    match fuel with
    | O => None
    | S fuel =>
      if Keyed.is_empty equations_this_round
      then Some (relations, env_this_round)
      else
        let* '(relations, env_next_round) :=
          n_way_join_round choices n_way_join_type env_this_round
            equations_this_round relations
        in
        n_way_join_delayed_equations fuel choices n_way_join_type
          env_next_round relations
          (new_equations_in_joined_envs env_next_round env_this_round)
    end.

  (* The types of the [Simple]s that new existential variables stand
     for. *)
  Fixpoint equations_of_existentials (p : parameters) (env : t)
      (existentials :
         list (variable * (Simples_in_joined_envs.t * flambda_kind)))
      : option (finite_map (Keyed.finite_map Index.t flambda_type)) :=
    match existentials with
    | [] => Some []
    | (var, (simples, kind)) :: existentials =>
      let* types :=
        Joined_envs.find_simples_in_joined_envs (expand_const p)
          (joined_envs env) simples kind
      in
      let* equations := equations_of_existentials p env existentials in
      Some ((var, types) :: equations)
    end.

  (* [n_way_join_loop]: the types of what the new existential variables
     stand for are joined, which may make more of them, as many times
     as the depth allows. Those made in the last round get no
     equation. *)
  Fixpoint n_way_join_loop (depth_left : nat) (p : parameters) (fuel : nat)
      (choices : choices) (n_way_join_type : n_way_join_type)
      (env_this_round : t)
      (relations : inverse_relations)
      (existentials_this_round :
         finite_map (Simples_in_joined_envs.t * flambda_kind))
      : option (inverse_relations * t) :=
    match depth_left with
    | O => Some (relations, env_this_round)
    | S depth_left =>
      if Keyed.is_empty existentials_this_round
      then Some (relations, env_this_round)
      else
        let* equations_in_joined_envs :=
          equations_of_existentials p env_this_round
            (compact existentials_this_round)
        in
        let* '(relations, env_next_round) :=
          n_way_join_delayed_equations fuel choices n_way_join_type
            env_this_round relations equations_in_joined_envs
        in
        n_way_join_loop depth_left p fuel choices n_way_join_type
          env_next_round relations
          (new_definitions_of_existentials env_next_round env_this_round)
    end.

  (* ---------------------------------------------------------------- *)
  (* What the join of types calls                                      *)
  (* ---------------------------------------------------------------- *)

  (* [Join_env.joined_env]. *)
  Definition joined_env (env : t) (index : Index.t) : option Typing_env.t :=
    Joined_envs.get_nth_joined_env (joined_envs env) index.

  (* [Join_env.import_type]. *)
  Definition import_type (p : parameters) (fuel : nat) (env : t)
      (ty : flambda_type) : option t :=
    import_type_transitive p fuel env ty.

  (* [Join_env.n_way_join_simples]: the [Simple] of the target that
     stands for [Simple]s of the joined environments. *)
  Definition n_way_join_simples (p : parameters) (fuel : nat) (env : t)
      (kind : flambda_kind) (simples : list (Index.t * simple))
      : option (Or_bottom.t simple * t) :=
    match simples with
    | [] => Some (Or_bottom.Bottom, env)
    | _ :: _ =>
      let* canonicals :=
        Joined_envs.get_canonical_simples_ignoring_name_mode (joined_envs env)
          simples
      in
      let* canonical := n_way_join_canonical_simples env canonicals in
      match canonical with
      | Canonical_in_source_env simple => Some (Or_bottom.Ok simple, env)
      | Import_from_all_joined_envs var =>
        let* env := import_var_transitive p fuel env var in
        Some (Or_bottom.Ok (Var var), env)
      | Existential_for_these_canonicals canonicals =>
        let '(var, env) :=
          existential_for_these_canonicals p env canonicals kind
        in
        Some (Or_bottom.Ok (Var var), env)
      end
    end.

  (* [Join_env.n_way_join_env_extension]: the extension of the first
     input if the others are the same, with its names and the variables
     of its types imported; the empty extension otherwise. (The
     compiler's function also takes the join of types and the meet, and
     uses neither.) Every equation of the list is visited, and not only
     the first for each name: an extension of the compiler has one
     equation for a name, and for such an extension it is the same.
     With every equation visited, each variable that the extension
     mentions is imported whether or not its names are distinct. *)
  Definition n_way_join_env_extension (p : parameters) (fuel : nat) (env : t)
      (extensions : list (Index.t * env_extension))
      : option (Or_bottom.t (env_extension * t)) :=
    match extensions with
    | [] => Some Or_bottom.Bottom
    | (_, first_extension) :: other_extensions =>
      if forallb
           (fun binding => env_extension_eqb p (snd binding) first_extension)
           other_extensions
      then
        let* env :=
          List.fold_left
            (fun env equation =>
               let* env := env in
               let* env := import_name_transitive p fuel env (fst equation) in
               import_type_transitive p fuel env (snd equation))
            (equations_of first_extension) (Some env)
        in
        Some (Or_bottom.Ok (first_extension, env))
      else Some (Or_bottom.Ok (empty_env_extension, env))
    end.

  (* ---------------------------------------------------------------- *)
  (* Defining the variables of the target                              *)
  (* ---------------------------------------------------------------- *)

  (* What the join uses of the environment of the meet
     (types/env/meet_env.ml), as functions that it is given:
     [Meet_env.create], [current_typing_env], [add_variable_definition],
     [add_env_extension] (which is given the meet of expanded types;
     [None] is a fatal error or no fuel), [make_bottom] and
     [final_typing_env]. The last two fields are not in the compiler:
     the environment of the meet carries the names that new variables
     must avoid (FlambdaMeetEnv.v). *)
  Record meet_env_functions (meet_env meet_expanded_head : Type) : Type := {
    create_meet_env : Typing_env.t -> meet_env;
    current_typing_env : meet_env -> Typing_env.t;
    add_variable_definition :
      meet_env -> variable -> flambda_kind -> Name_mode.t -> meet_env;
    add_env_extension :
      meet_env -> env_extension -> meet_expanded_head -> option meet_env;
    make_bottom : meet_env -> meet_env;
    final_typing_env : meet_env -> Typing_env.t;
    names_to_avoid : meet_env -> list variable;
    with_names_to_avoid : meet_env -> list variable -> meet_env
  }.

  Arguments create_meet_env {meet_env meet_expanded_head} _.
  Arguments current_typing_env {meet_env meet_expanded_head} _.
  Arguments add_variable_definition {meet_env meet_expanded_head} _.
  Arguments add_env_extension {meet_env meet_expanded_head} _.
  Arguments make_bottom {meet_env meet_expanded_head} _.
  Arguments final_typing_env {meet_env meet_expanded_head} _.
  Arguments names_to_avoid {meet_env meet_expanded_head} _.
  Arguments with_names_to_avoid {meet_env meet_expanded_head} _.

  (* [move_definition]. *)
  Definition move_definition
      (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind))
      (from to_ : variable)
      : finite_map (Simples_in_joined_envs.t * flambda_kind) :=
    match lookup from definitions with
    | None => definitions
    | Some defn => (to_, defn) :: unbind from definitions
    end.

  (* [move_equation]. [None] is a fatal error of the compiler: the
     variable that the equation is moved to has one. *)
  Definition move_equation (equations : finite_map flambda_type)
      (from to_ : variable) : option (finite_map flambda_type) :=
    if is_bound to_ equations
    then None
    else
      match lookup from equations with
      | None => Some equations
      | Some ty => Some ((to_, ty) :: unbind from equations)
      end.

  (* [move_inverse_relation]. *)
  Definition move_inverse_relation (relations : inverse_relations)
      (from to_ : variable) : inverse_relations :=
    match lookup from relations with
    | None => relations
    | Some names =>
      let relations := unbind from relations in
      (to_,
       names
       ++ match lookup to_ relations with
          | Some existing => existing
          | None => []
          end)
      :: relations
    end.

  (* A variable is among [names_in_inverse_relations]: it holds the
     result of a relation, or a relation is on it. *)
  Definition in_inverse_relations (relations : inverse_relations)
      (var : variable) : bool :=
    is_bound var relations
    || existsb
         (fun binding =>
            existsb (fun pair => name_eqb (snd pair) (Name_var var))
              (snd binding))
         (compact relations).

  Definition count_occurrences (var : variable) (vars : list variable) : nat :=
    length (filter (Pos.eqb var) vars).

  (* The variable that stands for an existential variable, among the
     variables of the target that are its aliases: the one that the
     target binds first. (The compiler takes the least in the order of
     variables, [Set.choose_opt]; the model does not order names.) The
     outer [None] is for a variable that the target does not bind. *)
  Fixpoint earliest_bound (tenv : Typing_env.t) (best : variable)
      (candidates : list variable) : option variable :=
    match candidates with
    | [] => Some best
    | var :: candidates =>
      let* time := Typing_env.binding_time_and_mode tenv (Name_var var) in
      let* best_time :=
        Typing_env.binding_time_and_mode tenv (Name_var best)
      in
      earliest_bound tenv
        (if Binding_time.strictly_earlier
              (Binding_time.With_name_mode.binding_time time)
              (Binding_time.With_name_mode.binding_time best_time)
         then var
         else best)
        candidates
    end.

  Definition choose_alias (tenv : Typing_env.t) (aliases : list variable)
      : option (option variable) :=
    match aliases with
    | [] => Some None
    | var :: aliases => option_map Some (earliest_bound tenv var aliases)
    end.

  (* [expand], in [define_or_eliminate_variables_and_add_equations]:
     the type to put for a variable that is removed, in which the
     variables that are removed have been replaced in turn. The
     compiler's recursion ends because a variable that is removed is
     mentioned once; here it is on fuel. [None] is also the compiler's
     [assert]: the variable is not one of those removed. *)
  Fixpoint expand_variable (fuel : nat) (to_expand : finite_map flambda_type)
      (var : variable) : option flambda_type :=
    match fuel with
    | O => None
    | S fuel =>
      let* ty := lookup var to_expand in
      match get_alias_opt ty with
      | None =>
        project_variables_out (domain to_expand)
          (expand_variable fuel to_expand) ty
      | Some (Var var') =>
        if among var' (domain to_expand)
        then expand_variable fuel to_expand var'
        else Some ty
      | Some (Symbol _) | Some (Const _) => Some ty
      end
    end.

  (* [define_or_eliminate_variables_and_add_equations]: the imported
     variables are defined in the target; each existential variable is
     replaced by a variable of the target that is its alias, if it has
     one, or removed if it is mentioned once and has no part in an
     inverse relation, or defined; and the equations, with the types of
     the variables removed put in their place, are added to the target,
     and then the inverse relations. It answers the target and the
     definitions of the variables that are left. *)
  Definition define_or_eliminate_variables_and_add_equations
      {meet_env meet_expanded_head : Type}
      (ops : meet_env_functions meet_env meet_expanded_head) (fuel : nat)
      (meet_expanded_head_fn : meet_expanded_head) (env : t)
      (source_env : meet_env) (relations : inverse_relations)
      : option
          (meet_env * finite_map (Simples_in_joined_envs.t * flambda_kind)) :=
    let target_env :=
      List.fold_left
        (fun target_env binding =>
           add_variable_definition ops target_env (fst binding) (snd binding)
             Name_mode.in_types)
        (Bindings_in_target_env.imported_variables_in_order (bindings env))
        source_env
    in
    let equations := types_in_target_env env in
    let free_vars_in_equations :=
      flat_map (fun binding => free_variables (snd binding))
        (compact equations)
    in
    let unique_occurrence_is_in_equations (var : variable) : bool :=
      negb (in_inverse_relations relations var)
      && Nat.leb (count_occurrences var free_vars_in_equations) 1
    in
    let tenv_of_aliases := current_typing_env ops target_env in
    let* '(target_env, to_expand, equations, relations, definitions) :=
      List.fold_left
        (fun acc binding =>
           let* '(target_env, to_expand, equations, relations, definitions) :=
             acc
           in
           let var := fst binding in
           let var_kind := snd binding in
           let aliases_of_var :=
             Aliases_of_existentials.aliases_of_existential_var
               (aliases_of_existentials env) var
           in
           let* canon := choose_alias tenv_of_aliases aliases_of_var in
           match canon with
           | None =>
             if unique_occurrence_is_in_equations var
             then
               let ty :=
                 match lookup var equations with
                 | Some ty => ty
                 | None => unknown var_kind
                 end
               in
               Some
                 (target_env, (var, ty) :: to_expand, unbind var equations,
                  relations, unbind var definitions)
             else
               Some
                 (add_variable_definition ops target_env var var_kind
                    Name_mode.in_types,
                  to_expand, equations, relations, definitions)
           | Some canon_var =>
             let demoted_aliases :=
               filter (fun alias => negb (Pos.eqb alias canon_var))
                 aliases_of_var
             in
             let definitions := move_definition definitions var canon_var in
             let* equations := move_equation equations var canon_var in
             let relations := move_inverse_relation relations var canon_var in
             let ty := alias_type_of var_kind (Var canon_var) in
             let equations :=
               List.fold_left
                 (fun equations alias => (alias, ty) :: equations)
                 demoted_aliases equations
             in
             Some
               (target_env, (var, ty) :: to_expand, equations, relations,
                definitions)
           end)
        (Bindings_in_target_env.existential_variables_in_order (bindings env))
        (Some
           (target_env, [], equations, relations,
            definitions_of_existentials env))
    in
    let to_project := domain to_expand in
    let expand := expand_variable fuel to_expand in
    let* equations :=
      traverse
        (fun binding =>
           let* ty := project_variables_out to_project expand (snd binding) in
           Some (Name_var (fst binding), ty))
        (compact equations)
    in
    let* equations_for_inverse_relations :=
      traverse
        (fun binding =>
           let* ty :=
             project_variables_out to_project expand
               (Type_naked_immediate
                  (Ok
                     (No_alias
                        (Head_of_kind_naked_immediate Or_unknown.Unknown
                           (snd binding)))))
           in
           Some (Name_var (fst binding), ty))
        (compact relations)
    in
    let* target_env :=
      add_env_extension ops target_env (Env_extension equations)
        meet_expanded_head_fn
    in
    let* target_env :=
      add_env_extension ops target_env
        (Env_extension equations_for_inverse_relations) meet_expanded_head_fn
    in
    Some (target_env, definitions).

  (* ---------------------------------------------------------------- *)
  (* Cut and n-way join                                                *)
  (* ---------------------------------------------------------------- *)

  (* [cut_for_join], for each environment that is not bottom: the
     equations of the level cut from it. [None] is a fatal error of
     [Typing_env.cut]. *)
  Fixpoint cut_for_joins (cut_after : Scope.t)
      (envs : list (Index.t * Typing_env.t)) : option Joined_envs.t :=
    match envs with
    | [] => Some []
    | (index, tenv) :: envs =>
      let* rest := cut_for_joins cut_after envs in
      if Typing_env.is_bottom tenv
      then Some rest
      else
        let* level := Typing_env.cut tenv cut_after in
        Some ((index, (tenv, Typing_env_level.equations level)) :: rest)
    end.

  (* [cut_and_n_way_join0]: the target, the definitions of the
     variables that the join made, and (not in the compiler) the names
     of the join: those that it was given to avoid and the variables
     that it made. The last phase is given the environment of the
     meet with those names to avoid (not in the compiler either: see
     the header). If no environment is joined, the target is
     bottom. *)
  Definition cut_and_n_way_join0 {meet_env meet_expanded_head : Type}
      (ops : meet_env_functions meet_env meet_expanded_head) (p : parameters)
      (fuel : nat) (choices : choices) (n_way_join_type : n_way_join_type)
      (meet_expanded_head_fn : meet_expanded_head) (avoid : list variable)
      (source_env : meet_env) (source_tenv : Typing_env.t)
      (equations_to_join : Joined_envs.t)
      : option
          (meet_env * finite_map (Simples_in_joined_envs.t * flambda_kind)
           * list variable) :=
    if Keyed.is_empty equations_to_join
    then Some (make_bottom ops source_env, [], avoid)
    else
      let empty_bindings :=
        Bindings_in_target_env.from_source_env avoid source_tenv
      in
      let empty_env := create equations_to_join empty_bindings in
      let* env :=
        join_aliases_into_bindings p fuel empty_env equations_to_join
      in
      let* '(relations, env) :=
        n_way_join_delayed_equations fuel choices n_way_join_type env []
          (new_equations_in_joined_envs env empty_env)
      in
      let* '(relations, env) :=
        n_way_join_loop (join_depth p) p fuel choices n_way_join_type env
          relations (new_definitions_of_existentials env empty_env)
      in
      let names := Bindings_in_target_env.names_to_avoid (bindings env) in
      let* '(target_env, definitions) :=
        define_or_eliminate_variables_and_add_equations ops fuel
          meet_expanded_head_fn env
          (with_names_to_avoid ops source_env names) relations
      in
      Some (target_env, definitions, names).

  (* The variables that an environment has a type for: those that it
     knows. *)
  Definition variables_of_env (tenv : Typing_env.t) : list variable :=
    flat_map
      (fun binding =>
         match fst binding with
         | Name_var var => [var]
         | Name_symbol _ => []
         end)
      (Typing_env.names_to_types tenv).

  (* [Join_env.cut_and_n_way_join]: the join that the meet calls. The
     names to avoid are those of the environment of the meet and the
     variables that the environments given know (the compiler's new
     variables differ from all of them by construction). The target
     is answered as the last phase leaves it: its names to avoid are
     those, the variables that the join made, and what the meets of
     the last phase added. *)
  Definition cut_and_n_way_join {meet_env meet_expanded_head : Type}
      (ops : meet_env_functions meet_env meet_expanded_head) (p : parameters)
      (fuel : nat) (choices : choices) (n_way_join_type : n_way_join_type)
      (meet_expanded_head_fn : meet_expanded_head) (cut_after : Scope.t)
      (source_env : meet_env) (source_tenv : Typing_env.t)
      (joined_envs : list Typing_env.t) : option meet_env :=
    let* equations_to_join :=
      cut_for_joins cut_after (Index.number joined_envs)
    in
    let avoid :=
      names_to_avoid ops source_env ++ variables_of_env source_tenv
      ++ flat_map variables_of_env joined_envs
    in
    let* '(target_env, _, _) :=
      cut_and_n_way_join0 ops p fuel choices n_way_join_type
        meet_expanded_head_fn avoid source_env source_tenv equations_to_join
    in
    Some target_env.

End Join_env.


(* ======================================================================== *)
(* The analysis of a join                                                   *)
(* ======================================================================== *)

Module Analysis.

  (* [Join_env.Analysis.t]. [definitions_in_joined_envs]: for some
     variables of the joined environment, the canonical [Simple] that
     each stands for in the environments joined.
     [canonical_definitions_at_normal_mode]: the same for those of the
     variables that stand, in every environment where they stand for
     something, for a [Simple] that may occur in terms there.
     [external_ids]: the identifier that the caller gave each
     environment joined that is not bottom. *)
  Record t (A : Type) : Type := {
    definitions_in_joined_envs :
      finite_map (Simples_in_joined_envs.t * flambda_kind);
    canonical_definitions_at_normal_mode :
      finite_map (Simples_in_joined_envs.t * flambda_kind);
    external_ids : Keyed.finite_map Index.t A
  }.

  Arguments definitions_in_joined_envs {A} _.
  Arguments canonical_definitions_at_normal_mode {A} _.
  Arguments external_ids {A} _.

  (* The analysis that says nothing: that of a join of no environment. *)
  Definition empty {A : Type} : t A :=
    {| definitions_in_joined_envs := [];
       canonical_definitions_at_normal_mode := [];
       external_ids := [] |}.

  (* The values that a variable has at the uses, as
     [Variable_refined_at_join.fold_values_at_uses] gives them: a
     constant where the variable stands for one, and unknown where it
     stands for a name. [None] is a fatal error of the compiler: an
     index with no identifier. *)
  Definition value_at_use (simple : simple) : Or_unknown.t const :=
    match simple with
    | Const const => Or_unknown.Known const
    | Var _ | Symbol _ => Or_unknown.Unknown
    end.

  Fixpoint values_of_bindings {A : Type}
      (external_ids : Keyed.finite_map Index.t A)
      (canonicals_in_joined_envs : list (Index.t * simple))
      : option (list (A * Or_unknown.t const)) :=
    match canonicals_in_joined_envs with
    | [] => Some []
    | (index, simple) :: canonicals_in_joined_envs =>
      let* external_id := Keyed.lookup Index.equal index external_ids in
      let* values :=
        values_of_bindings external_ids canonicals_in_joined_envs
      in
      Some ((external_id, value_at_use simple) :: values)
    end.

  (* Each index of the vector is visited once: a binding that another
     shadows is not one of the map. *)
  Definition values_at_uses {A : Type}
      (external_ids : Keyed.finite_map Index.t A)
      (canonicals_in_joined_envs : Simples_in_joined_envs.t)
      : option (list (A * Or_unknown.t const)) :=
    values_of_bindings external_ids
      (Keyed.compact Index.equal canonicals_in_joined_envs).

  (* [Join_env.Analysis.simple_refined_at_join]. *)
  Inductive simple_refined_at_join : Type :=
    | Not_refined_at_join
    | Invariant_in_all_uses (simple : simple)
    | Variable_refined_at_these_uses
        (canonicals_in_joined_envs : Simples_in_joined_envs.t)
        (kind : flambda_kind).

  (* What the join did to a [Simple] of the environment [env], which is
     the joined environment or one made from it. *)
  Definition simple_refined_at_join_of {A : Type} (analysis : t A)
      (env : Typing_env.t) (simple : simple) : simple_refined_at_join :=
    match Typing_env.get_canonical_simple_ignoring_name_mode env simple with
    | Const const => Invariant_in_all_uses (Const const)
    | Symbol symbol => Invariant_in_all_uses (Symbol symbol)
    | Var var =>
      match lookup var (definitions_in_joined_envs analysis) with
      | None => Not_refined_at_join
      | Some (canonicals_in_joined_envs, kind) =>
        Variable_refined_at_these_uses canonicals_in_joined_envs kind
      end
    end.

  (* The [Simple] that may be given, at each use, for a variable that
     the join made: [Simples_at_join.fold_definitions_at_uses]. [None]
     as for [values_at_uses]. *)
  Fixpoint definitions_of_bindings {A : Type}
      (external_ids : Keyed.finite_map Index.t A)
      (canonicals_in_joined_envs : list (Index.t * simple))
      : option (list (A * simple)) :=
    match canonicals_in_joined_envs with
    | [] => Some []
    | (index, simple) :: canonicals_in_joined_envs =>
      let* external_id := Keyed.lookup Index.equal index external_ids in
      let* definitions :=
        definitions_of_bindings external_ids canonicals_in_joined_envs
      in
      Some ((external_id, simple) :: definitions)
    end.

  Definition definitions_at_uses {A : Type}
      (external_ids : Keyed.finite_map Index.t A)
      (canonicals_in_joined_envs : Simples_in_joined_envs.t)
      : option (list (A * simple)) :=
    definitions_of_bindings external_ids
      (Keyed.compact Index.equal canonicals_in_joined_envs).

  (* [Join_env.Analysis.fold_variables_created_at_join]: the variables
     that have a definition at the normal name mode, each with that
     definition and its kind. *)
  Definition variables_created_at_join {A : Type} (analysis : t A)
      : list (variable * Simples_in_joined_envs.t * flambda_kind) :=
    map
      (fun binding =>
         (fst binding, fst (snd binding), snd (snd binding)))
      (compact (canonical_definitions_at_normal_mode analysis)).

  (* The canonical elements of the [Simple]s of a vector that may occur
     in terms, each in its environment. The outer [None] is a fatal
     error of the compiler (an index with no environment, a name that
     is not bound); the inner one is its [Not_found]: one of them has
     no alias that may occur in terms. *)
  Fixpoint simples_at_normal_mode (joined_envs : Joined_envs.t)
      (simples : list (Index.t * simple))
      : option (option Simples_in_joined_envs.t) :=
    match simples with
    | [] => Some (Some [])
    | (index, simple) :: simples =>
      let* typing_env := Joined_envs.get_nth_joined_env joined_envs index in
      let* canonical :=
        Typing_env.get_canonical_simple_exn typing_env
          (Some Name_mode.normal) None simple
      in
      let* rest := simples_at_normal_mode joined_envs simples in
      Some
        match canonical, rest with
        | Some canonical, Some rest => Some ((index, canonical) :: rest)
        | _, _ => None
        end
    end.

  (* The definitions that are, at every index, a [Simple] that may
     occur in terms. *)
  Fixpoint definitions_at_normal_mode (joined_envs : Joined_envs.t)
      (definitions :
         list (variable * (Simples_in_joined_envs.t * flambda_kind)))
      : option (finite_map (Simples_in_joined_envs.t * flambda_kind)) :=
    match definitions with
    | [] => Some []
    | (var, (simples, kind)) :: definitions =>
      let* at_normal_mode :=
        simples_at_normal_mode joined_envs
          (Keyed.compact Index.equal simples)
      in
      let* rest := definitions_at_normal_mode joined_envs definitions in
      Some
        match at_normal_mode with
        | Some simples => (var, (simples, kind)) :: rest
        | None => rest
        end
    end.

  (* [Join_env.Analysis.create]. *)
  Definition create {A : Type} (external_ids : Keyed.finite_map Index.t A)
      (joined_envs : Joined_envs.t)
      (definitions_in_joined_envs :
         finite_map (Simples_in_joined_envs.t * flambda_kind))
      : option (t A) :=
    let* canonical_definitions_at_normal_mode :=
      definitions_at_normal_mode joined_envs
        (compact definitions_in_joined_envs)
    in
    Some
      {| definitions_in_joined_envs := definitions_in_joined_envs;
         canonical_definitions_at_normal_mode :=
           canonical_definitions_at_normal_mode;
         external_ids := external_ids |}.

End Analysis.

(* ======================================================================== *)
(* The join that Simplify calls                                             *)
(* ======================================================================== *)

(* [Join_env.cut_and_n_way_join_with_analysis]: the environments that
   are bottom are left out, with their identifiers; the target is the
   source with the level that the join makes; the analysis is made of
   the definitions that are left. The first argument is the names that
   new variables must avoid, which the compiler has no need of. *)
Definition cut_and_n_way_join_with_analysis {A meet_env meet_expanded_head
    : Type}
    (ops : Join_env.meet_env_functions meet_env meet_expanded_head)
    (p : Join_env.parameters) (fuel : nat) (choices : Join_env.choices)
    (n_way_join_type : Join_env.n_way_join_type)
    (meet_expanded_head_fn : meet_expanded_head) (avoid : list variable)
    (cut_after : Scope.t) (source_tenv : Typing_env.t)
    (joined_envs : list (A * Typing_env.t))
    : option (Typing_env.t * Analysis.t A) :=
  let numbered := Index.number joined_envs in
  let external_ids :=
    flat_map
      (fun binding =>
         if Typing_env.is_bottom (snd (snd binding))
         then []
         else [(fst binding, fst (snd binding))])
      numbered
  in
  let* equations_to_join :=
    Join_env.cut_for_joins cut_after
      (map (fun binding => (fst binding, snd (snd binding))) numbered)
  in
  let source_env := Join_env.create_meet_env ops source_tenv in
  let* '(target_env, definitions, _) :=
    Join_env.cut_and_n_way_join0 ops p fuel choices n_way_join_type
      meet_expanded_head_fn avoid source_env source_tenv equations_to_join
  in
  let target_env := Join_env.final_typing_env ops target_env in
  let* join_analysis :=
    Analysis.create external_ids equations_to_join definitions
  in
  Some (target_env, join_analysis).

(* [Join_levels.cut_and_n_way_join] (types/join_levels.ml) where the
   join algorithm is the n-way one: the two checks of
   [check_join_inputs] (each parameter of the continuation, and each
   symbol of a constant lifted while its body was simplified, is
   defined in the environment at the fork), then the join, with the
   uses in the reverse of the order given. So the index of a use is
   its position from the end of the list that Simplify gives. *)
Definition join_levels_cut_and_n_way_join {A meet_env meet_expanded_head
    : Type}
    (ops : Join_env.meet_env_functions meet_env meet_expanded_head)
    (p : Join_env.parameters) (fuel : nat) (choices : Join_env.choices)
    (n_way_join_type : Join_env.n_way_join_type)
    (meet_expanded_head_fn : meet_expanded_head) (avoid : list variable)
    (definition_typing_env : Typing_env.t)
    (ts_and_use_ids : list (Typing_env.t * A)) (params : list variable)
    (cut_after : Scope.t) (extra_lifted_consts_in_use_envs : list symbol)
    : option (Typing_env.t * Analysis.t A) :=
  if forallb
       (fun param =>
          Typing_env.mem None definition_typing_env (Name_var param))
       params
     && forallb
          (fun symbol =>
             Typing_env.mem None definition_typing_env (Name_symbol symbol))
          extra_lifted_consts_in_use_envs
  then
    cut_and_n_way_join_with_analysis ops p fuel choices n_way_join_type
      meet_expanded_head_fn avoid cut_after definition_typing_env
      (rev (map (fun use => (snd use, fst use)) ts_and_use_ids))
  else None.
