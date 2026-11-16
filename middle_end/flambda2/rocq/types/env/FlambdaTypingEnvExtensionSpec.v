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

(* The extensions of a typing environment: what the functions on them
   answer, and what an extension taken from an environment means.

   This file is to be reviewed, with FlambdaTypingEnvExtension.v,
   against middle_end/flambda2/types/env/typing_env_extension.ml and
   its interface. The functions there are read against the OCaml; what
   they answer is stated here.

   An extension is equations on names: types that names are known to
   have under some condition. Types hold them (the extension of a case
   of a variant or of a row-like type holds when the value is in that
   case), the meet adds them to an environment, and a level cut out of
   an environment is made into one. What an extension means is in the
   meaning of types ([env_extension_satisfied],
   FlambdaTypeMeaningSpec.v): every name with an equation has a value,
   of the type of the equation.

   What is stated.
   - The equations: [bindings] has exactly the equations that count,
     each name once; where no name has two equations it is the list
     itself, so that [fold] goes through the equations as they are
     written.
   - Of each function that makes an extension, the equation that the
     answer has for a name, and what satisfies the answer: [empty],
     [one_equation], [add_or_replace_equation] (and [replace_equation],
     which is the same function), [disjoint_union].
   - An extension whose equations are types that a typing environment
     has for the names is satisfied by what satisfies the environment
     strictly and gives those names values
     ([environment_extension_satisfied_statement]). This is what makes
     the extension of a level cut out of an environment true: the
     statement about [Typing_env.cut] that gives the first premise is
     with the invariant of the typing environment.
   - An extension with extra variables is satisfied when some values of
     the extra variables satisfy its equations
     ([with_extra_variables_satisfied]); with no extra variable it is a
     plain extension, and one more extra variable asks less.
   - Renaming. The renaming and the free names of an extension are
     those of the grammar of types, and their laws are in
     FlambdaTypeRenameSpec.v. Here: the functions on extensions commute
     with renaming, [fold] for a function that commutes with it, and
     the two checks do not see it; the renamings act on extensions with
     extra variables as they do on types (identity, composition, the
     free names of a renamed extension, a renaming that fixes the free
     names); the functions on them commute with renaming; and the
     renamed environment satisfies the renamed extension.

   Differences from the OCaml, all deliberate.
   - The equations are an association list of which the first binding
     of a name counts (FlambdaMapSpec.v). [add_or_replace_equation]
     takes out the earlier equations of the name, so that it keeps a
     list in which no name has two equations as such; the OCaml's map
     has one equation for a name by construction.
   - The OCaml's [fold] goes through the equations in the order of
     names; the model's in the order of the list. What is stated of the
     equation that a name has, and of what satisfies an extension, does
     not depend on the order. Some statements are about the list as it
     is written, and keep the model's order: that [bindings] is the
     list itself where no name has two equations, and that the
     functions commute with renaming, the folds among them, which go
     through the renamed equations in the same order. (The extra
     variables are a list in the OCaml too, and its fold goes through
     them in the order of the list.)
   - [Name.Map.disjoint_union] stops with a fatal error on a name that
     both have; [disjoint_union] answers [None] there.
   - The checks that the OCaml makes only under a flag
     ([More_type_creators.check_equation], [invariant], the warning of
     [add_or_replace_equation] when it replaces an equation) are not
     made by the functions; [check_equation] and [invariant] are
     functions of their own that answer whether the check passes.
     Of them it is stated only that they do not see a renaming.
   - [With_extra_variables.add_definition] does not check the kind it
     is given: a variable of the model has no kind.
   - Not modelled: [print], [ids_for_export].
   - The OCaml does not treat the extra variables of an extension as
     bound by it ("At present [existential_vars] is not treated as a
     binding site", typing_env_extension.mli): its [free_names] lists
     them and its [apply_renaming] renames them, and so do the model's.
     The meaning given here binds them: what satisfies an extension
     does not depend on the values of its extra variables
     ([with_extra_variables_bound_statement]). The two agree as far as
     a renaming goes, since a renaming is a permutation and renames a
     bound variable with its uses
     ([with_extra_variables_satisfied_rename_statement]); the free
     names are more than the names that the meaning depends on.

   The proofs are in FlambdaTypingEnvExtensionProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaNominalSpec FlambdaAlphaSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypeRenameSpec FlambdaBindingTime
  FlambdaTypingEnv FlambdaTypingEnvSpec FlambdaTypingEnvExtension.
Import ListNotations.

(* ======================================================================== *)
(* The equations                                                            *)
(* ======================================================================== *)

Definition to_map_from_map_statement : Prop :=
  (∀ equations : Keyed.finite_map name flambda_type,
    Typing_env_extension.to_map (Typing_env_extension.from_map equations)
    = equations)
  ∧ (∀ t : Typing_env_extension.t,
      Typing_env_extension.from_map (Typing_env_extension.to_map t) = t).

(* [bindings] has the equations that count, each name once. *)
Definition bindings_lookup_statement : Prop :=
  ∀ (t : Typing_env_extension.t) (n : name) (ty : flambda_type),
    (n, ty) ∈ Typing_env_extension.bindings t
    <-> Keyed.lookup name_eqb n (Typing_env_extension.to_map t) = Some ty.

Definition bindings_distinct_statement : Prop :=
  ∀ t : Typing_env_extension.t,
    Keyed.distinct name_eqb (map fst (Typing_env_extension.bindings t))
    = true.

(* Where no name has two equations, they are the equations as they are
   written, in that order. *)
Definition bindings_of_distinct_statement : Prop :=
  ∀ t : Typing_env_extension.t,
    Keyed.distinct name_eqb (map fst (Typing_env_extension.to_map t)) = true ->
    Typing_env_extension.bindings t = Typing_env_extension.to_map t.

Definition is_empty_lookup_statement : Prop :=
  ∀ t : Typing_env_extension.t,
    Typing_env_extension.is_empty t = true
    <-> (∀ n : name,
          Keyed.lookup name_eqb n (Typing_env_extension.to_map t) = None).

Definition has_equation_lookup_statement : Prop :=
  ∀ (t : Typing_env_extension.t) (n : name),
    Typing_env_extension.has_equation n t = true
    <-> (∃ ty : flambda_type,
          Keyed.lookup name_eqb n (Typing_env_extension.to_map t) = Some ty).

(* ======================================================================== *)
(* Making an extension                                                      *)
(* ======================================================================== *)

(* A name has a value of a type. *)
Definition name_has_type (statics : static_env) (vars : env) (n : name)
    (ty : flambda_type) : Prop :=
  ∃ v : value, value_of_name vars n = Some v ∧ type_satisfied statics vars v ty.

Definition empty_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env),
    env_extension_satisfied statics vars Typing_env_extension.empty.

Definition one_equation_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (n : name) (ty : flambda_type),
    env_extension_satisfied statics vars
      (Typing_env_extension.one_equation n ty)
    <-> name_has_type statics vars n ty.

(* The equation that the answer of [add_or_replace_equation] has for a
   name: the new one for the name given, and for another name the one
   that there was. *)
Definition add_or_replace_equation_lookup_statement : Prop :=
  ∀ (t : Typing_env_extension.t) (n : name) (ty : flambda_type) (n' : name),
    Keyed.lookup name_eqb n'
      (Typing_env_extension.to_map
         (Typing_env_extension.add_or_replace_equation t n ty))
    = if name_eqb n' n
      then Some ty
      else Keyed.lookup name_eqb n' (Typing_env_extension.to_map t).

(* It keeps a list in which no name has two equations as such. *)
Definition add_or_replace_equation_distinct_statement : Prop :=
  ∀ (t : Typing_env_extension.t) (n : name) (ty : flambda_type),
    Keyed.distinct name_eqb (map fst (Typing_env_extension.to_map t)) = true ->
    Keyed.distinct name_eqb
      (map fst
         (Typing_env_extension.to_map
            (Typing_env_extension.add_or_replace_equation t n ty)))
    = true.

(* What satisfies the answer: the name has the new type, and the other
   names have the types that they had. *)
Definition add_or_replace_equation_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (t : Typing_env_extension.t)
    (n : name) (ty : flambda_type),
    env_extension_satisfied statics vars
      (Typing_env_extension.add_or_replace_equation t n ty)
    <-> name_has_type statics vars n ty
        ∧ (∀ (n' : name) (ty' : flambda_type),
            n' <> n ->
            Keyed.lookup name_eqb n' (Typing_env_extension.to_map t)
            = Some ty' ->
            name_has_type statics vars n' ty').

Definition replace_equation_statement : Prop :=
  ∀ (t : Typing_env_extension.t) (n : name) (ty : flambda_type),
    Typing_env_extension.replace_equation t n ty
    = Typing_env_extension.add_or_replace_equation t n ty.

(* [disjoint_union] answers when no name has an equation in both... *)
Definition disjoint_union_defined_statement : Prop :=
  ∀ t1 t2 : Typing_env_extension.t,
    Typing_env_extension.disjoint_union t1 t2 <> None
    <-> (∀ n : name,
          Typing_env_extension.has_equation n t1 = true ->
          Typing_env_extension.has_equation n t2 = false).

(* ... and what satisfies the answer satisfies both, and conversely. *)
Definition disjoint_union_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (t1 t2 t : Typing_env_extension.t),
    Typing_env_extension.disjoint_union t1 t2 = Some t ->
    (env_extension_satisfied statics vars t
     <-> env_extension_satisfied statics vars t1
         ∧ env_extension_satisfied statics vars t2).

(* ======================================================================== *)
(* An extension taken from an environment                                   *)
(* ======================================================================== *)

(* An extension each of whose equations is the type that a typing
   environment has for the name is satisfied by a valuation that
   satisfies the environment strictly, if the names of the equations
   have values in it. *)

(* CR mshinwell: the last premise is needed because an extension is
   satisfied only if every name with an equation has a value
   ([env_extension_satisfied]), while a valuation that satisfies a
   typing environment may give a name of the environment no value, and
   then nothing is asked of its type (FlambdaTypingEnvSpec.v,
   [names_have_their_types_but]). So the extension of a level cut out
   of an environment is not, as it stands, a consequence of the
   environment: the valuation has to give values to the names of the
   level's equations. It is not certain which of the two meanings is
   to move: an extension could ask a type only of a name that has a
   value, as the environment does. Decided overnight: the premise is
   kept and the meanings are left as they are. *)
Definition environment_extension_satisfied_statement : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (t : Typing_env_extension.t),
    typing_env_satisfied_strictly statics valuation tenv ->
    (∀ (n : name) (ty : flambda_type),
      Keyed.lookup name_eqb n (Typing_env_extension.to_map t) = Some ty ->
      ∃ binding_time_and_mode : Binding_time.With_name_mode.t,
        Keyed.lookup name_eqb n (Typing_env.names_to_types tenv)
        = Some (ty, binding_time_and_mode)) ->
    (∀ (n : name) (ty : flambda_type),
      Keyed.lookup name_eqb n (Typing_env_extension.to_map t) = Some ty ->
      ∃ v : value, value_of_name valuation n = Some v) ->
    env_extension_satisfied statics valuation t.

(* The same of the extension that a level is made into
   ([Typing_env_level.as_extension_without_bindings]), in terms of the
   equations of the level. *)
Definition level_extension_satisfied_statement : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (level : Typing_env_level.t),
    typing_env_satisfied_strictly statics valuation tenv ->
    (∀ (n : name) (ty : flambda_type),
      Keyed.lookup name_eqb n (Typing_env_level.equations level) = Some ty ->
      ∃ binding_time_and_mode : Binding_time.With_name_mode.t,
        Keyed.lookup name_eqb n (Typing_env.names_to_types tenv)
        = Some (ty, binding_time_and_mode)) ->
    (∀ (n : name) (ty : flambda_type),
      Keyed.lookup name_eqb n (Typing_env_level.equations level) = Some ty ->
      ∃ v : value, value_of_name valuation n = Some v) ->
    env_extension_satisfied statics valuation
      (Typing_env_level.as_extension_without_bindings level).

(* ======================================================================== *)
(* Extensions with extra variables                                          *)
(* ======================================================================== *)

(* An extension with extra variables is satisfied by an environment when
   some values of the extra variables satisfy its equations: an
   environment that differs only at the extra variables satisfies
   them. *)

(* CR mshinwell: this meaning binds the extra variables: what they hold
   in the environment given does not count. The OCaml does not treat
   them as bound ([With_extra_variables.free_names] lists them and
   [apply_renaming] renames them; the interface says "At present
   [existential_vars] is not treated as a binding site"). They are made
   by [Expand_head.make_suitable_for_environment] as fresh variables
   that stand for names the target environment does not have, which is
   why they are read here as existential. It is not certain that every
   use agrees. Decided overnight. *)
Definition with_extra_variables_satisfied (statics : static_env) (vars : env)
    (t : Typing_env_extension.With_extra_variables.t) : Prop :=
  ∃ vars' : env,
    (∀ x : variable,
      ~ x ∈ Typing_env_extension.With_extra_variables.existential_vars t ->
      lookup x vars' = lookup x vars)
    ∧ env_extension_satisfied statics vars'
        (Env_extension
           (Typing_env_extension.With_extra_variables.equations t)).

Definition with_extra_variables_empty_statement : Prop :=
  ∀ (statics : static_env) (vars : env),
    with_extra_variables_satisfied statics vars
      Typing_env_extension.With_extra_variables.empty.

(* With no extra variable it is the plain extension. *)
Definition with_no_extra_variables_statement : Prop :=
  ∀ (statics : static_env) (vars : env)
    (equations : Keyed.finite_map name flambda_type),
    with_extra_variables_satisfied statics vars
      (Typing_env_extension.With_extra_variables.create [] equations)
    <-> env_extension_satisfied statics vars (Env_extension equations).

(* One more extra variable asks less. *)
Definition add_definition_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env)
    (t : Typing_env_extension.With_extra_variables.t) (var : variable)
    (kind : flambda_kind),
    with_extra_variables_satisfied statics vars t ->
    with_extra_variables_satisfied statics vars
      (Typing_env_extension.With_extra_variables.add_definition t var kind).

(* The equation that [With_extra_variables.add_or_replace_equation]
   answers for a name, as for [add_or_replace_equation]; the extra
   variables are as they were. *)
Definition with_extra_variables_add_or_replace_equation_statement : Prop :=
  ∀ (t : Typing_env_extension.With_extra_variables.t) (n : name)
    (ty : flambda_type) (n' : name),
    Typing_env_extension.With_extra_variables.existential_vars
      (Typing_env_extension.With_extra_variables.add_or_replace_equation t n
         ty)
    = Typing_env_extension.With_extra_variables.existential_vars t
    ∧ Keyed.lookup name_eqb n'
        (Typing_env_extension.With_extra_variables.equations
           (Typing_env_extension.With_extra_variables.add_or_replace_equation
              t n ty))
      = if name_eqb n' n
        then Some ty
        else
          Keyed.lookup name_eqb n'
            (Typing_env_extension.With_extra_variables.equations t).

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

(* The functions on extensions commute with renaming. *)

Definition apply_renaming_to_map_statement : Prop :=
  ∀ (r : renaming) (t : Typing_env_extension.t),
    Typing_env_extension.to_map (Typing_env_extension.apply_renaming t r)
    = Keyed.map_bindings (rename_name r) (rename_type r)
        (Typing_env_extension.to_map t).

Definition apply_renaming_bindings_statement : Prop :=
  ∀ (r : renaming) (t : Typing_env_extension.t),
    Typing_env_extension.bindings (Typing_env_extension.apply_renaming t r)
    = Keyed.map_bindings (rename_name r) (rename_type r)
        (Typing_env_extension.bindings t).

Definition apply_renaming_is_empty_statement : Prop :=
  ∀ (r : renaming) (t : Typing_env_extension.t),
    Typing_env_extension.is_empty (Typing_env_extension.apply_renaming t r)
    = Typing_env_extension.is_empty t.

Definition apply_renaming_has_equation_statement : Prop :=
  ∀ (r : renaming) (t : Typing_env_extension.t) (n : name),
    Typing_env_extension.has_equation (rename_name r n)
      (Typing_env_extension.apply_renaming t r)
    = Typing_env_extension.has_equation n t.

Definition apply_renaming_one_equation_statement : Prop :=
  ∀ (r : renaming) (n : name) (ty : flambda_type),
    Typing_env_extension.apply_renaming
      (Typing_env_extension.one_equation n ty) r
    = Typing_env_extension.one_equation (rename_name r n) (rename_type r ty).

Definition apply_renaming_add_or_replace_equation_statement : Prop :=
  ∀ (r : renaming) (t : Typing_env_extension.t) (n : name)
    (ty : flambda_type),
    Typing_env_extension.apply_renaming
      (Typing_env_extension.add_or_replace_equation t n ty) r
    = Typing_env_extension.add_or_replace_equation
        (Typing_env_extension.apply_renaming t r) (rename_name r n)
        (rename_type r ty).

Definition apply_renaming_disjoint_union_statement : Prop :=
  ∀ (r : renaming) (t1 t2 : Typing_env_extension.t),
    Typing_env_extension.disjoint_union
      (Typing_env_extension.apply_renaming t1 r)
      (Typing_env_extension.apply_renaming t2 r)
    = option_map (fun t => Typing_env_extension.apply_renaming t r)
        (Typing_env_extension.disjoint_union t1 t2).

(* [fold] over the renamed extension, with a function that does to the
   renamed state, for the renamed name and type, what [equation] does:
   the answer is the renamed answer. [state] is how the state is
   renamed. *)
Definition apply_renaming_fold_statement : Prop :=
  ∀ (B B' : Type) (r : renaming) (state : B -> B')
    (equation : name -> flambda_type -> B -> B)
    (equation' : name -> flambda_type -> B' -> B')
    (t : Typing_env_extension.t) (acc : B),
    (∀ (n : name) (ty : flambda_type) (acc : B),
      equation' (rename_name r n) (rename_type r ty) (state acc)
      = state (equation n ty acc)) ->
    Typing_env_extension.fold equation'
      (Typing_env_extension.apply_renaming t r) (state acc)
    = state (Typing_env_extension.fold equation t acc).

(* The two checks do not see a renaming. *)
Definition apply_renaming_checks_statement : Prop :=
  (∀ (r : renaming) (n : name) (ty : flambda_type),
    Typing_env_extension.check_equation (rename_name r n) (rename_type r ty)
    = Typing_env_extension.check_equation n ty)
  ∧ (∀ (r : renaming) (t : Typing_env_extension.t),
      Typing_env_extension.invariant (Typing_env_extension.apply_renaming t r)
      = Typing_env_extension.invariant t).

(* Extensions with extra variables: the renamings act on them as they do
   on types. *)

Definition with_extra_variables_rename_identity_statement : Prop :=
  ∀ t : Typing_env_extension.With_extra_variables.t,
    Typing_env_extension.With_extra_variables.apply_renaming t
      identity_renaming
    = t.

Definition with_extra_variables_rename_compose_statement : Prop :=
  ∀ (second first : renaming)
    (t : Typing_env_extension.With_extra_variables.t),
    Typing_env_extension.With_extra_variables.apply_renaming t
      (compose_renaming second first)
    = Typing_env_extension.With_extra_variables.apply_renaming
        (Typing_env_extension.With_extra_variables.apply_renaming t first)
        second.

(* The free names of the renamed extension are the renamed free names. *)
Definition with_extra_variables_free_names_renamed_statement : Prop :=
  ∀ (r : renaming) (t : Typing_env_extension.With_extra_variables.t),
    Typing_env_extension.With_extra_variables.free_names
      (Typing_env_extension.With_extra_variables.apply_renaming t r)
    = map (rename_name r)
        (Typing_env_extension.With_extra_variables.free_names t).

(* A renaming that fixes the variables among the free names of an
   extension leaves it as it is. *)
Definition with_extra_variables_rename_fixes_statement : Prop :=
  ∀ (r : renaming) (t : Typing_env_extension.With_extra_variables.t),
    (∀ x : variable,
      Name_var x ∈ Typing_env_extension.With_extra_variables.free_names t ->
      rename_variable r x = x) ->
    Typing_env_extension.With_extra_variables.apply_renaming t r = t.

(* The functions on them commute with renaming. *)
Definition with_extra_variables_rename_add_definition_statement : Prop :=
  ∀ (r : renaming) (t : Typing_env_extension.With_extra_variables.t)
    (var : variable) (kind : flambda_kind),
    Typing_env_extension.With_extra_variables.apply_renaming
      (Typing_env_extension.With_extra_variables.add_definition t var kind) r
    = Typing_env_extension.With_extra_variables.add_definition
        (Typing_env_extension.With_extra_variables.apply_renaming t r)
        (rename_variable r var) kind.

Definition with_extra_variables_rename_add_equation_statement : Prop :=
  ∀ (r : renaming) (t : Typing_env_extension.With_extra_variables.t)
    (n : name) (ty : flambda_type),
    Typing_env_extension.With_extra_variables.apply_renaming
      (Typing_env_extension.With_extra_variables.add_or_replace_equation t n
         ty)
      r
    = Typing_env_extension.With_extra_variables.add_or_replace_equation
        (Typing_env_extension.With_extra_variables.apply_renaming t r)
        (rename_name r n) (rename_type r ty).

(* [map_types] with a function that commutes with the renaming. *)
Definition with_extra_variables_rename_map_types_statement : Prop :=
  ∀ (r : renaming) (f f' : flambda_type -> flambda_type)
    (t : Typing_env_extension.With_extra_variables.t),
    (∀ ty : flambda_type, f' (rename_type r ty) = rename_type r (f ty)) ->
    Typing_env_extension.With_extra_variables.apply_renaming
      (Typing_env_extension.With_extra_variables.map_types t f) r
    = Typing_env_extension.With_extra_variables.map_types
        (Typing_env_extension.With_extra_variables.apply_renaming t r) f'.

(* [fold], as for a plain extension: the variables in the order of
   their list, then the equations. *)
Definition with_extra_variables_rename_fold_statement : Prop :=
  ∀ (B B' : Type) (r : renaming) (state : B -> B')
    (variable : FlambdaModel.variable -> B -> B)
    (variable' : FlambdaModel.variable -> B' -> B')
    (equation : name -> flambda_type -> B -> B)
    (equation' : name -> flambda_type -> B' -> B')
    (t : Typing_env_extension.With_extra_variables.t) (acc : B),
    (∀ (x : FlambdaModel.variable) (acc : B),
      variable' (rename_variable r x) (state acc) = state (variable x acc)) ->
    (∀ (n : name) (ty : flambda_type) (acc : B),
      equation' (rename_name r n) (rename_type r ty) (state acc)
      = state (equation n ty acc)) ->
    Typing_env_extension.With_extra_variables.fold variable' equation'
      (Typing_env_extension.With_extra_variables.apply_renaming t r)
      (state acc)
    = state
        (Typing_env_extension.With_extra_variables.fold variable equation t
           acc).

(* The renamed environment satisfies the renamed extension exactly when
   the environment satisfies the extension. *)
Definition with_extra_variables_satisfied_rename_statement : Prop :=
  ∀ (r : renaming) (statics : static_env) (vars : env)
    (t : Typing_env_extension.With_extra_variables.t),
    with_extra_variables_satisfied statics (rename_env r vars)
      (Typing_env_extension.With_extra_variables.apply_renaming t r)
    <-> with_extra_variables_satisfied statics vars t.

(* The extra variables are bound by the meaning: what satisfies an
   extension does not depend on the values that it gives them. *)
Definition with_extra_variables_bound_statement : Prop :=
  ∀ (statics : static_env) (vars vars' : env)
    (t : Typing_env_extension.With_extra_variables.t),
    (∀ x : variable,
      ~ x ∈ Typing_env_extension.With_extra_variables.existential_vars t ->
      lookup x vars' = lookup x vars) ->
    with_extra_variables_satisfied statics vars t ->
    with_extra_variables_satisfied statics vars' t.
