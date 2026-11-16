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

(* The extensions of a typing environment: equations on names, to be
   added to an environment.

   This file is the model's copy of
   middle_end/flambda2/types/env/typing_env_extension.ml: the functions
   on extensions and on extensions with extra variables, written as the
   OCaml is and under its names, in a module [Typing_env_extension]. It
   extracts. The type of an extension is that of the grammar of types
   ([env_extension], FlambdaTypeGrammar.v), since types hold
   extensions. What the functions answer is
   FlambdaTypingEnvExtensionSpec.v, whose header lists the differences
   from the OCaml; in short:
   - the equations are an association list, and the functions that add
     an equation take the earlier equation of the name out, so that a
     name has one equation, as in the OCaml's map;
   - the OCaml goes through the equations in the order of names; the
     model in the order of the list ([bindings]);
   - a fatal error of the OCaml that no flag guards is [None]; a check
     made only under a flag is a function of its own that answers
     whether the check passes ([check_equation], [invariant]);
   - a variable of the model has no kind, so [add_definition] cannot
     check the kind it is given;
   - a renaming renames variables only, and the free names are the
     variables and the symbols, as for types
     (FlambdaTypeRenameSpec.v). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaNominalSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeRenameSpec.
Import ListNotations.

Module Typing_env_extension.

(* [Typing_env_extension.t], which is [Type_grammar.Env_extension.t]. *)
Definition t : Type := env_extension.

Definition to_map (t : t) : Keyed.finite_map name flambda_type :=
  match t with
  | Env_extension equations => equations
  end.

Definition from_map (equations : Keyed.finite_map name flambda_type) : t :=
  Env_extension equations.

(* The equations that count, one for each name that has one, in the
   order of the list. The OCaml's map has them in the order of names. *)
Definition bindings (t : t) : list (name * flambda_type) :=
  Keyed.compact name_eqb (to_map t).

(* [fold], which goes through [bindings]. *)
Definition fold {B : Type} (equation : name -> flambda_type -> B -> B)
    (t : t) (acc : B) : B :=
  fold_left (fun acc binding => equation (fst binding) (snd binding) acc)
    (bindings t) acc.

(* [More_type_creators.is_alias_of_name]. *)
Definition is_alias_of_name (ty : flambda_type) (name : name) : bool :=
  match get_alias_opt ty with
  | None => false
  | Some (Var var) => name_eqb name (Name_var var)
  | Some (Symbol symbol) => name_eqb name (Name_symbol symbol)
  | Some (Const _) => false
  end.

(* What [More_type_creators.check_equation] checks, when the flag for
   invariants is set: the equation does not make the name an alias of
   itself. The answer is [true] where the OCaml goes on. *)
Definition check_equation (name : name) (ty : flambda_type) : bool :=
  negb (is_alias_of_name ty name).

(* What [invariant] checks, when the flag is set. *)
Definition invariant (t : t) : bool :=
  forallb (fun binding => check_equation (fst binding) (snd binding))
    (bindings t).

Definition empty : t := empty_env_extension.

Definition is_empty (t : t) : bool := Keyed.is_empty (to_map t).

Definition has_equation (name : name) (t : t) : bool :=
  Keyed.is_bound name_eqb name (to_map t).

Definition one_equation (name : name) (ty : flambda_type) : t :=
  Env_extension [(name, ty)].

(* The equation of the name, if it had one, is replaced. *)
Definition add_or_replace_equation (t : t) (name : name)
    (ty : flambda_type) : Typing_env_extension.t :=
  Env_extension ((name, ty) :: Keyed.unbind name_eqb name (to_map t)).

Definition replace_equation (t : t) (name : name) (ty : flambda_type)
    : Typing_env_extension.t :=
  Env_extension ((name, ty) :: Keyed.unbind name_eqb name (to_map t)).

(* [None] is the fatal error of [Name.Map.disjoint_union]: a name has an
   equation in both. *)
Definition disjoint_union (t1 t2 : t) : option t :=
  if Keyed.is_empty
       (Keyed.inter name_eqb
          (Keyed.domain name_eqb (to_map t1))
          (Keyed.domain name_eqb (to_map t2)))
  then Some (Env_extension (to_map t1 ++ to_map t2))
  else None.

(* [free_names] and [apply_renaming], which are those of
   [Type_grammar.Env_extension]: the names that have an equation are
   free names, and are renamed. *)
Definition free_names (t : t) : list name := free_names_of_env_extension t.

Definition apply_renaming (t : t) (renaming : renaming)
    : Typing_env_extension.t :=
  rename_env_extension renaming t.

(* ======================================================================== *)
(* Extensions with extra variables                                          *)
(* ======================================================================== *)

Module With_extra_variables.

  (* [Typing_env_extension.With_extra_variables.t], with [create]:
     equations, and variables that the equations may mention and that
     the environment the extension is for does not have. *)
  Record t : Type := create {
    existential_vars : list variable;
    equations : Keyed.finite_map name flambda_type
  }.

  (* The variables, then the equations that count in the order of the
     list. *)
  Definition fold {B : Type} (variable : variable -> B -> B)
      (equation : name -> flambda_type -> B -> B) (t : t) (acc : B) : B :=
    let acc :=
      fold_left (fun acc var => variable var acc) (existential_vars t) acc
    in
    fold_left (fun acc binding => equation (fst binding) (snd binding) acc)
      (Keyed.compact name_eqb (equations t))
      acc.

  Definition empty : t := create [] [].

  (* The OCaml checks, with light invariant checks, that the kind given
     is that of the variable. A variable of the model has no kind. *)
  Definition add_definition (t : t) (var : variable) (kind : flambda_kind)
      : With_extra_variables.t :=
    create (var :: existential_vars t) (equations t).

  Definition add_or_replace_equation (t : t) (name : name)
      (ty : flambda_type) : With_extra_variables.t :=
    create (existential_vars t)
      ((name, ty) :: Keyed.unbind name_eqb name (equations t)).

  Definition map_types (t : t) (f : flambda_type -> flambda_type)
      : With_extra_variables.t :=
    create (existential_vars t) (Keyed.map_values f (equations t)).

  (* The extra variables are free names, and are renamed: "At present
     [existential_vars] is not treated as a binding site"
     (typing_env_extension.mli). *)
  Definition free_names (t : t) : list name :=
    map Name_var (existential_vars t)
    ++ free_names_of_env_extension (Env_extension (equations t)).

  Definition apply_renaming (t : t) (renaming : renaming)
      : With_extra_variables.t :=
    create (map (rename_variable renaming) (existential_vars t))
      (Keyed.map_bindings (rename_name renaming) (rename_type renaming)
         (equations t)).

End With_extra_variables.

End Typing_env_extension.
