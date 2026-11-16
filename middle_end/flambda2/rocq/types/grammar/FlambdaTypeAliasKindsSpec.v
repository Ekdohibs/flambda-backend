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


(* The aliases of a type with their kinds: what the creation functions
   give.

   This file is to be reviewed: the statements, which say what
   [aliases_with_kinds] (FlambdaTypeGrammar.v) is of the types that the
   creation functions make. They are for whoever asks that the aliases
   of a type stand at the kinds of their targets, which is a premise of
   what is stated of meet and of join and no part of a type being well
   formed: a type made by a creation function has the aliases of the
   types it is made of, and no other but its own.

   The last section says the same of [equation_names_with_kinds], the
   names that have an equation in an extension inside a type, with the
   kinds of their equations.

   Each statement holds by computation. The proofs are in
   FlambdaTypeAliasKindsProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar.
Import ListNotations.

(* ======================================================================== *)
(* Unknown, bottom, aliases and constants                                   *)
(* ======================================================================== *)

Definition aliases_with_kinds_unknown_statement : Prop :=
  ∀ k : flambda_kind, aliases_with_kinds (unknown k) = [].

Definition aliases_with_kinds_bottom_statement : Prop :=
  ∀ k : flambda_kind, aliases_with_kinds (bottom k) = [].

(* An alias type has one alias, at the kind it is made at, which is its
   kind. *)
Definition aliases_with_kinds_alias_statement : Prop :=
  ∀ (k : flambda_kind) (s : simple),
    aliases_with_kinds (alias_type_of k s) = [(s, k)].

Definition aliases_with_kinds_get_alias_statement : Prop :=
  ∀ (t : flambda_type) (s : simple),
    get_alias_opt t = Some s -> aliases_with_kinds t = [(s, kind t)].

(* The type of a constant has no alias but the constant, at its kind. *)
Definition aliases_with_kinds_type_for_const_statement : Prop :=
  ∀ (c : const) (s : simple) (k : flambda_kind),
    (s, k) ∈ aliases_with_kinds (type_for_const c) ->
    s = Const c ∧ k = kind_of_const c.

Definition aliases_with_kinds_these_naked_immediates_statement : Prop :=
  ∀ (imms : list Z) (s : simple) (k : flambda_kind),
    (s, k) ∈ aliases_with_kinds (these_naked_immediates imms) ->
    ∃ i : Z,
      imms = [i]
      ∧ s = Const (Naked_immediate i)
      ∧ k = Naked_number Kind_naked_immediate.

Definition aliases_with_kinds_these_tagged_immediates_statement : Prop :=
  ∀ (imms : list Z) (s : simple) (k : flambda_kind),
    (s, k) ∈ aliases_with_kinds (these_tagged_immediates imms) ->
    ∃ i : Z, imms = [i] ∧ s = Const (Tagged_immediate i) ∧ k = Value.

(* ======================================================================== *)
(* The types that have no alias                                             *)
(* ======================================================================== *)

(* The types of the results of the relations have none: the name of an
   inverse relation is not listed. *)
Definition aliases_with_kinds_none_statement : Prop :=
  aliases_with_kinds null = []
  ∧ aliases_with_kinds any_non_null_value = []
  ∧ aliases_with_kinds any_tagged_immediate = []
  ∧ aliases_with_kinds any_block = []
  ∧ (∀ alloc_mode : Alloc_mode.t,
      aliases_with_kinds (mutable_block alloc_mode) = [])
  ∧ (∀ bytes : list Z, aliases_with_kinds (this_immutable_string bytes) = [])
  ∧ (∀ (tags : list Z) (alloc_mode : Alloc_mode.t),
      aliases_with_kinds (blocks_with_these_tags tags alloc_mode) = [])
  ∧ (∀ n : name,
      aliases_with_kinds (is_int_for_scrutinee n) = []
      ∧ aliases_with_kinds (get_tag_for_block n) = []
      ∧ aliases_with_kinds (is_null_for_scrutinee n) = []).

(* ======================================================================== *)
(* The types made of types                                                  *)
(* ======================================================================== *)

Definition aliases_with_kinds_create_variant_statement : Prop :=
  ∀ (is_unique : bool) (immediates : Or_unknown.t flambda_type)
    (blocks : Or_unknown.t row_like_for_blocks)
    (extensions : variant_extensions),
    aliases_with_kinds (create_variant is_unique immediates blocks extensions)
    = match immediates with
      | Or_unknown.Known ty => aliases_with_kinds ty
      | Or_unknown.Unknown => []
      end
      ++ match blocks with
         | Or_unknown.Known blocks => aliases_of_row_like_for_blocks blocks
         | Or_unknown.Unknown => []
         end
      ++ match extensions with
         | No_extensions => []
         | Ext when_immediate when_block =>
           aliases_of_env_extension when_immediate
           ++ aliases_of_env_extension when_block
         end.

Definition aliases_with_kinds_tag_immediate_statement : Prop :=
  ∀ t : flambda_type,
    aliases_with_kinds (tag_immediate t) = aliases_with_kinds t.

Definition aliases_with_kinds_box_number_statement : Prop :=
  ∀ (k : boxable_number) (contents : flambda_type)
    (alloc_mode : Alloc_mode.t),
    aliases_with_kinds (box_number k contents alloc_mode)
    = aliases_with_kinds contents.

Definition aliases_with_kinds_immutable_block_statement : Prop :=
  ∀ (is_unique : bool) (tag : Z) (shape : block_shape)
    (alloc_mode : Alloc_mode.t) (fields : list flambda_type),
    aliases_with_kinds (immutable_block is_unique tag shape alloc_mode fields)
    = flat_map aliases_with_kinds fields.

Definition aliases_with_kinds_create_closures_statement : Prop :=
  ∀ (alloc_mode : Alloc_mode.t) (by_function_slot : row_like_for_closures),
    aliases_with_kinds (create_closures alloc_mode by_function_slot)
    = aliases_of_row_like_for_closures by_function_slot.

(* The closure types and the value slot types; the function types have
   no alias. *)
Definition aliases_with_kinds_this_closure_statement : Prop :=
  ∀ (slot : function_slot)
    (function_types : finite_map (Or_unknown.t function_type))
    (closure_types value_slot_types : finite_map flambda_type)
    (alloc_mode : Alloc_mode.t),
    aliases_with_kinds
      (exactly_this_closure slot function_types closure_types
         value_slot_types alloc_mode)
    = aliases_of_closures_entry
        (Closures_entry function_types closure_types value_slot_types)
    ∧ aliases_with_kinds
        (at_least_this_closure slot function_types closure_types
           value_slot_types alloc_mode)
      = aliases_of_closures_entry
          (Closures_entry function_types closure_types value_slot_types).

(* ======================================================================== *)
(* The names of equations, with their kinds                                 *)
(* ======================================================================== *)

(* The types that hold no extension have none. *)
Definition equation_names_none_statement : Prop :=
  (∀ k : flambda_kind, equation_names_with_kinds (unknown k) = [])
  ∧ (∀ k : flambda_kind, equation_names_with_kinds (bottom k) = [])
  ∧ (∀ (k : flambda_kind) (s : simple),
      equation_names_with_kinds (alias_type_of k s) = [])
  ∧ (∀ c : const, equation_names_with_kinds (type_for_const c) = []).

(* An extension has the name of each of its equations, at the kind of
   the type of the equation, and the names of the equations inside
   that type. *)
Definition equation_names_of_equations_statement : Prop :=
  ∀ (equations : Keyed.finite_map name flambda_type) (n : name)
    (ty : flambda_type),
    (n, ty) ∈ equations ->
    (n, kind ty) ∈ equation_names_of_env_extension (Env_extension equations)
    ∧ ∀ equation : name * flambda_kind,
        equation ∈ equation_names_with_kinds ty ->
        equation ∈ equation_names_of_env_extension (Env_extension equations).

(* A variant has those of its immediates, of its blocks and of its two
   extensions. *)
Definition equation_names_create_variant_statement : Prop :=
  ∀ (is_unique : bool) (immediates : Or_unknown.t flambda_type)
    (blocks : Or_unknown.t row_like_for_blocks)
    (extensions : variant_extensions),
    equation_names_with_kinds
      (create_variant is_unique immediates blocks extensions)
    = match immediates with
      | Or_unknown.Known ty => equation_names_with_kinds ty
      | Or_unknown.Unknown => []
      end
      ++ match blocks with
         | Or_unknown.Known blocks =>
           equation_names_of_row_like_for_blocks blocks
         | Or_unknown.Unknown => []
         end
      ++ match extensions with
         | No_extensions => []
         | Ext when_immediate when_block =>
           equation_names_of_env_extension when_immediate
           ++ equation_names_of_env_extension when_block
         end.

(* ======================================================================== *)
(* The variables of the aliases and of the names of equations are free      *)
(* ======================================================================== *)

(* A variable among the simples that [aliases_with_kinds] lists is a
   free variable of the type: whoever brings the free variables of a
   type into an environment has brought its aliases. *)
Definition aliases_free_variables_statement : Prop :=
  (∀ (t : flambda_type) (s : simple) (k : flambda_kind) (x : variable),
    (s, k) ∈ aliases_with_kinds t ->
    x ∈ free_variables_of_simple s -> x ∈ free_variables t)
  ∧ (∀ (extension : env_extension) (s : simple) (k : flambda_kind)
       (x : variable),
      (s, k) ∈ aliases_of_env_extension extension ->
      x ∈ free_variables_of_simple s ->
      x ∈ free_variables_of_env_extension extension).

(* The same of the names of equations. *)
Definition equation_names_free_variables_statement : Prop :=
  (∀ (t : flambda_type) (n : name) (k : flambda_kind) (x : variable),
    (n, k) ∈ equation_names_with_kinds t ->
    x ∈ free_variables_of_name n -> x ∈ free_variables t)
  ∧ (∀ (extension : env_extension) (n : name) (k : flambda_kind)
       (x : variable),
      (n, k) ∈ equation_names_of_env_extension extension ->
      x ∈ free_variables_of_name n ->
      x ∈ free_variables_of_env_extension extension).
