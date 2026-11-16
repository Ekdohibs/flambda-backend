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


(* Renaming the variables of a type, and the names that a type mentions.

   This file is to be reviewed: [rename_type] and the functions defined
   with it, against [Type_grammar.apply_renaming]; [free_names], against
   [Type_grammar.free_names]; and the statements.

   A renaming is a permutation of the variables (FlambdaNominalSpec.v;
   its permutation of continuations has nothing to act on in a type).
   [rename_type r t] renames every variable that [t] mentions: the
   simple of an alias, the variables that hold the results of [Is_int],
   [Get_tag] and [Is_null], the names of which a naked immediate is the
   result of a relation, and, in an environment extension, the names
   that have an equation as well as the types of the equations. It is
   the action that an interface for things that contain names asks
   for, as the compiler's [Contains_names] does: the statements are
   the laws of such an interface for variables, with the equality of
   Rocq, and what the functions on types owe it. The free names of
   the interface are [free_variables] (FlambdaTypeGrammar.v).

   What the compiler's function does that this one does not, the
   renamings of the model having less: a [Renaming.t] also renames
   symbols and code ids, when a unit is imported, so the compiler
   renames the symbols of a type and the code ids of its function
   types; and it drops, from the index of a case of closures and from
   the value slot types of its entry, the value slots that the
   renaming says are not used
   ([Set_of_closures_contents.apply_renaming],
   [apply_renaming_value_slot_indexed_product]). Here a symbol and a
   code id are fixed and nothing is dropped. The compiler also renames
   the variables of a [Rec_info_expr.t], in the head of a type of kind
   [Rec_info] and in the [rec_info] of a function type. The grammar
   has neither (FlambdaTypeGrammar.v), so there is nothing of them to
   rename here, nor to list in [free_names]. The compiler keeps the
   parts that do not change as they are in memory, which the model
   does not see.

   [free_names] is the names that a type mentions, the symbols with
   the variables; [free_variables] is its variables
   ([free_names_variables_statement]). The compiler's function also
   counts the code ids, the function slots and the value slots of a
   type, and records at which name mode each name occurs (in types);
   the model has neither.

   The statements, in order: the renamings act as a group on types
   (identity, composition, inverse), and two renamings that agree on
   the free variables of a type rename it alike, so that one that
   fixes them leaves it as it is; the free variables, the free names
   and the aliases of a renamed type are those of the type, renamed;
   its kind is the same, and so is the constant that
   [must_be_singleton] reads of it; a value has the renamed type under
   the renamed environment exactly when it has the type under the
   environment; the tests of types do not see a renaming; the names
   of the equations of a renamed type are those of the type, renamed;
   and a creation function given renamed parts makes the renamed
   type.

   The proofs are in FlambdaTypeRenameProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaNominalSpec FlambdaAlphaSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypeWellFormedSpec FlambdaTypeEqualSpec.
Import ListNotations.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

(* [Renaming.apply_name]: a variable is renamed, a symbol is not. *)
Definition rename_name (r : renaming) (n : name) : name :=
  match n with
  | Name_var x => Name_var (rename_variable r x)
  | Name_symbol s => Name_symbol s
  end.

(* [Or_unknown.map], [Or_bottom.map] and [Or_unknown_or_bottom.map]. *)
Definition or_unknown_map {A B : Type} (f : A -> B) (a : Or_unknown.t A)
    : Or_unknown.t B :=
  match a with
  | Or_unknown.Known a => Or_unknown.Known (f a)
  | Or_unknown.Unknown => Or_unknown.Unknown
  end.

Definition or_bottom_map {A B : Type} (f : A -> B) (a : Or_bottom.t A)
    : Or_bottom.t B :=
  match a with
  | Or_bottom.Ok a => Or_bottom.Ok (f a)
  | Or_bottom.Bottom => Or_bottom.Bottom
  end.

Definition or_unknown_or_bottom_map {A B : Type} (f : A -> B)
    (a : or_unknown_or_bottom A) : or_unknown_or_bottom B :=
  match a with
  | Unknown => Unknown
  | Ok a => Ok (f a)
  | Bottom => Bottom
  end.

(* A type that is unknown, bottom, an alias or given by a head. *)
Definition rename_descr {head : Type} (r : renaming)
    (rename_head : head -> head) (ty : type_descr head) : type_descr head :=
  or_unknown_or_bottom_map
    (fun descr =>
       match descr with
       | No_alias head => No_alias (rename_head head)
       | Equals s => Equals (rename_simple r s)
       end)
    ty.

Definition rename_is_null (r : renaming) (null : is_null) : is_null :=
  match null with
  | Not_null => Not_null
  | Maybe_null x => Maybe_null (option_map (rename_variable r) x)
  end.

Definition rename_head_of_kind_naked_immediate (r : renaming)
    (head : head_of_kind_naked_immediate) : head_of_kind_naked_immediate :=
  Head_of_kind_naked_immediate (naked_immediates head)
    (map (fun relation => (fst relation, rename_name r (snd relation)))
       (inverse_relations head)).

(* A case of blocks. The two arguments stand for the functions being
   defined below. *)
Definition rename_block_case (rename_type : flambda_type -> flambda_type)
    (rename_env_extension : env_extension -> env_extension)
    (case : row_like_block_case) : row_like_block_case :=
  Row_like_case (map rename_type (case_maps_to case)) (case_index case)
    (rename_env_extension (case_env_extension case)).

(* A case of closures: the index has slots only. *)
Definition rename_closures_case
    (rename_closures_entry : closures_entry -> closures_entry)
    (rename_env_extension : env_extension -> env_extension)
    (case : row_like_closures_case) : row_like_closures_case :=
  Row_like_case (rename_closures_entry (case_maps_to case)) (case_index case)
    (rename_env_extension (case_env_extension case)).

(* [Type_grammar.apply_renaming], and one function for each structure
   of the grammar. A map keyed by tags or by slots keeps its keys
   ([map_values]); the equations of an environment extension are
   renamed on both sides. *)
Fixpoint rename_type (r : renaming) (t : flambda_type) {struct t}
    : flambda_type :=
  match t with
  | Type_value ty =>
    Type_value (rename_descr r (rename_head_of_kind_value r) ty)
  | Type_naked_immediate ty =>
    Type_naked_immediate
      (rename_descr r (rename_head_of_kind_naked_immediate r) ty)
  | Type_naked_number k ty =>
    Type_naked_number k (rename_descr r (fun numbers => numbers) ty)
  | Type_rec_info ty => Type_rec_info (rename_descr r (fun head => head) ty)
  | Type_region ty => Type_region (rename_descr r (fun head => head) ty)
  end

with rename_head_of_kind_value (r : renaming) (head : head_of_kind_value)
    {struct head} : head_of_kind_value :=
  match head with
  | Head_of_kind_value non_null is_null =>
    Head_of_kind_value
      (or_unknown_or_bottom_map (rename_head_of_kind_value_non_null r)
         non_null)
      (rename_is_null r is_null)
  end

with rename_head_of_kind_value_non_null (r : renaming)
    (head : head_of_kind_value_non_null) {struct head}
    : head_of_kind_value_non_null :=
  match head with
  | Variant is_int immediates get_tag blocks extensions is_unique =>
    Variant (option_map (rename_variable r) is_int)
      (or_unknown_map (rename_type r) immediates)
      (option_map (rename_variable r) get_tag)
      (or_unknown_map (rename_row_like_for_blocks r) blocks)
      match extensions with
      | No_extensions => No_extensions
      | Ext when_immediate when_block =>
        Ext (rename_env_extension r when_immediate)
          (rename_env_extension r when_block)
      end
      is_unique
  | Mutable_block alloc_mode => Mutable_block alloc_mode
  | Boxed_number k contents alloc_mode =>
    Boxed_number k (rename_type r contents) alloc_mode
  | Closures by_function_slot alloc_mode =>
    Closures (rename_row_like_for_closures r by_function_slot) alloc_mode
  | String strs => String strs
  end

with rename_row_like_for_blocks (r : renaming) (blocks : row_like_for_blocks)
    {struct blocks} : row_like_for_blocks :=
  match blocks with
  | Row_like_for_blocks known_tags other_tags alloc_mode =>
    Row_like_for_blocks
      (Keyed.map_values
         (or_unknown_map
            (rename_block_case (rename_type r) (rename_env_extension r)))
         known_tags)
      (or_bottom_map
         (rename_block_case (rename_type r) (rename_env_extension r))
         other_tags)
      alloc_mode
  end

with rename_row_like_for_closures (r : renaming)
    (closures : row_like_for_closures) {struct closures}
    : row_like_for_closures :=
  match closures with
  | Row_like_for_closures known_closures =>
    Row_like_for_closures
      (map_values
         (rename_closures_case (rename_closures_entry r)
            (rename_env_extension r))
         known_closures)
  end

with rename_closures_entry (r : renaming) (entry : closures_entry)
    {struct entry} : closures_entry :=
  match entry with
  | Closures_entry function_types closure_types value_slot_types =>
    Closures_entry function_types
      (map_values (rename_type r) closure_types)
      (map_values (rename_type r) value_slot_types)
  end

with rename_env_extension (r : renaming) (extension : env_extension)
    {struct extension} : env_extension :=
  match extension with
  | Env_extension equations =>
    Env_extension
      (map
         (fun equation =>
            (rename_name r (fst equation), rename_type r (snd equation)))
         equations)
  end.

(* ======================================================================== *)
(* Free names                                                               *)
(* ======================================================================== *)

(* [Type_grammar.free_names], of which the variables and the symbols
   are kept. A name may be listed more than once, and the names of a
   binding of a map that is shadowed are listed too. *)

Definition free_names_of_simple (s : simple) : list name :=
  match s with
  | Var x => [Name_var x]
  | Symbol sym => [Name_symbol sym]
  | Const _ => []
  end.

Definition free_names_of_option (x : option variable) : list name :=
  match x with
  | Some x => [Name_var x]
  | None => []
  end.

Definition free_names_of_descr {head : Type}
    (free_names_of_head : head -> list name) (ty : type_descr head)
    : list name :=
  match ty with
  | Ok (No_alias head) => free_names_of_head head
  | Ok (Equals s) => free_names_of_simple s
  | Unknown | Bottom => []
  end.

Definition free_names_of_naked_immediate
    (head : head_of_kind_naked_immediate) : list name :=
  map snd (inverse_relations head).

(* The two arguments stand for the functions being defined below. *)
Definition free_names_of_block_case
    (free_names : flambda_type -> list name)
    (free_names_of_env_extension : env_extension -> list name)
    (case : row_like_block_case) : list name :=
  flat_map free_names (case_maps_to case)
  ++ free_names_of_env_extension (case_env_extension case).

Fixpoint free_names (t : flambda_type) : list name :=
  match t with
  | Type_value ty => free_names_of_descr free_names_of_head_of_kind_value ty
  | Type_naked_immediate ty =>
    free_names_of_descr free_names_of_naked_immediate ty
  | Type_naked_number _ ty => free_names_of_descr (fun _ => []) ty
  | Type_rec_info ty => free_names_of_descr (fun _ => []) ty
  | Type_region ty => free_names_of_descr (fun _ => []) ty
  end

with free_names_of_head_of_kind_value (head : head_of_kind_value)
    : list name :=
  match head with
  | Head_of_kind_value non_null is_null =>
    match non_null with
    | Ok head => free_names_of_head_of_kind_value_non_null head
    | Unknown | Bottom => []
    end
    ++ match is_null with
       | Not_null => []
       | Maybe_null is_null => free_names_of_option is_null
       end
  end

with free_names_of_head_of_kind_value_non_null
    (head : head_of_kind_value_non_null) : list name :=
  match head with
  | Variant is_int immediates get_tag blocks extensions _ =>
    free_names_of_option is_int
    ++ match immediates with
       | Or_unknown.Known ty => free_names ty
       | Or_unknown.Unknown => []
       end
    ++ free_names_of_option get_tag
    ++ match blocks with
       | Or_unknown.Known blocks => free_names_of_row_like_for_blocks blocks
       | Or_unknown.Unknown => []
       end
    ++ match extensions with
       | No_extensions => []
       | Ext when_immediate when_block =>
         free_names_of_env_extension when_immediate
         ++ free_names_of_env_extension when_block
       end
  | Mutable_block _ | String _ => []
  | Boxed_number _ contents _ => free_names contents
  | Closures by_function_slot _ =>
    free_names_of_row_like_for_closures by_function_slot
  end

with free_names_of_row_like_for_blocks (blocks : row_like_for_blocks)
    : list name :=
  match blocks with
  | Row_like_for_blocks known_tags other_tags _ =>
    flat_map
      (fun case =>
         match snd case with
         | Or_unknown.Known case =>
           free_names_of_block_case free_names free_names_of_env_extension
             case
         | Or_unknown.Unknown => []
         end)
      known_tags
    ++ match other_tags with
       | Or_bottom.Ok case =>
         free_names_of_block_case free_names free_names_of_env_extension
           case
       | Or_bottom.Bottom => []
       end
  end

with free_names_of_row_like_for_closures (closures : row_like_for_closures)
    : list name :=
  match closures with
  | Row_like_for_closures known_closures =>
    flat_map
      (fun case =>
         free_names_of_closures_entry (case_maps_to (snd case))
         ++ free_names_of_env_extension (case_env_extension (snd case)))
      known_closures
  end

with free_names_of_closures_entry (entry : closures_entry) : list name :=
  match entry with
  | Closures_entry _ closure_types value_slot_types =>
    flat_map (fun binding => free_names (snd binding)) closure_types
    ++ flat_map (fun binding => free_names (snd binding)) value_slot_types
  end

with free_names_of_env_extension (extension : env_extension) : list name :=
  match extension with
  | Env_extension equations =>
    flat_map (fun binding => fst binding :: free_names (snd binding))
      equations
  end.

(* The variables among the names that a type mentions are its free
   variables, in the same order. *)
Definition free_names_variables_statement : Prop :=
  (∀ t : flambda_type,
    free_variables t = flat_map free_variables_of_name (free_names t))
  ∧ (∀ extension : env_extension,
      free_variables_of_env_extension extension
      = flat_map free_variables_of_name
          (free_names_of_env_extension extension)).

(* ======================================================================== *)
(* The renamings act on types                                               *)
(* ======================================================================== *)

Definition rename_type_identity_statement : Prop :=
  (∀ t : flambda_type, rename_type identity_renaming t = t)
  ∧ (∀ extension : env_extension,
      rename_env_extension identity_renaming extension = extension).

Definition rename_type_compose_statement : Prop :=
  (∀ (second first : renaming) (t : flambda_type),
    rename_type (compose_renaming second first) t
    = rename_type second (rename_type first t))
  ∧ (∀ (second first : renaming) (extension : env_extension),
      rename_env_extension (compose_renaming second first) extension
      = rename_env_extension second (rename_env_extension first extension)).

(* Two renamings that rename the free variables of a type alike rename
   the type alike. *)
Definition rename_type_extensional_statement : Prop :=
  (∀ (r r' : renaming) (t : flambda_type),
    (∀ x : variable,
      x ∈ free_variables t -> rename_variable r x = rename_variable r' x) ->
    rename_type r t = rename_type r' t)
  ∧ (∀ (r r' : renaming) (extension : env_extension),
      (∀ x : variable,
        x ∈ free_variables_of_env_extension extension ->
        rename_variable r x = rename_variable r' x) ->
      rename_env_extension r extension = rename_env_extension r' extension).

(* So a renaming that fixes the free variables of a type leaves the
   type as it is: the free variables are a support. *)
Definition rename_type_fixes_statement : Prop :=
  (∀ (r : renaming) (t : flambda_type),
    (∀ x : variable, x ∈ free_variables t -> rename_variable r x = x) ->
    rename_type r t = t)
  ∧ (∀ (r : renaming) (extension : env_extension),
      (∀ x : variable,
        x ∈ free_variables_of_env_extension extension ->
        rename_variable r x = x) ->
      rename_env_extension r extension = extension).

Definition rename_type_inverse_statement : Prop :=
  (∀ (r : renaming) (t : flambda_type),
    rename_type (inverse_renaming r) (rename_type r t) = t)
  ∧ (∀ (r : renaming) (extension : env_extension),
      rename_env_extension (inverse_renaming r)
        (rename_env_extension r extension)
      = extension).

(* ======================================================================== *)
(* What a renamed type has                                                  *)
(* ======================================================================== *)

(* Its free variables, its free names and its aliases are those of the
   type, renamed. *)
Definition free_variables_rename_type_statement : Prop :=
  (∀ (r : renaming) (t : flambda_type),
    free_variables (rename_type r t)
    = map (rename_variable r) (free_variables t))
  ∧ (∀ (r : renaming) (extension : env_extension),
      free_variables_of_env_extension (rename_env_extension r extension)
      = map (rename_variable r) (free_variables_of_env_extension extension)).

Definition free_names_rename_type_statement : Prop :=
  (∀ (r : renaming) (t : flambda_type),
    free_names (rename_type r t) = map (rename_name r) (free_names t))
  ∧ (∀ (r : renaming) (extension : env_extension),
      free_names_of_env_extension (rename_env_extension r extension)
      = map (rename_name r) (free_names_of_env_extension extension)).

Definition aliases_with_kinds_rename_type_statement : Prop :=
  (∀ (r : renaming) (t : flambda_type),
    aliases_with_kinds (rename_type r t)
    = map (fun alias => (rename_simple r (fst alias), snd alias))
        (aliases_with_kinds t))
  ∧ (∀ (r : renaming) (extension : env_extension),
      aliases_of_env_extension (rename_env_extension r extension)
      = map (fun alias => (rename_simple r (fst alias), snd alias))
          (aliases_of_env_extension extension)).

(* Its kind is that of the type. *)
Definition kind_rename_type_statement : Prop :=
  ∀ (r : renaming) (t : flambda_type), kind (rename_type r t) = kind t.

(* The constant that it shows for its only value, if it shows one, is
   that of the type: a constant holds no variable. *)
Definition must_be_singleton_rename_type_statement : Prop :=
  ∀ (r : renaming) (t : flambda_type),
    must_be_singleton (rename_type r t) = must_be_singleton t.

(* The meaning of types is equivariant: a value has the renamed type
   under the renamed environment exactly when it has the type under
   the environment. The value is not renamed: it holds no variable.
   FlambdaTypeNominalSpec.v has the two statements in the form of the
   interface for things that contain names, with [env_names] for the
   environment ([type_satisfied_equivariant_statement],
   [env_extension_satisfied_equivariant_statement]). *)
Definition type_satisfied_rename_statement : Prop :=
  ∀ (r : renaming) (statics : static_env) (vars : env) (v : value)
    (t : flambda_type),
    type_satisfied statics (rename_env r vars) v (rename_type r t)
    <-> type_satisfied statics vars v t.

Definition env_extension_satisfied_rename_statement : Prop :=
  ∀ (r : renaming) (statics : static_env) (vars : env)
    (extension : env_extension),
    env_extension_satisfied statics (rename_env r vars)
      (rename_env_extension r extension)
    <-> env_extension_satisfied statics vars extension.

(* ======================================================================== *)
(* The tests of types do not see a renaming                                 *)
(* ======================================================================== *)

(* A renamed type is well formed exactly when the type is. *)
Definition type_well_formed_rename_statement : Prop :=
  (∀ (r : renaming) (t : flambda_type),
    type_well_formed (rename_type r t) = type_well_formed t)
  ∧ (∀ (r : renaming) (extension : env_extension),
      env_extension_well_formed (rename_env_extension r extension)
      = env_extension_well_formed extension).

(* Two renamed types are the same type exactly when the types are. *)
Definition type_eqb_rename_statement : Prop :=
  (∀ (r : renaming) (t t' : flambda_type),
    type_eqb (rename_type r t) (rename_type r t') = type_eqb t t')
  ∧ (∀ (r : renaming) (extension extension' : env_extension),
      env_extension_eqb (rename_env_extension r extension)
        (rename_env_extension r extension')
      = env_extension_eqb extension extension').

(* ======================================================================== *)
(* The names of the equations of a renamed type                             *)
(* ======================================================================== *)

(* They are those of the type, renamed, at the same kinds. *)
Definition equation_names_rename_type_statement : Prop :=
  (∀ (r : renaming) (t : flambda_type),
    equation_names_with_kinds (rename_type r t)
    = map (fun equation => (rename_name r (fst equation), snd equation))
        (equation_names_with_kinds t))
  ∧ (∀ (r : renaming) (extension : env_extension),
      equation_names_of_env_extension (rename_env_extension r extension)
      = map (fun equation => (rename_name r (fst equation), snd equation))
          (equation_names_of_env_extension extension)).

(* ======================================================================== *)
(* The creation functions and the renaming                                  *)
(* ======================================================================== *)

(* A creation function given renamed parts makes the renamed type, and
   what tells how a type is written does not see a renaming. With the
   statements above, and those of FlambdaTypeAccessSpec.v, every
   function of the grammar commutes with the renaming. *)

(* What is made of no name and no type is left as it is. *)
Definition rename_type_closed_statement : Prop :=
  ∀ r : renaming,
    (∀ k : flambda_kind, rename_type r (unknown k) = unknown k)
    ∧ (∀ k : flambda_kind, rename_type r (bottom k) = bottom k)
    ∧ (∀ c : const, rename_type r (this_const c) = this_const c)
    ∧ (∀ c : const, rename_type r (type_for_const c) = type_for_const c)
    ∧ rename_type r null = null
    ∧ rename_type r any_non_null_value = any_non_null_value
    ∧ (∀ alloc_mode : Alloc_mode.t,
        rename_type r (mutable_block alloc_mode) = mutable_block alloc_mode)
    ∧ (∀ bytes : list Z,
        rename_type r (this_immutable_string bytes)
        = this_immutable_string bytes)
    ∧ (∀ imms : list Z,
        rename_type r (these_naked_immediates imms)
        = these_naked_immediates imms)
    ∧ (∀ imms : list Z,
        rename_type r (these_tagged_immediates imms)
        = these_tagged_immediates imms)
    ∧ rename_type r any_tagged_immediate = any_tagged_immediate
    ∧ rename_type r any_block = any_block
    ∧ (∀ (tags : list Z) (alloc_mode : Alloc_mode.t),
        rename_type r (blocks_with_these_tags tags alloc_mode)
        = blocks_with_these_tags tags alloc_mode)
    ∧ rename_row_like_for_blocks r row_like_for_blocks_bottom
      = row_like_for_blocks_bottom
    ∧ rename_env_extension r empty_env_extension = empty_env_extension
    ∧ rename_head_of_kind_value r head_of_kind_value_null
      = head_of_kind_value_null
    ∧ (∀ imm : Z,
        rename_head_of_kind_value r (head_of_kind_value_tagged_immediate imm)
        = head_of_kind_value_tagged_immediate imm).

(* An alias: the simple is renamed. *)
Definition rename_type_alias_statement : Prop :=
  ∀ r : renaming,
    (∀ (k : flambda_kind) (s : simple),
      rename_type r (alias_type_of k s) = alias_type_of k (rename_simple r s))
    ∧ (∀ t : flambda_type,
        get_alias_opt (rename_type r t)
        = option_map (rename_simple r) (get_alias_opt t)).

(* Whether a type, or a structure of blocks, is written as bottom or
   as unknown. *)
Definition is_obviously_rename_type_statement : Prop :=
  ∀ r : renaming,
    (∀ t : flambda_type,
      is_obviously_bottom (rename_type r t) = is_obviously_bottom t)
    ∧ (∀ t : flambda_type,
        is_obviously_unknown (rename_type r t) = is_obviously_unknown t)
    ∧ (∀ blocks : row_like_for_blocks,
        row_like_for_blocks_is_bottom (rename_row_like_for_blocks r blocks)
        = row_like_for_blocks_is_bottom blocks).

(* The extensions of a variant, as [rename_type] renames them. *)
Definition rename_variant_extensions (r : renaming)
    (extensions : variant_extensions) : variant_extensions :=
  match extensions with
  | No_extensions => No_extensions
  | Ext when_immediate when_block =>
    Ext (rename_env_extension r when_immediate)
      (rename_env_extension r when_block)
  end.

(* The types of values that are not null. *)
Definition rename_type_of_head_statement : Prop :=
  ∀ r : renaming,
    (∀ head : head_of_kind_value_non_null,
      rename_type r (non_null_value head)
      = non_null_value (rename_head_of_kind_value_non_null r head))
    ∧ (∀ (is_unique : bool) (immediates : Or_unknown.t flambda_type)
         (blocks : Or_unknown.t row_like_for_blocks)
         (extensions : variant_extensions),
        rename_type r (create_variant is_unique immediates blocks extensions)
        = create_variant is_unique (or_unknown_map (rename_type r) immediates)
            (or_unknown_map (rename_row_like_for_blocks r) blocks)
            (rename_variant_extensions r extensions))
    ∧ (∀ (alloc_mode : Alloc_mode.t) (closures : row_like_for_closures),
        rename_type r (create_closures alloc_mode closures)
        = create_closures alloc_mode
            (rename_row_like_for_closures r closures))
    ∧ (∀ (k : boxable_number) (contents : flambda_type)
         (alloc_mode : Alloc_mode.t),
        rename_type r (box_number k contents alloc_mode)
        = box_number k (rename_type r contents) alloc_mode)
    ∧ (∀ t : flambda_type,
        rename_type r (tag_immediate t) = tag_immediate (rename_type r t)).

(* Blocks: the types of the fields are renamed. *)
Definition rename_type_of_blocks_statement : Prop :=
  ∀ (r : renaming) (tag : Z) (size : nat) (shape : block_shape)
    (maps_to : list flambda_type) (alloc_mode : Alloc_mode.t),
    rename_row_like_for_blocks r
      (row_like_for_blocks_create_exactly tag size shape maps_to alloc_mode)
    = row_like_for_blocks_create_exactly tag size shape
        (map (rename_type r) maps_to) alloc_mode
    ∧ rename_row_like_for_blocks r
        (row_like_for_blocks_create_at_least tag size shape maps_to
           alloc_mode)
      = row_like_for_blocks_create_at_least tag size shape
          (map (rename_type r) maps_to) alloc_mode
    ∧ rename_row_like_for_blocks r
        (row_like_for_blocks_create_at_least_unknown_tag size shape maps_to
           alloc_mode)
      = row_like_for_blocks_create_at_least_unknown_tag size shape
          (map (rename_type r) maps_to) alloc_mode
    ∧ (∀ is_unique : bool,
        rename_type r (immutable_block is_unique tag shape alloc_mode maps_to)
        = immutable_block is_unique tag shape alloc_mode
            (map (rename_type r) maps_to)).

(* The result of a relation: the name it is about is renamed. *)
Definition rename_type_of_relation_statement : Prop :=
  ∀ (r : renaming) (n : name),
    rename_type r (is_int_for_scrutinee n)
    = is_int_for_scrutinee (rename_name r n)
    ∧ rename_type r (get_tag_for_block n) = get_tag_for_block (rename_name r n)
    ∧ rename_type r (is_null_for_scrutinee n)
      = is_null_for_scrutinee (rename_name r n).

(* Closures: the closure types and the value slot types are renamed. *)
Definition rename_type_of_closure_statement : Prop :=
  ∀ (r : renaming) (slot : function_slot)
    (function_types : finite_map (Or_unknown.t function_type))
    (closure_types value_slot_types : finite_map flambda_type)
    (alloc_mode : Alloc_mode.t),
    rename_type r
      (exactly_this_closure slot function_types closure_types
         value_slot_types alloc_mode)
    = exactly_this_closure slot function_types
        (map_values (rename_type r) closure_types)
        (map_values (rename_type r) value_slot_types) alloc_mode
    ∧ rename_type r
        (at_least_this_closure slot function_types closure_types
           value_slot_types alloc_mode)
      = at_least_this_closure slot function_types
          (map_values (rename_type r) closure_types)
          (map_values (rename_type r) value_slot_types) alloc_mode.
