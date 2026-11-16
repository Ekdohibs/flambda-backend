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


(* When two Flambda types are the same type.

   This file is to be reviewed: [type_eqb] and the tests defined with
   it, against the compiler's [Type_grammar.equal], and the statements.

   The joins of the compiler ask in some places whether two types are
   the same, to keep one of them as it is. They asked whether the two
   were one value in memory; they are to ask, with
   [-flambda2-join-sharing structural], whether the two are the same
   structure, with [Type_grammar.equal] (decision D30 of the project).
   [type_eqb] is that function. Nothing of the model speaks of sharing
   in memory.

   What is compared, as in the compiler: the kind; whether the type is
   unknown, bottom, an alias or given by a head (an alias is the same
   as an alias only, never as what it expands to); the simple of an
   alias, a name being the same as itself only (no environment is
   consulted, and nothing is up to renaming); every part of a head,
   allocation modes and [is_unique] included; an environment extension
   name by name.

   A map is compared as a map and a set as a set, and not as the lists
   that the model writes them with: two maps are the same when each key
   is bound in both or in neither, to things that are the same
   ([map_eqb], [map_eqb_iff_statement]); two sets, when they have the
   same members. So [type_eqb] does not decide the equality of Rocq on
   types: it decides [type_equal], which is coarser, and is what the
   statements are about. It is an equivalence, and two types that it
   identifies have the same kind and the same meaning.

   They need not have the same [free_variables]: that function lists
   the variables of a binding of a map that is shadowed, which nothing
   here compares. In two well-formed types (FlambdaTypeWellFormedSpec.v),
   where a map binds a key once, the two lists would have the same
   members; they need not be the same list, since the bindings of two
   maps that are the same may come in another order. It is not
   stated. A join that finds its inputs the same keeps the first as it
   is, so that what is known of its variables is known of the result.

   Where the compiler's function sees more than this one, because the
   grammar here has less: the [rec_info] of a function type, the
   expression of a [Rec_info] head, the string of a poison constant, a
   relation of a naked immediate with no name, coercions, arrays. Two
   types of the compiler that differ only there are the same type here,
   so the compiler may join where the model keeps a type as it is, and
   never the other way.

   The proofs are in FlambdaTypeEqualProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaCompareSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypeMeaningBoolSpec.
Import ListNotations.

(* ======================================================================== *)
(* What types are made of                                                   *)
(* ======================================================================== *)

(* Two maps are the same map: each key that one binds, the other binds
   to the same thing. [eqb] is asked of what the first map binds,
   so that a recursive definition may go through it. *)
Definition map_eqb {K A : Type} (key_eqb : K -> K -> bool)
    (eqb : A -> A -> bool) (bindings bindings' : Keyed.finite_map K A)
    : bool :=
  Keyed.for_all_bindings_b key_eqb
    (fun key a =>
       match Keyed.lookup key_eqb key bindings' with
       | Some a' => eqb a a'
       | None => false
       end)
    bindings
  && Keyed.for_all_bindings_b key_eqb
       (fun key _ => Keyed.is_bound key_eqb key bindings)
       bindings'.

Definition map_eqb_iff_statement : Prop :=
  ∀ (K : Type) (key_eqb : K -> K -> bool),
    decides_equality key_eqb ->
    ∀ (A : Type) (eqb : A -> A -> bool)
      (bindings bindings' : Keyed.finite_map K A),
      map_eqb key_eqb eqb bindings bindings' = true
      <-> (∀ key : K,
            match
              Keyed.lookup key_eqb key bindings,
              Keyed.lookup key_eqb key bindings'
            with
            | Some a, Some a' => eqb a a' = true
            | None, None => True
            | Some _, None | None, Some _ => False
            end).

(* Names, simples and the small types. A name is the same as itself
   only. *)
Definition simple_eqb (s s' : simple) : bool :=
  match s, s' with
  | Var x, Var x' => Pos.eqb x x'
  | Symbol sym, Symbol sym' => Pos.eqb sym sym'
  | Const c, Const c' => const_eqb c c'
  | _, _ => false
  end.

Definition option_variable_eqb (x x' : option variable) : bool :=
  match x, x' with
  | Some x, Some x' => Pos.eqb x x'
  | None, None => true
  | _, _ => false
  end.

Definition alloc_mode_eqb (mode mode' : Alloc_mode.t) : bool :=
  match mode, mode' with
  | Alloc_mode.Heap, Alloc_mode.Heap
  | Alloc_mode.Local, Alloc_mode.Local
  | Alloc_mode.Heap_or_local, Alloc_mode.Heap_or_local => true
  | _, _ => false
  end.

Definition relation_eqb (relation relation' : Relation.t) : bool :=
  match relation, relation' with
  | Relation.Is_null, Relation.Is_null
  | Relation.Is_int, Relation.Is_int
  | Relation.Get_tag, Relation.Get_tag => true
  | _, _ => false
  end.

Definition naked_number_kind_eqb (k k' : naked_number_kind) : bool :=
  kind_eqb (Naked_number k) (Naked_number k').

Definition boxable_number_eqb (k k' : boxable_number) : bool :=
  kind_eqb (unboxed_kind k) (unboxed_kind k').

Definition is_null_eqb (is_null is_null' : is_null) : bool :=
  match is_null, is_null' with
  | Not_null, Not_null => true
  | Maybe_null x, Maybe_null x' => option_variable_eqb x x'
  | _, _ => false
  end.

Definition block_shape_eqb (shape shape' : block_shape) : bool :=
  match shape, shape' with
  | Scannable Value_only, Scannable Value_only => true
  | Scannable (Mixed_record value_prefix_size flat_suffix),
    Scannable (Mixed_record value_prefix_size' flat_suffix') =>
    Nat.eqb value_prefix_size value_prefix_size'
    && lists_eqb naked_number_kind_eqb flat_suffix flat_suffix'
  | Float_record, Float_record => true
  | _, _ => false
  end.

Definition function_type_eqb (function_type function_type' : function_type)
    : bool :=
  Pos.eqb (function_type_code_id function_type)
    (function_type_code_id function_type').

(* The index of a case of blocks: exact or from below, the size, the
   shape. *)
Definition block_index_eqb (index index' : row_like_index nat block_shape)
    : bool :=
  match index_domain index, index_domain index' with
  | Row_like_index_domain.Known size, Row_like_index_domain.Known size'
  | Row_like_index_domain.At_least size, Row_like_index_domain.At_least size'
    => Nat.eqb size size'
  | _, _ => false
  end
  && block_shape_eqb (index_shape index) (index_shape index').

(* The index of a case of closures: exact or from below, the function
   slots and the value slots, as sets. *)
Definition closures_index_eqb
    (index index' : row_like_index set_of_closures_contents unit) : bool :=
  match index_domain index, index_domain index' with
  | Row_like_index_domain.Known contents,
    Row_like_index_domain.Known contents'
  | Row_like_index_domain.At_least contents,
    Row_like_index_domain.At_least contents' =>
    Keyed.same_keys_b Pos.eqb (contents_closures contents)
      (contents_closures contents')
    && Keyed.same_keys_b Pos.eqb (contents_value_slots contents)
         (contents_value_slots contents')
  | _, _ => false
  end.

(* [Or_unknown.t], [Or_bottom.t] and [Or_unknown_or_bottom.t]. *)
Definition or_unknown_eqb {A : Type} (eqb : A -> A -> bool)
    (a a' : Or_unknown.t A) : bool :=
  match a, a' with
  | Or_unknown.Known a, Or_unknown.Known a' => eqb a a'
  | Or_unknown.Unknown, Or_unknown.Unknown => true
  | _, _ => false
  end.

Definition or_bottom_eqb {A : Type} (eqb : A -> A -> bool)
    (a a' : Or_bottom.t A) : bool :=
  match a, a' with
  | Or_bottom.Ok a, Or_bottom.Ok a' => eqb a a'
  | Or_bottom.Bottom, Or_bottom.Bottom => true
  | _, _ => false
  end.

Definition or_unknown_or_bottom_eqb {A : Type} (eqb : A -> A -> bool)
    (a a' : or_unknown_or_bottom A) : bool :=
  match a, a' with
  | Unknown, Unknown | Bottom, Bottom => true
  | Ok a, Ok a' => eqb a a'
  | _, _ => false
  end.

(* A type that is unknown, bottom, an alias or given by a head. An
   alias is the same as an alias to the same simple, and as nothing
   else. *)
Definition descr_eqb {head : Type} (head_eqb : head -> head -> bool)
    (ty ty' : type_descr head) : bool :=
  or_unknown_or_bottom_eqb
    (fun descr descr' =>
       match descr, descr' with
       | No_alias head, No_alias head' => head_eqb head head'
       | Equals s, Equals s' => simple_eqb s s'
       | _, _ => false
       end)
    ty ty'.

(* A head of naked immediates: the numbers, as a set, or their absence;
   the relations with their names, as a set. *)
Definition head_of_kind_naked_immediate_eqb
    (head head' : head_of_kind_naked_immediate) : bool :=
  or_unknown_eqb (Keyed.same_keys_b Z.eqb) (naked_immediates head)
    (naked_immediates head')
  && Keyed.same_keys_b
       (fun relation relation' =>
          relation_eqb (fst relation) (fst relation')
          && name_eqb (snd relation) (snd relation'))
       (inverse_relations head) (inverse_relations head').

(* A case of blocks: the index, the types of the product one by one,
   the extension. The two arguments stand for the functions being
   defined below. *)
Definition block_case_eqb (type_eqb : flambda_type -> flambda_type -> bool)
    (env_extension_eqb : env_extension -> env_extension -> bool)
    (case case' : row_like_block_case) : bool :=
  block_index_eqb (case_index case) (case_index case')
  && lists_eqb type_eqb (case_maps_to case) (case_maps_to case')
  && env_extension_eqb (case_env_extension case) (case_env_extension case').

(* A case of closures: the index, the entry, the extension. The two
   arguments stand for the functions being defined below. *)
Definition closures_case_eqb
    (closures_entry_eqb : closures_entry -> closures_entry -> bool)
    (env_extension_eqb : env_extension -> env_extension -> bool)
    (case case' : row_like_closures_case) : bool :=
  closures_index_eqb (case_index case) (case_index case')
  && closures_entry_eqb (case_maps_to case) (case_maps_to case')
  && env_extension_eqb (case_env_extension case) (case_env_extension case').

(* The extensions of a variant: none, or the two. The argument stands
   for the function being defined below. *)
Definition variant_extensions_eqb
    (env_extension_eqb : env_extension -> env_extension -> bool)
    (extensions extensions' : variant_extensions) : bool :=
  match extensions, extensions' with
  | No_extensions, No_extensions => true
  | Ext when_immediate when_block, Ext when_immediate' when_block' =>
    env_extension_eqb when_immediate when_immediate'
    && env_extension_eqb when_block when_block'
  | _, _ => false
  end.

(* ======================================================================== *)
(* The same type                                                            *)
(* ======================================================================== *)

(* [Type_grammar.equal], and the six tests defined with it, one for
   each structure of the grammar. *)
Fixpoint type_eqb (t t' : flambda_type) {struct t} : bool :=
  match t, t' with
  | Type_value ty, Type_value ty' => descr_eqb head_of_kind_value_eqb ty ty'
  | Type_naked_immediate ty, Type_naked_immediate ty' =>
    descr_eqb head_of_kind_naked_immediate_eqb ty ty'
  | Type_naked_number k ty, Type_naked_number k' ty' =>
    naked_number_kind_eqb k k' && descr_eqb (Keyed.same_keys_b Z.eqb) ty ty'
  | Type_rec_info ty, Type_rec_info ty' => descr_eqb (fun _ _ => true) ty ty'
  | Type_region ty, Type_region ty' => descr_eqb (fun _ _ => true) ty ty'
  | _, _ => false
  end

with head_of_kind_value_eqb (head head' : head_of_kind_value) {struct head}
    : bool :=
  match head, head' with
  | Head_of_kind_value non_null is_null,
    Head_of_kind_value non_null' is_null' =>
    or_unknown_or_bottom_eqb head_of_kind_value_non_null_eqb non_null
      non_null'
    && is_null_eqb is_null is_null'
  end

with head_of_kind_value_non_null_eqb
    (head head' : head_of_kind_value_non_null) {struct head} : bool :=
  match head, head' with
  | Variant is_int immediates get_tag blocks extensions is_unique,
    Variant is_int' immediates' get_tag' blocks' extensions' is_unique' =>
    option_variable_eqb is_int is_int'
    && or_unknown_eqb type_eqb immediates immediates'
    && option_variable_eqb get_tag get_tag'
    && or_unknown_eqb row_like_for_blocks_eqb blocks blocks'
    && variant_extensions_eqb env_extension_eqb extensions extensions'
    && Bool.eqb is_unique is_unique'
  | Mutable_block alloc_mode, Mutable_block alloc_mode' =>
    alloc_mode_eqb alloc_mode alloc_mode'
  | Boxed_number k contents alloc_mode, Boxed_number k' contents' alloc_mode'
    =>
    boxable_number_eqb k k'
    && type_eqb contents contents'
    && alloc_mode_eqb alloc_mode alloc_mode'
  | Closures by_function_slot alloc_mode,
    Closures by_function_slot' alloc_mode' =>
    row_like_for_closures_eqb by_function_slot by_function_slot'
    && alloc_mode_eqb alloc_mode alloc_mode'
  | String strs, String strs' =>
    Keyed.same_keys_b (lists_eqb Z.eqb) strs strs'
  | _, _ => false
  end

with row_like_for_blocks_eqb (blocks blocks' : row_like_for_blocks)
    {struct blocks} : bool :=
  match blocks, blocks' with
  | Row_like_for_blocks known_tags other_tags alloc_mode,
    Row_like_for_blocks known_tags' other_tags' alloc_mode' =>
    map_eqb Z.eqb
      (or_unknown_eqb (block_case_eqb type_eqb env_extension_eqb))
      known_tags known_tags'
    && or_bottom_eqb (block_case_eqb type_eqb env_extension_eqb) other_tags
         other_tags'
    && alloc_mode_eqb alloc_mode alloc_mode'
  end

with row_like_for_closures_eqb (closures closures' : row_like_for_closures)
    {struct closures} : bool :=
  match closures, closures' with
  | Row_like_for_closures known_closures,
    Row_like_for_closures known_closures' =>
    map_eqb Pos.eqb (closures_case_eqb closures_entry_eqb env_extension_eqb)
      known_closures known_closures'
  end

with closures_entry_eqb (entry entry' : closures_entry) {struct entry}
    : bool :=
  match entry, entry' with
  | Closures_entry function_types closure_types value_slot_types,
    Closures_entry function_types' closure_types' value_slot_types' =>
    map_eqb Pos.eqb (or_unknown_eqb function_type_eqb) function_types
      function_types'
    && map_eqb Pos.eqb type_eqb closure_types closure_types'
    && map_eqb Pos.eqb type_eqb value_slot_types value_slot_types'
  end

with env_extension_eqb (extension extension' : env_extension)
    {struct extension} : bool :=
  match extension, extension' with
  | Env_extension equations, Env_extension equations' =>
    map_eqb name_eqb type_eqb equations equations'
  end.

(* The relation that [type_eqb] decides, and the same for environment
   extensions. *)
Definition type_equal (t t' : flambda_type) : Prop := type_eqb t t' = true.

Definition env_extension_equal (extension extension' : env_extension)
    : Prop :=
  env_extension_eqb extension extension' = true.

(* ======================================================================== *)
(* What it is to be the same type                                           *)
(* ======================================================================== *)

(* An equivalence. *)
Definition type_equal_equivalence_statement : Prop :=
  (∀ t : flambda_type, type_equal t t)
  ∧ (∀ t t' : flambda_type, type_equal t t' -> type_equal t' t)
  ∧ (∀ t t' t'' : flambda_type,
      type_equal t t' -> type_equal t' t'' -> type_equal t t'').

Definition env_extension_equal_equivalence_statement : Prop :=
  (∀ extension : env_extension, env_extension_equal extension extension)
  ∧ (∀ extension extension' : env_extension,
      env_extension_equal extension extension' ->
      env_extension_equal extension' extension)
  ∧ (∀ extension extension' extension'' : env_extension,
      env_extension_equal extension extension' ->
      env_extension_equal extension' extension'' ->
      env_extension_equal extension extension'').

(* Two environment extensions are the same when they have equations on
   the same names, with types that are the same. *)
Definition env_extension_equal_lookup_statement : Prop :=
  ∀ equations equations' : Keyed.finite_map name flambda_type,
    env_extension_equal (Env_extension equations) (Env_extension equations')
    <-> (∀ n : name,
          match
            Keyed.lookup name_eqb n equations,
            Keyed.lookup name_eqb n equations'
          with
          | Some t, Some t' => type_equal t t'
          | None, None => True
          | Some _, None | None, Some _ => False
          end).

(* The same kind. *)
Definition type_equal_kind_statement : Prop :=
  ∀ t t' : flambda_type, type_equal t t' -> kind t = kind t'.

(* The same meaning. *)
Definition type_equal_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (t t' : flambda_type),
    type_equal t t' ->
    (type_satisfied statics vars v t <-> type_satisfied statics vars v t').

Definition env_extension_equal_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (extension extension' : env_extension),
    env_extension_equal extension extension' ->
    (env_extension_satisfied statics vars extension
     <-> env_extension_satisfied statics vars extension').
