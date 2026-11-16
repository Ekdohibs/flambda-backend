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


(* What the provers and reification read of a type: the functions of the
   grammar that look into a row-like structure or an entry of closures.

   This file is to be reviewed: the definitions, against
   [Type_grammar.Row_like_for_blocks.all_tags], [all_tags_and_sizes],
   [get_singleton] and [get_field],
   [Type_grammar.Row_like_for_closures.get_single_tag], [get_closure]
   and [get_env_var], and [Type_grammar.Closures_entry.find_function_type];
   and the statements.

   These are the functions through which provers.ml and reify.ml read
   the blocks and the closures of a type, once [Expand_head] has given
   them a head. Each has here a statement of what its answer says of a
   value of the type, so that the provers can be stated from these and
   need not go back to the meaning of types.

   How the definitions differ from the compiler's.
   - A size and the index of a field are natural numbers; the compiler
     has target integers.
   - [all_tags_and_sizes] is written with a test of the cases and not
     with a flag that a function over the map sets.
   - [get_single_tag] stops the compiler when the one case has an
     exact index without the function slot of the case. Here it
     answers [None] there, and a well-formed type has no such case
     ([get_single_tag_well_formed_statement]).
   - A map is read by [lookup], so a binding that is shadowed is not
     seen: the keys of a map are its domain, and its one binding, if
     it has one, is that of the one key bound.

   What an answer is not. [get_singleton] and [get_single_tag] drop the
   environment extension of the case, as the compiler does (it holds
   that what the extension says is in the environment by then). The
   statements say what holds of a value of the type, of which the
   extension holds too; they do not say that a value with what is
   answered has the type.

   One answer rests on well-formedness.
   [find_function_type] with [exact] answers [Bottom] when the entry
   has no function type for the slot. A prover reads that as "the type
   has no value" ([Provers.prove_single_closures_entry]). That is
   right of a well-formed type only, in which an exact case has a
   function type for each function of its index, the slot of the case
   among them (condition 8 of FlambdaTypeWellFormedSpec.v):
   [find_function_type_exact_statement] says that the answer is then
   never [Bottom]. Of a type that is not well formed the reading is
   wrong: the meaning does not ask an entry to have a function type.

   The last section says that each of these functions commutes with
   the renaming of variables (FlambdaTypeRenameSpec.v), which a
   function that reads types owes to the interface for things that
   contain names.

   The proofs are in FlambdaTypeAccessProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec FlambdaNominalSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypeWellFormedSpec FlambdaTypeRenameSpec.
Import ListNotations.

(* ======================================================================== *)
(* Blocks                                                                   *)
(* ======================================================================== *)

(* [Row_like_for_blocks.all_tags]: the tags that a block of the
   structure may have, if they are known. *)
Definition row_like_for_blocks_all_tags (blocks : row_like_for_blocks)
    : Or_unknown.t (list Z) :=
  match blocks with
  | Row_like_for_blocks known_tags other_tags _ =>
    match other_tags with
    | Or_bottom.Ok _ => Or_unknown.Unknown
    | Or_bottom.Bottom => Or_unknown.Known (Keyed.domain Z.eqb known_tags)
    end
  end.

(* The size and the shape of a case, if the case is known and its size
   is exact. *)
Definition block_case_size_and_shape
    (case : Or_unknown.t row_like_block_case) : option (nat * block_shape) :=
  match case with
  | Or_unknown.Known case =>
    match index_domain (case_index case) with
    | Row_like_index_domain.Known size =>
      Some (size, index_shape (case_index case))
    | Row_like_index_domain.At_least _ => None
    end
  | Or_unknown.Unknown => None
  end.

(* [Row_like_for_blocks.all_tags_and_sizes]: the size and the shape of
   the blocks of each tag, if the tags are known and each has a case
   of an exact size. The size and the shape given for a case that has
   none are never read. *)
Definition row_like_for_blocks_all_tags_and_sizes
    (blocks : row_like_for_blocks)
    : Or_unknown.t (Keyed.finite_map Z (nat * block_shape)) :=
  match blocks with
  | Row_like_for_blocks known_tags other_tags _ =>
    match other_tags with
    | Or_bottom.Ok _ => Or_unknown.Unknown
    | Or_bottom.Bottom =>
      if
        Keyed.for_all_bindings_b Z.eqb
          (fun _ case =>
             match block_case_size_and_shape case with
             | Some _ => true
             | None => false
             end)
          known_tags
      then
        Or_unknown.Known
          (Keyed.map_values
             (fun case =>
                match block_case_size_and_shape case with
                | Some size_and_shape => size_and_shape
                | None => (0, Scannable Value_only)
                end)
             known_tags)
      else Or_unknown.Unknown
    end
  end.

(* [Row_like_for_blocks.get_singleton]: the tag, the shape, the size
   and the types of the fields of the blocks of the structure, if they
   all have one tag and one exact size. *)
Definition row_like_for_blocks_get_singleton (blocks : row_like_for_blocks)
    : option (Z * block_shape * nat * list flambda_type * Alloc_mode.t) :=
  match blocks with
  | Row_like_for_blocks known_tags other_tags alloc_mode =>
    match other_tags with
    | Or_bottom.Ok _ => None
    | Or_bottom.Bottom =>
      match Keyed.get_singleton_binding Z.eqb known_tags with
      | None | Some (_, Or_unknown.Unknown) => None
      | Some (tag, Or_unknown.Known case) =>
        match index_domain (case_index case) with
        | Row_like_index_domain.At_least _ => None
        | Row_like_index_domain.Known size =>
          Some
            (tag, index_shape (case_index case), size, case_maps_to case,
             alloc_mode)
        end
      end
    end
  end.

(* [Row_like_for_blocks.get_field]: the type of the field at [index] of
   the blocks of the structure, if [get_singleton] has them; bottom if
   they have no such field. *)
Definition row_like_for_blocks_get_field (blocks : row_like_for_blocks)
    (index : nat) : or_unknown_or_bottom flambda_type :=
  match row_like_for_blocks_get_singleton blocks with
  | None => Unknown
  | Some (_, _, size, maps_to, _) =>
    if Nat.leb size index
    then Bottom
    else
      match nth_error maps_to index with
      | None => Unknown
      | Some ty => Ok ty
      end
  end.

(* ======================================================================== *)
(* Closures                                                                 *)
(* ======================================================================== *)

(* [Closures_entry.find_function_type]: the function type of a slot.
   An entry of a case that is exact has one for every function of the
   set, so the absence of one is bottom there. *)
Definition closures_entry_find_function_type (entry : closures_entry)
    (exact : bool) (slot : function_slot)
    : or_unknown_or_bottom function_type :=
  match entry with
  | Closures_entry function_types _ _ =>
    match lookup slot function_types with
    | None => if exact then Bottom else Unknown
    | Some Or_unknown.Unknown => Unknown
    | Some (Or_unknown.Known function_type) => Ok function_type
    end
  end.

(* [Row_like_for_closures.get_single_tag_result]. *)
Inductive get_single_tag_result : Type :=
  | No_singleton
  | Exact_closure (slot : function_slot) (entry : closures_entry)
  | Incomplete_closure (slot : function_slot) (entry : closures_entry).

(* [Row_like_for_closures.get_single_tag]: the function slot and the
   entry of the closures of the structure, if they are all at one
   slot, and whether the functions and value slots of their set are
   known exactly. [None] is where the compiler stops. *)
Definition row_like_for_closures_get_single_tag
    (closures : row_like_for_closures) : option get_single_tag_result :=
  match closures with
  | Row_like_for_closures known_closures =>
    match Keyed.get_singleton_binding Pos.eqb known_closures with
    | None => Some No_singleton
    | Some (slot, case) =>
      match index_domain (case_index case) with
      | Row_like_index_domain.At_least contents =>
        if among slot (contents_closures contents)
        then Some (Incomplete_closure slot (case_maps_to case))
        else Some No_singleton
      | Row_like_index_domain.Known contents =>
        if among slot (contents_closures contents)
        then Some (Exact_closure slot (case_maps_to case))
        else None
      end
    end
  end.

(* The entry that [get_single_tag] answers, if it answers one. *)
Definition single_closures_entry (result : option get_single_tag_result)
    : option (function_slot * closures_entry) :=
  match result with
  | Some (Exact_closure slot entry) | Some (Incomplete_closure slot entry) =>
    Some (slot, entry)
  | Some No_singleton | None => None
  end.

(* [Row_like_for_closures.get_closure]: the type of the closure at a
   function slot of the set of the closures of the structure. The
   outer [None] is where [get_single_tag] stops. *)
Definition row_like_for_closures_get_closure
    (closures : row_like_for_closures) (slot : function_slot)
    : option (Or_unknown.t flambda_type) :=
  match row_like_for_closures_get_single_tag closures with
  | None => None
  | Some No_singleton => Some Or_unknown.Unknown
  | Some (Exact_closure _ (Closures_entry _ closure_types _))
  | Some (Incomplete_closure _ (Closures_entry _ closure_types _)) =>
    match lookup slot closure_types with
    | None => Some Or_unknown.Unknown
    | Some ty => Some (Or_unknown.Known ty)
    end
  end.

(* [Row_like_for_closures.get_env_var]: the type of a value slot of
   the closures of the structure. *)
Definition row_like_for_closures_get_env_var
    (closures : row_like_for_closures) (slot : value_slot)
    : option (Or_unknown.t flambda_type) :=
  match row_like_for_closures_get_single_tag closures with
  | None => None
  | Some No_singleton => Some Or_unknown.Unknown
  | Some (Exact_closure _ (Closures_entry _ _ value_slot_types))
  | Some (Incomplete_closure _ (Closures_entry _ _ value_slot_types)) =>
    match lookup slot value_slot_types with
    | None => Some Or_unknown.Unknown
    | Some ty => Some (Or_unknown.Known ty)
    end
  end.

(* ======================================================================== *)
(* What the answers say of a block                                          *)
(* ======================================================================== *)

(* The tag of a block of the structure is one of the tags answered. *)
Definition all_tags_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (tag : Z) (fields : list value)
    (blocks : row_like_for_blocks) (tags : list Z),
    row_like_for_blocks_all_tags blocks = Or_unknown.Known tags ->
    row_like_for_blocks_satisfied statics vars tag fields blocks ->
    tag ∈ tags.

(* A block of the structure has the size and the shape answered for
   its tag. *)
Definition all_tags_and_sizes_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (tag : Z) (fields : list value)
    (blocks : row_like_for_blocks)
    (sizes : Keyed.finite_map Z (nat * block_shape)),
    row_like_for_blocks_all_tags_and_sizes blocks = Or_unknown.Known sizes ->
    row_like_for_blocks_satisfied statics vars tag fields blocks ->
    ∃ (size : nat) (shape : block_shape),
      Keyed.lookup Z.eqb tag sizes = Some (size, shape)
      ∧ length fields = size
      ∧ block_has_shape shape tag fields.

(* A block of the structure has the tag, the size and the shape
   answered, and its fields have the types answered. *)
Definition get_singleton_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (tag' : Z) (fields : list value)
    (blocks : row_like_for_blocks) (tag : Z) (shape : block_shape)
    (size : nat) (tys : list flambda_type) (alloc_mode : Alloc_mode.t),
    row_like_for_blocks_get_singleton blocks
    = Some (tag, shape, size, tys, alloc_mode) ->
    row_like_for_blocks_satisfied statics vars tag' fields blocks ->
    tag' = tag
    ∧ length fields = size
    ∧ block_has_shape shape tag fields
    ∧ fields_satisfied (type_satisfied statics vars) tys fields.

(* The field of a block of the structure, at the index asked, has the
   type answered; if the answer is bottom the block has no such
   field. *)
Definition get_field_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (tag : Z) (fields : list value)
    (blocks : row_like_for_blocks) (index : nat),
    row_like_for_blocks_satisfied statics vars tag fields blocks ->
    match row_like_for_blocks_get_field blocks index with
    | Ok ty =>
      ∃ field : value,
        nth_error fields index = Some field
        ∧ type_satisfied statics vars field ty
    | Bottom => nth_error fields index = None
    | Unknown => True
    end.

(* ======================================================================== *)
(* What the answers say of a closure                                        *)
(* ======================================================================== *)

(* The function type answered for a function of the set of a closure
   names the code of that function. *)
Definition find_function_type_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (slot : function_slot)
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value)) (entry : closures_entry)
    (exact : bool) (g : function_slot) (function_type : function_type)
    (code : code_id),
    closures_entry_satisfied statics vars v slot functions value_slots entry ->
    closures_entry_find_function_type entry exact g = Ok function_type ->
    lookup g functions = Some code ->
    code = function_type_code_id function_type.

(* In a well-formed case whose index is exact, the answer for the slot
   of the case is not bottom. *)
(* CR mshinwell: the compiler reads the answer bottom as a type with no
   value ([Provers.prove_single_closures_entry] answers [Invalid]).
   Here that is not stated: a type whose exact case has no function
   type for a function of its set has values, since the meaning asks
   nothing of an entry that says nothing. The other choice is to make
   the meaning of an exact case ask for a function type of each
   function, which would make the meaning depend on how an entry is
   written. Nothing in the compiler checks that an exact case has
   them (T1-7). No decision has a number. *)
Definition find_function_type_exact_statement : Prop :=
  ∀ (slot : function_slot) (entry : closures_entry)
    (contents : set_of_closures_contents) (extension : env_extension),
    row_like_closures_case_well_formed slot
      (Row_like_case entry
         (Row_like_index (Row_like_index_domain.Known contents) tt)
         extension)
    = true ->
    closures_entry_find_function_type entry true slot <> Bottom.

(* A closure of the structure is at the slot answered, and the entry
   answered holds of it. *)
Definition get_single_tag_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (slot' : function_slot)
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value))
    (closures : row_like_for_closures) (slot : function_slot)
    (entry : closures_entry),
    single_closures_entry (row_like_for_closures_get_single_tag closures)
    = Some (slot, entry) ->
    row_like_for_closures_satisfied statics vars v slot' functions
      value_slots closures ->
    slot' = slot
    ∧ closures_entry_satisfied statics vars v slot functions value_slots
        entry.

(* When the answer is exact, the set of the closure has the functions
   and the value slots of a set of closures contents, and no other,
   and the case answered is the one case, which is exact. *)
Definition get_single_tag_exact_statement : Prop :=
  ∀ (known_closures : finite_map row_like_closures_case)
    (slot : function_slot) (entry : closures_entry),
    row_like_for_closures_get_single_tag
      (Row_like_for_closures known_closures)
    = Some (Exact_closure slot entry) ->
    ∃ (contents : set_of_closures_contents) (extension : env_extension),
      (∀ slot' : function_slot,
        lookup slot' known_closures
        = if Pos.eqb slot' slot
          then
            Some
              (Row_like_case entry
                 (Row_like_index (Row_like_index_domain.Known contents) tt)
                 extension)
          else None)
      ∧ among slot (contents_closures contents) = true.

(* The compiler does not stop on a well-formed structure. *)
Definition get_single_tag_well_formed_statement : Prop :=
  ∀ closures : row_like_for_closures,
    row_like_for_closures_well_formed closures = true ->
    row_like_for_closures_get_single_tag closures <> None.

(* The closure at a function slot of the set of a closure of the
   structure has the type answered. *)
Definition get_closure_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (slot : function_slot)
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value))
    (closures : row_like_for_closures) (g : function_slot)
    (ty : flambda_type) (closure : value),
    row_like_for_closures_satisfied statics vars v slot functions value_slots
      closures ->
    row_like_for_closures_get_closure closures g
    = Some (Or_unknown.Known ty) ->
    project_function_slot statics slot g v = Some closure ->
    type_satisfied statics vars closure ty.

(* The contents of a value slot of a closure of the structure have the
   type answered. *)
Definition get_env_var_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (slot : function_slot)
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value))
    (closures : row_like_for_closures) (w : value_slot) (ty : flambda_type)
    (contents : value),
    row_like_for_closures_satisfied statics vars v slot functions value_slots
      closures ->
    row_like_for_closures_get_env_var closures w
    = Some (Or_unknown.Known ty) ->
    lookup w value_slots = Some contents ->
    type_satisfied statics vars contents ty.

(* ======================================================================== *)
(* The answers on a renamed type                                            *)
(* ======================================================================== *)

(* What each function answers of a renamed structure is its answer,
   renamed. A tag, a size, a shape, a slot and a function type hold no
   variable, so the answers made of those are the same. *)

(* The answer of [get_single_tag], renamed: the entry is. *)
Definition rename_get_single_tag_result (r : renaming)
    (result : get_single_tag_result) : get_single_tag_result :=
  match result with
  | No_singleton => No_singleton
  | Exact_closure slot entry =>
    Exact_closure slot (rename_closures_entry r entry)
  | Incomplete_closure slot entry =>
    Incomplete_closure slot (rename_closures_entry r entry)
  end.

Definition all_tags_rename_statement : Prop :=
  ∀ (r : renaming) (blocks : row_like_for_blocks),
    row_like_for_blocks_all_tags (rename_row_like_for_blocks r blocks)
    = row_like_for_blocks_all_tags blocks
    ∧ row_like_for_blocks_all_tags_and_sizes
        (rename_row_like_for_blocks r blocks)
      = row_like_for_blocks_all_tags_and_sizes blocks.

Definition get_singleton_rename_statement : Prop :=
  ∀ (r : renaming) (blocks : row_like_for_blocks),
    row_like_for_blocks_get_singleton (rename_row_like_for_blocks r blocks)
    = option_map
        (fun singleton =>
           match singleton with
           | (tag, shape, size, tys, alloc_mode) =>
             (tag, shape, size, map (rename_type r) tys, alloc_mode)
           end)
        (row_like_for_blocks_get_singleton blocks).

Definition get_field_rename_statement : Prop :=
  ∀ (r : renaming) (blocks : row_like_for_blocks) (index : nat),
    row_like_for_blocks_get_field (rename_row_like_for_blocks r blocks) index
    = or_unknown_or_bottom_map (rename_type r)
        (row_like_for_blocks_get_field blocks index).

Definition find_function_type_rename_statement : Prop :=
  ∀ (r : renaming) (entry : closures_entry) (exact : bool)
    (slot : function_slot),
    closures_entry_find_function_type (rename_closures_entry r entry) exact
      slot
    = closures_entry_find_function_type entry exact slot.

(* The answer of [get_single_tag] is renamed, and the entry read from
   it with it. *)
Definition get_single_tag_rename_statement : Prop :=
  ∀ (r : renaming) (closures : row_like_for_closures),
    row_like_for_closures_get_single_tag
      (rename_row_like_for_closures r closures)
    = option_map (rename_get_single_tag_result r)
        (row_like_for_closures_get_single_tag closures)
    ∧ single_closures_entry
        (row_like_for_closures_get_single_tag
           (rename_row_like_for_closures r closures))
      = option_map
          (fun answer => (fst answer, rename_closures_entry r (snd answer)))
          (single_closures_entry
             (row_like_for_closures_get_single_tag closures)).

Definition get_closure_rename_statement : Prop :=
  ∀ (r : renaming) (closures : row_like_for_closures),
    (∀ slot : function_slot,
      row_like_for_closures_get_closure
        (rename_row_like_for_closures r closures) slot
      = option_map (or_unknown_map (rename_type r))
          (row_like_for_closures_get_closure closures slot))
    ∧ (∀ slot : value_slot,
        row_like_for_closures_get_env_var
          (rename_row_like_for_closures r closures) slot
        = option_map (or_unknown_map (rename_type r))
            (row_like_for_closures_get_env_var closures slot)).
