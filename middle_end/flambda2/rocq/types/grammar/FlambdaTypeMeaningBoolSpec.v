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


(* The meaning of a Flambda type, as a test.

   [type_satisfied] (FlambdaTypeMeaningSpec.v) says which values have a
   type. It is decidable, and [type_satisfied_b] here decides it: a
   boolean function that extracts, so that the meaning can be run on the
   values of a run against the types that the compiler infers. Each of
   the functions defined with the meaning has its test here, under its
   name followed by [_b], and a statement that the test answers [true]
   exactly when the meaning holds ([type_satisfied_b_iff_statement] and the
   others).

   What is to be reviewed here is the statements, which are short. The
   tests need no review of their own: the statements tie them to the
   meaning, and nothing but the statements is to be used of them in a
   proof.

   One definition at the head of this file is of the machine and not of
   the types, and is here until the machine has it: [value_eqb], the
   equality test of values, which the test of an alias needs. It goes
   by the constructors of values, as an equality has to, and so has a
   case to gain with each new form of value. [lists_eqb] is [forall2b]
   of FlambdaCompareSpec.v, written so that a recursive definition may
   go through it (see [every] in FlambdaMapSpec.v). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaCompareSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec.
Import ListNotations.

(* ======================================================================== *)
(* Equality tests of the machine                                            *)
(* ======================================================================== *)

(* Two lists are the same, element by element. *)
Definition lists_eqb {A : Type} (eqb : A -> A -> bool)
    : list A -> list A -> bool :=
  fix lists_eqb (l l' : list A) : bool :=
    match l, l' with
    | [], [] => true
    | a :: l, a' :: l' => eqb a a' && lists_eqb l l'
    | [], _ :: _ | _ :: _, [] => false
    end.

(* Two values are the same value. *)
Fixpoint value_eqb (v v' : value) : bool :=
  match v, v' with
  | Value_const c, Value_const c' => const_eqb c c'
  | Value_closure code value_slots, Value_closure code' value_slots' =>
    Pos.eqb code code'
    && lists_eqb
         (fun slot slot' =>
            Pos.eqb (fst slot) (fst slot')
            && value_eqb (snd slot) (snd slot'))
         value_slots value_slots'
  | Value_closure_of_set slot function_decls value_slots,
    Value_closure_of_set slot' function_decls' value_slots' =>
    Pos.eqb slot slot'
    && lists_eqb
         (fun decl decl' =>
            Pos.eqb (fst decl) (fst decl') && Pos.eqb (snd decl) (snd decl'))
         function_decls function_decls'
    && lists_eqb
         (fun slot slot' =>
            Pos.eqb (fst slot) (fst slot')
            && value_eqb (snd slot) (snd slot'))
         value_slots value_slots'
  | Value_symbol s, Value_symbol s' => Pos.eqb s s'
  | Value_block tag fields, Value_block tag' fields' =>
    Z.eqb tag tag' && lists_eqb value_eqb fields fields'
  | Value_immutable_array elements, Value_immutable_array elements' =>
    lists_eqb value_eqb elements elements'
  | Value_boxed_number c, Value_boxed_number c' => const_eqb c c'
  | Value_location l, Value_location l' => Pos.eqb l l'
  | Value_partial_application callee args args_arity,
    Value_partial_application callee' args' args_arity' =>
    value_eqb callee callee'
    && lists_eqb value_eqb args args'
    && arity_eqb args_arity args_arity'
  | _, _ => false
  end.

Definition value_eqb_decides_equality_statement : Prop :=
  decides_equality value_eqb.

(* ======================================================================== *)
(* The relations, the shapes and the indexes                                *)
(* ======================================================================== *)

(* [is_poison]. *)
Definition is_poison_b (v : value) : bool :=
  match v with
  | Value_const (Poison _) => true
  | _ => false
  end.

(* [is_tag]. *)
Definition is_tag_b (n : Z) : bool := Z.leb 0 n && Z.leb n 255.

(* [relation_holds]. *)
Definition relation_holds_b (statics : static_env) (relation : Relation.t)
    (v : value) (n : Z) : bool :=
  match relation with
  | Relation.Is_null =>
    match v with
    | Value_const Null => Z.eqb n 1
    | _ => kind_eqb (kind_of_value v) Value && Z.eqb n 0
    end
  | Relation.Is_int =>
    match v with
    | Value_const (Tagged_immediate _) => Z.eqb n 1
    | _ =>
      match block_of statics v with
      | Some _ => Z.eqb n 0
      | None => false
      end
    end
  | Relation.Get_tag =>
    match v with
    | Value_const (Tagged_immediate _) => true
    | _ =>
      match block_of statics v with
      | Some (tag, _) => Z.eqb tag n && is_tag_b n
      | None => false
      end
    end
  end.

Definition relation_holds_b_iff_statement : Prop :=
  ∀ (statics : static_env) (relation : Relation.t) (v : value) (n : Z),
    relation_holds_b statics relation v n = true
    <-> relation_holds statics relation v n.

(* [block_has_shape]. *)
Definition block_has_shape_b (shape : block_shape) (tag : Z)
    (fields : list value) : bool :=
  match shape with
  | Scannable _ => Z.ltb tag no_scan_tag
  | Float_record => Z.eqb tag double_array_tag
  end
  && lists_eqb kind_eqb (map kind_of_value fields)
       (field_kinds shape (length fields)).

(* [block_index_satisfied]. *)
Definition block_index_satisfied_b (tag : Z) (fields : list value)
    (index : row_like_index nat block_shape) : bool :=
  match index_domain index with
  | Row_like_index_domain.Known size => Nat.eqb (length fields) size
  | Row_like_index_domain.At_least size => Nat.leb size (length fields)
  end
  && block_has_shape_b (index_shape index) tag fields.

(* [closures_index_satisfied]. *)
Definition closures_index_satisfied_b
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value))
    (index : row_like_index set_of_closures_contents unit) : bool :=
  match index_domain index with
  | Row_like_index_domain.Known contents =>
    Keyed.same_keys_b Pos.eqb (contents_closures contents) (domain functions)
    && Keyed.same_keys_b Pos.eqb (contents_value_slots contents)
         (domain value_slots)
  | Row_like_index_domain.At_least contents =>
    all_among (contents_closures contents) (domain functions)
    && all_among (contents_value_slots contents) (domain value_slots)
  end.

(* ======================================================================== *)
(* The meaning of a type, as a test                                         *)
(* ======================================================================== *)

Section Type_satisfied_b.

Variable statics : static_env.
Variable vars : env.

(* [variable_is]. *)
Definition variable_is_b (x : option variable) (n : Z) : bool :=
  match x with
  | Some x =>
    match lookup x vars with
    | Some (Value_const (Naked_immediate i)) => Z.eqb i n
    | _ => false
    end
  | None => true
  end.

(* [descr_satisfied]. *)
Definition descr_satisfied_b {head : Type} (k : flambda_kind)
    (head_satisfied_b : head -> bool) (v : value) (ty : type_descr head)
    : bool :=
  match ty with
  | Unknown => kind_eqb (kind_of_value v) k
  | Bottom => false
  | Ok (No_alias head) => head_satisfied_b head
  | Ok (Equals s) =>
    match eval_simple vars s with
    | Some w => value_eqb w v && kind_eqb (kind_of_value v) k
    | None => false
    end
  end.

(* [head_of_kind_naked_immediate_satisfied]. *)
Definition head_of_kind_naked_immediate_satisfied_b (v : value)
    (head : head_of_kind_naked_immediate) : bool :=
  match v with
  | Value_const (Naked_immediate i) =>
    match naked_immediates head with
    | Or_unknown.Unknown => true
    | Or_unknown.Known numbers => Keyed.among Z.eqb i numbers
    end
    && every_b
         (fun relation =>
            match value_of_name vars (snd relation) with
            | Some w => relation_holds_b statics (fst relation) w i
            | None => false
            end)
         (inverse_relations head)
  | _ => false
  end.

(* [naked_numbers_satisfied]. *)
Definition naked_numbers_satisfied_b (k : naked_number_kind) (v : value)
    (numbers : list Z) : bool :=
  match v with
  | Value_const c =>
    kind_eqb (kind_of_const c) (Naked_number k)
    && match number_of_const c with
       | Some n => Keyed.among Z.eqb n numbers
       | None => false
       end
  | _ => false
  end.

(* [fields_satisfied]. *)
Definition fields_satisfied_b
    (type_satisfied_b : value -> flambda_type -> bool)
    : list flambda_type -> list value -> bool :=
  fix fields_satisfied_b (tys : list flambda_type) (fields : list value)
      : bool :=
    match tys, fields with
    | ty :: tys, field :: fields =>
      type_satisfied_b field ty && fields_satisfied_b tys fields
    | _, _ => true
    end.

(* [block_case_satisfied]. *)
Definition block_case_satisfied_b
    (type_satisfied_b : value -> flambda_type -> bool)
    (env_extension_satisfied_b : env_extension -> bool) (tag : Z)
    (fields : list value) (case : row_like_block_case) : bool :=
  block_index_satisfied_b tag fields (case_index case)
  && fields_satisfied_b type_satisfied_b (case_maps_to case) fields
  && env_extension_satisfied_b (case_env_extension case).

(* [closures_case_satisfied]. *)
Definition closures_case_satisfied_b
    (closures_entry_satisfied_b : closures_entry -> bool)
    (env_extension_satisfied_b : env_extension -> bool)
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value)) (case : row_like_closures_case)
    : bool :=
  closures_index_satisfied_b functions value_slots (case_index case)
  && closures_entry_satisfied_b (case_maps_to case)
  && env_extension_satisfied_b (case_env_extension case).

(* [type_satisfied] and the six functions defined with it. *)
Fixpoint type_satisfied_b (v : value) (t : flambda_type) {struct t} : bool :=
  match t with
  | Type_value ty =>
    descr_satisfied_b Value (head_of_kind_value_satisfied_b v) v ty
  | Type_naked_immediate ty =>
    descr_satisfied_b (Naked_number Kind_naked_immediate)
      (head_of_kind_naked_immediate_satisfied_b v) v ty
  | Type_naked_number k ty =>
    descr_satisfied_b (Naked_number k) (naked_numbers_satisfied_b k v) v ty
  | Type_rec_info ty =>
    descr_satisfied_b Rec_info (fun _ => kind_eqb (kind_of_value v) Rec_info)
      v ty
  | Type_region ty =>
    descr_satisfied_b Region (fun _ => kind_eqb (kind_of_value v) Region) v
      ty
  end

with head_of_kind_value_satisfied_b (v : value) (head : head_of_kind_value)
    {struct head} : bool :=
  match head with
  | Head_of_kind_value non_null is_null =>
    match v with
    | Value_const Null =>
      match is_null with
      | Not_null => false
      | Maybe_null is_null => variable_is_b is_null 1
      end
    | _ =>
      match non_null with
      | Unknown => kind_eqb (kind_of_value v) Value
      | Bottom => false
      | Ok head => head_of_kind_value_non_null_satisfied_b v head
      end
      && match is_null with
         | Not_null => true
         | Maybe_null is_null => variable_is_b is_null 0
         end
    end
  end

with head_of_kind_value_non_null_satisfied_b (v : value)
    (head : head_of_kind_value_non_null) {struct head} : bool :=
  match head with
  | Variant is_int immediates get_tag blocks extensions _ =>
    match v with
    | Value_const (Tagged_immediate i) =>
      variable_is_b is_int 1
      && match immediates with
         | Or_unknown.Unknown => true
         | Or_unknown.Known ty =>
           type_satisfied_b (Value_const (Naked_immediate i)) ty
         end
      && match extensions with
         | No_extensions => true
         | Ext when_immediate _ => env_extension_satisfied_b when_immediate
         end
    | _ =>
      match block_of statics v with
      | Some (tag, fields) =>
        variable_is_b is_int 0
        && variable_is_b get_tag tag
        && match blocks with
           | Or_unknown.Unknown => true
           | Or_unknown.Known blocks =>
             row_like_for_blocks_satisfied_b tag fields blocks
           end
        && match extensions with
           | No_extensions => true
           | Ext _ when_block => env_extension_satisfied_b when_block
           end
      | None => false
      end
    end
  | Mutable_block _ => false
  | Boxed_number k contents _ =>
    match boxed_number_of statics v with
    | Some c =>
      kind_eqb (kind_of_const c) (unboxed_kind k)
      && type_satisfied_b (Value_const c) contents
    | None => false
    end
  | Closures by_function_slot _ =>
    match closure_in_set statics v with
    | Some (slot, functions, value_slots) =>
      row_like_for_closures_satisfied_b v slot functions value_slots
        by_function_slot
    | None => false
    end
  | String strs =>
    match string_of statics v with
    | Some bytes => Keyed.among (lists_eqb Z.eqb) bytes strs
    | None => false
    end
  end

with row_like_for_blocks_satisfied_b (tag : Z) (fields : list value)
    (blocks : row_like_for_blocks) {struct blocks} : bool :=
  match blocks with
  | Row_like_for_blocks known_tags other_tags _ =>
    Keyed.at_binding_b Z.eqb
      (fun case =>
         match case with
         | Or_unknown.Unknown => true
         | Or_unknown.Known case =>
           block_case_satisfied_b type_satisfied_b env_extension_satisfied_b
             tag fields case
         end)
      tag known_tags
    && match Keyed.lookup Z.eqb tag known_tags with
       | Some _ => true
       | None =>
         match other_tags with
         | Or_bottom.Bottom => false
         | Or_bottom.Ok case =>
           block_case_satisfied_b type_satisfied_b env_extension_satisfied_b
             tag fields case
         end
       end
  end

with row_like_for_closures_satisfied_b (v : value) (slot : function_slot)
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value))
    (closures : row_like_for_closures) {struct closures} : bool :=
  match closures with
  | Row_like_for_closures known_closures =>
    is_bound slot known_closures
    && at_binding_b
         (closures_case_satisfied_b
            (closures_entry_satisfied_b v slot functions value_slots)
            env_extension_satisfied_b functions value_slots)
         slot known_closures
  end

with closures_entry_satisfied_b (v : value) (slot : function_slot)
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value)) (entry : closures_entry)
    {struct entry} : bool :=
  match entry with
  | Closures_entry function_types closure_types value_slot_types =>
    for_all_bindings_b
      (fun g function_type =>
         match function_type, lookup g functions with
         | Or_unknown.Known function_type, Some code =>
           Pos.eqb code (function_type_code_id function_type)
         | Or_unknown.Unknown, _ | _, None => true
         end)
      function_types
    && for_all_bindings_b
         (fun g ty =>
            match project_function_slot statics slot g v with
            | Some closure => type_satisfied_b closure ty
            | None => true
            end)
         closure_types
    && for_all_bindings_b
         (fun w ty =>
            match lookup w value_slots with
            | Some contents => type_satisfied_b contents ty
            | None => true
            end)
         value_slot_types
  end

with env_extension_satisfied_b (extension : env_extension) {struct extension}
    : bool :=
  match extension with
  | Env_extension equations =>
    Keyed.for_all_bindings_b name_eqb
      (fun n ty =>
         match value_of_name vars n with
         | Some v => type_satisfied_b v ty
         | None => false
         end)
      equations
  end.

End Type_satisfied_b.

(* [row_like_block_case_satisfied] and
   [row_like_closures_case_satisfied]. *)
Definition row_like_block_case_satisfied_b (statics : static_env)
    (vars : env) (tag : Z) (fields : list value) (case : row_like_block_case)
    : bool :=
  block_case_satisfied_b (type_satisfied_b statics vars)
    (env_extension_satisfied_b statics vars) tag fields case.

Definition row_like_closures_case_satisfied_b (statics : static_env)
    (vars : env) (v : value) (slot : function_slot)
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value)) (case : row_like_closures_case)
    : bool :=
  closures_case_satisfied_b
    (closures_entry_satisfied_b statics vars v slot functions value_slots)
    (env_extension_satisfied_b statics vars) functions value_slots case.

(* ======================================================================== *)
(* Each test answers [true] exactly when the meaning holds                  *)
(* ======================================================================== *)

Definition type_satisfied_b_iff_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (t : flambda_type),
    type_satisfied_b statics vars v t = true
    <-> type_satisfied statics vars v t.

Definition head_of_kind_value_satisfied_b_iff_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value)
    (head : head_of_kind_value),
    head_of_kind_value_satisfied_b statics vars v head = true
    <-> head_of_kind_value_satisfied statics vars v head.

Definition head_of_kind_value_non_null_satisfied_b_iff_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value)
    (head : head_of_kind_value_non_null),
    head_of_kind_value_non_null_satisfied_b statics vars v head = true
    <-> head_of_kind_value_non_null_satisfied statics vars v head.

Definition row_like_for_blocks_satisfied_b_iff_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (tag : Z) (fields : list value)
    (blocks : row_like_for_blocks),
    row_like_for_blocks_satisfied_b statics vars tag fields blocks = true
    <-> row_like_for_blocks_satisfied statics vars tag fields blocks.

Definition row_like_for_closures_satisfied_b_iff_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (slot : function_slot)
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value))
    (closures : row_like_for_closures),
    row_like_for_closures_satisfied_b statics vars v slot functions
      value_slots closures
    = true
    <-> row_like_for_closures_satisfied statics vars v slot functions
          value_slots closures.

Definition closures_entry_satisfied_b_iff_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (slot : function_slot)
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value)) (entry : closures_entry),
    closures_entry_satisfied_b statics vars v slot functions value_slots
      entry
    = true
    <-> closures_entry_satisfied statics vars v slot functions value_slots
          entry.

Definition env_extension_satisfied_b_iff_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (extension : env_extension),
    env_extension_satisfied_b statics vars extension = true
    <-> env_extension_satisfied statics vars extension.

Definition row_like_block_case_satisfied_b_iff_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (tag : Z) (fields : list value)
    (case : row_like_block_case),
    row_like_block_case_satisfied_b statics vars tag fields case = true
    <-> row_like_block_case_satisfied statics vars tag fields case.

Definition row_like_closures_case_satisfied_b_iff_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (slot : function_slot)
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value)) (case : row_like_closures_case),
    row_like_closures_case_satisfied_b statics vars v slot functions
      value_slots case
    = true
    <-> row_like_closures_case_satisfied statics vars v slot functions
          value_slots case.
