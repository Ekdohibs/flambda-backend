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


(* What the other creation functions make.

   This file is to be reviewed: the statements. The functions are in
   the last section of FlambdaTypeGrammar.v ("More creation
   functions"), to be reviewed there against more_type_creators.ml and
   type_grammar.ml.

   These are the types that Simplify gives to the results of
   primitives ([these_naked_numbers], [any_naked_bool],
   [this_boxed_number], ...), to parameters and to the results of
   calls ([unknown_types_from_arity]), and to what a primitive takes
   apart: unboxing a number, untagging an immediate, loading a field
   and projecting a slot each say of their argument that it is the
   boxed number, the tagged immediate, the block or the closure whose
   part the result is ([boxed_number_alias_to],
   [tagged_immediate_alias_to], [immutable_block_with_size_at_least],
   [closure_with_at_least_these_function_slots],
   [closure_with_at_least_this_value_slot]).

   For each function there is what the type that it makes means: an
   equivalence with a condition on the value in which that type does
   not appear. The types that the function is given still do (those
   of the fields given to [row_like_for_blocks_create]), and the
   statement of [unknown_from_shape] has one direction. Then,
   for all of them: that the type is well formed
   (FlambdaTypeWellFormedSpec.v), what it is under a renaming of
   variables (FlambdaTypeRenameSpec.v), and what its aliases are, with
   their kinds (FlambdaTypeGrammar.v).

   A variable that such a type is an alias to holds a value under the
   environment: the meaning of an alias asks it. So a statement here
   with [lookup x vars = Some w] says that [x] is bound.

   The proofs are in FlambdaTypeCreatorsProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec FlambdaNominalSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypeWellFormedSpec FlambdaTypeRenameSpec.
Import ListNotations.

(* ======================================================================== *)
(* Unknown types, kinds and aliases                                         *)
(* ======================================================================== *)

(* The unknown type of the kind of a type, and the bottom one. *)
Definition unknown_like_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (t : flambda_type),
    (type_satisfied statics vars v (unknown_like t)
     <-> kind_of_value v = kind t)
    ∧ ~ type_satisfied statics vars v (bottom_like t).

(* A field of a block of a shape has the type that
   [unknown_from_shape] gives its index. *)
Definition unknown_from_shape_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (shape : block_shape) (tag : Z)
    (fields : list value) (index : nat) (field : value) (t : flambda_type),
    block_has_shape shape tag fields ->
    nth_error fields index = Some field ->
    unknown_from_shape shape index = Some t ->
    type_satisfied statics vars field t.

(* The type of a kind with a subkind that says nothing has the values
   of the kind, as the unknown type has. *)
Definition unknown_with_subkind_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (k : flambda_kind),
    type_satisfied statics vars v (unknown_with_subkind k)
    <-> kind_of_value v = k.

(* Values have the types that an arity gives when they are of its
   kinds, one for one. *)
Definition unknown_types_from_arity_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (vs : list value) (a : arity),
    Forall2 (type_satisfied statics vars) vs (unknown_types_from_arity a)
    <-> map kind_of_value vs = unarize a.

Definition kind_for_const_of_const_statement : Prop :=
  ∀ c : const, kind_for_const c = kind_of_const c.

Definition is_alias_of_name_iff_statement : Prop :=
  ∀ (t : flambda_type) (n : name),
    is_alias_of_name t n = true <-> get_alias_opt t = Some (simple_of_name n).

(* ======================================================================== *)
(* Numbers                                                                  *)
(* ======================================================================== *)

(* The numbers of a kind that are in a set. *)
Definition these_naked_numbers_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (k : naked_number_kind)
    (numbers : list Z),
    type_satisfied statics vars v (these_naked_numbers k numbers)
    <-> ∃ (c : const) (n : Z),
          v = Value_const c
          ∧ kind_of_const c = Naked_number k
          ∧ number_of_const c = Some n
          ∧ n ∈ numbers.

Definition any_bool_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value),
    (type_satisfied statics vars v any_tagged_bool
     <-> v = Value_const (Tagged_immediate 0)
         ∨ v = Value_const (Tagged_immediate 1))
    ∧ (type_satisfied statics vars v any_naked_bool
       <-> v = Value_const (Naked_immediate 0)
           ∨ v = Value_const (Naked_immediate 1)).

Definition any_tagged_immediate_or_null_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value),
    type_satisfied statics vars v any_tagged_immediate_or_null
    <-> is_immediate v ∨ v = Value_const Null.

(* The tagged immediate whose number the variable holds: what
   untagging says of its argument, the variable being its result. *)
Definition tagged_immediate_alias_to_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (x : variable),
    type_satisfied statics vars v (tagged_immediate_alias_to x)
    <-> ∃ i : Z,
          v = Value_const (Tagged_immediate i)
          ∧ lookup x vars = Some (Value_const (Naked_immediate i)).

(* ======================================================================== *)
(* Boxed numbers                                                            *)
(* ======================================================================== *)

Definition unboxed_kind_naked_number_statement : Prop :=
  ∀ k : boxable_number, unboxed_kind k = Naked_number (unboxed_number_kind k).

(* The boxed number whose contents the variable holds: what unboxing
   says of its argument, the variable being its result. *)
Definition boxed_number_alias_to_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (k : boxable_number)
    (x : variable) (alloc_mode : Alloc_mode.t),
    type_satisfied statics vars v (boxed_number_alias_to k x alloc_mode)
    <-> ∃ c : const,
          boxed_number_of statics v = Some c
          ∧ kind_of_const c = unboxed_kind k
          ∧ lookup x vars = Some (Value_const c).

Definition this_boxed_number_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (k : boxable_number)
    (i : Z) (alloc_mode : Alloc_mode.t),
    type_satisfied statics vars v (this_boxed_number k i alloc_mode)
    <-> boxed_number_of statics v
        = Some (const_of_naked_number (unboxed_number_kind k) i).

Definition these_boxed_numbers_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (k : boxable_number)
    (numbers : list Z) (alloc_mode : Alloc_mode.t),
    type_satisfied statics vars v (these_boxed_numbers k numbers alloc_mode)
    <-> ∃ (c : const) (n : Z),
          boxed_number_of statics v = Some c
          ∧ kind_of_const c = unboxed_kind k
          ∧ number_of_const c = Some n
          ∧ n ∈ numbers.

(* Any boxed number of a kind; its contents may be poison. *)
Definition any_boxed_number_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (k : boxable_number),
    type_satisfied statics vars v (any_boxed_number k)
    <-> ∃ c : const,
          boxed_number_of statics v = Some c
          ∧ kind_of_const c = unboxed_kind k.

(* ======================================================================== *)
(* Blocks                                                                   *)
(* ======================================================================== *)

(* [Row_like_for_blocks.create] answers but for a tag, given or known,
   that the shape cannot have. *)
Definition row_like_for_blocks_create_defined_statement : Prop :=
  ∀ (shape : block_shape) (field_tys : list flambda_type)
    (open_or_closed : Open_or_closed.t) (alloc_mode : Alloc_mode.t),
    row_like_for_blocks_create shape field_tys open_or_closed alloc_mode
    = None
    <-> ∃ tag : Z,
          (open_or_closed = Open_or_closed.Closed tag
           ∨ open_or_closed = Open_or_closed.Open (Or_unknown.Known tag))
          ∧ match shape with
            | Scannable _ => (no_scan_tag <= tag)%Z
            | Float_record => tag <> double_array_tag
            end.

(* The blocks it answers: of the tag, if one is given; of the shape;
   with as many fields as there are types, or at least as many; whose
   fields have the types. *)
Definition row_like_for_blocks_create_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (tag : Z) (fields : list value)
    (shape : block_shape) (field_tys : list flambda_type)
    (open_or_closed : Open_or_closed.t) (alloc_mode : Alloc_mode.t)
    (blocks : row_like_for_blocks),
    row_like_for_blocks_create shape field_tys open_or_closed alloc_mode
    = Some blocks ->
    (row_like_for_blocks_satisfied statics vars tag fields blocks
     <-> match open_or_closed with
         | Open_or_closed.Closed tag' =>
           tag = tag' ∧ length fields = length field_tys
         | Open_or_closed.Open (Or_unknown.Known tag') =>
           tag = tag' ∧ length field_tys <= length fields
         | Open_or_closed.Open Or_unknown.Unknown =>
           length field_tys <= length fields
         end
         ∧ block_has_shape shape tag fields
         ∧ fields_satisfied (type_satisfied statics vars) field_tys fields).

(* [immutable_block] is made of what [create] answers for a closed
   tag, where it answers. *)
Definition immutable_block_create_statement : Prop :=
  ∀ (is_unique : bool) (tag : Z) (shape : block_shape)
    (alloc_mode : Alloc_mode.t) (fields : list flambda_type)
    (blocks : row_like_for_blocks),
    row_like_for_blocks_create shape fields (Open_or_closed.Closed tag)
      alloc_mode
    = Some blocks ->
    immutable_block is_unique tag shape alloc_mode fields
    = create_variant is_unique
        (Or_unknown.Known (bottom (Naked_number Kind_naked_immediate)))
        (Or_unknown.Known blocks) No_extensions.

(* The blocks of at least [n] fields, of the tag if it is known and of
   the shape, of which the variable holds the field before the [n]th:
   what a load of that field says of its argument, the variable being
   its result. *)
Definition immutable_block_with_size_at_least_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (tag : Or_unknown.t Z)
    (n : nat) (shape : block_shape) (x : variable) (t : flambda_type),
    immutable_block_with_size_at_least tag n shape x = Some t ->
    (type_satisfied statics vars v t
     <-> ∃ (tag' : Z) (fields : list value),
           block_of statics v = Some (tag', fields)
           ∧ match tag with
             | Or_unknown.Known tag => tag' = tag
             | Or_unknown.Unknown => True
             end
           ∧ n <= length fields
           ∧ block_has_shape shape tag' fields
           ∧ match n with
             | O => True
             | S index =>
               ∃ field : value,
                 nth_error fields index = Some field
                 ∧ lookup x vars = Some field
             end).

(* ======================================================================== *)
(* Closures                                                                 *)
(* ======================================================================== *)

(* The closure at a slot of a set that has the function slots given,
   each of which, projected from the closure, is what its simple
   evaluates to: what a projection of a function slot says of its
   argument. *)
Definition closure_with_these_function_slots_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (slot : function_slot)
    (bindings : finite_map simple),
    type_satisfied statics vars v
      (closure_with_at_least_these_function_slots slot bindings)
    <-> ∃ (functions : list (function_slot * code_id))
          (value_slots : list (value_slot * value)),
          closure_in_set statics v = Some (slot, functions, value_slots)
          ∧ all_among (domain bindings) (domain functions) = true
          ∧ ∀ (g : function_slot) (s : simple) (closure : value),
              lookup g bindings = Some s ->
              project_function_slot statics slot g v = Some closure ->
              eval_simple vars s = Some closure.

(* A closure at a slot whose function has the code given. If a symbol
   is given, what is projected from the closure at its own slot,
   where the projection has a value, is the symbol. The statement
   holds of every static environment. The projection has a value
   where what the symbols are bound to is valid
   ([well_formed_closures_case_closure_type_statement], in
   FlambdaTypeWellFormedSpec.v), and the symbol is the closure itself
   where the static sets are bound
   (FlambdaClosuresSpec.static_sets_bound). *)
Definition static_closure_with_this_code_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (slot : function_slot)
    (closure_symbol : option symbol) (code : code_id),
    type_satisfied statics vars v
      (static_closure_with_this_code slot closure_symbol code)
    <-> ∃ (functions : list (function_slot * code_id))
          (value_slots : list (value_slot * value)),
          closure_in_set statics v = Some (slot, functions, value_slots)
          ∧ lookup slot functions = Some code
          ∧ match closure_symbol with
            | Some closure_symbol =>
              ∀ closure : value,
                project_function_slot statics slot slot v = Some closure ->
                closure = Value_symbol closure_symbol
            | None => True
            end.

(* The closure at a slot of a set that has the value slots given, each
   of which holds what its variable holds, of the kind given: what a
   projection of a value slot says of its argument. *)
Definition closure_with_these_value_slots_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (slot : function_slot)
    (slots : finite_map (variable * flambda_kind)),
    type_satisfied statics vars v
      (closure_with_at_least_these_value_slots slot slots)
    <-> ∃ (functions : list (function_slot * code_id))
          (value_slots : list (value_slot * value)),
          closure_in_set statics v = Some (slot, functions, value_slots)
          ∧ all_among (domain slots) (domain value_slots) = true
          ∧ ∀ (w : value_slot) (x : variable) (k : flambda_kind)
              (contents : value),
              lookup w slots = Some (x, k) ->
              lookup w value_slots = Some contents ->
              lookup x vars = Some contents ∧ kind_of_value contents = k.

Definition closure_with_this_value_slot_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (slot : function_slot)
    (w : value_slot) (x : variable) (k : flambda_kind),
    type_satisfied statics vars v
      (closure_with_at_least_this_value_slot slot w x k)
    <-> ∃ (functions : list (function_slot * code_id))
          (value_slots : list (value_slot * value)) (contents : value),
          closure_in_set statics v = Some (slot, functions, value_slots)
          ∧ lookup w value_slots = Some contents
          ∧ lookup x vars = Some contents
          ∧ kind_of_value contents = k.

(* ======================================================================== *)
(* They are well formed                                                     *)
(* ======================================================================== *)

(* Those that take no map and no shape give a well-formed type,
   whatever they are given. *)
Definition more_creators_well_formed_statement : Prop :=
  (∀ t : flambda_type,
    type_well_formed (unknown_like t) = true
    ∧ type_well_formed (bottom_like t) = true)
  ∧ (∀ (shape : block_shape) (index : nat) (t : flambda_type),
      unknown_from_shape shape index = Some t -> type_well_formed t = true)
  ∧ (∀ k : flambda_kind, type_well_formed (unknown_with_subkind k) = true)
  ∧ (∀ (a : arity) (t : flambda_type),
      t ∈ unknown_types_from_arity a -> type_well_formed t = true)
  ∧ (∀ (k : naked_number_kind) (numbers : list Z),
      type_well_formed (these_naked_numbers k numbers) = true)
  ∧ type_well_formed any_tagged_bool = true
  ∧ type_well_formed any_naked_bool = true
  ∧ type_well_formed any_tagged_immediate_or_null = true
  ∧ (∀ x : variable, type_well_formed (tagged_immediate_alias_to x) = true)
  ∧ (∀ (k : boxable_number) (x : variable) (alloc_mode : Alloc_mode.t),
      type_well_formed (boxed_number_alias_to k x alloc_mode) = true)
  ∧ (∀ (k : boxable_number) (i : Z) (alloc_mode : Alloc_mode.t),
      type_well_formed (this_boxed_number k i alloc_mode) = true)
  ∧ (∀ (k : boxable_number) (numbers : list Z) (alloc_mode : Alloc_mode.t),
      type_well_formed (these_boxed_numbers k numbers alloc_mode) = true)
  ∧ (∀ k : boxable_number, type_well_formed (any_boxed_number k) = true)
  ∧ (∀ (slot : function_slot) (closure_symbol : option symbol)
       (code : code_id),
      type_well_formed
        (static_closure_with_this_code slot closure_symbol code)
      = true)
  ∧ (∀ (slot : function_slot) (w : value_slot) (x : variable)
       (k : flambda_kind),
      type_well_formed (closure_with_at_least_this_value_slot slot w x k)
      = true).

(* Blocks: the caller owes a shape written in the one way, and
   well-formed types. *)
Definition row_like_for_blocks_create_checked_statement : Prop :=
  ∀ (shape : block_shape) (field_tys : list flambda_type)
    (open_or_closed : Open_or_closed.t) (alloc_mode : Alloc_mode.t)
    (blocks : row_like_for_blocks),
    row_like_for_blocks_create shape field_tys open_or_closed alloc_mode
    = Some blocks ->
    (row_like_for_blocks_well_formed blocks = true
     <-> block_shape_canonical shape
         ∧ ∀ ty : flambda_type,
             ty ∈ field_tys -> type_well_formed ty = true).

Definition immutable_block_with_size_at_least_well_formed_statement : Prop :=
  ∀ (tag : Or_unknown.t Z) (n : nat) (shape : block_shape) (x : variable)
    (t : flambda_type),
    immutable_block_with_size_at_least tag n shape x = Some t ->
    (type_well_formed t = true <-> block_shape_canonical shape).

(* Closures: the caller owes a map that binds a key once, as a map of
   the compiler does. *)
Definition closure_with_at_least_well_formed_statement : Prop :=
  (∀ (slot : function_slot) (bindings : finite_map simple),
    type_well_formed
      (closure_with_at_least_these_function_slots slot bindings)
    = true
    <-> distinct (keys bindings) = true)
  ∧ (∀ (slot : function_slot)
       (slots : finite_map (variable * flambda_kind)),
      type_well_formed (closure_with_at_least_these_value_slots slot slots)
      = true
      <-> distinct (keys slots) = true).

(* ======================================================================== *)
(* Under a renaming                                                         *)
(* ======================================================================== *)

(* What is made of no name is left as it is. *)
Definition more_creators_closed_statement : Prop :=
  ∀ r : renaming,
    (∀ t : flambda_type,
      rename_type r (unknown_like t) = unknown_like t
      ∧ rename_type r (bottom_like t) = bottom_like t
      ∧ unknown_like (rename_type r t) = unknown_like t
      ∧ bottom_like (rename_type r t) = bottom_like t)
    ∧ (∀ (shape : block_shape) (index : nat) (t : flambda_type),
        unknown_from_shape shape index = Some t -> rename_type r t = t)
    ∧ (∀ k : flambda_kind,
        rename_type r (unknown_with_subkind k) = unknown_with_subkind k)
    ∧ (∀ a : arity,
        map (rename_type r) (unknown_types_from_arity a)
        = unknown_types_from_arity a)
    ∧ (∀ (k : naked_number_kind) (numbers : list Z),
        rename_type r (these_naked_numbers k numbers)
        = these_naked_numbers k numbers)
    ∧ rename_type r any_tagged_bool = any_tagged_bool
    ∧ rename_type r any_naked_bool = any_naked_bool
    ∧ rename_type r any_tagged_immediate_or_null
      = any_tagged_immediate_or_null
    ∧ (∀ (k : boxable_number) (i : Z) (alloc_mode : Alloc_mode.t),
        rename_type r (this_boxed_number k i alloc_mode)
        = this_boxed_number k i alloc_mode)
    ∧ (∀ (k : boxable_number) (numbers : list Z)
         (alloc_mode : Alloc_mode.t),
        rename_type r (these_boxed_numbers k numbers alloc_mode)
        = these_boxed_numbers k numbers alloc_mode)
    ∧ (∀ k : boxable_number,
        rename_type r (any_boxed_number k) = any_boxed_number k)
    ∧ (∀ (slot : function_slot) (closure_symbol : option symbol)
         (code : code_id),
        rename_type r (static_closure_with_this_code slot closure_symbol code)
        = static_closure_with_this_code slot closure_symbol code).

(* What is made of a name, a simple or types is made of them, renamed. *)
Definition more_creators_rename_statement : Prop :=
  ∀ r : renaming,
    (∀ (t : flambda_type) (n : name),
      is_alias_of_name (rename_type r t) (rename_name r n)
      = is_alias_of_name t n)
    ∧ (∀ x : variable,
        rename_type r (tagged_immediate_alias_to x)
        = tagged_immediate_alias_to (rename_variable r x))
    ∧ (∀ (k : boxable_number) (x : variable) (alloc_mode : Alloc_mode.t),
        rename_type r (boxed_number_alias_to k x alloc_mode)
        = boxed_number_alias_to k (rename_variable r x) alloc_mode)
    ∧ (∀ (shape : block_shape) (field_tys : list flambda_type)
         (open_or_closed : Open_or_closed.t) (alloc_mode : Alloc_mode.t),
        option_map (rename_row_like_for_blocks r)
          (row_like_for_blocks_create shape field_tys open_or_closed
             alloc_mode)
        = row_like_for_blocks_create shape (map (rename_type r) field_tys)
            open_or_closed alloc_mode)
    ∧ (∀ (tag : Or_unknown.t Z) (n : nat) (shape : block_shape)
         (x : variable),
        option_map (rename_type r)
          (immutable_block_with_size_at_least tag n shape x)
        = immutable_block_with_size_at_least tag n shape
            (rename_variable r x))
    ∧ (∀ (slot : function_slot) (bindings : finite_map simple),
        rename_type r
          (closure_with_at_least_these_function_slots slot bindings)
        = closure_with_at_least_these_function_slots slot
            (map_values (rename_simple r) bindings))
    ∧ (∀ (slot : function_slot)
         (slots : finite_map (variable * flambda_kind)),
        rename_type r (closure_with_at_least_these_value_slots slot slots)
        = closure_with_at_least_these_value_slots slot
            (map_values
               (fun value_slot =>
                  (rename_variable r (fst value_slot), snd value_slot))
               slots))
    ∧ (∀ (slot : function_slot) (w : value_slot) (x : variable)
         (k : flambda_kind),
        rename_type r (closure_with_at_least_this_value_slot slot w x k)
        = closure_with_at_least_this_value_slot slot w (rename_variable r x)
            k).

(* ======================================================================== *)
(* Their aliases, with their kinds                                          *)
(* ======================================================================== *)

(* The types that are made of no name and no simple have no alias but
   to a constant, which those of numbers have; these have one alias
   each, to their variable at the kind of what it holds. *)
Definition more_creators_aliases_statement : Prop :=
  (∀ x : variable,
    aliases_with_kinds (tagged_immediate_alias_to x)
    = [(Var x, Naked_number Kind_naked_immediate)])
  ∧ (∀ (k : boxable_number) (x : variable) (alloc_mode : Alloc_mode.t),
      aliases_with_kinds (boxed_number_alias_to k x alloc_mode)
      = [(Var x, unboxed_kind k)])
  ∧ (∀ (tag : Or_unknown.t Z) (index : nat) (shape : block_shape)
       (x : variable) (t : flambda_type),
      immutable_block_with_size_at_least tag (S index) shape x = Some t ->
      ∃ k : flambda_kind,
        block_shape_element_kind shape index = Some k
        ∧ aliases_with_kinds t = [(Var x, k)])
  ∧ (∀ (slot : function_slot) (bindings : finite_map simple),
      aliases_with_kinds
        (closure_with_at_least_these_function_slots slot bindings)
      = map (fun binding => (snd binding, Value)) bindings)
  ∧ (∀ (slot : function_slot)
       (slots : finite_map (variable * flambda_kind)),
      aliases_with_kinds
        (closure_with_at_least_these_value_slots slot slots)
      = map
          (fun binding => (Var (fst (snd binding)), snd (snd binding)))
          slots).
