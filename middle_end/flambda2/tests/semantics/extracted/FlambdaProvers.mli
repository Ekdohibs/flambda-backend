open BinInt
open BinNums
open BinPos
open Datatypes
open FlambdaBindingTime
open FlambdaCompareSpec
open FlambdaExpandHead
open FlambdaKindModel
open FlambdaMapSpec
open FlambdaModel
open FlambdaTypeAccessSpec
open FlambdaTypeEqualSpec
open FlambdaTypeGrammar
open FlambdaTypeMeaningBoolSpec
open FlambdaTypingEnv
open List
open PeanoNat

type descr_oub =
  | Oub_value of head_of_kind_value or_unknown_or_bottom
  | Oub_naked_immediate of head_of_kind_naked_immediate or_unknown_or_bottom
  | Oub_naked_number of naked_number_kind * coq_Z list or_unknown_or_bottom
  | Oub_rec_info of unit or_unknown_or_bottom
  | Oub_region of unit or_unknown_or_bottom

val descr_oub_of_expanded : Expanded_type.t -> descr_oub

module Meet_shortcut : sig
  type 'a t =
    | Known_result of 'a
    | Need_meet
    | Invalid
end

module Proof_of_property : sig
  type 'a t =
    | Proved of 'a
    | Unknown
end

module Generic_proof : sig
  type 'a t =
    | Proved of 'a
    | Unknown
    | Invalid
end

module Boxed_or_tagged_number : sig
  type t =
    | Boxed of Alloc_mode.t * boxable_number * flambda_type
    | Tagged_immediate
end

type variant_like_proof =
  { const_ctors : coq_Z list Or_unknown.t;
    non_const_ctors_with_sizes : (coq_Z, nat * block_shape) Keyed.finite_map
  }

module Provers : sig
  val is_bottom : Typing_env.t -> flambda_type -> bool option

  val expand_head : Typing_env.t -> flambda_type -> descr_oub option

  val wrong_kind : 'a1 -> 'a1

  val as_meet_shortcut : 'a1 Generic_proof.t -> 'a1 Meet_shortcut.t

  val as_property : 'a1 Generic_proof.t -> 'a1 Proof_of_property.t

  val gen_value_to_gen :
    (Typing_env.t -> head_of_kind_value_non_null -> 'a1 Generic_proof.t option) ->
    Typing_env.t ->
    flambda_type ->
    'a1 Generic_proof.t option

  val gen_value_to_proof :
    (Typing_env.t -> head_of_kind_value_non_null -> 'a1 Generic_proof.t option) ->
    Typing_env.t ->
    flambda_type ->
    'a1 Proof_of_property.t option

  val gen_value_to_meet :
    (Typing_env.t -> head_of_kind_value_non_null -> 'a1 Generic_proof.t option) ->
    Typing_env.t ->
    flambda_type ->
    'a1 Meet_shortcut.t option

  val prove_is_int_generic_value :
    bool ->
    Typing_env.t ->
    head_of_kind_value_non_null ->
    bool Generic_proof.t option

  val prove_is_int_generic :
    bool -> Typing_env.t -> flambda_type -> bool Generic_proof.t option

  val prove_is_int :
    Typing_env.t -> flambda_type -> bool Proof_of_property.t option

  val meet_is_int_variant_only :
    Typing_env.t -> flambda_type -> bool Meet_shortcut.t option

  val prove_is_not_a_pointer_generic_value :
    Typing_env.t -> flambda_type -> bool Generic_proof.t option

  val prove_is_not_a_pointer :
    Typing_env.t -> flambda_type -> bool Proof_of_property.t option

  val prove_get_tag_generic_value :
    Typing_env.t ->
    head_of_kind_value_non_null ->
    coq_Z list Generic_proof.t option

  val prove_get_tag_generic :
    Typing_env.t -> flambda_type -> coq_Z list Generic_proof.t option

  val prove_get_tag :
    Typing_env.t -> flambda_type -> coq_Z list Proof_of_property.t option

  val prove_is_null_generic :
    Typing_env.t -> flambda_type -> bool Generic_proof.t option

  val meet_is_null : Typing_env.t -> flambda_type -> bool Meet_shortcut.t option

  val these_immediates : coq_Z list -> coq_Z list Generic_proof.t

  val meet_these_immediates :
    coq_Z list -> coq_Z list Generic_proof.t -> coq_Z list Generic_proof.t

  val meet_this_immediate :
    coq_Z -> coq_Z list Generic_proof.t -> coq_Z list Generic_proof.t

  val prove_naked_immediates_relation :
    Typing_env.t ->
    Relation.t ->
    name ->
    coq_Z list Generic_proof.t ->
    coq_Z list Generic_proof.t option

  val prove_naked_immediates_generic :
    Typing_env.t -> flambda_type -> coq_Z list Generic_proof.t option

  val meet_naked_immediates :
    Typing_env.t -> flambda_type -> coq_Z list Meet_shortcut.t option

  val prove_equals_tagged_immediates_value :
    Typing_env.t ->
    head_of_kind_value_non_null ->
    coq_Z list Generic_proof.t option

  val prove_equals_tagged_immediates :
    Typing_env.t -> flambda_type -> coq_Z list Proof_of_property.t option

  val meet_equals_tagged_immediates_value :
    Typing_env.t ->
    head_of_kind_value_non_null ->
    coq_Z list Generic_proof.t option

  val meet_equals_tagged_immediates :
    Typing_env.t -> flambda_type -> coq_Z list Meet_shortcut.t option

  val meet_equals_single_tagged_immediate :
    Typing_env.t -> flambda_type -> coq_Z Meet_shortcut.t option

  val meet_naked_number :
    naked_number_kind ->
    Typing_env.t ->
    flambda_type ->
    coq_Z list Meet_shortcut.t option

  val meet_naked_float32s :
    Typing_env.t -> flambda_type -> coq_Z list Meet_shortcut.t option

  val meet_naked_floats :
    Typing_env.t -> flambda_type -> coq_Z list Meet_shortcut.t option

  val meet_naked_int8s :
    Typing_env.t -> flambda_type -> coq_Z list Meet_shortcut.t option

  val meet_naked_int16s :
    Typing_env.t -> flambda_type -> coq_Z list Meet_shortcut.t option

  val meet_naked_int32s :
    Typing_env.t -> flambda_type -> coq_Z list Meet_shortcut.t option

  val meet_naked_int64s :
    Typing_env.t -> flambda_type -> coq_Z list Meet_shortcut.t option

  val meet_naked_nativeints :
    Typing_env.t -> flambda_type -> coq_Z list Meet_shortcut.t option

  val meet_naked_vec128s :
    Typing_env.t -> flambda_type -> coq_Z list Meet_shortcut.t option

  val meet_naked_vec256s :
    Typing_env.t -> flambda_type -> coq_Z list Meet_shortcut.t option

  val meet_naked_vec512s :
    Typing_env.t -> flambda_type -> coq_Z list Meet_shortcut.t option

  val meet_naked_masks :
    Typing_env.t -> flambda_type -> coq_Z list Meet_shortcut.t option

  val canonical_simple_of_alias :
    Typing_env.t -> Name_mode.t -> flambda_type -> simple option option

  val prove_equals_to_simple_of_kind :
    Typing_env.t ->
    flambda_type ->
    flambda_kind ->
    simple Proof_of_property.t option

  val prove_is_a_boxed_or_tagged_number_value :
    Typing_env.t ->
    head_of_kind_value_non_null ->
    Boxed_or_tagged_number.t Generic_proof.t option

  val prove_nothing :
    Typing_env.t -> flambda_type -> 'a1 Proof_of_property.t option

  val prove_is_a_boxed_or_tagged_number :
    Typing_env.t ->
    flambda_type ->
    Boxed_or_tagged_number.t Proof_of_property.t option

  val prove_is_a_tagged_immediate :
    Typing_env.t -> flambda_type -> unit Proof_of_property.t option

  val prove_is_a_boxed_number :
    boxable_number ->
    Typing_env.t ->
    flambda_type ->
    unit Proof_of_property.t option

  val prove_is_a_boxed_float32 :
    Typing_env.t -> flambda_type -> unit Proof_of_property.t option

  val prove_is_a_boxed_float :
    Typing_env.t -> flambda_type -> unit Proof_of_property.t option

  val prove_is_a_boxed_int32 :
    Typing_env.t -> flambda_type -> unit Proof_of_property.t option

  val prove_is_a_boxed_int64 :
    Typing_env.t -> flambda_type -> unit Proof_of_property.t option

  val prove_is_a_boxed_nativeint :
    Typing_env.t -> flambda_type -> unit Proof_of_property.t option

  val prove_is_a_boxed_vec128 :
    Typing_env.t -> flambda_type -> unit Proof_of_property.t option

  val prove_is_a_boxed_vec256 :
    Typing_env.t -> flambda_type -> unit Proof_of_property.t option

  val prove_is_a_boxed_vec512 :
    Typing_env.t -> flambda_type -> unit Proof_of_property.t option

  val prove_is_a_boxed_mask :
    Typing_env.t -> flambda_type -> unit Proof_of_property.t option

  val prove_is_or_is_not_a_boxed_float_value :
    Typing_env.t -> head_of_kind_value_non_null -> bool Generic_proof.t option

  val prove_is_or_is_not_a_boxed_float :
    Typing_env.t -> flambda_type -> bool Proof_of_property.t option

  val prove_strings_value :
    Typing_env.t ->
    head_of_kind_value_non_null ->
    coq_Z list list Generic_proof.t option

  val meet_strings :
    Typing_env.t -> flambda_type -> coq_Z list list Meet_shortcut.t option

  val prove_strings :
    Typing_env.t -> flambda_type -> coq_Z list list Proof_of_property.t option

  type tagging_proof_kind =
    | Prove
    | Meet

  val inspect_tagging_immediates :
    Name_mode.t ->
    Typing_env.t ->
    flambda_type Or_unknown.t ->
    simple Generic_proof.t option

  val inspect_tagging_of_simple_value :
    tagging_proof_kind ->
    Name_mode.t ->
    Typing_env.t ->
    head_of_kind_value_non_null ->
    simple Generic_proof.t option

  val prove_tagging_of_simple :
    Typing_env.t ->
    Name_mode.t ->
    flambda_type ->
    simple Proof_of_property.t option

  val meet_tagging_of_simple :
    Typing_env.t -> Name_mode.t -> flambda_type -> simple Meet_shortcut.t option

  val meet_boxed_number_containing_simple :
    boxable_number ->
    Typing_env.t ->
    Name_mode.t ->
    flambda_type ->
    simple Meet_shortcut.t option

  val meet_boxed_float32_containing_simple :
    Typing_env.t -> Name_mode.t -> flambda_type -> simple Meet_shortcut.t option

  val meet_boxed_float_containing_simple :
    Typing_env.t -> Name_mode.t -> flambda_type -> simple Meet_shortcut.t option

  val meet_boxed_int32_containing_simple :
    Typing_env.t -> Name_mode.t -> flambda_type -> simple Meet_shortcut.t option

  val meet_boxed_int64_containing_simple :
    Typing_env.t -> Name_mode.t -> flambda_type -> simple Meet_shortcut.t option

  val meet_boxed_nativeint_containing_simple :
    Typing_env.t -> Name_mode.t -> flambda_type -> simple Meet_shortcut.t option

  val meet_boxed_vec128_containing_simple :
    Typing_env.t -> Name_mode.t -> flambda_type -> simple Meet_shortcut.t option

  val meet_boxed_vec256_containing_simple :
    Typing_env.t -> Name_mode.t -> flambda_type -> simple Meet_shortcut.t option

  val meet_boxed_vec512_containing_simple :
    Typing_env.t -> Name_mode.t -> flambda_type -> simple Meet_shortcut.t option

  val meet_boxed_mask_containing_simple :
    Typing_env.t -> Name_mode.t -> flambda_type -> simple Meet_shortcut.t option

  val tag_is_scannable : coq_Z -> bool

  val prove_variant_like_generic_value :
    Typing_env.t ->
    head_of_kind_value_non_null ->
    variant_like_proof Generic_proof.t option

  val meet_variant_like :
    Typing_env.t -> flambda_type -> variant_like_proof Meet_shortcut.t option

  val prove_variant_like :
    Typing_env.t ->
    flambda_type ->
    variant_like_proof Proof_of_property.t option

  val prove_unique_tag_and_size_value :
    Typing_env.t ->
    head_of_kind_value_non_null ->
    ((((coq_Z * block_shape) * nat) * flambda_type list) * Alloc_mode.t)
    Generic_proof.t
    option

  val prove_unique_tag_and_size :
    Typing_env.t ->
    flambda_type ->
    ((coq_Z * block_shape) * nat) Proof_of_property.t option

  val field_aliases_rev :
    simple list Proof_of_property.t ->
    flambda_type ->
    simple list Proof_of_property.t

  val prove_unique_fully_constructed_immutable_heap_block :
    Typing_env.t ->
    flambda_type ->
    (((coq_Z * block_shape) * nat) * simple list) Proof_of_property.t option

  val meet_block_field_simple_value :
    Name_mode.t ->
    flambda_kind ->
    nat ->
    Typing_env.t ->
    head_of_kind_value_non_null ->
    simple Generic_proof.t option

  val meet_block_field_simple :
    Typing_env.t ->
    Name_mode.t ->
    flambda_kind ->
    flambda_type ->
    nat ->
    simple Meet_shortcut.t option

  val single_closures_entry_of_singleton :
    Alloc_mode.t ->
    bool ->
    function_slot ->
    closures_entry ->
    (((function_slot * Alloc_mode.t) * closures_entry) * function_type)
    Generic_proof.t

  val prove_single_closures_entry_generic_value :
    Typing_env.t ->
    head_of_kind_value_non_null ->
    (((function_slot * Alloc_mode.t) * closures_entry) * function_type)
    Generic_proof.t
    option

  val meet_single_closures_entry :
    Typing_env.t ->
    flambda_type ->
    (((function_slot * Alloc_mode.t) * closures_entry) * function_type)
    Meet_shortcut.t
    option

  val prove_single_closures_entry :
    Typing_env.t ->
    flambda_type ->
    (((function_slot * Alloc_mode.t) * closures_entry) * function_type)
    Proof_of_property.t
    option

  val add_code_id_of_case :
    code_id list Or_unknown.t ->
    function_slot * row_like_closures_case ->
    code_id list Or_unknown.t

  val prove_code_ids_generic_value :
    Typing_env.t ->
    head_of_kind_value_non_null ->
    code_id list Generic_proof.t option

  val meet_code_ids :
    Typing_env.t -> flambda_type -> code_id list Meet_shortcut.t option

  val prove_code_ids :
    Typing_env.t -> flambda_type -> code_id list Proof_of_property.t option

  val meet_project_function_slot_simple_value :
    Name_mode.t ->
    function_slot ->
    Typing_env.t ->
    head_of_kind_value_non_null ->
    simple Generic_proof.t option

  val meet_project_function_slot_simple :
    Typing_env.t ->
    Name_mode.t ->
    flambda_type ->
    function_slot ->
    simple Meet_shortcut.t option

  val meet_project_value_slot_simple_value :
    Name_mode.t ->
    value_slot ->
    flambda_kind ->
    Typing_env.t ->
    head_of_kind_value_non_null ->
    simple Generic_proof.t option

  val meet_project_value_slot_simple :
    Typing_env.t ->
    Name_mode.t ->
    flambda_type ->
    value_slot ->
    flambda_kind ->
    simple Meet_shortcut.t option

  val incompatible_naked_numbers :
    Typing_env.t -> flambda_type -> flambda_type -> bool option

  val physical_equality_of_immediates :
    Typing_env.t -> flambda_type -> flambda_type -> bool Generic_proof.t option

  val physical_equality_of_blocks :
    row_like_for_blocks -> row_like_for_blocks -> bool Generic_proof.t

  val no_immediate_and_no_block :
    flambda_type Or_unknown.t -> row_like_for_blocks Or_unknown.t -> bool

  val physical_equality_of_parts :
    bool Generic_proof.t -> bool Generic_proof.t -> bool Proof_of_property.t

  val physical_equality_of_variants :
    Typing_env.t ->
    flambda_type Or_unknown.t ->
    flambda_type Or_unknown.t ->
    row_like_for_blocks Or_unknown.t ->
    row_like_for_blocks Or_unknown.t ->
    bool Proof_of_property.t option

  val physical_equality_of_heads :
    Typing_env.t ->
    head_of_kind_value_non_null ->
    head_of_kind_value_non_null ->
    bool Proof_of_property.t option

  val physical_equality_check_heads :
    Typing_env.t ->
    flambda_type ->
    flambda_type ->
    bool Proof_of_property.t option

  val prove_physical_equality :
    Typing_env.t ->
    flambda_type ->
    flambda_type ->
    bool Proof_of_property.t option

  val prove_alloc_mode_of_boxed_number_value :
    Typing_env.t ->
    head_of_kind_value_non_null ->
    Alloc_mode.t Generic_proof.t option

  val prove_alloc_mode_of_boxed_number :
    Typing_env.t -> flambda_type -> Alloc_mode.t Proof_of_property.t option

  val never_holds_locally_allocated_values :
    Typing_env.t -> variable -> unit Proof_of_property.t option
end
