open BinInt
open BinPos
open Bool
open Datatypes
open FlambdaCompareSpec
open FlambdaKindModel
open FlambdaMapSpec
open FlambdaModel
open FlambdaTypeGrammar
open FlambdaTypeMeaningBoolSpec
open PeanoNat

val map_eqb :
  ('a1 -> 'a1 -> bool) ->
  ('a2 -> 'a2 -> bool) ->
  ('a1, 'a2) Keyed.finite_map ->
  ('a1, 'a2) Keyed.finite_map ->
  bool

val simple_eqb : simple -> simple -> bool

val option_variable_eqb : variable option -> variable option -> bool

val alloc_mode_eqb : Alloc_mode.t -> Alloc_mode.t -> bool

val relation_eqb : Relation.t -> Relation.t -> bool

val naked_number_kind_eqb : naked_number_kind -> naked_number_kind -> bool

val is_null_eqb : is_null -> is_null -> bool

val block_shape_eqb : block_shape -> block_shape -> bool

val function_type_eqb : function_type -> function_type -> bool

val block_index_eqb :
  (nat, block_shape) row_like_index -> (nat, block_shape) row_like_index -> bool

val closures_index_eqb :
  (set_of_closures_contents, unit) row_like_index ->
  (set_of_closures_contents, unit) row_like_index ->
  bool

val or_unknown_eqb :
  ('a1 -> 'a1 -> bool) -> 'a1 Or_unknown.t -> 'a1 Or_unknown.t -> bool

val or_bottom_eqb :
  ('a1 -> 'a1 -> bool) -> 'a1 Or_bottom.t -> 'a1 Or_bottom.t -> bool

val or_unknown_or_bottom_eqb :
  ('a1 -> 'a1 -> bool) ->
  'a1 or_unknown_or_bottom ->
  'a1 or_unknown_or_bottom ->
  bool

val descr_eqb : ('a1 -> 'a1 -> bool) -> 'a1 type_descr -> 'a1 type_descr -> bool

val head_of_kind_naked_immediate_eqb :
  head_of_kind_naked_immediate -> head_of_kind_naked_immediate -> bool

val block_case_eqb :
  (flambda_type -> flambda_type -> bool) ->
  (env_extension -> env_extension -> bool) ->
  row_like_block_case ->
  row_like_block_case ->
  bool

val closures_case_eqb :
  (closures_entry -> closures_entry -> bool) ->
  (env_extension -> env_extension -> bool) ->
  row_like_closures_case ->
  row_like_closures_case ->
  bool

val variant_extensions_eqb :
  (env_extension -> env_extension -> bool) ->
  variant_extensions ->
  variant_extensions ->
  bool

val type_eqb : flambda_type -> flambda_type -> bool

val head_of_kind_value_eqb : head_of_kind_value -> head_of_kind_value -> bool

val head_of_kind_value_non_null_eqb :
  head_of_kind_value_non_null -> head_of_kind_value_non_null -> bool

val row_like_for_blocks_eqb : row_like_for_blocks -> row_like_for_blocks -> bool

val row_like_for_closures_eqb :
  row_like_for_closures -> row_like_for_closures -> bool

val closures_entry_eqb : closures_entry -> closures_entry -> bool

val env_extension_eqb : env_extension -> env_extension -> bool
