open BinInt
open BinNums
open BinPos
open Datatypes
open FlambdaMapSpec
open FlambdaModel
open FlambdaTypeGrammar
open List
open PeanoNat

val row_like_for_blocks_all_tags :
  row_like_for_blocks -> coq_Z list Or_unknown.t

val block_case_size_and_shape :
  row_like_block_case Or_unknown.t -> (nat * block_shape) option

val row_like_for_blocks_all_tags_and_sizes :
  row_like_for_blocks ->
  (coq_Z, nat * block_shape) Keyed.finite_map Or_unknown.t

val row_like_for_blocks_get_singleton :
  row_like_for_blocks ->
  ((((coq_Z * block_shape) * nat) * flambda_type list) * Alloc_mode.t) option

val row_like_for_blocks_get_field :
  row_like_for_blocks -> nat -> flambda_type or_unknown_or_bottom

val closures_entry_find_function_type :
  closures_entry -> bool -> function_slot -> function_type or_unknown_or_bottom

type get_single_tag_result =
  | No_singleton
  | Exact_closure of function_slot * closures_entry
  | Incomplete_closure of function_slot * closures_entry

val row_like_for_closures_get_single_tag :
  row_like_for_closures -> get_single_tag_result option

val row_like_for_closures_get_closure :
  row_like_for_closures -> function_slot -> flambda_type Or_unknown.t option

val row_like_for_closures_get_env_var :
  row_like_for_closures -> value_slot -> flambda_type Or_unknown.t option
