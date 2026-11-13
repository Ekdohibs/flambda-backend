open Datatypes
open FlambdaBlockLoadSpec
open FlambdaKindModel
open FlambdaModel
open FlambdaNormaliseSpec
open FlambdaPrimitiveModel
open FlambdaPrimitiveSpec
open List

val surely_value : simple -> bool

val surely_immediate : simple -> bool

val of_known_block_surely : unary_primitive -> known_block -> simple option

val load_known_named_surely : known_blocks -> named -> named

val load_blocks_knowing_surely : known_blocks -> expr -> expr

val load_blocks_in_handler_surely :
  known_blocks -> continuation_handler -> continuation_handler

val load_known_blocks_surely : expr -> expr

val allocation_of_constants : named -> bool

val remove_constant_allocations : expr -> expr
