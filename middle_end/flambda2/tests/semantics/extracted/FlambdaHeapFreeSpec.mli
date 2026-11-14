open Datatypes
open FlambdaHeapSpec
open FlambdaModel
open List

val expr_free_of_heap : expr -> bool

val handler_free_of_heap : continuation_handler -> bool

val binding_free_of_heap : static_binding -> bool

val code_free_of_heap : code -> bool
