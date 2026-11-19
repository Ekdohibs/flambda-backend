open Datatypes
open FlambdaHeapSpec
open FlambdaModel
open List

val expr_free_of_heap : expr -> bool

val handler_free_of_heap : continuation_handler -> bool

val binding_free_of_heap : static_binding -> bool

val code_free_of_heap : code -> bool

val calls_nothing : expr -> bool

val handler_calls_nothing : continuation_handler -> bool

val expr_free_of_heap_outside_code : expr -> bool

val handler_free_of_heap_outside_code : continuation_handler -> bool

val free_of_heap_fine : expr -> bool
