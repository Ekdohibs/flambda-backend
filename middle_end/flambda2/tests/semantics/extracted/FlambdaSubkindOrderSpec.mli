open BinInt
open BinNums
open Datatypes
open FlambdaKindModel
open List

val shape_of_tag : coq_Z -> constructor_shapes -> field_subkinds option option

val subkind_says_at_least : subkind -> subkind -> bool

val shapes_say_at_least :
  coq_Z list -> constructor_shapes -> constructor_shapes -> bool

val fields_say_at_least : field_subkinds -> field_subkinds -> bool

val param_says_at_least : param_subkind -> param_subkind -> bool

val params_say_at_least : param_subkind list -> param_subkind list -> bool
