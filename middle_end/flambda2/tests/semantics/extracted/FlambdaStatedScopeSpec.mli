open BinInt
open BinNums
open Datatypes
open FlambdaKindModel
open FlambdaMapSpec
open FlambdaModel
open FlambdaPrimitiveModel
open FlambdaPrimitiveSpec
open FlambdaSubkindBinderSpec
open FlambdaSubkindOrderSpec
open FlambdaSubkindSpec
open FlambdaValueSpec
open List

type stated_in_scope = (variable * param_subkind) list

val stated_of_var : stated_in_scope -> variable -> param_subkind

val not_null : param_subkind

val tagged : param_subkind

val stated_of_simple : stated_in_scope -> simple -> param_subkind

val fields_stated : stated_in_scope -> simple list -> field_subkinds

val stated_of_field : param_subkind -> coq_Z -> param_subkind

val stated_of_named : stated_in_scope -> named -> param_subkind

val params_stated :
  variable list -> param_subkind list -> stated_in_scope -> stated_in_scope

val is_stated : stated_in_scope -> param_subkind -> simple -> bool

val are_stated : stated_in_scope -> simple list -> param_subkind list -> bool

val code_stated : variable list -> variable -> code_metadata -> stated_in_scope

val is_stated_at :
  stated_in_scope -> param_subkind list -> nat -> simple -> bool
