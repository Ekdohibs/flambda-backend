open BinInt
open BinNums
open BinPos
open Datatypes
open FlambdaGiveScrutineeSpec
open FlambdaKindModel
open FlambdaMapSpec
open FlambdaModel
open FlambdaNormaliseSpec
open FlambdaSemanticsSpec
open FlambdaUnusedContsSpec
open List
open ListDef
open PeanoNat

val give_to : continuation -> variable list -> apply_cont -> apply_cont

val give : continuation -> variable list -> expr -> expr

val give_in_handler :
  continuation -> variable list -> continuation_handler -> continuation_handler

val applies_only : continuation -> bool -> apply_cont -> bool

val only_applied : continuation -> variable list -> bool -> expr -> bool

val handler_only_applied :
  continuation -> variable list -> bool -> continuation_handler -> bool

val read_params :
  variable list -> flambda_kind list -> expr -> (variable * flambda_kind) list

val give_all : (continuation * variable list) list -> expr -> expr

val lift_spine :
  expr ->
  variable list ->
  flambda_kind list ->
  (continuation * variable list) list ->
  expr ->
  ((continuation * continuation_handler) list * expr) option

val position : variable -> variable list -> nat option

val tested : variable list -> expr -> nat option

val gives_immediate : continuation -> nat -> apply_cont -> bool

val all_give_immediate : continuation -> nat -> expr -> bool

val handler_all_give_immediate :
  continuation -> nat -> continuation_handler -> bool

val tested_through : expr -> variable option

val arm_through : continuation -> expr -> coq_Z option

val facts_in :
  continuation -> expr -> (variable * coq_Z) list -> (variable * coq_Z) list

val forgetting :
  variable list -> (variable * coq_Z) list -> (variable * coq_Z) list

val in_arm :
  simple -> coq_Z -> (variable * coq_Z) list -> (variable * coq_Z) list

val keep_known :
  continuation ->
  nat ->
  (variable * coq_Z) list ->
  apply_cont ->
  variable list ->
  variable list

val still_known :
  continuation ->
  nat ->
  (variable * coq_Z) list ->
  expr ->
  variable list ->
  variable list

val handler_still_known :
  continuation ->
  nat ->
  (variable * coq_Z) list ->
  continuation_handler ->
  variable list ->
  variable list

val scrutinees : expr -> variable list

val handler_scrutinees : continuation_handler -> variable list

val number_of_a_scrutinee : continuation -> nat -> expr -> bool

val only_read_given : variable list -> simple list -> apply_cont -> bool

val gives_what_is_read :
  continuation ->
  nat ->
  variable list ->
  (coq_Z * apply_cont) list ->
  apply_cont ->
  bool

val all_give_read :
  continuation ->
  nat ->
  variable list ->
  (coq_Z * apply_cont) list ->
  expr ->
  bool

val handler_all_give_read :
  continuation ->
  nat ->
  variable list ->
  (coq_Z * apply_cont) list ->
  continuation_handler ->
  bool

val read_where_given :
  continuation -> nat -> variable list -> expr -> expr -> bool

val bind_lifted : (continuation * continuation_handler) list -> expr -> expr

val lift_conts : continuation -> continuation_handler -> expr -> expr option

val lift_handler_conts : expr -> expr
