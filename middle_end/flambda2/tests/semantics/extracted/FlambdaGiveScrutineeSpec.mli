open BinInt
open BinNums
open BinPos
open Datatypes
open FlambdaMapSpec
open FlambdaModel
open FlambdaStatedScopeSpec
open FlambdaUnaliasParamsSpec
open FlambdaUnusedContsSpec
open List
open ListDef

val first_arm_of : continuation -> (coq_Z * apply_cont) list -> coq_Z option

val only_arm : continuation -> coq_Z -> (coq_Z * apply_cont) list -> bool

val arm_fact : variable -> continuation -> expr -> coq_Z option

val arm_known : variable -> simple -> coq_Z -> coq_Z option -> coq_Z option

val known_under : variable -> variable list -> coq_Z option -> coq_Z option

val give_nth : variable -> coq_Z -> nat -> simple list -> simple list

val give_in_apply_cont :
  variable -> continuation -> nat -> coq_Z option -> apply_cont -> apply_cont

val give_scrutinee :
  variable -> continuation -> nat -> coq_Z option -> expr -> expr

val give_in_handler :
  variable ->
  continuation ->
  nat ->
  coq_Z option ->
  continuation_handler ->
  continuation_handler

val scrutinee_by :
  continuation -> nat -> variable list -> apply_cont -> variable list option

val scrutinee_var : variable list -> simple -> variable list

val scrutinee_of : variable list -> expr -> variable list

val first_scrutinee :
  continuation -> nat -> variable list -> expr -> variable list option

val handler_first_scrutinee :
  continuation ->
  nat ->
  variable list ->
  continuation_handler ->
  variable list option

val give_first :
  stated_in_scope ->
  variable list ->
  nat ->
  continuation ->
  continuation_handler ->
  expr ->
  expr

val give_at :
  stated_in_scope -> nat -> continuation -> continuation_handler -> expr -> expr

val give_from :
  stated_in_scope -> nat -> continuation -> continuation_handler -> expr -> expr

val give_scrutinees_under : stated_in_scope -> expr -> expr
