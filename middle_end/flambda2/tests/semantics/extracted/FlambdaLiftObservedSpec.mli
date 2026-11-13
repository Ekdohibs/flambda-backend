open FlambdaCopyCodeSpec
open FlambdaLiftAroundSpec
open FlambdaLiftFromArmsSpec
open FlambdaLiftFromCodeSpec
open FlambdaLiftSpec
open FlambdaModel
open FlambdaMyClosureSpec
open FlambdaUnusedHeadStaticsSpec
open FlambdaWellFormedSpec

val lift_to_head_unit_observed :
  unit_interface -> (path * lift_to) list -> expr -> expr option

val lift_from_code_unit_observed :
  unit_interface -> lets_in_code -> expr -> expr option

val lift_unit_observed : lifting -> expr -> expr option

val lift_around_unit_observed :
  unit_interface ->
  (path * lift_to) list ->
  (path * path list) list ->
  expr ->
  expr option

val to_head_observed : lifting -> expr -> expr

val out_of_code_observed : lifting -> expr -> expr

val lets_observed : lifting -> expr -> expr

val around_observed : lifting -> expr -> expr

val liftings_observed : lifting -> expr -> expr

val with_lifting_observed : lifting -> expr -> expr
