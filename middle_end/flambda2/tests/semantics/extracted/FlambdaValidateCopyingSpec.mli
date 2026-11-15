open Datatypes
open FlambdaCopyLetsSpec
open FlambdaLiftSpec
open FlambdaLoopifySpec
open FlambdaModel
open FlambdaNeverReturnsSpec
open FlambdaStubCallsSpec
open FlambdaUnusedStaticsSpec
open FlambdaValidateImportsSpec
open FlambdaValidateKnowingSpec
open FlambdaValidateSpec
open FlambdaValueSpec
open ListDef

val lifted_steps :
  nat -> continuation list -> lifting list -> code_id list -> expr -> expr

val copied_avoiding :
  loopified -> lifting list -> lifting -> lifting list -> continuation list

val before_copied :
  nat ->
  loopified ->
  lifting list ->
  lifting ->
  let_copy list ->
  lifting list ->
  code_id list ->
  expr ->
  expr

val validate_unit_copying :
  nat ->
  dropped ->
  value_slot list ->
  loopified ->
  lifting list ->
  lifting ->
  let_copy list ->
  lifting list ->
  code_id list ->
  expr ->
  expr ->
  validation

val validate_unit_knowing_copying :
  nat ->
  dropped ->
  value_slot list ->
  loopified ->
  lifting list ->
  lifting ->
  let_copy list ->
  lifting list ->
  code_id list ->
  expr ->
  expr ->
  validation

val lifted_steps_free_in :
  static_env ->
  nat ->
  continuation list ->
  lifting list ->
  code_id list ->
  expr ->
  bool

val before_copied_holds_from :
  static_env ->
  nat ->
  loopified ->
  lifting list ->
  lifting ->
  lifting list ->
  code_id list ->
  expr ->
  bool
