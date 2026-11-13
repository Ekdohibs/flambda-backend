open Datatypes
open FlambdaArrayLoadSpec
open FlambdaCompareSpec
open FlambdaGuardScopeSpec
open FlambdaLiftSpec
open FlambdaLoopifySpec
open FlambdaModel
open FlambdaNeverReturnsSpec
open FlambdaNormaliseSpec
open FlambdaRecHandlersSpec
open FlambdaSimplifyCarriedSpec
open FlambdaStubCallsSpec
open FlambdaUnusedStaticsSpec
open FlambdaValidateKnowingSpec
open FlambdaValidateSpec
open ListDef

val read_arrays_in_code : expr -> expr

val read_arrays : expr -> expr

val normalise_before_avoiding :
  continuation list -> nat -> code_id list -> expr -> expr

val reading_round :
  continuation list -> nat -> loopified -> code_id list -> expr -> expr

val normalise_reading :
  nat -> continuation list -> loopified -> code_id list -> expr -> expr

val before_read :
  nat -> loopified -> lifting list -> code_id list -> expr -> expr

val validate_unit_reading :
  nat ->
  dropped ->
  value_slot list ->
  loopified ->
  lifting list ->
  code_id list ->
  expr ->
  expr ->
  validation

val validate_unit_knowing_reading :
  nat ->
  dropped ->
  value_slot list ->
  loopified ->
  lifting list ->
  code_id list ->
  expr ->
  expr ->
  validation
