open Datatypes
open FlambdaDemoteSpec
open FlambdaLiftSpec
open FlambdaLoopifySpec
open FlambdaModel
open FlambdaUnusedStaticsSpec
open FlambdaValidateKnowingSpec
open FlambdaValidateReadingSpec
open FlambdaValidateSpec

val after_demoted : expr -> expr

val validate_unit_demoted :
  nat ->
  dropped ->
  value_slot list ->
  loopified ->
  lifting list ->
  code_id list ->
  expr ->
  expr ->
  validation

val validate_unit_knowing_demoted :
  nat ->
  dropped ->
  value_slot list ->
  loopified ->
  lifting list ->
  code_id list ->
  expr ->
  expr ->
  validation

val validate_unit_reading_demoted :
  nat ->
  dropped ->
  value_slot list ->
  loopified ->
  lifting list ->
  code_id list ->
  expr ->
  expr ->
  validation

val validate_unit_knowing_reading_demoted :
  nat ->
  dropped ->
  value_slot list ->
  loopified ->
  lifting list ->
  code_id list ->
  expr ->
  expr ->
  validation
