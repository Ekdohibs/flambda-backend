open Datatypes
open FlambdaInlineCallsAtSpec
open FlambdaLiftSpec
open FlambdaLoopifySpec
open FlambdaModel
open FlambdaRenameApartSpec
open FlambdaUnusedStaticsSpec
open FlambdaValidateKnowingSpec
open FlambdaValidateSpec
open ListDef

type inlining_choice = path list * calls_in_code

val inlined_in_turn : continuation list -> inlining_choice list -> expr -> expr

val before_inlined :
  inlining_choice list -> loopified -> lifting list -> expr -> expr

val validate_unit_inlining :
  inlining_choice list ->
  nat ->
  dropped ->
  value_slot list ->
  loopified ->
  lifting list ->
  code_id list ->
  expr ->
  expr ->
  validation

val validate_unit_knowing_inlining :
  inlining_choice list ->
  nat ->
  dropped ->
  value_slot list ->
  loopified ->
  lifting list ->
  code_id list ->
  expr ->
  expr ->
  validation
