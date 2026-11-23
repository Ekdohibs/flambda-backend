open Datatypes
open FlambdaBindersSpec
open FlambdaCompareOrderSpec
open FlambdaFloatCodeSpec
open FlambdaImportsSpec
open FlambdaKnownSlotsSpec
open FlambdaLiftObservedSpec
open FlambdaLiftSpec
open FlambdaLoopifySpec
open FlambdaModel
open FlambdaNeverReturnsSpec
open FlambdaStaticsOrderSpec
open FlambdaStubCallsSpec
open FlambdaUnusedStaticsSpec
open FlambdaValidateImportsSpec
open FlambdaValidateSpec
open FlambdaValueSpec

val lifted_in_turn_observed :
  nat -> continuation list -> lifting list -> code_id list -> expr -> expr

val lifting_observed_free_in : static_env -> lifting -> expr -> bool

val lifted_observed_free_in :
  static_env ->
  nat ->
  continuation list ->
  lifting list ->
  code_id list ->
  expr ->
  bool

type related_validation =
  | Related_once_lifted
  | Not_related

val validate_unit_related :
  (symbol * import_class) list ->
  nat ->
  dropped ->
  value_slot list ->
  loopified ->
  lifting list ->
  code_id list ->
  expr ->
  expr ->
  related_validation
