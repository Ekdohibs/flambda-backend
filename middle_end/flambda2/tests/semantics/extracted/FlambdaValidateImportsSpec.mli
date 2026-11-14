open Datatypes
open FlambdaCompareSpec
open FlambdaFloatCodeSpec
open FlambdaGuardScopeSpec
open FlambdaHoistStaticsSpec
open FlambdaImportsSpec
open FlambdaKnownClosuresSpec
open FlambdaKnownSlotsSpec
open FlambdaKnownSymbolsSpec
open FlambdaLiftAroundSpec
open FlambdaLiftFromArmsSpec
open FlambdaLiftFromCodeSpec
open FlambdaLiftSpec
open FlambdaLoopifySpec
open FlambdaModel
open FlambdaNeverReturnsSpec
open FlambdaSharedReadsSpec
open FlambdaSimplifyCarriedSpec
open FlambdaStaticsOrderSpec
open FlambdaStubCallsSpec
open FlambdaUnusedHeadStaticsSpec
open FlambdaUnusedStaticsSpec
open FlambdaValidateKnowingSpec
open FlambdaValidateSpec
open FlambdaValueSpec
open ListDef

val holds_from : static_env -> nat -> code_id list -> expr -> validation -> bool

val made_free_in : static_env -> expr option -> bool

val tables_free_in : static_env -> lifting -> expr -> bool

val lifting_free_in : static_env -> lifting -> expr -> bool

val known_free_in :
  static_env -> nat -> continuation list -> code_id list -> expr -> bool

val known_calling_free_in :
  static_env -> bool -> nat -> continuation list -> code_id list -> expr -> bool

val normalise_lifted_calling_free_in :
  static_env -> bool -> nat -> continuation list -> code_id list -> expr -> bool

val tidy_known_free_in : static_env -> nat -> nat -> expr -> bool

val demote_and_tidy_known_free_in : static_env -> nat -> expr -> bool

val lifted_free_in :
  static_env ->
  nat ->
  continuation list ->
  lifting list ->
  code_id list ->
  expr ->
  bool

val statics_first_free_in : static_env -> lifting list -> expr -> bool

val head_removal_free_in : static_env -> dropped -> expr -> bool

val lifted_holds_from :
  static_env ->
  nat ->
  dropped ->
  value_slot list ->
  loopified ->
  lifting list ->
  code_id list ->
  expr ->
  expr ->
  bool

val verdict_holds_from :
  static_env ->
  nat ->
  dropped ->
  value_slot list ->
  loopified ->
  lifting list ->
  code_id list ->
  expr ->
  expr ->
  validation ->
  bool

val verdict_knowing_holds_from :
  static_env ->
  nat ->
  dropped ->
  value_slot list ->
  loopified ->
  lifting list ->
  code_id list ->
  expr ->
  expr ->
  validation ->
  bool
