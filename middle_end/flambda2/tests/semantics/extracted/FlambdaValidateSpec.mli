open Datatypes
open FlambdaAliasLetsSpec
open FlambdaAliasesSpec
open FlambdaArmNumbersSpec
open FlambdaBlockLoadSpec
open FlambdaCodeLetsSpec
open FlambdaCompareSpec
open FlambdaCseSpec
open FlambdaDeadParamsSpec
open FlambdaDemoteSpec
open FlambdaFloatCodeSpec
open FlambdaForwardKindedSpec
open FlambdaForwardSpec
open FlambdaGiveScrutineeSpec
open FlambdaGroupsUpSpec
open FlambdaGuardScopeSpec
open FlambdaHoistCodeSpec
open FlambdaHoistStaticsSpec
open FlambdaIdentitiesSpec
open FlambdaInlineSiblingsSpec
open FlambdaInlineTailSpec
open FlambdaKnownClosuresSpec
open FlambdaKnownSlotsSpec
open FlambdaKnownSymbolsSpec
open FlambdaLiftAroundSpec
open FlambdaLiftFromArmsSpec
open FlambdaLiftFromCodeSpec
open FlambdaLiftSpec
open FlambdaLoopAliasesSpec
open FlambdaLoopLetsSpec
open FlambdaLoopifySpec
open FlambdaMergeSwitchesSpec
open FlambdaMergeTablesSpec
open FlambdaModel
open FlambdaNeverReturnsSpec
open FlambdaNormaliseSpec
open FlambdaParamAliasesSpec
open FlambdaRecHandlersSpec
open FlambdaRecoverComparisonSpec
open FlambdaRenameApartSpec
open FlambdaSameComparisonSpec
open FlambdaSameHandlersSpec
open FlambdaShadowParamsSpec
open FlambdaShareProjectionsSpec
open FlambdaSharedReadsSpec
open FlambdaShortcutSpec
open FlambdaSimplifyCarriedSpec
open FlambdaSinkContsSpec
open FlambdaSinkUsesSpec
open FlambdaStaticsOrderSpec
open FlambdaStubCallsSpec
open FlambdaTwinParamsSpec
open FlambdaUnaliasParamsSpec
open FlambdaUnboxInScopeSpec
open FlambdaUnboxLoopsSpec
open FlambdaUnboxParamsSpec
open FlambdaUnboxSpec
open FlambdaUnboxTogetherSpec
open FlambdaUnboxVariantsSpec
open FlambdaUntagParamsSpec
open FlambdaUnusedContsSpec
open FlambdaUnusedHeadStaticsSpec
open FlambdaUnusedParamsSpec
open FlambdaUnusedProjectionsSpec
open FlambdaUnusedSlotsSpec
open FlambdaUnusedStaticsSpec
open FlambdaWellFormedSpec
open FlambdaWrapReturnsSpec
open List

val iterate : nat -> (expr -> expr) -> expr -> expr

val normalise_expr_before : expr -> expr

val normalise_code_before : expr -> expr

val normalise_before_inlined : expr -> expr

val with_twins : continuation list -> expr -> expr

val normalise_before_round_avoiding :
  continuation list -> code_id list -> expr -> expr

val normalise_before_round : code_id list -> expr -> expr

val normalise_expr_params_knowing : facts -> expr -> expr

val normalise_code_params_knowing : facts -> expr -> expr

val normalise_code_params : expr -> expr

val normalise_params_round : expr -> expr

val demote_and_tidy : nat -> expr -> expr

val normalise_before : nat -> code_id list -> expr -> expr

val inline_once_applied_after : expr -> expr

val normalise_expr_after : expr -> expr

val normalise_code_after : expr -> expr

val normalise_after_round : code_id list -> expr -> expr

val normalise_after : nat -> code_id list -> expr -> expr

val without_statics : dropped -> expr -> expr

val without_head_statics : dropped -> expr -> expr

val with_lifting : lifting -> expr -> expr

val with_known : expr -> expr

val normalise_lifted_round : continuation list -> code_id list -> expr -> expr

val normalise_lifted : nat -> continuation list -> code_id list -> expr -> expr

val unit_continuations : lifting list -> continuation list

val demote_and_tidy_known_round : nat -> expr -> expr

val demote_and_tidy_known : nat -> expr -> expr

val lifted_in_turn :
  nat -> continuation list -> lifting list -> code_id list -> expr -> expr

val without_slots : value_slot list -> expr -> expr

val with_code_first : expr -> expr

val with_statics_first : lifting list -> expr -> expr

val place_conts : expr -> expr

val placed : expr -> expr

val tidy_expr_before : expr -> expr

val tidy_code_before : expr -> expr

val tidy_before_round : expr -> expr

val tidied_before : nat -> expr -> expr

val tidy_expr_after : expr -> expr

val tidy_code_after : expr -> expr

val tidy_after_round : expr -> expr

val tidied_after : nat -> expr -> expr

val take_returns_on : expr -> expr

val returns_taken_on : expr -> expr

val forwarded_in_unit : bool -> expr -> expr

val keeps_a_return : expr -> bool

val with_returns : bool -> expr -> expr

type validation =
  | Same_up_to_bound_names
  | Same_once_normalised
  | Same_once_unused_statics_removed
  | Same_once_head_statics_removed
  | Same_once_lifted
  | Not_validated

val validate_unit :
  nat ->
  dropped ->
  value_slot list ->
  loopified ->
  lifting list ->
  code_id list ->
  expr ->
  expr ->
  validation
