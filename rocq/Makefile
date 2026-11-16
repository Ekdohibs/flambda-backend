# Check the Rocq models and proofs, and regenerate the extracted OxCaml.
#
#   make          check every .v file, then run the extraction
#   make assumptions  print global assumptions and declaration types, and
#                 check the axioms against the trust ledger (TRUST-LEDGER.md)
#   make claims   check that every statement a Spec file binds is proved
#   make review   typeset what a reviewer reads as rocq/review.pdf
#   make clean    remove everything generated
#
# The sources live in five directories. Four are each a Rocq library:
#
#   rocq/                   OxCamlStdlib: models of OxCaml's own types (int,
#                           Int64, lists, exceptions as options), generic
#                           integer arithmetic, and their extraction
#                           directives; see HACKING.md there
#   backend/rocq/           OxCamlBackend: backend-wide definitions, such as
#                           the semantics of integer comparisons and the
#                           operations of the backend's instructions
#   backend/ssa/rocq/       OxCamlSsa: the SSA graph as the models see it
#                           (abstract blocks and values, with the few
#                           observations the passes make of them), and the
#                           properties of terminators, uses and dominance
#                           that any pass may need
#   backend/ssa/loop/rocq/  OxCamlLoop: the loop analyses and their
#                           extraction (Extract.v, one .ml per model); the
#                           passes using them are backend/ssa/loop/
#                           delete_empty_loops.ml and bounds_check_elim.ml
#
# The fifth, middle_end/flambda2/rocq/, holds two libraries, in folders by
# subject (README.md there has the tree and the rules of a folder):
#
#   OxCamlFlambda2          the terms of Flambda 2 and their semantics, with
#                           rewrites proved against it; extracted for tests
#                           only (validation/checks/FlambdaExtract.v)
#   OxCamlFlambda2Types     its folders types/ and simplify/: Flambda types
#                           and typing environments, and refinement relative
#                           to a typing environment; extracted likewise
#
# The extracted files are run through ocamlformat so that they satisfy the
# repository's `make fmt` check like any other source under backend/ssa/.

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
ROCQ ?= rocq
OCAMLFORMAT ?= ocamlformat

STDLIB_DIR := $(ROOT)/rocq
BACKEND_DIR := $(ROOT)/backend/rocq
SSA_DIR := $(ROOT)/backend/ssa/rocq
LOOP_DIR := $(ROOT)/backend/ssa/loop/rocq
FLAMBDA2_DIR := $(ROOT)/middle_end/flambda2/rocq
# The folders of $(FLAMBDA2_DIR) that hold .v files, by the library that
# their files are compiled into; each is given to Rocq with -Q. A folder is
# of one library, the folders are not nested, and each .v file is directly
# in one of them: Rocq binds the folders under a folder that it is given, so
# a file of an inner folder would have two names.
FLAMBDA2_FOLDERS := terms semantics wellformed heap regions/machine \
  regions/erasure regions/check regions/ends nominal rewrites/common \
  rewrites/calls rewrites/closures rewrites/statics rewrites/lifting \
  rewrites/continuations rewrites/switches rewrites/params rewrites/loops \
  rewrites/unboxing rewrites/lets validation/checks validation/names
FLAMBDA2_TYPES_FOLDERS := types/grammar types/env types/aliases types/meet \
  types/join types/provers types/naive simplify

FLAGS := -Q $(STDLIB_DIR) OxCamlStdlib -Q $(BACKEND_DIR) OxCamlBackend \
  -Q $(SSA_DIR) OxCamlSsa -Q $(LOOP_DIR) OxCamlLoop \
  $(foreach d,$(FLAMBDA2_FOLDERS),-Q $(FLAMBDA2_DIR)/$(d) OxCamlFlambda2) \
  $(foreach d,$(FLAMBDA2_TYPES_FOLDERS),-Q $(FLAMBDA2_DIR)/$(d) \
    OxCamlFlambda2Types)

STDLIB := $(addprefix $(STDLIB_DIR)/, \
  OxCamlLogic.v OxCamlOption.v ZTactics.v OxCamlInt.v OxCamlIntProofs.v \
  OxCamlInt64.v OxCamlInt64Proofs.v OxCamlList.v OxCamlListProofs.v \
  IntArith.v IntArithProofs.v OxCamlExtract.v IntArithExtract.v)
BACKEND := $(addprefix $(BACKEND_DIR)/, \
  IntegerComparison.v IntegerComparisonProofs.v IntegerComparisonExtract.v \
  Operation.v OperationExtract.v)
SSA := $(addprefix $(SSA_DIR)/, \
  SsaModel.v SsaModelExtract.v SsaGraphSpec.v SsaGraph.v SsaGraphProofs.v \
  SsaDominanceSpec.v SsaDominance.v SsaDominanceProofs.v SsaSemanticsSpec.v \
  SsaSemantics.v SsaSemanticsProofs.v SsaExecutionSpec.v \
  SsaExecutionProofs.v SsaLivenessSpec.v SsaLivenessProofs.v \
  SsaProgressSpec.v SsaProgressProofs.v SsaContextSpec.v SsaContextProofs.v)
LOOP := $(addprefix $(LOOP_DIR)/, \
  FourierMotzkin.v FourierMotzkinProofs.v FourierMotzkinExtract.v \
  LoopComparisons.v LoopComparisonsProofs.v AffineExpr.v AffineExprProofs.v \
  AffineExprExtract.v NaturalLoop.v NaturalLoopProofs.v AffineSsa.v \
  AffineSsaProofs.v AffineSsaExtract.v InductionVar.v InductionVarProofs.v \
  InductionVarExtract.v LoopTermination.v LoopTerminationProofs.v \
  DeadInductionVar.v DeadInductionVarProofs.v DeleteEmptyLoopsSpec.v \
  DeleteEmptyLoops.v DeleteEmptyLoopsProofs.v InductionRange.v \
  InductionRangeProofs.v BoundsCheckElim.v BoundsCheckElimProofs.v \
  LoopExecutionProofs.v LoopInvariantExecutionProofs.v \
  LoopCounterExecutionProofs.v LoopLiveOutProofs.v LoopEligibilitySpec.v \
  LoopEligibilityProofs.v LoopValueSnapshotProofs.v \
  LoopGuardExecutionProofs.v LoopRangeExecutionProofs.v \
  InductionRangeExecutionProofs.v BoundsCheckExecutionSpec.v \
  BoundsCheckExecutionProofs.v)
EXTRACT := $(LOOP_DIR)/Extract.v
# The files of the two Flambda 2 libraries: one to a line, each with its
# folder, in the order that the names of the files had here without their
# folders, which is nearly alphabetical. Put a new line where the name of
# its file belongs in alphabetical order, whatever its folder, so that
# branches that each add a file merge. The order means nothing: make takes
# the dependencies from rocq dep, and the review document puts the files in
# dependency order itself (review-pdf.pl).
FLAMBDA2 := $(addprefix $(FLAMBDA2_DIR)/, \
  rewrites/lets/FlambdaAliasLetsProofs.v \
  rewrites/lets/FlambdaAliasLetsSpec.v \
  nominal/FlambdaAlphaFreshProofs.v \
  nominal/FlambdaAlphaFreshSpec.v \
  nominal/FlambdaAlphaGroupProofs.v \
  nominal/FlambdaAlphaGroupSpec.v \
  nominal/FlambdaAlphaProofs.v \
  nominal/FlambdaAlphaSpec.v \
  semantics/FlambdaApplyProofs.v \
  semantics/FlambdaApplySpec.v \
  rewrites/switches/FlambdaArmNumbersNamesProofs.v \
  rewrites/switches/FlambdaArmNumbersProofs.v \
  rewrites/switches/FlambdaArmNumbersSpec.v \
  rewrites/lets/FlambdaArrayLoadProofs.v \
  rewrites/lets/FlambdaArrayLoadScopeProofs.v \
  rewrites/lets/FlambdaArrayLoadScopeSpec.v \
  rewrites/lets/FlambdaArrayLoadSpec.v \
  rewrites/params/FlambdaBindConstParamsNamesProofs.v \
  rewrites/params/FlambdaBindConstParamsProofs.v \
  rewrites/params/FlambdaBindConstParamsSpec.v \
  terms/FlambdaBindersProofs.v \
  terms/FlambdaBindersSpec.v \
  rewrites/lets/FlambdaBlockLoadProofs.v \
  rewrites/lets/FlambdaBlockLoadSpec.v \
  nominal/FlambdaCanonicalFreeNamesProofs.v \
  nominal/FlambdaCanonicalFreeNamesSpec.v \
  rewrites/calls/FlambdaClosureCallsProofs.v \
  rewrites/calls/FlambdaClosureCallsSpec.v \
  rewrites/closures/FlambdaClosureOfCodeProofs.v \
  rewrites/closures/FlambdaClosureOfCodeSpec.v \
  semantics/FlambdaClosuresProofs.v \
  semantics/FlambdaClosuresSpec.v \
  rewrites/closures/FlambdaCodeLetsProofs.v \
  rewrites/closures/FlambdaCodeLetsSpec.v \
  nominal/FlambdaCompareAlphaProofs.v \
  nominal/FlambdaCompareAlphaSpec.v \
  nominal/FlambdaCompareConverseProofs.v \
  nominal/FlambdaCompareConverseSpec.v \
  nominal/FlambdaCompareProofs.v \
  nominal/FlambdaCompareSpec.v \
  semantics/FlambdaContAlikeProofs.v \
  semantics/FlambdaContextProofs.v \
  semantics/FlambdaContextMemoryProofs.v \
  semantics/FlambdaContextSpec.v \
  semantics/FlambdaContextWhereProofs.v \
  semantics/FlambdaContextWhereSpec.v \
  semantics/FlambdaControlFlowProofs.v \
  semantics/FlambdaControlFlowSpec.v \
  rewrites/closures/FlambdaCopyCodeProofs.v \
  rewrites/closures/FlambdaCopyCodeSpec.v \
  rewrites/closures/FlambdaCopyLetsProofs.v \
  rewrites/closures/FlambdaCopyLetsSpec.v \
  rewrites/lets/FlambdaCseEquivalentProofs.v \
  rewrites/lets/FlambdaCseEquivalentSpec.v \
  rewrites/lets/FlambdaCseProofs.v \
  rewrites/lets/FlambdaCseSpec.v \
  rewrites/params/FlambdaDeadParamsProofs.v \
  rewrites/params/FlambdaDeadParamsSpec.v \
  rewrites/continuations/FlambdaDemoteProofs.v \
  rewrites/continuations/FlambdaDemoteScopeProofs.v \
  rewrites/continuations/FlambdaDemoteScopeSpec.v \
  rewrites/continuations/FlambdaDemoteSpec.v \
  nominal/FlambdaEquivarianceProofs.v \
  nominal/FlambdaEquivarianceSpec.v \
  nominal/FlambdaEquivarianceTactic.v \
  semantics/FlambdaExitBoundProofs.v \
  semantics/FlambdaExitBoundSpec.v \
  nominal/FlambdaExitWorldActionsProofs.v \
  nominal/FlambdaExitWorldActionsSpec.v \
  semantics/FlambdaExitWorldProofs.v \
  semantics/FlambdaExitWorldSpec.v \
  semantics/FlambdaExtcallProofs.v \
  semantics/FlambdaExtcallSpec.v \
  rewrites/closures/FlambdaFloatCodeProofs.v \
  rewrites/closures/FlambdaFloatCodeSpec.v \
  rewrites/continuations/FlambdaForwardArmProofs.v \
  rewrites/continuations/FlambdaForwardArmSpec.v \
  rewrites/continuations/FlambdaForwardJumpsProofs.v \
  rewrites/continuations/FlambdaForwardJumpsSpec.v \
  rewrites/continuations/FlambdaForwardKindedProofs.v \
  rewrites/continuations/FlambdaForwardKindedSpec.v \
  rewrites/continuations/FlambdaForwardProofs.v \
  rewrites/continuations/FlambdaForwardSpec.v \
  nominal/FlambdaFreeNamesProofs.v \
  nominal/FlambdaFreeNamesSpec.v \
  nominal/FlambdaFreshAbstractionProofs.v \
  nominal/FlambdaFreshAbstractionSpec.v \
  nominal/FlambdaFreshBindersProofs.v \
  nominal/FlambdaFreshBindersSpec.v \
  nominal/FlambdaFreshQuantifierProofs.v \
  nominal/FlambdaFreshQuantifierSpec.v \
  rewrites/switches/FlambdaGiveScrutineeNamesProofs.v \
  rewrites/switches/FlambdaGiveScrutineeProofs.v \
  rewrites/switches/FlambdaGiveScrutineeSpec.v \
  rewrites/statics/FlambdaGroupsUpProofs.v \
  rewrites/statics/FlambdaGroupsUpSpec.v \
  rewrites/common/FlambdaGuardScopeProofs.v \
  rewrites/common/FlambdaGuardScopeSpec.v \
  nominal/FlambdaHeapEquivarianceProofs.v \
  nominal/FlambdaHeapEquivarianceSpec.v \
  heap/FlambdaHeapMachineProofs.v \
  heap/FlambdaHeapMachineSpec.v \
  heap/FlambdaHeapFreeProofs.v \
  heap/FlambdaHeapFreeSpec.v \
  heap/FlambdaHeapProofs.v \
  heap/FlambdaHeapSpec.v \
  rewrites/closures/FlambdaHoistCodeProofs.v \
  rewrites/closures/FlambdaHoistCodeSpec.v \
  rewrites/statics/FlambdaHoistStaticsObservedProofs.v \
  rewrites/statics/FlambdaHoistStaticsObservedSpec.v \
  rewrites/statics/FlambdaHoistStaticsProofs.v \
  rewrites/statics/FlambdaHoistStaticsSpec.v \
  rewrites/lets/FlambdaIdentitiesProofs.v \
  rewrites/lets/FlambdaIdentitiesSpec.v \
  semantics/FlambdaImportsProofs.v \
  semantics/FlambdaImportsSpec.v \
  rewrites/calls/FlambdaInlineFreshProofs.v \
  rewrites/calls/FlambdaInlineFreshSpec.v \
  rewrites/calls/FlambdaInlineCallsAtProofs.v \
  rewrites/calls/FlambdaInlineCallsAtSpec.v \
  rewrites/calls/FlambdaInlineSiblingsProofs.v \
  rewrites/calls/FlambdaInlineSiblingsSpec.v \
  rewrites/continuations/FlambdaInlineTailProofs.v \
  rewrites/continuations/FlambdaInlineTailSpec.v \
  validation/names/FlambdaKeepsScopeProofs.v \
  validation/names/FlambdaKeepsScopeSpec.v \
  terms/FlambdaKindModel.v \
  rewrites/closures/FlambdaKnownClosuresProofs.v \
  rewrites/closures/FlambdaKnownClosuresSpec.v \
  rewrites/calls/FlambdaKnownSiblingsProofs.v \
  rewrites/calls/FlambdaKnownSiblingsSpec.v \
  rewrites/closures/FlambdaKnownSlotsProofs.v \
  rewrites/closures/FlambdaKnownSlotsSpec.v \
  rewrites/statics/FlambdaKnownSymbolsProofs.v \
  rewrites/statics/FlambdaKnownSymbolsSpec.v \
  semantics/FlambdaLaggingProofs.v \
  rewrites/lifting/FlambdaLiftAroundProofs.v \
  rewrites/lifting/FlambdaLiftAroundSpec.v \
  rewrites/continuations/FlambdaLiftContsNamesProofs.v \
  rewrites/continuations/FlambdaLiftContsProofs.v \
  rewrites/continuations/FlambdaLiftContsScopeProofs.v \
  rewrites/continuations/FlambdaLiftContsScopeSpec.v \
  rewrites/continuations/FlambdaLiftContsSpec.v \
  rewrites/lifting/FlambdaLiftFromArmsProofs.v \
  rewrites/lifting/FlambdaLiftFromArmsSpec.v \
  rewrites/lifting/FlambdaLiftFromCodeProofs.v \
  rewrites/lifting/FlambdaLiftFromCodeSpec.v \
  rewrites/lifting/FlambdaLiftObservedProofs.v \
  rewrites/lifting/FlambdaLiftObservedSpec.v \
  rewrites/lifting/FlambdaLiftProofs.v \
  rewrites/lifting/FlambdaLiftSpec.v \
  nominal/FlambdaLocationsMachineProofs.v \
  nominal/FlambdaLocationsMachineSpec.v \
  nominal/FlambdaLocationsPrimitiveProofs.v \
  nominal/FlambdaLocationsPrimitiveSpec.v \
  nominal/FlambdaLocationsProofs.v \
  nominal/FlambdaLocationsSpec.v \
  rewrites/loops/FlambdaLoopAliasesProofs.v \
  rewrites/loops/FlambdaLoopAliasesSpec.v \
  rewrites/loops/FlambdaLoopLetsProofs.v \
  rewrites/loops/FlambdaLoopLetsSpec.v \
  rewrites/loops/FlambdaLoopifyProofs.v \
  rewrites/loops/FlambdaLoopifySpec.v \
  validation/names/FlambdaLoopsScopeProofs.v \
  validation/names/FlambdaLoopsScopeSpec.v \
  rewrites/lets/FlambdaMadeLengthsProofs.v \
  rewrites/lets/FlambdaMadeLengthsSpec.v \
  terms/FlambdaMapProofs.v \
  terms/FlambdaMapSpec.v \
  rewrites/switches/FlambdaMergeSwitchesProofs.v \
  rewrites/switches/FlambdaMergeSwitchesSpec.v \
  rewrites/switches/FlambdaMergeTablesProofs.v \
  rewrites/switches/FlambdaMergeTablesScopeProofs.v \
  rewrites/switches/FlambdaMergeTablesScopeSpec.v \
  rewrites/switches/FlambdaMergeTablesSpec.v \
  terms/FlambdaModel.v \
  rewrites/closures/FlambdaMyClosureProofs.v \
  rewrites/closures/FlambdaMyClosureSpec.v \
  nominal/FlambdaNameAbstractionProofs.v \
  nominal/FlambdaNameAbstractionSpec.v \
  rewrites/calls/FlambdaNeverReturnsProofs.v \
  rewrites/calls/FlambdaNeverReturnsScopeProofs.v \
  rewrites/calls/FlambdaNeverReturnsScopeSpec.v \
  rewrites/calls/FlambdaNeverReturnsSpec.v \
  heap/FlambdaNoArrayValueObserversProofs.v \
  heap/FlambdaNoArrayValueObserversSpec.v \
  heap/FlambdaNoArrayValueProofs.v \
  heap/FlambdaNoArrayValueSpec.v \
  nominal/FlambdaNominalInstancesProofs.v \
  nominal/FlambdaNominalInstancesSpec.v \
  nominal/FlambdaNominalProofs.v \
  nominal/FlambdaNominalSpec.v \
  rewrites/common/FlambdaNormaliseProofs.v \
  rewrites/common/FlambdaNormaliseSpec.v \
  nominal/FlambdaObserversActionsProofs.v \
  nominal/FlambdaObserversActionsSpec.v \
  semantics/FlambdaObserversProofs.v \
  semantics/FlambdaObserversSpec.v \
  rewrites/params/FlambdaParamAliasesProofs.v \
  rewrites/params/FlambdaParamAliasesSpec.v \
  rewrites/params/FlambdaParamOrderProofs.v \
  rewrites/params/FlambdaParamOrderSpec.v \
  semantics/FlambdaPartialApplicationProofs.v \
  semantics/FlambdaPartialApplicationSpec.v \
  terms/FlambdaPrimitiveModel.v \
  semantics/FlambdaPrimitiveProofs.v \
  semantics/FlambdaPrimitiveSpec.v \
  wellformed/FlambdaRangeProofs.v \
  wellformed/FlambdaRangeSpec.v \
  rewrites/lets/FlambdaReadsAfterProofs.v \
  rewrites/lets/FlambdaReadsAfterSpec.v \
  rewrites/loops/FlambdaRecHandlersProofs.v \
  rewrites/loops/FlambdaRecHandlersSpec.v \
  rewrites/lets/FlambdaRecoverComparisonProofs.v \
  rewrites/lets/FlambdaRecoverComparisonScopeProofs.v \
  rewrites/lets/FlambdaRecoverComparisonScopeSpec.v \
  rewrites/lets/FlambdaRecoverComparisonSpec.v \
  regions/machine/FlambdaRegionsCallsProofs.v \
  regions/machine/FlambdaRegionsCallsSpec.v \
  regions/machine/FlambdaRegionsCellsProofs.v \
  regions/check/FlambdaRegionsCheckProofs.v \
  regions/check/FlambdaRegionsCheckSoundProofs.v \
  regions/check/FlambdaRegionsCheckSoundSpec.v \
  regions/check/FlambdaRegionsCheckSpec.v \
  regions/check/FlambdaRegionsCheckedAppliedProofs.v \
  regions/check/FlambdaRegionsCheckedBasicsProofs.v \
  regions/check/FlambdaRegionsCheckedCallProofs.v \
  regions/check/FlambdaRegionsCheckedDeepProofs.v \
  regions/check/FlambdaRegionsCheckedFaultProofs.v \
  regions/check/FlambdaRegionsCheckedGivenProofs.v \
  regions/check/FlambdaRegionsCheckedHeadProofs.v \
  regions/check/FlambdaRegionsCheckedInitialProofs.v \
  regions/check/FlambdaRegionsCheckedJumpProofs.v \
  regions/check/FlambdaRegionsCheckedLetProofs.v \
  regions/check/FlambdaRegionsCheckedMarksProofs.v \
  regions/check/FlambdaRegionsCheckedMovesProofs.v \
  regions/check/FlambdaRegionsCheckedOutlivesProofs.v \
  regions/check/FlambdaRegionsCheckedParamsProofs.v \
  regions/check/FlambdaRegionsCheckedProofs.v \
  regions/check/FlambdaRegionsCheckedRankProofs.v \
  regions/check/FlambdaRegionsCheckedReadsProofs.v \
  regions/check/FlambdaRegionsCheckedSameStoreProofs.v \
  regions/check/FlambdaRegionsCheckedSharedProofs.v \
  regions/check/FlambdaRegionsCheckedSpec.v \
  regions/check/FlambdaRegionsCheckedStableProofs.v \
  regions/check/FlambdaRegionsCheckedStateProofs.v \
  regions/check/FlambdaRegionsCheckedStoreProofs.v \
  regions/check/FlambdaRegionsCheckedUnknownProofs.v \
  regions/check/FlambdaRegionsCheckedViewsProofs.v \
  regions/ends/FlambdaRegionsEndsBasicsProofs.v \
  regions/ends/FlambdaRegionsEndsCallProofs.v \
  regions/ends/FlambdaRegionsEndsHandlersProofs.v \
  regions/ends/FlambdaRegionsEndsJumpProofs.v \
  regions/ends/FlambdaRegionsEndsLetProofs.v \
  regions/ends/FlambdaRegionsEndsProofs.v \
  regions/ends/FlambdaRegionsEndsSpec.v \
  regions/ends/FlambdaRegionsEndsStableProofs.v \
  regions/ends/FlambdaRegionsEndsViewProofs.v \
  regions/erasure/FlambdaRegionsErasedCallsProofs.v \
  regions/erasure/FlambdaRegionsErasedProofs.v \
  regions/erasure/FlambdaRegionsErasedReadsProofs.v \
  regions/erasure/FlambdaRegionsErasureProofs.v \
  regions/erasure/FlambdaRegionsErasureSpec.v \
  regions/machine/FlambdaRegionsFreshProofs.v \
  regions/machine/FlambdaRegionsInvariantProofs.v \
  regions/machine/FlambdaRegionsInvariantSpec.v \
  regions/machine/FlambdaRegionsNoPoisonProofs.v \
  regions/erasure/FlambdaRegionsNoStepProofs.v \
  regions/erasure/FlambdaRegionsOnceProofs.v \
  regions/erasure/FlambdaRegionsPlainStepProofs.v \
  regions/machine/FlambdaRegionsProofs.v \
  regions/machine/FlambdaRegionsReadsProofs.v \
  regions/machine/FlambdaRegionsSpec.v \
  regions/machine/FlambdaRegionsStepsProofs.v \
  nominal/FlambdaRenameApartAlphaProofs.v \
  nominal/FlambdaRenameApartAlphaSpec.v \
  nominal/FlambdaRenameApartCanonicalProofs.v \
  nominal/FlambdaRenameApartCanonicalSpec.v \
  nominal/FlambdaRenameApartProofs.v \
  nominal/FlambdaRenameApartSpec.v \
  rewrites/lets/FlambdaSameComparisonProofs.v \
  rewrites/lets/FlambdaSameComparisonScopeProofs.v \
  rewrites/lets/FlambdaSameComparisonScopeSpec.v \
  rewrites/lets/FlambdaSameComparisonSpec.v \
  rewrites/continuations/FlambdaSameHandlersProofs.v \
  rewrites/continuations/FlambdaSameHandlersSpec.v \
  semantics/FlambdaRoundsProofs.v \
  semantics/FlambdaRunUnderProofs.v \
  semantics/FlambdaRunUnderSpec.v \
  wellformed/FlambdaScopeProofs.v \
  wellformed/FlambdaScopeSpec.v \
  rewrites/common/FlambdaScopeTactic.v \
  rewrites/common/FlambdaScopedInductionProofs.v \
  rewrites/common/FlambdaScopedProofs.v \
  semantics/FlambdaSemanticsProofs.v \
  semantics/FlambdaSemanticsSpec.v \
  rewrites/closures/FlambdaSetsOfCodeProofs.v \
  rewrites/closures/FlambdaSetsOfCodeSpec.v \
  rewrites/unboxing/FlambdaShadowParamsNamesProofs.v \
  rewrites/unboxing/FlambdaShadowParamsProofs.v \
  rewrites/unboxing/FlambdaShadowParamsSpec.v \
  rewrites/closures/FlambdaShareProjectionsProofs.v \
  rewrites/closures/FlambdaShareProjectionsScopeProofs.v \
  rewrites/closures/FlambdaShareProjectionsScopeSpec.v \
  rewrites/closures/FlambdaShareProjectionsSpec.v \
  rewrites/statics/FlambdaSharedReadsPairsProofs.v \
  rewrites/statics/FlambdaSharedReadsPairsSpec.v \
  rewrites/statics/FlambdaSharedReadsProofs.v \
  rewrites/statics/FlambdaSharedReadsSpec.v \
  rewrites/continuations/FlambdaShortcutProofs.v \
  rewrites/continuations/FlambdaShortcutSpec.v \
  semantics/FlambdaSimulationProofs.v \
  rewrites/continuations/FlambdaSinkContsProofs.v \
  rewrites/continuations/FlambdaSinkContsSpec.v \
  rewrites/continuations/FlambdaSinkUsesProofs.v \
  rewrites/continuations/FlambdaSinkUsesSpec.v \
  rewrites/closures/FlambdaSlotsOfCodeProofs.v \
  rewrites/closures/FlambdaSlotsOfCodeSpec.v \
  rewrites/common/FlambdaSpineWalkNamesProofs.v \
  rewrites/common/FlambdaSpineWalkNamesSpec.v \
  rewrites/common/FlambdaSpineWalkProofs.v \
  rewrites/common/FlambdaSpineWalkSpec.v \
  nominal/FlambdaStateSupportProofs.v \
  nominal/FlambdaStateSupportSpec.v \
  semantics/FlambdaStaticProofs.v \
  semantics/FlambdaStaticSpec.v \
  wellformed/FlambdaStaticsInScopeProofs.v \
  wellformed/FlambdaStaticsInScopeSpec.v \
  rewrites/statics/FlambdaStaticsOrderProofs.v \
  rewrites/statics/FlambdaStaticsOrderSpec.v \
  validation/names/FlambdaStaticsScopeProofs.v \
  validation/names/FlambdaStaticsScopeSpec.v \
  rewrites/calls/FlambdaStubCallsProofs.v \
  rewrites/calls/FlambdaStubCallsSpec.v \
  terms/FlambdaTermInductionProofs.v \
  wellformed/FlambdaTrapProofs.v \
  wellformed/FlambdaTrapSpec.v \
  semantics/FlambdaTupledProofs.v \
  semantics/FlambdaTupledSpec.v \
  rewrites/unboxing/FlambdaTwinParamsNamesProofs.v \
  rewrites/unboxing/FlambdaTwinParamsProofs.v \
  rewrites/unboxing/FlambdaTwinParamsSpec.v \
  rewrites/params/FlambdaUnaliasParamsNamesProofs.v \
  rewrites/params/FlambdaUnaliasParamsProofs.v \
  rewrites/params/FlambdaUnaliasParamsSpec.v \
  rewrites/loops/FlambdaUnaliasRecParamsNamesProofs.v \
  rewrites/loops/FlambdaUnaliasRecParamsProofs.v \
  rewrites/loops/FlambdaUnaliasRecParamsSpec.v \
  rewrites/unboxing/FlambdaUnboxInScopeProofs.v \
  rewrites/unboxing/FlambdaUnboxInScopeRenameProofs.v \
  rewrites/unboxing/FlambdaUnboxInScopeRenameSpec.v \
  rewrites/unboxing/FlambdaUnboxInScopeSpec.v \
  rewrites/loops/FlambdaUnboxLoopsProofs.v \
  rewrites/loops/FlambdaUnboxLoopsSpec.v \
  rewrites/unboxing/FlambdaUnboxParamsExamplesProofs.v \
  rewrites/unboxing/FlambdaUnboxParamsExamplesSpec.v \
  rewrites/unboxing/FlambdaUnboxParamsNamesProofs.v \
  rewrites/unboxing/FlambdaUnboxParamsProofs.v \
  rewrites/unboxing/FlambdaUnboxParamsRenameProofs.v \
  rewrites/unboxing/FlambdaUnboxParamsRenameSpec.v \
  rewrites/unboxing/FlambdaUnboxParamsSpec.v \
  rewrites/unboxing/FlambdaUnboxProofs.v \
  rewrites/unboxing/FlambdaUnboxSpec.v \
  rewrites/unboxing/FlambdaUnboxTogetherNamesProofs.v \
  rewrites/unboxing/FlambdaUnboxTogetherProofs.v \
  rewrites/unboxing/FlambdaUnboxTogetherSpec.v \
  rewrites/unboxing/FlambdaUnboxVariantsExamplesProofs.v \
  rewrites/unboxing/FlambdaUnboxVariantsExamplesSpec.v \
  rewrites/unboxing/FlambdaUnboxVariantsProofs.v \
  rewrites/unboxing/FlambdaUnboxVariantsRenameProofs.v \
  rewrites/unboxing/FlambdaUnboxVariantsRenameSpec.v \
  rewrites/unboxing/FlambdaUnboxVariantsSpec.v \
  semantics/FlambdaUnitFromProofs.v \
  rewrites/unboxing/FlambdaUntagParamsNamesProofs.v \
  rewrites/unboxing/FlambdaUntagParamsProofs.v \
  rewrites/unboxing/FlambdaUntagParamsSpec.v \
  rewrites/continuations/FlambdaUnusedContsProofs.v \
  rewrites/continuations/FlambdaUnusedContsSpec.v \
  rewrites/params/FlambdaUnusedParamsProofs.v \
  rewrites/params/FlambdaUnusedParamsSpec.v \
  rewrites/closures/FlambdaUnusedSlotsObservedProofs.v \
  rewrites/closures/FlambdaUnusedSlotsObservedSpec.v \
  rewrites/closures/FlambdaUnusedSlotsProofs.v \
  rewrites/closures/FlambdaUnusedSlotsSpec.v \
  rewrites/closures/FlambdaUnusedProjectionsProofs.v \
  rewrites/closures/FlambdaUnusedProjectionsSpec.v \
  rewrites/statics/FlambdaUnusedHeadStaticsProofs.v \
  rewrites/statics/FlambdaUnusedHeadStaticsSpec.v \
  rewrites/statics/FlambdaUnusedStaticsProofs.v \
  rewrites/statics/FlambdaUnusedStaticsSpec.v \
  validation/checks/FlambdaValidateCopyingProofs.v \
  validation/checks/FlambdaValidateCopyingSpec.v \
  validation/checks/FlambdaValidateImportsProofs.v \
  validation/names/FlambdaValidateImportsRenameProofs.v \
  validation/names/FlambdaValidateImportsRenameSpec.v \
  validation/checks/FlambdaValidateImportsSpec.v \
  validation/checks/FlambdaValidateInliningProofs.v \
  validation/names/FlambdaValidateInliningRenameProofs.v \
  validation/names/FlambdaValidateInliningRenameSpec.v \
  validation/checks/FlambdaValidateInliningSpec.v \
  validation/checks/FlambdaValidateKnowingProofs.v \
  validation/checks/FlambdaValidateKnowingSpec.v \
  validation/checks/FlambdaValidateProofs.v \
  validation/checks/FlambdaValidateReadingProofs.v \
  validation/names/FlambdaValidateReadingRenameProofs.v \
  validation/names/FlambdaValidateReadingRenameSpec.v \
  validation/checks/FlambdaValidateReadingSpec.v \
  validation/checks/FlambdaValidateRelatedProofs.v \
  validation/names/FlambdaValidateRelatedRenameProofs.v \
  validation/names/FlambdaValidateRelatedRenameSpec.v \
  validation/checks/FlambdaValidateRelatedSpec.v \
  validation/names/FlambdaValidateScopeProofs.v \
  validation/names/FlambdaValidateScopeSpec.v \
  validation/checks/FlambdaValidateSpec.v \
  validation/names/FlambdaValidatorEquivarianceBaseProofs.v \
  validation/names/FlambdaValidatorEquivarianceProofs.v \
  validation/names/FlambdaValidatorEquivarianceSpec.v \
  semantics/FlambdaValueProofs.v \
  semantics/FlambdaValueSpec.v \
  wellformed/FlambdaWellFormedProofs.v \
  wellformed/FlambdaWellFormedSpec.v \
  wellformed/FlambdaWellKindedProofs.v \
  wellformed/FlambdaWellKindedSpec.v \
  rewrites/params/FlambdaWrapReturnsNamesProofs.v \
  rewrites/params/FlambdaWrapReturnsProofs.v \
  rewrites/params/FlambdaWrapReturnsSpec.v \
)
FLAMBDA2_EXTRACT := $(FLAMBDA2_DIR)/validation/checks/FlambdaExtract.v
FLAMBDA2_EXTRACTED_DIR := $(ROOT)/middle_end/flambda2/tests/semantics/extracted
FLAMBDA2_TYPES := $(addprefix $(FLAMBDA2_DIR)/, \
  types/naive/FlambdaAliasesProofs.v \
  types/naive/FlambdaAliasesSpec.v \
  types/env/FlambdaBindingTime.v \
  types/aliases/FlambdaEnvAliases.v \
  types/aliases/FlambdaEnvAliasesFollowProofs.v \
  types/aliases/FlambdaEnvAliasesFollowScopeProofs.v \
  types/aliases/FlambdaEnvAliasesFollowScopeSpec.v \
  types/aliases/FlambdaEnvAliasesFollowSpec.v \
  types/aliases/FlambdaEnvAliasesInvariantProofs.v \
  types/aliases/FlambdaEnvAliasesInvariantSpec.v \
  types/aliases/FlambdaEnvAliasesProofs.v \
  types/aliases/FlambdaEnvAliasesRenamingProofs.v \
  types/aliases/FlambdaEnvAliasesRenamingSpec.v \
  types/aliases/FlambdaEnvAliasesReplay.v \
  types/aliases/FlambdaEnvAliasesSetProofs.v \
  types/aliases/FlambdaEnvAliasesSetSpec.v \
  types/aliases/FlambdaEnvAliasesSpec.v \
  types/join/FlambdaEnvRenameJoinProofs.v \
  types/join/FlambdaEnvRenameJoinSpec.v \
  types/join/FlambdaEnvRenameJoinTypesProofs.v \
  types/meet/FlambdaEnvRenameMeetPartsProofs.v \
  types/meet/FlambdaEnvRenameMeetProductsProofs.v \
  types/meet/FlambdaEnvRenameMeetProofs.v \
  types/meet/FlambdaEnvRenameMeetValuesProofs.v \
  types/env/FlambdaEnvRenameProofs.v \
  types/env/FlambdaEnvRenameSpec.v \
  types/meet/FlambdaExpandHead.v \
  types/meet/FlambdaExpandHeadProofs.v \
  types/meet/FlambdaExpandHeadSpec.v \
  types/join/FlambdaJoinAddsProofs.v \
  types/join/FlambdaJoinAddsSpec.v \
  types/join/FlambdaJoinDefineProofs.v \
  types/join/FlambdaJoinDefineSpec.v \
  types/join/FlambdaJoinEnv.v \
  types/join/FlambdaJoinEnvProofs.v \
  types/join/FlambdaJoinEnvSpec.v \
  types/join/FlambdaJoinEquationNamesProofs.v \
  types/join/FlambdaJoinEquationNamesSpec.v \
  types/join/FlambdaJoinFitsProofs.v \
  types/join/FlambdaJoinFitsSpec.v \
  types/join/FlambdaJoinInTermsProofs.v \
  types/join/FlambdaJoinInTermsSpec.v \
  types/join/FlambdaJoinKindsProofs.v \
  types/join/FlambdaJoinKindsSpec.v \
  types/join/FlambdaJoinMeet.v \
  types/join/FlambdaJoinMeetExamplesProofs.v \
  types/join/FlambdaJoinMeetExamplesSpec.v \
  types/join/FlambdaJoinMeetFitsProofs.v \
  types/join/FlambdaJoinMeetFitsSpec.v \
  types/join/FlambdaJoinMeetProofs.v \
  types/join/FlambdaJoinMeetRenameExampleProofs.v \
  types/join/FlambdaJoinMeetRenameExampleSpec.v \
  types/join/FlambdaJoinMeetRenameProofs.v \
  types/join/FlambdaJoinMeetRenameSpec.v \
  types/join/FlambdaJoinMeetSpec.v \
  types/join/FlambdaJoinNamesProofs.v \
  types/join/FlambdaJoinNamesSpec.v \
  types/join/FlambdaJoinNothingProofs.v \
  types/join/FlambdaJoinNothingSpec.v \
  types/join/FlambdaJoinProofs.v \
  types/join/FlambdaJoinRepairProofs.v \
  types/join/FlambdaJoinRepairSpec.v \
  types/join/FlambdaJoinSpec.v \
  types/join/FlambdaJoinStateProofs.v \
  types/join/FlambdaJoinStateSpec.v \
  types/join/FlambdaJoinTypes.v \
  types/join/FlambdaJoinTypesExamplesProofs.v \
  types/join/FlambdaJoinTypesExamplesSpec.v \
  types/join/FlambdaJoinTypesKeepsProofs.v \
  types/join/FlambdaJoinTypesKeepsSpec.v \
  types/join/FlambdaJoinTypesProofs.v \
  types/join/FlambdaJoinTypesSpec.v \
  types/join/FlambdaJoinTypesWithEnvProofs.v \
  types/join/FlambdaJoinTypesWithEnvSpec.v \
  types/join/FlambdaJoinUsesProofs.v \
  types/join/FlambdaJoinUsesSpec.v \
  types/join/FlambdaJoinWellFormedProofs.v \
  types/join/FlambdaJoinWellFormedSpec.v \
  types/meet/FlambdaMeet.v \
  types/meet/FlambdaMeetChoicesProofs.v \
  types/meet/FlambdaMeetChoicesSpec.v \
  types/meet/FlambdaMeetClosuresProofs.v \
  types/meet/FlambdaMeetClosuresSpec.v \
  types/meet/FlambdaMeetDepthProofs.v \
  types/meet/FlambdaMeetDepthSpec.v \
  types/meet/FlambdaMeetEnv.v \
  types/meet/FlambdaMeetEnvProofs.v \
  types/meet/FlambdaMeetEnvSpec.v \
  types/meet/FlambdaMeetExamplesProofs.v \
  types/meet/FlambdaMeetExamplesSpec.v \
  types/meet/FlambdaMeetFuelProofs.v \
  types/meet/FlambdaMeetFuelSpec.v \
  types/meet/FlambdaMeetGrammar.v \
  types/meet/FlambdaMeetHeadsProofs.v \
  types/meet/FlambdaMeetHeadsSpec.v \
  types/meet/FlambdaMeetImmediateProofs.v \
  types/meet/FlambdaMeetImmediateSpec.v \
  types/meet/FlambdaMeetKeepsProofs.v \
  types/meet/FlambdaMeetKeepsSpec.v \
  types/meet/FlambdaMeetMatchExamplesProofs.v \
  types/meet/FlambdaMeetMatchExamplesSpec.v \
  types/meet/FlambdaMeetParamsProofs.v \
  types/meet/FlambdaMeetParamsSpec.v \
  types/meet/FlambdaMeetProductsProofs.v \
  types/meet/FlambdaMeetProductsSpec.v \
  types/meet/FlambdaMeetProjectionExamplesProofs.v \
  types/meet/FlambdaMeetProjectionExamplesSpec.v \
  types/meet/FlambdaMeetProofs.v \
  types/meet/FlambdaMeetRelationFuelProofs.v \
  types/meet/FlambdaMeetRelationFuelSpec.v \
  types/meet/FlambdaMeetRenameHeadsProofs.v \
  types/meet/FlambdaMeetRenameProofs.v \
  types/meet/FlambdaMeetRenameRowLikeProofs.v \
  types/meet/FlambdaMeetRenameScopesProofs.v \
  types/meet/FlambdaMeetRenameSpec.v \
  types/meet/FlambdaMeetRowLikeProofs.v \
  types/meet/FlambdaMeetRowLikeSpec.v \
  types/meet/FlambdaMeetScopesProofs.v \
  types/meet/FlambdaMeetScopesSpec.v \
  types/meet/FlambdaMeetSpec.v \
  types/meet/FlambdaMeetStepsProofs.v \
  types/meet/FlambdaMeetStepsSpec.v \
  types/meet/FlambdaMeetTagExamplesProofs.v \
  types/meet/FlambdaMeetTagExamplesSpec.v \
  types/meet/FlambdaMeetTermsProofs.v \
  types/meet/FlambdaMeetTermsSpec.v \
  types/meet/FlambdaMeetTerminationProofs.v \
  types/meet/FlambdaMeetTerminationSpec.v \
  types/meet/FlambdaMeetValidity.v \
  types/provers/FlambdaProvers.v \
  types/provers/FlambdaProversExamplesProofs.v \
  types/provers/FlambdaProversExamplesSpec.v \
  types/provers/FlambdaProversPoisonProofs.v \
  types/provers/FlambdaProversPoisonSpec.v \
  types/provers/FlambdaProversProofs.v \
  types/provers/FlambdaProversRenameProofs.v \
  types/provers/FlambdaProversRenameSpec.v \
  types/provers/FlambdaProversSpec.v \
  types/provers/FlambdaReify.v \
  types/provers/FlambdaReifyProofs.v \
  types/provers/FlambdaReifySpec.v \
  simplify/FlambdaSimplifyApplyProofs.v \
  simplify/FlambdaSimplifyApplySpec.v \
  simplify/FlambdaSimplifyArithProofs.v \
  simplify/FlambdaSimplifyArithSpec.v \
  simplify/FlambdaSimplifyArmsProofs.v \
  simplify/FlambdaSimplifyArmsSpec.v \
  simplify/FlambdaSimplifyBoxingProofs.v \
  simplify/FlambdaSimplifyBoxingSpec.v \
  simplify/FlambdaSimplifyCarriedProofs.v \
  simplify/FlambdaSimplifyCarriedSpec.v \
  simplify/FlambdaSimplifyClosuresUnionProofs.v \
  simplify/FlambdaSimplifyClosuresUnionSpec.v \
  simplify/FlambdaSimplifyCodeEnvProofs.v \
  simplify/FlambdaSimplifyCodeEnvSpec.v \
  simplify/FlambdaSimplifyHandlersProofs.v \
  simplify/FlambdaSimplifyHandlersSpec.v \
  simplify/FlambdaSimplifyPassProofs.v \
  simplify/FlambdaSimplifyPassRenamingProofs.v \
  simplify/FlambdaSimplifyPassRenamingSpec.v \
  simplify/FlambdaSimplifyPassSpec.v \
  simplify/FlambdaSimplifyPrimitivesProofs.v \
  simplify/FlambdaSimplifyPrimitivesRenamingProofs.v \
  simplify/FlambdaSimplifyPrimitivesRenamingSpec.v \
  simplify/FlambdaSimplifyPrimitivesSpec.v \
  simplify/FlambdaSimplifyProjectionsProofs.v \
  simplify/FlambdaSimplifyProjectionsSpec.v \
  simplify/FlambdaSimplifyPruningProofs.v \
  simplify/FlambdaSimplifyPruningRenamingProofs.v \
  simplify/FlambdaSimplifyPruningRenamingSpec.v \
  simplify/FlambdaSimplifyPruningSpec.v \
  simplify/FlambdaSimplifyStaticsProofs.v \
  simplify/FlambdaSimplifyStaticsSpec.v \
  simplify/FlambdaSimplifySwitchArmsProofs.v \
  simplify/FlambdaSimplifySwitchArmsSpec.v \
  simplify/FlambdaSimplifySwitchFuelProofs.v \
  simplify/FlambdaSimplifySwitchFuelSpec.v \
  simplify/FlambdaSimplifySwitchMergeProofs.v \
  simplify/FlambdaSimplifySwitchMergeSpec.v \
  simplify/FlambdaSimplifySwitchProofs.v \
  simplify/FlambdaSimplifySwitchRewriteProofs.v \
  simplify/FlambdaSimplifySwitchRewriteSpec.v \
  simplify/FlambdaSimplifySwitchSpec.v \
  simplify/FlambdaSimplifyUsesProofs.v \
  simplify/FlambdaSimplifyUsesSpec.v \
  simplify/FlambdaSimplifyValueSlotsProofs.v \
  simplify/FlambdaSimplifyValueSlotsSpec.v \
  types/grammar/FlambdaTypeAccessProofs.v \
  types/grammar/FlambdaTypeAccessSpec.v \
  types/grammar/FlambdaTypeAliasKindsProofs.v \
  types/grammar/FlambdaTypeAliasKindsSpec.v \
  types/grammar/FlambdaTypeCreatorsProofs.v \
  types/grammar/FlambdaTypeCreatorsSpec.v \
  types/grammar/FlambdaTypeEqualProofs.v \
  types/grammar/FlambdaTypeEqualSpec.v \
  types/grammar/FlambdaTypeGrammar.v \
  types/grammar/FlambdaTypeGrammarProofs.v \
  types/grammar/FlambdaTypeMeaningBoolProofs.v \
  types/grammar/FlambdaTypeMeaningBoolSpec.v \
  types/grammar/FlambdaTypeMeaningProofs.v \
  types/grammar/FlambdaTypeMeaningSpec.v \
  types/grammar/FlambdaTypeNominalProofs.v \
  types/grammar/FlambdaTypeNominalSpec.v \
  types/grammar/FlambdaTypeProject.v \
  types/grammar/FlambdaTypeProjectRenameProofs.v \
  types/grammar/FlambdaTypeRenameProofs.v \
  types/grammar/FlambdaTypeRenameSpec.v \
  types/grammar/FlambdaTypeWellFormedProofs.v \
  types/grammar/FlambdaTypeWellFormedSpec.v \
  types/naive/FlambdaTypesProofs.v \
  types/naive/FlambdaTypesSpec.v \
  types/env/FlambdaTypingEnv.v \
  types/env/FlambdaTypingEnvExamplesProofs.v \
  types/env/FlambdaTypingEnvExamplesSpec.v \
  types/env/FlambdaTypingEnvExtension.v \
  types/env/FlambdaTypingEnvExtensionProofs.v \
  types/env/FlambdaTypingEnvExtensionSpec.v \
  types/env/FlambdaTypingEnvInvariantProofs.v \
  types/env/FlambdaTypingEnvInvariantSpec.v \
  types/env/FlambdaTypingEnvKindedProofs.v \
  types/env/FlambdaTypingEnvKindedSpec.v \
  types/env/FlambdaTypingEnvOperationsProofs.v \
  types/env/FlambdaTypingEnvOperationsSpec.v \
  types/env/FlambdaTypingEnvProofs.v \
  types/env/FlambdaTypingEnvRangeProofs.v \
  types/env/FlambdaTypingEnvRangeSpec.v \
  types/env/FlambdaTypingEnvRefinementProofs.v \
  types/env/FlambdaTypingEnvRefinementSpec.v \
  types/env/FlambdaTypingEnvSpec.v \
)

# The extraction writes one .ml and .mli per model into the loop directory;
# those of the SSA models are then moved next to their sources. Those of the
# rocq/ models that are not mapped onto OxCaml's own functions (OxCamlList,
# IntArith) stay in the loop directory.
SSA_MODULES := SsaModel SsaGraphSpec SsaGraph SsaDominance SsaSemanticsSpec \
  SsaSemantics
SSA_EXTRACTED := $(addprefix $(SSA_DIR)/, \
  $(foreach m, $(SSA_MODULES), $(m).ml $(m).mli))
LOOP_EXTRACTED := $(addprefix $(LOOP_DIR)/, \
  OxCamlList.ml OxCamlList.mli IntArith.ml IntArith.mli \
  FourierMotzkin.ml FourierMotzkin.mli \
  LoopComparisons.ml LoopComparisons.mli AffineExpr.ml AffineExpr.mli \
  NaturalLoop.ml NaturalLoop.mli AffineSsa.ml AffineSsa.mli \
  InductionVar.ml InductionVar.mli LoopTermination.ml LoopTermination.mli \
  DeadInductionVar.ml DeadInductionVar.mli \
  DeleteEmptyLoops.ml DeleteEmptyLoops.mli \
  InductionRange.ml InductionRange.mli \
  BoundsCheckElim.ml BoundsCheckElim.mli)
EXTRACTED := $(SSA_EXTRACTED) $(LOOP_EXTRACTED)

SOURCES := $(STDLIB) $(BACKEND) $(SSA) $(LOOP) $(EXTRACT) $(FLAMBDA2) \
  $(FLAMBDA2_TYPES) $(FLAMBDA2_EXTRACT)
MODULES := $(patsubst $(STDLIB_DIR)/%.v,OxCamlStdlib.%,$(STDLIB)) \
  $(patsubst $(BACKEND_DIR)/%.v,OxCamlBackend.%,$(BACKEND)) \
  $(patsubst $(SSA_DIR)/%.v,OxCamlSsa.%,$(SSA)) \
  $(patsubst $(LOOP_DIR)/%.v,OxCamlLoop.%,$(LOOP) $(EXTRACT)) \
  $(patsubst %.v,OxCamlFlambda2.%,$(notdir $(FLAMBDA2) \
    $(FLAMBDA2_EXTRACT))) \
  $(patsubst %.v,OxCamlFlambda2Types.%,$(notdir $(FLAMBDA2_TYPES)))
DEPS := $(STDLIB_DIR)/.rocqdeps.d

.PHONY: all assumptions claims layout review clean

all: layout $(SOURCES:.v=.vo) claims

# The folders of middle_end/flambda2/rocq/ (README.md there). A .v file under
# it that is in neither list above would not be compiled, and nothing else
# would say so; a file that is not directly in a folder of its library would
# have no name, or the name of the other library; and two files of one name,
# in two folders, would be one module to Rocq. So each of these is an error
# here, with the name of the file, and so is a .v file in
# middle_end/flambda2/types/rocq/, where the files of types/ and simplify/
# were. A folder in both lists of folders, and a folder of the lists inside
# another, are errors with the name of the folder. The lines of _CoqProject
# there, which an editor reads, must be the flags of FLAGS: the same
# directories under the same libraries. The recipe itself does not run Rocq.
FLAMBDA2_LISTED := $(patsubst $(FLAMBDA2_DIR)/%,%,$(FLAMBDA2) \
  $(FLAMBDA2_EXTRACT))
FLAMBDA2_TYPES_LISTED := $(patsubst $(FLAMBDA2_DIR)/%,%,$(FLAMBDA2_TYPES))

layout:
	@set -eu; \
	  layout_dir=$$(mktemp -d); \
	  trap 'rm -rf "$$layout_dir"' 0; \
	  cd $(FLAMBDA2_DIR); \
	  bad=0; \
	  for f in ../types/rocq/*.v; do \
	    [ -e "$$f" ] || continue; bad=1; \
	    echo "error: $$f: no .v file is in that directory any more; the" \
	      "folders types/ and simplify/ of $(FLAMBDA2_DIR) hold them" >&2; \
	  done; \
	  for f in $(FLAMBDA2_LISTED); do \
	    case " $(FLAMBDA2_FOLDERS) " in *" $${f%/*} "*) ;; *) bad=1; \
	      echo "error: $$f: not in a folder of FLAMBDA2_FOLDERS" >&2;; \
	    esac; \
	  done; \
	  for f in $(FLAMBDA2_TYPES_LISTED); do \
	    case " $(FLAMBDA2_TYPES_FOLDERS) " in *" $${f%/*} "*) ;; *) bad=1; \
	      echo "error: $$f: not in a folder of FLAMBDA2_TYPES_FOLDERS" >&2;; \
	    esac; \
	  done; \
	  for d in $(FLAMBDA2_FOLDERS); do \
	    case " $(FLAMBDA2_TYPES_FOLDERS) " in *" $$d "*) bad=1; \
	      echo "error: the folder $$d is in FLAMBDA2_FOLDERS and in" \
	        "FLAMBDA2_TYPES_FOLDERS: a folder is of one library" >&2;; \
	    esac; \
	  done; \
	  for d in $(FLAMBDA2_FOLDERS) $(FLAMBDA2_TYPES_FOLDERS); do \
	    for e in $(FLAMBDA2_FOLDERS) $(FLAMBDA2_TYPES_FOLDERS); do \
	      case "$$e" in "$$d"/*) bad=1; \
	        echo "error: the folder $$e is inside the folder $$d: the" \
	          "folders given to Rocq are not nested" >&2;; \
	      esac; \
	    done; \
	  done; \
	  printf '%s\n' $(FLAMBDA2_LISTED) $(FLAMBDA2_TYPES_LISTED) \
	    | LC_ALL=C sort > "$$layout_dir/listed"; \
	  find . -name '*.v' | sed 's,^\./,,' | LC_ALL=C sort \
	    > "$$layout_dir/found"; \
	  LC_ALL=C comm -13 "$$layout_dir/listed" "$$layout_dir/found" \
	    > "$$layout_dir/unlisted"; \
	  if [ -s "$$layout_dir/unlisted" ]; then bad=1; \
	    echo "error: .v files under $(FLAMBDA2_DIR) that are in neither" \
	      "FLAMBDA2 nor FLAMBDA2_TYPES of rocq/Makefile:" >&2; \
	    sed 's/^/  /' "$$layout_dir/unlisted" >&2; \
	  fi; \
	  for name in $$(sed 's,.*/,,' "$$layout_dir/found" | LC_ALL=C sort \
	      | uniq -d); do bad=1; \
	    echo "error: more than one file is called $$name:" >&2; \
	    grep -e "/$$name\$$" -e "^$$name\$$" "$$layout_dir/found" \
	      | sed 's/^/  /' >&2; \
	  done; \
	  set -- $(FLAGS); \
	  while [ $$# -ge 3 ]; do echo "$$2 $$3"; shift 3; done \
	    | sed 's,^$(FLAMBDA2_DIR)/,,; s,^$(ROOT)/,../../../,' \
	    | LC_ALL=C sort > "$$layout_dir/flags"; \
	  sed -n 's/^-Q  *//p' _CoqProject | LC_ALL=C sort \
	    > "$$layout_dir/project"; \
	  if ! cmp -s "$$layout_dir/flags" "$$layout_dir/project"; then bad=1; \
	    echo "error: the lines of $(FLAMBDA2_DIR)/_CoqProject are not the" \
	      "flags of rocq/Makefile (< Makefile, > _CoqProject):" >&2; \
	    diff "$$layout_dir/flags" "$$layout_dir/project" | grep '^[<>]' >&2; \
	  fi; \
	  [ $$bad = 0 ]

# The review document, rocq/review.pdf: ROCQ-REVIEW-GUIDE.md, in parts, with
# every reviewed .v file (definitions and statements; the implementation and
# Proofs files are left out), the extraction directives and the axioms the
# statements rest on, typeset by rocq doc and LuaLaTeX (HACKING.md, "The
# review document").
review: $(SOURCES:.v=.vo)
	perl $(STDLIB_DIR)/review-pdf.pl $(STDLIB_DIR)/_review \
	  $(STDLIB_DIR)/review.pdf $(FLAGS) $(STDLIB) $(BACKEND) $(SSA) $(LOOP) \
	  $(EXTRACT) $(FLAMBDA2) $(FLAMBDA2_TYPES) $(FLAMBDA2_EXTRACT)

# Every statement a Spec file binds as a proposition ([Definition X_statement
# : Prop]) must be proved as the theorem [X] of the matching Proofs module
# (HACKING.md, "The layout of a model"). Each is checked by defining a
# constant of the statement's type whose body is the theorem; on success the
# claims checked are listed. The constant is named after the Spec file and
# the statement, so that two Spec files may each have a statement of one name.
# A statement is found by its first line, so [Definition X_statement : Prop]
# must be on one line: a Spec file with a [Definition] of a name ending in
# [_statement] that is written otherwise is an error, as such a statement
# would not be checked.
SPECS := $(patsubst %,OxCamlSsa:%,$(filter %Spec.v,$(SSA))) \
  $(patsubst %,OxCamlLoop:%,$(filter %Spec.v,$(LOOP))) \
  $(patsubst %,OxCamlFlambda2:%,$(filter %Spec.v,$(FLAMBDA2))) \
  $(patsubst %,OxCamlFlambda2Types:%,$(filter %Spec.v,$(FLAMBDA2_TYPES)))

claims: $(SOURCES:.v=.vo)
	@set -eu; \
	  claims_dir=$$(mktemp -d); \
	  trap 'rm -rf "$$claims_dir"' 0; \
	  : > "$$claims_dir/list"; \
	  for entry in $(SPECS); do \
	    lib=$${entry%%:*}; spec=$${entry#*:}; \
	    module=$$(basename $$spec .v); proofs=$${module%Spec}Proofs; \
	    if grep -n '^Definition [^ ]*_statement\( \|$$\)' $$spec \
	        | grep -v '^[0-9]*:Definition [A-Za-z0-9_]*_statement : Prop' \
	        >&2; then \
	      echo "$$spec: a statement that make claims would not check;" \
	        "write [Definition X_statement : Prop] on one line" >&2; \
	      exit 1; \
	    fi; \
	    for claim in $$(sed -n \
	        's/^Definition \([A-Za-z0-9_]*\)_statement : Prop.*/\1/p' $$spec); \
	    do \
	      printf 'From %s Require %s %s.\n' $$lib $$module $$proofs; \
	      printf 'Definition %s_%s_claim : %s.%s_statement := %s.%s.\n' \
	        $$module $$claim $$module $$claim $$proofs $$claim; \
	      printf '%s.%s proves %s.%s_statement\n' \
	        $$proofs $$claim $$module $$claim >> "$$claims_dir/list"; \
	    done; \
	  done > "$$claims_dir/Claims.v"; \
	  $(ROCQ) repl -q -batch $(FLAGS) -l "$$claims_dir/Claims.v"; \
	  cat "$$claims_dir/list"

# The checker reports global assumptions, not theorem premises. Constant
# types also expose Section arguments and contract fields,
# including declarations inside nested modules.
#
# The axioms are checked against the trust ledger (TRUST-LEDGER.md), and the
# target fails if they differ: an axiom that Rocq reports must be listed
# there, and an axiom listed there must be reported. An axiom is listed by a
# list item beginning with its fully qualified name between backquotes.
# Then every statement of the Flambda 2 libraries (the theorems that claims
# checks) must depend only on axioms of the ledger's section "Flambda 2".
# Print Assumptions names an axiom by the shortest name that is not
# ambiguous, so one is taken to be listed if a listed name ends with it.
LEDGER := $(STDLIB_DIR)/TRUST-LEDGER.md
FLAMBDA2_SPECS := $(patsubst %,OxCamlFlambda2:%,$(filter %Spec.v,$(FLAMBDA2))) \
  $(patsubst %,OxCamlFlambda2Types:%,$(filter %Spec.v,$(FLAMBDA2_TYPES)))

assumptions: all
	@printf '%s\n' '=== Global assumptions (including dependencies) ==='
	@set -eu; \
	  ledger_dir=$$(mktemp -d); \
	  trap 'rm -rf "$$ledger_dir"' 0; \
	  cd "$$ledger_dir"; \
	  $(ROCQ) check -silent -o $(FLAGS) $(MODULES) > check 2>&1 \
	    || { cat check; exit 1; }; \
	  cat check; \
	  sed -n '/^\* Axioms:/,/^\* /p' check \
	    | sed -n 's/^    \([^ ][^ ]*\)$$/\1/p' | LC_ALL=C sort -u > reported; \
	  sed -n 's/^- `\([A-Za-z0-9_.'"'"']*\)`.*/\1/p' $(LEDGER) \
	    | LC_ALL=C sort -u > listed; \
	  LC_ALL=C comm -23 reported listed > unlisted; \
	  LC_ALL=C comm -13 reported listed > stale; \
	  printf '\n%s\n' '=== The axioms against the trust ledger ==='; \
	  if [ -s unlisted ]; then \
	    echo "error: axioms that $(LEDGER) does not list:" >&2; \
	    sed 's/^/  /' unlisted >&2; \
	  fi; \
	  if [ -s stale ]; then \
	    echo "error: axioms that $(LEDGER) lists and Rocq does not report:" \
	      >&2; \
	    sed 's/^/  /' stale >&2; \
	  fi; \
	  [ ! -s unlisted ] && [ ! -s stale ] || exit 1; \
	  echo "The $$(wc -l < reported | tr -d ' ') axioms reported are those of" \
	    "the trust ledger."; \
	  for entry in $(FLAMBDA2_SPECS); do \
	    lib=$${entry%%:*}; spec=$${entry#*:}; \
	    module=$$(basename $$spec .v); proofs=$${module%Spec}Proofs; \
	    for claim in $$(sed -n \
	        's/^Definition \([A-Za-z0-9_]*\)_statement : Prop.*/\1/p' $$spec); \
	    do \
	      printf 'From %s Require %s.\nPrint Assumptions %s.%s.\n' \
	        $$lib $$proofs $$proofs $$claim; \
	    done; \
	  done > Statements.v; \
	  $(ROCQ) repl -q -batch $(FLAGS) -l Statements.v > statements 2>&1 \
	    || { cat statements; exit 1; }; \
	  sed -n 's/^\([A-Za-z_][A-Za-z0-9_.'"'"']*\)\( :.*\)\{0,1\}$$/\1/p' \
	    statements | grep -v '^Axioms$$' | LC_ALL=C sort -u > used; \
	  sed -n '/^## Flambda 2$$/,/^## /p' $(LEDGER) \
	    | sed -n 's/^- `\([A-Za-z0-9_.'"'"']*\)`.*/\1/p' > flambda2; \
	  : > foreign; \
	  while read -r axiom; do \
	    grep -q -x -e "$$axiom" -e ".*\.$$axiom" flambda2 \
	      || echo "$$axiom" >> foreign; \
	  done < used; \
	  if [ -s foreign ]; then \
	    echo "error: statements of the Flambda 2 libraries depend on axioms" \
	      "that are not in the section \"Flambda 2\" of $(LEDGER):" >&2; \
	    sed 's/^/  /' foreign >&2; \
	    exit 1; \
	  fi; \
	  echo "The statements of the Flambda 2 libraries depend on" \
	    "$$(wc -l < used | tr -d ' ') of the $$(wc -l < flambda2 | tr -d ' ')" \
	    "axioms of the ledger's section \"Flambda 2\", and on no other."
	@printf '\n%s\n' '=== Declaration types and contract fields ===' \
	  'Section hypotheses appear as arguments in these types.' \
	  'This section includes proved theorems, not just assumptions.'
	@set -eu; \
	  audit_dir=$$(mktemp -d); \
	  trap 'rm -rf "$$audit_dir"' 0; \
	  { printf '%s\n' 'Set Printing Width 78.' \
	      'Set Printing Depth 1000000.'; \
	    for module in $(MODULES); do \
	      printf 'Require %s.\nPrint Namespace %s.\n' "$$module" "$$module"; \
	    done; \
	  } > "$$audit_dir/Assumptions.v"; \
	  $(ROCQ) repl -q -batch $(FLAGS) -l "$$audit_dir/Assumptions.v"

# Extract.v writes the .ml and .mli files into the loop directory. Separate
# extraction also writes an empty pair for every Rocq module whose contents
# are all mapped onto OxCaml's own (OxCamlInt, Uint63, ...); those are removed.
# It also puts the same `open M` lines at the top of a .ml and of its .mli;
# in the .mli only types can be referred to, so the open of a module that
# defines no types or submodules (OxCamlList) is unused there, which warning
# 33 rejects. Such opens are removed from the .mli files.
$(EXTRACT:.v=.vo): $(EXTRACT)
	$(ROCQ) compile $(FLAGS) -output-directory $(LOOP_DIR) $<
	for f in $(LOOP_DIR)/*.ml $(LOOP_DIR)/*.mli; do \
	  grep -q '[^[:space:]]' $$f || rm -f $$f; done
	mv $(foreach m, $(SSA_MODULES), $(LOOP_DIR)/$(m).ml $(LOOP_DIR)/$(m).mli) \
	  $(SSA_DIR)/
	@for f in $(LOOP_DIR)/*.ml $(LOOP_DIR)/*.mli; do \
	  case " $(LOOP_EXTRACTED) " in *" $$f "*) ;; *) \
	    echo "error: unexpected extracted file $$f: a model computes with" \
	      "something that has no OxCaml realization (see OxCamlExtract.v)"; \
	    exit 1;; \
	  esac; \
	done
	for f in $(LOOP_DIR)/*.mli $(SSA_DIR)/*.mli; do \
	  for m in `sed -n 's/^open \(.*\)$$/\1/p' $$f`; do \
	    cat $(LOOP_DIR)/$$m.mli $(SSA_DIR)/$$m.mli 2>/dev/null \
	      | grep -q -E '^ *(type|module) ' || \
	      { grep -v "^open $$m$$" $$f > $$f.tmp && mv $$f.tmp $$f; }; \
	  done; \
	done
	$(OCAMLFORMAT) -i $(EXTRACTED)

# FlambdaExtract.v writes the machine of the Flambda 2 semantics as OCaml
# modules into the directory of the test tool that runs it
# (middle_end/flambda2/tests/semantics/extracted/); nothing there is written
# by hand but the dune file.
$(FLAMBDA2_EXTRACT:.v=.vo): $(FLAMBDA2_EXTRACT)
	cd $(FLAMBDA2_EXTRACTED_DIR) && $(ROCQ) compile $(FLAGS) $<
	$(OCAMLFORMAT) -i $(FLAMBDA2_EXTRACTED_DIR)/*.ml \
	  $(FLAMBDA2_EXTRACTED_DIR)/*.mli

%.vo: %.v
	$(ROCQ) compile $(FLAGS) $<

# The dependencies between the .v files, computed by rocq dep. They are
# written to another file, which is renamed when rocq dep has succeeded:
# a run that fails (rocq is not on the path, say) must not leave an empty
# file that is newer than the sources, as later builds would take it for
# the dependencies and rebuild nothing. They depend on this file too: a
# file added to one of the lists above may be older than the dependencies
# computed before it was listed, and would then be compiled without its
# own.
$(DEPS): $(SOURCES) $(firstword $(MAKEFILE_LIST))
	$(ROCQ) dep $(FLAGS) $(SOURCES) > $@.tmp || { rm -f $@.tmp; exit 1; }
	mv $@.tmp $@

-include $(DEPS)

clean:
	rm -f $(DEPS) $(DEPS).tmp $(EXTRACTED) $(STDLIB_DIR)/review.pdf
	rm -rf $(STDLIB_DIR)/_review
	for d in $(STDLIB_DIR) $(BACKEND_DIR) $(SSA_DIR) $(LOOP_DIR) \
	    $$(find $(FLAMBDA2_DIR) -name .coq-native -prune -o -type d -print); \
	  do \
	  rm -rf $$d/*.vo $$d/*.vos $$d/*.vok $$d/*.glob $$d/.*.aux \
	    $$d/.lia.cache $$d/.nia.cache $$d/.coq-native; \
	done
