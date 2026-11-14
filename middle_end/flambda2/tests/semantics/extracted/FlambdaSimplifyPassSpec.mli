open BinInt
open BinNums
open BinPos
open Datatypes
open FlambdaBindingTime
open FlambdaHeapSpec
open FlambdaKindModel
open FlambdaMapSpec
open FlambdaModel
open FlambdaNormaliseSpec
open FlambdaPrimitiveModel
open FlambdaPrimitiveSpec
open FlambdaProvers
open FlambdaSemanticsSpec
open FlambdaSimplifyApplySpec
open FlambdaSimplifyArithSpec
open FlambdaSimplifyPrimitivesSpec
open FlambdaSimplifyProjectionsSpec
open FlambdaSimplifyValueSlotsSpec
open FlambdaTypeGrammar
open FlambdaTypingEnv
open FlambdaTypingEnvInvariantSpec
open FlambdaUnusedContsSpec
open List
open ListDef
open OxCamlOption
open PeanoNat

type outcome =
  | Becomes_invalid
  | Becomes_simple of simple
  | Stays_with_type of flambda_type
  | Stays_with_kind of flambda_kind
  | Not_known

val naked_bool_const : bool -> simple

val of_bool_shortcut : bool Meet_shortcut.t option -> outcome -> outcome

val of_simple_shortcut : simple Meet_shortcut.t option -> outcome -> outcome

val simplify_unary :
  Typing_env.t -> unary_primitive -> simple -> flambda_type -> outcome

type one_side_outcome =
  | Exactly of coq_Z
  | The_other_side
  | No_result
  | Cannot_simplify

val symmetric_op_one_side_unknown :
  binary_int_arith_op -> coq_Z -> one_side_outcome

val op_lhs_unknown : binary_int_arith_op -> coq_Z -> one_side_outcome

val op_rhs_unknown : binary_int_arith_op -> coq_Z -> one_side_outcome

type possible_result =
  | Result_exactly of coq_Z
  | Result_other_side

val possible_result_eqb : possible_result -> possible_result -> bool

val one_side_results :
  (coq_Z -> one_side_outcome) -> coq_Z list -> possible_result list option

val both_sides_results :
  binary_int_arith_op -> coq_Z list -> coq_Z list -> possible_result list

val exact_results : possible_result list -> coq_Z list

val check_possible_results : possible_result list -> simple -> outcome

val max_num_possible_results : nat

val simplify_int_arith_tagged :
  Typing_env.t ->
  binary_int_arith_op ->
  simple ->
  flambda_type ->
  simple ->
  flambda_type ->
  outcome

val comparison_unknown : int_comparison_behaviour -> outcome

val simplify_int_comp_tagged :
  Typing_env.t ->
  int_comparison_behaviour ->
  flambda_type ->
  flambda_type ->
  outcome

val simplify_phys_equal :
  Typing_env.t -> equality_comparison -> flambda_type -> flambda_type -> outcome

val simplify_binary :
  Typing_env.t ->
  binary_primitive ->
  simple ->
  flambda_type ->
  simple ->
  flambda_type ->
  outcome

val simplify_variadic : variadic_primitive -> flambda_type list -> outcome

val simplify_named : Typing_env.t -> named -> outcome

val canonical_named : Typing_env.t -> named -> named

type let_step =
  | Step_invalid
  | Step_let of named * Typing_env.t
  | Step_stop

val simple_in_terms_b : Typing_env.t -> simple -> bool

val simple_of_kind_b : Typing_env.t -> simple -> flambda_kind -> bool

val bind_simple : Typing_env.t -> variable -> simple -> let_step

val type_in_scope_b : Typing_env.t -> flambda_type -> bool

val resolve_value_slot : Typing_env.t -> flambda_kind -> named -> named

val simplify_apply_cont : Typing_env.t -> apply_cont -> apply_cont

val simplify_switch :
  Typing_env.t -> simple -> (coq_Z * apply_cont) list -> expr

val arm_env : Typing_env.t -> simple -> coq_Z -> Typing_env.t

val not_among : variable list -> simple -> bool

val leaf_uses :
  Typing_env.t ->
  continuation ->
  expr ->
  (Typing_env.t * simple list) list option

val binders_not_free :
  variable list -> continuation list -> continuation_handler -> bool
