open BinNums
open BinPos
open Bool
open Datatypes
open FlambdaApplySpec
open FlambdaCompareSpec
open FlambdaKindModel
open FlambdaMapSpec
open FlambdaModel
open FlambdaNominalSpec
open FlambdaPrimitiveModel
open FlambdaPrimitiveSpec
open FlambdaSemanticsSpec
open FlambdaStatedScopeSpec
open FlambdaStatedSymbolsSpec
open FlambdaTupledSpec
open FlambdaValueSpec
open List
open ListDef
open PeanoNat

val simple_in_scope : variable list -> simple -> bool

val simples_in_scope : variable list -> simple list -> bool

val named_in_scope : variable list -> named -> bool

val static_binding_in_scope : variable list -> static_binding -> bool

val apply_cont_in_scope :
  variable list -> continuation list -> apply_cont -> bool

val expr_in_scope : variable list -> continuation list -> expr -> bool

val handler_in_scope :
  variable list -> continuation list -> continuation_handler -> bool

val inline_renaming :
  continuation -> continuation -> continuation -> continuation -> renaming

val var_in_simples : variable -> simple list -> bool

val inlinable_b : apply_expr -> code -> bool

val tupled_inlinable_b : apply_expr -> code -> bool

val over_applied_b : apply_expr -> code -> bool

val split_over_application : code_id -> code -> apply_expr -> expr

val codes_of :
  static_binding list -> (code_id * code) list -> (code_id * code) list

val inline_apply : code_id list -> (code_id * code) list -> apply_expr -> expr

val inline_calls : code_id list -> (code_id * code) list -> expr -> expr

val inline_calls_in_handler :
  code_id list ->
  (code_id * code) list ->
  continuation_handler ->
  continuation_handler

val inline_calls_in_binding :
  code_id list -> (code_id * code) list -> static_binding -> static_binding

val inline_calls_in_code : code_id list -> (code_id * code) list -> code -> code

val var_in_simple : variable -> simple -> bool

val var_in_value_slots : variable -> (value_slot * simple) list -> bool

val var_in_named : variable -> named -> bool

val var_in_static_binding : variable -> static_binding -> bool

val var_free_b : variable -> expr -> bool

val var_free_in_handler_b : variable -> continuation_handler -> bool

val prim_may_fold : prim -> bool

val named_may_go : named -> bool

val remove_unused_lets : expr -> expr

val remove_unused_lets_in_handler : continuation_handler -> continuation_handler

val inline_cont_at_tail :
  continuation ->
  variable list ->
  handler_metadata ->
  expr ->
  expr ->
  expr option

val inline_tail_conts : expr -> expr

val binds_something : static_binding -> bool

val let_static_if_any : static_binding list -> expr -> expr

val drop_deleted_code : expr -> expr

val consts_of_simples : simple list -> const list option

val fold_named : named -> named

val fold_switch : simple -> (coq_Z * apply_cont) list -> expr

val fold_constants : expr -> expr

val named_is_total : flambda_kind -> named -> bool

val remove_unused_total_lets : expr -> expr

type kinds_in_scope = (variable * flambda_kind) list

val kind_of_simple_in_scope : kinds_in_scope -> simple -> flambda_kind option

val named_defined : kinds_in_scope -> flambda_kind -> named -> bool

val params_in_scope :
  variable list -> flambda_kind list -> kinds_in_scope -> kinds_in_scope

val remove_unused_defined_lets : kinds_in_scope -> expr -> expr

val symbols_of :
  static_binding list ->
  (symbol * code_id option) list ->
  (symbol * code_id option) list

val call_agrees_stated :
  symbols_stated -> stated_in_scope -> apply_expr -> code -> bool

val inline_known_apply :
  stated_in_scope ->
  code_id list ->
  (code_id * code) list ->
  (symbol * code_id option) list ->
  symbols_stated ->
  apply_expr ->
  expr

val inline_known_calls_under :
  stated_in_scope ->
  code_id list ->
  (code_id * code) list ->
  (symbol * code_id option) list ->
  symbols_stated ->
  expr ->
  expr

val inline_known_calls_in_handler_under :
  stated_in_scope ->
  code_id list ->
  (code_id * code) list ->
  (symbol * code_id option) list ->
  symbols_stated ->
  continuation_handler ->
  continuation_handler

val inline_known_calls_in_binding_knowing :
  code_id list ->
  (code_id * code) list ->
  (symbol * code_id option) list ->
  symbols_stated ->
  static_binding ->
  static_binding

val inline_known_calls_in_code_knowing :
  code_id list ->
  (code_id * code) list ->
  (symbol * code_id option) list ->
  symbols_stated ->
  code ->
  code

val inline_known_calls_in_binding :
  code_id list ->
  (code_id * code) list ->
  (symbol * code_id option) list ->
  static_binding ->
  static_binding

val inline_known_calls :
  code_id list ->
  (code_id * code) list ->
  (symbol * code_id option) list ->
  expr ->
  expr

val binds_code : static_binding -> bool

val code_first : static_binding list -> static_binding list

val merge_statics : expr -> expr

val simple_eq_dec : simple -> simple -> bool

val trap_action_eq_dec : trap_action -> trap_action -> bool

val apply_cont_eq_dec : apply_cont -> apply_cont -> bool

val collapse_switch : simple -> (coq_Z * apply_cont) list -> expr

val collapse_switches : expr -> expr

val rewrite_code_body : (expr -> expr) -> expr -> expr

val rewrite_code_in_binding : (expr -> expr) -> static_binding -> static_binding

val rewrite_code : (expr -> expr) -> expr -> expr

val rewrite_code_in_handler :
  (expr -> expr) -> continuation_handler -> continuation_handler

val thread_arm :
  continuation ->
  (coq_Z * apply_cont) list ->
  coq_Z * apply_cont ->
  coq_Z * apply_cont

val arms_apply : continuation -> (coq_Z * apply_cont) list -> bool

val thread_switch : continuation -> continuation_handler -> expr -> expr

val thread_switches : expr -> expr
