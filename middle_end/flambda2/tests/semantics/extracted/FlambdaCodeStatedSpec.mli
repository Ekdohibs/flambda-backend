open FlambdaModel
open FlambdaNormaliseSpec
open FlambdaStatedScopeSpec
open ListDef

val rewrite_code_stated_in_binding :
  (stated_in_scope -> expr -> expr) -> static_binding -> static_binding

val rewrite_code_stated : (stated_in_scope -> expr -> expr) -> expr -> expr

val rewrite_code_stated_in_handler :
  (stated_in_scope -> expr -> expr) ->
  continuation_handler ->
  continuation_handler
