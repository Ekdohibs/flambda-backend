open FlambdaModel
open FlambdaNormaliseSpec
open FlambdaStatedScopeSpec
open ListDef

val rewrite_rec_handlers_stated :
  (stated_in_scope -> expr -> expr) -> stated_in_scope -> expr -> expr

val rewrite_rec_handlers_stated_in_handler :
  (stated_in_scope -> expr -> expr) ->
  stated_in_scope ->
  continuation_handler ->
  continuation_handler

val rewrite_rec_handler_stated :
  (stated_in_scope -> expr -> expr) ->
  stated_in_scope ->
  continuation_handler ->
  continuation_handler
