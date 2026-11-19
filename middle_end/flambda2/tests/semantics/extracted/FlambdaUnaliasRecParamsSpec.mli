open Datatypes
open FlambdaMapSpec
open FlambdaModel
open FlambdaStatedScopeSpec
open FlambdaUnaliasParamsSpec
open List

val unalias_rec_param :
  stated_in_scope ->
  nat ->
  continuation ->
  continuation_handler ->
  expr ->
  (continuation_handler * expr) option

val unalias_rec_from :
  stated_in_scope ->
  nat ->
  continuation ->
  continuation_handler ->
  expr ->
  continuation_handler * expr

val unalias_rec_params_under : stated_in_scope -> expr -> expr
