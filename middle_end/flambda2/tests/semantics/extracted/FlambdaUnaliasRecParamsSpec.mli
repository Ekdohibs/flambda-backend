open Datatypes
open FlambdaMapSpec
open FlambdaModel
open FlambdaUnaliasParamsSpec
open List

val unalias_rec_param :
  nat ->
  continuation ->
  continuation_handler ->
  expr ->
  (continuation_handler * expr) option

val unalias_rec_from :
  nat ->
  continuation ->
  continuation_handler ->
  expr ->
  continuation_handler * expr

val unalias_rec_params : expr -> expr
