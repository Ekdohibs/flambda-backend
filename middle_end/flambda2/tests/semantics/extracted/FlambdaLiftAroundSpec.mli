open BinNums
open BinPos
open Bool
open Datatypes
open FlambdaBindersSpec
open FlambdaHeapFreeSpec
open FlambdaLiftSpec
open FlambdaMapSpec
open FlambdaModel
open FlambdaSemanticsSpec
open FlambdaWellFormedSpec
open List
open ListDef

type known_constants = (symbol * static_const) list

val constants_of_group : static_binding list -> known_constants

val constant_known : known_constants -> static_binding -> bool

val forget : variable list -> known_constants -> known_constants

val alias_of :
  (path * lift_to) list ->
  path ->
  named ->
  (static_binding list * symbol) option

val group_at : (path * static_binding list) list -> path -> static_binding list

val lift_around :
  (path * lift_to) list ->
  (path * static_binding list) list ->
  path ->
  expr ->
  expr

val lift_around_handler :
  (path * lift_to) list ->
  (path * static_binding list) list ->
  path ->
  continuation_handler ->
  continuation_handler

val total_const : static_const -> bool

val putable_group : static_binding list -> bool

val group_in_scope : static_binding list -> expr -> bool

val around_ok :
  (path * lift_to) list ->
  (path * static_binding list) list ->
  known_constants ->
  path ->
  expr ->
  bool

val handler_around_ok :
  (path * lift_to) list ->
  (path * static_binding list) list ->
  known_constants ->
  path ->
  continuation_handler ->
  bool

val bindings_of_lets :
  (path * lift_to) list -> path list -> path -> expr -> static_binding list

val handler_bindings_of_lets :
  (path * lift_to) list ->
  path list ->
  path ->
  continuation_handler ->
  static_binding list

val groups_of :
  (path * lift_to) list ->
  (path * path list) list ->
  expr ->
  (path * static_binding list) list

val lift_around_unit :
  unit_interface ->
  (path * lift_to) list ->
  (path * path list) list ->
  expr ->
  expr option
