open BinInt
open BinNums
open BinPos
open Datatypes
open FlambdaMapSpec
open FlambdaModel
open FlambdaPrimitiveModel
open List
open ListDef
open OxCamlOption

type static_place =
  | Place_slot of value_slot
  | Place_field of coq_Z

val same_place : static_place -> static_place -> bool

val place_held : static_const -> static_place -> simple option

val const_places : static_const -> static_place list

type place_class = symbol * variable

val same_class : place_class -> place_class -> bool

type held_places = ((symbol * static_place) * place_class) list

val first_symbol : static_binding list -> symbol option

val place_entry :
  symbol -> symbol -> static_const -> static_place -> held_places

val binding_places : symbol -> static_binding -> held_places

val group_places : static_binding list -> held_places

val places_held : expr -> held_places

val places_held_of_handler : continuation_handler -> held_places

val group_const : symbol -> static_binding list -> static_const option

val entry_meets_group :
  static_binding list -> (symbol * static_place) * place_class -> bool

val entry_of_group :
  static_binding list -> (symbol * static_place) * place_class -> bool

val group_holds : held_places -> static_binding list -> bool

val groups_hold : held_places -> expr -> bool

val handler_groups_hold : held_places -> continuation_handler -> bool

val code_of_binding_groups_hold : held_places -> static_binding -> bool

val code_groups_hold : held_places -> code -> bool

val read_of : named -> (symbol * static_place) option

val class_of : held_places -> symbol -> static_place -> place_class option

val read_class : held_places -> named -> place_class option

type reads_made = (place_class * variable) list

val forget_read : variable -> reads_made -> reads_made

val forget_reads : variable list -> reads_made -> reads_made

val read_before : reads_made -> place_class -> variable option

val share_reads : held_places -> reads_made -> expr -> expr

val share_reads_in_handler :
  held_places -> reads_made -> continuation_handler -> continuation_handler

val share_reads_in_binding : held_places -> static_binding -> static_binding

val share_reads_in_code : held_places -> code -> code

val share_static_reads : expr -> expr
