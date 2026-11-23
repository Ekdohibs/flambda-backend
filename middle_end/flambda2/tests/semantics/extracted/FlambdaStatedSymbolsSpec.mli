open Datatypes
open FlambdaKindModel
open FlambdaMapSpec
open FlambdaModel
open FlambdaStatedScopeSpec
open FlambdaSubkindOrderSpec

type symbols_stated = (symbol * param_subkind) list

val stated_of_symbol : symbols_stated -> symbol -> param_subkind

val static_field_stated : symbols_stated -> simple -> param_subkind option

val static_fields_stated :
  symbols_stated -> simple list -> field_subkinds option

val stated_of_static_const : symbols_stated -> static_const -> param_subkind

val statics_stated : static_binding list -> symbols_stated -> symbols_stated

val is_stated_knowing :
  symbols_stated -> stated_in_scope -> param_subkind -> simple -> bool

val are_stated_knowing :
  symbols_stated -> stated_in_scope -> simple list -> param_subkind list -> bool
