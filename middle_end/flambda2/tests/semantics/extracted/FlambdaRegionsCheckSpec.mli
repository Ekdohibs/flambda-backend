open BinPos
open FlambdaKindModel
open FlambdaMapSpec
open FlambdaModel

type lifetime =
  | Forever
  | Of_caller
  | Of_region of variable
  | Gone

val regions_below : variable -> variable list -> variable list

val in_order : variable list -> variable list -> bool

type handler_entry =
  { he_open : variable list;
    he_params : lifetime list
  }

type code_entry =
  { ce_params_local : bool list;
    ce_closure_local : bool;
    ce_returns_local : bool
  }

type certificate =
  { cert_handlers : (continuation * handler_entry) list;
    cert_code : (code_id * code_entry) list;
    cert_indirect_local : bool;
    cert_unknown_returns_local : bool;
    cert_partial_local : bool
  }

val region_at_jump : (variable * variable) list -> variable -> variable

val region_params_of : variable list -> flambda_kind list -> variable list
