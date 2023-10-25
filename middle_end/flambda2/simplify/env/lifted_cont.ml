(**************************************************************************)
(*                                                                        *)
(*                                 OCaml                                  *)
(*                                                                        *)
(*                        Guillaume Bury, OCamlPro                        *)
(*                                                                        *)
(*   Copyright 2023--2023 OCamlPro SAS                                    *)
(*   Copyright 2023--2023 Jane Street Group LLC                           *)
(*                                                                        *)
(*   All rights reserved.  This file is distributed under the terms of    *)
(*   the GNU Lesser General Public License version 2.1, with the          *)
(*   special exception on linking described in the file LICENSE.          *)
(*                                                                        *)
(**************************************************************************)

type one_recursive_handler =
  { params : Bound_parameters.t;
    handler : Flambda.Expr.t;
    is_cold : bool
  }

type non_recursive_handler =
  { cont : Continuation.t;
    params : Bound_parameters.t;
    lifted_params : Lifted_cont_params.t;
    handler : Flambda.Expr.t;
    is_exn_handler : bool;
    is_cold : bool;
  }

type original_handlers =
  | Recursive of
      { invariant_params : Bound_parameters.t;
        lifted_params : Lifted_cont_params.t;
        continuation_handlers : one_recursive_handler Continuation.Map.t
      }
  | Non_recursive of non_recursive_handler

let print_one_recursive_handler ppf ({ params; handler; is_cold; } : one_recursive_handler) =
  Format.fprintf ppf
    "@[<hov 1>(\
     @[<hv 1>(params@ %a)@]@ \
     @[<hv 1>(is_cold@ %b)@]@ \
     @[<hv 1>(handler@ %a)@]\
     )@]"
    Bound_parameters.print params is_cold
    Flambda.Expr.print handler

let print_non_recursive_handler ppf
    { cont; params; lifted_params; handler; is_exn_handler; is_cold; } =
  Format.fprintf ppf
    "@[<hov 1>(\
     @[<hv 1>(cont@ %a)@]@ \
     @[<hv 1>(params@ %a)@]@ \
     @[<hv 1>(lifted_params@ %a)@]@ \
     @[<hv 1>(is_exn_handler@ %b)@]@ \
     @[<hv 1>(is_cold@ %b)@]@ \
     @[<hv 1>(handler@ %a)@]@ \
     )@]"
    Continuation.print cont
    Bound_parameters.print params
    Lifted_cont_params.print lifted_params
    is_exn_handler is_cold
    Flambda.Expr.print handler

let print_original_handlers ppf original_handlers =
  match (original_handlers : original_handlers) with
  | Non_recursive non_rec_handler ->
    print_non_recursive_handler ppf non_rec_handler
  | Recursive { invariant_params; lifted_params; continuation_handlers; } ->
    Format.fprintf ppf
      "@[<hov 1>(\
       @[<hv 1>(invariant params@ %a)@]@ \
       @[<hv 1>(lifted_params@ %a)@]@ \
       @[<hv 1>(continuation handlers@ %a)@]\
       )@]"
      Bound_parameters.print invariant_params
      Lifted_cont_params.print lifted_params
      (Continuation.Map.print print_one_recursive_handler) continuation_handlers
