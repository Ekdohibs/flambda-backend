open Datatypes
open FlambdaKindModel
open FlambdaMapSpec
open FlambdaModel
open FlambdaNormaliseSpec
open FlambdaPrimitiveModel
open FlambdaUnboxParamsSpec
open FlambdaUnboxVariantsSpec
open FlambdaUntagParamsSpec
open List
open ListDef

(** val shown_number : boxable_number -> variable list -> facts -> simple ->
    simple option **)

let shown_number boxed unboxed known s =
  match
    certain_projection known (Unbox_number boxed) (unboxed_kind boxed) s
  with
  | Some number ->
    if none_in unboxed (fun p -> var_in_simple p number)
    then Some number
    else None
  | None -> None

(** val unboxed_simple : boxable_number -> variable list -> facts -> simple ->
    simple option **)

let unboxed_simple boxed unboxed known s =
  match s with
  | Var x ->
    if among x unboxed then Some s else shown_number boxed unboxed known s
  | _ -> shown_number boxed unboxed known s

(** val unbox_arg : boxable_number -> variable list -> facts -> simple -> simple
    **)

let unbox_arg boxed unboxed known s =
  match unboxed_simple boxed unboxed known s with Some s' -> s' | None -> s

(** val together_args : boxable_number -> nat list -> variable list -> facts ->
    simple list -> simple list **)

let together_args boxed us unboxed known args =
  app (kept_from us O args)
    (rev (map (unbox_arg boxed unboxed known) (moved_from us O args)))

(** val together_apply_cont : boxable_number -> positions -> variable list ->
    facts -> apply_cont -> apply_cont **)

let together_apply_cont boxed active unboxed known ac =
  { ac_continuation = ac.ac_continuation;
    ac_args =
      together_args boxed
        (positions_of active ac.ac_continuation)
        unboxed known ac.ac_args;
    ac_trap_action = ac.ac_trap_action
  }

(** val unbox_of : boxable_number -> named -> variable option **)

let unbox_of boxed = function
  | Prim (p0, args) -> (
    match p0 with
    | Unary p1 -> (
      match p1 with
      | Unbox_number boxed' -> (
        match args with
        | [] -> None
        | s :: l -> (
          match s with
          | Var p -> (
            match l with
            | [] ->
              if kind_eqb (unboxed_kind boxed') (unboxed_kind boxed)
              then Some p
              else None
            | _ :: _ -> None)
          | _ -> None))
      | _ -> None)
    | _ -> None)
  | _ -> None

(** val together_named : boxable_number -> variable list -> named -> named **)

let together_named boxed unboxed d =
  match unbox_of boxed d with
  | Some p -> if among p unboxed then Simple (Var p) else d
  | None -> d

(** val together_metadata : boxable_number -> nat list -> handler_metadata ->
    handler_metadata **)

let together_metadata boxed us metadata =
  { hm_params_kinds =
      app
        (kept_from us O metadata.hm_params_kinds)
        (map
           (fun _ -> unboxed_kind boxed)
           (rev (moved_from us O metadata.hm_params_kinds)));
    hm_params_subkinds =
      app
        (kept_from us O (stated_of_params metadata))
        (map
           (fun _ -> any_value)
           (rev (moved_from us O (stated_of_params metadata))));
    hm_is_exn_handler = metadata.hm_is_exn_handler
  }

(** val together_expr : boxable_number -> positions -> positions -> variable
    list -> facts -> expr -> expr **)

let rec together_expr boxed sites active unboxed known e =
  match e with
  | Let (x, kind, d, body) ->
    Let
      ( x,
        kind,
        together_named boxed unboxed d,
        together_expr boxed sites active
          (without (x :: []) unboxed)
          (facts_under_let known x d)
          body )
  | Let_static (bindings, body) ->
    Let_static
      ( bindings,
        together_expr boxed sites active unboxed
          (facts_under_statics known bindings)
          body )
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive
      ( k,
        together_handler boxed sites active unboxed known (positions_of sites k)
          h,
        together_expr boxed sites
          ((k, positions_of sites k) :: active)
          unboxed known body )
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive
      ( map
          (fun pat ->
            let k, h = pat in
            ( k,
              together_handler boxed sites
                (active_under sites handlers active)
                unboxed known (positions_of sites k) h ))
          handlers,
        together_expr boxed sites
          (active_under sites handlers active)
          unboxed known body )
  | Apply_cont ac ->
    Apply_cont (together_apply_cont boxed active unboxed known ac)
  | Switch (scrutinee, arms) ->
    Switch
      ( scrutinee,
        map
          (fun pat ->
            let i, ac = pat in
            i, together_apply_cont boxed active unboxed known ac)
          arms )
  | _ -> e

(** val together_handler : boxable_number -> positions -> positions -> variable
    list -> facts -> nat list -> continuation_handler -> continuation_handler **)

and together_handler boxed sites active unboxed known us = function
  | Handler (params, metadata, handler) ->
    Handler
      ( untag_params_list us params,
        together_metadata boxed us metadata,
        together_expr boxed sites active
          (untagged_under us params unboxed)
          (facts_under_params known params)
          handler )

(** val unboxed_arg : boxable_number -> variable list -> facts -> simple -> bool
    **)

let unboxed_arg boxed unboxed known s =
  match unboxed_simple boxed unboxed known s with
  | Some _ -> true
  | None -> false

(** val together_args_ok : boxable_number -> variable list -> facts -> nat list
    -> simple list -> bool **)

let together_args_ok boxed unboxed known us args =
  none_in unboxed (fun p -> var_in_simples p (kept_from us O args))
  && forallb (unboxed_arg boxed unboxed known) (moved_from us O args)

(** val together_apply_cont_ok : boxable_number -> positions -> variable list ->
    facts -> apply_cont -> bool **)

let together_apply_cont_ok boxed active unboxed known ac =
  together_args_ok boxed unboxed known
    (positions_of active ac.ac_continuation)
    ac.ac_args

(** val reads_unboxed : boxable_number -> variable list -> named -> bool **)

let reads_unboxed boxed unboxed d =
  match unbox_of boxed d with Some p -> among p unboxed | None -> false

(** val together_named_ok : boxable_number -> variable list -> named -> bool **)

let together_named_ok boxed unboxed d =
  reads_unboxed boxed unboxed d || none_in unboxed (fun p -> var_in_named p d)

(** val together_ok : boxable_number -> positions -> positions -> variable list
    -> facts -> expr -> bool **)

let rec together_ok boxed sites active unboxed known = function
  | Let (x, _, d, body) ->
    together_named_ok boxed unboxed d
    && together_ok boxed sites active
         (without (x :: []) unboxed)
         (facts_under_let known x d)
         body
  | Let_static (bindings, body) ->
    none_in unboxed (fun p -> existsb (var_in_static_binding p) bindings)
    && together_ok boxed sites active unboxed
         (facts_under_statics known bindings)
         body
  | Let_cont_non_recursive (k, h, body) ->
    together_handler_ok boxed sites active unboxed known (positions_of sites k)
      h
    && together_ok boxed sites
         ((k, positions_of sites k) :: active)
         unboxed known body
  | Let_cont_recursive (handlers, body) ->
    forallb
      (fun pat ->
        let k, h = pat in
        together_handler_ok boxed sites
          (active_under sites handlers active)
          unboxed known (positions_of sites k) h)
      handlers
    && together_ok boxed sites
         (active_under sites handlers active)
         unboxed known body
  | Apply ap ->
    (none_in unboxed (fun p ->
         (var_in_simple p ap.ap_callee || var_in_simples p ap.ap_args)
         || var_in_simples p (exn_extra_simples ap))
    && no_positions active ap.ap_continuation)
    && no_positions active ap.ap_exn_continuation
  | Apply_cont ac -> together_apply_cont_ok boxed active unboxed known ac
  | Switch (scrutinee, arms) ->
    none_in unboxed (fun p -> var_in_simple p scrutinee)
    && forallb
         (fun pat ->
           let _, ac = pat in
           together_apply_cont_ok boxed active unboxed known ac)
         arms
  | Invalid -> true

(** val together_handler_ok : boxable_number -> positions -> positions ->
    variable list -> facts -> nat list -> continuation_handler -> bool **)

and together_handler_ok boxed sites active unboxed known us = function
  | Handler (params, metadata, handler) ->
    handler_fits us params metadata
    && together_ok boxed sites active
         (untagged_under us params unboxed)
         (facts_under_params known params)
         handler

(** val together_at : boxable_number -> positions -> facts -> expr -> expr **)

let together_at boxed sites known e =
  if together_ok boxed sites [] [] known e
  then together_expr boxed sites [] [] known e
  else e

(** val together_args_blame : boxable_number -> origin_map -> facts ->
    continuation -> nat list -> nat -> simple list -> (continuation * nat) list
    **)

let rec together_args_blame boxed origin known k us i = function
  | [] -> []
  | a :: args0 ->
    app
      (if at_position us i
       then
         if unboxed_arg boxed (map fst origin) known a then [] else (k, i) :: []
       else origins origin (fun x -> var_in_simple x a))
      (together_args_blame boxed origin known k us (S i) args0)

(** val together_apply_cont_blame : boxable_number -> positions -> origin_map ->
    facts -> apply_cont -> (continuation * nat) list **)

let together_apply_cont_blame boxed active origin known ac =
  together_args_blame boxed origin known ac.ac_continuation
    (positions_of active ac.ac_continuation)
    O ac.ac_args

(** val together_blame : boxable_number -> positions -> positions -> origin_map
    -> facts -> expr -> (continuation * nat) list **)

let rec together_blame boxed sites active origin known = function
  | Let (x, _, d, body) ->
    app
      (if reads_unboxed boxed (map fst origin) d
       then []
       else origins origin (fun y -> var_in_named y d))
      (together_blame boxed sites active (unbind x origin)
         (facts_under_let known x d)
         body)
  | Let_static (bindings, body) ->
    app
      (origins origin (fun y -> existsb (var_in_static_binding y) bindings))
      (together_blame boxed sites active origin
         (facts_under_statics known bindings)
         body)
  | Let_cont_non_recursive (k, h, body) ->
    app
      (together_handler_blame boxed sites active origin known k h)
      (together_blame boxed sites
         ((k, positions_of sites k) :: active)
         origin known body)
  | Let_cont_recursive (handlers, body) ->
    app
      (flat_map
         (fun pat ->
           let k, h = pat in
           together_handler_blame boxed sites
             (active_under sites handlers active)
             origin known k h)
         handlers)
      (together_blame boxed sites
         (active_under sites handlers active)
         origin known body)
  | Apply ap ->
    app
      (origins origin (fun y ->
           (var_in_simple y ap.ap_callee || var_in_simples y ap.ap_args)
           || var_in_simples y (exn_extra_simples ap)))
      (app
         (all_of active ap.ap_continuation)
         (all_of active ap.ap_exn_continuation))
  | Apply_cont ac -> together_apply_cont_blame boxed active origin known ac
  | Switch (scrutinee, arms) ->
    app
      (origins origin (fun y -> var_in_simple y scrutinee))
      (flat_map
         (fun pat ->
           let _, ac = pat in
           together_apply_cont_blame boxed active origin known ac)
         arms)
  | Invalid -> []

(** val together_handler_blame : boxable_number -> positions -> positions ->
    origin_map -> facts -> continuation -> continuation_handler -> (continuation
    * nat) list **)

and together_handler_blame boxed sites active origin known k = function
  | Handler (params, metadata, handler) ->
    app
      (if handler_fits (positions_of sites k) params metadata
       then []
       else map (fun u -> k, u) (positions_of sites k))
      (together_blame boxed sites active
         (origins_under k (positions_of sites k) params origin)
         (facts_under_params known params)
         handler)

(** val together_settle : boxable_number -> nat -> positions -> facts -> expr ->
    positions **)

let rec together_settle boxed fuel sites known e =
  match fuel with
  | O -> sites
  | S fuel0 -> (
    match together_blame boxed sites [] [] known e with
    | [] -> sites
    | p :: l -> together_settle boxed fuel0 (prune (p :: l) sites) known e)

(** val together_positions : boxable_number -> facts -> expr -> positions **)

let together_positions boxed known e =
  together_settle boxed
    (S (count_positions (candidates e)))
    (candidates e) known e

(** val unbox_together_of : boxable_number -> facts -> expr -> expr **)

let unbox_together_of boxed known e =
  together_at boxed (together_positions boxed known e) known e

(** val unboxed_sorts : expr -> boxable_number list **)

let rec unboxed_sorts = function
  | Let (_, _, d, body) -> (
    match d with
    | Prim (p, args) -> (
      match p with
      | Unary p0 -> (
        match p0 with
        | Unbox_number boxed -> (
          match args with
          | [] -> unboxed_sorts body
          | s :: l -> (
            match s with
            | Var _ -> (
              match l with
              | [] -> boxed :: unboxed_sorts body
              | _ :: _ -> unboxed_sorts body)
            | _ -> unboxed_sorts body))
        | _ -> unboxed_sorts body)
      | _ -> unboxed_sorts body)
    | _ -> unboxed_sorts body)
  | Let_static (_, body) -> unboxed_sorts body
  | Let_cont_non_recursive (_, h, body) ->
    app (handler_unboxed_sorts h) (unboxed_sorts body)
  | Let_cont_recursive (handlers, body) ->
    app
      (flat_map
         (fun pat ->
           let _, h = pat in
           handler_unboxed_sorts h)
         handlers)
      (unboxed_sorts body)
  | _ -> []

(** val handler_unboxed_sorts : continuation_handler -> boxable_number list **)

and handler_unboxed_sorts = function
  | Handler (_, _, handler) -> unboxed_sorts handler

(** val sorts_once : boxable_number list -> boxable_number list ->
    boxable_number list **)

let rec sorts_once seen = function
  | [] -> []
  | boxed :: sorts0 ->
    if
      existsb
        (fun boxed' -> kind_eqb (unboxed_kind boxed') (unboxed_kind boxed))
        seen
    then sorts_once seen sorts0
    else boxed :: sorts_once (boxed :: seen) sorts0

(** val unbox_together_all : boxable_number list -> facts -> expr -> expr **)

let rec unbox_together_all boxes known e =
  match boxes with
  | [] -> e
  | boxed :: boxes0 ->
    unbox_together_all boxes0 known (unbox_together_of boxed known e)

(** val unbox_together_knowing : facts -> expr -> expr **)

let unbox_together_knowing known e =
  unbox_together_all (sorts_once [] (unboxed_sorts e)) known e
