(******************************************************************************
 *                                  OxCaml                                    *
 * -------------------------------------------------------------------------- *
 *                               MIT License                                  *
 *                                                                            *
 * Copyright (c) 2026 Jane Street Group LLC                                   *
 * opensource-contacts@janestreet.com                                         *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included    *
 * in all copies or substantial portions of the Software.                     *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
 * DEALINGS IN THE SOFTWARE.                                                  *
 ******************************************************************************)

(* The rebuild of the reaper, transcribed: the term that the compiler's
   reaper makes of a unit, given the flow graph of the unit and the
   solution of its rules.

   This file transcribes middle_end/flambda2/reaper/rebuild.ml (the
   functions named below), the queries of the solution it calls in
   analysis.ml ([has_use], [has_source], [field_used],
   [code_id_actually_directly_called],
   [arguments_used_by_known_arity_call],
   [arguments_used_by_unknown_arity_call]) and the queries of
   points_to_analysis.ml that answer the last three
   ([get_direct_sources], [get_field_sources], [cofield_has_use],
   [arguments_used_by_call], [code_id_actually_directly_called]), and
   [Rebuilt_expr]. It needs no review: what the rebuilt term means is to
   be stated and proved elsewhere, and the test tool compares the term
   with the one the compiler's reaper gives. It is extracted
   ([rebuild_unit], and [rebuild_unit_from_tables] for a solution given
   by its tables).

   What is transcribed is the reaper run with
   [-no-reaper-change-calling-conventions], which implies
   [-no-reaper-unbox]: without the first flag,
   [Unboxing_analysis.cannot_change_calling_convention] holds of every
   code id, so that [Analysis.get_calling_convention_change] is
   [Not_changing_calling_convention] for all code, and
   [Unboxing_analysis.perform_analysis] computes no unboxing, so that
   [get_unboxed_fields] and [get_changed_representation] find nothing.
   Every branch of rebuild.ml that reads either is left out: the
   rebuild of a binding that is unboxed or whose representation changes,
   [rewrite_field_access], the second half of [rebuild_apply], the
   [Unbox] decisions of parameters, and in [rewrite_simple] the poison of
   an unboxed name.

   The inputs are the facts of the graph and the solution
   (FlambdaReaperGraphSpec.v), read through boolean queries: [has_use]
   is [s_has_usage], [has_source] is [s_has_source], [field_used] is
   [s_field_used]. The queries about calls go through the facts as the
   Datalog queries of points_to_analysis.ml go through the relations of
   the graph: every set of nodes they compute is a test, and every
   existential is over a list of facts. The facts are needed for these,
   and for the closures of a set (below).

   The shape follows the OCaml: [rebuild_expr] is [rebuild_expr] and
   [rebuild_holed] at once (the model has terms, not the zipper of
   rev_expr.ml, so the body of a binder is rebuilt before the binder,
   as the zipper does), and each construct has the function of the same
   name. What [rebuild] computes once for the unit is carried down the
   term in an environment ([rebuild_env]): the parameters to keep of
   each continuation in scope ([cont_params_to_keep]), and the kinds of
   the variables in scope ([Variable.kind]).

   Fresh names. A wrapper continuation ([make_apply_wrapper]) and its
   parameters are given names from [first_fresh], an argument: the
   continuation is [first_fresh], and the parameters are [first_fresh]
   and the names after it. One continuation name serves every wrapper,
   since the scope of a wrapper is one call, which binds nothing; and
   the parameters of a wrapper are in scope in its handler only, which
   names nothing else but the continuation it applies. [rebuild_unit]
   takes for [first_fresh] the name above every variable and
   continuation of the unit ([fresh_name] of [names_of]), so that a
   wrapper captures no continuation of a call.

   The deliberate differences from the OCaml, each where the model
   cannot follow it or does not have what it reads:
   - Sets of closures. The model's [Let] binds one variable, to the
     closure at [slot] of a [Closure_of_set]; the compiler's binds one
     to each closure of the set, and the others are bound in the model
     by [Let]s of [Project_function_slot] from the first
     (FlambdaModel.let_closures). The variables of the set are taken to
     be the one of the [Let] and those of the projections that follow
     it in the shape [let_closures] gives them ([projected_closures]).
     As in [rebuild_let_expr_holed_set_of_closures], the [Let] of the set
     is [Invalid] if one of them has no source, is removed with them if
     none has a use, and is kept with all of them otherwise; the
     projections are then kept as they are. A value slot is kept if a
     closure of the set has a use of it ([rewrite_set_of_closures]); the
     closures of the set are read off the graph: the variable of the
     [Let] and the nodes it holds in its function slots
     ([set_closure_nodes]). On the compiler's graph these are the
     variables bound to the closures; a traversal of the model may
     stand for them by other nodes. The older form [Set_of_closures] is
     a set of one closure.
   - Static sets of closures. A static set is one [Bind_symbol] for
     each of its symbols (FlambdaModel.bind_closures). Each is kept if
     one symbol of the set has a use, as the set is in
     [rebuild_let_expr_static_consts], and keeps the value slots that one
     symbol of the set has a use of; so a set is kept or removed whole.
   - Code whose closure is not called. [rewrite_set_of_closures] makes a
     function slot whose closure has no use of a call witness
     [Deleted]; the model has no deleted function slot, and the slot
     keeps its code id. The code itself is still [Bind_deleted_code]
     where its code id has no use.
   - The callee of a call. [rebuild_apply] gives a call no callee where
     the callee has no use ([rewrite_simple_opt]); the model has no call
     without a callee, and the callee is kept as it is. The callee has a
     use wherever the node of the call is aliased to its witness (the
     node [called] has [any_usage], which reaches the witness, then the
     node of the call, then the callee by the accessor of the call
     witness): always at toplevel, and in code where the code id has
     [any_usage] ([cond_alias] is a [propagate] there). Elsewhere the
     model keeps a callee that the compiler drops, and whose binding
     the rebuild may have removed.
   - Call kinds. The three call kinds are the compiler's. The rebuild
     keeps a call of unknown arity. The call kind that
     [decide_whether_apply_needs_calling_convention_change] gives a
     direct call or a call of known arity is [Direct] of the one code
     it calls, or of known arity, with the code ids it may call if they
     are known; the arguments are those of a call of known arity in
     every case, as in the OCaml, where the new call kind is of known
     arity when the old one is.
   - [should_preserve_direct_calls] is a boolean, [true] for the modes
     [Auto] and [Always], which the rebuild treats alike, and [false]
     for [Never] and [Zero_alloc] (the model has no code that is checked
     for allocations). A code id that no [Let_static] of the unit binds
     ([unit_code_ids]) is taken for one of another unit.
   - The masks of the arguments of a call are computed at the widths of
     the call, where [Analysis.answer_call_queries] computes them once
     for each callee at the greatest widths of its calls and each call
     takes a prefix; the mask of an argument depends only on its
     position, so the two are the same.
   - Code is rebuilt where it is bound, with no continuation in scope
     (its return and exception continuations are of fixed arity, so
     every argument given them is kept), where [rebuild] rebuilds all
     code first, in [ordered_code_ids], and binds deleted code where the
     code id has no use; the terms made are the same.
   - Regions. A [Begin_region] or a [Begin_try_region] binds a variable
     of the kind [Region], which [is_var_used] says is used, as the
     OCaml says of every region variable: so its [Let] is kept, as
     [is_begin_region] keeps it. The [Let] of an end of a region is
     kept where its variable has a use, which the traversal gives it
     wherever the code around it is used ([cond_any_usage]). The [Let]
     of a [Local_allocation] binds again, at the same kind, the variable
     of the allocation just before it (FlambdaPrimitiveModel.v): the
     test that keeps or removes the [Let] of the allocation keeps or
     removes it, and makes it [Invalid] where it makes that one so, so
     it is kept exactly where the allocation is. (FlambdaReaperUnboxing.v
     removes it where the allocation is unboxed.) Two differences
     follow, on terms with regions only. Of a set of closures, the
     [Local_allocation] comes between the [Let] of the first closure
     and the projections of the others, where [projected_closures] does
     not look: the projections are then rebuilt as any other, each kept
     where it has a use, which is a use of the first closure (the
     accessor fact of the projection) and so keeps the set and its
     [Local_allocation]; rebuild.ml keeps every closure of a set it
     keeps, and the model removes a projection that has no use. And
     where a boxed number whose field has no use is made poison
     ([rebuild_named]), the mode of the compiler goes with the
     allocation, while the model keeps the [Local_allocation] after the
     poison, of which the machine of FlambdaRegionsSpec.v has no step.
   - Subkinds. [Analysis.rewrite_kind_with_subkind] rewrites what the
     parameters of code that keeps its calling convention state
     ([Unboxing_analysis.compute_code_changes]), and what a parameter
     of a continuation that is kept states ([should_keep_param]). The
     rebuild takes it as [types] ([subkinds_rewriting]). The reaper
     without types ([Traverse.With_types.Without_types], [None]) uses
     [erase_subkind] ([Flambda_kind.With_subkind.erase_subkind]:
     [Anything] and nullable for the kind [Value], nothing stated of a
     naked kind), so that the rebuilt code ([erase_code_subkinds]) and
     handlers ([with_params_kinds]) state nothing: the empty list, as
     the test tool converts the compiler's term
     ([convert_params_subkinds]). The reaper with types ([Some rewrite],
     FlambdaReaperTypes.rebuild_unit_with_types) gives each parameter
     kept what [rewrite] gives of it and of what it stated
     ([kept_handler_metadata], [kept_code_metadata]); a list of which
     nothing says anything is the empty list. Left out, with types: what
     code states of its results, which the compiler rewrites under the
     nodes of its code dep and the machine does not test, and what the
     parameters of a wrapper state, which the compiler takes from the
     parameters of the continuation that it applies; both state nothing
     here. The arities of a call have no subkinds in the model.
   - The model has no phantom binding, no region of code, no
     [Rec_info], no [Never_returns], no cost metrics, inlining
     decisions, [is_my_closure_used], [only_full_applications],
     [is_cold],
     debuginfo or message of [Invalid]; and nothing of what [rebuild]
     returns but the term ([exported_code], [slot_offsets], the typing
     environment). Without phantom bindings, the rule of
     [rebuild_let_expr_holed] that a phantom binding never makes code
     dead has nothing to act on here.
   - A variable that is bound nowhere in the term is taken to be of kind
     [Value].
   - The arms of a [Switch] are gone through by their bindings that
     count ([Keyed.compact]), as the compiler goes through a map.
   - A static boxed number holds a [Simple]; a symbol there, which the
     compiler's [Or_variable.t] cannot hold, is kept as it is.
   - Where the OCaml stops with a fatal error or an exception, the model
     does something defined, which no term that the compiler accepts
     reaches: where a list of decisions is shorter than the arguments
     it is for ([apply_mask], [List.fold_left2]), the arguments beyond
     it are kept; a continuation that has no decisions (one that is not
     bound in the term, as the return and exception continuations of
     the unit and of code are not) has all its arguments kept, as a
     continuation of fixed arity has; a direct call or a call of known
     arity whose callee calls no code
     ([code_id_actually_directly_called] finds an empty set) keeps its
     call kind; a name found where a code id is expected is
     ignored. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaBindersSpec FlambdaWellFormedSpec
  FlambdaNameAbstractionSpec FlambdaFreshQuantifierSpec
  FlambdaRenameApartSpec FlambdaReaperGraphSpec.
Import ListNotations.

(* ======================================================================== *)
(* The environment                                                          *)
(* ======================================================================== *)

(* What the rebuild knows of the names in scope: for each continuation
   bound in the term, whether each of its parameters is kept
   ([cont_params_to_keep], [Keep] being [true]); the kind of each
   variable; and the variables of the closures of a set whose
   projections ([let_closures]) are next, to be kept as they are. *)
Record rebuild_env : Type := Rebuild_env {
  env_conts : finite_map (list bool);
  env_kinds : finite_map flambda_kind;
  env_projections : list variable
}.

Definition empty_env : rebuild_env :=
  {| env_conts := []; env_kinds := []; env_projections := [] |}.

(* [Variable.kind]. *)
Definition kind_of_variable (en : rebuild_env) (x : variable)
    : flambda_kind :=
  match lookup x (env_kinds en) with
  | Some kind => kind
  | None => Value
  end.

(* The variables bound, a later one shadowing an earlier one, and the
   projections that are next. *)
Definition bind_variables_then (en : rebuild_env)
    (bound : list (variable * flambda_kind)) (projections : list variable)
    : rebuild_env :=
  {| env_conts := env_conts en;
     env_kinds := rev bound ++ env_kinds en;
     env_projections := projections |}.

Definition bind_variables (en : rebuild_env)
    (bound : list (variable * flambda_kind)) : rebuild_env :=
  bind_variables_then en bound [].

Definition bind_continuations (en : rebuild_env)
    (bound : list (continuation * list bool)) : rebuild_env :=
  {| env_conts := bound ++ env_conts en;
     env_kinds := env_kinds en;
     env_projections := [] |}.

(* The parameters of a handler with their kinds; a parameter that has
   no kind in the list is of kind [Value]. *)
Fixpoint with_kinds (params : list variable) (kinds : list flambda_kind)
    : list (variable * flambda_kind) :=
  match params, kinds with
  | [], _ => []
  | x :: params, [] => (x, Value) :: with_kinds params []
  | x :: params, kind :: kinds => (x, kind) :: with_kinds params kinds
  end.

(* The elements kept: those at a [true] of the mask, and those beyond
   it. *)
Fixpoint filter_kept {A : Type} (mask : list bool) (l : list A) : list A :=
  match l, mask with
  | [], _ => []
  | a :: l, [] => a :: filter_kept [] l
  | a :: l, keep :: mask =>
    if keep then a :: filter_kept mask l else filter_kept mask l
  end.

(* The metadata of code with what its parameters and its results state
   erased ([compute_code_changes], for code that keeps its calling
   convention, without types): nothing is stated. *)
Definition erase_code_subkinds (metadata : code_metadata) : code_metadata :=
  {| cm_params_arity := cm_params_arity metadata;
     cm_params_subkinds := [];
     cm_result_arity := cm_result_arity metadata;
     cm_result_subkinds := [];
     cm_is_tupled := cm_is_tupled metadata;
     cm_is_stub := cm_is_stub metadata |}.

(* The metadata of a handler with other kinds of parameters. It states
   nothing of them beyond their kinds: what a parameter that is kept
   states is erased ([should_keep_param], [erase_subkind]). *)
Definition with_params_kinds (metadata : handler_metadata)
    (kinds : list flambda_kind) : handler_metadata :=
  if hm_is_exn_handler metadata
  then make_exn_handler_metadata kinds
  else make_handler_metadata kinds.

(* How the rebuild rewrites what a parameter that it keeps states: not
   at all, so that it states nothing ([None], the reaper without types),
   or by a function of the parameter and of what it stated ([Some]). *)
Definition subkinds_rewriting : Type :=
  option (node -> param_subkind -> param_subkind).

(* Each parameter with what it states; nothing where the list is
   shorter. *)
Fixpoint with_subkinds (params : list variable) (sks : list param_subkind)
    : list (variable * param_subkind) :=
  match params, sks with
  | [], _ => []
  | x :: params, [] => (x, any_value) :: with_subkinds params []
  | x :: params, ps :: sks => (x, ps) :: with_subkinds params sks
  end.

(* A list of which nothing says anything is the empty list. *)
Definition stated (sks : list param_subkind) : list param_subkind :=
  if subkinds_say_nothing sks then [] else sks.

(* What the parameters kept by [mask] state, rewritten. *)
Definition rewritten_subkinds
    (rewrite : node -> param_subkind -> param_subkind) (mask : list bool)
    (params : list variable) (sks : list param_subkind)
    : list param_subkind :=
  stated
    (filter_kept mask
       (map (fun '(x, ps) => rewrite (Node_variable x) ps)
          (with_subkinds params sks))).

(* The metadata of a handler that keeps the parameters [params] at a
   [true] of [mask]. *)
Definition kept_handler_metadata (types : subkinds_rewriting)
    (mask : list bool) (params : list variable)
    (metadata : handler_metadata) : handler_metadata :=
  match types with
  | None =>
    with_params_kinds metadata (filter_kept mask (hm_params_kinds metadata))
  | Some rewrite =>
    {| hm_params_kinds := filter_kept mask (hm_params_kinds metadata);
       hm_params_subkinds :=
         rewritten_subkinds rewrite mask params (hm_params_subkinds metadata);
       hm_is_exn_handler := hm_is_exn_handler metadata |}
  end.

(* The same metadata, the parameters stating [sks]. *)
Definition with_code_params_subkinds (sks : list param_subkind)
    (metadata : code_metadata) : code_metadata :=
  {| cm_params_arity := cm_params_arity metadata;
     cm_params_subkinds := sks;
     cm_result_arity := cm_result_arity metadata;
     cm_result_subkinds := cm_result_subkinds metadata;
     cm_is_tupled := cm_is_tupled metadata;
     cm_is_stub := cm_is_stub metadata |}.

(* The metadata of code that keeps its calling convention, whose
   parameters are [params]: with types, what they state is rewritten,
   and nothing is stated of the results. *)
Definition kept_code_metadata (types : subkinds_rewriting)
    (params : list variable) (metadata : code_metadata) : code_metadata :=
  match types with
  | None => erase_code_subkinds metadata
  | Some rewrite =>
    with_code_params_subkinds
      (rewritten_subkinds rewrite [] params (cm_params_subkinds metadata))
      (erase_code_subkinds metadata)
  end.

(* The continuation of a call changed. *)
Definition with_continuation (ap : apply_expr) (k : continuation)
    : apply_expr :=
  {| ap_callee := ap_callee ap;
     ap_call_kind := ap_call_kind ap;
     ap_args := ap_args ap;
     ap_continuation := k;
     ap_exn_continuation := ap_exn_continuation ap;
     ap_exn_extra_args := ap_exn_extra_args ap;
     ap_args_arity := ap_args_arity ap;
     ap_return_arity := ap_return_arity ap |}.

(* The code ids that a binding binds, deleted code included. *)
Definition code_ids_bound (binding : static_binding) : list code_id :=
  match binding with
  | Bind_code id _ | Bind_deleted_code id => [id]
  | Bind_symbol _ _ => []
  end.

(* The zero of a kind of naked number ([Float.zero], [Int32.zero], ...,
   of [rewrite_static_const]). *)
Definition zero_of_kind (kind : flambda_kind) : const :=
  match kind with
  | Naked_number k =>
    match k with
    | Kind_naked_immediate => Naked_immediate 0
    | Kind_naked_float32 => Naked_float32 0
    | Kind_naked_float => Naked_float 0
    | Kind_naked_int8 => Naked_int8 0
    | Kind_naked_int16 => Naked_int16 0
    | Kind_naked_int32 => Naked_int32 0
    | Kind_naked_int64 => Naked_int64 0
    | Kind_naked_nativeint => Naked_nativeint 0
    | Kind_naked_vec128 => Naked_vec128 0
    | Kind_naked_vec256 => Naked_vec256 0
    | Kind_naked_vec512 => Naked_vec512 0
    | Kind_naked_mask => Naked_mask 0
    end
  | Value | Region | Rec_info => Poison kind
  end.

Section Rebuild.

Variable F : facts.
Variable sol : solution.
(* [should_preserve_direct_calls] is [Yes] or [Auto]. *)
Variable preserve_direct_calls : bool.
(* The code ids of the unit. *)
Variable unit_code_ids : list code_id.
(* The first of the names that the rebuild makes. *)
Variable first_fresh : positive.
(* How what a parameter kept states is rewritten. *)
Variable types : subkinds_rewriting.

(* ======================================================================== *)
(* Queries of the solution                                                  *)
(* ======================================================================== *)

(* [raw_is_var_used]. *)
Definition is_var_used (x : variable) (kind : flambda_kind) : bool :=
  match kind with
  | Region | Rec_info => true
  | Value | Naked_number _ => s_has_usage sol (Node_variable x)
  end.

(* [is_symbol_used]. *)
Definition is_symbol_used (s : symbol) : bool :=
  s_has_usage sol (Node_symbol s).

(* [is_code_id_used], of a code id of the unit. *)
Definition is_code_id_used (id : code_id) : bool :=
  s_has_usage sol (Node_code_id id).

(* [is_dead_var]. *)
Definition is_dead_var (x : variable) (kind : flambda_kind) : bool :=
  match kind with
  | Region | Rec_info => false
  | Value | Naked_number _ => negb (s_has_source sol (Node_variable x))
  end.

(* [Analysis.field_used]. *)
Definition field_used (n : node) (f : field) : bool := s_field_used sol n f.

(* [simple_is_dead]. *)
Definition simple_is_dead (en : rebuild_env) (s : simple) : bool :=
  match s with
  | Var x => is_dead_var x (kind_of_variable en x)
  | Symbol sym => negb (s_has_source sol (Node_symbol sym))
  | Const _ => false
  end.

(* The test of [rewrite_apply_cont_expr]: a name that has no source,
   whatever its kind. *)
Definition simple_has_no_source (s : simple) : bool :=
  match s with
  | Var x => negb (s_has_source sol (Node_variable x))
  | Symbol sym => negb (s_has_source sol (Node_symbol sym))
  | Const _ => false
  end.

(* ======================================================================== *)
(* Queries about calls                                                      *)
(* ======================================================================== *)

(* [Points_to_analysis.sources]: a set of allocations, as a test. *)
Inductive allocation_sources : Type :=
  | Any_source
  | Sources (mem : node -> bool).

(* [get_direct_sources], of one node. *)
Definition get_direct_sources (x : node) : allocation_sources :=
  if s_any_source sol x
  then Any_source
  else Sources (fun y => s_sources sol x y && s_has_source sol y).

(* [get_field_sources]: the sources of what the nodes of [in_] hold in
   the field [f]. *)
Definition get_field_sources (in_ : node -> bool) (f : field)
    : allocation_sources :=
  if existsb
       (fun '(x, f', y) => in_ x && field_eqb f' f && s_any_source sol y)
       (f_constructor F)
  then Any_source
  else
    Sources
      (fun z =>
         existsb
           (fun '(x, f', y) =>
              in_ x && field_eqb f' f && s_sources sol y z
              && s_has_source sol z)
           (f_constructor F)).

(* [cofield_has_use]: a node of [in_] has a parameter number [i] that has
   a use. *)
Definition cofield_has_use (in_ : node -> bool) (i : nat) : bool :=
  existsb (fun '(x, j, y) => in_ x && Nat.eqb j i && s_has_usage sol y)
    (f_parameter F).

(* [arguments_used_by_call]: for each group of arguments, of the sizes
   [num_args], whether each argument is kept. *)
Fixpoint arguments_used_by_call (ep : closure_entry_point)
    (callee_sources : allocation_sources) (num_args : list nat)
    : list (list bool) :=
  match num_args with
  | [] => []
  | num_first_arg :: num_args_rest =>
    let all_kept := map (fun n => repeat true n) num_args in
    match callee_sources with
    | Any_source => all_kept
    | Sources callee_sources =>
      match get_field_sources callee_sources (Field_call_witness ep) with
      | Any_source => all_kept
      | Sources witness_sources =>
        map (cofield_has_use witness_sources) (seq 0 num_first_arg)
        :: match num_args_rest with
           | [] => []
           | _ :: _ =>
             arguments_used_by_call ep
               (get_field_sources witness_sources
                  (Field_return_of_call (Normal 0)))
               num_args_rest
           end
      end
    end
  end.

(* [arguments_used_by_known_arity_call], as a mask of the arguments. *)
Definition arguments_used_by_known_arity_call (callee : node)
    (args : list simple) : list bool :=
  concat
    (arguments_used_by_call Known_arity_code_pointer
       (get_direct_sources callee) [length args]).

(* [arguments_used_by_unknown_arity_call], flattened: the arguments are
   grouped by the parameters of the arity of the call
   ([Flambda_arity.group_by_parameter]). *)
Definition arguments_used_by_unknown_arity_call (callee : node)
    (args_arity : arity) : list bool :=
  concat
    (arguments_used_by_call Unknown_arity_code_pointer
       (get_direct_sources callee)
       (map (fun c => length (unarize_component c)) args_arity)).

(* [code_id_actually_directly_called]: [None] for [Unknown], or the code
   ids of the witnesses of known arity that the closure may hold. *)
Definition code_id_actually_directly_called (closure : node)
    : option (list code_id) :=
  let apply_widgets :=
    filter
      (fun '(_, f, base) =>
         node_eqb base closure
         && field_eqb f (Field_call_witness Known_arity_code_pointer))
      (f_accessor F)
  in
  if existsb (fun '(apply_widget, _, _) => s_any_source sol apply_widget)
       apply_widgets
  then None
  else
    Some
      (flat_map
         (fun '(apply_widget, _, _) =>
            flat_map
              (fun '(call_witness, f, codeid) =>
                 match codeid with
                 | Node_code_id id =>
                   if field_eqb f Field_code_id_of_call_witness
                      && s_sources sol apply_widget call_witness
                      && s_has_source sol call_witness
                   then [id]
                   else []
                 | Node_variable _ | Node_symbol _ | Node_all_constants
                 | Node_made _ => []
                 end)
              (f_constructor F))
         apply_widgets).

(* ======================================================================== *)
(* Simples                                                                  *)
(* ======================================================================== *)

(* [rewrite_simple]. *)
Definition rewrite_simple (en : rebuild_env) (s : simple) : simple :=
  match s with
  | Var x =>
    let kind := kind_of_variable en x in
    if is_var_used x kind then s else Const (Poison kind)
  | Symbol sym => if is_symbol_used sym then s else Const (Poison Value)
  | Const _ => s
  end.

(* [name_poison] of a name; a constant stays. *)
Definition poison_argument (en : rebuild_env) (s : simple) : simple :=
  match s with
  | Var x => Const (Poison (kind_of_variable en x))
  | Symbol _ => Const (Poison Value)
  | Const _ => s
  end.

(* [rewrite_or_variable]. *)
Definition rewrite_or_variable (en : rebuild_env) (s : simple) : simple :=
  match s with
  | Var x =>
    let kind := kind_of_variable en x in
    if is_var_used x kind then s else Const (zero_of_kind kind)
  | Symbol _ | Const _ => s
  end.

(* ======================================================================== *)
(* Blocks and sets of closures                                              *)
(* ======================================================================== *)

(* The fields of a block bound to [base], from field [i]
   ([rebuild_make_block_default_case], and [rewrite_static_const] of a
   [Block]): every field of a block of the model is a value. *)
Fixpoint rebuild_fields (en : rebuild_env) (base : node) (i : nat)
    (fields : list simple) : list simple :=
  match fields with
  | [] => []
  | s :: fields =>
    (if field_used base (Field_block i Value)
     then rewrite_simple en s
     else Const (Poison Value))
    :: rebuild_fields en base (Datatypes.S i) fields
  end.

(* The value slots kept by [rewrite_set_of_closures], for a set whose
   closures are [bound]. *)
Definition rebuild_value_slots (en : rebuild_env) (bound : list node)
    (value_slots : list (value_slot * simple))
    : list (value_slot * simple) :=
  flat_map
    (fun '(w, s) =>
       if existsb (fun n => field_used n (Field_value_slot w)) bound
       then [(w, rewrite_simple en s)]
       else [])
    value_slots.

(* The closures of the set of which [x] is a closure, as the graph has
   them: [x], and what it holds in its function slots. *)
Definition set_closure_nodes (x : node) : list node :=
  x
  :: flat_map
       (fun '(base, f, from) =>
          if node_eqb base x && is_function_slot f then [from] else [])
       (f_constructor F).

(* The function slots of a set other than [slot], if [slot] is the first,
   in the order in which [let_closures] projects them. *)
Definition other_slots (slot : function_slot)
    (function_decls : list (function_slot * code_id))
    : list function_slot :=
  match function_decls with
  | (first, _) :: others => if Pos.eqb first slot then map fst others else []
  | [] => []
  end.

(* The variables bound by the projections from [x], at [slot], of the
   closures at [others] that begin [e] (FlambdaModel.let_projections). *)
Fixpoint projected_closures (x : variable) (slot : function_slot)
    (others : list function_slot) (e : expr) {struct e} : list variable :=
  match others, e with
  | other :: others,
    Let y _ (Project_function_slot move_from move_to (Var x')) body =>
    if Pos.eqb x' x && Pos.eqb move_from slot && Pos.eqb move_to other
       && negb (Pos.eqb y x)
    then y :: projected_closures x slot others body
    else []
  | _, _ => []
  end.

Definition projections_of_set (x : variable) (defining_expr : named)
    (body : expr) : list variable :=
  match defining_expr with
  | Closure_of_set slot function_decls _ =>
    projected_closures x slot (other_slots slot function_decls) body
  | _ => []
  end.

(* [rewrite_set_of_closures], of a set whose closures are [bound]. *)
Definition rebuild_set (en : rebuild_env) (bound : list node)
    (defining_expr : named) : named :=
  match defining_expr with
  | Set_of_closures id value_slots =>
    Set_of_closures id (rebuild_value_slots en bound value_slots)
  | Closure_of_set slot function_decls value_slots =>
    Closure_of_set slot function_decls
      (rebuild_value_slots en bound value_slots)
  | _ => defining_expr
  end.

(* The next binding is a projection of the closures of a set. *)
Definition is_pending_projection (en : rebuild_env) (x : variable)
    (defining_expr : named) : bool :=
  match env_projections en, defining_expr with
  | y :: _, Project_function_slot _ _ _ => Pos.eqb x y
  | _, _ => false
  end.

(* ======================================================================== *)
(* Lets                                                                     *)
(* ======================================================================== *)

(* [rebuild_let_expr_singleton] once the variable is known to be used,
   and [rebuild_named_default_case]: the defining expression of [x]. *)
Definition rebuild_named (en : rebuild_env) (x : variable) (d : named)
    : named :=
  match d with
  | Prim (Variadic (Make_block kind)) fields =>
    Prim (Variadic (Make_block kind))
      (rebuild_fields en (Node_variable x) 0 fields)
  | Prim (Heap (Make_block_with_identity mut tag)) fields =>
    Prim (Heap (Make_block_with_identity mut tag))
      (rebuild_fields en (Node_variable x) 0 fields)
  | Prim (Unary (Box_number bn)) args =>
    if field_used (Node_variable x) (Field_boxed_number bn)
    then Prim (Unary (Box_number bn)) (map (rewrite_simple en) args)
    else Simple (Const (Poison Value))
  | Prim p args => Prim p (map (rewrite_simple en) args)
  | Simple s => Simple (rewrite_simple en s)
  | Project_value_slot w closure =>
    Project_value_slot w (rewrite_simple en closure)
  | Project_value_slot_from project_from w closure =>
    Project_value_slot_from project_from w (rewrite_simple en closure)
  | Project_function_slot move_from move_to closure =>
    Project_function_slot move_from move_to (rewrite_simple en closure)
  | Set_of_closures _ _ | Closure_of_set _ _ _ => d
  end.

(* ======================================================================== *)
(* Static constants                                                         *)
(* ======================================================================== *)

(* Whether the binding of [s] is kept ([rebuild_let_expr_static_consts]):
   a closure of a static set is kept if a symbol of the set has a use. *)
Definition static_const_is_used (s : symbol) (const : static_const)
    : bool :=
  match const with
  | Static_const_closure_of_set _ closure_symbols _ _ =>
    existsb (fun '(_, sym) => is_symbol_used sym) closure_symbols
  | _ => is_symbol_used s
  end.

(* [rewrite_static_const], and [rewrite_set_of_closures] of a static
   set. *)
Definition rebuild_static_const (en : rebuild_env) (s : symbol)
    (const : static_const) : static_const :=
  match const with
  | Static_const_set_of_closures id value_slots =>
    Static_const_set_of_closures id
      (rebuild_value_slots en [Node_symbol s] value_slots)
  | Static_const_closure_of_set slot closure_symbols function_decls
      value_slots =>
    Static_const_closure_of_set slot closure_symbols function_decls
      (rebuild_value_slots en
         (map (fun '(_, sym) => Node_symbol sym) closure_symbols)
         value_slots)
  | Static_const_block tag fields =>
    Static_const_block tag (rebuild_fields en (Node_symbol s) 0 fields)
  | Static_const_boxed_number contents =>
    Static_const_boxed_number (rewrite_or_variable en contents)
  | Static_const_immutable_string _ => const
  | Static_const_immutable_value_array elements =>
    Static_const_immutable_value_array (map (rewrite_simple en) elements)
  end.

(* ======================================================================== *)
(* Continuations                                                            *)
(* ======================================================================== *)

(* [should_keep_param] of [rebuild], for a continuation that is not of
   fixed arity: a parameter is kept if it is used, or if it is the first
   parameter of an exception handler. *)
Definition params_to_keep (is_exn_handler : bool)
    (params : list (variable * flambda_kind)) : list bool :=
  map
    (fun '(x, kind) =>
       is_var_used x kind
       || (is_exn_handler
           && match params with
              | (first, _) :: _ => Pos.eqb x first
              | [] => false
              end))
    params.

Definition handler_mask (is_exn_handler : bool) (h : continuation_handler)
    : list bool :=
  match h with
  | Handler params metadata _ =>
    params_to_keep is_exn_handler (with_kinds params (hm_params_kinds metadata))
  end.

(* [rewrite_apply_cont_expr]: [None] if an argument has no source. *)
Definition rewrite_apply_cont (en : rebuild_env) (ac : apply_cont)
    : option apply_cont :=
  if existsb simple_has_no_source (ac_args ac)
  then None
  else
    let args := map (rewrite_simple en) (ac_args ac) in
    Some
      {| ac_continuation := ac_continuation ac;
         ac_args :=
           match lookup (ac_continuation ac) (env_conts en) with
           | Some mask => filter_kept mask args
           | None => args
           end;
         ac_trap_action := ac_trap_action ac |}.

(* The [Apply_cont] case of [rebuild_expr]. *)
Definition rebuild_apply_cont (en : rebuild_env) (ac : apply_cont) : expr :=
  match rewrite_apply_cont en ac with
  | Some ac => Apply_cont ac
  | None => Invalid
  end.

(* The [Switch] case of [rebuild_expr]: an arm whose action has an
   argument with no source is removed. *)
Definition rebuild_switch (en : rebuild_env) (scrutinee : simple)
    (arms : list (Z * apply_cont)) : expr :=
  let arms :=
    flat_map
      (fun '(d, ac) =>
         match rewrite_apply_cont en ac with
         | Some ac => [(d, ac)]
         | None => []
         end)
      (Keyed.compact Z.eqb arms)
  in
  if Keyed.is_empty arms
  then Invalid
  else Switch (rewrite_simple en scrutinee) arms.

(* ======================================================================== *)
(* Calls                                                                    *)
(* ======================================================================== *)

(* [called] of [decide_whether_apply_needs_calling_convention_change]:
   the call kind of a call of known arity or a direct call, given that
   of a callee whose code is not known. *)
Definition called_call_kind (ap : apply_expr) (call_kind_if_unknown : call_kind)
    : call_kind :=
  match ap_callee ap with
  | Const _ => call_kind_if_unknown
  | Var _ | Symbol _ =>
    match code_id_actually_directly_called (node_of_simple (ap_callee ap))
    with
    | None => call_kind_if_unknown
    | Some ids =>
      match Keyed.get_singleton Pos.eqb ids with
      | Some id' => Direct id'
      | None =>
        if Keyed.is_empty ids
        then ap_call_kind ap
        else Indirect_known_arity (Some ids)
      end
    end
  end.

(* The call kind of
   [decide_whether_apply_needs_calling_convention_change]. *)
Definition rewrite_call_kind (ap : apply_expr) : call_kind :=
  match ap_call_kind ap with
  | Indirect_unknown_arity => Indirect_unknown_arity
  | Direct id =>
    called_call_kind ap
      (if preserve_direct_calls || negb (among id unit_code_ids)
       then Direct id
       else Indirect_known_arity None)
  | Indirect_known_arity _ => called_call_kind ap (Indirect_known_arity None)
  end.

(* The arguments of a call, at the mask of the arguments used: one not
   used is poison. *)
Fixpoint mask_arguments (en : rebuild_env) (mask : list bool)
    (args : list simple) : list simple :=
  match args, mask with
  | [], _ => []
  | arg :: args, [] => arg :: mask_arguments en [] args
  | arg :: args, keep :: mask =>
    (if keep then arg else poison_argument en arg)
    :: mask_arguments en mask args
  end.

(* [args] of [rebuild_apply]: an argument kept is as it was, one deleted
   is poison; if the callee is a constant every argument is kept and
   rewritten. *)
Definition rebuild_arguments (en : rebuild_env) (ap : apply_expr)
    : list simple :=
  match ap_callee ap with
  | Const _ => map (rewrite_simple en) (ap_args ap)
  | Var _ | Symbol _ =>
    let callee := node_of_simple (ap_callee ap) in
    mask_arguments en
      (match ap_call_kind ap with
       | Direct _ | Indirect_known_arity _ =>
         arguments_used_by_known_arity_call callee (ap_args ap)
       | Indirect_unknown_arity =>
         arguments_used_by_unknown_arity_call callee (ap_args_arity ap)
       end)
      (ap_args ap)
  end.

(* [exn_continuation] of [rebuild_apply]: the extra arguments of the
   exception continuation, at the parameters kept of its handler after
   the first. *)
Definition rebuild_exn_extra_args (en : rebuild_env) (ap : apply_expr)
    : list (simple * flambda_kind) :=
  let extra_args :=
    map (fun '(s, kind) => (rewrite_simple en s, kind)) (ap_exn_extra_args ap)
  in
  match lookup (ap_exn_continuation ap) (env_conts en) with
  | Some mask => filter_kept (tl mask) extra_args
  | None => extra_args
  end.

(* [make_apply_wrapper], for a call whose results are all kept by the
   callee ([func_decisions] of [rebuild_apply]): if the return
   continuation of the call does not keep all its parameters, the call
   returns to a wrapper, which applies it to those it keeps. *)
Definition make_apply_wrapper (en : rebuild_env) (ap : apply_expr) : expr :=
  match lookup (ap_continuation ap) (env_conts en) with
  | Some mask =>
    if forallb (fun keep => keep) mask
    then Apply ap
    else
      let kinds := unarize (ap_return_arity ap) in
      let params := names_from (length kinds) first_fresh in
      Let_cont_non_recursive first_fresh
        (Handler params (make_handler_metadata kinds)
           (Apply_cont
              {| ac_continuation := ap_continuation ap;
                 ac_args := filter_kept mask (map Var params);
                 ac_trap_action := None |}))
        (Apply (with_continuation ap first_fresh))
  | None => Apply ap
  end.

(* [rebuild_apply]. *)
Definition rebuild_apply (en : rebuild_env) (ap : apply_expr) : expr :=
  if simple_is_dead en (ap_callee ap)
     || existsb (simple_is_dead en) (ap_args ap)
  then Invalid
  else
    make_apply_wrapper en
      {| ap_callee := ap_callee ap;
         ap_call_kind := rewrite_call_kind ap;
         ap_args := rebuild_arguments en ap;
         ap_continuation := ap_continuation ap;
         ap_exn_continuation := ap_exn_continuation ap;
         ap_exn_extra_args := rebuild_exn_extra_args en ap;
         ap_args_arity := ap_args_arity ap;
         ap_return_arity := ap_return_arity ap |}.

(* ======================================================================== *)
(* The rebuild                                                              *)
(* ======================================================================== *)

(* [rebuild_expr] and [rebuild_holed]. A [Let] whose variable has no
   source is [Invalid], one whose variable has no use is removed
   ([rebuild_let_expr_holed], [rebuild_let_expr_singleton]); a
   [Let_static] keeps the bindings used ([rebuild_let_expr_static_consts]);
   a non-recursive [Let_cont] whose continuation is not applied in the
   rebuilt body is removed, and the parameters that a handler does not
   keep are removed, from the handler and from the arguments given it. *)
Fixpoint rebuild_expr (en : rebuild_env) (e : expr) {struct e} : expr :=
  match e with
  | Let x kind defining_expr body =>
    if is_pending_projection en x defining_expr
    then
      Let x kind defining_expr
        (rebuild_expr
           (bind_variables_then en [(x, kind)] (tl (env_projections en)))
           body)
    else
      match defining_expr with
      | Set_of_closures _ _ | Closure_of_set _ _ _ =>
        let projections := projections_of_set x defining_expr body in
        if is_dead_var x kind || existsb (fun y => is_dead_var y Value)
                                   projections
        then Invalid
        else if is_var_used x kind
                || existsb (fun y => is_var_used y Value) projections
        then
          Let x kind
            (rebuild_set en (set_closure_nodes (Node_variable x))
               defining_expr)
            (rebuild_expr (bind_variables_then en [(x, kind)] projections)
               body)
        else rebuild_expr (bind_variables en [(x, kind)]) body
      | Simple _ | Prim _ _ | Project_value_slot _ _
      | Project_value_slot_from _ _ _ | Project_function_slot _ _ _ =>
        if is_dead_var x kind
        then Invalid
        else if is_var_used x kind
        then
          Let x kind (rebuild_named en x defining_expr)
            (rebuild_expr (bind_variables en [(x, kind)]) body)
        else rebuild_expr (bind_variables en [(x, kind)]) body
      end
  | Let_static bindings body =>
    Let_static (flat_map (rebuild_binding en) bindings)
      (rebuild_expr (bind_variables en []) body)
  | Let_cont_non_recursive k h body =>
    let mask := handler_mask (handler_is_exn_handler h) h in
    let body := rebuild_expr (bind_continuations en [(k, mask)]) body in
    if among k (free_continuations_of_expr body)
    then Let_cont_non_recursive k (rebuild_handler en true mask h) body
    else body
  | Let_cont_recursive handlers body =>
    let en :=
      bind_continuations en
        (map (fun '(k, h) => (k, handler_mask false h)) handlers)
    in
    Let_cont_recursive
      (map
         (fun '(k, h) =>
            (k, rebuild_handler en false (handler_mask false h) h))
         handlers)
      (rebuild_expr en body)
  | Apply ap => rebuild_apply en ap
  | Apply_cont ac => rebuild_apply_cont en ac
  | Switch scrutinee arms => rebuild_switch en scrutinee arms
  | Invalid => Invalid
  end
(* A handler: the parameters kept by [mask]. The handler of a
   non-recursive [Let_cont] with a parameter that has no source is
   [Invalid] ([dead_check]). *)
with rebuild_handler (en : rebuild_env) (dead_check : bool)
    (mask : list bool) (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    let bound := with_kinds params (hm_params_kinds metadata) in
    Handler (filter_kept mask params)
      (kept_handler_metadata types mask params metadata)
      (if dead_check && existsb (fun '(x, kind) => is_dead_var x kind) bound
       then Invalid
       else rebuild_expr (bind_variables en bound) handler)
  end
(* [rebuild_let_expr_static_consts] and [rebuild_static_const_or_code], of
   one binding: code whose code id has no use is deleted, a symbol that
   has no use is not bound. *)
with rebuild_binding (en : rebuild_env) (binding : static_binding)
    : list static_binding :=
  match binding with
  | Bind_code id c =>
    if is_code_id_used id
    then [Bind_code id (rebuild_code c)]
    else [Bind_deleted_code id]
  | Bind_deleted_code _ => [binding]
  | Bind_symbol s const =>
    if static_const_is_used s const
    then [Bind_symbol s (rebuild_static_const en s const)]
    else []
  end
(* [rebuild_code] and [rebuild_function_params_and_body]: the body is
   [Invalid] if [my_closure] or a parameter has no source, and what the
   parameters and the results state is erased. *)
with rebuild_code (c : code) : code :=
  match c with
  | Code params my_closure return_continuation exn_continuation metadata
      body =>
    let bound :=
      (my_closure, Value)
      :: with_kinds params (unarize (cm_params_arity metadata))
    in
    Code params my_closure return_continuation exn_continuation
      (kept_code_metadata types params metadata)
      (if existsb (fun '(x, kind) => is_dead_var x kind) bound
       then Invalid
       else rebuild_expr (bind_variables empty_env bound) body)
  end.

End Rebuild.

(* [Rebuild.rebuild]: the rebuilt unit, given the facts of its graph and
   the solution, with what the parameters kept state rewritten as
   [types] says. *)
Definition rebuild_unit_with (types : subkinds_rewriting) (F : facts)
    (sol : solution) (preserve_direct_calls : bool) (e : expr) : expr :=
  rebuild_expr F sol preserve_direct_calls (static_names code_ids_bound e)
    (fresh_name (names_of e)) types empty_env e.

(* The reaper without types. *)
Definition rebuild_unit (F : facts) (sol : solution)
    (preserve_direct_calls : bool) (e : expr) : expr :=
  rebuild_expr F sol preserve_direct_calls (static_names code_ids_bound e)
    (fresh_name (names_of e)) None empty_env e.

(* The same, with the solution given by its tables. *)
Definition rebuild_unit_from_tables (F : facts) (T : tables)
    (preserve_direct_calls : bool) (e : expr) : expr :=
  rebuild_unit F (solution_of_tables T) preserve_direct_calls e.

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

Module Examples.

Local Open Scope positive_scope.

(* The units return to 1 and raise to 2. *)
Definition return_to (k : continuation) (args : list simple) : expr :=
  Apply_cont {| ac_continuation := k; ac_args := args;
                ac_trap_action := None |}.

Definition load (field : Z) (block : simple) : named :=
  Prim (Unary (Block_load (Block_access_values None None Any_value)
                 Immutable field)) [block].

Definition make_block (fields : list simple) : named :=
  Prim (Variadic (Make_block (Block_kind_values 0))) fields.

Definition no_tables : tables := Tables [] [] [] [] [] [] [] [] [] [].

(* [let x10 = 5 in let x11 = 7 in let x12 = [x11; x11] in
   let x13 = x12.(0) in return x13]: [x10] has no use, and nothing reads
   field 1 of [x12]. The [Let] of [x10] is removed and field 1 is
   poison. *)
Definition unused_let_and_field : expr :=
  Let 10 Value (Simple (Const (Tagged_immediate 5)))
    (Let 11 Value (Simple (Const (Tagged_immediate 7)))
       (Let 12 Value (make_block [Var 11; Var 11])
          (Let 13 Value (load 0 (Var 12)) (return_to 1 [Var 13])))).

Definition unused_let_and_field_tables : tables :=
  {| t_alias := []; t_any_usage := []; t_any_source := [];
     t_has_usage := [Node_variable 11; Node_variable 12; Node_variable 13];
     t_has_source :=
       [Node_variable 10; Node_variable 11; Node_variable 12;
        Node_variable 13];
     t_usages := []; t_sources := []; t_reading_field := [];
     t_escaping_field := [];
     t_field_used := [(Node_variable 12, Field_block 0 Value)] |}.

Example rebuild_unused_let_and_field :
  rebuild_unit_from_tables no_facts unused_let_and_field_tables true
    unused_let_and_field =
  Let 11 Value (Simple (Const (Tagged_immediate 7)))
    (Let 12 Value (make_block [Var 11; Const (Poison Value)])
       (Let 13 Value (load 0 (Var 12)) (return_to 1 [Var 13]))).
Proof. vm_compute. reflexivity. Qed.

(* The handler of 20 has two parameters, of which only [x30] is used:
   [x31] goes, and so does the argument given it, [x41], whose [Let] has
   no use left. *)
Definition unused_continuation_parameter : expr :=
  Let_cont_non_recursive 20
    (Handler [30; 31] (make_handler_metadata [Value; Value])
       (return_to 1 [Var 30]))
    (Let 40 Value (Simple (Const (Tagged_immediate 1)))
       (Let 41 Value (Simple (Const (Tagged_immediate 2)))
          (return_to 20 [Var 40; Var 41]))).

Definition unused_continuation_parameter_tables : tables :=
  {| t_alias := []; t_any_usage := []; t_any_source := [];
     t_has_usage := [Node_variable 30; Node_variable 40];
     t_has_source :=
       [Node_variable 30; Node_variable 31; Node_variable 40;
        Node_variable 41];
     t_usages := []; t_sources := []; t_reading_field := [];
     t_escaping_field := []; t_field_used := [] |}.

Example rebuild_unused_continuation_parameter :
  rebuild_unit_from_tables no_facts unused_continuation_parameter_tables true
    unused_continuation_parameter =
  Let_cont_non_recursive 20
    (Handler [30] (make_handler_metadata [Value]) (return_to 1 [Var 30]))
    (Let 40 Value (Simple (Const (Tagged_immediate 1)))
       (return_to 20 [Var 40])).
Proof. vm_compute. reflexivity. Qed.

(* The code 50 returns its first parameter [x40] and does not use its
   second, [x42]. The unit makes the closure [x60] of it at the function
   slot 70 and calls it directly with 9 and [x61], returning to 20, whose
   parameter [x32] has no use. The second argument is poison, the [Let]
   of [x61] goes, and the call returns to a wrapper, the continuation 62
   (the name after every name of the unit), which drops the result and
   applies 20.

   The graph has what the queries about the call read: the accessor of
   the call witness from [x60] (made node 1), with the code id 50
   (through the witness, made node 2), and the parameters of that
   witness. *)
Definition call_code : code :=
  Code [40; 42] 41 3 4
    (make_code_metadata [Singleton Value; Singleton Value] [Singleton Value])
    (return_to 3 [Var 40]).

Definition call_through_wrapper_call (args : list simple) (k : continuation)
    : apply_expr :=
  {| ap_callee := Var 60; ap_call_kind := Direct 50; ap_args := args;
     ap_continuation := k; ap_exn_continuation := 2;
     ap_exn_extra_args := [];
     ap_args_arity := [Singleton Value; Singleton Value];
     ap_return_arity := [Singleton Value] |}.

Definition call_through_wrapper : expr :=
  Let_static [Bind_code 50 call_code]
    (Let 61 Value (Simple (Const (Tagged_immediate 3)))
       (Let 60 Value (Closure_of_set 70 [(70, 50)] [])
          (Let_cont_non_recursive 20
             (Handler [32] (make_handler_metadata [Value])
                (return_to 1 [Const (Tagged_immediate 0)]))
             (Apply
                (call_through_wrapper_call
                   [Const (Tagged_immediate 9); Var 61] 20))))).

Definition call_through_wrapper_facts : facts :=
  {| f_alias := []; f_use := [];
     f_accessor :=
       [(Node_made 1, Field_call_witness Known_arity_code_pointer,
         Node_variable 60)];
     f_constructor :=
       [(Node_variable 60, Field_call_witness Known_arity_code_pointer,
         Node_made 2);
        (Node_made 2, Field_code_id_of_call_witness, Node_code_id 50)];
     f_argument := [];
     f_parameter :=
       [(Node_made 2, 0%nat, Node_variable 40);
        (Node_made 2, 1%nat, Node_variable 42)];
     f_propagate := []; f_alias_if_any_source := []; f_any_usage := [];
     f_any_source := []; f_any_source_if_used := [];
     f_zero_alloc_source := [] |}.

Definition call_through_wrapper_tables : tables :=
  {| t_alias := []; t_any_usage := []; t_any_source := [];
     t_has_usage :=
       [Node_code_id 50; Node_variable 60; Node_variable 40];
     t_has_source :=
       [Node_variable 60; Node_variable 61; Node_variable 32;
        Node_variable 40; Node_variable 41; Node_variable 42; Node_made 2];
     t_usages := [];
     t_sources :=
       [(Node_variable 60, Node_variable 60); (Node_made 1, Node_made 2);
        (Node_made 2, Node_made 2)];
     t_reading_field := []; t_escaping_field := []; t_field_used := [] |}.

Example rebuild_call_through_wrapper :
  rebuild_unit_from_tables call_through_wrapper_facts
    call_through_wrapper_tables true call_through_wrapper =
  Let_static [Bind_code 50 call_code]
    (Let 60 Value (Closure_of_set 70 [(70, 50)] [])
       (Let_cont_non_recursive 20
          (Handler [] (make_handler_metadata [])
             (return_to 1 [Const (Tagged_immediate 0)]))
          (Let_cont_non_recursive 62
             (Handler [62] (make_handler_metadata [Value]) (return_to 20 []))
             (Apply
                (call_through_wrapper_call
                   [Const (Tagged_immediate 9); Const (Poison Value)]
                   62))))).
Proof. vm_compute. reflexivity. Qed.

End Examples.
