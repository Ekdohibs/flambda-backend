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

(* What the rebuild with unboxing is worth where the unboxing is local:
   the unit rebuilt with unboxing from a solution that passes the checks
   refines the unit, as the plain rebuild does, if the allocations that
   are unboxed are blocks and boxed numbers that are only read, aliased,
   and held in fields of blocks that are unboxed, in the scope where they
   are bound or in the handlers that they are given to, whose parameters
   are unboxed alike.

   This file is to be reviewed: the fragment ([local_unboxing]), the
   statements, and the examples, against FlambdaReaperUnboxing.v (the
   rebuild with unboxing, which needs no review) and
   FlambdaReaperRebuildSpec.v (the plain statement). The proofs are in
   FlambdaReaperUnboxingRefinesProofs.v, which needs no review.

   [rebuild_unbox_refines_rebuild_statement]: for a unit [e] of the fragment of
   FlambdaReaperFlowSpec.v ([in_flow_fragment]), with the interface [u], any
   mode of the traversal ([flag]) and of the rebuild ([preserve_direct_calls]),
   and tables [T] that are closed under the rules ([check_closed]), the unit
   rebuilt with unboxing ([rebuild_unit_unbox]: no change of representation)
   refines the unit rebuilt without
   ([FlambdaReaperRebuild.rebuild_unit_from_tables]), run on its own
   ([program_refines]), if the unboxing that the tables of
   [dominated_by_allocation_point] and [to_unbox] give is local
   ([local_unboxing]) and the reads that the rebuild keeps of the names
   unboxed find their fields ([check_unboxed_reads]). The two rebuilds make
   the same term but for the [Let] of
   an unboxed variable, which the unboxing makes [Let]s of the variables of its
   fields, each bound to what the allocation holds there or, for an alias, to
   the variable of the field of the variable it is an alias of, and, for a field
   unboxed in turn, the variables of its fields to those of the unboxed variable
   that it holds; a read of a field of an unboxed variable, which it makes the
   variable of the field or, for a field unboxed in turn, [Let]s of the
   variables of the fields of the variable read into to those of the field; and
   a handler with an unboxed parameter, which the unboxing gives the variables
   of the fields of the parameter as parameters in its place, and to which an
   [Apply_cont] gives the variables of the fields of its argument. A field
   unboxed is read, with a use, from a use of the variable ([get_fields] of
   [mk_unboxed_fields]; the graph of a unit of the fragment reads no function
   slot and makes no call, so that no use is through one), so the rules give it
   [field_of_constructor_is_used] and give what it holds a use: the plain
   rebuild keeps the field, and binds what it holds. That is what [check_closed]
   is asked for; [is_local_slots_only] says that the fields of a block and of a
   boxed number are not local, and [in_flow_fragment] is the fragment in which
   FlambdaReaperFlowProofs.v says what facts the traversal gives.

   [rebuild_unbox_refines_observably_statement]: with the premises of
   FlambdaReaperRebuildSpec.rebuild_refines_observably_statement, the
   unit rebuilt with unboxing refines the unit up to observation
   ([program_refines_observably]); it is the plain statement composed
   with the first.

   The fragment ([local_unboxing]), for the unboxed fields of the names
   that the tables give ([unit_unboxed_fields]) and the solution:
   - [local_expr]: no closure, no projection, no call and no code (the
     terms of FlambdaReaperRebuildSpec.closure_free); a variable that is
     unboxed is bound by a [Let] of a block of values, of a boxed number,
     of another unboxed variable or of a read of a field unboxed in turn
     ([unboxed_binding_ok]) or is a parameter of a handler ([param_ok]),
     and is otherwise only read, by a [Block_load] or an [Unbox_number]
     ([unboxed_read]), given to a handler of the unit ([apply_cont_ok]),
     or held in a field of an unboxed block that is unboxed in turn: it
     is not an argument of another primitive, of a way out of the unit,
     of a [Switch] or of a static block. A symbol that is unboxed is not
     used.
   - [unboxed_binding_ok]: the unboxed fields of a block are fields of
     the block, of kind [Value], and the argument held there is of kind
     [Value] where the rebuild looks it up: a variable, holding a simple
     of [plain_simple], or a field unboxed in turn, holding another
     variable, unboxed alike and used ([unboxed_arg_ok]); the unboxed
     field of a boxed number is its contents, a variable; an alias is of
     another variable, unboxed alike and used; the read of a field
     unboxed in turn is unboxed alike. Alike is at any depth: each field
     is one of the other's, a variable as a variable and a field unboxed
     in turn as one whose fields are alike ([fields_alike], the OCaml's
     [fold2_subset_u]). Otherwise the OCaml stops with a fatal error, or
     the model makes a field poison or the [Let] [Invalid].
   - [plain_binding_ok]: a read of an unboxed variable is not from a
     mutable field and is into a value or a naked number
     ([read_shape_ok]). That such a read, if the rebuild keeps it (its
     variable has a source and a use), reads a field that is unboxed, a
     variable, is not asked of the fragment but of the solution
     ([check_unboxed_reads], a premise of the statements). Otherwise the
     OCaml stops with a fatal error ([rewrite_field_access]) and the
     model makes the read [Invalid]. It does not follow from
     [check_closed]: a solution that is closed and not least may give
     the variable read into a source and the field read none
     ([unboxed_reads_example_statement]); the least solution passes it.
   - [param_ok]: an unboxed parameter is of kind [Value], and the plain
     rebuild keeps it ([handler_mask]) unless it has no field; a handler
     has as many kinds as parameters. Where the plain rebuild deletes a
     parameter that has fields, it deletes its argument too, and the
     rebuild with unboxing gives the handler the fields: the parameter
     has no use, and its fields are read from uses of it; only a
     solution that is not least gives both.
   - [args_ok]: an [Apply_cont] to a handler of the unit gives an
     unboxed parameter a variable that is unboxed, that has the fields of
     the parameter alike ([unboxed_arg_ok]), and that is used where the
     parameter is kept; it gives a simple of [plain_simple] to a
     parameter that the plain rebuild keeps and that is not unboxed,
     and anything to one that it deletes. Where an unboxed parameter is
     given anything else, the OCaml stops with a fatal error ([get_args]
     of [rewrite_apply_cont_expr]) and the model makes the [Apply_cont]
     [Invalid]; where an unboxed variable is given to a parameter kept
     and not unboxed, the rebuild gives it poison
     ([solution_without_unboxed]).
   Nothing is asked of the variables that stand for the fields: the
   model names them after every name of the unit, each once, so that
   they are fresh (FlambdaReaperUnboxingRefinesProofs.v proves it).

   Of the checks of FlambdaReaperUnboxing.v none is asked. What
   [args_ok] asks of an unboxed parameter [p], dominated by an
   allocation [a] to unbox, they would give, though that is not proved
   here: an argument [y] of [p] has no source but those of [p]
   ([sources_alias]), so that, if it has one, it is dominated by [a]
   ([check_dominated]) and is unboxed (a constant or a symbol gives [p]
   another source); the uses of [p] are among those of [y]
   ([usages_alias]: [y] and [p] have no [any_usage], or [a] would be in
   [cannot_unbox0]), so that the fields of [p] are among those of [y],
   each read into names
   all dominated by the one allocation that the field holds, hence
   unboxed in [p] as in [y]; [y] is used if [p] is ([has_usage_alias]).
   That an unboxed variable is given to no parameter kept and not
   unboxed does not follow from them ([spurious_keep_example]); where a
   solution keeps such a parameter, the parameter has no use, and its
   poison is harmless, but the proof here does not show it. In the least
   solution, which the compiler computes, the parameter has no use and
   is deleted, but for the first parameter of an exception handler,
   which the plain rebuild keeps anyway. The two checks of the fixes of
   FlambdaReaperUnboxingSpec.v ([check_called_my_closure] and
   [check_result_receivers]) are about calls and code. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaUnusedStaticsSpec.
From OxCamlFlambda2 Require FlambdaReaperTupledSpec
  FlambdaReaperCallingConventionsSpec FlambdaReaperUnboxingSpec
  FlambdaReaperSolutionSpec.
Import ListNotations.

(* ======================================================================== *)
(* The fragment                                                             *)
(* ======================================================================== *)

Section Fragment.

(* The unboxed fields of the names, and the solution. *)
Variable unboxed : list (node * unboxed_fields).
Variable sol : solution.

Definition fields_of (x : variable) : option unboxed_fields :=
  Keyed.lookup node_eqb (Node_variable x) unboxed.

(* A variable that is not unboxed. *)
Definition plain_variable (x : variable) : bool :=
  negb (Keyed.is_bound node_eqb (Node_variable x) unboxed).

(* A simple that is not an unboxed name. *)
Definition plain_simple (s : simple) : bool :=
  match s with
  | Var x => plain_variable x
  | Symbol sym => negb (Keyed.is_bound node_eqb (Node_symbol sym) unboxed)
  | Const _ => true
  end.

Definition kinds_env (kinds : finite_map flambda_kind) : rebuild_env :=
  {| env_conts := []; env_kinds := kinds; env_projections := [] |}.

(* The unboxed fields [fields] are among [fields_y], alike: each field of
   [fields] is one of [fields_y], a variable as a variable, and a field
   unboxed in turn as a field unboxed in turn whose fields are alike, at
   any depth ([unboxed_field_pairs], the OCaml's [fold2_subset_u], does
   not fail). *)
Definition fields_alike (fields fields_y : unboxed_fields) : bool :=
  match unboxed_field_pairs Value (Unboxed fields) (Unboxed fields_y) with
  | Some _ => true
  | None => false
  end.

(* The argument [a] of a parameter unboxed with the fields [fields], which
   the plain rebuild keeps if [keep], or the variable that an unboxed
   variable is an alias of or that a field unboxed in turn holds ([keep]
   then): a variable unboxed that has the fields alike, and that is used
   if the parameter is kept. *)
Definition unboxed_arg_ok (kinds : finite_map flambda_kind) (keep : bool)
    (fields : unboxed_fields) (a : simple) : bool :=
  match a with
  | Var y =>
    match fields_of y with
    | Some fields_y =>
      fields_alike fields fields_y
      && (negb keep
          || is_var_used sol y (kind_of_variable (kinds_env kinds) y))
    | None => false
    end
  | Symbol _ | Const _ => false
  end.

(* The [Let] of an unboxed variable [x], of kind [kind], to [d], the
   unboxed fields of the variable being [fields]: a block of values, of
   which each unboxed field is a field, holding an argument of kind
   [Value] where the rebuild looks it up, of [plain_simple] if the field
   is a variable, or else a variable unboxed alike other than [x], used,
   whose fields the field has, alike; a boxed number, whose unboxed field
   is its contents, a variable; another variable unboxed alike, of which
   [x] is an alias; or the read of a field of another unboxed variable
   that is unboxed in turn, alike. *)
Definition unboxed_binding_ok (kinds : finite_map flambda_kind)
    (x : variable) (kind : flambda_kind) (d : named) (fields : unboxed_fields)
    : bool :=
  match kind, d with
  | Value, Prim (Variadic (Make_block (Block_kind_values _))) args =>
    forallb
      (fun '(f, u) =>
         match f, u with
         | Field_block i Value, Not_unboxed _ =>
           match nth_error args i with
           | Some arg =>
             plain_simple arg
             && kind_eqb (simple_kind (kinds_env kinds) arg) Value
           | None => false
           end
         | Field_block i Value, Unboxed nested =>
           match nth_error args i with
           | Some (Var b) =>
             negb (Pos.eqb x b)
             && kind_eqb (simple_kind (kinds_env kinds) (Var b)) Value
             && unboxed_arg_ok kinds true nested (Var b)
           | _ => false
           end
         | _, _ => false
         end)
      fields
  | Value, Prim (Unary (Box_number bn)) [contents] =>
    plain_simple contents
    && forallb
         (fun '(f, u) =>
            match u with
            | Not_unboxed _ => field_eqb f (Field_boxed_number bn)
            | Unboxed _ => false
            end)
         fields
  | Value, Simple (Var y) =>
    negb (Pos.eqb x y) && unboxed_arg_ok kinds true fields (Var y)
  | Value, Prim (Unary (Block_load _ _ i)) [Var y] =>
    negb (Pos.eqb x y)
    && match fields_of y with
       | Some fields_y =>
         match Keyed.lookup field_eqb (Field_block (Z.to_nat i) Value) fields_y
         with
         | Some (Unboxed nested) => fields_alike fields nested
         | _ => false
         end
       | None => false
       end
  | _, _ => false
  end.

(* The read of a field of an unboxed variable: the field, and the unboxed
   fields of the variable. *)
Definition unboxed_read (d : named) : option (field * unboxed_fields) :=
  match d with
  | Prim (Unary (Block_load _ _ _)) [Var x]
  | Prim (Unary (Unbox_number _)) [Var x] =>
    match field_read d, fields_of x with
    | Some (f, _), Some fields => Some (f, fields)
    | _, _ => None
    end
  | _ => None
  end.

(* The read [d] of an unboxed variable into a variable of kind [kind]:
   into a value or a naked number (whose use and source the solution
   has), and not from a mutable field (the traversal gives a mutable
   read any source, whatever the field holds, so that the least solution
   may fail [check_unboxed_reads] at a mutable read, in code that never
   runs, of a field with no source; the OCaml makes such a read
   [Invalid]). *)
Definition read_shape_ok (kind : flambda_kind) (d : named) : bool :=
  match kind, d with
  | Region, _ | Rec_info, _ => false
  | _, Prim (Unary (Block_load _ Mutable _)) _ => false
  | _, _ => true
  end.

(* The [Let] of [x], which is not unboxed: a read of an unboxed variable
   of [read_shape_ok]; or a simple or a primitive of simples of
   [plain_simple]. That a read that the rebuild keeps reads a field
   unboxed as a variable is [check_unboxed_reads], of the solution. *)
Definition plain_binding_ok (x : variable) (kind : flambda_kind) (d : named)
    : bool :=
  match unboxed_read d with
  | Some _ => read_shape_ok kind d
  | None =>
    match d with
    | Simple s => plain_simple s
    | Prim _ args => forallb plain_simple args
    | _ => false
    end
  end.

Definition local_binding (binding : static_binding) : bool :=
  match binding with
  | Bind_symbol _ (Static_const_block _ fields) =>
    forallb plain_simple fields
  | _ => false
  end.

(* The arguments [args] of an [Apply_cont] to a handler of the unit whose
   parameters are [params], each with whether the plain rebuild keeps it:
   an unboxed parameter is given an argument unboxed alike, a parameter
   kept that is not unboxed a simple of [plain_simple]. *)
Fixpoint args_ok (kinds : finite_map flambda_kind)
    (params : list (variable * bool)) (args : list simple) : bool :=
  match params, args with
  | _, [] => true
  | [], _ :: _ => forallb plain_simple args
  | (p, keep) :: params, a :: args =>
    match fields_of p with
    | Some fields => unboxed_arg_ok kinds keep fields a
    | None => negb keep || plain_simple a
    end
    && args_ok kinds params args
  end.

(* [conts] are the parameters of the handlers in scope, with what the
   plain rebuild keeps of them; any other continuation is given simples
   of [plain_simple]. *)
Definition apply_cont_ok (kinds : finite_map flambda_kind)
    (conts : finite_map (list (variable * bool))) (ac : apply_cont) : bool :=
  match lookup (ac_continuation ac) conts with
  | Some params => args_ok kinds params (ac_args ac)
  | None => forallb plain_simple (ac_args ac)
  end.

(* A parameter [p] of a handler, of kind [kind], which the plain rebuild
   keeps if [keep]: if it is unboxed, of kind [Value], and kept unless it
   has no field. *)
Definition param_ok (p : variable) (kind : flambda_kind) (keep : bool)
    : bool :=
  match fields_of p with
  | Some fields =>
    kind_eqb kind Value
    && (keep || match fields with [] => true | _ :: _ => false end)
  | None => true
  end.

Definition handler_variables (h : continuation_handler) : list variable :=
  match h with Handler params _ _ => params end.

(* [kinds] are those of the variables in scope, as the rebuild has them
   ([env_kinds]), and [conts] the parameters of the handlers in scope. *)
Fixpoint local_expr (kinds : finite_map flambda_kind)
    (conts : finite_map (list (variable * bool))) (e : expr) : bool :=
  match e with
  | Let x kind d body =>
    match fields_of x with
    | Some fields => unboxed_binding_ok kinds x kind d fields
    | None => plain_binding_ok x kind d
    end
    && local_expr ((x, kind) :: kinds) conts body
  | Let_static bindings body =>
    forallb local_binding bindings && local_expr kinds conts body
  | Let_cont_non_recursive k h body =>
    let mask := handler_mask sol (handler_is_exn_handler h) h in
    local_handler kinds conts mask h
    && local_expr kinds ((k, combine (handler_variables h) mask) :: conts)
         body
  | Let_cont_recursive handlers body =>
    let conts :=
      map
        (fun '(k, h) =>
           (k, combine (handler_variables h) (handler_mask sol false h)))
        handlers
      ++ conts
    in
    forallb
      (fun '(_, h) => local_handler kinds conts (handler_mask sol false h) h)
      handlers
    && local_expr kinds conts body
  | Apply _ => false
  | Apply_cont ac => apply_cont_ok kinds conts ac
  | Switch scrutinee arms =>
    plain_simple scrutinee
    && forallb (fun '(_, ac) => apply_cont_ok kinds conts ac) arms
  | Invalid => true
  end
with local_handler (kinds : finite_map flambda_kind)
    (conts : finite_map (list (variable * bool))) (mask : list bool)
    (h : continuation_handler) : bool :=
  match h with
  | Handler params metadata handler =>
    Nat.eqb (length params) (length (hm_params_kinds metadata))
    && forallb (fun '((p, kind), keep) => param_ok p kind keep)
         (combine (combine params (hm_params_kinds metadata)) mask)
    && local_expr (rev (combine params (hm_params_kinds metadata)) ++ kinds)
         conts handler
  end.

End Fragment.

(* The unit [e] is of the fragment, for the unboxing that the tables
   give from its graph [F]. *)
Definition local_unboxing (F : facts) (T : tables)
    (dominated : list (node * node)) (to_unbox : list node) (e : expr)
    : bool :=
  let unboxed := fst (unit_unboxed_fields F T dominated to_unbox e) in
  local_expr unboxed (solution_of_tables T) [] [] e.

(* The reads that the rebuild keeps of a name unboxed find their field
   ([rewrite_field_access] does not stop with a fatal error): for each
   read [(x, f, y)] of the graph into a node [x] that has a source and a
   use and that no allocation to unbox dominates, of a node [y] that an
   allocation [a] to unbox dominates, [y] has no [any_usage] and the
   field [f] of [a] has a source ([get_single_field_source] is not
   [No_source]). The compiler's tables pass it, on the units of the
   fragment: [a] is not to be unboxed if [y], an alias of it, has any
   usage ([cannot_unbox0]); and in the least solution, a source of [x],
   read immutably off [y] (whose only allocation with a source is [a],
   [check_dominated]), is one of a node held in the field [f] of an
   allocation that [y] may be ([alias_accessor_source] or
   [alias_constructor_usage]), hence of [a], so that the field of [a]
   has a source: a node that the least solution gives a source, and not
   [any_source], has among its [sources] an allocation with a source.
   A solution closed under the rules but not least may give [x] a
   source and the field none ([unboxed_reads_example_statement]). *)
Definition check_unboxed_reads (F : facts) (T : tables)
    (dominated : list (node * node)) (to_unbox : list node) : bool :=
  forallb
    (fun '(x, f, y) =>
       negb (mem_node x (t_has_source T) && mem_node x (t_has_usage T)
             && negb (has_to_be_unboxed dominated to_unbox x))
       || forallb
            (fun '(a, y') =>
               negb (node_eqb y y' && mem_node a to_unbox)
               || (negb (mem_node y (t_any_usage T))
                   && match get_single_field_source F T a f with
                      | No_source => false
                      | One_source _ | Many_sources => true
                      end))
            dominated)
    (f_accessor F).

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Definition rebuild_unbox_refines_rebuild_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    local_unboxing (traverse_unit flag u e) T dominated to_unbox e = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines
      (rebuild_unit_from_tables (traverse_unit flag u e) T
         preserve_direct_calls e)
      (rebuild_unit_unbox (traverse_unit flag u e) T cannot_change code_deps
         dominated to_unbox preserve_direct_calls e).

Definition rebuild_unbox_refines_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    static_constants (unit_codes e) e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    local_unboxing (traverse_unit flag u e) T dominated to_unbox e = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

(* ======================================================================== *)
(* An example                                                               *)
(* ======================================================================== *)

(* The fragment holds of more than nothing: a unit that unboxes a block,
   read in its body and in a handler, and a boxed number:

     let x10 = [7; 8] in
     let x11 = box_int64 5 in
     let k20 x30 = (let x31 = x10.(0) in let x32 = [x30; x31] in
                    return x32) in
     let x12 = x10.(1) in
     let x13 = unbox_int64 x11 in
     let x14 = box_int64 x13 in
     let x15 = [x12; x14] in
     k20 x15

   Its solution, as the solver of the test tool computes it, passes the
   checks of the plain statement, and those of the unboxing with the
   tables of [dominated_by_allocation_point] and [to_unbox] that the rules
   give, which unbox [x10] and [x11] ([x14] escapes); the unit meets the
   premises of [rebuild_unbox_refines_observably_statement]. The rebuild
   binds the variables of the fields, 34 and 35 for [x10] and 33 for
   [x11], to 7, 8 and 5, and reads them in place of the loads; the unit and
   the rebuilt unit exit with the same block. *)
Module LocalExample.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition box64 (s : simple) : named :=
  Prim (Unary (Box_number Boxable_naked_int64)) [s].

Definition unbox64 (s : simple) : named :=
  Prim (Unary (Unbox_number Boxable_naked_int64)) [s].

Definition unit_expr : expr :=
  Let 10 Value (make_block [int 7; int 8])
    (Let 11 Value (box64 (Const (Naked_int64 5)))
       (Let_cont_non_recursive 20
          (Handler [30] (make_handler_metadata [Value])
             (Let 31 Value (load 0 (Var 10))
                (Let 32 Value (make_block [Var 30; Var 31])
                   (return_to 1 [Var 32]))))
          (Let 12 Value (load 1 (Var 10))
             (Let 13 (Naked_number Kind_naked_int64) (unbox64 (Var 11))
                (Let 14 Value (box64 (Var 13))
                   (Let 15 Value (make_block [Var 12; Var 14])
                      (return_to 20 [Var 15]))))))).

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition int64 : field := Field_boxed_number Boxable_naked_int64.

Definition tables : tables :=
  {| t_alias :=
       [(v 30, v 15); (m 1, v 32); (v 13, consts); (v 31, consts);
        (v 12, consts)];
     t_any_usage :=
       [m 1; m 2; v 32; v 30; v 31; consts; v 15; v 12; v 14; v 13];
     t_any_source := [consts; v 13; v 31; v 12];
     t_has_usage :=
       [m 1; m 2; v 32; v 30; v 31; consts; v 15; v 12; v 14; v 13; v 11;
        v 10];
     t_has_source :=
       [consts; v 15; v 32; v 11; v 10; v 30; m 1; v 13; v 31; v 12; v 14];
     t_usages := [(v 11, v 11); (v 10, v 10)];
     t_sources :=
       [(v 15, v 15); (v 14, v 14); (v 32, v 32); (v 11, v 11); (v 10, v 10);
        (v 30, v 15); (m 1, v 32)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used :=
       [(v 15, Field_block 0 Value); (v 15, Field_block 1 Value);
        (v 15, Field_is_int); (v 15, Field_get_tag); (v 14, int64);
        (v 32, Field_block 0 Value); (v 32, Field_block 1 Value);
        (v 32, Field_is_int); (v 32, Field_get_tag); (v 11, int64);
        (v 10, Field_block 0 Value); (v 10, Field_block 1 Value)] |}.

Definition cannot_change : list node := derive_cannot_change facts tables.

Definition dominated : list (node * node) := derive_dominated tables.

Definition to_unbox : list node :=
  derive_to_unbox no_local facts tables cannot_change max_unbox_size false.

Definition rebuilt : expr :=
  Let 35 Value (Simple (int 8))
    (Let 34 Value (Simple (int 7))
       (Let 33 (Naked_number Kind_naked_int64)
          (Simple (Const (Naked_int64 5)))
          (Let_cont_non_recursive 20
             (Handler [30] (make_handler_metadata [Value])
                (Let 31 Value (Simple (Var 34))
                   (Let 32 Value (make_block [Var 30; Var 31])
                      (return_to 1 [Var 32]))))
             (Let 12 Value (Simple (Var 35))
                (Let 13 (Naked_number Kind_naked_int64) (Simple (Var 33))
                   (Let 14 Value (box64 (Var 13))
                      (Let 15 Value (make_block [Var 12; Var 14])
                         (return_to 20 [Var 15])))))))).

Definition result : value :=
  Value_block 0
    [Value_block 0
       [Value_const (Tagged_immediate 8); Value_boxed_number (Naked_int64 5)];
     Value_const (Tagged_immediate 7)].

End LocalExample.

Definition local_example_statement : Prop :=
  is_local_slots_only Examples.no_local
  ∧ in_flow_fragment LocalExample.unit_expr = true
  ∧ static_constants (unit_codes LocalExample.unit_expr)
      LocalExample.unit_expr
    = true
  ∧ results_kinded [] [] LocalExample.unit_expr = true
  ∧ well_formed_unit Examples.interface LocalExample.unit_expr = true
  ∧ exits_one_value Examples.interface LocalExample.unit_expr = true
  ∧ check_closed Examples.no_local LocalExample.facts LocalExample.tables
    = true
  ∧ check_field_used_exact Examples.no_local LocalExample.facts
      LocalExample.tables
    = true
  ∧ check_dominated LocalExample.tables LocalExample.dominated = true
  ∧ check_to_unbox Examples.no_local LocalExample.facts LocalExample.tables
      LocalExample.cannot_change Examples.max_unbox_size false
      LocalExample.to_unbox
    = true
  ∧ LocalExample.to_unbox
    = [Node_variable 11%positive; Node_variable 10%positive]
  ∧ local_unboxing LocalExample.facts LocalExample.tables
      LocalExample.dominated LocalExample.to_unbox LocalExample.unit_expr
    = true
  ∧ rebuild_unit_unbox LocalExample.facts LocalExample.tables
      LocalExample.cannot_change
      (unit_code_deps Auto Examples.interface LocalExample.unit_expr)
      LocalExample.dominated LocalExample.to_unbox true
      LocalExample.unit_expr
    = LocalExample.rebuilt
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own LocalExample.unit_expr)
      (Exits 1%positive [LocalExample.result] empty_heap)
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own LocalExample.rebuilt)
      (Exits 1%positive [LocalExample.result] empty_heap).

(* An unboxed block and an unboxed boxed number given to a handler whose
   parameters are unboxed alike, the block twice, once to a parameter
   that is not used:

     let k20 x30 x31 x37 =
       (let x32 = x30.(0) in let x33 = x30.(1) in
        let x34 = unbox_int64 x31 in let x35 = box_int64 x34 in
        let x36 = [x32; x33; x35] in return x36) in
     let x10 = [7; 8] in
     let x11 = box_int64 5 in
     k20 x10 x11 x10

   The tables are those of the solver of the test tool; they pass the
   checks, and the rules unbox [x10] and [x11], hence [x30], [x31] and
   [x37], which dominates them. The rebuild binds the variables of the
   fields of [x10], 39 and 40, and of [x11], 38, gives them to [k20],
   whose parameters are the variables of the fields of [x30], 41 and 42,
   and of [x31], 43; [x37], which has no field read and no use, is a
   parameter neither of the plain rebuild nor of this one. The unit and
   the rebuilt unit exit with the same block. *)
Module ContinuationExample.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition unit_expr : expr :=
  Let_cont_non_recursive 20
    (Handler [30; 31; 37] (make_handler_metadata [Value; Value; Value])
       (Let 32 Value (load 0 (Var 30))
          (Let 33 Value (load 1 (Var 30))
             (Let 34 (Naked_number Kind_naked_int64)
                (LocalExample.unbox64 (Var 31))
                (Let 35 Value (LocalExample.box64 (Var 34))
                   (Let 36 Value (make_block [Var 32; Var 33; Var 35])
                      (return_to 1 [Var 36])))))))
    (Let 10 Value (make_block [int 7; int 8])
       (Let 11 Value (LocalExample.box64 (Const (Naked_int64 5)))
          (return_to 20 [Var 10; Var 11; Var 10]))).

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition tables : tables :=
  {| t_alias :=
       [(v 30, v 10); (v 31, v 11); (v 37, v 10); (m 1, v 36);
        (v 34, consts); (v 32, consts); (v 33, consts)];
     t_any_usage := [m 1; m 2; v 36; v 32; v 33; v 35; consts; v 34];
     t_any_source := [consts; v 34; v 32; v 33];
     t_has_usage :=
       [m 1; m 2; v 36; v 32; v 33; v 35; consts; v 34; v 31; v 30; v 10;
        v 11];
     t_has_source :=
       [consts; v 11; v 10; v 36; v 30; v 31; v 37; m 1; v 34; v 32; v 33;
        v 35];
     t_usages := [(v 31, v 31); (v 30, v 30); (v 10, v 30); (v 11, v 31)];
     t_sources :=
       [(v 11, v 11); (v 10, v 10); (v 36, v 36); (v 35, v 35);
        (v 30, v 10); (v 31, v 11); (v 37, v 10); (m 1, v 36)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used :=
       [(v 11, LocalExample.int64); (v 10, Field_block 0 Value);
        (v 10, Field_block 1 Value); (v 36, Field_block 0 Value);
        (v 36, Field_block 1 Value); (v 36, Field_block 2 Value);
        (v 36, Field_is_int); (v 36, Field_get_tag);
        (v 35, LocalExample.int64)] |}.

Definition cannot_change : list node := derive_cannot_change facts tables.

Definition dominated : list (node * node) := derive_dominated tables.

Definition to_unbox : list node :=
  derive_to_unbox no_local facts tables cannot_change max_unbox_size false.

Definition rebuilt : expr :=
  Let_cont_non_recursive 20
    (Handler [41; 42; 43]
       (make_handler_metadata [Value; Value; Naked_number Kind_naked_int64])
       (Let 32 Value (Simple (Var 41))
          (Let 33 Value (Simple (Var 42))
             (Let 34 (Naked_number Kind_naked_int64) (Simple (Var 43))
                (Let 35 Value (LocalExample.box64 (Var 34))
                   (Let 36 Value (make_block [Var 32; Var 33; Var 35])
                      (return_to 1 [Var 36])))))))
    (Let 40 Value (Simple (int 8))
       (Let 39 Value (Simple (int 7))
          (Let 38 (Naked_number Kind_naked_int64)
             (Simple (Const (Naked_int64 5)))
             (return_to 20 [Var 39; Var 40; Var 38])))).

Definition result : value :=
  Value_block 0
    [Value_const (Tagged_immediate 7); Value_const (Tagged_immediate 8);
     Value_boxed_number (Naked_int64 5)].

End ContinuationExample.

Definition continuation_example_statement : Prop :=
  in_flow_fragment ContinuationExample.unit_expr = true
  ∧ static_constants (unit_codes ContinuationExample.unit_expr)
      ContinuationExample.unit_expr
    = true
  ∧ results_kinded [] [] ContinuationExample.unit_expr = true
  ∧ well_formed_unit Examples.interface ContinuationExample.unit_expr = true
  ∧ exits_one_value Examples.interface ContinuationExample.unit_expr = true
  ∧ check_closed Examples.no_local ContinuationExample.facts
      ContinuationExample.tables
    = true
  ∧ check_field_used_exact Examples.no_local ContinuationExample.facts
      ContinuationExample.tables
    = true
  ∧ check_dominated ContinuationExample.tables ContinuationExample.dominated
    = true
  ∧ check_to_unbox Examples.no_local ContinuationExample.facts
      ContinuationExample.tables ContinuationExample.cannot_change
      Examples.max_unbox_size false ContinuationExample.to_unbox
    = true
  ∧ ContinuationExample.to_unbox
    = [Node_variable 10%positive; Node_variable 11%positive]
  ∧ local_unboxing ContinuationExample.facts ContinuationExample.tables
      ContinuationExample.dominated ContinuationExample.to_unbox
      ContinuationExample.unit_expr
    = true
  ∧ rebuild_unit_unbox ContinuationExample.facts ContinuationExample.tables
      ContinuationExample.cannot_change
      (unit_code_deps Auto Examples.interface ContinuationExample.unit_expr)
      ContinuationExample.dominated ContinuationExample.to_unbox true
      ContinuationExample.unit_expr
    = ContinuationExample.rebuilt
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own ContinuationExample.unit_expr)
      (Exits 1%positive [ContinuationExample.result] empty_heap)
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own ContinuationExample.rebuilt)
      (Exits 1%positive [ContinuationExample.result] empty_heap).

(* A loop: a block given to a recursive handler, whose unboxed parameter
   is given back to it by an arm of a [Switch]:

     let x10 = [7; 8] in
     let rec k20 x30 x31 =
       (let k21 () = (let x32 = x30.(1) in return x32) in
        switch x31 with 0 -> k21 () | 1 -> k20 x30 0) in
     k20 x10 1

   With the tables of the solver, which pass the checks, the rules unbox
   [x10] and [x30]; the rebuild binds the variable of the field of
   [x10], 33, to 8, and the parameter of [k20] is the variable of the
   field of [x30], 34. The unit and the rebuilt unit exit with 8. *)
Module LoopExample.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition imm (i : Z) : simple := Const (Naked_immediate i).

Definition goto_args (k : continuation) (args : list simple) : apply_cont :=
  {| ac_continuation := k; ac_args := args; ac_trap_action := None |}.

Definition unit_expr : expr :=
  Let 10 Value (make_block [int 7; int 8])
    (Let_cont_recursive
       [(20%positive,
         Handler [30; 31]
           (make_handler_metadata [Value; Naked_number Kind_naked_immediate])
           (Let_cont_non_recursive 21
              (Handler [] (make_handler_metadata [])
                 (Let 32 Value (load 1 (Var 30)) (return_to 1 [Var 32])))
              (Switch (Var 31)
                 [(0%Z, goto_args 21 []);
                  (1%Z, goto_args 20 [Var 30; imm 0])])))]
       (return_to 20 [Var 10; imm 1])).

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition tables : tables :=
  {| t_alias :=
       [(v 30, v 10); (v 31, consts); (v 30, v 30); (m 1, v 32);
        (v 32, consts)];
     t_any_usage := [m 1; v 31; m 2; consts; v 32];
     t_any_source := [consts; v 31; v 32; m 1];
     t_has_usage := [m 1; v 31; m 2; consts; v 32; v 30; v 10];
     t_has_source := [consts; v 31; v 10; v 30; v 32; m 1];
     t_usages := [(v 30, v 30); (v 10, v 30)];
     t_sources := [(v 10, v 10); (v 30, v 10)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used := [(v 10, Field_block 1 Value)] |}.

Definition cannot_change : list node := derive_cannot_change facts tables.

Definition dominated : list (node * node) := derive_dominated tables.

Definition to_unbox : list node :=
  derive_to_unbox no_local facts tables cannot_change max_unbox_size false.

Definition rebuilt : expr :=
  Let 33 Value (Simple (int 8))
    (Let_cont_recursive
       [(20%positive,
         Handler [34; 31]
           (make_handler_metadata [Value; Naked_number Kind_naked_immediate])
           (Let_cont_non_recursive 21
              (Handler [] (make_handler_metadata [])
                 (Let 32 Value (Simple (Var 34)) (return_to 1 [Var 32])))
              (Switch (Var 31)
                 [(0%Z, goto_args 21 []);
                  (1%Z, goto_args 20 [Var 34; imm 0])])))]
       (return_to 20 [Var 33; imm 1])).

End LoopExample.

Definition loop_example_statement : Prop :=
  in_flow_fragment LoopExample.unit_expr = true
  ∧ static_constants (unit_codes LoopExample.unit_expr) LoopExample.unit_expr
    = true
  ∧ results_kinded [] [] LoopExample.unit_expr = true
  ∧ well_formed_unit Examples.interface LoopExample.unit_expr = true
  ∧ exits_one_value Examples.interface LoopExample.unit_expr = true
  ∧ check_closed Examples.no_local LoopExample.facts LoopExample.tables
    = true
  ∧ check_field_used_exact Examples.no_local LoopExample.facts
      LoopExample.tables
    = true
  ∧ check_dominated LoopExample.tables LoopExample.dominated = true
  ∧ check_to_unbox Examples.no_local LoopExample.facts LoopExample.tables
      LoopExample.cannot_change Examples.max_unbox_size false
      LoopExample.to_unbox
    = true
  ∧ LoopExample.to_unbox = [Node_variable 10%positive]
  ∧ local_unboxing LoopExample.facts LoopExample.tables
      LoopExample.dominated LoopExample.to_unbox LoopExample.unit_expr
    = true
  ∧ rebuild_unit_unbox LoopExample.facts LoopExample.tables
      LoopExample.cannot_change
      (unit_code_deps Auto Examples.interface LoopExample.unit_expr)
      LoopExample.dominated LoopExample.to_unbox true LoopExample.unit_expr
    = LoopExample.rebuilt
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own LoopExample.unit_expr)
      (Exits 1%positive [Value_const (Tagged_immediate 8)] empty_heap)
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own LoopExample.rebuilt)
      (Exits 1%positive [Value_const (Tagged_immediate 8)] empty_heap).

(* A block in a block, both unboxed ([Examples.nested_block]):

     let x10 = [7; 8] in
     let x11 = [x10] in
     let x12 = x11.(0) in
     let x13 = x12.(1) in
     return x13

   With the tables of FlambdaReaperUnboxing.v, which pass the checks, the
   rules unbox [x11] and [x10], hence [x12], which [x10] dominates. Field 0
   of [x11] is unboxed in turn, its field 1 the variable 14; the rebuild
   binds the variable of field 1 of [x10], 15, to 8, 14 to 15, the
   variable of field 1 of [x12], 16, to 14, and [x13] to 16. The unit and
   the rebuilt unit exit with 8. *)
Definition nested_example_statement : Prop :=
  let F := Examples.nested_block_facts in
  let T := Examples.nested_block_tables in
  let e := Examples.nested_block in
  let cannot_change := derive_cannot_change F T in
  let dominated := derive_dominated T in
  let to_unbox :=
    derive_to_unbox Examples.no_local F T cannot_change
      Examples.max_unbox_size false
  in
  let rebuilt :=
    Let 15%positive Value (Simple (Examples.int 8))
      (Let 14%positive Value (Simple (Var 15%positive))
         (Let 16%positive Value (Simple (Var 14%positive))
            (Let 13%positive Value (Simple (Var 16%positive))
               (Examples.return_to 1%positive [Var 13%positive]))))
  in
  in_flow_fragment e = true
  ∧ static_constants (unit_codes e) e = true
  ∧ results_kinded [] [] e = true
  ∧ well_formed_unit Examples.interface e = true
  ∧ exits_one_value Examples.interface e = true
  ∧ check_closed Examples.no_local F T = true
  ∧ check_field_used_exact Examples.no_local F T = true
  ∧ check_dominated T dominated = true
  ∧ check_to_unbox Examples.no_local F T cannot_change
      Examples.max_unbox_size false to_unbox
    = true
  ∧ to_unbox = [Node_variable 11%positive; Node_variable 10%positive]
  ∧ local_unboxing F T dominated to_unbox e = true
  ∧ rebuild_unit_unbox F T cannot_change
      (unit_code_deps Auto Examples.interface e) dominated to_unbox true e
    = rebuilt
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own e)
      (Exits 1%positive [Value_const (Tagged_immediate 8)] empty_heap)
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own rebuilt)
      (Exits 1%positive [Value_const (Tagged_immediate 8)] empty_heap).

(* A block in a block given to a handler whose parameter is unboxed
   alike, with a field unboxed in turn:

     let k20 x30 = (let x31 = x30.(0) in let x32 = x31.(1) in
                    return x32) in
     let x10 = [7; 8] in
     let x11 = [x10] in
     k20 x11

   With the tables of the solver, which pass the checks, the rules unbox
   [x11] and [x10], hence [x30] and [x31], which they dominate. Field 0
   of [x11] and of [x30] is unboxed in turn, its field 1 the variables
   33 and 35. The rebuild binds the variable of field 1 of [x10], 34, to
   8, and 33 to 34, gives 33 to [k20], whose parameter is 35, binds the
   variable of field 1 of [x31], 36, to 35, and [x32] to 36. The unit
   and the rebuilt unit exit with 8. *)
Module NestedParamExample.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition unit_expr : expr :=
  Let_cont_non_recursive 20
    (Handler [30] (make_handler_metadata [Value])
       (Let 31 Value (load 0 (Var 30))
          (Let 32 Value (load 1 (Var 31)) (return_to 1 [Var 32]))))
    (Let 10 Value (make_block [int 7; int 8])
       (Let 11 Value (make_block [Var 10]) (return_to 20 [Var 11]))).

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition tables : tables :=
  {| t_alias := [(v 30, v 11); (m 1, v 32); (v 31, v 10); (v 32, consts)];
     t_any_usage := [m 1; m 2; v 32; consts];
     t_any_source := [consts; v 32; m 1];
     t_has_usage := [m 1; m 2; v 32; v 31; v 30; v 11; v 10; consts];
     t_has_source := [consts; v 11; v 10; v 30; v 31; v 32; m 1];
     t_usages := [(v 31, v 31); (v 30, v 30); (v 11, v 30); (v 10, v 31)];
     t_sources := [(v 11, v 11); (v 10, v 10); (v 30, v 11); (v 31, v 10)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used :=
       [(v 11, Field_block 0 Value); (v 10, Field_block 1 Value)] |}.

Definition cannot_change : list node := derive_cannot_change facts tables.

Definition dominated : list (node * node) := derive_dominated tables.

Definition to_unbox : list node :=
  derive_to_unbox no_local facts tables cannot_change max_unbox_size false.

Definition rebuilt : expr :=
  Let_cont_non_recursive 20
    (Handler [35] (make_handler_metadata [Value])
       (Let 36 Value (Simple (Var 35))
          (Let 32 Value (Simple (Var 36)) (return_to 1 [Var 32]))))
    (Let 34 Value (Simple (int 8))
       (Let 33 Value (Simple (Var 34)) (return_to 20 [Var 33]))).

End NestedParamExample.

Definition nested_param_example_statement : Prop :=
  in_flow_fragment NestedParamExample.unit_expr = true
  ∧ static_constants (unit_codes NestedParamExample.unit_expr)
      NestedParamExample.unit_expr
    = true
  ∧ results_kinded [] [] NestedParamExample.unit_expr = true
  ∧ well_formed_unit Examples.interface NestedParamExample.unit_expr = true
  ∧ exits_one_value Examples.interface NestedParamExample.unit_expr = true
  ∧ check_closed Examples.no_local NestedParamExample.facts
      NestedParamExample.tables
    = true
  ∧ check_field_used_exact Examples.no_local NestedParamExample.facts
      NestedParamExample.tables
    = true
  ∧ check_dominated NestedParamExample.tables NestedParamExample.dominated
    = true
  ∧ check_to_unbox Examples.no_local NestedParamExample.facts
      NestedParamExample.tables NestedParamExample.cannot_change
      Examples.max_unbox_size false NestedParamExample.to_unbox
    = true
  ∧ NestedParamExample.to_unbox
    = [Node_variable 11%positive; Node_variable 10%positive]
  ∧ fst
      (unit_unboxed_fields NestedParamExample.facts NestedParamExample.tables
         NestedParamExample.dominated NestedParamExample.to_unbox
         NestedParamExample.unit_expr)
    = [(Node_variable 11%positive,
        [(Field_block 0 Value,
          Unboxed [(Field_block 1 Value, Not_unboxed 33%positive)])]);
       (Node_variable 10%positive,
        [(Field_block 1 Value, Not_unboxed 34%positive)]);
       (Node_variable 30%positive,
        [(Field_block 0 Value,
          Unboxed [(Field_block 1 Value, Not_unboxed 35%positive)])]);
       (Node_variable 31%positive,
        [(Field_block 1 Value, Not_unboxed 36%positive)])]
  ∧ local_unboxing NestedParamExample.facts NestedParamExample.tables
      NestedParamExample.dominated NestedParamExample.to_unbox
      NestedParamExample.unit_expr
    = true
  ∧ rebuild_unit_unbox NestedParamExample.facts NestedParamExample.tables
      NestedParamExample.cannot_change
      (unit_code_deps Auto Examples.interface NestedParamExample.unit_expr)
      NestedParamExample.dominated NestedParamExample.to_unbox true
      NestedParamExample.unit_expr
    = NestedParamExample.rebuilt
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own NestedParamExample.unit_expr)
      (Exits 1%positive [Value_const (Tagged_immediate 8)] empty_heap)
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own NestedParamExample.rebuilt)
      (Exits 1%positive [Value_const (Tagged_immediate 8)] empty_heap).

(* An alias of an unboxed block whose field is unboxed in turn:

     let x10 = [7; 8] in
     let x11 = [x10] in
     let x12 = x11 in
     let x13 = x12.(0) in
     let x14 = x13.(1) in
     return x14

   With the tables of the solver, which pass the checks, the rules unbox
   [x11] and [x10], hence [x12] and [x13], which they dominate. Field 0
   of [x11] and of [x12] is unboxed in turn, its field 1 the variables
   15 and 17. The rebuild binds the variable of field 1 of [x10], 16, to
   8, 15 to 16, 17 to 15 (the alias), the variable of field 1 of [x13],
   18, to 17, and [x14] to 18. The unit and the rebuilt unit exit with
   8. *)
Module NestedAliasExample.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition unit_expr : expr :=
  Let 10 Value (make_block [int 7; int 8])
    (Let 11 Value (make_block [Var 10])
       (Let 12 Value (Simple (Var 11))
          (Let 13 Value (load 0 (Var 12))
             (Let 14 Value (load 1 (Var 13)) (return_to 1 [Var 14]))))).

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition tables : tables :=
  {| t_alias := [(m 1, v 14); (v 12, v 11); (v 13, v 10); (v 14, consts)];
     t_any_usage := [m 1; m 2; v 14; consts];
     t_any_source := [consts; v 14; m 1];
     t_has_usage := [m 1; m 2; v 14; v 13; v 12; v 11; v 10; consts];
     t_has_source := [consts; v 11; v 10; v 12; v 13; v 14; m 1];
     t_usages := [(v 13, v 13); (v 12, v 12); (v 11, v 12); (v 10, v 13)];
     t_sources := [(v 11, v 11); (v 10, v 10); (v 12, v 11); (v 13, v 10)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used :=
       [(v 11, Field_block 0 Value); (v 10, Field_block 1 Value)] |}.

Definition cannot_change : list node := derive_cannot_change facts tables.

Definition dominated : list (node * node) := derive_dominated tables.

Definition to_unbox : list node :=
  derive_to_unbox no_local facts tables cannot_change max_unbox_size false.

Definition rebuilt : expr :=
  Let 16 Value (Simple (int 8))
    (Let 15 Value (Simple (Var 16))
       (Let 17 Value (Simple (Var 15))
          (Let 18 Value (Simple (Var 17))
             (Let 14 Value (Simple (Var 18)) (return_to 1 [Var 14]))))).

End NestedAliasExample.

Definition nested_alias_example_statement : Prop :=
  in_flow_fragment NestedAliasExample.unit_expr = true
  ∧ static_constants (unit_codes NestedAliasExample.unit_expr)
      NestedAliasExample.unit_expr
    = true
  ∧ results_kinded [] [] NestedAliasExample.unit_expr = true
  ∧ well_formed_unit Examples.interface NestedAliasExample.unit_expr = true
  ∧ exits_one_value Examples.interface NestedAliasExample.unit_expr = true
  ∧ check_closed Examples.no_local NestedAliasExample.facts
      NestedAliasExample.tables
    = true
  ∧ check_field_used_exact Examples.no_local NestedAliasExample.facts
      NestedAliasExample.tables
    = true
  ∧ check_dominated NestedAliasExample.tables NestedAliasExample.dominated
    = true
  ∧ check_to_unbox Examples.no_local NestedAliasExample.facts
      NestedAliasExample.tables NestedAliasExample.cannot_change
      Examples.max_unbox_size false NestedAliasExample.to_unbox
    = true
  ∧ NestedAliasExample.to_unbox
    = [Node_variable 11%positive; Node_variable 10%positive]
  ∧ fst
      (unit_unboxed_fields NestedAliasExample.facts NestedAliasExample.tables
         NestedAliasExample.dominated NestedAliasExample.to_unbox
         NestedAliasExample.unit_expr)
    = [(Node_variable 11%positive,
        [(Field_block 0 Value,
          Unboxed [(Field_block 1 Value, Not_unboxed 15%positive)])]);
       (Node_variable 10%positive,
        [(Field_block 1 Value, Not_unboxed 16%positive)]);
       (Node_variable 12%positive,
        [(Field_block 0 Value,
          Unboxed [(Field_block 1 Value, Not_unboxed 17%positive)])]);
       (Node_variable 13%positive,
        [(Field_block 1 Value, Not_unboxed 18%positive)])]
  ∧ local_unboxing NestedAliasExample.facts NestedAliasExample.tables
      NestedAliasExample.dominated NestedAliasExample.to_unbox
      NestedAliasExample.unit_expr
    = true
  ∧ rebuild_unit_unbox NestedAliasExample.facts NestedAliasExample.tables
      NestedAliasExample.cannot_change
      (unit_code_deps Auto Examples.interface NestedAliasExample.unit_expr)
      NestedAliasExample.dominated NestedAliasExample.to_unbox true
      NestedAliasExample.unit_expr
    = NestedAliasExample.rebuilt
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own NestedAliasExample.unit_expr)
      (Exits 1%positive [Value_const (Tagged_immediate 8)] empty_heap)
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own NestedAliasExample.rebuilt)
      (Exits 1%positive [Value_const (Tagged_immediate 8)] empty_heap).

(* Blocks nested two deep, given to a handler whose parameter is unboxed
   alike:

     let k20 x30 = (let x31 = x30.(0) in let x32 = x31.(0) in
                    let x33 = x32.(1) in return x33) in
     let x10 = [7; 8] in
     let x11 = [x10] in
     let x12 = [x11] in
     k20 x12

   With the tables of the solver, which pass the checks, the rules unbox
   [x12], [x11] and [x10], hence [x30], [x31] and [x32], which they
   dominate. Field 0 of [x12] and of [x30] is unboxed in turn, and so is
   its field 0, whose field 1 is the variable 34, and 37. The rebuild
   binds the variable of field 1 of [x10], 36, to 8, that of [x11], 35,
   to 36, 34 to 35, gives 34 to [k20], whose parameter is 37, binds the
   variables of [x31] and [x32], 38 and 39, to 37 and 38, and [x33] to
   39. The unit and the rebuilt unit exit with 8. *)
Module DeepExample.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition unit_expr : expr :=
  Let_cont_non_recursive 20
    (Handler [30] (make_handler_metadata [Value])
       (Let 31 Value (load 0 (Var 30))
          (Let 32 Value (load 0 (Var 31))
             (Let 33 Value (load 1 (Var 32)) (return_to 1 [Var 33])))))
    (Let 10 Value (make_block [int 7; int 8])
       (Let 11 Value (make_block [Var 10])
          (Let 12 Value (make_block [Var 11]) (return_to 20 [Var 12])))).

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition tables : tables :=
  {| t_alias :=
       [(v 30, v 12); (m 1, v 33); (v 31, v 11); (v 32, v 10);
        (v 33, consts)];
     t_any_usage := [m 1; m 2; v 33; consts];
     t_any_source := [consts; v 33; m 1];
     t_has_usage :=
       [m 1; m 2; v 33; v 32; v 31; v 30; v 12; v 11; v 10; consts];
     t_has_source :=
       [consts; v 12; v 11; v 10; v 30; v 31; v 32; v 33; m 1];
     t_usages :=
       [(v 32, v 32); (v 31, v 31); (v 30, v 30); (v 12, v 30);
        (v 11, v 31); (v 10, v 32)];
     t_sources :=
       [(v 12, v 12); (v 11, v 11); (v 10, v 10); (v 30, v 12);
        (v 31, v 11); (v 32, v 10)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used :=
       [(v 12, Field_block 0 Value); (v 11, Field_block 0 Value);
        (v 10, Field_block 1 Value)] |}.

Definition cannot_change : list node := derive_cannot_change facts tables.

Definition dominated : list (node * node) := derive_dominated tables.

Definition to_unbox : list node :=
  derive_to_unbox no_local facts tables cannot_change max_unbox_size false.

Definition rebuilt : expr :=
  Let_cont_non_recursive 20
    (Handler [37] (make_handler_metadata [Value])
       (Let 38 Value (Simple (Var 37))
          (Let 39 Value (Simple (Var 38))
             (Let 33 Value (Simple (Var 39)) (return_to 1 [Var 33])))))
    (Let 36 Value (Simple (int 8))
       (Let 35 Value (Simple (Var 36))
          (Let 34 Value (Simple (Var 35)) (return_to 20 [Var 34])))).

End DeepExample.

Definition deep_example_statement : Prop :=
  in_flow_fragment DeepExample.unit_expr = true
  ∧ static_constants (unit_codes DeepExample.unit_expr) DeepExample.unit_expr
    = true
  ∧ results_kinded [] [] DeepExample.unit_expr = true
  ∧ well_formed_unit Examples.interface DeepExample.unit_expr = true
  ∧ exits_one_value Examples.interface DeepExample.unit_expr = true
  ∧ check_closed Examples.no_local DeepExample.facts DeepExample.tables
    = true
  ∧ check_field_used_exact Examples.no_local DeepExample.facts
      DeepExample.tables
    = true
  ∧ check_dominated DeepExample.tables DeepExample.dominated = true
  ∧ check_to_unbox Examples.no_local DeepExample.facts DeepExample.tables
      DeepExample.cannot_change Examples.max_unbox_size false
      DeepExample.to_unbox
    = true
  ∧ DeepExample.to_unbox
    = [Node_variable 12%positive; Node_variable 11%positive;
       Node_variable 10%positive]
  ∧ fst
      (unit_unboxed_fields DeepExample.facts DeepExample.tables
         DeepExample.dominated DeepExample.to_unbox DeepExample.unit_expr)
    = [(Node_variable 12%positive,
        [(Field_block 0 Value,
          Unboxed
            [(Field_block 0 Value,
              Unboxed [(Field_block 1 Value, Not_unboxed 34%positive)])])]);
       (Node_variable 11%positive,
        [(Field_block 0 Value,
          Unboxed [(Field_block 1 Value, Not_unboxed 35%positive)])]);
       (Node_variable 10%positive,
        [(Field_block 1 Value, Not_unboxed 36%positive)]);
       (Node_variable 30%positive,
        [(Field_block 0 Value,
          Unboxed
            [(Field_block 0 Value,
              Unboxed [(Field_block 1 Value, Not_unboxed 37%positive)])])]);
       (Node_variable 31%positive,
        [(Field_block 0 Value,
          Unboxed [(Field_block 1 Value, Not_unboxed 38%positive)])]);
       (Node_variable 32%positive,
        [(Field_block 1 Value, Not_unboxed 39%positive)])]
  ∧ local_unboxing DeepExample.facts DeepExample.tables DeepExample.dominated
      DeepExample.to_unbox DeepExample.unit_expr
    = true
  ∧ rebuild_unit_unbox DeepExample.facts DeepExample.tables
      DeepExample.cannot_change
      (unit_code_deps Auto Examples.interface DeepExample.unit_expr)
      DeepExample.dominated DeepExample.to_unbox true DeepExample.unit_expr
    = DeepExample.rebuilt
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own DeepExample.unit_expr)
      (Exits 1%positive [Value_const (Tagged_immediate 8)] empty_heap)
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own DeepExample.rebuilt)
      (Exits 1%positive [Value_const (Tagged_immediate 8)] empty_heap).

(* An alias of an unboxed block:

     let x10 = [7; 8] in
     let x11 = x10 in
     let x12 = x11.(1) in
     let x13 = x10.(0) in
     let x14 = [x12; x13] in
     return x14

   With the tables of the solver, which pass the checks, the rules unbox
   [x10], and [x11], which it dominates. The rebuild binds the variables
   of the fields of [x10], 15 and 16, to 7 and 8, and that of the field
   of [x11], 17, to 16; the unit and the rebuilt unit exit with the same
   block. *)
Module AliasExample.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition unit_expr : expr :=
  Let 10 Value (make_block [int 7; int 8])
    (Let 11 Value (Simple (Var 10))
       (Let 12 Value (load 1 (Var 11))
          (Let 13 Value (load 0 (Var 10))
             (Let 14 Value (make_block [Var 12; Var 13])
                (return_to 1 [Var 14]))))).

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition tables : tables :=
  {| t_alias := [(m 1, v 14); (v 11, v 10); (v 13, consts); (v 12, consts)];
     t_any_usage := [m 1; m 2; v 14; v 12; v 13; consts];
     t_any_source := [consts; v 13; v 12];
     t_has_usage := [m 1; m 2; v 14; v 12; v 13; consts; v 10; v 11];
     t_has_source := [consts; v 14; v 10; m 1; v 11; v 13; v 12];
     t_usages := [(v 10, v 10); (v 11, v 11); (v 10, v 11)];
     t_sources := [(v 14, v 14); (v 10, v 10); (m 1, v 14); (v 11, v 10)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used :=
       [(v 14, Field_block 0 Value); (v 14, Field_block 1 Value);
        (v 14, Field_is_int); (v 14, Field_get_tag);
        (v 10, Field_block 0 Value); (v 10, Field_block 1 Value)] |}.

Definition cannot_change : list node := derive_cannot_change facts tables.

Definition dominated : list (node * node) := derive_dominated tables.

Definition to_unbox : list node :=
  derive_to_unbox no_local facts tables cannot_change max_unbox_size false.

Definition rebuilt : expr :=
  Let 16 Value (Simple (int 8))
    (Let 15 Value (Simple (int 7))
       (Let 17 Value (Simple (Var 16))
          (Let 12 Value (Simple (Var 17))
             (Let 13 Value (Simple (Var 15))
                (Let 14 Value (make_block [Var 12; Var 13])
                   (return_to 1 [Var 14])))))).

Definition result : value :=
  Value_block 0
    [Value_const (Tagged_immediate 8); Value_const (Tagged_immediate 7)].

End AliasExample.

Definition alias_example_statement : Prop :=
  in_flow_fragment AliasExample.unit_expr = true
  ∧ static_constants (unit_codes AliasExample.unit_expr) AliasExample.unit_expr
    = true
  ∧ results_kinded [] [] AliasExample.unit_expr = true
  ∧ well_formed_unit Examples.interface AliasExample.unit_expr = true
  ∧ exits_one_value Examples.interface AliasExample.unit_expr = true
  ∧ check_closed Examples.no_local AliasExample.facts AliasExample.tables
    = true
  ∧ check_field_used_exact Examples.no_local AliasExample.facts
      AliasExample.tables
    = true
  ∧ check_dominated AliasExample.tables AliasExample.dominated = true
  ∧ check_to_unbox Examples.no_local AliasExample.facts AliasExample.tables
      AliasExample.cannot_change Examples.max_unbox_size false
      AliasExample.to_unbox
    = true
  ∧ AliasExample.to_unbox = [Node_variable 10%positive]
  ∧ local_unboxing AliasExample.facts AliasExample.tables
      AliasExample.dominated AliasExample.to_unbox AliasExample.unit_expr
    = true
  ∧ rebuild_unit_unbox AliasExample.facts AliasExample.tables
      AliasExample.cannot_change
      (unit_code_deps Auto Examples.interface AliasExample.unit_expr)
      AliasExample.dominated AliasExample.to_unbox true AliasExample.unit_expr
    = AliasExample.rebuilt
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own AliasExample.unit_expr)
      (Exits 1%positive [AliasExample.result] empty_heap)
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own AliasExample.rebuilt)
      (Exits 1%positive [AliasExample.result] empty_heap).

(* The fragment asks that an unboxed variable be given to no parameter
   that the plain rebuild keeps and that is not unboxed. The checks do not
   give it: a solution that is closed and not least may give such a
   parameter a use that it has not. In

     let x10 = [7] in
     let x13 = [9] in
     let k20 x30 x31 = return x31 in
     let x11 = x10.(0) in
     switch 0 with 0 -> k20 x10 x11 | 1 -> k20 x13 x11

   [x30] has no use: with the solution of the solver ([least]) the
   rebuilds delete it, and the fragment holds. The tables [tables] add
   to that solution a use of [x30], and so of [x13]: they pass the
   checks, the rules unbox [x10] and [x13], [x30], which they may come
   from, is not unboxed but kept, and the rebuild gives it poison; the
   fragment does not hold. That is no miscompilation: [x30] has no use,
   and the unit and the rebuilt unit exit alike. A parameter that is
   given an unboxed variable and that has a real use does not occur: a
   read of it with a use makes the allocation
   [cannot_change_representation0] ([check_crr0_other_source],
   [check_crr0_any_source_usage]), and a use of any other kind makes it
   of [any_usage], as is the variable given, which is then in
   [cannot_unbox0]. *)
Module SpuriousKeepExample.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition goto_args (k : continuation) (args : list simple) : apply_cont :=
  {| ac_continuation := k; ac_args := args; ac_trap_action := None |}.

Definition unit_expr : expr :=
  Let 10 Value (make_block [int 7])
    (Let 13 Value (make_block [int 9])
       (Let_cont_non_recursive 20
          (Handler [30; 31] (make_handler_metadata [Value; Value])
             (return_to 1 [Var 31]))
          (Let 11 Value (load 0 (Var 10))
             (Switch (Const (Naked_immediate 0))
                [(0%Z, goto_args 20 [Var 10; Var 11]);
                 (1%Z, goto_args 20 [Var 13; Var 11])])))).

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition least : tables :=
  {| t_alias :=
       [(v 30, v 13); (v 31, v 11); (v 30, v 10); (m 1, v 31);
        (v 11, consts)];
     t_any_usage := [m 1; consts; m 2; v 31; v 11];
     t_any_source := [consts; v 11; v 31; m 1];
     t_has_usage := [m 1; consts; m 2; v 31; v 11; v 10];
     t_has_source := [consts; v 13; v 10; v 30; v 11; v 31; m 1];
     t_usages := [(v 10, v 10)];
     t_sources := [(v 13, v 13); (v 10, v 10); (v 30, v 13); (v 30, v 10)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used := [(v 10, Field_block 0 Value)] |}.

Definition tables : tables :=
  {| t_alias := t_alias least;
     t_any_usage := t_any_usage least;
     t_any_source := t_any_source least;
     t_has_usage := t_has_usage least ++ [v 30; v 13];
     t_has_source := t_has_source least;
     t_usages := t_usages least;
     t_sources := t_sources least;
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used := t_field_used least |}.

Definition to_unbox (T : FlambdaReaperGraphSpec.tables) : list node :=
  derive_to_unbox no_local facts T (derive_cannot_change facts T)
    max_unbox_size false.

Definition rebuilt : expr :=
  Let 32 Value (Simple (int 7))
    (Let_cont_non_recursive 20
       (Handler [30; 31] (make_handler_metadata [Value; Value])
          (return_to 1 [Var 31]))
       (Let 11 Value (Simple (Var 32))
          (Switch (Const (Naked_immediate 0))
             [(0%Z, goto_args 20 [Const (Poison Value); Var 11]);
              (1%Z, goto_args 20 [Const (Poison Value); Var 11])]))).

End SpuriousKeepExample.

Definition spurious_keep_example_statement : Prop :=
  let T := SpuriousKeepExample.tables in
  let L := SpuriousKeepExample.least in
  let F := SpuriousKeepExample.facts in
  let e := SpuriousKeepExample.unit_expr in
  let cannot_change := derive_cannot_change F T in
  let dominated := derive_dominated T in
  let to_unbox := SpuriousKeepExample.to_unbox T in
  in_flow_fragment e = true
  ∧ check_closed Examples.no_local F L = true
  ∧ local_unboxing F L (derive_dominated L) (SpuriousKeepExample.to_unbox L)
      e
    = true
  ∧ check_closed Examples.no_local F T = true
  ∧ check_field_used_exact Examples.no_local F T = true
  ∧ check_dominated T dominated = true
  ∧ check_to_unbox Examples.no_local F T cannot_change Examples.max_unbox_size
      false to_unbox
    = true
  ∧ to_unbox = [Node_variable 10%positive; Node_variable 13%positive]
  ∧ local_unboxing F T dominated to_unbox e = false
  ∧ rebuild_unit_unbox F T cannot_change
      (unit_code_deps Auto Examples.interface e) dominated to_unbox true e
    = SpuriousKeepExample.rebuilt
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own e)
      (Exits 1%positive [Value_const (Tagged_immediate 7)] empty_heap)
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own SpuriousKeepExample.rebuilt)
      (Exits 1%positive [Value_const (Tagged_immediate 7)] empty_heap).

(* Of the four units of blocks of FlambdaReaperUnboxing.v, with the tables
   there, the fragment holds of the first two, whose block is only read,
   in the scope where it is bound or in a handler it is given to, and of
   the fourth, whose block is held in a field of another block that is
   unboxed, and not of the third: the block is returned by code. *)
Definition fragment_examples_statement : Prop :=
  let v (x : positive) : node := Node_variable x in
  local_unboxing Examples.read_block_facts Examples.read_block_tables
    [(v 10%positive, v 10%positive)] [v 10%positive] Examples.read_block
  = true
  ∧ local_unboxing Examples.block_to_continuation_facts
      Examples.block_to_continuation_tables
      [(v 10%positive, v 10%positive); (v 10%positive, v 30%positive)]
      [v 10%positive] Examples.block_to_continuation
    = true
  ∧ local_unboxing Examples.block_from_code_facts
      Examples.block_from_code_tables Examples.block_from_code_dominated
      [v 42%positive] Examples.block_from_code
    = false
  ∧ local_unboxing Examples.nested_block_facts Examples.nested_block_tables
      [(v 11%positive, v 11%positive); (v 10%positive, v 10%positive);
       (v 10%positive, v 12%positive)]
      [v 11%positive; v 10%positive] Examples.nested_block
    = true.

(* ======================================================================== *)
(* The reads find their fields                                              *)
(* ======================================================================== *)

(* [check_unboxed_reads] holds of the tables of the examples above, as
   the solver of the test tool or the rules give them, with their tables
   of [dominated_by_allocation_point] and [to_unbox]. It is not vacuous
   there: in [LocalExample], for one, the rebuild keeps the reads of
   [x10] and [x11], unboxed, into [x31], [x12] and [x13]. *)
Definition unboxed_reads_examples_statement : Prop :=
  let v (x : positive) : node := Node_variable x in
  let to_unbox (F : facts) (T : tables) : list node :=
    derive_to_unbox Examples.no_local F T (derive_cannot_change F T)
      Examples.max_unbox_size false
  in
  check_unboxed_reads LocalExample.facts LocalExample.tables
    LocalExample.dominated LocalExample.to_unbox
  = true
  ∧ check_unboxed_reads ContinuationExample.facts ContinuationExample.tables
      ContinuationExample.dominated ContinuationExample.to_unbox
    = true
  ∧ check_unboxed_reads LoopExample.facts LoopExample.tables
      LoopExample.dominated LoopExample.to_unbox
    = true
  ∧ check_unboxed_reads Examples.nested_block_facts
      Examples.nested_block_tables
      (derive_dominated Examples.nested_block_tables)
      (to_unbox Examples.nested_block_facts Examples.nested_block_tables)
    = true
  ∧ check_unboxed_reads NestedParamExample.facts NestedParamExample.tables
      NestedParamExample.dominated NestedParamExample.to_unbox
    = true
  ∧ check_unboxed_reads NestedAliasExample.facts NestedAliasExample.tables
      NestedAliasExample.dominated NestedAliasExample.to_unbox
    = true
  ∧ check_unboxed_reads DeepExample.facts DeepExample.tables
      DeepExample.dominated DeepExample.to_unbox
    = true
  ∧ check_unboxed_reads AliasExample.facts AliasExample.tables
      AliasExample.dominated AliasExample.to_unbox
    = true
  ∧ check_unboxed_reads SpuriousKeepExample.facts SpuriousKeepExample.least
      (derive_dominated SpuriousKeepExample.least)
      (SpuriousKeepExample.to_unbox SpuriousKeepExample.least)
    = true
  ∧ check_unboxed_reads SpuriousKeepExample.facts SpuriousKeepExample.tables
      (derive_dominated SpuriousKeepExample.tables)
      (SpuriousKeepExample.to_unbox SpuriousKeepExample.tables)
    = true
  ∧ check_unboxed_reads Examples.read_block_facts Examples.read_block_tables
      [(v 10%positive, v 10%positive)] [v 10%positive]
    = true
  ∧ check_unboxed_reads Examples.block_to_continuation_facts
      Examples.block_to_continuation_tables
      [(v 10%positive, v 10%positive); (v 10%positive, v 30%positive)]
      [v 10%positive]
    = true.

(* A table closed under the rules, and not least, that fails
   [check_unboxed_reads] where the least solution passes it: a read, in a
   handler that is never applied, of a block whose field holds the
   parameter, which has no source:

     let rec k20 x30 = (let x31 = [x30] in let x32 = x31.(0) in
                        return x32) in
     return 7

   The least solution ([least], whose tables less any one tuple are not
   closed) gives [x30], hence [x32], no source: the [Let] of [x32] is
   [Invalid] for both rebuilds. The table [tables] is [least] with a
   source for [x32]; it is closed and passes the other checks, the rules
   unbox [x31] in both, and the unit is of the fragment for both. With
   [tables], [x32] has a source and a use, the field 0 of [x31] none, so
   that it is not unboxed: the plain rebuild keeps the read, and the
   rebuild with unboxing makes it [Invalid] (the OCaml stops with a fatal
   error in [rewrite_field_access]). The handler never runs, so that the
   two rebuilds behave alike. *)
Module DeadReadExample.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition unit_expr : expr :=
  Let_cont_recursive
    [(20%positive,
      Handler [30] (make_handler_metadata [Value])
        (Let 31 Value (make_block [Var 30])
           (Let 32 Value (load 0 (Var 31))
              (return_to 1 [Var 32]))))]
    (return_to 1 [int 7]).

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition least : tables :=
  {| t_alias := [(m 1, consts); (m 1, v 32); (v 32, v 30)];
     t_any_usage := [m 1; m 2; consts; v 32; v 30];
     t_any_source := [consts; m 1];
     t_has_usage := [m 1; m 2; consts; v 32; v 30; v 31];
     t_has_source := [consts; m 1; v 31];
     t_usages := [(v 31, v 31)];
     t_sources := [(v 31, v 31)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used := [(v 31, Field_block 0 Value)] |}.

Definition tables : tables :=
  {| t_alias := t_alias least;
     t_any_usage := t_any_usage least;
     t_any_source := t_any_source least;
     t_has_usage := t_has_usage least;
     t_has_source := v 32 :: t_has_source least;
     t_usages := t_usages least;
     t_sources := t_sources least;
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used := t_field_used least |}.

Definition to_unbox (T : FlambdaReaperGraphSpec.tables) : list node :=
  derive_to_unbox no_local facts T (derive_cannot_change facts T)
    max_unbox_size false.

Definition handler_with (body : expr) : expr :=
  Let_cont_recursive
    [(20%positive, Handler [30] (make_handler_metadata [Value]) body)]
    (return_to 1 [int 7]).

End DeadReadExample.

Definition unboxed_reads_example_statement : Prop :=
  let F := DeadReadExample.facts in
  let L := DeadReadExample.least in
  let T := DeadReadExample.tables in
  let e := DeadReadExample.unit_expr in
  let to_unbox := DeadReadExample.to_unbox in
  in_flow_fragment e = true
  ∧ check_closed Examples.no_local F L = true
  ∧ forallb (fun T' => negb (check_closed Examples.no_local F T'))
      (FlambdaReaperSolutionSpec.one_fewer_tables L)
    = true
  ∧ check_field_used_exact Examples.no_local F L = true
  ∧ to_unbox L = [Node_variable 31%positive]
  ∧ local_unboxing F L (derive_dominated L) (to_unbox L) e = true
  ∧ check_unboxed_reads F L (derive_dominated L) (to_unbox L) = true
  ∧ check_closed Examples.no_local F T = true
  ∧ check_field_used_exact Examples.no_local F T = true
  ∧ check_dominated T (derive_dominated T) = true
  ∧ check_to_unbox Examples.no_local F T (derive_cannot_change F T)
      Examples.max_unbox_size false (to_unbox T)
    = true
  ∧ to_unbox T = [Node_variable 31%positive]
  ∧ local_unboxing F T (derive_dominated T) (to_unbox T) e = true
  ∧ check_unboxed_reads F T (derive_dominated T) (to_unbox T) = false
  ∧ rebuild_unit_from_tables F T true e = e
  ∧ rebuild_unit_unbox F T (derive_cannot_change F T)
      (unit_code_deps Auto Examples.interface e) (derive_dominated T)
      (to_unbox T) true e
    = DeadReadExample.handler_with Invalid.

(* ======================================================================== *)
(* An example with a call                                                   *)
(* ======================================================================== *)

(* Code whose parameter and result are unboxed, called directly, the call
   returning to a handler:

     code 50 x40 = (let x43 = x40.(1) in let x42 = [x43; 9] in
                    return x42)
     let x60 = closure of 50 in
     let k20 x30 = (let x31 = x30.(0) in return x31) in
     let x10 = [7; 8] in
     x60 x10 -> k20

   The tables are those of the solver of the test tool. They pass the
   checks of the plain statement, those of the change of calling
   conventions ([check_cc_closed] with no code that cannot change,
   [check_apply_widgets], [check_witness_code_ids]), [check_dominated],
   [check_to_unbox], and [check_result_receivers], the check of the fix
   of FlambdaReaperUnboxingSpec.kept_result_unboxed; the unit is of
   [cc_fragment], and not of [local_unboxing], which has no call. The
   rules unbox [x10], which dominates the parameter [x40], and [x42],
   which dominates the result of the code dep and [x30]: the code 50
   changes its calling convention, its parameter unboxed with its field
   1 and its result with its field 0. The rebuild binds the variable 61
   of field 1 of [x10] to 8 and gives it to the call, a product of one
   value; the code has the variable 66 of field 1 of [x40] for
   parameter, which it reads in place of the load, binds the variable
   62 of field 0 of [x42] to [x43] and returns it; [k20] has the
   variable 64 of field 0 of [x30] for parameter, which it reads in
   place of the load. The decisions of the call and of the callee for
   the result agree, so there is no wrapper. The unit, the unit rebuilt
   with the changes of calling convention alone
   ([rebuild_unit_cc_from_tables], which keeps both blocks, poison in
   the fields that are not read, and the code's parameter and result),
   and the rebuilt unit exit with 8. *)
Module CallExample.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition callee : code :=
  Code [40] 41 3 4 (make_code_metadata [Singleton Value] [Singleton Value])
    (Let 43 Value (load 1 (Var 40))
       (Let 42 Value (make_block [Var 43; int 9]) (return_to 3 [Var 42]))).

Definition unit_expr : expr :=
  Let_static [Bind_code 50 callee]
    (Let 60 Value (Closure_of_set 70 [(70, 50)] [])
       (Let_cont_non_recursive 20
          (Handler [30] (make_handler_metadata [Value])
             (Let 31 Value (load 0 (Var 30)) (return_to 1 [Var 31])))
          (Let 10 Value (make_block [int 7; int 8])
             (Apply (call_50 [Var 10] 20))))).

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition tables : tables :=
  {| t_alias :=
       [(m 13, m 7); (m 5, m 12); (m 9, m 11); (m 1, v 31); (m 3, v 42);
        (v 41, m 5); (v 40, m 6); (m 11, m 7); (v 30, m 3); (m 2, m 4);
        (m 10, c 50); (m 6, v 10); (v 43, consts); (v 31, v 43); (m 5, v 60)];
     t_any_usage := [m 1; m 10; m 2; v 31; m 4; c 50; v 43; consts];
     t_any_source := [c 50; consts; m 10; v 43; v 31; m 1];
     t_has_usage :=
       [m 1; m 10; m 2; v 31; m 9; v 30; m 11; m 7; m 3; m 4; c 50; v 43; v 42;
        consts; v 60; v 40; m 6; v 10];
     t_has_source :=
       [c 50; consts; v 10; v 60; v 42; m 8; m 7; m 13; m 3; m 11; v 30; m 10;
        m 6; v 43; v 31; m 9; m 1; v 40; m 5; v 41];
     t_usages :=
       [(v 60, v 60); (m 9, m 9); (v 30, v 30); (v 40, v 40); (m 11, m 9);
        (m 6, v 40); (m 7, m 9); (m 3, v 30); (v 10, v 40); (v 42, v 30)];
     t_sources :=
       [(v 60, v 60); (v 10, v 10); (v 42, v 42); (m 8, m 8); (m 7, m 7);
        (m 13, m 7); (m 3, v 42); (m 11, m 7); (m 9, m 7); (v 30, v 42);
        (m 6, v 10); (v 40, v 10); (m 5, v 60); (v 41, v 60)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used :=
       [(v 60, known); (v 10, Field_block 1 Value);
        (v 42, Field_block 0 Value); (m 7, result 0);
        (m 7, Field_return_of_call Exn);
        (m 7, Field_code_id_of_call_witness)] |}.

Definition cannot_change : list node := derive_cannot_change facts tables.

Definition dominated : list (node * node) := derive_dominated tables.

Definition to_unbox : list node :=
  derive_to_unbox no_local facts tables cannot_change max_unbox_size false.

Definition code_deps : finite_map code_dep :=
  unit_code_deps Auto interface unit_expr.

Definition callee_rebuilt : code :=
  Code [66] 41 3 4
    {| cm_params_arity := [Unboxed_product [Singleton Value]];
       cm_result_arity := [Unboxed_product [Singleton Value]];
       cm_is_tupled := false;
       cm_is_stub := false |}
    (Let 43 Value (Simple (Var 66))
       (Let 62 Value (Simple (Var 43)) (return_to 3 [Var 62]))).

Definition rebuilt : expr :=
  Let_static [Bind_code 50 callee_rebuilt]
    (Let 60 Value (Closure_of_set 70 [(70, 50)] [])
       (Let_cont_non_recursive 20
          (Handler [64] (make_handler_metadata [Value])
             (Let 31 Value (Simple (Var 64)) (return_to 1 [Var 31])))
          (Let 61 Value (Simple (int 8))
             (Apply
                {| ap_callee := Var 60; ap_call_kind := Direct 50;
                   ap_args := [Var 61]; ap_continuation := 20;
                   ap_exn_continuation := 2; ap_exn_extra_args := [];
                   ap_args_arity := [Unboxed_product [Singleton Value]];
                   ap_return_arity := [Unboxed_product [Singleton Value]]
                |})))).

End CallExample.

Definition unboxed_call_example_statement : Prop :=
  is_local_slots_only Examples.no_local
  ∧ in_flow_fragment CallExample.unit_expr = true
  ∧ static_constants (unit_codes CallExample.unit_expr)
      CallExample.unit_expr
    = true
  ∧ results_kinded [] [] CallExample.unit_expr = true
  ∧ well_formed_unit Examples.interface CallExample.unit_expr = true
  ∧ exits_one_value Examples.interface CallExample.unit_expr = true
  ∧ check_closed Examples.no_local CallExample.facts CallExample.tables
    = true
  ∧ check_field_used_exact Examples.no_local CallExample.facts
      CallExample.tables
    = true
  ∧ CallExample.cannot_change = []
  ∧ check_cc_closed CallExample.facts CallExample.tables
      CallExample.cannot_change
    = true
  ∧ FlambdaReaperCallingConventionsSpec.check_apply_widgets
      CallExample.facts CallExample.tables
    = true
  ∧ FlambdaReaperCallingConventionsSpec.check_witness_code_ids
      CallExample.facts
    = true
  ∧ FlambdaReaperCallingConventionsSpec.cc_fragment CallExample.unit_expr
    = true
  ∧ check_dominated CallExample.tables CallExample.dominated = true
  ∧ check_to_unbox Examples.no_local CallExample.facts CallExample.tables
      CallExample.cannot_change Examples.max_unbox_size false
      CallExample.to_unbox
    = true
  ∧ CallExample.to_unbox
    = [Node_variable 42%positive; Node_variable 10%positive]
  ∧ FlambdaReaperUnboxingSpec.check_result_receivers CallExample.facts
      CallExample.tables CallExample.dominated CallExample.to_unbox
    = true
  ∧ local_unboxing CallExample.facts CallExample.tables
      CallExample.dominated CallExample.to_unbox CallExample.unit_expr
    = false
  ∧ unit_calling_convention_changes_u CallExample.facts CallExample.tables
      CallExample.cannot_change CallExample.code_deps CallExample.dominated
      CallExample.to_unbox CallExample.unit_expr
    = [(50%positive,
        Changing_u [Unbox [(Field_block 1 Value, Not_unboxed 65%positive)]]
          [Unbox [(Field_block 0 Value, Not_unboxed 63%positive)]])]
  ∧ rebuild_unit_unbox CallExample.facts CallExample.tables
      CallExample.cannot_change CallExample.code_deps CallExample.dominated
      CallExample.to_unbox true CallExample.unit_expr
    = CallExample.rebuilt
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own CallExample.unit_expr)
      (Exits 1%positive [Value_const (Tagged_immediate 8)] empty_heap)
  ∧ behaves
      (FlambdaReaperTupledSpec.on_its_own
         (rebuild_unit_cc_from_tables CallExample.facts CallExample.tables
            CallExample.cannot_change CallExample.code_deps true
            CallExample.unit_expr))
      (Exits 1%positive [Value_const (Tagged_immediate 8)] empty_heap)
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own CallExample.rebuilt)
      (Exits 1%positive [Value_const (Tagged_immediate 8)] empty_heap).

(* The same code called twice, the first call returning to a handler
   that has no use of the result, and calling again, returning to the
   handler that reads it:

     let x10 = [7; 8] in
     let k20 x30 = (let x31 = x30.(0) in return x31) in
     let k21 x32 = x60 x10 -> k20 in
     x60 x10 -> k21

   The tables are those of the solver; they pass the same checks, and
   the code changes as above. The plain rebuild deletes the parameter
   [x32] of [k21], as it has no use; the callee unboxes the result: the
   first call returns to a wrapper, the continuation 67, whose
   parameter, the field of the result, is dropped ([make_apply_wrapper]:
   a result that the continuation deletes is dropped, whatever the
   callee does with it). The second call needs no wrapper. The unit and
   the rebuilt unit exit with 8. *)
Module WrapperExample.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition unit_expr : expr :=
  Let_static [Bind_code 50 CallExample.callee]
    (Let 60 Value (Closure_of_set 70 [(70, 50)] [])
       (Let 10 Value (make_block [int 7; int 8])
          (Let_cont_non_recursive 20
             (Handler [30] (make_handler_metadata [Value])
                (Let 31 Value (load 0 (Var 30)) (return_to 1 [Var 31])))
             (Let_cont_non_recursive 21
                (Handler [32] (make_handler_metadata [Value])
                   (Apply (call_50 [Var 10] 20)))
                (Apply (call_50 [Var 10] 21)))))).

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition tables : tables :=
  {| t_alias :=
       [(m 18, m 7); (m 5, m 17); (m 13, m 7); (m 5, m 12); (m 14, m 16);
        (m 9, m 11); (m 1, v 31); (m 3, v 42); (v 41, m 5); (v 40, m 6);
        (m 11, m 7); (m 16, m 7); (v 32, m 3); (v 30, m 3); (m 2, m 4);
        (m 15, c 50); (m 10, c 50); (m 6, v 10); (v 43, consts); (v 31, v 43);
        (m 5, v 60)];
     t_any_usage := [m 1; m 15; m 10; m 2; v 31; m 4; c 50; v 43; consts];
     t_any_source := [c 50; consts; m 15; m 10; v 43; v 31; m 1];
     t_has_usage :=
       [m 1; m 15; m 10; m 2; v 31; m 14; m 9; v 30; m 16; m 11; m 7; m 3; m 4;
        c 50; v 43; v 42; consts; v 60; v 40; m 6; v 10];
     t_has_source :=
       [c 50; consts; v 10; v 60; v 42; m 8; m 7; m 18; m 13; m 3; m 11; m 16;
        v 32; v 30; m 15; m 10; m 6; v 43; v 31; m 14; m 9; m 1; v 40; m 5;
        v 41];
     t_usages :=
       [(v 60, v 60); (m 14, m 14); (m 9, m 9); (v 30, v 30); (v 40, v 40);
        (m 16, m 14); (m 11, m 9); (m 6, v 40); (m 7, m 9); (m 7, m 14);
        (m 3, v 30); (v 10, v 40); (v 42, v 30)];
     t_sources :=
       [(v 60, v 60); (v 10, v 10); (v 42, v 42); (m 8, m 8); (m 7, m 7);
        (m 18, m 7); (m 13, m 7); (m 3, v 42); (m 11, m 7); (m 16, m 7);
        (m 14, m 7); (m 9, m 7); (v 32, v 42); (v 30, v 42); (m 6, v 10);
        (v 40, v 10); (m 5, v 60); (v 41, v 60)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used :=
       [(v 60, known); (v 10, Field_block 1 Value);
        (v 42, Field_block 0 Value); (m 7, result 0);
        (m 7, Field_return_of_call Exn);
        (m 7, Field_code_id_of_call_witness)] |}.

Definition cannot_change : list node := derive_cannot_change facts tables.

Definition dominated : list (node * node) := derive_dominated tables.

Definition to_unbox : list node :=
  derive_to_unbox no_local facts tables cannot_change max_unbox_size false.

Definition code_deps : finite_map code_dep :=
  unit_code_deps Auto interface unit_expr.

(* The call of 50 with the variable of field 1 of [x10], returning to
   [k]. *)
Definition call_unboxed (k : continuation) : apply_expr :=
  {| ap_callee := Var 60; ap_call_kind := Direct 50; ap_args := [Var 61];
     ap_continuation := k; ap_exn_continuation := 2;
     ap_exn_extra_args := [];
     ap_args_arity := [Unboxed_product [Singleton Value]];
     ap_return_arity := [Unboxed_product [Singleton Value]] |}.

Definition rebuilt : expr :=
  Let_static [Bind_code 50 CallExample.callee_rebuilt]
    (Let 60 Value (Closure_of_set 70 [(70, 50)] [])
       (Let 61 Value (Simple (int 8))
          (Let_cont_non_recursive 20
             (Handler [64] (make_handler_metadata [Value])
                (Let 31 Value (Simple (Var 64)) (return_to 1 [Var 31])))
             (Let_cont_non_recursive 21
                (Handler [] (make_handler_metadata [])
                   (Apply (call_unboxed 20)))
                (Let_cont_non_recursive 67
                   (Handler [67] (make_handler_metadata [Value])
                      (return_to 21 []))
                   (Apply (call_unboxed 67))))))).

End WrapperExample.

Definition unboxed_wrapper_example_statement : Prop :=
  is_local_slots_only Examples.no_local
  ∧ in_flow_fragment WrapperExample.unit_expr = true
  ∧ static_constants (unit_codes WrapperExample.unit_expr)
      WrapperExample.unit_expr
    = true
  ∧ results_kinded [] [] WrapperExample.unit_expr = true
  ∧ well_formed_unit Examples.interface WrapperExample.unit_expr = true
  ∧ exits_one_value Examples.interface WrapperExample.unit_expr = true
  ∧ check_closed Examples.no_local WrapperExample.facts
      WrapperExample.tables
    = true
  ∧ check_field_used_exact Examples.no_local WrapperExample.facts
      WrapperExample.tables
    = true
  ∧ WrapperExample.cannot_change = []
  ∧ check_cc_closed WrapperExample.facts WrapperExample.tables
      WrapperExample.cannot_change
    = true
  ∧ FlambdaReaperCallingConventionsSpec.check_apply_widgets
      WrapperExample.facts WrapperExample.tables
    = true
  ∧ FlambdaReaperCallingConventionsSpec.check_witness_code_ids
      WrapperExample.facts
    = true
  ∧ FlambdaReaperCallingConventionsSpec.cc_fragment WrapperExample.unit_expr
    = true
  ∧ check_dominated WrapperExample.tables WrapperExample.dominated = true
  ∧ check_to_unbox Examples.no_local WrapperExample.facts
      WrapperExample.tables WrapperExample.cannot_change
      Examples.max_unbox_size false WrapperExample.to_unbox
    = true
  ∧ WrapperExample.to_unbox
    = [Node_variable 42%positive; Node_variable 10%positive]
  ∧ FlambdaReaperUnboxingSpec.check_result_receivers WrapperExample.facts
      WrapperExample.tables WrapperExample.dominated WrapperExample.to_unbox
    = true
  ∧ unit_calling_convention_changes_u WrapperExample.facts
      WrapperExample.tables WrapperExample.cannot_change
      WrapperExample.code_deps WrapperExample.dominated
      WrapperExample.to_unbox WrapperExample.unit_expr
    = [(50%positive,
        Changing_u [Unbox [(Field_block 1 Value, Not_unboxed 65%positive)]]
          [Unbox [(Field_block 0 Value, Not_unboxed 63%positive)]])]
  ∧ rebuild_unit_unbox WrapperExample.facts WrapperExample.tables
      WrapperExample.cannot_change WrapperExample.code_deps
      WrapperExample.dominated WrapperExample.to_unbox true
      WrapperExample.unit_expr
    = WrapperExample.rebuilt
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own WrapperExample.unit_expr)
      (Exits 1%positive [Value_const (Tagged_immediate 8)] empty_heap)
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own WrapperExample.rebuilt)
      (Exits 1%positive [Value_const (Tagged_immediate 8)] empty_heap).

(* ======================================================================== *)
(* With nothing unboxed                                                     *)
(* ======================================================================== *)

(* A first piece of the calls (M4), by the route that compares the rebuild
   with unboxing with the rebuild of FlambdaReaperCallingConventions.v
   ([rebuild_unit_cc_from_tables], the same table of
   [cannot_change_calling_convention]): where the unboxing unboxes nothing
   ([unit_unboxed_fields] gives no name), the two make the same term
   ([rebuild_unbox_without_unboxing_statement]), calls, code and wrappers
   included, so that the rebuild with unboxing refines the unit wherever
   the change of calling conventions is proved to
   ([rebuild_unbox_without_unboxing_observably_statement], with the
   premises of
   FlambdaReaperCallingConventionsSpec.rebuild_cc_refines_observably_statement).

   It holds of units whose lengths and kinds agree as those of the
   compiler do ([arities_match]): each handler has as many kinds as
   parameters; each code as many parameters as its arity has, unarized,
   and no more than its code dep in [code_deps]; a call whose continuation
   is a handler of the term, or the return continuation of its code
   ([conts]: their kinds, or those of the results of the code), has
   results whose kinds begin with those ([is_prefix]); and a call known to
   call code of the unit ([call_results]) expects results of the kinds
   that the code returns. Where these fail the two rebuilds differ, in
   ways that the machine does not run past (the call or the continuation
   goes wrong): FlambdaReaperRebuild.make_apply_wrapper (which the rebuild
   of calling conventions uses for a callee that does not change) tests
   every parameter of the continuation, make_apply_wrapper_cc and
   make_apply_wrapper_u only as many as the call has results; the
   parameters of the wrapper of make_apply_wrapper_cc have the kinds of
   the decisions of the callee, those of make_apply_wrapper_u the kinds of
   the return arity of the call; and the poison that wrapper_args_u gives
   a result that the callee deletes has the kind of the continuation's
   parameter, that of wrapper_args the kind of the call's result. *)

(* [l] begins [l']. *)
Fixpoint is_prefix (l l' : list flambda_kind) : bool :=
  match l, l' with
  | [], _ => true
  | a :: l, b :: l' => kind_eqb a b && is_prefix l l'
  | _ :: _, [] => false
  end.

(* The kinds of the parameters of a handler. *)
Definition handler_kinds (h : continuation_handler) : list flambda_kind :=
  match h with Handler _ metadata _ => hm_params_kinds metadata end.

(* See above; [results] gives, for a call, the kinds of the results of
   the code it is known to call ([call_results]). *)
Fixpoint arities_match (results : apply_expr -> option (list flambda_kind))
    (code_deps : finite_map code_dep) (conts : finite_map (list flambda_kind))
    (e : expr) : bool :=
  match e with
  | Let _ _ _ body => arities_match results code_deps conts body
  | Let_static bindings body =>
    forallb
      (fun binding =>
         match binding with
         | Bind_code id (Code params _ ret _ metadata cbody) =>
           Nat.eqb (length params)
             (length (unarize (cm_params_arity metadata)))
           && match lookup id code_deps with
              | Some cd => Nat.leb (length params) (length (cd_params cd))
              | None => true
              end
           && arities_match results code_deps
                [(ret, unarize (cm_result_arity metadata))] cbody
         | Bind_deleted_code _ | Bind_symbol _ _ => true
         end)
      bindings
    && arities_match results code_deps conts body
  | Let_cont_non_recursive k h body =>
    handler_arities_match results code_deps conts h
    && arities_match results code_deps ((k, handler_kinds h) :: conts) body
  | Let_cont_recursive handlers body =>
    let conts :=
      map (fun '(k, h) => (k, handler_kinds h)) handlers ++ conts
    in
    forallb
      (fun '(_, h) => handler_arities_match results code_deps conts h)
      handlers
    && arities_match results code_deps conts body
  | Apply ap =>
    match lookup (ap_continuation ap) conts with
    | Some ks => is_prefix ks (unarize (ap_return_arity ap))
    | None => true
    end
    && match results ap with
       | Some ks => kinds_eqb ks (unarize (ap_return_arity ap))
       | None => true
       end
  | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_arities_match
    (results : apply_expr -> option (list flambda_kind))
    (code_deps : finite_map code_dep) (conts : finite_map (list flambda_kind))
    (h : continuation_handler) : bool :=
  match h with
  | Handler params metadata handler =>
    Nat.eqb (length params) (length (hm_params_kinds metadata))
    && arities_match results code_deps conts handler
  end.

(* The kinds of the results of the code that a call is known to call
   ([code_id_actually_called]), among the code of the unit [e]. *)
Definition call_results (F : facts) (T : tables) (e : expr)
    (ap : apply_expr) : option (list flambda_kind) :=
  match code_id_actually_called F (solution_of_tables T) ap with
  | Some id =>
    option_map (fun md => unarize (cm_result_arity md))
      (lookup id (bound_code_metadata e))
  | None => None
  end.

Definition rebuild_unbox_without_unboxing_statement : Prop :=
  ∀ (F : facts) (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool) (e : expr),
    fst (unit_unboxed_fields F T dominated to_unbox e) = [] ->
    arities_match (call_results F T e) code_deps [] e = true ->
    rebuild_unit_unbox F T cannot_change code_deps dominated to_unbox
      preserve_direct_calls e
    = rebuild_unit_cc_from_tables F T cannot_change code_deps
        preserve_direct_calls e.

Definition rebuild_unbox_without_unboxing_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    static_constants (unit_codes e) e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    FlambdaReaperCallingConventionsSpec.check_witness_code_ids
      (traverse_unit flag u e)
    = true ->
    fst (unit_unboxed_fields (traverse_unit flag u e) T dominated to_unbox e)
    = [] ->
    arities_match (call_results (traverse_unit flag u e) T e)
      (unit_code_deps flag u e) [] e
    = true ->
    program_refines_observably e
      (rebuild_unit_unbox (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

(* The premise holds of the units of the examples with calls, the two
   above and those of FlambdaReaperCallingConventions.v and
   FlambdaReaperUnboxing.v (where, unboxing nothing, the unit of
   [kept_result_unboxed] is rebuilt as by the change of calling
   conventions). *)
Definition without_unboxing_examples_statement : Prop :=
  arities_match
    (call_results CallExample.facts CallExample.tables CallExample.unit_expr)
    CallExample.code_deps [] CallExample.unit_expr
  = true
  ∧ arities_match
      (call_results WrapperExample.facts WrapperExample.tables
         WrapperExample.unit_expr)
      WrapperExample.code_deps [] WrapperExample.unit_expr
    = true
  ∧ arities_match
      (call_results FlambdaReaperCallingConventions.Examples.two_calls_facts
         FlambdaReaperCallingConventions.Examples.two_calls_tables
         FlambdaReaperCallingConventions.Examples.two_calls)
      (unit_code_deps Auto FlambdaReaperCallingConventions.Examples.interface
         FlambdaReaperCallingConventions.Examples.two_calls)
      [] FlambdaReaperCallingConventions.Examples.two_calls
    = true
  ∧ arities_match
      (call_results Examples.kept_result_unboxed_facts
         Examples.kept_result_unboxed_tables Examples.kept_result_unboxed)
      (unit_code_deps Auto Examples.interface Examples.kept_result_unboxed)
      [] Examples.kept_result_unboxed
    = true
  ∧ fst
      (unit_unboxed_fields CallExample.facts CallExample.tables
         CallExample.dominated [] CallExample.unit_expr)
    = [].
