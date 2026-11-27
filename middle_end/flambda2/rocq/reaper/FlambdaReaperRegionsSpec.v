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

(* The rebuild of the reaper on units with regions, through the erasure
   of the core (FlambdaRegionsErasureSpec.v).

   This file is to be reviewed: the fragment [regions_plain], the
   relation [regions_refine_observably], the examples and the
   statements. FlambdaReaperRegionsProofs.v proves them, with the lemmas
   of FlambdaReaperRegionsRebuildProofs.v (the commutation),
   FlambdaReaperRegionsTraverseProofs.v (the two traversals) and
   FlambdaReaperRegionsBaseProofs.v (the refinement of the plain rebuild
   for any first fresh name), which need no review.

   WHY

   The statements of the reaper are of the machine of
   FlambdaSemanticsSpec.v, which has no step for a primitive of regions:
   of a unit with regions they say nothing, as its run goes wrong there.
   The compiler's reaper is run on units with regions, and its traversal
   reads their primitives (FlambdaReaperTraversal.v). The core has a
   machine with regions (FlambdaRegionsSpec.v), and the erasure: a run on
   that machine that does not stop without a step is the run of the term
   with its regions dropped ([erase]), on the machine of
   FlambdaSemanticsSpec.v. The statements here carry the refinement of the
   plain rebuild to the machine of regions through the erasure.

   WHAT THE REBUILD DOES WITH REGIONS

   (FlambdaReaperRebuild.v, "Regions".) The [Let] of a begin binds a
   variable of the kind [Region], which is always used, and is kept. The
   [Let] of an end is kept where its variable has a use, which the
   traversal gives it wherever the code around it is used. The [Let] of a
   [Local_allocation] that binds again the variable of the allocation
   just before it is kept or removed with that allocation, by the same
   test. Region parameters are kept, and their arguments.

   THE COMMUTATION ([rebuild_erase_commutes_statement])

   On the fragment [regions_plain], and for any tables: erasing the
   rebuilt unit is rebuilding the erased unit, with the facts of the
   traversal of the erased unit, the same tables, and the first fresh name
   of the unit with its regions. (That of the erased unit may be smaller,
   as the erasure drops the variables of the regions; the rebuild uses it
   only to name the wrappers of calls.) The two traversals give the same
   constructors, accessors, parameters and arguments, which are all that
   the rebuild reads of the facts; their other facts differ by the facts
   of the regions, which the erased graph has not, and by an alias of the
   variable of each end to the constants, which the erased graph has, as
   the erasure binds that variable to the unit value.

   THE FRAGMENT ([regions_plain])

   No binder of the kind [Region] but the begin of a region (no region
   parameter of a handler or of code, no result of a call of that kind);
   begins and ends applied as the conversion applies them; a
   [Local_allocation] directly after the [Let] of an allocation of the
   variable that it binds again, at the same kind; no variable of a region
   evaluated, but by an end or a [Local_allocation]; no primitive of
   regions among the projections of the closures of a set. The conversion
   of the test tool makes all but the last of every unit that has no
   region parameter: on the corpus of the tool, three units of
   tests/semantics keep one after Simplify, and none compiled from OCaml.
   The last refuses a local set of several closures, where the conversion
   puts the [Local_allocation] between the first closure and the
   projections of the others; there the commutation fails
   ([local_set_counterexample_statement], below).

   THE REFINEMENT ([least_rebuild_regions_refines_statement])

   For a least solution of the graph of the unit with its regions, which
   is the analysis that the compiler runs, and the premises of
   FlambdaReaperLeastSpec.least_rebuild_refines_observably_statement on
   the erased unit: if the rebuilt unit, as the unit, is erasable and has
   no symbol at or above the base (FlambdaRegionsSpec.erasable,
   region_names_fresh, with the empty static environment), and passes the
   check of regions (FlambdaRegionsCheckSpec.region_check, with any
   certificate), it refines the unit on the machine of regions
   ([regions_refine_observably]). The check is asked of the rebuilt unit
   alone, as FlambdaRegionsCheckSoundSpec.v asks it of the term after a
   run of Simplify: by its soundness, a run of the rebuilt unit on the
   machine of regions stops without a step only where the run of its
   erasure goes wrong.

   The other premises on the rebuilt unit follow from those on the unit.
   On the fragment, for any facts and tables, the rebuild keeps
   [erasable] ([rebuild_keeps_erasable_statement]): it makes no poison of
   the kind [Region], as a variable of that kind always has a use and is
   the argument of no call, and no handler with a parameter of that kind,
   as no parameter of a handler is of that kind and no call returns one.
   And it makes no symbol ([rebuild_keeps_fresh_statement]). So the
   refinement holds with the check of regions as the one premise on the
   rebuilt unit ([least_rebuild_regions_refines_checked_statement]).

   The proof: the erased unit is refined by its rebuild with the facts of
   its traversal and the least solution of the unit with the tuples that
   the erasure adds (the alias of each variable that has any source to the
   constants, which then have every usage). That solution is closed under
   the rules on the erased graph, as no fact of the graph of a unit puts
   something into the constants, and a least solution has no alias from
   them; and the rebuild does not ask whether the constants have a use.
   By the commutation that rebuild is the erasure of the rebuilt unit, and
   the statements of the erasure relate each run on the machine of
   regions to the run of the erased term
   (FlambdaRegionsCallsSpec.erasure_exits_erased_statement,
   erasure_adds_no_exit_erased_statement).

   WHAT FAILS

   - [boxed_poison_counterexample_statement]. The rebuild makes poison a
     boxed number whose contents nothing reads, and keeps the
     [Local_allocation] after it: on the machine of regions the rebuilt
     unit has no step there (a [Local_allocation] after something that is
     not an allocation), where the unit exits. The rebuilt unit fails the
     check of regions with every certificate, so the statement of the
     refinement holds of it by its premise. The commutation holds of it.
     The unit passes the check of regions, so the check is not kept by the
     rebuild ([region_check_not_kept_statement]): no certificate for the
     rebuilt unit can be built from one for the unit where the rebuild
     makes a local boxed number poison. An artefact of the model: in the
     compiler the mode [Local] is a part of the [Box_number] primitive,
     which rebuild.ml replaces whole by poison ([rebuild_let_expr_holed],
     the case of [Box_number]).
   - [local_set_counterexample_statement]. On a local set of two
     closures, the rebuild with regions removes the projection of the
     second closure, which has no use: the [Local_allocation] hides the
     projections from the rebuild of the set
     (FlambdaReaperRebuild.projections_of_set does not look past it, where
     the traversal does), so the projection is rebuilt as any [Let]. The
     rebuild of the erased unit keeps it with the set. The two rebuilt
     units differ, and both exit as the unit does. An artefact of the
     model: rebuild.ml keeps every closure of a set that it keeps
     ([rebuild_let_expr_holed_set_of_closures]), and binds them all at
     once, as the model does not.
   - The example [regions_example_statement]: a region with two local
     blocks, of which the second has no use; the statement of the
     refinement applies to it, and its rebuild drops the unused block.

   WHAT IS NOT STATED

   - The rebuild with unboxing across calls. That with changes of
     calling convention is stated in FlambdaReaperRegionsCcSpec.v, and
     the local unboxing in FlambdaReaperRegionsUnboxSpec.v.
   - Units with a parameter of the kind [Region], or a local set of
     several closures.
   - That the rebuilt unit passes the check of regions wherever the unit
     does: false ([region_check_not_kept_statement]).
   - Anything of a run that goes on for ever, or of the heap at the
     exit. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaPrimitiveModel
  FlambdaModel FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaWellFormedSpec FlambdaNameAbstractionSpec FlambdaRenameApartSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperLeastSpec FlambdaRegionsSpec FlambdaRegionsErasureSpec
  FlambdaRegionsCheckSpec.
Import ListNotations.

(* ======================================================================== *)
(* The fragment                                                             *)
(* ======================================================================== *)

(* Whether a variable is of the kind [Region] in [kinds], which has the
   kinds of the variables in scope as the rebuild has them
   (FlambdaReaperRebuild.kind_of_variable). *)
Definition region_variable (kinds : finite_map flambda_kind) (x : variable)
    : bool :=
  match lookup x kinds with
  | Some kind => kind_eqb kind Region
  | None => false
  end.

(* A simple that is not a variable of the kind [Region]. *)
Definition simple_not_region (kinds : finite_map flambda_kind) (s : simple)
    : bool :=
  match s with
  | Var x => negb (region_variable kinds x)
  | Symbol _ | Const _ => true
  end.

Definition kind_not_region (kind : flambda_kind) : bool :=
  negb (kind_eqb kind Region).

(* The projections of the other closures of a set whose first closure
   is bound to [x] at [slot] (FlambdaModel.let_closures), as the
   rebuild finds them ([projected_closures]): no primitive of regions
   is among them, so that they are the same with the regions dropped. A
   [Local_allocation] after the first closure of a set of several
   closures is such a primitive. *)
Fixpoint projections_plain (x : variable) (slot : function_slot)
    (others : list function_slot) (e : expr) {struct e} : bool :=
  match others, e with
  | [], _ => true
  | other :: others',
    Let y _ (Project_function_slot move_from move_to (Var x')) body =>
    if Pos.eqb x' x && Pos.eqb move_from slot && Pos.eqb move_to other
       && negb (Pos.eqb y x)
    then projections_plain x slot others' body
    else true
  | _ :: _, Let _ _ (Prim (Region_primitive _) _) _ => false
  | _ :: _, _ => true
  end.

Definition set_plain (x : variable) (defining_expr : named) (body : expr)
    : bool :=
  match defining_expr with
  | Closure_of_set slot function_decls _ =>
    projections_plain x slot (other_slots slot function_decls) body
  | _ => true
  end.

(* The regions of a term are as the conversion makes them, and kept
   apart from everything else ([kinds] is as above):
   - the only binder of the kind [Region] is the [Let] of a begin of a
     region, with no argument; no parameter of a handler or of code,
     and no result of a call, is of that kind;
   - an end of a region has one argument, and binds a variable of
     another kind;
   - a [Local_allocation] comes directly after the [Let] of an
     allocation (FlambdaRegionsSpec.is_allocation, is_cell_allocation)
     of the variable that it applies to, at the same kind, and binds
     that variable again;
   - a variable of the kind [Region] is evaluated nowhere: it is only
     the argument of an end or an argument of a [Local_allocation]
     after the first;
   - no primitive of regions is among the projections of the closures
     of a set ([set_plain]). *)
Fixpoint regions_plain (kinds : finite_map flambda_kind) (e : expr)
    {struct e} : bool :=
  match e with
  | Let x kind (Prim (Region_primitive p) args) body =>
    match p, args with
    | (Begin_region | Begin_try_region), [] =>
      kind_eqb kind Region && regions_plain ((x, Region) :: kinds) body
    | (End_region | End_try_region), [_] =>
      kind_not_region kind && regions_plain ((x, kind) :: kinds) body
    | _, _ => false
    end
  | Let x kind defining_expr body =>
    kind_not_region kind
    && forallb (simple_not_region kinds) (simples_of_named defining_expr)
    && set_plain x defining_expr body
    && match body with
       | Let y kind' (Prim (Region_primitive Local_allocation) args) body' =>
         (is_allocation defining_expr || is_cell_allocation defining_expr)
         && Pos.eqb y x && kind_eqb kind' kind
         && match args with
            | Var x' :: _ => Pos.eqb x' x
            | _ => false
            end
         && regions_plain ((x, kind) :: (x, kind) :: kinds) body'
       | _ => regions_plain ((x, kind) :: kinds) body
       end
  | Let_static bindings body =>
    forallb (binding_regions_plain kinds) bindings
    && regions_plain kinds body
  | Let_cont_non_recursive _ h body =>
    handler_regions_plain kinds h && regions_plain kinds body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_regions_plain kinds h) handlers
    && regions_plain kinds body
  | Apply ap =>
    forallb (simple_not_region kinds)
      (ap_callee ap :: ap_args ap ++ exn_extra_simples ap)
    && forallb kind_not_region (unarize (ap_return_arity ap))
  | Apply_cont ac => forallb (simple_not_region kinds) (ac_args ac)
  | Switch scrutinee arms =>
    simple_not_region kinds scrutinee
    && forallb (fun '(_, ac) => forallb (simple_not_region kinds) (ac_args ac))
         arms
  | Invalid => true
  end
with handler_regions_plain (kinds : finite_map flambda_kind)
    (h : continuation_handler) {struct h} : bool :=
  match h with
  | Handler params metadata handler =>
    forallb kind_not_region (hm_params_kinds metadata)
    && regions_plain
         (rev (with_kinds params (hm_params_kinds metadata)) ++ kinds) handler
  end
with binding_regions_plain (kinds : finite_map flambda_kind)
    (binding : static_binding) {struct binding} : bool :=
  match binding with
  | Bind_code _ c => code_regions_plain c
  | Bind_deleted_code _ => true
  | Bind_symbol _ const =>
    forallb (simple_not_region kinds) (simples_of_static_const const)
  end
with code_regions_plain (c : code) {struct c} : bool :=
  match c with
  | Code params my_closure _ _ metadata body =>
    forallb kind_not_region (unarize (cm_params_arity metadata))
    && regions_plain
         (rev (with_kinds params (unarize (cm_params_arity metadata)))
          ++ [(my_closure, Value)])
         body
  end.

(* ======================================================================== *)
(* Refinement on the machine of regions                                     *)
(* ======================================================================== *)

(* [target] refines [source] on the machine of regions
   (FlambdaRegionsSpec.v), from their first states in [statics] with the
   symbols drawn from [base], as far as the erasure relates the runs of
   that machine to the runs of the terms with their regions dropped:
   - an exit of the source is an exit of the target through the same
     continuation, and the values of the two exits are erased
     (FlambdaRegionsErasureSpec.value_erased, in the state that each run
     exits in) to values that are observed equal ([observed_equal]);
   - an exit of the target is an exit of the source alike, or the
     source stops without a step, on the machine of regions or with its
     regions dropped;
   - the target stops without a step only where the source, with its
     regions dropped, goes wrong. *)
Definition regions_refine_observably (statics : static_env) (base : symbol)
    (source target : expr) : Prop :=
  (∀ (fuel : nat) (k : continuation) (vs : list value) (rs : region_state),
     region_run fuel (initial_region_state source statics base)
     = Region_exit k vs rs ->
     ∃ (fuel' : nat) (ws : list value) (rs' : region_state)
       (us us' : list value),
       region_run fuel' (initial_region_state target statics base)
       = Region_exit k ws rs'
       ∧ Forall2 (value_erased rs) vs us
       ∧ Forall2 (value_erased rs') ws us'
       ∧ values_related observed_equal us us')
  ∧ (∀ (fuel' : nat) (k : continuation) (ws : list value)
       (rs' : region_state),
       region_run fuel' (initial_region_state target statics base)
       = Region_exit k ws rs' ->
       (∃ (fuel : nat) (vs : list value) (rs : region_state)
          (us us' : list value),
          region_run fuel (initial_region_state source statics base)
          = Region_exit k vs rs
          ∧ Forall2 (value_erased rs) vs us
          ∧ Forall2 (value_erased rs') ws us'
          ∧ values_related observed_equal us us')
       ∨ (∃ (fuel : nat) (fault : region_fault),
            region_run fuel (initial_region_state source statics base)
            = Region_undefined fault)
       ∨ behaves (erased_initial_state source statics) Goes_wrong)
  ∧ (∀ (fuel' : nat) (fault : region_fault),
       region_run fuel' (initial_region_state target statics base)
       = Region_undefined fault ->
       behaves (erased_initial_state source statics) Goes_wrong).

(* ======================================================================== *)
(* Statements                                                               *)
(* ======================================================================== *)

(* Erasing the rebuilt unit is rebuilding the erased unit: with the
   facts of the erased unit, the same tables, and the first fresh name
   of the unit with its regions. *)
Definition rebuild_erase_commutes_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (u : unit_interface) (e : expr)
    (T : tables) (preserve_direct_calls : bool),
    regions_plain [] e = true ->
    erase [] (rebuild_unit_from_tables (traverse_unit flag u e) T
                preserve_direct_calls e)
    = rebuild_expr (traverse_unit flag u (erase [] e)) (solution_of_tables T)
        preserve_direct_calls (static_names code_ids_bound (erase [] e))
        (fresh_name (names_of e)) None FlambdaReaperRebuild.empty_env
        (erase [] e).

(* The rebuild, for the least solution of the graph of the unit with its
   regions, refines the unit on the machine of regions, if the rebuilt
   unit passes the check of regions. The premises on the term with its
   regions dropped are those of
   FlambdaReaperLeastSpec.least_rebuild_refines_observably_statement. *)
Definition least_rebuild_regions_refines_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) (cert : certificate) (base : symbol),
    regions_plain [] e = true ->
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free (erase [] e) = true ->
    in_flow_fragment (erase [] e) = true ->
    any_kind_arguments (erase [] e) = true ->
    results_kinded [] [] (erase [] e) = true ->
    well_formed_unit u (erase [] e) = true ->
    exits_one_value u (erase [] e) = true ->
    least_solution is_local (traverse_unit flag u e) T ->
    region_names_fresh base e empty_static_env = true ->
    erasable e empty_static_env = true ->
    region_names_fresh base
      (rebuild_unit_from_tables (traverse_unit flag u e) T
         preserve_direct_calls e)
      empty_static_env
    = true ->
    erasable
      (rebuild_unit_from_tables (traverse_unit flag u e) T
         preserve_direct_calls e)
      empty_static_env
    = true ->
    region_check cert empty_static_env
      (rebuild_unit_from_tables (traverse_unit flag u e) T
         preserve_direct_calls e)
    = true ->
    regions_refine_observably empty_static_env base e
      (rebuild_unit_from_tables (traverse_unit flag u e) T
         preserve_direct_calls e).

(* On the fragment, for any facts and tables, the rebuilt unit is
   erasable wherever the unit is: the rebuild makes no poison of the kind
   [Region] (a variable of that kind is always used, and none is the
   argument of a call), and no handler with a parameter of that kind. *)
Definition rebuild_keeps_erasable_statement : Prop :=
  ∀ (F : facts) (T : tables) (preserve_direct_calls : bool) (e : expr),
    regions_plain [] e = true ->
    erasable e empty_static_env = true ->
    erasable (rebuild_unit_from_tables F T preserve_direct_calls e)
      empty_static_env
    = true.

(* And the rebuilt unit has no symbol at or above the base wherever the
   unit has none: the rebuild makes no symbol. *)
Definition rebuild_keeps_fresh_statement : Prop :=
  ∀ (F : facts) (T : tables) (preserve_direct_calls : bool) (e : expr)
    (base : symbol),
    regions_plain [] e = true ->
    region_names_fresh base e empty_static_env = true ->
    region_names_fresh base
      (rebuild_unit_from_tables F T preserve_direct_calls e) empty_static_env
    = true.

(* So [least_rebuild_regions_refines_statement] holds with no premise on
   the rebuilt unit but the check of regions. *)
Definition least_rebuild_regions_refines_checked_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) (cert : certificate) (base : symbol),
    regions_plain [] e = true ->
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free (erase [] e) = true ->
    in_flow_fragment (erase [] e) = true ->
    any_kind_arguments (erase [] e) = true ->
    results_kinded [] [] (erase [] e) = true ->
    well_formed_unit u (erase [] e) = true ->
    exits_one_value u (erase [] e) = true ->
    least_solution is_local (traverse_unit flag u e) T ->
    region_names_fresh base e empty_static_env = true ->
    erasable e empty_static_env = true ->
    region_check cert empty_static_env
      (rebuild_unit_from_tables (traverse_unit flag u e) T
         preserve_direct_calls e)
    = true ->
    regions_refine_observably empty_static_env base e
      (rebuild_unit_from_tables (traverse_unit flag u e) T
         preserve_direct_calls e).

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

Module RegionsExamples.

Local Open Scope positive_scope.

(* The units return to 1 and raise to 2; the machine of regions draws
   its symbols from 100 on. *)
Definition interface : unit_interface :=
  {| unit_return := 1; unit_exn := 2 |}.

Definition base : symbol := 100.

(* The certificate that says that nothing is local but what a local
   allocation makes. *)
Definition cert : certificate := Certificate [] [] false false false.

Definition return_to (k : continuation) (args : list simple) : expr :=
  Apply_cont {| ac_continuation := k; ac_args := args;
                ac_trap_action := None |}.

Definition region (p : region_primitive) (args : list simple) : named :=
  Prim (Region_primitive p) args.

Definition make_block (fields : list simple) : named :=
  Prim (Variadic (Make_block (Block_kind_values 0))) fields.

(* A load of an immediate: what it reads lives for ever, for the check
   of regions. *)
Definition load_immediate (field : Z) (block : simple) : named :=
  Prim (Unary (Block_load (Block_access_values None None Immediate)
                 Immutable field)) [block].

(* A region with two local blocks, of which the second has no use:

     let r10 = begin_region in
     let x11 = local [7] in
     let x12 = local [8] in
     let x13 = x11.(0) in
     let x14 = end_region r10 in
     return x13

   The rebuild removes the [Let]s of [x12], its [Local_allocation]
   with it, and keeps the rest. *)
Definition two_blocks : expr :=
  Let 10 Region (region Begin_region [])
    (Let 11 Value (make_block [Const (Tagged_immediate 7)])
       (Let 11 Value (region Local_allocation [Var 11; Var 10])
          (Let 12 Value (make_block [Const (Tagged_immediate 8)])
             (Let 12 Value (region Local_allocation [Var 12; Var 10])
                (Let 13 Value (load_immediate 0 (Var 11))
                   (Let 14 Value (region End_region [Var 10])
                      (return_to 1 [Var 13]))))))).

Definition two_blocks_tables : tables :=
  {| t_alias :=
       [(Node_variable 13, Node_all_constants);
        (Node_made 1, Node_variable 13)];
     t_any_usage :=
       [Node_all_constants; Node_variable 10; Node_variable 13; Node_made 2;
        Node_variable 14; Node_made 1];
     t_any_source :=
       [Node_made 1; Node_variable 13; Node_all_constants; Node_variable 10;
        Node_variable 14];
     t_has_usage :=
       [Node_variable 11; Node_variable 10; Node_all_constants;
        Node_variable 13; Node_made 1; Node_variable 14; Node_made 2];
     t_has_source :=
       [Node_made 1; Node_variable 11; Node_variable 12; Node_variable 13;
        Node_variable 14; Node_variable 10; Node_all_constants];
     t_usages := [(Node_variable 11, Node_variable 11)];
     t_sources :=
       [(Node_variable 11, Node_variable 11);
        (Node_variable 12, Node_variable 12)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used := [(Node_variable 11, Field_block 0 Value)] |}.

Definition two_blocks_rebuilt : expr :=
  Let 10 Region (region Begin_region [])
    (Let 11 Value (make_block [Const (Tagged_immediate 7)])
       (Let 11 Value (region Local_allocation [Var 11; Var 10])
          (Let 13 Value (load_immediate 0 (Var 11))
             (Let 14 Value (region End_region [Var 10])
                (return_to 1 [Var 13]))))).

(* A local boxed number whose contents nothing reads; [x11] is read in a
   handler that nothing applies, by [Is_int] (which the flow takes for a
   read of another field):

     let r10 = begin_region in
     let x11 = local (box 5) in
     let_cont k20 () = let x12 = is_int x11 in return x12 in
     let x13 = end_region r10 in
     return 0

   The rebuild makes [x11] poison and keeps its [Local_allocation]. *)
Definition boxed : expr :=
  Let 10 Region (region Begin_region [])
    (Let 11 Value
       (Prim (Unary (Box_number Boxable_naked_int64)) [Const (Naked_int64 5)])
       (Let 11 Value (region Local_allocation [Var 11; Var 10])
          (Let_cont_non_recursive 20
             (Handler [] (make_handler_metadata [])
                (Let 12 Value (Prim (Unary (Is_int true)) [Var 11])
                   (return_to 1 [Var 12])))
             (Let 13 Value (region End_region [Var 10])
                (return_to 1 [Const (Tagged_immediate 0)]))))).

Definition boxed_tables : tables :=
  {| t_alias :=
       [(Node_made 1, Node_variable 12); (Node_made 1, Node_all_constants)];
     t_any_usage :=
       [Node_variable 10; Node_all_constants; Node_variable 12; Node_made 2;
        Node_variable 13; Node_made 1];
     t_any_source :=
       [Node_made 1; Node_all_constants; Node_variable 10; Node_variable 13];
     t_has_usage :=
       [Node_variable 11; Node_variable 10; Node_variable 12;
        Node_all_constants; Node_made 1; Node_variable 13; Node_made 2];
     t_has_source :=
       [Node_variable 11; Node_made 1; Node_variable 13; Node_variable 10;
        Node_all_constants];
     t_usages := [(Node_variable 11, Node_variable 11)];
     t_sources := [(Node_variable 11, Node_variable 11)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used := [] |}.

Definition boxed_rebuilt : expr :=
  Let 10 Region (region Begin_region [])
    (Let 11 Value (Simple (Const (Poison Value)))
       (Let 11 Value (region Local_allocation [Var 11; Var 10])
          (Let 13 Value (region End_region [Var 10])
             (return_to 1 [Const (Tagged_immediate 0)])))).

(* A local set of two closures of deleted code, the second bound by a
   projection that has no use; the first is read in a handler that
   nothing applies:

     let r10 = begin_region in
     let x11 = local (closure at 70 of { 70 = f50; 71 = f51 }) in
     let x12 = x11 moved from 70 to 71 in
     let_cont k20 () = let x15 = get_tag x11 in return x15 in
     let x14 = end_region r10 in
     return 0 *)
Definition local_set : expr :=
  Let_static [Bind_deleted_code 50; Bind_deleted_code 51]
    (Let 10 Region (region Begin_region [])
       (Let 11 Value (Closure_of_set 70 [(70, 50); (71, 51)] [])
          (Let 11 Value (region Local_allocation [Var 11; Var 10])
             (Let 12 Value (Project_function_slot 70 71 (Var 11))
                (Let_cont_non_recursive 20
                   (Handler [] (make_handler_metadata [])
                      (Let 15 Value (Prim (Unary Get_tag) [Var 11])
                         (return_to 1 [Var 15])))
                   (Let 14 Value (region End_region [Var 10])
                      (return_to 1 [Const (Tagged_immediate 0)]))))))).

Definition local_set_tables : tables :=
  {| t_alias :=
       [(Node_variable 12, Node_variable 12); (Node_made 1, Node_variable 15);
        (Node_made 1, Node_all_constants)];
     t_any_usage :=
       [Node_variable 10; Node_all_constants; Node_variable 15; Node_made 2;
        Node_variable 14; Node_made 1];
     t_any_source :=
       [Node_made 1; Node_all_constants; Node_variable 10; Node_variable 14;
        Node_made 3; Node_made 4];
     t_has_usage :=
       [Node_variable 11; Node_variable 10; Node_variable 15;
        Node_all_constants; Node_made 1; Node_variable 14; Node_made 2];
     t_has_source :=
       [Node_made 1; Node_made 4; Node_made 3; Node_variable 14;
        Node_variable 10; Node_all_constants; Node_variable 11;
        Node_variable 12];
     t_usages := [(Node_variable 11, Node_variable 11)];
     t_sources :=
       [(Node_variable 12, Node_variable 12);
        (Node_variable 11, Node_variable 11)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used := [] |}.

(* The rebuild with regions, then erased: the projection of [x12] is
   gone. *)
Definition local_set_rebuilt_erased : expr :=
  Let_static [Bind_deleted_code 50; Bind_deleted_code 51]
    (Let 11 Value (Closure_of_set 70 [(70, 50); (71, 51)] [])
       (Let 14 Value (Simple (Const (Tagged_immediate 0)))
          (return_to 1 [Const (Tagged_immediate 0)]))).

(* The rebuild of the erased unit: the projection is kept. *)
Definition local_set_erased_rebuilt : expr :=
  Let_static [Bind_deleted_code 50; Bind_deleted_code 51]
    (Let 11 Value (Closure_of_set 70 [(70, 50); (71, 51)] [])
       (Let 12 Value (Project_function_slot 70 71 (Var 11))
          (Let 14 Value (Simple (Const (Tagged_immediate 0)))
             (return_to 1 [Const (Tagged_immediate 0)])))).

End RegionsExamples.

Import RegionsExamples.

(* The statement applies to [two_blocks]: it meets every premise, its
   rebuild drops the unused local block, and both exit with 7 on the
   machine of regions. *)
Definition regions_example_statement : Prop :=
  regions_plain [] two_blocks = true
  ∧ least_solution (fun _ => false)
      (traverse_unit Auto interface two_blocks) two_blocks_tables
  ∧ in_flow_fragment (erase [] two_blocks) = true
  ∧ any_kind_arguments (erase [] two_blocks) = true
  ∧ results_kinded [] [] (erase [] two_blocks) = true
  ∧ well_formed_unit interface (erase [] two_blocks) = true
  ∧ exits_one_value interface (erase [] two_blocks) = true
  ∧ region_names_fresh base two_blocks empty_static_env = true
  ∧ erasable two_blocks empty_static_env = true
  ∧ rebuild_unit_from_tables (traverse_unit Auto interface two_blocks)
      two_blocks_tables true two_blocks
    = two_blocks_rebuilt
  ∧ region_names_fresh base two_blocks_rebuilt empty_static_env = true
  ∧ erasable two_blocks_rebuilt empty_static_env = true
  ∧ region_check cert empty_static_env two_blocks_rebuilt = true
  ∧ (∃ rs : region_state,
       region_run 10 (initial_region_state two_blocks empty_static_env base)
       = Region_exit 1%positive [Value_const (Tagged_immediate 7)] rs)
  ∧ (∃ rs : region_state,
       region_run 10
         (initial_region_state two_blocks_rebuilt empty_static_env base)
       = Region_exit 1%positive [Value_const (Tagged_immediate 7)] rs)
  ∧ regions_refine_observably empty_static_env base two_blocks
      two_blocks_rebuilt.

(* The rebuild keeps the [Local_allocation] after a boxed number that it
   makes poison: [boxed] meets the premises of
   [least_rebuild_regions_refines_statement] but the check of its
   rebuild, which fails with every certificate; it exits with 0 on the
   machine of regions, and its rebuild has no step there, at the
   [Local_allocation] after something that is not an allocation. The
   rebuild does not refine it on that machine. *)
Definition boxed_poison_counterexample_statement : Prop :=
  regions_plain [] boxed = true
  ∧ least_solution (fun _ => false) (traverse_unit Auto interface boxed)
      boxed_tables
  ∧ in_flow_fragment (erase [] boxed) = true
  ∧ any_kind_arguments (erase [] boxed) = true
  ∧ results_kinded [] [] (erase [] boxed) = true
  ∧ well_formed_unit interface (erase [] boxed) = true
  ∧ exits_one_value interface (erase [] boxed) = true
  ∧ region_names_fresh base boxed empty_static_env = true
  ∧ erasable boxed empty_static_env = true
  ∧ rebuild_unit_from_tables (traverse_unit Auto interface boxed)
      boxed_tables true boxed
    = boxed_rebuilt
  ∧ region_names_fresh base boxed_rebuilt empty_static_env = true
  ∧ erasable boxed_rebuilt empty_static_env = true
  ∧ (∀ c : certificate, region_check c empty_static_env boxed_rebuilt = false)
  ∧ (∃ rs : region_state,
       region_run 10 (initial_region_state boxed empty_static_env base)
       = Region_exit 1%positive [Value_const (Tagged_immediate 0)] rs)
  ∧ region_run 10 (initial_region_state boxed_rebuilt empty_static_env base)
    = Region_undefined Not_of_the_conversion
  ∧ ~ regions_refine_observably empty_static_env base boxed boxed_rebuilt.

(* The check of regions is not kept by the rebuild: [boxed] passes it,
   with the certificate that says that nothing is local but what a local
   allocation makes, and its rebuild for the least solution fails it with
   every certificate ([boxed_poison_counterexample_statement]). *)
Definition region_check_not_kept_statement : Prop :=
  regions_plain [] boxed = true
  ∧ least_solution (fun _ => false) (traverse_unit Auto interface boxed)
      boxed_tables
  ∧ region_check cert empty_static_env boxed = true
  ∧ (∀ c : certificate,
       region_check c empty_static_env
         (rebuild_unit_from_tables (traverse_unit Auto interface boxed)
            boxed_tables true boxed)
       = false).

(* The rebuild does not commute with the erasure on a local set of two
   closures ([local_set], which [regions_plain] refuses): rebuilt with
   its regions, the projection of the second closure, which has no use,
   is removed, as the [Local_allocation] between the first closure and
   the projection hides the projection from the rebuild of the set;
   rebuilt with its regions dropped, the projection is kept with the
   set. Both rebuilds exit as the unit does. *)
Definition local_set_counterexample_statement : Prop :=
  regions_plain [] local_set = false
  ∧ least_solution (fun _ => false) (traverse_unit Auto interface local_set)
      local_set_tables
  ∧ region_names_fresh base local_set empty_static_env = true
  ∧ erasable local_set empty_static_env = true
  ∧ erase []
      (rebuild_unit_from_tables (traverse_unit Auto interface local_set)
         local_set_tables true local_set)
    = local_set_rebuilt_erased
  ∧ rebuild_expr (traverse_unit Auto interface (erase [] local_set))
      (solution_of_tables local_set_tables) true
      (static_names code_ids_bound (erase [] local_set))
      (fresh_name (names_of local_set)) None FlambdaReaperRebuild.empty_env
      (erase [] local_set)
    = local_set_erased_rebuilt
  ∧ local_set_rebuilt_erased <> local_set_erased_rebuilt
  ∧ (∃ rs : region_state,
       region_run 10 (initial_region_state local_set empty_static_env base)
       = Region_exit 1%positive [Value_const (Tagged_immediate 0)] rs)
  ∧ (∃ rs : region_state,
       region_run 10
         (initial_region_state
            (rebuild_unit_from_tables (traverse_unit Auto interface local_set)
               local_set_tables true local_set)
            empty_static_env base)
       = Region_exit 1%positive [Value_const (Tagged_immediate 0)] rs).
