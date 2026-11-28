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

(* The result types of code at the parameters of the code, end to end, at
   the nodes of the parameters of its code dep, as the compiler reads them:
   the type that the result types give a parameter of code, rewritten as
   the compiler rewrites it, holds of the value of the parameter in the
   rebuilt unit.

   This file is to be reviewed: [param_node], [at_param], the check
   ([only_read_by], [params_aliased]), the premises ([param_premises])
   and the statements, against FlambdaReaperTypesReturnsSpec.v (the
   results of code, whose definitions these follow),
   FlambdaReaperTypesCallsUsedSpec.v (whose premises these take) and
   FlambdaReaperIndirectDeadFramesSpec.only_alias_into. The proofs are in
   FlambdaReaperTypesParamsProofs.v, which needs no review.

   The nodes of the parameters. The compiler gives [rewrite_result_types]
   the variables of the code dep of the code (unboxing_analysis.ml:
   [code_dep.params] and [code_dep.return]), and
   [Types_rewriter.rewrite_result_types] rewrites the type of each
   parameter under the metadata of the variable of [code_dep.params] at
   its rank ([variable_pattern], [metadata_for_name context var]). Those
   variables are made by [Traverse.prepare_code], one for each parameter
   of the code; the model's traversal makes a node for each
   (FlambdaReaperTraversal.prepare_code, [cd_params]), so the node of the
   parameter of rank [i] is the node of rank [i] of [cd_params] of the
   code dep of the code ([param_node]), and its type is rewritten under
   the metadata of that node
   (FlambdaReaperTypesReturnsSpec.rewrite_result_type_at). The
   statements of FlambdaReaperTypesCallsSpec.v are of the parameters of
   the code (FlambdaReaperTypes.rewrite_result_type at the node of the
   variable).

   The parameter and its node. The traversal aliases each parameter of
   the code to the node of its rank (FlambdaReaperTraversal.traverse_code,
   an alias into the node of the parameter from the node of the code
   dep), and the calls of the code give their arguments to the node of
   the code dep: it stands for the arguments of every call. The rule of
   aliases gives the parameter the sources of the node, and the node the
   usages of the parameter. The simulations of the rebuilds relate the
   value of a parameter at its own node (FlambdaReaperTypesSpec
   .rebuilt_value at the node of the variable). Where no other fact puts
   anything into the parameter (FlambdaReaperIndirectDeadFramesSpec
   .only_alias_into) and no other fact reads the node ([only_read_by]),
   a least solution gives the two the same sources and usages, so that
   what the value of the parameter may come from is described by the
   metadata of the node of the code dep: the value is rebuilt under that
   metadata ([param_value_at_node_statement]). [params_aliased] checks it
   of every parameter of the code of the unit, a test of the facts.

   The statements; those of types each with [_landed] for the fixes to
   land (FlambdaReaperTypes.landed_fixes), and those end to end with
   [_local] and [_local_landed] for a least solution of the rules with
   the fix of local fields (FlambdaReaperTypesCallsSpec.least_with):
   - [param_value_at_node_statement]: for a least solution, a unit that
     passes [params_aliased] and the parameter [p] of rank [i] of the
     code [id], whose node [r] of the code dep has a use, a value rebuilt
     at the node of [p] is rebuilt under the metadata of [r], and [p] has
     a use.
   - [param_type_end_to_end_statement]: at each state that the run of the
     unit comes to and from which it steps, where the parameter [p] of
     rank [i] of the code [id] has the value [v], the type that the
     result types give [type_var], which holds of [v], rewritten under
     the metadata of the node [r] of rank [i] of the parameters of the
     code dep of [id], holds of the value of [p] at a state that the run
     of the plain rebuild comes to.
   - [cc_param_type_end_to_end_statement]: the same of the rebuild with
     changes of calling convention; the node has a use, so the change
     keeps the parameter.
   - [param_example_statement]: the unit of
     FlambdaReaperUnboxingRefinesSpec.CallExample, whose code 50 takes the
     block [x40] of which it reads the field 1. The node of its parameter
     is the made node 6, whose only source is the block [x10] of 7 and 8,
     passed by the call. The result types give the parameter, the
     variable 90 of the result types, the type of a block of tag 0 whose
     field 1 is 8 (FlambdaReaperTypesCallsSpec.UnboxCallsExample), which
     holds of [7; 8]. The unit passes the check; the type rewritten under
     the metadata of the node is the type itself (the field 0 was
     unknown); the plain rebuild makes the field 0 of [x10], which no one
     reads, poison, and its run binds [x40] to the block of poison and 8,
     of which the type holds.

   Of the premises: those of FlambdaReaperTypesCallsUsedSpec.v
   ([used_premises], and [check_cc_closed] and [check_apply_widgets] for
   the rebuild with changes of calling convention), that the node of the
   parameter has a use (the rebuild makes poison of a parameter with no
   use, and the change of calling conventions deletes it, and the
   compiler drops its type: [Delete] in [rewrite_result_types]), and the
   check [params_aliased].

   The traversal of every unit of the fragment whose variables are bound
   once passes [params_aliased], and the statements here hold without it
   (FlambdaReaperTypesParamsTraversalSpec.v).

   What is left out:
   - The parameters that the unboxing across calls unboxes: the compiler
     rewrites their types into the types of their fields
     (FlambdaReaperTypesCallsSpec
     .calls_unboxed_result_type_end_to_end_statement, at the node of the
     variable). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaPrimitiveSpec FlambdaSemanticsSpec
  FlambdaWellFormedSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperRebuild FlambdaReaperFlowSpec
  FlambdaReaperLeastSpec FlambdaReaperLocalFields FlambdaReaperTypes
  FlambdaReaperTypesSpec FlambdaReaperTypesTermSpec
  FlambdaReaperCallingConventions FlambdaReaperTypesCallsSpec
  FlambdaReaperTypesReturnsSpec FlambdaReaperTypesUsedValuesSpec
  FlambdaReaperTypesCallsUsedSpec FlambdaReaperIndirectDeadFramesSpec.
From OxCamlFlambda2 Require FlambdaReaperRebuildSpec FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperCallingConventionsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaTypingEnv
  FlambdaTypingEnvSpec.
Import ListNotations.

(* ======================================================================== *)
(* The parameters of code                                                   *)
(* ======================================================================== *)

(* The node of the parameter of rank [i] of the code [id]: the node of
   that rank of [cd_params] of its code dep, which stands for the variable
   of [code_dep.params] that the compiler makes. *)
Definition param_node (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) (id : code_id) (i : nat) : option node :=
  match lookup id (unit_code_deps flag u e) with
  | Some cd => nth_error (cd_params cd) i
  | None => None
  end.

(* [p] is the parameter of rank [i] of the code [id] that the unit
   binds. *)
Definition at_param (e : expr) (id : code_id) (i : nat) (p : variable)
    : Prop :=
  ∃ c : code, (id, c) ∈ unit_codes e ∧ nth_error (code_params c) i = Some p.

(* No fact of [F] puts [r] where a rule may conclude [alias _ r],
   [usages r _], [any_usage r] or [has_usage r], but aliases and
   propagations from [r] into [x]. *)
Definition only_read_by (F : facts) (r x : node) : bool :=
  forallb (fun '(to_, y) => negb (node_eqb y r) || node_eqb to_ x) (f_alias F)
  && forallb (fun '(_, to_, y) => negb (node_eqb y r) || node_eqb to_ x)
       (f_propagate F)
  && forallb (fun '(_, _, y) => negb (node_eqb y r)) (f_alias_if_any_source F)
  && forallb (fun '(_, _, y) => negb (node_eqb y r)) (f_constructor F)
  && forallb (fun '(y, _, b) => negb (node_eqb y r || node_eqb b r))
       (f_argument F)
  && forallb (fun '(_, _, b) => negb (node_eqb b r)) (f_accessor F)
  && forallb (fun '(_, y) => negb (node_eqb y r)) (f_use F)
  && negb (mem_node r (f_any_usage F)).

(* Each parameter of the code of the unit [e] is put something into only
   by the alias from the node of its rank of the code dep of the code
   ([cds]), which is read only by that alias. *)
Definition params_aliased (F : facts) (cds : finite_map code_dep) (e : expr)
    : bool :=
  forallb
    (fun '(id, c) =>
       match lookup id cds with
       | Some cd =>
         forallb
           (fun '(p, r) =>
              only_alias_into F (Node_variable p) r
              && only_read_by F r (Node_variable p))
           (combine (code_params c) (cd_params cd))
       | None => true
       end)
    (unit_codes e).

(* The premises on a state [s] of the run of the unit where the parameter
   [p] of rank [i] of the code [id] has the value [v]: the run comes to
   [s] and steps from it, the node [r] of the parameter has a use, and a
   valuation of the names of the result types [tenv] that gives
   [type_var] the value [v] satisfies them, with the copies of the
   rewriting valued as [copies_valued] says. *)
Definition param_premises (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) (T : tables) (n : nat) (s : state)
    (id : code_id) (i : nat) (p : variable) (r : node) (tenv : Typing_env.t)
    (canon : canon_fn) (valuation valuation' : env) (type_var : variable)
    (v : value) : Prop :=
  run n (initial_state e) = Step s
  ∧ step s <> Undefined
  ∧ at_param e id i p
  ∧ lookup p (state_vars s) = Some v
  ∧ param_node flag u e id i = Some r
  ∧ s_has_usage (solution_of_tables T) r = true
  ∧ typing_env_satisfied_strictly (state_statics s) valuation tenv
  ∧ env_in_fragment valuation tenv
  ∧ copies_valued tenv canon valuation valuation'
  ∧ Typing_env.find tenv (Name_var type_var) None <> None
  ∧ lookup type_var valuation = Some v.

(* ======================================================================== *)
(* The value of a parameter at the node of the code dep                     *)
(* ======================================================================== *)

Definition param_value_at_node_statement : Prop :=
  ∀ (local : bool) (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (sets : list set_of_closures) (id : code_id) (i : nat) (p : variable)
    (r : node) (v v' : value),
    is_local_slots_only is_local ->
    least_with local is_local (traverse_unit flag u e) T e ->
    params_aliased (traverse_unit flag u e) (unit_code_deps flag u e) e
    = true ->
    at_param e id i p ->
    param_node flag u e id i = Some r ->
    s_has_usage (solution_of_tables T) r = true ->
    rebuilt_value (graph_of_facts (traverse_unit flag u e))
      (solution_of_tables T) v (Node_variable p) v' ->
    s_has_usage (solution_of_tables T) (Node_variable p) = true
    ∧ rebuilt_under (graph_of_facts (traverse_unit flag u e))
        (solution_of_tables T)
        (metadata_for_name
           (plain_context is_local (traverse_unit flag u e) T sets) r)
        v v'.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* The type of the parameter, rewritten under the metadata of the node of
   the code dep, holds of the value of the parameter at a state that the
   run of the plain rebuild comes to. *)
Definition param_type_end_to_end_with (local : bool) (fx : fixes) : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (id : code_id) (i : nat) (p : variable)
    (r : node) (type_var : variable) (v : value),
    used_premises local flag is_local u e T sets ->
    params_aliased (traverse_unit flag u e) (unit_code_deps flag u e) e
    = true ->
    param_premises flag u e T n s id i p r tenv canon valuation valuation'
      type_var v ->
    ∃ (m : nat) (t : state) (v' : value),
      run m
        (initial_state
           (rebuild_unit_from_tables (traverse_unit flag u e) T
              preserve_direct_calls e))
      = Step t
      ∧ lookup p (state_vars t) = Some v'
      ∧ type_satisfied (state_statics t) valuation' v'
          (rewrite_result_type_at fx
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon r type_var).

Definition param_type_end_to_end_statement : Prop :=
  param_type_end_to_end_with false all_fixes.

Definition param_type_end_to_end_landed_statement : Prop :=
  param_type_end_to_end_with false landed_fixes.

Definition param_type_end_to_end_local_statement : Prop :=
  param_type_end_to_end_with true all_fixes.

Definition param_type_end_to_end_local_landed_statement : Prop :=
  param_type_end_to_end_with true landed_fixes.

(* The same of the rebuild with changes of calling convention, for a
   table [cannot_change] that passes its checks. *)
Definition cc_param_type_end_to_end_with (local : bool) (fx : fixes)
    : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (id : code_id) (i : nat) (p : variable)
    (r : node) (type_var : variable) (v : value),
    used_premises local flag is_local u e T sets ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    params_aliased (traverse_unit flag u e) (unit_code_deps flag u e) e
    = true ->
    param_premises flag u e T n s id i p r tenv canon valuation valuation'
      type_var v ->
    ∃ (m : nat) (t : state) (v' : value),
      run m
        (initial_state
           (rebuild_unit_cc_from_tables (traverse_unit flag u e) T
              cannot_change (unit_code_deps flag u e) preserve_direct_calls
              e))
      = Step t
      ∧ lookup p (state_vars t) = Some v'
      ∧ type_satisfied (state_statics t) valuation' v'
          (rewrite_result_type_at fx
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon r type_var).

Definition cc_param_type_end_to_end_statement : Prop :=
  cc_param_type_end_to_end_with false all_fixes.

Definition cc_param_type_end_to_end_landed_statement : Prop :=
  cc_param_type_end_to_end_with false landed_fixes.

Definition cc_param_type_end_to_end_local_statement : Prop :=
  cc_param_type_end_to_end_with true all_fixes.

Definition cc_param_type_end_to_end_local_landed_statement : Prop :=
  cc_param_type_end_to_end_with true landed_fixes.

(* ======================================================================== *)
(* An example                                                               *)
(* ======================================================================== *)

(* The unit of FlambdaReaperUnboxingRefinesSpec.CallExample
   (FlambdaReaperTypesCallsSpec.UnboxCallsExample), whose code 50 takes the
   block [x40], of which it reads the field 1:

     code 50 (x40) =                  (my_closure 41, return 3)
       let x43 = x40.(1) in
       let x42 = [x43; 9] in
       return x42
     let x60 = closure 70 of the code 50 in
     let_cont k20 (x30) = let x31 = x30.(0) in return x31 in
     let x10 = [7; 8] in
     apply x60 x10 -> k20

   The node of its parameter is the made node 6, whose only source is
   [x10]. The result types give the parameter, the variable 90 of the
   result types, the type [UnboxCallsExample.param_type] of a block of
   tag 0 whose field 1 is 8. *)
Module ParamExample.

Local Open Scope positive_scope.

(* What the plain rebuild binds [x40] to: the field 0 of [x10], which no
   one reads, is poison. *)
Definition rebuilt_param : value :=
  Value_block 0 [Value_const (Poison Value); Value_const (Tagged_immediate 8)].

End ParamExample.

(* The unit meets the premises of [param_type_end_to_end_statement] with
   the tables of the least solution, no local field and the set of
   closures of [x60]: it passes [params_aliased], the parameter of rank 0
   of the code 50 is [x40], whose node is the made node 6, and the run of
   the unit comes, in 5 steps, to the [Let] of [x43] in the code, where
   [x40] is bound to [7; 8], the variable 90 of the result types at
   [7; 8]. The type of the parameter, rewritten under the metadata of the
   node, is the type itself; the run of the plain rebuild comes, in 5
   steps, to a state where [x40] is bound to [rebuilt_param], of which it
   holds. *)
Definition param_example_with (fx : fixes) : Prop :=
  used_premises false Auto FlambdaReaperUnboxing.Examples.no_local
    FlambdaReaperUnboxing.Examples.interface UnboxCallsExample.unit_expr
    UnboxCallsExample.tables UnboxCallsExample.sets
  ∧ params_aliased UnboxCallsExample.facts
      (unit_code_deps Auto FlambdaReaperUnboxing.Examples.interface
         UnboxCallsExample.unit_expr)
      UnboxCallsExample.unit_expr
    = true
  ∧ param_node Auto FlambdaReaperUnboxing.Examples.interface
      UnboxCallsExample.unit_expr 50%positive 0
    = Some (Node_made 6)
  ∧ (∃ s : state,
       param_premises Auto FlambdaReaperUnboxing.Examples.interface
         UnboxCallsExample.unit_expr UnboxCallsExample.tables 5 s 50%positive
         0 40%positive (Node_made 6) UnboxCallsExample.tenv
         UnboxCallsExample.canon UnboxCallsExample.valuation
         UnboxCallsExample.valuation 90%positive UnboxCallsExample.pair)
  ∧ rewrite_result_type_at fx
      (plain_context FlambdaReaperUnboxing.Examples.no_local
         UnboxCallsExample.facts UnboxCallsExample.tables
         UnboxCallsExample.sets)
      UnboxCallsExample.tenv UnboxCallsExample.canon (Node_made 6)
      90%positive
    = UnboxCallsExample.param_type
  ∧ ∃ t : state,
      run 5
        (initial_state
           (rebuild_unit_from_tables UnboxCallsExample.facts
              UnboxCallsExample.tables true UnboxCallsExample.unit_expr))
      = Step t
      ∧ lookup 40%positive (state_vars t) = Some ParamExample.rebuilt_param
      ∧ type_satisfied (state_statics t) UnboxCallsExample.valuation
          ParamExample.rebuilt_param UnboxCallsExample.param_type.

Definition param_example_statement : Prop :=
  param_example_with all_fixes.

Definition param_example_landed_statement : Prop :=
  param_example_with landed_fixes.
