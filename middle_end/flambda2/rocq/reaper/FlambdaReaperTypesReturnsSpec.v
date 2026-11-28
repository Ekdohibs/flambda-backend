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

(* The result types of code at the results of the code, end to end: the
   type that the result types give a result of code, rewritten as the
   compiler rewrites it, holds of the value that the rebuilt code returns.

   This file is to be reviewed: [result_node] and
   [rewrite_result_type_at] (the correspondence with the compiler), the
   premises ([at_return], [return_premises]) and the statements, against
   FlambdaReaperTypes.v, FlambdaReaperTypesCallsSpec.v (whose premises
   these take) and FlambdaReaperDeadFramesSpec.applies_to (the returns of
   code). The proofs are in FlambdaReaperTypesReturnsProofs.v, which needs
   no review.

   The nodes of the results. The compiler gives [rewrite_result_types]
   the variables of the code dep of the code
   (unboxing_analysis.ml: [code_dep.params] and [code_dep.return]), and
   [Types_rewriter.rewrite_result_types] reads the metadata of each
   ([variable_pattern], [metadata_for_name context var]). The variables
   of [code_dep.return] are made by [Traverse.prepare_code]
   ([Variable.create "function_return_..."]), one for each result of the
   code; the model's traversal makes a node for each
   (FlambdaReaperTraversal.prepare_code, [cd_return]), as it makes a
   [Node_made] wherever the compiler makes a variable. So the node of the
   result of rank [i] is the node of rank [i] of [cd_return] of the code
   dep of the code ([result_node]), and its type is rewritten under the
   metadata of that node ([rewrite_result_type_at]):
   FlambdaReaperTypes.rewrite_result_type is the same at the node of a
   variable ([rewrite_result_type fx ctx tenv canon x type_var] is
   [rewrite_result_type_at fx ctx tenv canon (Node_variable x) type_var]
   by definition), and the statements of FlambdaReaperTypesCallsSpec.v
   are of the variables that the run binds.

   The returns of code. The code returns where its body applies its
   return continuation, not under a binding of it
   (FlambdaReaperDeadFramesSpec.applies_to, the arms of a switch kept by
   the rebuilds among them): the traversal aliases the node of each
   result to the argument at its rank there
   (FlambdaReaperTraversal.apply_cont_deps, under the continuations of
   the body of code, [code_env]). A tail call returns what its callee
   returns, at the returns of the callee.

   The statements, each with [_landed] for the fixes to land
   (FlambdaReaperTypes.landed_fixes) and [_local] for a least solution of
   the rules with the fix of local fields
   (FlambdaReaperTypesCallsSpec.least_with):
   - [code_result_type_end_to_end_statement]: at each state that the run
     of the unit comes to and from which it steps, at a return of the
     code [id] whose argument of rank [i] is [arg] and has the value [v],
     the type that the result types give [type_var], which holds of [v],
     rewritten under the metadata of the node [r] of the result of rank
     [i] of [id], holds of the value of [arg] at a state that the run of
     the plain rebuild comes to: the state at the rebuild of that return,
     where [arg], which has a use, is kept.
   - [cc_code_result_type_end_to_end_statement]: the same of the rebuild
     with changes of calling convention; the result has a use, so the
     change keeps it.
   - [code_result_example_statement]: the unit of
     FlambdaReaperUnboxingRefinesSpec.CallExample, whose code 50 returns
     the block [x42] of the field [x43] and of 9, of which the caller
     reads the field 0. The node of its result is the made node 3; the
     result types give it the type of a block of tag 0 whose fields are 8
     and 9, which holds of [8; 9]. The type rewritten under the metadata
     of the node keeps the field 0 and forgets the field 1, which the
     plain rebuild makes poison: the run of the rebuilt unit binds [x42]
     to the block of 8 and of poison there, of which the type holds.

   Of the premises: those of FlambdaReaperTypesCallsSpec.v
   ([unit_premises], [check_cc_closed] and [check_apply_widgets] for the
   rebuild with changes of calling convention), and that the result has
   a use: the rebuild makes poison of an argument with no use, and the
   change of calling conventions deletes a result with no use, whose
   type the compiler drops ([Delete] in [rewrite_result_types]).

   The parameters of code at the nodes of [code_dep.params] are in
   FlambdaReaperTypesParamsSpec.v, and the statements here with the wider
   check of static blocks of FlambdaReaperTypesUsedValuesSpec.v
   ([used_values] in place of [unit_of_values]) in
   FlambdaReaperTypesReturnsUsedSpec.v.

   What is left out:
   - The unboxing across calls: its simulation relates the states within
     a wrapper of both runs, about to apply a continuation, with no
     environment (FlambdaReaperTypesCallsSpec.v), and a return of code
     is such an [Apply_cont]. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaPrimitiveSpec FlambdaSemanticsSpec
  FlambdaWellFormedSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperRebuild FlambdaReaperFlowSpec
  FlambdaReaperLeastSpec FlambdaReaperLocalFields FlambdaReaperTypes
  FlambdaReaperTypesSpec FlambdaReaperTypesTermSpec
  FlambdaReaperCallingConventions FlambdaReaperTypesCallsSpec.
From OxCamlFlambda2 Require FlambdaReaperRebuildSpec FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperCallingConventionsSpec
  FlambdaReaperDeadFramesSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaTypingEnv
  FlambdaTypingEnvSpec.
Import ListNotations.

(* ======================================================================== *)
(* The results of code                                                      *)
(* ======================================================================== *)

(* The node of the result of rank [i] of the code [id]: the node of that
   rank of [cd_return] of its code dep, which stands for the variable of
   [code_dep.return] that the compiler makes. *)
Definition result_node (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) (id : code_id) (i : nat) : option node :=
  match lookup id (unit_code_deps flag u e) with
  | Some cd => nth_error (cd_return cd) i
  | None => None
  end.

(* FlambdaReaperTypes.rewrite_result_type at the node [r] of a result of
   code: the type that the result types give [type_var], rewritten under
   the metadata of [r] ([variable_pattern] of [Keep]). *)
Definition rewrite_result_type_at (fx : fixes) (ctx : rewrite_context)
    (tenv : Typing_env.t) (canon : canon_fn) (r : node) (type_var : variable)
    : flambda_type :=
  rewrite_concrete_type_of fx ctx tenv canon (Name_var type_var)
    (kind_of_name tenv (Name_var type_var)) (metadata_for_name ctx r).

(* The state [s] is at a return of the code [id] that the unit binds: at
   an [Apply_cont] of its body to its return continuation, not under a
   binding of it, whose argument of rank [i] is [arg]. *)
Definition at_return (e : expr) (s : state) (id : code_id) (i : nat)
    (arg : simple) : Prop :=
  ∃ (c : code) (ac : apply_cont),
    (id, c) ∈ unit_codes e
    ∧ state_expr s = Apply_cont ac
    ∧ ac
      ∈ FlambdaReaperDeadFramesSpec.applies_to (code_return_continuation c)
          (code_body c)
    ∧ nth_error (ac_args ac) i = Some arg.

(* The premises on a state [s] of the run of the unit at a return of the
   code [id]: the run comes to [s] and steps from it, the argument [arg]
   of rank [i] has the value [v] there, the node [r] of the result of
   rank [i] has a use, and a valuation of the names of the result types
   [tenv] (whose types are of the fragment of [loose]) that gives
   [type_var] the value [v] satisfies them, with the copies of the
   rewriting valued as [copies_valued] says. *)
Definition return_premises (loose : bool) (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) (T : tables) (n : nat) (s : state)
    (id : code_id) (i : nat) (arg : simple) (r : node) (tenv : Typing_env.t)
    (canon : canon_fn) (valuation valuation' : env) (type_var : variable)
    (v : value) : Prop :=
  run n (initial_state e) = Step s
  ∧ step s <> Undefined
  ∧ at_return e s id i arg
  ∧ eval_simple (state_vars s) arg = Some v
  ∧ result_node flag u e id i = Some r
  ∧ s_has_usage (solution_of_tables T) r = true
  ∧ typing_env_satisfied_strictly (state_statics s) valuation tenv
  ∧ env_in_fragment_with loose valuation tenv
  ∧ copies_valued tenv canon valuation valuation'
  ∧ Typing_env.find tenv (Name_var type_var) None <> None
  ∧ lookup type_var valuation = Some v.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* The type of the result, rewritten under the metadata of its node, holds
   of the value of [arg] at a state that the run of the plain rebuild
   comes to. *)
Definition code_result_type_end_to_end_with (local : bool) (fx : fixes)
    : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (id : code_id) (i : nat) (arg : simple) (r : node)
    (type_var : variable) (v : value),
    unit_premises local flag is_local u e T sets ->
    return_premises false flag u e T n s id i arg r tenv canon valuation
      valuation' type_var v ->
    ∃ (m : nat) (t : state) (v' : value),
      run m
        (initial_state
           (rebuild_unit_from_tables (traverse_unit flag u e) T
              preserve_direct_calls e))
      = Step t
      ∧ eval_simple (state_vars t) arg = Some v'
      ∧ type_satisfied (state_statics t) valuation' v'
          (rewrite_result_type_at fx
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon r type_var).

Definition code_result_type_end_to_end_statement : Prop :=
  code_result_type_end_to_end_with false all_fixes.

Definition code_result_type_end_to_end_landed_statement : Prop :=
  code_result_type_end_to_end_with false landed_fixes.

Definition code_result_type_end_to_end_local_statement : Prop :=
  code_result_type_end_to_end_with true all_fixes.

Definition code_result_type_end_to_end_local_landed_statement : Prop :=
  code_result_type_end_to_end_with true landed_fixes.

(* The same of the rebuild with changes of calling convention, for a
   table [cannot_change] that passes its checks. *)
Definition cc_code_result_type_end_to_end_with (local : bool) (fx : fixes)
    : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (id : code_id) (i : nat) (arg : simple) (r : node)
    (type_var : variable) (v : value),
    unit_premises local flag is_local u e T sets ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    return_premises false flag u e T n s id i arg r tenv canon valuation
      valuation' type_var v ->
    ∃ (m : nat) (t : state) (v' : value),
      run m
        (initial_state
           (rebuild_unit_cc_from_tables (traverse_unit flag u e) T
              cannot_change (unit_code_deps flag u e) preserve_direct_calls
              e))
      = Step t
      ∧ eval_simple (state_vars t) arg = Some v'
      ∧ type_satisfied (state_statics t) valuation' v'
          (rewrite_result_type_at fx
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon r type_var).

Definition cc_code_result_type_end_to_end_statement : Prop :=
  cc_code_result_type_end_to_end_with false all_fixes.

Definition cc_code_result_type_end_to_end_landed_statement : Prop :=
  cc_code_result_type_end_to_end_with false landed_fixes.

Definition cc_code_result_type_end_to_end_local_statement : Prop :=
  cc_code_result_type_end_to_end_with true all_fixes.

Definition cc_code_result_type_end_to_end_local_landed_statement : Prop :=
  cc_code_result_type_end_to_end_with true landed_fixes.

(* ======================================================================== *)
(* An example                                                               *)
(* ======================================================================== *)

(* The unit of FlambdaReaperUnboxingRefinesSpec.CallExample
   (FlambdaReaperTypesCallsSpec.UnboxCallsExample), whose code 50 returns
   the block [x42] of [x43] and 9, of which the caller reads the field 0:

     code 50 (x40) =                  (my_closure 41, return 3)
       let x43 = x40.(1) in
       let x42 = [x43; 9] in
       return x42

   The node of its result is the made node 3, whose only source is
   [x42]. The result types give the result, the variable 91 of the result
   types, the type of a block of tag 0 whose fields are 8 and 9. *)
Module CodeResultExample.

Local Open Scope positive_scope.

Definition result_type : flambda_type :=
  immutable_block false 0%Z (Scannable Value_only) Alloc_mode.Heap
    [this_tagged_immediate 8%Z; this_tagged_immediate 9%Z].

(* The result types: the variable 91 of the type [result_type]. *)
Definition tenv : Typing_env.t :=
  Typing_env.replace_equation
    (Typing_env.add_variable_definition Typing_env.create 91 Value
       Name_mode.normal)
    (Name_var 91) result_type.

(* What the code returns in the run of the unit. *)
Definition result : value :=
  Value_block 0 [Value_const (Tagged_immediate 8);
                 Value_const (Tagged_immediate 9)].

Definition valuation : env := [(91, result)].

(* The type rewritten under the metadata of the made node 3: the field 0,
   which the caller reads, is kept, and the field 1 forgotten. *)
Definition rewritten : flambda_type :=
  immutable_block false 0%Z (Scannable Value_only) Alloc_mode.Heap
    [this_tagged_immediate 8%Z; unknown Value].

(* What the rebuilt code returns: the field 1 is poison. *)
Definition rebuilt_result : value :=
  Value_block 0 [Value_const (Tagged_immediate 8); Value_const (Poison Value)].

End CodeResultExample.

(* The unit meets the premises of [code_result_type_end_to_end_statement]
   with the tables of the least solution, no local field and the set of
   closures of [x60]: the result of rank 0 of the code 50 is the made node
   3, and the run of the unit comes, in 7 steps, to the return of [x42],
   bound to [8; 9], with the variable 91 of the result types at [8; 9].
   The type of the result, rewritten under the metadata of the node, is
   [rewritten]; the run of the plain rebuild comes, in 7 steps, to a state
   where [x42] is bound to [rebuilt_result], of which it holds. *)
Definition code_result_example_with (fx : fixes) : Prop :=
  unit_premises false Auto FlambdaReaperUnboxing.Examples.no_local
    FlambdaReaperUnboxing.Examples.interface UnboxCallsExample.unit_expr
    UnboxCallsExample.tables UnboxCallsExample.sets
  ∧ result_node Auto FlambdaReaperUnboxing.Examples.interface
      UnboxCallsExample.unit_expr 50%positive 0
    = Some (Node_made 3)
  ∧ (∃ s : state,
       return_premises false Auto FlambdaReaperUnboxing.Examples.interface
         UnboxCallsExample.unit_expr UnboxCallsExample.tables 7 s 50%positive
         0 (Var 42%positive) (Node_made 3) CodeResultExample.tenv
         UnboxCallsExample.canon CodeResultExample.valuation
         CodeResultExample.valuation 91%positive CodeResultExample.result)
  ∧ rewrite_result_type_at fx
      (plain_context FlambdaReaperUnboxing.Examples.no_local
         UnboxCallsExample.facts UnboxCallsExample.tables
         UnboxCallsExample.sets)
      CodeResultExample.tenv UnboxCallsExample.canon (Node_made 3)
      91%positive
    = CodeResultExample.rewritten
  ∧ ∃ t : state,
      run 7
        (initial_state
           (rebuild_unit_from_tables UnboxCallsExample.facts
              UnboxCallsExample.tables true UnboxCallsExample.unit_expr))
      = Step t
      ∧ eval_simple (state_vars t) (Var 42%positive)
        = Some CodeResultExample.rebuilt_result
      ∧ type_satisfied (state_statics t) CodeResultExample.valuation
          CodeResultExample.rebuilt_result CodeResultExample.rewritten.

Definition code_result_example_statement : Prop :=
  code_result_example_with all_fixes.

Definition code_result_example_landed_statement : Prop :=
  code_result_example_with landed_fixes.
