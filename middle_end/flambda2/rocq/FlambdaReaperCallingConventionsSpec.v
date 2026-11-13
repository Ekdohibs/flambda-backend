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

(* Which code may change its calling convention, what the check of the
   table that says so means, and the rebuild when no code changes.

   This file is to be reviewed: [cc_closed] against the rules
   [calling_convention_changes_rules] of
   middle_end/flambda2/reaper/unboxing_analysis.ml, and the statements.
   The check, the table and the rebuild are
   FlambdaReaperCallingConventions.v, which needs no review; the proofs
   are FlambdaReaperCallingConventionsProofs.v.

   The reaper changes the calling convention of code of the unit (it
   removes its unused parameters and results) unless
   [cannot_change_calling_convention] holds of its code id. That relation
   is derived by eight rules from the solution of points_to_analysis.ml
   (FlambdaReaperGraphSpec.solution), through a second relation,
   [cannot_change_witness_calling_convention], of the call witnesses
   (the allocations of [Field.code_id_of_call_witness]). A witness
   cannot change if it escapes ([cc_witness_any_usage]); if a call that
   uses it and reads the code id of what it calls may also find another
   witness there, or any ([cc_witness_other_source],
   [cc_witness_any_source_usage]); if it may come from another
   allocation, or from any ([cc_witness_other_own_source],
   [cc_witness_any_source]); or if it is used and is not its own source
   ([cc_witness_not_own_source]). The code of a used witness that cannot
   change cannot change ([cc_code_id_of_witness]), nor can code whose
   witness of unknown arity is used ([cc_code_id_unknown_arity]).
   [cc_closed] says that two tables are closed under the eight rules,
   each a field named [cc_] and the rule. What the rebuild is to rely on
   is that code not in the table is called only by calls that find its
   witness and no other.

   The table of [cannot_change_calling_convention] is read off the
   compiler, and nothing is believed of it: [check_cc_closed] tests it,
   and a table that passes is, with a table of the witnesses, closed
   under the rules ([check_cc_closed_sound_statement]). The table of the
   witnesses is not an input: no rule concludes it from itself, and the
   check computes it from the solution. [derive_cannot_change] is the
   least table: it passes the check, and is in every table that is
   closed under the rules ([derive_cannot_change_least_statement]).
   Nothing is said of a table that fails. The examples of
   FlambdaReaperCallingConventions.v have a table that the check
   rejects.

   [rebuild_unit_cc], the rebuild with changes of calling convention, is
   [FlambdaReaperRebuild.rebuild_unit] when every code of the unit is in
   the table ([rebuild_unit_cc_unchanged_statement]): what is proved of
   the one is to be carried to the other, for the code that does not
   change.

   Where code changes, the rebuilt unit runs as the unit does on three
   units, each of which meets the premises of
   FlambdaReaperRebuildSpec.rebuild_refines_observably_statement, with
   its least table, which passes [check_cc_closed]: the unit of
   FlambdaReaperCallingConventions.v, whose code loses a parameter and a
   result, called once without a wrapper and once through one
   ([two_calls_example_statement]); a unit in which code that cannot
   change, called at unknown arity, calls in tail position code that
   changes and no longer returns its second result, so that the wrapper
   of the call gives the return continuation poison in its place, which
   the continuation of the call of unknown arity does not use
   ([poison_result_example_statement]); and the same unit with both codes
   changing, the call in tail position returning to the return
   continuation of the code it is in, which takes the results that both
   keep ([masked_return_example_statement]). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions.
Import ListNotations.

(* ======================================================================== *)
(* The rules                                                                *)
(* ======================================================================== *)

Section Rules.

Variable G : graph.
Variable S : solution.
(* [cannot_change_witness_calling_convention]. *)
Variable cannot_change_witness : node -> bool.
(* [cannot_change_calling_convention]. *)
Variable cannot_change : node -> bool.

Abbreviation any_usage := (s_any_usage S).
Abbreviation any_source := (s_any_source S).
Abbreviation has_usage := (s_has_usage S).
Abbreviation has_source := (s_has_source S).
Abbreviation usages := (s_usages S).
Abbreviation sources := (s_sources S).

(* The rules of [calling_convention_changes_rules], in their order there,
   each a field named [cc_] and the rule; each says that the tables have
   the conclusion of the rule wherever they have the premises. *)
Record cc_closed : Prop := {
  cc_witness_any_usage : ∀ x,
    any_usage x = true -> cannot_change_witness x = true;
  cc_witness_other_source : ∀ allocation_id alias alias_source v,
    usages allocation_id alias = true ->
    sources alias alias_source = true -> has_source alias_source = true ->
    alias_source <> allocation_id ->
    g_accessor G v Field_code_id_of_call_witness alias ->
    has_usage v = true ->
    cannot_change_witness allocation_id = true;
  cc_witness_any_source_usage : ∀ allocation_id alias v,
    usages allocation_id alias = true -> any_source alias = true ->
    g_accessor G v Field_code_id_of_call_witness alias ->
    has_usage v = true ->
    cannot_change_witness allocation_id = true;
  cc_witness_other_own_source : ∀ allocation_id source,
    sources allocation_id source = true -> has_source source = true ->
    source <> allocation_id ->
    cannot_change_witness allocation_id = true;
  cc_witness_not_own_source : ∀ allocation_id usage,
    usages allocation_id usage = true -> has_usage usage = true ->
    sources allocation_id allocation_id = false ->
    cannot_change_witness allocation_id = true;
  cc_witness_any_source : ∀ allocation_id,
    any_source allocation_id = true ->
    cannot_change_witness allocation_id = true;
  cc_code_id_of_witness : ∀ call_witness code_id,
    g_constructor G call_witness Field_code_id_of_call_witness code_id ->
    has_usage call_witness = true ->
    cannot_change_witness call_witness = true ->
    cannot_change code_id = true;
  cc_code_id_unknown_arity : ∀ set_of_closures call_witness code_id,
    g_constructor G set_of_closures
      (Field_call_witness Unknown_arity_code_pointer) call_witness ->
    has_usage call_witness = true ->
    g_constructor G call_witness Field_code_id_of_call_witness code_id ->
    cannot_change code_id = true
}.

End Rules.

(* ======================================================================== *)
(* The check                                                                *)
(* ======================================================================== *)

Definition check_cc_closed_sound_statement : Prop :=
  ∀ (F : facts) (T : tables) (cannot_change : list node),
    check_cc_closed F T cannot_change = true ->
    ∃ cannot_change_witness : node -> bool,
      cc_closed (graph_of_facts F) (solution_of_tables T)
        cannot_change_witness (fun n => mem_node n cannot_change).

Definition derive_cannot_change_least_statement : Prop :=
  ∀ (F : facts) (T : tables),
    check_cc_closed F T (derive_cannot_change F T) = true
    ∧ ∀ cannot_change_witness cannot_change : node -> bool,
        cc_closed (graph_of_facts F) (solution_of_tables T)
          cannot_change_witness cannot_change ->
        ∀ n : node,
          mem_node n (derive_cannot_change F T) = true ->
          cannot_change n = true.

(* ======================================================================== *)
(* The rebuild                                                              *)
(* ======================================================================== *)

Definition rebuild_unit_cc_unchanged_statement : Prop :=
  ∀ (F : facts) (sol : solution) (cannot_change : node -> bool)
    (code_deps : finite_map code_dep) (preserve_direct_calls : bool)
    (e : expr),
    (∀ id : code_id,
       among id (static_names code_ids_bound e) = true ->
       cannot_change (Node_code_id id) = true) ->
    rebuild_unit_cc F sol cannot_change code_deps preserve_direct_calls e
    = rebuild_unit F sol preserve_direct_calls e.

(* ======================================================================== *)
(* Runs of units whose code changes                                         *)
(* ======================================================================== *)

(* The unit [two_calls] of FlambdaReaperCallingConventions.v meets the
   premises of FlambdaReaperRebuildSpec.rebuild_refines_observably_statement;
   no code is in its least table, and the code 50 loses its second
   parameter and its third result. The rebuilt unit exits as the unit
   does, with the block of the first two results of the second call. *)
Definition two_calls_example_statement : Prop :=
  in_flow_fragment Examples.two_calls = true
  ∧ static_constants (unit_codes Examples.two_calls) Examples.two_calls = true
  ∧ results_kinded [] [] Examples.two_calls = true
  ∧ well_formed_unit Examples.interface Examples.two_calls = true
  ∧ exits_one_value Examples.interface Examples.two_calls = true
  ∧ check_closed Examples.no_local Examples.two_calls_facts
      Examples.two_calls_tables
    = true
  ∧ check_field_used_exact Examples.no_local Examples.two_calls_facts
      Examples.two_calls_tables
    = true
  ∧ derive_cannot_change Examples.two_calls_facts Examples.two_calls_tables
    = []
  ∧ check_cc_closed Examples.two_calls_facts Examples.two_calls_tables []
    = true
  ∧ unit_calling_convention_changes Examples.two_calls_tables []
      (unit_code_deps Auto Examples.interface Examples.two_calls)
      Examples.two_calls
    = [(50%positive,
        Changing_calling_convention [Keep Value; Delete]
          [Keep Value; Keep Value; Delete])]
  ∧ behaves (State Examples.two_calls [] [] empty_static_env empty_heap)
      (Exits 1%positive
         [Value_block 0
            [Value_const (Tagged_immediate 9);
             Value_const (Tagged_immediate 9)]]
         empty_heap)
  ∧ behaves
      (State
         (rebuild_unit_cc_from_tables Examples.two_calls_facts
            Examples.two_calls_tables []
            (unit_code_deps Auto Examples.interface Examples.two_calls) true
            Examples.two_calls)
         [] [] empty_static_env empty_heap)
      (Exits 1%positive
         [Value_block 0
            [Value_const (Tagged_immediate 9);
             Value_const (Tagged_immediate 9)]]
         empty_heap).

(* Code that cannot change calls, in tail position, code that changes and
   no longer returns its second result:

     let code f51 x40 x41 my_closure x45 return k3 exn k4 = k3 x40 x41 in
     let code f52 x42 my_closure x46 return k5 exn k6 =
       let x47 = x46 moved from 71 to 70 in
       x47 x42 7 return k5 exn k6            (direct, to f51)
     in
     let x11 = closure at 71 of { 70 = f51; 71 = f52 } in
     let_cont k20 x30 x31 = return x30 in   ([x31] has no use)
     x11 5 return k20 exn k2                (of unknown arity)

   The call of unknown arity uses the witness of unknown arity of [f52],
   which so cannot change ([cc_code_id_unknown_arity]); [f51] is called
   by the direct call only, and changes: its parameter [x41] has no use,
   nor its second result, which [f52] returns to the call of unknown
   arity, whose continuation does not use it. The rebuild gives the
   direct call the first argument only, and has it return to a wrapper,
   the continuation 48, which applies [k5] to the result and to poison in
   place of the second ([make_apply_wrapper_cc]); the call of unknown
   arity returns to a wrapper too, which drops the poison
   ([make_apply_wrapper]). Its least solution, as the solver of the test
   tool computes it, passes the checks. *)
Module PoisonResultExample.

Local Open Scope positive_scope.
Import FlambdaReaperCallingConventions.Examples.

Definition code_51 : code :=
  Code [40; 41] 45 3 4
    (make_code_metadata [Singleton Value; Singleton Value]
       [Singleton Value; Singleton Value])
    (return_to 3 [Var 40; Var 41]).

Definition call_51 : apply_expr :=
  {| ap_callee := Var 47; ap_call_kind := Direct 51;
     ap_args := [Var 42; Const (Tagged_immediate 7)];
     ap_continuation := 5; ap_exn_continuation := 6; ap_exn_extra_args := [];
     ap_args_arity := [Singleton Value; Singleton Value];
     ap_return_arity := [Singleton Value; Singleton Value] |}.

Definition code_52 : code :=
  Code [42] 46 5 6
    (make_code_metadata [Singleton Value] [Singleton Value; Singleton Value])
    (Let 47 Value (Project_function_slot 71 70 (Var 46)) (Apply call_51)).

Definition call_52 : apply_expr :=
  {| ap_callee := Var 11; ap_call_kind := Indirect;
     ap_args := [Const (Tagged_immediate 5)];
     ap_continuation := 20; ap_exn_continuation := 2; ap_exn_extra_args := [];
     ap_args_arity := [Singleton Value];
     ap_return_arity := [Singleton Value; Singleton Value] |}.

Definition unit_expr : expr :=
  Let_static [Bind_code 51 code_51; Bind_code 52 code_52]
    (Let 11 Value (Closure_of_set 71 [(70, 51); (71, 52)] [])
       (Let_cont_non_recursive 20
          (Handler [30; 31] (make_handler_metadata [Value; Value])
             (return_to 1 [Var 30]))
          (Apply call_52))).

Definition unit_facts : facts := traverse_unit Auto interface unit_expr.

Definition solution : tables :=
  {| t_alias :=
       [(m 27, m 28); (m 1, v 30); (v 46, m 16); (v 42, m 17); (m 3, v 40);
        (m 4, v 41); (v 45, m 6); (v 40, m 7); (v 41, m 8); (m 28, m 19);
        (m 2, m 15); (v 30, m 13); (v 31, m 14); (m 26, c 52); (m 17, consts);
        (m 16, v 11); (m 24, m 9); (m 6, m 23); (m 20, m 22); (v 47, m 25);
        (m 22, m 9); (m 13, m 3); (m 14, m 4); (m 15, m 5); (m 21, c 51);
        (m 7, v 42); (m 8, consts); (m 6, m 25)];
     t_any_usage :=
       [m 1; m 26; m 21; m 2; v 30; m 15; m 13; c 52; m 3; m 5; c 51; v 40; m 7;
        v 42; m 17; consts];
     t_any_source :=
       [c 52; c 51; consts; m 26; m 17; v 42; m 21; m 7; m 8; v 40; v 41; m 3;
        m 4; m 13; m 14; v 30; v 31; m 1];
     t_has_usage :=
       [m 1; m 26; m 21; m 2; v 30; m 27; m 20; m 28; m 19; m 15; m 13; c 52;
        m 22; v 11; v 47; v 46; m 16; m 25; m 9; m 3; m 5; c 51; v 40; m 7;
        v 42; m 17; consts];
     t_has_source :=
       [c 52; c 51; consts; v 11; m 25; m 19; m 18; m 11; m 12; m 10; m 9; m 28;
        m 26; m 17; m 27; v 42; m 16; m 24; v 46; v 47; m 22; m 21; m 7; m 8;
        v 40; v 41; m 20; m 6; m 3; m 4; v 45; m 13; m 14; v 30; v 31; m 1];
     t_usages :=
       [(v 11, v 11); (m 27, m 27); (v 47, v 47); (m 20, m 20); (v 46, v 46);
        (m 28, m 27); (m 16, v 46); (m 19, m 27); (v 11, v 46); (m 22, m 20);
        (m 25, v 47); (m 9, m 20)];
     t_sources :=
       [(m 25, m 25); (v 11, v 11); (m 19, m 19); (m 18, m 18); (m 11, m 11);
        (m 10, m 10); (m 12, m 12); (m 9, m 9); (m 28, m 19); (m 27, m 19);
        (m 16, v 11); (m 24, m 9); (v 46, v 11); (v 47, m 25); (m 22, m 9);
        (m 20, m 9); (m 6, m 25); (v 45, m 25)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used :=
       [(m 25, known); (v 11, unknown); (v 11, Field_function_slot 70);
        (m 19, Field_return_of_call Exn); (m 19, result 0);
        (m 19, Field_code_id_of_call_witness); (m 9, result 0);
        (m 9, Field_return_of_call Exn);
        (m 9, Field_code_id_of_call_witness)] |}.

(* The code 51 rebuilt: one parameter, one result. *)
Definition code_51_changed : code :=
  Code [40] 45 3 4
    {| cm_params_arity :=
         [Unboxed_product [Singleton Value]; Unboxed_product []];
       cm_result_arity := [Unboxed_product [Singleton Value]];
       cm_is_tupled := false;
       cm_is_stub := false |}
    (return_to 3 [Var 40]).

Definition call_51_changed (k : continuation) : apply_expr :=
  {| ap_callee := Var 47; ap_call_kind := Direct 51; ap_args := [Var 42];
     ap_continuation := k; ap_exn_continuation := 6; ap_exn_extra_args := [];
     ap_args_arity := [Unboxed_product [Singleton Value]; Unboxed_product []];
     ap_return_arity := [Unboxed_product [Singleton Value]] |}.

Definition rebuilt : expr :=
  Let_static
    [Bind_code 51 code_51_changed;
     Bind_code 52
       (Code [42] 46 5 6
          (make_code_metadata [Singleton Value]
             [Singleton Value; Singleton Value])
          (Let 47 Value (Project_function_slot 71 70 (Var 46))
             (Let_cont_non_recursive 48
                (Handler [48] (make_handler_metadata [Value])
                   (return_to 5 [Var 48; Const (Poison Value)]))
                (Apply (call_51_changed 48)))))]
    (Let 11 Value (Closure_of_set 71 [(70, 51); (71, 52)] [])
       (Let_cont_non_recursive 20
          (Handler [30] (make_handler_metadata [Value])
             (return_to 1 [Var 30]))
          (Let_cont_non_recursive 48
             (Handler [48; 49] (make_handler_metadata [Value; Value])
                (return_to 20 [Var 48]))
             (Apply (with_continuation call_52 48))))).

End PoisonResultExample.

Definition poison_result_example_statement : Prop :=
  in_flow_fragment PoisonResultExample.unit_expr = true
  ∧ static_constants (unit_codes PoisonResultExample.unit_expr)
      PoisonResultExample.unit_expr
    = true
  ∧ results_kinded [] [] PoisonResultExample.unit_expr = true
  ∧ well_formed_unit Examples.interface PoisonResultExample.unit_expr = true
  ∧ exits_one_value Examples.interface PoisonResultExample.unit_expr = true
  ∧ check_closed Examples.no_local PoisonResultExample.unit_facts
      PoisonResultExample.solution
    = true
  ∧ check_field_used_exact Examples.no_local PoisonResultExample.unit_facts
      PoisonResultExample.solution
    = true
  ∧ derive_cannot_change PoisonResultExample.unit_facts
      PoisonResultExample.solution
    = [Node_code_id 52%positive]
  ∧ check_cc_closed PoisonResultExample.unit_facts
      PoisonResultExample.solution [Node_code_id 52%positive]
    = true
  ∧ unit_calling_convention_changes PoisonResultExample.solution
      [Node_code_id 52%positive]
      (unit_code_deps Auto Examples.interface PoisonResultExample.unit_expr)
      PoisonResultExample.unit_expr
    = [(51%positive,
        Changing_calling_convention [Keep Value; Delete] [Keep Value; Delete]);
       (52%positive, Not_changing_calling_convention)]
  ∧ rebuild_unit_cc_from_tables PoisonResultExample.unit_facts
      PoisonResultExample.solution [Node_code_id 52%positive]
      (unit_code_deps Auto Examples.interface PoisonResultExample.unit_expr)
      true PoisonResultExample.unit_expr
    = PoisonResultExample.rebuilt
  ∧ behaves
      (State PoisonResultExample.unit_expr [] [] empty_static_env empty_heap)
      (Exits 1%positive [Value_const (Tagged_immediate 5)] empty_heap)
  ∧ behaves
      (State PoisonResultExample.rebuilt [] [] empty_static_env empty_heap)
      (Exits 1%positive [Value_const (Tagged_immediate 5)] empty_heap).

(* The unit of [PoisonResultExample] with a second parameter of [f52],
   which has no use, and a direct call of [x11], given 5 and 8: both codes
   change, each losing its second parameter and its second result. In
   the body of [f52], the return continuation [k5] takes the first result
   only ([cont_params_to_keep]), which is what [f51] keeps: the call in
   tail position returns to it directly, without a wrapper, as the call of
   [x11] returns to [k20], which keeps the first result only. *)
Module MaskedReturnExample.

Local Open Scope positive_scope.
Import FlambdaReaperCallingConventions.Examples.

Definition code_52 : code :=
  Code [42; 43] 46 5 6
    (make_code_metadata [Singleton Value; Singleton Value]
       [Singleton Value; Singleton Value])
    (Let 47 Value (Project_function_slot 71 70 (Var 46))
       (Apply PoisonResultExample.call_51)).

Definition call_52 : apply_expr :=
  {| ap_callee := Var 11; ap_call_kind := Direct 52;
     ap_args := [Const (Tagged_immediate 5); Const (Tagged_immediate 8)];
     ap_continuation := 20; ap_exn_continuation := 2; ap_exn_extra_args := [];
     ap_args_arity := [Singleton Value; Singleton Value];
     ap_return_arity := [Singleton Value; Singleton Value] |}.

Definition unit_expr : expr :=
  Let_static [Bind_code 51 PoisonResultExample.code_51; Bind_code 52 code_52]
    (Let 11 Value (Closure_of_set 71 [(70, 51); (71, 52)] [])
       (Let_cont_non_recursive 20
          (Handler [30; 31] (make_handler_metadata [Value; Value])
             (return_to 1 [Var 30]))
          (Apply call_52))).

Definition unit_facts : facts := traverse_unit Auto interface unit_expr.

Definition solution : tables :=
  {| t_alias :=
       [(m 33, m 19); (m 16, m 32); (m 29, m 31); (m 1, v 30); (v 46, m 16);
        (v 42, m 17); (v 43, m 18); (m 3, v 40); (m 4, v 41); (v 45, m 6);
        (v 40, m 7); (v 41, m 8); (m 31, m 19); (v 30, m 13); (v 31, m 14);
        (m 2, m 15); (m 30, c 52); (m 17, consts); (m 18, consts); (m 16, v 11);
        (m 27, m 9); (m 6, m 26); (m 23, m 25); (v 47, m 28); (m 25, m 9);
        (m 13, m 3); (m 14, m 4); (m 15, m 5); (m 24, c 51); (m 7, v 42);
        (m 8, consts); (m 6, m 28)];
     t_any_usage :=
       [m 1; m 30; m 24; m 2; v 30; m 13; m 15; c 52; m 3; m 5; c 51; v 40; m 7;
        v 42; m 17; consts];
     t_any_source :=
       [c 52; c 51; consts; m 30; m 17; m 18; v 42; v 43; m 24; m 7; m 8; v 40;
        v 41; m 3; m 4; m 13; m 14; v 30; v 31; m 1];
     t_has_usage :=
       [m 1; m 30; m 24; m 2; v 30; m 29; m 23; m 31; m 19; m 13; m 15; c 52;
        m 25; v 11; v 47; v 46; m 16; m 28; m 9; m 3; m 5; c 51; m 32; v 40;
        m 7; v 42; m 17; consts];
     t_has_source :=
       [c 52; c 51; consts; v 11; m 28; m 21; m 22; m 20; m 19; m 11; m 12;
        m 10; m 9; m 33; m 31; m 30; m 17; m 18; m 29; v 42; v 43; m 16; m 27;
        v 46; v 47; m 25; m 24; m 7; m 8; v 40; v 41; m 23; m 6; m 3; m 4; v 45;
        m 13; m 14; v 30; v 31; m 1];
     t_usages :=
       [(v 11, v 11); (m 29, m 29); (v 47, v 47); (m 23, m 23); (v 46, v 46);
        (m 31, m 29); (m 16, v 46); (m 32, v 46); (m 19, m 29); (v 11, v 46);
        (m 25, m 23); (m 28, v 47); (m 9, m 23)];
     t_sources :=
       [(m 28, m 28); (v 11, v 11); (m 21, m 21); (m 20, m 20); (m 22, m 22);
        (m 19, m 19); (m 11, m 11); (m 10, m 10); (m 12, m 12); (m 9, m 9);
        (m 33, m 19); (m 31, m 19); (m 29, m 19); (m 16, v 11); (m 27, m 9);
        (v 46, v 11); (v 47, m 28); (m 25, m 9); (m 23, m 9); (m 6, m 28);
        (v 45, m 28)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used :=
       [(m 28, known); (v 11, known); (v 11, Field_function_slot 70);
        (m 19, result 0); (m 19, Field_return_of_call Exn);
        (m 19, Field_code_id_of_call_witness); (m 9, result 0);
        (m 9, Field_return_of_call Exn);
        (m 9, Field_code_id_of_call_witness)] |}.

Definition rebuilt : expr :=
  Let_static
    [Bind_code 51 PoisonResultExample.code_51_changed;
     Bind_code 52
       (Code [42] 46 5 6
          {| cm_params_arity :=
               [Unboxed_product [Singleton Value]; Unboxed_product []];
             cm_result_arity := [Unboxed_product [Singleton Value]];
             cm_is_tupled := false;
             cm_is_stub := false |}
          (Let 47 Value (Project_function_slot 71 70 (Var 46))
             (Apply (PoisonResultExample.call_51_changed 5))))]
    (Let 11 Value (Closure_of_set 71 [(70, 51); (71, 52)] [])
       (Let_cont_non_recursive 20
          (Handler [30] (make_handler_metadata [Value])
             (return_to 1 [Var 30]))
          (Apply
             {| ap_callee := Var 11; ap_call_kind := Direct 52;
                ap_args := [Const (Tagged_immediate 5)];
                ap_continuation := 20; ap_exn_continuation := 2;
                ap_exn_extra_args := [];
                ap_args_arity :=
                  [Unboxed_product [Singleton Value]; Unboxed_product []];
                ap_return_arity := [Unboxed_product [Singleton Value]] |}))).

End MaskedReturnExample.

Definition masked_return_example_statement : Prop :=
  in_flow_fragment MaskedReturnExample.unit_expr = true
  ∧ static_constants (unit_codes MaskedReturnExample.unit_expr)
      MaskedReturnExample.unit_expr
    = true
  ∧ results_kinded [] [] MaskedReturnExample.unit_expr = true
  ∧ well_formed_unit Examples.interface MaskedReturnExample.unit_expr = true
  ∧ exits_one_value Examples.interface MaskedReturnExample.unit_expr = true
  ∧ check_closed Examples.no_local MaskedReturnExample.unit_facts
      MaskedReturnExample.solution
    = true
  ∧ check_field_used_exact Examples.no_local MaskedReturnExample.unit_facts
      MaskedReturnExample.solution
    = true
  ∧ derive_cannot_change MaskedReturnExample.unit_facts
      MaskedReturnExample.solution
    = []
  ∧ check_cc_closed MaskedReturnExample.unit_facts
      MaskedReturnExample.solution []
    = true
  ∧ unit_calling_convention_changes MaskedReturnExample.solution []
      (unit_code_deps Auto Examples.interface MaskedReturnExample.unit_expr)
      MaskedReturnExample.unit_expr
    = [(51%positive,
        Changing_calling_convention [Keep Value; Delete] [Keep Value; Delete]);
       (52%positive,
        Changing_calling_convention [Keep Value; Delete] [Keep Value; Delete])]
  ∧ rebuild_unit_cc_from_tables MaskedReturnExample.unit_facts
      MaskedReturnExample.solution []
      (unit_code_deps Auto Examples.interface MaskedReturnExample.unit_expr)
      true MaskedReturnExample.unit_expr
    = MaskedReturnExample.rebuilt
  ∧ behaves
      (State MaskedReturnExample.unit_expr [] [] empty_static_env empty_heap)
      (Exits 1%positive [Value_const (Tagged_immediate 5)] empty_heap)
  ∧ behaves
      (State MaskedReturnExample.rebuilt [] [] empty_static_env empty_heap)
      (Exits 1%positive [Value_const (Tagged_immediate 5)] empty_heap).
