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

(* Tupled code at an unknown arity, and the reaper: what the facts that
   keep the tuple alive are for.

   This file is to be reviewed: the unit, the removal of the two facts,
   and the statements, against
   middle_end/flambda2/reaper/traverse_acc.ml
   ([create_unknown_arity_tupled_call_witnesses]) and
   testsuite/tests/reaper/unused_tupled_function_param.ml, and against
   FlambdaSemanticsSpec.tuple_fields, which says how the machine reads a
   tuple. The proofs are in FlambdaReaperTupledProofs.v, which needs no
   review; each statement is a computation.

   A call of unknown arity of tupled code goes, at run time, through the
   entry point [caml_tuplifyN] of the closure (backend/cmm_helpers.ml,
   [tuplify_function]), which loads every field of the tuple it is given
   and calls the code with them. The loads are not in the term. In the
   graph, the parameters of the code are read off the one parameter of
   the witness of unknown arity, [untuple_var], by accessors of its
   fields; if the code uses none of them, nothing uses the tuple, and
   the reaper made the tuple poison, of which [caml_tuplifyN] then
   loaded the fields. The fix (#7072) gives the traversal two facts
   more: an accessor of [Is_int] of [untuple_var] into a node of its own,
   [keep_tuple_alive_var], and a use of that node by the code id, so
   that the tuple has a use wherever the code has one.
   FlambdaReaperTraversal.v transcribes the traversal with the two
   facts.

   The machine does what [caml_tuplifyN] does: every call gives tupled
   code one argument, the tuple, and the machine reads its fields
   (FlambdaSemanticsSpec.tuple_fields, [arguments_for_code]). A value
   that is not an immutable block of tag 0 with a field of kind [Value]
   for each parameter has no fields, and the call is undefined. Poison
   is not a block.

   The unit is that of the test of the fix. The tupled code [ignore]
   takes a pair and uses neither component; the code [pass] returns its
   argument. The unit calls [pass] directly with the closure of
   [ignore], calls the closure that [pass] returns, which the unit does
   not know statically, at unknown arity ([Indirect]) with a pair, and
   returns the result. Through [pass] the reaper follows the closure to
   the code, and sees that the code uses no field of the pair. Through
   [Opaque_identity] it would not: the closure could then be any, the
   pair would escape ([any_usage]) and be kept, with or without the two
   facts. The pair is a static block, as Simplify makes it.
   middle_end/flambda2/tests/semantics/reaper/tupled_unused_components.fl
   is the same unit in the fexpr language, with the module block that
   an example returns; the test tool runs the compiler's reaper on it
   beside the model.

   - [without_keep_alive] removes the two facts from a graph: the graph
     of the unit without them is the one that the traversal with the fix
     reverted would make, but for the numbers of the nodes that the
     traversal makes after [keep_tuple_alive_var], which mean nothing
     (FlambdaReaperTraversal.v). In the graph of the unit it removes the
     two facts and nothing else ([keep_alive_facts_statement]).
   - The two solutions are the least solutions of the two graphs: the
     first as the test tool's solver computes it, the second the first
     less what came of the two facts. Both pass [check_closed] and
     [check_field_used_exact], with and without local slots
     ([with_keep_alive_solution_passes_statement],
     [without_keep_alive_solution_passes_statement]), so they are closed
     under the rules (FlambdaReaperSolutionSpec.v); the second is not
     closed for the first graph ([keep_alive_facts_needed_statement]).
   - Rebuilt with the first, the pair is kept, each of its fields
     poison; rebuilt with the second, the pair is gone and the call is
     given poison ([with_keep_alive_rebuilt_statement],
     [without_keep_alive_rebuilt_statement]).
   - The unit is well-formed and well-kinded, it returns 1 and does not
     go wrong; rebuilt with the first graph and solution it returns 1
     too, and refines it ([unit_well_formed_statement],
     [with_keep_alive_rebuild_runs_statement],
     [unit_does_not_go_wrong_statement],
     [with_keep_alive_refines_statement]).
   - Rebuilt with the second it goes wrong, at the call of [ignore],
     where the tuple is poison and has no fields; so it does not refine
     the unit ([without_keep_alive_rebuild_goes_wrong_statement],
     [without_keep_alive_tuple_is_poison_statement],
     [without_keep_alive_not_refinement_statement]).

   What this shows. The statement of S7 is to be that the unit rebuilt
   with any solution of its graph that passes [check_closed] refines
   the unit run on its own (status/S7.md). With the fix reverted it is
   false: the second solution passes the check that the rebuild relies
   on, and the rebuilt unit goes wrong where the unit returns, at the
   loads that the entry point of tupled code makes of the tuple. The
   model detects that the two facts are needed: the tuple must stay a
   block of as many fields as the code has parameters, whatever is in
   them, for as long as the code may be called, which is what the
   comment of the fix says of [caml_tuplifyN]. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaUnusedStaticsSpec FlambdaWellFormedSpec FlambdaWellKindedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperRebuild.
Import ListNotations.

(* ======================================================================== *)
(* The two facts of the fix                                                 *)
(* ======================================================================== *)

(* The accessor of [Is_int] that keeps a tuple alive, told by its shape:
   it reads into a node that the traversal made. The other accessors of
   [Is_int] are those of the primitive [Is_int], which read into the
   variable that their [Let] binds
   (FlambdaReaperTraversal.traverse_prim). *)
Definition is_keep_alive_accessor (fct : node * field * node) : bool :=
  match fct with
  | (Node_made _, Field_is_int, _) => true
  | _ => false
  end.

(* The nodes [keep_tuple_alive_var] of a graph. *)
Definition keep_alive_nodes (F : facts) : list node :=
  map (fun '(to_, _, _) => to_) (filter is_keep_alive_accessor (f_accessor F)).

(* The use of such a node by a code id. No other use of a traversal is
   of one: the others are of the names of the unit, of the node of
   constants, or of a callee that [add_apply] makes, and no accessor of
   [Is_int] reads into any of these. *)
Definition is_keep_alive_use (F : facts) (fct : node * node) : bool :=
  match fct with
  | (Node_code_id _, from) => mem_node from (keep_alive_nodes F)
  | _ => false
  end.

(* The graph without the two facts: the traversal with the fix
   reverted. *)
Definition without_keep_alive (F : facts) : facts :=
  {| f_alias := f_alias F;
     f_use := filter (fun u => negb (is_keep_alive_use F u)) (f_use F);
     f_accessor :=
       filter (fun a => negb (is_keep_alive_accessor a)) (f_accessor F);
     f_constructor := f_constructor F;
     f_argument := f_argument F;
     f_parameter := f_parameter F;
     f_propagate := f_propagate F;
     f_alias_if_any_source := f_alias_if_any_source F;
     f_any_usage := f_any_usage F;
     f_any_source := f_any_source F;
     f_any_source_if_used := f_any_source_if_used F;
     f_zero_alloc_source := f_zero_alloc_source F |}.

(* ======================================================================== *)
(* The unit                                                                 *)
(* ======================================================================== *)

Definition a : variable := 1%positive.
Definition b : variable := 2%positive.
Definition ignore_my_closure : variable := 3%positive.
Definition f : variable := 4%positive.
Definition pass_my_closure : variable := 5%positive.
Definition func : variable := 6%positive.

Definition done : continuation := 1%positive.
Definition error : continuation := 2%positive.
Definition ignore_return : continuation := 3%positive.
Definition ignore_exn : continuation := 4%positive.
Definition pass_return : continuation := 5%positive.
Definition pass_exn : continuation := 6%positive.
Definition after_pass : continuation := 7%positive.

Definition ignore_code_id : code_id := 1%positive.
Definition pass_code_id : code_id := 2%positive.
Definition ignore_slot : function_slot := 1%positive.
Definition pass_slot : function_slot := 2%positive.
Definition ignore_symbol : symbol := 1%positive.
Definition pass_symbol : symbol := 2%positive.
Definition pair_symbol : symbol := 3%positive.

Definition interface : unit_interface := Unit_interface done error.

Definition a_value : arity := [Singleton Value].

(* [let ignore (_a, _b) = 1]: tupled, of two parameters. *)
Definition ignore_code : code :=
  Code [a; b] ignore_my_closure ignore_return ignore_exn
    (make_tupled_code_metadata [Singleton Value; Singleton Value] a_value)
    (Apply_cont (Apply_cont_expr ignore_return [Const (Tagged_immediate 1)]
                   None)).

(* [let pass f = f]. *)
Definition pass_code : code :=
  Code [f] pass_my_closure pass_return pass_exn
    (make_code_metadata a_value a_value)
    (Apply_cont (Apply_cont_expr pass_return [Var f] None)).

(* The call of the closure that [pass] returns, with [tuple], at
   unknown arity, returning to [done]. *)
Definition call_of_func (tuple : simple) : apply_expr :=
  Apply_expr (Var func) Indirect [tuple] done error [] a_value a_value.

(* The unit with [pair_bindings] for the binding of the pair, giving
   [tuple] to the closure that [pass] returns:

     let code ignore, pass; let $ignore, $pass = closures; pair_bindings
     let_cont after_pass (func) = apply func (tuple) -> done * error in
     apply direct(pass) $pass ($ignore) -> after_pass * error *)
Definition unit_with_tuple (pair_bindings : list static_binding)
    (tuple : simple) : expr :=
  Let_static
    ([Bind_code ignore_code_id ignore_code; Bind_code pass_code_id pass_code]
     ++ bind_closures [(ignore_slot, ignore_symbol)]
          [(ignore_slot, ignore_code_id)] []
     ++ bind_closures [(pass_slot, pass_symbol)] [(pass_slot, pass_code_id)] []
     ++ pair_bindings)
    (Let_cont_non_recursive after_pass
       (Handler [func] (make_handler_metadata [Value])
          (Apply (call_of_func tuple)))
       (Apply
          (Apply_expr (Symbol pass_symbol) (Direct pass_code_id)
             [Symbol ignore_symbol] after_pass error [] a_value a_value))).

(* The binding of the pair, with [fields], to [pair_symbol]. *)
Definition pair_with (fields : list simple) : list static_binding :=
  [Bind_symbol pair_symbol (Static_const_block 0 fields)].

(* The unit: the pair is [(0, 0)]. *)
Definition tupled_unit : expr :=
  unit_with_tuple
    (pair_with [Const (Tagged_immediate 0); Const (Tagged_immediate 0)])
    (Symbol pair_symbol).

(* A term run on its own. *)
Definition on_its_own (e : expr) : state :=
  State e [] [] empty_static_env empty_heap.

(* ======================================================================== *)
(* The two graphs and their solutions                                       *)
(* ======================================================================== *)

(* The graph of the unit, as the reaper builds it with the default of
   [-reaper-preserve-direct-calls], and the graph without the two
   facts. *)
Definition with_keep_alive_graph : facts :=
  traverse_unit Auto interface tupled_unit.

Definition without_keep_alive_graph : facts :=
  without_keep_alive with_keep_alive_graph.

(* No field is local (the default, without [-reaper-local-fields]), or
   the slots are, as all slots are of the unit (with the flag). *)
Definition no_local_fields (relation : field) : bool := false.

Definition slots_local (relation : field) : bool :=
  match relation with
  | Field_value_slot _ | Field_function_slot _ => true
  | _ => false
  end.

(* The nodes that the traversal makes, by their numbers: 1 and 2 the
   toplevel return and exception; 3 to 11 for the code [ignore] (its
   result 3, its exception 4, its [my_closure] 5, its parameters 6 and
   7, its witnesses 8 of known arity and 9 of unknown arity, the tuple
   10 that is the parameter of 9, [untuple_var], and 11,
   [keep_tuple_alive_var]); 12 to 17 for the code [pass] (its result
   12, its exception 13, its [my_closure] 14, its parameter 15, its
   witnesses 16 and 17); 18 to 20 for the call of [func] ([called], its
   witness, the call); 21 to 25 for the call of [pass] (its witness,
   [called], the call, and the callee and the call if the callee has
   any source). *)
Section Solutions.

Let made (n : positive) : node := Node_made n.
Let var (x : variable) : node := Node_variable x.
Let sym (s : symbol) : node := Node_symbol s.
Let ignore_node : node := Node_code_id ignore_code_id.
Let pass_node : node := Node_code_id pass_code_id.
Let consts : node := Node_all_constants.

(* What the solution has in common for the two graphs. *)
Let common_alias : list (node * node) :=
  [(made 25, made 16); (made 14, made 24); (made 21, made 23);
   (made 19, made 20); (made 12, var f); (var pass_my_closure, made 14);
   (var f, made 15); (made 3, consts); (var ignore_my_closure, made 5);
   (var a, made 6); (var b, made 7); (made 23, made 16);
   (var func, made 12); (made 2, made 13); (made 22, pass_node);
   (made 15, sym ignore_symbol); (made 20, made 9); (made 1, made 3);
   (made 2, made 4); (made 18, ignore_node); (made 10, sym pair_symbol);
   (made 6, consts); (made 7, consts); (made 5, sym ignore_symbol);
   (made 14, sym pass_symbol)].

Let common_any_usage : list node :=
  [made 1; made 22; made 18; made 2; made 13; pass_node; made 3; made 4;
   ignore_node; consts].

Let common_any_source : list node :=
  [pass_node; ignore_node; consts; made 3; made 22; made 1; made 18;
   made 6; made 7; var a; var b].

Let common_has_usage : list node :=
  [made 1; made 22; made 18; made 2; made 21; made 19; made 23; made 20;
   made 16; made 13; pass_node; made 9; made 3; made 4; ignore_node;
   consts; sym pass_symbol; var func; made 12; var f; made 15;
   sym ignore_symbol].

Let common_has_source : list node :=
  [pass_node; ignore_node; consts; made 3; sym pair_symbol;
   sym pass_symbol; made 17; made 16; sym ignore_symbol; made 9; made 8;
   made 25; made 23; made 22; made 15; made 20; made 1; made 18; made 10;
   made 6; made 7; made 21; made 19; var f; var a; var b; made 5;
   made 14; made 12; var pass_my_closure; var ignore_my_closure;
   var func].

Let common_usages : list (node * node) :=
  [(sym pass_symbol, sym pass_symbol); (made 21, made 21);
   (var func, var func); (made 19, made 19); (made 10, made 10);
   (made 23, made 21); (made 20, made 19); (made 16, made 21);
   (made 12, var func); (var f, var func); (made 15, var func);
   (sym ignore_symbol, var func); (made 9, made 19);
   (sym pair_symbol, made 10)].

Let common_sources : list (node * node) :=
  [(sym ignore_symbol, sym ignore_symbol); (sym pass_symbol, sym pass_symbol);
   (sym pair_symbol, sym pair_symbol); (made 17, made 17);
   (made 16, made 16); (made 9, made 9); (made 8, made 8);
   (made 25, made 16); (made 23, made 16); (made 21, made 16);
   (made 15, sym ignore_symbol); (var f, sym ignore_symbol);
   (made 12, sym ignore_symbol); (var func, sym ignore_symbol);
   (made 20, made 9); (made 19, made 9); (made 10, sym pair_symbol);
   (made 5, sym ignore_symbol); (made 14, sym pass_symbol);
   (var pass_my_closure, sym pass_symbol);
   (var ignore_my_closure, sym ignore_symbol)].

Let common_field_used : list (node * field) :=
  [(sym ignore_symbol, Field_call_witness Unknown_arity_code_pointer);
   (sym pass_symbol, Field_call_witness Known_arity_code_pointer);
   (made 16, Field_return_of_call (Normal 0));
   (made 16, Field_return_of_call Exn);
   (made 16, Field_code_id_of_call_witness);
   (made 9, Field_return_of_call (Normal 0));
   (made 9, Field_return_of_call Exn);
   (made 9, Field_code_id_of_call_witness)].

(* The least solution of the first graph. The code id of [ignore] has a
   use, from the call of [func] ([called], 18, is an alias of it); so 11
   has [any_usage] by the use of the fix, and the tuple 10 a use, by the
   accessor of [Is_int], and so the pair, of which 10 is an alias by the
   call. Only the field [Is_int] of the pair is used: its fields 0 and
   1 flow to the parameters of [ignore], which have no use. *)
Definition with_keep_alive_solution : tables :=
  {| t_alias := (made 11, consts) :: common_alias;
     t_any_usage := made 11 :: common_any_usage;
     t_any_source := made 11 :: common_any_source;
     t_has_usage := made 11 :: made 10 :: sym pair_symbol :: common_has_usage;
     t_has_source := made 11 :: common_has_source;
     t_usages := common_usages;
     t_sources := common_sources;
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used := (sym pair_symbol, Field_is_int) :: common_field_used |}.

(* The least solution of the second: the first without what came of the
   two facts. The node 11 is in no fact, and neither the tuple 10 nor
   the pair has a use. *)
Definition without_keep_alive_solution : tables :=
  {| t_alias := common_alias;
     t_any_usage := common_any_usage;
     t_any_source := common_any_source;
     t_has_usage := common_has_usage;
     t_has_source := common_has_source;
     t_usages := common_usages;
     t_sources := common_sources;
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used := common_field_used |}.

End Solutions.

(* The unit rebuilt with each graph and its solution, with the default
   of [-reaper-preserve-direct-calls] ([Auto]: [true] here). *)
Definition rebuilt_with_keep_alive : expr :=
  rebuild_unit_from_tables with_keep_alive_graph with_keep_alive_solution
    true tupled_unit.

Definition rebuilt_without_keep_alive : expr :=
  rebuild_unit_from_tables without_keep_alive_graph
    without_keep_alive_solution true tupled_unit.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* The two facts that [without_keep_alive] removes from the graph of the
   unit, and nothing else: the accessor of [Is_int] of the tuple 10 into
   11, and the use of 11 by the code id of [ignore]. *)
Definition keep_alive_facts_statement : Prop :=
  filter is_keep_alive_accessor (f_accessor with_keep_alive_graph)
  = [(Node_made 11, Field_is_int, Node_made 10)]
  ∧ filter (is_keep_alive_use with_keep_alive_graph)
      (f_use with_keep_alive_graph)
    = [(Node_code_id ignore_code_id, Node_made 11)].

Definition unit_well_formed_statement : Prop :=
  well_formed_unit interface tupled_unit = true
  ∧ well_kinded_unit [] done error tupled_unit = true.

Definition with_keep_alive_solution_passes_statement : Prop :=
  ∀ is_local : field -> bool,
    is_local ∈ [no_local_fields; slots_local] ->
    check_closed is_local with_keep_alive_graph with_keep_alive_solution
    = true
    ∧ check_field_used_exact is_local with_keep_alive_graph
        with_keep_alive_solution
      = true.

Definition without_keep_alive_solution_passes_statement : Prop :=
  ∀ is_local : field -> bool,
    is_local ∈ [no_local_fields; slots_local] ->
    check_closed is_local without_keep_alive_graph without_keep_alive_solution
    = true
    ∧ check_field_used_exact is_local without_keep_alive_graph
        without_keep_alive_solution
      = true.

(* The second solution is not one of the first graph: the code id of
   [ignore] has a use, and the use of the fix asks that 11 have
   [any_usage]. *)
Definition keep_alive_facts_needed_statement : Prop :=
  ∀ is_local : field -> bool,
    ~ closed is_local (graph_of_facts with_keep_alive_graph)
        (solution_of_tables without_keep_alive_solution).

(* Rebuilt with the first, the pair is kept, its fields poison; rebuilt
   with the second, the pair is not bound and the call is given
   poison. *)
Definition with_keep_alive_rebuilt_statement : Prop :=
  rebuilt_with_keep_alive
  = unit_with_tuple (pair_with [Const (Poison Value); Const (Poison Value)])
      (Symbol pair_symbol).

Definition without_keep_alive_rebuilt_statement : Prop :=
  rebuilt_without_keep_alive = unit_with_tuple [] (Const (Poison Value)).

Definition with_keep_alive_rebuild_runs_statement : Prop :=
  behaves (on_its_own tupled_unit)
    (Exits done [Value_const (Tagged_immediate 1)] empty_heap)
  ∧ behaves (on_its_own rebuilt_with_keep_alive)
      (Exits done [Value_const (Tagged_immediate 1)] empty_heap).

Definition unit_does_not_go_wrong_statement : Prop :=
  ~ behaves (on_its_own tupled_unit) Goes_wrong.

Definition with_keep_alive_refines_statement : Prop :=
  program_refines tupled_unit rebuilt_with_keep_alive.

Definition without_keep_alive_rebuild_goes_wrong_statement : Prop :=
  behaves (on_its_own rebuilt_without_keep_alive) Goes_wrong.

(* Why: the run comes to the call of the closure of [ignore], tupled code
   of two parameters, with poison for the tuple; poison has no fields,
   and the call is undefined. *)
Definition without_keep_alive_tuple_is_poison_statement : Prop :=
  ∃ (fuel : nat) (s : state),
    run fuel (on_its_own rebuilt_without_keep_alive) = Step s
    ∧ state_expr s = Apply (call_of_func (Const (Poison Value)))
    ∧ eval_simple (state_vars s) (Var func) = Some (Value_symbol ignore_symbol)
    ∧ closure_of (state_statics s) (Value_symbol ignore_symbol)
      = Some (ignore_code_id, [])
    ∧ lookup ignore_code_id (static_code (state_statics s)) = Some ignore_code
    ∧ tuple_fields (state_statics s) 2 (Value_const (Poison Value)) = None
    ∧ step s = Undefined.

Definition without_keep_alive_not_refinement_statement : Prop :=
  ~ program_refines tupled_unit rebuilt_without_keep_alive.
