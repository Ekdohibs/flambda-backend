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

   Where code changes, the rebuilt unit runs as the unit does on four
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
   keep ([masked_return_example_statement]); and a unit in which code
   that changes calls in tail position code that does not, [let f x y =
   g x] with [g] called at unknown arity too, so that the call returns
   through the wrapper of [make_apply_wrapper] to the return continuation
   of [f], which keeps the first result only
   ([tail_escape_example_statement]).

   The rebuild with changes of calling convention refines its unit
   ([rebuild_cc_refines_observably_statement]) as the rebuild without
   does (FlambdaReaperRebuildSpec.rebuild_refines_observably_statement).
   Calls return anywhere: to the continuations of the unit, to handlers,
   and in tail position, in code that changes or not, to code that
   changes or not or of unknown arity, in chains of calls through code
   that changes, keeping other results or not. An earlier step asked
   that a call in tail position in code that changes be known to call
   code that changes ([check_cc_returns]); the statement no longer asks
   it. The four units above are instances ([two_calls_refines_statement],
   [poison_result_refines_statement],
   [masked_return_refines_statement], [tail_escape_refines_statement]).
   The statement asks of the table more than that it be closed and pass
   [check_cc_closed]: that the witnesses of known arity that calls read
   off a node have the same sources ([check_apply_widgets]), which the
   rules derive of them but a table closed under the rules need not
   have; without it, a call of code that changes is not known to call it
   and goes wrong ([spurious_widget_counterexample_statement]). That each
   witness of the facts give one code id ([check_witness_code_ids]) is
   not asked: the traversal makes it so of every unit
   ([traverse_witness_code_ids_statement]). *)

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
(* The fragment                                                             *)
(* ======================================================================== *)

(* No call of [e] returns or raises to [ret]. *)
Fixpoint no_call_returns_to (ret : continuation) (e : expr) : bool :=
  match e with
  | Let _ _ _ body | Let_static _ body => no_call_returns_to ret body
  | Let_cont_non_recursive _ h body =>
    handler_no_call_returns_to ret h && no_call_returns_to ret body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_no_call_returns_to ret h) handlers
    && no_call_returns_to ret body
  | Apply ap =>
    negb (Pos.eqb (ap_continuation ap) ret
          || Pos.eqb (ap_exn_continuation ap) ret)
  | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_no_call_returns_to (ret : continuation)
    (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => no_call_returns_to ret handler
  end.

(* Every call of [e] returns to a continuation that [e] binds, or that
   [bound] has. *)
Fixpoint calls_return_to_handlers (bound : list continuation) (e : expr)
    : bool :=
  match e with
  | Let _ _ _ body | Let_static _ body => calls_return_to_handlers bound body
  | Let_cont_non_recursive k h body =>
    handler_calls_return_to_handlers bound h
    && calls_return_to_handlers (k :: bound) body
  | Let_cont_recursive handlers body =>
    forallb
      (fun '(_, h) =>
         handler_calls_return_to_handlers (map fst handlers ++ bound) h)
      handlers
    && calls_return_to_handlers (map fst handlers ++ bound) body
  | Apply ap => among (ap_continuation ap) bound
  | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_calls_return_to_handlers (bound : list continuation)
    (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => calls_return_to_handlers bound handler
  end.

(* The units whose calls return to handlers: no call of the unit returns
   to a continuation of the unit, nor one of code to a continuation of
   the code, and none raises to the return continuation of its code. In
   particular, the code makes no call in tail position. An earlier step
   of the proof that the rebuild with changes of calling convention
   refines, which the unit of [two_calls_example_statement] meets, and
   those of [poison_result_example_statement] and
   [masked_return_example_statement] do not
   ([cc_fragment_examples_statement]); the statement no longer asks
   it. *)
Definition cc_fragment (e : expr) : bool :=
  calls_return_to_handlers [] e
  && forallb
       (fun '(_, c) =>
          calls_return_to_handlers [] (code_body c)
          && no_call_returns_to (code_return_continuation c) (code_body c))
       (unit_codes e).

(* Each call witness gives one code id. The traversal gives a witness a
   code id once, where it makes it: the witnesses of code of the unit
   give its code id, and the witness of a closure whose code is not of
   the unit gives the closure ([traverse_witness_code_ids_statement]). *)
Definition check_witness_code_ids (F : facts) : bool :=
  forallb
    (fun '(w, f, c) =>
       negb (field_eqb f Field_code_id_of_call_witness)
       || forallb
            (fun '(w', f', c') =>
               negb (node_eqb w' w
                     && field_eqb f' Field_code_id_of_call_witness)
               || node_eqb c' c)
            (f_constructor F))
    (f_constructor F).

(* The facts of every unit pass [check_witness_code_ids]. *)
Definition traverse_witness_code_ids_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (u : unit_interface) (e : expr),
    check_witness_code_ids (traverse_unit flag u e) = true.

(* The witnesses of known arity that calls read off a node have the same
   sources in the table: [code_id_actually_directly_called] reads them
   all, the rules of [cannot_change_calling_convention] those of the
   calls of code that is used. Each is read off the node by its call
   only, so the rules derive the same of all
   ([spurious_widget_counterexample_statement]). *)
Definition check_apply_widgets (F : facts) (T : tables) : bool :=
  forallb
    (fun '(a, f, n) =>
       forallb
         (fun '(a', f', n') =>
            negb
              (field_eqb f (Field_call_witness Known_arity_code_pointer)
               && field_eqb f' f && node_eqb n' n)
            || (Bool.eqb (mem_node a (t_any_source T))
                  (mem_node a' (t_any_source T))
                && forallb
                     (fun '(x, y) =>
                        negb (node_eqb x a) || mem_pair a' y (t_sources T))
                     (t_sources T)))
         (f_accessor F))
    (f_accessor F).

Section Returns.

Variable F : facts.
Variable sol : solution.
Variable cannot_change : node -> bool.
Variable code_deps : finite_map code_dep.
Variable unit_code_ids : list code_id.
Variable codes : finite_map code_metadata.

(* What returns in [e] to [r], the return continuation of code, whose
   results are of the kinds [K], the variables having the kinds [E]: an
   [Apply_cont] gives it simples of the kinds [K]; a call that returns
   to it, to code that changes its calling convention or not, or of
   unknown arity, has the return arity [K]. Under a [Let_cont] that
   binds [r] again, nothing; nor in a handler that does not have as many
   kinds as parameters, which is never entered. The kinds are those that
   [FlambdaReaperRebuildSpec.results_kinded] checks. *)
Fixpoint ret_calls_ok (E : finite_map flambda_kind) (r : continuation)
    (K : list flambda_kind) (e : expr) : bool :=
  match e with
  | Let x kind _ body => ret_calls_ok ((x, kind) :: E) r K body
  | Let_static _ body => ret_calls_ok E r K body
  | Let_cont_non_recursive k h body =>
    handler_ret_calls_ok E r K h
    && (Pos.eqb k r || ret_calls_ok E r K body)
  | Let_cont_recursive handlers body =>
    among r (map fst handlers)
    || (forallb (fun '(_, h) => handler_ret_calls_ok E r K h) handlers
        && ret_calls_ok E r K body)
  | Apply ap =>
    negb (Pos.eqb (ap_continuation ap) r)
    || kinds_eqb K (unarize (ap_return_arity ap))
  | Apply_cont ac =>
    negb (Pos.eqb (ac_continuation ac) r) || simples_of_kinds E (ac_args ac) K
  | Switch _ arms =>
    forallb
      (fun '(_, ac) =>
         negb (Pos.eqb (ac_continuation ac) r)
         || simples_of_kinds E (ac_args ac) K)
      arms
  | Invalid => true
  end
with handler_ret_calls_ok (E : finite_map flambda_kind) (r : continuation)
    (K : list flambda_kind) (h : continuation_handler) : bool :=
  match h with
  | Handler params metadata handler =>
    negb (Nat.eqb (length params) (length (hm_params_kinds metadata)))
    || ret_calls_ok (rev (with_kinds params (hm_params_kinds metadata)) ++ E)
         r K handler
  end.

(* A call in [e] returns to [r] only if it is known to call code that
   changes its calling convention. Under a [Let_cont] that binds [r]
   again, nothing. *)
Fixpoint tail_calls_known (r : continuation) (e : expr) : bool :=
  match e with
  | Let _ _ _ body | Let_static _ body => tail_calls_known r body
  | Let_cont_non_recursive k h body =>
    handler_tail_calls_known r h && (Pos.eqb k r || tail_calls_known r body)
  | Let_cont_recursive handlers body =>
    among r (map fst handlers)
    || (forallb (fun '(_, h) => handler_tail_calls_known r h) handlers
        && tail_calls_known r body)
  | Apply ap =>
    negb (Pos.eqb (ap_continuation ap) r)
    || match
         apply_calling_convention_change F sol cannot_change code_deps
           unit_code_ids codes ap
       with
       | Changing_calling_convention _ _ => true
       | Not_changing_calling_convention => false
       end
  | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_tail_calls_known (r : continuation) (h : continuation_handler)
    : bool :=
  match h with
  | Handler _ _ handler => tail_calls_known r handler
  end.

End Returns.

(* The code of the unit that changes its calling convention: a call in
   tail position in it is known to call code that changes
   ([tail_calls_known]). An earlier premise of
   [rebuild_cc_refines_observably_statement], no longer asked. *)
Definition check_cc_returns (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep) (e : expr)
    : bool :=
  forallb
    (fun '(id, c) =>
       match c with
       | Code _ _ ret _ metadata body =>
         match
           code_calling_convention_change (solution_of_tables T)
             (fun n => mem_node n cannot_change) code_deps
             (static_names code_ids_bound e) id metadata
         with
         | Changing_calling_convention _ _ =>
           tail_calls_known F (solution_of_tables T)
                (fun n => mem_node n cannot_change) code_deps
                (static_names code_ids_bound e) (bound_code_metadata e) ret
                body
         | Not_changing_calling_convention => true
         end
       end)
    (unit_codes e).

(* ======================================================================== *)
(* The rebuild with changes of calling convention refines                    *)
(* ======================================================================== *)

(* The rebuild with changes of calling convention refines its unit, as
   FlambdaReaperRebuildSpec.rebuild_refines_observably_statement says of
   the rebuild without, with a table that passes [check_cc_closed], and a
   table of the solution whose witnesses read off a node agree
   ([check_apply_widgets]). That the witnesses of the facts give one code
   id each ([check_witness_code_ids]) is not asked: the traversal makes
   it so ([traverse_witness_code_ids_statement]). A call in tail position
   in code that changes may call code that changes, code that does not,
   or be of unknown arity: [check_cc_returns] is not asked. *)
Definition rebuild_cc_refines_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    check_apply_widgets (traverse_unit flag u e) T = true ->
    program_refines_observably e
      (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) preserve_direct_calls e).

(* The same of the units with the primitives of the heap
   ([in_heap_fragment]), the exits compared up to a correspondence of
   their locations, as
   FlambdaReaperRebuildSpec.rebuild_refines_up_to_heap_statement says of
   the rebuild without changes of calling convention. *)
Definition rebuild_cc_refines_up_to_heap_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_heap_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    check_apply_widgets (traverse_unit flag u e) T = true ->
    program_refines_up_to_heap e
      (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) preserve_direct_calls e).

(* And it gives the statement before it on the units of
   [in_flow_fragment], as
   FlambdaReaperRebuildSpec.rebuild_refines_up_to_heap_flow_statement. *)
Definition rebuild_cc_refines_up_to_heap_flow_statement : Prop :=
  rebuild_cc_refines_up_to_heap_statement ->
  rebuild_cc_refines_observably_statement.

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
  ∧ behaves (State Examples.two_calls [] [] empty_static_env
               empty_heap no_reads)
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
         [] [] empty_static_env empty_heap no_reads)
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
      (State PoisonResultExample.unit_expr [] [] empty_static_env
         empty_heap no_reads)
      (Exits 1%positive [Value_const (Tagged_immediate 5)] empty_heap)
  ∧ behaves
      (State PoisonResultExample.rebuilt [] [] empty_static_env
         empty_heap no_reads)
      (Exits 1%positive [Value_const (Tagged_immediate 5)] empty_heap).

(* A unit in which code that changes its calling convention calls code
   that does not in tail position: [let f x y = g x], with [g] returning
   two results. The code [f52] is called directly only, and loses its
   second parameter, which has no use, and its second result, which its
   caller does not use. The code [g51] is called at unknown arity too
   (the call of [x12]), so it keeps its calling convention. The call of
   [g51] in the body of [f52] returns to the return continuation [k5] of
   [f52], which keeps the first result only: the rebuild gives it the
   wrapper of [make_apply_wrapper] (the continuation 48), which applies
   [k5] to the first result. Its least solution, as the solver of the
   test tool computes it, passes the checks; [check_cc_returns] excluded
   it. *)
Module TailEscapeExample.

Local Open Scope positive_scope.
Import FlambdaReaperCallingConventions.Examples.

Definition code_51 : code :=
  Code [40] 45 3 4
    (make_code_metadata [Singleton Value] [Singleton Value; Singleton Value])
    (return_to 3 [Var 40; Const (Tagged_immediate 7)]).

Definition call_51 (k : continuation) : apply_expr :=
  {| ap_callee := Var 47; ap_call_kind := Direct 51; ap_args := [Var 42];
     ap_continuation := k; ap_exn_continuation := 6; ap_exn_extra_args := [];
     ap_args_arity := [Singleton Value];
     ap_return_arity := [Singleton Value; Singleton Value] |}.

Definition code_52 : code :=
  Code [42; 43] 46 5 6
    (make_code_metadata [Singleton Value; Singleton Value]
       [Singleton Value; Singleton Value])
    (Let 47 Value (Project_function_slot 71 70 (Var 46)) (Apply (call_51 5))).

Definition call_52 : apply_expr :=
  {| ap_callee := Var 11; ap_call_kind := Direct 52;
     ap_args := [Const (Tagged_immediate 5); Const (Tagged_immediate 8)];
     ap_continuation := 21; ap_exn_continuation := 2; ap_exn_extra_args := [];
     ap_args_arity := [Singleton Value; Singleton Value];
     ap_return_arity := [Singleton Value; Singleton Value] |}.

Definition call_12 (k : continuation) : apply_expr :=
  {| ap_callee := Var 12; ap_call_kind := Indirect; ap_args := [Var 31];
     ap_continuation := k; ap_exn_continuation := 2; ap_exn_extra_args := [];
     ap_args_arity := [Singleton Value];
     ap_return_arity := [Singleton Value; Singleton Value] |}.

Definition unit_expr : expr :=
  Let_static [Bind_code 51 code_51; Bind_code 52 code_52]
    (Let 11 Value (Closure_of_set 71 [(70, 51); (71, 52)] [])
       (Let 12 Value (Project_function_slot 71 70 (Var 11))
          (Let_cont_non_recursive 20
             (Handler [30; 33] (make_handler_metadata [Value; Value])
                (return_to 1 [Var 30]))
             (Let_cont_non_recursive 21
                (Handler [31; 32] (make_handler_metadata [Value; Value])
                   (Apply (call_12 20)))
                (Apply call_52))))).

Definition unit_facts : facts := traverse_unit Auto interface unit_expr.

Definition solution : tables :=
  {| t_alias :=
       [(m 33, m 16); (m 13, m 32); (m 29, m 31); (m 27, m 28); (m 1, v 30);
        (v 46, m 13); (v 42, m 14); (v 43, m 15); (m 3, v 40); (m 4, consts);
        (v 45, m 6); (v 40, m 7); (m 31, m 16); (v 12, m 25); (m 28, m 9);
        (v 31, m 10); (v 32, m 11); (m 2, m 12); (m 30, c 52); (m 14, consts);
        (m 15, consts); (m 2, m 5); (v 30, m 3); (v 33, m 4); (m 26, c 51);
        (m 7, v 31); (m 6, m 25); (m 13, v 11); (m 24, m 8); (m 6, m 23);
        (m 20, m 22); (v 47, m 25); (m 22, m 8); (m 10, m 3); (m 11, m 4);
        (m 12, m 5); (m 21, c 51); (m 7, v 42)];
     t_any_usage :=
       [m 1; m 30; m 26; m 21; m 2; v 30; m 12; c 52; m 5; m 3; c 51; v 40; m 7;
        v 31; m 10; v 42; m 14; consts];
     t_any_source :=
       [c 52; c 51; consts; m 4; m 30; m 14; m 15; v 33; m 26; v 42; v 43; m 11;
        m 21; m 7; v 40; v 32; m 3; v 30; m 10; m 1; v 31];
     t_has_usage :=
       [m 1; m 30; m 26; m 21; m 2; v 30; m 29; m 27; m 20; m 31; m 28; m 16;
        m 9; m 12; c 52; m 5; m 3; c 51; v 40; m 7; v 31; m 22; m 10; v 11;
        v 12; v 47; v 46; m 13; m 25; m 8; v 42; m 32; m 14; consts];
     t_has_source :=
       [c 52; c 51; consts; m 4; v 11; m 25; m 18; m 19; m 17; m 16; m 9; m 8;
        m 33; m 31; v 12; m 28; m 30; m 14; m 15; v 33; m 26; m 29; m 27; v 42;
        v 43; m 6; m 13; m 24; v 46; v 45; v 47; m 22; m 11; m 21; m 7; v 40;
        v 32; m 20; m 3; v 30; m 10; m 1; v 31];
     t_usages :=
       [(v 11, v 11); (m 29, m 29); (v 12, v 12); (m 27, m 27); (v 47, v 47);
        (m 20, m 20); (v 46, v 46); (m 31, m 29); (m 28, m 27); (m 13, v 46);
        (m 32, v 46); (m 16, m 29); (m 25, v 12); (m 9, m 27); (v 11, v 46);
        (m 22, m 20); (m 25, v 47); (m 8, m 20)];
     t_sources :=
       [(m 25, m 25); (v 11, v 11); (m 18, m 18); (m 17, m 17); (m 19, m 19);
        (m 16, m 16); (m 9, m 9); (m 8, m 8); (m 33, m 16); (m 31, m 16);
        (v 12, m 25); (m 29, m 16); (m 28, m 9); (m 27, m 9); (m 6, m 25);
        (m 13, v 11); (m 24, m 8); (v 46, v 11); (v 45, m 25); (v 47, m 25);
        (m 22, m 8); (m 20, m 8)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used :=
       [(m 25, known); (m 25, unknown); (v 11, known);
        (v 11, Field_function_slot 70); (m 16, result 0);
        (m 16, Field_return_of_call Exn); (m 16, Field_code_id_of_call_witness);
        (m 9, Field_return_of_call Exn); (m 9, result 0);
        (m 9, Field_code_id_of_call_witness); (m 8, result 0);
        (m 8, Field_return_of_call Exn);
        (m 8, Field_code_id_of_call_witness)] |}.

(* The wrapper of [make_apply_wrapper] over [k]: it takes the two
   results, and gives [k] the first. *)
Definition wrapper_over (k : continuation) (body : expr) : expr :=
  Let_cont_non_recursive 48
    (Handler [48; 49] (make_handler_metadata [Value; Value])
       (return_to k [Var 48]))
    body.

Definition rebuilt : expr :=
  Let_static
    [Bind_code 51 code_51;
     Bind_code 52
       (Code [42] 46 5 6
          {| cm_params_arity :=
               [Unboxed_product [Singleton Value]; Unboxed_product []];
             cm_result_arity := [Unboxed_product [Singleton Value]];
             cm_is_tupled := false;
             cm_is_stub := false |}
          (Let 47 Value (Project_function_slot 71 70 (Var 46))
             (wrapper_over 5 (Apply (call_51 48)))))]
    (Let 11 Value (Closure_of_set 71 [(70, 51); (71, 52)] [])
       (Let 12 Value (Project_function_slot 71 70 (Var 11))
          (Let_cont_non_recursive 20
             (Handler [30] (make_handler_metadata [Value])
                (return_to 1 [Var 30]))
             (Let_cont_non_recursive 21
                (Handler [31] (make_handler_metadata [Value])
                   (wrapper_over 20 (Apply (call_12 48))))
                (Apply
                   {| ap_callee := Var 11; ap_call_kind := Direct 52;
                      ap_args := [Const (Tagged_immediate 5)];
                      ap_continuation := 21; ap_exn_continuation := 2;
                      ap_exn_extra_args := [];
                      ap_args_arity :=
                        [Unboxed_product [Singleton Value];
                         Unboxed_product []];
                      ap_return_arity :=
                        [Unboxed_product [Singleton Value]] |}))))).

End TailEscapeExample.

Definition tail_escape_example_statement : Prop :=
  in_flow_fragment TailEscapeExample.unit_expr = true
  ∧ static_constants (unit_codes TailEscapeExample.unit_expr)
      TailEscapeExample.unit_expr
    = true
  ∧ results_kinded [] [] TailEscapeExample.unit_expr = true
  ∧ well_formed_unit Examples.interface TailEscapeExample.unit_expr = true
  ∧ exits_one_value Examples.interface TailEscapeExample.unit_expr = true
  ∧ check_closed Examples.no_local TailEscapeExample.unit_facts
      TailEscapeExample.solution
    = true
  ∧ check_field_used_exact Examples.no_local TailEscapeExample.unit_facts
      TailEscapeExample.solution
    = true
  ∧ derive_cannot_change TailEscapeExample.unit_facts
      TailEscapeExample.solution
    = [Node_code_id 51%positive]
  ∧ check_cc_closed TailEscapeExample.unit_facts
      TailEscapeExample.solution [Node_code_id 51%positive]
    = true
  ∧ check_apply_widgets TailEscapeExample.unit_facts
      TailEscapeExample.solution
    = true
  ∧ check_witness_code_ids TailEscapeExample.unit_facts = true
  ∧ unit_calling_convention_changes TailEscapeExample.solution
      [Node_code_id 51%positive]
      (unit_code_deps Auto Examples.interface TailEscapeExample.unit_expr)
      TailEscapeExample.unit_expr
    = [(51%positive, Not_changing_calling_convention);
       (52%positive,
        Changing_calling_convention [Keep Value; Delete] [Keep Value; Delete])]
  ∧ check_cc_returns TailEscapeExample.unit_facts TailEscapeExample.solution
      [Node_code_id 51%positive]
      (unit_code_deps Auto Examples.interface TailEscapeExample.unit_expr)
      TailEscapeExample.unit_expr
    = false
  ∧ rebuild_unit_cc_from_tables TailEscapeExample.unit_facts
      TailEscapeExample.solution [Node_code_id 51%positive]
      (unit_code_deps Auto Examples.interface TailEscapeExample.unit_expr)
      true TailEscapeExample.unit_expr
    = TailEscapeExample.rebuilt
  ∧ behaves
      (State TailEscapeExample.unit_expr [] [] empty_static_env
         empty_heap no_reads)
      (Exits 1%positive [Value_const (Tagged_immediate 5)] empty_heap)
  ∧ behaves
      (State TailEscapeExample.rebuilt [] [] empty_static_env
         empty_heap no_reads)
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
      (State MaskedReturnExample.unit_expr [] [] empty_static_env
         empty_heap no_reads)
      (Exits 1%positive [Value_const (Tagged_immediate 5)] empty_heap)
  ∧ behaves
      (State MaskedReturnExample.rebuilt [] [] empty_static_env
         empty_heap no_reads)
      (Exits 1%positive [Value_const (Tagged_immediate 5)] empty_heap).

Definition cc_fragment_examples_statement : Prop :=
  cc_fragment Examples.two_calls = true
  ∧ check_apply_widgets Examples.two_calls_facts Examples.two_calls_tables
    = true
  ∧ check_witness_code_ids Examples.two_calls_facts = true
  ∧ cc_fragment PoisonResultExample.unit_expr = false
  ∧ check_apply_widgets PoisonResultExample.unit_facts
      PoisonResultExample.solution
    = true
  ∧ cc_fragment MaskedReturnExample.unit_expr = false
  ∧ check_apply_widgets MaskedReturnExample.unit_facts
      MaskedReturnExample.solution
    = true.

(* A closed table with which the rebuild with changes of calling
   convention does not refine its unit:

     let code f50 x40 x42 my_closure x41 return k3 exn k4 = k3 x40 in
     let code f51 x43 my_closure x44 return k5 exn k6 =  (never called)
       let_cont k7 x45 = k5 x45 in
       s80 x43 x43 return k7 exn k6               (direct, to f50)
     in
     let symbol s80 = closure at 70 of { 70 = f50 } in
     let_cont k20 x30 = return x30 in
     s80 9 7 return k20 exn k2                    (direct, to f50)

   [x42] has no use, and [f50] loses it. With the least solution, as the
   solver of the test tool computes it ([least]), no code is in the
   least table, and the call of the unit is known to call [f50]
   ([code_id_actually_called]): it gives the first argument only, and
   the rebuilt unit exits as the unit does. The table [spurious] is
   [least] with one node more of [any_source]: the witness that the call
   in [f51] reads off [s80] (the node 20, of
   [Field_call_witness Known_arity_code_pointer]). It is closed, as
   nothing reads that node but through an alias that holds only where
   [f51] is used, which it is not; and it passes [check_cc_closed] with
   no code in the table, as the rules read the witnesses of the calls
   only through that alias; and it passes [check_cc_returns], no code
   making a call in tail position. But
   [code_id_actually_directly_called] reads every witness read off
   [s80], that of [f51] included, and finds one with any source: the
   call of the unit is not known to call [f50], is
   rebuilt by [FlambdaReaperRebuild.rebuild_apply], with its two
   arguments, and calls the rebuilt [f50], which takes one. The rebuilt
   unit goes wrong, whatever [preserve_direct_calls].

   So the refinement with changes of calling convention needs more of
   the table than [check_closed] and [check_cc_closed]: the witnesses
   read off a node with the same field have the same sources, as the
   rules derive the same of them ([check_apply_widgets], which [least]
   passes and [spurious] fails). *)
Module SpuriousWidgetExample.

Local Open Scope positive_scope.
Import FlambdaReaperCallingConventions.Examples.

Definition s80 : node := Node_symbol 80.

Definition code_50 : code :=
  Code [40; 42] 41 3 4
    (make_code_metadata [Singleton Value; Singleton Value] [Singleton Value])
    (return_to 3 [Var 40]).

Definition call_50 (args : list simple) (k k_exn : continuation)
    : apply_expr :=
  {| ap_callee := Symbol 80; ap_call_kind := Direct 50; ap_args := args;
     ap_continuation := k; ap_exn_continuation := k_exn;
     ap_exn_extra_args := [];
     ap_args_arity := [Singleton Value; Singleton Value];
     ap_return_arity := [Singleton Value] |}.

Definition code_51 : code :=
  Code [43] 44 5 6 (make_code_metadata [Singleton Value] [Singleton Value])
    (Let_cont_non_recursive 7
       (Handler [45] (make_handler_metadata [Value]) (return_to 5 [Var 45]))
       (Apply (call_50 [Var 43; Var 43] 7 6))).

Definition unit_expr : expr :=
  Let_static
    [Bind_code 50 code_50; Bind_code 51 code_51;
     Bind_symbol 80 (Static_const_closure_of_set 70 [(70, 80)] [(70, 50)] [])]
    (Let_cont_non_recursive 20
       (Handler [30] (make_handler_metadata [Value]) (return_to 1 [Var 30]))
       (Apply
          (call_50 [Const (Tagged_immediate 9); Const (Tagged_immediate 7)]
             20 2))).

Definition unit_facts : facts := traverse_unit Auto interface unit_expr.

Definition least : tables :=
  {| t_alias :=
       [(m 27, m 8); (m 5, m 26); (m 23, m 25); (m 1, v 30); (m 12, v 45);
        (v 44, m 14); (v 43, m 15); (m 3, v 40); (v 41, m 5); (v 40, m 6);
        (v 42, m 7); (m 20, m 8); (m 25, m 8); (v 30, m 3); (m 2, m 4);
        (m 24, c 50); (m 6, consts); (m 7, consts); (m 5, s80)];
     t_any_usage :=
       [m 1; m 24; m 19; m 2; v 30; m 3; m 4; c 50; v 40; m 6; consts];
     t_any_source :=
       [c 51; c 50; consts; m 24; m 6; m 7; v 40; v 42; m 3; v 30; m 1];
     t_has_usage :=
       [m 1; m 24; m 19; m 2; v 30; m 23; m 18; m 25; m 8; m 3; m 4; c 50; v 40;
        m 6; consts; s80];
     t_has_source :=
       [c 51; c 50; consts; s80; m 17; m 16; m 10; m 11; m 9; m 8;
        m 27; m 20; m 25; m 24; m 6; m 7; m 23; v 40; v 42; m 5; m 3; v 41;
        v 30; m 1];
     t_usages :=
       [(s80, s80); (m 23, m 23); (m 18, m 18);
        (m 25, m 23); (m 8, m 23)];
     t_sources :=
       [(s80, s80); (m 17, m 17); (m 16, m 16);
        (m 10, m 10); (m 9, m 9); (m 11, m 11); (m 8, m 8); (m 27, m 8);
        (m 20, m 8); (m 25, m 8); (m 23, m 8); (m 5, s80);
        (v 41, s80)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used :=
       [(s80, known); (m 8, result 0);
        (m 8, Field_return_of_call Exn);
        (m 8, Field_code_id_of_call_witness)] |}.

Definition spurious : tables :=
  {| t_alias := t_alias least; t_any_usage := t_any_usage least;
     t_any_source := m 20 :: t_any_source least;
     t_has_usage := t_has_usage least; t_has_source := t_has_source least;
     t_usages := t_usages least; t_sources := t_sources least;
     t_reading_field := t_reading_field least;
     t_escaping_field := t_escaping_field least;
     t_field_used := t_field_used least |}.

(* The rebuilt unit, with no code in the table. *)
Definition rebuilt (T : tables) (preserve_direct_calls : bool) : expr :=
  rebuild_unit_cc_from_tables unit_facts T []
    (unit_code_deps Auto interface unit_expr) preserve_direct_calls
    unit_expr.

End SpuriousWidgetExample.

Definition spurious_widget_counterexample_statement : Prop :=
  in_flow_fragment SpuriousWidgetExample.unit_expr = true
  ∧ static_constants (unit_codes SpuriousWidgetExample.unit_expr)
      SpuriousWidgetExample.unit_expr
    = true
  ∧ cc_fragment SpuriousWidgetExample.unit_expr = true
  ∧ results_kinded [] [] SpuriousWidgetExample.unit_expr = true
  ∧ well_formed_unit Examples.interface SpuriousWidgetExample.unit_expr
    = true
  ∧ exits_one_value Examples.interface SpuriousWidgetExample.unit_expr
    = true
  ∧ check_closed Examples.no_local SpuriousWidgetExample.unit_facts
      SpuriousWidgetExample.least
    = true
  ∧ check_field_used_exact Examples.no_local SpuriousWidgetExample.unit_facts
      SpuriousWidgetExample.least
    = true
  ∧ derive_cannot_change SpuriousWidgetExample.unit_facts
      SpuriousWidgetExample.least
    = []
  ∧ behaves
      (State (SpuriousWidgetExample.rebuilt SpuriousWidgetExample.least true)
         [] [] empty_static_env empty_heap no_reads)
      (Exits 1%positive [Value_const (Tagged_immediate 9)] empty_heap)
  ∧ check_closed Examples.no_local SpuriousWidgetExample.unit_facts
      SpuriousWidgetExample.spurious
    = true
  ∧ check_field_used_exact Examples.no_local SpuriousWidgetExample.unit_facts
      SpuriousWidgetExample.spurious
    = true
  ∧ check_cc_closed SpuriousWidgetExample.unit_facts
      SpuriousWidgetExample.spurious []
    = true
  ∧ check_cc_returns SpuriousWidgetExample.unit_facts
      SpuriousWidgetExample.spurious []
      (unit_code_deps Auto Examples.interface SpuriousWidgetExample.unit_expr)
      SpuriousWidgetExample.unit_expr
    = true
  ∧ check_apply_widgets SpuriousWidgetExample.unit_facts
      SpuriousWidgetExample.least
    = true
  ∧ check_apply_widgets SpuriousWidgetExample.unit_facts
      SpuriousWidgetExample.spurious
    = false
  ∧ behaves
      (State SpuriousWidgetExample.unit_expr [] [] empty_static_env
         empty_heap no_reads)
      (Exits 1%positive [Value_const (Tagged_immediate 9)] empty_heap)
  ∧ ∀ preserve_direct_calls : bool,
      ~ program_refines_observably SpuriousWidgetExample.unit_expr
          (SpuriousWidgetExample.rebuilt SpuriousWidgetExample.spurious
             preserve_direct_calls).

(* The unit [two_calls] meets the premises of
   [rebuild_cc_refines_observably_statement] (with
   [two_calls_example_statement] and [cc_fragment_examples_statement]):
   its rebuild, in which code changes its calling convention, refines
   it. So do the units of [poison_result_example_statement], where a call
   in tail position in code that keeps its calling convention returns
   through the wrapper of [make_apply_wrapper_cc], and of
   [masked_return_example_statement], where one in code that changes
   returns to its return continuation directly. *)
Definition two_calls_refines_statement : Prop :=
  program_refines_observably Examples.two_calls
    (rebuild_unit_cc_from_tables Examples.two_calls_facts
       Examples.two_calls_tables []
       (unit_code_deps Auto Examples.interface Examples.two_calls) true
       Examples.two_calls).

Definition poison_result_refines_statement : Prop :=
  program_refines_observably PoisonResultExample.unit_expr
    (rebuild_unit_cc_from_tables PoisonResultExample.unit_facts
       PoisonResultExample.solution [Node_code_id 52%positive]
       (unit_code_deps Auto Examples.interface PoisonResultExample.unit_expr)
       true PoisonResultExample.unit_expr).

Definition masked_return_refines_statement : Prop :=
  program_refines_observably MaskedReturnExample.unit_expr
    (rebuild_unit_cc_from_tables MaskedReturnExample.unit_facts
       MaskedReturnExample.solution []
       (unit_code_deps Auto Examples.interface MaskedReturnExample.unit_expr)
       true MaskedReturnExample.unit_expr).

(* The same of the unit of [tail_escape_example_statement], where a call
   in tail position in code that changes calls code that does not. *)
Definition tail_escape_refines_statement : Prop :=
  program_refines_observably TailEscapeExample.unit_expr
    (rebuild_unit_cc_from_tables TailEscapeExample.unit_facts
       TailEscapeExample.solution [Node_code_id 51%positive]
       (unit_code_deps Auto Examples.interface TailEscapeExample.unit_expr)
       true TailEscapeExample.unit_expr).
