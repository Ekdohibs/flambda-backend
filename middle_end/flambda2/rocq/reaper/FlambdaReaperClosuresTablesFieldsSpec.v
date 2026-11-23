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

(* The checks of the fields of a representation in the fragment of the
   refinement across changes of representation ([repr_field_ok] of
   FlambdaReaperUnboxingRefinesClosuresSpec.v, which
   FlambdaReaperClosuresTablesSpec.changed_set_frag_ok still asks) that
   follow from the tables of the unboxing analysis, and the refinement
   with a fragment that asks only the others.

   This file is to be reviewed: [repr_field_frag_ok],
   [changed_set_frag2_ok], [repr_expr_with] (a copy of
   FlambdaReaperClosuresTablesSpec.repr_expr_frag that takes the check
   of a set of closures that changes as an argument),
   [repr_unboxing_frag2_nr], [analysis_tables], [ChangedInSlot],
   [CyclicInSlot] and the statements. The
   proofs are in FlambdaReaperClosuresTablesFieldsProofs.v, which needs
   no review.

   The tables: [to_unbox] and [tcr] as the rules of the unboxing analysis
   with the fix give them ([check_to_change_representation_fixed], which
   asks [check_to_unbox_fixed] of [to_unbox]; the compiler has the fix),
   the closures of a set unboxed and changed alike ([check_sets_whole]),
   [dominated] the derived relation ([check_dominated]), and a least
   solution of the traversal with [Auto] of a unit of the flow fragment,
   well formed and binding each variable once. For the representation
   [r] of the variable [x] of a [Let x _ (Closure_of_set slot decls vs)]:
   - [changed_field_used_statement]: each field of [r] is a value slot
     [w] with a use in a closure of the set ([s_field_used]). The field
     has a constructor at the allocation of [tcr] whose set gives [r]
     ([get_single_field_source] has a source), which holds no field of a
     block ([cannot_change_representation]), and is read
     ([get_fields_of_constructors]) from a closure of that set, which the
     rules make a use of the field; [x] is a closure of that set
     (FlambdaReaperTypesTraversalProofs.dominated_self), and the
     closures of a set hold the same function slots
     ([function_slots_of_sets] of the traversal).
   - [changed_field_kept_statement]: if [vs] has the slot [w], at [a],
     [a] is a constant or a name with a use ([kept_name]): the closures
     of the set hold at [w] what [x] holds, [a]
     (FlambdaReaperClosuresExactProofs.closure_node_exact and the edges
     of the traversal), and the rules give a use to what a field with a
     use holds.
   - [unboxed_changed_disjoint_statement]: no name is both unboxed and
     changed: it would be dominated by an allocation to unbox and by a
     closure of a set of [tcr], one allocation, and a set is unboxed or
     changed whole.
   - [changed_in_slot_counterexample_statement]: a field not unboxed may
     hold a changed closure, so [plain_c] of it is false of the
     construction. In [ChangedInSlot], the closure [x60], called through
     the slot 80 of the closure [x61], and [x61], called by the unit,
     both change their representation; the field 80 of that of [x61] is
     not unboxed and holds [x60]. The compiler rewrites the read of the
     slot, a name dominated by [x60], with the representation of [x60],
     as it does the closure that the new slot holds: only the proof needs
     the conjunct (its relation has no changed closure in a slot). The
     unit is not of the fragment for that reason too: the variable that
     reads the slot is changed and bound to no set.
   - [cyclic_in_slot_counterexample_statement]: a field unboxed may not
     be unboxed as the variable it holds is, so [unboxed_field_pairs] of
     the two is not derived. The model bounds the depth of an unboxing by
     [S (length to_unbox)] (FlambdaReaperUnboxing.mk_unboxed_fields), as
     a chain of allocations to unbox, each in a field of the one before,
     was taken not to exceed it; in [CyclicInSlot] a block made by a loop
     holds the block of the iteration before, a chain of one allocation.
     The slot, unboxed from the closure, is cut one level before the
     block, unboxed from itself. The compiler has no bound: both are
     unboxed as deep as the reads go, and the fields of the slot are
     among those of the block (by hand, not run). A fault of the bound of
     the model, not of the compiler.

   What is left as checks ([repr_field_frag_ok]): that [vs] has the slot
   of the field, holding a plain simple ([plain_c]) for a field not
   unboxed, and a variable unboxed alike ([unboxed_field_pairs]) for a
   field unboxed. That the slot is one of the set would need an invariant
   of the traversal on the value slots of the sets
   (FlambdaReaperClosuresExactProofs.sg_ok does not keep them). That the
   simple is not unboxed would need the inversion of the rules of a least
   solution (the reads of the slot of an unboxed name would be dominated
   by its allocation, and the field unboxed). Above, [plain_c] and
   [unboxed_field_pairs] are each false of the construction.

   [repr_unboxing_frag2_sound_statement]: for such a unit and tables,
   [repr_unboxing_frag2_nr] implies
   FlambdaReaperClosuresTablesSpec.repr_unboxing_frag_nr.
   [rebuild_repr_frag2_refines_cc_statement],
   [rebuild_repr_frag2_observably_statement] and
   [rebuild_repr_frag2_local_statement]: the [rebuild_repr_frag_]
   statements of FlambdaReaperClosuresTablesSpec.v with
   [repr_unboxing_frag2_nr] and the checks of the tables above. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaBindersSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperFlowSpec FlambdaReaperLeastSpec
  FlambdaReaperRebuild
  FlambdaReaperRebuildSpec FlambdaReaperCallingConventions
  FlambdaReaperUnboxing FlambdaUnusedStaticsSpec
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaReaperLocalFields FlambdaReaperUnboxingRefinesClosuresSpec
  FlambdaReaperClosuresTablesSpec.
From OxCamlFlambda2 Require FlambdaReaperCallingConventionsSpec
  FlambdaReaperLeast.
Import ListNotations.

(* ======================================================================== *)
(* The fragment                                                             *)
(* ======================================================================== *)

Section Fragment_frag2.

Variable unboxed : list (node * unboxed_fields).
Variable changed : list (node * closure_repr).

(* The conjuncts of [repr_field_ok] that are not derived, of a field of
   the representation of a set whose value slots are [vs]. A field other
   than a value slot is not asked anything: there is none. *)
Definition repr_field_frag_ok (vs : list (value_slot * simple))
    (fu : field * unboxed_field) : bool :=
  match fu with
  | (Field_value_slot w, u) =>
    match lookup w vs with
    | Some a =>
      match u with
      | Not_unboxed _ => plain_c unboxed changed a
      | Unboxed _ =>
        match a with
        | Var b =>
          match fields_of unboxed b with
          | Some fb => negb (is_none (unboxed_field_pairs Value u (Unboxed fb)))
          | None => false
          end
        | Symbol _ | Const _ => false
        end
      end
    | None => false
    end
  | _ => true
  end.

(* The conjuncts of [changed_set_frag_ok] that are not derived. *)
Definition changed_set_frag2_ok (x : variable) (kind : flambda_kind)
    (d : named) (body : expr) (r : closure_repr) : bool :=
  match d with
  | Closure_of_set _ _ vs =>
    kind_eqb kind Value
    && match projections_of_set x d body with [] => true | _ :: _ => false end
    && forallb (repr_field_frag_ok vs) (cr_fields r)
  | _ => false
  end.

End Fragment_frag2.

Section Fragment_with.

Variable F : facts.
Variable sol : solution.
Variable cannot_change : node -> bool.
Variable code_deps : finite_map code_dep.
Variable unit_code_ids : list code_id.
Variable codes : finite_map code_metadata.
Variable unboxed : list (node * unboxed_fields).
Variable changed : list (node * closure_repr).
Variable my_closures : finite_map variable.
Variable nr : code_id -> bool.
(* The check of a [Let] of a closure whose set changes. *)
Variable cs : variable -> flambda_kind -> named -> expr -> closure_repr -> bool.

(* [repr_expr_frag], with [cs] for [changed_set_frag_ok]. *)
Fixpoint repr_expr_with (kinds : finite_map flambda_kind)
    (C : finite_map (list decision * bool)) (e : expr) {struct e} : bool :=
  match e with
  | Let x kind d body =>
    kind_ok kind
    && match fields_of unboxed x, changed_name changed (Node_variable x) with
       | Some fields, None =>
         (unboxed_binding_ok unboxed sol kinds x kind d fields
          && no_changed changed d)
         || unboxed_slot_read_ok unboxed changed x kind d fields
       | None, None =>
         (plain_binding_ok unboxed x kind d && no_changed changed d)
         || plain_set_ok unboxed changed x d body
         || plain_slot_read_ok unboxed changed d
       | None, Some r => cs x kind d body r
       | Some _, Some _ => false
       end
    && repr_expr_with ((x, kind) :: kinds) C body
  | Let_static bindings body =>
    forallb repr_binding_with bindings && repr_expr_with kinds C body
  | Let_cont_non_recursive k h body =>
    let ds := handler_decisions true sol unboxed (handler_is_exn_handler h) h in
    repr_handler_with kinds C ds h
    && repr_expr_with kinds ((k, (ds, true)) :: C) body
  | Let_cont_recursive handlers body =>
    let C' :=
      map
        (fun '(k, h) =>
           (k, (handler_decisions true sol unboxed false h, true)))
        handlers
      ++ C
    in
    forallb
      (fun '(_, h) =>
         repr_handler_with kinds C' (handler_decisions true sol unboxed false h)
           h)
      handlers
    && repr_expr_with kinds C' body
  | Apply ap =>
    repr_call F sol cannot_change code_deps unit_code_ids codes unboxed changed
      my_closures nr C ap
  | Apply_cont ac => apply_cont_ok_c unboxed changed C ac
  | Switch scrutinee arms =>
    plain_c unboxed changed scrutinee
    && forallb (fun '(_, ac) => apply_cont_ok_c unboxed changed C ac) arms
  | Invalid => true
  end
with repr_handler_with (kinds : finite_map flambda_kind)
    (C : finite_map (list decision * bool)) (ds : list decision)
    (h : continuation_handler) {struct h} : bool :=
  match h with
  | Handler params metadata handler =>
    Nat.eqb (length params) (length (hm_params_kinds metadata))
    && forallb kind_ok (hm_params_kinds metadata)
    && forallb decided ds
    && forallb (fun x => plain_c unboxed changed (Var x)) params
    && repr_expr_with
         (rev (combine params (hm_params_kinds metadata)) ++ kinds) C handler
  end
with repr_binding_with (binding : static_binding) {struct binding} : bool :=
  match binding with
  | Bind_code id c => repr_code_with id c
  | Bind_deleted_code _ => true
  | Bind_symbol _ (Static_const_block _ fields) =>
    forallb (plain_c unboxed changed) fields
  | Bind_symbol _ _ => false
  end
with repr_code_with (id : code_id) (c : code) {struct c} : bool :=
  match c with
  | Code params my_closure ret exn metadata body =>
    match lookup id codes with
    | Some metadata' => code_metadata_eqb metadata metadata'
    | None => false
    end
    && match lookup id my_closures with
       | Some mc => Pos.eqb mc my_closure
       | None => false
       end
    && match
         code_change_u sol cannot_change code_deps unit_code_ids unboxed id
           metadata
       with
       | Changing_u pd rd =>
         forallb decided pd && forallb decided rd
         && Nat.eqb (length params) (length pd)
         && Nat.eqb (length pd) (length (unarize (cm_params_arity metadata)))
         && Nat.eqb (length rd) (length (unarize (cm_result_arity metadata)))
         && forallb (fun x => plain_c unboxed changed (Var x)) params
         && is_none
              (my_closure_decision_u cannot_change code_deps unit_code_ids
                 unboxed id)
         && is_none (fields_of unboxed my_closure)
         && negb (Pos.eqb ret exn)
         && repr_expr_with
              (rev ((my_closure, Value)
                    :: with_kinds params (unarize (cm_params_arity metadata))))
              [(ret, (rd, true)); (exn, ([], false))] body
       | Not_changing_u => false
       end
    && (negb (nr id)
        || negb (among ret (live_conts F sol code_deps codes body)))
  end.

End Fragment_with.

(* [repr_unboxing_frag_nr], with [changed_set_frag2_ok] for
   [changed_set_frag_ok]. *)
Definition repr_unboxing_frag2_nr (nr : code_id -> bool)
    (is_local : field -> bool) (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox tcr : list node) (e : expr)
    : bool :=
  let unboxed := fst (unit_unboxed_fields F T dominated to_unbox e) in
  let changed :=
    unit_changed_representations is_local F T dominated to_unbox tcr e
  in
  repr_expr_with F (solution_of_tables T) (fun n => mem_node n cannot_change)
    code_deps (static_names code_ids_bound e) (bound_code_metadata e) unboxed
    changed (bound_code_my_closures e) nr
    (changed_set_frag2_ok unboxed changed) [] [] e
  && check_unboxed_uses F T unboxed.

(* ======================================================================== *)
(* The tables                                                               *)
(* ======================================================================== *)

(* The tables of the unboxing analysis, with the fix, for the facts [F]:
   [dominated] derived, [to_unbox] and [tcr] as the rules give them, and
   the sets unboxed and changed whole. *)
Definition analysis_tables (is_local : field -> bool) (F : facts) (T : tables)
    (cannot_change : list node) (max_unbox_size : nat)
    (lambda_lifting : bool) (dominated : list (node * node))
    (to_unbox tcr : list node) : bool :=
  check_dominated T dominated
  && check_to_change_representation_fixed is_local F T cannot_change
       max_unbox_size lambda_lifting to_unbox tcr
  && check_sets_whole F T to_unbox tcr.

(* ======================================================================== *)
(* A changed closure in a slot                                              *)
(* ======================================================================== *)

(* The code 51 of the closure [x61] reads its slot 80, the closure [x60]
   of the code 50, and calls it; the unit calls [x61]. Both are called
   and neither escapes: both change their representation, and the field
   80 of [x61] is not unboxed, as [x60] is not. *)
Module ChangedInSlot.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition call (id : code_id) (callee : variable) (args : list simple)
    (k exn : continuation) : apply_expr :=
  {| ap_callee := Var callee; ap_call_kind := Direct id; ap_args := args;
     ap_continuation := k; ap_exn_continuation := exn;
     ap_exn_extra_args := [];
     ap_args_arity := [Singleton Value];
     ap_return_arity := [Singleton Value] |}.

Definition code_50 : code :=
  Code [40] 41 3 4 (make_code_metadata [Singleton Value] [Singleton Value])
    (return_to 3 [Var 40]).

Definition code_51 : code :=
  Code [42] 43 5 6 (make_code_metadata [Singleton Value] [Singleton Value])
    (Let 44 Value (Project_value_slot_from 71 80 (Var 43))
       (Apply (call 50 44 [Var 42] 5 6))).

Definition unit_expr : expr :=
  Let_static [Bind_code 50 code_50; Bind_code 51 code_51]
    (Let 60 Value (Closure_of_set 70 [(70, 50)] [])
       (Let 61 Value (Closure_of_set 71 [(71, 51)] [(80, Var 60)])
          (Apply (call 51 61 [int 5] 1 2)))).

Definition facts : facts := traverse_unit Auto interface unit_expr.

(* The least solution, as the model computes it. *)
Definition tables : tables :=
  match FlambdaReaperLeast.compute_least no_local facts 100 with
  | Some T => T
  | None => FlambdaReaperLeast.empty_tables
  end.

Definition dominated : list (node * node) := derive_dominated tables.

Definition to_unbox : list node :=
  derive_to_unbox_fixed no_local facts tables [] max_unbox_size false.

Definition tcr : list node :=
  derive_to_change_representation no_local facts tables [] max_unbox_size
    false to_unbox.

Definition unboxed : list (node * unboxed_fields) :=
  fst (unit_unboxed_fields facts tables dominated to_unbox unit_expr).

Definition reprs : list (node * closure_repr) :=
  unit_changed_representations no_local facts tables dominated to_unbox tcr
    unit_expr.

End ChangedInSlot.

(* ======================================================================== *)
(* A block of a loop in a slot                                              *)
(* ======================================================================== *)

(* The block [x10], made by the loop 21 of field 0 its parameter [x30],
   which only [x10] is given (the loop is entered from the handler 20,
   never applied, whose parameter has no source), is in the slot 80 of the
   closure [x61]. The code 51 reads the slot, field 0 of it, field 0 of
   that and field 1 of that. [x10] is the one allocation to unbox, so that
   the unboxing has [S (length to_unbox)] = 2 levels: the slot, unboxed
   from the closure, has one for [x10], and [x10], unboxed from itself,
   two; the field 0 of field 0 is a variable in the first and unboxed in
   the second. *)
Module CyclicInSlot.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition goto (k : continuation) (args : list simple) : apply_cont :=
  {| ac_continuation := k; ac_args := args; ac_trap_action := None |}.

Definition code_51 : code :=
  Code [42] 43 5 6 (make_code_metadata [Singleton Value] [Singleton Value])
    (Let 44 Value (Project_value_slot_from 71 80 (Var 43))
       (Let 45 Value (load 0 (Var 44))
          (Let 46 Value (load 0 (Var 45))
             (Let 47 Value (load 1 (Var 46))
                (Let 48 (Naked_number Kind_naked_immediate)
                   (Prim (Unary Untag_immediate) [Var 47])
                   (Switch (Var 48) [(0%Z, goto 5 [Var 42])])))))).

Definition loop : expr :=
  Let 10 Value (make_block [Var 30; int 7])
    (Let 61 Value (Closure_of_set 71 [(71, 51)] [(80, Var 10)])
       (Let_cont_non_recursive 22
          (Handler [] (make_handler_metadata [])
             (Apply (ChangedInSlot.call 51 61 [int 5] 1 2)))
          (Let_cont_non_recursive 23
             (Handler [] (make_handler_metadata [])
                (Apply_cont (goto 21 [Var 10])))
             (Switch (Const (Naked_immediate 0))
                [(0%Z, goto 22 []); (1%Z, goto 23 [])])))).

Definition unit_expr : expr :=
  Let_static [Bind_code 51 code_51]
    (Let_cont_non_recursive 20
       (Handler [31] (make_handler_metadata [Value])
          (Let_cont_recursive
             [(21, Handler [30] (make_handler_metadata [Value]) loop)]
             (Apply_cont (goto 21 [Var 31]))))
       Invalid).

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition tables : tables :=
  match FlambdaReaperLeast.compute_least no_local facts 200 with
  | Some T => T
  | None => FlambdaReaperLeast.empty_tables
  end.

Definition dominated : list (node * node) := derive_dominated tables.

Definition to_unbox : list node :=
  derive_to_unbox_fixed no_local facts tables [] max_unbox_size false.

Definition tcr : list node :=
  derive_to_change_representation no_local facts tables [] max_unbox_size
    false to_unbox.

Definition unboxed : list (node * unboxed_fields) :=
  fst (unit_unboxed_fields facts tables dominated to_unbox unit_expr).

Definition reprs : list (node * closure_repr) :=
  unit_changed_representations no_local facts tables dominated to_unbox tcr
    unit_expr.

(* The slot 80 in the representation of [x61], and [x10]. *)
Definition slot_80 : unboxed_field :=
  Unboxed
    [(Field_block 0 Value, Unboxed [(Field_block 0 Value, Not_unboxed 81)])].

Definition fields_10 : unboxed_fields :=
  [(Field_block 0 Value,
    Unboxed
      [(Field_block 0 Value, Unboxed [(Field_block 1 Value, Not_unboxed 67)]);
       (Field_block 1 Value, Not_unboxed 68)]);
   (Field_block 1 Value, Not_unboxed 69)].

End CyclicInSlot.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Definition changed_field_used_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (max_unbox_size : nat)
    (lambda_lifting : bool) (dominated : list (node * node))
    (to_unbox tcr : list node) (x : variable) (slot : function_slot)
    (decls : list (function_slot * code_id))
    (vs : list (value_slot * simple)) (r : closure_repr)
    (fu : field * unboxed_field),
    in_flow_fragment e = true -> well_formed_unit u e = true ->
    variables_bound_once e = true ->
    least_solution is_local (traverse_unit Auto u e) T ->
    analysis_tables is_local (traverse_unit Auto u e) T cannot_change
      max_unbox_size lambda_lifting dominated to_unbox tcr
    = true ->
    (x, Closure_of_set slot decls vs) ∈ let_bindings e ->
    Keyed.lookup node_eqb (Node_variable x)
      (unit_changed_representations is_local (traverse_unit Auto u e) T
         dominated to_unbox tcr e)
    = Some r ->
    fu ∈ cr_fields r ->
    ∃ (w : value_slot) (uf : unboxed_field),
      fu = (Field_value_slot w, uf)
      ∧ existsb
          (fun n => s_field_used (solution_of_tables T) n (Field_value_slot w))
          (set_closure_nodes (traverse_unit Auto u e) (Node_variable x))
        = true.

Definition changed_field_kept_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (max_unbox_size : nat)
    (lambda_lifting : bool) (dominated : list (node * node))
    (to_unbox tcr : list node) (x : variable) (slot : function_slot)
    (decls : list (function_slot * code_id))
    (vs : list (value_slot * simple)) (r : closure_repr) (w : value_slot)
    (uf : unboxed_field) (a : simple),
    in_flow_fragment e = true -> well_formed_unit u e = true ->
    variables_bound_once e = true ->
    least_solution is_local (traverse_unit Auto u e) T ->
    analysis_tables is_local (traverse_unit Auto u e) T cannot_change
      max_unbox_size lambda_lifting dominated to_unbox tcr
    = true ->
    (x, Closure_of_set slot decls vs) ∈ let_bindings e ->
    Keyed.lookup node_eqb (Node_variable x)
      (unit_changed_representations is_local (traverse_unit Auto u e) T
         dominated to_unbox tcr e)
    = Some r ->
    (Field_value_slot w, uf) ∈ cr_fields r ->
    lookup w vs = Some a ->
    kept_name (solution_of_tables T) a = true.

Definition unboxed_changed_disjoint_statement : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables)
    (cannot_change : list node) (max_unbox_size : nat)
    (lambda_lifting : bool) (dominated : list (node * node))
    (to_unbox tcr : list node) (e : expr) (n : node),
    analysis_tables is_local F T cannot_change max_unbox_size lambda_lifting
      dominated to_unbox tcr
    = true ->
    Keyed.lookup node_eqb n (fst (unit_unboxed_fields F T dominated to_unbox e))
    <> None ->
    Keyed.lookup node_eqb n
      (unit_changed_representations is_local F T dominated to_unbox tcr e)
    = None.

(* The unit is of the premises of the statements above, and the field 80
   of the representation of [x61], not unboxed, holds [x60], which is
   changed. *)
Definition changed_in_slot_counterexample_statement : Prop :=
  in_flow_fragment ChangedInSlot.unit_expr = true
  ∧ well_formed_unit FlambdaReaperUnboxing.Examples.interface
      ChangedInSlot.unit_expr
    = true
  ∧ variables_bound_once ChangedInSlot.unit_expr = true
  ∧ least_solution FlambdaReaperUnboxing.Examples.no_local
      ChangedInSlot.facts ChangedInSlot.tables
  ∧ derive_cannot_change ChangedInSlot.facts ChangedInSlot.tables = []
  ∧ analysis_tables FlambdaReaperUnboxing.Examples.no_local
      ChangedInSlot.facts ChangedInSlot.tables []
      FlambdaReaperUnboxing.Examples.max_unbox_size false
      ChangedInSlot.dominated ChangedInSlot.to_unbox ChangedInSlot.tcr
    = true
  ∧ Keyed.lookup node_eqb (Node_variable 61%positive) ChangedInSlot.reprs
    = Some (Closure_repr [(Field_value_slot 80, Not_unboxed 82)] [(71, 83)]
              71)%positive
  ∧ plain_c ChangedInSlot.unboxed ChangedInSlot.reprs (Var 60%positive)
    = false.

(* The unit is of the premises of the statements above, [x10] is the one
   allocation to unbox, and the slot 80 of the representation of [x61],
   which holds [x10], is not unboxed as [x10] is. *)
Definition cyclic_in_slot_counterexample_statement : Prop :=
  in_flow_fragment CyclicInSlot.unit_expr = true
  ∧ well_formed_unit FlambdaReaperUnboxing.Examples.interface
      CyclicInSlot.unit_expr
    = true
  ∧ variables_bound_once CyclicInSlot.unit_expr = true
  ∧ least_solution FlambdaReaperUnboxing.Examples.no_local
      CyclicInSlot.facts CyclicInSlot.tables
  ∧ derive_cannot_change CyclicInSlot.facts CyclicInSlot.tables = []
  ∧ analysis_tables FlambdaReaperUnboxing.Examples.no_local
      CyclicInSlot.facts CyclicInSlot.tables []
      FlambdaReaperUnboxing.Examples.max_unbox_size false
      CyclicInSlot.dominated CyclicInSlot.to_unbox CyclicInSlot.tcr
    = true
  ∧ CyclicInSlot.to_unbox = [Node_variable 10%positive]
  ∧ Keyed.lookup node_eqb (Node_variable 61%positive) CyclicInSlot.reprs
    = Some (Closure_repr [(Field_value_slot 80, CyclicInSlot.slot_80)]
              [(71, 82)] 71)%positive
  ∧ fields_of CyclicInSlot.unboxed 10%positive = Some CyclicInSlot.fields_10
  ∧ unboxed_field_pairs Value CyclicInSlot.slot_80
      (Unboxed CyclicInSlot.fields_10)
    = None.

Definition repr_unboxing_frag2_sound_statement : Prop :=
  ∀ (nr : code_id -> bool) (is_local : field -> bool) (u : unit_interface)
    (e : expr) (T : tables) (cannot_change : list node)
    (max_unbox_size : nat) (lambda_lifting : bool)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox tcr : list node),
    in_flow_fragment e = true -> well_formed_unit u e = true ->
    variables_bound_once e = true ->
    least_solution is_local (traverse_unit Auto u e) T ->
    analysis_tables is_local (traverse_unit Auto u e) T cannot_change
      max_unbox_size lambda_lifting dominated to_unbox tcr
    = true ->
    repr_unboxing_frag2_nr nr is_local (traverse_unit Auto u e) T
      cannot_change code_deps dominated to_unbox tcr e
    = true ->
    repr_unboxing_frag_nr nr is_local (traverse_unit Auto u e) T cannot_change
      code_deps dominated to_unbox tcr e
    = true.

Definition rebuild_repr_frag2_refines_cc_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (max_unbox_size : nat)
    (lambda_lifting : bool) (dominated : list (node * node))
    (to_unbox tcr : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    well_formed_unit u e = true ->
    variables_bound_once e = true ->
    least_solution is_local (traverse_unit Auto u e) T ->
    check_closed is_local (traverse_unit Auto u e) T = true ->
    analysis_tables is_local (traverse_unit Auto u e) T cannot_change
      max_unbox_size lambda_lifting dominated to_unbox tcr
    = true ->
    repr_unboxing_frag2_nr nr is_local (traverse_unit Auto u e) T
      cannot_change (unit_code_deps Auto u e) dominated to_unbox tcr e
    = true ->
    check_unboxed_reads (traverse_unit Auto u e) T dominated to_unbox = true ->
    program_refines
      (rebuild_unit_cc_from_tables (traverse_unit Auto u e) T cannot_change
         (unit_code_deps Auto u e) preserve_direct_calls e)
      (rebuild_unit_unbox_repr_fixed is_local (traverse_unit Auto u e) T
         cannot_change (unit_code_deps Auto u e) dominated to_unbox tcr
         preserve_direct_calls e).

Definition rebuild_repr_frag2_observably_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (max_unbox_size : nat)
    (lambda_lifting : bool) (dominated : list (node * node))
    (to_unbox tcr : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    variables_bound_once e = true ->
    exits_one_value u e = true ->
    least_solution is_local (traverse_unit Auto u e) T ->
    check_closed is_local (traverse_unit Auto u e) T = true ->
    check_field_used_exact is_local (traverse_unit Auto u e) T = true ->
    check_cc_closed (traverse_unit Auto u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit Auto u e) T
    = true ->
    analysis_tables is_local (traverse_unit Auto u e) T cannot_change
      max_unbox_size lambda_lifting dominated to_unbox tcr
    = true ->
    repr_unboxing_frag2_nr nr is_local (traverse_unit Auto u e) T
      cannot_change (unit_code_deps Auto u e) dominated to_unbox tcr e
    = true ->
    check_unboxed_reads (traverse_unit Auto u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_repr_fixed is_local (traverse_unit Auto u e) T
         cannot_change (unit_code_deps Auto u e) dominated to_unbox tcr
         preserve_direct_calls e).

(* The same with local fields: the tables pass the check of the fix
   too (FlambdaReaperLocalFieldsSpec.v). *)
Definition rebuild_repr_frag2_local_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (max_unbox_size : nat)
    (lambda_lifting : bool) (dominated : list (node * node))
    (to_unbox tcr : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    variables_bound_once e = true ->
    exits_one_value u e = true ->
    least_solution is_local (traverse_unit Auto u e) T ->
    check_closed is_local (traverse_unit Auto u e) T = true ->
    check_local_reads_used is_local (traverse_unit Auto u e) T = true ->
    check_field_used_exact is_local (traverse_unit Auto u e) T = true ->
    check_cc_closed (traverse_unit Auto u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit Auto u e) T
    = true ->
    analysis_tables is_local (traverse_unit Auto u e) T cannot_change
      max_unbox_size lambda_lifting dominated to_unbox tcr
    = true ->
    repr_unboxing_frag2_nr nr is_local (traverse_unit Auto u e) T
      cannot_change (unit_code_deps Auto u e) dominated to_unbox tcr e
    = true ->
    check_unboxed_reads (traverse_unit Auto u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_repr_fixed is_local (traverse_unit Auto u e) T
         cannot_change (unit_code_deps Auto u e) dominated to_unbox tcr
         preserve_direct_calls e).
