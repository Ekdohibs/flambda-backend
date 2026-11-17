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

(* The rebuild with unboxing refines its unit across the changes of
   representation of sets of closures, and what can be said of the
   unboxing of closures ([Unbox_my_closure]) on this machine.

   This file is to be reviewed: the fragment ([repr_unboxing]), the
   statements and the examples, against FlambdaReaperUnboxing.v. The
   proofs are in FlambdaReaperUnboxingRefinesClosuresProofs.v and
   FlambdaReaperUnboxingRefinesClosuresSimProofs.v, which need no review.

   [rebuild_repr_fixed_refines_cc_statement]: for a unit [e] of the
   fragment of FlambdaReaperFlowSpec.v ([in_flow_fragment]), tables [T]
   closed under the rules ([check_closed]), any table of
   [cannot_change_calling_convention] ([cannot_change]) and of
   [to_change_representation] ([tcr]), the unit rebuilt with unboxing and
   changes of representation in the mode of the fix
   ([rebuild_unit_unbox_repr_fixed]) refines the unit rebuilt with the
   change of calling conventions alone, from the same table
   ([rebuild_unit_cc_from_tables]), run on its own ([program_refines]),
   if the unit is of [repr_unboxing] and the reads that the rebuild keeps
   of the names unboxed find their fields ([check_unboxed_reads]). The
   two rebuilt terms are the same but for the unboxing of blocks and
   boxed numbers (as in FlambdaReaperUnboxingRefinesSpec.v), the sets
   whose representation changes, made in the second with the new slots
   ([rewrite_set_of_closures]: a new value slot for each field of the
   representation, holding what the old slot held or, for a block
   unboxed in turn, the variables of its fields; and the new function
   slots), and the reads of their value slots, at the new slots
   ([rewrite_field_access_chg_repr], and
   [load_field_from_value_which_is_being_unboxed] for a value read that
   is unboxed). A closure of the second run is not the closure of the
   first: the relation of the proof relates the two, slot for slot. Code
   called through a closure whose representation changes runs with its
   [my_closure] bound to the closure of its run, and reads the new slots
   of it.
   [rebuild_repr_fixed_refines_observably_statement]: with the premises
   of
   FlambdaReaperCallingConventionsSpec.rebuild_cc_refines_observably_statement,
   the rebuilt unit refines the unit up to observation. It is that
   statement composed with the first; no closure whose representation
   changes reaches an exit, in this fragment, so that exits compare
   exactly. The same two statements hold of the variant of the fix
   ([rebuild_repr_fixed_tail_refines_cc_statement],
   [rebuild_repr_fixed_tail_refines_observably_statement]). The [_local]
   versions of the two statements up to observation hold with local
   fields: they ask the check of the fix of local fields
   ([check_local_reads_used], FlambdaReaperLocalFieldsSpec.v) in place
   of no local field or no projection, and compose the [_local]
   statement of FlambdaReaperCallingConventionsSpec.v with the first.
   The statements with the old premise are their corollaries.

   The fragment ([repr_unboxing]), for the unboxed fields that the tables
   give ([unit_unboxed_fields]), the changes of representation
   ([unit_changed_representations]), the solution and the table:
   - Names are plain, unboxed, or changed (a name that the changes of
     representation give a representation, [changed_name]), and no name
     is both unboxed and changed. A plain name is neither ([plain_c]).
     Every kind is a [Value] or a naked number ([kind_ok]).
   - An unboxed variable is bound as in FlambdaReaperUnboxingRefinesSpec.v
     ([unboxed_binding_ok]), of names that are not changed
     ([no_changed]), or to the read of a value slot of a changed closure
     whose representation unboxes that slot alike
     ([unboxed_slot_read_ok]).
   - A plain variable is bound to a simple or a primitive of plain
     simples, or to the read of a field of an unboxed name
     ([plain_binding_ok], [no_changed]); to a set of closures that does
     not change, with plain value slots and no other closure projected
     ([plain_set_ok]); or to the read of a value slot of a changed
     closure that the representation keeps as a slot
     ([plain_slot_read_ok]).
   - A changed variable is bound to a set of closures, at the function
     slot of its representation, whose other closures are not projected
     and whose new function slot finds the code of the old
     ([changed_set_ok]). Each field of the representation is a value slot
     of the set that the first rebuild keeps (it has a use in a closure
     of the set) and that holds a plain simple with a use, or, if the
     field is unboxed in turn, a used name unboxed alike; the new slots
     are distinct ([repr_field_ok]). These are checks of the tables and
     of the representation, which the proof does not derive from the
     rules; the tables of the examples pass them.
   - Continuations: a handler keeps or deletes each of its parameters,
     plain, with no unboxing and no parameter kept with no use (its
     decisions are [Decided]: the change of calling conventions keeps the
     same). The continuations in scope are described by those decisions
     and by whether they may be applied: a handler and the return
     continuation of code may be, the exception continuation of code is
     only that of the calls in its body. An [Apply_cont] gives plain
     simples to a continuation that may be applied, or out of the unit.
   - Code ([repr_code]) changes its calling convention, with decisions
     that keep or delete each parameter and result ([Decided]), keeps its
     [my_closure] ([my_closure_decision_u] is [None]), which is not
     unboxed, and has plain parameters, as many as its decisions.
   - Calls ([repr_call]): direct, of the code they are known to call, at
     the arities of that code, with plain arguments and no extra argument
     for the exception continuation, by a variable that is plain if the
     [my_closure] of the code is, and changed with the same
     representation if that is changed; they return to a continuation
     whose decisions are those of the results of the callee (to a way out
     of the unit, if the callee keeps all of them), so that neither
     rebuild makes a wrapper.
   - Static constants are code of the fragment and blocks of plain
     simples.
   - [check_unboxed_uses] of FlambdaReaperUnboxingRefinesCallsSpec.v.

   What is left out, for the proof: unboxed parameters, parameters kept
   with no use, wrappers, raising from code, code that keeps its calling
   convention, calls of unknown arity, projections of other closures of
   a set, static sets whose representation changes (the static
   environments of the two runs would differ: [static_set_example]
   runs it), and closures that are unboxed (lambda lifting, below).

   [block_in_slot_example_statement]: the unit
   FlambdaReaperUnboxing.Examples.block_in_slot (a block in the slot of a
   closure whose set changes its representation, read by the code
   through its [my_closure]), with the tables of the solver, is of the
   fragment; the rebuild in the mode of the fix is the term of
   [rebuild_block_in_slot]; it refines the unit up to observation.
   [two_fields_example_statement]: the same of a unit that unboxes a
   block of two fields into two new slots, read and added by the code,
   as unbox_block_in_value_slot.fl does.

   The unboxing of closures ([Unbox_my_closure], lambda lifting). The
   rebuild makes a call of code that takes the fields of its closure as
   parameters a call with no callee in the OCaml, poison here: the
   machine runs a call by the closure that its callee is, and finds a
   call of poison undefined. So no unit that runs such a call refines on
   this machine: [unboxed_closure_goes_wrong_statement] (the unit
   FlambdaReaperUnboxing.Examples.unboxed_closure returns 7, and its
   rebuild goes wrong). What can be said is said of a machine extended
   with calls with no callee ([step_lifted]: a direct call whose callee
   is poison runs the code named, with its [my_closure] bound to the
   poison), which is the machine of the compiler, where such code never
   reads its [my_closure]: [unboxed_closure_lifted_statement], the
   rebuilt unit, run on that machine, exits as the unit does on this
   one. The statement for every unit is left for when the machine has
   calls with no callee (a request to the core): it would relate a
   closure of the first run to the variables of its fields in the
   second, as an unboxed block is, and the [my_closure] of code that
   unboxes it to the parameters of its fields. Stating it of a machine
   defined here would duplicate the semantics and every lemma about it,
   for a statement the core is to replace.

   [static_set_example_statement]: the unit
   FlambdaReaperUnboxing.Examples.static_set, whose static set changes
   its representation, rebuilt in the mode of the fix, refines the unit
   up to observation (both return 7). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaBindersSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaUnusedStaticsSpec FlambdaReaperUnboxingRefinesSpec
  FlambdaReaperUnboxingRefinesCallsSpec FlambdaReaperLocalFields.
From OxCamlFlambda2 Require FlambdaReaperCallingConventionsSpec
  FlambdaReaperLeast.
Import ListNotations.

(* ======================================================================== *)
(* The fragment                                                             *)
(* ======================================================================== *)

(* Equality of unboxed fields and of representations. *)
Fixpoint unboxed_field_eqb (u u' : unboxed_field) {struct u} : bool :=
  match u, u' with
  | Not_unboxed x, Not_unboxed x' => Pos.eqb x x'
  | Unboxed fields, Unboxed fields' =>
    (fix go (l l' : list (field * unboxed_field)) : bool :=
       match l, l' with
       | [], [] => true
       | (f, u) :: l, (f', u') :: l' =>
         field_eqb f f' && unboxed_field_eqb u u' && go l l'
       | _, _ => false
       end) fields fields'
  | _, _ => false
  end.

Fixpoint slot_pairs_eqb (l l' : list (function_slot * function_slot))
    : bool :=
  match l, l' with
  | [], [] => true
  | (a, b) :: l, (a', b') :: l' =>
    Pos.eqb a a' && Pos.eqb b b' && slot_pairs_eqb l l'
  | _, _ => false
  end.

Definition closure_repr_eqb (r r' : closure_repr) : bool :=
  unboxed_field_eqb (Unboxed (cr_fields r)) (Unboxed (cr_fields r'))
  && slot_pairs_eqb (cr_function_slots r) (cr_function_slots r')
  && Pos.eqb (cr_current r) (cr_current r').

Definition repr_option_eqb (o o' : option closure_repr) : bool :=
  match o, o' with
  | None, None => true
  | Some r, Some r' => closure_repr_eqb r r'
  | _, _ => false
  end.

Definition option_code_eqb (o o' : option code_id) : bool :=
  match o, o' with
  | None, None => true
  | Some a, Some a' => Pos.eqb a a'
  | _, _ => false
  end.

(* A decision that keeps or deletes, with no unboxing and no use
   forgotten. *)
Definition decided (d : decision) : bool :=
  match d with
  | Decided _ => true
  | Unbox _ | Keep_unused _ => false
  end.

Definition keeps (d : decision) : bool :=
  match d with
  | Decided (Keep _) => true
  | _ => false
  end.

(* A parameter of a continuation and the result given it: both kept or
   both deleted. *)
Definition same_decision (d d' : decision) : bool :=
  match d, d' with
  | Decided (Keep _), Decided (Keep _) | Decided Delete, Decided Delete =>
    true
  | _, _ => false
  end.

(* What a defining expression reads. *)
Definition named_simples (d : named) : list simple :=
  match d with
  | Simple s => [s]
  | Prim _ args => args
  | Set_of_closures _ vs | Closure_of_set _ _ vs => map snd vs
  | Project_value_slot _ s | Project_value_slot_from _ _ s
  | Project_function_slot _ _ s => [s]
  end.

(* A simple whose value is that of the name in both runs: a constant, or
   a name with a use. *)
Definition kept_name (sol : solution) (s : simple) : bool :=
  match s with
  | Var z => s_has_usage sol (Node_variable z)
  | Symbol sym => s_has_usage sol (Node_symbol sym)
  | Const _ => true
  end.

Section Fragment.

Variable F : facts.
Variable sol : solution.
Variable cannot_change : node -> bool.
Variable code_deps : finite_map code_dep.
Variable unit_code_ids : list code_id.
Variable codes : finite_map code_metadata.
Variable unboxed : list (node * unboxed_fields).
Variable changed : list (node * closure_repr).
(* The [my_closure] of each code of the unit. *)
Variable my_closures : finite_map variable.

Definition changed_name (n : node) : option closure_repr :=
  changed_of_node changed n.

Definition not_changed (s : simple) : bool :=
  match s with
  | Var x => is_none (changed_name (Node_variable x))
  | Symbol sym => is_none (changed_name (Node_symbol sym))
  | Const _ => true
  end.

(* A simple that is neither unboxed nor changed. *)
Definition plain_c (s : simple) : bool :=
  plain_simple unboxed s && not_changed s.

Definition no_changed (d : named) : bool :=
  forallb not_changed (named_simples d).

(* The set of closures of a plain variable: value slots plain, and no
   other closure of the set projected. *)
Definition plain_set_ok (x : variable) (d : named) (body : expr) : bool :=
  match d with
  | Closure_of_set _ _ vs =>
    forallb (fun ws => plain_c (snd ws)) vs
    && match projections_of_set x d body with [] => true | _ :: _ => false end
  | _ => false
  end.

(* The read of a value slot of a changed closure [y] that is not unboxed,
   the representation giving the field [u]. *)
Definition slot_read_field (d : named) : option (variable * unboxed_field) :=
  match field_read d with
  | Some (Field_value_slot w, Var y) =>
    match fields_of unboxed y, changed_name (Node_variable y) with
    | None, Some r => option_map (fun u => (y, u))
                        (Keyed.lookup field_eqb (Field_value_slot w)
                           (cr_fields r))
    | _, _ => None
    end
  | _ => None
  end.

Definition plain_slot_read_ok (d : named) : bool :=
  match slot_read_field d with
  | Some (_, Not_unboxed _) => true
  | _ => false
  end.

Definition unboxed_slot_read_ok (x : variable) (kind : flambda_kind)
    (d : named) (fields : unboxed_fields) : bool :=
  kind_eqb kind Value
  && match slot_read_field d with
     | Some (y, Unboxed nested) =>
       negb (Pos.eqb x y) && fields_alike fields nested
     | _ => false
     end.

(* A field of the representation of the set of [x], whose value slots are
   [vs]. *)
Definition repr_field_ok (x : variable) (vs : list (value_slot * simple))
    (fu : field * unboxed_field) : bool :=
  match fu with
  | (Field_value_slot w, u) =>
    existsb (fun n => s_field_used sol n (Field_value_slot w))
      (set_closure_nodes F (Node_variable x))
    && match lookup w vs with
       | Some a =>
         match u with
         | Not_unboxed _ => plain_c a && kept_name sol a
         | Unboxed _ =>
           match a with
           | Var b =>
             match fields_of unboxed b with
             | Some fb =>
               s_has_usage sol (Node_variable b) && not_changed (Var b)
               && negb (is_none (unboxed_field_pairs Value u (Unboxed fb)))
             | None => false
             end
           | Symbol _ | Const _ => false
           end
         end
       | None => false
       end
  | _ => false
  end.

Definition changed_set_ok (x : variable) (kind : flambda_kind) (d : named)
    (body : expr) (r : closure_repr) : bool :=
  match d with
  | Closure_of_set slot decls vs =>
    kind_eqb kind Value && Pos.eqb slot (cr_current r)
    && option_code_eqb (lookup (current_slot r) (rename_function_decls r decls))
         (lookup slot decls)
    && match projections_of_set x d body with [] => true | _ :: _ => false end
    && distinct (map fst (fields_leaves (cr_fields r)))
    && forallb (repr_field_ok x vs) (cr_fields r)
  | _ => false
  end.

(* An [Apply_cont] to [C]'s continuations. *)
Definition apply_cont_ok_c (C : finite_map (list decision * bool))
    (ac : apply_cont) : bool :=
  forallb plain_c (ac_args ac)
  && match lookup (ac_continuation ac) C with
     | Some (_, false) => false
     | Some (_, true) | None => true
     end.

(* The continuation [k] of a call, given the decisions [rd] of the
   results of the callee. *)
Definition returns_ok (C : finite_map (list decision * bool))
    (k : continuation) (rd : list decision) : bool :=
  match lookup k C with
  | Some (ds, true) => forallb2 same_decision ds rd
  | Some (_, false) => false
  | None => forallb keeps rd
  end.

Definition repr_call (C : finite_map (list decision * bool))
    (ap : apply_expr) : bool :=
  match ap_call_kind ap, ap_callee ap, code_id_actually_called F sol ap with
  | Direct id, Var f, Some id' =>
    Pos.eqb id id'
    && match lookup id codes, lookup id my_closures with
       | Some metadata, Some mc =>
         match
           code_change_u sol cannot_change code_deps unit_code_ids unboxed id
             metadata
         with
         | Changing_u pd rd =>
           forallb decided pd && forallb decided rd
           && is_none
                (my_closure_decision_u cannot_change code_deps unit_code_ids
                   unboxed id)
           && repr_option_eqb (changed_name (Node_variable f))
                (changed_name (Node_variable mc))
           && is_none (fields_of unboxed f)
           && arity_eqb (ap_args_arity ap) (cm_params_arity metadata)
           && arity_eqb (ap_return_arity ap) (cm_result_arity metadata)
           && Nat.eqb (length rd) (length (unarize (cm_result_arity metadata)))
           && forallb plain_c (ap_args ap)
           && match ap_exn_extra_args ap with [] => true | _ :: _ => false end
           && returns_ok C (ap_continuation ap) rd
         | Not_changing_u => false
         end
       | _, _ => false
       end
  | _, _, _ => false
  end.

Fixpoint repr_expr (kinds : finite_map flambda_kind)
    (C : finite_map (list decision * bool)) (e : expr) {struct e} : bool :=
  match e with
  | Let x kind d body =>
    kind_ok kind
    && match fields_of unboxed x, changed_name (Node_variable x) with
       | Some fields, None =>
         (unboxed_binding_ok unboxed sol kinds x kind d fields && no_changed d)
         || unboxed_slot_read_ok x kind d fields
       | None, None =>
         (plain_binding_ok unboxed x kind d && no_changed d)
         || plain_set_ok x d body || plain_slot_read_ok d
       | None, Some r => changed_set_ok x kind d body r
       | Some _, Some _ => false
       end
    && repr_expr ((x, kind) :: kinds) C body
  | Let_static bindings body =>
    forallb repr_binding bindings && repr_expr kinds C body
  | Let_cont_non_recursive k h body =>
    let ds := handler_decisions true sol unboxed (handler_is_exn_handler h) h in
    repr_handler kinds C ds h && repr_expr kinds ((k, (ds, true)) :: C) body
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
         repr_handler kinds C' (handler_decisions true sol unboxed false h) h)
      handlers
    && repr_expr kinds C' body
  | Apply ap => repr_call C ap
  | Apply_cont ac => apply_cont_ok_c C ac
  | Switch scrutinee arms =>
    plain_c scrutinee && forallb (fun '(_, ac) => apply_cont_ok_c C ac) arms
  | Invalid => true
  end
with repr_handler (kinds : finite_map flambda_kind)
    (C : finite_map (list decision * bool)) (ds : list decision)
    (h : continuation_handler) {struct h} : bool :=
  match h with
  | Handler params metadata handler =>
    Nat.eqb (length params) (length (hm_params_kinds metadata))
    && forallb kind_ok (hm_params_kinds metadata)
    && forallb decided ds
    && forallb (fun x => plain_c (Var x)) params
    && repr_expr (rev (combine params (hm_params_kinds metadata)) ++ kinds) C
         handler
  end
with repr_binding (binding : static_binding) {struct binding} : bool :=
  match binding with
  | Bind_code id c => repr_code id c
  | Bind_deleted_code _ => true
  | Bind_symbol _ (Static_const_block _ fields) => forallb plain_c fields
  | Bind_symbol _ _ => false
  end
with repr_code (id : code_id) (c : code) {struct c} : bool :=
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
         && forallb (fun x => plain_c (Var x)) params
         && is_none
              (my_closure_decision_u cannot_change code_deps unit_code_ids
                 unboxed id)
         && is_none (fields_of unboxed my_closure)
         && negb (Pos.eqb ret exn)
         && repr_expr
              (rev ((my_closure, Value)
                    :: with_kinds params (unarize (cm_params_arity metadata))))
              [(ret, (rd, true)); (exn, ([], false))] body
       | Not_changing_u => false
       end
  end.

End Fragment.

(* The unit [e] is of the fragment, for the unboxing and the changes of
   representation that the tables give from its graph [F]. *)
Definition repr_unboxing (is_local : field -> bool) (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox tcr : list node) (e : expr)
    : bool :=
  let unboxed := fst (unit_unboxed_fields F T dominated to_unbox e) in
  repr_expr F (solution_of_tables T) (fun n => mem_node n cannot_change)
    code_deps (static_names code_ids_bound e) (bound_code_metadata e) unboxed
    (unit_changed_representations is_local F T dominated to_unbox tcr e)
    (bound_code_my_closures e) [] [] e
  && check_unboxed_uses F T unboxed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Definition rebuild_repr_fixed_refines_cc_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    repr_unboxing is_local (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox tcr e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines
      (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) preserve_direct_calls e)
      (rebuild_unit_unbox_repr_fixed is_local (traverse_unit flag u e) T
         cannot_change (unit_code_deps flag u e) dominated to_unbox tcr
         preserve_direct_calls e).

Definition rebuild_repr_fixed_tail_refines_cc_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    repr_unboxing is_local (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox tcr e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines
      (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) preserve_direct_calls e)
      (rebuild_unit_unbox_repr_fixed_tail is_local (traverse_unit flag u e) T
         cannot_change (unit_code_deps flag u e) dominated to_unbox tcr
         preserve_direct_calls e).

Definition rebuild_repr_fixed_refines_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox tcr : list node)
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
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    repr_unboxing is_local (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox tcr e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_repr_fixed is_local (traverse_unit flag u e) T
         cannot_change (unit_code_deps flag u e) dominated to_unbox tcr
         preserve_direct_calls e).

(* The same with local fields: the tables pass the check of the fix
   too (FlambdaReaperLocalFieldsSpec.v). *)
Definition rebuild_repr_fixed_refines_observably_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    repr_unboxing is_local (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox tcr e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_repr_fixed is_local (traverse_unit flag u e) T
         cannot_change (unit_code_deps flag u e) dominated to_unbox tcr
         preserve_direct_calls e).

Definition rebuild_repr_fixed_tail_refines_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox tcr : list node)
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
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    repr_unboxing is_local (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox tcr e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_repr_fixed_tail is_local (traverse_unit flag u e) T
         cannot_change (unit_code_deps flag u e) dominated to_unbox tcr
         preserve_direct_calls e).

(* The same with local fields: the tables pass the check of the fix
   too (FlambdaReaperLocalFieldsSpec.v). *)
Definition rebuild_repr_fixed_tail_refines_observably_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    repr_unboxing is_local (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox tcr e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_repr_fixed_tail is_local (traverse_unit flag u e) T
         cannot_change (unit_code_deps flag u e) dominated to_unbox tcr
         preserve_direct_calls e).

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

Import FlambdaReaperUnboxing.Examples.

(* A unit run on its own. *)
Definition on_its_own (e : expr) : state :=
  State e [] [] empty_static_env empty_heap no_reads.

(* The block in a slot of FlambdaReaperUnboxing.Examples, with its tables:
   the rebuild in the mode of the fix is that of the branch
   ([rebuild_block_in_slot] has the term). *)
Definition block_in_slot_example_statement : Prop :=
  repr_unboxing no_local block_in_slot_facts block_in_slot_tables []
    (unit_code_deps Auto interface block_in_slot) block_in_slot_dominated
    [v 10] [v 60] block_in_slot
  = true
  ∧ check_unboxed_reads block_in_slot_facts block_in_slot_tables
      block_in_slot_dominated [v 10]
    = true
  ∧ rebuild_unit_unbox_repr_fixed no_local block_in_slot_facts
      block_in_slot_tables [] (unit_code_deps Auto interface block_in_slot)
      block_in_slot_dominated [v 10] [v 60] true block_in_slot
    = rebuild_unit_unbox_repr no_local block_in_slot_facts
        block_in_slot_tables [] (unit_code_deps Auto interface block_in_slot)
        block_in_slot_dominated [v 10] [v 60] true block_in_slot
  ∧ program_refines_observably block_in_slot
      (rebuild_unit_unbox_repr_fixed no_local block_in_slot_facts
         block_in_slot_tables [] (unit_code_deps Auto interface block_in_slot)
         block_in_slot_dominated [v 10] [v 60] true block_in_slot).

(* The unit of unbox_block_in_value_slot.fl, with constants for the
   arguments of [run]: the code 54 reads the slot 80 of its closure, a
   block of two fields, adds them and its parameter. The block [x10] is
   unboxed, as the set it is stored in changes its representation; the
   two fields are the new slots 81 and 82, the function slot 83. The
   tables are the least solution, and the others are derived from them by
   the rules of the model ([tcr] has the two made nodes too, which are not
   closures and get no representation). The unit returns 47, and so does
   its rebuild. *)
Module TwoFieldsExample.

Local Open Scope positive_scope.

Definition add (a b : simple) : named :=
  Prim (Binary (Int_arith Standard_tagged_immediate Add)) [a; b].

Definition code_54 : code :=
  Code [40] 41 3 4 (make_code_metadata [Singleton Value] [Singleton Value])
    (Let 45 Value (Project_value_slot_from 70 80 (Var 41))
       (Let 46 Value (load 0 (Var 45))
          (Let 47 Value (load 1 (Var 45))
             (Let 48 Value (add (Var 46) (Var 47))
                (Let 49 Value (add (Var 48) (Var 40))
                   (return_to 3 [Var 49])))))).

Definition call (callee : simple) (args : list simple) : apply_expr :=
  {| ap_callee := callee; ap_call_kind := Direct 54; ap_args := args;
     ap_continuation := 1; ap_exn_continuation := 2;
     ap_exn_extra_args := [];
     ap_args_arity := [Singleton Value];
     ap_return_arity := [Singleton Value] |}.

Definition unit_expr : expr :=
  Let_static [Bind_code 54 code_54]
    (Let 10 Value (make_block [int 7; int 8])
       (Let 60 Value (Closure_of_set 70 [(70, 54)] [(80, Var 10)])
          (Apply (call (Var 60) [int 32])))).

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition tables : tables :=
  match FlambdaReaperLeast.compute_least no_local facts 40 with
  | Some T => T
  | None => FlambdaReaperLeast.empty_tables
  end.

Definition dominated : list (node * node) := derive_dominated tables.

Definition to_unbox : list node :=
  derive_to_unbox no_local facts tables [] max_unbox_size false.

Definition tcr : list node :=
  derive_to_change_representation no_local facts tables [] max_unbox_size
    false to_unbox.

Definition code_deps : finite_map code_dep :=
  unit_code_deps Auto interface unit_expr.

(* The rebuild: the block is the variables 63 and 64 of its fields, the
   set is made with the new slots, and the code reads them into the
   variables 61 and 62 of the fields of [x45]. *)
Definition rebuilt : expr :=
  Let_static
    [Bind_code 54
       (Code [40] 41 3 4
          {| cm_params_arity := [Unboxed_product [Singleton Value]];
             cm_result_arity := [Unboxed_product [Singleton Value]];
             cm_is_tupled := false;
             cm_is_stub := false |}
          (Let 62 Value (Project_value_slot_from 83 82 (Var 41))
             (Let 61 Value (Project_value_slot_from 83 81 (Var 41))
                (Let 46 Value (Simple (Var 61))
                   (Let 47 Value (Simple (Var 62))
                      (Let 48 Value (add (Var 46) (Var 47))
                         (Let 49 Value (add (Var 48) (Var 40))
                            (return_to 3 [Var 49]))))))))]
    (Let 64 Value (Simple (int 8))
       (Let 63 Value (Simple (int 7))
          (Let 60 Value
             (Closure_of_set 83 [(83, 54)] [(81, Var 63); (82, Var 64)])
             (Apply
                {| ap_callee := Var 60; ap_call_kind := Direct 54;
                   ap_args := [int 32]; ap_continuation := 1;
                   ap_exn_continuation := 2; ap_exn_extra_args := [];
                   ap_args_arity := [Unboxed_product [Singleton Value]];
                   ap_return_arity :=
                     [Unboxed_product [Singleton Value]] |})))).

End TwoFieldsExample.

Definition two_fields_example_statement : Prop :=
  check_dominated TwoFieldsExample.tables TwoFieldsExample.dominated = true
  ∧ check_to_unbox no_local TwoFieldsExample.facts TwoFieldsExample.tables []
      max_unbox_size false TwoFieldsExample.to_unbox
    = true
  ∧ check_to_change_representation no_local TwoFieldsExample.facts
      TwoFieldsExample.tables [] max_unbox_size false
      TwoFieldsExample.to_unbox TwoFieldsExample.tcr
    = true
  ∧ check_sets_whole TwoFieldsExample.facts TwoFieldsExample.tables
      TwoFieldsExample.to_unbox TwoFieldsExample.tcr
    = true
  ∧ repr_unboxing no_local TwoFieldsExample.facts TwoFieldsExample.tables []
      TwoFieldsExample.code_deps TwoFieldsExample.dominated
      TwoFieldsExample.to_unbox TwoFieldsExample.tcr TwoFieldsExample.unit_expr
    = true
  ∧ rebuild_unit_unbox_repr_fixed no_local TwoFieldsExample.facts
      TwoFieldsExample.tables [] TwoFieldsExample.code_deps
      TwoFieldsExample.dominated TwoFieldsExample.to_unbox
      TwoFieldsExample.tcr true TwoFieldsExample.unit_expr
    = TwoFieldsExample.rebuilt
  ∧ run 30 (on_its_own TwoFieldsExample.unit_expr)
    = Exit 1%positive [Value_const (Tagged_immediate 47)] empty_heap no_reads
  ∧ program_refines_observably TwoFieldsExample.unit_expr
      TwoFieldsExample.rebuilt.

(* The unboxing of closures: the unit unboxed_closure of
   FlambdaReaperUnboxing.Examples returns 7; its rebuild with lambda
   lifting, in the mode of the fix, calls the code that takes the field of
   its closure with poison for a callee, and goes wrong on this machine, so
   that it refines the unit on no machine that is this one. *)
Definition unboxed_closure_rebuilt : expr :=
  rebuild_unit_unbox_repr_fixed no_local unboxed_closure_facts
    unboxed_closure_tables [] (unit_code_deps Auto interface unboxed_closure)
    unboxed_closure_dominated [v 60] [m 2; m 4] true unboxed_closure.

Definition unboxed_closure_goes_wrong_statement : Prop :=
  run 10 (on_its_own unboxed_closure)
  = Exit 1%positive [Value_const (Tagged_immediate 7)] empty_heap no_reads
  ∧ run 10 (on_its_own unboxed_closure_rebuilt) = Undefined
  ∧ ~ program_refines_observably unboxed_closure unboxed_closure_rebuilt.

(* The machine with calls that have no callee: a direct call whose callee
   is poison, with no extra argument for its exception continuation, runs
   the code it names as [call_code] does, its [my_closure] bound to the
   poison; any other step is [step]. It is what the compiler does of a
   call with no callee ([rebuild_apply], [Unbox_my_closure]), whose code
   takes the fields of its closure as parameters. *)
Definition step_lifted (s : state) : step_result :=
  match s with
  | State (Apply ap) vars conts statics hp rd =>
    match ap_callee ap, ap_call_kind ap, ap_exn_extra_args ap with
    | Const (Poison kind), Direct id, [] =>
      match lookup id (static_code statics), eval_simples vars (ap_args ap)
      with
      | Some c, Some vs =>
        call_code (Value_const (Poison kind)) c vs (ap_args_arity ap)
          (ap_return_arity ap) (resolve (ap_continuation ap) conts)
          (resolve (ap_exn_continuation ap) conts) statics hp rd
      | _, _ => Undefined
      end
    | _, _, _ => step s
    end
  | _ => step s
  end.

Fixpoint run_lifted (fuel : nat) (s : state) : step_result :=
  match fuel with
  | O => Step s
  | S fuel =>
    match step_lifted s with
    | Step s' => run_lifted fuel s'
    | result => result
    end
  end.

(* On that machine the rebuilt unit returns 7, as the unit does on
   this one (and on that one: it makes no call with no callee). *)
Definition unboxed_closure_lifted_statement : Prop :=
  run_lifted 10 (on_its_own unboxed_closure_rebuilt)
  = Exit 1%positive [Value_const (Tagged_immediate 7)] empty_heap no_reads
  ∧ run_lifted 10 (on_its_own unboxed_closure)
    = Exit 1%positive [Value_const (Tagged_immediate 7)] empty_heap no_reads.

(* A static set whose representation changes: the unit static_set of
   FlambdaReaperUnboxing.Examples and its rebuild in the mode of the fix
   (that of [rebuild_static_set]) both return 7. *)
Definition static_set_rebuilt : expr :=
  rebuild_unit_unbox_repr_fixed no_local static_set_facts static_set_tables
    [] (unit_code_deps Auto interface static_set)
    (derive_dominated static_set_tables) [] [sym 90] true static_set.

Definition static_set_example_statement : Prop :=
  static_set_rebuilt
  = rebuild_unit_unbox_repr no_local static_set_facts static_set_tables []
      (unit_code_deps Auto interface static_set)
      (derive_dominated static_set_tables) [] [sym 90] true static_set
  ∧ run 10 (on_its_own static_set)
    = Exit 1%positive [Value_const (Tagged_immediate 7)] empty_heap no_reads
  ∧ run 10 (on_its_own static_set_rebuilt)
    = Exit 1%positive [Value_const (Tagged_immediate 7)] empty_heap no_reads
  ∧ program_refines_observably static_set static_set_rebuilt.
