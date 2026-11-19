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

(* The unboxing of the reaper, transcribed: which allocations are
   unboxed, the variables that stand for their fields, which closures
   change their representation, and the rebuild of the unit with them,
   the changes of calling convention included.

   This file transcribes, in middle_end/flambda2/reaper/:
   - points_to_analysis.ml: the rules of [post_processing_rules] that
     conclude [multiple_allocation_points] and
     [dominated_by_allocation_point] ([derive_dominated], and the check
     [check_dominated]); [get_direct_usages],
     [add_usages_through_function_slots], [get_fields],
     [get_single_field_source], [get_set_of_closures_def]
     ([set_function_slots]) and [get_fields_usage_of_constructors]
     ([get_fields_of_constructors]), with the rules of
     [field_of_constructor_is_used_top] and
     [field_of_constructor_is_used_as] that it reads ([field_used_top],
     [field_used_as]);
   - unboxing_analysis.ml: [unboxing_rules], with the seeds of
     [cannot_unbox0] of [perform_analysis0] ([unboxing_relations], the
     checks [check_unboxing_closed], [check_to_unbox] and
     [check_to_change_representation], [derive_to_unbox] and
     [derive_to_change_representation]); [has_to_be_unboxed],
     [mk_unboxed_fields] and the unboxed fields of [perform_analysis0]
     ([unboxed_fields_of_names]); the [changed_representation] of
     [perform_analysis0] for closures ([Closure_representation]:
     [changed_representations]); [param_decision] with [Unbox],
     [my_closure_param_decision] ([my_closure_decision_u]),
     [arity_of_decisions], [get_arity_and_modes] and
     [compute_code_changes] ([code_change_u]);
   - rebuild.ml: [rewrite_simple] of an unboxed name ([poison]),
     [rewrite_simple_opt] of an unboxed callee, [get_args],
     [get_args_with_kinds], [bind_fields], [bind_field_to_simple],
     [should_keep_param] of [rebuild] ([handler_decisions]),
     [rewrite_apply_cont_expr] ([get_args]),
     [load_field_from_value_which_is_being_unboxed] (of an unboxed value
     and of a closure whose representation changes),
     [rebuild_singleton_binding_which_is_being_unboxed],
     [rebuild_set_of_closures_binding_which_is_being_unboxed]
     ([rebuild_unboxed_set]), [rewrite_set_of_closures] with a change of
     representation ([rebuild_set_u], [changed_value_slots]),
     [rebuild_singleton_binding_whose_representation_is_being_changed]
     of a closure ([rebuild_changed_binding]), [rewrite_field_access] and
     [rewrite_field_access_chg_repr] of [rebuild_named_default_case]
     ([rebuild_let_u]), the static sets of
     [rebuild_let_expr_static_consts] whose representation changes
     ([rebuild_changed_static_const]), [freshen_decisions],
     [make_apply_wrapper] with every pair of decisions, [rebuild_apply]
     with an unboxed callee, with [Unbox] decisions and with
     [Unbox_my_closure], and [rebuild_function_params_and_body] with
     them.
   It needs no review, but for the fix of faults of the compiler's
   unboxing, which is not on the branch (the seeds of its rules, its
   check, [unboxing_relations_with], the mode [fixed] of the rebuild and
   its fatal errors, [rebuild_fails_u]), which
   FlambdaReaperUnboxingSpec.v names for review
   with its statements: that file says what the checks mean, and the test tool
   compares the decisions and the rebuilt term with those of the
   compiler's reaper. It is extracted ([derive_dominated],
   [check_dominated], [derive_to_unbox], [check_to_unbox],
   [to_unbox_of_unboxed_names],
   [derive_to_change_representation], [check_to_change_representation],
   [to_change_representation_of_alloc_points], [check_sets_whole],
   [unit_unboxed_fields], [unit_changed_representations],
   [unit_calling_convention_changes_u], [unit_my_closure_decisions_u],
   [rebuild_unit_unbox], [rebuild_unit_unbox_repr], and, for the model
   with the fix, [fix_seeds], [derive_to_unbox_fixed],
   [check_to_unbox_fixed], [check_to_change_representation_fixed],
   [rebuild_unit_unbox_repr_fixed] and
   [rebuild_unit_unbox_repr_fixed_fails]).

   What is transcribed is the reaper run with
   [-reaper-change-calling-conventions] and [-reaper-unbox], the
   compiler's defaults, for blocks, boxed numbers and closures. The flags
   that the rules read are parameters: the local fields ([is_local]),
   [-reaper-max-unbox-size] ([max_unbox_size]) and the extra option
   [reaper-lambda-lifting] ([lambda_lifting]).

   The tables. [dominated_by_allocation_point] is exported by
   points_to_analysis.mli ([Relations]) and is an input, read off the
   compiler, as the solution is. [to_unbox] is not exported, and nor are
   the relations it is derived from; the unboxed fields are, by
   [Analysis.get_unboxed_fields], for each name dominated by an
   allocation to unbox. So the tool computes [to_unbox] from them
   ([to_unbox_of_unboxed_names]: the allocations that dominate a name
   with unboxed fields), which finds every allocation of [to_unbox] that
   has a source. Both tables are checked, each against the rules that
   conclude it, both ways ([check_dominated], [check_to_unbox]); the
   relations [cannot_change_representation0], [1],
   [cannot_change_representation], [cannot_unbox0] and [cannot_unbox],
   which no rule concludes from [to_unbox], are computed from the tables
   by their rules ([unboxing_relations], [cannot_unbox0] by rounds of its
   one recursive rule), and tested to be closed under them. The table of
   [cannot_change_calling_convention] is that of
   FlambdaReaperCallingConventions.v, checked there.

   [to_change_representation] is not exported either; the changes of
   representation are, by [Analysis.get_changed_representation], each
   with the closure that the name is dominated by. The tool takes for
   [to_change_representation] those closures that have a use
   ([to_change_representation_of_alloc_points]), which finds every
   allocation of the table that has a source and is a closure (the
   others would have a [Block_representation], below); the check
   [check_to_change_representation] tests it against the last rule, both
   ways, with the tables that the model computes, and
   [check_sets_whole] that the closures of a set with a use are unboxed
   alike and change their representation alike.

   The unboxed fields. Each name dominated by an allocation to unbox has
   its own variables, one for each field read from it that is not
   unboxed in turn, as [perform_analysis0] makes them; the model gives
   them the names after every name of the unit ([fresh_name] of
   [names_of]), in the order of the table [dominated], and in each name
   in the order of the fields. A map of fields is a list in the order of
   [Field.compare_view] ([field_lt]), the order in which the OCaml goes
   through it ([Field.Map.ordered_fold]) to make parameters, arguments
   and [Let]s. The wrappers have the names after those
   (FlambdaReaperRebuild.v says how they are named).

   The changes of representation. A closure of [to_change_representation]
   gives its set a new representation ([changed_representations]): the
   value slots that its closures have a use of
   ([get_fields_of_constructors]), each a new value slot or, if what it
   is read into is to be unboxed, unboxed in turn into new value slots
   ([mk_unboxed_fields], whose leaves are here value slots), and a new
   function slot for each of its function slots. Every name dominated by
   a closure of the set has that representation, with the function slot
   of that closure. The OCaml makes new slots of its own
   ([Value_slot.create], [Function_slot.create]); the model picks its
   own: the names after every slot of the unit ([fresh_name] of
   [slots_of]), the value slots of a set in the order of the fields, then
   its function slots, the sets in the order of the table [tcr]. The
   test tool relates the compiler's new slots to the model's through the
   two representations of each name, field for field.

   The rebuild. [rebuild_expr_u] is [rebuild_expr_cc] of
   FlambdaReaperCallingConventions.v case for case, with the unboxing.
   The functions of FlambdaReaperRebuild.v that it calls are given a
   solution in which the unboxed names have no use
   ([solution_without_unboxed]): their [rewrite_simple] is then that of
   rebuild.ml, which makes an unboxed name poison. Where rebuild.ml reads
   an unboxing, the model has a function of its own: the [Let] of an
   unboxed name binds the variables of its fields, a read from an unboxed
   name is the variable of the field, a continuation has the decisions of
   its parameters ([decision]: [Keep], [Delete] or [Unbox]) in place of
   the masks of FlambdaReaperRebuild.v, and code that changes its
   calling convention has [Unbox] decisions for the parameters and
   results that the code dep has unboxed, and the fields of its
   [my_closure] for parameters if the code dep has it unboxed. A set of
   closures that is unboxed binds the variables of the fields of its
   closures; one whose representation changes is made with the new
   slots, and a read from one of its closures is at the new slots. With
   nothing unboxed and no change of representation, the rebuild is that
   of FlambdaReaperCallingConventions.v (an example below shows it on the
   examples of that file).

   [make_apply_wrapper] is transcribed as the compiler has it. Where the
   continuation of a call keeps a parameter for a result that the callee
   unboxes, or the reverse, the OCaml has a test [has_any_source] to stop
   with a fatal error, stubbed to [false] ("CR ncourant: restore this
   check"), and makes the call one that never returns. That is a
   miscompile where the callee returns: a parameter with two sources,
   the unboxed result of the callee and a closure, that is only called,
   is kept. The model makes the call return to a continuation whose
   handler is [Invalid], which is how the conversion of the test tool
   reads a call that never returns; [kept_result_unboxed] below is such a
   unit, which returns 10, and whose rebuild goes wrong. A result that
   the continuation deletes is dropped, whatever the callee does with it,
   since the fix "reaper: fix miscompilation with unused unboxed
   variables" (#7952); before, a result that the callee unboxes and the
   continuation deletes made the call one that never returns too, a
   miscompile where a parameter with two sources, the unboxed result of
   the callee and a constant, has no use. FlambdaReaperUnboxingSpec.v
   has the unit.

   The fix. Its two rules of the analysis are seeds of [cannot_unbox0]
   ([fix_seeds]), with which the tables are computed again
   ([unboxing_relations_with]). Its rebuild is the mode [fixed] of
   [rebuild_expr_u]. The compiler gives decisions to every continuation,
   those of fixed arity included (the return and exception
   continuations of code, whose parameters are those of the code dep);
   the model gives decisions to these only in the mode of the fix
   ([code_continuation_decisions]), where they make a difference.
   [make_apply_wrapper] passes a result that the callee keeps to a
   parameter kept, and gives poison to one kept for a result that the
   callee deletes or unboxes; one unboxed for a result that the callee
   does not unbox is a fatal error. A call some of whose results have no
   source ([never_returns]: a result of the code of the call, or one
   read off the call witness of its callee, with no source) is made one
   that never returns, with no wrapper, which the model writes as the
   wrapper [Invalid] (below). A call of code that keeps its calling
   convention gives the arguments that the code dep uses, the others
   being poison ([used_params]: [kept_code_mask]); a callee that is
   unboxed there makes the call [Invalid] ("Unboxed callee ... cannot
   actually be a function"; the fix stopped there with a fatal error
   for a while, and no longer does); and code with [never_returns]
   whose rebuilt body still has its return continuation free is a fatal
   error ([check_never_returns]). The model rebuilds the unit as before
   at these two errors (an [Invalid]); whether the compiler stops is
   [rebuild_fails_u], which walks the unit as the compiler rebuilds it
   ([rebuild_unit_unbox_repr_fixed_fails]).
   [Keep_unused] is the [Keep] of a parameter kept with no use, which the
   fix no longer distinguishes: the rebuild treats it as [Keep], and the
   statements read the mark (an earlier fix gave such a parameter poison,
   whatever the callee returned, and lost a tail call).

   The deliberate differences from the OCaml, each where the model cannot
   follow it or does not have what it reads:
   - Where the OCaml stops with a fatal error, the model does something
     defined, which no term that the compiler accepts reaches: a [Let]
     whose fields cannot be bound (a field that is not of the variable
     read from, or unboxed in one and not in the other: [fold2_subset]),
     a read of a field that the unboxed value read from has not, or that
     is unboxed in turn, a call or an [Apply_cont] whose argument for an
     unboxed parameter is not unboxed alike, are [Invalid]; a field that
     may come from several allocations or is read into names of which
     some are to be unboxed and some not is a variable
     ([mk_unboxed_fields]); a field of an unexpected kind is left out of
     the unboxing; the nesting of an unboxing is bounded by the number of
     allocations to unbox; a parameter of code that the code dep unboxes
     and the parameter does not keeps the fields of the code dep.
   - A load of a mutable field that the unboxed value has not is the
     primitive [Invalid] in the OCaml; the model has no such primitive,
     and the [Let] is [Invalid].
   - [rewrite_or_variable] reads [has_usage] of a variable that may be
     unboxed, which the model's solution says has no use: the contents of
     a static boxed number would be zero where the OCaml keeps the
     variable. No well-kinded term has an unboxed name of kind
     [Naked_number].
   - The decisions of a continuation that has none (a continuation of
     fixed arity, not bound in the term) keep every result, and the
     arguments beyond the decisions of a continuation are kept, as in
     FlambdaReaperRebuild.v. A result that the callee no longer returns
     and the continuation keeps is poison of the kind of the parameter
     of the continuation, as in the OCaml (FlambdaReaperCallingConventions.v
     takes the kind of the return arity of the call, the same where the
     kinds agree).
   - The fixpoints of [add_usages_through_function_slots] and of
     [cannot_unbox0] are computed by rounds, at most one more than the
     tuples of the tables that a round adds to; [check_to_unbox] fails if
     the rounds of [cannot_unbox0] are not done.
   - The relation [code_id_my_closure] is read off the term
     ([bound_code_my_closures]), the facts not having it.
   - The new slots of a change of representation are the model's
     (above).
   - Sets of closures are [let_closures] (FlambdaReaperRebuild.v): the
     [Let] of the first closure, and the projections of the others from
     it. The closures of the set are the variable of the [Let] and those
     of the projections. Where the set is unboxed, the [Let] binds the
     variables of the fields of each of its closures with a use, in the
     order of the closures, as the compiler's [Let] of the set does, and
     the projections are dropped; where its representation changes, the
     projections are at the new function slots.
   - In the fix, a call whose code is not known and whose callee is a
     constant gives its arguments as they are; the model rewrites them,
     as on the branch ([rebuild_arguments]). A call of unknown arity with
     no argument, a fatal error of the analysis in the fix, is not one
     that never returns ([never_returns_of_call]).
   - A call of code that changes its calling convention under
     [Keep_my_closure] whose callee has no use has no callee in the OCaml
     ([rewrite_simple_opt]). The model keeps it: its calls have a callee,
     and its machine runs a call by the closure that its callee is. The
     code does not read the closure, so that the call runs as the
     compiler's does without it.
   - A call whose callee is unboxed has no callee in the OCaml, whether
     the code unboxes its [my_closure] ([Unbox_my_closure]) or keeps it
     ([Keep_my_closure], [rewrite_simple_opt]); the model's calls always
     have one, and it is the poison that [rewrite_simple] makes of the
     unboxed callee. The compiler drops the callee under
     [Keep_my_closure] since its fix of a crash with lambda lifting
     (the commit "reaper: fix a few bugs"): two closures of one code,
     each only called, are unboxed, and the [my_closure] of the code,
     of which both are sources, is not; the call kept a closure whose
     binding the unboxing removed. FlambdaReaperUnboxingSpec.v has the
     unit.
   - Regions. The [Let] of a [Local_allocation] binds again the variable
     of the allocation just before it (FlambdaPrimitiveModel.v). Where
     that variable is unboxed it is removed, as the mode of the
     compiler goes with the allocation ([rebuild_unboxed_binding]);
     where the representation of a closure changes it is kept, as the
     closure is still made. Otherwise it is as in
     FlambdaReaperRebuild.v, which says why a local set of closures is
     rebuilt without the projections of its other closures.
   - The older forms [Set_of_closures] and [Static_const_set_of_closures]
     have no fact of a function slot (FlambdaReaperTraversal.v): to
     [get_set_of_closures_def] they are not sets, and they never change
     their representation. They are unboxed as the other form is.
   - Where the OCaml stops with a fatal error in a change of
     representation or an unboxing of a closure, the model does something
     defined: a field of the representation that the set has not, or
     that is unboxed in one of the two and not in the other, is left out
     of the set; a function slot that the representation has not stays;
     a read of a field that the representation has not, or that is
     unboxed in turn, is [Invalid]; a closure with a use of an unboxed
     set that is not unboxed itself, a field of an unboxed closure that
     the set has not, or a callee that is not unboxed for code that
     unboxes its [my_closure], make the term [Invalid]; with no group of
     parameters, the fields of [my_closure] or of the callee are left
     out.

   Left out: [Block_representation]. The compiler makes one for an
   allocation that its traversal makes, such as the closure of a partial
   application, and gives it to the names that the allocation dominates,
   which may be names of the term: the model gives no representation to
   an allocation of [tcr] that is not a closure of a set, and the tool
   reports a unit that gives a [Block_representation] to a name of the
   term as out of the model. *)

From Stdlib Require Import ZArith List Bool PeanoNat.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaBindersSpec FlambdaWellFormedSpec
  FlambdaNameAbstractionSpec FlambdaFreshQuantifierSpec
  FlambdaRenameApartSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperRebuild FlambdaReaperSolution
  FlambdaReaperCallingConventions.
From OxCamlFlambda2 Require FlambdaValueSpec FlambdaSemanticsSpec.
Import ListNotations.

(* ======================================================================== *)
(* Fields                                                                   *)
(* ======================================================================== *)

(* [Field.kind]. A virtual field has no kind (a fatal error in the
   OCaml): it is given [Value]. Every value slot of the model is a
   value. *)
Definition field_kind (f : field) : flambda_kind :=
  match f with
  | Field_block _ kind => kind
  | Field_value_slot _ | Field_function_slot _ => Value
  | Field_is_int | Field_get_tag => Naked_number Kind_naked_immediate
  | Field_boxed_number bn => unboxed_kind bn
  | Field_call_witness _ | Field_return_of_call _
  | Field_code_id_of_call_witness => Value
  end.

(* The fields of a block or a boxed number, those of which the
   representation of an allocation can never change
   ([cannot_change_representation] of [unboxing_rules]). *)
Definition is_block_like_field (f : field) : bool :=
  match f with
  | Field_block _ _ | Field_is_int | Field_get_tag | Field_boxed_number _ =>
    true
  | Field_value_slot _ | Field_function_slot _ | Field_call_witness _
  | Field_return_of_call _ | Field_code_id_of_call_witness => false
  end.

Definition is_return_of_call (f : field) : bool :=
  match f with
  | Field_return_of_call _ => true
  | _ => false
  end.

Definition is_symbol_node (n : node) : bool :=
  match n with
  | Node_symbol _ => true
  | _ => false
  end.

(* The order of [Stdlib.compare] on [Flambda_kind.t]: the constant
   constructors first, in their order, then [Naked_number] by its
   argument. *)
Definition naked_number_kind_index (k : naked_number_kind) : nat :=
  match k with
  | Kind_naked_immediate => 0
  | Kind_naked_float32 => 1
  | Kind_naked_float => 2
  | Kind_naked_int8 => 3
  | Kind_naked_int16 => 4
  | Kind_naked_int32 => 5
  | Kind_naked_int64 => 6
  | Kind_naked_nativeint => 7
  | Kind_naked_vec128 => 8
  | Kind_naked_vec256 => 9
  | Kind_naked_vec512 => 10
  | Kind_naked_mask => 11
  end.

Definition kind_index (k : flambda_kind) : nat :=
  match k with
  | Value => 0
  | Region => 1
  | Rec_info => 2
  | Naked_number k => 3 + naked_number_kind_index k
  end.

(* The order of [Stdlib.compare] on [Flambda_kind.Boxable_number.t]. *)
Definition boxable_number_index (bn : boxable_number) : nat :=
  match bn with
  | Boxable_naked_float32 => 0
  | Boxable_naked_float => 1
  | Boxable_naked_int32 => 2
  | Boxable_naked_int64 => 3
  | Boxable_naked_nativeint => 4
  | Boxable_naked_vec128 => 5
  | Boxable_naked_vec256 => 6
  | Boxable_naked_vec512 => 7
  | Boxable_naked_mask => 8
  end.

(* [constructor_numbering] of [Field.compare_view]. *)
Definition field_constructor_index (f : field) : nat :=
  match f with
  | Field_block _ _ => 0
  | Field_value_slot _ => 1
  | Field_function_slot _ => 2
  | Field_call_witness _ => 3
  | Field_is_int => 4
  | Field_get_tag => 5
  | Field_boxed_number _ => 6
  | Field_return_of_call _ => 7
  | Field_code_id_of_call_witness => 8
  end.

Definition entry_point_index (ep : closure_entry_point) : nat :=
  match ep with
  | Unknown_arity_code_pointer => 0
  | Known_arity_code_pointer => 1
  end.

(* [Field.compare_view]: the order in which [Field.Map.ordered_fold] goes
   through the fields of an unboxing, which is that of the parameters and
   arguments it makes. Slots compare by their numbers, which stand for the
   compiler's own order of slots. *)
Definition field_lt (f f' : field) : bool :=
  match f, f' with
  | Field_block i k, Field_block i' k' =>
    Nat.ltb i i' || (Nat.eqb i i' && Nat.ltb (kind_index k) (kind_index k'))
  | Field_value_slot w, Field_value_slot w' => Pos.ltb w w'
  | Field_function_slot s, Field_function_slot s' => Pos.ltb s s'
  | Field_call_witness ep, Field_call_witness ep' =>
    Nat.ltb (entry_point_index ep) (entry_point_index ep')
  | Field_boxed_number bn, Field_boxed_number bn' =>
    Nat.ltb (boxable_number_index bn) (boxable_number_index bn')
  | Field_return_of_call (Normal i), Field_return_of_call (Normal i') =>
    Nat.ltb i i'
  | Field_return_of_call (Normal _), Field_return_of_call Exn => true
  | Field_return_of_call _, Field_return_of_call _ => false
  | _, _ => Nat.ltb (field_constructor_index f) (field_constructor_index f')
  end.

(* A field put in its place in a list ordered by [field_lt]. *)
Fixpoint insert_field {A : Type} (f : field) (a : A)
    (l : list (field * A)) : list (field * A) :=
  match l with
  | [] => [(f, a)]
  | (f', a') :: l' =>
    if field_lt f f' then (f, a) :: l else (f', a') :: insert_field f a l'
  end.

Definition sort_fields {A : Type} (l : list (field * A)) : list (field * A) :=
  fold_right (fun '(f, a) acc => insert_field f a acc) [] l.

(* ======================================================================== *)
(* The rules: [unboxing_rules]                                              *)
(* ======================================================================== *)

Section Rules.

Variable is_local : field -> bool.
Variable F : facts.
Variable T : tables.
(* The table of [cannot_change_calling_convention]
   (FlambdaReaperCallingConventions.v). *)
Variable cannot_change : list node.
(* [Flambda_features.reaper_max_unbox_size], and the extra option
   [reaper-lambda-lifting]. *)
Variable max_unbox_size : nat.
Variable lambda_lifting : bool.

Abbreviation any_usage x := (mem_node x (t_any_usage T)).
Abbreviation any_source x := (mem_node x (t_any_source T)).
Abbreviation has_usage x := (mem_node x (t_has_usage T)).
Abbreviation has_source x := (mem_node x (t_has_source T)).
Abbreviation sources x y := (mem_pair x y (t_sources T)).
Abbreviation field_of_constructor_is_used base relation :=
  (mem_node_field base relation (t_field_used T)).

(* [multiple_allocation_points] of [post_processing_rules]
   (points_to_analysis.ml): [x] may come from any allocation, or from two
   that have a source. *)
Definition multiple_allocation_points (x : node) : bool :=
  any_source x
  || existsb
       (fun '(x', y) =>
          node_eqb x x' && has_source y
          && existsb
               (fun '(x'', z) =>
                  node_eqb x x'' && has_source z && negb (node_eqb y z))
               (t_sources T))
       (t_sources T).

(* [dominated_by_allocation_point a x] as the rule of
   [post_processing_rules] gives it, from the tables: the pairs [(a, x)]
   where [a] is the one allocation with a source that [x] may come from.
   This is the relation of the compiler for these tables. *)
Definition derive_dominated : list (node * node) :=
  flat_map
    (fun '(x, y) =>
       if has_source y && negb (multiple_allocation_points x)
       then [(y, x)]
       else [])
    (t_sources T).

(* Whether a table of [dominated_by_allocation_point] is that relation:
   it has every pair that the rule gives, and only those. *)
Definition check_dominated (dominated : list (node * node)) : bool :=
  forallb (fun '(a, x) => mem_pair a x dominated) derive_dominated
  && forallb
       (fun '(a, x) =>
          sources x a && has_source a && negb (multiple_allocation_points x))
       dominated.

(* [rev_accessor ~base:alias field ~to_:v; has_usage v] for a real field
   [field] and some [v]. *)
Definition real_field_read_with_usage (alias : node) : bool :=
  existsb
    (fun '(v, field, base) =>
       node_eqb base alias && is_real_field field && has_usage v)
    (f_accessor F).

(* [constructor ~base:call_witness code_id_of_call_witness ~from:codeid;
   cannot_change_calling_convention codeid] for some [codeid]. *)
Definition witness_cannot_change (call_witness : node) : bool :=
  existsb
    (fun '(call_witness', field, codeid) =>
       node_eqb call_witness call_witness'
       && field_eqb field Field_code_id_of_call_witness
       && mem_node codeid cannot_change)
    (f_constructor F).

(* The fields of [x] of which [field_of_constructor_is_used] holds, real
   and not function slots, each once: those that [perform_analysis0]
   counts against [max_unbox_size]. *)
Definition used_fields (x : node) : list field :=
  Keyed.dedup field_eqb
    (flat_map
       (fun '(y, f) =>
          if node_eqb y x && is_real_field f && negb (is_function_slot f)
          then [f]
          else [])
       (t_field_used T)).

(* What the nine rules that conclude [cannot_change_representation0]
   give, in their order, each a pass over the facts and the tuples that
   its premises name. No rule concludes it from itself, so this is the
   relation of the compiler for these tables. *)
Definition derived_cannot_change_representation0 : list node :=
  flat_map
    (fun '(x, field, _) =>
       if any_usage x && negb (is_local field) && is_real_field field
       then [x]
       else [])
    (f_constructor F)
  ++ flat_map
       (fun '(x, field, _) =>
          if any_usage x && is_local field
             && existsb (fun '(field', _) => field_eqb field field')
                  (t_reading_field T)
          then [x]
          else [])
       (f_constructor F)
  ++ flat_map
       (fun '(v, field, usage) =>
          if has_usage v && is_local field
          then
            flat_map
              (fun '(usage1, source1) =>
                 if node_eqb usage usage1 && has_source source1
                    && existsb
                         (fun '(usage2, source2) =>
                            node_eqb usage usage2 && has_source source2
                            && negb (node_eqb source1 source2))
                         (t_sources T)
                 then [source1]
                 else [])
              (t_sources T)
          else [])
       (f_accessor F)
  ++ flat_map
       (fun '(allocation_id, alias) =>
          if existsb
               (fun '(alias', alias_source) =>
                  node_eqb alias alias' && has_source alias_source
                  && negb (node_eqb alias_source allocation_id))
               (t_sources T)
             && real_field_read_with_usage alias
          then [allocation_id]
          else [])
       (t_usages T)
  ++ flat_map
       (fun '(allocation_id, alias) =>
          if any_source alias && real_field_read_with_usage alias
          then [allocation_id]
          else [])
       (t_usages T)
  ++ flat_map
       (fun '(allocation_id, source) =>
          if has_source source && negb (node_eqb source allocation_id)
          then [allocation_id]
          else [])
       (t_sources T)
  ++ flat_map
       (fun '(allocation_id, usage) =>
          if has_usage usage && negb (sources allocation_id allocation_id)
          then [allocation_id]
          else [])
       (t_usages T)
  ++ t_any_source T
  ++ flat_map
       (fun '(call_witness, field, _) =>
          if field_eqb field Field_code_id_of_call_witness
          then [call_witness]
          else [])
       (f_constructor F).

(* The closures of a set whose closures, one of them in [l], hold them in
   a function slot. *)
Definition by_function_slots (l : list node) : list node :=
  flat_map
    (fun '(base, field, y) =>
       if is_function_slot field && mem_node base l then [y] else [])
    (f_constructor F).

(* What the rules that conclude [cannot_unbox0] from relations other than
   itself give, for a table [crr1] of [cannot_change_representation1]:
   the seeds of the one recursive rule. *)
Definition derived_cannot_unbox0_seeds (crr1 : list node) : list node :=
  crr1 ++ t_any_usage T
  ++ flat_map
       (fun '(x, field, call_witness) =>
          if field_eqb field (Field_call_witness Known_arity_code_pointer)
             && witness_cannot_change call_witness
          then [x]
          else [])
       (f_constructor F)
  ++ (if lambda_lifting
      then []
      else
        flat_map
          (fun '(x, field, call_witness) =>
             if field_eqb field (Field_call_witness Known_arity_code_pointer)
                && has_usage call_witness
             then [x]
             else [])
          (f_constructor F))
  ++ flat_map
       (fun '(alias, allocation_id) =>
          if has_source allocation_id
             && existsb
                  (fun '(call_witness, relation, alias') =>
                     node_eqb alias alias' && is_return_of_call relation
                     && witness_cannot_change call_witness)
                  (f_constructor F)
          then [allocation_id]
          else [])
       (t_sources T)
  ++ flat_map
       (fun '(alias, allocation_id) =>
          if has_source allocation_id
             && existsb
                  (fun '(call_witness, _, alias') =>
                     node_eqb alias alias'
                     && witness_cannot_change call_witness)
                  (f_parameter F)
          then [allocation_id]
          else [])
       (t_sources T)
  ++ flat_map (fun '(x, _) => if is_symbol_node x then [x] else [])
       (t_sources T)
  ++ filter (fun x => Nat.ltb max_unbox_size (length (used_fields x)))
       (Keyed.dedup node_eqb (map fst (t_field_used T))).

(* One round of the rule that concludes [cannot_unbox0] from itself, for
   tables [crr] of [cannot_change_representation] and [cu0] of
   [cannot_unbox0]: the allocations not in [cu0] stored in a used real
   field of one that can neither change its representation nor be
   unboxed. *)
Definition stored_round (crr cu0 : list node) : list node :=
  flat_map
    (fun '(alias, allocation_id) =>
       if negb (mem_node allocation_id cu0)
          && existsb
               (fun '(to_, relation, alias') =>
                  node_eqb alias alias'
                  && field_of_constructor_is_used to_ relation
                  && mem_node to_ crr && is_real_field relation
                  && mem_node to_ cu0)
               (f_constructor F)
       then [allocation_id]
       else [])
    (t_sources T).

(* Rounds until one adds nothing or [fuel] rounds have been made. A round
   that adds something adds an allocation of the tables that was not
   there, so that [S (length (t_sources T))] rounds are enough. *)
Fixpoint stored_rounds (crr : list node) (fuel : nat) (cu0 : list node)
    : list node :=
  match fuel with
  | O => cu0
  | S fuel =>
    match Keyed.dedup node_eqb (stored_round crr cu0) with
    | [] => cu0
    | added => stored_rounds crr fuel (added ++ cu0)
    end
  end.

(* The tables of the relations of [unboxing_rules] but [to_unbox] and
   [to_change_representation], as lists. *)
Record unboxing_tables : Type := Unboxing_tables {
  ut_crr0 : list node;
  ut_crr1 : list node;
  ut_crr : list node;
  ut_cu0 : list node;
  ut_cu : list node
}.

(* The tables that the model computes, each from the one before, as the
   rules give them: the least ones, if the rounds of [cannot_unbox0] are
   done, which [check_unboxing_closed] tests. *)
Definition unboxing_relations : unboxing_tables :=
  let crr0 := Keyed.dedup node_eqb derived_cannot_change_representation0 in
  let crr1 := Keyed.dedup node_eqb (crr0 ++ by_function_slots crr0) in
  let crr :=
    Keyed.dedup node_eqb
      (crr1
       ++ flat_map
            (fun '(base, field, _) =>
               if is_block_like_field field then [base] else [])
            (f_constructor F))
  in
  let cu0 :=
    stored_rounds crr (S (length (t_sources T)))
      (Keyed.dedup node_eqb (derived_cannot_unbox0_seeds crr1))
  in
  let cu := Keyed.dedup node_eqb (cu0 ++ by_function_slots cu0) in
  {| ut_crr0 := crr0; ut_crr1 := crr1; ut_crr := crr; ut_cu0 := cu0;
     ut_cu := cu |}.

(* The checks that tables are closed under the rules: one test for each
   rule, named after the field of
   FlambdaReaperUnboxingSpec.unboxing_closed that states it, as in
   FlambdaReaperSolution.v: a pass over the facts and the tuples that the
   premises name, which asks of the tables, at each match, the
   conclusion. *)
Section Check.

Variable R : unboxing_tables.

Abbreviation crr0 x := (mem_node x (ut_crr0 R)).
Abbreviation crr1 x := (mem_node x (ut_crr1 R)).
Abbreviation crr x := (mem_node x (ut_crr R)).
Abbreviation cu0 x := (mem_node x (ut_cu0 R)).
Abbreviation cu x := (mem_node x (ut_cu R)).

Definition check_crr0_any_usage : bool :=
  forallb
    (fun '(x, field, _) =>
       negb (any_usage x) || is_local field || negb (is_real_field field)
       || crr0 x)
    (f_constructor F).

Definition check_crr0_local_reading : bool :=
  forallb
    (fun '(x, field, _) =>
       negb (any_usage x) || negb (is_local field)
       || forallb
            (fun '(field', _) => negb (field_eqb field field') || crr0 x)
            (t_reading_field T))
    (f_constructor F).

Definition check_crr0_local_sources : bool :=
  forallb
    (fun '(v, field, usage) =>
       negb (has_usage v) || negb (is_local field)
       || forallb
            (fun '(usage1, source1) =>
               negb (node_eqb usage usage1) || negb (has_source source1)
               || forallb
                    (fun '(usage2, source2) =>
                       negb (node_eqb usage usage2)
                       || negb (has_source source2)
                       || node_eqb source1 source2 || crr0 source1)
                    (t_sources T))
            (t_sources T))
    (f_accessor F).

Definition check_crr0_other_source : bool :=
  forallb
    (fun '(allocation_id, alias) =>
       forallb
         (fun '(alias', alias_source) =>
            negb (node_eqb alias alias') || negb (has_source alias_source)
            || node_eqb alias_source allocation_id
            || forallb
                 (fun '(v, field, alias'') =>
                    negb (node_eqb alias alias'')
                    || negb (is_real_field field) || negb (has_usage v)
                    || crr0 allocation_id)
                 (f_accessor F))
         (t_sources T))
    (t_usages T).

Definition check_crr0_any_source_usage : bool :=
  forallb
    (fun '(allocation_id, alias) =>
       negb (any_source alias)
       || forallb
            (fun '(v, field, alias') =>
               negb (node_eqb alias alias') || negb (is_real_field field)
               || negb (has_usage v) || crr0 allocation_id)
            (f_accessor F))
    (t_usages T).

Definition check_crr0_other_own_source : bool :=
  forallb
    (fun '(allocation_id, source) =>
       negb (has_source source) || node_eqb source allocation_id
       || crr0 allocation_id)
    (t_sources T).

Definition check_crr0_not_own_source : bool :=
  forallb
    (fun '(allocation_id, usage) =>
       negb (has_usage usage) || sources allocation_id allocation_id
       || crr0 allocation_id)
    (t_usages T).

Definition check_crr0_any_source : bool :=
  forallb (fun x => crr0 x) (t_any_source T).

Definition check_crr0_call_witness : bool :=
  forallb
    (fun '(call_witness, field, _) =>
       negb (field_eqb field Field_code_id_of_call_witness)
       || crr0 call_witness)
    (f_constructor F).

Definition check_crr1_crr0 : bool := forallb (fun x => crr1 x) (ut_crr0 R).

Definition check_crr1_function_slot : bool :=
  forallb
    (fun '(x, field, y) =>
       negb (is_function_slot field) || negb (crr0 x) || crr1 y)
    (f_constructor F).

Definition check_crr_crr1 : bool := forallb (fun x => crr x) (ut_crr1 R).

Definition check_crr_block_like : bool :=
  forallb
    (fun '(x, field, _) => negb (is_block_like_field field) || crr x)
    (f_constructor F).

Definition check_cu0_cannot_change_representation : bool :=
  forallb (fun x => cu0 x) (ut_crr1 R).

Definition check_cu0_any_usage : bool :=
  forallb (fun x => cu0 x) (t_any_usage T).

Definition check_cu0_called_cannot_change : bool :=
  forallb
    (fun '(x, field, call_witness) =>
       negb (field_eqb field (Field_call_witness Known_arity_code_pointer))
       || forallb
            (fun '(call_witness', field', codeid) =>
               negb (node_eqb call_witness call_witness')
               || negb (field_eqb field' Field_code_id_of_call_witness)
               || negb (mem_node codeid cannot_change) || cu0 x)
            (f_constructor F))
    (f_constructor F).

Definition check_cu0_called : bool :=
  lambda_lifting
  || forallb
       (fun '(x, field, call_witness) =>
          negb (field_eqb field (Field_call_witness Known_arity_code_pointer))
          || negb (has_usage call_witness) || cu0 x)
       (f_constructor F).

Definition check_cu0_result : bool :=
  forallb
    (fun '(alias, allocation_id) =>
       negb (has_source allocation_id)
       || forallb
            (fun '(call_witness, relation, alias') =>
               negb (node_eqb alias alias')
               || negb (is_return_of_call relation)
               || forallb
                    (fun '(call_witness', field, codeid) =>
                       negb (node_eqb call_witness call_witness')
                       || negb (field_eqb field Field_code_id_of_call_witness)
                       || negb (mem_node codeid cannot_change)
                       || cu0 allocation_id)
                    (f_constructor F))
            (f_constructor F))
    (t_sources T).

Definition check_cu0_parameter : bool :=
  forallb
    (fun '(alias, allocation_id) =>
       negb (has_source allocation_id)
       || forallb
            (fun '(call_witness, _, alias') =>
               negb (node_eqb alias alias')
               || forallb
                    (fun '(call_witness', field, codeid) =>
                       negb (node_eqb call_witness call_witness')
                       || negb (field_eqb field Field_code_id_of_call_witness)
                       || negb (mem_node codeid cannot_change)
                       || cu0 allocation_id)
                    (f_constructor F))
            (f_parameter F))
    (t_sources T).

Definition check_cu0_symbol : bool :=
  forallb (fun '(x, _) => negb (is_symbol_node x) || cu0 x) (t_sources T).

Definition check_cu0_stored : bool :=
  forallb
    (fun '(alias, allocation_id) =>
       forallb
         (fun '(to_, relation, alias') =>
            negb (node_eqb alias alias')
            || negb (field_of_constructor_is_used to_ relation)
            || negb (crr to_) || negb (is_real_field relation)
            || negb (cu0 to_) || cu0 allocation_id)
         (f_constructor F))
    (t_sources T).

Definition check_cu0_max_unbox_size : bool :=
  forallb
    (fun '(x, _) =>
       negb (Nat.ltb max_unbox_size (length (used_fields x))) || cu0 x)
    (t_field_used T).

Definition check_cu_cu0 : bool := forallb (fun x => cu x) (ut_cu0 R).

Definition check_cu_function_slot : bool :=
  forallb
    (fun '(x, field, y) =>
       negb (cu0 x) || negb (is_function_slot field) || cu y)
    (f_constructor F).

(* Whether the tables are closed under every rule. *)
Definition check_unboxing_closed : bool :=
  check_crr0_any_usage && check_crr0_local_reading
  && check_crr0_local_sources && check_crr0_other_source
  && check_crr0_any_source_usage && check_crr0_other_own_source
  && check_crr0_not_own_source && check_crr0_any_source
  && check_crr0_call_witness && check_crr1_crr0 && check_crr1_function_slot
  && check_crr_crr1 && check_crr_block_like
  && check_cu0_cannot_change_representation && check_cu0_any_usage
  && check_cu0_called_cannot_change && check_cu0_called && check_cu0_result
  && check_cu0_parameter && check_cu0_symbol && check_cu0_stored
  && check_cu0_max_unbox_size && check_cu_cu0 && check_cu_function_slot.

End Check.

(* [to_unbox] as the rule gives it: what has a use and is not in
   [cannot_unbox]. *)
Definition derive_to_unbox : list node :=
  let R := unboxing_relations in
  filter (fun x => negb (mem_node x (ut_cu R)))
    (Keyed.dedup node_eqb (t_has_usage T)).

(* Whether a table of [to_unbox] is the relation that the rules give:
   the tables that the model computes are closed under the rules, each
   allocation of [to_unbox] has a use and is not in that of
   [cannot_unbox], and each allocation with a use and a source that is
   not in it is in [to_unbox]. An allocation with no source, which the
   rule may unbox but which dominates no name and whose [Let] is
   [Invalid], may be left out. *)
Definition check_to_unbox (to_unbox : list node) : bool :=
  let R := unboxing_relations in
  check_unboxing_closed R
  && forallb (fun x => has_usage x && negb (mem_node x (ut_cu R))) to_unbox
  && forallb
       (fun x =>
          negb (has_source x) || mem_node x (ut_cu R) || mem_node x to_unbox)
       (t_has_usage T).

(* [to_change_representation] as the last rule gives it, for a table
   [to_unbox]: what has a use, is not in [cannot_change_representation]
   and is not to be unboxed. *)
Definition derive_to_change_representation (to_unbox : list node)
    : list node :=
  let R := unboxing_relations in
  filter
    (fun x => negb (mem_node x (ut_crr R)) && negb (mem_node x to_unbox))
    (Keyed.dedup node_eqb (t_has_usage T)).

(* Whether a table [tcr] of [to_change_representation] is the relation
   that the rule gives, for a table of [to_unbox] that passes
   [check_to_unbox]: each allocation of [tcr] has a use, is not in the
   table of [cannot_change_representation] that the model computes, and
   is not in [to_unbox]; each allocation with a use and a source that is
   in none of the three is in [tcr]. An allocation with no source, which
   dominates no name, may be left out. *)
Definition check_to_change_representation (to_unbox tcr : list node)
    : bool :=
  let R := unboxing_relations in
  check_to_unbox to_unbox
  && forallb
       (fun x =>
          has_usage x && negb (mem_node x (ut_crr R))
          && negb (mem_node x to_unbox))
       tcr
  && forallb
       (fun x =>
          negb (has_source x) || mem_node x (ut_crr R) || mem_node x to_unbox
          || mem_node x tcr)
       (t_has_usage T).

(* Whether the closures of a set that have a use are unboxed alike and
   change their representation alike: the rebuild unboxes a set, or
   changes its representation, whole. A closure of a set holds each
   closure of the set in a function slot. *)
Definition check_sets_whole (to_unbox tcr : list node) : bool :=
  forallb
    (fun '(x, f, y) =>
       negb (is_function_slot f) || negb (has_usage x) || negb (has_usage y)
       || (Bool.eqb (mem_node x to_unbox) (mem_node y to_unbox)
           && Bool.eqb (mem_node x tcr) (mem_node y tcr)))
    (f_constructor F).

(* ------------------------------------------------------------------------ *)
(* The fix                                                                  *)
(* ------------------------------------------------------------------------ *)

(* The fix of the faults of the compiler's unboxing that
   FlambdaReaperUnboxingSpec.v states, of which this is the half of the
   analysis: two rules that conclude [cannot_unbox0], not on the branch,
   whose seeds are allocations that they forbid to unbox. The half of the
   rebuild is the mode [fixed] of [rebuild_expr_u] below. They are here
   so that the test tool can run the model with the fix
   ([derive_to_unbox_fixed], [rebuild_unit_unbox_repr_fixed]). *)

(* The tables of the rules with the allocations [seeds] added to
   [cannot_unbox0], as a rule that concludes it from relations other than
   itself adds them ([derived_cannot_unbox0_seeds]): before the rounds of
   [cu0_stored], which are made again. The other tables are those of
   [unboxing_relations], which no seed of [cannot_unbox0] changes. *)
Definition unboxing_relations_with (seeds : list node) : unboxing_tables :=
  let R := unboxing_relations in
  let cu0 :=
    stored_rounds (ut_crr R) (S (length (t_sources T)))
      (Keyed.dedup node_eqb (seeds ++ ut_cu0 R))
  in
  {| ut_crr0 := ut_crr0 R; ut_crr1 := ut_crr1 R; ut_crr := ut_crr R;
     ut_cu0 := cu0;
     ut_cu := Keyed.dedup node_eqb (cu0 ++ by_function_slots cu0) |}.

(* [derive_to_unbox] for those tables. *)
Definition derive_to_unbox_with (seeds : list node) : list node :=
  let R := unboxing_relations_with seeds in
  filter (fun x => negb (mem_node x (ut_cu R)))
    (Keyed.dedup node_eqb (t_has_usage T)).

(* [check_to_unbox] for those tables: they are closed under the rules and
   have the seeds in [cannot_unbox0], so that they are closed under the
   rules with the seeds; each allocation of [to_unbox] has a use and is
   not in their [cannot_unbox], and each allocation with a use and a
   source that is not in it is in [to_unbox]. *)
Definition check_to_unbox_with (seeds to_unbox : list node) : bool :=
  let R := unboxing_relations_with seeds in
  check_unboxing_closed R
  && forallb (fun x => mem_node x (ut_cu0 R)) seeds
  && forallb (fun x => has_usage x && negb (mem_node x (ut_cu R))) to_unbox
  && forallb
       (fun x =>
          negb (has_source x) || mem_node x (ut_cu R) || mem_node x to_unbox)
       (t_has_usage T).

(* [check_to_change_representation] for a table of [to_unbox] that passes
   [check_to_unbox_with], the table of [cannot_change_representation]
   being that of [unboxing_relations]. *)
Definition check_to_change_representation_with (seeds to_unbox tcr : list node)
    : bool :=
  let R := unboxing_relations in
  check_to_unbox_with seeds to_unbox
  && forallb
       (fun x =>
          has_usage x && negb (mem_node x (ut_crr R))
          && negb (mem_node x to_unbox))
       tcr
  && forallb
       (fun x =>
          negb (has_source x) || mem_node x (ut_crr R) || mem_node x to_unbox
          || mem_node x tcr)
       (t_has_usage T).

(* [rev_accessor ~base:alias field ~to_:v; has_usage v] for any field
   [field] and some [v]: unlike [real_field_read_with_usage], the fields
   of calls too (a call witness read off a callee, a result read off a
   witness). *)
Definition field_read_with_usage (alias : node) : bool :=
  existsb (fun '(v, _, base) => node_eqb base alias && has_usage v)
    (f_accessor F).

(* The first rule of the fix, as what it adds to [cannot_unbox0]: an
   allocation with a use [alias] that has another source, and from which
   a field is read into a name with a use,
     [usages allocation_id alias; sources alias alias_source;
      has_source alias_source; distinct alias_source allocation_id;
      rev_accessor ~base:alias field ~to_:v; has_usage v]
     ==> cannot_unbox0 allocation_id.
   It is the fourth rule of [derived_cannot_change_representation0]
   without [is_real_field]. *)
Definition other_source_reader_seeds : list node :=
  flat_map
    (fun '(allocation_id, alias) =>
       if existsb
            (fun '(alias', alias_source) =>
               node_eqb alias alias' && has_source alias_source
               && negb (node_eqb alias_source allocation_id))
            (t_sources T)
          && field_read_with_usage alias
       then [allocation_id]
       else [])
    (t_usages T).

(* The second rule of the fix: an allocation with a use [alias] of any
   source, from which a field is read into a name with a use,
     [usages allocation_id alias; any_source alias;
      rev_accessor ~base:alias field ~to_:v; has_usage v]
     ==> cannot_unbox0 allocation_id;
   the fifth rule of [derived_cannot_change_representation0] without
   [is_real_field]. *)
Definition any_source_reader_seeds : list node :=
  flat_map
    (fun '(allocation_id, alias) =>
       if any_source alias && field_read_with_usage alias
       then [allocation_id]
       else [])
    (t_usages T).

(* The seeds of the two rules. *)
Definition fix_seeds : list node :=
  other_source_reader_seeds ++ any_source_reader_seeds.

(* The fix, as what it asks of a table of [to_unbox]: no allocation that
   it adds to [cannot_unbox0] is unboxed. *)
Definition check_alias_readers (to_unbox : list node) : bool :=
  forallb (fun x => negb (mem_node x to_unbox)) fix_seeds.

(* [to_unbox] as the rules with the fix give it, and the checks of a table
   of [to_unbox] and of [to_change_representation] against those rules.
   [check_to_unbox] rejects a table of [derive_to_unbox_fixed] that
   differs from [derive_to_unbox]: it is not what the rules without the
   fix give. *)
Definition derive_to_unbox_fixed : list node := derive_to_unbox_with fix_seeds.

Definition check_to_unbox_fixed (to_unbox : list node) : bool :=
  check_to_unbox_with fix_seeds to_unbox.

Definition check_to_change_representation_fixed (to_unbox tcr : list node)
    : bool :=
  check_to_change_representation_with fix_seeds to_unbox tcr.

End Rules.

(* [to_change_representation] from what the compiler exports: the
   allocations with a use among those that [changed_representation] of
   [perform_analysis0] gives with each name ([alloc_points]). For a set,
   these are its closures, which go together. *)
Definition to_change_representation_of_alloc_points (T : tables)
    (alloc_points : list node) : list node :=
  filter (fun x => mem_node x (t_has_usage T))
    (Keyed.dedup node_eqb alloc_points).

(* [to_unbox] from what the compiler exports: the allocations that
   dominate a name that [Analysis.get_unboxed_fields] unboxes ([names]).
   An allocation that dominates no name, which has no source and whose
   [Let] is [Invalid], is not found, and makes no difference to the
   rebuild. *)
Definition to_unbox_of_unboxed_names (dominated : list (node * node))
    (names : list node) : list node :=
  Keyed.dedup node_eqb
    (flat_map
       (fun '(a, x) => if mem_node x names then [a] else [])
       dominated).

(* ======================================================================== *)
(* The unboxed fields: [mk_unboxed_fields]                                  *)
(* ======================================================================== *)

(* [Unboxing_analysis.Unboxed_fields.u], of variables: a field is a
   variable of its own, or unboxed in turn. [Unboxed_fields.t] is a list
   of fields in the order of [field_lt], each once. *)
#[local] Set Warnings "-register-all".
Inductive unboxed_field : Type :=
  | Not_unboxed (x : variable)
  | Unboxed (fields : list (field * unboxed_field)).

Definition unboxed_fields : Type := list (field * unboxed_field).

(* [Unboxed_fields.fold_with_kind]: the variables of an unboxing, each with
   the kind of the field it stands for, in the order of the fields. *)
Fixpoint unboxed_field_leaves (kind : flambda_kind) (u : unboxed_field)
    {struct u} : list (variable * flambda_kind) :=
  match u with
  | Not_unboxed x => [(x, kind)]
  | Unboxed fields =>
    (fix leaves (fields : list (field * unboxed_field))
         : list (variable * flambda_kind) :=
       match fields with
       | [] => []
       | (f, u) :: fields =>
         unboxed_field_leaves (field_kind f) u ++ leaves fields
       end) fields
  end.

Definition fields_leaves (fields : unboxed_fields)
    : list (variable * flambda_kind) :=
  unboxed_field_leaves Value (Unboxed fields).

(* [Unboxed_fields.fold2_subset_u]: each variable of [u], with the one of
   [u'] for the same field and the kind of the field, in the order of the
   fields of [u]. [None] where a field of [u] is not in [u'], or the two
   are not unboxed alike (a fatal error in the OCaml). *)
Fixpoint unboxed_field_pairs (kind : flambda_kind) (u u' : unboxed_field)
    {struct u} : option (list (variable * variable * flambda_kind)) :=
  match u, u' with
  | Not_unboxed x, Not_unboxed x' => Some [(x, x', kind)]
  | Unboxed fields, Unboxed fields' =>
    (fix pairs (fields : list (field * unboxed_field))
         : option (list (variable * variable * flambda_kind)) :=
       match fields with
       | [] => Some []
       | (f, u) :: fields =>
         match Keyed.lookup field_eqb f fields' with
         | Some u' =>
           match unboxed_field_pairs (field_kind f) u u', pairs fields with
           | Some p, Some q => Some (p ++ q)
           | _, _ => None
           end
         | None => None
         end
       end) fields
  | Not_unboxed _, Unboxed _ | Unboxed _, Not_unboxed _ => None
  end.

(* [Or_unknown.t] of the nodes a field is read into ([get_fields]). *)
Inductive field_use : Type :=
  | Use_unknown
  | Use_known (flow_to : list node).

(* [Points_to_analysis.single_field_source]. *)
Inductive single_field_source : Type :=
  | No_source
  | One_source (source : node)
  | Many_sources.

(* The fields that an unboxing gives a variable of their own. The others
   are [Call_witness], which is left out, and the fields that are read
   only from call witnesses and function slots (a fatal error in the
   OCaml), which are left out too. *)
Definition is_data_field (f : field) : bool :=
  match f with
  | Field_block _ _ | Field_value_slot _ | Field_is_int | Field_get_tag
  | Field_boxed_number _ => true
  | Field_function_slot _ | Field_call_witness _ | Field_return_of_call _
  | Field_code_id_of_call_witness => false
  end.

(* The [my_closure] of the code that the [Let_static]s of an expression
   bind, where [static_names] looks: the relation [code_id_my_closure] of
   the compiler's graph, which the facts do not have. *)
Fixpoint bound_code_my_closures (e : expr) : finite_map variable :=
  match e with
  | Let _ _ _ body => bound_code_my_closures body
  | Let_static bindings body =>
    flat_map
      (fun binding =>
         match binding with
         | Bind_code id c => [(id, code_my_closure c)]
         | Bind_deleted_code _ | Bind_symbol _ _ => []
         end)
      bindings
    ++ bound_code_my_closures body
  | Let_cont_non_recursive _ h body =>
    handler_bound_code_my_closures h ++ bound_code_my_closures body
  | Let_cont_recursive handlers body =>
    flat_map (fun '(_, h) => handler_bound_code_my_closures h) handlers
    ++ bound_code_my_closures body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => []
  end
with handler_bound_code_my_closures (h : continuation_handler)
    : finite_map variable :=
  match h with
  | Handler _ _ handler => bound_code_my_closures handler
  end.

(* The slots, value and function, that a defining expression or a static
   constant names. *)
Definition slots_of_named (d : named) : list positive :=
  match d with
  | Set_of_closures _ value_slots => map fst value_slots
  | Closure_of_set slot function_decls value_slots =>
    slot :: map fst function_decls ++ map fst value_slots
  | Project_value_slot w _ => [w]
  | Project_value_slot_from project_from w _ => [project_from; w]
  | Project_function_slot move_from move_to _ => [move_from; move_to]
  | Simple _ | Prim _ _ => []
  end.

Definition slots_of_static_const (const : static_const) : list positive :=
  match const with
  | Static_const_set_of_closures _ value_slots => map fst value_slots
  | Static_const_closure_of_set slot closure_symbols function_decls
      value_slots =>
    slot :: map fst closure_symbols ++ map fst function_decls
    ++ map fst value_slots
  | Static_const_block _ _ | Static_const_boxed_number _
  | Static_const_immutable_string _
  | Static_const_immutable_value_array _ => []
  end.

(* Every slot that the unit names, in its code too: the slots of a change
   of representation are the names after them. *)
Fixpoint slots_of (e : expr) : list positive :=
  match e with
  | Let _ _ d body => slots_of_named d ++ slots_of body
  | Let_static bindings body =>
    flat_map binding_slots bindings ++ slots_of body
  | Let_cont_non_recursive _ h body => handler_slots h ++ slots_of body
  | Let_cont_recursive handlers body =>
    flat_map (fun '(_, h) => handler_slots h) handlers ++ slots_of body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => []
  end
with handler_slots (h : continuation_handler) : list positive :=
  match h with
  | Handler _ _ handler => slots_of handler
  end
with binding_slots (binding : static_binding) : list positive :=
  match binding with
  | Bind_code _ (Code _ _ _ _ _ body) => slots_of body
  | Bind_deleted_code _ => []
  | Bind_symbol _ const => slots_of_static_const const
  end.

(* [Unboxing_analysis.changed_representation] of a closure
   ([Closure_representation]): the value slots of the set kept, each a
   new slot or unboxed in turn into new slots ([cr_fields], whose leaves
   are value slots); the new function slot of each function slot of the
   set ([cr_function_slots]); and the function slot of the closure, before
   the change ([cr_current]). *)
Record closure_repr : Type := Closure_repr {
  cr_fields : unboxed_fields;
  cr_function_slots : list (function_slot * function_slot);
  cr_current : function_slot
}.

Section Unboxed_fields.

Variable F : facts.
Variable T : tables.
Variable code_my_closures : finite_map variable.
(* The tables of [dominated_by_allocation_point] and [to_unbox]. *)
Variable dominated : list (node * node).
Variable to_unbox : list node.

Abbreviation any_usage x := (mem_node x (t_any_usage T)).
Abbreviation any_source x := (mem_node x (t_any_source T)).
Abbreviation has_usage x := (mem_node x (t_has_usage T)).
Abbreviation has_source x := (mem_node x (t_has_source T)).

(* [has_to_be_unboxed]: [x] is dominated by an allocation to unbox. *)
Definition has_to_be_unboxed (x : node) : bool :=
  existsb (fun '(a, x') => node_eqb x x' && mem_node a to_unbox) dominated.

(* [get_direct_usages]. *)
Definition get_direct_usages (in_ : list node) : list node :=
  Keyed.dedup node_eqb
    (flat_map
       (fun '(x, y) => if mem_node x in_ && has_usage y then [y] else [])
       (t_usages T)).

(* The usages of the closures in [out] through a call of known arity
   ([follow]: the uses of the [my_closure] of the code called) and through
   their function slots: one round of the two rules of
   [add_usages_through_function_slots]. *)
Definition usages_through_function_slots (follow : bool) (out : list node)
    : list node :=
  flat_map
    (fun x =>
       (if follow
        then
          flat_map
            (fun '(apply_witness, f, base) =>
               if node_eqb base x
                  && field_eqb f (Field_call_witness Known_arity_code_pointer)
               then
                 flat_map
                   (fun '(apply_witness', call_witness) =>
                      if node_eqb apply_witness apply_witness'
                      then
                        flat_map
                          (fun '(call_witness', f', code_id) =>
                             if node_eqb call_witness call_witness'
                                && field_eqb f' Field_code_id_of_call_witness
                             then
                               match code_id with
                               | Node_code_id id =>
                                 match lookup id code_my_closures with
                                 | Some my_closure =>
                                   flat_map
                                     (fun '(my_closure', y) =>
                                        if node_eqb (Node_variable my_closure)
                                             my_closure'
                                           && has_usage y
                                        then [y]
                                        else [])
                                     (t_usages T)
                                 | None => []
                                 end
                               | _ => []
                               end
                             else [])
                          (f_constructor F)
                      else [])
                   (t_sources T)
               else [])
            (f_accessor F)
        else [])
       ++ flat_map
            (fun '(y, f, base) =>
               if node_eqb base x && is_function_slot f
               then
                 flat_map
                   (fun '(y', z) =>
                      if node_eqb y y' && has_usage z then [z] else [])
                   (t_usages T)
               else [])
            (f_accessor F))
    out.

Fixpoint add_usages_rounds (follow : bool) (fuel : nat) (out : list node)
    : list node :=
  match fuel with
  | O => out
  | S fuel =>
    match
      filter (fun y => negb (mem_node y out))
        (Keyed.dedup node_eqb (usages_through_function_slots follow out))
    with
    | [] => out
    | added => add_usages_rounds follow fuel (out ++ added)
    end
  end.

(* [add_usages_through_function_slots], to the fixpoint: a round that adds
   a node adds one of the tables, so that [S (length (t_usages T))]
   rounds are enough. *)
Definition add_usages_through_function_slots (follow : bool)
    (in_ : list node) : list node :=
  add_usages_rounds follow (S (length (t_usages T))) in_.

(* [get_fields]: each field read from the nodes [uses], other than a
   function slot, with the nodes it is read into that have a use, or
   [Use_unknown] if one of them has any usage; in the order of the
   fields. *)
Definition get_fields (uses : list node) : list (field * field_use) :=
  let reads :=
    filter
      (fun '(_, f, base) => mem_node base uses && negb (is_function_slot f))
      (f_accessor F)
  in
  sort_fields
    (flat_map
       (fun f =>
          let read_into :=
            flat_map (fun '(v, f', _) => if field_eqb f f' then [v] else [])
              reads
          in
          if existsb (fun v => any_usage v) read_into
          then [(f, Use_unknown)]
          else
            match Keyed.dedup node_eqb (filter (fun v => has_usage v) read_into)
            with
            | [] => []
            | flow_to => [(f, Use_known flow_to)]
            end)
       (Keyed.dedup field_eqb (map (fun '(_, f, _) => f) reads))).

(* [get_single_field_source]. *)
Definition get_single_field_source (block : node) (f : field)
    : single_field_source :=
  if existsb
       (fun '(base, f', source) =>
          node_eqb base block && field_eqb f f' && any_source source)
       (f_constructor F)
  then Many_sources
  else
    match
      Keyed.dedup node_eqb
        (flat_map
           (fun '(base, f', field_source) =>
              if node_eqb base block && field_eqb f f'
              then
                flat_map
                  (fun '(field_source', source) =>
                     if node_eqb field_source field_source' && has_source source
                     then [source]
                     else [])
                  (t_sources T)
              else [])
           (f_constructor F))
    with
    | [] => No_source
    | [source] => One_source source
    | _ :: _ :: _ => Many_sources
    end.

(* [mk_unboxed_fields] for the allocation [block] and the fields read
   [fields]: a field with no source is left out, one read into names that
   are all to be unboxed is unboxed in turn, from the one allocation it
   may come from, and any other is a variable, the names from [next] on,
   in the order of the fields. The OCaml stops with a fatal error where
   such a field may come from several allocations, or is read into names
   of which some are to be unboxed and some not; the model makes it a
   variable. [fuel] bounds the depth of the unboxing, which a chain of
   allocations to unbox, each in a field of the one before, cannot
   exceed: where it runs out, the field is a variable. *)
Fixpoint mk_unboxed_fields (fuel : nat) (block : node)
    (fields : list (field * field_use)) (next : positive) {struct fuel}
    : unboxed_fields * positive :=
  (fix go (fields : list (field * field_use)) (next : positive)
       : unboxed_fields * positive :=
     match fields with
     | [] => ([], next)
     | (f, use) :: fields =>
       let '(u, next) :=
         if negb (is_data_field f)
         then (None, next)
         else
           match get_single_field_source block f with
           | No_source => (None, next)
           | field_source =>
             let default := (Some (Not_unboxed next), Pos.succ next) in
             match use with
             | Use_unknown => default
             | Use_known flow_to =>
               if forallb has_to_be_unboxed flow_to
               then
                 match field_source, fuel with
                 | One_source source, S fuel =>
                   let '(nested, next) :=
                     mk_unboxed_fields fuel source
                       (get_fields
                          (add_usages_through_function_slots true
                             (get_direct_usages flow_to)))
                       next
                   in
                   (Some (Unboxed nested), next)
                 | _, _ => default
                 end
               else default
             end
           end
       in
       let '(rest, next) := go fields next in
       (match u with
        | Some u => (f, u) :: rest
        | None => rest
        end,
        next)
     end) fields next.

(* The unboxed fields of [x], dominated by the allocation [a] to unbox
   ([query_to_unbox] of [perform_analysis0]): the fields of [a] read from
   the uses of [x]. *)
Definition name_unboxed_fields (a x : node) (next : positive)
    : unboxed_fields * positive :=
  mk_unboxed_fields (S (length to_unbox)) a
    (get_fields
       (add_usages_through_function_slots true (get_direct_usages [x])))
    next.

(* [unboxed_fields] of [perform_analysis0]: the unboxed fields of each
   name dominated by an allocation to unbox, the names in the order of
   [dominated], each once, and their variables from [next] on, in that
   order. Returns the name after the last variable. *)
Definition unboxed_fields_of_names (next : positive)
    : list (node * unboxed_fields) * positive :=
  fold_left
    (fun '(acc, next) '(a, x) =>
       if mem_node a to_unbox
          && negb (existsb (fun '(y, _) => node_eqb x y) acc)
       then
         let '(fields, next) := name_unboxed_fields a x next in
         (acc ++ [(x, fields)], next)
       else (acc, next))
    dominated ([], next).

(* The changes of representation: the fields of a set of closures as
   [get_fields_usage_of_constructors] gives them, read off the facts and
   the tables by the rules of [field_of_constructor_is_used_top] and
   [field_of_constructor_is_used_as] of [post_processing_rules]
   (points_to_analysis.ml), which read the local fields. *)
Variable is_local : field -> bool.

(* The nodes that a field [f] of [usage] is read into, which have a use
   ([rev_accessor ~base:usage f ~to_:v; has_usage v]), and whether one of
   them has any usage. *)
Definition read_into (usage : node) (f : field) : list node :=
  flat_map
    (fun '(v, f', usage') =>
       if node_eqb usage usage' && field_eqb f f' && has_usage v
       then [v]
       else [])
    (f_accessor F).

Definition read_into_any_usage (usage : node) (f : field) : bool :=
  existsb
    (fun '(v, f', usage') =>
       node_eqb usage usage' && field_eqb f f' && any_usage v)
    (f_accessor F).

Definition has_constructor (base : node) (f : field) : bool :=
  existsb (fun '(base', f', _) => node_eqb base base' && field_eqb f f')
    (f_constructor F).

(* The uses of [base] ([usages base usage]), and, for a local field, the
   names that it is a source of ([sources usage base]). *)
Definition uses_of (base : node) : list node :=
  flat_map (fun '(x, usage) => if node_eqb x base then [usage] else [])
    (t_usages T).

Definition sourced_by (base : node) : list node :=
  flat_map (fun '(usage, x) => if node_eqb x base then [usage] else [])
    (t_sources T).

(* [field_of_constructor_is_used_top base f]: the four rules that
   conclude it. *)
Definition field_used_top (base : node) (f : field) : bool :=
  has_constructor base f
  && ((any_usage base && negb (is_local f))
      || existsb (fun usage => read_into_any_usage usage f) (uses_of base)
      || (any_usage base
          && existsb (fun '(f', x) => field_eqb f f' && any_usage x)
               (t_reading_field T))
      || (is_local f && any_usage base
          && existsb (fun usage => read_into_any_usage usage f)
               (sourced_by base))).

(* The [v] of [field_of_constructor_is_used_as base f v]: the three rules
   that conclude it. *)
Definition field_used_as (base : node) (f : field) : list node :=
  if has_constructor base f
  then
    flat_map (fun usage => read_into usage f) (uses_of base)
    ++ (if any_usage base
        then
          flat_map
            (fun '(f', x) => if field_eqb f f' && has_usage x then [x] else [])
            (t_reading_field T)
        else [])
    ++ (if is_local f && any_usage base
        then flat_map (fun usage => read_into usage f) (sourced_by base)
        else [])
  else [].

(* [get_fields_usage_of_constructors] of the closures [closures]: each
   field of one of them other than a function slot, [Use_unknown] if
   [field_used_top] holds of one, else the nodes of [field_used_as], if
   any; in the order of the fields. *)
Definition get_fields_of_constructors (closures : list node)
    : list (field * field_use) :=
  sort_fields
    (flat_map
       (fun f =>
          if existsb (fun x => field_used_top x f) closures
          then [(f, Use_unknown)]
          else
            match
              Keyed.dedup node_eqb
                (flat_map (fun x => field_used_as x f) closures)
            with
            | [] => []
            | flow_to => [(f, Use_known flow_to)]
            end)
       (Keyed.dedup field_eqb
          (flat_map
             (fun '(base, f, _) =>
                if mem_node base closures && negb (is_function_slot f)
                then [f]
                else [])
             (f_constructor F)))).

(* [get_set_of_closures_def]: each function slot of the set of which [x]
   is a closure, with its closure; none if [x] is not a closure of a
   set. *)
Definition set_function_slots (x : node) : list (function_slot * node) :=
  compact
    (flat_map
       (fun '(base, f, y) =>
          match f with
          | Field_function_slot fs => if node_eqb base x then [(fs, y)] else []
          | _ => []
          end)
       (f_constructor F)).

(* Each slot given a new one, the names from [next] on. *)
Fixpoint rename_slots (slots : list function_slot) (next : positive)
    : list (function_slot * function_slot) * positive :=
  match slots with
  | [] => ([], next)
  | fs :: slots =>
    let '(rest, next') := rename_slots slots (Pos.succ next) in
    ((fs, next) :: rest, next')
  end.

(* [changed_representation] of [perform_analysis0], for the allocations
   of the table [tcr] in its order: the first closure of a set met gives
   the representation of the set, [mk_unboxed_fields] of the fields of
   its closures with a new value slot for each field not unboxed in turn,
   and a new function slot for each of its function slots; each name
   dominated by a closure of the set has it, with the function slot of
   that closure. The new slots are the names from [next] on: the value
   slots in the order of the fields, then the function slots. Returns the
   name after the last. An allocation that is not a closure of a set
   would have a [Block_representation], which is left out. *)
Definition changed_representations (tcr : list node) (next : positive)
    : list (node * closure_repr) * positive :=
  let '(acc, _, next) :=
    fold_left
      (fun '(acc, seen, next) x =>
         if mem_node x seen
         then (acc, seen, next)
         else
           match set_function_slots x with
           | [] => (acc, seen, next)
           | l =>
             let '(fields, next) :=
               mk_unboxed_fields (S (length to_unbox)) x
                 (get_fields_of_constructors (map snd l)) next
             in
             let '(fss, next) := rename_slots (map fst l) next in
             (acc
              ++ flat_map
                   (fun '(fs, f) =>
                      flat_map
                        (fun '(a, c) =>
                           if node_eqb a f
                           then [(c, Closure_repr fields fss fs)]
                           else [])
                        dominated)
                   l,
              seen ++ map snd l, next)
           end)
      tcr ([], [], next)
  in
  (acc, next).

End Unboxed_fields.

(* The unboxed fields of the names of the unit [e], with variables after
   every name of the unit. *)
Definition unit_unboxed_fields (F : facts) (T : tables)
    (dominated : list (node * node)) (to_unbox : list node) (e : expr)
    : list (node * unboxed_fields) * positive :=
  unboxed_fields_of_names F T (bound_code_my_closures e) dominated to_unbox
    (fresh_name (names_of e)).

(* The changes of representation of the names of the unit [e], for a
   table [tcr] of [to_change_representation], with slots after every slot
   of the unit. *)
Definition unit_changed_representations (is_local : field -> bool)
    (F : facts) (T : tables) (dominated : list (node * node))
    (to_unbox tcr : list node) (e : expr) : list (node * closure_repr) :=
  fst
    (changed_representations F T (bound_code_my_closures e) dominated
       to_unbox is_local tcr (fresh_name (slots_of e))).

(* ======================================================================== *)
(* The decisions: [compute_code_changes] with [Unbox]                       *)
(* ======================================================================== *)

(* [Unboxing_analysis.param_decision]: one of
   FlambdaReaperCallingConventions.v, [Keep] or [Delete], or [Unbox] with
   the unboxed fields of the parameter. In the mode of the fix,
   [Keep_unused] marks the [Keep] of a parameter of a continuation kept
   though it has no use; the rebuild treats it as [Keep]. *)
Inductive decision : Type :=
  | Decided (d : param_decision)
  | Unbox (fields : unboxed_fields)
  | Keep_unused (kind : flambda_kind).

(* The parameters that a decision makes of [x] ([get_parameters]): [x] if
   kept, the variables of its fields if unboxed. *)
Definition decision_params (x : variable) (d : decision)
    : list (variable * flambda_kind) :=
  match d with
  | Decided (Keep kind) | Keep_unused kind => [(x, kind)]
  | Decided Delete => []
  | Unbox fields => fields_leaves fields
  end.

Definition decision_kinds (d : decision) : list flambda_kind :=
  map snd (decision_params xH d).

Definition decisions_kinds (decisions : list decision) : list flambda_kind :=
  flat_map decision_kinds decisions.

(* [Unboxing_analysis.arity_of_decisions]: one product. *)
Definition arity_of_decisions_u (decisions : list decision) : arity :=
  [Unboxed_product (map Singleton (decisions_kinds decisions))].

(* The arity of [get_arity_and_modes]: a product for each parameter of
   the code. *)
Definition arity_of_grouped_decisions_u (groups : list (list decision))
    : arity :=
  map (fun ds => Unboxed_product (map Singleton (decisions_kinds ds))) groups.

(* A list put at the front of the first group, as [compute_code_changes]
   puts the decision of [my_closure] and [rebuild_apply] the fields of an
   unboxed callee; with no group (a fatal error in the OCaml) it is left
   out. *)
Definition prepend_first {A : Type} (l : list A) (groups : list (list A))
    : list (list A) :=
  match groups with
  | [] => []
  | first :: rest => (l ++ first) :: rest
  end.

(* The metadata of code that changes its calling convention
   ([compute_code_changes]), the decision of [my_closure] being
   [my_closure_decision]: [Decided Delete] for [Keep_my_closure], whose
   closure is not a parameter, and [Unbox] for [Unbox_my_closure]. *)
Definition changed_code_metadata_u (metadata : code_metadata)
    (my_closure_decision : decision)
    (params_decisions return_decisions : list decision) : code_metadata :=
  {| cm_params_arity :=
       arity_of_grouped_decisions_u
         (prepend_first [my_closure_decision]
            (group_by_parameter (cm_params_arity metadata)
               params_decisions));
     cm_result_arity := arity_of_decisions_u return_decisions;
     cm_is_tupled := false;
     cm_is_stub := cm_is_stub metadata |}.

(* [Unboxing_analysis.calling_convention_change], without the decision of
   [my_closure], which is apart ([my_closure_decision_u]). *)
Inductive calling_convention_change_u : Type :=
  | Not_changing_u
  | Changing_u (params_decisions return_decisions : list decision).

Definition all_some {A : Type} (l : list (option A)) : option (list A) :=
  fold_right
    (fun o acc =>
       match o, acc with
       | Some a, Some acc => Some (a :: acc)
       | _, _ => None
       end)
    (Some []) l.

(* ======================================================================== *)
(* The rebuild                                                              *)
(* ======================================================================== *)

(* What the rebuild knows of the names in scope: that of
   FlambdaReaperRebuild.v, and the decisions of each continuation bound in
   the term ([cont_params_to_keep]), whose masks are those of the
   continuations in [ue_env]. *)
Record unbox_env : Type := Unbox_env {
  ue_env : rebuild_env;
  ue_decisions : finite_map (list decision);
  (* The projections of the closures of a set that are next are dropped:
     the set is unboxed. *)
  ue_drop_projections : bool
}.

Definition empty_unbox_env : unbox_env :=
  {| ue_env := empty_env; ue_decisions := []; ue_drop_projections := false |}.

(* The variables bound, and the projections of the closures of a set that
   are next, to be dropped if [drop]. *)
Definition ubind_set (en : unbox_env) (bound : list (variable * flambda_kind))
    (projections : list variable) (drop : bool) : unbox_env :=
  {| ue_env := bind_variables_then (ue_env en) bound projections;
     ue_decisions := ue_decisions en;
     ue_drop_projections := drop |}.

Definition ubind_variables_then (en : unbox_env)
    (bound : list (variable * flambda_kind)) (projections : list variable)
    : unbox_env :=
  ubind_set en bound projections false.

Definition ubind_variables (en : unbox_env)
    (bound : list (variable * flambda_kind)) : unbox_env :=
  ubind_variables_then en bound [].

Definition is_kept_decision (d : decision) : bool :=
  match d with
  | Decided d => is_kept d
  | Unbox _ | Keep_unused _ => true
  end.

Definition ubind_continuations (en : unbox_env)
    (bound : list (continuation * list decision)) : unbox_env :=
  {| ue_env :=
       bind_continuations (ue_env en)
         (map (fun '(k, ds) => (k, map is_kept_decision ds)) bound);
     ue_decisions := bound ++ ue_decisions en;
     ue_drop_projections := false |}.

(* [get_simple_kind]. *)
Definition simple_kind (en : rebuild_env) (s : simple) : flambda_kind :=
  match s with
  | Var x => kind_of_variable en x
  | Symbol _ => Value
  | Const c => kind_of_const c
  end.

(* The parameters that the decisions make of the parameters of a handler
   or of code; a parameter beyond the decisions is kept. *)
Fixpoint params_of_decisions (bound : list (variable * flambda_kind))
    (decisions : list decision) : list (variable * flambda_kind) :=
  match bound, decisions with
  | [], _ => []
  | (x, kind) :: bound, [] => (x, kind) :: params_of_decisions bound []
  | (x, _) :: bound, d :: decisions =>
    decision_params x d ++ params_of_decisions bound decisions
  end.

(* [freshen_decisions] of the decisions of a callee for its results, each
   with the kind of the result in the return arity of the call: the
   variables of the wrapper of [make_apply_wrapper], the names from
   [next] on. *)
Inductive wrapper_decision : Type :=
  | Wrapper_keep (x : variable)
  | Wrapper_delete
  | Wrapper_unbox (fields : unboxed_fields).

Fixpoint rename_unboxed_field (u : unboxed_field) (next : positive)
    {struct u} : unboxed_field * positive :=
  match u with
  | Not_unboxed _ => (Not_unboxed next, Pos.succ next)
  | Unboxed fields =>
    let '(fields, next) :=
      (fix rename (fields : list (field * unboxed_field)) (next : positive)
           : unboxed_fields * positive :=
         match fields with
         | [] => ([], next)
         | (f, u) :: fields =>
           let '(u, next) := rename_unboxed_field u next in
           let '(fields, next) := rename fields next in
           ((f, u) :: fields, next)
         end) fields next
    in
    (Unboxed fields, next)
  end.

Fixpoint freshen_decisions (decisions : list (decision * flambda_kind))
    (next : positive) : list (wrapper_decision * flambda_kind) :=
  match decisions with
  | [] => []
  | (Decided (Keep _), kind) :: decisions
  | (Keep_unused _, kind) :: decisions =>
    (Wrapper_keep next, kind) :: freshen_decisions decisions (Pos.succ next)
  | (Decided Delete, kind) :: decisions =>
    (Wrapper_delete, kind) :: freshen_decisions decisions next
  | (Unbox fields, kind) :: decisions =>
    match rename_unboxed_field (Unboxed fields) next with
    | (Unboxed fields, next) =>
      (Wrapper_unbox fields, kind) :: freshen_decisions decisions next
    | (Not_unboxed _, next) =>
      (Wrapper_delete, kind) :: freshen_decisions decisions next
    end
  end.

(* [get_parameters] of the decisions of the wrapper. *)
Definition wrapper_params (func : list (wrapper_decision * flambda_kind))
    : list (variable * flambda_kind) :=
  flat_map
    (fun '(d, kind) =>
       match d with
       | Wrapper_keep x => [(x, kind)]
       | Wrapper_delete => []
       | Wrapper_unbox fields => fields_leaves fields
       end)
    func.

(* [rev_args_or_invalid] of [make_apply_wrapper]: for each result, what
   the continuation of the call is given, by the decision of the
   continuation for its parameter and that of the callee. A result that
   the continuation deletes is dropped, whatever the callee does with it;
   otherwise [None] where one decision unboxes and the other does not
   ([Invalid]). A result beyond the decisions of the continuation, or of
   a continuation that has none, is kept, as a continuation of fixed
   arity keeps it. In the mode of the fix ([fixed]), a parameter kept is
   given poison for a result that the callee unboxes, as for one that it
   deletes; one unboxed for a result that the callee does not unbox is
   the fatal error of the fix ([wrapper_fails_u]), [None] here. *)
Fixpoint wrapper_args_u (fixed : bool) (apply_decisions : list decision)
    (func : list (wrapper_decision * flambda_kind)) : option (list simple) :=
  match func with
  | [] => Some []
  | (fd, kind) :: func =>
    let '(ad, apply_decisions) :=
      match apply_decisions with
      | [] => (Decided (Keep kind), [])
      | ad :: apply_decisions => (ad, apply_decisions)
      end
    in
    match wrapper_args_u fixed apply_decisions func with
    | None => None
    | Some rest =>
      match ad, fd with
      (* First since the fix "reaper: fix miscompilation with unused
         unboxed variables" (#7952): before, a result that the callee
         unboxes and the continuation deletes was the case below. *)
      | Decided Delete, _ => Some rest
      | Decided (Keep _), Wrapper_keep x | Keep_unused _, Wrapper_keep x =>
        Some (Var x :: rest)
      | Decided (Keep kind), Wrapper_delete
      | Keep_unused kind, (Wrapper_delete | Wrapper_unbox _) =>
        Some (Const (Poison kind) :: rest)
      (* The case where the compiler on the branch stops with a fatal
         error if [has_any_source], a test that is [false] there ("CR
         ncourant: restore this check"); the fix gives poison to a
         parameter kept, and stops with a fatal error for one
         unboxed. *)
      | Decided (Keep kind), Wrapper_unbox _ =>
        if fixed then Some (Const (Poison kind) :: rest) else None
      | Unbox _, (Wrapper_keep _ | Wrapper_delete) => None
      | Unbox fields_apply, Wrapper_unbox fields_func =>
        match
          unboxed_field_pairs Value (Unboxed fields_apply)
            (Unboxed fields_func)
        with
        | Some pairs => Some (map (fun '(_, x, _) => Var x) pairs ++ rest)
        | None => None
        end
      end
    end
  end.

(* The fatal error of [make_apply_wrapper] in the fix: a parameter that
   the continuation unboxes for a result that the callee keeps or
   deletes, at some position of the decisions of the continuation
   [apply_decisions] and those of the callee [func]. *)
Fixpoint wrapper_fails_u (apply_decisions func : list decision) : bool :=
  match apply_decisions, func with
  | ad :: apply_decisions, fd :: func =>
    match ad, fd with
    | Unbox _, (Decided _ | Keep_unused _) => true
    | _, _ => wrapper_fails_u apply_decisions func
    end
  | _, _ => false
  end.

(* The arguments are the parameters, in order. *)
Fixpoint args_are_params (args : list simple) (params : list variable)
    : bool :=
  match args, params with
  | [], [] => true
  | Var x :: args, y :: params => Pos.eqb x y && args_are_params args params
  | _, _ => false
  end.

(* The field that a defining expression reads, and from what: the cases
   of [rebuild_named_default_case] that [rewrite_field_access] rebuilds
   when the value read from is unboxed. *)
Definition field_read (d : named) : option (field * simple) :=
  match d with
  | Prim (Unary (Block_load _ _ i)) [arg] =>
    Some (Field_block (Z.to_nat i) Value, arg)
  | Prim (Unary (Is_int true)) [arg] => Some (Field_is_int, arg)
  | Prim (Unary Get_tag) [arg] => Some (Field_get_tag, arg)
  | Prim (Unary (Unbox_number bn)) [arg] => Some (Field_boxed_number bn, arg)
  | Project_value_slot w arg | Project_value_slot_from _ w arg =>
    Some (Field_value_slot w, arg)
  | _ => None
  end.

Section Rebuild.

(* The mode: [false] for the rebuild of the compiler on the branch,
   [true] for that of the fix. *)
Variable fixed : bool.
Variable F : facts.
Variable sol : solution.
Variable cannot_change : node -> bool.
Variable code_deps : finite_map code_dep.
(* [should_preserve_direct_calls] is [Yes] or [Auto]. *)
Variable preserve_direct_calls : bool.
(* The code ids of the unit, and the metadata of its code. *)
Variable unit_code_ids : list code_id.
Variable codes : finite_map code_metadata.
(* The unboxed fields of the names ([unboxed_fields_of_names]). *)
Variable unboxed : list (node * unboxed_fields).
(* The first of the names of the wrappers. *)
Variable first_fresh : positive.
(* The changes of representation of the names
   ([changed_representations]). *)
Variable changed : list (node * closure_repr).

(* [Analysis.get_unboxed_fields]. *)
Definition unboxed_of_node (n : node) : option unboxed_fields :=
  Keyed.lookup node_eqb n unboxed.

(* [get_simple_unboxable]: the unboxed fields of a name. *)
Definition unboxed_of (s : simple) : option unboxed_fields :=
  match s with
  | Var x => unboxed_of_node (Node_variable x)
  | Symbol sym => unboxed_of_node (Node_symbol sym)
  | Const _ => None
  end.

(* [Analysis.get_changed_representation], and [get_simple_changed_repr]. *)
Definition changed_of_node (n : node) : option closure_repr :=
  Keyed.lookup node_eqb n changed.

Definition changed_of (s : simple) : option closure_repr :=
  match s with
  | Var x => changed_of_node (Node_variable x)
  | Symbol sym => changed_of_node (Node_symbol sym)
  | Const _ => None
  end.

(* The new function slot of a function slot of the set ([Function_slot.Map.find
   ... function_slots]); one that the set has not (a fatal error in the
   OCaml) stays. *)
Definition rename_function_slot (repr : closure_repr) (fs : function_slot)
    : function_slot :=
  match lookup fs (cr_function_slots repr) with
  | Some fs' => fs'
  | None => fs
  end.

(* The new function slot of the closure, which a projection names. *)
Definition current_slot (repr : closure_repr) : function_slot :=
  rename_function_slot repr (cr_current repr).

Definition is_unboxed_name (n : node) : bool :=
  match n with
  | Node_variable _ | Node_symbol _ => Keyed.is_bound node_eqb n unboxed
  | Node_code_id _ | Node_all_constants | Node_made _ => false
  end.

(* The solution in which the names unboxed have no use. Given it,
   FlambdaReaperRebuild.rewrite_simple is [rewrite_simple] of rebuild.ml,
   which makes an unboxed name poison of its kind
   ("reaper_rewrite_simple_unboxed"); so are the functions of
   FlambdaReaperRebuild.v that go through it. Nothing else of those that
   the rebuild calls with it reads [has_usage] of a variable or a symbol
   but [rewrite_or_variable]. *)
Definition solution_without_unboxed : solution :=
  {| s_alias := s_alias sol;
     s_any_usage := s_any_usage sol;
     s_any_source := s_any_source sol;
     s_has_usage := fun n => s_has_usage sol n && negb (is_unboxed_name n);
     s_has_source := s_has_source sol;
     s_usages := s_usages sol;
     s_sources := s_sources sol;
     s_reading_field := s_reading_field sol;
     s_escaping_field := s_escaping_field sol;
     s_field_used := s_field_used sol |}.

Abbreviation sol_u := solution_without_unboxed.

(* [bind_fields]: the variables of [u] bound to those of [u'] for the same
   fields; the first is the innermost. *)
Definition bind_fields (u u' : unboxed_field) (body : expr) : option expr :=
  match unboxed_field_pairs Value u u' with
  | Some pairs =>
    Some
      (fold_left (fun body '(x, x', kind) => Let x kind (Simple (Var x')) body)
         pairs body)
  | None => None
  end.

(* [bind_field_to_simple]. *)
Definition bind_field_to_simple (u : unboxed_field) (kind : flambda_kind)
    (s : simple) (body : expr) : option expr :=
  match u with
  | Not_unboxed x => Some (Let x kind (Simple s) body)
  | Unboxed _ => None
  end.

(* The fields of a block made unboxed ([Make_block] of
   [rebuild_singleton_binding_which_is_being_unboxed]): each bound to its
   argument, or to the variables of its argument if that is unboxed; a
   field whose argument is missing or of another kind is poison; [Is_int]
   is false and [Get_tag] the tag. The first field is the innermost. *)
Definition bind_block_fields (en : rebuild_env) (tag : Z) (args : list simple)
    (to_bind : unboxed_fields) (body : expr) : option expr :=
  fold_left
    (fun acc '(f, u) =>
       match acc with
       | None => None
       | Some body =>
         match f with
         | Field_block nth kind =>
           let arg :=
             match nth_error args nth with
             | Some arg =>
               if kind_eqb kind (simple_kind en arg)
               then arg
               else Const (Poison kind)
             | None => Const (Poison kind)
             end
           in
           match unboxed_of arg with
           | Some arg_fields => bind_fields u (Unboxed arg_fields) body
           | None => bind_field_to_simple u kind arg body
           end
         | Field_is_int =>
           bind_field_to_simple u (field_kind f) (Const (Naked_immediate 0))
             body
         | Field_get_tag =>
           bind_field_to_simple u (field_kind f) (Const (Naked_immediate tag))
             body
         | _ => None
         end
       end)
    to_bind (Some body).

(* The field of a boxed number made unboxed ([Box_number]). *)
Definition bind_boxed_number_field (bn : boxable_number) (contents : simple)
    (to_bind : unboxed_fields) (body : expr) : option expr :=
  fold_left
    (fun acc '(f, u) =>
       match acc with
       | None => None
       | Some body =>
         match f with
         | Field_boxed_number bn' =>
           if Nat.eqb (boxable_number_index bn) (boxable_number_index bn')
           then bind_field_to_simple u (field_kind f) contents body
           else None
         | _ => None
         end
       end)
    to_bind (Some body).

(* [load_field_from_value_which_is_being_unboxed], for an unboxed value
   read from: the variables of what is read are those of the field. *)
Definition load_unboxed_field (f : field) (arg : simple)
    (to_bind : unboxed_fields) (body : expr) : option expr :=
  match unboxed_of arg with
  | Some arg_fields =>
    match Keyed.lookup field_eqb f arg_fields with
    | Some u => bind_fields (Unboxed to_bind) u body
    | None => None
    end
  | None =>
    (* A value slot of a closure whose representation changes: the
       variables of what is read are projections of the new slots of the
       field. *)
    match changed_of arg, f with
    | Some repr, Field_value_slot _ =>
      match Keyed.lookup field_eqb f (cr_fields repr) with
      | Some u =>
        match unboxed_field_pairs Value (Unboxed to_bind) u with
        | Some pairs =>
          Some
            (fold_left
               (fun body '(x, slot, kind) =>
                  Let x kind
                    (Project_value_slot_from (current_slot repr) slot arg)
                    body)
               pairs body)
        | None => None
        end
      | None => None
      end
    | _, _ => None
    end
  end.

(* [rebuild_singleton_binding_which_is_being_unboxed]: the [Let] of a
   variable unboxed binds the variables of its fields. *)
Definition rebuild_unboxed_binding (en : rebuild_env)
    (to_bind : unboxed_fields) (d : named) (body : expr) : expr :=
  let rebuilt :=
    match d with
    | Prim (Variadic (Make_block (Block_kind_values tag))) args
    | Prim (Heap (Make_block_with_identity _ tag)) args =>
      bind_block_fields en tag args to_bind body
    | Prim (Unary (Box_number bn)) [contents] =>
      bind_boxed_number_field bn contents to_bind body
    | Prim (Unary (Block_load _ _ i)) [arg] =>
      load_unboxed_field (Field_block (Z.to_nat i) Value) arg to_bind body
    | Project_value_slot w arg | Project_value_slot_from _ w arg =>
      load_unboxed_field (Field_value_slot w) arg to_bind body
    | Project_function_slot _ _ arg | Simple arg =>
      match unboxed_of arg with
      | Some arg_fields =>
        bind_fields (Unboxed to_bind) (Unboxed arg_fields) body
      | None => None
      end
    (* The local allocation of [x] was unboxed by the [Let] before. *)
    | Prim (Region_primitive Local_allocation) _ => Some body
    | _ => None
    end
  in
  match rebuilt with
  | Some e => e
  | None => Invalid
  end.

(* [rebuild_let_expr_singleton] of a variable that is not unboxed: a read
   from an unboxed value is the variable of the field read
   ([rewrite_field_access]); anything else is [rebuild_named] of
   FlambdaReaperRebuild.v. *)
Definition rebuild_let_u (en : rebuild_env) (x : variable)
    (kind : flambda_kind) (d : named) (body : expr) : expr :=
  match field_read d with
  | Some (f, arg) =>
    match unboxed_of arg with
    | Some arg_fields =>
      match Keyed.lookup field_eqb f arg_fields with
      | Some (Not_unboxed v) => Let x kind (Simple (Var v)) body
      | Some (Unboxed _) | None => Invalid
      end
    | None =>
      (* [rewrite_field_access_chg_repr]: a value slot of a closure whose
         representation changes is projected at its new slot. *)
      match changed_of arg, f with
      | None, _ | Some _, Field_boxed_number _ =>
        Let x kind (rebuild_named sol_u en x d) body
      | Some repr, _ =>
        match f, Keyed.lookup field_eqb f (cr_fields repr) with
        | Field_value_slot _, Some (Not_unboxed slot) =>
          Let x kind (Project_value_slot_from (current_slot repr) slot arg)
            body
        | _, _ => Invalid
        end
      end
    end
  | None => Let x kind (rebuild_named sol_u en x d) body
  end.

(* [rebuild_singleton_binding_whose_representation_is_being_changed]: a
   projection of a closure is at the new function slots; anything else is
   [rebuild_let_u]. *)
Definition rebuild_changed_binding (en : rebuild_env) (repr : closure_repr)
    (x : variable) (kind : flambda_kind) (d : named) (body : expr) : expr :=
  match d with
  | Project_function_slot move_from move_to arg =>
    Let x kind
      (Project_function_slot (rename_function_slot repr move_from)
         (rename_function_slot repr move_to) arg)
      body
  | _ => rebuild_let_u en x kind d body
  end.

(* The value slots of a set whose representation changes
   ([rewrite_set_of_closures]): for each field of the representation, in
   its order, the new slot bound to the argument of the old slot, or the
   new slots bound to the variables of that argument if it is unboxed. A
   field that the set has not, or that is unboxed in one and not in the
   other (a fatal error in the OCaml), is left out. *)
Definition changed_value_slots (en : rebuild_env) (repr : closure_repr)
    (value_slots : list (value_slot * simple)) : list (value_slot * simple) :=
  flat_map
    (fun '(f, u) =>
       match f with
       | Field_value_slot w =>
         match lookup w value_slots with
         | Some arg =>
           match unboxed_of arg with
           | Some arg_fields =>
             match unboxed_field_pairs Value u (Unboxed arg_fields) with
             | Some pairs => map (fun '(slot, x, _) => (slot, Var x)) pairs
             | None => []
             end
           | None =>
             match u with
             | Not_unboxed slot => [(slot, rewrite_simple sol_u en arg)]
             | Unboxed _ => []
             end
           end
         | None => []
         end
       | _ => []
       end)
    (cr_fields repr).

Definition rename_function_decls (repr : closure_repr)
    (function_decls : list (function_slot * code_id))
    : list (function_slot * code_id) :=
  map (fun '(fs, id) => (rename_function_slot repr fs, id)) function_decls.

(* [rewrite_set_of_closures] of the set bound to [x]: with the new
   representation of [x] if it changes, as FlambdaReaperRebuild.rebuild_set
   otherwise. *)
Definition rebuild_set_u (en : rebuild_env) (x : variable) (d : named)
    : named :=
  match changed_of_node (Node_variable x), d with
  | Some repr, Closure_of_set slot function_decls value_slots =>
    Closure_of_set (rename_function_slot repr slot)
      (rename_function_decls repr function_decls)
      (changed_value_slots en repr value_slots)
  | Some repr, Set_of_closures id value_slots =>
    Set_of_closures id (changed_value_slots en repr value_slots)
  | _, _ => rebuild_set sol_u en (set_closure_nodes F (Node_variable x)) d
  end.

(* The projection of a closure of a set from the first ([let_closures]),
   at the new function slots if the representation of the set changes:
   the compiler binds the closures of the set together. *)
Definition rename_projection (d : named) : named :=
  match d with
  | Project_function_slot move_from move_to closure =>
    match changed_of closure with
    | Some repr =>
      Project_function_slot (rename_function_slot repr move_from)
        (rename_function_slot repr move_to) closure
    | None => d
    end
  | _ => d
  end.

Definition set_value_slots (d : named) : list (value_slot * simple) :=
  match d with
  | Set_of_closures _ value_slots | Closure_of_set _ _ value_slots =>
    value_slots
  | _ => []
  end.

(* The fields of a closure of a set that is unboxed, each bound to the
   argument of its value slot, or to the variables of that argument if it
   is unboxed; a field that is not a value slot of the set is [None]. *)
Definition bind_closure_fields (value_slots : list (value_slot * simple))
    (to_bind : unboxed_fields) (body : expr) : option expr :=
  fold_left
    (fun acc '(f, u) =>
       match acc, f with
       | Some body, Field_value_slot w =>
         match lookup w value_slots with
         | Some arg =>
           match unboxed_of arg with
           | Some arg_fields => bind_fields u (Unboxed arg_fields) body
           | None => bind_field_to_simple u Value arg body
           end
         | None => None
         end
       | _, _ => None
       end)
    to_bind (Some body).

(* [rebuild_set_of_closures_binding_which_is_being_unboxed]: each closure
   of the set that has a use, in the order of [bound], binds the
   variables of its fields; the first is the innermost. A closure with a
   use that is not unboxed (an assertion of the OCaml) makes it
   [Invalid]. *)
Definition rebuild_unboxed_set (bound : list variable)
    (value_slots : list (value_slot * simple)) (body : expr) : expr :=
  match
    fold_left
      (fun acc x =>
         match acc with
         | None => None
         | Some body =>
           if is_var_used sol x Value
           then
             match unboxed_of_node (Node_variable x) with
             | Some to_bind => bind_closure_fields value_slots to_bind body
             | None => None
             end
           else Some body
         end)
      bound (Some body)
  with
  | Some e => e
  | None => Invalid
  end.

(* [rebuild_let_expr_static_consts] of a closure of a static set: the
   representation of its set, if it changes, which keeps it. *)
Definition static_set_repr (s : symbol) (const : static_const)
    : option closure_repr :=
  match const with
  | Static_const_closure_of_set _ closure_symbols _ _ =>
    fold_right
      (fun '(_, sym) acc =>
         match changed_of_node (Node_symbol sym) with
         | Some repr => Some repr
         | None => acc
         end)
      None closure_symbols
  | Static_const_set_of_closures _ _ => changed_of_node (Node_symbol s)
  | _ => None
  end.

(* [rewrite_set_of_closures] of a static set whose representation
   changes, and the renaming of the function slots of its pattern. *)
Definition rebuild_changed_static_const (en : rebuild_env)
    (repr : closure_repr) (const : static_const) : static_const :=
  match const with
  | Static_const_closure_of_set slot closure_symbols function_decls
      value_slots =>
    Static_const_closure_of_set (rename_function_slot repr slot)
      (map (fun '(fs, sym) => (rename_function_slot repr fs, sym))
         closure_symbols)
      (rename_function_decls repr function_decls)
      (changed_value_slots en repr value_slots)
  | Static_const_set_of_closures id value_slots =>
    Static_const_set_of_closures id (changed_value_slots en repr value_slots)
  | _ => const
  end.

(* [should_keep_param] of [rebuild]: an unboxed parameter is [Unbox] of
   its fields; any other is kept as FlambdaReaperRebuild.params_to_keep
   says, and in the mode of the fix, a parameter kept with no use (the
   first of an exception handler) is marked [Keep_unused]. *)
Definition handler_decisions (is_exn_handler : bool)
    (h : continuation_handler) : list decision :=
  match h with
  | Handler params metadata _ =>
    let bound := with_kinds params (hm_params_kinds metadata) in
    map
      (fun (p : (variable * flambda_kind) * bool) =>
         let '((x, kind), keep) := p in
         match unboxed_of_node (Node_variable x) with
         | Some fields => Unbox fields
         | None =>
           if fixed && keep && negb (is_var_used sol x kind)
           then Keep_unused kind
           else Decided (if keep then Keep kind else Delete)
         end)
      (combine bound (params_to_keep sol is_exn_handler bound))
  end.

(* [should_keep_param] of [rebuild] for the continuations of fixed arity
   that the model gives decisions, the return and exception
   continuations of code, in the mode of the fix: every parameter is kept,
   marked [Keep_unused] if it has no use, and an unboxed one is
   [Unbox]. *)
Definition fixed_arity_decisions (nodes : list node)
    (kinds : list flambda_kind) : list decision :=
  map
    (fun '(n, kind) =>
       match unboxed_of_node n with
       | Some fields => Unbox fields
       | None =>
         if is_node_used sol n kind then Decided (Keep kind)
         else Keep_unused kind
       end)
    (combine nodes kinds).

(* The arguments given to an unboxed parameter: the variables of the
   argument for the fields of the parameter. *)
Definition unboxed_args (fields : unboxed_fields) (arg : simple)
    : option (list (simple * flambda_kind)) :=
  match unboxed_of arg with
  | Some arg_fields =>
    match unboxed_field_pairs Value (Unboxed fields) (Unboxed arg_fields) with
    | Some pairs => Some (map (fun '(_, x, kind) => (Var x, kind)) pairs)
    | None => None
    end
  | None => None
  end.

(* [get_args]: the arguments of an [Apply_cont] for the parameters kept,
   and for those unboxed the variables of the argument; the arguments
   beyond the decisions are kept, as by FlambdaReaperRebuild.filter_kept.
   [None] where an argument given to an unboxed parameter is not unboxed
   alike (a fatal error in the OCaml). *)
Fixpoint get_args (en : rebuild_env) (decisions : list decision)
    (args : list simple) : option (list simple) :=
  match args, decisions with
  | [], _ => Some []
  | arg :: args, [] =>
    option_map (cons (rewrite_simple sol_u en arg)) (get_args en [] args)
  | arg :: args, d :: decisions =>
    match get_args en decisions args with
    | None => None
    | Some rest =>
      match d with
      | Decided (Keep _) | Keep_unused _ =>
        Some (rewrite_simple sol_u en arg :: rest)
      | Decided Delete => Some rest
      | Unbox fields =>
        option_map (fun l => map fst l ++ rest) (unboxed_args fields arg)
      end
    end
  end.

(* [get_args_with_kinds] of one group of the arguments of a call. *)
Fixpoint get_args_with_kinds_u (en : rebuild_env) (decisions : list decision)
    (args : list simple) : option (list (simple * flambda_kind)) :=
  match args, decisions with
  | arg :: args, d :: decisions =>
    match get_args_with_kinds_u en decisions args with
    | None => None
    | Some rest =>
      match d with
      | Decided (Keep kind) | Keep_unused kind =>
        Some ((rewrite_simple sol_u en arg, kind) :: rest)
      | Decided Delete => Some rest
      | Unbox fields =>
        option_map (fun l => l ++ rest) (unboxed_args fields arg)
      end
    end
  | _, _ => Some []
  end.

(* The extra arguments of the exception continuation of a call, at the
   decisions of its handler after the first ([exn_continuation] of
   [rebuild_apply]); an argument kept has its kind, and those beyond the
   decisions are kept, as in FlambdaReaperRebuild.rebuild_exn_extra_args. *)
Fixpoint get_extra_args (en : rebuild_env) (decisions : list decision)
    (args : list (simple * flambda_kind))
    : option (list (simple * flambda_kind)) :=
  match args, decisions with
  | [], _ => Some []
  | (s, kind) :: args, [] =>
    option_map (cons (rewrite_simple sol_u en s, kind))
      (get_extra_args en [] args)
  | (s, kind) :: args, d :: decisions =>
    match get_extra_args en decisions args with
    | None => None
    | Some rest =>
      match d with
      | Decided (Keep _) | Keep_unused _ =>
        Some ((rewrite_simple sol_u en s, kind) :: rest)
      | Decided Delete => Some rest
      | Unbox fields => option_map (fun l => l ++ rest) (unboxed_args fields s)
      end
    end
  end.

Definition rebuild_exn_extra_args_u (en : unbox_env) (ap : apply_expr)
    : option (list (simple * flambda_kind)) :=
  get_extra_args (ue_env en)
    (match lookup (ap_exn_continuation ap) (ue_decisions en) with
     | Some decisions => tl decisions
     | None => []
     end)
    (ap_exn_extra_args ap).

(* [rewrite_apply_cont_expr]: [None] if an argument has no source, or
   where [get_args] fails. *)
Definition rewrite_apply_cont_u (en : unbox_env) (ac : apply_cont)
    : option apply_cont :=
  if existsb (simple_has_no_source sol) (ac_args ac)
  then None
  else
    match
      match lookup (ac_continuation ac) (ue_decisions en) with
      | Some decisions => get_args (ue_env en) decisions (ac_args ac)
      | None => Some (map (rewrite_simple sol_u (ue_env en)) (ac_args ac))
      end
    with
    | Some args =>
      Some
        {| ac_continuation := ac_continuation ac;
           ac_args := args;
           ac_trap_action := ac_trap_action ac |}
    | None => None
    end.

Definition rebuild_apply_cont_u (en : unbox_env) (ac : apply_cont) : expr :=
  match rewrite_apply_cont_u en ac with
  | Some ac => Apply_cont ac
  | None => Invalid
  end.

Definition rebuild_switch_u (en : unbox_env) (scrutinee : simple)
    (arms : list (Z * apply_cont)) : expr :=
  let arms :=
    flat_map
      (fun '(d, ac) =>
         match rewrite_apply_cont_u en ac with
         | Some ac => [(d, ac)]
         | None => []
         end)
      (Keyed.compact Z.eqb arms)
  in
  if Keyed.is_empty arms
  then Invalid
  else Switch (rewrite_simple sol_u (ue_env en) scrutinee) arms.

(* The decisions of [compute_code_changes] for the nodes of a code dep and
   their kinds: [Unbox] for a node with unboxed fields, and those of
   FlambdaReaperCallingConventions.decisions_of for the others. *)
Definition decisions_u (nodes : list node) (kinds : list flambda_kind)
    : list decision :=
  map
    (fun '(n, kind) =>
       match unboxed_of_node n with
       | Some fields => Unbox fields
       | None => Decided (if is_node_used sol n kind then Keep kind else Delete)
       end)
    (combine nodes kinds).

(* [compute_code_changes], of the code [id] with [metadata]. *)
Definition code_change_u (id : code_id) (metadata : code_metadata)
    : calling_convention_change_u :=
  if cannot_change_calling_convention cannot_change unit_code_ids id
  then Not_changing_u
  else
    match lookup id code_deps with
    | Some cd =>
      Changing_u
        (decisions_u (cd_params cd) (unarize (cm_params_arity metadata)))
        (decisions_u (cd_return cd) (unarize (cm_result_arity metadata)))
    | None => Not_changing_u
    end.

(* The change of calling convention of a call: that of the code it is
   known to call. *)
Definition apply_change_u (ap : apply_expr) : calling_convention_change_u :=
  match code_id_actually_called F sol ap with
  | Some id =>
    match lookup id codes with
    | Some metadata => code_change_u id metadata
    | None => Not_changing_u
    end
  | None => Not_changing_u
  end.

(* The decision of [my_closure] of [compute_code_changes], for code that
   changes its calling convention: [Some] of the unboxed fields of the
   [my_closure] of the code dep ([Unbox_my_closure]), or [None]
   ([Keep_my_closure]). *)
Definition my_closure_decision_u (id : code_id) : option unboxed_fields :=
  if cannot_change_calling_convention cannot_change unit_code_ids id
  then None
  else
    match lookup id code_deps with
    | Some cd => unboxed_of_node (cd_my_closure cd)
    | None => None
    end.

(* The decision of [my_closure] of the code that a call is known to
   call. *)
Definition apply_my_closure_u (ap : apply_expr) : option unboxed_fields :=
  match code_id_actually_called F sol ap with
  | Some id =>
    match lookup id codes with
    | Some _ => my_closure_decision_u id
    | None => None
    end
  | None => None
  end.

(* [never_returns] of [compute_code_changes], in the fix: some result of
   the code dep of [id] has no source. *)
Definition code_never_returns (id : code_id) : bool :=
  match lookup id code_deps with
  | Some cd => existsb (fun r => negb (s_has_source sol r)) (cd_return cd)
  | None => false
  end.

(* [get_returns_with_source] of points_to_analysis.ml: a node of [ws]
   holds at the result of rank [i] a node with a source. *)
Definition result_has_source_of (ws : node -> bool) (i : nat) : bool :=
  existsb
    (fun '(x, f, y) =>
       ws x && field_eqb f (Field_return_of_call (Normal i))
       && s_has_source sol y)
    (f_constructor F).

(* The [never_returns] of [arguments_used_by_call], in the fix, for a
   call of [nres] results with groups of arguments of the sizes
   [num_args]: some result of rank below [nres], read off the witness of
   the last group, has no source; [false] where a callee or a witness
   has any source. *)
Fixpoint never_returns_of_call (ep : closure_entry_point)
    (callee_sources : allocation_sources) (num_args : list nat)
    (nres : nat) : bool :=
  match num_args with
  | [] => false
  | _ :: num_args_rest =>
    match callee_sources with
    | Any_source => false
    | Sources callee_sources =>
      match get_field_sources F sol callee_sources (Field_call_witness ep)
      with
      | Any_source => false
      | Sources ws =>
        match num_args_rest with
        | [] => existsb (fun i => negb (result_has_source_of ws i)) (seq 0 nres)
        | _ :: _ =>
          never_returns_of_call ep
            (get_field_sources F sol ws (Field_return_of_call (Normal 0)))
            num_args_rest nres
        end
      end
    end
  end.

(* The [never_returns] of a call ([decide_whether_apply_needs_calling_
   convention_change] in the fix): that of the code it is known to call,
   if of the unit; otherwise that read off its callee, at the field of
   its arity; [false] for a constant callee or code of no unit. *)
Definition apply_never_returns (ap : apply_expr) : bool :=
  match code_id_actually_called F sol ap with
  | Some id =>
    match lookup id codes with
    | Some _ => code_never_returns id
    | None => false
    end
  | None =>
    match ap_callee ap with
    | Const _ => false
    | Var _ | Symbol _ =>
      let callee := get_direct_sources sol (node_of_simple (ap_callee ap)) in
      let nres := length (unarize (ap_return_arity ap)) in
      match ap_call_kind ap with
      | Direct _ | Indirect_known_arity _ =>
        never_returns_of_call Known_arity_code_pointer callee
          [length (ap_args ap)] nres
      | Indirect_unknown_arity =>
        never_returns_of_call Unknown_arity_code_pointer callee
          (map (fun c => length (unarize_component c)) (ap_args_arity ap))
          nres
      end
    end
  end.

(* The [used_params] of a call of code that keeps its calling convention,
   in the fix: for code of the unit known to be called, the parameters of
   its code dep with a use ([compute_code_changes]); for code of no unit,
   every argument; [None] where the code is not known, whose mask is
   read off the callee ([rebuild_arguments]). *)
Definition kept_code_mask (ap : apply_expr) : option (list bool) :=
  match code_id_actually_called F sol ap with
  | Some id =>
    match lookup id codes, lookup id code_deps with
    | Some _, Some cd => Some (map (s_has_usage sol) (cd_params cd))
    | _, _ => Some (repeat true (length (ap_args ap)))
    end
  | None => None
  end.

(* The arguments of a call of code that keeps its calling convention: in
   the mode of the fix, at [kept_code_mask] if the code is known. *)
Definition kept_arguments_u (en : unbox_env) (ap : apply_expr)
    : list simple :=
  match (if fixed then kept_code_mask ap else None) with
  | Some mask => mask_arguments (ue_env en) mask (ap_args ap)
  | None => rebuild_arguments F sol_u (ue_env en) ap
  end.

(* [make_apply_wrapper], given the decisions of the callee for its
   results, each with the kind of the result in the return arity of the
   call: no wrapper if what the continuation of the call is given is what
   the callee returns, in order; otherwise the call returns to a wrapper,
   the continuation [first_fresh], whose parameters are what the callee
   returns. Where the two decisions disagree on unboxing a result that
   the continuation does not delete, the compiler makes a call that never
   returns, which the conversion of the test tool writes as a call
   returning to a continuation whose handler is [Invalid]: the wrapper,
   with that handler. In the mode of the fix, so is a call that
   [never_returns], whatever the decisions. *)
Definition make_apply_wrapper_u (en : unbox_env) (ap : apply_expr)
    (func_decisions : list (decision * flambda_kind)) (never_returns : bool)
    : expr :=
  let func := freshen_decisions func_decisions first_fresh in
  let params := wrapper_params func in
  let wrapper handler :=
    Let_cont_non_recursive first_fresh
      (Handler (map fst params) (make_handler_metadata (map snd params))
         handler)
      (Apply (with_continuation ap first_fresh))
  in
  let apply_decisions :=
    match lookup (ap_continuation ap) (ue_decisions en) with
    | Some decisions => decisions
    | None => []
    end
  in
  if fixed && never_returns
  then wrapper Invalid
  else
    match wrapper_args_u fixed apply_decisions func with
    | Some args =>
      if args_are_params args (map fst params)
      then Apply ap
      else
        wrapper
          (Apply_cont
             {| ac_continuation := ap_continuation ap; ac_args := args;
                ac_trap_action := None |})
    | None => wrapper Invalid
    end.

(* [rebuild_apply]. A call whose callee does not change its calling
   convention is that of FlambdaReaperRebuild.rebuild_apply, but for an
   unboxed callee, which makes it [Invalid] (in the fix too), and, in the
   mode of the fix, for the arguments
   ([kept_arguments_u]); one whose callee changes gives the parameters
   kept and the fields of those unboxed, a product for each group, and
   expects the results kept and the fields of those unboxed. *)
Definition rebuild_apply_u (en : unbox_env) (ap : apply_expr) : expr :=
  if simple_is_dead sol (ue_env en) (ap_callee ap)
     || existsb (simple_is_dead sol (ue_env en)) (ap_args ap)
  then Invalid
  else
    match rebuild_exn_extra_args_u en ap with
    | None => Invalid
    | Some exn_extra_args =>
      match apply_change_u ap with
      | Not_changing_u =>
        match unboxed_of (ap_callee ap) with
        | Some _ => Invalid
        | None =>
          make_apply_wrapper_u en
            {| ap_callee := ap_callee ap;
               ap_call_kind :=
                 rewrite_call_kind F sol preserve_direct_calls unit_code_ids
                   ap;
               ap_args := kept_arguments_u en ap;
               ap_continuation := ap_continuation ap;
               ap_exn_continuation := ap_exn_continuation ap;
               ap_exn_extra_args := exn_extra_args;
               ap_args_arity := ap_args_arity ap;
               ap_return_arity := ap_return_arity ap |}
            (map (fun kind => (Decided (Keep kind), kind))
               (unarize (ap_return_arity ap)))
            (apply_never_returns ap)
        end
      | Changing_u params_decisions return_decisions =>
        (* [Unbox_my_closure]: the fields of the callee are given at the
           front of the first group. A callee with unboxed fields is
           dropped, with [Unbox_my_closure] or [Keep_my_closure]
           ([rewrite_simple_opt]): the call has no callee, which is
           poison here (an unboxed name, [rewrite_simple]). *)
        let callee_fields := apply_my_closure_u ap in
        match
          match callee_fields with
          | Some fields => unboxed_args fields (ap_callee ap)
          | None => Some []
          end,
          all_some
            (map
               (fun '(args, decisions) =>
                  get_args_with_kinds_u (ue_env en) decisions args)
               (combine (group_by_parameter (ap_args_arity ap) (ap_args ap))
                  (group_by_parameter (ap_args_arity ap) params_decisions)))
        with
        | None, _ | _, None => Invalid
        | Some from_callee, Some groups =>
          let groups := prepend_first from_callee groups in
          make_apply_wrapper_u en
            {| ap_callee :=
                 match unboxed_of (ap_callee ap) with
                 | Some _ => rewrite_simple sol_u (ue_env en) (ap_callee ap)
                 | None => ap_callee ap
                 end;
               ap_call_kind :=
                 rewrite_call_kind F sol preserve_direct_calls unit_code_ids
                   ap;
               ap_args := flat_map (map fst) groups;
               ap_continuation := ap_continuation ap;
               ap_exn_continuation := ap_exn_continuation ap;
               ap_exn_extra_args := exn_extra_args;
               ap_args_arity := arity_of_groups groups;
               ap_return_arity := arity_of_decisions_u return_decisions |}
            (combine return_decisions (unarize (ap_return_arity ap)))
            (apply_never_returns ap)
        end
      end
    end.

(* The decisions of the return and exception continuations of the code
   [id], with [metadata] and the change of calling convention [change],
   in the mode of the fix: for its results those of [change] if the code
   changes its calling convention, and those of a continuation of fixed
   arity otherwise; for its exception, those of a continuation of fixed
   arity. Without a code dep (a fatal error in the OCaml), the decisions
   of [change] alone. *)
Definition code_continuation_decisions (id : code_id)
    (metadata : code_metadata) (change : calling_convention_change_u)
    (return_continuation exn_continuation : continuation)
    : list (continuation * list decision) :=
  let return_decisions :=
    match change with
    | Not_changing_u => []
    | Changing_u _ return_decisions => [(return_continuation, return_decisions)]
    end
  in
  match lookup id code_deps with
  | Some cd =>
    match change with
    | Not_changing_u =>
      [(return_continuation,
        fixed_arity_decisions (cd_return cd)
          (unarize (cm_result_arity metadata)))]
    | Changing_u _ _ => return_decisions
    end
    ++ [(exn_continuation, fixed_arity_decisions [cd_exn cd] [Value])]
  | None => return_decisions
  end.

(* The decision of a parameter of code that changes its calling
   convention, at the parameter ([rebuild_function_params_and_body]): an
   unboxed one has the fields of the parameter itself, which have the
   shape of those of the code dep. *)
Definition param_decision_at (x : variable) (d : decision) : decision :=
  match d with
  | Unbox fields =>
    match unboxed_of_node (Node_variable x) with
    | Some fields' => Unbox fields'
    | None => Unbox fields
    end
  | Decided _ | Keep_unused _ => d
  end.

(* [rebuild_expr] of FlambdaReaperCallingConventions.v, with unboxing. *)
Fixpoint rebuild_expr_u (en : unbox_env) (e : expr) {struct e} : expr :=
  match e with
  | Let x kind defining_expr body =>
    if is_pending_projection (ue_env en) x defining_expr
    then
      let body :=
        rebuild_expr_u
          (ubind_set en [(x, kind)] (tl (env_projections (ue_env en)))
             (ue_drop_projections en))
          body
      in
      if ue_drop_projections en
      then body
      else Let x kind (rename_projection defining_expr) body
    else
      match defining_expr with
      | Set_of_closures _ _ | Closure_of_set _ _ _ =>
        let projections := projections_of_set x defining_expr body in
        if is_dead_var sol x kind
           || existsb (fun y => is_dead_var sol y Value) projections
        then Invalid
        else if existsb (fun y => is_unboxed_name (Node_variable y))
                  (x :: projections)
        then
          rebuild_unboxed_set (x :: projections)
            (set_value_slots defining_expr)
            (rebuild_expr_u (ubind_set en [(x, kind)] projections true) body)
        else if is_var_used sol x kind
                || existsb (fun y => is_var_used sol y Value) projections
        then
          Let x kind (rebuild_set_u (ue_env en) x defining_expr)
            (rebuild_expr_u (ubind_variables_then en [(x, kind)] projections)
               body)
        else rebuild_expr_u (ubind_variables en [(x, kind)]) body
      | Simple _ | Prim _ _ | Project_value_slot _ _
      | Project_value_slot_from _ _ _ | Project_function_slot _ _ _ =>
        if is_dead_var sol x kind
        then Invalid
        else if is_var_used sol x kind
        then
          let body := rebuild_expr_u (ubind_variables en [(x, kind)]) body in
          match unboxed_of_node (Node_variable x) with
          | Some to_bind =>
            rebuild_unboxed_binding (ue_env en) to_bind defining_expr body
          | None =>
            match changed_of_node (Node_variable x) with
            | Some repr =>
              rebuild_changed_binding (ue_env en) repr x kind defining_expr
                body
            | None => rebuild_let_u (ue_env en) x kind defining_expr body
            end
          end
        else rebuild_expr_u (ubind_variables en [(x, kind)]) body
      end
  | Let_static bindings body =>
    Let_static (flat_map (rebuild_binding_u en) bindings)
      (rebuild_expr_u (ubind_variables en []) body)
  | Let_cont_non_recursive k h body =>
    let decisions := handler_decisions (handler_is_exn_handler h) h in
    let body := rebuild_expr_u (ubind_continuations en [(k, decisions)]) body in
    if among k (free_continuations_of_expr body)
    then Let_cont_non_recursive k (rebuild_handler_u en true decisions h) body
    else body
  | Let_cont_recursive handlers body =>
    let en :=
      ubind_continuations en
        (map (fun '(k, h) => (k, handler_decisions false h)) handlers)
    in
    Let_cont_recursive
      (map
         (fun '(k, h) =>
            (k, rebuild_handler_u en false (handler_decisions false h) h))
         handlers)
      (rebuild_expr_u en body)
  | Apply ap => rebuild_apply_u en ap
  | Apply_cont ac => rebuild_apply_cont_u en ac
  | Switch scrutinee arms => rebuild_switch_u en scrutinee arms
  | Invalid => Invalid
  end
(* [rebuild_handler]: the parameters are those that the decisions make. *)
with rebuild_handler_u (en : unbox_env) (dead_check : bool)
    (decisions : list decision) (h : continuation_handler)
    : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    let bound := with_kinds params (hm_params_kinds metadata) in
    let params := params_of_decisions bound decisions in
    Handler (map fst params) (with_params_kinds metadata (map snd params))
      (if dead_check
          && existsb (fun '(x, kind) => is_dead_var sol x kind) bound
       then Invalid
       else rebuild_expr_u (ubind_variables en bound) handler)
  end
(* [rebuild_binding]. *)
with rebuild_binding_u (en : unbox_env) (binding : static_binding)
    : list static_binding :=
  match binding with
  | Bind_code id c =>
    if is_code_id_used sol id
    then [Bind_code id (rebuild_code_u id c)]
    else [Bind_deleted_code id]
  | Bind_deleted_code _ => [binding]
  | Bind_symbol s const =>
    match static_set_repr s const with
    | Some repr =>
      [Bind_symbol s (rebuild_changed_static_const (ue_env en) repr const)]
    | None =>
      if static_const_is_used sol s const
      then [Bind_symbol s (rebuild_static_const sol_u (ue_env en) s const)]
      else []
    end
  end
(* [rebuild_code] and [rebuild_function_params_and_body]: code that
   changes its calling convention has the parameters that its decisions
   make, the arities of [changed_code_metadata_u], and its return
   continuation the decisions of its results. In the mode of the fix, the
   return continuation of code that keeps its calling convention, and the
   exception continuation of all code, have the decisions of continuations
   of fixed arity, for the nodes of the code dep. *)
with rebuild_code_u (id : code_id) (c : code) : code :=
  match c with
  | Code params my_closure return_continuation exn_continuation metadata
      body =>
    let change := code_change_u id metadata in
    let bound :=
      (my_closure, Value)
      :: with_kinds params (unarize (cm_params_arity metadata))
    in
    let en :=
      if fixed
      then
        ubind_continuations empty_unbox_env
          (code_continuation_decisions id metadata change return_continuation
             exn_continuation)
      else
        match change with
        | Not_changing_u => empty_unbox_env
        | Changing_u _ return_decisions =>
          ubind_continuations empty_unbox_env
            [(return_continuation, return_decisions)]
        end
    in
    let body :=
      if existsb (fun '(x, kind) => is_dead_var sol x kind) bound
      then Invalid
      else rebuild_expr_u (ubind_variables en bound) body
    in
    match change with
    | Not_changing_u =>
      Code params my_closure return_continuation exn_continuation metadata
        body
    | Changing_u params_decisions return_decisions =>
      (* [Unbox_my_closure]: the fields of [my_closure] are parameters, in
         front of the others. *)
      let my_closure_decision :=
        match my_closure_decision_u id with
        | Some fields => param_decision_at my_closure (Unbox fields)
        | None => Decided Delete
        end
      in
      Code
        (map fst
           (decision_params my_closure my_closure_decision
            ++ flat_map
                 (fun '(x, d) => decision_params x (param_decision_at x d))
                 (combine params params_decisions)))
        my_closure return_continuation exn_continuation
        (changed_code_metadata_u metadata my_closure_decision
           params_decisions return_decisions)
        body
    end
  end.

(* The fatal error of the fix at a call ([rebuild_apply]): the arm of
   [make_apply_wrapper] ([wrapper_fails_u]) for a call that does not
   [never_returns]. A call that the rebuild makes [Invalid] before (a
   dead name, an unboxed callee of code that keeps its calling
   convention, or the fatal error of another check) does not reach
   it. *)
Definition apply_fails_u (en : unbox_env) (ap : apply_expr) : bool :=
  if simple_is_dead sol (ue_env en) (ap_callee ap)
     || existsb (simple_is_dead sol (ue_env en)) (ap_args ap)
  then false
  else
    match rebuild_exn_extra_args_u en ap with
    | None => false
    | Some _ =>
      let apply_decisions :=
        match lookup (ap_continuation ap) (ue_decisions en) with
        | Some decisions => decisions
        | None => []
        end
      in
      match apply_change_u ap with
      | Not_changing_u =>
        match unboxed_of (ap_callee ap) with
        | Some _ => false
        | None =>
          negb (apply_never_returns ap)
          && wrapper_fails_u apply_decisions
               (map (fun kind => Decided (Keep kind))
                  (unarize (ap_return_arity ap)))
        end
      | Changing_u params_decisions return_decisions =>
        match
          match apply_my_closure_u ap with
          | Some fields => unboxed_args fields (ap_callee ap)
          | None => Some []
          end,
          all_some
            (map
               (fun '(args, decisions) =>
                  get_args_with_kinds_u (ue_env en) decisions args)
               (combine (group_by_parameter (ap_args_arity ap) (ap_args ap))
                  (group_by_parameter (ap_args_arity ap) params_decisions)))
        with
        | Some _, Some _ =>
          negb (apply_never_returns ap)
          && wrapper_fails_u apply_decisions return_decisions
        | _, _ => false
        end
      end
    end.

(* Whether the rebuild of the fix stops with a fatal error, walking the
   unit as the compiler rebuilds it: under the decisions of the
   continuations in scope, as [rebuild_expr_u] binds them, the calls
   ([apply_fails_u]), the body of a [Let] whatever its binding (the
   compiler rebuilds the body first), a handler where the rebuilt body
   applies it and its parameters are not dead, and every code of the unit,
   used or not (the compiler rebuilds all code), whose body is rebuilt
   where its parameters are not dead. Code that [never_returns] stops
   the compiler where its rebuilt body has its return continuation free
   ([check_never_returns]). *)
Fixpoint rebuild_fails_u (en : unbox_env) (e : expr) {struct e} : bool :=
  match e with
  | Let x kind defining_expr body =>
    let en' :=
      if is_pending_projection (ue_env en) x defining_expr
      then
        ubind_set en [(x, kind)] (tl (env_projections (ue_env en)))
          (ue_drop_projections en)
      else
        match defining_expr with
        | Set_of_closures _ _ | Closure_of_set _ _ _ =>
          let projections := projections_of_set x defining_expr body in
          if existsb (fun y => is_unboxed_name (Node_variable y))
               (x :: projections)
          then ubind_set en [(x, kind)] projections true
          else if is_var_used sol x kind
                  || existsb (fun y => is_var_used sol y Value) projections
          then ubind_variables_then en [(x, kind)] projections
          else ubind_variables en [(x, kind)]
        | _ => ubind_variables en [(x, kind)]
        end
    in
    rebuild_fails_u en' body
  | Let_static bindings body =>
    existsb binding_fails_u bindings
    || rebuild_fails_u (ubind_variables en []) body
  | Let_cont_non_recursive k h body =>
    let decisions := handler_decisions (handler_is_exn_handler h) h in
    let en' := ubind_continuations en [(k, decisions)] in
    rebuild_fails_u en' body
    || among k (free_continuations_of_expr (rebuild_expr_u en' body))
       && handler_fails_u en true h
  | Let_cont_recursive handlers body =>
    let en :=
      ubind_continuations en
        (map (fun '(k, h) => (k, handler_decisions false h)) handlers)
    in
    existsb (fun '(_, h) => handler_fails_u en false h) handlers
    || rebuild_fails_u en body
  | Apply ap => apply_fails_u en ap
  | Apply_cont _ | Switch _ _ | Invalid => false
  end
with handler_fails_u (en : unbox_env) (dead_check : bool)
    (h : continuation_handler) : bool :=
  match h with
  | Handler params metadata handler =>
    let bound := with_kinds params (hm_params_kinds metadata) in
    negb (dead_check
          && existsb (fun '(x, kind) => is_dead_var sol x kind) bound)
    && rebuild_fails_u (ubind_variables en bound) handler
  end
with binding_fails_u (binding : static_binding) : bool :=
  match binding with
  | Bind_code id c => code_fails_u id c
  | Bind_deleted_code _ | Bind_symbol _ _ => false
  end
with code_fails_u (id : code_id) (c : code) : bool :=
  match c with
  | Code params my_closure return_continuation exn_continuation metadata
      body =>
    let bound :=
      (my_closure, Value)
      :: with_kinds params (unarize (cm_params_arity metadata))
    in
    let en :=
      ubind_continuations empty_unbox_env
        (code_continuation_decisions id metadata (code_change_u id metadata)
           return_continuation exn_continuation)
    in
    negb (existsb (fun '(x, kind) => is_dead_var sol x kind) bound)
    && rebuild_fails_u (ubind_variables en bound) body
    || code_never_returns id
       && among return_continuation
            (free_continuations_of_expr (code_body (rebuild_code_u id c)))
  end.

End Rebuild.

(* [Rebuild.rebuild] with unboxing, changes of calling convention and
   changes of representation: the rebuilt unit, given the facts of its
   graph, the solution, the tables of [cannot_change_calling_convention],
   [dominated_by_allocation_point], [to_unbox] and
   [to_change_representation] ([tcr]), and the code deps of the
   traversal. The variables of the unboxed fields are the names after
   every name of the unit, and the wrappers the names after them; the new
   slots are the names after every slot of the unit. [fixed] is the mode
   of [rebuild_expr_u]. *)
Definition rebuild_unit_unbox_repr_mode (fixed : bool)
    (is_local : field -> bool) (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (preserve_direct_calls : bool) (e : expr) : expr :=
  let '(unboxed, next) := unit_unboxed_fields F T dominated to_unbox e in
  rebuild_expr_u fixed F (solution_of_tables T)
    (fun n => mem_node n cannot_change) code_deps preserve_direct_calls
    (static_names code_ids_bound e) (bound_code_metadata e) unboxed next
    (unit_changed_representations is_local F T dominated to_unbox tcr e)
    empty_unbox_env e.

(* The rebuild of the compiler on the branch, and that of the fix. *)
Definition rebuild_unit_unbox_repr := rebuild_unit_unbox_repr_mode false.

Definition rebuild_unit_unbox_repr_fixed :=
  rebuild_unit_unbox_repr_mode true.

(* Whether the rebuild of the fix stops with a fatal error, with the
   arguments of [rebuild_unit_unbox_repr_fixed]. *)
Definition rebuild_unit_unbox_repr_fixed_fails (is_local : field -> bool)
    (F : facts) (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox tcr : list node) (preserve_direct_calls : bool) (e : expr)
    : bool :=
  let '(unboxed, next) := unit_unboxed_fields F T dominated to_unbox e in
  rebuild_fails_u true F (solution_of_tables T)
    (fun n => mem_node n cannot_change) code_deps preserve_direct_calls
    (static_names code_ids_bound e) (bound_code_metadata e) unboxed next
    (unit_changed_representations is_local F T dominated to_unbox tcr e)
    empty_unbox_env e.

(* The rebuild with no change of representation: a set of closures that
   is not unboxed keeps its slots. *)
Definition rebuild_unit_unbox (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) (e : expr) : expr :=
  rebuild_unit_unbox_repr (fun _ => false) F T cannot_change code_deps
    dominated to_unbox [] preserve_direct_calls e.

(* The change of calling convention of each code of the unit, with
   unboxing, for the test tool to compare with the compiler's
   ([Analysis.get_calling_convention_change]). *)
Definition unit_calling_convention_changes_u (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox : list node) (e : expr)
    : list (code_id * calling_convention_change_u) :=
  let '(unboxed, _) := unit_unboxed_fields F T dominated to_unbox e in
  map
    (fun '(id, metadata) =>
       (id,
        code_change_u (solution_of_tables T)
          (fun n => mem_node n cannot_change) code_deps
          (static_names code_ids_bound e) unboxed id metadata))
    (bound_code_metadata e).

(* The decision of [my_closure] of each code of the unit
   ([my_closure_decision_u]), for the test tool to compare with the
   compiler's: [Some] of the unboxed fields for [Unbox_my_closure]. *)
Definition unit_my_closure_decisions_u (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox : list node) (e : expr)
    : list (code_id * option unboxed_fields) :=
  let '(unboxed, _) := unit_unboxed_fields F T dominated to_unbox e in
  map
    (fun '(id, _) =>
       (id,
        my_closure_decision_u (fun n => mem_node n cannot_change) code_deps
          (static_names code_ids_bound e) unboxed id))
    (bound_code_metadata e).

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

Module Examples.

Local Open Scope positive_scope.

(* The units return to 1 and raise to 2. *)
Definition interface : unit_interface :=
  {| unit_return := 1; unit_exn := 2 |}.

Definition return_to (k : continuation) (args : list simple) : expr :=
  Apply_cont {| ac_continuation := k; ac_args := args;
                ac_trap_action := None |}.

Definition make_block (fields : list simple) : named :=
  Prim (Variadic (Make_block (Block_kind_values 0))) fields.

Definition load (field : Z) (block : simple) : named :=
  Prim (Unary (Block_load (Block_access_values None None Any_value)
                 Immutable field)) [block].

Definition int (i : Z) : simple := Const (Tagged_immediate i).

(* [let x10 = [7; 8] in let x11 = x10.(0) in return x11]: the block is
   allocated and only read. *)
Definition read_block : expr :=
  Let 10 Value (make_block [int 7; int 8])
    (Let 11 Value (load 0 (Var 10)) (return_to 1 [Var 11])).

(* The block is given to the continuation 20, whose handler reads it. *)
Definition block_to_continuation : expr :=
  Let_cont_non_recursive 20
    (Handler [30] (make_handler_metadata [Value])
       (Let 31 Value (load 0 (Var 30)) (return_to 1 [Var 31])))
    (Let 10 Value (make_block [int 7; int 8]) (return_to 20 [Var 10])).

(* The code 50 returns a block of its parameter [x40] and 8; the unit
   calls it directly through the closure [x60] and reads field 0 of what
   it returns. *)
Definition code_50 : code :=
  Code [40] 41 3 4
    (make_code_metadata [Singleton Value] [Singleton Value])
    (Let 42 Value (make_block [Var 40; int 8]) (return_to 3 [Var 42])).

Definition call_50 (args : list simple) (k : continuation) : apply_expr :=
  {| ap_callee := Var 60; ap_call_kind := Direct 50; ap_args := args;
     ap_continuation := k; ap_exn_continuation := 2;
     ap_exn_extra_args := [];
     ap_args_arity := [Singleton Value];
     ap_return_arity := [Singleton Value] |}.

Definition block_from_code : expr :=
  Let_static [Bind_code 50 code_50]
    (Let 60 Value (Closure_of_set 70 [(70, 50)] [])
       (Let_cont_non_recursive 20
          (Handler [30] (make_handler_metadata [Value])
             (Let 31 Value (load 0 (Var 30)) (return_to 1 [Var 31])))
          (Apply (call_50 [int 9] 20)))).

(* A block [x10] in a block [x11]: the unit reads field 0 of [x11], and
   field 1 of that. *)
Definition nested_block : expr :=
  Let 10 Value (make_block [int 7; int 8])
    (Let 11 Value (make_block [Var 10])
       (Let 12 Value (load 0 (Var 11))
          (Let 13 Value (load 1 (Var 12)) (return_to 1 [Var 13])))).

(* The case of [make_apply_wrapper] where the continuation of a call keeps
   a result that the callee unboxes. The code 50 returns a block of its
   parameter and 8, and the code 51 its parameter. The unit calls 50
   twice: once returning to 20, which reads field 0 of the block, so that
   the result of 50 is unboxed; and once returning to 21, whose parameter
   [x32] is also given the closure [x61] of 51. [x32] has two sources and
   is only called, by the arm 1 of the [Switch] of 21; it is kept. The
   arms 0, taken at run time, make the call returning to 21, then return
   10. *)
Definition code_51 : code :=
  Code [43] 44 5 6 (make_code_metadata [Singleton Value] [Singleton Value])
    (return_to 5 [Var 43]).

Definition goto (k : continuation) : apply_cont :=
  {| ac_continuation := k; ac_args := []; ac_trap_action := None |}.

Definition no_params : handler_metadata := make_handler_metadata [].

Definition call_32 : apply_expr :=
  {| ap_callee := Var 32; ap_call_kind := Indirect_unknown_arity;
     ap_args := [int 3];
     ap_continuation := 1; ap_exn_continuation := 2;
     ap_exn_extra_args := [];
     ap_args_arity := [Singleton Value];
     ap_return_arity := [Singleton Value] |}.

Definition kept_result_unboxed : expr :=
  Let_static [Bind_code 50 code_50; Bind_code 51 code_51]
    (Let 60 Value (Closure_of_set 70 [(70, 50)] [])
       (Let 61 Value (Closure_of_set 71 [(71, 51)] [])
          (Let_cont_non_recursive 20
             (Handler [30] (make_handler_metadata [Value])
                (Let 31 Value (load 0 (Var 30)) (return_to 1 [Var 31])))
             (Let_cont_non_recursive 21
                (Handler [32] (make_handler_metadata [Value])
                   (Let_cont_non_recursive 26
                      (Handler [] no_params (return_to 1 [int 10]))
                      (Let_cont_non_recursive 27
                         (Handler [] no_params (Apply call_32))
                         (Switch (Const (Naked_immediate 0))
                            [(0%Z, goto 26); (1%Z, goto 27)]))))
                (Let_cont_non_recursive 23
                   (Handler [] no_params (Apply (call_50 [int 9] 21)))
                   (Let_cont_non_recursive 24
                      (Handler [] no_params (return_to 21 [Var 61]))
                      (Let_cont_non_recursive 25
                         (Handler [] no_params (Apply (call_50 [int 9] 20)))
                         (Switch (Const (Naked_immediate 0))
                            [(0%Z, goto 23); (1%Z, goto 24);
                             (2%Z, goto 25)])))))))).

Definition read_block_facts : facts := traverse_unit Auto interface read_block.

Definition block_to_continuation_facts : facts :=
  traverse_unit Auto interface block_to_continuation.

Definition block_from_code_facts : facts :=
  traverse_unit Auto interface block_from_code.

Definition nested_block_facts : facts :=
  traverse_unit Auto interface nested_block.

Definition kept_result_unboxed_facts : facts :=
  traverse_unit Auto interface kept_result_unboxed.

(* Short names for the nodes and fields of the tables below. *)
Definition v (x : positive) : node := Node_variable x.
Definition m (n : positive) : node := Node_made n.
Definition c (id : positive) : node := Node_code_id id.
Definition consts : node := Node_all_constants.
Definition known : field := Field_call_witness Known_arity_code_pointer.
Definition unknown : field := Field_call_witness Unknown_arity_code_pointer.
Definition result (i : nat) : field := Field_return_of_call (Normal i).

(* No field is local, the size of an unboxing is the compiler's default,
   and closures are not lambda-lifted. *)
Definition no_local (_ : field) : bool := false.
Definition max_unbox_size : nat := 10.

(* The solutions of the four units, as the solver of the test tool
   computes them. *)
Definition read_block_tables : tables :=
  {| t_alias := [(m 1, v 11); (v 11, consts)];
     t_any_usage := [m 1; m 2; v 11; consts];
     t_any_source := [consts; v 11; m 1];
     t_has_usage := [m 1; m 2; v 11; v 10; consts];
     t_has_source := [consts; v 10; v 11; m 1];
     t_usages := [(v 10, v 10)];
     t_sources := [(v 10, v 10)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used := [(v 10, Field_block 0 Value)] |}.

Definition block_to_continuation_tables : tables :=
  {| t_alias := [(v 30, v 10); (m 1, v 31); (v 31, consts)];
     t_any_usage := [m 1; m 2; v 31; consts];
     t_any_source := [consts; v 31; m 1];
     t_has_usage := [m 1; m 2; v 31; v 30; v 10; consts];
     t_has_source := [consts; v 10; v 30; v 31; m 1];
     t_usages := [(v 30, v 30); (v 10, v 30)];
     t_sources := [(v 10, v 10); (v 30, v 10)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used := [(v 10, Field_block 0 Value)] |}.

Definition block_from_code_tables : tables :=
  {| t_alias := [(m 13, m 7); (m 5, m 12); (m 9, m 11); (m 1, v 31);
       (m 3, v 42); (v 41, m 5); (v 40, m 6); (m 11, m 7); (v 30, m 3);
       (m 2, m 4); (m 10, c 50); (m 6, consts); (v 31, v 40); (m 5, v 60)];
     t_any_usage := [m 1; m 10; m 2; v 31; m 4; c 50; v 40; m 6; consts];
     t_any_source := [c 50; consts; m 10; m 6; v 40; v 31; m 1];
     t_has_usage := [m 1; m 10; m 2; v 31; m 9; v 30; m 11; m 7; m 3; m 4;
       c 50; v 40; v 42; m 6; consts; v 60];
     t_has_source := [c 50; consts; v 60; v 42; m 8; m 7; m 13; m 3; m 11;
       v 30; m 10; m 6; m 9; v 40; v 31; m 5; m 1; v 41];
     t_usages := [(v 60, v 60); (m 9, m 9); (v 30, v 30); (m 11, m 9);
       (m 7, m 9); (m 3, v 30); (v 42, v 30)];
     t_sources := [(v 60, v 60); (v 42, v 42); (m 8, m 8); (m 7, m 7);
       (m 13, m 7); (m 3, v 42); (m 11, m 7); (m 9, m 7); (v 30, v 42);
       (m 5, v 60); (v 41, v 60)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used := [(v 60, known); (v 42, Field_block 0 Value);
       (m 7, result 0); (m 7, Field_return_of_call Exn);
       (m 7, Field_code_id_of_call_witness)] |}.

Definition nested_block_tables : tables :=
  {| t_alias := [(m 1, v 13); (v 12, v 10); (v 13, consts)];
     t_any_usage := [m 1; m 2; v 13; consts];
     t_any_source := [consts; v 13; m 1];
     t_has_usage := [m 1; m 2; v 13; v 12; v 11; v 10; consts];
     t_has_source := [consts; v 11; v 10; v 12; v 13; m 1];
     t_usages := [(v 12, v 12); (v 11, v 11); (v 10, v 12)];
     t_sources := [(v 11, v 11); (v 10, v 10); (v 12, v 10)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used := [(v 11, Field_block 0 Value);
       (v 10, Field_block 1 Value)] |}.

Example solutions :
  check_closed no_local read_block_facts read_block_tables = true
  ∧ check_field_used_exact no_local read_block_facts read_block_tables = true
  ∧ check_closed no_local block_to_continuation_facts
      block_to_continuation_tables = true
  ∧ check_field_used_exact no_local block_to_continuation_facts
      block_to_continuation_tables = true
  ∧ check_closed no_local block_from_code_facts block_from_code_tables = true
  ∧ check_field_used_exact no_local block_from_code_facts
      block_from_code_tables = true
  ∧ check_closed no_local nested_block_facts nested_block_tables = true
  ∧ check_field_used_exact no_local nested_block_facts nested_block_tables
    = true.
Proof. vm_compute. auto 10. Qed.

(* A block allocated and only read. [x10] is its own allocation point and
   that of no other name, and is unboxed: it has a use (its field 0 is
   read into [x11]) and no rule forbids it. [x11] has any usage, as the
   unit returns it: the check rejects a table that unboxes it, as it
   rejects tables of [dominated_by_allocation_point] with a pair missing
   or a pair more, and a table of [to_unbox] that leaves [x10] out. With
   an unboxing size of 0, the one field used is one too many. *)
Example read_block_tables_checked :
  derive_dominated read_block_tables = [(v 10, v 10)]
  ∧ check_dominated read_block_tables [(v 10, v 10)] = true
  ∧ check_dominated read_block_tables [] = false
  ∧ check_dominated read_block_tables [(v 10, v 10); (v 10, v 11)] = false
  ∧ derive_to_unbox no_local read_block_facts read_block_tables []
      max_unbox_size false = [v 10]
  ∧ check_to_unbox no_local read_block_facts read_block_tables []
      max_unbox_size false [v 10] = true
  ∧ check_to_unbox no_local read_block_facts read_block_tables []
      max_unbox_size false [v 10; v 11] = false
  ∧ check_to_unbox no_local read_block_facts read_block_tables []
      max_unbox_size false [] = false
  ∧ check_to_unbox no_local read_block_facts read_block_tables [] 0 false
      [v 10] = false.
Proof. vm_compute. auto 10. Qed.

(* Field 0 of [x10], the one read, is given the variable 12, the name
   after every name of the unit. What rebuild.ml makes of the unit: the
   [Let] of the block binds that variable to the field
   ([rebuild_singleton_binding_which_is_being_unboxed], with
   [bind_field_to_simple], which does not rewrite the field), and the
   load of field 0 is that variable ([rewrite_field_access]). *)
Example read_block_unboxed_fields :
  unit_unboxed_fields read_block_facts read_block_tables [(v 10, v 10)]
    [v 10] read_block
  = ([(v 10, [(Field_block 0 Value, Not_unboxed 12)])], 13).
Proof. vm_compute. reflexivity. Qed.

Example rebuild_read_block :
  rebuild_unit_unbox read_block_facts read_block_tables []
    (unit_code_deps Auto interface read_block) [(v 10, v 10)] [v 10] true
    read_block =
  Let 12 Value (Simple (int 7))
    (Let 11 Value (Simple (Var 12)) (return_to 1 [Var 11])).
Proof. vm_compute. reflexivity. Qed.

(* A block given to a continuation: [x10] dominates [x10] and the
   parameter [x30], whose field 0 is read. Each of the two has a variable
   for field 0, 32 and 33. The handler of 20 has the variable of [x30]
   for parameter ([should_keep_param]: [Unbox]), and is applied to that
   of [x10] ([get_args]). *)
Example block_to_continuation_unboxed_fields :
  derive_dominated block_to_continuation_tables
  = [(v 10, v 10); (v 10, v 30)]
  ∧ derive_to_unbox no_local block_to_continuation_facts
      block_to_continuation_tables [] max_unbox_size false = [v 10]
  ∧ check_to_unbox no_local block_to_continuation_facts
      block_to_continuation_tables [] max_unbox_size false [v 10] = true
  ∧ unit_unboxed_fields block_to_continuation_facts
      block_to_continuation_tables [(v 10, v 10); (v 10, v 30)] [v 10]
      block_to_continuation
    = ([(v 10, [(Field_block 0 Value, Not_unboxed 32)]);
        (v 30, [(Field_block 0 Value, Not_unboxed 33)])], 34).
Proof. vm_compute. auto. Qed.

Example rebuild_block_to_continuation :
  rebuild_unit_unbox block_to_continuation_facts block_to_continuation_tables
    [] (unit_code_deps Auto interface block_to_continuation)
    [(v 10, v 10); (v 10, v 30)] [v 10] true block_to_continuation =
  Let_cont_non_recursive 20
    (Handler [33] (make_handler_metadata [Value])
       (Let 31 Value (Simple (Var 33)) (return_to 1 [Var 31])))
    (Let 32 Value (Simple (int 7)) (return_to 20 [Var 32])).
Proof. vm_compute. reflexivity. Qed.

(* A block returned by code whose calling convention changes. The code
   dep of 50 has the result 3, the exception 4, [my_closure] 5, the
   parameter 6 and the witnesses 7 (of known arity) and 8; the call
   makes the witness 9. No code is in the least table of
   [cannot_change_calling_convention], and the block [x42] is unboxed:
   it dominates the result 3 of the code dep and the parameter [x30] of
   the continuation of the call. If the code could not change its calling
   convention, the block could not be unboxed ([cu0_result]). *)
Definition block_from_code_dominated : list (node * node) :=
  [(v 60, v 60); (v 42, v 42); (m 8, m 8); (m 7, m 7); (m 7, m 13);
   (v 42, m 3); (m 7, m 11); (m 7, m 9); (v 42, v 30); (v 60, m 5);
   (v 60, v 41)].

Example block_from_code_tables_checked :
  derive_cannot_change block_from_code_facts block_from_code_tables = []
  ∧ check_cc_closed block_from_code_facts block_from_code_tables [] = true
  ∧ derive_dominated block_from_code_tables = block_from_code_dominated
  ∧ check_dominated block_from_code_tables block_from_code_dominated = true
  ∧ derive_to_unbox no_local block_from_code_facts block_from_code_tables []
      max_unbox_size false = [v 42]
  ∧ check_to_unbox no_local block_from_code_facts block_from_code_tables []
      max_unbox_size false [v 42] = true
  ∧ check_to_unbox no_local block_from_code_facts block_from_code_tables
      [c 50] max_unbox_size false [v 42] = false.
Proof. vm_compute. auto 10. Qed.

(* The code keeps its parameter and unboxes its result
   ([compute_code_changes]: [Unbox] of the unboxed fields of the result 3
   of the code dep, with the variable 62). *)
Example block_from_code_changes :
  unit_calling_convention_changes_u block_from_code_facts
    block_from_code_tables [] (unit_code_deps Auto interface block_from_code)
    block_from_code_dominated [v 42] block_from_code
  = [(50, Changing_u [Decided (Keep Value)]
            [Unbox [(Field_block 0 Value, Not_unboxed 62)]])].
Proof. vm_compute. reflexivity. Qed.

(* What rebuild.ml makes of the unit. In the code, the [Let] of the block
   binds the variable 61 of its field 0 to [x40], and the return gives
   that variable for the result unboxed ([get_args]); the arities are a
   product for the parameter and one for the results
   ([compute_code_changes]). The handler of 20 has the variable 63 of
   [x30] for parameter. The decisions of the call and of the callee both
   unbox the result, with the same field: the arguments the wrapper would
   give 20 are its parameters, and there is no wrapper
   ([make_apply_wrapper]). *)
Definition code_50_unboxed : code :=
  Code [40] 41 3 4
    {| cm_params_arity := [Unboxed_product [Singleton Value]];
       cm_result_arity := [Unboxed_product [Singleton Value]];
       cm_is_tupled := false;
       cm_is_stub := false |}
    (Let 61 Value (Simple (Var 40)) (return_to 3 [Var 61])).

Example rebuild_block_from_code :
  rebuild_unit_unbox block_from_code_facts block_from_code_tables []
    (unit_code_deps Auto interface block_from_code) block_from_code_dominated
    [v 42] true block_from_code =
  Let_static [Bind_code 50 code_50_unboxed]
    (Let 60 Value (Closure_of_set 70 [(70, 50)] [])
       (Let_cont_non_recursive 20
          (Handler [63] (make_handler_metadata [Value])
             (Let 31 Value (Simple (Var 63)) (return_to 1 [Var 31])))
          (Apply
             {| ap_callee := Var 60; ap_call_kind := Direct 50;
                ap_args := [int 9]; ap_continuation := 20;
                ap_exn_continuation := 2; ap_exn_extra_args := [];
                ap_args_arity := [Unboxed_product [Singleton Value]];
                ap_return_arity := [Unboxed_product [Singleton Value]] |}))).
Proof. vm_compute. reflexivity. Qed.

(* A nested block: [x10] is stored in field 0 of [x11], which is read
   into [x12], which [x10] dominates. Both are unboxed: [x11] has its
   field 0 unboxed in turn, as what it is read into is to be unboxed
   ([mk_unboxed_fields]), with the field 1 of [x10] that [x12] reads. A
   table of [to_unbox] that unboxes [x10] alone fails the check: [x11]
   could be unboxed, and would hold the fields of [x10]. *)
Example nested_block_tables_checked :
  derive_dominated nested_block_tables
  = [(v 11, v 11); (v 10, v 10); (v 10, v 12)]
  ∧ derive_to_unbox no_local nested_block_facts nested_block_tables []
      max_unbox_size false = [v 11; v 10]
  ∧ check_to_unbox no_local nested_block_facts nested_block_tables []
      max_unbox_size false [v 11; v 10] = true
  ∧ check_to_unbox no_local nested_block_facts nested_block_tables []
      max_unbox_size false [v 10] = false
  ∧ unit_unboxed_fields nested_block_facts nested_block_tables
      [(v 11, v 11); (v 10, v 10); (v 10, v 12)] [v 11; v 10] nested_block
    = ([(v 11,
         [(Field_block 0 Value,
           Unboxed [(Field_block 1 Value, Not_unboxed 14)])]);
        (v 10, [(Field_block 1 Value, Not_unboxed 15)]);
        (v 12, [(Field_block 1 Value, Not_unboxed 16)])], 17).
Proof. vm_compute. auto. Qed.

(* What rebuild.ml makes of it: the [Let] of [x10] binds the variable of
   its field 1; that of [x11] binds the variable of field 1 of its field 0
   to that of [x10] ([bind_fields]); the load into [x12], which is
   unboxed, binds the variable of [x12] to that of field 0 of [x11]
   ([load_field_from_value_which_is_being_unboxed]); the load into [x13]
   is the variable of field 1 of [x12]. *)
Example rebuild_nested_block :
  rebuild_unit_unbox nested_block_facts nested_block_tables []
    (unit_code_deps Auto interface nested_block)
    [(v 11, v 11); (v 10, v 10); (v 10, v 12)] [v 11; v 10] true
    nested_block =
  Let 15 Value (Simple (int 8))
    (Let 14 Value (Simple (Var 15))
       (Let 16 Value (Simple (Var 14))
          (Let 13 Value (Simple (Var 16)) (return_to 1 [Var 13])))).
Proof. vm_compute. reflexivity. Qed.

(* With nothing unboxed, the rebuild is that of
   FlambdaReaperCallingConventions.v, on its examples. *)
Example rebuild_unit_unbox_without_unboxing :
  rebuild_unit_unbox FlambdaReaperCallingConventions.Examples.two_calls_facts
    FlambdaReaperCallingConventions.Examples.two_calls_tables []
    (unit_code_deps Auto FlambdaReaperCallingConventions.Examples.interface
       FlambdaReaperCallingConventions.Examples.two_calls)
    [] [] true FlambdaReaperCallingConventions.Examples.two_calls
  = rebuild_unit_cc_from_tables
      FlambdaReaperCallingConventions.Examples.two_calls_facts
      FlambdaReaperCallingConventions.Examples.two_calls_tables []
      (unit_code_deps Auto FlambdaReaperCallingConventions.Examples.interface
         FlambdaReaperCallingConventions.Examples.two_calls)
      true FlambdaReaperCallingConventions.Examples.two_calls.
Proof. vm_compute. reflexivity. Qed.

(* The case of [make_apply_wrapper] that the compiler gets wrong. The
   result 3 of the code dep of 50 and [x30] are dominated by the block
   [x42], which is unboxed; [x32] is not, and its decision is [Keep]. The
   call of 50 that returns to 21 has the decision [Unbox] of the callee
   for a result that the continuation keeps: the compiler makes it a call
   that never returns, the conversion of the test tool a call that
   returns to a continuation whose handler is [Invalid], which the model
   makes, with the parameter 65 for the field of the result. The unit
   returns 10, and the rebuilt unit goes wrong, as the call returns. *)
Definition kept_result_unboxed_tables : tables :=
  {| t_alias := [(m 27, m 7); (m 5, m 26); (m 22, m 7); (m 5, m 21);
       (m 23, m 25); (v 32, v 61); (m 18, m 20); (m 16, m 17); (m 1, consts);
       (m 1, v 31); (m 9, v 43); (v 44, m 11); (v 43, m 12); (m 3, v 42);
       (v 41, m 5); (v 40, m 6); (m 20, m 7); (m 25, m 7); (m 17, m 14);
       (m 2, m 10); (m 1, m 9); (m 15, c 51); (v 30, m 3); (v 32, m 3);
       (m 2, m 4); (m 24, c 50); (m 19, c 50); (m 12, consts); (m 6, consts);
       (v 31, v 40); (m 5, v 60); (m 11, v 61)];
     t_any_usage := [m 1; consts; m 24; m 19; m 15; m 2; v 31; m 10; m 9;
       c 51; m 4; c 50; v 40; v 43; m 12; m 6];
     t_any_source := [c 51; c 50; consts; m 1; m 15; m 24; m 19; m 12; m 6;
       v 43; v 40; v 31; m 9];
     t_has_usage := [m 1; consts; m 24; m 19; m 15; m 2; v 31; m 23; m 18;
       m 16; v 30; m 25; m 20; m 17; m 7; m 14; m 10; m 9; c 51; m 3; m 4;
       c 50; v 40; v 43; m 12; v 42; m 6; v 60; v 32; v 61];
     t_has_source := [c 51; c 50; consts; m 1; v 61; v 60; v 42; m 14; m 13;
       m 8; m 7; m 27; m 22; v 32; m 3; m 20; m 25; m 17; m 15; v 30; m 24;
       m 19; m 12; m 6; m 23; m 18; m 16; v 43; v 40; v 31; m 5; m 11; m 9;
       v 44; v 41];
     t_usages := [(v 60, v 60); (m 23, m 23); (m 18, m 18); (v 32, v 32);
       (m 16, m 16); (v 30, v 30); (m 25, m 23); (v 61, v 32); (m 20, m 18);
       (m 17, m 16); (m 7, m 18); (m 7, m 23); (m 14, m 16); (m 3, v 30);
       (m 3, v 32); (v 42, v 32); (v 42, v 30)];
     t_sources := [(v 60, v 60); (v 61, v 61); (v 42, v 42); (m 14, m 14);
       (m 13, m 13); (m 8, m 8); (m 7, m 7); (m 27, m 7); (m 22, m 7);
       (v 32, v 61); (m 3, v 42); (m 20, m 7); (m 25, m 7); (m 17, m 14);
       (m 23, m 7); (m 18, m 7); (m 16, m 14); (v 30, v 42); (v 32, v 42);
       (m 5, v 60); (m 11, v 61); (v 44, v 61); (v 41, v 60)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used := [(v 60, known); (v 61, unknown);
       (v 42, Field_block 0 Value); (m 14, Field_return_of_call Exn);
       (m 14, result 0); (m 14, Field_code_id_of_call_witness);
       (m 7, result 0); (m 7, Field_return_of_call Exn);
       (m 7, Field_code_id_of_call_witness)] |}.

Definition kept_result_unboxed_dominated : list (node * node) :=
  derive_dominated kept_result_unboxed_tables.

Definition kept_result_unboxed_rebuilt : expr :=
  rebuild_unit_unbox kept_result_unboxed_facts kept_result_unboxed_tables
    [c 51] (unit_code_deps Auto interface kept_result_unboxed)
    kept_result_unboxed_dominated [v 42] true kept_result_unboxed.

Example kept_result_unboxed_tables_checked :
  check_closed no_local kept_result_unboxed_facts kept_result_unboxed_tables
  = true
  ∧ check_field_used_exact no_local kept_result_unboxed_facts
      kept_result_unboxed_tables = true
  ∧ derive_cannot_change kept_result_unboxed_facts kept_result_unboxed_tables
    = [c 51]
  ∧ check_dominated kept_result_unboxed_tables kept_result_unboxed_dominated
    = true
  ∧ check_to_unbox no_local kept_result_unboxed_facts
      kept_result_unboxed_tables [c 51] max_unbox_size false [v 42] = true
  ∧ unit_unboxed_fields kept_result_unboxed_facts kept_result_unboxed_tables
      kept_result_unboxed_dominated [v 42] kept_result_unboxed
    = ([(v 42, [(Field_block 0 Value, Not_unboxed 62)]);
        (m 3, [(Field_block 0 Value, Not_unboxed 63)]);
        (v 30, [(Field_block 0 Value, Not_unboxed 64)])], 65).
Proof. vm_compute. auto 10. Qed.

Example kept_result_unboxed_wrapper :
  rebuild_apply_u false kept_result_unboxed_facts
    (solution_of_tables kept_result_unboxed_tables)
    (fun n => mem_node n [c 51])
    (unit_code_deps Auto interface kept_result_unboxed) true [50; 51]
    (bound_code_metadata kept_result_unboxed)
    (fst (unit_unboxed_fields kept_result_unboxed_facts
            kept_result_unboxed_tables kept_result_unboxed_dominated [v 42]
            kept_result_unboxed))
    65
    (ubind_continuations empty_unbox_env [(21, [Decided (Keep Value)])])
    (call_50 [int 9] 21)
  = Let_cont_non_recursive 65
      (Handler [65] (make_handler_metadata [Value]) Invalid)
      (Apply
         {| ap_callee := Var 60; ap_call_kind := Direct 50;
            ap_args := [int 9]; ap_continuation := 65;
            ap_exn_continuation := 2; ap_exn_extra_args := [];
            ap_args_arity := [Unboxed_product [Singleton Value]];
            ap_return_arity := [Unboxed_product [Singleton Value]] |}).
Proof. vm_compute. reflexivity. Qed.

Example kept_result_unboxed_goes_wrong :
  FlambdaSemanticsSpec.run 100
    (FlambdaSemanticsSpec.State kept_result_unboxed [] []
       FlambdaValueSpec.empty_static_env FlambdaValueSpec.empty_heap
         FlambdaValueSpec.no_reads)
  = FlambdaSemanticsSpec.Exit 1
      [FlambdaValueSpec.Value_const (Tagged_immediate 10)]
      FlambdaValueSpec.empty_heap FlambdaValueSpec.no_reads
  ∧ FlambdaSemanticsSpec.run 100
      (FlambdaSemanticsSpec.State kept_result_unboxed_rebuilt [] []
         FlambdaValueSpec.empty_static_env FlambdaValueSpec.empty_heap
         FlambdaValueSpec.no_reads)
    = FlambdaSemanticsSpec.Undefined.
Proof. vm_compute. auto. Qed.

(* ---- Closures ---------------------------------------------------------- *)

Definition sym (s : positive) : node := Node_symbol s.

(* A direct call of the code [id] through [callee], with the argument 9,
   returning from the unit. *)
Definition call (id : code_id) (callee : simple) : apply_expr :=
  {| ap_callee := callee; ap_call_kind := Direct id; ap_args := [int 9];
     ap_continuation := 1; ap_exn_continuation := 2;
     ap_exn_extra_args := [];
     ap_args_arity := [Singleton Value];
     ap_return_arity := [Singleton Value] |}.

(* The code 52 reads the value slot 80 of its closure [x41] into [x45],
   and returns field 0 of that; its parameter is not used. The unit puts
   the block [x10] in the slot 80 of the closure [x60] of a set, and
   calls 52 through it. *)
Definition code_52 : code :=
  Code [40] 41 3 4 (make_code_metadata [Singleton Value] [Singleton Value])
    (Let 45 Value (Project_value_slot_from 70 80 (Var 41))
       (Let 46 Value (load 0 (Var 45)) (return_to 3 [Var 46]))).

Definition block_in_slot : expr :=
  Let_static [Bind_code 52 code_52]
    (Let 10 Value (make_block [int 7; int 8])
       (Let 60 Value (Closure_of_set 70 [(70, 52)] [(80, Var 10)])
          (Apply (call 52 (Var 60))))).

(* The code 53 returns the value slot 80 of its closure. The unit calls
   it through the closure [x60], whose slot 80 holds 7, or through the
   closure of a static set, bound to the symbol 90. *)
Definition code_53 : code :=
  Code [40] 41 3 4 (make_code_metadata [Singleton Value] [Singleton Value])
    (Let 45 Value (Project_value_slot_from 70 80 (Var 41))
       (return_to 3 [Var 45])).

Definition unboxed_closure : expr :=
  Let_static [Bind_code 53 code_53]
    (Let 60 Value (Closure_of_set 70 [(70, 53)] [(80, int 7)])
       (Apply (call 53 (Var 60)))).

Definition static_set : expr :=
  Let_static
    [Bind_code 53 code_53;
     Bind_symbol 90
       (Static_const_closure_of_set 70 [(70, 90)] [(70, 53)] [(80, int 7)])]
    (Apply (call 53 (Symbol 90))).

Definition block_in_slot_facts : facts :=
  traverse_unit Auto interface block_in_slot.

Definition unboxed_closure_facts : facts :=
  traverse_unit Auto interface unboxed_closure.

Definition static_set_facts : facts :=
  traverse_unit Auto interface static_set.

(* The solutions of the three units, as the solver of the test tool
   computes them. The code deps of the code of each have the result 3,
   the exception 4, [my_closure] 5, the parameter 6 and the witnesses 7
   (of known arity) and 8; the call makes the witness 9. *)
Definition block_in_slot_tables : tables :=
  {| t_alias := [(m 13, m 7); (m 5, m 12); (m 9, m 11); (m 3, v 46);
       (v 41, m 5); (v 40, m 6); (m 11, m 7); (m 1, m 3); (m 2, m 4);
       (m 10, c 52); (m 6, consts); (m 5, v 60); (v 45, v 10);
       (v 46, consts)];
     t_any_usage := [m 1; m 10; m 2; m 3; m 4; c 52; v 46; consts];
     t_any_source := [c 52; consts; m 10; m 6; v 40; v 46; m 3; m 1];
     t_has_usage := [m 1; m 10; m 2; m 9; m 11; m 7; m 3; m 4; c 52; v 46;
       v 60; v 45; v 41; m 5; v 10; consts; m 12];
     t_has_source := [c 52; consts; v 60; v 10; m 8; m 7; m 13; m 11; m 10;
       m 6; m 9; v 40; m 5; v 41; v 45; v 46; m 3; m 1];
     t_usages := [(v 60, v 60); (m 9, m 9); (v 45, v 45); (v 41, v 41);
       (m 11, m 9); (m 5, v 41); (m 12, v 41); (m 7, m 9); (v 60, v 41);
       (v 10, v 45)];
     t_sources := [(v 60, v 60); (v 10, v 10); (m 8, m 8); (m 7, m 7);
       (m 13, m 7); (m 11, m 7); (m 9, m 7); (m 5, v 60); (v 41, v 60);
       (v 45, v 10)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used := [(v 60, known); (v 60, Field_value_slot 80);
       (v 10, Field_block 0 Value); (m 7, result 0);
       (m 7, Field_return_of_call Exn);
       (m 7, Field_code_id_of_call_witness)] |}.

Definition unboxed_closure_tables : tables :=
  {| t_alias := [(m 13, m 7); (m 5, m 12); (m 9, m 11); (m 3, v 45);
       (v 41, m 5); (v 40, m 6); (m 11, m 7); (m 1, m 3); (m 2, m 4);
       (m 10, c 53); (m 6, consts); (m 5, v 60); (v 45, consts)];
     t_any_usage := [m 1; m 10; m 2; m 3; m 4; c 53; v 45; consts];
     t_any_source := [c 53; consts; m 10; m 6; v 40; v 45; m 3; m 1];
     t_has_usage := [m 1; m 10; m 2; m 9; m 11; m 7; m 3; m 4; c 53; v 45;
       v 60; v 41; m 5; consts; m 12];
     t_has_source := [c 53; consts; v 60; m 8; m 7; m 13; m 11; m 10; m 6;
       m 9; v 40; m 5; v 41; v 45; m 3; m 1];
     t_usages := [(v 60, v 60); (m 9, m 9); (v 41, v 41); (m 11, m 9);
       (m 5, v 41); (m 12, v 41); (m 7, m 9); (v 60, v 41)];
     t_sources := [(v 60, v 60); (m 8, m 8); (m 7, m 7); (m 13, m 7);
       (m 11, m 7); (m 9, m 7); (m 5, v 60); (v 41, v 60)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used := [(v 60, known); (v 60, Field_value_slot 80);
       (m 7, result 0); (m 7, Field_return_of_call Exn);
       (m 7, Field_code_id_of_call_witness)] |}.

Definition static_set_tables : tables :=
  {| t_alias := [(m 13, m 7); (m 5, m 12); (m 9, m 11); (m 3, v 45);
       (v 41, m 5); (v 40, m 6); (m 11, m 7); (m 1, m 3); (m 2, m 4);
       (m 10, c 53); (m 6, consts); (m 5, sym 90); (v 45, consts)];
     t_any_usage := [m 1; m 10; m 2; m 3; m 4; c 53; v 45; consts];
     t_any_source := [c 53; consts; m 10; m 6; v 40; v 45; m 3; m 1];
     t_has_usage := [m 1; m 10; m 2; m 9; m 11; m 7; m 3; m 4; c 53; v 45;
       sym 90; v 41; m 5; consts; m 12];
     t_has_source := [c 53; consts; sym 90; m 8; m 7; m 13; m 11; m 10;
       m 6; m 9; v 40; m 5; v 41; v 45; m 3; m 1];
     t_usages := [(sym 90, sym 90); (m 9, m 9); (v 41, v 41); (m 11, m 9);
       (m 5, v 41); (m 12, v 41); (m 7, m 9); (sym 90, v 41)];
     t_sources := [(sym 90, sym 90); (m 8, m 8); (m 7, m 7); (m 13, m 7);
       (m 11, m 7); (m 9, m 7); (m 5, sym 90); (v 41, sym 90)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used := [(sym 90, known); (sym 90, Field_value_slot 80);
       (m 7, result 0); (m 7, Field_return_of_call Exn);
       (m 7, Field_code_id_of_call_witness)] |}.

Example closure_solutions :
  check_closed no_local block_in_slot_facts block_in_slot_tables = true
  ∧ check_field_used_exact no_local block_in_slot_facts block_in_slot_tables
    = true
  ∧ check_closed no_local unboxed_closure_facts unboxed_closure_tables = true
  ∧ check_field_used_exact no_local unboxed_closure_facts
      unboxed_closure_tables = true
  ∧ check_closed no_local static_set_facts static_set_tables = true
  ∧ check_field_used_exact no_local static_set_facts static_set_tables
    = true.
Proof. vm_compute. auto 10. Qed.

(* A block in a slot of a closure whose set changes its representation.
   The closure [x60] is called, and closures are not lambda-lifted: it
   cannot be unboxed ([cu0_called]). It can change its representation:
   it has a use, and nothing escapes. The block [x10] is unboxed, as the
   set it is stored in changes its representation ([cu0_stored] does not
   hold); it dominates [x45], which reads the slot. The made nodes 2 and
   4 have a use and are in the table of [to_change_representation]; they
   are not closures, have no source, and the compiler's table may leave
   them out: the allocations of the
   names that [changed_representation] gives, those of [x60], [x41] and
   the [my_closure] 5 of the code dep, are [x60]. A table without [x60],
   or with [x10], fails the check. *)
Definition block_in_slot_dominated : list (node * node) :=
  [(v 60, v 60); (v 10, v 10); (m 8, m 8); (m 7, m 7); (m 7, m 13);
   (m 7, m 11); (m 7, m 9); (v 60, m 5); (v 60, v 41); (v 10, v 45)].

Example block_in_slot_tables_checked :
  derive_cannot_change block_in_slot_facts block_in_slot_tables = []
  ∧ derive_dominated block_in_slot_tables = block_in_slot_dominated
  ∧ derive_to_unbox no_local block_in_slot_facts block_in_slot_tables []
      max_unbox_size false = [v 10]
  ∧ derive_to_change_representation no_local block_in_slot_facts
      block_in_slot_tables [] max_unbox_size false [v 10] = [m 2; m 4; v 60]
  ∧ to_change_representation_of_alloc_points block_in_slot_tables
      [v 60; v 60; v 60] = [v 60]
  ∧ check_to_change_representation no_local block_in_slot_facts
      block_in_slot_tables [] max_unbox_size false [v 10] [v 60] = true
  ∧ check_to_change_representation no_local block_in_slot_facts
      block_in_slot_tables [] max_unbox_size false [v 10] [m 2] = false
  ∧ check_to_change_representation no_local block_in_slot_facts
      block_in_slot_tables [] max_unbox_size false [v 10] [v 60; v 10]
    = false
  ∧ check_sets_whole block_in_slot_facts block_in_slot_tables [v 10] [v 60]
    = true.
Proof. vm_compute. auto 10. Qed.

(* The unboxed fields: field 0 of [x10] is the variable 61, that of [x45]
   62, the names after every name of the unit. The representation of the
   set, which every name that [x60] dominates has: its slot 80 holds the
   block unboxed, field 0 of which is the new value slot 81; its function
   slot 70 is the new one 82. The slots of the unit are 70 and 80, and the
   new ones the names after them, the value slots first. *)
Definition block_in_slot_repr : closure_repr :=
  {| cr_fields :=
       [(Field_value_slot 80,
         Unboxed [(Field_block 0 Value, Not_unboxed 81)])];
     cr_function_slots := [(70, 82)];
     cr_current := 70 |}.

Example block_in_slot_representations :
  unit_unboxed_fields block_in_slot_facts block_in_slot_tables
    block_in_slot_dominated [v 10] block_in_slot
  = ([(v 10, [(Field_block 0 Value, Not_unboxed 61)]);
      (v 45, [(Field_block 0 Value, Not_unboxed 62)])], 63)
  ∧ unit_changed_representations no_local block_in_slot_facts
      block_in_slot_tables block_in_slot_dominated [v 10] [v 60]
      block_in_slot
    = [(v 60, block_in_slot_repr); (m 5, block_in_slot_repr);
       (v 41, block_in_slot_repr)].
Proof. vm_compute. auto. Qed.

(* What rebuild.ml makes of the unit. The [Let] of the block binds the
   variable 61 of its field 0 to 7. The set is made with the new slots
   ([rewrite_set_of_closures]): the slot 81 holds the variable of field 0
   of the block, and the function slot is 82. In the code, which deletes
   its parameter, [x45] is unboxed and read from [x41], whose
   representation changes: its variable 62 is the projection of the slot
   81, at the function slot 82
   ([load_field_from_value_which_is_being_unboxed]); the load into [x46]
   is that variable ([rewrite_field_access]). *)
Example rebuild_block_in_slot :
  rebuild_unit_unbox_repr no_local block_in_slot_facts block_in_slot_tables
    [] (unit_code_deps Auto interface block_in_slot) block_in_slot_dominated
    [v 10] [v 60] true block_in_slot =
  Let_static
    [Bind_code 52
       (Code [] 41 3 4
          {| cm_params_arity := [Unboxed_product []];
             cm_result_arity := [Unboxed_product [Singleton Value]];
             cm_is_tupled := false;
             cm_is_stub := false |}
          (Let 62 Value (Project_value_slot_from 82 81 (Var 41))
             (Let 46 Value (Simple (Var 62)) (return_to 3 [Var 46]))))]
    (Let 61 Value (Simple (int 7))
       (Let 60 Value (Closure_of_set 82 [(82, 52)] [(81, Var 61)])
          (Apply
             {| ap_callee := Var 60; ap_call_kind := Direct 52;
                ap_args := []; ap_continuation := 1;
                ap_exn_continuation := 2; ap_exn_extra_args := [];
                ap_args_arity := [Unboxed_product []];
                ap_return_arity := [Unboxed_product [Singleton Value]] |}))).
Proof. vm_compute. reflexivity. Qed.

(* An unboxed closure. With closures lambda-lifted, the closure [x60],
   which is only called, is unboxed; without, it is not. It dominates
   [x41] and the [my_closure] 5 of the code dep, whose slot 80 is the
   variable 62: the decision of [my_closure] is [Unbox_my_closure]. *)
Definition unboxed_closure_dominated : list (node * node) :=
  derive_dominated unboxed_closure_tables.

Example unboxed_closure_tables_checked :
  derive_cannot_change unboxed_closure_facts unboxed_closure_tables = []
  ∧ derive_to_unbox no_local unboxed_closure_facts unboxed_closure_tables []
      max_unbox_size true = [v 60]
  ∧ derive_to_unbox no_local unboxed_closure_facts unboxed_closure_tables []
      max_unbox_size false = []
  ∧ check_to_change_representation no_local unboxed_closure_facts
      unboxed_closure_tables [] max_unbox_size true [v 60] [m 2; m 4] = true
  ∧ check_sets_whole unboxed_closure_facts unboxed_closure_tables [v 60] []
    = true
  ∧ unit_unboxed_fields unboxed_closure_facts unboxed_closure_tables
      unboxed_closure_dominated [v 60] unboxed_closure
    = ([(v 60, [(Field_value_slot 80, Not_unboxed 61)]);
        (m 5, [(Field_value_slot 80, Not_unboxed 62)]);
        (v 41, [(Field_value_slot 80, Not_unboxed 63)])], 64)
  ∧ unit_calling_convention_changes_u unboxed_closure_facts
      unboxed_closure_tables [] (unit_code_deps Auto interface unboxed_closure)
      unboxed_closure_dominated [v 60] unboxed_closure
    = [(53, Changing_u [Decided Delete] [Decided (Keep Value)])]
  ∧ unit_my_closure_decisions_u unboxed_closure_facts unboxed_closure_tables
      [] (unit_code_deps Auto interface unboxed_closure)
      unboxed_closure_dominated [v 60] unboxed_closure
    = [(53, Some [(Field_value_slot 80, Not_unboxed 62)])].
Proof. vm_compute. auto 10. Qed.

(* What rebuild.ml makes of it. The set is not made: the variable 61 of
   its slot 80 is bound to 7
   ([rebuild_set_of_closures_binding_which_is_being_unboxed]). The code
   has the variable 63 of the slot of its [my_closure] for parameter, in
   front of the others, here none ([rebuild_function_params_and_body]),
   and the projection is that variable. The call gives the variable of
   the slot of the callee, at the front of the first group, and has no
   callee ([rebuild_apply], [Unbox_my_closure]): here, poison. *)
Example rebuild_unboxed_closure :
  rebuild_unit_unbox_repr no_local unboxed_closure_facts
    unboxed_closure_tables [] (unit_code_deps Auto interface unboxed_closure)
    unboxed_closure_dominated [v 60] [m 2; m 4] true unboxed_closure =
  Let_static
    [Bind_code 53
       (Code [63] 41 3 4
          {| cm_params_arity := [Unboxed_product [Singleton Value]];
             cm_result_arity := [Unboxed_product [Singleton Value]];
             cm_is_tupled := false;
             cm_is_stub := false |}
          (Let 45 Value (Simple (Var 63)) (return_to 3 [Var 45])))]
    (Let 61 Value (Simple (int 7))
       (Apply
          {| ap_callee := Const (Poison Value); ap_call_kind := Direct 53;
             ap_args := [Var 61]; ap_continuation := 1;
             ap_exn_continuation := 2; ap_exn_extra_args := [];
             ap_args_arity := [Unboxed_product [Singleton Value]];
             ap_return_arity := [Unboxed_product [Singleton Value]] |})).
Proof. vm_compute. reflexivity. Qed.

(* A static set. The symbol 90 cannot be unboxed ([cu0_symbol]), and
   changes its representation: its slot 80, read into [x45], which the
   unit returns, is the new value slot 81, and its function slot 70 the
   new one 82. *)
Definition static_set_repr : closure_repr :=
  {| cr_fields := [(Field_value_slot 80, Not_unboxed 81)];
     cr_function_slots := [(70, 82)];
     cr_current := 70 |}.

Example static_set_tables_checked :
  derive_cannot_change static_set_facts static_set_tables = []
  ∧ derive_to_unbox no_local static_set_facts static_set_tables []
      max_unbox_size false = []
  ∧ derive_to_change_representation no_local static_set_facts
      static_set_tables [] max_unbox_size false [] = [m 2; m 4; sym 90]
  ∧ check_to_change_representation no_local static_set_facts
      static_set_tables [] max_unbox_size false [] [sym 90] = true
  ∧ unit_changed_representations no_local static_set_facts static_set_tables
      (derive_dominated static_set_tables) [] [sym 90] static_set
    = [(sym 90, static_set_repr); (m 5, static_set_repr);
       (v 41, static_set_repr)].
Proof. vm_compute. auto 10. Qed.

(* What rebuild.ml makes of it: the pattern of the set and its
   function declarations have the new function slot
   ([rebuild_let_expr_static_consts]), its value slots the new one
   ([rewrite_set_of_closures]); the projection in the code is at the new
   slots ([rewrite_field_access_chg_repr]). *)
Example rebuild_static_set :
  rebuild_unit_unbox_repr no_local static_set_facts static_set_tables []
    (unit_code_deps Auto interface static_set)
    (derive_dominated static_set_tables) [] [sym 90] true static_set =
  Let_static
    [Bind_code 53
       (Code [] 41 3 4
          {| cm_params_arity := [Unboxed_product []];
             cm_result_arity := [Unboxed_product [Singleton Value]];
             cm_is_tupled := false;
             cm_is_stub := false |}
          (Let 45 Value (Project_value_slot_from 82 81 (Var 41))
             (return_to 3 [Var 45])));
     Bind_symbol 90
       (Static_const_closure_of_set 82 [(82, 90)] [(82, 53)] [(81, int 7)])]
    (Apply
       {| ap_callee := Symbol 90; ap_call_kind := Direct 53; ap_args := [];
          ap_continuation := 1; ap_exn_continuation := 2;
          ap_exn_extra_args := [];
          ap_args_arity := [Unboxed_product []];
          ap_return_arity := [Unboxed_product [Singleton Value]] |}).
Proof. vm_compute. reflexivity. Qed.

End Examples.
