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
   unboxed, the variables that stand for their fields, and the rebuild of
   the unit with them, the changes of calling convention included.

   This file transcribes, in middle_end/flambda2/reaper/:
   - points_to_analysis.ml: the rules of [post_processing_rules] that
     conclude [multiple_allocation_points] and
     [dominated_by_allocation_point] ([derive_dominated], and the check
     [check_dominated]); [get_direct_usages],
     [add_usages_through_function_slots], [get_fields] and
     [get_single_field_source];
   - unboxing_analysis.ml: [unboxing_rules] but those of
     [to_change_representation], with the seeds of [cannot_unbox0] of
     [perform_analysis0] ([unboxing_relations], the checks
     [check_unboxing_closed] and [check_to_unbox], and [derive_to_unbox]);
     [has_to_be_unboxed], [mk_unboxed_fields] and the unboxed fields of
     [perform_analysis0] ([unboxed_fields_of_names]); [param_decision]
     with [Unbox], [arity_of_decisions], [get_arity_and_modes] and
     [compute_code_changes] ([code_change_u]);
   - rebuild.ml: [rewrite_simple] of an unboxed name ([poison]),
     [get_args], [get_args_with_kinds], [bind_fields],
     [bind_field_to_simple], [should_keep_param] of [rebuild]
     ([handler_decisions]), [rewrite_apply_cont_expr] ([get_args]),
     [load_field_from_value_which_is_being_unboxed],
     [rebuild_singleton_binding_which_is_being_unboxed],
     [rewrite_field_access] of [rebuild_named_default_case]
     ([rebuild_let_u]), [freshen_decisions], [make_apply_wrapper] with
     every pair of decisions, [rebuild_apply] with an unboxed callee and
     with [Unbox] decisions, and [rebuild_function_params_and_body] with
     them.
   It needs no review: FlambdaReaperUnboxingSpec.v says what the checks
   mean, and the test tool compares the decisions and the rebuilt term
   with those of the compiler's reaper. It is extracted
   ([derive_dominated], [check_dominated], [derive_to_unbox],
   [check_to_unbox], [to_unbox_of_unboxed_names], [unit_unboxed_fields],
   [unit_calling_convention_changes_u], [rebuild_unit_unbox]).

   What is transcribed is the reaper run with
   [-reaper-change-calling-conventions] and [-reaper-unbox], the
   compiler's defaults, for blocks and boxed numbers. The flags that the
   rules read are parameters: the local fields ([is_local]),
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
   results that the code dep has unboxed. With nothing unboxed, the
   rebuild is that of FlambdaReaperCallingConventions.v (an example
   below shows it on the examples of that file).

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
   unit, which returns 10, and whose rebuild goes wrong.

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

   Left out: closures. The change of representation of closures
   ([to_change_representation], [Closure_representation], the renaming
   of their value and function slots in [rewrite_set_of_closures],
   [rebuild_singleton_binding_whose_representation_is_being_changed],
   [rewrite_field_access_chg_repr], the static sets of
   [rebuild_let_expr_static_consts]), the unboxing of a set of closures
   ([rebuild_set_of_closures_binding_which_is_being_unboxed]) and of
   [my_closure] ([Unbox_my_closure], the fields of an unboxed callee
   given as arguments). The model's decision of [my_closure] is
   [Keep_my_closure], a set of closures is rebuilt as if no closure were
   unboxed, and a projection of a value slot from an unboxed closure is
   the variable of the slot. [Block_representation] is left out too: the
   compiler makes one only for names that its traversal makes, which are
   not in the term. The test tool reports a unit that unboxes a closure,
   or changes the representation of a name of the term other than by
   renaming the slots of a closure, as out of the model. *)

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

End Rules.

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

End Unboxed_fields.

(* The unboxed fields of the names of the unit [e], with variables after
   every name of the unit. *)
Definition unit_unboxed_fields (F : facts) (T : tables)
    (dominated : list (node * node)) (to_unbox : list node) (e : expr)
    : list (node * unboxed_fields) * positive :=
  unboxed_fields_of_names F T (bound_code_my_closures e) dominated to_unbox
    (fresh_name (names_of e)).

(* ======================================================================== *)
(* The decisions: [compute_code_changes] with [Unbox]                       *)
(* ======================================================================== *)

(* [Unboxing_analysis.param_decision]: one of
   FlambdaReaperCallingConventions.v, [Keep] or [Delete], or [Unbox] with
   the unboxed fields of the parameter. *)
Inductive decision : Type :=
  | Decided (d : param_decision)
  | Unbox (fields : unboxed_fields).

(* The parameters that a decision makes of [x] ([get_parameters]): [x] if
   kept, the variables of its fields if unboxed. *)
Definition decision_params (x : variable) (d : decision)
    : list (variable * flambda_kind) :=
  match d with
  | Decided (Keep kind) => [(x, kind)]
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

(* The metadata of code that changes its calling convention
   ([compute_code_changes]). *)
Definition changed_code_metadata_u (metadata : code_metadata)
    (params_decisions return_decisions : list decision) : code_metadata :=
  {| cm_params_arity :=
       arity_of_grouped_decisions_u
         (group_by_parameter (cm_params_arity metadata) params_decisions);
     cm_result_arity := arity_of_decisions_u return_decisions;
     cm_is_tupled := false;
     cm_is_stub := cm_is_stub metadata |}.

(* [Unboxing_analysis.calling_convention_change], without the decision of
   [my_closure], which is [Keep_my_closure]. *)
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
  ue_decisions : finite_map (list decision)
}.

Definition empty_unbox_env : unbox_env :=
  {| ue_env := empty_env; ue_decisions := [] |}.

Definition ubind_variables_then (en : unbox_env)
    (bound : list (variable * flambda_kind)) (projections : list variable)
    : unbox_env :=
  {| ue_env := bind_variables_then (ue_env en) bound projections;
     ue_decisions := ue_decisions en |}.

Definition ubind_variables (en : unbox_env)
    (bound : list (variable * flambda_kind)) : unbox_env :=
  ubind_variables_then en bound [].

Definition is_kept_decision (d : decision) : bool :=
  match d with
  | Decided d => is_kept d
  | Unbox _ => true
  end.

Definition ubind_continuations (en : unbox_env)
    (bound : list (continuation * list decision)) : unbox_env :=
  {| ue_env :=
       bind_continuations (ue_env en)
         (map (fun '(k, ds) => (k, map is_kept_decision ds)) bound);
     ue_decisions := bound ++ ue_decisions en |}.

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
  | (Decided (Keep _), kind) :: decisions =>
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
   continuation for its parameter and that of the callee; [None] where
   one decision unboxes and the other does not ([Invalid]). A result
   beyond the decisions of the continuation, or of a continuation that has
   none, is kept, as a continuation of fixed arity keeps it. *)
Fixpoint wrapper_args_u (apply_decisions : list decision)
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
    match wrapper_args_u apply_decisions func with
    | None => None
    | Some rest =>
      match ad, fd with
      (* The case where the compiler stops with a fatal error if
         [has_any_source], a test that is [false] there ("CR ncourant:
         restore this check"). *)
      | Unbox _, (Wrapper_keep _ | Wrapper_delete)
      | Decided _, Wrapper_unbox _ => None
      | Decided Delete, _ => Some rest
      | Decided (Keep _), Wrapper_keep x => Some (Var x :: rest)
      | Decided (Keep kind), Wrapper_delete =>
        Some (Const (Poison kind) :: rest)
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
  | None => None
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
    | None => Let x kind (rebuild_named sol_u en x d) body
    end
  | None => Let x kind (rebuild_named sol_u en x d) body
  end.

(* [should_keep_param] of [rebuild]: an unboxed parameter is [Unbox] of
   its fields; any other is kept as FlambdaReaperRebuild.params_to_keep
   says. *)
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
         | None => Decided (if keep then Keep kind else Delete)
         end)
      (combine bound (params_to_keep sol is_exn_handler bound))
  end.

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
      | Decided (Keep _) => Some (rewrite_simple sol_u en arg :: rest)
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
      | Decided (Keep kind) =>
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
      | Decided (Keep _) => Some ((rewrite_simple sol_u en s, kind) :: rest)
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

(* [make_apply_wrapper], given the decisions of the callee for its
   results, each with the kind of the result in the return arity of the
   call: no wrapper if what the continuation of the call is given is what
   the callee returns, in order; otherwise the call returns to a wrapper,
   the continuation [first_fresh], whose parameters are what the callee
   returns. Where the two decisions disagree on unboxing, the compiler
   makes a call that never returns, which the conversion of the test tool
   writes as a call returning to a continuation whose handler is
   [Invalid]: the wrapper, with that handler. *)
Definition make_apply_wrapper_u (en : unbox_env) (ap : apply_expr)
    (func_decisions : list (decision * flambda_kind)) : expr :=
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
  match wrapper_args_u apply_decisions func with
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
   unboxed callee, which makes it [Invalid]; one whose callee changes gives
   the parameters kept and the fields of those unboxed, a product for each
   group, and expects the results kept and the fields of those
   unboxed. *)
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
               ap_args := rebuild_arguments F sol_u (ue_env en) ap;
               ap_continuation := ap_continuation ap;
               ap_exn_continuation := ap_exn_continuation ap;
               ap_exn_extra_args := exn_extra_args;
               ap_args_arity := ap_args_arity ap;
               ap_return_arity := ap_return_arity ap |}
            (map (fun kind => (Decided (Keep kind), kind))
               (unarize (ap_return_arity ap)))
        end
      | Changing_u params_decisions return_decisions =>
        match
          all_some
            (map
               (fun '(args, decisions) =>
                  get_args_with_kinds_u (ue_env en) decisions args)
               (combine (group_by_parameter (ap_args_arity ap) (ap_args ap))
                  (group_by_parameter (ap_args_arity ap) params_decisions)))
        with
        | None => Invalid
        | Some groups =>
          make_apply_wrapper_u en
            {| ap_callee := ap_callee ap;
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
        end
      end
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
  | Decided _ => d
  end.

(* [rebuild_expr] of FlambdaReaperCallingConventions.v, with unboxing. *)
Fixpoint rebuild_expr_u (en : unbox_env) (e : expr) {struct e} : expr :=
  match e with
  | Let x kind defining_expr body =>
    if is_pending_projection (ue_env en) x defining_expr
    then
      Let x kind defining_expr
        (rebuild_expr_u
           (ubind_variables_then en [(x, kind)]
              (tl (env_projections (ue_env en))))
           body)
    else
      match defining_expr with
      | Set_of_closures _ _ | Closure_of_set _ _ _ =>
        let projections := projections_of_set x defining_expr body in
        if is_dead_var sol x kind
           || existsb (fun y => is_dead_var sol y Value) projections
        then Invalid
        else if is_var_used sol x kind
                || existsb (fun y => is_var_used sol y Value) projections
        then
          Let x kind
            (rebuild_set sol_u (ue_env en)
               (set_closure_nodes F (Node_variable x)) defining_expr)
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
          | None => rebuild_let_u (ue_env en) x kind defining_expr body
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
    if static_const_is_used sol s const
    then [Bind_symbol s (rebuild_static_const sol_u (ue_env en) s const)]
    else []
  end
(* [rebuild_code] and [rebuild_function_params_and_body]: code that
   changes its calling convention has the parameters that its decisions
   make, the arities of [changed_code_metadata_u], and its return
   continuation the decisions of its results. *)
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
      Code
        (map fst
           (flat_map (fun '(x, d) => decision_params x (param_decision_at x d))
              (combine params params_decisions)))
        my_closure return_continuation exn_continuation
        (changed_code_metadata_u metadata params_decisions return_decisions)
        body
    end
  end.

End Rebuild.

(* [Rebuild.rebuild] with unboxing and changes of calling convention: the
   rebuilt unit, given the facts of its graph, the solution, the tables of
   [cannot_change_calling_convention], [dominated_by_allocation_point] and
   [to_unbox], and the code deps of the traversal. The variables of the
   unboxed fields are the names after every name of the unit, and the
   wrappers the names after them. *)
Definition rebuild_unit_unbox (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) (e : expr) : expr :=
  let '(unboxed, next) := unit_unboxed_fields F T dominated to_unbox e in
  rebuild_expr_u F (solution_of_tables T) (fun n => mem_node n cannot_change)
    code_deps preserve_direct_calls (static_names code_ids_bound e)
    (bound_code_metadata e) unboxed next empty_unbox_env e.

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
  {| ap_callee := Var 32; ap_call_kind := Indirect; ap_args := [int 3];
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
  rebuild_apply_u kept_result_unboxed_facts
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
       FlambdaValueSpec.empty_static_env FlambdaValueSpec.empty_heap)
  = FlambdaSemanticsSpec.Exit 1
      [FlambdaValueSpec.Value_const (Tagged_immediate 10)]
      FlambdaValueSpec.empty_heap
  ∧ FlambdaSemanticsSpec.run 100
      (FlambdaSemanticsSpec.State kept_result_unboxed_rebuilt [] []
         FlambdaValueSpec.empty_static_env FlambdaValueSpec.empty_heap)
    = FlambdaSemanticsSpec.Undefined.
Proof. vm_compute. auto. Qed.

End Examples.
