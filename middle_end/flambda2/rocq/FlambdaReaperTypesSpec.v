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

(* What the rewriting of the types of the reaper is worth: the types it
   rewrites are true of the values of the rebuilt unit.

   This file is to be reviewed: [describes], [metadata_at],
   [rebuilt_value] and [statics_rebuilt] (what the rebuild makes of a
   value and of the static environment, with [set_sites], [slot_held]
   and [value_slot_rebuilt] for closures), the fragment ([shaped_type],
   [type_in_fragment], [env_in_fragment]), [sets_of_graph],
   [copies_valued], [rebuild_exit_relation] (what the statement of the
   two runs asks of the simulation of the rebuild), [statics_of_values]
   and [exits_of_values] (its premise on static blocks), [unboxed_values] and
   [block_fields_of_one_kind] (for unboxed variables of code), the
   statements, and the examples, against
   FlambdaReaperTypes.v (the rewriting, which needs no review),
   FlambdaReaperFlowSpec.v (where values may be),
   FlambdaReaperLeastSpec.v (the least solution) and
   FlambdaReaperRebuildSpec.v (what the rebuild does). The meaning of a
   type is FlambdaTypeMeaningSpec.type_satisfied. The proofs are in
   FlambdaReaperTypesProofs.v, which needs no review.

   The setting. The reaper rewrites the types of the final typing
   environment of a unit, which the .cmx file carries to the units that
   use it: the type of each symbol of the unit, under the metadata of
   its node ([metadata_for_name]), and, under the metadata that an alias
   to a name is reached with, the type of a copy of that name
   ([get_canonical_with]). Two runs are compared: that of the unit, with
   a static environment [statics] and a valuation [valuation] of the
   names of the typing environment that satisfies it
   ([typing_env_satisfied_strictly]); and that of the rebuilt unit, with
   its static environment [statics'], and a valuation [valuation'] of the
   names that the rewritten types mention.

   What the rebuild makes of a value ([rebuilt_value v n v']: [v] is a
   value of the first run that may be at the node [n], and [v'] what the
   second run has for it), as the simulation of
   FlambdaReaperRebuildProofs.v relates the values of the two runs
   ([rel]): a constant is itself; a symbol is itself, and has a use if
   the node has one; a block has its tag and as many fields, each field
   that the allocation [a] it may come from uses related, at the node
   that [a] holds it in, and the others poison of their kind; a boxed
   number is itself, or poison where its contents have no use; an
   immutable array or a partial application is one, of the same arity.
   A closure of a set ([Value_closure_of_set]) is one of the same set at
   the same slot, with the value slots that the rebuild keeps. It has an
   allocation for each function slot of its set ([sites]), that at its
   slot may come to the node; each holds, at each function slot of the
   set, the allocation of the closure there, and at no other
   ([set_sites]). A value slot that one of them uses is held by all of
   them at one node ([slot_held]), which has a source and a use, and the
   second run has it, related there ([value_slot_rebuilt]); the others
   may be gone. The static environment of the second run binds each
   symbol that has a use as the first does, a block with its fields
   related as those of a block whose allocation is the symbol, a closure
   of a static set as a closure of a set is related, the allocation at
   its slot being the symbol, a boxed number and a string as they are,
   and no closure of a set of one function with no slot named
   ([Static_closure], which [statics_rel] has not either)
   ([statics_rebuilt]). The fields of a block are taken at their kind
   ([field_rebuilt]), as the compiler reads the field of a static block,
   at the kind that the shape of the block gives it
   (Static_const.block_field_kind). That is a choice to review: the
   model's traversal reads every field of a block at the kind value
   ([Field_block i Value]) and its rebuild makes an unused field poison
   of kind value, so the two readings agree on blocks of values, which
   the blocks that a primitive makes are, and the statement of the two
   runs asks it of the static blocks where the unit exits
   ([exits_of_values]). The model's reading, at the kind value, was
   weighed and set aside: an unused field of another kind would then be
   rebuilt as poison of kind value, of which the type of the field,
   rewritten at its own kind, does not hold, and the statements of the
   rewriting would be false of a static block of a naked number.

   What a metadata says of an allocation ([describes]), as the comment
   before [rewrite_metadata] in types_rewriter.ml says it: [No_source],
   that there is none; [Single_source a], that it is [a];
   [Many_sources_any_usage], that it has any usage, so the rebuild has
   kept it whole; [Many_sources_usages us], that it has any usage or each
   of [us] as a use, which has a use; [At_least_one_source_no_usages],
   nothing. A metadata holds at a node if it describes every allocation
   that a value at the node may come from ([metadata_at]); a value is
   rebuilt under a metadata if it is a constant, rebuilt as itself, or
   is at a node where the metadata holds, which has a use unless the
   metadata forgets the type ([rebuilt_under]).

   The fragment ([type_in_fragment]): types of a shape ([shaped_type])
   whose aliases are to names that hold, in the first run, a constant or
   a symbol ([unallocated]). The shape: in each case of closures, the
   slots of which the entry has a closure type or a type of a value slot
   are among those of the index (condition 8a of
   FlambdaTypeWellFormedSpec.v, which the creation functions of the
   compiler give, but for the closure types of
   [exactly_this_closure]), and the closure types are aliases, as those
   that Simplify gives a set of closures are. A closure type of a
   closure bound by a [Let] has then none for a function of its set: the
   closure there is allocated. The second condition excludes the aliases
   to a variable
   that holds a block: the machine has no identity of allocations, so an
   alias says that two values are equal, not that they are one
   allocation, and two positions of one block may be rebuilt
   differently. The rewriting gives the copy of a name, for a metadata,
   one name, which stands for every position that the name is reached at
   under the metadata ([copies_valued]: in the second run, the copy has
   the value that the name has in the first). For a constant or a symbol
   that is what the rebuild keeps at a position with a use.

   The sets of closures given to the rewriting ([rc_sets_of_closures])
   are to be those of the graph ([sets_of_graph]): allocations that hold
   one another at the function slots of a set, and nothing else, are
   the closures of one of them, at the same slots. The solution is to be
   the least ([least_solution]): [refine_sets_of_closures] recovers the
   sets that a closure may come from from the usages of the closure,
   which, in a solution that is only closed, may not have the
   allocation among their sources (in a least one, an allocation is a
   source of each of its usages that has not any source,
   [usages_carry_sources] in the proofs); and the metadata of a value
   slot read off the tables ([field_of_constructor_is_used_top],
   [field_used_as], [get_single_field_source]) is what the rules give
   only where no tuple is stale ([least_no_stale_statement]) and where
   a node with a source that has not any source has a source that has
   one.

   The statements:
   - [rewrite_type_sound_statement]: a type of the fragment that holds of
     a value of the first run, rewritten under a metadata that the value
     is rebuilt under, holds of the value of the second run, with every
     fix of FlambdaReaperTypes.v on, for the least solution of the
     graph and the sets of closures of the graph. Closure types are of
     the fragment: the closure case of the rewriting
     ([refine_sets_of_closures], the slots of the sets kept, of the
     sources outside the unit, and their join) is covered.
   - [rewrite_symbol_type_sound_statement]: the type of a symbol of the
     environment, rewritten as [rewrite_typing_env] rewrites it, holds of
     the symbol in the second run, for every naming of the copies that
     [copies_valued] allows, if the symbol is bound in the first run and
     an allocation of its own there ([statics_allocated], which
     FlambdaReaperFlowSpec.statics_flow gives).
   - [rewrite_result_type_sound_statement]: the type that the result
     types of code give a parameter or a result, rewritten as
     [rewrite_result_types] rewrites it under the metadata of the
     variable of the code, holds of what the second run has at that
     variable, if its type held of the value of the first run there.
     It is the type of the copy that [rewrite_result_types] makes, to
     which the new parameter is an alias. Of the runs, it asks the values
     of the variable in the two, related, as the simulation has them at
     a return of code.
   - [rewrite_unboxed_result_type_sound_statement]: the same of a
     parameter or a result whose variable of code the unboxing of
     FlambdaReaperUnboxing.v unboxes ([unit_unboxed_fields]): the code
     rebuilt with unboxing has a new parameter or result for each of its
     fields, to which [rewrite_result_types_u] gives the type at that
     field of the type of the variable ([fold_destructuring]), rewritten
     under the metadata that [patterns_for_unboxed_fields] gives the
     field. Each holds of the value of that variable in the run of the
     unit rebuilt with unboxing ([rebuild_unit_unbox_repr_fixed] and
     [_fixed_tail], with no change of representation). Of the runs, it
     asks the value [v_mid] that the run of the unit rebuilt without
     unboxing has at the variable, related to the value of the first run
     there as for a variable kept ([rebuilt_value]), and the values [ws]
     of the variables of the fields in the run with unboxing, the values
     at their paths in [v_mid] ([unboxed_values], the relation
     [fields_vals] of the simulation of
     FlambdaReaperUnboxingRefinesCallsSimProofs.v). It asks of the unit,
     as the fragment of FlambdaReaperUnboxingRefinesCallsSpec.v does,
     that no use of a name unboxed is called or has a function slot read
     from it ([check_unboxed_uses]): [patterns_for_unboxed_fields] reads
     the fields of the uses of the variable and of the closures read
     from them and called ([add_usages_through_function_slots]), and a
     field read only from the latter need not be one that the
     allocation uses, nor its metadata hold where it is. It asks of the
     graph that an allocation hold a field of a block at one kind
     ([block_fields_of_one_kind]), as the traversal gives: the field
     unboxed in turn is read at the kind of the read, and its source
     is taken among what the allocation holds at that kind.
   - [exported_types_sound_statement]: the statement of the two runs.
     Where the unit exits, from a state whose static environment and a
     valuation satisfy a typing environment of the fragment, the rebuilt
     unit exits through the same continuation, and the type of each
     symbol bound, rewritten, holds of the symbol in the static
     environment of that exit. It is proved from the second statement and
     [rebuild_exit_relation], which it takes as a premise: what it asks
     of the simulation of FlambdaReaperRebuildProofs.v, that the
     static environments of the two exits are related by
     [statics_rebuilt], and that every symbol bound in the first is an
     allocation of its own.
   - [rebuild_exit_relation_holds_statement]: [rebuild_exit_relation]
     holds under the premises of
     FlambdaReaperRebuildSpec.rebuild_refines_observably_statement, of a
     unit that binds each variable once
     (FlambdaReaperLeastSpec.variables_bound_once) and whose static
     blocks hold values where it exits ([exits_of_values]). The first
     makes the allocations of the closures of a set hold, at their
     function slots and value slots, the facts of the set and no other,
     which [set_sites] and [slot_held] ask; the second is what
     [field_rebuilt] needs of a field of a static block, and is needed
     ([naked_field_exit_relation_statement]).
   - [exported_types_sound_end_to_end_statement]: the statement of the
     two runs with no premise on the relation of the runs, for a unit
     of those premises and the least solution of its graph.
   - [boxed_fix_needed_statement] and [alias_fix_needed_statement]: the
     first statement is false without the fix of a boxed number whose
     contents have no use, and without that of the aliases under a
     metadata that forgets; [compiler_rewrite_unsound_statement]: so it
     is false of the compiler's rewriting. Each is a counterexample on a
     small graph whose least solution passes the checks of
     FlambdaReaperSolution.v. The program of the second, compiled from
     OCaml, gives a wrong type in the .cmx file: the type of a block says
     two fields are one name where the rebuild has kept the first and
     made the second poison. [boxed_result_fix_needed_statement]: the
     statement of result types is false without the first fix, on a
     result that is the box: the compiler's [rewrite_result_types] gives
     it the type of the box, which the rebuild makes poison (compiled
     from OCaml, [-flambda2-result-types-all-functions] exports such a
     result type; no use of it was found that goes wrong).
   - [refine_fix_needed_statement]: the first statement is false without
     the fix of [refine_sets_of_closures], on a closure type of the
     closure of one of two copies of a set of closures, with local value
     slots (RefineCounterexample). The closure escapes, so
     [join_metadata] makes its usages none; the sets recovered from the
     usages of the other copy are its own; and the rewritten type says
     that the closure has a value slot that only the other copy uses,
     which the rebuild removes from it. The compiler's fault is the
     same, with the representations of the two sets changed by the
     unboxing of closures, which the model leaves out: compiled from
     OCaml with [-reaper-local-fields] and result types for all
     functions, another unit inlines code that reads the value slot of
     one set from a closure that the type says is of the other, and
     reaches [Invalid]. [refine_example_statement]: the two rewritings
     of that closure type, computed.
   - [rewrite_keeps_null_statement]: [Null], which every rebuild keeps
     as it is, has every type that it had, rewritten, of any type,
     closure types included, under any metadata and any solution, with
     every fix on; [null_fix_needed_statement]: not without the fix of
     the nullability of closure types. The counterexample is a closure
     type that may be null, of the closure of a set of the unit, which
     the compiler rewrites into one that is not null. Compiled from
     OCaml, a unit whose symbol is [Null] at the type
     [(int -> int) or_null] exports that type, and another unit that
     matches on the symbol takes the branch of [This]: it prints 2
     where the program prints [null].
   - [example_statement]: the rewriting of an environment of two symbols
     and a block, computed; [result_types_example_statement]: the
     rewriting of the result types of code that returns the box, with
     the fixes and as the compiler has it.

   What is left out:
   - Of closure types, the closure types of a case that are not aliases:
     the type of the closure of another function slot of the set is
     rewritten under the metadata [Single_source] of the allocation of
     that closure, which may have no use, where a value is rebuilt under
     a metadata only at a node with a use ([rebuilt_under]), and which
     may have other sources than itself, where the graph does not say
     that it has not. Closures of a set of one function with no slot
     named ([Value_closure], [Static_closure]), which [rel] and
     [statics_rel] have not either. The partial applications, of which
     the closure types say nothing ([closure_in_set]).
   - The equations of the copies: the statements give each copy the
     value of the name it copies, and say nothing of its own equation.
     That equation is the type of the name rewritten under the metadata
     of the copy; it holds of that value where an alias reaches the copy
     in a case of a type that the value takes, and may not elsewhere (an
     alias in the case of another tag of a variant, whose fields have the
     metadata of the fields of the case taken); there the copy is to have
     no value.
   - For result types, the relation of the values of the runs at a
     return of code (of the two runs, and for an unboxed variable, of
     the three).
   - A premise on the term in place of [exits_of_values], a premise on
     the runs: the fields of the static blocks of kind value. The check
     of FlambdaWellKindedSpec.v asks it, among much else; the compiler
     reads a static block at its shape, which the model has not got.
   - Unboxing: the rewriting of the types of the closures whose
     representation changes, which the model forgets (FlambdaReaperTypes.v
     says what the compiler does), and so the rebuild with a change of
     representation; the unboxing of closures and of names whose uses
     are called (not of [check_unboxed_uses]); the fields of an unboxed
     [my_closure], which the result types give no equation, so that
     there is nothing to state of them. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaPrimitiveSpec FlambdaSemanticsSpec
  FlambdaWellFormedSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperRebuild FlambdaReaperFlowSpec
  FlambdaReaperLeastSpec FlambdaReaperTypes.
From OxCamlFlambda2 Require FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesCallsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaTypingEnv
  FlambdaTypingEnvSpec.
Import ListNotations.

(* ======================================================================== *)
(* The fragment                                                             *)
(* ======================================================================== *)

(* Whether a type is an alias. *)
Definition is_alias (t : flambda_type) : bool :=
  match get_alias_opt t with
  | Some _ => true
  | None => false
  end.

(* The shape of the types of the fragment, at any depth: in every case of
   closures, the function slots of which the entry has a closure type,
   and the value slots of which it has a type, are among those of the
   index (condition 8a of FlambdaTypeWellFormedSpec.v), and every closure
   type is an alias. *)
Fixpoint shaped_type (t : flambda_type) : bool :=
  match t with
  | Type_value (Ok (No_alias (Head_of_kind_value (Ok head) _))) =>
    shaped_non_null head
  | Type_value _ | Type_naked_immediate _ | Type_naked_number _ _
  | Type_rec_info _ | Type_region _ => true
  end

with shaped_non_null (head : head_of_kind_value_non_null) : bool :=
  match head with
  | Variant _ immediates _ blocks extensions _ =>
    match immediates with
    | Or_unknown.Known ty => shaped_type ty
    | Or_unknown.Unknown => true
    end
    && match blocks with
       | Or_unknown.Known blocks => shaped_blocks blocks
       | Or_unknown.Unknown => true
       end
  | Boxed_number _ contents _ => shaped_type contents
  | Closures closures _ => shaped_closures closures
  | Mutable_block _ | String _ => true
  end

with shaped_blocks (blocks : row_like_for_blocks) : bool :=
  match blocks with
  | Row_like_for_blocks known_tags other_tags _ =>
    forallb
      (fun p =>
         match snd p with
         | Or_unknown.Known case => forallb shaped_type (case_maps_to case)
         | Or_unknown.Unknown => true
         end)
      known_tags
    && match other_tags with
       | Or_bottom.Ok case => forallb shaped_type (case_maps_to case)
       | Or_bottom.Bottom => true
       end
  end

with shaped_closures (closures : row_like_for_closures) : bool :=
  match closures with
  | Row_like_for_closures known_closures =>
    forallb
      (fun p =>
         match case_maps_to (snd p) with
         | Closures_entry _ closure_types value_slot_types =>
           all_among (domain closure_types)
             (contents_closures (closures_index_contents (case_index (snd p))))
           && all_among (domain value_slot_types)
                (contents_value_slots
                   (closures_index_contents (case_index (snd p))))
           && forallb (fun q => is_alias (snd q)) closure_types
           && forallb (fun q => shaped_type (snd q)) value_slot_types
         end)
      known_closures
  end.

(* A value that is not allocated: a constant or a symbol. *)
Definition unallocated (v : value) : Prop :=
  (∃ c : const, v = Value_const c) ∨ (∃ s : symbol, v = Value_symbol s).

(* A type of the fragment: of the shape of the fragment, and every alias,
   of a variable of a relation or of an extension too, to a name that
   holds a constant or a symbol in [valuation]. *)
Definition type_in_fragment (valuation : env) (t : flambda_type) : Prop :=
  shaped_type t = true
  ∧ ∀ (s : simple) (k : flambda_kind) (w : value),
      (s, k) ∈ aliases_with_kinds t ->
      eval_simple valuation s = Some w ->
      unallocated w.

(* Every type of the environment is of the fragment. *)
Definition env_in_fragment (valuation : env) (tenv : Typing_env.t) : Prop :=
  ∀ (n : name) (ty : flambda_type)
    (binding_time_and_mode : Binding_time.With_name_mode.t),
    Keyed.lookup name_eqb n (Typing_env.names_to_types tenv)
    = Some (ty, binding_time_and_mode) ->
    type_in_fragment valuation ty.

(* ======================================================================== *)
(* What the rebuild makes of a value                                        *)
(* ======================================================================== *)

(* [P i field field'] of the fields of two blocks, position by position,
   the first at [i]; the two have as many fields. *)
Definition fields_rebuilt (P : nat -> value -> value -> Prop)
    : nat -> list value -> list value -> Prop :=
  fix fields_rebuilt (i : nat) (fields fields' : list value) {struct fields}
      : Prop :=
    match fields, fields' with
    | [], [] => True
    | field :: fields, field' :: fields' =>
      P i field field' ∧ fields_rebuilt (S i) fields fields'
    | _, _ => False
    end.

Section Rebuilt.

Variable G : graph.
Variable sol : solution.

(* The field [i] of a block whose allocation is [a], in the two runs:
   [a] holds it at a node [from], at its kind; if [a] uses it, it has a
   use there and is related ([rebuilt] stands for [rebuilt_value]); if
   not, it is poison of its kind. *)
Definition field_rebuilt (rebuilt : value -> node -> value -> Prop)
    (a : node) (i : nat) (field field' : value) : Prop :=
  ∃ from : node,
    g_constructor G a (Field_block i (kind_of_value field)) from
    ∧ (s_field_used sol a (Field_block i (kind_of_value field)) = true ->
       s_has_usage sol from = true ∧ rebuilt field from field')
    ∧ (s_field_used sol a (Field_block i (kind_of_value field)) = false ->
       field' = Value_const (Poison (kind_of_value field))).

(* The allocations [sites] of the closures of a set whose functions are
   [function_decls], one for each function slot: each holds, at each
   function slot of the set, the allocation of the closure there, and at
   no other. *)
Definition set_sites (function_decls : list (function_slot * code_id))
    (sites : function_slot -> node) : Prop :=
  ∀ (f : function_slot) (id : code_id),
    lookup f function_decls = Some id ->
    ∀ (g : function_slot) (b : node),
      g_constructor G (sites f) (Field_function_slot g) b
      <-> is_bound g function_decls = true ∧ b = sites g.

(* Each of these allocations holds [from] at the value slot [w], and
   nothing else. *)
Definition slot_held (function_decls : list (function_slot * code_id))
    (sites : function_slot -> node) (w : value_slot) (from : node) : Prop :=
  ∀ (f : function_slot) (id : code_id),
    lookup f function_decls = Some id ->
    ∀ b : node, g_constructor G (sites f) (Field_value_slot w) b <-> b = from.

(* The value slot [w] of a closure of the set, whose value is [x] in the
   first run, where the second has the value slots [value_slots']: if
   one of the allocations of the closures uses it, they all hold it at a
   node [from], which has a source and a use, and the second run has it,
   related ([rebuilt] stands for [rebuilt_value]). *)
Definition value_slot_rebuilt (rebuilt : value -> node -> value -> Prop)
    (function_decls : list (function_slot * code_id))
    (sites : function_slot -> node)
    (value_slots' : list (value_slot * value)) (w : value_slot) (x : value)
    : Prop :=
  ∀ (f : function_slot) (id : code_id),
    lookup f function_decls = Some id ->
    s_field_used sol (sites f) (Field_value_slot w) = true ->
    ∃ (from : node) (x' : value),
      slot_held function_decls sites w from
      ∧ s_has_source sol from = true ∧ s_has_usage sol from = true
      ∧ lookup w value_slots' = Some x' ∧ rebuilt x from x'.

(* The value [v] of the first run may be at the node [n], and [v'] is
   what the second run has for it. *)
Fixpoint rebuilt_value (v : value) (n : node) (v' : value) {struct v}
    : Prop :=
  match v with
  | Value_const _ => s_any_source sol n = true ∧ v' = v
  | Value_symbol s =>
    v' = v ∧ may_come_from sol (Node_symbol s) n
    ∧ (s_has_usage sol n = true -> s_has_usage sol (Node_symbol s) = true)
  | Value_block tag fields =>
    ∃ (a : node) (fields' : list value),
      may_come_from sol a n ∧ s_has_source sol a = true
      ∧ v' = Value_block tag fields'
      ∧ fields_rebuilt
          (fun i field field' =>
             ∃ from : node,
               g_constructor G a (Field_block i (kind_of_value field)) from
               ∧ (s_field_used sol a (Field_block i (kind_of_value field))
                  = true ->
                  s_has_usage sol from = true
                  ∧ rebuilt_value field from field')
               ∧ (s_field_used sol a (Field_block i (kind_of_value field))
                  = false ->
                  field' = Value_const (Poison (kind_of_value field))))
          0 fields fields'
  | Value_boxed_number c =>
    ∃ (a : node) (bn : boxable_number) (from : node),
      may_come_from sol a n ∧ s_has_source sol a = true
      ∧ is_number_of_kind bn c = true
      ∧ g_constructor G a (Field_boxed_number bn) from
      ∧ v' = (if s_field_used sol a (Field_boxed_number bn) then v
              else Value_const (Poison Value))
  | Value_immutable_array _ =>
    ∃ elements' : list value, v' = Value_immutable_array elements'
  | Value_closure_of_set slot function_decls value_slots =>
    ∃ (sites : function_slot -> node)
      (value_slots' : list (value_slot * value)),
      v' = Value_closure_of_set slot function_decls value_slots'
      ∧ may_come_from sol (sites slot) n
      ∧ set_sites function_decls sites
      ∧ for_all_bindings
          (fun w x =>
             ∀ (f : function_slot) (id : code_id),
               lookup f function_decls = Some id ->
               s_field_used sol (sites f) (Field_value_slot w) = true ->
               ∃ (from : node) (x' : value),
                 slot_held function_decls sites w from
                 ∧ s_has_source sol from = true
                 ∧ s_has_usage sol from = true
                 ∧ lookup w value_slots' = Some x'
                 ∧ rebuilt_value x from x')
          value_slots
  | Value_partial_application _ _ args_arity =>
    ∃ (closure' : value) (args' : list value),
      v' = Value_partial_application closure' args' args_arity
  | Value_closure _ _ | Value_location _ => False
  end.

(* What the second static environment binds to a symbol that the first
   binds to [allocated] and that has a use. *)
Definition static_rebuilt (s : symbol) (allocated : statically_allocated_value)
    (found : option statically_allocated_value) : Prop :=
  match allocated with
  | Static_block tag fields =>
    ∃ fields' : list value,
      found = Some (Static_block tag fields')
      ∧ fields_rebuilt (field_rebuilt rebuilt_value (Node_symbol s)) 0 fields
          fields'
  | Static_boxed_number _ | Static_string _ | Static_imported _ =>
    found = Some allocated
  | Static_immutable_array _ =>
    ∃ elements' : list value, found = Some (Static_immutable_array elements')
  | Static_closure_of_set slot closure_symbols function_decls value_slots =>
    ∃ (sites : function_slot -> node)
      (value_slots' : list (value_slot * value)),
      found
      = Some
          (Static_closure_of_set slot closure_symbols function_decls
             value_slots')
      ∧ sites slot = Node_symbol s
      ∧ set_sites function_decls sites
      ∧ for_all_bindings
          (value_slot_rebuilt rebuilt_value function_decls sites
             value_slots')
          value_slots
  | Static_closure _ _ => False
  end.

(* The static environment of the second run binds every symbol that has a
   use as [static_rebuilt] says. *)
Definition statics_rebuilt (statics statics' : static_env) : Prop :=
  ∀ (s : symbol) (allocated : statically_allocated_value),
    lookup s (static_values statics) = Some allocated ->
    s_has_usage sol (Node_symbol s) = true ->
    static_rebuilt s allocated (lookup s (static_values statics')).

(* Every symbol bound is an allocation of its own
   (FlambdaReaperFlowSpec.static_value_flows). *)
Definition statics_allocated (statics : static_env) : Prop :=
  ∀ (s : symbol) (allocated : statically_allocated_value),
    lookup s (static_values statics) = Some allocated ->
    may_come_from sol (Node_symbol s) (Node_symbol s).

(* The static values of the fragment of the model: a static block holds
   values, and a static boxed number a number of a kind that can be
   boxed (as FlambdaWellKindedSpec.allocated_well_kinded asks of them).
   The model's static block has no shape: its traversal and its rebuild
   read every field as a value. *)
Definition statics_of_values (statics : static_env) : Prop :=
  ∀ (s : symbol) (allocated : statically_allocated_value),
    lookup s (static_values statics) = Some allocated ->
    match allocated with
    | Static_block _ fields =>
      ∀ v : value, v ∈ fields -> kind_of_value v = Value
    | Static_boxed_number contents =>
      boxable_number_of_kind (kind_of_value contents) <> None
    | Static_closure _ _ | Static_closure_of_set _ _ _ _ | Static_string _
    | Static_immutable_array _ | Static_imported _ =>
      True
    end.

(* ======================================================================== *)
(* What a metadata says                                                     *)
(* ======================================================================== *)

(* What a metadata says of an allocation that the value may come from. *)
Definition describes (m : rewrite_metadata) (a : node) : Prop :=
  match m with
  | No_source => False
  | At_least_one_source_no_usages => True
  | Single_source source => a = source
  | Many_sources_any_usage => s_any_usage sol a = true
  | Many_sources_usages usages =>
    s_any_usage sol a = true
    ∨ ∀ u : node,
        u ∈ usages -> s_usages sol a u = true ∧ s_has_usage sol u = true
  end.

(* The metadata describes every allocation that a value at [n] may come
   from. *)
Definition metadata_at (m : rewrite_metadata) (n : node) : Prop :=
  ∀ a : node,
    may_come_from sol a n -> s_has_source sol a = true -> describes m a.

(* The value [v] of the first run is rebuilt as [v'] under the metadata
   [m]: a constant, rebuilt as itself; or a value at a node where [m]
   holds, which has a use unless [m] forgets the type. *)
Definition rebuilt_under (m : rewrite_metadata) (v v' : value) : Prop :=
  (∃ c : const, v = Value_const c ∧ v' = v)
  ∨ ∃ n : node,
      rebuilt_value v n v' ∧ metadata_at m n
      ∧ (forgets m = false -> s_has_usage sol n = true).

End Rebuilt.

(* The sets of closures given to the rewriting are those of the graph:
   allocations that hold one another at the function slots of a set, as
   the closures of a set do ([set_sites]), are the closures of one of
   [sets], at the same function slots. *)
Definition sets_of_graph (G : graph) (sets : list set_of_closures) : Prop :=
  ∀ (function_decls : list (function_slot * code_id))
    (sites : function_slot -> node) (slot : function_slot) (id : code_id),
    lookup slot function_decls = Some id ->
    set_sites G function_decls sites ->
    ∃ set : set_of_closures,
      set ∈ sets
      ∧ (∀ (g : function_slot) (c : node) (code : Or_unknown.t code_id),
           (g, (c, code)) ∈ set ->
           is_bound g function_decls = true ∧ c = sites g)
      ∧ ∀ g : function_slot,
          is_bound g function_decls = true -> is_bound g set = true.

(* ======================================================================== *)
(* The copies of names                                                      *)
(* ======================================================================== *)

(* The copy that [canon] gives of a name, for any metadata, has in the
   second run the value that the name has in the first: that of every
   [Simple] whose canonical element the name is. *)
Definition copies_valued (tenv : Typing_env.t) (canon : canon_fn)
    (valuation valuation' : env) : Prop :=
  ∀ (s : simple) (x : name) (k : flambda_kind) (m : rewrite_metadata),
    Typing_env.get_canonical_simple_exn tenv (Some Name_mode.in_types) None s
    = Some (Some (simple_of_name x)) ->
    value_of_name valuation' (canon x k m) = eval_simple valuation s.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* A type of the fragment that holds of a value of the first run holds,
   rewritten under a metadata that the value is rebuilt under, of the
   value of the second run; with the fixes [fx]. *)
Definition rewrite_type_sound_with (fx : fixes) : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (statics statics' : static_env) (valuation valuation' : env)
    (fuel : nat) (m : rewrite_metadata) (t : flambda_type) (v v' : value),
    least_solution is_local F T ->
    is_local_slots_only is_local ->
    sets_of_graph (graph_of_facts F) sets ->
    typing_env_satisfied_strictly statics valuation tenv ->
    env_in_fragment valuation tenv ->
    statics_allocated (solution_of_tables T) statics ->
    statics_rebuilt (graph_of_facts F) (solution_of_tables T) statics
      statics' ->
    copies_valued tenv canon valuation valuation' ->
    type_in_fragment valuation t ->
    rebuilt_under (graph_of_facts F) (solution_of_tables T) m v v' ->
    type_satisfied statics valuation v t ->
    type_satisfied statics' valuation' v'
      (rewrite_arbitrary fx (plain_context is_local F T sets) tenv canon fuel
         m t).

Definition rewrite_type_sound_statement : Prop :=
  rewrite_type_sound_with all_fixes.

(* The type of a symbol, rewritten as the rewriting of the environment
   rewrites it, holds of the symbol in the second run. *)
Definition rewrite_symbol_type_sound_statement : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (statics statics' : static_env) (valuation valuation' : env)
    (s : symbol) (allocated : statically_allocated_value),
    least_solution is_local F T ->
    is_local_slots_only is_local ->
    sets_of_graph (graph_of_facts F) sets ->
    typing_env_satisfied_strictly statics valuation tenv ->
    env_in_fragment valuation tenv ->
    statics_allocated (solution_of_tables T) statics ->
    statics_rebuilt (graph_of_facts F) (solution_of_tables T) statics
      statics' ->
    copies_valued tenv canon valuation valuation' ->
    lookup s (static_values statics) = Some allocated ->
    type_satisfied statics' valuation' (Value_symbol s)
      (rewrite_symbol_type all_fixes (plain_context is_local F T sets) tenv
         canon s).

(* The type that the result types of code give a parameter or a result,
   [type_var], which holds of its value [v] in the first run (the
   environment of the result types has a type for it), rewritten
   under the metadata of the variable of the code [code_var] as
   [rewrite_result_types] rewrites it, holds of what the second run has
   at that variable; with the fixes [fx]. *)
Definition rewrite_result_type_sound_with (fx : fixes) : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (statics statics' : static_env) (valuation valuation' : env)
    (code_var type_var : variable) (v v' : value),
    least_solution is_local F T ->
    is_local_slots_only is_local ->
    sets_of_graph (graph_of_facts F) sets ->
    typing_env_satisfied_strictly statics valuation tenv ->
    env_in_fragment valuation tenv ->
    statics_allocated (solution_of_tables T) statics ->
    statics_rebuilt (graph_of_facts F) (solution_of_tables T) statics
      statics' ->
    copies_valued tenv canon valuation valuation' ->
    Typing_env.find tenv (Name_var type_var) None <> None ->
    lookup type_var valuation = Some v ->
    rebuilt_value (graph_of_facts F) (solution_of_tables T) v
      (Node_variable code_var) v' ->
    type_satisfied statics' valuation' v'
      (rewrite_result_type fx (plain_context is_local F T sets) tenv canon
         code_var type_var).

Definition rewrite_result_type_sound_statement : Prop :=
  rewrite_result_type_sound_with all_fixes.

(* The field [f] of a value: a field of a block, or the contents of a
   boxed number (as FlambdaReaperUnboxingRefinesProofs.field_value). *)
Definition field_of_value (f : field) (w : value) : option value :=
  match f, w with
  | Field_block i _, Value_block _ fields => nth_error fields i
  | Field_boxed_number _, Value_boxed_number c => Some (Value_const c)
  | _, _ => None
  end.

(* The value at a path of fields. *)
Fixpoint value_at_path (path : list field) (w : value) : option value :=
  match path with
  | [] => Some w
  | f :: path =>
    match field_of_value f w with
    | Some w1 => value_at_path path w1
    | None => None
    end
  end.

(* The values [ws] of the variables of the unboxed fields [fields] of a
   value [w]: in the order of the leaves of [fields], the value at the
   path of each, of its kind. It is
   FlambdaReaperUnboxingRefinesCallsSimProofs.fields_vals, how the
   simulation of the rebuild with unboxing relates a value of the
   rebuild without to the values of the variables of its fields. *)
Definition unboxed_values (fields : FlambdaReaperUnboxing.unboxed_fields)
    (w : value) (ws : list value) : Prop :=
  Forall2
    (fun (pk : list field * flambda_kind) (w' : value) =>
       value_at_path (fst pk) w = Some w' ∧ kind_of_value w' = snd pk)
    (FlambdaReaperUnboxingRefinesCallsSpec.fields_paths fields) ws.

(* An allocation holds the field of a block at a position at one kind,
   as those of the traversal do, which are all of kind value
   (FlambdaReaperTraversal.v). *)
Definition block_fields_of_one_kind (G : graph) : Prop :=
  ∀ (a from from' : node) (i : nat) (k k' : flambda_kind),
    g_constructor G a (Field_block i k) from ->
    g_constructor G a (Field_block i k') from' -> k = k'.

(* The types that the result types of code give the new parameters of an
   unboxed variable of code [code_var], one for each of its fields
   [fields], rewritten as [rewrite_result_types_u] rewrites them from the
   type of [type_var], hold of the values of the variables of the fields
   in the run of the unit rebuilt with unboxing, if the type of
   [type_var] held of the value [v] of the first run there. The unboxing
   is that of the analysis ([unit_unboxed_fields]), and no use of a name
   unboxed is called or has a function slot read from it
   ([check_unboxed_uses], of the fragment of
   FlambdaReaperUnboxingRefinesCallsSpec.v). Of the runs, it asks the
   value of the variable in the rebuild without unboxing, [v_mid],
   related to [v] as the simulation of the rebuild relates them
   ([rebuilt_value]), and the values of the variables of the fields
   in the rebuild with unboxing, [ws], related to [v_mid] as the
   simulation of the unboxing relates them ([unboxed_values]). *)
Definition rewrite_unboxed_result_type_sound_with (fx : fixes) : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables)
    (sets : list set_of_closures) (dominated : list (node * node))
    (to_unbox : list node) (e : expr) (tenv : Typing_env.t)
    (canon : canon_fn) (statics statics' : static_env)
    (valuation valuation' : env) (code_var type_var : variable)
    (fields : FlambdaReaperUnboxing.unboxed_fields) (v v_mid : value)
    (ws : list value),
    least_solution is_local F T ->
    is_local_slots_only is_local ->
    sets_of_graph (graph_of_facts F) sets ->
    block_fields_of_one_kind (graph_of_facts F) ->
    typing_env_satisfied_strictly statics valuation tenv ->
    env_in_fragment valuation tenv ->
    statics_allocated (solution_of_tables T) statics ->
    statics_rebuilt (graph_of_facts F) (solution_of_tables T) statics
      statics' ->
    copies_valued tenv canon valuation valuation' ->
    FlambdaReaperUnboxingRefinesCallsSpec.check_unboxed_uses F T
      (fst (FlambdaReaperUnboxing.unit_unboxed_fields F T dominated to_unbox
              e))
    = true ->
    Keyed.lookup node_eqb (Node_variable code_var)
      (fst (FlambdaReaperUnboxing.unit_unboxed_fields F T dominated to_unbox
              e))
    = Some fields ->
    lookup type_var valuation = Some v ->
    rebuilt_value (graph_of_facts F) (solution_of_tables T) v
      (Node_variable code_var) v_mid ->
    unboxed_values fields v_mid ws ->
    Forall2 (type_satisfied statics' valuation') ws
      (rewrite_unboxed_result_type fx (plain_context is_local F T sets) tenv
         (FlambdaReaperUnboxing.bound_code_my_closures e) canon code_var
         type_var fields).

Definition rewrite_unboxed_result_type_sound_statement : Prop :=
  rewrite_unboxed_result_type_sound_with all_fixes.

(* ======================================================================== *)
(* The two runs                                                             *)
(* ======================================================================== *)

(* What the statement of the two runs asks of the simulation of the
   rebuild (FlambdaReaperRebuildProofs.v, [related]): where the run of
   the unit [e] exits, from a state whose static environment is
   [statics], the run of the rebuilt unit [e'] exits through the same
   continuation, with the same heap, from a state whose static
   environment [statics'] [statics_rebuilt] relates to [statics]; and
   every symbol that [statics] binds is an allocation of its own
   ([statics_allocated], which FlambdaReaperFlowSpec.statics_flow
   gives). It holds under the premises of
   FlambdaReaperRebuildSpec.rebuild_refines_observably_statement, each
   variable bound once and the static blocks holding values where the
   unit exits ([rebuild_exit_relation_holds_statement]). Without the last
   it is false of a unit whose static block holds a value of another kind
   than [Value] ([naked_field_exit_relation_statement]); what it asks but
   [statics_rebuilt] holds under the first alone
   ([rebuild_exits_statement]). *)
Definition rebuild_exit_relation (G : graph) (sol : solution) (e e' : expr)
    : Prop :=
  ∀ (n : nat) (s : state) (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record),
    run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
    step s = Exit k vs hp rd ->
    statics_allocated sol (state_statics s)
    ∧ ∃ (m : nat) (t : state) (vs' : list value) (rd' : read_record),
        run m (State e' [] [] empty_static_env empty_heap no_reads) = Step t
        ∧ step t = Exit k vs' hp rd'
        ∧ statics_rebuilt G sol (state_statics s) (state_statics t).

(* Where the unit exits, its static values are of the fragment
   ([statics_of_values]): its static blocks hold values. *)
Definition exits_of_values (e : expr) : Prop :=
  ∀ (n : nat) (s : state) (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record),
    run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
    step s = Exit k vs hp rd ->
    statics_of_values (state_statics s).

(* The types that the reaper exports for the symbols of a unit hold of
   them where the rebuilt unit exits: where the unit exits from a state
   whose static environment and a valuation satisfy a typing environment
   of the fragment, the rebuilt unit exits through the same continuation,
   and the type of each symbol bound, rewritten, holds of the symbol
   there, given [rebuild_exit_relation]. *)
Definition exported_types_sound_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record) (sym : symbol)
    (allocated : statically_allocated_value),
    is_local_slots_only is_local ->
    least_solution is_local (traverse_unit flag u e) T ->
    sets_of_graph (graph_of_facts (traverse_unit flag u e)) sets ->
    rebuild_exit_relation (graph_of_facts (traverse_unit flag u e))
      (solution_of_tables T) e
      (rebuild_unit_from_tables (traverse_unit flag u e) T
         preserve_direct_calls e) ->
    run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
    step s = Exit k vs hp rd ->
    typing_env_satisfied_strictly (state_statics s) valuation tenv ->
    env_in_fragment valuation tenv ->
    copies_valued tenv canon valuation valuation' ->
    lookup sym (static_values (state_statics s)) = Some allocated ->
    ∃ (m : nat) (t : state) (vs' : list value) (rd' : read_record),
      run m
        (State
           (rebuild_unit_from_tables (traverse_unit flag u e) T
              preserve_direct_calls e)
           [] [] empty_static_env empty_heap no_reads)
      = Step t
      ∧ step t = Exit k vs' hp rd'
      ∧ type_satisfied (state_statics t) valuation' (Value_symbol sym)
          (rewrite_symbol_type all_fixes
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon sym).

(* ======================================================================== *)
(* The fixes that are needed                                                *)
(* ======================================================================== *)

(* A boxed number whose contents no use reads: the variable [x10] is
   bound to a boxed float, of which only the tag is read ([Is_int], into
   [x11], which has any usage). Its metadata is [Single_source x10]; the
   rebuild makes it poison ([rebuilt_value]: the field of the contents
   is not used), and the type [box_number] is no type of poison. The
   least solution of the graph, worked out by hand, passes the check of
   closure. *)
Module BoxedCounterexample.

Definition x10 : node := Node_variable 10%positive.
Definition x11 : node := Node_variable 11%positive.

Definition is_local (f : field) : bool := false.

Definition graph : facts :=
  Facts [] [] [(x11, Field_is_int, x10)]
    [(x10, Field_boxed_number Boxable_naked_float, Node_all_constants)]
    [] [] [] [] [x11] [Node_all_constants] [] [].

Definition solution : tables :=
  Tables [] [x11] [Node_all_constants] [x11; x10] [Node_all_constants; x10]
    [(x10, x10)] [(x10, x10)] [] [] [].

Definition metadata : rewrite_metadata := Single_source x10.

Definition boxed_type : flambda_type :=
  box_number Boxable_naked_float (unknown (Naked_number Kind_naked_float))
    Alloc_mode.Heap.

Definition boxed : value := Value_boxed_number (Naked_float 5%Z).

Definition rebuilt : value := Value_const (Poison Value).

End BoxedCounterexample.

(* Two fields that the type says are one symbol: the block [x10] holds
   the symbol [s50] in its fields 0 and 1, and has two uses, itself, of
   which the field 0 is read into [x12], and its alias [x11], of which
   the tag is read into [x13]; [x12] and [x13] have any usage. Field 0
   is used, field 1 is not, and the rebuild makes field 1 poison. Under
   the metadata [Many_sources_usages [x11]], whose uses read neither
   field, [follow_field] forgets the type of both, and the compiler
   rewrites the two aliases into aliases to one copy of [s50]: no value
   of that copy is both [s50] and poison. *)
Module AliasCounterexample.

Definition x10 : node := Node_variable 10%positive.
Definition x11 : node := Node_variable 11%positive.
Definition x12 : node := Node_variable 12%positive.
Definition x13 : node := Node_variable 13%positive.
Definition s50 : node := Node_symbol 50%positive.

Definition is_local (f : field) : bool := false.

Definition graph : facts :=
  Facts [(x11, x10)] []
    [(x12, Field_block 0 Value, x10); (x13, Field_is_int, x11)]
    [(x10, Field_block 0 Value, s50); (x10, Field_block 1 Value, s50)]
    [] [] [] [] [x12; x13] [s50] [] [].

Definition solution : tables :=
  Tables [(x11, x10); (x12, s50)] [x12; x13; s50] [s50; x12]
    [x12; x13; s50; x10; x11] [s50; x12; x10; x11]
    [(x10, x10); (x11, x11); (x10, x11)] [(x10, x10); (x11, x10)] [] []
    [(x10, Field_block 0 Value)].

Definition metadata : rewrite_metadata := Many_sources_usages [x11].

Definition block_type : flambda_type :=
  immutable_block false 0%Z (Scannable Value_only) Alloc_mode.Heap
    [alias_type_of Value (Symbol 50%positive);
     alias_type_of Value (Symbol 50%positive)].

Definition block : value :=
  Value_block 0%Z [Value_symbol 50%positive; Value_symbol 50%positive].

Definition rebuilt : value :=
  Value_block 0%Z [Value_symbol 50%positive; Value_const (Poison Value)].

End AliasCounterexample.

(* A closure that escapes and whose set the rewriting does not recover:
   two sets of closures of the unit have the function slot 70, of code
   20, one whose closure is [x10] and one whose closure is [x11] (two
   copies of a set, as inlining makes them). Their value slot 1 is local
   ([-reaper-local-fields]). [x10] escapes (any usage) and nothing reads
   its value slot, so the rebuild removes it; the value slot of [x11] is
   read into [x12], which has any usage. Under the metadata
   [Many_sources_usages [x11]], which [join_metadata] gives to the join of
   the two closures, [x10] being described by its any usage,
   [refine_sets_of_closures] recovers the sources of [x11] only and keeps
   the set of [x11]; the closure type of [x10] rewritten says that it has
   the value slot 1, which the rebuilt closure has not. The tables are
   the least solution. *)
Module RefineCounterexample.

Definition x10 : node := Node_variable 10%positive.
Definition x11 : node := Node_variable 11%positive.
Definition x12 : node := Node_variable 12%positive.

Definition is_local (f : field) : bool :=
  match f with
  | Field_value_slot _ => true
  | _ => false
  end.

Definition graph : facts :=
  Facts [] [] [(x12, Field_value_slot 1%positive, x11)]
    [(x10, Field_function_slot 70%positive, x10);
     (x10, Field_value_slot 1%positive, Node_all_constants);
     (x11, Field_function_slot 70%positive, x11);
     (x11, Field_value_slot 1%positive, Node_all_constants)]
    [] [] [] [] [x10; x12] [Node_all_constants] [] [].

Definition solution : tables :=
  Tables [(x12, Node_all_constants)] [Node_all_constants; x12; x10]
    [x12; Node_all_constants] [x11; Node_all_constants; x10; x12]
    [x12; Node_all_constants; x11; x10] [(x11, x11)] [(x11, x11); (x10, x10)]
    [] [(Field_value_slot 1%positive, Node_all_constants)]
    [(x11, Field_value_slot 1%positive);
     (x10, Field_function_slot 70%positive)].

Definition sets : list set_of_closures :=
  [[(70%positive, (x10, Or_unknown.Known 20%positive))];
   [(70%positive, (x11, Or_unknown.Known 20%positive))]].

Definition metadata : rewrite_metadata := Many_sources_usages [x11].

Definition closure_type : flambda_type :=
  at_least_this_closure 70%positive
    [(70%positive, Or_unknown.Known (Function_type 20%positive))] []
    [(1%positive, unknown Value)] Alloc_mode.Heap.

Definition closure : value :=
  Value_closure_of_set 70%positive [(70%positive, 20%positive)]
    [(1%positive, Value_const (Tagged_immediate 5%Z))].

Definition rebuilt : value :=
  Value_closure_of_set 70%positive [(70%positive, 20%positive)] [].

End RefineCounterexample.

(* Without the fix of boxed numbers, and without that of aliases, the
   first statement is false; so it is of the compiler. *)
(* The same box as the result of code: the result types give the
   parameter [x20] of the result types the type [box_number], and the
   code returns [x10]. *)
Module BoxedResultCounterexample.

Definition tenv : Typing_env.t :=
  Typing_env.replace_equation
    (Typing_env.add_variable_definition Typing_env.create 20%positive Value
       Name_mode.normal)
    (Name_var 20%positive) BoxedCounterexample.boxed_type.

Definition valuation : env := [(20%positive, BoxedCounterexample.boxed)].

End BoxedResultCounterexample.

Definition boxed_fix_needed_statement : Prop :=
  ~ rewrite_type_sound_with (Fixes false true true true).

Definition alias_fix_needed_statement : Prop :=
  ~ rewrite_type_sound_with (Fixes true false true true).

Definition compiler_rewrite_unsound_statement : Prop :=
  ~ rewrite_type_sound_with compiler_fixes.

Definition boxed_result_fix_needed_statement : Prop :=
  ~ rewrite_result_type_sound_with (Fixes false true true true).

(* Without the fix of [refine_sets_of_closures], the first statement is
   false. *)
Definition refine_fix_needed_statement : Prop :=
  ~ rewrite_type_sound_with (Fixes true true false true).

(* The closure type of that counterexample, rewritten: with the fixes,
   the type of a closure at the slot 70; without the fix of
   [refine_sets_of_closures], that of a closure at the slot 70 that has
   the value slot 1. *)
Definition refine_example_statement : Prop :=
  let context :=
    plain_context RefineCounterexample.is_local RefineCounterexample.graph
      RefineCounterexample.solution RefineCounterexample.sets
  in
  rewrite_arbitrary all_fixes context Typing_env.create (fun n _ _ => n) 5
    RefineCounterexample.metadata RefineCounterexample.closure_type
  = at_least_this_closure 70%positive [] [] [] Alloc_mode.Heap
  ∧ rewrite_arbitrary (Fixes true true false true) context Typing_env.create
      (fun n _ _ => n) 5 RefineCounterexample.metadata
      RefineCounterexample.closure_type
    = at_least_this_closure 70%positive [] []
        [(1%positive, unknown Value)] Alloc_mode.Heap.

(* ======================================================================== *)
(* Null keeps its types                                                     *)
(* ======================================================================== *)

(* [Null], a constant, is what every rebuild has for [Null]. So a type
   that [Null] has in the first run, rewritten under any metadata, is to
   be one that [Null] has in the second: of any type, closures included,
   with any solution and any context; with the fixes [fx]. *)
Definition rewrite_keeps_null_with (fx : fixes) : Prop :=
  ∀ (ctx : rewrite_context) (tenv : Typing_env.t) (canon : canon_fn)
    (statics statics' : static_env) (valuation valuation' : env)
    (fuel : nat) (m : rewrite_metadata) (t : flambda_type),
    typing_env_satisfied_strictly statics valuation tenv ->
    copies_valued tenv canon valuation valuation' ->
    type_satisfied statics valuation (Value_const Null) t ->
    type_satisfied statics' valuation' (Value_const Null)
      (rewrite_arbitrary fx ctx tenv canon fuel m t).

Definition rewrite_keeps_null_statement : Prop :=
  rewrite_keeps_null_with all_fixes.

(* A closure type that may be null: the closure at the function slot 70
   of a set that has at least that function, of code 20, or [Null]. The
   unit has one set of closures, whose closure at 70 is [x10]; under the
   metadata [Single_source x10] the rewriting of the compiler gives the
   type of a closure that is not null ([MTC.at_least_this_closure]). *)
Module NullCounterexample.

Definition x10 : node := Node_variable 10%positive.

Definition context : rewrite_context :=
  plain_context (fun _ => false) no_facts
    (Tables [] [] [] [] [] [] [] [] [] [])
    [[(70%positive, (x10, Or_unknown.Known 20%positive))]].

Definition closure_or_null : flambda_type :=
  Type_value
    (Ok
       (No_alias
          (Head_of_kind_value
             (Ok
                (Closures
                   (Row_like_for_closures
                      [(70%positive,
                        Row_like_case
                          (Closures_entry
                             [(70%positive,
                               Or_unknown.Known (Function_type 20%positive))]
                             [] [])
                          (Row_like_index
                             (Row_like_index_domain.At_least
                                (Set_of_closures_contents [70%positive] []))
                             tt)
                          empty_env_extension)])
                   Alloc_mode.Heap))
             (Maybe_null None)))).

Definition metadata : rewrite_metadata := Single_source x10.

End NullCounterexample.

(* Without the fix of the nullability of closure types, the statement is
   false: so it is of the compiler. *)
Definition null_fix_needed_statement : Prop :=
  ~ rewrite_keeps_null_with (Fixes true true true false).

(* ======================================================================== *)
(* An example                                                               *)
(* ======================================================================== *)

(* The unit

     let symbol s51 = { 3 } in
     let symbol s50 = { 7; s51 } in
     let x10 = s50.(0) in ...          (x10 has any usage)

   at its end: the environment has [s50] of the type of the blocks of
   two fields [= 7] and [= s51], and [s51] of those of one field [= 3].
   Only the field 0 of [s50] is read: [s51] has no use, and its type is
   forgotten; the field 1 of [s50] is not used, and its type is
   forgotten; the field 0 keeps its type. *)
Module RewriteExample.

Definition s50 : node := Node_symbol 50%positive.
Definition s51 : node := Node_symbol 51%positive.
Definition x10 : node := Node_variable 10%positive.

Definition is_local (f : field) : bool := false.

Definition graph : facts :=
  Facts [] [] [(x10, Field_block 0 Value, s50)]
    [(s50, Field_block 0 Value, Node_all_constants);
     (s50, Field_block 1 Value, s51);
     (s51, Field_block 0 Value, Node_all_constants)]
    [] [] [] [] [x10] [Node_all_constants] [] [].

Definition solution : tables :=
  Tables [(x10, Node_all_constants)] [x10; Node_all_constants]
    [Node_all_constants; x10] [x10; Node_all_constants; s50]
    [Node_all_constants; x10; s50; s51] [(s50, s50)] [(s50, s50); (s51, s51)]
    [] [] [(s50, Field_block 0 Value)].

Definition context : rewrite_context :=
  plain_context is_local graph solution [].

Definition type_s50 : flambda_type :=
  immutable_block false 0%Z (Scannable Value_only) Alloc_mode.Heap
    [this_tagged_immediate 7%Z; alias_type_of Value (Symbol 51%positive)].

Definition type_s51 : flambda_type :=
  immutable_block false 0%Z (Scannable Value_only) Alloc_mode.Heap
    [this_tagged_immediate 3%Z].

Definition tenv : Typing_env.t :=
  Typing_env.replace_equation
    (Typing_env.replace_equation
       (Typing_env.add_symbol_definition
          (Typing_env.add_symbol_definition Typing_env.create 51%positive)
          50%positive)
       (Name_symbol 51%positive) type_s51)
    (Name_symbol 50%positive) type_s50.

Definition rewritten_s50 : flambda_type :=
  immutable_block false 0%Z (Scannable Value_only) Alloc_mode.Heap
    [this_tagged_immediate 7%Z; unknown Value].

End RewriteExample.

(* The least solution passes the check of closure, and the rewriting
   gives [s50] the type of the blocks of two fields, [= 7] and unknown,
   and [s51] the unknown type. *)
Definition example_statement : Prop :=
  check_closed RewriteExample.is_local RewriteExample.graph
    RewriteExample.solution = true
  ∧ rewrite_typing_env all_fixes RewriteExample.context RewriteExample.tenv
    = [(Name_symbol 50%positive, RewriteExample.rewritten_s50);
       (Name_symbol 51%positive, unknown Value)].

(* The result types of code that returns the box of
   [BoxedCounterexample], kept: the new parameter [x21] is an alias to the
   copy of [x20], whose type is forgotten to that of a value that is not
   null; the compiler keeps the type of the box, which the rebuild makes
   poison. *)
Definition result_types_example_statement : Prop :=
  let context :=
    plain_context BoxedCounterexample.is_local BoxedCounterexample.graph
      BoxedCounterexample.solution []
  in
  rewrite_result_types all_fixes context BoxedResultCounterexample.tenv
    [(10%positive, Keep, 20%positive)] []
  = Rewritten_result_types [21%positive] [] [20%positive]
      [(Name_var 21%positive, alias_type_of Value (Var 20%positive));
       (Name_var 20%positive, any_non_null_value)]
  ∧ rr_equations
      (rewrite_result_types compiler_fixes context
         BoxedResultCounterexample.tenv [(10%positive, Keep, 20%positive)]
         [])
    = [(Name_var 21%positive, alias_type_of Value (Var 20%positive));
       (Name_var 20%positive, BoxedCounterexample.boxed_type)].

(* ======================================================================== *)
(* The exit relation and the rebuild                                        *)
(* ======================================================================== *)

(* [rebuild_exit_relation] does not follow from the premises of
   FlambdaReaperRebuildSpec.rebuild_refines_observably_statement, even
   with the least solution and each variable bound once: a field of a
   static block may hold a value of a kind other than [Value]
   (FlambdaSemanticsSpec.eval_static_const tests no kind, and neither
   [in_flow_fragment] nor [well_formed_unit] looks at one), the traversal
   gives every field of a block the kind value
   (FlambdaReaperTraversal.block_deps), and [field_rebuilt] asks for a
   constructor at the kind of the field. The symbol [s50] is bound to a
   block of a naked immediate, and the unit returns it:

     let symbol s50 = { #3 } in
     return s50

   The gap is the model's: a static block of the compiler has a shape,
   and its traversal gives each field the kind that the shape gives it
   (reaper/traverse.ml, [traverse_block_like_static_const], by
   [Static_const.block_field_kind]), as [field_rebuilt] reads it; the
   model's static block has no shape, the blocks of its fragment being
   blocks of values (FlambdaWellKindedSpec.v), and nothing makes a unit
   keep to that. So the statement needs a premise that the static blocks
   of the unit hold values: [exits_of_values], with which it holds
   ([rebuild_exit_relation_holds_statement]). The reading of
   [field_rebuilt] is the compiler's, at the kind of the field; at the
   kind value, as the model's traversal reads it, the relation would hold
   of this unit, and the statements of the rewriting would be false of a
   unit whose static block holds a naked number that nothing reads (the
   rebuild makes it poison of kind value, and the type of the field,
   rewritten at its own kind, does not hold of that). *)
Module NakedFieldExample.

Local Open Scope positive_scope.

Definition unit_expr : expr :=
  Let_static
    [Bind_symbol 50 (Static_const_block 0 [Const (Naked_immediate 3)])]
    (FlowExample.return_to 1 [Symbol 50]).

Definition facts : facts :=
  traverse_unit Never FlowExample.interface unit_expr.

Definition s50 : node := Node_symbol 50.
Definition returned : node := Node_made 1.
Definition raised : node := Node_made 2.
Definition constants : node := Node_all_constants.

Definition solution : tables :=
  {| t_alias := [(returned, s50)];
     t_any_usage := [constants; s50; raised; returned];
     t_any_source := [constants];
     t_has_usage := [constants; s50; returned; raised];
     t_has_source := [returned; s50; constants];
     t_usages := [];
     t_sources := [(returned, s50); (s50, s50)];
     t_reading_field := []; t_escaping_field := [];
     t_field_used :=
       [(s50, Field_get_tag); (s50, Field_is_int); (s50, Field_block 0 Value)]
  |}.

(* The state from which the unit exits. *)
Definition exit_state : state :=
  State (FlowExample.return_to 1 [Symbol 50]) [] []
    {| static_code := [];
       static_values := [(50, Static_block 0 [Value_const (Naked_immediate 3)])]
    |}
    empty_heap no_reads.

End NakedFieldExample.

(* The unit meets the premises of [rebuild_refines_observably_statement],
   with its least solution, binds each variable once, and exits from
   [exit_state] returning [s50]; [sets_of_graph] holds of no set, the graph
   having no constructor at a function slot. Still, whatever the rebuild,
   [rebuild_exit_relation] fails. *)
Definition naked_field_exit_relation_statement : Prop :=
  least_solution FlowExample.is_local NakedFieldExample.facts
    NakedFieldExample.solution
  ∧ is_local_slots_only FlowExample.is_local
  ∧ (∀ f : field, FlowExample.is_local f = false)
  ∧ in_flow_fragment NakedFieldExample.unit_expr = true
  ∧ FlambdaReaperRebuildSpec.any_kind_arguments NakedFieldExample.unit_expr
    = true
  ∧ FlambdaReaperRebuildSpec.results_kinded [] [] NakedFieldExample.unit_expr
    = true
  ∧ well_formed_unit FlowExample.interface NakedFieldExample.unit_expr = true
  ∧ FlambdaReaperRebuildSpec.exits_one_value FlowExample.interface
      NakedFieldExample.unit_expr
    = true
  ∧ variables_bound_once NakedFieldExample.unit_expr = true
  ∧ check_closed FlowExample.is_local NakedFieldExample.facts
      NakedFieldExample.solution
    = true
  ∧ check_field_used_exact FlowExample.is_local NakedFieldExample.facts
      NakedFieldExample.solution
    = true
  ∧ sets_of_graph (graph_of_facts NakedFieldExample.facts) []
  ∧ run 1 (State NakedFieldExample.unit_expr [] [] empty_static_env
             empty_heap no_reads)
    = Step NakedFieldExample.exit_state
  ∧ step NakedFieldExample.exit_state
    = Exit 1%positive [Value_symbol 50%positive] empty_heap no_reads
  ∧ ∀ preserve_direct_calls : bool,
      ~ rebuild_exit_relation (graph_of_facts NakedFieldExample.facts)
          (solution_of_tables NakedFieldExample.solution)
          NakedFieldExample.unit_expr
          (rebuild_unit_from_tables NakedFieldExample.facts
             NakedFieldExample.solution preserve_direct_calls
             NakedFieldExample.unit_expr).

(* What [rebuild_exit_relation] asks but [statics_rebuilt] holds under the
   premises of [rebuild_refines_observably_statement]: every symbol bound
   where the unit exits is an allocation of its own (from
   FlambdaReaperFlowSpec.flow_sound_statement), and the rebuilt unit exits
   through the same continuation with the same heap (from
   [rebuild_refines_observably_statement]). What is left is
   [statics_rebuilt] of the static environments of the two states that
   exit, which [naked_field_exit_relation_statement] makes false of some
   units, and [rebuild_exit_relation_holds_statement] gives, each
   variable bound once and the static blocks holding values. *)
Definition rebuild_exits_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) (n : nat) (s : state)
    (k : continuation) (vs : list value) (hp : heap) (rd : read_record),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
    step s = Exit k vs hp rd ->
    statics_allocated (solution_of_tables T) (state_statics s)
    ∧ ∃ (m : nat) (t : state) (vs' : list value) (rd' : read_record),
        run m
          (State
             (rebuild_unit_from_tables (traverse_unit flag u e) T
                preserve_direct_calls e)
             [] [] empty_static_env empty_heap no_reads)
        = Step t
        ∧ step t = Exit k vs' hp rd'.

(* [rebuild_exit_relation] holds under the premises of
   [rebuild_refines_observably_statement], if the unit binds each
   variable once (FlambdaReaperLeastSpec.variables_bound_once) and its
   static values are of the fragment where it exits ([exits_of_values]).
   The first gives the closures what [statics_rebuilt] asks of them: the
   allocations of the closures of a set, as the traversal makes them,
   hold at their function slots and their value slots the facts of their
   set and no other ([set_sites], [slot_held]). The second gives the
   fields of a static block the kind at which the traversal reads them,
   which [field_rebuilt] asks, and a static boxed number a number, which
   the rebuild keeps as it is; [naked_field_exit_relation_statement]
   shows it needed. *)
Definition rebuild_exit_relation_holds_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    variables_bound_once e = true ->
    exits_of_values e ->
    rebuild_exit_relation (graph_of_facts (traverse_unit flag u e))
      (solution_of_tables T) e
      (rebuild_unit_from_tables (traverse_unit flag u e) T
         preserve_direct_calls e).

(* [exported_types_sound_statement] with no premise on the runs but
   [exits_of_values]: for a unit of the premises of
   [rebuild_exit_relation_holds_statement] and the least solution of its
   graph (which passes the checks of the solution that the rebuild asks),
   where the unit exits from a state whose static environment and a
   valuation satisfy a typing environment of the fragment, the rebuilt
   unit exits through the same continuation, and the type of each symbol
   bound, rewritten, holds of the symbol there. *)
Definition exported_types_sound_end_to_end_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record) (sym : symbol)
    (allocated : statically_allocated_value),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    least_solution is_local (traverse_unit flag u e) T ->
    variables_bound_once e = true ->
    exits_of_values e ->
    sets_of_graph (graph_of_facts (traverse_unit flag u e)) sets ->
    run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
    step s = Exit k vs hp rd ->
    typing_env_satisfied_strictly (state_statics s) valuation tenv ->
    env_in_fragment valuation tenv ->
    copies_valued tenv canon valuation valuation' ->
    lookup sym (static_values (state_statics s)) = Some allocated ->
    ∃ (m : nat) (t : state) (vs' : list value) (rd' : read_record),
      run m
        (State
           (rebuild_unit_from_tables (traverse_unit flag u e) T
              preserve_direct_calls e)
           [] [] empty_static_env empty_heap no_reads)
      = Step t
      ∧ step t = Exit k vs' hp rd'
      ∧ type_satisfied (state_statics t) valuation' (Value_symbol sym)
          (rewrite_symbol_type all_fixes
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon sym).
