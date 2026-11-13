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
   value and of the static environment), the fragment
   ([closure_free_type], [type_in_fragment], [env_in_fragment]),
   [copies_valued], the statements, and the examples, against
   FlambdaReaperTypes.v (the rewriting, which needs no review),
   FlambdaReaperFlowSpec.v (where values may be) and
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
   immutable array, a closure or a partial application is one, of the
   same set and arity. The static environment of the second run binds
   each symbol that has a use as the first does, a block with its fields
   related as those of a block whose allocation is the symbol, a boxed
   number and a string as they are ([statics_rebuilt]). Two differences
   with [rel]: the fields of a block are taken at their kind (the
   traversal of the model gives every field of a block the kind value,
   [Field_block i Value], and [rel] makes an unused field poison of kind
   value); and a used field has a use at the node that holds it, which
   follows from [field_used_exact] where an allocation holds one node in
   each field, as those of the traversal do.

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

   The fragment ([type_in_fragment]): types without closure types, whose
   aliases are to names that hold, in the first run, a constant or a
   symbol ([unallocated]). The second excludes the aliases to a variable
   that holds a block: the machine has no identity of allocations, so an
   alias says that two values are equal, not that they are one
   allocation, and two positions of one block may be rebuilt
   differently. The rewriting gives the copy of a name, for a metadata,
   one name, which stands for every position that the name is reached at
   under the metadata ([copies_valued]: in the second run, the copy has
   the value that the name has in the first). For a constant or a symbol
   that is what the rebuild keeps at a position with a use.

   The statements:
   - [rewrite_type_sound_statement]: a type of the fragment that holds of
     a value of the first run, rewritten under a metadata that the value
     is rebuilt under, holds of the value of the second run, with every
     fix of FlambdaReaperTypes.v on.
   - [rewrite_symbol_type_sound_statement]: the type of a symbol of the
     environment, rewritten as [rewrite_typing_env] rewrites it, holds of
     the symbol in the second run, for every naming of the copies that
     [copies_valued] allows, if the symbol is bound in the first run and
     an allocation of its own there ([statics_allocated], which
     FlambdaReaperFlowSpec.statics_flow gives).
   - [boxed_fix_needed_statement] and [alias_fix_needed_statement]: the
     first statement is false without the fix of a boxed number whose
     contents have no use, and without that of the aliases under a
     metadata that forgets; [compiler_rewrite_unsound_statement]: so it
     is false of the compiler's rewriting. Each is a counterexample on a
     small graph whose least solution passes the checks of
     FlambdaReaperSolution.v. The program of the second, compiled from
     OCaml, gives a wrong type in the .cmx file: the type of a block says
     two fields are one name where the rebuild has kept the first and
     made the second poison.
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
     and a block, computed.

   What is left out:
   - Closures: the types of closures (and so the fault of
     [refine_sets_of_closures], which [fix_refine] fixes), and the values
     of closures beyond their shape; of the nullability of closure types
     only what [rewrite_keeps_null_statement] says.
   - The equations of the copies: the statements give each copy the
     value of the name it copies, and say nothing of its own equation.
     That equation is the type of the name rewritten under the metadata
     of the copy; it holds of that value where an alias reaches the copy
     in a case of a type that the value takes, and may not elsewhere (an
     alias in the case of another tag of a variant, whose fields have the
     metadata of the fields of the case taken); there the copy is to have
     no value.
   - The link with the runs: that the static environments of the unit
     and of its rebuild, where they exit, are related by
     [statics_rebuilt], and that every symbol bound is an allocation of
     its own ([statics_allocated]). The simulation of
     FlambdaReaperRebuildProofs.v relates them ([statics_rel],
     [flow_invariant]) at every state, where the second differs from
     [statics_rebuilt] as above.
   - The result types of code ([rewrite_result_types]), and the
     unboxing. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaPrimitiveSpec FlambdaSemanticsSpec
  FlambdaReaperGraphSpec FlambdaReaperSolution FlambdaReaperFlowSpec
  FlambdaReaperTypes.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaTypingEnv
  FlambdaTypingEnvSpec.
Import ListNotations.

(* ======================================================================== *)
(* The fragment                                                             *)
(* ======================================================================== *)

(* A type has no closure type, at any depth. *)
Fixpoint closure_free_type (t : flambda_type) : bool :=
  match t with
  | Type_value (Ok (No_alias (Head_of_kind_value (Ok head) _))) =>
    closure_free_non_null head
  | Type_value _ | Type_naked_immediate _ | Type_naked_number _ _
  | Type_rec_info _ | Type_region _ => true
  end

with closure_free_non_null (head : head_of_kind_value_non_null) : bool :=
  match head with
  | Variant _ immediates _ blocks extensions _ =>
    match immediates with
    | Or_unknown.Known ty => closure_free_type ty
    | Or_unknown.Unknown => true
    end
    && match blocks with
       | Or_unknown.Known blocks => closure_free_blocks blocks
       | Or_unknown.Unknown => true
       end
  | Boxed_number _ contents _ => closure_free_type contents
  | Closures _ _ => false
  | Mutable_block _ | String _ => true
  end

with closure_free_blocks (blocks : row_like_for_blocks) : bool :=
  match blocks with
  | Row_like_for_blocks known_tags other_tags _ =>
    forallb
      (fun p =>
         match snd p with
         | Or_unknown.Known case =>
           forallb closure_free_type (case_maps_to case)
         | Or_unknown.Unknown => true
         end)
      known_tags
    && match other_tags with
       | Or_bottom.Ok case => forallb closure_free_type (case_maps_to case)
       | Or_bottom.Bottom => true
       end
  end.

(* A value that is not allocated: a constant or a symbol. *)
Definition unallocated (v : value) : Prop :=
  (∃ c : const, v = Value_const c) ∨ (∃ s : symbol, v = Value_symbol s).

(* A type of the fragment: no closure type, and every alias, of a
   variable of a relation or of an extension too, to a name that holds a
   constant or a symbol in [valuation]. *)
Definition type_in_fragment (valuation : env) (t : flambda_type) : Prop :=
  closure_free_type t = true
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
  | Value_closure_of_set slot function_decls _ =>
    ∃ value_slots' : list (value_slot * value),
      v' = Value_closure_of_set slot function_decls value_slots'
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
  | Static_closure_of_set slot closure_symbols function_decls _ =>
    ∃ value_slots' : list (value_slot * value),
      found
      = Some
          (Static_closure_of_set slot closure_symbols function_decls
             value_slots')
  | Static_closure code _ =>
    ∃ value_slots' : list (value_slot * value),
      found = Some (Static_closure code value_slots')
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
    closed is_local (graph_of_facts F) (solution_of_tables T) ->
    is_local_slots_only is_local ->
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
    closed is_local (graph_of_facts F) (solution_of_tables T) ->
    is_local_slots_only is_local ->
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

(* Without the fix of boxed numbers, and without that of aliases, the
   first statement is false; so it is of the compiler. *)
Definition boxed_fix_needed_statement : Prop :=
  ~ rewrite_type_sound_with (Fixes false true true true).

Definition alias_fix_needed_statement : Prop :=
  ~ rewrite_type_sound_with (Fixes true false true true).

Definition compiler_rewrite_unsound_statement : Prop :=
  ~ rewrite_type_sound_with compiler_fixes.

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
