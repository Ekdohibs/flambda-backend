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

(* The reaper and the code age relation, on the machine of the core that
   is given one (FlambdaCodeAgeCallsSpec.step_aged): bug 12 of the
   report as a counterexample, what the reaper makes of a direct call of
   a closure of a newer version of the code that it names, and the two
   fixes that the report proposes.

   This file is to be reviewed: [refines_observably_aged],
   [unit_statics], [same_convention], [sound_conventions_for],
   [conventions_agree], [age_fixed_table], [cut_relation], the three
   units and the statements. The proofs are in
   FlambdaReaperCodeAgeProofs.v, which needs no review.

   The setting. The terms of the model do not say of which code their
   code is a newer version, and the reaper reads no relation. Nor does
   the compiler's: it copies the relation of the unit unchanged into the
   typing environment that it exports (types/traversals.ml,
   [with_code_age_relation]), keeps [newer_version_of] of the code that
   it rewrites, and changes the calling convention of each code on its
   own (unboxing_analysis.ml, [compute_code_changes]). So the relation
   here is the parameter [age_of] of FlambdaCodeAgeCallsSpec.v, one and
   the same for a unit and for its rebuild. [unit_statics] is the static
   environment that the first step of a unit that begins with its
   [Let_static] binds ([unit_statics_step_statement]): the code and the
   symbols that a client of the unit runs with. [sound_for]
   (FlambdaCodeAgeCallsSpec.v) is of a static environment and a closure;
   its first three conjuncts, of the codes alone, are
   [sound_conventions_for] ([sound_for_conventions_statement]), and
   [conventions_agree] says the same of every link between two codes that
   a unit binds, which gives [sound_conventions_for] of its statics
   ([unit_statics_codes_statement],
   [conventions_agree_statics_statement]).

   1. The bug ([conventions_counterexample_statement],
      [conventions_client_statement]). The reproducer of the report in
      the model: [f51] a newer version of [f50], the two one code, the
      closure of [f50] escaping, the static closure of [f51] called
      directly only. The unit meets the premises of the statement of the
      rebuild with changes of calling convention for the least solution,
      and its rebuild refines it, on [step] and on the machine. The
      relation is sound for the closure of [f51] in the unit; in the
      rebuild [f51] has lost a parameter and [f50] has not, the
      conventions do not agree, and the relation, unchanged, is not
      sound. A client that holds the closure of [f51] and calls it
      directly naming [f50] exits with the code of the unit, and also
      with [f51] named, as
      FlambdaCodeAgeCallsSpec.direct_call_newer_version_refines_statement
      allows; with the code of the rebuilt unit the call naming [f50]
      exits, and the call naming [f51], which Simplify makes of it when
      the type of the callee names [f51], is undefined.

   2. A direct call of a closure of a newer version within the unit
      ([aged_call_counterexample_statement],
      [aged_call_sound_example_statement]). The traversal reads the
      witness of a direct call off its callee, and with [Auto] links the
      call to the code that it names only where the callee has any
      source: the graph takes the call for a call of the code of the
      closure called, which on the machine it is only where that code is
      the code named. The rebuild then names the code that the flow finds
      (FlambdaReaperRebuild.rewrite_call_kind). Where a direct call names
      [f50] and its callee is a closure of [f51], the unit runs [f50] with
      that closure and the rebuilt unit runs [f51]: the replacement of
      direct_call_newer_version_refines_statement, made whatever the
      relation. With a relation that is not sound, neither rebuild, with
      or without changes of calling convention (and with the table of
      the fix), refines the unit on the machine, though the unit meets
      the premises of the statement of the rebuild for the least
      solution; that statement holds of it on [step], where the unit goes
      wrong. With a sound relation both rebuilds refine it on the
      machine; but the solution does not describe that run: [f50] has no
      use and its closure no source, and the run enters the body of
      [f50] with its closure bound to the closure of [f51]. What is
      proved of every unit on the machine is the statements of the
      rebuild for a unit that does not go wrong on [step]
      ([least_rebuild_refines_aged_statement],
      [least_rebuild_cc_auto_refines_aged_statement]), from
      [refines_aged_of_defined_statement] and
      [refines_observably_aged_of_defined_statement]: such a unit makes
      no direct call of a closure of another code than the code named,
      and its runs and those of its rebuild are those of [step]. The
      changes of representation of closures (FlambdaReaperUnboxing.v)
      are not looked at here.

   3. The fixes. [age_fixed_table] adds every code of the relation to
      the table of [cannot_change_calling_convention]. It passes the check
      where the table does ([age_fixed_table_check_statement]), so the
      statements of the rebuild hold with it
      ([rebuild_cc_fixed_refines_observably_statement],
      [least_rebuild_cc_fixed_refines_observably_statement], and on the
      machine [least_rebuild_cc_fixed_refines_aged_statement]). Where
      every code that a link relates is in the table
      ([age_fixed_table_links_statement]), the rebuild keeps the
      conventions of the linked codes
      ([rebuild_cc_keeps_conventions_statement]), so that the statics of
      the rebuilt unit meet [sound_conventions_for] for every closure if
      the conventions of the unit agree
      ([rebuild_cc_fixed_sound_conventions_statement]). [cut_relation]
      keeps the links between codes of the table, and the conventions
      agree for it in the rebuild, whatever the table
      ([rebuild_cc_cut_keeps_conventions_statement]). Both fixes repair
      the unit of the bug ([conventions_fixed_statement]).
      What neither keeps is the rest of [sound_for], the refinement of
      the older code by the newer for the closure.
      [sound_for_not_kept_statement] is a unit in which no code changes
      its calling convention, whose relation is sound for a closure in
      the unit and not in its rebuild: the older code is undefined with
      that closure, which has not the value slot that it reads into a
      variable of no use; the rebuild removes the read, and the older
      code then returns what the newer does not. The rebuild refines
      each code for the runs of the unit, and not for every call that
      [sound_for] speaks of.

   Not reached: a statement of the rebuild on the machine for a unit
   that makes a direct call of a closure of a newer version of the code
   named, under a relation sound for each closure so called. The proof
   of the rebuild rests on the invariant of the flow, which is of the
   runs of [step] (FlambdaReaperFlowSpec.v) and does not hold of such a
   run ([aged_call_sound_example_statement]). It would need that
   invariant of the run in which each such call goes by the code of its
   closure, and that this run refines the run of the machine, from the
   soundness of the relation at each call; neither is done. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperCallingConventionsSpec
  FlambdaReaperLeastSpec FlambdaCodeAgeCallsSpec.
From OxCamlFlambda2 Require FlambdaReaperLeast.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaCodeAgeRelation FlambdaCodeAgeRelationSpec.
Import ListNotations.

(* ======================================================================== *)
(* Refinement up to observation, on the machine                             *)
(* ======================================================================== *)

(* FlambdaReaperRebuildSpec.refines_observably, word for word, with
   [behaves_aged age_of] in the place of [behaves]. *)
Definition refines_observably_aged (age_of : static_env -> Code_age_relation.t)
    (source target : state) : Prop :=
  (∀ (k : continuation) (vs : list value) (hp : heap),
     behaves_aged age_of source (Exits k vs hp) ->
     ∃ vs' : list value,
       behaves_aged age_of target (Exits k vs' hp)
       ∧ values_related observed_equal vs vs')
  ∧ (∀ (k : continuation) (vs' : list value) (hp : heap),
       behaves_aged age_of target (Exits k vs' hp) ->
       (∃ vs : list value,
          behaves_aged age_of source (Exits k vs hp)
          ∧ values_related observed_equal vs vs')
       ∨ behaves_aged age_of source Goes_wrong)
  ∧ (behaves_aged age_of target Goes_wrong ->
     behaves_aged age_of source Goes_wrong).

Definition program_refines_observably_aged
    (age_of : static_env -> Code_age_relation.t) (source target : expr)
    : Prop :=
  refines_observably_aged age_of
    (State source [] [] empty_static_env empty_heap no_reads)
    (State target [] [] empty_static_env empty_heap no_reads).

(* A refinement on [step] whose source does not go wrong on [step] holds
   on the machine, whatever the relation. *)
Definition refines_aged_of_defined_statement : Prop :=
  ∀ (age_of : static_env -> Code_age_relation.t) (source target : state),
    refines source target ->
    ~ behaves source Goes_wrong ->
    refines_aged age_of source target.

Definition refines_observably_aged_of_defined_statement : Prop :=
  ∀ (age_of : static_env -> Code_age_relation.t) (source target : state),
    refines_observably source target ->
    ~ behaves source Goes_wrong ->
    refines_observably_aged age_of source target.

(* FlambdaReaperLeastSpec.least_rebuild_refines_observably_statement and
   least_rebuild_cc_auto_refines_observably_statement on the machine, of
   a unit that does not go wrong on [step]. *)
Definition least_rebuild_refines_aged_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    least_solution is_local (traverse_unit flag u e) T ->
    ~ behaves (State e [] [] empty_static_env empty_heap no_reads)
        Goes_wrong ->
    program_refines_observably_aged age_of e
      (rebuild_unit_from_tables (traverse_unit flag u e) T
         preserve_direct_calls e).

Definition least_rebuild_cc_auto_refines_aged_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    least_solution is_local (traverse_unit Auto u e) T ->
    ~ behaves (State e [] [] empty_static_env empty_heap no_reads)
        Goes_wrong ->
    program_refines_observably_aged age_of e
      (rebuild_unit_cc_from_tables (traverse_unit Auto u e) T
         (derive_cannot_change (traverse_unit Auto u e) T)
         (unit_code_deps Auto u e) preserve_direct_calls e).

(* ======================================================================== *)
(* The code of a unit and its conventions                                   *)
(* ======================================================================== *)

(* The static environment that the first step of a unit binds, where the
   unit begins with a [Let_static]: its code and its symbols. The empty
   one otherwise, or where the bindings are undefined. *)
Definition unit_statics (e : expr) : static_env :=
  match e with
  | Let_static bindings _ =>
    match eval_static_bindings [] bindings empty_static_env with
    | Some statics => statics
    | None => empty_static_env
    end
  | _ => empty_static_env
  end.

Definition unit_statics_step_statement : Prop :=
  ∀ (bindings : list static_binding) (body : expr) (statics : static_env),
    eval_static_bindings [] bindings empty_static_env = Some statics ->
    unit_statics (Let_static bindings body) = statics
    ∧ step
        (State (Let_static bindings body) [] [] empty_static_env empty_heap
           no_reads)
      = Step (State body [] [] statics empty_heap no_reads).

(* The code that it binds is code that the unit binds. *)
Definition unit_statics_codes_statement : Prop :=
  ∀ (e : expr) (id : code_id) (c : code),
    lookup id (static_code (unit_statics e)) = Some c ->
    (id, c) ∈ unit_codes e.

(* The first three conjuncts of [code_refines_for]: [c'] takes its
   arguments and gives its results as [c] does. *)
Definition same_convention (c c' : code) : Prop :=
  code_params_arity c' = code_params_arity c
  ∧ code_result_arity c' = code_result_arity c
  ∧ code_is_tupled c' = code_is_tupled c.

(* The part of [sound_for] that is of the codes alone: each link above
   the code of the closure [v] is between codes of the same
   convention. *)
Definition sound_conventions_for (age_of : static_env -> Code_age_relation.t)
    (statics : static_env) (v : value) : Prop :=
  ∀ (newer older : code_id) (c c' : code),
    closure_at_least (age_of statics) statics v newer ->
    Code_age_relation.get_older_version_of (age_of statics) newer
    = Some older ->
    lookup older (static_code statics) = Some c ->
    lookup newer (static_code statics) = Some c' ->
    same_convention c c'.

(* Every two codes of [codes] that a link of [rel] relates have the same
   convention. *)
Definition conventions_agree (rel : Code_age_relation.t)
    (codes : list (code_id * code)) : Prop :=
  ∀ (newer older : code_id) (c c' : code),
    Code_age_relation.get_older_version_of rel newer = Some older ->
    (older, c) ∈ codes ->
    (newer, c') ∈ codes ->
    same_convention c c'.

Definition sound_for_conventions_statement : Prop :=
  ∀ (age_of : static_env -> Code_age_relation.t) (statics : static_env)
    (v : value),
    sound_for age_of statics v -> sound_conventions_for age_of statics v.

(* What a unit binds gives the statics of its first step. *)
Definition conventions_agree_statics_statement : Prop :=
  ∀ (age_of : static_env -> Code_age_relation.t) (e : expr) (v : value),
    conventions_agree (age_of (unit_statics e)) (unit_codes e) ->
    sound_conventions_for age_of (unit_statics e) v.

(* ======================================================================== *)
(* The fixes                                                                *)
(* ======================================================================== *)

(* The first fix of the report, as a table of
   [cannot_change_calling_convention]: every code id of the relation,
   newer or older, is added to it. *)
Definition age_fixed_table (rel : Code_age_relation.t)
    (cannot_change : list node) : list node :=
  cannot_change
  ++ flat_map
       (fun '(newer, older) => [Node_code_id newer; Node_code_id older])
       rel.

(* The second: the relation cut where code changes. A pair of [rel] is
   kept if it is the link of its newer code ([get_older_version_of]
   finds it) and [keep] holds of both its codes. *)
Definition cut_relation (keep : code_id -> bool) (rel : Code_age_relation.t)
    : Code_age_relation.t :=
  filter
    (fun '(newer, older) =>
       keep newer && keep older
       && match Code_age_relation.get_older_version_of rel newer with
          | Some older' => Pos.eqb older' older
          | None => false
          end)
    rel.

(* Where no code that a link relates is in the table, the rebuild keeps
   the conventions of the linked codes that it binds. *)
Definition rebuild_cc_keeps_conventions_statement : Prop :=
  ∀ (types : subkinds_rewriting) (F : facts) (sol : solution)
    (cannot_change : node -> bool) (code_deps : finite_map code_dep)
    (preserve_direct_calls : bool) (e : expr) (rel : Code_age_relation.t),
    (∀ newer older : code_id,
       Code_age_relation.get_older_version_of rel newer = Some older ->
       cannot_change (Node_code_id newer) = true
       ∧ cannot_change (Node_code_id older) = true) ->
    conventions_agree rel (unit_codes e) ->
    conventions_agree rel
      (unit_codes
         (rebuild_unit_cc_with types F sol cannot_change code_deps
            preserve_direct_calls e)).

(* The relation cut at the code that changes, for any table. *)
Definition rebuild_cc_cut_keeps_conventions_statement : Prop :=
  ∀ (types : subkinds_rewriting) (F : facts) (sol : solution)
    (cannot_change : node -> bool) (code_deps : finite_map code_dep)
    (preserve_direct_calls : bool) (e : expr) (rel : Code_age_relation.t),
    conventions_agree rel (unit_codes e) ->
    conventions_agree
      (cut_relation (fun id => cannot_change (Node_code_id id)) rel)
      (unit_codes
         (rebuild_unit_cc_with types F sol cannot_change code_deps
            preserve_direct_calls e)).

Definition age_fixed_table_links_statement : Prop :=
  ∀ (rel : Code_age_relation.t) (cannot_change : list node)
    (newer older : code_id),
    Code_age_relation.get_older_version_of rel newer = Some older ->
    mem_node (Node_code_id newer) (age_fixed_table rel cannot_change) = true
    ∧ mem_node (Node_code_id older) (age_fixed_table rel cannot_change)
      = true.

(* The table of the fix passes the check where the table does. *)
Definition age_fixed_table_check_statement : Prop :=
  ∀ (F : facts) (T : tables) (cannot_change : list node)
    (rel : Code_age_relation.t),
    check_cc_closed F T cannot_change = true ->
    check_cc_closed F T (age_fixed_table rel cannot_change) = true.

(* With the table of the fix, the statics of the first step of the
   rebuilt unit meet [sound_conventions_for] for every closure, the
   relation being [rel]. *)
Definition rebuild_cc_fixed_sound_conventions_statement : Prop :=
  ∀ (F : facts) (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (preserve_direct_calls : bool)
    (e : expr) (rel : Code_age_relation.t)
    (age_of : static_env -> Code_age_relation.t) (v : value),
    let e' :=
      rebuild_unit_cc_from_tables F T (age_fixed_table rel cannot_change)
        code_deps preserve_direct_calls e
    in
    conventions_agree rel (unit_codes e) ->
    age_of (unit_statics e') = rel ->
    sound_conventions_for age_of (unit_statics e') v.

(* The rebuild with the table of the fix refines the unit, as
   FlambdaReaperCallingConventionsSpec.rebuild_cc_refines_observably_statement
   says of the table. *)
Definition rebuild_cc_fixed_refines_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (preserve_direct_calls : bool) (rel : Code_age_relation.t),
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
      (rebuild_unit_cc_from_tables (traverse_unit flag u e) T
         (age_fixed_table rel cannot_change) (unit_code_deps flag u e)
         preserve_direct_calls e).

(* The same for the least solution and its table, and on the machine for
   a unit that does not go wrong on [step]. *)
Definition least_rebuild_cc_fixed_refines_observably_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) (rel : Code_age_relation.t),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    least_solution is_local (traverse_unit Auto u e) T ->
    program_refines_observably e
      (rebuild_unit_cc_from_tables (traverse_unit Auto u e) T
         (age_fixed_table rel (derive_cannot_change (traverse_unit Auto u e) T))
         (unit_code_deps Auto u e) preserve_direct_calls e).

Definition least_rebuild_cc_fixed_refines_aged_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) (rel : Code_age_relation.t)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    least_solution is_local (traverse_unit Auto u e) T ->
    ~ behaves (State e [] [] empty_static_env empty_heap no_reads)
        Goes_wrong ->
    program_refines_observably_aged age_of e
      (rebuild_unit_cc_from_tables (traverse_unit Auto u e) T
         (age_fixed_table rel (derive_cannot_change (traverse_unit Auto u e) T))
         (unit_code_deps Auto u e) preserve_direct_calls e).

(* ======================================================================== *)
(* The bug: code that the relation links changes on its own                 *)
(* ======================================================================== *)

(* Two codes of two parameters, the code 51 a newer version of the code
   50, the reproducer of the report (relation_mixed_conventions.ml) in
   the model:

     let code f50 x40 x42 my_closure x41 return k3 exn k4 = k3 x40 in
     let code f51 x43 x45 my_closure x44 return k5 exn k6 = k5 x43 in
     let symbol s80 = closure at 70 of { 70 = f51 } in
     let x60 = closure at 71 of { 71 = f50 } in
     let_cont k20 x30 = let x36 = [x30; x60] in return x36 in
     s80 9 7 return k20 exn k2                    (direct, to f51)

   The closure [x60] of [f50] escapes, and [f50] keeps its calling
   convention; [s80] is called by the direct call only, and [f51] loses
   [x45], which it does not use. A client of the unit that holds [s80]
   calls it directly naming [f50], with two arguments, as
   FlambdaCodeAgeCallsSpec.step_aged allows ([client]). *)
Module ConventionsExample.

Local Open Scope positive_scope.
Import FlambdaReaperCallingConventions.Examples.

Definition older : code :=
  Code [40; 42] 41 3 4
    (make_code_metadata [Singleton Value; Singleton Value] [Singleton Value])
    (return_to 3 [Var 40]).

Definition newer : code :=
  Code [43; 45] 44 5 6
    (make_code_metadata [Singleton Value; Singleton Value] [Singleton Value])
    (return_to 5 [Var 43]).

Definition unit_expr : expr :=
  Let_static
    [Bind_code 50 older; Bind_code 51 newer;
     Bind_symbol 80 (Static_const_closure_of_set 70 [(70, 80)] [(70, 51)] [])]
    (Let 60 Value (Closure_of_set 71 [(71, 50)] [])
       (Let_cont_non_recursive 20
          (Handler [30] (make_handler_metadata [Value])
             (Let 36 Value (make_block [Var 30; Var 60])
                (return_to 1 [Var 36])))
          (Apply
             {| ap_callee := Symbol 80; ap_call_kind := Direct 51;
                ap_args :=
                  [Const (Tagged_immediate 9); Const (Tagged_immediate 7)];
                ap_continuation := 20; ap_exn_continuation := 2;
                ap_exn_extra_args := [];
                ap_args_arity := [Singleton Value; Singleton Value];
                ap_return_arity := [Singleton Value] |}))).

(* [f51] is a newer version of [f50]. *)
Definition relation : Code_age_relation.t :=
  Code_age_relation.add Code_age_relation.empty 51 50.

Definition age_of (_ : static_env) : Code_age_relation.t := relation.

Definition facts : facts := traverse_unit Auto interface unit_expr.

(* The least solution, as FlambdaReaperLeast.v computes it. *)
Definition tables : tables :=
  match FlambdaReaperLeast.compute_least no_local facts 40 with
  | Some T => T
  | None => FlambdaReaperLeast.empty_tables
  end.

(* The least table of [cannot_change_calling_convention], and that of
   the fix. *)
Definition table : list node := derive_cannot_change facts tables.

Definition fixed : list node := age_fixed_table relation table.

Definition code_deps : finite_map code_dep :=
  unit_code_deps Auto interface unit_expr.

Definition rebuilt : expr :=
  rebuild_unit_cc_from_tables facts tables table code_deps true unit_expr.

Definition rebuilt_fixed : expr :=
  rebuild_unit_cc_from_tables facts tables fixed code_deps true unit_expr.

(* What the rebuild makes of [f51]. *)
Definition newer_rebuilt : code :=
  Code [43] 44 5 6
    {| cm_params_arity :=
         [Unboxed_product [Singleton Value]; Unboxed_product []];
       cm_params_subkinds := [];
       cm_result_arity := [Unboxed_product [Singleton Value]];
       cm_result_subkinds := [];
       cm_is_tupled := false;
       cm_is_stub := false |}
    (return_to 5 [Var 43]).

(* A client of a unit whose first step binds [statics]: it calls [s80]
   directly, naming the code [id], with 5 and 7 at the arity of [f50],
   and returns to the continuation 1. *)
Definition client (statics : static_env) (id : code_id) : state :=
  State
    (Apply
       {| ap_callee := Symbol 80; ap_call_kind := Direct id;
          ap_args := [Const (Tagged_immediate 5); Const (Tagged_immediate 7)];
          ap_continuation := 1; ap_exn_continuation := 2;
          ap_exn_extra_args := [];
          ap_args_arity := [Singleton Value; Singleton Value];
          ap_return_arity := [Singleton Value] |})
    [] [(1, Outer 1); (2, Outer 2)] statics empty_heap no_reads.

End ConventionsExample.

(* The unit meets the premises of
   FlambdaReaperLeastSpec.least_rebuild_cc_auto_refines_observably_statement,
   and does not go wrong: its rebuild refines it, on [step] and on the
   machine whatever the relation. [f50] is in the least table, [f51] is
   not and loses its second parameter. The relation is sound for [s80]
   in the unit, where the two codes are one code, and the conventions of
   the codes it links agree; in the rebuilt unit, with the same relation,
   they do not, and the relation is not sound for [s80]. *)
Definition conventions_counterexample_statement : Prop :=
  let e := ConventionsExample.unit_expr in
  let F := ConventionsExample.facts in
  let T := ConventionsExample.tables in
  let rel := ConventionsExample.relation in
  let e' := ConventionsExample.rebuilt in
  least_solution Examples.no_local F T
  ∧ in_flow_fragment e = true ∧ any_kind_arguments e = true
  ∧ results_kinded [] [] e = true
  ∧ well_formed_unit Examples.interface e = true
  ∧ exits_one_value Examples.interface e = true
  ∧ ~ behaves (State e [] [] empty_static_env empty_heap no_reads) Goes_wrong
  ∧ ConventionsExample.table = repeat (Node_code_id 50%positive) 5
  ∧ unit_calling_convention_changes T ConventionsExample.table
      ConventionsExample.code_deps e
    = [(50%positive, Not_changing_calling_convention);
       (51%positive,
        Changing_calling_convention [Keep Value; Delete] [Keep Value])]
  ∧ program_refines_observably e e'
  ∧ (∀ age_of : static_env -> Code_age_relation.t,
       program_refines_observably_aged age_of e e')
  ∧ conventions_agree rel (unit_codes e)
  ∧ sound_for ConventionsExample.age_of (unit_statics e)
      (Value_symbol 80%positive)
  ∧ lookup 50%positive (static_code (unit_statics e'))
    = Some ConventionsExample.older
  ∧ lookup 51%positive (static_code (unit_statics e'))
    = Some ConventionsExample.newer_rebuilt
  ∧ ~ conventions_agree rel (unit_codes e')
  ∧ ~ sound_for ConventionsExample.age_of (unit_statics e')
        (Value_symbol 80%positive).

(* The call that the client makes, naming [f50], of [s80], a closure of
   [f51]. With the code of the unit it exits with 5, and so does the
   call naming [f51], which refines it, as
   FlambdaCodeAgeCallsSpec.direct_call_newer_version_refines_statement
   says. With the code of the rebuilt unit, the call naming [f50] exits
   with 5 (on [step] it is undefined, the callee not being a closure of
   [f50]), and the call naming [f51] is undefined: [f51] takes one
   argument now. It does not refine the first. *)
Definition conventions_client_statement : Prop :=
  let age_of := ConventionsExample.age_of in
  let before := unit_statics ConventionsExample.unit_expr in
  let after := unit_statics ConventionsExample.rebuilt in
  let five :=
    Exit 1%positive [Value_const (Tagged_immediate 5)] empty_heap no_reads
  in
  run_aged age_of 2 (ConventionsExample.client before 50%positive) = five
  ∧ run_aged age_of 2 (ConventionsExample.client before 51%positive) = five
  ∧ refines_aged age_of (ConventionsExample.client before 50%positive)
      (ConventionsExample.client before 51%positive)
  ∧ step (ConventionsExample.client after 50%positive) = Undefined
  ∧ run_aged age_of 2 (ConventionsExample.client after 50%positive) = five
  ∧ run_aged age_of 1 (ConventionsExample.client after 51%positive)
    = Undefined
  ∧ ~ refines_aged age_of (ConventionsExample.client after 50%positive)
        (ConventionsExample.client after 51%positive).

(* The two fixes on the unit. With the table of the fix, which passes the
   check, no code changes, the rebuild refines the unit, the conventions
   agree, the relation is sound for [s80] in the rebuilt unit, and the
   call of the client naming [f51] refines the call naming [f50]. With
   the least table and the relation cut where code changes, nothing is
   left of the relation, which says nothing. *)
Definition conventions_fixed_statement : Prop :=
  let e := ConventionsExample.unit_expr in
  let F := ConventionsExample.facts in
  let T := ConventionsExample.tables in
  let rel := ConventionsExample.relation in
  let e' := ConventionsExample.rebuilt_fixed in
  let age_of := ConventionsExample.age_of in
  check_cc_closed F T ConventionsExample.fixed = true
  ∧ unit_calling_convention_changes T ConventionsExample.fixed
      ConventionsExample.code_deps e
    = [(50%positive, Not_changing_calling_convention);
       (51%positive, Not_changing_calling_convention)]
  ∧ program_refines_observably e e'
  ∧ conventions_agree rel (unit_codes e')
  ∧ sound_for age_of (unit_statics e') (Value_symbol 80%positive)
  ∧ refines_aged age_of
      (ConventionsExample.client (unit_statics e') 50%positive)
      (ConventionsExample.client (unit_statics e') 51%positive)
  ∧ cut_relation
      (fun id => mem_node (Node_code_id id) ConventionsExample.table) rel
    = [].

(* ======================================================================== *)
(* A direct call of a closure of a newer version, within the unit           *)
(* ======================================================================== *)

(* A unit that calls the static closure [s80] of the code 51 directly,
   naming the code 50, of which 51 is a newer version:

     let code f50 x40 my_closure x41 return k3 exn k4 = k3 ra in
     let code f51 x43 my_closure x44 return k5 exn k6 = k5 rb in
     let symbol s80 = closure at 70 of { 70 = f51 } in
     let_cont k20 x30 = return x30 in
     s80 9 return k20 exn k2                      (direct, to f50)

   On [step] the call is undefined; on the machine it runs [f50] with
   [s80] for its closure. The traversal reads the witness of the call off
   [s80] and finds that of [f51], and links the call to no other code
   ([Auto], the callee having no source but [s80]): [f50] has no use,
   its closure [x41] no source. The rebuild deletes [f50] and makes the
   call name [f51], the code that the flow finds
   (FlambdaReaperRebuild.rewrite_call_kind). *)
Module AgedCallExample.

Local Open Scope positive_scope.
Import FlambdaReaperCallingConventions.Examples.

(* Code of one parameter [x] that returns [result]. *)
Definition code_returning (x my_closure k k_exn : variable) (result : simple)
    : code :=
  Code [x] my_closure k k_exn
    (make_code_metadata [Singleton Value] [Singleton Value])
    (return_to k [result]).

Definition unit_with (ra rb : simple) : expr :=
  Let_static
    [Bind_code 50 (code_returning 40 41 3 4 ra);
     Bind_code 51 (code_returning 43 44 5 6 rb);
     Bind_symbol 80 (Static_const_closure_of_set 70 [(70, 80)] [(70, 51)] [])]
    (Let_cont_non_recursive 20
       (Handler [30] (make_handler_metadata [Value]) (return_to 1 [Var 30]))
       (Apply
          {| ap_callee := Symbol 80; ap_call_kind := Direct 50;
             ap_args := [Const (Tagged_immediate 9)];
             ap_continuation := 20; ap_exn_continuation := 2;
             ap_exn_extra_args := [];
             ap_args_arity := [Singleton Value];
             ap_return_arity := [Singleton Value] |})).

(* [f50] returns 0 and [f51] returns 1: the relation is not sound. *)
Definition unsound_unit : expr :=
  unit_with (Const (Tagged_immediate 0)) (Const (Tagged_immediate 1)).

(* Both return their argument: it is. *)
Definition sound_unit : expr := unit_with (Var 40) (Var 43).

Definition relation : Code_age_relation.t :=
  Code_age_relation.add Code_age_relation.empty 51 50.

Definition age_of (_ : static_env) : Code_age_relation.t := relation.

Definition facts_of (e : expr) : facts := traverse_unit Auto interface e.

Definition tables_of (e : expr) : tables :=
  match FlambdaReaperLeast.compute_least no_local (facts_of e) 40 with
  | Some T => T
  | None => FlambdaReaperLeast.empty_tables
  end.

(* The rebuild without changes of calling convention, and with them and
   the table of the fix. *)
Definition rebuilt (e : expr) : expr :=
  rebuild_unit_from_tables (facts_of e) (tables_of e) true e.

Definition rebuilt_fixed (e : expr) : expr :=
  rebuild_unit_cc_from_tables (facts_of e) (tables_of e)
    (age_fixed_table relation
       (derive_cannot_change (facts_of e) (tables_of e)))
    (unit_code_deps Auto interface e) true e.

Definition initial (e : expr) : state :=
  State e [] [] empty_static_env empty_heap no_reads.

End AgedCallExample.

(* The unit with [f50] returning 0 and [f51] returning 1 meets the
   premises of
   FlambdaReaperLeastSpec.least_rebuild_refines_observably_statement,
   and goes wrong on [step]: the statement holds of it and says nothing.
   On the machine it exits with 0, and both rebuilds exit with 1: neither
   refines it on the machine. The relation is not sound for [s80], and
   the rebuild makes the replacement that
   FlambdaCodeAgeCallsSpec.direct_call_newer_version_refines_statement
   allows only where it is: the refinement of the rebuild does not hold
   on the machine of every relation. *)
Definition aged_call_counterexample_statement : Prop :=
  let e := AgedCallExample.unsound_unit in
  let F := AgedCallExample.facts_of e in
  let T := AgedCallExample.tables_of e in
  let age_of := AgedCallExample.age_of in
  let exit_with (n : Z) :=
    Exit 1%positive [Value_const (Tagged_immediate n)] empty_heap no_reads
  in
  least_solution Examples.no_local F T
  ∧ in_flow_fragment e = true ∧ any_kind_arguments e = true
  ∧ results_kinded [] [] e = true
  ∧ well_formed_unit Examples.interface e = true
  ∧ exits_one_value Examples.interface e = true
  ∧ behaves (AgedCallExample.initial e) Goes_wrong
  ∧ program_refines_observably e (AgedCallExample.rebuilt e)
  ∧ run_aged age_of 5 (AgedCallExample.initial e) = exit_with 0%Z
  ∧ run_aged age_of 5 (AgedCallExample.initial (AgedCallExample.rebuilt e))
    = exit_with 1%Z
  ∧ run_aged age_of 5
      (AgedCallExample.initial (AgedCallExample.rebuilt_fixed e))
    = exit_with 1%Z
  ∧ ~ program_refines_observably_aged age_of e (AgedCallExample.rebuilt e)
  ∧ ~ program_refines_observably_aged age_of e
        (AgedCallExample.rebuilt_fixed e).

(* The unit with both codes returning their argument. The relation is
   sound for [s80]. The solution gives [f50] no use and its closure
   [x41] no source, but the run of the unit on the machine enters the
   body of [f50] with [x41] bound to [s80]: the solution does not
   describe the runs of the machine, as it does those of [step]
   (FlambdaReaperFlowSpec.v). The rebuild deletes [f50]; both rebuilds
   refine the unit on the machine. *)
Definition aged_call_sound_example_statement : Prop :=
  let e := AgedCallExample.sound_unit in
  let sol := solution_of_tables (AgedCallExample.tables_of e) in
  let age_of := AgedCallExample.age_of in
  sound_for age_of (unit_statics e) (Value_symbol 80%positive)
  ∧ s_has_usage sol (Node_code_id 50%positive) = false
  ∧ s_has_source sol (Node_variable 41%positive) = false
  ∧ (∃ s : state,
       run_aged age_of 3 (AgedCallExample.initial e) = Step s
       ∧ state_expr s = Examples.return_to 3%positive [Var 40%positive]
       ∧ lookup 41%positive (state_vars s) = Some (Value_symbol 80%positive))
  ∧ lookup 50%positive (unit_codes (AgedCallExample.rebuilt e)) = None
  ∧ program_refines_observably_aged age_of e (AgedCallExample.rebuilt e)
  ∧ program_refines_observably_aged age_of e
      (AgedCallExample.rebuilt_fixed e).

(* ======================================================================== *)
(* What the fix does not keep                                               *)
(* ======================================================================== *)

(* A relation sound for a closure in the unit and not in its rebuild,
   though no code changes its calling convention:

     let code f50 x40 my_closure x41 return k3 exn k4 =
       let x47 = x41.(90 at 71) in k3 0          ([x47] has no use)
     let code f51 x43 my_closure x44 return k5 exn k6 = k5 1 in
     let symbol s80 = closure at 70 of { 70 = f51 } in
     let x60 = closure at 71 of { 71 = f50 } with { 90 = 3 } in
     let x36 = [x60; s80] in return x36

   Both closures escape, and both codes are in the least table: the table
   of the fix is that table, and the conventions agree in the rebuild.
   [f50] reads the value slot 90 of its closure, which [s80] has not: a
   call of [f50] with [s80] for its closure is undefined, and [f51]
   refines it, whatever it does. The rebuild removes the read, which has
   no use: [f50] then returns 0 with [s80], and [f51] returns 1, which
   does not refine it. So what the fix keeps of [sound_for] is the
   conventions ([rebuild_cc_fixed_sound_conventions_statement]), and not
   the refinement of the codes: the rebuild refines each code of the unit
   for the runs of the unit, and a newer version may differ from the
   older where the older is undefined. *)
Module BehaviourExample.

Local Open Scope positive_scope.
Import FlambdaReaperCallingConventions.Examples.

Definition older : code :=
  Code [40] 41 3 4 (make_code_metadata [Singleton Value] [Singleton Value])
    (Let 47 Value (Project_value_slot_from 71 90 (Var 41))
       (return_to 3 [Const (Tagged_immediate 0)])).

Definition newer : code :=
  Code [43] 44 5 6 (make_code_metadata [Singleton Value] [Singleton Value])
    (return_to 5 [Const (Tagged_immediate 1)]).

Definition unit_expr : expr :=
  Let_static
    [Bind_code 50 older; Bind_code 51 newer;
     Bind_symbol 80 (Static_const_closure_of_set 70 [(70, 80)] [(70, 51)] [])]
    (Let 60 Value
       (Closure_of_set 71 [(71, 50)] [(90, Const (Tagged_immediate 3))])
       (Let 36 Value (make_block [Var 60; Symbol 80])
          (return_to 1 [Var 36]))).

Definition relation : Code_age_relation.t :=
  Code_age_relation.add Code_age_relation.empty 51 50.

Definition age_of (_ : static_env) : Code_age_relation.t := relation.

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition tables : tables :=
  match FlambdaReaperLeast.compute_least no_local facts 40 with
  | Some T => T
  | None => FlambdaReaperLeast.empty_tables
  end.

Definition table : list node := derive_cannot_change facts tables.

Definition rebuilt : expr :=
  rebuild_unit_cc_from_tables facts tables (age_fixed_table relation table)
    (unit_code_deps Auto interface unit_expr) true unit_expr.

(* What the rebuild makes of [f50]. *)
Definition older_rebuilt : code :=
  Code [40] 41 3 4 (make_code_metadata [Singleton Value] [Singleton Value])
    (return_to 3 [Const (Tagged_immediate 0)]).

End BehaviourExample.

Definition sound_for_not_kept_statement : Prop :=
  let e := BehaviourExample.unit_expr in
  let F := BehaviourExample.facts in
  let T := BehaviourExample.tables in
  let rel := BehaviourExample.relation in
  let e' := BehaviourExample.rebuilt in
  let age_of := BehaviourExample.age_of in
  least_solution Examples.no_local F T
  ∧ in_flow_fragment e = true ∧ any_kind_arguments e = true
  ∧ results_kinded [] [] e = true
  ∧ well_formed_unit Examples.interface e = true
  ∧ exits_one_value Examples.interface e = true
  ∧ ~ behaves (State e [] [] empty_static_env empty_heap no_reads) Goes_wrong
  ∧ (∀ newer older : code_id,
       Code_age_relation.get_older_version_of rel newer = Some older ->
       mem_node (Node_code_id newer) BehaviourExample.table = true
       ∧ mem_node (Node_code_id older) BehaviourExample.table = true)
  ∧ e'
    = rebuild_unit_cc_from_tables F T BehaviourExample.table
        (unit_code_deps Auto Examples.interface e) true e
  ∧ program_refines_observably e e'
  ∧ (∀ age_of' : static_env -> Code_age_relation.t,
       program_refines_observably_aged age_of' e e')
  ∧ conventions_agree rel (unit_codes e)
  ∧ conventions_agree rel (unit_codes e')
  ∧ sound_for age_of (unit_statics e) (Value_symbol 80%positive)
  ∧ lookup 50%positive (static_code (unit_statics e'))
    = Some BehaviourExample.older_rebuilt
  ∧ ~ sound_for age_of (unit_statics e') (Value_symbol 80%positive).
