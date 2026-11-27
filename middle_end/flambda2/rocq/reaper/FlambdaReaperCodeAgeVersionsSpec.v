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

(* Two versions of a code that the unboxing treats differently, on the
   machine of the core that is given a code age relation
   (FlambdaCodeAgeCallsSpec.step_aged): a change of representation of the
   set of one and not of the other, an unboxed parameter of one only.

   This file is to be reviewed: the unit and the statements. The proofs
   are in FlambdaReaperCodeAgeVersionsProofs.v, which needs no review.

   Within the unit the rebuild makes no such call mix two versions: at a
   direct call whose callee has a source, it names the code that the flow
   finds, the code of the closure called (rewrite_call_kind, and
   [apply_change_u] for the change of calling convention), and so takes
   the convention and the representation of that code; at one whose
   callee has any source, it keeps the code named, whose [my_closure]
   then has any source, which forbids a change of representation of its
   sets ([derived_cannot_change_representation0]). The compiler does the
   same (rebuild.ml, [decide_whether_apply_needs_calling_convention_change];
   unboxing_analysis.ml, [cannot_change_representation0]).

   What is not kept is a client's call. A client of the unit runs with
   the statics of its first step ([unit_statics]), and a direct call of
   it naming a code of the unit may have for callee a closure of a newer
   version.

   [rebuild_unbox_sound_conventions_statement]: with the table of the fix
   of bug 12 ([age_fixed_table]), every rebuild with unboxing (each mode,
   with or without types, with any changes of representation: the
   rebuilds of FlambdaReaperUnboxingRefines*Spec.v are instances of
   [rebuild_unit_unbox_repr_mode_with]) keeps the conventions of the
   linked codes in the statics of a unit that begins with its
   [Let_static]: they meet [sound_conventions_for] for every closure where
   the conventions agree in the unit. So no parameter of a linked code is
   unboxed or deleted; the code of the table keeps its calling
   convention.

   [repr_client_counterexample_statement]: the table does not reach the
   changes of representation. The unit of [ReprExample]: the codes 53 and
   55, 55 a newer version of 53, read the value slot 80 of their closure
   at the function slot 70; the closure [x60] of 53 is called directly;
   the static closure [s90] of 55 escapes, and the unit calls it directly
   naming 53. With the table of the fix no code changes its calling
   convention, but the set of [x60] changes its representation, and 53
   reads the new slots. The rebuild names 55 at the call of [s90], and
   the rebuilt unit refines the unit on the machine. The relation is
   sound for [s90] in the unit and in the rebuilt unit (where 53 with
   [s90] is undefined), and the conventions agree in both. A client that
   calls [s90] directly naming 53 exits with 7 with the code of the unit,
   and its call is undefined with the code of the rebuilt unit: 53 reads
   slots that [s90] has not. Without the change of representation (the
   table of [to_change_representation] empty) the call exits with 7.

   Not reached: a fix stated for every unit. One would leave out of the
   changes of representation the sets of a code that the relation links
   to a newer version, or rename the slots of the closures of the newer
   versions alike; and it would need a statement of what a client may
   call, which the statements of the rebuild, of the unit run on its own,
   do not give. Not attempted: a reproducer in the compiler. Its reaper
   changes the representation of every set it can (to_change_representation),
   with new function and value slots, and reads [newer_version_of]
   nowhere but a CR of types_rewriter.ml; but a client names the older
   code of a closure of the newer only where a type says so, and the types
   of Simplify name the code of the allocation (the join of two code ids
   is unknown, their meet the newer). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperCallingConventionsSpec
  FlambdaReaperUnboxing FlambdaReaperLeastSpec FlambdaCodeAgeCallsSpec
  FlambdaReaperCodeAgeSpec.
From OxCamlFlambda2 Require FlambdaReaperLeast.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaCodeAgeRelation FlambdaCodeAgeRelationSpec.
Import ListNotations.

(* ======================================================================== *)
(* The conventions                                                          *)
(* ======================================================================== *)

(* With the table of the fix, the statics of the first step of a unit
   rebuilt with unboxing, which begins with its [Let_static], meet
   [sound_conventions_for] for every closure, the relation being [rel]
   and the conventions of the unit agreeing for it. *)
Definition rebuild_unbox_sound_conventions_statement : Prop :=
  ∀ (types : subkinds_rewriting) (fixed : bool) (is_local : field -> bool)
    (F : facts) (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox tcr : list node) (preserve_direct_calls : bool)
    (bindings : list static_binding) (body : expr)
    (rel : Code_age_relation.t)
    (age_of : static_env -> Code_age_relation.t) (v : value),
    let e := Let_static bindings body in
    let e' :=
      rebuild_unit_unbox_repr_mode_with types fixed is_local F T
        (age_fixed_table rel cannot_change) code_deps dominated to_unbox tcr
        preserve_direct_calls e
    in
    conventions_agree rel (unit_codes e) ->
    age_of (unit_statics e') = rel ->
    sound_conventions_for age_of (unit_statics e') v.

(* ======================================================================== *)
(* A change of representation of the set of the older version               *)
(* ======================================================================== *)

(* The unit:

     let code f53 x40 my_closure x41 return k3 exn k4 =
       let x45 = x41.(80 at 70) in k3 x45
     let code f55 x42 my_closure x43 return k5 exn k6 =
       let x46 = x43.(80 at 70) in k5 x46
     let symbol s90 = closure at 70 of { 70 = f55 } with { 80 = 7 } in
     let x60 = closure at 70 of { 70 = f53 } with { 80 = 7 } in
     let_cont k20 x30 =
       let_cont k21 x31 = let x36 = [x30; x31; s90] in return x36 in
       s90 9 return k21 exn k2                    (direct, to f53)
     in
     x60 9 return k20 exn k2                      (direct, to f53)

   [f55] is a newer version of [f53]. *)
Module ReprExample.

Local Open Scope positive_scope.
Import FlambdaReaperUnboxing.Examples.

(* Code of one parameter that returns the value slot 80 of its closure,
   read at the function slot 70. *)
Definition slot_reader (x my_closure k k_exn y : variable) : code :=
  Code [x] my_closure k k_exn
    (make_code_metadata [Singleton Value] [Singleton Value])
    (Let y Value (Project_value_slot_from 70 80 (Var my_closure))
       (return_to k [Var y])).

Definition older : code := slot_reader 40 41 3 4 45.

Definition newer : code := slot_reader 42 43 5 6 46.

(* A direct call naming [id], with 9, returning to [k]. *)
Definition call (id : code_id) (callee : simple) (k : continuation)
    : apply_expr :=
  {| ap_callee := callee; ap_call_kind := Direct id; ap_args := [int 9];
     ap_continuation := k; ap_exn_continuation := 2;
     ap_exn_extra_args := [];
     ap_args_arity := [Singleton Value];
     ap_return_arity := [Singleton Value] |}.

Definition unit_expr : expr :=
  Let_static
    [Bind_code 53 older; Bind_code 55 newer;
     Bind_symbol 90
       (Static_const_closure_of_set 70 [(70, 90)] [(70, 55)] [(80, int 7)])]
    (Let 60 Value (Closure_of_set 70 [(70, 53)] [(80, int 7)])
       (Let_cont_non_recursive 20
          (Handler [30] (make_handler_metadata [Value])
             (Let_cont_non_recursive 21
                (Handler [31] (make_handler_metadata [Value])
                   (Let 36 Value (make_block [Var 30; Var 31; Symbol 90])
                      (return_to 1 [Var 36])))
                (Apply (call 53 (Symbol 90) 21))))
          (Apply (call 53 (Var 60) 20)))).

Definition relation : Code_age_relation.t :=
  Code_age_relation.add Code_age_relation.empty 55 53.

Definition age_of (_ : static_env) : Code_age_relation.t := relation.

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition tables : tables :=
  match FlambdaReaperLeast.compute_least no_local facts 40 with
  | Some T => T
  | None => FlambdaReaperLeast.empty_tables
  end.

Definition code_deps : finite_map code_dep :=
  unit_code_deps Auto interface unit_expr.

(* The table of the fix, and the tables of the unboxing that the rules
   give with it. *)
Definition fixed : list node :=
  age_fixed_table relation (derive_cannot_change facts tables).

Definition dominated : list (node * node) := derive_dominated tables.

Definition to_unbox : list node :=
  derive_to_unbox no_local facts tables fixed max_unbox_size false.

Definition tcr : list node :=
  derive_to_change_representation no_local facts tables fixed max_unbox_size
    false to_unbox.

(* The rebuild in the mode of the fix, and the same with no change of
   representation. *)
Definition rebuilt : expr :=
  rebuild_unit_unbox_repr_fixed no_local facts tables fixed code_deps
    dominated to_unbox tcr true unit_expr.

Definition rebuilt_same_repr : expr :=
  rebuild_unit_unbox_repr_fixed no_local facts tables fixed code_deps
    dominated to_unbox [] true unit_expr.

(* The representation of the set of [x60]: the value slot 81 for 80, and
   the function slot 82 for 70. *)
Definition repr : closure_repr :=
  {| cr_fields := [(Field_value_slot 80, Not_unboxed 81)];
     cr_function_slots := [(70, 82)];
     cr_current := 70 |}.

(* What the rebuild makes of [f53]: it reads the new slots. *)
Definition older_rebuilt : code :=
  Code [40] 41 3 4 (make_code_metadata [Singleton Value] [Singleton Value])
    (Let 45 Value (Project_value_slot_from 82 81 (Var 41))
       (return_to 3 [Var 45])).

Definition initial (e : expr) : state :=
  State e [] [] empty_static_env empty_heap no_reads.

(* A client of a unit whose first step binds [statics]: it calls [s90]
   directly, naming the code [id], with 5, and returns to the
   continuation 1. *)
Definition client (statics : static_env) (id : code_id) : state :=
  State
    (Apply
       {| ap_callee := Symbol 90; ap_call_kind := Direct id;
          ap_args := [int 5];
          ap_continuation := 1; ap_exn_continuation := 2;
          ap_exn_extra_args := [];
          ap_args_arity := [Singleton Value];
          ap_return_arity := [Singleton Value] |})
    [] [(1, Outer 1); (2, Outer 2)] statics empty_heap no_reads.

End ReprExample.

(* The table of the fix passes the check, and no code changes its calling
   convention; the set of [x60] changes its representation. The unit
   goes wrong on [step] (its call of [s90] naming [f53]); on the machine
   it exits with [[7; 7; s90]], and so does the rebuilt unit, which
   refines it. The conventions agree in the unit and in the rebuilt unit,
   and the relation is sound for [s90] in both. [f53] of the rebuilt unit
   reads the new slots. The client's call naming [f53] exits with 7 with
   the statics of the unit, and is undefined with those of the rebuilt
   unit, which does not refine it; naming [f55] it exits with 7. With no
   change of representation, the call naming [f53] exits with 7. *)
Definition repr_client_counterexample_statement : Prop :=
  let e := ReprExample.unit_expr in
  let F := ReprExample.facts in
  let T := ReprExample.tables in
  let rel := ReprExample.relation in
  let age_of := ReprExample.age_of in
  let e' := ReprExample.rebuilt in
  let before := unit_statics e in
  let after := unit_statics e' in
  let seven :=
    Exit 1%positive [Value_const (Tagged_immediate 7)] empty_heap no_reads
  in
  let result :=
    [Value_block 0
       [Value_const (Tagged_immediate 7); Value_const (Tagged_immediate 7);
        Value_symbol 90%positive]]
  in
  least_solution Examples.no_local F T
  ∧ check_cc_closed F T ReprExample.fixed = true
  ∧ unit_calling_convention_changes_u F T ReprExample.fixed
      ReprExample.code_deps ReprExample.dominated ReprExample.to_unbox e
    = [(53%positive, Not_changing_u); (55%positive, Not_changing_u)]
  ∧ unit_changed_representations Examples.no_local F T
      ReprExample.dominated ReprExample.to_unbox ReprExample.tcr e
    = [(Node_variable 41%positive, ReprExample.repr);
       (Node_made 5, ReprExample.repr);
       (Node_variable 60%positive, ReprExample.repr)]
  ∧ behaves (ReprExample.initial e) Goes_wrong
  ∧ run_aged age_of 20 (ReprExample.initial e)
    = Exit 1%positive result empty_heap no_reads
  ∧ run_aged age_of 20 (ReprExample.initial e')
    = Exit 1%positive result empty_heap no_reads
  ∧ program_refines_observably_aged age_of e e'
  ∧ conventions_agree rel (unit_codes e)
  ∧ conventions_agree rel (unit_codes e')
  ∧ sound_for age_of before (Value_symbol 90%positive)
  ∧ sound_for age_of after (Value_symbol 90%positive)
  ∧ lookup 53%positive (static_code after) = Some ReprExample.older_rebuilt
  ∧ run_aged age_of 3 (ReprExample.client before 53%positive) = seven
  ∧ run_aged age_of 2 (ReprExample.client after 53%positive) = Undefined
  ∧ ~ refines_aged age_of (ReprExample.client before 53%positive)
        (ReprExample.client after 53%positive)
  ∧ run_aged age_of 3 (ReprExample.client after 55%positive) = seven
  ∧ run_aged age_of 3
      (ReprExample.client (unit_statics ReprExample.rebuilt_same_repr)
         53%positive)
    = seven.
