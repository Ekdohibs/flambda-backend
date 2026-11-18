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

(* The fix of local fields: the rule that it adds to the reaper's, its
   check, and the least solution of the rules with it, computed.

   This file is the implementation; it needs no review.
   FlambdaReaperLocalFieldsSpec.v says what the rule is and what its
   answers mean, and FlambdaReaperLocalFieldsProofs.v shows that they
   do.

   The rule ([local_reads_used] in the Spec file, the last of
   [local_rules] in points_to_analysis.ml with the fix): a read of a
   local field from a node that has any source gives that node
   [has_usage]. [check_local_reads_used] tests it, over the accessors
   of the graph, as FlambdaReaperSolution.v tests the other rules;
   [check_closed_local] tests every rule, those of [check_closed] and
   this one.

   [step_local] is [FlambdaReaperLeast.step] with the new rule, and
   [saturate_local], [inflate_local], [alternate_local] and
   [compute_least_local] are those of FlambdaReaperLeast.v with
   [step_local] for [step] and [check_closed_local] for
   [check_closed]. *)

From Stdlib Require Import List Bool PeanoNat.
From OxCamlFlambda2 Require Import FlambdaReaperGraphSpec
  FlambdaReaperSolution FlambdaReaperLeast.
Import ListNotations.

Section Rule.

Variable is_local : field -> bool.
Variable F : facts.
Variable T : tables.

(* The rule: [any_source base], [accessor ~to_ relation ~base] and
   [is_local relation] give [has_usage base]. *)
Definition check_local_reads_used : bool :=
  forallb
    (fun '(_, relation, base) =>
       negb (mem_node base (t_any_source T) && is_local relation)
       || mem_node base (t_has_usage T))
    (f_accessor F).

(* Every rule: those of the compiler before the fix, and the new one. *)
Definition check_closed_local : bool :=
  check_closed is_local F T && check_local_reads_used.

(* What the new rule concludes from [T]. *)
Definition new_has_usage_local : list node :=
  flat_map
    (fun '(_, relation, base) =>
       when (mem_node base (t_any_source T) && is_local relation) [base])
    (f_accessor F).

End Rule.

Section Iterate.

Variable is_local : field -> bool.
Variable F : facts.

(* One application of every rule, the new one with the others. *)
Definition step_local (Tg T : tables) : tables :=
  let T' := step is_local F Tg T in
  {| t_alias := t_alias T';
     t_any_usage := t_any_usage T';
     t_any_source := t_any_source T';
     t_has_usage :=
       add_new node_eqb (t_has_usage T') (new_has_usage_local is_local F T);
     t_has_source := t_has_source T';
     t_usages := t_usages T';
     t_sources := t_sources T';
     t_reading_field := t_reading_field T';
     t_escaping_field := t_escaping_field T';
     t_field_used := t_field_used T' |}.

Fixpoint saturate_local (Tg : tables) (fuel : nat) (T : tables) : tables :=
  match fuel with
  | O => T
  | S fuel =>
    let T' := step_local Tg T in
    if Nat.eqb (tables_size T') (tables_size T) then T
    else saturate_local Tg fuel T'
  end.

Fixpoint inflate_local (fuel : nat) (T : tables) : tables :=
  match fuel with
  | O => T
  | S fuel =>
    let T' := step_local T T in
    if Nat.eqb (tables_size T') (tables_size T) then T
    else inflate_local fuel T'
  end.

Fixpoint alternate_local (fuel rounds : nat) (Tg : tables)
    : option tables :=
  let T := saturate_local Tg fuel empty_tables in
  if check_closed_local is_local F T && tops_within T Tg then Some T
  else
    match rounds with
    | O => None
    | S rounds => alternate_local fuel rounds T
    end.

(* A least solution of the rules with the fix. *)
Definition compute_least_local (fuel : nat) : option tables :=
  alternate_local fuel fuel (inflate_local fuel empty_tables).

End Iterate.
