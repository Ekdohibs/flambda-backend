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

(* What the check of a solution of the reaper's rules means.

   This file is to be reviewed: the statements, and the example against
   the rules of middle_end/flambda2/reaper/points_to_analysis.ml (or the
   fields of FlambdaReaperGraphSpec.closed, which transcribe them). The
   checks themselves are FlambdaReaperSolution.v, which needs no review;
   the proofs are FlambdaReaperSolutionProofs.v.

   The solution of the reaper is read off the compiler's Datalog
   database, and nothing is believed of it: [check_closed] tests it, and
   a solution whose tables pass is closed under the rules
   ([check_closed_sound_statement]), which is all that the statements
   about the rebuild ask of it. Likewise [check_field_used_exact] tests
   that [field_of_constructor_is_used] holds only where a rule of the
   post-processing gives it ([check_field_used_exact_sound_statement]).
   Nothing is said of a solution that fails: the check may be
   conservative.

   That the premise can hold, and that the check tests something, is the
   example of the last part: the least solution of a small graph, worked
   out by hand from the rules, passes both checks
   ([example_passes_statement]); without any one of its tuples it fails
   [check_closed] ([example_each_tuple_needed_statement]), so the
   example is the least solution; and with a field that is built but
   never read said to be used, it fails [check_field_used_exact]
   ([example_unread_field_rejected_statement]). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaReaperGraphSpec FlambdaReaperSolution.
Import ListNotations.

(* ======================================================================== *)
(* Soundness                                                                *)
(* ======================================================================== *)

Definition check_closed_sound_statement : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables),
    check_closed is_local F T = true ->
    closed is_local (graph_of_facts F) (solution_of_tables T).

Definition check_field_used_exact_sound_statement : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables),
    check_field_used_exact is_local F T = true ->
    field_used_exact is_local (graph_of_facts F) (solution_of_tables T).

(* ======================================================================== *)
(* An example                                                               *)
(* ======================================================================== *)

(* The lists that are [l] without one of its elements, one for each. *)
Fixpoint one_fewer {A : Type} (l : list A) : list (list A) :=
  match l with
  | [] => []
  | x :: l' => l' :: map (cons x) (one_fewer l')
  end.

(* The tables that are [T] without one of its tuples, one for each. *)
Definition one_fewer_tables (T : tables) : list tables :=
  let '(Tables a u s hu hs us so rf ef fu) := T in
  map (fun a => Tables a u s hu hs us so rf ef fu) (one_fewer a)
  ++ map (fun u => Tables a u s hu hs us so rf ef fu) (one_fewer u)
  ++ map (fun s => Tables a u s hu hs us so rf ef fu) (one_fewer s)
  ++ map (fun hu => Tables a u s hu hs us so rf ef fu) (one_fewer hu)
  ++ map (fun hs => Tables a u s hu hs us so rf ef fu) (one_fewer hs)
  ++ map (fun us => Tables a u s hu hs us so rf ef fu) (one_fewer us)
  ++ map (fun so => Tables a u s hu hs us so rf ef fu) (one_fewer so)
  ++ map (fun rf => Tables a u s hu hs us so rf ef fu) (one_fewer rf)
  ++ map (fun ef => Tables a u s hu hs us so rf ef fu) (one_fewer ef)
  ++ map (fun fu => Tables a u s hu hs us so rf ef fu) (one_fewer fu).

(* The graph is that of

     let b = { x; x } in          (block [b], two fields)
     let y = b.0 in               (only the first field is read)
     let c = closure of k, with the value slot w = x in
     ... in the code [k], whose closure is [d]: let e = d.w in ...
     g y, where the code of [g] has the parameter [p]
     r is computed from y, e and p, and used out of the unit; c escapes

   [x] comes from outside the unit ([any_source]), and so does the
   closure [d] that the code [k] is given. The value slot [w] is local;
   no other field is. *)
Module Example.

Definition x : node := Node_variable 1%positive.
Definition b : node := Node_variable 2%positive.
Definition y : node := Node_variable 3%positive.
Definition r : node := Node_variable 4%positive.
Definition c : node := Node_variable 5%positive.
Definition d : node := Node_variable 6%positive.
Definition e : node := Node_variable 7%positive.
Definition g : node := Node_variable 8%positive.
Definition p : node := Node_variable 9%positive.
Definition k : node := Node_code_id 1%positive.

Definition field_0 : field := Field_block 0 Value.
Definition field_1 : field := Field_block 1 Value.
Definition w : field := Field_value_slot 1%positive.
Definition fs : field := Field_function_slot 1%positive.

Definition is_local (f : field) : bool := field_eqb f w.

Definition graph : facts :=
  {| f_alias := [];
     f_use := [(r, y); (r, e); (r, p)];
     f_accessor := [(y, field_0, b); (e, w, d)];
     f_constructor :=
       [(b, field_0, x); (b, field_1, x); (c, w, x); (c, fs, k)];
     f_argument := [(y, 0, g)];
     f_parameter := [(g, 0, p)];
     f_propagate := [];
     f_alias_if_any_source := [];
     f_any_usage := [r; c];
     f_any_source := [x; d];
     f_any_source_if_used := [];
     f_zero_alloc_source := [] |}.

(* The least solution, worked out by hand. [r] and [c] are used out of
   the unit; [y], [e] and [p] are used to compute [r] ([use]), and [k]
   is in a field of [c], which is not local. The read of [b.0] in [y]
   and the block [b] built with [x] make [y] an alias of [x]
   ([alias_constructor_usage], [alias_accessor_source]); the value slot
   [w] of [c], which escapes, holds [x], and the read of [w] in the
   closure [d], which comes from outside, is [e], so [e] is an alias of
   [x] ([escaping_field_rule], [reading_field_rule],
   [alias_escaping_reading]); the argument [y] of the call of [g] makes
   the parameter [p] an alias of [y] ([alias_parameter_usage],
   [alias_argument_source]). Then [any_usage] goes back along the
   aliases to [x], and [any_source] forward from [x] to [y], [e] and
   [p], and to [r] by [use]. The field 0 of [b] is used because [y] is;
   [w] because [e] is; [fs] because [c] escapes. The field 1 of [b] is
   not used. *)
Definition solution : tables :=
  {| t_alias := [(y, x); (e, x); (p, y)];
     t_any_usage := [r; c; y; e; p; k; x];
     t_any_source := [x; d; y; e; p; r];
     t_has_usage := [r; c; y; e; p; k; x; b; d; g];
     t_has_source := [x; d; y; e; p; b; c; g; r];
     t_usages := [(b, b); (d, d); (g, g)];
     t_sources := [(b, b); (c, c); (g, g)];
     t_reading_field := [(w, e)];
     t_escaping_field := [(w, x)];
     t_field_used := [(b, field_0); (c, fs); (c, w)] |}.

(* The same, with the field 1 of [b], which nothing reads, said to be
   used. *)
Definition solution_with_unread_field_used : tables :=
  let '(Tables a u s hu hs us so rf ef fu) := solution in
  Tables a u s hu hs us so rf ef ((b, field_1) :: fu).

End Example.

Definition example_passes_statement : Prop :=
  check_closed Example.is_local Example.graph Example.solution = true
  ∧ check_field_used_exact Example.is_local Example.graph Example.solution
    = true.

Definition example_each_tuple_needed_statement : Prop :=
  ∀ T : tables,
    T ∈ one_fewer_tables Example.solution ->
    check_closed Example.is_local Example.graph T = false.

Definition example_unread_field_rejected_statement : Prop :=
  check_closed Example.is_local Example.graph
    Example.solution_with_unread_field_used = true
  ∧ check_field_used_exact Example.is_local Example.graph
      Example.solution_with_unread_field_used = false.
