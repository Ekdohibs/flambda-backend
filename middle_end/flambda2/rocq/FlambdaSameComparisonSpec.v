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

(* A [Phys_equal] of a simple against itself, and of two different
   symbols, becomes its answer: a normalisation that refines.

   This file is to be reviewed: [fold_same_comparisons] and what it is
   made of, which are extracted and run by the validation of
   middle_end/flambda2/tests/semantics/
   (FlambdaValidateSpec.normalise_expr_before), against what Simplify
   does; and the statements.

   What the source has. A handler [with Not_found -> ...] compares what
   was raised with the exception: a [Phys_equal] of the parameter of
   the exception handler against the symbol of the exception. Where
   every raise that reaches the handler raises that exception, the
   parameter is the symbol, and the comparison is of the symbol against
   itself. The same arises of any value compared with itself by [==].
   Where a raise that reaches the handler raises another exception that
   the term names, the comparison is of two different symbols.

   What Simplify does. [Simplify_binary_primitive.simplify_phys_equal]
   (middle_end/flambda2/simplify/simplify_binary_primitive.ml) asks
   [Provers.prove_physical_equality]
   (middle_end/flambda2/types/provers.ml) of the types of the two
   arguments. Each is the type of an alias, and where the two are
   aliases of one simple, the same constant, the same symbol or the
   same variable, the answer is that they are equal (the last match of
   that function), whatever the value is. The primitive becomes the
   naked immediate 1 for [Phys_eq] and 0 for [Phys_neq]. Where the two
   are aliases of two different symbols, the answer of the same match
   is that they differ, and the primitive becomes 0 for [Phys_eq] and 1
   for [Phys_neq]. That match answers that a symbol and a constant
   differ too, and two different constants; this pass leaves the first
   as it is, and FlambdaNormaliseSpec.fold_constants computes the
   second where the primitive has a value.

   [fold_same_comparisons] does the same with what can be read off the
   term: a [Let] of [Phys_equal] applied to two arguments that are the
   same simple becomes a [Let] of that constant, and applied to two
   symbols that are not the same a [Let] of the constant of the other
   answer ([fold_same_comparison]). It looks at one [Let] at a time and knows
   nothing of where it is; the rewrites of the validation that follow
   the values put one simple in the place of two that are equal before
   it runs, and those that fold constants and remove what is unused
   run after it. Like FlambdaNormaliseSpec.fold_constants, which this
   file follows, it goes into the handler of a non-recursive
   continuation, and leaves the handlers of recursive ones and the
   bodies of code, for the validation to take the pass into them as it
   does the other rewrites of a round.

   Why it is sound. Where the comparison of a value against itself has
   an answer, the answer is that the two are the same
   (FlambdaPrimitiveSpec.phys_equal_same_statement), which is the
   constant. Where the comparison of two different symbols has an
   answer, the answer is that the two differ
   (FlambdaPrimitiveSpec.phys_equal_symbols_statement), which is the
   other constant. So the defining expression that the pass makes has the
   value that the one it replaces has, wherever that one has a value
   ([fold_same_comparison_sound_statement]), and the result refines
   the term ([fold_same_comparisons_refines_statement]).

   It is a refinement and no equivalence. [Phys_equal] has no value on
   an allocated value that has no identity, on null and on poison (the
   CRs before FlambdaPrimitiveSpec.eval_binary_primitive), and of two
   different symbols it has a value only where one of the two is an
   identity (the comment of [phys_equal_symbols_statement]); elsewhere
   the term is undefined and the result is not. So the validation
   applies the pass to the term before Simplify only.

   The machine of FlambdaSemanticsSpec.v computes the primitive of a
   [Let] by FlambdaPrimitiveSpec.eval_prim, and that is the machine of
   the two statements above. The machine with a heap computes it by
   FlambdaHeapSpec.exec_prim, which compares a location too, and may
   change the heap. [phys_equal_same_heap_statement] says of it what
   the law says of [eval_prim], and that the heap is left as it was:
   the comparison of a value against itself, a location included, has
   the same answer where it has one. [phys_equal_symbols_heap_statement]
   says the same of two different symbols, neither of which is a
   location. They are what the refinement needs when the machine of
   the semantics runs [exec_prim].

   The pass commutes with the renamings of FlambdaNominalSpec.v
   ([fold_same_comparisons_equivariant_statement]): it compares two
   simples for equality, which a renaming keeps as it is injective, it
   asks whether the two are symbols, which a renaming leaves as they
   are, and it makes no name.

   The proofs are in FlambdaSameComparisonProofs.v, which needs no
   review. That the pass keeps scope is stated in
   FlambdaSameComparisonScopeSpec.v. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaPrimitiveModel
  FlambdaPrimitiveSpec FlambdaSemanticsSpec FlambdaNominalSpec
  FlambdaHeapSpec.
Import ListNotations.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

(* The answer of [Phys_equal op] where the two values are the same, as
   a constant: FlambdaPrimitiveSpec.naked_bool of [true] for [Phys_eq]
   and of [false] for [Phys_neq]. *)
Definition same_answer (op : equality_comparison) : const :=
  Naked_immediate
    match op with
    | Phys_eq => 1
    | Phys_neq => 0
    end.

(* The answer of [Phys_equal op] where the two values differ:
   [naked_bool] of [false] for [Phys_eq] and of [true] for [Phys_neq]. *)
Definition different_answer (op : equality_comparison) : const :=
  Naked_immediate
    match op with
    | Phys_eq => 0
    | Phys_neq => 1
    end.

Definition fold_same_comparison (d : named) : named :=
  match d with
  | Prim (Binary (Phys_equal op)) [a; b] =>
    if simple_eq_dec a b
    then Simple (Const (same_answer op))
    else
      match a, b with
      | Symbol _, Symbol _ => Simple (Const (different_answer op))
      | Symbol _, (Var _ | Const _) | (Var _ | Const _), _ => d
      end
  | Prim _ _ | Simple _ | Set_of_closures _ _ | Closure_of_set _ _ _
  | Project_value_slot _ _ | Project_value_slot_from _ _ _
  | Project_function_slot _ _ _ =>
    d
  end.

Fixpoint fold_same_comparisons (e : expr) : expr :=
  match e with
  | Let x kind d body =>
    Let x kind (fold_same_comparison d) (fold_same_comparisons body)
  | Let_static bindings body =>
    Let_static bindings (fold_same_comparisons body)
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    Let_cont_non_recursive k
      (Handler params metadata (fold_same_comparisons handler))
      (fold_same_comparisons body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (fold_same_comparisons body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* Where the defining expression has a value, what the pass makes of it
   has that value. Nothing is said where it has none. *)
Definition fold_same_comparison_sound_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (d : named) (v : value),
    eval_named statics vars d = Some v ->
    eval_named statics vars (fold_same_comparison d) = Some v.

Definition fold_same_comparisons_refines_statement : Prop :=
  ∀ e : expr, expr_refines e (fold_same_comparisons e).

(* The law for the machine with a heap: [Phys_equal] of a value against
   itself, where FlambdaHeapSpec.exec_prim has an answer, has the
   answer that the two are the same, and leaves the heap as it was.
   The value may be a location. *)
Definition phys_equal_same_heap_statement : Prop :=
  ∀ (statics : static_env) (h h' : heap) (op : equality_comparison)
    (v result : value),
    exec_prim statics h (Binary (Phys_equal op)) [v; v] = Some (result, h') ->
    result = Value_const (same_answer op) ∧ h' = h.

(* The same law for two different symbols: where [exec_prim] has an
   answer, the answer is that the two differ, and the heap is left as
   it was. *)
Definition phys_equal_symbols_heap_statement : Prop :=
  ∀ (statics : static_env) (h h' : heap) (op : equality_comparison)
    (s1 s2 : symbol) (result : value),
    s1 <> s2 ->
    exec_prim statics h (Binary (Phys_equal op))
      [Value_symbol s1; Value_symbol s2]
    = Some (result, h') ->
    result = Value_const (different_answer op) ∧ h' = h.

Definition fold_same_comparisons_equivariant_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    fold_same_comparisons (rename_expr r e)
    = rename_expr r (fold_same_comparisons e).
