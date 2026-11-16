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

(* A test against zero of the result of a three-way comparison becomes
   the comparison of the two operands: a normalisation that refines.

   This file is to be reviewed: [recover_known_comparisons] and what it
   is made of, which are extracted and run by the validation of
   middle_end/flambda2/tests/semantics/
   (FlambdaValidateSpec.normalise_expr_before); the table
   [against_zero], against the compiler's; and the statements.

   What the source has. [compare a b < 0], [compare a b = 0], and the
   first test of a [match] on [compare a b], reach Simplify as three
   primitives: [Int_comp kind (Yielding_int_like_compare_functions s)]
   of the two operands, whose result is the naked immediate -1, 0 or 1;
   [Tag_immediate] of that; and a comparison of the tagged result with
   the tagged immediate 0, which is [Int_comp] at the tagged immediates
   yielding a boolean, or [Phys_equal], the form that [=] and [<>] at
   type [int] take (FlambdaPrimitiveSpec.v, at [eval_phys_equal]). The
   kind is that of the integers compared: tagged immediates for [int],
   and the naked numbers for the boxed integers, once they are unboxed.

   What Simplify does. It makes the third primitive a comparison of the
   two operands themselves, at their kind:
   [Simplify_binary_primitive.recover_comparison_primitive]
   (middle_end/flambda2/simplify/simplify_binary_primitive.ml), which
   [simplify_binary_primitive] applies to every binary primitive before
   anything else. That function looks at [Phys_equal] and at [Int_comp]
   on tagged immediates yielding a boolean, where one argument is the
   constant tagged immediate 0 and the other a variable for which the
   environment has a comparison result
   ([Downwards_env.find_comparison_result]). The environment records
   one where a primitive is recorded for common subexpression
   elimination ([Downwards_env.add_cse], with
   [Comparison_result.create] of
   middle_end/flambda2/simplify/comparison_result.ml): for a variable
   bound to a three-way [Int_comp], its kind, its signedness and its
   two arguments; and for a variable bound to [Tag_immediate] of such a
   variable, the same, marked as tagged. The new comparison is given by
   [Comparison_result.convert_result_compared_to_tagged_zero]. A test
   with the zero on the left is taken the other way round ([Lt] for
   [Gt], [Le] for [Ge]) before that table is applied. Once the test no
   longer reads the tagged result, the three-way comparison and its
   tagging go where nothing else uses them.

   [recover_known_comparisons] does the same with what can be read off
   the term. It keeps what FlambdaCseSpec.share_primitives_knowing
   keeps: for a variable bound by a [Let] to a primitive that may be
   shared, that primitive and its arguments ([computed]; [Int_comp] and
   [Tag_immediate] may be shared). It looks through every entry for a
   variable and not at the first only ([first_known]), so that it does
   not rest on there being one entry for each variable. A [Let] of a
   primitive that tests two tagged immediates ([tested]), one being
   the constant 0 and the other
   a variable known to hold [Tag_immediate] of a variable known to hold
   a three-way [Int_comp] ([compared]), becomes a [Let] of the
   comparison that [against_zero] gives, at the kind of the three-way
   comparison and applied to its arguments ([recover_comparison]).
   Nothing else is rewritten, and the [Let]s of the three-way
   comparison and of its tagging stay: the validation has other
   rewrites that remove a [Let] that nothing uses. As in
   FlambdaCseSpec.v, which this file follows: what is known of a
   variable, and what mentions it in an argument, is forgotten under a
   binder of it, so that the operands are what they were where the
   comparison is put; a handler of a non-recursive continuation is
   gone into, with what is known where it is defined; the handlers of
   recursive ones and the bodies of code are left as they are, for the
   validation to take the pass into them as it does the other rewrites
   of a round.

   [against_zero] is the compiler's table, case for case. The test
   [c] is of the tagged result [r] against 0, [r] being -1, 0 or 1 as
   the first operand is less than, equal to or greater than the second
   in the reading [s]:
   - [r = 0] is that the operands are equal, and [r <> 0] that they are
     not;
   - a signed [r < 0], [r > 0], [r <= 0], [r >= 0] is the same
     comparison of the operands in the reading [s];
   - an unsigned [r > 0] is [r <> 0], as no unsigned integer is below
     0, so it is that the operands are not equal; an unsigned [r <= 0]
     is [r = 0], that they are equal;
   - an unsigned [r < 0] is never true and an unsigned [r >= 0] always:
     the compiler leaves those two as they are, and so does this file.

   Why it is sound. Where the test has a value, the variable it reads
   has the value that [Tag_immediate] gave of the result of the
   three-way comparison, so that comparison had a value on the values
   that its arguments still have: two integers of its kind, in range.
   The new comparison has a value on them, and it is the boolean of the
   test by the table. For the two cases of equality, two integers in
   the range of a kind that have the same unsigned reading are the
   same.

   The result refines the term
   ([recover_known_comparisons_refines_statement]). It is not claimed
   to be equivalent to it, though it most likely is; a refinement is
   what the term before Simplify needs.

   The pass commutes with the renamings of FlambdaNominalSpec.v, what
   it knows being renamed with the term
   ([recover_comparisons_knowing_equivariant_statement]): it compares
   variables for equality and makes no name.

   The proofs are in FlambdaRecoverComparisonProofs.v, which needs no
   review. That the pass keeps scope is stated in
   FlambdaRecoverComparisonScopeSpec.v. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaPrimitiveModel FlambdaPrimitiveSpec
  FlambdaSemanticsSpec FlambdaNormaliseSpec FlambdaNominalSpec
  FlambdaCseSpec.
Import ListNotations.

(* ======================================================================== *)
(* What is known of a variable                                              *)
(* ======================================================================== *)

(* The first answer that [f] gives of an entry of what is known. What
   is known may have several entries for one variable; every one that
   this file reads holds where the pass runs
   (FlambdaRecoverComparisonProofs.comparisons_hold), so any will do. *)
Fixpoint first_known {A : Type}
    (f : variable * (prim * list simple) -> option A) (known : computed)
    : option A :=
  match known with
  | [] => None
  | entry :: known =>
    match f entry with
    | Some a => Some a
    | None => first_known f known
    end
  end.

(* What a three-way comparison is: the kind of the integers compared,
   how they are read, and the arguments of the comparison. *)
Definition comparison : Type := standard_int * signedness * list simple.

(* An entry that says that [u] holds the result of a three-way
   comparison: that comparison. *)
Definition comparison_held (u : variable)
    (entry : variable * (prim * list simple)) : option comparison :=
  match entry with
  | (y,
     (Binary (Int_comp kind (Yielding_int_like_compare_functions s)),
      operands)) =>
    if Pos.eqb u y then Some (kind, s, operands) else None
  | _ => None
  end.

(* An entry that says that [t] holds [Tag_immediate] of a variable,
   where that variable is known to hold the result of a three-way
   comparison: that comparison. *)
Definition tagged_comparison_held (known : computed) (t : variable)
    (entry : variable * (prim * list simple)) : option comparison :=
  match entry with
  | (y, (Unary Tag_immediate, [Var u])) =>
    if Pos.eqb t y then first_known (comparison_held u) known else None
  | _ => None
  end.

(* A simple that is a variable known to hold the tagged result of a
   three-way comparison: that comparison. *)
Definition compared (known : computed) (tagged : simple)
    : option comparison :=
  match tagged with
  | Var t => first_known (tagged_comparison_held known t) known
  | _ => None
  end.

(* ======================================================================== *)
(* The test, and the comparison it becomes                                  *)
(* ======================================================================== *)

(* The constant tagged immediate 0. *)
Definition is_tagged_zero (s : simple) : bool :=
  match s with
  | Const (Tagged_immediate i) => Z.eqb i 0
  | _ => false
  end.

(* The comparison that a primitive makes of two tagged immediates, to
   yield a boolean: [Phys_equal], which on two tagged immediates is
   whether they are the same integer, and [Int_comp] at the tagged
   immediates. *)
Definition tested (p : prim) : option int_comparison :=
  match p with
  | Binary (Phys_equal Phys_eq) => Some Int_eq
  | Binary (Phys_equal Phys_neq) => Some Int_neq
  | Binary (Int_comp Standard_tagged_immediate (Yielding_bool c)) => Some c
  | _ => None
  end.

(* The comparison with its arguments the other way round. *)
Definition swap_comparison (c : int_comparison) : int_comparison :=
  match c with
  | Int_eq => Int_eq
  | Int_neq => Int_neq
  | Int_lt s => Int_gt s
  | Int_gt s => Int_lt s
  | Int_le s => Int_ge s
  | Int_ge s => Int_le s
  end.

(* The comparison of the two operands that is the test [c] against 0 of
   the result of comparing them three ways in the reading [s]
   ([Comparison_result.convert_result_compared_to_tagged_zero]). *)
Definition against_zero (s : signedness) (c : int_comparison)
    : option int_comparison :=
  match c with
  | Int_eq => Some Int_eq
  | Int_neq => Some Int_neq
  | Int_lt Signed => Some (Int_lt s)
  | Int_gt Signed => Some (Int_gt s)
  | Int_le Signed => Some (Int_le s)
  | Int_ge Signed => Some (Int_ge s)
  | Int_lt Unsigned => None
  | Int_gt Unsigned => Some Int_neq
  | Int_le Unsigned => Some Int_eq
  | Int_ge Unsigned => None
  end.

(* The test [c] of [tagged] against [zero], where [zero] is the
   constant 0 and [tagged] is known to hold the tagged result of a
   three-way comparison: the comparison of its operands. *)
Definition recover_one_way (known : computed) (tagged zero : simple)
    (c : int_comparison) : option named :=
  if is_tagged_zero zero
  then
    match compared known tagged with
    | Some (kind, s, operands) =>
      match against_zero s c with
      | Some c' =>
        Some (Prim (Binary (Int_comp kind (Yielding_bool c'))) operands)
      | None => None
      end
    | None => None
    end
  else None.

(* A test with the zero on the right, or else with the zero on the
   left, which is the test the other way round with it on the right. *)
Definition recover_comparison (known : computed) (d : named) : named :=
  match d with
  | Prim p [arg1; arg2] =>
    match tested p with
    | Some c =>
      match recover_one_way known arg1 arg2 c with
      | Some d' => d'
      | None =>
        match recover_one_way known arg2 arg1 (swap_comparison c) with
        | Some d' => d'
        | None => d
        end
      end
    | None => d
    end
  | _ => d
  end.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

Fixpoint recover_comparisons_knowing (known : computed) (e : expr) : expr :=
  match e with
  | Let x kind d body =>
    Let x kind (recover_comparison known d)
      (recover_comparisons_knowing (computed_under_let known x d) body)
  | Let_static bindings body =>
    Let_static bindings (recover_comparisons_knowing known body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (recover_comparisons_in_handler known h)
      (recover_comparisons_knowing known body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (recover_comparisons_knowing known body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with recover_comparisons_in_handler (known : computed)
    (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata
      (recover_comparisons_knowing (forget_computed_vars params known)
         handler)
  end.

Definition recover_known_comparisons (e : expr) : expr :=
  recover_comparisons_knowing [] e.

Definition recover_known_comparisons_refines_statement : Prop :=
  ∀ e : expr, expr_refines e (recover_known_comparisons e).

(* ======================================================================== *)
(* The pass commutes with renaming                                          *)
(* ======================================================================== *)

(* What is known is renamed with the term
   (FlambdaCseSpec.rename_computed). *)
Definition recover_comparisons_knowing_equivariant_statement : Prop :=
  ∀ (r : renaming) (known : computed) (e : expr),
    recover_comparisons_knowing (rename_computed r known) (rename_expr r e)
    = rename_expr r (recover_comparisons_knowing known e).

(* The pass as the validation runs it, with nothing known. *)
Definition recover_known_comparisons_equivariant_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    recover_known_comparisons (rename_expr r e)
    = rename_expr r (recover_known_comparisons e).
