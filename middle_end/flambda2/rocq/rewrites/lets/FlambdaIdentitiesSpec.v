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



(* The identities of integer arithmetic that Simplify applies, and the
   comparison of a variable with itself: a normalisation that refines.

   This file is to be reviewed: [apply_identities] and what it is made
   of, which are extracted and run by the validation of
   middle_end/flambda2/tests/semantics/ as one of the normalisations of
   the term before Simplify (FlambdaValidateSpec.v), and the statement.

   Simplify computes an arithmetic operation or a shift of two integers
   that it knows. Where it knows one of the two only, it applies an
   identity of the operation, if the integer it knows has one
   ([Simplify_binary_primitive]: [Int_ops_for_binary_arith], with
   [symmetric_op_one_side_unknown], [op_lhs_unknown] and
   [op_rhs_unknown], and [Int_ops_for_binary_shift]). The result is the
   other argument, a constant, the opposite of the other argument,
   written [0 - x], or its complement. With [x] for the argument that
   is not known:

   - [x + 0], [0 + x], [x - 0], [x * 1], [1 * x], [x / 1], [x land -1],
     [-1 land x], [x lor 0], [0 lor x], [x lxor 0], [0 lxor x] and a
     shift of [x] by 0 are [x];
   - [x * 0], [0 * x], [x land 0], [0 land x], [x mod 1], the signed
     [x mod -1] and a shift of 0 by [x] are 0;
   - [x lor -1], [-1 lor x] and the arithmetic shift right of -1 by [x]
     are -1;
   - [x * -1], [-1 * x] and the signed [x / -1] are [0 - x], which is
     also what [0 - x] is made into: that one is left as it is;
   - [x lxor -1] and [-1 lxor x] are the complement of [x], the unary
     [Int_arith] with [Not].

   The functions below are Simplify's, with its names, on integers. Its
   divisions and remainders by 0, which it makes [Invalid], are not
   here: they are left as they are. Simplify has no identity for an
   operation on a variable and itself ([x - x], [x lxor x], [x lor x]),
   and there is none here.

   Simplify also computes [Phys_equal] of two arguments that it proves
   to be the same value or different ones
   ([Simplify_binary_primitive.simplify_phys_equal], with
   [Provers.prove_physical_equality]). One of its proofs needs nothing
   but the term: two arguments that are the same variable are the same
   value, so [Phys_eq] of them is 1 and [Phys_neq] is 0. That one is
   here. (It says the same of a symbol and itself, and that two symbols
   are different; neither is here, as [Phys_equal] has no value on two
   allocated values, FlambdaPrimitiveSpec.eval_phys_equal.)

   This normalisation uses no types: it does the same where the integer
   that is known is a constant of the term, of the kind of the
   operation, and the other argument is not a constant. An operation on
   two constants is left alone: computing it is another normalisation
   (FlambdaNormaliseSpec.fold_constants). A [Let] of such an operation
   becomes a [Let] of what the identity gives ([identity_named]), with
   the kind it had. Nothing else is rewritten, and the [Let] stays, as in
   FlambdaBlockLoadSpec.v, which this file follows: a handler of a
   non-recursive continuation is gone into; the handlers of recursive
   ones and the bodies of code are left as they are.

   The result refines the term ([apply_identities_refines_statement]).
   Each identity is true of every integer in the range of the kind,
   wrapping included, and an operation has a value on integers of its
   kind, in range, only (FlambdaPrimitiveSpec.int_of_value): so where
   the operation has a value, what replaces it has the same. It is not
   an equivalence: where the other argument is not an integer of the
   kind the operation has no value, and a constant, or the other
   argument, has one; and [Phys_equal] has no value on a variable that
   holds an allocated value.

   The pass commutes with renaming
   ([apply_identities_equivariant_statement]): it binds no name and
   makes none, and it looks at a name only to see that the two
   arguments of [Phys_equal] are the same variable, which a renaming,
   being injective, keeps true or false.

   The proofs are in FlambdaIdentitiesProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaValueSpec FlambdaPrimitiveModel FlambdaPrimitiveSpec
  FlambdaSemanticsSpec FlambdaNominalSpec.
Import ListNotations.
Open Scope Z_scope.

(* ======================================================================== *)
(* The identities                                                           *)
(* ======================================================================== *)

(* What an operation with one argument known comes to:
   [Simplify_binary_primitive.binary_arith_outcome_for_one_side_only],
   without [Invalid] and without the negation of floats. *)
Inductive outcome : Type :=
  | Exactly (i : Z)
  | The_other_side
  | Negation_of_the_other_side
  | Bitwise_not_of_the_other_side
  | Cannot_simplify.

(* [symmetric_op_one_side_unknown]: the operations that give the same
   whichever side the known integer is on. *)
Definition symmetric_op_one_side_unknown (op : binary_int_arith_op)
    (this_side : Z) : outcome :=
  match op with
  | Add => if this_side =? 0 then The_other_side else Cannot_simplify
  | Mul =>
    if this_side =? 0 then Exactly 0
    else if this_side =? 1 then The_other_side
    else if this_side =? -1 then Negation_of_the_other_side
    else Cannot_simplify
  | And =>
    if this_side =? -1 then The_other_side
    else if this_side =? 0 then Exactly 0
    else Cannot_simplify
  | Or =>
    if this_side =? -1 then Exactly (-1)
    else if this_side =? 0 then The_other_side
    else Cannot_simplify
  | Xor =>
    if this_side =? 0 then The_other_side
    else if this_side =? -1 then Bitwise_not_of_the_other_side
    else Cannot_simplify
  | Sub | Div _ | Mod _ => Cannot_simplify
  end.

Definition is_signed (s : signedness) : bool :=
  match s with
  | Signed => true
  | Unsigned => false
  end.

(* [op_lhs_unknown]: the right argument is known. *)
Definition op_lhs_unknown (op : binary_int_arith_op) (rhs : Z) : outcome :=
  match op with
  | Add | Mul | And | Or | Xor => symmetric_op_one_side_unknown op rhs
  | Sub => if rhs =? 0 then The_other_side else Cannot_simplify
  | Div s =>
    if rhs =? 1 then The_other_side
    else if (rhs =? -1) && is_signed s then Negation_of_the_other_side
    else Cannot_simplify
  | Mod s =>
    if rhs =? 1 then Exactly 0
    else if (rhs =? -1) && is_signed s then Exactly 0
    else Cannot_simplify
  end.

(* [op_rhs_unknown]: the left argument is known. *)
Definition op_rhs_unknown (op : binary_int_arith_op) (lhs : Z) : outcome :=
  match op with
  | Add | Mul | And | Or | Xor => symmetric_op_one_side_unknown op lhs
  | Sub => if lhs =? 0 then Negation_of_the_other_side else Cannot_simplify
  | Div _ | Mod _ => Cannot_simplify
  end.

(* The same of the shifts: the amount is known, or the integer that is
   shifted. *)
Definition shift_lhs_unknown (op : int_shift_op) (rhs : Z) : outcome :=
  if rhs =? 0 then The_other_side else Cannot_simplify.

Definition shift_rhs_unknown (op : int_shift_op) (lhs : Z) : outcome :=
  match op with
  | Lsl | Lsr => if lhs =? 0 then Exactly 0 else Cannot_simplify
  | Asr =>
    if lhs =? 0 then Exactly 0
    else if lhs =? -1 then Exactly (-1)
    else Cannot_simplify
  end.

(* ======================================================================== *)
(* Applying one                                                             *)
(* ======================================================================== *)

(* The integer that an argument is, where it is a constant of the kind,
   in range. *)
Definition int_of_simple (kind : standard_int) (s : simple) : option Z :=
  match s with
  | Const c => int_of_value kind (Value_const c)
  | Var _ | Symbol _ => None
  end.

Definition is_const (s : simple) : bool :=
  match s with
  | Const _ => true
  | Var _ | Symbol _ => false
  end.

(* What an outcome makes of the operation, at a kind, [other] being the
   argument that is not known. *)
Definition named_of_outcome (kind : standard_int) (o : outcome)
    (other : simple) : option named :=
  match o with
  | Exactly i => Some (Simple (Const (const_of_int kind i)))
  | The_other_side => Some (Simple other)
  | Negation_of_the_other_side =>
    Some
      (Prim (Binary (Int_arith kind Sub))
         [Const (const_of_int kind 0); other])
  | Bitwise_not_of_the_other_side =>
    Some (Prim (Unary (Unary_int_arith kind Not)) [other])
  | Cannot_simplify => None
  end.

(* What [Phys_equal] is of a value and itself: a naked immediate, 1 for
   [Phys_eq] and 0 for [Phys_neq]. *)
Definition same_value_const (op : equality_comparison) : const :=
  Naked_immediate
    match op with
    | Phys_eq => 1
    | Phys_neq => 0
    end.

(* What one argument known makes of a binary primitive: an identity of
   an arithmetic operation or of a shift, where one argument is a
   constant of the right kind and the other is not a constant; and what
   [Phys_equal] is of a variable and itself. The amount of a shift is a
   naked immediate. *)
Definition identity_of_binary (p : binary_primitive) (arg1 arg2 : simple)
    : option named :=
  match p with
  | Int_arith kind op =>
    match int_of_simple kind arg1, int_of_simple kind arg2 with
    | Some lhs, None =>
      if is_const arg2 then None
      else named_of_outcome kind (op_rhs_unknown op lhs) arg2
    | None, Some rhs =>
      if is_const arg1 then None
      else named_of_outcome kind (op_lhs_unknown op rhs) arg1
    | _, _ => None
    end
  | Int_shift kind op =>
    match
      int_of_simple kind arg1, int_of_simple Standard_naked_immediate arg2
    with
    | Some lhs, None =>
      if is_const arg2 then None
      else named_of_outcome kind (shift_rhs_unknown op lhs) arg2
    | None, Some rhs =>
      if is_const arg1 then None
      else named_of_outcome kind (shift_lhs_unknown op rhs) arg1
    | _, _ => None
    end
  | Phys_equal op =>
    match arg1, arg2 with
    | Var x, Var y =>
      if Pos.eqb x y then Some (Simple (Const (same_value_const op)))
      else None
    | _, _ => None
    end
  | Array_load _ _ _ | String_or_bigstring_load _ | Int_comp _ _ => None
  end.

(* A primitive applied to arguments, as a binary primitive applied to
   two. *)
Definition binary_applied (p : prim) (args : list simple)
    : option (binary_primitive * simple * simple) :=
  match p, args with
  | Binary p, [arg1; arg2] => Some (p, arg1, arg2)
  | _, _ => None
  end.

Definition identity_named (d : named) : named :=
  match d with
  | Prim p args =>
    match binary_applied p args with
    | Some (p, arg1, arg2) =>
      match identity_of_binary p arg1 arg2 with
      | Some d' => d'
      | None => d
      end
    | None => d
    end
  | _ => d
  end.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

Fixpoint apply_identities (e : expr) : expr :=
  match e with
  | Let x kind d body =>
    Let x kind (identity_named d) (apply_identities body)
  | Let_static bindings body => Let_static bindings (apply_identities body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (apply_identities_in_handler h)
      (apply_identities body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (apply_identities body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with apply_identities_in_handler (h : continuation_handler)
    : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (apply_identities handler)
  end.

Definition apply_identities_refines_statement : Prop :=
  ∀ e : expr, expr_refines e (apply_identities e).

(* The pass commutes with renaming, and so does what it makes of one
   defining expression. *)
Definition identity_named_equivariant_statement : Prop :=
  ∀ (r : renaming) (d : named),
    identity_named (rename_named r d) = rename_named r (identity_named d).

Definition apply_identities_equivariant_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    apply_identities (rename_expr r e) = rename_expr r (apply_identities e).
