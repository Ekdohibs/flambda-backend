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


(* Simplifying the arithmetic of tagged immediates from the types of
   its arguments.

   This file is to be reviewed: the definition and the statements,
   against middle_end/flambda2/simplify/simplify_binary_primitive.ml
   (the functor [Binary_arith_like] at [Int_ops_for_binary_arith] of
   the tagged immediates). It follows FlambdaSimplifyPrimitivesSpec.v,
   which has the type of an argument ([argument_type]), the rewrites
   to a constant and to [Invalid], and what is assumed: strict
   satisfaction of the environment, and well-formed types for the
   provers of tagged immediates.

   What Simplify does. It asks
   [Provers.meet_equals_tagged_immediates] about the type of each
   argument.
   - Where both answer numbers (no more than ten each), it computes
     the operation on every pair; a pair on which the operation has no
     result (a division by zero) gives none. If there is no result the
     term is [Invalid]; if there is one, the primitive is that
     constant; if there are several, the primitive stays and its
     variable is given the type of the tagged immediates of those
     results.
   - Where one answers numbers and the other does not know, it looks
     each number up in a table of the operation
     ([op_lhs_unknown], [op_rhs_unknown]): the result is a number
     whatever the other argument is ([Exactly]), or is the other
     argument ([The_other_side]), or there is none ([Invalid]), or
     nothing is known. If every number that gives a result gives the
     same one of the first two, the primitive is that constant, or
     the other argument; if every number gives none, the term is
     [Invalid].
   - Where either answers that the type has no tagged immediate, the
     term is [Invalid].
   - Otherwise the primitive stays, and its variable is given the type
     of the tagged immediates.

   What is here. The numbers that an argument may be, by the answer of
   the prover ([possible_tagged_immediate]): those answered, or any in
   range if the prover does not know, or none. One statement,
   [let_int_arith_tagged_statement], has the four conclusions in terms
   of the operation on those numbers: if the operation has no result
   but one number on them, the primitive is that constant; if it has
   no result but its second argument, or its first, the primitive is
   that argument; if it has no result, the term is [Invalid]. It does
   not go by the table: [int_arith_identities_statement] says of each
   line of the table that is [Exactly], [The_other_side] or [Invalid]
   what the operation is there, which is what the statement asks. (So
   the statement has no limit of ten numbers, and holds where both
   answer numbers and the result is always one of the arguments, which
   Simplify does not look for.) The lines of the table that make
   another primitive are not here: the negation of the other side,
   which is a subtraction from zero, and its bitwise complement, a
   unary primitive that the model has not.

   The types of the variable where the primitive stays:
   [let_int_arith_tagged_these_statement] (the results, for any list
   that has them all) and [let_int_arith_tagged_unknown_statement]
   (the tagged immediates).

   Two examples: [example_add_constants_statement] and
   [example_add_zero_statement].

   Not here: the arithmetic of the naked integers, which is the same
   functor at other provers ([Provers.meet_naked_immediates],
   [meet_naked_number]); shifts; the unary arithmetic.

   The proofs are in FlambdaSimplifyArithProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaValueSpec FlambdaPrimitiveModel FlambdaPrimitiveSpec
  FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaEnvAliasesSpec FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvInvariantSpec FlambdaTypingEnvExamplesSpec
  FlambdaTypingEnvRefinementSpec FlambdaProvers
  FlambdaProversSpec FlambdaSimplifyPrimitivesSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* The numbers that an argument may be                                      *)
(* ======================================================================== *)

(* The width of the tagged immediates. *)
Definition tagged_width : Z := int_width Standard_tagged_immediate.

(* The numbers that an argument of the type may be, by what
   [Provers.meet_equals_tagged_immediates] answers: those answered; any
   number, if the prover does not know; none, if it answers that the
   type has no tagged immediate. They are in the range of the tagged
   immediates: the arithmetic has no value on another. *)
Definition possible_tagged_immediate (tenv : Typing_env.t)
    (ty : flambda_type) (i : Z) : Prop :=
  in_range tagged_width i = true
  ∧ match Provers.meet_equals_tagged_immediates tenv ty with
    | Some (Meet_shortcut.Known_result imms) => i ∈ imms
    | Some Meet_shortcut.Invalid => False
    | Some Meet_shortcut.Need_meet | None => True
    end.

(* An argument that is a tagged immediate in range, in a state that
   satisfies the environment, is one of those numbers. *)
Definition argument_possible_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars : env) (arg : simple)
    (ty : flambda_type) (i : Z),
    env_satisfies_strictly tenv statics vars ->
    argument_type tenv arg ty ->
    well_formed_input tenv ty ->
    eval_simple vars arg = Some (Value_const (Tagged_immediate i)) ->
    in_range tagged_width i = true ->
    possible_tagged_immediate tenv ty i.

(* ======================================================================== *)
(* The four conclusions                                                     *)
(* ======================================================================== *)

(* [Simplify_binary_primitive], for [Int_arith] on tagged immediates.
   Over the numbers that the two arguments may be:
   - if the operation has no result but one number, the primitive is
     that constant;
   - if it has no result but its second argument, the primitive is
     that argument; likewise for the first;
   - if it has no result, the term is [Invalid]. *)
Definition let_int_arith_tagged_statement : Prop :=
  ∀ (tenv : Typing_env.t) (x : variable) (declared : flambda_kind)
    (op : binary_int_arith_op) (arg1 arg2 : simple) (ty1 ty2 : flambda_type)
    (body : expr),
    argument_type tenv arg1 ty1 ->
    argument_type tenv arg2 ty2 ->
    well_formed_input tenv ty1 ->
    well_formed_input tenv ty2 ->
    let arith :=
      Prim (Binary (Int_arith Standard_tagged_immediate op)) [arg1; arg2]
    in
    let results (holds : Z -> Z -> option Z -> Prop) : Prop :=
      ∀ i1 i2 : Z,
        possible_tagged_immediate tenv ty1 i1 ->
        possible_tagged_immediate tenv ty2 i2 ->
        holds i1 i2 (eval_int_arith tagged_width op i1 i2)
    in
    (∀ result : Z,
      results (fun _ _ r => ∀ r' : Z, r = Some r' -> r' = result) ->
      expr_refines_under tenv (Let x declared arith body)
        (Let x declared (Simple (Const (Tagged_immediate result))) body))
    ∧ (results (fun _ i2 r => ∀ r' : Z, r = Some r' -> r' = i2) ->
       expr_refines_under tenv (Let x declared arith body)
         (Let x declared (Simple arg2) body))
    ∧ (results (fun i1 _ r => ∀ r' : Z, r = Some r' -> r' = i1) ->
       expr_refines_under tenv (Let x declared arith body)
         (Let x declared (Simple arg1) body))
    ∧ (results (fun _ _ r => r = None) ->
       expr_equivalent_under tenv (Let x declared arith body) Invalid).

(* The lines of the tables of [Int_ops_for_binary_arith]
   ([symmetric_op_one_side_unknown], [op_lhs_unknown],
   [op_rhs_unknown]) that are [The_other_side], [Exactly] or
   [Invalid]: what the operation is there, on a number in range, at
   the width of any kind of integer. *)
Definition int_arith_identities_statement : Prop :=
  ∀ (kind : standard_int) (j : Z),
    let width := int_width kind in
    in_range width j = true ->
    (* [Add]: zero on either side. *)
    eval_int_arith width Add 0 j = Some j
    ∧ eval_int_arith width Add j 0 = Some j
    (* [Sub]: zero on the right. *)
    ∧ eval_int_arith width Sub j 0 = Some j
    (* [Mul]: zero and one on either side. *)
    ∧ eval_int_arith width Mul 0 j = Some 0
    ∧ eval_int_arith width Mul j 0 = Some 0
    ∧ eval_int_arith width Mul 1 j = Some j
    ∧ eval_int_arith width Mul j 1 = Some j
    (* [And]: minus one and zero on either side. *)
    ∧ eval_int_arith width And (-1) j = Some j
    ∧ eval_int_arith width And j (-1) = Some j
    ∧ eval_int_arith width And 0 j = Some 0
    ∧ eval_int_arith width And j 0 = Some 0
    (* [Or]: minus one and zero on either side. *)
    ∧ eval_int_arith width Or (-1) j = Some (-1)
    ∧ eval_int_arith width Or j (-1) = Some (-1)
    ∧ eval_int_arith width Or 0 j = Some j
    ∧ eval_int_arith width Or j 0 = Some j
    (* [Xor]: zero on either side. *)
    ∧ eval_int_arith width Xor 0 j = Some j
    ∧ eval_int_arith width Xor j 0 = Some j
    (* [Div] and [Mod]: zero and one on the right; for [Mod], minus one
       on the right where it is signed. *)
    ∧ (∀ s : signedness,
        eval_int_arith width (Div s) j 0 = None
        ∧ eval_int_arith width (Mod s) j 0 = None
        ∧ eval_int_arith width (Div s) j 1 = Some j
        ∧ eval_int_arith width (Mod s) j 1 = Some 0)
    ∧ eval_int_arith width (Mod Signed) j (-1) = Some 0.

(* ======================================================================== *)
(* Where the primitive stays                                                *)
(* ======================================================================== *)

(* The variable is given the type of the tagged immediates of the
   results, where both provers answer numbers. Stated for any list
   that has every result. *)
Definition let_int_arith_tagged_these_statement : Prop :=
  ∀ (tenv : Typing_env.t) (x : variable) (declared : flambda_kind)
    (op : binary_int_arith_op) (arg1 arg2 : simple) (ty1 ty2 : flambda_type)
    (results : list Z) (body body' : expr),
    typing_env_well_formed tenv = true ->
    Typing_env.mem None tenv (Name_var x) = false ->
    argument_type tenv arg1 ty1 ->
    argument_type tenv arg2 ty2 ->
    well_formed_input tenv ty1 ->
    well_formed_input tenv ty2 ->
    (∀ i1 i2 r : Z,
      possible_tagged_immediate tenv ty1 i1 ->
      possible_tagged_immediate tenv ty2 i2 ->
      eval_int_arith tagged_width op i1 i2 = Some r ->
      r ∈ results) ->
    let arith :=
      Prim (Binary (Int_arith Standard_tagged_immediate op)) [arg1; arg2]
    in
    expr_refines_under
      (Typing_env.replace_equation
         (Typing_env.add_variable_definition tenv x Value Name_mode.normal)
         (Name_var x) (these_tagged_immediates results))
      body body' ->
    expr_refines_under tenv (Let x declared arith body)
      (Let x declared arith body').

(* Otherwise it is given the type of the tagged immediates. *)
Definition let_int_arith_tagged_unknown_statement : Prop :=
  ∀ (tenv : Typing_env.t) (x : variable) (declared : flambda_kind)
    (op : binary_int_arith_op) (arg1 arg2 : simple) (body body' : expr),
    typing_env_well_formed tenv = true ->
    Typing_env.mem None tenv (Name_var x) = false ->
    let arith :=
      Prim (Binary (Int_arith Standard_tagged_immediate op)) [arg1; arg2]
    in
    expr_refines_under
      (Typing_env.replace_equation
         (Typing_env.add_variable_definition tenv x Value Name_mode.normal)
         (Name_var x) any_tagged_immediate)
      body body' ->
    expr_refines_under tenv (Let x declared arith body)
      (Let x declared arith body').

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

(* The variables are those of FlambdaTypingEnvExamplesSpec.v. *)

(* [let x = 3 in let y = x + 4 in apply_cont k y], closed but for its
   continuation, from the environment that knows nothing: [x] is made
   equal to 3; both provers answer one number, and [y] is made equal
   to 7. *)
Definition example_add_constants_statement : Prop :=
  ∀ k : continuation,
    expr_refines
      (Let example_x Value (Simple (Const (Tagged_immediate 3)))
         (Let example_y Value
            (Prim (Binary (Int_arith Standard_tagged_immediate Add))
               [Var example_x; Const (Tagged_immediate 4)])
            (Apply_cont (Apply_cont_expr k [Var example_y] None))))
      (Let example_x Value (Simple (Const (Tagged_immediate 3)))
         (Let example_y Value (Simple (Const (Tagged_immediate 7)))
            (Apply_cont
               (Apply_cont_expr k [Const (Tagged_immediate 7)] None)))).

(* [let y = x + 0 in apply_cont k y], under the environment in which
   [x] is defined at the kind value and nothing more is known of it:
   the prover does not know for [x] and answers zero for the constant,
   and [y] is made equal to [x]. *)
Definition example_add_zero_statement : Prop :=
  ∀ k : continuation,
    expr_refines_under
      (Typing_env.add_definitions_of_params Typing_env.create
         [(example_x, Value)])
      (Let example_y Value
         (Prim (Binary (Int_arith Standard_tagged_immediate Add))
            [Var example_x; Const (Tagged_immediate 0)])
         (Apply_cont (Apply_cont_expr k [Var example_y] None)))
      (Let example_y Value (Simple (Var example_x))
         (Apply_cont (Apply_cont_expr k [Var example_x] None))).
