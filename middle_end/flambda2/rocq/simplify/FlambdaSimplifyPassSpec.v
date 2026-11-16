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


(* A function on terms that carries a typing environment down a term
   and rewrites it from the types: the rewrites of primitives, put
   together.

   This file is to be reviewed: every definition, and the
   statements. The function is meant to be extracted and run by the
   validation of middle_end/flambda2/tests/semantics/; nothing runs it
   yet, and it has not been extracted.

   [simplify_expr tenv e] goes down the term [e] with the typing
   environment [tenv], which it extends at each binder as Simplify
   does, and at each [Let] asks the provers about the types of the
   arguments, as the statements of FlambdaSimplifyPrimitivesSpec.v,
   FlambdaSimplifyProjectionsSpec.v, FlambdaSimplifyBoxingSpec.v and
   FlambdaSimplifyArithSpec.v say that Simplify may. Its answer refines
   the term under the environment
   ([simplify_expr_refines_under_statement]); from the environment that
   knows nothing, it refines the term outright
   ([simplify_by_types_refines_statement]), with no premise on the
   term.

   It is not Simplify, and is not a definition of what Simplify does.
   It is one function that is right, made of rewrites that Simplify
   makes, for the validator to bring a term before Simplify nearer to
   the term after. What it does:
   - At a [Let] of a [Simple]: the [Simple] is replaced by its
     canonical element, and the variable is made an alias of it
     ([bind_simple]).
   - At a [Let] of a primitive of the heap (an allocation with an
     identity, a store): nothing. The [Let] is left as it is, and its
     variable is defined at the kind that the [Let] declares.
   - At a [Let] of another primitive: the arguments are looked up
     ([simplify_simples], FlambdaSimplifyPrimitivesSpec.v); if their
     types are not of the kinds that
     the primitive takes, the term is [Invalid]; otherwise
     [simplify_unary], [simplify_binary] or [simplify_variadic] asks
     the provers, and the [Let] becomes [Invalid], or a [Let] of a
     [Simple] (a constant, an argument, or what a prover answers),
     which is then treated as above; or the primitive stays, with its
     arguments canonical, and its variable is defined with the type
     that Simplify gives it ([Make_block], [Tag_immediate],
     [Box_number], the arithmetic of tagged immediates) or with the
     kind of its result and nothing more.
   - At a [Let] of a projection of a function slot: the same, with
     the prover of that projection. At a [Let] of a set of closures,
     or of a closure of a set, the variable is defined with the type
     of the closure ([set_of_closures_type], [closure_of_set_type],
     FlambdaSimplifyApplySpec.v): its code ids, and for each value
     slot the type of the [Simple] that it holds; or at the kind
     value, where a value slot holds a [Simple] that the environment
     does not know.
   - At a [Let] of the projection of a value slot: where the prover
     of that projection answers a [Simple], asked about the type of
     the closure at the kind that the [Let] declares
     ([resolve_value_slot], on [simplify_project_value_slot] of
     FlambdaSimplifyValueSlotsSpec.v), the projection becomes that
     [Simple], which is then treated as above; otherwise the
     projection stays, and nothing is known of its variable but the
     kind that the [Let] declares (below). (The model's value slots
     have no kind; Simplify asks at the kind of the slot.)
   - A call is given to [simplify_apply] (FlambdaSimplifyApplySpec.v):
     an indirect call whose callee is proved, from its type, to be a
     closure of one piece of code becomes a direct call of that code,
     and the callee and the arguments are replaced by their canonical
     elements.
   - Under a [Let_static] its symbols are defined and given the types
     of their constants: a block, a boxed number, a string, a closure
     ([add_static_bindings], FlambdaSimplifyStaticsSpec.v). The body
     of each piece of code that it binds is rewritten too
     ([simplify_code]), under the environment of the body of the
     [Let_static] with its variables out of scope, [my_closure]
     defined at the kind value and the parameters at the kinds of the
     arity of the code ([code_env_in], FlambdaSimplifyCodeEnvSpec.v).
     The symbols keep their types there, those of the group included;
     a variable of the place where the code is bound stays known to
     the types and is not put into the body. The body is rewritten
     only if neither it nor what it becomes binds something static: a
     call of code whose body does is undefined
     (FlambdaSemanticsSpec.binds_statics).
   - The body of a [Let_cont] is gone into, and so is each handler
     ([simplify_handler]), under the environment of the [Let_cont]
     with the parameters of the handler defined at the kinds that the
     handler gives them ([handler_env],
     FlambdaSimplifyHandlersSpec.v). A handler of a recursive
     [Let_cont] is kept as it was if the rewrite would change whether
     it binds something static, as such a handler may bind nothing
     static. The handler of a non-recursive [Let_cont] is rewritten
     after the body, and under more where what the body has become
     applies the continuation at one place (below).
   - The arguments of an [Apply_cont] are replaced by their canonical
     elements. A [Switch] whose scrutinee has a constant for
     canonical element becomes the [Apply_cont] of its arm.
   - Where it knows nothing of the defining expression of a [Let] (a
     [Simple] that the environment does not know, the projection of a
     value slot that no prover answers, a test that fails on the way)
     it defines the variable at the kind that the [Let] declares and
     goes on into the body ([declare_variable]): the machine has no
     step at a [Let] unless the value is of that kind. Where the
     variable is one that the environment has already, or the
     environment fails its test of well-formedness, it leaves the term
     from there down as it is.

   The handler of a continuation under what is known where the
   continuation is applied. At a non-recursive [Let_cont] the body is
   rewritten first. Where what it has become applies the continuation
   at one place ([find_use]: at the end of the [Let]s and the
   non-recursive [Let_cont]s that it begins with, by an [Apply_cont]
   or by one arm of a [Switch]), the handler is rewritten under the
   environment of that application: the environment at the
   application ([let_env] at each [Let] on the way, [arm_env] where
   the application is an arm of a [Switch]) with the parameters
   defined and made aliases of the arguments ([use_env]). What the
   handler becomes is kept if it mentions none of the binders between
   the [Let_cont] and the application ([binders_not_free]); if it
   mentions one, and where there is no such application, the handler
   is rewritten as the bullet above says, with nothing known of its
   parameters but their kinds. FlambdaSimplifyUsesSpec.v says what
   Simplify does there, where the model parts from it, and has the
   statements for a [Let_cont] whose body is the application;
   [find_use_refines_statement] here is for one whose body binds
   something first. [rewrite_handler_at_use] is that step alone, with
   a theorem of its own and one example. The function that prunes
   (FlambdaSimplifyPruningSpec.v) makes the same step.

   The applications gathered. [gather_uses] gathers the applications
   of a continuation at the end of the body and in the handlers of
   the continuations that the body defines, each with the environment
   that the function has there; the handler may be rewritten under
   any environment that each of them admits ([application_admits],
   [gather_uses_refines_statement]). Simplify gathers the uses of a
   continuation in the same places, as it goes down the body.

   One application gathered. Where [find_use] finds no application
   and [gather_uses] gathers one and no more (one in the handler of a
   continuation that the body defines, say), the handler is rewritten
   under the environment made for that application
   ([find_gathered_use]: [use_env] of it, the parameters made aliases
   of the arguments that may occur in terms at the [Let_cont]), and
   kept on the same condition on the binders passed ([find_one_use]
   is [find_use] and then this). Simplify gives the handler the
   environment of its uses where the continuation has at most one,
   whatever the flags (simplify/join_points.ml, [should_do_join]).
   The function that prunes makes the same step.

   Several applications. There is the statement above for them and no
   rewrite yet: where two or more are gathered no environment is
   computed, and none is until a join is called. Simplify gives the
   handler the join of the environments of its uses there, under
   -flambda2-join-points.

   What Simplify does and this does not:
   - There are no joins of the environments at the uses of a
     continuation, which Simplify has for the parameters of a
     non-recursive [Let_cont] and for the invariant parameters of a
     recursive one, and no subkinds (FlambdaSimplifyHandlersSpec.v
     says where the two part). So [simplify_expr] knows nothing of a
     parameter but its kind in every handler of a recursive
     [Let_cont], and in the handler of a non-recursive one whose
     continuation is applied at more than one place, or at a place
     that neither [find_use] nor [gather_uses] reaches: under a
     [Let_static] or a recursive [Let_cont], or by a call that
     returns or raises to it.
   - In the body of code nothing is known of a parameter but its
     kind, nor of [my_closure] but that it is a value: Simplify gives
     [my_closure] the type of the closure. The code is rewritten
     where it is bound, under what is known just after its group;
     Simplify rewrites it where a set of closures uses it
     (FlambdaSimplifyCodeEnvSpec.v says where the two part).
   - Nothing goes through the meet. Where Simplify meets the type of
     an argument with a shape ([Simplify_common.simplify_projection]:
     the unboxing primitives, the loads and projections that a prover
     does not answer) or adds an equation by a meet (the relational
     primitives [Is_int], [Get_tag], [Is_null]), the function asks
     the provers that the statements are about, in the environment
     before the [Let], and otherwise records the kind of the result
     only. No arm of a [Switch] is removed from the type of the
     scrutinee.
   - Nothing is lifted; there are no common subexpressions, no
     inlining, no unboxing of parameters. The value slots of a set of
     closures are left as they are written (Simplify replaces them by
     their canonical elements).
   - Of the unary primitives, [Is_int], [Is_null], [Get_tag], a
     [Block_load] that is not marked [Mutable], and the tagging,
     untagging, boxing and unboxing of a number are looked at; of the
     binary ones, the arithmetic and the comparison of tagged
     immediates, [Phys_equal], and the comparisons of integers of the
     other kinds that yield a boolean, of which the type of the two
     booleans is recorded and nothing more (below). For every other
     primitive only the kind of its result is recorded. (Simplify
     makes a [Block_load] marked [Mutable] of a constant [Invalid],
     [simplify_mutable_block_load]; this leaves it.)
   - Of that arithmetic, of the tables for one side known
     ([op_lhs_unknown], [op_rhs_unknown]) the lines that make another
     primitive are left out. The comparison is computed on every
     pair of numbers, however many each argument may be: Simplify
     gives up above ten on a side, as [simplify_int_arith_tagged]
     does here, so this rewrites a comparison that Simplify leaves.
     A comparison of integers of any kind that yields a boolean and
     is not decided, and a [Phys_equal] that is not decided, give
     their variable the type of the two booleans, as Simplify does;
     of a comparison that yields the result of a compare function
     only the kind is recorded (Simplify records -1, 0 or 1).
   - The tests that it makes before a rewrite (that the environment
     passes [typing_env_well_formed], that the variable of a [Let] is
     new to it, that the types of the environment and the types
     asked about are well formed where a prover of tagged immediates
     or the prover of physical equality is asked) are not Simplify's:
     they are the premises of the statements, tested so that the
     function needs none. The same of the tests of [code_env_in]:
     the arity of the code has as many kinds as the code has
     parameters, and [my_closure] and the parameters are different
     variables that the environment does not have, each defined in
     an environment that is well formed; and of those of
     [handler_env], which are the same for the parameters of a
     handler. Where a test fails the
     rewrite that it guards is
     not made: the function goes on knowing less, or leaves the term
     from there down, or the code, as it is. So a failure costs
     rewrites and not correctness; that the tests pass on the
     environments that the function builds, for a term whose binders
     are all different, is not stated here.

   Eighteen terms on which the function is computed are at the end:
   what it answers is stated as an equation, with no premise. A
   nineteenth is for [rewrite_handler_at_use].

   That the function commutes with renaming ([simplify_expr] of the
   renamed environment and the renamed term is the renamed answer),
   as the other rewrites of the validation do
   (FlambdaValidatorEquivarianceSpec.v), is in
   FlambdaSimplifyPassRenamingSpec.v, with the equations of its
   parts. The functions of the section "Where a continuation is
   applied" and [rewrite_handler_at_use] have equations of their own
   there.

   Nothing is assumed: the two statements of [simplify_expr] have no
   premise, and the others none but what they say.

   The proofs are in FlambdaSimplifyPassProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaValueSpec FlambdaPrimitiveModel FlambdaPrimitiveSpec
  FlambdaHeapSpec FlambdaSemanticsSpec FlambdaNormaliseSpec
  FlambdaUnusedContsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypeWellFormedSpec FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaTypingEnv
  FlambdaTypingEnvSpec FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvExamplesSpec FlambdaTypingEnvRefinementSpec
  FlambdaProvers FlambdaProversSpec FlambdaSimplifyPrimitivesSpec
  FlambdaSimplifyProjectionsSpec FlambdaSimplifyBoxingSpec
  FlambdaSimplifyArithSpec FlambdaSimplifyApplySpec
  FlambdaSimplifyValueSlotsSpec FlambdaSimplifyStaticsSpec
  FlambdaSimplifyCodeEnvSpec FlambdaSimplifyHandlersSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* What is decided of a defining expression                                 *)
(* ======================================================================== *)

Inductive outcome : Type :=
  (* The [Let] is [Invalid]. *)
  | Becomes_invalid
  (* The defining expression becomes this [Simple]. *)
  | Becomes_simple (s : simple)
  (* It stays, and its variable has this type. *)
  | Stays_with_type (ty : flambda_type)
  (* It stays, and its variable is of this kind. *)
  | Stays_with_kind (kind : flambda_kind)
  (* Nothing is known. *)
  | Not_known.

(* A boolean, as the naked immediate that a primitive answers. *)
Definition naked_bool_const (b : bool) : simple :=
  Const (Naked_immediate (if b then 1 else 0)).

(* The outcome on the answer of a meet shortcut. *)
Definition of_bool_shortcut (answer : option (Meet_shortcut.t bool))
    (otherwise : outcome) : outcome :=
  match answer with
  | Some (Meet_shortcut.Known_result b) => Becomes_simple (naked_bool_const b)
  | Some Meet_shortcut.Invalid => Becomes_invalid
  | Some Meet_shortcut.Need_meet | None => otherwise
  end.

Definition of_simple_shortcut (answer : option (Meet_shortcut.t simple))
    (otherwise : outcome) : outcome :=
  match answer with
  | Some (Meet_shortcut.Known_result s) => Becomes_simple s
  | Some Meet_shortcut.Invalid => Becomes_invalid
  | Some Meet_shortcut.Need_meet | None => otherwise
  end.

(* A unary primitive, on its argument and the type of the argument. A
   primitive that has no line of its own stays, and its variable is of
   the kind of its result. *)
Definition simplify_unary (tenv : Typing_env.t) (p : unary_primitive)
    (arg : simple) (ty : flambda_type) : outcome :=
  let unknown := Stays_with_kind (result_kind_of_unary_primitive p) in
  match p with
  | Is_int false =>
    match Provers.prove_is_int tenv ty with
    | Some (Proof_of_property.Proved b) => Becomes_simple (naked_bool_const b)
    | Some Proof_of_property.Unknown | None => unknown
    end
  | Is_int true =>
    of_bool_shortcut (Provers.meet_is_int_variant_only tenv ty) unknown
  | Is_null => of_bool_shortcut (Provers.meet_is_null tenv ty) unknown
  | Get_tag =>
    match Provers.prove_is_int tenv ty with
    | Some (Proof_of_property.Proved true) => Becomes_invalid
    | Some (Proof_of_property.Proved false)
    | Some Proof_of_property.Unknown | None =>
      match Provers.prove_get_tag tenv ty with
      | Some (Proof_of_property.Proved [tag]) =>
        Becomes_simple (Const (Naked_immediate tag))
      | Some (Proof_of_property.Proved _)
      | Some Proof_of_property.Unknown | None =>
        unknown
      end
    end
  | Block_load _ (Immutable | Immutable_unique) field =>
    of_simple_shortcut
      (Provers.meet_block_field_simple tenv Name_mode.normal Value ty
         (Z.to_nat field))
      unknown
  | Tag_immediate => Stays_with_type (tag_immediate ty)
  | Box_number kind => Stays_with_type (box_number kind ty Alloc_mode.Heap)
  | Untag_immediate =>
    if types_well_formed_b tenv
    then
      of_simple_shortcut
        (Provers.meet_tagging_of_simple tenv Name_mode.normal
           (alias_type_of Value arg))
        unknown
    else unknown
  | Unbox_number kind =>
    of_simple_shortcut
      (Provers.meet_boxed_number_containing_simple kind tenv Name_mode.normal
         (alias_type_of Value arg))
      unknown
  | _ => unknown
  end.

(* ======================================================================== *)
(* The arithmetic of tagged immediates                                      *)
(* ======================================================================== *)

(* [Simplify_binary_primitive.binary_arith_outcome_for_one_side_only],
   without the outcomes that make another primitive: what is known of
   the result from one argument alone. [No_result] is the compiler's
   [Invalid]. *)
Inductive one_side_outcome : Type :=
  | Exactly (result : Z)
  | The_other_side
  | No_result
  | Cannot_simplify.

(* [Int_ops_for_binary_arith.symmetric_op_one_side_unknown]. Where the
   compiler answers the negation or the bitwise complement of the other
   side, this answers [Cannot_simplify]. *)
Definition symmetric_op_one_side_unknown (op : binary_int_arith_op)
    (this_side : Z) : one_side_outcome :=
  match op with
  | Add => if this_side =? 0 then The_other_side else Cannot_simplify
  | Mul =>
    if this_side =? 0 then Exactly 0
    else if this_side =? 1 then The_other_side
    else Cannot_simplify
  | And =>
    if this_side =? -1 then The_other_side
    else if this_side =? 0 then Exactly 0
    else Cannot_simplify
  | Or =>
    if this_side =? -1 then Exactly (-1)
    else if this_side =? 0 then The_other_side
    else Cannot_simplify
  | Xor => if this_side =? 0 then The_other_side else Cannot_simplify
  | Sub | Div _ | Mod _ => Cannot_simplify
  end.

(* [Int_ops_for_binary_arith.op_lhs_unknown]: the right argument is
   known. *)
Definition op_lhs_unknown (op : binary_int_arith_op) (rhs : Z)
    : one_side_outcome :=
  match op with
  | Add | Mul | And | Or | Xor => symmetric_op_one_side_unknown op rhs
  | Sub => if rhs =? 0 then The_other_side else Cannot_simplify
  | Div _ =>
    if rhs =? 0 then No_result
    else if rhs =? 1 then The_other_side
    else Cannot_simplify
  | Mod s =>
    if rhs =? 0 then No_result
    else if rhs =? 1 then Exactly 0
    else
      match s with
      | Signed => if rhs =? -1 then Exactly 0 else Cannot_simplify
      | Unsigned => Cannot_simplify
      end
  end.

(* [Int_ops_for_binary_arith.op_rhs_unknown]: the left argument is
   known. *)
Definition op_rhs_unknown (op : binary_int_arith_op) (lhs : Z)
    : one_side_outcome :=
  match op with
  | Add | Mul | And | Or | Xor => symmetric_op_one_side_unknown op lhs
  | Sub | Div _ | Mod _ => Cannot_simplify
  end.

(* The possible results of [Binary_arith_like.simplify]: a number, or
   the other argument. *)
Inductive possible_result : Type :=
  | Result_exactly (result : Z)
  | Result_other_side.

Definition possible_result_eqb (r r' : possible_result) : bool :=
  match r, r' with
  | Result_exactly i, Result_exactly i' => i =? i'
  | Result_other_side, Result_other_side => true
  | Result_exactly _, Result_other_side
  | Result_other_side, Result_exactly _ =>
    false
  end.

(* The local function [only_one_side_known] of
   [Binary_arith_like.simplify]: the possible results for the numbers
   that one argument may be, or [None] if nothing is known for one of
   them. *)
Fixpoint one_side_results (table : Z -> one_side_outcome) (nums : list Z)
    : option (list possible_result) :=
  match nums with
  | [] => Some []
  | i :: nums =>
    let* results := one_side_results table nums in
    match table i with
    | Exactly result => Some (Result_exactly result :: results)
    | The_other_side => Some (Result_other_side :: results)
    | No_result => Some results
    | Cannot_simplify => None
    end
  end.

(* The results of the operation on every pair of numbers that the two
   arguments may be, where both are known. *)
Definition both_sides_results (op : binary_int_arith_op) (nums1 nums2 : list Z)
    : list possible_result :=
  flat_map
    (fun i1 =>
       flat_map
         (fun i2 =>
            match eval_int_arith tagged_width op i1 i2 with
            | Some result => [Result_exactly result]
            | None => []
            end)
         nums2)
    nums1.

(* The numbers among possible results. *)
Fixpoint exact_results (results : list possible_result) : list Z :=
  match results with
  | [] => []
  | Result_exactly result :: results => result :: exact_results results
  | Result_other_side :: results => exact_results results
  end.

(* The local function [check_possible_results] of
   [Binary_arith_like.simplify]: no result is [Invalid]; one result is
   the constant or the other argument; several numbers are the type of
   those numbers; otherwise the type of the tagged immediates. *)
Definition check_possible_results (results : list possible_result)
    (other_side : simple) : outcome :=
  match results with
  | [] => Becomes_invalid
  | result :: rest =>
    if forallb (possible_result_eqb result) rest
    then
      match result with
      | Result_exactly i => Becomes_simple (Const (Tagged_immediate i))
      | Result_other_side => Becomes_simple other_side
      end
    else if forallb
              (fun r =>
                 match r with
                 | Result_exactly _ => true
                 | Result_other_side => false
                 end)
              results
    then Stays_with_type (these_tagged_immediates (exact_results results))
    else Stays_with_type any_tagged_immediate
  end.

(* [Binary_arith_like.max_num_possible_results]. *)
Definition max_num_possible_results : nat := 10.

(* [Binary_arith_like.simplify] at [Int_ops_for_binary_arith] of the
   tagged immediates. *)
Definition simplify_int_arith_tagged (tenv : Typing_env.t)
    (op : binary_int_arith_op) (arg1 : simple) (ty1 : flambda_type)
    (arg2 : simple) (ty2 : flambda_type) : outcome :=
  let unknown := Stays_with_type any_tagged_immediate in
  if well_formed_input_b tenv ty1 && well_formed_input_b tenv ty2
  then
    match
      Provers.meet_equals_tagged_immediates tenv ty1,
      Provers.meet_equals_tagged_immediates tenv ty2
    with
    | Some (Meet_shortcut.Known_result nums1),
      Some (Meet_shortcut.Known_result nums2) =>
      if (max_num_possible_results <? length nums1)%nat
         || (max_num_possible_results <? length nums2)%nat
      then unknown
      else check_possible_results (both_sides_results op nums1 nums2) arg1
    | Some (Meet_shortcut.Known_result nums1),
      (Some Meet_shortcut.Need_meet | None) =>
      match one_side_results (op_rhs_unknown op) nums1 with
      | Some results => check_possible_results results arg2
      | None => unknown
      end
    | (Some Meet_shortcut.Need_meet | None),
      Some (Meet_shortcut.Known_result nums2) =>
      match one_side_results (op_lhs_unknown op) nums2 with
      | Some results => check_possible_results results arg1
      | None => unknown
      end
    | Some Meet_shortcut.Invalid, _ | _, Some Meet_shortcut.Invalid =>
      Becomes_invalid
    | (Some Meet_shortcut.Need_meet | None),
      (Some Meet_shortcut.Need_meet | None) =>
      unknown
    end
  else unknown.

(* What is known of the result of a comparison that is not decided: one
   that yields a boolean is 0 or 1 ([unknown] of
   [Int_ops_for_binary_comp], [T.any_naked_bool]). Of one that yields
   the result of a compare function this records the kind only
   (Simplify records that it is -1, 0 or 1). *)
Definition comparison_unknown (behaviour : int_comparison_behaviour)
    : outcome :=
  match behaviour with
  | Yielding_bool _ => Stays_with_type any_naked_bool
  | Yielding_int_like_compare_functions _ =>
    Stays_with_kind naked_immediate_kind
  end.

(* The same functor at [Int_ops_for_binary_comp] of the tagged
   immediates, where both arguments are known: the one result of the
   comparison on every pair, if there is one. *)
Definition simplify_int_comp_tagged (tenv : Typing_env.t)
    (behaviour : int_comparison_behaviour) (ty1 ty2 : flambda_type)
    : outcome :=
  let unknown := comparison_unknown behaviour in
  if well_formed_input_b tenv ty1 && well_formed_input_b tenv ty2
  then
    match
      Provers.meet_equals_tagged_immediates tenv ty1,
      Provers.meet_equals_tagged_immediates tenv ty2
    with
    | Some (Meet_shortcut.Known_result nums1),
      Some (Meet_shortcut.Known_result nums2) =>
      let results :=
        flat_map
          (fun i1 =>
             map (fun i2 => eval_int_comp tagged_width behaviour i1 i2) nums2)
          nums1
      in
      match results with
      | result :: rest =>
        if forallb (Z.eqb result) rest
        then Becomes_simple (Const (Naked_immediate result))
        else unknown
      | [] => unknown
      end
    | Some Meet_shortcut.Invalid, _ | _, Some Meet_shortcut.Invalid =>
      Becomes_invalid
    | _, _ => unknown
    end
  else unknown.

(* [Simplify_binary_primitive.simplify_phys_equal]: where the prover
   proves a boolean the primitive is that boolean; where it does not,
   the result is 0 or 1 ([T.any_naked_bool]). *)
Definition simplify_phys_equal (tenv : Typing_env.t)
    (op : equality_comparison) (ty1 ty2 : flambda_type) : outcome :=
  let unknown := Stays_with_type any_naked_bool in
  if well_formed_input_b tenv ty1 && well_formed_input_b tenv ty2
  then
    match Provers.prove_physical_equality tenv ty1 ty2 with
    | Some (Proof_of_property.Proved b) =>
      Becomes_simple
        (naked_bool_const match op with Phys_eq => b | Phys_neq => negb b end)
    | Some Proof_of_property.Unknown | None => unknown
    end
  else unknown.

(* A binary primitive: the arithmetic and the comparison of tagged
   immediates, and physical equality; a comparison of other integers
   stays, and its result is 0 or 1 where it yields a boolean; every
   other stays, at the kind of its result. *)
Definition simplify_binary (tenv : Typing_env.t) (p : binary_primitive)
    (arg1 : simple) (ty1 : flambda_type) (arg2 : simple) (ty2 : flambda_type)
    : outcome :=
  match p with
  | Phys_equal op => simplify_phys_equal tenv op ty1 ty2
  | Int_arith Standard_tagged_immediate op =>
    simplify_int_arith_tagged tenv op arg1 ty1 arg2 ty2
  | Int_comp Standard_tagged_immediate behaviour =>
    simplify_int_comp_tagged tenv behaviour ty1 ty2
  | Int_comp _ behaviour => comparison_unknown behaviour
  | _ => Stays_with_kind (result_kind_of_binary_primitive p)
  end.

(* [Simplify_variadic_primitive.simplify_make_block]. *)
Definition simplify_variadic (p : variadic_primitive)
    (tys : list flambda_type) : outcome :=
  match p with
  | Make_block (Block_kind_values tag) =>
    if tag <? no_scan_tag
    then Stays_with_type (make_block_type tag tys)
    else Stays_with_kind Value
  | Make_immutable_array _ => Stays_with_kind Value
  end.

(* ======================================================================== *)
(* A defining expression                                                    *)
(* ======================================================================== *)

(* Nothing is decided of a primitive of the heap
   (FlambdaHeapSpec.named_changes_heap): the machine runs it, it
   changes the heap, and [prim_args_well_kinded] knows the kinds of
   the arguments of none of them, so the test of the kinds below would
   make every one of them [Invalid]. *)
Definition simplify_named (tenv : Typing_env.t) (defining_expr : named)
    : outcome :=
  if named_changes_heap defining_expr
  then Not_known
  else
  match defining_expr with
  | Simple s => Becomes_simple s
  | Prim p args =>
    match simplify_simples tenv args with
    | None => Not_known
    | Some answers =>
      let tys := map fst answers in
      if prim_args_well_kinded p (map FlambdaTypeGrammar.kind tys)
      then
        match p, args, tys with
        | Unary p, [arg], [ty] => simplify_unary tenv p arg ty
        | Binary p, [arg1; arg2], [ty1; ty2] =>
          simplify_binary tenv p arg1 ty1 arg2 ty2
        | Variadic p, _, _ => simplify_variadic p tys
        | _, _, _ => Not_known
        end
      else Becomes_invalid
    end
  | Set_of_closures code value_slots =>
    match simplify_simples tenv (map snd value_slots) with
    | Some answers =>
      Stays_with_type
        (set_of_closures_type code (map fst value_slots) (map fst answers))
    | None => Stays_with_kind Value
    end
  | Closure_of_set slot function_decls value_slots =>
    match simplify_simples tenv (map snd value_slots) with
    | Some answers =>
      Stays_with_type
        (closure_of_set_type slot function_decls (map fst value_slots)
           (map fst answers))
    | None => Stays_with_kind Value
    end
  | Project_function_slot _ move_to closure =>
    match simplify_simple tenv closure with
    | Some (ty, _) =>
      of_simple_shortcut
        (Provers.meet_project_function_slot_simple tenv Name_mode.normal ty
           move_to)
        (Stays_with_kind Value)
    | None => Not_known
    end
  | Project_value_slot _ _ | Project_value_slot_from _ _ _ => Not_known
  end.

(* The arguments of a primitive replaced by their canonical elements. *)
Definition canonical_named (tenv : Typing_env.t) (defining_expr : named)
    : named :=
  match defining_expr with
  | Prim p args =>
    match simplify_simples tenv args with
    | Some answers => Prim p (map snd answers)
    | None => defining_expr
    end
  | Simple _ | Set_of_closures _ _ | Closure_of_set _ _ _
  | Project_value_slot _ _ | Project_value_slot_from _ _ _
  | Project_function_slot _ _ _ =>
    defining_expr
  end.

(* ======================================================================== *)
(* A [Let]                                                                  *)
(* ======================================================================== *)

(* What becomes of a [Let]: it is [Invalid]; or it has this defining
   expression, and its body is rewritten under this environment; or it
   is left as it is, with its body. *)
Inductive let_step : Type :=
  | Step_invalid
  | Step_let (defining_expr : named) (tenv : Typing_env.t)
  | Step_stop.

(* [simple_in_terms] and [simple_of_kind], as tests. *)
Definition simple_in_terms_b (tenv : Typing_env.t) (s : simple) : bool :=
  match s with
  | Var x => Typing_env.mem (Some Name_mode.normal) tenv (Name_var x)
  | Symbol _ | Const _ => true
  end.

Definition simple_of_kind_b (tenv : Typing_env.t) (s : simple)
    (kind : flambda_kind) : bool :=
  match s with
  | Const c => kind_eqb (FlambdaTypeGrammar.kind (type_for_const c)) kind
  | Var x => is_some (Typing_env.find tenv (Name_var x) (Some kind))
  | Symbol sym => is_some (Typing_env.find tenv (Name_symbol sym) (Some kind))
  end.

(* The variable of a [Let] is bound to a [Simple]: the [Simple] is
   replaced by its canonical element, and the variable is defined at
   its kind and made an alias of it. *)
Definition bind_simple (tenv : Typing_env.t) (x : variable) (s : simple)
    : let_step :=
  match simplify_simple tenv s with
  | Some (ty, canonical) =>
    let kind := FlambdaTypeGrammar.kind ty in
    if simple_in_terms_b tenv canonical && simple_of_kind_b tenv canonical kind
    then
      match
        Typing_env.add_alias
          (Typing_env.add_variable_definition tenv x kind Name_mode.normal)
          (Var x) canonical
      with
      | Some (Ok result) =>
        Step_let (Simple canonical) (Typing_env.add_alias_result_t result)
      | Some Unknown | Some Bottom | None => Step_stop
      end
    else Step_stop
  | None => Step_stop
  end.

(* Every variable of a type is in scope. *)
Definition type_in_scope_b (tenv : Typing_env.t) (ty : flambda_type) : bool :=
  forallb
    (fun y => Typing_env.mem (Some Name_mode.normal) tenv (Name_var y))
    (free_variables ty).

Definition simplify_let (tenv : Typing_env.t) (x : variable)
    (defining_expr : named) : let_step :=
  if typing_env_well_formed tenv
     && negb (Typing_env.mem None tenv (Name_var x))
  then
    match simplify_named tenv defining_expr with
    | Becomes_invalid => Step_invalid
    | Becomes_simple s => bind_simple tenv x s
    | Stays_with_type ty =>
      if type_in_scope_b tenv ty
      then
        Step_let (canonical_named tenv defining_expr)
          (Typing_env.replace_equation
             (Typing_env.add_variable_definition tenv x
                (FlambdaTypeGrammar.kind ty) Name_mode.normal)
             (Name_var x) ty)
      else Step_stop
    | Stays_with_kind kind =>
      Step_let (canonical_named tenv defining_expr)
        (Typing_env.add_variable_definition tenv x kind Name_mode.normal)
    | Not_known => Step_stop
    end
  else Step_stop.

(* The projection of a value slot, at a [Let] that declares the kind
   [declared]: where the prover answers a [Simple] for the slot, asked
   at that kind about the type of the closure, the defining expression
   is that [Simple]. Every other defining expression is left as it
   is. *)
Definition resolve_value_slot (tenv : Typing_env.t) (declared : flambda_kind)
    (defining_expr : named) : named :=
  match defining_expr with
  | Project_value_slot slot closure
  | Project_value_slot_from _ slot closure =>
    match simplify_simple tenv closure with
    | Some (ty, _) =>
      match simplify_project_value_slot tenv declared slot ty with
      | Some s => Simple s
      | None => defining_expr
      end
    | None => defining_expr
    end
  | Simple _ | Prim _ _ | Set_of_closures _ _ | Closure_of_set _ _ _
  | Project_function_slot _ _ _ =>
    defining_expr
  end.

(* A [Let] of which nothing is known: its variable is defined at the
   kind that the [Let] declares, where the environment passes its test
   of well-formedness and does not have the variable. *)
Definition declare_variable (tenv : Typing_env.t) (x : variable)
    (declared : flambda_kind) : option Typing_env.t :=
  if typing_env_well_formed tenv
     && negb (Typing_env.mem None tenv (Name_var x))
  then
    Some
      (Typing_env.add_variable_definition tenv x declared Name_mode.normal)
  else None.

(* ======================================================================== *)
(* The other expressions                                                    *)
(* ======================================================================== *)

Definition simplify_apply_cont (tenv : Typing_env.t) (ac : apply_cont)
    : apply_cont :=
  match simplify_simples tenv (ac_args ac) with
  | Some answers =>
    Apply_cont_expr (ac_continuation ac) (map snd answers)
      (ac_trap_action ac)
  | None => ac
  end.

(* A [Switch] whose scrutinee has a constant for canonical element is
   the [Apply_cont] of its arm. *)
(* CR mshinwell: no arm is removed here from the type of the scrutinee,
   as Simplify removes the arms whose discriminant the type excludes.
   The law of that rewrite (the workstream on the grammar of types has
   it) rests on the meet, and so holds of the states whose static
   environment is valid; the theorem of this function is of every
   state, which is what the validation asks of a rewrite of the term
   before. The choice made: the pruning stays out of this function,
   and is in a second one with a theorem about valid static
   environments (FlambdaSimplifyPruningSpec.v). The other choice is
   to carry the
   premise into [simplify_by_types_refines_statement], and so into the
   theorem of the validation. *)
Definition simplify_switch (tenv : Typing_env.t) (scrutinee : simple)
    (arms : list (Z * apply_cont)) : expr :=
  match simplify_simple tenv scrutinee with
  | Some (_, Const (Naked_immediate i)) =>
    match find_arm i arms with
    | Some ac => Apply_cont (simplify_apply_cont tenv ac)
    | None => Switch scrutinee arms
    end
  | Some (_, (Const _ | Var _ | Symbol _)) | None => Switch scrutinee arms
  end.

(* ======================================================================== *)
(* Where a continuation is applied                                          *)
(* ======================================================================== *)

(* The environment in the arm of discriminant [d] of a [Switch]. Where
   the canonical element of the scrutinee is a variable that may occur
   in terms and is a naked immediate, that variable is made an alias of
   the constant [d]; elsewhere the environment is as it was. *)
Definition arm_env (tenv : Typing_env.t) (scrutinee : simple) (d : Z)
    : Typing_env.t :=
  match simplify_simple tenv scrutinee with
  | Some (_, Var c) =>
    if simple_in_terms_b tenv (Var c)
       && simple_of_kind_b tenv (Var c) naked_immediate_kind
    then
      match Typing_env.add_alias tenv (Var c) (Const (Naked_immediate d)) with
      | Some (Ok result) => Typing_env.add_alias_result_t result
      | Some Unknown | Some Bottom | None => tenv
      end
    else tenv
  | Some (_, (Const _ | Symbol _)) | None => tenv
  end.

(* Parameters are defined in order at their kinds, each after the two
   tests of [declare_variable], and each is made an alias of the
   [Simple] given with it where [aliased] says so and that [Simple] may
   occur in terms and is of the kind. [None] where a test fails. *)
Fixpoint bind_params_to_args (aliased : simple -> bool) (tenv : Typing_env.t)
    (bindings : list (variable * flambda_kind * simple))
    : option Typing_env.t :=
  match bindings with
  | [] => Some tenv
  | (x, kind, canonical) :: bindings =>
    if
      typing_env_well_formed tenv
      && negb (Typing_env.mem None tenv (Name_var x))
    then
      let defined :=
        Typing_env.add_variable_definition tenv x kind Name_mode.normal
      in
      bind_params_to_args aliased
        (if
           aliased canonical && simple_in_terms_b tenv canonical
           && simple_of_kind_b tenv canonical kind
         then
           match Typing_env.add_alias defined (Var x) canonical with
           | Some (Ok result) => Typing_env.add_alias_result_t result
           | Some Unknown | Some Bottom | None => defined
           end
         else defined)
        bindings
    else None
  end.

(* A [Simple] is none of the variables given. *)
Definition not_among (xs : list variable) (s : simple) : bool :=
  match s with
  | Var x => negb (among x xs)
  | Symbol _ | Const _ => true
  end.

(* The environment in which a handler runs when it is applied to
   [args] from a place whose environment is [tenv]: the parameters are
   defined at the kinds that the handler gives them, and each is made
   an alias of the canonical element of its argument, if that element
   may occur in terms under [scope] (the environment of the place
   where the handler is defined) and is of the kind of the parameter.
   [None] where the environment does not know an argument, the handler
   has not as many kinds as parameters or the application as many
   arguments, or a test of [bind_params_to_args] fails. *)
Definition use_env (scope tenv : Typing_env.t) (params : list variable)
    (metadata : handler_metadata) (args : list simple)
    : option Typing_env.t :=
  let kinds := hm_params_kinds metadata in
  match simplify_simples tenv args with
  | Some answers =>
    if
      Nat.eqb (length kinds) (length params)
      && Nat.eqb (length answers) (length params)
    then
      bind_params_to_args
        (fun canonical =>
           simple_in_terms_b scope canonical && not_among params canonical)
        tenv
        (combine (combine params kinds) (map snd answers))
    else None
  | None => None
  end.

(* The applications of the continuation [k] that an [Apply_cont] or a
   [Switch] may come to, each with the environment at that application:
   that of the arm, for an arm of a [Switch]. [None] of any other
   expression. *)
Definition leaf_uses (tenv : Typing_env.t) (k : continuation) (e : expr)
    : option (list (Typing_env.t * list simple)) :=
  match e with
  | Apply_cont ac =>
    Some (if Pos.eqb (ac_continuation ac) k then [(tenv, ac_args ac)] else [])
  | Switch scrutinee arms =>
    Some
      (flat_map
         (fun arm =>
            if Pos.eqb (ac_continuation (snd arm)) k
            then [(arm_env tenv scrutinee (fst arm), ac_args (snd arm))]
            else [])
         arms)
  | Let _ _ _ _ | Let_static _ _ | Let_cont_non_recursive _ _ _
  | Let_cont_recursive _ _ | Apply _ | Invalid =>
    None
  end.

(* The environment under which the body of a [Let] is rewritten:
   that of [simplify_let], or of [declare_variable] where
   [simplify_let] stops. [None] where the [Let] is [Invalid] or the
   term is left as it is from there down. *)
Definition let_env (tenv : Typing_env.t) (x : variable)
    (declared : flambda_kind) (defining_expr : named)
    : option Typing_env.t :=
  match
    simplify_let tenv x (resolve_value_slot tenv declared defining_expr)
  with
  | Step_let _ tenv' => Some tenv'
  | Step_invalid => None
  | Step_stop => declare_variable tenv x declared
  end.

(* Where the one application of the continuation [k], of handler [h],
   is, in the body of its [Let_cont]: the environment in which the
   handler runs there ([use_env]), with the variables of the [Let]s
   and the continuations of the [Let_cont]s that are bound between the
   [Let_cont] of [k] and that application. The application is looked
   for under [Let]s whose variable is not free in [h] and under
   non-recursive [Let_cont]s of another name whose handler does not
   mention [k] and whose name [h] does not mention, down to an
   [Apply_cont] or a [Switch] that applies [k] once. [None] where
   there is no such application. [scope] is the environment of the
   [Let_cont] of [k], and [tenv] that of [body]. *)
Fixpoint find_use (scope tenv : Typing_env.t) (k : continuation)
    (h : continuation_handler) (body : expr)
    : option (Typing_env.t * list variable * list continuation) :=
  match body with
  | Let x declared defining_expr rest =>
    if var_free_in_handler_b x h then None
    else
      match let_env tenv x declared defining_expr with
      | Some tenv' =>
        match find_use scope tenv' k h rest with
        | Some (target, xs, ks) => Some (target, x :: xs, ks)
        | None => None
        end
      | None => None
      end
  | Let_cont_non_recursive k' h' rest =>
    if
      negb (Pos.eqb k k')
      && negb (cont_free_in_handler_b k h')
      && negb (cont_free_in_handler_b k' h)
    then
      match find_use scope tenv k h rest with
      | Some (target, xs, ks) => Some (target, xs, k' :: ks)
      | None => None
      end
    else None
  | Apply_cont _ | Switch _ _ =>
    match h, leaf_uses tenv k body with
    | Handler params metadata _, Some [(tenv_use, args)] =>
      match use_env scope tenv_use params metadata args with
      | Some target => Some (target, [], [])
      | None => None
      end
    | Handler _ _ _, (Some ([] | _ :: _ :: _) | None) => None
    end
  | Let_static _ _ | Let_cont_recursive _ _ | Apply _ | Invalid => None
  end.

(* None of the variables and none of the continuations given is free
   in the handler. *)
Definition binders_not_free (xs : list variable) (ks : list continuation)
    (h : continuation_handler) : bool :=
  forallb (fun x => negb (var_free_in_handler_b x h)) xs
  && forallb (fun k => negb (cont_free_in_handler_b k h)) ks.

(* The applications of the continuation [k], of handler [h], in
   [body], each with the environment at the application and its
   arguments; and the binders passed on the way to them. Where
   [find_use] looks for one application at the end of the [Let]s and
   the non-recursive [Let_cont]s that the body begins with, this
   gathers every application: those at the end ([leaf_uses]), and
   those in the handler of a [Let_cont] passed, if that handler
   mentions [k]. Such a handler is gone into in the same way, under
   the environment of its [Let_cont] with its parameters defined
   ([handler_env]: nothing is asked of where that handler is applied).
   The binders are the variables of the [Let]s, the parameters of the
   handlers gone into, the names of the [Let_cont]s and, where a
   handler is gone into, [k] itself: in the proof the handler of [k]
   is copied to the head of that handler, so neither [h] nor what it
   becomes may mention them. [None] where the body, or a handler that
   mentions [k], is not of that form, or where [h] mentions one of
   those binders. *)
Fixpoint gather_uses (tenv : Typing_env.t) (k : continuation)
    (h : continuation_handler) (body : expr)
    : option
        (list (Typing_env.t * list simple) * list variable
         * list continuation) :=
  match body with
  | Let x declared defining_expr rest =>
    if var_free_in_handler_b x h then None
    else
      match let_env tenv x declared defining_expr with
      | Some tenv' =>
        match gather_uses tenv' k h rest with
        | Some (uses, xs, ks) => Some (uses, x :: xs, ks)
        | None => None
        end
      | None => None
      end
  | Let_cont_non_recursive k' h' rest =>
    if negb (Pos.eqb k k') && negb (cont_free_in_handler_b k' h)
    then
      match gather_uses tenv k h rest with
      | Some (uses, xs, ks) =>
        if cont_free_in_handler_b k h'
        then
          match h' with
          | Handler params' metadata' body' =>
            if
              negb (cont_free_in_handler_b k h)
              && forallb (fun x => negb (var_free_in_handler_b x h)) params'
            then
              match handler_env tenv params' metadata' with
              | Some tenv_handler =>
                match gather_uses tenv_handler k h body' with
                | Some (uses', xs', ks') =>
                  Some
                    (uses' ++ uses, params' ++ xs' ++ xs,
                     k :: k' :: ks' ++ ks)
                | None => None
                end
              | None => None
              end
            else None
          end
        else Some (uses, xs, k' :: ks)
      | None => None
      end
    else None
  | Apply_cont _ | Switch _ _ =>
    match leaf_uses tenv k body with
    | Some uses => Some (uses, [], [])
    | None => None
    end
  | Let_static _ _ | Let_cont_recursive _ _ | Apply _ | Invalid => None
  end.

(* Where [gather_uses] gathers one application and no more, the
   environment that [use_env] makes for it, and the binders passed.
   [tenv] is the environment of the [Let_cont], and is the scope that
   [use_env] is given: a parameter is made an alias of its argument
   only if the argument may occur in terms there. [None] where the
   gathering answers nothing, no application or several, or where
   [use_env] answers nothing. *)
Definition find_gathered_use (tenv : Typing_env.t) (k : continuation)
    (h : continuation_handler) (body : expr)
    : option (Typing_env.t * list variable * list continuation) :=
  match h, gather_uses tenv k h body with
  | Handler params metadata _, Some ([(tenv_use, args)], xs, ks) =>
    match use_env tenv tenv_use params metadata args with
    | Some target => Some (target, xs, ks)
    | None => None
    end
  | Handler _ _ _, (Some (([] | _ :: _ :: _), _, _) | None) => None
  end.

(* The one application under whose environment the handler of [k] is
   rewritten: the one that [find_use] finds from the environment of
   the [Let_cont], and where it finds none, the one gathered. *)
Definition find_one_use (tenv : Typing_env.t) (k : continuation)
    (h : continuation_handler) (body : expr)
    : option (Typing_env.t * list variable * list continuation) :=
  match find_use tenv tenv k h body with
  | Some found => Some found
  | None => find_gathered_use tenv k h body
  end.

(* ======================================================================== *)
(* A term                                                                   *)
(* ======================================================================== *)

Fixpoint simplify_expr (tenv : Typing_env.t) (e : expr) : expr :=
  match e with
  | Let x declared defining_expr body =>
    match
      simplify_let tenv x (resolve_value_slot tenv declared defining_expr)
    with
    | Step_invalid => Invalid
    | Step_let defining_expr' tenv' =>
      Let x declared defining_expr' (simplify_expr tenv' body)
    | Step_stop =>
      match declare_variable tenv x declared with
      | Some tenv' => Let x declared defining_expr (simplify_expr tenv' body)
      | None => e
      end
    end
  | Let_static bindings body =>
    let tenv' := add_static_bindings tenv bindings in
    Let_static (map (simplify_static_binding tenv') bindings)
      (simplify_expr tenv' body)
  | Let_cont_non_recursive k handler body =>
    let body' := simplify_expr tenv body in
    Let_cont_non_recursive k
      (match handler with
       | Handler params metadata handler_body =>
         match find_one_use tenv k handler body' with
         | Some (target, xs, ks) =>
           let rewritten :=
             Handler params metadata (simplify_expr target handler_body)
           in
           if binders_not_free xs ks rewritten then rewritten
           else simplify_handler false tenv handler
         | None => simplify_handler false tenv handler
         end
       end)
      body'
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map (fun '(k, h) => (k, simplify_handler true tenv h)) handlers)
      (simplify_expr tenv body)
  | Apply ap => simplify_apply tenv ap
  | Apply_cont ac => Apply_cont (simplify_apply_cont tenv ac)
  | Switch scrutinee arms => simplify_switch tenv scrutinee arms
  | Invalid => Invalid
  end

(* A handler, where [tenv] is the environment of its [Let_cont]: it is
   rewritten under the environment of the handler made from [tenv]
   ([handler_env], FlambdaSimplifyHandlersSpec.v). [recursive] says
   whether the [Let_cont] is recursive: a handler of one is rewritten
   only if that does not change whether it binds something static. *)
with simplify_handler (recursive : bool) (tenv : Typing_env.t)
    (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    match handler_env tenv params metadata with
    | Some tenv_handler =>
      let handler' := simplify_expr tenv_handler handler in
      if negb recursive
         || Bool.eqb (binds_statics handler) (binds_statics handler')
      then Handler params metadata handler'
      else h
    | None => h
    end
  end

(* The code that a static binding binds, where [tenv] is the
   environment of the body of the [Let_static]. *)
with simplify_static_binding (tenv : Typing_env.t) (binding : static_binding)
    : static_binding :=
  match binding with
  | Bind_code id c => Bind_code id (simplify_code tenv c)
  | Bind_deleted_code _ | Bind_symbol _ _ => binding
  end

(* The body of code is rewritten under the environment of the code
   made from [tenv] ([code_env_in], FlambdaSimplifyCodeEnvSpec.v), if
   neither it nor what it becomes binds something static. *)
with simplify_code (tenv : Typing_env.t) (c : code) : code :=
  match c with
  | Code params my_closure ret exn metadata body =>
    match code_env_in tenv params my_closure metadata with
    | Some tenv_code =>
      let body' := simplify_expr tenv_code body in
      if negb (binds_statics body) && negb (binds_statics body')
      then Code params my_closure ret exn metadata body'
      else c
    | None => c
    end
  end.

(* From the environment that knows nothing. *)
Definition simplify_by_types (e : expr) : expr :=
  simplify_expr Typing_env.create e.

(* The handler [h] of the continuation [k] rewritten under the
   environment of the one application of [k] in [body] ([find_use]),
   where [tenv] is the environment of the [Let_cont]. [None] where
   there is no such application, or where what the handler becomes
   mentions one of the binders between the [Let_cont] and the
   application: it could not then stay where the handler is. *)
Definition rewrite_handler_at_use (tenv : Typing_env.t) (k : continuation)
    (h : continuation_handler) (body : expr) : option continuation_handler :=
  match h with
  | Handler params metadata handler =>
    match find_use tenv tenv k h body with
    | Some (target, xs, ks) =>
      let rewritten :=
        Handler params metadata (simplify_expr target handler)
      in
      if binders_not_free xs ks rewritten then Some rewritten else None
    | None => None
    end
  end.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* What the function answers refines the term, in every state that
   satisfies the environment. *)
Definition simplify_expr_refines_under_statement : Prop :=
  ∀ (tenv : Typing_env.t) (e : expr),
    expr_refines_under tenv e (simplify_expr tenv e).

(* From the environment that knows nothing, it refines the term. *)
Definition simplify_by_types_refines_statement : Prop :=
  ∀ e : expr, expr_refines e (simplify_by_types e).

(* The environment of the body of a [Let] holds once the [Let] has
   taken its step: the variable is bound to the value that the machine
   has for the defining expression, as it was written, on the heap and
   with the record of the step (FlambdaSemanticsSpec.exec_named),
   which is of the kind that the [Let] declares. *)
Definition let_env_satisfied_statement : Prop :=
  ∀ (tenv tenv' : Typing_env.t) (x : variable) (declared : flambda_kind)
    (defining_expr : named) (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (vars : env) (v : value),
    let_env tenv x declared defining_expr = Some tenv' ->
    env_satisfies_strictly tenv statics vars ->
    exec_named statics hp rd vars defining_expr = Some (v, hp', rd') ->
    kind_of_value v = declared ->
    env_satisfies_strictly tenv' statics ((x, v) :: vars).

(* The handler of a continuation may be rewritten under the
   environment that [find_use] answers, if what it becomes mentions
   none of the binders between the [Let_cont] and the application.
   The body is left as it is. *)
Definition find_use_refines_statement : Prop :=
  ∀ (scope tenv : Typing_env.t) (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler handler' body : expr)
    (target : Typing_env.t) (xs : list variable) (ks : list continuation),
    find_use scope tenv k (Handler params metadata handler) body
    = Some (target, xs, ks) ->
    binders_not_free xs ks (Handler params metadata handler') = true ->
    expr_refines_under target handler handler' ->
    expr_refines_under tenv
      (Let_cont_non_recursive k (Handler params metadata handler) body)
      (Let_cont_non_recursive k (Handler params metadata handler') body).

(* So the handler that [rewrite_handler_at_use] answers may take the
   place of the handler. *)
Definition rewrite_handler_at_use_refines_statement : Prop :=
  ∀ (tenv : Typing_env.t) (k : continuation) (h h' : continuation_handler)
    (body : expr),
    rewrite_handler_at_use tenv k h body = Some h' ->
    expr_refines_under tenv (Let_cont_non_recursive k h body)
      (Let_cont_non_recursive k h' body).

(* An application admits an environment for the handler: from a state
   at the application, which satisfies the environment there, the
   handler is entered in a state that satisfies [target]. The handler
   is taken to run under the variables of the application with its
   parameters bound, as it does once its [Let_cont] has been moved to
   the application. The environment that [use_env] answers for an
   application is one that it admits
   (FlambdaSimplifyUsesSpec.use_env_satisfied_statement). *)
Definition application_admits (params : list variable)
    (metadata : handler_metadata) (tenv_use : Typing_env.t)
    (args : list simple) (target : Typing_env.t) : Prop :=
  ∀ (statics : static_env) (vars : env) (vs : list value)
    (handler_vars : env),
    env_satisfies_strictly tenv_use statics vars ->
    eval_simples vars args = Some vs ->
    map kind_of_value vs = hm_params_kinds metadata ->
    bind_params params vs vars = Some handler_vars ->
    env_satisfies_strictly target statics handler_vars.

(* Several applications. The handler of a continuation may be
   rewritten under an environment that every application gathered
   admits, if what it becomes mentions none of the binders passed. No
   environment is computed here: the statement is of any that the
   applications admit, as a join of their environments is meant to
   be. *)
Definition gather_uses_refines_statement : Prop :=
  ∀ (tenv : Typing_env.t) (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler handler' body : expr)
    (target : Typing_env.t) (uses : list (Typing_env.t * list simple))
    (xs : list variable) (ks : list continuation),
    gather_uses tenv k (Handler params metadata handler) body
    = Some (uses, xs, ks) ->
    binders_not_free xs ks (Handler params metadata handler') = true ->
    (∀ use : Typing_env.t * list simple,
      use ∈ uses ->
      application_admits params metadata (fst use) (snd use) target) ->
    expr_refines_under target handler handler' ->
    expr_refines_under tenv
      (Let_cont_non_recursive k (Handler params metadata handler) body)
      (Let_cont_non_recursive k (Handler params metadata handler') body).

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

(* What the function answers, computed. The variables are those of
   FlambdaTypingEnvExamplesSpec.v. A variable of which nothing is to be
   known is bound to [Opaque_identity] of a constant. *)

Definition example_tagged_add (arg1 arg2 : simple) : named :=
  Prim (Binary (Int_arith Standard_tagged_immediate Add)) [arg1; arg2].

Definition example_opaque (kind : flambda_kind) (c : const) : named :=
  Prim (Unary (Opaque_identity kind)) [Const c].

(* [let x = 3 in let y = x + 4 in apply_cont k y]. *)
Definition example_pass_add_constants_statement : Prop :=
  ∀ k : continuation,
    simplify_by_types
      (Let example_x Value (Simple (Const (Tagged_immediate 3)))
         (Let example_y Value
            (example_tagged_add (Var example_x) (Const (Tagged_immediate 4)))
            (Apply_cont (Apply_cont_expr k [Var example_y] None))))
    = Let example_x Value (Simple (Const (Tagged_immediate 3)))
        (Let example_y Value (Simple (Const (Tagged_immediate 7)))
           (Apply_cont
              (Apply_cont_expr k [Const (Tagged_immediate 7)] None))).

(* [let x = 5 in let i = Is_int x in switch i with 0 -> k0 | 1 -> k1]. *)
Definition example_pass_is_int_statement : Prop :=
  ∀ k0 k1 : continuation,
    simplify_by_types
      (Let example_x Value (Simple (Const (Tagged_immediate 5)))
         (Let example_i naked_immediate_kind
            (Prim (Unary (Is_int false)) [Var example_x])
            (Switch (Var example_i)
               [(0, Apply_cont_expr k0 [] None);
                (1, Apply_cont_expr k1 [] None)])))
    = Let example_x Value (Simple (Const (Tagged_immediate 5)))
        (Let example_i naked_immediate_kind
           (Simple (Const (Naked_immediate 1)))
           (Apply_cont (Apply_cont_expr k1 [] None))).

(* [let p = Make_block 0 [x; y] in let q = Block_load p 1 in
   apply_cont k q], for [x] and [y] unknown. *)
Definition example_pass_second_of_pair_statement : Prop :=
  ∀ k : continuation,
    let unknown (x : variable) (c : Z) (body : expr) : expr :=
      Let x Value (example_opaque Value (Tagged_immediate c)) body
    in
    let pair :=
      Prim (Variadic (Make_block (Block_kind_values 0)))
        [Var example_x; Var example_y]
    in
    simplify_by_types
      (unknown example_x 5
         (unknown example_y 6
            (Let example_p Value pair
               (Let example_q Value
                  (Prim
                     (Unary
                        (Block_load
                           (Block_access_values (Some 0) (Some 2) Any_value)
                           Immutable 1))
                     [Var example_p])
                  (Apply_cont (Apply_cont_expr k [Var example_q] None))))))
    = unknown example_x 5
        (unknown example_y 6
           (Let example_p Value pair
              (Let example_q Value (Simple (Var example_y))
                 (Apply_cont (Apply_cont_expr k [Var example_y] None))))).

(* [let y = x + 0 in apply_cont k y], for [x] unknown. *)
Definition example_pass_add_zero_statement : Prop :=
  ∀ k : continuation,
    let unknown (body : expr) : expr :=
      Let example_x Value (example_opaque Value (Tagged_immediate 5)) body
    in
    simplify_by_types
      (unknown
         (Let example_y Value
            (example_tagged_add (Var example_x) (Const (Tagged_immediate 0)))
            (Apply_cont (Apply_cont_expr k [Var example_y] None))))
    = unknown
        (Let example_y Value (Simple (Var example_x))
           (Apply_cont (Apply_cont_expr k [Var example_x] None))).

(* [let p = Box_number x in let q = Unbox_number p in apply_cont k q],
   for a naked float [x] unknown. *)
Definition example_pass_unbox_of_box_statement : Prop :=
  ∀ k : continuation,
    let float := Naked_number Kind_naked_float in
    let unknown (body : expr) : expr :=
      Let example_x float (example_opaque float (Naked_float 1)) body
    in
    let boxed := Prim (Unary (Box_number Boxable_naked_float)) [Var example_x]
    in
    simplify_by_types
      (unknown
         (Let example_p Value boxed
            (Let example_q float
               (Prim (Unary (Unbox_number Boxable_naked_float))
                  [Var example_p])
               (Apply_cont (Apply_cont_expr k [Var example_q] None)))))
    = unknown
        (Let example_p Value boxed
           (Let example_q float (Simple (Var example_x))
              (Apply_cont (Apply_cont_expr k [Var example_x] None)))).

(* [let p = Tag_immediate x in let q = Untag_immediate p in
   apply_cont k q], for a naked immediate [x] unknown. *)
Definition example_pass_untag_of_tag_statement : Prop :=
  ∀ k : continuation,
    let unknown (body : expr) : expr :=
      Let example_x naked_immediate_kind
        (example_opaque naked_immediate_kind (Naked_immediate 1)) body
    in
    let tagged := Prim (Unary Tag_immediate) [Var example_x] in
    simplify_by_types
      (unknown
         (Let example_p Value tagged
            (Let example_q naked_immediate_kind
               (Prim (Unary Untag_immediate) [Var example_p])
               (Apply_cont (Apply_cont_expr k [Var example_q] None)))))
    = unknown
        (Let example_p Value tagged
           (Let example_q naked_immediate_kind (Simple (Var example_x))
              (Apply_cont (Apply_cont_expr k [Var example_x] None)))).

(* Under a [Let_static], in the body of a [Let_cont] and in its
   handler: the symbol is known to the environment; the handler is
   rewritten under the environment of the one application of its
   continuation, where its parameter [p] is 3, so [q] is 3. The
   continuations are the numbers 1 and 2. *)
Definition example_pass_under_binders_statement : Prop :=
  let k : continuation := 1%positive in
  let k' : continuation := 2%positive in
    let handler (defining_expr : named) (result : simple)
        : continuation_handler :=
      Handler [example_p] (make_handler_metadata [Value])
        (Let example_q Value defining_expr
           (Apply_cont (Apply_cont_expr k [result] None)))
    in
    let bindings :=
      [Bind_symbol example_s
         (Static_const_block 0 [Const (Tagged_immediate 1)])]
    in
    simplify_by_types
      (Let_static bindings
         (Let_cont_non_recursive k'
            (handler
               (example_tagged_add (Var example_p)
                  (Const (Tagged_immediate 0)))
               (Var example_q))
            (Let example_x Value (Simple (Symbol example_s))
               (Let example_y Value
                  (example_tagged_add (Const (Tagged_immediate 1))
                     (Const (Tagged_immediate 2)))
                  (Apply_cont
                     (Apply_cont_expr k' [Var example_y] None))))))
    = Let_static bindings
        (Let_cont_non_recursive k'
           (handler (Simple (Const (Tagged_immediate 3)))
              (Const (Tagged_immediate 3)))
           (Let example_x Value (Simple (Symbol example_s))
              (Let example_y Value (Simple (Const (Tagged_immediate 3)))
                 (Apply_cont
                    (Apply_cont_expr k' [Const (Tagged_immediate 3)]
                       None))))).

(* In the body of code. [let code f (x) = (let y = x + 0 in
   apply_cont ret y) in apply_cont k], where [x] is a parameter of
   kind value: [y] is [x]. The closure variable is [example_p]. *)
Definition example_pass_in_code_statement : Prop :=
  ∀ k ret exn : continuation,
    let code (body : expr) : static_binding :=
      Bind_code example_f
        (Code [example_x] example_p ret exn
           (make_code_metadata [Singleton Value] [Singleton Value]) body)
    in
    simplify_by_types
      (Let_static
         [code
            (Let example_y Value
               (example_tagged_add (Var example_x)
                  (Const (Tagged_immediate 0)))
               (Apply_cont (Apply_cont_expr ret [Var example_y] None)))]
         (Apply_cont (Apply_cont_expr k [] None)))
    = Let_static
        [code
           (Let example_y Value (Simple (Var example_x))
              (Apply_cont (Apply_cont_expr ret [Var example_x] None)))]
        (Apply_cont (Apply_cont_expr k [] None)).

(* The kinds of the arity of the code are those of its parameters.
   [let code f (x) = (let p = Box_number x in let q = Unbox_number p in
   apply_cont ret q) in apply_cont k], where [x] is a parameter of kind
   naked int64: [q] is [x]. The closure variable is [example_c]. *)
Definition example_pass_unbox_in_code_statement : Prop :=
  ∀ k ret exn : continuation,
    let int64 := Naked_number Kind_naked_int64 in
    let code (body : expr) : static_binding :=
      Bind_code example_f
        (Code [example_x] example_c ret exn
           (make_code_metadata [Singleton int64] [Singleton int64]) body)
    in
    let boxed := Prim (Unary (Box_number Boxable_naked_int64)) [Var example_x]
    in
    simplify_by_types
      (Let_static
         [code
            (Let example_p Value boxed
               (Let example_q int64
                  (Prim (Unary (Unbox_number Boxable_naked_int64))
                     [Var example_p])
                  (Apply_cont (Apply_cont_expr ret [Var example_q] None))))]
         (Apply_cont (Apply_cont_expr k [] None)))
    = Let_static
        [code
           (Let example_p Value boxed
              (Let example_q int64 (Simple (Var example_x))
                 (Apply_cont (Apply_cont_expr ret [Var example_x] None))))]
        (Apply_cont (Apply_cont_expr k [] None)).

(* [let x = 5 in let i = (x == 5) in switch i with 0 -> k0 | 1 -> k1]:
   the two arguments of [Phys_equal] are aliases of one constant. *)
Definition example_pass_phys_equal_statement : Prop :=
  ∀ k0 k1 : continuation,
    simplify_by_types
      (Let example_x Value (Simple (Const (Tagged_immediate 5)))
         (Let example_i naked_immediate_kind
            (Prim (Binary (Phys_equal Phys_eq))
               [Var example_x; Const (Tagged_immediate 5)])
            (Switch (Var example_i)
               [(0, Apply_cont_expr k0 [] None);
                (1, Apply_cont_expr k1 [] None)])))
    = Let example_x Value (Simple (Const (Tagged_immediate 5)))
        (Let example_i naked_immediate_kind
           (Simple (Const (Naked_immediate 1)))
           (Apply_cont (Apply_cont_expr k1 [] None))).

(* A call of a local function. [let x = opaque in let g = closure of
   [example_f] with [x] in its value slot in g x]: [g] is given the
   type of the closure, and the call through [g] is a direct call of
   [example_f]. *)
Definition example_pass_call_statement : Prop :=
  ∀ k exn : continuation,
    let unknown (body : expr) : expr :=
      Let example_x Value (example_opaque Value (Tagged_immediate 5)) body
    in
    let closure := Set_of_closures example_f [(example_w, Var example_x)] in
    let call (callee : simple) (kind : call_kind) : expr :=
      Apply
        (Apply_expr callee kind [Var example_x] k exn [] [Singleton Value]
           [Singleton Value])
    in
    simplify_by_types
      (unknown (Let example_c Value closure (call (Var example_c)
        Indirect_unknown_arity)))
    = unknown
        (Let example_c Value closure
           (call (Var example_c) (Direct example_f))).

(* The same closure read back from a pair: [let p = Make_block 0 [g; x]
   in let q = Block_load p 0 in q x]. The load is [g], and the call
   through [q] is a direct call of [example_f] through [g]. *)
Definition example_pass_call_from_pair_statement : Prop :=
  ∀ k exn : continuation,
    let unknown (body : expr) : expr :=
      Let example_x Value (example_opaque Value (Tagged_immediate 5)) body
    in
    let closure := Set_of_closures example_f [(example_w, Var example_x)] in
    let pair :=
      Prim (Variadic (Make_block (Block_kind_values 0)))
        [Var example_c; Var example_x]
    in
    let load :=
      Prim
        (Unary
           (Block_load (Block_access_values (Some 0) (Some 2) Any_value)
              Immutable 0))
        [Var example_p]
    in
    let call (callee : simple) (kind : call_kind) : expr :=
      Apply
        (Apply_expr callee kind [Var example_x] k exn [] [Singleton Value]
           [Singleton Value])
    in
    simplify_by_types
      (unknown
         (Let example_c Value closure
            (Let example_p Value pair
               (Let example_q Value load (call (Var example_q)
                 Indirect_unknown_arity)))))
    = unknown
        (Let example_c Value closure
           (Let example_p Value pair
              (Let example_q Value (Simple (Var example_c))
                 (call (Var example_c) (Direct example_f))))).

(* A value slot read back. [let x = opaque in let c = closure of
   [example_f] with [x] in its value slot in let y = the value slot of
   [c] in apply_cont k y]: [y] is [x]. *)
Definition example_pass_value_slot_statement : Prop :=
  ∀ k : continuation,
    let unknown (body : expr) : expr :=
      Let example_x Value (example_opaque Value (Tagged_immediate 5)) body
    in
    let closure := Set_of_closures example_f [(example_w, Var example_x)] in
    simplify_by_types
      (unknown
         (Let example_c Value closure
            (Let example_y Value (Project_value_slot example_w (Var example_c))
               (Apply_cont (Apply_cont_expr k [Var example_y] None)))))
    = unknown
        (Let example_c Value closure
           (Let example_y Value (Simple (Var example_x))
              (Apply_cont (Apply_cont_expr k [Var example_x] None)))).

(* A load from a static block. [let x = opaque in let_static s = Block
   0 (x, 7) in let q = Block_load s 0 in apply_cont k q]: the symbol is
   given the type of the block, and the load is [x]. *)
Definition example_pass_static_block_statement : Prop :=
  ∀ k : continuation,
    let unknown (body : expr) : expr :=
      Let example_x Value (example_opaque Value (Tagged_immediate 5)) body
    in
    let bindings :=
      [Bind_symbol example_s
         (Static_const_block 0 [Var example_x; Const (Tagged_immediate 7)])]
    in
    let load :=
      Prim
        (Unary
           (Block_load (Block_access_values (Some 0) (Some 2) Any_value)
              Immutable 0))
        [Symbol example_s]
    in
    simplify_by_types
      (unknown
         (Let_static bindings
            (Let example_q Value load
               (Apply_cont (Apply_cont_expr k [Var example_q] None)))))
    = unknown
        (Let_static bindings
           (Let example_q Value (Simple (Var example_x))
              (Apply_cont (Apply_cont_expr k [Var example_x] None)))).

(* A [Let] of which nothing is known. [let c = opaque in let y = the
   value slot of [c] in let p = Make_block 0 [y; y] in let q =
   Block_load p 0 in apply_cont k q]: no prover answers for the value
   slot of [c]; [y] is defined at the kind that its [Let] declares, and
   the load is [y]. *)
Definition example_pass_unknown_let_statement : Prop :=
  ∀ k : continuation,
    let unknown (body : expr) : expr :=
      Let example_c Value (example_opaque Value (Tagged_immediate 5)) body
    in
    let project := Project_value_slot example_w (Var example_c) in
    let pair :=
      Prim (Variadic (Make_block (Block_kind_values 0)))
        [Var example_y; Var example_y]
    in
    let load :=
      Prim
        (Unary
           (Block_load (Block_access_values (Some 0) (Some 2) Any_value)
              Immutable 0))
        [Var example_p]
    in
    simplify_by_types
      (unknown
         (Let example_y Value project
            (Let example_p Value pair
               (Let example_q Value load
                  (Apply_cont (Apply_cont_expr k [Var example_q] None))))))
    = unknown
        (Let example_y Value project
           (Let example_p Value pair
              (Let example_q Value (Simple (Var example_y))
                 (Apply_cont (Apply_cont_expr k [Var example_y] None))))).

(* In the body of code a symbol has its type. [let_static s = Block 0
   (7) and code f (x) = (let q = Block_load s 0 in apply_cont ret q) in
   apply_cont k]: in the code the load from the symbol, which the group
   of the code binds, is the constant. *)
Definition example_pass_static_in_code_statement : Prop :=
  ∀ k ret exn : continuation,
    let bindings (code_body : expr) : list static_binding :=
      [Bind_symbol example_s
         (Static_const_block 0 [Const (Tagged_immediate 7)]);
       Bind_code example_f
         (Code [example_x] example_p ret exn
            (make_code_metadata [Singleton Value] [Singleton Value])
            code_body)]
    in
    let load :=
      Prim
        (Unary
           (Block_load (Block_access_values (Some 0) (Some 1) Any_value)
              Immutable 0))
        [Symbol example_s]
    in
    simplify_by_types
      (Let_static
         (bindings
            (Let example_q Value load
               (Apply_cont (Apply_cont_expr ret [Var example_q] None))))
         (Apply_cont (Apply_cont_expr k [] None)))
    = Let_static
        (bindings
           (Let example_q Value (Simple (Const (Tagged_immediate 7)))
              (Apply_cont
                 (Apply_cont_expr ret [Const (Tagged_immediate 7)] None))))
        (Apply_cont (Apply_cont_expr k [] None)).

(* A function that calls itself, and a call of it. [let_static s = the
   closure of the code f and code f (x) = (apply s x) in apply s 7],
   the two calls indirect: the symbol is given the type of the
   closures of the code, and each call is the direct call of the
   code, the one in the body of the [Let_static] and the one in the
   code. *)
Definition example_pass_static_closure_statement : Prop :=
  ∀ k exn ret exn' : continuation,
    let call (kind : call_kind) (arg : simple) (k exn : continuation)
        : expr :=
      Apply
        (Apply_expr (Symbol example_s) kind [arg] k exn []
           [Singleton Value] [Singleton Value])
    in
    let bindings (kind : call_kind) : list static_binding :=
      [Bind_symbol example_s (Static_const_set_of_closures example_f []);
       Bind_code example_f
         (Code [example_x] example_p ret exn'
            (make_code_metadata [Singleton Value] [Singleton Value])
            (call kind (Var example_x) ret exn'))]
    in
    simplify_by_types
      (Let_static (bindings Indirect_unknown_arity)
         (call Indirect_unknown_arity (Const (Tagged_immediate 7)) k exn))
    = Let_static (bindings (Direct example_f))
        (call (Direct example_f) (Const (Tagged_immediate 7)) k exn).

(* A loop. [let_cont rec k (p) = (let q = p + 0 in apply_cont k q) in
   apply_cont k 1], where [p] is a parameter of kind value: in the
   handler, [q] is [p]. *)
Definition example_pass_loop_statement : Prop :=
  ∀ k : continuation,
    let loop (defining_expr : named) (result : simple) : expr :=
      Let_cont_recursive
        [(k,
          Handler [example_p] (make_handler_metadata [Value])
            (Let example_q Value defining_expr
               (Apply_cont (Apply_cont_expr k [result] None))))]
        (Apply_cont (Apply_cont_expr k [Const (Tagged_immediate 1)] None))
    in
    simplify_by_types
      (loop
         (example_tagged_add (Var example_p) (Const (Tagged_immediate 0)))
         (Var example_q))
    = loop (Simple (Var example_p)) (Var example_p).

(* A primitive of the heap is left as it is. [let p = (a new mutable
   block of one field, which holds 1) in apply_cont k p]: nothing is
   decided of the allocation, and the term comes back. (The test of
   the kinds of the arguments, which [simplify_named] makes of every
   other primitive, would answer [Invalid] here, where the machine
   binds [p] to a new location.) *)
Definition example_pass_heap_primitive_statement : Prop :=
  ∀ k : continuation,
    let term : expr :=
      Let example_p Value
        (Prim (Heap (Make_block_with_identity Cell_mutable 0))
           [Const (Tagged_immediate 1)])
        (Apply_cont (Apply_cont_expr k [Var example_p] None))
    in
    simplify_by_types term = term.

(* A handler under the environment of its one application.
   [let x = opaque 0 in let_cont k1 () = (switch x | 0 -> k3 | 1 -> k4)
   in let_cont k2 () = apply_cont k4 in switch x | 0 -> k1 | 1 -> k2],
   where nothing is known of the naked immediate [x]: [k1] is applied
   from the arm of 0 only, where [x] is 0, so its handler is
   [apply_cont k3]. The continuations are the numbers 1 to 4. *)
Definition example_handler_at_use_statement : Prop :=
  let k1 : continuation := 1%positive in
  let k2 : continuation := 2%positive in
  let k3 : continuation := 3%positive in
  let k4 : continuation := 4%positive in
  let goto (k : continuation) : apply_cont := Apply_cont_expr k [] None in
  let switch (on_0 on_1 : continuation) : expr :=
    Switch (Var example_x) [(0, goto on_0); (1, goto on_1)]
  in
  let no_params : handler_metadata := make_handler_metadata [] in
  match
    let_env Typing_env.create example_x naked_immediate_kind
      (example_opaque naked_immediate_kind (Naked_immediate 0))
  with
  | Some tenv =>
    rewrite_handler_at_use tenv k1 (Handler [] no_params (switch k3 k4))
      (Let_cont_non_recursive k2
         (Handler [] no_params (Apply_cont (goto k4)))
         (switch k1 k2))
  | None => None
  end
  = Some (Handler [] no_params (Apply_cont (goto k3))).
