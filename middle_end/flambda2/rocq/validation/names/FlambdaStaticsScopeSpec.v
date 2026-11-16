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


(* The rewrites of static bindings that the check of the validation
   applies keep scope, and so does a rewrite of parameters that it
   applies once the continuations are placed.

   This file is to be reviewed: the statements.

   A rewrite keeps scope (FlambdaValidatorEquivarianceSpec.keeps_scope)
   if, on a term whose code is closed, the code of its result is closed
   and every variable or continuation that is free in the result is
   free in the term. Symbols and code ids are not counted: a rewrite
   that keeps scope may mention or bind one that the term does not. The
   check of the validation (FlambdaValidateSpec.validate_unit) applies
   rewrites to the term before that are not rewrites of a round: it
   removes static bindings, it lifts, it uses what the unit binds its
   symbols to and what is known of the static closures in sight, it
   removes value slots, and it moves the bindings of code up; and
   FlambdaStaticsOrderSpec.order_statics_as is for it to apply, with
   the term after as the pattern. Each keeps scope, and that is stated
   here. It is what a statement that the check answers the
   same on renamed terms needs of them, once a round of the check
   makes names (the last section of
   FlambdaValidatorEquivarianceSpec.v).

   - FlambdaUnusedStaticsSpec.erase_expr removes the bindings of the
     names given, in code and in handlers too, and adds nothing
     ([erase_expr_keeps_scope_statement]). The removal as the check
     applies it, FlambdaUnusedStaticsSpec.drop_statics, answers a term
     only where the term refers to none of the names, and the term it
     answers is that of [erase_expr]; the statement here does not need
     that test. FlambdaValidateSpec.without_statics is [drop_statics]
     and then two rewrites of a round,
     FlambdaNormaliseSpec.drop_deleted_code and [merge_statics], which
     keep scope (FlambdaKeepsScopeSpec.v): it keeps scope
     ([without_statics_keeps_scope_statement]).
   - FlambdaStaticsOrderSpec.order_statics_as puts the bindings of
     each group of static bindings in the order of another term. The
     bindings of a group are the same ones in another order, whatever
     the other term is ([order_statics_as_keeps_scope_statement]).
   - FlambdaLiftSpec.lift_expr makes of a [Let] a static binding of
     what the [Let] made and a [Let] of the symbol: the variables of
     the constant bound are those of the defining expression, and a
     symbol has none ([lift_expr_keeps_scope_statement]).
     FlambdaMyClosureSpec.bind_my_closures binds, in code, the closure
     of the code to a symbol: it adds a [Let] of the variable that the
     code binds for its closure ([bind_my_closures_keeps_scope_statement]).
     FlambdaLiftFromCodeSpec.lift_from_code puts a symbol in the place
     of a defining expression that has no variable, in code, and binds
     the symbol beside the code to a constant that has none
     ([lift_from_code_keeps_scope_statement]).
     FlambdaLiftFromArmsSpec.lift_to_head does the same outside code,
     and binds the symbols in a group around the unit
     ([lift_to_head_keeps_scope_statement]).
     FlambdaLiftAroundSpec.lift_around puts a symbol in the place of a
     defining expression, and a group of static bindings before a
     [Let_cont]. A group that the guess puts may mention variables,
     so this one needs a test: where FlambdaLiftAroundSpec.around_ok
     passes, a group put binds symbols only and mentions only
     variables that are free in the [Let_cont] that it is put before;
     where it does not pass the term is left as it is
     ([lift_around_keeps_scope_statement]).
     The lifting as the check applies it, FlambdaValidateSpec.with_lifting,
     is the fourth of these, then the third, then the first two, then
     the last, each made or not; and then the tables given, bound in
     a group at the head of the unit where the checks of
     FlambdaUnusedHeadStaticsSpec.add_head_tables pass: a table that
     is bound has constants and symbols only, so the group has no
     code and mentions no variable ([with_lifting_keeps_scope_statement]).
     The same is stated of
     FlambdaLiftObservedSpec.with_lifting_observed, which makes the
     same rewrites and leaves out one test of each, that of the exits
     ([with_lifting_observed_keeps_scope_statement]).
     None of the statements but that of [lift_around] needs the tests
     that FlambdaLiftSpec.lift_unit,
     FlambdaLiftFromCodeSpec.lift_from_code_unit,
     FlambdaLiftFromArmsSpec.lift_to_head_unit and
     FlambdaLiftAroundSpec.lift_around_unit make.
   - FlambdaKnownSymbolsSpec.fold_known puts a constant or a symbol in
     the place of a defining expression, and makes an indirect call a
     direct call of the same callee with the same arguments, whatever is
     known of constants and of sets
     ([fold_known_keeps_scope_statement]); [with_known] is that, or the
     term as it is ([with_known_keeps_scope_statement]).
   - FlambdaKnownSlotsSpec.read_slots puts a symbol or a constant in
     the place of the projection of a value slot, where what it is
     given lists symbols and constants only
     ([read_slots_keeps_scope_statement]); [read_known_slots] is that,
     with what it has checked, or the term as it is
     ([read_known_slots_keeps_scope_statement]), and so is
     [read_known_slots_anywhere], with other symbols taken to be in
     scope ([read_known_slots_anywhere_keeps_scope_statement]).
   - FlambdaStubCallsSpec.demote_stub_calls makes, in the code of the
     stubs of a unit, a direct call an indirect call of the same
     callee with the same arguments
     ([demote_stub_calls_keeps_scope_statement]).
   - FlambdaUnusedSlotsSpec.drop_slots removes value slots from static
     sets of closures, and with them the variables they hold
     ([drop_slots_keeps_scope_statement]); [without_slots] is that
     followed by the removal of the [Let]s that nothing uses, in the
     term and in its code, or the term as it is
     ([without_slots_keeps_scope_statement]).
   - FlambdaKnownClosuresSpec.use_known_closures puts, for the
     projection of a value slot of a static closure in sight, what the
     slot holds, and makes a call of such a closure direct. A variable
     that a slot of a static closure holds is free in the binding of
     the closure, which is in the term
     ([use_known_closures_keeps_scope_statement]).
   - FlambdaFloatCodeSpec.float_code moves the bindings of code out of
     [Let]s and continuations. Closed code has no free variable and no
     free continuation, so none leaves the scope of its binder
     ([float_code_keeps_scope_statement]).
   - FlambdaGroupsUpSpec.groups_up moves a group of static bindings
     from the head of the body of a [Let_cont] to before it. A
     [Let_cont] binds no variable in its body, and a group whose code
     is closed mentions no continuation
     ([groups_up_keeps_scope_statement]).
   - FlambdaHoistStaticsSpec.hoist_statics moves to the head of the
     unit the bindings of code, and the bindings of symbols whose
     constant has a value in the empty environment, which has no
     variable; [with_statics_first] is that with the groups tidied, or
     the term as it is ([with_statics_first_keeps_scope_statement]).
     The statement does not need the tests that [hoist_statics] makes.
   - FlambdaUnaliasParamsSpec.unalias_params, of the rounds that
     follow the placing of the continuations
     (FlambdaValidateSpec.tidied_before), takes away a parameter of a
     continuation that every
     application gives one variable, and has the handler begin with a
     [Let] of the parameter to that variable. The variable is free in
     the body of the [Let_cont], where an application gives it and no
     binder on the way binds it; the applications lose an argument
     and mention no continuation more
     ([unalias_params_keeps_scope_statement]).
   - FlambdaGiveScrutineeSpec.give_scrutinees puts the scrutinee of
     a [Switch] for an argument of a continuation that is the number
     of the arm that the application is
     in. The variable is put only in an arm of a [Switch] on it, or
     in a handler bound beside such a [Switch] that only arms of one
     number apply, or under those where no binder of the variable is
     between, so it is free in the term
     ([give_scrutinees_keeps_scope_statement]).
   - FlambdaArmNumbersSpec.numbers_in_arms puts a constant for a
     variable where it is read (in the arguments of continuations,
     in a defining expression that is the variable, in the arguments
     of primitives and in the scrutinee of a [Switch]), so a free
     variable of the result is one of the term
     ([numbers_in_arms_keeps_scope_statement]).
   - FlambdaWrapReturnsSpec.wrap_returns takes away a parameter that
     its handler does not use, of a continuation that a call returns
     to: the applications lose an argument, and each call returns to
     a wrapper bound around it, which has the name of the
     continuation and the parameters of its handler, and applies the
     continuation to the parameters that are left. The wrapper
     mentions the continuation, which is in scope where the call was,
     and its own parameters ([wrap_returns_keeps_scope_statement]).
   - FlambdaUntagParamsSpec.untag_params untags, for a set of positions
     at once, parameters of continuations that hold a tagged immediate
     and are only untagged or passed on: the handler has those
     parameters after the others, an application has its arguments in
     that order with a naked constant for a tagged one, and the [Let]
     of the untagging of such a parameter is a [Let] of the parameter.
     No variable and no continuation is mentioned that was not
     ([untag_params_keeps_scope_statement]).
   - FlambdaShadowParamsSpec.shadow_params binds again, at the head
     of the handler of a continuation, parameters of that handler to
     themselves: a [Let] that it puts mentions a parameter of the
     handler and binds it, so no variable is free in the result that
     was not in the term ([shadow_params_keeps_scope_statement]).
   - FlambdaTwinParamsSpec.twin_params gives a twin to a parameter
     of a continuation that the head of its handler binds again. The
     [Let] goes: its variable is a parameter of the handler in the
     result, and the twin is the parameter that the [Let] read. An
     argument or an untagging of that variable that is made the twin
     is under both. The twins are then untagged as
     FlambdaUntagParamsSpec.untag_params untags
     ([twin_params_keeps_scope_statement]).
   - FlambdaBindConstParamsSpec.bind_const_params takes away a
     parameter of a continuation that every application gives one
     symbol or one constant, and has the handler begin with a [Let]
     of the parameter to it. The [Let] mentions no variable and binds
     the parameter; the applications lose an argument and mention no
     continuation more ([bind_const_params_keeps_scope_statement]).

   Nothing is asked of a term but that its code is closed, which
   [keeps_scope] itself asks, and nothing of what a rewrite is given.

   Nothing is trusted here. The proofs are in
   FlambdaStaticsScopeProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaBindersSpec FlambdaUnusedStaticsSpec
  FlambdaStaticsOrderSpec FlambdaMyClosureSpec FlambdaLiftSpec
  FlambdaLiftFromCodeSpec FlambdaLiftFromArmsSpec FlambdaLiftAroundSpec
  FlambdaUnusedSlotsSpec
  FlambdaKnownSymbolsSpec FlambdaKnownClosuresSpec FlambdaFloatCodeSpec
  FlambdaGroupsUpSpec FlambdaKnownSlotsSpec FlambdaStubCallsSpec
  FlambdaUnaliasParamsSpec FlambdaWrapReturnsSpec FlambdaUntagParamsSpec
  FlambdaGiveScrutineeSpec FlambdaArmNumbersSpec FlambdaShadowParamsSpec
  FlambdaTwinParamsSpec FlambdaValidateSpec
  FlambdaValidatorEquivarianceSpec.
From OxCamlFlambda2 Require FlambdaBindConstParamsSpec.
From OxCamlFlambda2 Require Import FlambdaUnaliasRecParamsSpec.
From OxCamlFlambda2 Require Import FlambdaLiftObservedSpec.
Import ListNotations.

(* ======================================================================== *)
(* The removal of static bindings                                           *)
(* ======================================================================== *)

(* The bindings of the names given removed, whatever the names. *)
Definition erase_expr_keeps_scope_statement : Prop :=
  ∀ D : dropped, keeps_scope (erase_expr D).

(* The removal as the check applies it, whatever the names. *)
Definition without_statics_keeps_scope_statement : Prop :=
  ∀ unused : dropped, keeps_scope (without_statics unused).

(* ======================================================================== *)
(* The order of static bindings                                             *)
(* ======================================================================== *)

(* The bindings of each group in the order of another term, whatever
   that term. *)
Definition order_statics_as_keeps_scope_statement : Prop :=
  ∀ pattern : expr, keeps_scope (order_statics_as pattern).

(* ======================================================================== *)
(* The lifting                                                              *)
(* ======================================================================== *)

(* The [Let]s given lifted, whatever they are and wherever the term is
   said to be. *)
Definition lift_expr_keeps_scope_statement : Prop :=
  ∀ (lets : list (path * lift_to)) (at_path : path),
    keeps_scope (lift_expr lets at_path).

(* The [Let]s given lifted out of the code given, whatever they are. *)
Definition lift_from_code_keeps_scope_statement : Prop :=
  ∀ (K : lets_in_code) (lets : list (path * lift_to)) (at_path : path),
    keeps_scope (lift_from_code K lets at_path).

(* The [Let]s given lifted to the head of the unit, whatever they
   are. *)
Definition lift_to_head_keeps_scope_statement : Prop :=
  ∀ lets : list (path * lift_to), keeps_scope (lift_to_head lets).

(* The [Let]s given made aliases of the symbols given and the groups
   given put, where the check of FlambdaLiftAroundSpec.around_ok
   passes, from nothing known and at any path; the term as it is
   where it does not. *)
Definition lift_around_keeps_scope_statement : Prop :=
  ∀ (lets : list (path * lift_to))
    (groups : list (path * list static_binding)) (at_path : path),
    keeps_scope
      (fun e =>
         if around_ok lets groups [] at_path e
         then lift_around lets groups at_path e
         else e).

(* The closure of each code given bound to its symbol in the code. *)
Definition bind_my_closures_keeps_scope_statement : Prop :=
  ∀ known : list (code_id * symbol), keeps_scope (bind_my_closures known).

(* The lifting as the check applies it. *)
Definition with_lifting_keeps_scope_statement : Prop :=
  ∀ L : lifting, keeps_scope (with_lifting L).

(* The same without the two tests of the unit, that of its exits and
   that of the heap (FlambdaLiftObservedSpec.with_lifting_observed). *)
Definition with_lifting_observed_keeps_scope_statement : Prop :=
  ∀ L : lifting, keeps_scope (with_lifting_observed L).

(* ======================================================================== *)
(* What the unit binds its symbols to                                       *)
(* ======================================================================== *)

Definition fold_known_keeps_scope_statement : Prop :=
  ∀ statics sets : static_env, keeps_scope (fold_known statics sets).

Definition with_known_keeps_scope_statement : Prop := keeps_scope with_known.

(* ======================================================================== *)
(* The value slots that the unit gives one symbol or constant               *)
(* ======================================================================== *)

Definition read_slots_keeps_scope_statement : Prop :=
  ∀ (known : known_slots) (syms : list symbol),
    closed_slots known = true -> keeps_scope (read_slots known syms).

Definition read_known_slots_keeps_scope_statement : Prop :=
  keeps_scope read_known_slots.

Definition read_known_slots_anywhere_keeps_scope_statement : Prop :=
  keeps_scope read_known_slots_anywhere.

(* ======================================================================== *)
(* The direct calls of the code of a stub                                   *)
(* ======================================================================== *)

Definition demote_stub_calls_keeps_scope_statement : Prop :=
  keeps_scope demote_stub_calls.

(* ======================================================================== *)
(* The removal of value slots                                               *)
(* ======================================================================== *)

Definition drop_slots_keeps_scope_statement : Prop :=
  ∀ dropped : list value_slot, keeps_scope (drop_slots dropped).

Definition without_slots_keeps_scope_statement : Prop :=
  ∀ slots : list value_slot, keeps_scope (without_slots slots).

(* ======================================================================== *)
(* What is known of the static closures in sight                            *)
(* ======================================================================== *)

Definition use_known_closures_keeps_scope_statement : Prop :=
  keeps_scope use_known_closures.

(* ======================================================================== *)
(* The bindings of code moved up                                            *)
(* ======================================================================== *)

Definition float_code_keeps_scope_statement : Prop := keeps_scope float_code.

(* ======================================================================== *)
(* The groups moved above continuation bindings                             *)
(* ======================================================================== *)

Definition groups_up_keeps_scope_statement : Prop := keeps_scope groups_up.

(* ======================================================================== *)
(* The static bindings that need nothing bound first                        *)
(* ======================================================================== *)

Definition with_statics_first_keeps_scope_statement : Prop :=
  ∀ Ls : list lifting, keeps_scope (with_statics_first Ls).

(* ======================================================================== *)
(* A parameter that every application gives one variable                    *)
(* ======================================================================== *)

Definition unalias_params_keeps_scope_statement : Prop :=
  keeps_scope unalias_params.

(* ======================================================================== *)
(* A parameter that every application gives the number of its arm           *)
(* ======================================================================== *)

Definition give_scrutinees_keeps_scope_statement : Prop :=
  keeps_scope give_scrutinees.

(* ======================================================================== *)
(* The scrutinee is the number of the arm                                   *)
(* ======================================================================== *)

Definition numbers_in_arms_keeps_scope_statement : Prop :=
  keeps_scope numbers_in_arms.

(* ======================================================================== *)
(* An unused parameter of a continuation that a call returns to             *)
(* ======================================================================== *)

Definition wrap_returns_keeps_scope_statement : Prop :=
  keeps_scope wrap_returns.

(* ======================================================================== *)
(* Parameters that are only untagged or passed on                           *)
(* ======================================================================== *)

Definition untag_params_keeps_scope_statement : Prop :=
  keeps_scope untag_params.

(* ======================================================================== *)
(* Parameters bound again at the head of their handlers, and their twins    *)
(* ======================================================================== *)

Definition shadow_params_keeps_scope_statement : Prop :=
  keeps_scope shadow_params.

Definition twin_params_keeps_scope_statement : Prop :=
  keeps_scope twin_params.

(* ======================================================================== *)
(* A parameter that every application gives one symbol or one constant      *)
(* ======================================================================== *)

Definition bind_const_params_keeps_scope_statement : Prop :=
  keeps_scope FlambdaBindConstParamsSpec.bind_const_params.

(* ======================================================================== *)
(* A parameter of a loop that every turn passes on                          *)
(* ======================================================================== *)

Definition unalias_rec_params_keeps_scope_statement : Prop :=
  keeps_scope unalias_rec_params.
