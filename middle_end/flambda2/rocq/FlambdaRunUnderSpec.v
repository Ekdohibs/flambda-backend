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

(* A unit run under a continuation environment, and the unit run on its
   own.

   This file is to be reviewed: [same_but_conts], [static_code_closed],
   [step_result_behaves] and the four statements.

   A unit leaves when it applies a way out: a continuation that its
   continuation environment did not bind where the name was resolved
   ([FlambdaSemanticsSpec.Outer]; the name may be bound elsewhere in the
   unit). Run on its own, from an empty continuation environment, that
   is an exit. Run from a continuation environment [base], it is the
   application of what [base] binds the name to, if it binds it. Until
   then the two runs are the same run: the same term, variables, static
   environment and heap at each step, and continuation environments
   that differ only in what stands for a way out. So the run from
   [base] is the run of the unit on its own and then, if that exits,
   what [base] binds the continuation of the exit to, applied to the
   values of the exit under the static environment and the heap that
   the unit left.

   The code that the term binds is to be closed
   (FlambdaBindersSpec.code_closed), and so is the code of the static
   environment that the run starts from ([static_code_closed]). A call
   gives the body of code the continuations of the call and no other,
   so a body that applied a continuation that it does not bind would
   leave by it whatever [base] binds it to, and the statements would be
   false of such a term. Nothing is asked of [base].

   The run starts from any variables, static environment and heap, and
   from no continuation: so from the empty static environment and heap,
   and from the static environment of a unit's imports
   (FlambdaImportsSpec.imports_statics), neither of which binds code.
   The statements are what lets a statement about what a unit leaves at
   its exit (FlambdaExitWorldSpec.exits_world) be used of the unit
   followed by something that runs on what it left.

   The proofs are in FlambdaRunUnderProofs.v, which needs no review. *)

From Stdlib Require Import List.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaValueSpec
  FlambdaSemanticsSpec FlambdaBindersSpec FlambdaExitWorldSpec.
Import ListNotations.

(* Two states that differ in their continuation environments only. *)
Definition same_but_conts (s s' : state) : Prop :=
  state_expr s = state_expr s'
  ∧ state_vars s = state_vars s'
  ∧ state_statics s = state_statics s'
  ∧ state_heap s = state_heap s'
  ∧ state_reads s = state_reads s'.

(* The code that a static environment binds is closed. It is so of an
   environment that binds no code. *)
Definition static_code_closed (statics : static_env) : Prop :=
  Forall (fun binding => code_is_closed (snd binding) = true)
    (static_code statics).

(* A state that the unit reaches on its own is reached from [base] in as
   many steps, the same but for its continuations. *)
Definition run_under_statement : Prop :=
  ∀ (base : cont_env) (e : expr) (vars : env) (statics : static_env)
    (hp : heap) (rd : read_record) (n : nat) (s : state),
    code_closed e = true ->
    static_code_closed statics ->
    run n (State e vars [] statics hp rd) = Step s ->
    ∃ s' : state,
      run n (State e vars base statics hp rd) = Step s'
      ∧ same_but_conts s s'.

(* Where the unit exits on its own, the run from [base] reaches the same
   state but for its continuations, and its step is what [base] binds
   the continuation of the exit to, applied to the values of the exit:
   the exit itself if [base] does not bind it. *)
Definition exit_under_statement : Prop :=
  ∀ (base : cont_env) (e : expr) (vars : env) (statics : static_env)
    (hp : heap) (rd : read_record) (n : nat) (last : state) (k : continuation)
    (vs : list value) (hp' : heap) (rd' : read_record),
    code_closed e = true ->
    static_code_closed statics ->
    run n (State e vars [] statics hp rd) = Step last ->
    step last = Exit k vs hp' rd' ->
    ∃ last' : state,
      run n (State e vars base statics hp rd) = Step last'
      ∧ same_but_conts last last'
      ∧ step last'
        = apply_cont_value (resolve k base) vs (state_statics last) hp' rd'.

(* Where the unit goes wrong on its own, so does the run from [base],
   in as many steps. *)
Definition wrong_under_statement : Prop :=
  ∀ (base : cont_env) (e : expr) (vars : env) (statics : static_env)
    (hp : heap) (rd : read_record) (n : nat),
    code_closed e = true ->
    static_code_closed statics ->
    run n (State e vars [] statics hp rd) = Undefined ->
    run n (State e vars base statics hp rd) = Undefined.

(* The behaviours of what a step comes to: those of the state it steps
   to; the exit, if it is one; going wrong, if it is undefined. *)
Definition step_result_behaves (result : step_result) (b : behaviour)
    : Prop :=
  match result with
  | Step s => behaves s b
  | Exit k vs hp rd => b = Exits k vs hp
  | Undefined => b = Goes_wrong
  end.

(* So the behaviours of the run from [base] are those of the unit on its
   own, with an exit replaced by the behaviours of what [base] binds its
   continuation to, applied to the world that the unit left. *)
Definition behaves_under_statement : Prop :=
  ∀ (base : cont_env) (e : expr) (vars : env) (statics : static_env)
    (hp : heap) (rd : read_record),
    code_closed e = true ->
    static_code_closed statics ->
    let alone := State e vars [] statics hp rd in
    let under := State e vars base statics hp rd in
    (∀ (w : exit_world) (b : behaviour),
      exits_world alone w ->
      behaves under b
      <-> step_result_behaves
            (apply_cont_value (resolve (exit_cont w) base) (exit_values w)
               (exit_statics w) (exit_heap w) (exit_reads w))
            b)
    ∧ (behaves alone Goes_wrong -> behaves under Goes_wrong)
    ∧ (behaves alone Diverges -> behaves under Diverges).
