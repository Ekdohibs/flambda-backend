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

(* Calls on the machine of regions (FlambdaRegionsSpec.v): code cannot
   end a region of its caller, and what follows of the values that a
   run exits with.

   This file is to be reviewed: all of it. It adds nothing to the
   machine: [region_step] and the rules of FlambdaRegionsSpec.v keep
   their text, and so do the tests of FlambdaRegionsInvariantSpec.v.
   FlambdaRegionsCallsProofs.v proves the statements.

   WHAT IS SAID

   In a run from the state that the test tool starts in, with the
   symbols that the machine draws apart from those of the term and of
   the static environment that the run starts with
   (FlambdaRegionsSpec.region_names_fresh): from a call until the code
   that was called returns or raises, no step pops a frame that was on
   the stack at the call. The reason is that a call gives code no
   region: an [Apply] one of whose simples has, for its value, a mark
   that is on the stack has no step (FlambdaRegionsSpec.simples_fault,
   [Region_used_as_value]), and in such a run no value holds a mark
   inside. So the regions that code can end are those that it began,
   and those that code which it called began and gave back to it.
   None of this is said of a state that no such run reaches: a value
   of it may hold a mark inside.

   The machine has no stack of calls. A call is a step into the body
   of the code (FlambdaSemanticsSpec.call_code); the caller is what
   the two continuations of the code hold. So "the frames that were
   there at the call" is nothing that a state shows, and the sentence
   above is not a statement here ("what is not stated", below). What
   a state does show is what its continuations have held since a call:
   the extra arguments that a call gave its exception continuation
   ([Extra_args]), which were evaluated at the call and are given, as
   they are, when the code raises ([cont_held]). The statements are of
   those.

   - In every state of a run from the state that the test tool starts
     in, what the continuations have held since a call may be
     evaluated ([held_stays_usable_statement]).
   - So the values that such a run exits with may all be evaluated
     ([exit_values_usable_statement]). The arguments that the term
     gives at the exit are evaluated there (decision D45); the others
     are what a continuation has held.
   - So the two statements of FlambdaRegionsErasureSpec.v hold with
     FlambdaRegionsErasureSpec.value_erased of every value at the
     exit, where they have [value_erased_if_usable]
     ([erasure_exits_erased_statement],
     [erasure_adds_no_exit_erased_statement]).

   THE TEST OF A STATE ([calls_kept])

   The statements are proved from a test of a state, which the state
   that a run starts in passes and which a step keeps, of a state that
   has the invariant of FlambdaRegionsInvariantSpec.v. It has four
   parts.

   - A mark is held by variables only ([marks_in_variables]). The
     value of a variable, of the state or of a closure that a
     continuation holds, may be the mark of a frame. No other symbol
     that the state holds is one: not inside a value of a variable,
     however deep, the callee and the arguments of a partial
     application included; not among the extra arguments or the
     arguments that remain of an over-application; not in what the
     static environment or a frame binds. So no parameter of code is
     given a mark, whatever the call: with all its arguments, with too
     many, or of a partial application.
   - The frames are in the order of their marks ([frames_ordered]).
     The machine draws its symbols upwards, so a symbol says when it
     was drawn. The marks decrease along the stack, newest first, and
     what a frame binds was drawn after its mark and before the mark
     of the frame above it. So what the end of a region frees was
     drawn after the mark of the region.
   - What the continuations have held since a call may be evaluated.
   - Each symbol that is itself one of the values that the
     continuations have held since a call is smaller than the mark of
     every frame that the variables hold ([held_older]); and the same
     of each closure that the continuations hold, with its variables
     and its continuations ([cont_held_older]). Nothing is asked of a
     symbol inside a value that is held. The variables of code are
     given no mark, so the marks that they come to hold are drawn
     after the call; a symbol that the two continuations of the code
     have held was in the state at the call, and the machine draws
     above every symbol of a state that has the invariant
     (FlambdaRegionsInvariantSpec.symbols_fresh). Such a symbol need
     not be one that the machine drew: it may be of the term or of the
     static environment.

   With the second and the fourth, the end of a region that a variable
   of the state names frees no symbol that is itself one of the values
   that the continuations of the state have held since a call: that is
   what "code cannot end a region of its caller" comes to in a state.

   HOW FAR THIS IS ABOUT THE PROGRAM

   "May be evaluated" is said of this machine
   (FlambdaRegionsSpec.value_usable), as the check of regions is. It
   is not that the program has not freed the value. One place where
   the two differ is known: the helper that the backend makes for a
   generic application ([apply_function_body] in
   backend/cmm_helpers.ml). Where the closure that is called does not
   take as many arguments as the application gives, the helper
   applies the arguments one at a time. If the compiler has stack
   allocation and the result of the application is not local
   ([Not_alloc_stack]), the helper begins a region of its own before
   the first of those applications. Before the last it tests whether
   anything has been allocated in that region
   ([maybe_reset_current_region]). If nothing has, the last
   application is a tail call and the helper ends no region. If
   something has, the helper ends its region after the last
   application has returned; if that application raises, the helper
   does not end it. The term has no such region. The machine runs an
   application with too many arguments as two calls in the frame that
   is open, so where the helper ends its region the machine can keep
   local allocations that the first call left live longer than the
   program does (FlambdaRegionsSpec.v, "what this stage leaves out":
   the helpers that the backend makes for a generic application).
   This is one example and not a list: the backend begins such a
   region too where it splits an application of many arguments into
   two at the place of the call ([might_split_call_caml_apply]). Such
   a region is the backend's own and it is no region of this machine:
   nothing here says when the program frees a value, there or
   anywhere else.

   WHAT IS NOT STATED

   - The sentence at the head of this file, of the frames. It would
     need a run that counts its calls: the continuations of a state do
     not say which call is being run. A loop that calls the same code
     twice, with the same return continuation, gives the two calls the
     same continuations, and may end a region of its own between
     them.
   - That code leaves the stack as it found it. It is false: code may
     leave a frame that it pushed (decision D48), and with no region
     of its own its local allocations go in the newest frame, which is
     a frame of its caller (decision D44).
   - That the arguments that remain of an over-application
     ([Over_application]) may be evaluated when the result of the full
     call is called with them. They are arguments of the application,
     evaluated there, unless a partial application keeps at least as
     many arguments as its code has parameters; no call makes such a
     value, and no statement says so. Nothing here needs it: they
     become parameters of code, and a step tests a parameter where it
     evaluates it. The test asks only that none of them is a mark.
   - That what a closure holds in its variables may be evaluated. A
     variable may hold what is gone: decision D45 is about evaluating,
     not holding. Of those variables the test says that a mark is held
     there directly only, and what the fourth part says of the marks
     that they hold.
   - That what a partial application keeps, its callee and its
     arguments, may be evaluated. The test says that none of them
     holds a mark, however deep, and no more. At an exit a partial
     application is itself a value that may be evaluated, whatever it
     keeps, and FlambdaRegionsErasureSpec.value_erased asks a part of
     it to be erased only if the part may be evaluated
     (FlambdaRegionsErasureSpec.v, "the values"): the two statements
     of the erasure that are said again here keep that limit. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaRegionsSpec FlambdaRegionsInvariantSpec
  FlambdaRegionsErasureSpec.
Import ListNotations.

(* ======================================================================== *)
(* What a continuation has held since a call                                *)
(* ======================================================================== *)

(* The extra arguments of the continuation, and those of the
   continuations that it holds: in the environment of a closure, under
   extra arguments, in what is left of an over-application. *)
Fixpoint cont_held (c : cont_value) : list value :=
  match c with
  | Closure _ _ conts _ => flat_map (fun '(_, c) => cont_held c) conts
  | Outer _ => []
  | Extra_args c extra => extra ++ cont_held c
  | Over_application _ _ _ return_to exn_to =>
    cont_held return_to ++ cont_held exn_to
  end.

Definition conts_held (conts : cont_env) : list value :=
  flat_map (fun '(_, c) => cont_held c) conts.

(* What the continuations of a state have held since a call. *)
Definition held (rs : region_state) : list value :=
  conts_held (state_conts (rs_state rs)).

(* ======================================================================== *)
(* A mark is held by variables only                                         *)
(* ======================================================================== *)

(* No symbol that the value holds is the mark of a frame: the value
   itself if it is a symbol, and the symbols in its fields, its
   elements, its value slots, its callee and its arguments, however
   deep. Unlike FlambdaRegionsInvariantSpec.value_holds, this does not
   stop at a partial application. *)
Fixpoint value_no_mark (frames : list frame) (v : value) : bool :=
  match v with
  | Value_const _ | Value_boxed_number _ | Value_location _ => true
  | Value_closure _ value_slots | Value_closure_of_set _ _ value_slots =>
    forallb (fun '(_, v) => value_no_mark frames v) value_slots
  | Value_symbol s => negb (on_stack s frames)
  | Value_block _ fields => forallb (value_no_mark frames) fields
  | Value_immutable_array elements => forallb (value_no_mark frames) elements
  | Value_partial_application callee args _ =>
    value_no_mark frames callee && forallb (value_no_mark frames) args
  end.

(* What a variable may hold: any symbol, a mark included, or a value
   that is not a symbol and holds no mark. *)
Definition variable_no_mark (frames : list frame) (v : value) : bool :=
  match v with
  | Value_symbol _ => true
  | _ => value_no_mark frames v
  end.

(* The same as [value_no_mark], of what a symbol is bound to. *)
Definition allocated_no_mark (frames : list frame)
    (allocated : statically_allocated_value) : bool :=
  match allocated with
  | Static_closure _ value_slots =>
    forallb (fun '(_, v) => value_no_mark frames v) value_slots
  | Static_closure_of_set _ closure_symbols _ value_slots =>
    forallb (fun '(_, s) => negb (on_stack s frames)) closure_symbols
    && forallb (fun '(_, v) => value_no_mark frames v) value_slots
  | Static_block _ fields => forallb (value_no_mark frames) fields
  | Static_boxed_number contents => value_no_mark frames contents
  | Static_string _ | Static_imported _ => true
  | Static_immutable_array elements => forallb (value_no_mark frames) elements
  end.

(* A continuation: the variables of its closure hold marks directly
   only; its extra arguments and the arguments that remain of an
   over-application hold none; and the same of the continuations that
   it holds. *)
Fixpoint cont_no_mark (frames : list frame) (c : cont_value) : bool :=
  match c with
  | Closure _ vars conts _ =>
    forallb (fun '(_, v) => variable_no_mark frames v) vars
    && forallb (fun '(_, c) => cont_no_mark frames c) conts
  | Outer _ => true
  | Extra_args c extra =>
    cont_no_mark frames c && forallb (value_no_mark frames) extra
  | Over_application remaining _ _ return_to exn_to =>
    forallb (value_no_mark frames) remaining
    && cont_no_mark frames return_to
    && cont_no_mark frames exn_to
  end.

(* The first part of the test. *)
Definition marks_in_variables (rs : region_state) : bool :=
  let s := rs_state rs in
  let frames := rs_frames rs in
  forallb (fun '(_, v) => variable_no_mark frames v) (state_vars s)
  && forallb (fun '(_, c) => cont_no_mark frames c) (state_conts s)
  && forallb
       (fun '(_, allocated) => allocated_no_mark frames allocated)
       (static_values (state_statics s))
  && forallb
       (fun '(_, allocated) => allocated_no_mark frames allocated)
       (live_locals frames).

(* ======================================================================== *)
(* The frames are in the order of their marks                               *)
(* ======================================================================== *)

(* The frames are in order below [above]: the mark of each frame was
   drawn before what is above it, and so was each symbol that the
   frame binds, which was drawn after the mark. The newest frame is
   first. *)
Fixpoint frames_ordered_below (above : symbol) (frames : list frame) : bool :=
  match frames with
  | [] => true
  | f :: below =>
    Pos.ltb (frame_mark f) above
    && forallb
         (fun '(s, _) => Pos.ltb (frame_mark f) s && Pos.ltb s above)
         (frame_locals f)
    && frames_ordered_below (frame_mark f) below
  end.

(* The second part of the test: below the next symbol to be drawn. *)
Definition frames_ordered (rs : region_state) : bool :=
  frames_ordered_below (rs_next rs) (rs_frames rs).

(* ======================================================================== *)
(* What is held is older than the marks that the variables hold             *)
(* ======================================================================== *)

(* Each symbol among [vs] was drawn before the mark of each frame that
   the variables [vars] hold. *)
Definition held_older (frames : list frame) (vars : env) (vs : list value)
    : bool :=
  forallb
    (fun '(_, w) =>
       match w with
       | Value_symbol m =>
         negb (on_stack m frames)
         || forallb
              (fun v =>
                 match v with
                 | Value_symbol s => Pos.ltb s m
                 | _ => true
                 end)
              vs
       | _ => true
       end)
    vars.

(* The same of each closure that the continuation holds: of its
   variables, and of what the continuations of its environment have
   held; and so on for those. *)
Fixpoint cont_held_older (frames : list frame) (c : cont_value) : bool :=
  match c with
  | Closure _ vars conts _ =>
    held_older frames vars (flat_map (fun '(_, c) => cont_held c) conts)
    && forallb (fun '(_, c) => cont_held_older frames c) conts
  | Outer _ => true
  | Extra_args c _ => cont_held_older frames c
  | Over_application _ _ _ return_to exn_to =>
    cont_held_older frames return_to && cont_held_older frames exn_to
  end.

(* ======================================================================== *)
(* The test                                                                 *)
(* ======================================================================== *)

Definition calls_kept (rs : region_state) : bool :=
  let s := rs_state rs in
  let frames := rs_frames rs in
  marks_in_variables rs
  && frames_ordered rs
  && forallb (value_usable rs) (held rs)
  && held_older frames (state_vars s) (held rs)
  && forallb (fun '(_, c) => cont_held_older frames c) (state_conts s).

(* ======================================================================== *)
(* Statements                                                               *)
(* ======================================================================== *)

(* The state that a run starts in passes the test: it has no frame and
   no continuation. *)
Definition calls_kept_initial_statement : Prop :=
  ∀ (e : expr) (statics : static_env) (base : symbol),
    calls_kept (initial_region_state e statics base) = true.

(* A step keeps it, of a state that has the invariant. *)
Definition region_step_keeps_calls_statement : Prop :=
  ∀ rs rs' : region_state,
    region_invariant rs = true ->
    calls_kept rs = true ->
    region_step rs = Region_step rs' ->
    calls_kept rs' = true.

(* So every state of a run passes it. *)
Definition region_run_keeps_calls_statement : Prop :=
  ∀ (fuel : nat) (rs rs' : region_state),
    region_invariant rs = true ->
    calls_kept rs = true ->
    region_run fuel rs = Region_step rs' ->
    calls_kept rs' = true.

(* In every state of a run from the state that the test tool starts
   in, what the continuations have held since a call may be
   evaluated. *)
Definition held_stays_usable_statement : Prop :=
  ∀ (e : expr) (statics : static_env) (base : symbol) (fuel : nat)
    (rs : region_state),
    region_names_fresh base e statics = true ->
    region_run fuel (initial_region_state e statics base) = Region_step rs ->
    forallb (value_usable rs) (held rs) = true.

(* The values that such a run exits with may all be evaluated, under
   the state that the run left by. *)
Definition exit_values_usable_statement : Prop :=
  ∀ (e : expr) (statics : static_env) (base : symbol) (fuel : nat)
    (k : continuation) (vs : list value) (rs : region_state),
    region_names_fresh base e statics = true ->
    region_run fuel (initial_region_state e statics base)
    = Region_exit k vs rs ->
    forallb (value_usable rs) vs = true.

(* FlambdaRegionsErasureSpec.erasure_exits_statement, with each value
   of the run on the machine of regions erased to the value of the run
   of the erased term at its place. *)
Definition erasure_exits_erased_statement : Prop :=
  ∀ (e : expr) (statics : static_env) (base : symbol) (fuel : nat)
    (k : continuation) (vs : list value) (rs : region_state),
    region_names_fresh base e statics = true ->
    erasable e statics = true ->
    region_run fuel (initial_region_state e statics base)
    = Region_exit k vs rs ->
    ∃ (vs' : list value) (hp : heap),
      behaves (erased_initial_state e statics) (Exits k vs' hp)
      ∧ Forall2 (value_erased rs) vs vs'.

(* FlambdaRegionsErasureSpec.erasure_adds_no_exit_statement, in the
   same way. *)
Definition erasure_adds_no_exit_erased_statement : Prop :=
  ∀ (e : expr) (statics : static_env) (base : symbol) (k : continuation)
    (vs' : list value) (hp : heap),
    region_names_fresh base e statics = true ->
    erasable e statics = true ->
    behaves (erased_initial_state e statics) (Exits k vs' hp) ->
    ∃ fuel : nat,
      (∃ fault : region_fault,
        region_run fuel (initial_region_state e statics base)
        = Region_undefined fault)
      ∨ (∃ (vs : list value) (rs : region_state),
          region_run fuel (initial_region_state e statics base)
          = Region_exit k vs rs
          ∧ Forall2 (value_erased rs) vs vs').
