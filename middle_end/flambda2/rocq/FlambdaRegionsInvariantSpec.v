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

(* The invariant of the machine of regions (FlambdaRegionsSpec.v): what
   holds of every state of a run, and what follows of the values that
   are in it.

   This file is to be reviewed: all of it. It adds nothing to the
   machine: [region_step] and the rules of FlambdaRegionsSpec.v keep
   their text. FlambdaRegionsInvariantProofs.v proves the statements
   at the end.

   The statements of FlambdaRegionsSpec.v are laws of one step. Two
   things were said there in words and stated of no run: that the
   symbols which the machine draws are apart from those of the term,
   and that a value which is still there holds nothing that is gone.
   Here both are parts of one test of a state, [region_invariant],
   which the state that a run starts in passes and which a step keeps.
   A second test, [no_poison], is beside it, with the same two
   statements: no value of the state is the poison constant of the
   kind [Region].

   THE SYMBOLS THAT THE MACHINE DRAWS ([symbols_fresh])

   - What is written (the expression of the state, the code of its
     static environment, the handlers that its continuations hold)
     names no symbol at or above [rs_base]. So a [Let_static] of the
     term never binds a symbol that the machine drew.
   - What is held (the values of the variables and of the
     continuations, the symbols that the static environment and the
     frames bind and what they are bound to) names no symbol at or
     above [rs_next]. So the next symbol drawn is bound nowhere and
     held nowhere.
   - The marks and the local symbols of the frames were drawn: each is
     at or above [rs_base].
   - The mark of a frame is bound to nothing; and no symbol is bound
     both in a frame and in the static environment of the state.

   This is not FlambdaRegionsSpec.region_names_fresh said of every
   state. That test asks that the static environment binds no symbol
   at or above [rs_base], and a run does not keep that: a local
   allocation made with no frame binds a symbol that the machine drew
   in the static environment of the state, for ever (decision D44).
   Here the static environment is asked what the frames are asked:
   nothing at or above [rs_next]. [region_names_fresh] is what the
   test tool computes once, before the run, and it gives the invariant
   of the first state ([region_invariant_initial_statement]).

   WHAT A VALUE HOLDS ([values_live])

   A symbol is of the heap ([heap_symbol]) if the machine did not draw
   it, or if the static environment of the state binds it: a symbol of
   the term or of the imports, or a local allocation made with no
   frame, which nothing ends. Every other symbol that the machine drew
   is local: in a frame, or gone, or a mark. "Of the heap" is said of
   this machine, where such a symbol is never gone; it is not that the
   program allocates it on its heap, nor that the symbol is bound (a
   symbol of another unit is of the heap and bound to nothing).

   - What the static environment of the state binds a symbol to holds
     symbols of the heap only.
   - Of what a frame binds a symbol to, each value that it holds
     directly (a field, an element, a value slot, the contents of a
     boxed number) is a symbol of the heap, a symbol that the same
     frame or a frame below it binds, or a value that is not a symbol
     and holds symbols of the heap only; and the symbols that a set of
     closures gives its function slots are of the heap, of that frame
     or of a frame below. So it holds nothing that is gone before it
     is. This is what a local allocation is made of: the values of
     simples, of which the next point asks as much. To ask only that
     every symbol held, however deep, is of the heap, of that frame or
     of a frame below would not be kept by a step: a field that is a
     block which holds a local symbol could be read into a variable,
     and the next point refuses that.
   - A value of a variable, or one that a continuation holds (in the
     environment of a closure, among its extra arguments, among the
     arguments that remain of an over-application), is any symbol, or
     a value that is not a symbol and holds symbols of the heap only.
     A variable may hold a symbol that is gone, or a mark: decision
     D45 is a rule about evaluating it, not about holding it. A
     mark that is given to a region parameter is held by it as the
     variable of a begin holds its own.

   "Holds" is deep ([value_holds]): the fields of a block that is a
   field are looked at. It stops at a partial application, of which
   only this is asked: its callee and each of its arguments is any
   symbol, or holds symbols of the heap only. A partial application is
   a value that the machine makes with no rule of regions (decision
   D46 is not asked of it), so it may keep a local value among its
   arguments, and go on keeping it when it is gone.

   WHAT FOLLOWS

   - A symbol that is bound is bound to something that holds only
     symbols that are bound, or that the machine did not draw: a live
     value holds only live values
     ([live_holds_live_statement]).
   - What the static environment of the state binds holds no symbol of
     a frame: a heap value holds nothing local
     ([heap_holds_no_local_statement]). The values of the model that
     are not symbols are heap values too, and the same is said of
     those that a variable holds ([kept_holds_no_local_statement]).
   - The values that a run exits with: each is a symbol, or holds
     only what is live ([exit_values_kept_statement]). No more can be
     said from the invariant. The arguments that the term gives at
     the exit were evaluated there, and so are live (decision D45).
     But a continuation may hold extra arguments, which were
     evaluated where the continuation was made, at a call that gave
     them to its exception continuation; they go to the exit as they
     are, and the invariant asks of them what it asks of a variable.
     That they are still live when the exit is reached is not stated.
   - What is gone stays gone, over a run and not one step
     ([gone_stays_gone_run_statement]).
   Each stops at a partial application, as above.

   NO POISON OF THE KIND [Region] ([no_poison])

   A begin binds its variable to a mark, and a mark is no value that
   may be evaluated. The machine of
   FlambdaSemanticsSpec.v has one value of the kind [Region], the
   poison constant of that kind, and it would let a term give that to
   a region parameter (FlambdaRegionsSpec.v, "a region given to a
   continuation"). FlambdaRegionsSpec.erase is not meant for such a
   term. [no_poison] is the test that the state has no such constant
   anywhere: in what is written (the expression, the handlers that
   the continuations hold, the code) and, deeply, in what is held (the
   values of the variables and of the continuations, what the static
   environment and the frames bind). The state that a run starts in
   passes it if the term and the static environment have no such
   constant (FlambdaRegionsSpec.no_region_poison,
   [statics_no_region_poison]), and a step keeps it: no primitive
   makes the constant. What follows is what the erasure needs: a value
   of the state that may be evaluated is not of the kind [Region]
   ([usable_not_region_statement]).

   It is a test apart from [region_invariant]: neither needs the
   other.

   WHAT IS NOT HERE

   - Nothing about the plain run of the term with its regions dropped:
     that is FlambdaRegionsErasureSpec.v, which needs this file.
   - Nothing about the heap of the state: [step] does not run the
     primitives of the heap, and the machine of regions makes no cell.
   - That the symbols bound are bound once. A frame and the static
     environment bind no symbol in common, and the mark of a frame is
     bound to nothing; that two frames bind no symbol in common is
     true, and is not asked, as nothing here needs it. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaRegionsSpec.
Import ListNotations.

(* ======================================================================== *)
(* What a value holds                                                       *)
(* ======================================================================== *)

(* Every symbol that the value holds is one of [ok]: the value itself
   if it is a symbol, and the symbols in its fields, its elements and
   its value slots, however deep. Of a partial application: its callee
   and each of its arguments is any symbol, or holds symbols of [ok]
   only. *)
Fixpoint value_holds (ok : symbol -> bool) (v : value) : bool :=
  match v with
  | Value_const _ | Value_boxed_number _ | Value_location _ => true
  | Value_closure _ value_slots | Value_closure_of_set _ _ value_slots =>
    forallb (fun '(_, v) => value_holds ok v) value_slots
  | Value_symbol s => ok s
  | Value_block _ fields => forallb (value_holds ok) fields
  | Value_immutable_array elements => forallb (value_holds ok) elements
  | Value_partial_application callee args _ =>
    match callee with
    | Value_symbol _ => true
    | _ => value_holds ok callee
    end
    && forallb
         (fun arg =>
            match arg with
            | Value_symbol _ => true
            | _ => value_holds ok arg
            end)
         args
  end.

(* What a variable may hold: any symbol, or a value that is not a
   symbol and holds symbols of [ok] only. It is what [value_holds] asks
   of the callee and of the arguments of a partial application. *)
Definition value_kept (ok : symbol -> bool) (v : value) : bool :=
  match v with
  | Value_symbol _ => true
  | _ => value_holds ok v
  end.

(* The same of what a symbol is bound to. The symbols that a static set
   of closures gives its function slots are held too. *)
Definition allocated_holds (ok : symbol -> bool)
    (allocated : statically_allocated_value) : bool :=
  match allocated with
  | Static_closure _ value_slots =>
    forallb (fun '(_, v) => value_holds ok v) value_slots
  | Static_closure_of_set _ closure_symbols _ value_slots =>
    forallb (fun '(_, s) => ok s) closure_symbols
    && forallb (fun '(_, v) => value_holds ok v) value_slots
  | Static_block _ fields => forallb (value_holds ok) fields
  | Static_boxed_number contents => value_holds ok contents
  | Static_string _ | Static_imported _ => true
  | Static_immutable_array elements => forallb (value_holds ok) elements
  end.

(* What a local allocation holds directly: a symbol of [near], or a
   value that is not a symbol and holds symbols of [ok] only. With
   [near] and [ok] the same it is [value_holds]. *)
Definition value_placed (near ok : symbol -> bool) (v : value) : bool :=
  match v with
  | Value_symbol s => near s
  | _ => value_holds ok v
  end.

(* The same of what a symbol is bound to: each value that it holds
   directly, and the symbols that a static set of closures gives its
   function slots, which are to be of [near]. *)
Definition allocated_placed (near ok : symbol -> bool)
    (allocated : statically_allocated_value) : bool :=
  match allocated with
  | Static_closure _ value_slots =>
    forallb (fun '(_, v) => value_placed near ok v) value_slots
  | Static_closure_of_set _ closure_symbols _ value_slots =>
    forallb (fun '(_, s) => near s) closure_symbols
    && forallb (fun '(_, v) => value_placed near ok v) value_slots
  | Static_block _ fields => forallb (value_placed near ok) fields
  | Static_boxed_number contents => value_placed near ok contents
  | Static_string _ | Static_imported _ => true
  | Static_immutable_array elements =>
    forallb (value_placed near ok) elements
  end.

(* What a continuation holds is kept: the environment of a closure and
   the continuations that it was defined under, the extra arguments,
   the arguments that remain of an over-application. *)
Fixpoint cont_kept (ok : symbol -> bool) (c : cont_value) : bool :=
  match c with
  | Closure _ vars conts _ =>
    forallb (fun '(_, v) => value_kept ok v) vars
    && forallb (fun '(_, c) => cont_kept ok c) conts
  | Outer _ => true
  | Extra_args c extra => cont_kept ok c && forallb (value_kept ok) extra
  | Over_application remaining _ _ return_to exn_to =>
    forallb (value_kept ok) remaining
    && cont_kept ok return_to
    && cont_kept ok exn_to
  end.

(* ======================================================================== *)
(* The symbols that the machine draws                                       *)
(* ======================================================================== *)

(* What a continuation has that is written names no symbol at or above
   [base]: its handler, the handlers of its recursive group, and the
   same of the continuations that it holds. What it holds names no
   symbol at or above [next]. *)
Fixpoint cont_below (base next : symbol) (c : cont_value) : bool :=
  match c with
  | Closure h vars conts recursive =>
    handler_below base h
    && forallb (fun '(_, v) => value_below next v) vars
    && forallb (fun '(_, c) => cont_below base next c) conts
    && forallb (fun '(_, h) => handler_below base h) recursive
  | Outer _ => true
  | Extra_args c extra =>
    cont_below base next c && forallb (value_below next) extra
  | Over_application remaining _ _ return_to exn_to =>
    forallb (value_below next) remaining
    && cont_below base next return_to
    && cont_below base next exn_to
  end.

(* A symbol that the machine drew: at or above [base], below [next]. *)
Definition symbol_between (base next s : symbol) : bool :=
  Pos.leb base s && Pos.ltb s next.

(* The symbols of a frame were drawn, and what it binds them to names
   no symbol at or above [next]. *)
Definition frame_fresh (base next : symbol) (f : frame) : bool :=
  symbol_between base next (frame_mark f)
  && forallb
       (fun '(s, allocated) =>
          symbol_between base next s && allocated_below next allocated)
       (frame_locals f).

(* The first part of the invariant. *)
Definition symbols_fresh (rs : region_state) : bool :=
  let s := rs_state rs in
  let base := rs_base rs in
  let next := rs_next rs in
  let statics := state_statics s in
  Pos.leb base next
  (* What is written. *)
  && expr_below base (state_expr s)
  && forallb (fun '(_, c) => code_below base c) (static_code statics)
  (* What is held. *)
  && forallb (fun '(_, v) => value_below next v) (state_vars s)
  && forallb (fun '(_, c) => cont_below base next c) (state_conts s)
  && forallb
       (fun '(sym, allocated) =>
          symbol_below next sym && allocated_below next allocated)
       (static_values statics)
  && forallb (frame_fresh base next) (rs_frames rs)
  (* The mark of a frame is bound to nothing. *)
  && forallb
       (fun f =>
          negb
            (is_some (lookup (frame_mark f) (static_values (live_statics rs)))))
       (rs_frames rs)
  (* No symbol is bound in a frame and in the static environment. *)
  && forallb
       (fun '(sym, _) => negb (is_some (lookup sym (static_values statics))))
       (live_locals (rs_frames rs)).

(* ======================================================================== *)
(* What is live                                                             *)
(* ======================================================================== *)

(* A symbol of the heap: one that the machine did not draw, or one that
   the static environment of the state binds. *)
Definition heap_symbol (rs : region_state) (s : symbol) : bool :=
  negb (drawn rs s)
  || is_some (lookup s (static_values (state_statics (rs_state rs)))).

(* A symbol that the frames given bind. *)
Definition local_in (frames : list frame) (s : symbol) : bool :=
  is_some (lookup s (live_locals frames)).

(* Of what each frame binds, each value held directly is a symbol of the
   heap, of that frame or of a frame below it, or a value that is not a
   symbol and holds symbols of the heap only. The newest frame is
   first. *)
Fixpoint frames_hold (rs : region_state) (frames : list frame) : bool :=
  match frames with
  | [] => true
  | f :: below =>
    forallb
      (fun '(_, allocated) =>
         allocated_placed
           (fun s => heap_symbol rs s || local_in (f :: below) s)
           (heap_symbol rs) allocated)
      (frame_locals f)
    && frames_hold rs below
  end.

(* The second part of the invariant. *)
Definition values_live (rs : region_state) : bool :=
  let s := rs_state rs in
  forallb
    (fun '(_, allocated) => allocated_holds (heap_symbol rs) allocated)
    (static_values (state_statics s))
  && frames_hold rs (rs_frames rs)
  && forallb (fun '(_, v) => value_kept (heap_symbol rs) v) (state_vars s)
  && forallb (fun '(_, c) => cont_kept (heap_symbol rs) c) (state_conts s).

(* The invariant. *)
Definition region_invariant (rs : region_state) : bool :=
  symbols_fresh rs && values_live rs.

(* ======================================================================== *)
(* No poison of the kind [Region]                                           *)
(* ======================================================================== *)

(* A continuation has no poison constant of the kind [Region]: in its
   handler and the handlers of its recursive group, in what it holds,
   and the same of the continuations that it holds. *)
Fixpoint cont_no_region_poison (c : cont_value) : bool :=
  match c with
  | Closure h vars conts recursive =>
    handler_no_region_poison h
    && forallb (fun '(_, v) => value_no_region_poison v) vars
    && forallb (fun '(_, c) => cont_no_region_poison c) conts
    && forallb (fun '(_, h) => handler_no_region_poison h) recursive
  | Outer _ => true
  | Extra_args c extra =>
    cont_no_region_poison c && forallb value_no_region_poison extra
  | Over_application remaining _ _ return_to exn_to =>
    forallb value_no_region_poison remaining
    && cont_no_region_poison return_to
    && cont_no_region_poison exn_to
  end.

(* The test: nothing that the state has written, and nothing that it
   holds, is the poison constant of the kind [Region]. *)
Definition no_poison (rs : region_state) : bool :=
  let s := rs_state rs in
  no_region_poison (state_expr s)
  && statics_no_region_poison (state_statics s)
  && forallb (fun '(_, v) => value_no_region_poison v) (state_vars s)
  && forallb (fun '(_, c) => cont_no_region_poison c) (state_conts s)
  && forallb
       (fun '(_, allocated) => allocated_no_region_poison allocated)
       (live_locals (rs_frames rs)).

(* ======================================================================== *)
(* Statements                                                               *)
(* ======================================================================== *)

(* The state that a run starts in has the invariant, if the symbols
   that the machine draws are apart from those of the term and of the
   static environment that it is run in: the test that the test tool
   makes. *)
Definition region_invariant_initial_statement : Prop :=
  ∀ (e : expr) (statics : static_env) (base : symbol),
    region_names_fresh base e statics = true ->
    region_invariant (initial_region_state e statics base) = true.

(* A step keeps it. *)
Definition region_step_keeps_invariant_statement : Prop :=
  ∀ rs rs' : region_state,
    region_invariant rs = true ->
    region_step rs = Region_step rs' ->
    region_invariant rs' = true.

(* So every state of a run has it. *)
Definition region_run_keeps_invariant_statement : Prop :=
  ∀ (fuel : nat) (rs rs' : region_state),
    region_invariant rs = true ->
    region_run fuel rs = Region_step rs' ->
    region_invariant rs' = true.

(* A symbol that may be evaluated: FlambdaRegionsSpec.value_usable of
   it. One that the machine did not draw, or one that [live_statics]
   binds. *)
Definition live_symbol (rs : region_state) (s : symbol) : bool :=
  value_usable rs (Value_symbol s).

(* A symbol that may be evaluated and that is in no frame. *)
Definition not_local_symbol (rs : region_state) (s : symbol) : bool :=
  live_symbol rs s && negb (value_local rs (Value_symbol s)).

(* A live value holds only live values: what a symbol is bound to, in a
   frame or in the static environment of the state, holds only symbols
   that may be evaluated. *)
Definition live_holds_live_statement : Prop :=
  ∀ (rs : region_state) (s : symbol)
    (allocated : statically_allocated_value),
    region_invariant rs = true ->
    lookup s (static_values (live_statics rs)) = Some allocated ->
    allocated_holds (live_symbol rs) allocated = true.

(* A heap value holds nothing local: what the static environment of the
   state binds a symbol to holds no symbol of a frame, and none that is
   gone. *)
Definition heap_holds_no_local_statement : Prop :=
  ∀ (rs : region_state) (s : symbol)
    (allocated : statically_allocated_value),
    region_invariant rs = true ->
    lookup s (static_values (state_statics (rs_state rs))) = Some allocated ->
    allocated_holds (not_local_symbol rs) allocated = true.

(* The same of a value of a variable that is not a symbol. *)
Definition kept_holds_no_local_statement : Prop :=
  ∀ (rs : region_state) (x : variable) (v : value),
    region_invariant rs = true ->
    lookup x (state_vars (rs_state rs)) = Some v ->
    value_kept (not_local_symbol rs) v = true.

(* The values that a run exits with are each a symbol, or a value that
   is not a symbol and holds only symbols that may be evaluated. The
   state is the one that the run left by, which has the invariant. A
   symbol at the exit need not be one that may be evaluated: see the
   header. *)
Definition exit_values_kept_statement : Prop :=
  ∀ (fuel : nat) (rs rs' : region_state) (k : continuation)
    (vs : list value),
    region_invariant rs = true ->
    region_run fuel rs = Region_exit k vs rs' ->
    region_invariant rs' = true
    ∧ forallb (value_kept (live_symbol rs')) vs = true.

(* What is gone stays gone, over a run: FlambdaRegionsSpec.gone of a
   state that has the invariant holds of every state that a run from it
   comes to. *)
Definition gone_stays_gone_run_statement : Prop :=
  ∀ (fuel : nat) (rs rs' : region_state) (s : symbol),
    region_invariant rs = true ->
    region_run fuel rs = Region_step rs' ->
    gone rs s ->
    gone rs' s.

(* The state that a run starts in has no poison constant of the kind
   [Region], if the term and the static environment that it is run in
   have none. *)
Definition no_poison_initial_statement : Prop :=
  ∀ (e : expr) (statics : static_env) (base : symbol),
    no_region_poison e = true ->
    statics_no_region_poison statics = true ->
    no_poison (initial_region_state e statics base) = true.

(* A step keeps it, and so does a run. *)
Definition region_step_keeps_no_poison_statement : Prop :=
  ∀ rs rs' : region_state,
    no_poison rs = true ->
    region_step rs = Region_step rs' ->
    no_poison rs' = true.

Definition region_run_keeps_no_poison_statement : Prop :=
  ∀ (fuel : nat) (rs rs' : region_state),
    no_poison rs = true ->
    region_run fuel rs = Region_step rs' ->
    no_poison rs' = true.

(* In such a state, a simple whose value may be evaluated has a value
   that is not of the kind [Region]. So a parameter that is a region
   is given a mark, or nothing that the machine has a step for. *)
Definition usable_not_region_statement : Prop :=
  ∀ (rs : region_state) (s : simple) (v : value),
    no_poison rs = true ->
    simple_no_region_poison s = true ->
    eval_simple (state_vars (rs_state rs)) s = Some v ->
    value_usable rs v = true ->
    kind_of_value v <> Region.

