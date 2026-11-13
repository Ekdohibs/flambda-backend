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

(* Regions and local allocation: a machine beside that of
   FlambdaSemanticsSpec.v, which runs a term that has the primitives of
   regions (FlambdaPrimitiveModel.region_primitive) and says where a
   value allocated in a region is used after it is gone.

   This file is to be reviewed: all of it. It is a first stage. It
   checks a term; nothing is proved here of Simplify or of the
   validation, which still drops regions (decision D37).
   FlambdaRegionsProofs.v proves the statements at the end.

   WHAT RUNS

   When the program runs there are no regions. There is one stack of
   local allocations and a pointer to its top. A [Begin_region] or a
   [Begin_try_region] that is not a ghost reads the pointer; an
   [End_region] or an [End_try_region] that is not a ghost writes back
   the pointer that its region read; a local allocation moves the
   pointer and uses the space, whatever region its mode names
   ([To_cmm_primitive], [To_cmm_shared.alloc_mode_for_allocations_to_cmm]).
   Nothing is passed to a function for the region of its code. A raise
   does not write the pointer: the handler's [End_try_region] does. A
   return from the body of a [try] leaves the pointer where it is, by
   design, as the body may return something local.

   The conversion from Lambda relies on the write of the pointer
   ending every region begun since. Where a region is in tail position
   of another, the term that it makes has no end for the inner region
   on that way out: the body returns to the continuation that ends the
   outer one ([Lambda_to_flambda], the case of [Lregion]: "we will
   jump directly to the region closure continuation for the outermost
   open region. For this to be correct we rely on the fact that the
   code structure here [...] ensures this is equivalent to going
   through the sequence of nested [region_closure_continuation]s we
   generate").

   THE STATE

   The machine here is the state of FlambdaSemanticsSpec.v with a stack
   of frames, one for each region begun and not yet cut away, newest
   first. A frame has a mark, which is what the variable of the region
   is bound to, whether the region is that of a [try], and the local
   allocations made while it was the newest frame.

   - [Begin_region] and [Begin_try_region] push a frame with a fresh
     mark.
   - A local allocation is put in the newest frame. Not in the frame
     of the region that its mode names: decision D44.
   - [End_region] and [End_try_region] pop the frames down to and with
     the frame of their region. So the end of a region ends every
     region begun since that is still there, as the write of the
     pointer does.
   - A raise does nothing to the stack.
   - An [Apply_cont] may give a region to a continuation, as an
     argument. That does nothing to the stack either.

   The program has no stack of marks, and makes no test that a region
   is still there: a region is the number that its begin read. The
   stack of frames is the model's, and what has no step for it
   (decision D47) is a restriction of the model.

   A LOCAL ALLOCATION IS A SYMBOL

   The allocated values of the model have no identity, so "this block
   is gone" needs a name for the block. A local allocation binds a
   fresh symbol to what was made ([localise]), in its frame, and its
   value is the symbol. The machine of FlambdaSemanticsSpec.v already
   reads a symbol bound to a block as the block, because of lifted
   constants; [step] is given the static environment of the term with
   the local symbols of the frames in front ([live_statics]). When a
   frame is popped its symbols are bound to nothing any more: by
   decision D8 every operation on one is then undefined in [step], and
   this machine asks more (decision D45, below).

   The symbols are drawn from a counter that only goes up ([rs_next]),
   and popping a frame gives none back. So a symbol that is gone is
   never bound again ([region_step_draws_upwards_statement],
   [gone_stays_gone_statement]). That is here what "the static
   environment only grows" is to the machine of
   FlambdaSemanticsSpec.v: the static environment that [step] is given
   does shrink, by the symbols of the frames popped and by nothing
   else.

   A local allocation made with no frame on the stack is bound in the
   static environment of the state, and so for ever: no end of a
   region would free it (decision D44). It is counted
   ([count_no_frame]).

   The allocation primitives of the model have no mode. The mode
   [Local] of an allocation is a second [Let] directly after that of
   the allocation, of the primitive [Local_allocation] applied to the
   variable of the allocation and, where the conversion has a
   variable for the region that the mode names, to it. The two are
   one step here ([marker_of], [local_allocation]), which binds the
   variable of each of the two [Let]s to the symbol: the value is
   never bound as a heap value. What the conversion must make: the
   second [Let] directly under the first, at the kind [Value], its
   first argument the variable that the first binds. The conversion
   of the test tool has the second bind the same variable again. The
   step does not ask that: whether two binders have one name is not
   kept by a renaming of bound names, and the step is to be the same
   on two terms that differ by one. A [Local_allocation] anywhere
   else, or after a [Let] that is not an allocation, has no step
   ([Not_of_the_conversion]).

   A MARK

   A mark is a symbol that this machine draws and that is never bound
   to anything; the variable of a region is bound to [Value_symbol] of
   it. So no primitive of FlambdaPrimitiveSpec.v has a value on a mark
   (FlambdaPrimitiveSpec.eval_prim_symbols_bound_statement). [step]
   would pass a mark on where it reads nothing (a [Let] of a simple,
   an argument, a value slot, a field of a static block); this machine
   has no step where a simple that [step] is about to evaluate has a
   mark for its value, but for an argument that gives a region to a
   continuation (below). A mark is not a value that may be evaluated
   ([value_usable]): it is a drawn symbol that nothing binds. Apart
   from being given to a continuation, it is read in two places only:
   by the primitive that ends its region, and by a [Local_allocation]
   that names it.

   A REGION GIVEN TO A CONTINUATION

   Simplify may make the variable of a region a parameter of a
   continuation. Where it lifts a handler out of another, which it
   does with [-flambda2-match-in-match], what the handler uses of the
   other becomes its parameters, and the region that it ends or
   allocates in is one of them. When the program runs, that is a
   number given as an argument.

   Here an [Apply_cont] may have, for an argument, a mark that is on
   the stack, at a parameter that the handler of the continuation
   gives the kind [Region] ([region_places]). The parameter is bound
   to the mark ([region_apply_cont]); the stack is left as it is. The
   end of the region, and a [Local_allocation] that names it, read the
   mark through the parameter as through the variable that the begin
   bound. [step] has no step for such an [Apply_cont]: for it a mark is
   a symbol, of the kind [Value].

   That is all that may be done with a region. A mark that is on the
   stack and that a step evaluates anywhere else has no step
   ([Region_used_as_value]): as the callee or an argument of a call, in
   a block or a closure, given to a way out of the term or to a
   parameter of another kind. And of the symbols that the machine
   drew, only a mark that is on the stack may be given at a parameter
   of the kind [Region]: a region that is gone has no step there
   (decision D47).

   A value that may be evaluated and is given at such a parameter is
   left to [step], as before. [step] refuses it for its kind, unless
   it is a poison constant of the kind [Region], the one value that
   FlambdaValueSpec.kind_of_value gives that kind. Such a constant is
   no region: the parameter is bound to it, and its end, or a
   [Local_allocation] that names it, has no step. The conversion gives
   nothing but the variable of a region at such a parameter.

   The machine does not tell the handler that a call returns to from
   another. So code that gave a mark to its return continuation would
   have a step, where the call returns to a handler whose parameter is
   of the kind [Region]. No term of the compiler is known to do that,
   and the conversion refuses a call that returns or raises to such a
   handler ([calls_return_no_region]).

   The symbols drawn, for marks and for local allocations, are
   [rs_base], its successor, and so on. [region_names_fresh] is the
   test that a term and a static environment have no symbol at or
   above [rs_base]. The test tool computes it once, of the term that
   it runs and of the imports, before the run. Here it is the premise
   of one statement, about one step ([gone_stays_gone_statement]);
   that a step keeps it is not stated ("what is not stated", below).

   WHAT HAS NO STEP

   - Decision D45. A simple that the step evaluates
     ([evaluated_simples]) whose value is a symbol that this machine
     drew, that is not bound in [live_statics] and that is the mark of
     no frame: a local allocation whose frame is gone. This is more
     than the program asks. A use of what was freed goes wrong where
     the freed space has been written again, by a later local
     allocation or by the poison of a debugging runtime; where the
     collector has stopped tracing and updating the fields of the
     freed block; or where a root that is still live makes the
     collector look at the freed space. The mark of a region that is
     gone is such a symbol too, and the machine does not tell it from
     a local allocation: where it is evaluated it is refused under
     this rule, but at a parameter that is a region (D47, below).
   - A mark that is on the stack and that a step evaluates anywhere
     but as an argument at a parameter that is a region
     ([Region_used_as_value]). Nothing freed is used there; the
     machine has no meaning for a region as a value.
   - Decision D46. An allocation on the heap one of whose arguments is
     a local value that is in a frame ([value_local]): a [Let] of an
     allocation with no [Local_allocation] after it, and every
     [Let_static]. A local allocation is not asked this: what it holds
     is in a frame at or below its own, and lives as long.
   - Decision D47. The end of a region whose mark is in no frame; a
     [Local_allocation] that names such a region; and an [Apply_cont]
     that gives, at a parameter that is a region, a symbol that the
     machine drew, that nothing binds and that is the mark of no
     frame. The program has no such test: it writes whatever pointer
     the region read, and passes it on.
   - Where [step] has none.

   What is not undefined: leaving code, by a return or by a raise,
   with frames that the code pushed (decision D48); a
   [Local_allocation] that names another region than the newest plain
   one, which is the newest frame that is not of a [try]. Neither is
   found where it happens. What there is of the first: the frames
   that are left in the state when a run ends, and the count of the
   plain frames that the end of another region popped. Of the second:
   a count ([count_other_region]), and only of an allocation that
   names a region by a variable; one that names the region of its
   code names nothing here, and is not counted whatever is open. The
   allocation itself goes in the newest frame, of a [try] or not.

   THE SIMPLES THAT A STEP EVALUATES

   [evaluated_simples] is, for each form, the simples that [step]
   evaluates when it has a step:
   - [Let]: those of the defining expression ([simples_of_named]): the
     simple itself, the arguments of a primitive, the simples of the
     value slots of a set of closures, the closure of a projection;
   - [Let_static]: the fields of the blocks and arrays, the contents
     of the boxed numbers and the simples of the value slots of the
     constants bound; none for code, whose body is not run there;
   - [Apply]: the callee, the arguments, the extra arguments of the
     exception continuation;
   - [Apply_cont]: the arguments (a trap action has no simple). One
     of them may be at the place of a region ([region_places]); no
     other simple is;
   - [Switch]: the scrutinee (the arguments of the arm are those of
     the [Apply_cont] that the next step runs);
   - the two [Let_cont] and [Invalid]: none.
   Where [step] has no step it may stop before it evaluates them all
   (a group one of whose symbols is bound already, a closure at a
   function slot that its set has not got). Then this machine has no
   step either, and only the reason it gives may be another.

   A SYMBOL AND WHAT IT IS BOUND TO

   Where FlambdaValueSpec.v, FlambdaPrimitiveSpec.v and
   FlambdaSemanticsSpec.v treat a symbol and the value that it is
   bound to differently, and what is done here:
   1. [FlambdaValueSpec.project_function_slot]: of a
      [Value_closure_of_set] it is the same set at the other slot; of
      a symbol bound to a [Static_closure_of_set] it is the symbol
      that the binding lists for the other slot. [localise] binds a
      symbol for each function slot of the set, each to the closure at
      its slot with the list of them all, in one frame: the projection
      from the local closure has a value where that from the value
      has one, and it lives as long.
   2. [FlambdaValueSpec.boxed_number_of]: of a symbol, the contents
      must be a constant ([Static_boxed_number] holds a value).
      [localise] binds [Static_boxed_number] of the constant.
   3. [FlambdaValueSpec.string_of] reads symbols only,
      [partial_application_of] values only, [cell_of] locations only.
      Nothing local is a string; [localise] has no value on a partial
      application nor on a location, so neither is ever local here.
   4. [FlambdaPrimitiveSpec.bound_if_symbol] ([Opaque_identity],
      [Make_block], [Make_immutable_array]): true of every value that
      is not a symbol. This is the difference that the machine uses.
   5. [FlambdaValueSpec.imported_of], and by it [identity_of],
      [is_imported_pointer] and [FlambdaPrimitiveSpec.eval_phys_equal]:
      they tell a symbol bound to [Static_imported] from everything
      else. No local symbol is bound to that, so [Phys_equal] has on a
      local symbol what it has on the value: nothing against an
      allocated value, itself included, and 0 against a tagged
      immediate.
   6. [FlambdaSemanticsSpec.eval_simple] gives [Value_symbol] of a
      symbol of the term, bound or not; [eval_static_const],
      [eval_value_slots], the arguments of a call and of an
      [Apply_cont], and [call_code] for [my_closure], pass a symbol on
      without reading it. No symbol of the term is one that the
      machine drew ([region_names_fresh]), and the test of the simples
      ([simples_fault]) is made before [step] is asked.
   7. [FlambdaSemanticsSpec.eval_static_bindings] refuses a symbol that
      the static environment binds already. Here a [Let_static] is run
      on the static environment of the term, without the local
      symbols; it binds none of them ([region_names_fresh]).
   8. An exit ([Exit]) gives the values as they are. [Region_exit]
      gives the state with them: they are to be read under
      [live_statics] of it.
   9. [FlambdaPrimitiveSpec.value_in_range] and the checks of
      FlambdaWellFormedSpec.v do not look at what a symbol is bound
      to. They are not of the machine.
   The readings [closure_of], [closure_in_set], [project_value_slot],
   [block_of], [array_of], [is_allocated] and [kind_of_value] answer
   the same of a block, a boxed number, an array or a closure and of
   a symbol that [localise] binds to it.
   FlambdaHeapSpec.v is not run by [step].

   WHAT THIS STAGE LEAVES OUT

   - Ghost regions, the parent of a region, and the [alloc_region] of
     a mode: nothing runs for them, and the conversion makes nothing
     of them.
   - The region of the code that an allocation is in: nothing is
     passed for it, so the [Local_allocation] names no region, and the
     allocation goes in the newest frame like any other.
   - The modes of calls and the result mode of code, and so whether a
     function leaves a frame behind (decision D48).
   - A partial application: the machine makes it a value, whatever
     the mode that the compiler would give its closure. So it is
     never local here: D46 is not asked of it, and it is never found
     dead, though the program may have freed its closure. A dead
     value among the arguments it keeps is found (D45) where a simple
     that holds it is evaluated, as when the code that is at last
     called reads its parameter. What [step] reads from inside the
     partial application, its callee, is not a simple: if that is
     dead the state has no step for the reason of [step], and not for
     D45.
   - Blocks with an identity and mutable arrays: [step] does not run
     them. In the stage that does, a frame also lists the locations of
     the cells made while it was the newest, as it lists the symbols;
     the cells stay in the heap, and the region state keeps the
     locations of the frames that were popped: a location is dead iff
     it is among them.
   - Everything else that moves the pointer of the local stack when
     the program runs, which is no primitive of the term: local
     allocation by C code and the regions that the runtime begins and
     ends, callbacks, the switch of stack of an effect or of a thread,
     an asynchronous exception, and the helpers that the backend makes
     for a generic application. Among those helpers, the closure that
     currying makes for a partial application may be local: its
     lifetime is not here (the partial application, above).
   - Which variables are roots of the collector. The program keeps as
     a root whatever the liveness of the code says is live, on any
     path; D45 asks only of the simples that a step of the path taken
     evaluates.
   - What the backend does after: a term that passes here can still be
     compiled so that a load is moved after the end of its region.

   WHAT IS NOT STATED

   - The law of this machine under a renaming of variables and of
     continuations, and that a step is the same on two terms that
     differ by bound names. No rewrite is applied to a term with
     regions: the validation drops them (decision D37).
   - That a run does not depend on [rs_base]: of two bases that are
     both fresh ([region_names_fresh]) the runs should differ by the
     symbols drawn and by nothing else.
   - That the machine keeps
     FlambdaClosuresSpec.static_sets_bound of [live_statics]. It
     would be of sets whose function slots are distinct only (the
     comment of [localise]).
   - That a run keeps what [region_names_fresh] tests. As it is
     written it does not: a local allocation made with no frame binds
     a symbol that the machine drew in the static environment of the
     state. So [gone_stays_gone_statement] is a law of one step, under
     a premise on the state before that step, and nothing here says
     that the premise holds of every state of a run.
   - That the run of a term with its regions dropped ([erase]) on the
     machine of FlambdaSemanticsSpec.v is what its run here comes to.
     None of the statements is that.

   THE TERM WITH ITS REGIONS DROPPED ([erase])

   The validation is given terms with their regions dropped (decision
   D37). [erase] is that, as a function of the model, of a term that
   has them:
   - The [Let] of a [Begin_region] or of a [Begin_try_region] is
     dropped: its variable is bound to nothing.
   - The [Let] of an [End_region] binds the unit value. That of an
     [End_try_region] binds it through [Opaque_identity].
   - The [Let] of a [Local_allocation] applied to a variable binds that
     variable again, under the name that the [Let] gives; if the name
     is that of the variable, the [Let] is dropped.
   - A parameter of a handler that is of the kind [Region] is dropped,
     with its kind, and so is the argument at its place in every
     [Apply_cont] of the continuation that a [Let_cont] binds to the
     handler, in the arms of a [Switch] too.
   - Everything else is as it was, and the same is done in the
     handlers, in the bodies of code, and under every [Let].
   A begin or an end that is not applied as above (a begin with an
   argument, an end with none) is left: neither machine has a step
   for it. A [Local_allocation] is erased by its first argument
   alone, whatever follows it: [erase] does not test that it is as
   the conversion makes it ([marker_of]), and this machine has no
   step for one that is not.

   An [Apply_cont] of a continuation that no [Let_cont] of the body
   binds is left as it is: the return or exception continuation of
   code, a way out of the term. So [erase] is meant for a term in
   which no call returns or raises to a handler that has a region
   parameter ([calls_return_no_region]); of another term the erased
   one would give the handler a value that it has no parameter for.
   And it is meant for a term, and a static environment, that have no
   poison constant of the kind [Region] ([no_region_poison],
   [statics_no_region_poison]): given at a region parameter, such a
   constant is a value that the handler may go on to evaluate, where
   the erased handler has no parameter for it. [erasable] is the two
   tests together.

   The term that [erase] gives does not bind the variable of a region.
   A term that uses that variable anywhere but at the end of its
   region, in a [Local_allocation] or as an argument at a region
   parameter has no step on this machine where it does.

   This is, [Let] for [Let] and argument for argument, what the
   conversion of the test tool makes of a region that is not a ghost,
   of an allocation whose mode is [Local] and of a region parameter
   where it drops them: the conversion for the validation. The tool
   converts each term twice, once for the validation and once with
   regions kept, for this machine. It tests of every unit that it runs
   here that [erase] of the second is the first, up to bound names
   (FlambdaCompareSpec.expr_eqb), and that [erasable] holds of the
   second and of the imports; it runs nothing here of a unit that
   fails either. Nothing here states what the two runs have in
   common. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The state                                                                *)
(* ======================================================================== *)

(* A mark: what the variable of a region is bound to, as
   [Value_symbol]. *)
Definition mark : Type := symbol.

(* Whether a frame is of a [Begin_region] or of a [Begin_try_region]. The
   machine does the same with both; the counts tell them apart. *)
Inductive mark_kind : Type :=
  | Plain_mark
  | Try_mark.

(* A region begun and not yet cut away: its mark, its kind, and the
   local allocations made while it was the newest, newest first. *)
Record frame : Type := Frame {
  frame_mark : mark;
  frame_kind : mark_kind;
  frame_locals : list (symbol * statically_allocated_value)
}.

(* What is counted in a run. None of it changes what the machine does.
   - [count_other_region]: the local allocations that named, by a
     variable, a region which was not the newest plain one (the newest
     frame that is not of a [try]). One that names the region of its
     code names no variable, and is never counted.
   - [count_cut_by_end_region], [count_cut_by_end_try_region]: the
     plain frames popped by the end of another region, by the primitive
     that popped them.
   - [count_ends]: the ends of regions.
   - [count_local_allocations]: the local allocations.
   - [count_no_frame]: those of them made with no frame on the stack.
   - [count_local_uses]: the simples evaluated whose value was a local
     value in a frame. An evaluation: an argument that is passed on
     counts, and no memory need be read. *)
Record region_counts : Type := Region_counts {
  count_other_region : nat;
  count_cut_by_end_region : nat;
  count_cut_by_end_try_region : nat;
  count_ends : nat;
  count_local_allocations : nat;
  count_no_frame : nat;
  count_local_uses : nat
}.

Definition no_counts : region_counts := Region_counts 0 0 0 0 0 0 0.

(* The state: that of FlambdaSemanticsSpec.v, whose static environment
   is that of the term and of the local allocations made with no frame;
   the stack of frames, newest first; the first symbol that the machine
   drew and the next that it will draw; the counts. *)
Record region_state : Type := Region_state {
  rs_state : state;
  rs_frames : list frame;
  rs_base : symbol;
  rs_next : symbol;
  rs_counts : region_counts
}.

(* The local allocations that are in a frame. *)
Definition live_locals (frames : list frame)
    : list (symbol * statically_allocated_value) :=
  flat_map frame_locals frames.

(* The static environment that [step] is given: that of the state with
   the local allocations of the frames in front. *)
Definition live_statics (rs : region_state) : static_env :=
  let statics := state_statics (rs_state rs) in
  Static_env (static_code statics)
    (live_locals (rs_frames rs) ++ static_values statics).

(* A symbol that the machine drew. *)
Definition drawn (rs : region_state) (s : symbol) : bool :=
  Pos.leb (rs_base rs) s.

(* Decision D45: a value that may be evaluated. Any value but a symbol
   that the machine drew and that [live_statics] does not bind: the
   symbol of a local allocation whose frame is gone, or a mark. *)

(* CR mshinwell: the step that evaluates a simple whose value was
   allocated in a region that is gone is undefined, whatever is then
   done with the value. The program asks less: a use of what was
   freed goes wrong where the freed space has been written again (a
   later local allocation, the poison of a debugging runtime), where
   the collector has stopped tracing and updating the fields of the
   freed block, or where a root that is still live makes the collector
   look at the freed space. It is so here because a variable that
   holds such a pointer and is still used is a root, and because the
   rule is simple. It would be wrong if the compiler may keep and pass on a
   pointer to freed local memory that nothing then reads: such a term
   fails this check and is right. Decision D45. *)
Definition value_usable (rs : region_state) (v : value) : bool :=
  match v with
  | Value_symbol s =>
    negb (drawn rs s)
    || is_some (lookup s (static_values (live_statics rs)))
  | _ => true
  end.

(* A local value that is in a frame. *)
Definition value_local (rs : region_state) (v : value) : bool :=
  match v with
  | Value_symbol s => is_some (lookup s (live_locals (rs_frames rs)))
  | _ => false
  end.

Definition simple_usable (rs : region_state) (s : simple) : bool :=
  match eval_simple (state_vars (rs_state rs)) s with
  | Some v => value_usable rs v
  | None => true
  end.

Definition simple_local (rs : region_state) (s : simple) : bool :=
  match eval_simple (state_vars (rs_state rs)) s with
  | Some v => value_local rs v
  | None => false
  end.

(* ======================================================================== *)
(* The simples that a step evaluates                                        *)
(* ======================================================================== *)

Definition simples_of_named (defining_expr : named) : list simple :=
  match defining_expr with
  | Simple s => [s]
  | Prim _ args => args
  | Set_of_closures _ value_slots | Closure_of_set _ _ value_slots =>
    map snd value_slots
  | Project_value_slot _ closure
  | Project_value_slot_from _ _ closure
  | Project_function_slot _ _ closure =>
    [closure]
  end.

Definition simples_of_static_const (const : static_const) : list simple :=
  match const with
  | Static_const_set_of_closures _ value_slots
  | Static_const_closure_of_set _ _ _ value_slots =>
    map snd value_slots
  | Static_const_block _ fields => fields
  | Static_const_boxed_number contents => [contents]
  | Static_const_immutable_string _ => []
  | Static_const_immutable_value_array elements => elements
  end.

Definition simples_of_binding (binding : static_binding) : list simple :=
  match binding with
  | Bind_symbol _ const => simples_of_static_const const
  | Bind_code _ _ | Bind_deleted_code _ => []
  end.

(* The simples that [step] evaluates, by the form of the expression. A
   [Let] of a primitive of regions has none: its arguments are read by
   this machine. *)
Definition evaluated_simples (e : expr) : list simple :=
  match e with
  | Let _ _ (Prim (Region_primitive _) _) _ => []
  | Let _ _ defining_expr _ => simples_of_named defining_expr
  | Let_static bindings _ => flat_map simples_of_binding bindings
  | Let_cont_non_recursive _ _ _ | Let_cont_recursive _ _ | Invalid => []
  | Apply ap => ap_callee ap :: ap_args ap ++ exn_extra_simples ap
  | Apply_cont ac => ac_args ac
  | Switch scrutinee _ => [scrutinee]
  end.

(* ======================================================================== *)
(* Local allocation                                                         *)
(* ======================================================================== *)

(* The named expressions that allocate: the primitives that make a
   block, an immutable array or a boxed number, and the sets of
   closures. *)
Definition is_allocation (defining_expr : named) : bool :=
  match defining_expr with
  | Prim (Variadic (Make_block _ | Make_immutable_array _)) _
  | Prim (Unary (Box_number _)) _
  | Set_of_closures _ _
  | Closure_of_set _ _ _ =>
    true
  | _ => false
  end.

(* A symbol for each function slot of a set, from [next] on. *)
Fixpoint number_slots (function_decls : list (function_slot * code_id))
    (next : symbol) : list (function_slot * symbol) :=
  match function_decls with
  | [] => []
  | (slot, _) :: function_decls =>
    (slot, next) :: number_slots function_decls (Pos.succ next)
  end.

(* What a local allocation of the value [v] binds, the first symbol
   being [next], and the symbol that is its value. A block, a boxed
   number, an array and the closure of a set of one function are bound
   to one symbol. A closure of a set at named slots is bound with its
   set: a symbol for each function slot, each bound to the closure at
   that slot, with the symbols of them all. There is nothing for any
   other value: it is not what an allocation makes.

   This is the form that a set lifted to symbols has
   (FlambdaLiftSpec.value_lifted, the rule of a closure of a set). Of
   a set that names one function slot twice, the symbol of the second
   is bound to a closure whose list gives the first for that slot;
   nothing reads that symbol, as every lookup gives the first.
   [eval_named] does not refuse such a set;
   FlambdaWellFormedSpec.well_formed_unit does. *)
Definition localise (next : symbol) (v : value)
    : option (symbol * list (symbol * statically_allocated_value)) :=
  match v with
  | Value_block tag fields => Some (next, [(next, Static_block tag fields)])
  | Value_boxed_number c =>
    Some (next, [(next, Static_boxed_number (Value_const c))])
  | Value_immutable_array elements =>
    Some (next, [(next, Static_immutable_array elements)])
  | Value_closure code value_slots =>
    Some (next, [(next, Static_closure code value_slots)])
  | Value_closure_of_set slot function_decls value_slots =>
    let symbols := number_slots function_decls next in
    let* s := lookup slot symbols in
    Some
      (s,
       map
         (fun '(slot', s') =>
            (s',
             Static_closure_of_set slot' symbols function_decls value_slots))
         symbols)
  | _ => None
  end.

(* The symbol after [count] symbols from [next]. *)
Definition advance (next : symbol) (count : nat) : symbol :=
  Nat.iter count Pos.succ next.

(* The frame of the mark [m] on the stack. *)
Definition on_stack (m : mark) (frames : list frame) : bool :=
  existsb (fun f => Pos.eqb (frame_mark f) m) frames.

(* The frames above that of [m], and those below it; nothing if no
   frame is of [m]. *)
Fixpoint cut (m : mark) (frames : list frame)
    : option (list frame * list frame) :=
  match frames with
  | [] => None
  | f :: frames =>
    if Pos.eqb (frame_mark f) m then Some ([], frames)
    else
      let* '(above, below) := cut m frames in
      Some (f :: above, below)
  end.

Definition is_plain (f : frame) : bool :=
  match frame_kind f with
  | Plain_mark => true
  | Try_mark => false
  end.

(* The mark of the newest frame that is not of a [try]. *)
Definition newest_plain (frames : list frame) : option mark :=
  option_map frame_mark (find is_plain frames).

(* What follows the [Let] of [x]: nothing of regions; a [Let] of
   [Local_allocation] applied to [x], with the variable that it binds,
   the simple of the region it names if it names one, and what follows
   it; or a [Local_allocation] that is not as the conversion makes it.
   The variable that the second [Let] binds is not compared with [x]:
   see the header. The region that it names is not to be [x] itself,
   which is in scope there and is the allocation: of any other
   simple, the value is the same before [x] is bound, where
   [named_region] reads it, and after. *)
Inductive marker : Type :=
  | No_marker
  | Marker (y : variable) (region : option simple) (body : expr)
  | Bad_marker.

Definition marker_of (x : variable) (body : expr) : marker :=
  match body with
  | Let y kind (Prim (Region_primitive Local_allocation) args) body' =>
    match args with
    | Var x' :: region =>
      if Pos.eqb x x' && kind_eqb kind Value
      then
        match region with
        | [] => Marker y None body'
        | [Var z] =>
          if Pos.eqb z x then Bad_marker else Marker y (Some (Var z)) body'
        | [r] => Marker y (Some r) body'
        | _ :: _ :: _ => Bad_marker
        end
      else Bad_marker
    | _ => Bad_marker
    end
  | _ => No_marker
  end.

(* ======================================================================== *)
(* A region given to a continuation                                         *)
(* ======================================================================== *)

(* A mark that is on the stack: a symbol that may not be evaluated
   ([value_usable]) and is the mark of a frame. *)
Definition value_open_mark (rs : region_state) (v : value) : bool :=
  match v with
  | Value_symbol m => negb (value_usable rs v) && on_stack m (rs_frames rs)
  | _ => false
  end.

Definition simple_open_mark (rs : region_state) (s : simple) : bool :=
  match eval_simple (state_vars (rs_state rs)) s with
  | Some v => value_open_mark rs v
  | None => false
  end.

(* Which parameters of a handler are regions: those of the kind
   [Region], by the kinds that the handler gives its parameters. *)
Definition region_params (h : continuation_handler) : list bool :=
  map (fun kind => kind_eqb kind Region) (handler_kinds h).

(* For the simples that the step of a state evaluates
   ([evaluated_simples]), in order, whether each is at the place of a
   region. Only an argument of an [Apply_cont] can be: where the
   continuation is bound, in the state, to a handler, the argument is
   at the place of a region if the parameter at its place is one
   ([region_params]). The handler is the one that the machine finds
   when it applies the continuation ([resolve]), and the kind is the
   one that this handler gives the parameter; nothing is read from the
   [Apply_cont]. No argument of a continuation that is not bound to a
   handler is at the place of a region: a way out of the term, a
   continuation with extra arguments, what is left of an
   over-application. An argument beyond the handler's list of the
   kinds of its parameters is not either. *)
Definition region_places (rs : region_state) : list bool :=
  match state_expr (rs_state rs) with
  | Apply_cont ac =>
    match resolve (ac_continuation ac) (state_conts (rs_state rs)) with
    | Closure h _ _ _ => region_params h
    | _ => []
    end
  | _ => []
  end.

(* The kind of a value, for a parameter that it is given to: a mark
   that is on the stack is of the kind [Region], and any other value of
   the kind that FlambdaValueSpec.kind_of_value gives it, which is
   [Region] of a poison constant of that kind and of nothing else. *)
Definition region_kind_of_value (rs : region_state) (v : value)
    : flambda_kind :=
  if value_open_mark rs v then Region else kind_of_value v.

(* ======================================================================== *)
(* Steps                                                                    *)
(* ======================================================================== *)

(* Why a state has no step. *)
Inductive region_fault : Type :=
  (* Decision D45. *)
  | Dead_value_used
  (* Decision D46. *)
  | Heap_holds_local
  (* Decision D47. *)
  | Region_not_open
  (* A mark that is on the stack, evaluated where no region is to be
     given. *)
  | Region_used_as_value
  (* A primitive of regions that is not as the conversion makes it. *)
  | Not_of_the_conversion
  (* [step] has none. *)
  | No_plain_step.

(* What a step comes to. An exit has the state it left, for the frames
   that are still there, the counts, and the static environment under
   which its values are read. *)
Inductive region_step_result : Type :=
  | Region_step (rs : region_state)
  | Region_exit (k : continuation) (vs : list value) (rs : region_state)
  | Region_undefined (fault : region_fault).

Definition with_statics (s : state) (statics : static_env) : state :=
  State (state_expr s) (state_vars s) (state_conts s) statics (state_heap s).

Definition with_counts (rs : region_state) (counts : region_counts)
    : region_state :=
  Region_state (rs_state rs) (rs_frames rs) (rs_base rs) (rs_next rs) counts.

(* The step of [step], given [live_statics]. The static environment
   that it leaves is not looked at: this is not asked of a
   [Let_static], and at any other expression [step] leaves the static
   environment that it was given ([step_keeps_statics_statement]). *)
Definition plain_step (rs : region_state) : region_step_result :=
  let s := rs_state rs in
  match step (with_statics s (live_statics rs)) with
  | Step s' =>
    Region_step
      (Region_state (with_statics s' (state_statics s)) (rs_frames rs)
         (rs_base rs) (rs_next rs) (rs_counts rs))
  | Exit k vs _ => Region_exit k vs rs
  | Undefined => Region_undefined No_plain_step
  end.

(* A [Let_static]: the step of [step] on the static environment of the
   state, which the group is added to. *)
Definition static_step (rs : region_state) : region_step_result :=
  match step (rs_state rs) with
  | Step s' =>
    Region_step
      (Region_state s' (rs_frames rs) (rs_base rs) (rs_next rs)
         (rs_counts rs))
  | Exit k vs _ => Region_exit k vs rs
  | Undefined => Region_undefined No_plain_step
  end.

(* The state after a [Let] of [x] to [v]. *)
Definition bind_let (rs : region_state) (x : variable) (v : value)
    (body : expr) (statics : static_env) (frames : list frame)
    (next : symbol) (counts : region_counts) : region_state :=
  let s := rs_state rs in
  Region_state
    (State body ((x, v) :: state_vars s) (state_conts s) statics
       (state_heap s))
    frames (rs_base rs) next counts.

(* The state after a [Let] of [x] and then one of [y], both to [v]. *)
Definition bind_pair (rs : region_state) (x y : variable) (v : value)
    (body : expr) (statics : static_env) (frames : list frame)
    (next : symbol) (counts : region_counts) : region_state :=
  let s := rs_state rs in
  Region_state
    (State body ((y, v) :: (x, v) :: state_vars s) (state_conts s) statics
       (state_heap s))
    frames (rs_base rs) next counts.

(* [Begin_region] and [Begin_try_region]: a frame with a fresh mark,
   and the variable bound to the mark. The variable is of kind
   [Region]. *)
Definition begin_region (rs : region_state) (x : variable)
    (kind : flambda_kind) (mark_kind : mark_kind) (body : expr)
    : region_step_result :=
  if kind_eqb kind Region
  then
    let m := rs_next rs in
    Region_step
      (bind_let rs x (Value_symbol m) body (state_statics (rs_state rs))
         (Frame m mark_kind [] :: rs_frames rs) (Pos.succ m) (rs_counts rs))
  else Region_undefined Not_of_the_conversion.

Definition unit_value : value := Value_const (Tagged_immediate 0).

(* [End_region] and [End_try_region]: the frames down to and with that
   of the region are popped, and the variable is bound to the unit
   value. [ended_by] says which of the two it is, for the counts. *)

(* CR mshinwell: the end of a region whose mark is in no frame is
   undefined: a region ended twice, or ended after the end of a region
   begun before it. The program has no such test. It writes the
   pointer that the region read, which may free nothing, free what
   later regions hold, or move the pointer back over memory that was
   freed, by what was done since. Ending a region that is not the
   newest is defined, and ends those above it: the handler of a [try]
   does that, and so does every region around a [try] that returned.
   It would be wrong if the compiler makes, on purpose, a term that
   ends a region twice. Decision D47. *)
Definition end_region (rs : region_state) (x : variable)
    (kind : flambda_kind) (ended_by : mark_kind) (region : simple)
    (body : expr) : region_step_result :=
  if kind_eqb kind Value
  then
    match eval_simple (state_vars (rs_state rs)) region with
    | Some (Value_symbol m) =>
      match cut m (rs_frames rs) with
      | Some (above, below) =>
        let cut_away := length (filter is_plain above) in
        let counts := rs_counts rs in
        let counts' :=
          match ended_by with
          | Plain_mark =>
            Region_counts (count_other_region counts)
              (cut_away + count_cut_by_end_region counts)
              (count_cut_by_end_try_region counts) (S (count_ends counts))
              (count_local_allocations counts) (count_no_frame counts)
              (count_local_uses counts)
          | Try_mark =>
            Region_counts (count_other_region counts)
              (count_cut_by_end_region counts)
              (cut_away + count_cut_by_end_try_region counts)
              (S (count_ends counts)) (count_local_allocations counts)
              (count_no_frame counts) (count_local_uses counts)
          end
        in
        Region_step
          (bind_let rs x unit_value body (state_statics (rs_state rs)) below
             (rs_next rs) counts')
      | None => Region_undefined Region_not_open
      end
    | _ => Region_undefined Region_not_open
    end
  else Region_undefined Not_of_the_conversion.

(* The region that a [Local_allocation] names: whether it is another
   than the newest plain one, or nothing if its mark is in no frame.
   One that names no region is asked nothing. *)
Definition named_region (rs : region_state) (region : option simple)
    : option bool :=
  match region with
  | None => Some false
  | Some r =>
    match eval_simple (state_vars (rs_state rs)) r with
    | Some (Value_symbol m) =>
      if on_stack m (rs_frames rs)
      then
        match newest_plain (rs_frames rs) with
        | Some newest => Some (negb (Pos.eqb m newest))
        | None => Some true
        end
      else None
    | _ => None
    end
  end.

(* A local allocation: the [Let] of an allocation and the
   [Local_allocation] after it, in one step. The value is what [step]
   would bind; it is bound to fresh symbols ([localise]) in the newest
   frame, and the variable to the symbol. With no frame on the stack
   the symbols are bound in the static environment of the state: such
   an allocation is never freed. The region named must be on the
   stack; the allocation does not go there. *)

(* CR mshinwell: a local allocation lives as long as the newest region
   when it is made, whatever region its mode names, and the end of a
   region ends every region begun since. That is what the program
   does. The other choice is that it lives as long as the region it
   names. The two agree on a term in which every allocation names the
   newest plain region and regions end newest first, which is what the
   front end is read to make; they differ on a term in which a name is
   wrong or an end is lost, where this choice says what the program
   does and the other says that a value lives which the program has
   freed. And the front end does not end every region by its name: a
   region in tail position of another is ended by the end of the outer
   one (the header, "what runs"), so with the other choice such a
   region would never end. Decision D44. *)

(* CR mshinwell: a [Local_allocation] that names a region whose mark is
   in no frame is undefined, as the end of such a region is. The
   program does nothing with the name. It is so here because Simplify
   decides by the names whether a region is used, so such a name is a
   fault in the term even where the program runs it. Decision D47. *)
Definition local_allocation (rs : region_state) (x y : variable)
    (kind : flambda_kind) (defining_expr : named) (region : option simple)
    (body : expr) : region_step_result :=
  let s := rs_state rs in
  if is_allocation defining_expr && kind_eqb kind Value
  then
    match named_region rs region with
    | None => Region_undefined Region_not_open
    | Some other =>
      match eval_named (live_statics rs) (state_vars s) defining_expr with
      | None => Region_undefined No_plain_step
      | Some v =>
        match localise (rs_next rs) v with
        | None => Region_undefined Not_of_the_conversion
        | Some (symbol, bindings) =>
          let counts := rs_counts rs in
          let counts' :=
            Region_counts
              ((if other then 1 else 0) + count_other_region counts)
              (count_cut_by_end_region counts)
              (count_cut_by_end_try_region counts) (count_ends counts)
              (S (count_local_allocations counts))
              (match rs_frames rs with [] => 1 | _ :: _ => 0 end
               + count_no_frame counts)
              (count_local_uses counts)
          in
          let next := advance (rs_next rs) (length bindings) in
          match rs_frames rs with
          | Frame m mark_kind locals :: frames =>
            Region_step
              (bind_pair rs x y (Value_symbol symbol) body (state_statics s)
                 (Frame m mark_kind (bindings ++ locals) :: frames) next
                 counts')
          | [] =>
            Region_step
              (bind_pair rs x y (Value_symbol symbol) body
                 (Static_env (static_code (state_statics s))
                    (bindings ++ static_values (state_statics s)))
                 [] next counts')
          end
        end
      end
    end
  else Region_undefined Not_of_the_conversion.

(* The simples evaluated whose value is a local value in a frame are
   counted. *)
Definition count_uses (rs : region_state) : region_state :=
  let counts := rs_counts rs in
  let uses :=
    length
      (filter (simple_local rs) (evaluated_simples (state_expr (rs_state rs))))
  in
  with_counts rs
    (Region_counts (count_other_region counts)
       (count_cut_by_end_region counts) (count_cut_by_end_try_region counts)
       (count_ends counts) (count_local_allocations counts)
       (count_no_frame counts) (uses + count_local_uses counts)).

(* CR mshinwell: an allocation on the heap one of whose arguments is a
   local value in a frame is undefined: a block, an array or a closure
   made by a [Let] with no [Local_allocation] after it, and every
   constant that a [Let_static] binds. The program makes the block.
   It goes wrong if the argument is reached through it after its
   region is gone, and it can go wrong if a collection runs while the
   local value is reachable only through the heap block: the collector
   does not follow that edge into the local stack, so the fields of
   the local value are neither updated nor traced, and a debugging
   runtime poisons them. It is the rule
   that Simplify keeps when it lifts an allocation to a symbol. It
   would be wrong if the compiler makes, on purpose, a heap block that
   points into the local stack, is dead before the region ends, and
   is never the only thing that reaches the local value when a
   collection runs: the type checker forbids it of a source, and
   nothing checks it of a term. Decision D46. *)

(* CR mshinwell: leaving code with a region of its own still on the
   stack is not undefined, and nothing here says that it happened: by
   a return, where the function may be one that returns something
   local, which the model's code does not say; or by a raise, where
   the handler's [End_try_region] is what pops the frames. The marks
   left when the run of a unit ends are in the state of [Region_exit],
   and the plain frames that the end of another region pops are
   counted. The exception that a raise carries is not asked to be on
   the heap either: if it is local and its frame is popped by the
   handler, using it is D45. It would be wrong if a region that
   nothing ends is to be a fault of the term. Decision D48. *)

(* The test of the simples that a step evaluates: the first of them,
   in order, that may not be evaluated where it is, with the reason.
   [places] says of each whether it is at the place of a region
   ([region_places]); one beyond [places] is not.
   - A value that may be evaluated ([value_usable]) passes, wherever it
     is. At the place of a region it is not what the parameter is to be
     given, and the step that applies the continuation has none, for
     the kinds, unless the value is a poison constant of the kind
     [Region].
   - A mark that is on the stack passes at the place of a region, and
     has no step anywhere else ([Region_used_as_value]).
   - What is left is a symbol that the machine drew, that nothing
     binds and that is the mark of no frame: a local allocation whose
     frame is gone, or the mark of a region that is gone. The machine
     does not tell the two apart. At the place of a region it is taken
     to be a region (decision D47), anywhere else a value (decision
     D45). *)

(* CR mshinwell: giving a region whose mark is in no frame to a
   continuation, at a parameter that is a region, is undefined, as
   ending it is. The program passes the number that the region read,
   and nothing goes wrong there: a handler that does nothing with the
   number is harmless. It is so here so that the check says that a
   region is gone where the region is given, and does not wait for its
   end or for an allocation in it. That relies on the life that a
   region is meant to have, and refuses a path on which the handler
   ignores the region. It would be wrong if the compiler gives a
   region that has ended to a handler that, on that path, does nothing
   with it. Decision D47. *)
Fixpoint simples_fault (rs : region_state) (simples : list simple)
    (places : list bool) : option region_fault :=
  match simples with
  | [] => None
  | s :: simples =>
    let place := hd false places in
    if simple_usable rs s then simples_fault rs simples (tl places)
    else if simple_open_mark rs s
    then
      if place then simples_fault rs simples (tl places)
      else Some Region_used_as_value
    else Some (if place then Region_not_open else Dead_value_used)
  end.

(* An [Apply_cont] one of whose arguments is a mark that is on the
   stack. Where the continuation is bound to a handler this is
   FlambdaSemanticsSpec.apply_cont_value, but for the kind of a value:
   a mark that is on the stack is of the kind [Region]
   ([region_kind_of_value]), where [step] has it of the kind [Value],
   as any symbol. So the handler must have a parameter of the kind
   [Region] at the place of each mark, and the parameter is bound to
   the mark. The frames, the counter and the static environment are
   left as they are: giving a region ends nothing and begins nothing.
   Where the continuation is not bound to a handler this is
   [plain_step]; [region_step] does not come here then, as the test of
   the simples has refused the mark. *)
Definition region_apply_cont (rs : region_state) (ac : apply_cont)
    : region_step_result :=
  let s := rs_state rs in
  match resolve (ac_continuation ac) (state_conts s) with
  | Closure (Handler params metadata handler) def_vars def_conts recursive =>
    match eval_simples (state_vars s) (ac_args ac) with
    | Some vs =>
      if kinds_eqb (map (region_kind_of_value rs) vs)
           (hm_params_kinds metadata)
      then
        match bind_params params vs def_vars with
        | Some handler_vars =>
          Region_step
            (Region_state
               (State handler handler_vars
                  (rec_closures recursive def_vars def_conts ++ def_conts)
                  (state_statics s) (state_heap s))
               (rs_frames rs) (rs_base rs) (rs_next rs) (rs_counts rs))
        | None => Region_undefined No_plain_step
        end
      else Region_undefined No_plain_step
    | None => Region_undefined No_plain_step
    end
  | _ => plain_step rs
  end.

(* One step. First the test of every simple that [step] would evaluate
   ([simples_fault]). Then, by the form of the expression: a primitive
   that begins or ends a region; a [Let] with a [Local_allocation]
   after it; another [Let], with the test of D46 if it allocates; a
   [Let_static], with the test of D46; an [Apply_cont] that gives a
   region ([region_apply_cont]); and for the rest the step of [step]. *)
Definition region_step (rs : region_state) : region_step_result :=
  let e := state_expr (rs_state rs) in
  match simples_fault rs (evaluated_simples e) (region_places rs) with
  | Some fault => Region_undefined fault
  | None =>
    let rs := count_uses rs in
    match e with
    | Let x kind (Prim (Region_primitive p) args) body =>
      match p, args with
      | Begin_region, [] => begin_region rs x kind Plain_mark body
      | Begin_try_region, [] => begin_region rs x kind Try_mark body
      | End_region, [region] => end_region rs x kind Plain_mark region body
      | End_try_region, [region] => end_region rs x kind Try_mark region body
      | _, _ => Region_undefined Not_of_the_conversion
      end
    | Let x kind defining_expr body =>
      match marker_of x body with
      | Marker y region body' =>
        local_allocation rs x y kind defining_expr region body'
      | Bad_marker => Region_undefined Not_of_the_conversion
      | No_marker =>
        if is_allocation defining_expr
           && existsb (simple_local rs) (simples_of_named defining_expr)
        then Region_undefined Heap_holds_local
        else plain_step rs
      end
    | Let_static _ _ =>
      if existsb (simple_local rs) (evaluated_simples e)
      then Region_undefined Heap_holds_local
      else static_step rs
    | Apply_cont ac =>
      if existsb (simple_open_mark rs) (ac_args ac)
      then region_apply_cont rs ac
      else plain_step rs
    | _ => plain_step rs
    end
  end.

(* At most [fuel] steps. *)
Fixpoint region_run (fuel : nat) (rs : region_state) : region_step_result :=
  match fuel with
  | O => Region_step rs
  | S fuel =>
    match region_step rs with
    | Region_step rs' => region_run fuel rs'
    | result => result
    end
  end.

(* The state in which a term is run: no variable, no continuation, no
   frame, and [base] the first symbol that the machine draws. *)
Definition initial_region_state (e : expr) (statics : static_env)
    (base : symbol) : region_state :=
  Region_state (State e [] [] statics empty_heap) [] base base no_counts.

(* ======================================================================== *)
(* The symbols that the machine draws are not those of the term             *)
(* ======================================================================== *)

Definition symbol_below (base s : symbol) : bool := Pos.ltb s base.

Definition simple_below (base : symbol) (s : simple) : bool :=
  match s with
  | Symbol sym => symbol_below base sym
  | Var _ | Const _ => true
  end.

Definition static_const_below (base : symbol) (const : static_const) : bool :=
  forallb (simple_below base) (simples_of_static_const const)
  && match const with
     | Static_const_closure_of_set _ closure_symbols _ _ =>
       forallb (fun '(_, s) => symbol_below base s) closure_symbols
     | _ => true
     end.

Definition apply_cont_below (base : symbol) (ac : apply_cont) : bool :=
  forallb (simple_below base) (ac_args ac).

(* No symbol of the expression, of its handlers or of the code that it
   binds is [base] or above. *)
Fixpoint expr_below (base : symbol) (e : expr) : bool :=
  match e with
  | Let _ _ defining_expr body =>
    forallb (simple_below base) (simples_of_named defining_expr)
    && expr_below base body
  | Let_static bindings body =>
    forallb (binding_below base) bindings && expr_below base body
  | Let_cont_non_recursive _ h body =>
    handler_below base h && expr_below base body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_below base h) handlers
    && expr_below base body
  | Apply ap =>
    forallb (simple_below base)
      (ap_callee ap :: ap_args ap ++ exn_extra_simples ap)
  | Apply_cont ac => apply_cont_below base ac
  | Switch scrutinee arms =>
    simple_below base scrutinee
    && forallb (fun '(_, ac) => apply_cont_below base ac) arms
  | Invalid => true
  end
with handler_below (base : symbol) (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => expr_below base handler
  end
with binding_below (base : symbol) (binding : static_binding) : bool :=
  match binding with
  | Bind_code _ c => code_below base c
  | Bind_deleted_code _ => true
  | Bind_symbol s const => symbol_below base s && static_const_below base const
  end
with code_below (base : symbol) (c : code) : bool :=
  match c with
  | Code _ _ _ _ _ body => expr_below base body
  end.

Fixpoint value_below (base : symbol) (v : value) : bool :=
  match v with
  | Value_const _ | Value_boxed_number _ | Value_location _ => true
  | Value_closure _ value_slots | Value_closure_of_set _ _ value_slots =>
    forallb (fun '(_, v) => value_below base v) value_slots
  | Value_symbol s => symbol_below base s
  | Value_block _ fields => forallb (value_below base) fields
  | Value_immutable_array elements => forallb (value_below base) elements
  | Value_partial_application callee args _ =>
    value_below base callee && forallb (value_below base) args
  end.

Definition allocated_below (base : symbol)
    (allocated : statically_allocated_value) : bool :=
  match allocated with
  | Static_closure _ value_slots =>
    forallb (fun '(_, v) => value_below base v) value_slots
  | Static_closure_of_set _ closure_symbols _ value_slots =>
    forallb (fun '(_, s) => symbol_below base s) closure_symbols
    && forallb (fun '(_, v) => value_below base v) value_slots
  | Static_block _ fields => forallb (value_below base) fields
  | Static_boxed_number contents => value_below base contents
  | Static_string _ | Static_imported _ => true
  | Static_immutable_array elements => forallb (value_below base) elements
  end.

Definition statics_below (base : symbol) (statics : static_env) : bool :=
  forallb (fun '(_, c) => code_below base c) (static_code statics)
  && forallb
       (fun '(s, allocated) =>
          symbol_below base s && allocated_below base allocated)
       (static_values statics).

(* The test that the symbols which the machine draws from [base] on are
   none of the term or of the static environment that it is run in. *)
Definition region_names_fresh (base : symbol) (e : expr)
    (statics : static_env) : bool :=
  expr_below base e && statics_below base statics.

(* ======================================================================== *)
(* The term with its regions dropped                                        *)
(* ======================================================================== *)

(* The unit value, as a simple. *)
Definition unit_simple : simple := Const (Tagged_immediate 0).

(* What the [Let] of a primitive of regions becomes: nothing, where the
   [Let] is dropped; or a defining expression. A begin or an end that
   is not applied as the conversion applies it is left. A
   [Local_allocation] whose first argument is a variable is erased
   whatever its other arguments are. *)
Definition erase_region_primitive (x : variable) (p : region_primitive)
    (args : list simple) : option named :=
  match p, args with
  | (Begin_region | Begin_try_region), [] => None
  | End_region, [_] => Some (Simple unit_simple)
  | End_try_region, [_] =>
    Some (Prim (Unary (Opaque_identity Value)) [unit_simple])
  | Local_allocation, Var y :: _ =>
    if Pos.eqb x y then None else Some (Simple (Var y))
  | _, _ => Some (Prim (Region_primitive p) args)
  end.

(* The elements of a list that are not at the place of a region.
   [places] says of each, in order, whether it is; one beyond [places]
   is not, and is kept. *)
Fixpoint drop_regions {A : Type} (places : list bool) (l : list A) : list A :=
  match places, l with
  | [], _ => l
  | _ :: _, [] => []
  | place :: places, a :: l =>
    if place then drop_regions places l else a :: drop_regions places l
  end.

(* The continuations in scope that a [Let_cont] of the expression
   binds, each with [region_params] of its handler; the first of a name
   counts. The return and exception continuations of code, and the
   ways out of the term, are not among them. *)
Definition region_conts : Type := list (continuation * list bool).

(* An [Apply_cont] without the arguments that it gives to the region
   parameters of its continuation. *)
Definition erase_apply_cont (conts : region_conts) (ac : apply_cont)
    : apply_cont :=
  match lookup (ac_continuation ac) conts with
  | Some places =>
    Apply_cont_expr (ac_continuation ac) (drop_regions places (ac_args ac))
      (ac_trap_action ac)
  | None => ac
  end.

(* The term with its regions dropped. [conts] is as above: every
   continuation that a [Let_cont] binds is put in it, whether its
   handler has a region parameter or not, so that it hides one of the
   same name bound further out. The body of code is erased with none in
   scope. *)
Fixpoint erase (conts : region_conts) (e : expr) : expr :=
  match e with
  | Let x kind (Prim (Region_primitive p) args) body =>
    match erase_region_primitive x p args with
    | None => erase conts body
    | Some defining_expr => Let x kind defining_expr (erase conts body)
    end
  | Let x kind defining_expr body =>
    Let x kind defining_expr (erase conts body)
  | Let_static bindings body =>
    Let_static (map erase_binding bindings) (erase conts body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (erase_handler conts h)
      (erase ((k, region_params h) :: conts) body)
  | Let_cont_recursive handlers body =>
    let conts' :=
      map (fun '(k, h) => (k, region_params h)) handlers ++ conts
    in
    Let_cont_recursive
      (map (fun '(k, h) => (k, erase_handler conts' h)) handlers)
      (erase conts' body)
  | Apply _ | Invalid => e
  | Apply_cont ac => Apply_cont (erase_apply_cont conts ac)
  | Switch scrutinee arms =>
    Switch scrutinee
      (map (fun '(i, ac) => (i, erase_apply_cont conts ac)) arms)
  end
with erase_handler (conts : region_conts) (h : continuation_handler)
    : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    let places :=
      map (fun kind => kind_eqb kind Region) (hm_params_kinds metadata)
    in
    Handler (drop_regions places params)
      (Handler_metadata (drop_regions places (hm_params_kinds metadata))
         (hm_is_exn_handler metadata))
      (erase conts handler)
  end
with erase_binding (binding : static_binding) : static_binding :=
  match binding with
  | Bind_code id c => Bind_code id (erase_code c)
  | Bind_deleted_code _ | Bind_symbol _ _ => binding
  end
with erase_code (c : code) : code :=
  match c with
  | Code params my_closure return_continuation exn_continuation metadata
      body =>
    Code params my_closure return_continuation exn_continuation metadata
      (erase [] body)
  end.

(* A static environment with the regions of its code dropped. *)
Definition erase_statics (statics : static_env) : static_env :=
  Static_env (map (fun '(id, c) => (id, erase_code c)) (static_code statics))
    (static_values statics).

(* A continuation that is not one of [conts] with a region parameter. *)
Definition takes_no_region (conts : region_conts) (k : continuation) : bool :=
  match lookup k conts with
  | Some places => negb (existsb (fun place => place) places)
  | None => true
  end.

(* No call of the expression, of its handlers or of the code that it
   binds returns or raises to a continuation that has a region
   parameter. [conts] is as for [erase]. See the header: [erase] is
   meant for such a term only. *)
Fixpoint calls_return_no_region (conts : region_conts) (e : expr) : bool :=
  match e with
  | Let _ _ _ body => calls_return_no_region conts body
  | Let_static bindings body =>
    forallb binding_returns_no_region bindings
    && calls_return_no_region conts body
  | Let_cont_non_recursive k h body =>
    handler_returns_no_region conts h
    && calls_return_no_region ((k, region_params h) :: conts) body
  | Let_cont_recursive handlers body =>
    let conts' :=
      map (fun '(k, h) => (k, region_params h)) handlers ++ conts
    in
    forallb (fun '(_, h) => handler_returns_no_region conts' h) handlers
    && calls_return_no_region conts' body
  | Apply ap =>
    takes_no_region conts (ap_continuation ap)
    && takes_no_region conts (ap_exn_continuation ap)
  | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_returns_no_region (conts : region_conts)
    (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => calls_return_no_region conts handler
  end
with binding_returns_no_region (binding : static_binding) : bool :=
  match binding with
  | Bind_code _ c => code_returns_no_region c
  | Bind_deleted_code _ | Bind_symbol _ _ => true
  end
with code_returns_no_region (c : code) : bool :=
  match c with
  | Code _ _ _ _ _ body => calls_return_no_region [] body
  end.

Definition statics_return_no_region (statics : static_env) : bool :=
  forallb (fun '(_, c) => code_returns_no_region c) (static_code statics).

(* A constant that is not the poison constant of the kind [Region]: the
   one constant, and with a mark the one value, that a parameter of
   that kind can be given. *)
Definition const_no_region_poison (c : const) : bool :=
  match c with
  | Poison kind => negb (kind_eqb kind Region)
  | _ => true
  end.

Definition simple_no_region_poison (s : simple) : bool :=
  match s with
  | Const c => const_no_region_poison c
  | Var _ | Symbol _ => true
  end.

Definition apply_cont_no_region_poison (ac : apply_cont) : bool :=
  forallb simple_no_region_poison (ac_args ac).

(* No simple of the expression, of its handlers or of the code that it
   binds is the poison constant of the kind [Region]. *)
Fixpoint no_region_poison (e : expr) : bool :=
  match e with
  | Let _ _ defining_expr body =>
    forallb simple_no_region_poison (simples_of_named defining_expr)
    && no_region_poison body
  | Let_static bindings body =>
    forallb binding_no_region_poison bindings && no_region_poison body
  | Let_cont_non_recursive _ h body =>
    handler_no_region_poison h && no_region_poison body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_no_region_poison h) handlers
    && no_region_poison body
  | Apply ap =>
    forallb simple_no_region_poison
      (ap_callee ap :: ap_args ap ++ exn_extra_simples ap)
  | Apply_cont ac => apply_cont_no_region_poison ac
  | Switch scrutinee arms =>
    simple_no_region_poison scrutinee
    && forallb (fun '(_, ac) => apply_cont_no_region_poison ac) arms
  | Invalid => true
  end
with handler_no_region_poison (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => no_region_poison handler
  end
with binding_no_region_poison (binding : static_binding) : bool :=
  match binding with
  | Bind_code _ c => code_no_region_poison c
  | Bind_deleted_code _ => true
  | Bind_symbol _ const =>
    forallb simple_no_region_poison (simples_of_static_const const)
  end
with code_no_region_poison (c : code) : bool :=
  match c with
  | Code _ _ _ _ _ body => no_region_poison body
  end.

Fixpoint value_no_region_poison (v : value) : bool :=
  match v with
  | Value_const c | Value_boxed_number c => const_no_region_poison c
  | Value_location _ | Value_symbol _ => true
  | Value_closure _ value_slots | Value_closure_of_set _ _ value_slots =>
    forallb (fun '(_, v) => value_no_region_poison v) value_slots
  | Value_block _ fields => forallb value_no_region_poison fields
  | Value_immutable_array elements =>
    forallb value_no_region_poison elements
  | Value_partial_application callee args _ =>
    value_no_region_poison callee && forallb value_no_region_poison args
  end.

Definition allocated_no_region_poison
    (allocated : statically_allocated_value) : bool :=
  match allocated with
  | Static_closure _ value_slots
  | Static_closure_of_set _ _ _ value_slots =>
    forallb (fun '(_, v) => value_no_region_poison v) value_slots
  | Static_block _ fields => forallb value_no_region_poison fields
  | Static_boxed_number contents => value_no_region_poison contents
  | Static_string _ | Static_imported _ => true
  | Static_immutable_array elements =>
    forallb value_no_region_poison elements
  end.

(* No code and no value of the static environment has the poison
   constant of the kind [Region]. *)
Definition statics_no_region_poison (statics : static_env) : bool :=
  forallb (fun '(_, c) => code_no_region_poison c) (static_code statics)
  && forallb (fun '(_, allocated) => allocated_no_region_poison allocated)
       (static_values statics).

(* The terms that [erase] is meant for, with the static environment
   that they are run in: no call returns or raises to a handler that
   has a region parameter, and there is no poison constant of the kind
   [Region]. See the header. *)
Definition erasable (e : expr) (statics : static_env) : bool :=
  calls_return_no_region [] e
  && statics_return_no_region statics
  && no_region_poison e
  && statics_no_region_poison statics.

(* ======================================================================== *)
(* Statements                                                               *)
(* ======================================================================== *)

(* An expression that has nothing of regions at its head: no primitive
   of regions, and no [Local_allocation] after its [Let]. It is a test
   of the expression alone: an [Apply_cont] passes it whatever its
   arguments are, and with no frame none of them is a mark on the
   stack. *)
Definition head_plain (e : expr) : bool :=
  match e with
  | Let _ _ (Prim (Region_primitive _) _) _ => false
  | Let x _ _ body =>
    match marker_of x body with
    | No_marker => true
    | Marker _ _ _ | Bad_marker => false
    end
  | _ => true
  end.

(* The result of [step], as one of this machine: in the state [rs] with
   its counts changed to [counts]. *)
Definition of_plain_result (rs : region_state) (counts : region_counts)
    (result : step_result) : region_step_result :=
  match result with
  | Step s' =>
    Region_step
      (Region_state s' (rs_frames rs) (rs_base rs) (rs_next rs) counts)
  | Exit k vs _ => Region_exit k vs (with_counts rs counts)
  | Undefined => Region_undefined No_plain_step
  end.

(* With no frame, at an expression that has nothing of regions at its
   head, the machine does what [step] does, if every simple evaluated
   may be: nothing changes but the counts. *)
Definition region_step_plain_statement : Prop :=
  ∀ rs : region_state,
    rs_frames rs = [] ->
    head_plain (state_expr (rs_state rs)) = true ->
    forallb (simple_usable rs) (evaluated_simples (state_expr (rs_state rs)))
    = true ->
    region_step rs
    = of_plain_result rs (rs_counts (count_uses rs)) (step (rs_state rs)).

(* The primitives that end a region. *)
Definition ends_region (e : expr) : bool :=
  match e with
  | Let _ _ (Prim (Region_primitive (End_region | End_try_region)) _) _ =>
    true
  | _ => false
  end.

(* Only the end of a region pops a frame: after any other step every
   mark that was on the stack is on it. *)
Definition region_step_keeps_marks_statement : Prop :=
  ∀ (rs rs' : region_state) (m : mark),
    region_step rs = Region_step rs' ->
    ends_region (state_expr (rs_state rs)) = false ->
    on_stack m (rs_frames rs) = true ->
    on_stack m (rs_frames rs') = true.

(* And the end of a region pushes none: the stack after is what is left
   of the stack before. *)
Definition region_step_end_pops_statement : Prop :=
  ∀ rs rs' : region_state,
    region_step rs = Region_step rs' ->
    ends_region (state_expr (rs_state rs)) = true ->
    ∃ above : list frame, rs_frames rs = above ++ rs_frames rs'.

(* The machine draws symbols upwards only: a step keeps [rs_base], and
   [rs_next] does not go down. *)
Definition region_step_draws_upwards_statement : Prop :=
  ∀ rs rs' : region_state,
    region_step rs = Region_step rs' ->
    rs_base rs' = rs_base rs ∧ Pos.leb (rs_next rs) (rs_next rs') = true.

(* A symbol that the machine drew and that is gone: it is not the mark
   of a frame, and nothing binds it. *)
Definition gone (rs : region_state) (s : symbol) : Prop :=
  drawn rs s = true
  ∧ Pos.ltb s (rs_next rs) = true
  ∧ on_stack s (rs_frames rs) = false
  ∧ lookup s (static_values (live_statics rs)) = None.

(* What is gone stays gone. *)
Definition gone_stays_gone_statement : Prop :=
  ∀ (rs rs' : region_state) (s : symbol),
    region_step rs = Region_step rs' ->
    region_names_fresh (rs_base rs) (state_expr (rs_state rs))
      (state_statics (rs_state rs))
    = true ->
    gone rs s ->
    gone rs' s.

(* What [plain_step] relies on: at an expression that is not a
   [Let_static], a step of [step] leaves the static environment that
   it was given. So the static environment that [plain_step] does not
   look at is [live_statics] of the state before, and nothing that
   [step] did is lost. *)
Definition step_keeps_statics_statement : Prop :=
  ∀ s s' : state,
    step s = Step s' ->
    (∀ (bindings : list static_binding) (body : expr),
      state_expr s <> Let_static bindings body) ->
    state_statics s' = state_statics s.

(* The step of an [Apply_cont] that gives a region is that of [step]
   where it gives none: [region_apply_cont] differs from [plain_step]
   by the kind of a mark that is on the stack, and by nothing else. *)
Definition region_apply_cont_plain_statement : Prop :=
  ∀ (rs : region_state) (ac : apply_cont),
    state_expr (rs_state rs) = Apply_cont ac ->
    existsb (simple_open_mark rs) (ac_args ac) = false ->
    region_apply_cont rs ac = plain_step rs.
