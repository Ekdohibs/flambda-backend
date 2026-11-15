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

(* A second verdict of the check of regions, about ends: a check
   beside FlambdaRegionsCheckSpec.region_check, the property of states
   that a run of a term which passes it keeps, and what follows of the
   frames that the run leaves. It adds nothing to the machine, to
   [region_check] or to the certificate.

   WHAT IT IS FOR

   FlambdaRegionsCheckSpec.region_check does not say that a region is
   ended (decision D48): at a jump a handler is entered with regions
   of its own choice among those that are open, and a region that is
   open at the jump and not among them is forgotten. That is how a
   loop passes each turn of which begins a region and jumps back with
   it open.

   [ends_check] is a second test of the same term with the same
   certificate. It refuses a term in which a plain region is forgotten
   at a jump, unless the handler may be entered stale by the rules
   below, or is open where code returns. Decision D48 does not
   change: the machine has no new fault, and a term that
   [region_check] passes and [ends_check] refuses still has every
   property that is proved of [region_check].

   WHAT IT TESTS

   The check follows the regions that are open, by their variables,
   newest first, as [check_expr] does ([es_open] is its [cs_open]),
   and knows which of them are of a [Begin_try_region] ([es_tries]).

   - A JUMP to a handler: every region that is open at the jump and
     is not of a [try] is one of those that the handler is entered
     with (the regions of its entry in the certificate, with the
     regions given at its parameters put for them). None is
     forgotten. A jump to a handler that is ENTERED STALE has a
     second way to pass (below).
   - A RETURN: where the return continuation of code is applied, by a
     jump or as the continuation of a call, no region is open but
     those of a [try]. The same where the return continuation of the
     unit is applied: the check is given the two continuations of the
     unit, and checks the unit as it checks the body of code. A jump
     to a continuation that nothing binds, and that is neither, is
     refused: a unit has no third way out.
   - AN END, with the option [eo_own_end]: the regions that the
     check has open above the one that is ended, which end with it,
     are of a [try]. Without the option they may be plain. A plain
     region above it that the check has not open was forgotten at a
     jump to a handler that is entered stale (below). With the
     option only a handler that is marked as an exception handler
     may be entered stale, and an end in it pops the frames that
     were left above its regions, with the option as without. So
     the option is a test of the regions that the check follows,
     not of every end in a run.

   WHAT IS LEFT OPEN ON PURPOSE, AND WHAT IS STALE

   The end of a region pops every frame above the one that it ends
   (FlambdaRegionsSpec.end_region), and two things leave regions open
   on purpose for the end of an older region to pop. A raise: the
   handler ends the region of its [try]. And a jump out of several
   regions at once: the conversion jumps to the continuation that
   ends the outermost of them, and the inner ones get no end of
   their own (from_lambda/lambda_to_flambda.ml, the case of
   [Lregion]). So a handler may be ENTERED STALE ([enters_stale]).
   An exception handler is (the flag of its metadata). Without the
   option [eo_own_end], so is a handler whose body ends a region
   before any other primitive of regions, and before anything that
   is not a [Let] or a [Let_static] ([ends_first]). With the option
   only an exception handler is. Then:

   - a jump to a handler that is entered stale also passes if the
     regions which the handler is entered with are the OLDEST of
     those open (a suffix of the list): the ones forgotten, plain or
     of a [try], are then all above the newest that the handler
     knows. The exception continuation of a call is a handler that
     is entered stale, of which that is asked (its arguments are the
     exception and the extra arguments of the call), or the
     exception continuation of code or of the unit, of which nothing
     is asked; nor is anything asked of a jump to that continuation;
   - a handler that is entered stale is checked in a state that is
     STALE ([es_stale]), by whichever of the two tests a jump to it
     passes: frames that no open region accounts for may be on the
     stack above the newest open region, those that a raise or a
     jump left. A state that is stale may not begin a region (its
     frame would be above those that were left, and its end would
     not pop them) and may not return. It may jump to a handler only
     if the handler is entered stale, by the test of the oldest
     regions; it may jump to the exception continuation of code or
     of the unit; and the two continuations of a call from it are
     asked the same. The end of any open region, where it makes a
     step, pops the frames that were left, and the state of the
     check is not stale after it. In a handler whose body ends a
     region first there is one way to that end, through [Let]s that
     are not of regions and [Let_static]s. Nothing says that the
     run comes to it, or that the end makes a step: the statements
     are of steps that are made and of exits that are reached. An
     exception handler is stale until it successfully ends a region.
     One entered with no region cannot clear the flag by an end; it
     still has the jumps and calls just described, and may neither
     begin a region nor return. That is strict, on purpose: nothing
     in the term says that the callee which raised had no plain
     region open. The conversion makes the
     handler of a [try] end the region of the [try] first
     (from_lambda/lambda_to_flambda.ml), and Simplify takes away the
     end of a region that is not used only where it is an
     [End_region] (simplify/simplify_let_expr.ml,
     Flambda_primitive.is_end_region). Decision D55.

   A first form of the check let only an exception handler be entered
   stale. It refused the jump out of several regions that the
   conversion makes, whatever the certificate, though the run of
   such a term leaves no plain frame; that was found by running the
   check on generated units (decision D55).

   TRY MARKS

   The conversion ends the region of a [try] in the handler only, so a
   [try] whose body returns leaves its frame on the stack, and code may
   return with it there. A frame of a [try] holds whatever was
   allocated locally while it was the newest; nothing here says that
   it holds nothing. So frames of a [try] are left out everywhere: a
   region of a [try] may be forgotten at a jump and may be open at a
   return, and the statements are of the plain frames only
   ([plain_marks]).

   A parameter that is a region is not known to be of a [try],
   whatever it is given. It counts as plain in the ordinary test of a
   jump, in that of a return and in that of an end with the option:
   that is stricter than needed and sound. A jump to a handler that
   is entered stale may still forget it with the newer regions, as it
   may any plain region.

   WHAT IS STATED

   - [ends_unit_exit_statement]: the run of a term that passes
     [ends_check], from its first state, where it exits through the
     return continuation that the check was given, leaves no plain
     frame on the stack.
   - [ends_code_return_statement]: the body of code that passes, run
     on its own from a state whose frames are in the order of their
     marks, whose static environment has only code that passes and
     which binds no continuation (its two continuations then stand
     for ways out), where it exits through its return continuation,
     leaves the plain frames that the stack had: none more and none
     fewer. That the run of a term which calls the code is that run
     until the code applies one of its two continuations is a
     property of the machine (a continuation is only ever applied)
     that is not stated here.
   - [validated_and_ends_statement]: with the premises of
     FlambdaRegionsCheckSoundSpec.validated_and_checked_exits_agree_statement,
     and [ends_check] of the term after too, an exit of the term
     before through the return continuation is an exit of the term
     after through it, with no plain frame left. The validation's
     verdict is needed for the exit of the term after to exist, not
     for what it leaves.
   - [ends_own_end_statement]: with the option [eo_own_end], a step
     from a state that has the property, with a state of the check
     that is not stale, pops no plain frame but that of the region
     that it ends.

   WHAT IS NOT SAID

   - Nothing of an exit through the exception continuation: it is a
     raise, and a raise leaves its regions open.
   - Nothing of the frames of a [try], nor of what they hold.
   - Nothing of a run that does not exit: a loop that passes may still
     allocate in one region for ever.
   - Nothing that relates what [ends_check] answers of two terms: the
     check and [ends_unit_exit_statement] are of one term, and an
     answer is of the term that it was asked of.
     [validated_and_ends_statement] takes an exit of the term before
     to an exit of the term after, under the premises of the
     validation; it asks [ends_check] of the term after only, and
     its conclusion is of the frames that the term after leaves.
   - That the frame of a region is popped by its own end is said, with
     the option [eo_own_end], only of a step from a state that is not
     stale ([ends_own_end_statement]): after a raise the end of a
     region pops what the raise left. Whether a state is stale is the
     check's word, not the machine's, so nothing is said with the
     option of a whole run.
   - That a term is refused says nothing of its runs.

   THE PROPERTY OF STATES ([state_ends])

   It is a second property beside
   FlambdaRegionsCheckedSpec.state_checked, and does not use it: it
   needs of the machine only that the frames are in the order of their
   marks (FlambdaRegionsCallsSpec.frames_ordered). It has the shape of
   that property with the lifetimes taken out: a state of this check
   under which the expression passes; the open regions bound to marks
   that are on the stack; the continuations as their entries say
   ([cont_ends], the like of [cont_ok]); and, new, [frames_tight]: the
   plain frames whose marks were drawn since the running code was
   entered are the frames of its open regions, and the same of each
   caller for the frames drawn between its entry and its call.

   Two bounds are needed because [region_step_keeps_ends_statement]
   assumes the property of a state, not a run that reached it with
   these entry counters. The recorded entry of the running code,
   [floor], is no greater than [rs_next rs]. Without that bound the
   other parts can hold with the floor after the counter, and the
   mark that the next [Begin_region] draws is before the floor.
   Each saved caller was entered no earlier than [bottom], the
   entry counter of the unit or code body whose run is considered
   ([Ends_ok_return], [Ends_ok_raise]). Without that bound a return
   could resume a caller whose regions were drawn before [bottom],
   and end one. Both bounds were found by reading the property
   before anything was proved of it; a first form had neither.

   Symbols are times here too: the frames that an activation pushes
   have marks drawn after its entry ([eu_floor]), and code is given no
   region, so it ends none of its caller's. *)

From Stdlib Require Import ZArith List Bool PeanoNat.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaPrimitiveModel
  FlambdaModel FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaImportsSpec FlambdaCompareSpec FlambdaUnusedStaticsSpec
  FlambdaLoopifySpec FlambdaLiftSpec FlambdaValidateSpec
  FlambdaValidateImportsSpec FlambdaRegionsSpec FlambdaRegionsErasureSpec
  FlambdaRegionsCallsSpec FlambdaRegionsCheckSpec FlambdaRegionsCheckedSpec.
Import ListNotations.

(* ======================================================================== *)
(* The state of the check                                                   *)
(* ======================================================================== *)

(* The one option. With it, where a region is ended the regions that
   the check has open above it must be of a [try], and only an
   exception handler is entered stale. *)
Record ends_options : Type := Ends_options {
  eo_own_end : bool
}.

(* What a continuation in scope is: a handler with its entry and
   whether it is entered stale; a handler that the certificate has no
   entry for; the return continuation, of code or of the unit; the
   exception continuation, of code or of the unit. The check is
   given the two continuations of the unit, and refuses a jump to a
   continuation that is none of these: a unit has no other way out. *)
Inductive ends_cont : Type :=
  | Ends_handler (params : list variable) (kinds : list flambda_kind)
      (entry : handler_entry) (stale : bool)
  | Ends_unreachable
  | Ends_return
  | Ends_raise.

(* The regions that are open and have a name, newest first; the
   variables of the regions of a [try] in scope; every variable of a
   region in scope; whether frames that no open region accounts for
   may be above the newest open region; the continuations. *)
Record ends_state : Type := Ends_state {
  es_open : list variable;
  es_tries : list variable;
  es_regions : list variable;
  es_stale : bool;
  es_conts : list (continuation * ends_cont)
}.

(* The open regions above [r], [r] not included; all of them if [r] is
   not open. *)
Fixpoint regions_above (r : variable) (open : list variable)
    : list variable :=
  match open with
  | [] => []
  | x :: open => if Pos.eqb x r then [] else x :: regions_above r open
  end.

Fixpoint variables_eqb (xs ys : list variable) : bool :=
  match xs, ys with
  | [], [] => true
  | x :: xs, y :: ys => Pos.eqb x y && variables_eqb xs ys
  | _, _ => false
  end.

(* [xs] are the last of [ys]. *)
Definition oldest_of (xs ys : list variable) : bool :=
  variables_eqb xs (skipn (length ys - length xs) ys).

(* The regions given at the parameters that are regions, as
   FlambdaRegionsCheckSpec.region_args: each is the variable of a
   region that is open. [None] for the arguments: what a call returns,
   which is no region. *)
Fixpoint ends_region_args (open : list variable) (params : list variable)
    (kinds : list flambda_kind) (args : option (list simple))
    : option (list (variable * variable)) :=
  match params, kinds with
  | [], [] =>
    match args with
    | Some (_ :: _) => None
    | Some [] | None => Some []
    end
  | p :: params, kind :: kinds =>
    let rest_args :=
      match args with
      | Some (_ :: args) => Some (Some args)
      | Some [] => None
      | None => Some None
      end
    in
    let* rest_args := rest_args in
    let* rest := ends_region_args open params kinds rest_args in
    if kind_eqb kind Region
    then
      match args with
      | Some (Var r :: _) =>
        if among r open then Some ((p, r) :: rest) else None
      | _ => None
      end
    else Some rest
  | _, _ => None
  end.

(* ======================================================================== *)
(* A jump                                                                   *)
(* ======================================================================== *)

(* The regions that a handler is entered with, in the names of the
   jump. *)
Definition entry_at_jump (open : list variable) (params : list variable)
    (kinds : list flambda_kind) (entry : handler_entry)
    (args : option (list simple)) : option (list variable) :=
  let* given := ends_region_args open params kinds args in
  Some (map (region_at_jump given) (he_open entry)).

(* A jump to [k], with the arguments [args], or with what a call
   returns. To a handler it passes in one of two ways: the handler
   is entered stale, with the oldest of the open regions; or the
   state is not stale, and the handler is entered with regions that
   are open, in their order, every plain one among them. *)
Definition ends_jump (es : ends_state) (k : continuation)
    (args : option (list simple)) : bool :=
  match lookup k (es_conts es) with
  | None | Some Ends_unreachable => false
  | Some Ends_return =>
    negb (es_stale es)
    && forallb (fun r => among r (es_tries es)) (es_open es)
  | Some Ends_raise => true
  | Some (Ends_handler params kinds entry stale) =>
    match entry_at_jump (es_open es) params kinds entry args with
    | None => false
    | Some named =>
      (stale && oldest_of named (es_open es))
      || (negb (es_stale es)
          && in_order named (es_open es)
          && forallb (fun r => among r (es_tries es) || among r named)
               (es_open es))
    end
  end.

(* The exception continuation of a call, with the extra arguments of
   the call: the exception continuation of the code or of the unit, or
   a handler that is entered stale, with the oldest of the open
   regions. The handler is given the exception, which is no region,
   and then the extra arguments, as
   FlambdaRegionsCheckSpec.check_expr gives them to [jump]. *)
Definition ends_raise_to (es : ends_state) (k : continuation)
    (extra : list simple) : bool :=
  match lookup k (es_conts es) with
  | Some Ends_raise => true
  | None | Some Ends_unreachable | Some Ends_return => false
  | Some (Ends_handler params kinds entry stale) =>
    stale
    && match
         entry_at_jump (es_open es) params kinds entry
           (Some (unit_simple :: extra))
       with
       | None => false
       | Some named => oldest_of named (es_open es)
       end
  end.

(* ======================================================================== *)
(* The check of an expression                                               *)
(* ======================================================================== *)

(* An expression that ends a region before any other primitive of
   regions, and before anything that is not a [Let] or a
   [Let_static]. *)
Fixpoint ends_first (e : expr) : bool :=
  match e with
  | Let _ _ (Prim (Region_primitive p) args) _ =>
    match p, args with
    | (End_region | End_try_region), [Var _] => true
    | _, _ => false
    end
  | Let _ _ _ body => ends_first body
  | Let_static _ body => ends_first body
  | _ => false
  end.

(* Whether a handler is entered stale: an exception handler is, and,
   without the option, so is a handler whose body ends a region
   first. *)
Definition enters_stale (opts : ends_options) (h : continuation_handler)
    : bool :=
  match h with
  | Handler _ metadata handler =>
    hm_is_exn_handler metadata
    || (negb (eo_own_end opts) && ends_first handler)
  end.

Definition ends_cont_of (cert : certificate) (opts : ends_options)
    (k : continuation) (h : continuation_handler) : ends_cont :=
  match h, lookup k (cert_handlers cert) with
  | Handler params metadata _, Some entry =>
    Ends_handler params (hm_params_kinds metadata) entry
      (enters_stale opts h)
  | _, None => Ends_unreachable
  end.

Definition ends_with_conts (es : ends_state)
    (conts : list (continuation * ends_cont)) : ends_state :=
  Ends_state (es_open es) (es_tries es) (es_regions es) (es_stale es)
    (conts ++ es_conts es).

(* The state in which a handler is checked: the regions of its entry;
   its parameters that are regions are variables of regions, and are
   not of a [try]; it is stale if the handler is entered stale. *)
Definition ends_handler_state (es : ends_state) (params : list variable)
    (kinds : list flambda_kind) (entry : handler_entry) (stale : bool)
    : ends_state :=
  Ends_state (he_open entry) (es_tries es)
    (region_params_of params kinds ++ es_regions es) stale (es_conts es).

(* The state in which the body of code is checked: no region is open;
   its two continuations. *)
Definition ends_code_state (return_continuation exn_continuation
      : continuation) : ends_state :=
  Ends_state [] [] [] false
    [(return_continuation, Ends_return); (exn_continuation, Ends_raise)].

Section Ends.

Variable cert : certificate.

Variable opts : ends_options.

Fixpoint ends_expr (es : ends_state) (e : expr) {struct e} : bool :=
  match e with
  | Let x _ (Prim (Region_primitive p) args) body =>
    negb (among x (es_regions es))
    && match p, args with
       | Begin_region, [] =>
         negb (es_stale es)
         && ends_expr
              (Ends_state (x :: es_open es) (es_tries es)
                 (x :: es_regions es) false (es_conts es))
              body
       | Begin_try_region, [] =>
         negb (es_stale es)
         && ends_expr
              (Ends_state (x :: es_open es) (x :: es_tries es)
                 (x :: es_regions es) false (es_conts es))
              body
       | (End_region | End_try_region), [Var r] =>
         among r (es_open es)
         && (negb (eo_own_end opts)
             || forallb (fun r' => among r' (es_tries es))
                  (regions_above r (es_open es)))
         && ends_expr
              (Ends_state (regions_below r (es_open es)) (es_tries es)
                 (es_regions es) false (es_conts es))
              body
       | Local_allocation, _ => ends_expr es body
       | _, _ => false
       end
  | Let x _ _ body => negb (among x (es_regions es)) && ends_expr es body
  | Let_static bindings body =>
    forallb ends_binding bindings && ends_expr es body
  | Let_cont_non_recursive k h body =>
    ends_handler es k h
    && ends_expr (ends_with_conts es [(k, ends_cont_of cert opts k h)]) body
  | Let_cont_recursive handlers body =>
    let es' :=
      ends_with_conts es
        (map (fun '(k, h) => (k, ends_cont_of cert opts k h)) handlers)
    in
    forallb (fun '(k, h) => ends_handler es' k h) handlers
    && ends_expr es' body
  | Apply ap =>
    ends_jump es (ap_continuation ap) None
    && ends_raise_to es (ap_exn_continuation ap) (exn_extra_simples ap)
  | Apply_cont ac => ends_jump es (ac_continuation ac) (Some (ac_args ac))
  | Switch _ arms =>
    forallb
      (fun '(_, ac) => ends_jump es (ac_continuation ac) (Some (ac_args ac)))
      arms
  | Invalid => true
  end
(* A handler with no entry is not checked: nothing may jump to it. The
   regions that it is entered with are parameters or were open where it
   is bound, each once, and its parameters are new: tests of
   FlambdaRegionsCheckSpec.entry_fits, asked again so that this check
   stands on its own. *)
with ends_handler (es : ends_state) (k : continuation)
    (h : continuation_handler) {struct h} : bool :=
  match h with
  | Handler params metadata handler =>
    match lookup k (cert_handlers cert) with
    | None => true
    | Some entry =>
      let kinds := hm_params_kinds metadata in
      distinct params
      && forallb (fun p => negb (among p (es_regions es))) params
      && distinct (he_open entry)
      && forallb
           (fun r =>
              among r (region_params_of params kinds) || among r (es_open es))
           (he_open entry)
      && ends_expr
           (ends_handler_state es params kinds entry (enters_stale opts h))
           handler
    end
  end
with ends_binding (binding : static_binding) {struct binding} : bool :=
  match binding with
  | Bind_code _ c => ends_code c
  | Bind_deleted_code _ | Bind_symbol _ _ => true
  end
with ends_code (c : code) {struct c} : bool :=
  match c with
  | Code _ _ return_continuation exn_continuation _ body =>
    negb (Pos.eqb return_continuation exn_continuation)
    && ends_expr (ends_code_state return_continuation exn_continuation) body
  end.

End Ends.

(* ======================================================================== *)
(* The check of a term                                                      *)
(* ======================================================================== *)

(* The check: every code of the static environment passes; the term
   does, as the body of code does, with [ret] its return continuation
   and [exn] its exception continuation. *)

(* CR mshinwell: what this verdict refuses is a choice, in three
   parts. (1) A state that is stale, which is that of a handler that
   is entered stale until it ends a region, may not begin a region,
   jump to a handler that is not entered stale, or return: so an
   exception handler that begins and ends a region of its own before
   it ends the region of its [try] is refused, though its run leaves
   nothing, and one that is entered with no region stays stale. The
   exact rule would record where among the open regions the frames
   that were left are. (2) Regions of a [try] are left out: one may be
   forgotten at a jump and be open where code returns, because the
   conversion ends it in the handler only; so the verdict says
   nothing of the frames of a [try]. A parameter that is a region is
   taken to be plain, whatever it is given. (3) With the option
   [eo_own_end] the regions that the check has open above a region
   that is ended must be of a [try], and only an exception handler is
   entered stale; without it the end of a region may end plain
   regions above it, those that the check has open and those that a
   jump to a handler which ends a region first left. So the option
   refuses what the conversion makes where control leaves several
   regions at once, and what Simplify makes where it splits an
   over-application. It would be wrong if the compiler makes, on
   purpose, a handler of an exception that keeps working with what
   the raise left on the stack. Decision D55. *)
Definition ends_check (cert : certificate) (opts : ends_options)
    (ret exn : continuation) (statics : static_env) (e : expr) : bool :=
  forallb (fun '(_, c) => ends_code cert opts c) (static_code statics)
  && negb (Pos.eqb ret exn)
  && ends_expr cert opts (ends_code_state ret exn) e.

(* The two verdicts together. *)
Definition region_check_ends (cert : certificate) (opts : ends_options)
    (ret exn : continuation) (statics : static_env) (e : expr) : bool :=
  region_check cert statics e && ends_check cert opts ret exn statics e.

(* ======================================================================== *)
(* The plain frames                                                         *)
(* ======================================================================== *)

(* The marks of the frames that are not of a [try], newest first. *)
Definition plain_marks (frames : list frame) : list mark :=
  map frame_mark (filter is_plain frames).

(* The kind of the frame of a mark that is on the stack. *)
Definition kind_on_stack (m : mark) (frames : list frame)
    : option mark_kind :=
  option_map frame_kind (find (fun f => Pos.eqb (frame_mark f) m) frames).

(* The plain frames whose marks were drawn from [floor] on, and before
   [stop] if there is one, are frames of the marks [ms]. *)
Definition frames_tight (frames : list frame) (floor : symbol)
    (stop : option symbol) (ms : list mark) : Prop :=
  ∀ m : mark,
    In m (plain_marks frames) ->
    Pos.leb floor m = true ->
    before stop m = true ->
    In m ms.

(* Where a state is stale, nothing is asked of the frames above its
   newest open region: of none, if no region is open. *)
Definition stale_stop (stale : bool) (floor : symbol) (ms : list mark)
    : option symbol :=
  if stale then Some (hd floor ms) else None.

(* ======================================================================== *)
(* The state that runs                                                      *)
(* ======================================================================== *)

(* The state of the check describes the open regions of a state of the
   machine whose code was entered when the counter was [floor]: they
   are bound to the marks [ms], which are on the stack, decrease and
   were drawn since; a region of a [try] has a frame of a [try]; its
   variables of regions are in scope; and the plain frames drawn since
   are theirs, but for those above the newest if the state is
   stale. *)
Definition ends_placed (vars : env) (frames : list frame) (es : ends_state)
    (ms : list mark) (floor : symbol) : Prop :=
  open_marks vars (es_open es) = Some ms
  ∧ marks_placed frames ms = true
  ∧ forallb (fun m => Pos.leb floor m) ms = true
  ∧ all_among (es_open es) (es_regions es) = true
  ∧ all_among (es_tries es) (es_regions es) = true
  ∧ (∀ (r : variable) (m : mark),
       In (r, m) (combine (es_open es) ms) ->
       among r (es_tries es) = true ->
       kind_on_stack m frames = Some Try_mark).

Definition ends_running (rs : region_state) (es : ends_state)
    (ms : list mark) (floor : symbol) : Prop :=
  ends_placed (state_vars (rs_state rs)) (rs_frames rs) es ms floor
  ∧ frames_tight (rs_frames rs) floor (stale_stop (es_stale es) floor ms) ms.

(* ======================================================================== *)
(* Who may apply a continuation, and who called                             *)
(* ======================================================================== *)

(* What a continuation is as its entry says to: the variables, the
   variables of regions and those of the regions of a [try] of the
   state that may apply it, and the counter when its code was
   entered. *)
Record ends_user : Type := Ends_user {
  eu_vars : env;
  eu_regions : list variable;
  eu_tries : list variable;
  eu_floor : symbol
}.

(* A caller, as it was at the call: the state of the check, the
   variables, the marks of its open regions, and the counter when its
   own code was entered. *)
Record ends_caller : Type := Ends_caller {
  ec_state : ends_state;
  ec_vars : env;
  ec_marks : list mark;
  ec_floor : symbol
}.

Definition ends_caller_user (a : ends_caller) : ends_user :=
  Ends_user (ec_vars a) (es_regions (ec_state a)) (es_tries (ec_state a))
    (ec_floor a).

(* The caller of the code that the user runs: its open regions are
   still as they were, all drawn before the code of the user was
   entered, and the plain frames drawn between the entry of the caller
   and that of the user are theirs. *)
Definition ends_called_from (rs : region_state) (u : ends_user)
    (a : ends_caller) : Prop :=
  ends_placed (ec_vars a) (rs_frames rs) (ec_state a) (ec_marks a)
    (ec_floor a)
  ∧ forallb (fun m => Pos.ltb m (eu_floor u)) (ec_marks a) = true
  ∧ Pos.leb (ec_floor a) (eu_floor u) = true
  ∧ frames_tight (rs_frames rs) (ec_floor a)
      (Some
         (match stale_stop (es_stale (ec_state a)) (ec_floor a) (ec_marks a)
          with
          | Some stop => Pos.min stop (eu_floor u)
          | None => eu_floor u
          end))
      (ec_marks a).

(* ======================================================================== *)
(* The continuations                                                        *)
(* ======================================================================== *)

Section Ends_checked.

Variable cert : certificate.

Variable opts : ends_options.

(* The return continuation of what is run: of the unit, or of the code
   whose body is run on its own. It is the one continuation that
   stands for a way out and for a return. *)
Variable ret : continuation.

(* The counter when what is run was entered: every frame drawn before
   it is another's. *)
Variable bottom : symbol.

(* A closure of handlers was made in the state [es] of the check;
   [base] are the entries of the continuations that it holds and
   [recursive] its recursive group. The variables of regions that
   were in scope where it was made are bound, for the user, to what
   they were bound to, and those of the regions of a [try] are still
   known to be. *)
Definition ends_made_at (u : ends_user) (vars : env)
    (recursive : list (continuation * continuation_handler))
    (es : ends_state) (base : list (continuation * ends_cont)) : Prop :=
  es_conts es
  = map (fun '(k, h) => (k, ends_cont_of cert opts k h)) recursive ++ base
  ∧ forallb (fun '(k, h) => ends_handler cert opts es k h) recursive = true
  ∧ all_among (es_open es) (es_regions es) = true
  ∧ all_among (es_tries es) (es_regions es) = true
  ∧ (∀ r : variable,
       among r (es_regions es) = true ->
       lookup r (eu_vars u) = lookup r vars ∧ among r (eu_regions u) = true)
  ∧ (∀ r : variable,
       among r (es_tries es) = true -> among r (eu_tries u) = true).

(* A continuation is as its entry says, to a user ([cont_ends]); what
   a caller gave as the return continuation ([ret_ends]) and as the
   exception continuation ([raise_ends]) is a continuation of the
   caller that the check let the call return or raise to. The shape is
   that of FlambdaRegionsCheckedSpec.cont_ok. *)
Inductive cont_ends (rs : region_state)
    : ends_user -> ends_cont -> cont_value -> Prop :=
  | Ends_ok_handler (u : ends_user) (params : list variable)
      (metadata : handler_metadata) (handler : expr) (entry : handler_entry)
      (vars : env) (conts : cont_env)
      (recursive : list (continuation * continuation_handler))
      (es : ends_state) (base : list (continuation * ends_cont)) :
    ends_made_at u vars recursive es base ->
    ends_expr cert opts
      (ends_handler_state es params (hm_params_kinds metadata) entry
         (enters_stale opts (Handler params metadata handler)))
      handler
    = true ->
    distinct params = true ->
    forallb (fun p => negb (among p (es_regions es))) params = true ->
    distinct (he_open entry) = true ->
    forallb
      (fun r =>
         among r (region_params_of params (hm_params_kinds metadata))
         || among r (es_open es))
      (he_open entry)
    = true ->
    (∀ (k : continuation) (ec : ends_cont),
       lookup k base = Some ec ->
       cont_ends rs
         (Ends_user vars (es_regions es) (es_tries es) (eu_floor u)) ec
         (resolve k conts)) ->
    (∀ k : continuation, lookup k base = None -> lookup k conts = None) ->
    cont_ends rs u
      (Ends_handler params (hm_params_kinds metadata) entry
         (enters_stale opts (Handler params metadata handler)))
      (Closure (Handler params metadata handler) vars conts recursive)
  | Ends_ok_unreachable (u : ends_user) (c : cont_value) :
    cont_ends rs u Ends_unreachable c
  (* The return continuation of the unit, or of code that is run on
     its own: the way out [ret], for a user that was entered at
     [bottom]. *)
  | Ends_ok_exit (u : ends_user) :
    eu_floor u = bottom ->
    cont_ends rs u Ends_return (Outer ret)
  | Ends_ok_return (u : ends_user) (a : ends_caller) (c : cont_value) :
    Pos.leb bottom (ec_floor a) = true ->
    ends_called_from rs u a ->
    ret_ends rs a c ->
    cont_ends rs u Ends_return c
  (* The exception continuation of the unit, or of code that is run on
     its own: another way out than [ret]. *)
  | Ends_ok_raise_out (u : ends_user) (k : continuation)
      (extra : list value) :
    Pos.eqb k ret = false ->
    cont_ends rs u Ends_raise (with_extra_args (Outer k) extra)
  | Ends_ok_raise (u : ends_user) (a : ends_caller) (c : cont_value) :
    Pos.leb bottom (ec_floor a) = true ->
    ends_called_from rs u a ->
    raise_ends rs a c ->
    cont_ends rs u Ends_raise c
with ret_ends (rs : region_state) : ends_caller -> cont_value -> Prop :=
  | Ends_ret_to (a : ends_caller) (k : continuation) (c : cont_value) :
    ends_jump (ec_state a) k None = true ->
    (∀ ec : ends_cont,
       lookup k (es_conts (ec_state a)) = Some ec ->
       cont_ends rs (ends_caller_user a) ec c) ->
    ret_ends rs a c
  | Ends_ret_over (a : ends_caller) (remaining : list value)
      (remaining_arity return_arity : arity)
      (return_to exn_to : cont_value) :
    ret_ends rs a return_to ->
    raise_ends rs a exn_to ->
    ret_ends rs a
      (Over_application remaining remaining_arity return_arity return_to
         exn_to)
with raise_ends (rs : region_state) : ends_caller -> cont_value -> Prop :=
  (* A continuation of the caller, with the values that the extra
     arguments of the call had in the caller, as
     FlambdaRegionsCheckedSpec.raise_ok has them. *)
  | Ends_raise_to (a : ends_caller) (k : continuation)
      (simples : list simple) (extra : list value) (c : cont_value) :
    eval_simples (ec_vars a) simples = Some extra ->
    ends_raise_to (ec_state a) k simples = true ->
    (∀ ec : ends_cont,
       lookup k (es_conts (ec_state a)) = Some ec ->
       cont_ends rs (ends_caller_user a) ec c) ->
    raise_ends rs a (with_extra_args c extra).

(* The continuations of a state are as the entries say, and a
   continuation with no entry is bound to nothing. *)
Definition conts_ends (rs : region_state) (u : ends_user)
    (entries : list (continuation * ends_cont)) (conts : cont_env) : Prop :=
  (∀ (k : continuation) (ec : ends_cont),
     lookup k entries = Some ec -> cont_ends rs u ec (resolve k conts))
  ∧ (∀ k : continuation, lookup k entries = None -> lookup k conts = None).

(* ======================================================================== *)
(* The property                                                             *)
(* ======================================================================== *)

(* Every code of the static environment of the state passes. *)
Definition code_ends (rs : region_state) : Prop :=
  ∀ (id : code_id) (c : code),
    lookup id (static_code (state_statics (rs_state rs))) = Some c ->
    ends_code cert opts c = true.

(* The property, of a state all of whose frames drawn before [bottom]
   are another's: those of the run that it was entered from, for the
   body of code run on its own; none, for a term run from its first
   state. [state_ends_at] names the state of the check under which
   the expression passes. The running code was entered when the
   counter was [floor]: not before what is run was entered, and not
   after now, so that every mark still to be drawn is drawn from
   [floor] on; and a caller was entered no earlier than what is run
   was ([Ends_ok_return], [Ends_ok_raise]). *)
Definition state_ends_at (rs : region_state) (es : ends_state) : Prop :=
  frames_ordered rs = true
  ∧ code_ends rs
  ∧ ∃ (ms : list mark) (floor : symbol),
      ends_expr cert opts es (state_expr (rs_state rs)) = true
      ∧ Pos.leb bottom floor = true
      ∧ Pos.leb floor (rs_next rs) = true
      ∧ ends_running rs es ms floor
      ∧ conts_ends rs
          (Ends_user (state_vars (rs_state rs)) (es_regions es)
             (es_tries es) floor)
          (es_conts es) (state_conts (rs_state rs)).

Definition state_ends (rs : region_state) : Prop :=
  ∃ es : ends_state, state_ends_at rs es.

End Ends_checked.

(* ======================================================================== *)
(* Statements                                                               *)
(* ======================================================================== *)

(* The marks of the frames that were drawn before [bottom], with their
   kinds. *)
Definition frames_before (bottom : symbol) (frames : list frame)
    : list (mark * mark_kind) :=
  map (fun f => (frame_mark f, frame_kind f))
    (filter (fun f => Pos.ltb (frame_mark f) bottom) frames).

(* The state that a run of a term that passes starts in has the
   property, with nothing below it. *)
Definition ends_initial_statement : Prop :=
  ∀ (cert : certificate) (opts : ends_options) (ret exn : continuation)
    (statics : static_env) (base : symbol) (e : expr),
    ends_check cert opts ret exn statics e = true ->
    state_ends cert opts ret base (initial_region_state e statics base).

(* So has the body of code that passes, in any state whose frames are
   in the order of their marks and whose static environment has only
   code that passes, with its two continuations standing for ways
   out: everything that is on the stack then is below it. *)
Definition ends_code_entry_statement : Prop :=
  ∀ (cert : certificate) (opts : ends_options) (c : code)
    (rs : region_state),
    ends_code cert opts c = true ->
    frames_ordered rs = true ->
    code_ends cert opts rs ->
    state_expr (rs_state rs) = code_body c ->
    state_conts (rs_state rs) = [] ->
    state_ends cert opts (code_return_continuation c) (rs_next rs) rs.

(* A step keeps it, and keeps the frames below. *)
Definition region_step_keeps_ends_statement : Prop :=
  ∀ (cert : certificate) (opts : ends_options) (ret : continuation)
    (bottom : symbol) (rs rs' : region_state),
    state_ends cert opts ret bottom rs ->
    region_step rs = Region_step rs' ->
    state_ends cert opts ret bottom rs'
    ∧ frames_before bottom (rs_frames rs')
      = frames_before bottom (rs_frames rs).

(* A state that has it and exits through [ret] leaves no plain frame
   drawn from [bottom] on, and the frames below as they were. *)
Definition ends_exit_statement : Prop :=
  ∀ (cert : certificate) (opts : ends_options) (ret : continuation)
    (bottom : symbol) (rs rs' : region_state) (vs : list value),
    state_ends cert opts ret bottom rs ->
    region_step rs = Region_exit ret vs rs' ->
    frames_before bottom (rs_frames rs') = frames_before bottom (rs_frames rs)
    ∧ ∀ m : mark,
        In m (plain_marks (rs_frames rs')) -> Pos.ltb m bottom = true.

(* With the option, a step from a state that is not stale pops no
   plain frame but that of the region that it ends: the counts of the
   plain frames popped by the end of another region stay as they
   were. *)
Definition ends_own_end_statement : Prop :=
  ∀ (cert : certificate) (opts : ends_options) (ret : continuation)
    (bottom : symbol) (rs rs' : region_state) (es : ends_state),
    eo_own_end opts = true ->
    state_ends_at cert opts ret bottom rs es ->
    es_stale es = false ->
    region_step rs = Region_step rs' ->
    count_cut_by_end_region (rs_counts rs')
    = count_cut_by_end_region (rs_counts rs)
    ∧ count_cut_by_end_try_region (rs_counts rs')
      = count_cut_by_end_try_region (rs_counts rs).

(* The run of a term that passes, from its first state, where it exits
   through its return continuation, leaves no plain frame. Nothing is
   asked of the symbols that the machine draws: the frames of a run
   from the first state are in the order of their marks whatever the
   base. *)
Definition ends_unit_exit_statement : Prop :=
  ∀ (cert : certificate) (opts : ends_options) (ret exn : continuation)
    (statics : static_env) (base : symbol) (e : expr) (fuel : nat)
    (vs : list value) (rs : region_state),
    ends_check cert opts ret exn statics e = true ->
    region_run fuel (initial_region_state e statics base)
    = Region_exit ret vs rs ->
    plain_marks (rs_frames rs) = [].

(* The body of code that passes, run on its own from a state whose
   frames are in the order of their marks and whose static environment
   has only code that passes, its two continuations standing for ways
   out: where it exits through its return continuation, the plain
   frames are those that the stack had, none more and none fewer. *)
Definition ends_code_return_statement : Prop :=
  ∀ (cert : certificate) (opts : ends_options) (c : code)
    (rs rs' : region_state) (fuel : nat) (vs : list value),
    ends_code cert opts c = true ->
    frames_ordered rs = true ->
    code_ends cert opts rs ->
    state_expr (rs_state rs) = code_body c ->
    state_conts (rs_state rs) = [] ->
    region_run fuel rs = Region_exit (code_return_continuation c) vs rs' ->
    plain_marks (rs_frames rs') = plain_marks (rs_frames rs).

(* With the validation's verdict: the premises of
   FlambdaRegionsCheckSoundSpec.validated_and_checked_exits_agree_statement,
   and [ends_check] of the term after with [ret] and [exn] its two
   continuations. An exit of the term before, with its regions,
   through [ret] is an exit of the term after through it that leaves
   no plain frame. The agreement of the values is that statement's
   and is not said again. *)
Definition validated_and_ends_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (rounds : nat)
    (unused : dropped) (slots : list value_slot) (G : loopified)
    (Ls : list lifting) (kept : list code_id) (cert : certificate)
    (opts : ends_options) (ret exn : continuation) (base : symbol)
    (before after validated_before validated_after : expr),
    let statics := imports_statics imports in
    verdict_holds_from statics rounds unused slots G Ls kept
      validated_before validated_after
      (validate_unit rounds unused slots G Ls kept validated_before
         validated_after)
    = true ->
    expr_eqb [] [] (erase [] before) validated_before = true ->
    expr_eqb [] [] (erase [] after) validated_after = true ->
    ~ behaves (erased_initial_state before statics) Goes_wrong ->
    region_check_ends cert opts ret exn statics after = true ->
    region_names_fresh base before statics = true ->
    region_names_fresh base after statics = true ->
    erasable before statics = true ->
    erasable after statics = true ->
    ∀ (fuel : nat) (vs : list value) (rs : region_state),
      region_run fuel (initial_region_state before statics base)
      = Region_exit ret vs rs ->
      ∃ (fuel' : nat) (ws : list value) (rs' : region_state),
        region_run fuel' (initial_region_state after statics base)
        = Region_exit ret ws rs'
        ∧ plain_marks (rs_frames rs') = [].
