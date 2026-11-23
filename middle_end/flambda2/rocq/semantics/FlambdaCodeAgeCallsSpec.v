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

(* A direct call may name a newer version of the code that it names: the
   statement, on a machine that is given a code age relation.

   This file is to be reviewed: [step_aged], the machine, with
   [shape_of_aged_direct_call], against FlambdaSemanticsSpec.step and
   shape_of_closure_call and against what the compiler says of a direct
   call (below); [closure_at_least], [code_refines_for], [sound_for]
   and [versions_bound], which the statements suppose; and the
   statements. The relation, its order and the compiler's functions on
   it are FlambdaCodeAgeRelation.v and FlambdaCodeAgeRelationSpec.v.

   THIS IS A SECOND MACHINE. The machine of the tree is
   FlambdaSemanticsSpec.step, and every other statement of the tree is
   of [step]: the rewrites, the validation and what the tests run.
   This file changes nothing of the tree, and it does not say of any
   statement of the tree that it holds on [step_aged]. Until the tree is
   moved to this machine, the statements here compose with no statement
   of the validation: they say what holds of a direct call on this
   machine, and no more.

   Why a second machine. On [step] a direct call is undefined unless
   its callee is a closure of the code that the call names
   (FlambdaSemanticsSpec.shape_of_closure_call; discrepancy C0-1). So
   on [step] a direct call and the same call naming a newer version of
   the code are never both defined of one closure, and "the second
   refines the first" would hold there for that reason alone. To say it
   with content, the direct call of a closure of a newer version of the
   code named must mean something. It is not given a meaning in [step]:
   the statements of the tree are of every static environment and of
   every term, and with such a meaning some of them do not hold as they
   stand (the last part of this comment).

   What the machine is given. [age_of] gives each static environment a
   code age relation. It is a parameter of the machine and of every
   statement, which hold whatever it is. The terms of the model do not
   say of which code their code is a newer version (the compiler's
   [Code_metadata.newer_version_of] has no counterpart in
   FlambdaModel.code_metadata), so no relation is read off a term or
   off a static environment, and the conversion of the test tool makes
   none.

   What the machine does. [step_aged age_of] is [step] but at a direct
   call. There:
   - The callee must be a closure of the code named or of a newer
     version of it, in the relation of the static environment
     (FlambdaCodeAgeRelationSpec.same_or_newer_b), and the code NAMED
     must be bound. It is the code named that the call goes by,
     whatever the code of the closure: the compiler says of a direct
     call that "The [code_id] uniquely determines the function symbol
     that must be called" (terms/call_kind.mli).
   - Only that test and the choice of the code differ from [step]. The
     callee, the arguments and the extra arguments of the exception
     continuation are evaluated and the continuations are resolved as
     [step] does, and the call is the generic application that a direct
     call is on [step] (FlambdaSemanticsSpec.shape_of_code_call;
     discrepancy C3-9), told by the code named. A full call runs the
     body of the code named with the callee as [my_closure]
     ([call_code], with its tests of the kinds, of the arities and of
     what the parameters state). An over-application is that full call
     with the first arguments, whose result is then called with the
     rest. A partial application runs no code: it is the value that
     keeps the callee and the arguments, whichever code the call names,
     and a later call of that value is a call of unknown arity, which
     goes by the callee's own code. Tupled code is given the fields of
     the tuple ([arguments_for_code]).
   - Only the code named is looked up: the callee's own code need not
     be bound. On [step] the two are one code.
   - A direct call of a closure of any other code is undefined, as it
     is on [step].
   A state that [step] gives a step has the same step on the machine,
   whatever the relation ([step_aged_extends_statement]), and with the
   empty relation the machine is [step] ([step_aged_empty_statement]);
   so a state that does not go wrong on [step] behaves in the same way
   on the machine ([behaves_aged_extends_statement]). These are of one
   state. They do not carry a statement of the tree to the machine: a
   refinement of the tree may rest on its source being undefined at a
   direct call that has a meaning here.
   [direct_call_older_code_example_statement] has such a call: [step]
   gives it no step, and the machine runs it.

   What the statements suppose: a relation that is sound for a closure.
   - [code_refines_for age_of statics v c c']: a full call of the code
     [c'] with the closure [v] refines the full call of [c] with it,
     whatever the arguments, the arities of the call, the
     continuations, the heap and the read record; and the two codes
     have the same arity of parameters, the same arity of results and
     the same [is_tupled]. The three equalities are restrictions, and
     not consequences of the order: they are what gives a call one
     shape whichever of the two codes it names.
   - [sound_for age_of statics v]: for each code that [v] is a closure
     of, or of a newer version of, that code refines its older version
     for [v], where both are bound. It is refinement and not
     equivalence: the newer code may be defined where the older is not.
     It is a hypothesis, and not a consequence of having a relation or
     of the relation being well-formed.
   - It is of ONE closure, and not of every closure of the code.
     Simplify makes a newer version of code where a set of closures is
     made, with what it knows there of the value slots
     (FlambdaCopyCodeSpec.v, its head): the newer code is right for the
     closures that hold those values. The model has other closures of
     the same code, since a set of closures is made with any values in
     its slots and nothing checks them against its code. For such a
     closure the newer code need not refine the older. So the
     statement supposes soundness for the closure called. That a unit
     makes only closures for which its relation is sound would be an
     invariant of the unit; nothing here shows one, and nothing here
     says that a relation which the compiler makes is sound.

   The statement ([direct_call_newer_version_refines_statement]). A
   direct call names [named]; its callee is a closure of [newer] or of
   a newer version of it; [newer] is [named] or a newer version of it;
   the relation is sound for the callee; and every code from [newer]
   down to [named] is bound. Then the call that names [newer] refines
   it.
   - The condition on the callee cannot go: a newer version of the
     code named that the callee is not a closure of is no replacement.
     [direct_call_other_version_statement] is a unit with two newer
     versions of one code, in which every other hypothesis holds and
     the call that names the sibling version does not refine.
   - [versions_bound] asks for the code of [newer], of [named] and of
     every version between them. It is what this proof needs: the
     hypothesis is of links between bound code, and the proof goes down
     the versions link by link. It is not said to be necessary. The
     machine looks up the code named and no other, so the two calls may
     both be defined, and the second may refine the first, where a
     version between them is not bound; the statement does not say so.
   - An exit is compared as [refines] compares it: the same
     continuation, the same values and the same heap (decision D40).

   The form in which Simplify makes the replacement
   ([direct_call_meet_set_refines_statement],
   [direct_call_meet_set_bottom_statement]). Simplify meets the code id
   of the call with the code id that the type of the callee gives, as
   two sets of one code id, names the one code id of the answer, and
   puts [Invalid] where the answer is [Bottom]
   (simplify/simplify_apply_expr.ml, [simplify_direct_function_call]).
   The two statements are of [meet_set] for one compilation unit, with
   enough steps (FlambdaCodeAgeRelationSpec.v says why), and what the
   type says of the callee is a premise: the callee is a closure of
   the code of the type or of a newer version of it. The first has the
   hypotheses of the statement above for the code id answered. The
   second says that the call is undefined on the machine: nothing of
   the kind is tested when the program runs.

   Not here: the lists of code ids of calls of known arity (decision
   D59) and the code ids of function types (discrepancy T4-4), which
   mean "or a newer version" in the compiler too; a field of the code
   of the model for the older version, and a relation read off a term;
   a check of the links by the validation.

   THE STATEMENTS OF THE TREE THAT DO NOT HOLD ON THE MACHINE AS THEY
   STAND. "On the machine" is: with [step_aged age_of] in the place of
   [step], for every [age_of]. No search of every statement of the
   tree was made. What follows is what was found in the files of
   rewrites/calls/ and rewrites/closures/, with how each was found.
   - Shown here, by Rocq, on one unit
     ([direct_call_runs_code_named_statement]: the direct call exits
     with one value, and the same call made indirect with another):
     FlambdaStubCallsSpec.demote_call_step_statement
     ([demote_call_step_aged_fails_statement]),
     demote_calls_refines_statement
     ([demote_calls_refines_aged_fails_statement]) and
     demote_full_call_step_statement
     ([demote_full_call_step_aged_fails_statement]). A direct call that
     is made indirect goes by the callee's own code, where the direct
     call went by the code named.
   - By the same argument, read from the definitions and not checked by
     Rocq: the other refinements of FlambdaStubCallsSpec.v
     (demote_in_stubs_refines_statement,
     demote_stub_calls_refines_statement,
     demote_full_calls_refines_statement,
     demote_full_in_stubs_refines_statement,
     demote_full_stub_calls_refines_statement); and the statements of
     FlambdaCopyCodeSpec.v and FlambdaCopyLetsSpec.v that their rewrites
     refine (copy_static_code_program_refines_statement,
     with_copies_program_refines_statement,
     copy_let_code_program_refines_statement,
     with_let_copies_program_refines_statement, and those from the
     imports and for the observers), whose rewrites make indirect the
     direct calls that name copied code.
     What would repair them: for a statement that two steps are the
     same, that the callee is a closure of the code named; for a
     refinement, that, or that the relation is sound for the callee
     with the versions bound, since the callee's own code then refines
     the code named.
   - By an argument from the heads of their files, not checked by Rocq:
     the statements that rest on "the body of code runs only for
     closures of that code". On the machine it also runs for a closure
     of a newer version, at a direct call that names it. They are
     FlambdaClosureOfCodeSpec.know_code_closures_program_refines_statement,
     know_code_slots_program_refines_statement and the three statements
     of [know_own_closures] ("[my_closure] is known to be a closure of
     the code [id]: a call finds the body that it runs by the code of
     its callee"); the statements of FlambdaSlotsThroughSpec.v that are
     of [know_own_closures_through], which applies that rewrite; and
     FlambdaLiftSpec.lift_unit_program_refines_statement, which applies
     FlambdaMyClosureSpec.bind_my_closures ("It is right only if the
     closure called is the symbol whenever that code runs").
     What would repair them: that no direct call names an older version
     of the code of its callee, as with the empty relation. Soundness
     does not: these rewrites put the code named, or what is known of
     its own closures, in the place of the callee's code, which needs
     the older code to refine the newer.
   - And every statement that is built on those: the statements of
     validation/checks/ that apply these rewrites.
   Read and not found to fail, by the same reading and not checked by
   Rocq: the inlining of a direct call by the code that it names
   (FlambdaApplySpec.inline_statement and what is built on it), which
   is what the machine does at such a call. With the other meaning that
   such a call could be given, that the callee's own code runs, the
   statements of the first two groups would hold and the inlining
   would not; and the statement of this file would need no sound
   relation.

   The proofs are in FlambdaCodeAgeCallsProofs.v, which needs no
   review. *)

From Stdlib Require Import PArith ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaStubCallsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaCodeAgeRelation FlambdaCodeAgeRelationSpec.
Import ListNotations.

(* ======================================================================== *)
(* Runs, behaviours and refinement, of a machine given by its step          *)
(* ======================================================================== *)

(* FlambdaSemanticsSpec.run, [behaves] and [refines], with the step of
   the machine a parameter: the same definitions, word for word, with
   [machine_step] in the place of [step]. *)
Section Machine.

Variable machine_step : state -> step_result.

Fixpoint run_by (fuel : nat) (s : state) : step_result :=
  match fuel with
  | O => Step s
  | S fuel =>
    match machine_step s with
    | Step s' => run_by fuel s'
    | result => result
    end
  end.

Definition behaves_by (s : state) (b : behaviour) : Prop :=
  match b with
  | Exits k vs hp =>
    ∃ (fuel : nat) (rd : read_record), run_by fuel s = Exit k vs hp rd
  | Goes_wrong => ∃ fuel : nat, run_by fuel s = Undefined
  | Diverges => ∀ fuel : nat, ∃ s' : state, run_by fuel s = Step s'
  end.

Definition refines_by (source target : state) : Prop :=
  (∀ (k : continuation) (vs : list value) (hp : heap),
    behaves_by source (Exits k vs hp) -> behaves_by target (Exits k vs hp))
  ∧ (∀ b : behaviour,
      b <> Diverges ->
      behaves_by target b ->
      behaves_by source b ∨ behaves_by source Goes_wrong).

(* The same of what two steps come to: the first is undefined; or they
   are two states of which the second refines the first; or they are
   two exits by the same continuation, with the same values and the same
   heap. An exit is compared as [refines] compares it (decision D40);
   the read records are not compared, as a behaviour has none. *)
Definition result_refines_by (source target : step_result) : Prop :=
  match source, target with
  | Undefined, _ => True
  | Step s, Step s' => refines_by s s'
  | Exit k vs hp _, Exit k' vs' hp' _ => k = k' ∧ vs = vs' ∧ hp = hp'
  | Step _, (Exit _ _ _ _ | Undefined) => False
  | Exit _ _ _ _, (Step _ | Undefined) => False
  end.

End Machine.

(* With [step] for the step they are the definitions of the core. *)
Definition run_by_step_statement : Prop :=
  ∀ (fuel : nat) (s : state), run_by step fuel s = run fuel s.

Definition refines_by_step_statement : Prop :=
  ∀ source target : state,
    refines_by step source target <-> refines source target.

(* ======================================================================== *)
(* The machine that is given a code age relation                            *)
(* ======================================================================== *)

(* What a direct call that names [named] is, of the closure [closure],
   with the relation [rel]; [None] where it is undefined. The closure
   must be of the code named or of a newer version of it, and the code
   named must be bound. The call is then what
   FlambdaSemanticsSpec.shape_of_closure_call makes of a direct call,
   with the code named for the code: the arguments are those that the
   code named takes ([arguments_for_code]), and the shape of the call is
   told by the code named ([shape_of_code_call]). *)
(* CR mshinwell: a direct call of a closure whose code is not the code
   named, nor a newer version of it in the relation, is undefined here,
   as it is on [step] (discrepancy C0-1). The backend would run the
   code named with that closure. Simplify takes such a call for one
   that cannot be reached: [meet_set] of the two code ids answers
   [Bottom], and the call is made [Invalid]
   (simplify/simplify_apply_expr.ml, [simplify_direct_function_call]).
   Is it undefined? And a direct call is defined here where the code of
   its callee is not bound, if the code named is bound and the callee
   is a closure of a newer version of it: only the code named is
   looked up. May the code of a closure that a run holds be deleted,
   so that such a call is to have a meaning? (For a call of unknown
   arity the model says no, on trust: decision D20.) *)
Definition shape_of_aged_direct_call (rel : Code_age_relation.t)
    (statics : static_env) (closure : value) (named : code_id)
    (vs : list value) (args_arity return_arity : arity)
    : option call_shape :=
  match closure_of statics closure with
  | Some (called, _) =>
    if same_or_newer_b rel called named
    then
      match lookup named (static_code statics) with
      | Some c =>
        let* '(vs', args_arity') :=
          arguments_for_code statics c vs args_arity
        in
        shape_of_code_call closure c vs' args_arity' return_arity
      | None => None
      end
    else None
  | None => None
  end.

(* The step of the machine that is given, for each static environment,
   a code age relation. It is [step] but at a direct call. There the
   callee, the arguments and the extra arguments of the exception
   continuation are evaluated, and the continuations are resolved, as
   [step] does; the call is what [shape_of_aged_direct_call] says, and
   it runs as every call does ([run_call]). *)
Definition step_aged (age_of : static_env -> Code_age_relation.t)
    (s : state) : step_result :=
  match s with
  | State (Apply ap) vars conts statics hp rd =>
    match ap_call_kind ap with
    | Direct named =>
      match
        eval_simple vars (ap_callee ap), eval_simples vars (ap_args ap),
        eval_simples vars (exn_extra_simples ap)
      with
      | Some callee, Some vs, Some extra =>
        let return_to := resolve (ap_continuation ap) conts in
        run_call
          (fun partial => apply_cont_value return_to [partial] statics hp rd)
          (shape_of_aged_direct_call (age_of statics) statics callee named
             vs (ap_args_arity ap) (ap_return_arity ap))
          (ap_return_arity ap) return_to
          (with_extra_args (resolve (ap_exn_continuation ap) conts) extra)
          statics hp rd
      | _, _, _ => Undefined
      end
    | Indirect_unknown_arity | Indirect_known_arity _ => step s
    end
  | _ => step s
  end.

(* Its runs, its behaviours and its refinement. *)
Definition run_aged (age_of : static_env -> Code_age_relation.t)
    : nat -> state -> step_result :=
  run_by (step_aged age_of).

Definition behaves_aged (age_of : static_env -> Code_age_relation.t)
    : state -> behaviour -> Prop :=
  behaves_by (step_aged age_of).

Definition refines_aged (age_of : static_env -> Code_age_relation.t)
    : state -> state -> Prop :=
  refines_by (step_aged age_of).

Definition result_refines_aged (age_of : static_env -> Code_age_relation.t)
    : step_result -> step_result -> Prop :=
  result_refines_by (step_aged age_of).

(* The machine gives a state the step that [step] gives it, wherever
   [step] gives it one; and with the empty relation it is [step]. *)
Definition step_aged_extends_statement : Prop :=
  ∀ (age_of : static_env -> Code_age_relation.t) (s : state),
    step s <> Undefined -> step_aged age_of s = step s.

Definition step_aged_empty_statement : Prop :=
  ∀ s : state, step_aged (fun _ => Code_age_relation.empty) s = step s.

(* So a state that does not go wrong on [step] behaves in the same way
   on the machine. *)
Definition behaves_aged_extends_statement : Prop :=
  ∀ (age_of : static_env -> Code_age_relation.t) (s : state) (b : behaviour),
    b <> Goes_wrong -> behaves s b -> behaves_aged age_of s b.

(* ======================================================================== *)
(* A relation that is sound for a closure                                   *)
(* ======================================================================== *)

(* [v] is a closure of the code [id] or of a newer version of it. *)
Definition closure_at_least (rel : Code_age_relation.t)
    (statics : static_env) (v : value) (id : code_id) : Prop :=
  ∃ (called : code_id) (slots : list (value_slot * value)),
    closure_of statics v = Some (called, slots)
    ∧ same_or_newer rel called id.

(* The code [c'] in the place of the code [c], for the closure [v]: the
   two take their arguments in the same way, and a full call of [c']
   with that closure refines the full call of [c] with it, whatever the
   arguments, the arities of the call, the continuations, the heap and
   the read record. The first three conjuncts are what makes a call
   have one shape whichever of the two it names
   ([arguments_for_code], [shape_of_code_call]). *)
(* CR mshinwell: does a newer version of code always have the arity of
   the parameters, the arity of the results and [is_tupled] of the code
   that it is a newer version of? Simplify makes the new code of a
   function with the [params_arity] and the [is_tupled] of the old
   (simplify/simplify_set_of_closures.ml, [simplify_function0]). If
   another pass may change them and still say [newer_version_of], a
   direct call that names the older code and one that names the newer
   are not the same call, and this hypothesis is too strong. *)
Definition code_refines_for (age_of : static_env -> Code_age_relation.t)
    (statics : static_env) (v : value) (c c' : code) : Prop :=
  code_params_arity c' = code_params_arity c
  ∧ code_result_arity c' = code_result_arity c
  ∧ code_is_tupled c' = code_is_tupled c
  ∧ ∀ (vs : list value) (args_arity return_arity : arity)
      (return_to exn_to : cont_value) (hp : heap) (rd : read_record),
      result_refines_aged age_of
        (call_code v c vs args_arity return_arity return_to exn_to statics hp
           rd)
        (call_code v c' vs args_arity return_arity return_to exn_to statics
           hp rd).

(* The relation is sound for the closure [v], under [statics]: for each
   code that [v] is a closure of or of a newer version of, the code
   refines its older version for [v], where both are bound. *)
Definition sound_for (age_of : static_env -> Code_age_relation.t)
    (statics : static_env) (v : value) : Prop :=
  ∀ (newer older : code_id) (c c' : code),
    closure_at_least (age_of statics) statics v newer ->
    Code_age_relation.get_older_version_of (age_of statics) newer
    = Some older ->
    lookup older (static_code statics) = Some c ->
    lookup newer (static_code statics) = Some c' ->
    code_refines_for age_of statics v c c'.

(* Every code from [newer] down to [older] is bound: [newer], [older]
   and every code between them in the order. *)
Definition versions_bound (rel : Code_age_relation.t)
    (statics : static_env) (newer older : code_id) : Prop :=
  ∀ id : code_id,
    same_or_newer rel newer id ->
    same_or_newer rel id older ->
    ∃ c : code, lookup id (static_code statics) = Some c.

(* [Apply_expr.with_call_kind]: the same call, of another kind. *)
Definition with_call_kind (ap : apply_expr) (kind : call_kind)
    : apply_expr :=
  Apply_expr (ap_callee ap) kind (ap_args ap) (ap_continuation ap)
    (ap_exn_continuation ap) (ap_exn_extra_args ap) (ap_args_arity ap)
    (ap_return_arity ap).

(* ======================================================================== *)
(* A direct call may name a newer version                                   *)
(* ======================================================================== *)

(* A direct call that names [named], of a callee that is a closure of
   [newer] or of a newer version of it, where [newer] is [named] or a
   newer version of it: the call that names [newer] refines it. *)
Definition direct_call_newer_version_refines_statement : Prop :=
  ∀ (age_of : static_env -> Code_age_relation.t) (ap : apply_expr)
    (named newer : code_id) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) (callee : value),
    ap_call_kind ap = Direct named ->
    eval_simple vars (ap_callee ap) = Some callee ->
    closure_at_least (age_of statics) statics callee newer ->
    same_or_newer (age_of statics) newer named ->
    versions_bound (age_of statics) statics newer named ->
    sound_for age_of statics callee ->
    refines_aged age_of (State (Apply ap) vars conts statics hp rd)
      (State (Apply (with_call_kind ap (Direct newer))) vars conts statics hp
         rd).

(* The same in the form in which Simplify makes the replacement: the
   code id of the call is met with the code id that the type of the
   callee gives, as two sets of one code id, and the call names the one
   code id of the answer. The relation is that of one compilation unit;
   what the type says of the callee is the premise that the callee is a
   closure of [from_type] or of a newer version of it. *)
Definition direct_call_meet_set_refines_statement : Prop :=
  ∀ (age_of : static_env -> Code_age_relation.t) (steps : nat)
    (ap : apply_expr) (named from_type met : code_id) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record)
    (callee : value),
    length (age_of statics) < steps ->
    meet_set_in_unit steps (age_of statics) [named] [from_type]
    = Some (Or_bottom.Ok [met]) ->
    ap_call_kind ap = Direct named ->
    eval_simple vars (ap_callee ap) = Some callee ->
    closure_at_least (age_of statics) statics callee from_type ->
    versions_bound (age_of statics) statics met named ->
    sound_for age_of statics callee ->
    refines_aged age_of (State (Apply ap) vars conts statics hp rd)
      (State (Apply (with_call_kind ap (Direct met))) vars conts statics hp
         rd).

(* Where the answer is [Bottom], the call is undefined, and anything
   refines it: Simplify puts [Invalid] in its place. *)
Definition direct_call_meet_set_bottom_statement : Prop :=
  ∀ (age_of : static_env -> Code_age_relation.t) (steps : nat)
    (ap : apply_expr) (named from_type : code_id) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record)
    (callee : value),
    length (age_of statics) < steps ->
    meet_set_in_unit steps (age_of statics) [named] [from_type]
    = Some Or_bottom.Bottom ->
    ap_call_kind ap = Direct named ->
    eval_simple vars (ap_callee ap) = Some callee ->
    closure_at_least (age_of statics) statics callee from_type ->
    step_aged age_of (State (Apply ap) vars conts statics hp rd) = Undefined.

(* ======================================================================== *)
(* Two units                                                                *)
(* ======================================================================== *)

Section Examples.

Let x : variable := 1%positive.
Let my_closure : variable := 2%positive.
Let callee : variable := 3%positive.
Let ret : continuation := 1%positive.
Let exn : continuation := 2%positive.
Let original : code_id := 1%positive.
Let version_one : code_id := 2%positive.
Let version_two : code_id := 3%positive.

(* Code of one parameter that returns the simple given. *)
Definition age_example_code (result : simple) : code :=
  Code [x] my_closure ret exn
    (make_code_metadata [Singleton Value] [Singleton Value])
    (Apply_cont (Apply_cont_expr ret [result] None)).

(* A call of the variable [callee], of the kind given, with one
   argument. *)
Definition age_example_call (kind : call_kind) : apply_expr :=
  Apply_expr (Var callee) kind [Const (Tagged_immediate 5)] ret exn []
    [Singleton Value] [Singleton Value].

(* The state in which that call is made, of a closure of [version_one],
   under the code given. *)
Definition age_example_state (code : list (code_id * code))
    (kind : call_kind) : state :=
  State (Apply (age_example_call kind))
    [(callee, Value_closure version_one [])]
    [(ret, Outer ret); (exn, Outer exn)]
    {| static_code := code; static_values := [] |}
    empty_heap no_reads.

(* [version_one] and [version_two] are two newer versions of
   [original]. *)
Definition age_example_relation : Code_age_relation.t :=
  Code_age_relation.add
    (Code_age_relation.add Code_age_relation.empty version_one original)
    version_two original.

(* The three codes are one code, which returns its argument. *)
Definition age_example_same_code : list (code_id * code) :=
  [(original, age_example_code (Var x));
   (version_one, age_example_code (Var x));
   (version_two, age_example_code (Var x))].

(* The direct call that names [original], of the closure of
   [version_one]: [step] gives it no step; the machine runs it, and it
   exits with the argument, as the call that names [version_one]
   does. *)
Definition direct_call_older_code_example_statement : Prop :=
  let age_of := fun _ : static_env => age_example_relation in
  let exit :=
    Exit ret [Value_const (Tagged_immediate 5)] empty_heap no_reads
  in
  step (age_example_state age_example_same_code (Direct original))
  = Undefined
  ∧ run_aged age_of 2
      (age_example_state age_example_same_code (Direct original))
    = exit
  ∧ run_aged age_of 2
      (age_example_state age_example_same_code (Direct version_one))
    = exit.

(* The condition on the closure cannot go. [version_two] is a newer
   version of [original], every code is bound, and the relation is
   sound for the closure; but the closure is of [version_one], which is
   not [version_two] or a newer version of it, and the call that names
   [version_two] does not refine the call that names [original]. *)
Definition direct_call_other_version_statement : Prop :=
  let age_of := fun _ : static_env => age_example_relation in
  let statics :=
    {| static_code := age_example_same_code; static_values := [] |}
  in
  same_or_newer age_example_relation version_two original
  ∧ versions_bound age_example_relation statics version_two original
  ∧ sound_for age_of statics (Value_closure version_one [])
  ∧ ~ refines_aged age_of
        (age_example_state age_example_same_code (Direct original))
        (age_example_state age_example_same_code (Direct version_two)).

(* [version_one] is a newer version of [original], and the two codes
   differ: the older returns 0 and the newer returns 1. The relation is
   not sound. *)
Definition age_example_unsound_relation : Code_age_relation.t :=
  Code_age_relation.add Code_age_relation.empty version_one original.

Definition age_example_other_code : list (code_id * code) :=
  [(original, age_example_code (Const (Tagged_immediate 0)));
   (version_one, age_example_code (Const (Tagged_immediate 1)))].

(* The code named runs: the direct call that names [original], of the
   closure of [version_one], exits with 0, and the indirect call of the
   same closure exits with 1. *)
Definition direct_call_runs_code_named_statement : Prop :=
  let age_of := fun _ : static_env => age_example_unsound_relation in
  run_aged age_of 2
    (age_example_state age_example_other_code (Direct original))
  = Exit ret [Value_const (Tagged_immediate 0)] empty_heap no_reads
  ∧ run_aged age_of 2
      (age_example_state age_example_other_code Indirect_unknown_arity)
    = Exit ret [Value_const (Tagged_immediate 1)] empty_heap no_reads.

End Examples.

(* ======================================================================== *)
(* The statements of the tree are of [step]                                 *)
(* ======================================================================== *)

(* Three statements of FlambdaStubCallsSpec.v, with the machine for
   [step]: none holds. They are
   FlambdaStubCallsSpec.demote_call_step_statement,
   demote_calls_refines_statement and demote_full_call_step_statement,
   and the unit of [direct_call_runs_code_named_statement] refutes the
   three: the direct call has a step on the machine, the call made
   indirect has another, whether it is made a call of unknown arity or
   one of known arity, and the two runs exit with different values. *)
Definition demote_call_step_aged_fails_statement : Prop :=
  ~ (∀ (age_of : static_env -> Code_age_relation.t) (ids : list code_id)
       (ap : apply_expr) (vars : env) (conts : cont_env)
       (statics : static_env) (hp : heap) (rd : read_record),
       step_aged age_of (State (Apply ap) vars conts statics hp rd)
       <> Undefined ->
       step_aged age_of
         (State (Apply (demote_call ids ap)) vars conts statics hp rd)
       = step_aged age_of (State (Apply ap) vars conts statics hp rd)).

Definition demote_calls_refines_aged_fails_statement : Prop :=
  ~ (∀ (age_of : static_env -> Code_age_relation.t) (ids : list code_id)
       (e : expr) (vars : env) (conts : cont_env) (statics : static_env)
       (hp : heap) (rd : read_record),
       refines_aged age_of (State e vars conts statics hp rd)
         (State (demote_calls ids e) vars conts statics hp rd)).

Definition demote_full_call_step_aged_fails_statement : Prop :=
  ~ (∀ (age_of : static_env -> Code_age_relation.t)
       (known : list (code_id * code)) (ap : apply_expr) (vars : env)
       (conts : cont_env) (statics : static_env) (hp : heap)
       (rd : read_record),
       statics_know_signatures known statics ->
       step_aged age_of (State (Apply ap) vars conts statics hp rd)
       <> Undefined ->
       step_aged age_of
         (State (Apply (demote_full_call known ap)) vars conts statics hp rd)
       = step_aged age_of (State (Apply ap) vars conts statics hp rd)).
