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


(* Simplifying a [Switch] from the type of its scrutinee: the arms that
   cannot be taken are removed.

   This file is to be reviewed: the definitions and the statements,
   against [simplify_switch] of
   middle_end/flambda2/simplify/simplify_switch_expr.ml and
   [create_switch] of middle_end/flambda2/simplify/expr_builder.ml.
   Refinement relative to a typing environment is
   FlambdaTypingEnvRefinementSpec.v; the contract of the meet is
   FlambdaMeetSpec.v; the machine is FlambdaSemanticsSpec.v;
   what it is for a static environment to be valid is
   FlambdaClosuresSpec.v.

   What Simplify does. It reads the type of the scrutinee from its
   typing environment ([Simplify_simple.simplify_simple]: the alias
   type of the canonical element of the scrutinee at the normal name
   mode, with that element).
   For each arm it meets that type with the type of the number of the
   arm ([this_naked_immediate]), in the environment where the [Switch]
   is. Where the meet is bottom the arm is dropped. If one arm is left
   the [Switch] becomes the [Apply_cont] of that arm; if none is left
   it becomes [Invalid]; otherwise it is a [Switch] on the canonical
   element with the arms that are left. [simplify_switch] here is that
   function, with the meet as a parameter.

   What is stated.
   - Of the machine alone: a [Switch] with no arm is [Invalid]
     ([switch_no_arms_statement]), and a [Switch] refines what
     [switch_of_arms] makes of its arms
     ([switch_of_arms_refines_statement]). The second is a refinement
     and not an equivalence: a [Switch] with one arm is undefined where
     its scrutinee is not the number of the arm, and the [Apply_cont]
     is not.
   - Under a typing environment, in the states whose static
     environment is valid ([expr_refines_under_valid],
     [expr_equivalent_under_valid]; [under_valid_weaken_statement]
     says how they compare with the notions without that premise): the
     arms that cannot be taken in any such state that satisfies the
     environment may be removed ([arm_not_taken],
     [switch_without_arms_statement]). In every state that satisfies
     the environment, the scrutinee may be replaced by its canonical
     element ([switch_canonical_statement]). Both are equivalences.
   - That Simplify's test is right: an arm whose meet is bottom cannot
     be taken ([pruned_arm_not_taken_statement]). So the [Switch] with
     the arms that the test keeps is equivalent to the [Switch]
     ([prune_arms_equivalent_statement]), and what [simplify_switch]
     answers refines it ([simplify_switch_refines_statement]). The last
     is the statement for a pass over terms to use: its premises are
     the two on the meet and its environment below, and that the
     scrutinee may occur in terms, which is a test.
   - Of the transcribed meet that the contract is proved of
     ([meet_to_prove], FlambdaMeetSpec.v): it has what is asked of a
     meet here ([meet_to_prove_bottom_sound_statement]), so with it
     the statement for a pass has no premise on the meet, and one on
     the join ([simplify_switch_of_meet_to_prove_refines_statement]).
   - The functions commute with a renaming of variables
     ([arm_kept_rename_statement], [prune_arms_rename_statement],
     [simplify_switch_rename_statement]), for a meet that does
     ([meet_function_equivariant], FlambdaMeetRenameSpec.v). The
     transcribed meet that the contract is proved of does, with a join
     that does ([simplify_switch_of_meet_to_prove_rename_statement]).
   - Two small terms ([example_switch_no_arm_statement],
     [example_switch_comparison_statement]).

   What is assumed.
   - The contract of the meet, as far as it is needed:
     [meet_bottom_sound meet] (FlambdaMeetSpec.v), which says that
     where a meet answers bottom no value has both types. The
     statements are about any function [meet] that has it. The
     transcribed meet that the contract is proved of ([meet_to_prove])
     has it, whatever the fuel, with any join that has the contract
     of a join ([join_scoped_envs_sound]): that is
     [meet_to_prove_sound_statement] of FlambdaMeetClosuresSpec.v,
     which gives [meet_sound], from which [meet_bottom_sound] follows
     ([meet_answers_bottom_statement]). What that leaves, as that file
     says: [meet_to_prove] is not the compiler's meet ([compiler_meet],
     FlambdaMeetSpec.v, says where the two differ, and the compiler's
     has not [meet_sound]); that the compiler's join has the contract
     of a join is not shown; and nothing is said where the fuel runs
     out, where [simplify_switch] answers nothing. The statements are
     not made at a prover: Simplify asks the meet here, not a prover,
     and what is stated is about the test that it makes. The two
     examples run [meet_to_prove] with the join that forgets
     ([forgetful_join], FlambdaMeetExamplesSpec.v), which has the
     contract of a join, and have no hypothesis.
   - What the contract asks of the environment of a meet:
     [typing_env_fits tenv] (FlambdaMeetSpec.v, which says what in it
     is not known of the environments of the compiler). The two types
     that are met need no premise: the type of a number fits any
     environment, and the type that the environment answers for a
     [Simple] fits it ([type_simple_in_term_fits_statement]). Nor do
     their kinds: where the scrutinee is a naked immediate in the
     machine its type is of that kind, and where it is not the [Switch]
     is undefined.
   - That the static environment is valid ([statics_valid],
     FlambdaClosuresSpec.v): the contract of the meet
     supposes it (FlambdaMeetSpec.v says why), and the states of a run
     of a well-formed unit have it. [expr_refines_under] of
     FlambdaTypingEnvRefinementSpec.v is about every state that
     satisfies the environment, valid or not, so what rests on the
     meet is stated here of [expr_refines_under_valid], which is that
     notion with the one premise more. A statement of that file about
     a binder does not carry it: its premise is about every state. For
     the examples, the two about a [Let] that they use are proved
     again with the premise, in the Proofs file.
   - Strict satisfaction ([env_satisfies_strictly]), which is what
     [expr_refines_under] is made of and what the contract of the meet
     is about. After the unboxing of variants the environments that
     Simplify builds are satisfied only where poison is exempt
     (decision D26), and these statements do not apply there.
   - That the scrutinee may occur in terms
     ([Typing_env.mem_simple] at the normal name mode). Simplify does
     not test it: its terms are closed in its environment. It is what
     makes the value that the machine has for the scrutinee the one
     that a valuation has.

   Renaming. What is stated of [arm_kept], [prune_arms] and
   [simplify_switch] under a renaming supposes that the meet commutes
   with the renaming of its environment and of its two types
   ([meet_function_equivariant], FlambdaMeetRenameSpec.v). The
   transcribed meet that the contract is proved of has it, whatever
   the fuel, with a join that commutes with renaming
   ([meet_to_prove_equivariant_statement], in that file,
   [join_scoped_envs_equivariant]); the join that forgets, which the
   examples run, is such a join
   ([forgetful_join_equivariant_statement], there). So at that meet
   the equation of [simplify_switch] has one premise, on the join
   ([simplify_switch_of_meet_to_prove_rename_statement]). The renaming
   of typing environments and what it does to
   [Typing_env.type_simple_in_term_exn] are FlambdaEnvRenameSpec.v.

   The arms. In the compiler the arms are a map from numbers; here
   they are a list, and the machine takes the first arm for a number
   (FlambdaModel.v). The test of an arm reads its number
   only, so a list with two arms for one number loses both or neither,
   and the statements hold of any list. [switch_of_arms] counts the
   arms of the list; on a list that has two arms for one number and no
   other it answers the [Switch], where the compiler has no such
   [Switch] to answer about.

   Where the model answers nothing. [simplify_switch] is [None] where
   the scrutinee has no type or no canonical element at the normal name
   mode, which are fatal errors of the compiler, and where a meet
   answers nothing (FlambdaMeetSpec.v says what that stands for, at
   [meet_function]). The compiler makes the meets in the order of the
   numbers and stops at the first that fails; which one that is the
   model does not say.

   Not here: what each arm teaches (the environment that the meet
   answers for an arm, in which Simplify simplifies the arguments of
   the arm and which it records for the continuation); the arms that
   are removed on the way up ([rebuild_arm]); the arms that are the
   same [Apply_cont], which [create_switch] makes one
   ([switch_same_arms_statement], FlambdaControlFlowSpec.v,
   is the law); the merging of arms and the rewrites of a boolean
   [Switch] ([rebuild_switch]).

   The proofs are in FlambdaSimplifySwitchProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaPrimitiveModel FlambdaSemanticsSpec FlambdaClosuresSpec
  FlambdaNominalSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypeRenameSpec FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaTypingEnv
  FlambdaTypingEnvSpec FlambdaTypingEnvExamplesSpec
  FlambdaTypingEnvRefinementSpec FlambdaEnvRenameSpec FlambdaMeetGrammar
  FlambdaExpandHead FlambdaMeetEnv FlambdaMeet FlambdaMeetSpec
  FlambdaMeetRenameSpec.
From OxCamlFlambda2Types Require FlambdaMeetExamplesSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* What is made of the arms that are left                                   *)
(* ======================================================================== *)

(* What Simplify makes of the arms that are left. None: [Invalid]
   ([rebuild_switch]; [Expr_builder.create_switch] does the same). One:
   the [Apply_cont] of the arm ([simplify_switch], which takes this
   case before the arms are simplified, "so that it is an inlinable
   use"; [create_switch] does the same). Otherwise the [Switch].
   ([create_switch] also makes one [Apply_cont] of arms that are all
   the same; that is not here.) *)
Definition switch_of_arms (scrutinee : simple) (arms : list (Z * apply_cont))
    : expr :=
  match arms with
  | [] => Invalid
  | [(_, ac)] => Apply_cont ac
  | _ :: _ :: _ => Switch scrutinee arms
  end.

Definition switch_no_arms_statement : Prop :=
  ∀ scrutinee : simple, expr_equivalent (Switch scrutinee []) Invalid.

Definition switch_of_arms_refines_statement : Prop :=
  ∀ (scrutinee : simple) (arms : list (Z * apply_cont)),
    expr_refines (Switch scrutinee arms) (switch_of_arms scrutinee arms).

(* ======================================================================== *)
(* Refinement where the static environment is valid                         *)
(* ======================================================================== *)

(* [expr_refines_under] and [expr_equivalent_under]
   (FlambdaTypingEnvRefinementSpec.v), in the states whose static
   environment is valid. *)
(* CR mshinwell: these are here because the contract of the meet came
   to suppose a valid static environment after the notions of
   FlambdaTypingEnvRefinementSpec.v were made, and this is the first
   statement about terms that rests on the meet. Every rewrite of
   Simplify that asks the meet will need the premise. It cannot simply
   be added to [expr_refines_under], whose statements about handlers
   are about every state: a handler runs under a static environment
   that has grown since its [Let_cont], a [Let_static] may bind what
   is not valid, and that the static environment stays valid is known
   of the runs of a well-formed unit
   ([well_formed_values_valid_statement],
   FlambdaClosuresSpec.v) and not of every term.
   Decided with the owner of the function that simplifies a whole term
   by types: the pruning does not go into that function, whose theorem
   is about every state and is what the validator's rests on. It goes
   into a second function, with a theorem of its own; the premise is
   not carried through the first. The second function is that of
   FlambdaSimplifyPruningSpec.v, which takes a function that may
   rewrite a [Switch] and asks of it the notion that is here
   ([switch_rewrite_sound], there); [simplify_switch] is such a
   function ([simplify_switch_rewrite_sound_statement], there). Its
   theorem is from the states of the run of a unit whose sets of
   closures are well formed ([run_valid], there). A step keeps
   [run_valid], which is not so of a valid static environment by
   itself, and the static environment of a state that has it is
   valid; so what holds here holds from those states
   ([from_valid_weaken_statement], there). *)
Definition expr_refines_under_valid (tenv : Typing_env.t)
    (source target : expr) : Prop :=
  ∀ (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
    (rd : read_record),
    statics_valid statics ->
    env_satisfies_strictly tenv statics vars ->
    refines (State source vars conts statics hp rd)
      (State target vars conts statics hp rd).

Definition expr_equivalent_under_valid (tenv : Typing_env.t) (e e' : expr)
    : Prop :=
  ∀ (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
    (rd : read_record),
    statics_valid statics ->
    env_satisfies_strictly tenv statics vars ->
    equivalent (State e vars conts statics hp rd)
      (State e' vars conts statics hp rd).

(* What holds in every state that satisfies the environment holds in
   those whose static environment is valid; and an equivalence there
   is a refinement there. *)
Definition under_valid_weaken_statement : Prop :=
  ∀ (tenv : Typing_env.t) (e e' : expr),
    (expr_refines_under tenv e e' -> expr_refines_under_valid tenv e e')
    ∧ (expr_equivalent_under tenv e e' ->
       expr_equivalent_under_valid tenv e e')
    ∧ (expr_equivalent_under_valid tenv e e' ->
       expr_refines_under_valid tenv e e').

(* ======================================================================== *)
(* Arms that cannot be taken                                                *)
(* ======================================================================== *)

(* An arm cannot be taken under a typing environment: in no state of
   the machine that satisfies the environment strictly, and whose
   static environment is valid, is the scrutinee the number of the
   arm. *)
Definition arm_not_taken (tenv : Typing_env.t) (scrutinee : simple) (i : Z)
    : Prop :=
  ∀ (statics : static_env) (vars : env),
    statics_valid statics ->
    env_satisfies_strictly tenv statics vars ->
    eval_simple vars scrutinee <> Some (Value_const (Naked_immediate i)).

(* The arms whose numbers pass a test. *)
Definition keep_arms (keep : Z -> bool) (arms : list (Z * apply_cont))
    : list (Z * apply_cont) :=
  filter (fun arm => keep (fst arm)) arms.

(* The arms that cannot be taken may be removed. *)
Definition switch_without_arms_statement : Prop :=
  ∀ (tenv : Typing_env.t) (scrutinee : simple) (keep : Z -> bool)
    (arms : list (Z * apply_cont)),
    (∀ i : Z,
      i ∈ map fst arms -> keep i = false -> arm_not_taken tenv scrutinee i) ->
    expr_equivalent_under_valid tenv (Switch scrutinee arms)
      (Switch scrutinee (keep_arms keep arms)).

(* The scrutinee may be replaced by its canonical element, in every
   state that satisfies the environment. *)
Definition switch_canonical_statement : Prop :=
  ∀ (tenv : Typing_env.t) (scrutinee canonical : simple)
    (arms : list (Z * apply_cont)),
    canonical_in_terms tenv scrutinee canonical ->
    expr_equivalent_under tenv (Switch scrutinee arms)
      (Switch canonical arms).

(* ======================================================================== *)
(* Simplify's test                                                          *)
(* ======================================================================== *)

(* Whether Simplify keeps an arm: the meet of the type of the scrutinee
   with the type of the number of the arm is not bottom. [None] where
   the meet answers nothing. *)
Definition arm_kept (meet : meet_function) (tenv : Typing_env.t)
    (scrutinee_ty : flambda_type) (i : Z) : option bool :=
  match meet tenv scrutinee_ty (this_naked_immediate i) with
  | Some Or_bottom.Bottom => Some false
  | Some (Or_bottom.Ok _) => Some true
  | None => None
  end.

(* The arms that Simplify keeps (the [filter_map] of
   [simplify_switch]). The environment that the meet answers for an arm
   is not kept here. *)
Fixpoint prune_arms (meet : meet_function) (tenv : Typing_env.t)
    (scrutinee_ty : flambda_type) (arms : list (Z * apply_cont))
    : option (list (Z * apply_cont)) :=
  match arms with
  | [] => Some []
  | (i, ac) :: arms =>
    let* kept := arm_kept meet tenv scrutinee_ty i in
    let* arms := prune_arms meet tenv scrutinee_ty arms in
    Some (if kept then (i, ac) :: arms else arms)
  end.

(* [simplify_switch], as far as the term that it makes from the type of
   the scrutinee: the arguments of the arms are left as they are. *)
Definition simplify_switch (meet : meet_function) (tenv : Typing_env.t)
    (scrutinee : simple) (arms : list (Z * apply_cont)) : option expr :=
  let* answer :=
    Typing_env.type_simple_in_term_exn tenv (Some Name_mode.normal) scrutinee
  in
  let* '(scrutinee_ty, canonical) := answer in
  let* arms := prune_arms meet tenv scrutinee_ty arms in
  Some (switch_of_arms canonical arms).

(* The type that an environment answers for a [Simple] may be met in
   the environment. *)
Definition type_simple_in_term_fits_statement : Prop :=
  ∀ (tenv : Typing_env.t) (min_name_mode : option Name_mode.t)
    (s canonical : simple) (ty : flambda_type),
    typing_env_fits tenv ->
    Typing_env.type_simple_in_term_exn tenv min_name_mode s
    = Some (Some (ty, canonical)) ->
    type_fits tenv ty.

(* The test is right: an arm that Simplify drops cannot be taken. *)
Definition pruned_arm_not_taken_statement : Prop :=
  ∀ (meet : meet_function) (tenv : Typing_env.t)
    (scrutinee canonical : simple) (scrutinee_ty : flambda_type) (i : Z),
    meet_bottom_sound meet ->
    typing_env_fits tenv ->
    Typing_env.mem_simple (Some Name_mode.normal) tenv scrutinee = true ->
    Typing_env.type_simple_in_term_exn tenv (Some Name_mode.normal) scrutinee
    = Some (Some (scrutinee_ty, canonical)) ->
    arm_kept meet tenv scrutinee_ty i = Some false ->
    arm_not_taken tenv scrutinee i.

(* So the [Switch] with the arms that Simplify keeps is the [Switch]. *)
Definition prune_arms_equivalent_statement : Prop :=
  ∀ (meet : meet_function) (tenv : Typing_env.t)
    (scrutinee canonical : simple) (scrutinee_ty : flambda_type)
    (arms arms' : list (Z * apply_cont)),
    meet_bottom_sound meet ->
    typing_env_fits tenv ->
    Typing_env.mem_simple (Some Name_mode.normal) tenv scrutinee = true ->
    Typing_env.type_simple_in_term_exn tenv (Some Name_mode.normal) scrutinee
    = Some (Some (scrutinee_ty, canonical)) ->
    prune_arms meet tenv scrutinee_ty arms = Some arms' ->
    expr_equivalent_under_valid tenv (Switch scrutinee arms)
      (Switch scrutinee arms').

(* What Simplify makes of a [Switch] refines it. *)
Definition simplify_switch_refines_statement : Prop :=
  ∀ (meet : meet_function) (tenv : Typing_env.t) (scrutinee : simple)
    (arms : list (Z * apply_cont)) (e : expr),
    meet_bottom_sound meet ->
    typing_env_fits tenv ->
    Typing_env.mem_simple (Some Name_mode.normal) tenv scrutinee = true ->
    simplify_switch meet tenv scrutinee arms = Some e ->
    expr_refines_under_valid tenv (Switch scrutinee arms) e.

(* ======================================================================== *)
(* With the meet that the contract is proved of                             *)
(* ======================================================================== *)

(* The transcribed meet that the contract is proved of has what the
   statements above ask of a meet, with a join that has the contract
   of a join and whatever the fuel. *)
Definition meet_to_prove_bottom_sound_statement : Prop :=
  ∀ (join : join_scoped_envs) (fuel : nat),
    join_scoped_envs_sound join ->
    meet_bottom_sound (meet_to_prove join fuel).

(* So what [simplify_switch] answers with that meet refines the
   [Switch]. Where the fuel runs out in a meet it answers nothing. *)
Definition simplify_switch_of_meet_to_prove_refines_statement : Prop :=
  ∀ (join : join_scoped_envs) (fuel : nat) (tenv : Typing_env.t)
    (scrutinee : simple) (arms : list (Z * apply_cont)) (e : expr),
    join_scoped_envs_sound join ->
    typing_env_fits tenv ->
    Typing_env.mem_simple (Some Name_mode.normal) tenv scrutinee = true ->
    simplify_switch (meet_to_prove join fuel) tenv scrutinee arms = Some e ->
    expr_refines_under_valid tenv (Switch scrutinee arms) e.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

(* That a meet commutes with a renaming of the variables (of a renamed
   environment and renamed types it answers the renamed answer) is
   [meet_function_equivariant], FlambdaMeetRenameSpec.v. *)

(* The arms of a [Switch], renamed ([rename_expr],
   FlambdaNominalSpec.v). *)
Definition rename_arms (r : renaming) (arms : list (Z * apply_cont))
    : list (Z * apply_cont) :=
  map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms.

Definition arm_kept_rename_statement : Prop :=
  ∀ (meet : meet_function) (r : renaming) (tenv : Typing_env.t)
    (scrutinee_ty : flambda_type) (i : Z),
    meet_function_equivariant meet ->
    arm_kept meet (rename_typing_env r tenv) (rename_type r scrutinee_ty) i
    = arm_kept meet tenv scrutinee_ty i.

Definition prune_arms_rename_statement : Prop :=
  ∀ (meet : meet_function) (r : renaming) (tenv : Typing_env.t)
    (scrutinee_ty : flambda_type) (arms : list (Z * apply_cont)),
    meet_function_equivariant meet ->
    prune_arms meet (rename_typing_env r tenv) (rename_type r scrutinee_ty)
      (rename_arms r arms)
    = option_map (rename_arms r) (prune_arms meet tenv scrutinee_ty arms).

Definition simplify_switch_rename_statement : Prop :=
  ∀ (meet : meet_function) (r : renaming) (tenv : Typing_env.t)
    (scrutinee : simple) (arms : list (Z * apply_cont)),
    meet_function_equivariant meet ->
    simplify_switch meet (rename_typing_env r tenv) (rename_simple r scrutinee)
      (rename_arms r arms)
    = option_map (rename_expr r) (simplify_switch meet tenv scrutinee arms).

(* At the transcribed meet that the contract is proved of, with a join
   that commutes with renaming
   ([meet_to_prove_equivariant_statement], FlambdaMeetRenameSpec.v). *)
Definition simplify_switch_of_meet_to_prove_rename_statement : Prop :=
  ∀ (join : join_scoped_envs) (fuel : nat) (r : renaming)
    (tenv : Typing_env.t) (scrutinee : simple) (arms : list (Z * apply_cont)),
    join_scoped_envs_equivariant join ->
    simplify_switch (meet_to_prove join fuel) (rename_typing_env r tenv)
      (rename_simple r scrutinee) (rename_arms r arms)
    = option_map (rename_expr r)
        (simplify_switch (meet_to_prove join fuel) tenv scrutinee arms).

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

(* The variables are those of FlambdaTypingEnvExamplesSpec.v. The
   conclusions are plain; the examples are there to show that the
   premises of [simplify_switch_refines_statement] can be met by the
   environments that Simplify makes as it goes under a [Let]
   (FlambdaTypingEnvRefinementSpec.v), and that the transcribed meet,
   when it is run, drops the arms that are said. *)

(* The meet that is run: the transcribed meet that the contract is
   proved of, with the join that forgets, which has the contract of a
   join ([forgetful_join_sound_statement], FlambdaMeetExamplesSpec.v),
   and fuel enough for these meets. So it has [meet_bottom_sound]
   ([meet_to_prove_bottom_sound_statement]), and the examples have no
   hypothesis. *)
Definition example_meet : meet_function :=
  meet_to_prove FlambdaMeetExamplesSpec.forgetful_join 10.

(* [let i = 5 in switch i with 0 -> k0 | 1 -> k1], from the environment
   that knows nothing, where Simplify starts: so in every state whose
   static environment is valid. [i] is made equal to the constant, so
   the type of the scrutinee is the alias type of the constant. Its
   meet with the number of each arm is bottom, no arm is left, and the
   [Switch] is [Invalid]. *)
Definition example_switch_no_arm_statement : Prop :=
  ∀ k0 k1 : continuation,
    expr_refines_under_valid Typing_env.create
      (Let example_i (Naked_number Kind_naked_immediate)
         (Simple (Const (Naked_immediate 5)))
         (Switch (Var example_i)
            [(0, Apply_cont_expr k0 [] None);
             (1, Apply_cont_expr k1 [] None)]))
      (Let example_i (Naked_number Kind_naked_immediate)
         (Simple (Const (Naked_immediate 5))) Invalid).

(* [let i = (x < 7) in switch i with 0 -> k0 | 1 -> k1 | 2 -> k2], the
   comparison being of tagged immediates, signed, in the environment in
   which [x] is defined at the kind of values and nothing more is known
   of it: that of a handler whose parameter [x] is. Simplify cannot
   decide the comparison; it gives [i] the type of the two booleans
   ([Simplify_binary_primitive]: [unknown] of
   [Int_ops_for_binary_comp]), which every value of the comparison has.
   The scrutinee [i] is its own canonical element. The meet of its type
   with 2 is bottom, and with 0 or with 1 it is not: the arm for 2 is
   dropped. *)
Definition example_switch_comparison (arms : list (Z * apply_cont)) : expr :=
  Let example_i (Naked_number Kind_naked_immediate)
    (Prim
       (Binary
          (Int_comp Standard_tagged_immediate (Yielding_bool (Int_lt Signed))))
       [Var example_x; Const (Tagged_immediate 7)])
    (Switch (Var example_i) arms).

Definition example_switch_comparison_statement : Prop :=
  ∀ k0 k1 k2 : continuation,
    expr_refines_under_valid
      (Typing_env.add_variable_definition Typing_env.create example_x Value
         Name_mode.normal)
      (example_switch_comparison
         [(0, Apply_cont_expr k0 [] None); (1, Apply_cont_expr k1 [] None);
          (2, Apply_cont_expr k2 [] None)])
      (example_switch_comparison
         [(0, Apply_cont_expr k0 [] None); (1, Apply_cont_expr k1 [] None)]).
