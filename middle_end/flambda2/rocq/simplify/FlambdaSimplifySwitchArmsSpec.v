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


(* Simplifying a [Switch], second slice: the arms that are kept are
   simplified with what the meet has found of them.

   This file is to be reviewed: the definitions and the statements,
   against [simplify_switch] and [simplify_arm] of
   middle_end/flambda2/simplify/simplify_switch_expr.ml. The first
   slice is FlambdaSimplifySwitchSpec.v, whose notions and functions
   are used here; the contract of the meet is FlambdaMeetSpec.v; the
   function that replaces the arguments of an [Apply_cont] by their
   canonical elements is FlambdaSimplifyPassSpec.v.

   What Simplify does. For each arm it meets the type of the scrutinee
   with the type of the number of the arm, in the environment where
   the [Switch] is. Where the meet is bottom the arm is dropped: that
   is the first slice. Where it is not, the meet answers an
   environment as well, in which the scrutinee is known to be the
   number of the arm, and Simplify simplifies the arm in that
   environment: the arguments of its [Apply_cont] are replaced by
   their canonical elements there ([simplify_arm]:
   [Simplify_simple.simplify_simples] with [env_at_use]), or, where
   one arm is left, the whole [Apply_cont] is simplified there
   ([Simplify_apply_cont_expr.simplify_apply_cont]), which may do more
   than rewrite its arguments: the arm is then a use of the
   continuation that may be inlined, and the handler may take its
   place on the way up. So [switch x with 0 -> k x | 1 -> k' x]
   becomes [switch x with 0 -> k 0 | 1 -> k' 1].
   [simplify_switch_arms] here is that function as far as an arm
   stays an [Apply_cont]. The meet is a parameter, as in the first
   slice, and so is what is done to the [Apply_cont] of an arm in the
   environment of the arm ([apply_cont_rewrite]), which answers an
   [Apply_cont].

   What is stated.
   - What an arm knows ([arm_env_satisfied_statement]): a state that
     satisfies the environment of the [Switch], in which the scrutinee
     is the number of an arm, satisfies the environment that the meet
     answers for that arm.
   - Of the machine, under a typing environment
     ([switch_arms_refined_statement]): each arm of a [Switch] may be
     replaced by an [Apply_cont] that refines it in the states in
     which the arm is taken ([arm_refines_where_taken]).
   - So the arm that [simplify_arm] answers refines the arm where it
     is taken ([simplify_arm_refines_statement]), the [Switch] with
     the arms that [simplify_arms] answers refines the [Switch]
     ([simplify_arms_refines_statement]), and so does what
     [simplify_switch_arms] answers
     ([simplify_switch_arms_refines_statement]): for a rewrite of
     [Apply_cont]s that is right under every environment
     ([apply_cont_rewrite_sound]). The function of
     FlambdaSimplifyPassSpec.v is such a rewrite
     ([simplify_apply_cont_sound_statement]).
   - With the rewrite that changes nothing the functions are those of
     the first slice ([simplify_arms_identity_statement]).
   - Of the transcribed meet that the contract is proved of
     ([simplify_switch_arms_of_meet_to_prove_refines_statement]).
   - The functions commute with a renaming of variables
     ([simplify_arm_rename_statement], [simplify_arms_rename_statement],
     [simplify_switch_arms_rename_statement]), for a meet and a rewrite
     that do; the rewrite of FlambdaSimplifyPassSpec.v does
     ([simplify_apply_cont_rewrite_equivariant_statement]), and so
     does the transcribed meet that the contract is proved of, with a
     join that does
     ([simplify_switch_arms_of_meet_to_prove_rename_statement]).
   - One term ([example_switch_arms_statement]).

   What is assumed. What the first slice assumes, and its header says
   why: an environment in which a meet may be made
   ([typing_env_fits]), a static environment that is valid, strict
   satisfaction, a scrutinee that may occur in terms. And:
   - Of the meet, [meet_sound meet] (FlambdaMeetSpec.v) where the first
     slice asks [meet_bottom_sound meet]: what is used here is the
     environment that a meet answers, and [meet_sound] is what says
     that a valuation satisfies it. The transcribed meet that the
     contract is proved of has it, with a join that has the contract
     of a join ([meet_to_prove_sound_statement],
     FlambdaMeetClosuresSpec.v).
   - Of the meet, [meet_keeps_terms meet] (FlambdaMeetTermsSpec.v): a
     variable that may occur in terms under the environment that a
     meet answers may occur in terms under the environment that it was
     given. [meet_sound] does not give it: it gives a valuation that
     satisfies the environment answered and agrees with the first on
     the names that the first environment knew; the machine binds the
     variables that may occur in terms under the first environment,
     and nothing there says that it binds a variable that the meet has
     made. (The hypothesis also excludes that a variable which the
     first environment had at another name mode, or out of scope,
     comes to be in scope in terms.) The transcribed meet that the
     contract is proved of has it, with a join that has the contract
     of a join ([meet_to_prove_keeps_terms_statement], in that file),
     so the statement at that meet and the example do not suppose it.
   - Of the rewrite of an [Apply_cont], that it is right under every
     environment, in the states whose static environment is valid
     ([apply_cont_rewrite_sound]). It is asked under the environment
     that the meet answers for the arm, which is why
     [arm_env_satisfied_statement] is what the slice rests on.

   Renaming. As in the first slice, the three statements suppose
   [meet_function_equivariant meet] (FlambdaMeetRenameSpec.v), and
   here also that the rewrite commutes with the renaming of its
   environment and of the [Apply_cont]
   ([apply_cont_rewrite_equivariant]). The function of
   FlambdaSimplifyPassSpec.v does: that is
   [simplify_apply_cont_equivariant_statement] of
   FlambdaSimplifyPassRenamingSpec.v. The transcribed meet that the
   contract is proved of does, with a join that commutes with
   renaming ([meet_to_prove_equivariant_statement],
   FlambdaMeetRenameSpec.v): at that meet the equation of
   [simplify_switch_arms] has a premise on the join where it had one
   on the meet
   ([simplify_switch_arms_of_meet_to_prove_rename_statement]).

   Where the model answers nothing. As in the first slice:
   [simplify_switch_arms] is [None] where the scrutinee has no type or
   no canonical element at the normal name mode, and where a meet
   answers nothing.

   Not here: what Simplify makes of the arm that is alone where that
   is not an [Apply_cont]. A rewrite here takes an [Apply_cont] to an
   [Apply_cont], so the inlining of the continuation of that arm, and
   any other expression that
   [Simplify_apply_cont_expr.simplify_apply_cont] may answer for it,
   is not transcribed: the model has the part of that case in which
   the arm stays an [Apply_cont].

   Nor: what an arm tells its continuation. Simplify records
   each arm as a use of the continuation, with the environment of the
   arm and the types of the arguments there
   ([record_continuation_use]), and the handler is later simplified
   under the join of the environments of its uses. That is the join of
   the environments of a continuation, with the handlers. Nor: the
   arms that are removed on the way up, the merging of arms and the
   rewrites of a boolean [Switch] (the header of the first slice).

   The proofs are in FlambdaSimplifySwitchArmsProofs.v, which needs no
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
  FlambdaMeetTermsSpec FlambdaMeetRenameSpec FlambdaSimplifySwitchSpec.
From OxCamlFlambda2Types Require FlambdaSimplifyPassSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* What an arm knows                                                        *)
(* ======================================================================== *)

(* A state that satisfies the environment of a [Switch], in which the
   scrutinee is the number of an arm, satisfies the environment that
   the meet answers for the arm. *)
Definition arm_env_satisfied_statement : Prop :=
  ∀ (meet : meet_function) (tenv tenv' : Typing_env.t)
    (scrutinee canonical : simple) (scrutinee_ty ty : flambda_type) (i : Z)
    (statics : static_env) (vars : env),
    meet_sound meet ->
    meet_keeps_terms meet ->
    typing_env_fits tenv ->
    Typing_env.mem_simple (Some Name_mode.normal) tenv scrutinee = true ->
    Typing_env.type_simple_in_term_exn tenv (Some Name_mode.normal) scrutinee
    = Some (Some (scrutinee_ty, canonical)) ->
    meet tenv scrutinee_ty (this_naked_immediate i)
    = Some (Or_bottom.Ok (ty, tenv')) ->
    statics_valid statics ->
    env_satisfies_strictly tenv statics vars ->
    eval_simple vars scrutinee = Some (Value_const (Naked_immediate i)) ->
    env_satisfies_strictly tenv' statics vars.

(* ======================================================================== *)
(* Arms replaced by arms that refine them                                   *)
(* ======================================================================== *)

(* An [Apply_cont] refines another in the states in which the arm of a
   [Switch] for a number is taken: those that satisfy the environment
   strictly, whose static environment is valid, and in which the
   scrutinee is the number. *)
Definition arm_refines_where_taken (tenv : Typing_env.t) (scrutinee : simple)
    (i : Z) (ac ac' : apply_cont) : Prop :=
  ∀ (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
    (rd : read_record),
    statics_valid statics ->
    env_satisfies_strictly tenv statics vars ->
    eval_simple vars scrutinee = Some (Value_const (Naked_immediate i)) ->
    refines (State (Apply_cont ac) vars conts statics hp rd)
      (State (Apply_cont ac') vars conts statics hp rd).

(* The arms of a [Switch] may each be replaced by an arm for the same
   number that refines it where it is taken. *)
Definition switch_arms_refined_statement : Prop :=
  ∀ (tenv : Typing_env.t) (scrutinee : simple)
    (arms arms' : list (Z * apply_cont)),
    Forall2
      (fun arm arm' =>
         fst arm' = fst arm
         ∧ arm_refines_where_taken tenv scrutinee (fst arm) (snd arm)
             (snd arm'))
      arms arms' ->
    expr_refines_under_valid tenv (Switch scrutinee arms)
      (Switch scrutinee arms').

(* ======================================================================== *)
(* Simplify's function                                                      *)
(* ======================================================================== *)

(* What is done to the [Apply_cont] of an arm, in the environment of
   the arm. *)
Definition apply_cont_rewrite : Type :=
  Typing_env.t -> apply_cont -> apply_cont.

(* It is right under every environment, in the states whose static
   environment is valid. *)
Definition apply_cont_rewrite_sound (rewrite : apply_cont_rewrite) : Prop :=
  ∀ (tenv : Typing_env.t) (ac : apply_cont),
    expr_refines_under_valid tenv (Apply_cont ac)
      (Apply_cont (rewrite tenv ac)).

(* One arm ([simplify_switch] and [simplify_arm]): dropped where the
   meet of the type of the scrutinee with the type of its number is
   bottom, and otherwise rewritten in the environment that the meet
   answers. [None] where the meet answers nothing. *)
Definition simplify_arm (meet : meet_function) (rewrite : apply_cont_rewrite)
    (tenv : Typing_env.t) (scrutinee_ty : flambda_type) (i : Z)
    (ac : apply_cont) : option (option apply_cont) :=
  match meet tenv scrutinee_ty (this_naked_immediate i) with
  | Some Or_bottom.Bottom => Some None
  | Some (Or_bottom.Ok (_, tenv')) => Some (Some (rewrite tenv' ac))
  | None => None
  end.

(* The arms. *)
Fixpoint simplify_arms (meet : meet_function) (rewrite : apply_cont_rewrite)
    (tenv : Typing_env.t) (scrutinee_ty : flambda_type)
    (arms : list (Z * apply_cont)) : option (list (Z * apply_cont)) :=
  match arms with
  | [] => Some []
  | (i, ac) :: arms =>
    let* arm := simplify_arm meet rewrite tenv scrutinee_ty i ac in
    let* arms := simplify_arms meet rewrite tenv scrutinee_ty arms in
    Some
      match arm with
      | Some ac' => (i, ac') :: arms
      | None => arms
      end
  end.

(* [simplify_switch], as far as the term that it makes from the type of
   the scrutinee and from what the meet finds of each arm. *)
Definition simplify_switch_arms (meet : meet_function)
    (rewrite : apply_cont_rewrite) (tenv : Typing_env.t) (scrutinee : simple)
    (arms : list (Z * apply_cont)) : option expr :=
  let* answer :=
    Typing_env.type_simple_in_term_exn tenv (Some Name_mode.normal) scrutinee
  in
  let* '(scrutinee_ty, canonical) := answer in
  let* arms := simplify_arms meet rewrite tenv scrutinee_ty arms in
  Some (switch_of_arms canonical arms).

(* The arm that is answered refines the arm, where it is taken. *)
Definition simplify_arm_refines_statement : Prop :=
  ∀ (meet : meet_function) (rewrite : apply_cont_rewrite)
    (tenv : Typing_env.t) (scrutinee canonical : simple)
    (scrutinee_ty : flambda_type) (i : Z) (ac ac' : apply_cont),
    meet_sound meet ->
    meet_keeps_terms meet ->
    apply_cont_rewrite_sound rewrite ->
    typing_env_fits tenv ->
    Typing_env.mem_simple (Some Name_mode.normal) tenv scrutinee = true ->
    Typing_env.type_simple_in_term_exn tenv (Some Name_mode.normal) scrutinee
    = Some (Some (scrutinee_ty, canonical)) ->
    simplify_arm meet rewrite tenv scrutinee_ty i ac = Some (Some ac') ->
    arm_refines_where_taken tenv scrutinee i ac ac'.

(* The [Switch] with the arms that are answered refines the [Switch]. *)
Definition simplify_arms_refines_statement : Prop :=
  ∀ (meet : meet_function) (rewrite : apply_cont_rewrite)
    (tenv : Typing_env.t) (scrutinee canonical : simple)
    (scrutinee_ty : flambda_type) (arms arms' : list (Z * apply_cont)),
    meet_sound meet ->
    meet_keeps_terms meet ->
    apply_cont_rewrite_sound rewrite ->
    typing_env_fits tenv ->
    Typing_env.mem_simple (Some Name_mode.normal) tenv scrutinee = true ->
    Typing_env.type_simple_in_term_exn tenv (Some Name_mode.normal) scrutinee
    = Some (Some (scrutinee_ty, canonical)) ->
    simplify_arms meet rewrite tenv scrutinee_ty arms = Some arms' ->
    expr_refines_under_valid tenv (Switch scrutinee arms)
      (Switch scrutinee arms').

(* What Simplify makes of a [Switch] refines it. *)
Definition simplify_switch_arms_refines_statement : Prop :=
  ∀ (meet : meet_function) (rewrite : apply_cont_rewrite)
    (tenv : Typing_env.t) (scrutinee : simple) (arms : list (Z * apply_cont))
    (e : expr),
    meet_sound meet ->
    meet_keeps_terms meet ->
    apply_cont_rewrite_sound rewrite ->
    typing_env_fits tenv ->
    Typing_env.mem_simple (Some Name_mode.normal) tenv scrutinee = true ->
    simplify_switch_arms meet rewrite tenv scrutinee arms = Some e ->
    expr_refines_under_valid tenv (Switch scrutinee arms) e.

(* The rewrite of FlambdaSimplifyPassSpec.v, which replaces the
   arguments of an [Apply_cont] by their canonical elements at the
   normal name mode, is right under every environment. *)
Definition simplify_apply_cont_sound_statement : Prop :=
  apply_cont_rewrite_sound FlambdaSimplifyPassSpec.simplify_apply_cont.

(* With the rewrite that changes nothing, the functions are those of
   the first slice. *)
Definition simplify_arms_identity_statement : Prop :=
  ∀ (meet : meet_function) (tenv : Typing_env.t) (scrutinee : simple)
    (scrutinee_ty : flambda_type) (arms : list (Z * apply_cont)),
    simplify_arms meet (fun _ ac => ac) tenv scrutinee_ty arms
    = prune_arms meet tenv scrutinee_ty arms
    ∧ simplify_switch_arms meet (fun _ ac => ac) tenv scrutinee arms
      = simplify_switch meet tenv scrutinee arms.

(* ======================================================================== *)
(* With the meet that the contract is proved of                             *)
(* ======================================================================== *)

(* The statement for a pass over terms, at the transcribed meet that
   the contract is proved of: the contract of the join is where
   [meet_sound] and [meet_keeps_terms] were. *)
Definition simplify_switch_arms_of_meet_to_prove_refines_statement : Prop :=
  ∀ (join : join_scoped_envs) (fuel : nat) (rewrite : apply_cont_rewrite)
    (tenv : Typing_env.t) (scrutinee : simple) (arms : list (Z * apply_cont))
    (e : expr),
    join_scoped_envs_sound join ->
    apply_cont_rewrite_sound rewrite ->
    typing_env_fits tenv ->
    Typing_env.mem_simple (Some Name_mode.normal) tenv scrutinee = true ->
    simplify_switch_arms (meet_to_prove join fuel) rewrite tenv scrutinee arms
    = Some e ->
    expr_refines_under_valid tenv (Switch scrutinee arms) e.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

(* A rewrite commutes with a renaming of the variables. *)
Definition apply_cont_rewrite_equivariant (rewrite : apply_cont_rewrite)
    : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (ac : apply_cont),
    rewrite (rename_typing_env r tenv) (rename_apply_cont r ac)
    = rename_apply_cont r (rewrite tenv ac).

Definition simplify_arm_rename_statement : Prop :=
  ∀ (meet : meet_function) (rewrite : apply_cont_rewrite) (r : renaming)
    (tenv : Typing_env.t) (scrutinee_ty : flambda_type) (i : Z)
    (ac : apply_cont),
    meet_function_equivariant meet ->
    apply_cont_rewrite_equivariant rewrite ->
    simplify_arm meet rewrite (rename_typing_env r tenv)
      (rename_type r scrutinee_ty) i (rename_apply_cont r ac)
    = option_map (option_map (rename_apply_cont r))
        (simplify_arm meet rewrite tenv scrutinee_ty i ac).

Definition simplify_arms_rename_statement : Prop :=
  ∀ (meet : meet_function) (rewrite : apply_cont_rewrite) (r : renaming)
    (tenv : Typing_env.t) (scrutinee_ty : flambda_type)
    (arms : list (Z * apply_cont)),
    meet_function_equivariant meet ->
    apply_cont_rewrite_equivariant rewrite ->
    simplify_arms meet rewrite (rename_typing_env r tenv)
      (rename_type r scrutinee_ty) (rename_arms r arms)
    = option_map (rename_arms r)
        (simplify_arms meet rewrite tenv scrutinee_ty arms).

Definition simplify_switch_arms_rename_statement : Prop :=
  ∀ (meet : meet_function) (rewrite : apply_cont_rewrite) (r : renaming)
    (tenv : Typing_env.t) (scrutinee : simple) (arms : list (Z * apply_cont)),
    meet_function_equivariant meet ->
    apply_cont_rewrite_equivariant rewrite ->
    simplify_switch_arms meet rewrite (rename_typing_env r tenv)
      (rename_simple r scrutinee) (rename_arms r arms)
    = option_map (rename_expr r)
        (simplify_switch_arms meet rewrite tenv scrutinee arms).

(* At the transcribed meet that the contract is proved of, with a join
   that commutes with renaming
   ([meet_to_prove_equivariant_statement], FlambdaMeetRenameSpec.v). *)
Definition simplify_switch_arms_of_meet_to_prove_rename_statement : Prop :=
  ∀ (join : join_scoped_envs) (fuel : nat) (rewrite : apply_cont_rewrite)
    (r : renaming) (tenv : Typing_env.t) (scrutinee : simple)
    (arms : list (Z * apply_cont)),
    join_scoped_envs_equivariant join ->
    apply_cont_rewrite_equivariant rewrite ->
    simplify_switch_arms (meet_to_prove join fuel) rewrite
      (rename_typing_env r tenv) (rename_simple r scrutinee)
      (rename_arms r arms)
    = option_map (rename_expr r)
        (simplify_switch_arms (meet_to_prove join fuel) rewrite tenv scrutinee
           arms).

(* The rewrite of FlambdaSimplifyPassSpec.v commutes with a renaming
   ([simplify_apply_cont_equivariant_statement],
   FlambdaSimplifyPassRenamingSpec.v). *)
Definition simplify_apply_cont_rewrite_equivariant_statement : Prop :=
  apply_cont_rewrite_equivariant FlambdaSimplifyPassSpec.simplify_apply_cont.

(* ======================================================================== *)
(* An example                                                               *)
(* ======================================================================== *)

(* [let i = (x < 7) in switch i with 0 -> k0 i | 1 -> k1 i | 2 -> k2 i],
   the term of the second example of the first slice with the
   scrutinee passed to the continuations, in the same environment and
   with the same meet ([example_meet], FlambdaSimplifySwitchSpec.v),
   and with the rewrite of FlambdaSimplifyPassSpec.v. The arm for 2 is
   dropped. The meet of the type of [i] with 0 answers an environment
   in which [i] is 0, and there the canonical element of the argument
   [i] is the constant: the arm for 0 passes 0, and the arm for 1
   passes 1. *)
Definition example_arm (k : continuation) (arg : simple) : apply_cont :=
  Apply_cont_expr k [arg] None.

Definition example_switch_arms_statement : Prop :=
  ∀ k0 k1 k2 : continuation,
    expr_refines_under_valid
      (Typing_env.add_variable_definition Typing_env.create example_x Value
         Name_mode.normal)
      (example_switch_comparison
         [(0, example_arm k0 (Var example_i));
          (1, example_arm k1 (Var example_i));
          (2, example_arm k2 (Var example_i))])
      (example_switch_comparison
         [(0, example_arm k0 (Const (Naked_immediate 0)));
          (1, example_arm k1 (Const (Naked_immediate 1)))]).
