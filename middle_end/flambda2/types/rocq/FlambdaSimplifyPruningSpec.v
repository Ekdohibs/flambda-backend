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


(* The function that rewrites a term from the types, with the arms of a
   [Switch] removed that the type of the scrutinee excludes: a second
   function, whose theorem is about the states of the run of a unit
   whose sets of closures are well formed.

   This file is to be reviewed: every definition and every statement.
   The function is meant to be extracted and run by the validation of
   middle_end/flambda2/tests/semantics/; nothing runs it yet.

   Why a second function. [simplify_by_types]
   (FlambdaSimplifyPassSpec.v) refines the term in every state, and
   removes no arm of a [Switch] by a meet: where the scrutinee has a
   constant for canonical element it takes the arm of the constant,
   and that is all. The law by which Simplify removes the
   arms that the type of the scrutinee excludes
   (FlambdaSimplifySwitchSpec.v) rests on the meet, whose contract
   supposes that the static environment is valid
   (../../rocq/FlambdaClosuresSpec.statics_valid): it holds in the
   states that have a valid static environment and not in every state.
   A step keeps no such thing by itself: a [Let_static] of a term that
   is not well formed can bind what is not valid. What a step keeps is
   [run_valid]: every value that the state holds is valid
   ([state_valid], ../../rocq/FlambdaClosuresSpec.v), and the sets of
   closures of every term that it holds are well formed (its
   expression, the handlers that its continuations keep, the code that
   is bound: [sets_inv], the same file). The state that a unit starts
   in has it as soon as the sets of the unit are well formed
   (../../rocq/FlambdaWellFormedSpec.sets_well_formed, a test, and one
   of the conditions of a well-formed unit).

   What is stated.
   - Of [run_valid]: a step keeps it ([run_valid_kept_statement]); the
     state that a unit with well-formed sets starts in has it
     ([run_valid_initial_statement]); its static environment is valid
     ([run_valid_statics_statement]); and it reads the expression of
     the state through the test of its sets only
     ([run_valid_expr_statement]).
   - Refinement under a typing environment from the states that have
     [run_valid] ([expr_refines_from_valid]): it follows from
     refinement in every state that satisfies the environment, and
     from refinement in those whose static environment is valid
     ([from_valid_weaken_statement]); two steps make one where the
     term between has well-formed sets
     ([expr_refines_from_valid_trans_statement]); from the environment
     that knows nothing it is refinement of the two terms run on their
     own, if the sets of the first are well formed
     ([from_valid_program_statement]).
   - The function: [simplify_expr_pruning prune tenv fit e] is
     [simplify_expr tenv e] (FlambdaSimplifyPassSpec.v), with one more
     argument, [prune], a function that may rewrite a [Switch] from
     the environment, and a boolean, [fit], that says whether meets
     may be made in the environment (below). What [prune] is asked is
     [switch_rewrite_sound]; the function of
     FlambdaSimplifySwitchSpec.v that removes arms has it, for a meet
     that has [meet_bottom_sound]
     ([simplify_switch_rewrite_sound_statement]). What the function
     answers has well-formed sets if the term has
     ([simplify_expr_pruning_sets_statement]) and refines the term
     from the states that have [run_valid]
     ([simplify_expr_pruning_refines_statement]); from the environment
     that knows nothing, a unit with well-formed sets refines what the
     function makes of it
     ([simplify_by_types_pruning_refines_statement]). With a [prune]
     that answers nothing it is the first function
     ([simplify_expr_no_pruning_statement]).
   - The step by which the function, as the first, rewrites the
     handler of a continuation under the environment of its one
     application: it is right from the states that have [run_valid]
     for a rewrite of the handler that is right from them
     ([find_use_refines_from_valid_statement]; the statement for every
     state is FlambdaSimplifyPassSpec.find_use_refines_statement).
   - Of the applications that FlambdaSimplifyPassSpec.gather_uses
     gathers: a test ([gather_uses_keeps_fit]) under which each of
     their environments has the two conditions
     ([gather_uses_known_to_fit_statement]), and is satisfied only by
     valuations that satisfy the environment of the body
     ([gather_uses_weaken_statement]). The second is for the day a
     join is called on them. And
     FlambdaSimplifyPassSpec.gather_uses_refines_statement from the
     states that have [run_valid], for a rewrite of the handler that
     is right from them ([gather_uses_refines_from_valid_statement]).
   - The step by which the function, as the first, rewrites the
     handler under the environment of the one application gathered
     where [find_use] finds none
     (FlambdaSimplifyPassSpec.find_one_use): the test of the two
     conditions for the environment that it answers
     ([find_one_use_keeps_fit], [find_one_use_known_to_fit_statement]).
   - With the transcribed meet that the contract is proved of
     ([meet_to_prove], FlambdaMeetSpec.v):
     [simplify_by_types_pruning_with join fuel] refines a unit with
     well-formed sets for a join that has the contract of a join,
     whatever the fuel
     ([simplify_by_types_pruning_with_refines_statement]); and
     [prune_by_types fuel], which is that function at the join that
     forgets, applied where the sets of the unit pass their test and
     the identity elsewhere, has no premise at all
     ([prune_by_types_refines_statement]). That one is for the
     validation to run.
   - Two terms on which it is computed, with the transcribed meet, and
     what the first function answers on them
     ([example_pruning_no_arm_statement],
     [example_pruning_comparison_statement]).

   Where a meet may be made. The law of the pruning asks
   [typing_env_fits] of the environment (FlambdaMeetSpec.v). Two of
   its conditions have no test: [levels_agree] (in
   [typing_env_invariant]) and [aliases_kinded]. The function carries
   a boolean for them, [fit]: it is true where the two are known of
   the environment ([known_to_fit]). It is true of the environment
   that knows nothing; the environment of the body of code has the
   two conditions if the environment after the group that binds the
   code has them ([code_env_in_known_to_fit_statement]), so the
   boolean in the code is the one after the group; the same of the
   environment of a handler and the environment of its [Let_cont]
   ([handler_env_known_to_fit_statement]); where a handler is
   rewritten under the environment of the one application of its
   continuation, it stays true if each [Let] passed on the way and
   each alias added there has its test ([find_use_keeps_fit],
   [find_use_known_to_fit_statement]); and at a [Let], and at a
   [Let_static] for the types that its symbols are given, it stays
   true if the premises of the statements by which the function that
   extends the environment keeps the two conditions pass as tests
   ([let_keeps_fit], [let_keeps_fit_sound_statement];
   [statics_keep_fit], [statics_keep_fit_sound_statement]). At a
   [Switch] the rest of
   [typing_env_fits] is tested ([types_fit_b],
   [types_fit_b_fits_statement]), and that the scrutinee may occur in
   terms. Where [fit] is false or a test fails nothing is pruned: a
   failure costs rewrites and not correctness. None of these tests is
   Simplify's (the CR at [known_to_fit]).

   What is assumed. The theorem from the environment that knows
   nothing ([simplify_by_types_pruning_refines_statement]) has two
   premises: [switch_rewrite_sound prune], which for the pruning of
   FlambdaSimplifySwitchSpec.v is [meet_bottom_sound meet] (that file
   says what is and is not known of it for the transcribed meet); and
   that the sets of the term are well formed, which is a test. The two
   statements about any environment
   ([simplify_expr_pruning_sets_statement],
   [simplify_expr_pruning_refines_statement]) ask [known_to_fit tenv
   fit] as well: whoever calls the function with [fit] true owes the
   two conditions of the environment.

   Not here. The arms that are kept are not rewritten with what the
   meet finds of them, and the environment that the meet answers for
   an arm is not kept. In the handler of a continuation what is
   known is what the first function knows: the environment of the one
   application, where there is one that it finds, and otherwise
   nothing of a parameter but its kind. That it commutes with
   renaming is in FlambdaSimplifyPruningRenamingSpec.v. That it keeps
   scope (../../rocq/FlambdaValidatorEquivarianceSpec.v) is not
   stated: the validation can have it by the test of
   ../../rocq/FlambdaGuardScopeSpec.v.

   The proofs are in FlambdaSimplifyPruningProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaValueSpec FlambdaPrimitiveModel FlambdaSemanticsSpec
  FlambdaWellFormedSpec FlambdaClosuresSpec FlambdaContextWhereSpec
  FlambdaUnusedStaticsSpec FlambdaNormaliseSpec FlambdaUnusedContsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypeWellFormedSpec FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaEnvAliasesInvariantSpec
  FlambdaTypingEnv FlambdaTypingEnvSpec FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvKindedSpec FlambdaTypingEnvExamplesSpec
  FlambdaTypingEnvRefinementSpec FlambdaMeet FlambdaMeetSpec
  FlambdaSimplifyPrimitivesSpec FlambdaSimplifyApplySpec
  FlambdaSimplifyStaticsSpec FlambdaSimplifyCodeEnvSpec
  FlambdaSimplifyHandlersSpec FlambdaSimplifyPassSpec.
From OxCamlFlambda2Types Require FlambdaMeetExamplesSpec
  FlambdaSimplifySwitchSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* The states of the run of a unit whose sets are well formed               *)
(* ======================================================================== *)

(* What holds of every state of the run of a unit whose sets are well
   formed: the sets of every term that the state holds are well
   formed and the static sets are bound ([sets_inv],
   ../../rocq/FlambdaClosuresSpec.v), and every value that it holds is
   valid ([state_valid], the same file). *)
Definition run_valid (s : state) : Prop := sets_inv s ∧ state_valid s.

(* A step keeps it. *)
Definition run_valid_kept_statement : Prop := kept_by_step run_valid.

(* The state that a unit starts in has it, if the sets of the unit are
   well formed. *)
Definition run_valid_initial_statement : Prop :=
  ∀ e : expr,
    sets_well_formed e = true ->
    run_valid (State e [] [] empty_static_env empty_heap no_reads).

(* Its static environment is valid. *)
Definition run_valid_statics_statement : Prop :=
  ∀ s : state, run_valid s -> statics_valid (state_statics s).

(* It reads the expression through the test of its sets only: a state
   that has it has it with any expression whose sets are well formed. *)
Definition run_valid_expr_statement : Prop :=
  ∀ (e e' : expr) (vars : env) (conts : cont_env) (statics : static_env)
    (hp : heap) (rd : read_record),
    sets_well_formed e' = true ->
    run_valid (State e vars conts statics hp rd) ->
    run_valid (State e' vars conts statics hp rd).

(* ======================================================================== *)
(* Refinement from those states                                             *)
(* ======================================================================== *)

(* [expr_refines_under] (FlambdaTypingEnvRefinementSpec.v), from the
   states that have [run_valid]. The premise is on the state of the
   first expression. *)
Definition expr_refines_from_valid (tenv : Typing_env.t)
    (source target : expr) : Prop :=
  ∀ (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
    (rd : read_record),
    run_valid (State source vars conts statics hp rd) ->
    env_satisfies_strictly tenv statics vars ->
    refines (State source vars conts statics hp rd)
      (State target vars conts statics hp rd).

(* What holds in every state that satisfies the environment, and what
   holds in those whose static environment is valid, holds from the
   states that have [run_valid]. *)
Definition from_valid_weaken_statement : Prop :=
  ∀ (tenv : Typing_env.t) (e e' : expr),
    (expr_refines_under tenv e e' -> expr_refines_from_valid tenv e e')
    ∧ (FlambdaSimplifySwitchSpec.expr_refines_under_valid tenv e e' ->
       expr_refines_from_valid tenv e e').

(* Two steps make one, where the sets of the term between are well
   formed. *)
Definition expr_refines_from_valid_trans_statement : Prop :=
  ∀ (tenv : Typing_env.t) (e e' e'' : expr),
    sets_well_formed e' = true ->
    expr_refines_from_valid tenv e e' ->
    expr_refines_from_valid tenv e' e'' ->
    expr_refines_from_valid tenv e e''.

(* From the environment that knows nothing, a term whose sets are well
   formed refines the other, the two run on their own. *)
Definition from_valid_program_statement : Prop :=
  ∀ e e' : expr,
    sets_well_formed e = true ->
    expr_refines_from_valid Typing_env.create e e' ->
    program_refines e e'.

(* ======================================================================== *)
(* A function that rewrites a [Switch]                                      *)
(* ======================================================================== *)

(* A function that may rewrite a [Switch] from a typing environment,
   given the scrutinee and the arms. [None]: it does not. *)
Definition switch_rewrite : Type :=
  Typing_env.t -> simple -> list (Z * apply_cont) -> option expr.

(* What is asked of one: the sets of what it answers are well formed;
   and what it answers refines the [Switch] in the states that satisfy
   the environment and have a valid static environment, where meets
   may be made in the environment and the scrutinee may occur in
   terms. *)
Definition switch_rewrite_sound (prune : switch_rewrite) : Prop :=
  ∀ (tenv : Typing_env.t) (scrutinee : simple)
    (arms : list (Z * apply_cont)) (e : expr),
    prune tenv scrutinee arms = Some e ->
    sets_well_formed e = true
    ∧ (typing_env_fits tenv ->
       Typing_env.mem_simple (Some Name_mode.normal) tenv scrutinee = true ->
       FlambdaSimplifySwitchSpec.expr_refines_under_valid tenv
         (Switch scrutinee arms) e).

(* The function that removes the arms which a meet excludes has it. *)
Definition simplify_switch_rewrite_sound_statement : Prop :=
  ∀ meet : meet_function,
    meet_bottom_sound meet ->
    switch_rewrite_sound (FlambdaSimplifySwitchSpec.simplify_switch meet).

(* The function that rewrites no [Switch]. *)
Definition no_pruning : switch_rewrite := fun _ _ _ => None.

(* ======================================================================== *)
(* Where a meet may be made                                                 *)
(* ======================================================================== *)

(* What the boolean that the function carries stands for: where it is
   true, the environment has the two conditions of [typing_env_fits]
   that have no test. *)
(* CR mshinwell: the boolean and the tests of this section are not
   Simplify's, as the tests of the first function are not. They are
   the premises of the statements by which the pruning is right,
   tested so that the theorem of the function needs none. That they
   pass on the environments that the function builds is not stated;
   the example at the end is one where they do. The other choice is
   to prove [typing_env_fits] of every environment that the function
   builds, which needs each type that a rewrite gives a variable to
   be shown to fit. *)
Definition known_to_fit (tenv : Typing_env.t) (fit : bool) : Prop :=
  fit = true -> typing_env_invariant tenv ∧ aliases_kinded tenv.

(* A test of a type that the environment has for a name: it is well
   formed; each of its aliases, and each name that has an equation
   inside it, is of the kind at which the type has it; it is not
   obviously bottom; and the type of a symbol is of the kind of
   values. *)
Definition type_fits_b (tenv : Typing_env.t) (n : name)
    (entry : flambda_type * Binding_time.With_name_mode.t) : bool :=
  type_well_formed (fst entry)
  && forallb (fun alias => simple_of_kind_b tenv (fst alias) (snd alias))
       (aliases_with_kinds (fst entry))
  && forallb
       (fun equation =>
          simple_of_kind_b tenv (simple_of_name (fst equation))
            (snd equation))
       (equation_names_with_kinds (fst entry))
  && negb (is_obviously_bottom (fst entry))
  && match n with
     | Name_var _ => true
     | Name_symbol _ => kind_eqb (kind (fst entry)) Value
     end.

Definition types_fit_b (tenv : Typing_env.t) : bool :=
  Keyed.for_all_bindings_b name_eqb (type_fits_b tenv)
    (Typing_env.names_to_types tenv).

(* With the two conditions that have no test, the test is enough for
   meets to be made in the environment. *)
Definition types_fit_b_fits_statement : Prop :=
  ∀ tenv : Typing_env.t,
    typing_env_invariant tenv ->
    aliases_kinded tenv ->
    types_fit_b tenv = true ->
    typing_env_fits tenv.

(* The premises of [add_alias_invariant_statement]
   (FlambdaTypingEnvInvariantSpec.v) and of
   [add_alias_aliases_kinded_statement] (FlambdaTypingEnvKindedSpec.v),
   as a test: the two [Simple]s are canonical, bound, and of the kind
   given. *)
Definition alias_keeps_fit (tenv : Typing_env.t) (e1 e2 : simple)
    (kind : flambda_kind) : bool :=
  is_canonical_b (Typing_env.aliases tenv) e1
  && is_canonical_b (Typing_env.aliases tenv) e2
  && Typing_env.mem_simple None tenv e1
  && Typing_env.mem_simple None tenv e2
  && simple_of_kind_b tenv e1 kind
  && simple_of_kind_b tenv e2 kind.

(* The premises of [replace_equation_invariant_statement] and of
   [replace_equation_aliases_kinded_statement], as a test: the name is
   bound; the variables of the type are bound; the type is of the kind
   of the one that the name had, which is the kind of the name; if the
   type is an alias the name is not canonical; and either the type is
   not obviously unknown or no level before the current one has an
   equation on the name. *)
Definition equation_keeps_fit (tenv : Typing_env.t) (n : name)
    (ty : flambda_type) : bool :=
  Typing_env.mem None tenv n
  && forallb (fun y => Typing_env.mem None tenv (Name_var y))
       (free_variables ty)
  && match Keyed.lookup name_eqb n (Typing_env.names_to_types tenv) with
     | Some entry => kind_eqb (kind ty) (kind (fst entry))
     | None => true
     end
  && match get_alias_opt ty with
     | Some _ =>
       negb (is_canonical_b (Typing_env.aliases tenv) (simple_of_name n))
     | None => true
     end
  && (negb (is_obviously_unknown ty)
      || negb
           (is_some
              (Keyed.lookup name_eqb n
                 (flat_map Typing_env_level.equations
                    (map Typing_env.One_level.level
                       (Typing_env.prev_levels tenv))))))
  && simple_of_kind_b tenv (simple_of_name n) (kind ty).

(* Whether the environment that [simplify_let] answers is known to keep
   the two conditions, by the cases of [simplify_let]: a variable that
   is defined keeps them; one that is made an alias, or given a type,
   keeps them if the test above passes in the environment where the
   variable has been defined. *)
Definition let_keeps_fit (tenv : Typing_env.t) (x : variable)
    (defining_expr : named) : bool :=
  match simplify_named tenv defining_expr with
  | Becomes_simple s =>
    match simplify_simple tenv s with
    | Some (ty, canonical) =>
      alias_keeps_fit
        (Typing_env.add_variable_definition tenv x (kind ty)
           Name_mode.normal)
        (Var x) canonical (kind ty)
    | None => false
    end
  | Stays_with_type ty =>
    equation_keeps_fit
      (Typing_env.add_variable_definition tenv x (kind ty) Name_mode.normal)
      (Name_var x) ty
  | Stays_with_kind _ => true
  | Becomes_invalid | Not_known => false
  end.

Definition let_keeps_fit_sound_statement : Prop :=
  ∀ (tenv tenv' : Typing_env.t) (fit : bool) (x : variable)
    (defining_expr defining_expr' : named),
    known_to_fit tenv fit ->
    simplify_let tenv x defining_expr = Step_let defining_expr' tenv' ->
    known_to_fit tenv' (fit && let_keeps_fit tenv x defining_expr).

(* The same for the symbols of a [Let_static], by the cases of
   [add_static_bindings] (FlambdaSimplifyStaticsSpec.v): a symbol that
   is defined keeps the two conditions; one that is given the type of
   its constant keeps them if the test of [equation_keeps_fit] passes
   in the environment where the symbol has been defined. *)
Fixpoint block_like_bindings_keep_fit (tenv : Typing_env.t)
    (bindings : list static_binding) : bool :=
  match bindings with
  | [] => true
  | Bind_symbol s const :: bindings =>
    if is_closure_const const
    then block_like_bindings_keep_fit tenv bindings
    else
      match static_const_type_in tenv const with
      | Some ty =>
        equation_keeps_fit (Typing_env.add_symbol_definition tenv s)
          (Name_symbol s) ty
        && block_like_bindings_keep_fit (define_symbol_with_type tenv s ty)
             bindings
      | None =>
        block_like_bindings_keep_fit
          (Typing_env.add_symbol_definition tenv s) bindings
      end
  | (Bind_code _ _ | Bind_deleted_code _) :: bindings =>
    block_like_bindings_keep_fit tenv bindings
  end.

(* The same for the last pass of [add_static_bindings], which gives
   the symbols bound to closures their types: each symbol is defined
   already, and the test is made in the environment as it is when the
   symbol is given its type. *)
Fixpoint closure_bindings_keep_fit (prior tenv : Typing_env.t)
    (bindings : list static_binding) : bool :=
  match bindings with
  | [] => true
  | Bind_symbol s const :: bindings =>
    if is_closure_const const
    then
      match static_const_type_in prior const with
      | Some ty =>
        equation_keeps_fit tenv (Name_symbol s) ty
        && closure_bindings_keep_fit prior
             (Typing_env.replace_equation tenv (Name_symbol s) ty) bindings
      | None => closure_bindings_keep_fit prior tenv bindings
      end
    else closure_bindings_keep_fit prior tenv bindings
  | (Bind_code _ _ | Bind_deleted_code _) :: bindings =>
    closure_bindings_keep_fit prior tenv bindings
  end.

Definition statics_keep_fit (tenv : Typing_env.t)
    (bindings : list static_binding) : bool :=
  let defined :=
    Typing_env.add_symbol_definitions tenv (closure_symbols_of bindings)
  in
  let prior := add_block_like_bindings defined bindings in
  block_like_bindings_keep_fit defined bindings
  && closure_bindings_keep_fit prior prior bindings.

Definition statics_keep_fit_sound_statement : Prop :=
  ∀ (tenv : Typing_env.t) (fit : bool) (bindings : list static_binding),
    known_to_fit tenv fit ->
    known_to_fit (add_static_bindings tenv bindings)
      (fit && statics_keep_fit tenv bindings).

(* The environment of the body of code
   (FlambdaSimplifyCodeEnvSpec.code_env_in) has the two conditions if
   the environment that it is made from has them. *)
Definition code_env_in_known_to_fit_statement : Prop :=
  ∀ (tenv : Typing_env.t) (fit : bool) (params : list variable)
    (my_closure : variable) (metadata : code_metadata)
    (tenv_code : Typing_env.t),
    known_to_fit tenv fit ->
    code_env_in tenv params my_closure metadata = Some tenv_code ->
    known_to_fit tenv_code fit.

(* The same of the environment of a handler
   (FlambdaSimplifyHandlersSpec.handler_env). *)
Definition handler_env_known_to_fit_statement : Prop :=
  ∀ (tenv : Typing_env.t) (fit : bool) (params : list variable)
    (metadata : handler_metadata) (tenv_handler : Typing_env.t),
    known_to_fit tenv fit ->
    handler_env tenv params metadata = Some tenv_handler ->
    known_to_fit tenv_handler fit.

(* ======================================================================== *)
(* Where a continuation is applied                                          *)
(* ======================================================================== *)

(* Whether the environments that the functions of the section "Where a
   continuation is applied" of FlambdaSimplifyPassSpec.v answer are
   known to keep the two conditions: each alias that they add passes
   the test of [alias_keeps_fit] in the environment where it is
   added, and each [Let] on the way to the application passes the
   test of [let_keeps_fit]. The tests are sufficient, and no more is
   claimed of them: each follows the cases of its function in which
   an environment is answered, and where the function answers none
   (the lengths of the parameters and the arguments differ, a binder
   on the way is free in the handler) a test may still look at what
   the function did not, and answer false. A false answer costs
   pruning under the handler and nothing else. *)

(* [arm_env]. *)
Definition arm_env_keeps_fit (tenv : Typing_env.t) (scrutinee : simple)
    (d : Z) : bool :=
  match simplify_simple tenv scrutinee with
  | Some (_, Var c) =>
    if simple_in_terms_b tenv (Var c)
       && simple_of_kind_b tenv (Var c) naked_immediate_kind
    then
      match Typing_env.add_alias tenv (Var c) (Const (Naked_immediate d)) with
      | Some (Ok _) =>
        alias_keeps_fit tenv (Var c) (Const (Naked_immediate d))
          naked_immediate_kind
      | Some Unknown | Some Bottom | None => true
      end
    else true
  | Some (_, (Const _ | Symbol _)) | None => true
  end.

(* [bind_params_to_args]. *)
Fixpoint bind_params_to_args_keep_fit (aliased : simple -> bool)
    (tenv : Typing_env.t)
    (bindings : list (variable * flambda_kind * simple)) : bool :=
  match bindings with
  | [] => true
  | (x, kind, canonical) :: bindings =>
    if
      typing_env_well_formed tenv
      && negb (Typing_env.mem None tenv (Name_var x))
    then
      let defined :=
        Typing_env.add_variable_definition tenv x kind Name_mode.normal
      in
      if
        aliased canonical && simple_in_terms_b tenv canonical
        && simple_of_kind_b tenv canonical kind
      then
        match Typing_env.add_alias defined (Var x) canonical with
        | Some (Ok result) =>
          alias_keeps_fit defined (Var x) canonical kind
          && bind_params_to_args_keep_fit aliased
               (Typing_env.add_alias_result_t result) bindings
        | Some Unknown | Some Bottom | None =>
          bind_params_to_args_keep_fit aliased defined bindings
        end
      else bind_params_to_args_keep_fit aliased defined bindings
    else true
  end.

(* [use_env]. *)
Definition use_env_keeps_fit (scope tenv : Typing_env.t)
    (params : list variable) (metadata : handler_metadata)
    (args : list simple) : bool :=
  match simplify_simples tenv args with
  | Some answers =>
    bind_params_to_args_keep_fit
      (fun canonical =>
         simple_in_terms_b scope canonical && not_among params canonical)
      tenv
      (combine (combine params (hm_params_kinds metadata)) (map snd answers))
  | None => true
  end.

(* [leaf_uses]: the environment of each arm that applies [k]. *)
Definition leaf_uses_keep_fit (tenv : Typing_env.t) (k : continuation)
    (e : expr) : bool :=
  match e with
  | Switch scrutinee arms =>
    forallb
      (fun arm =>
         negb (Pos.eqb (ac_continuation (snd arm)) k)
         || arm_env_keeps_fit tenv scrutinee (fst arm))
      arms
  | Let _ _ _ _ | Let_static _ _ | Let_cont_non_recursive _ _ _
  | Let_cont_recursive _ _ | Apply _ | Apply_cont _ | Invalid =>
    true
  end.

(* [let_env]. *)
Definition let_env_keeps_fit (tenv : Typing_env.t) (x : variable)
    (declared : flambda_kind) (defining_expr : named) : bool :=
  let resolved := resolve_value_slot tenv declared defining_expr in
  match simplify_let tenv x resolved with
  | Step_let _ _ => let_keeps_fit tenv x resolved
  | Step_invalid | Step_stop => true
  end.

(* [find_use]. *)
Fixpoint find_use_keeps_fit (scope tenv : Typing_env.t) (k : continuation)
    (h : continuation_handler) (body : expr) : bool :=
  match body with
  | Let x declared defining_expr rest =>
    match let_env tenv x declared defining_expr with
    | Some tenv' =>
      let_env_keeps_fit tenv x declared defining_expr
      && find_use_keeps_fit scope tenv' k h rest
    | None => true
    end
  | Let_cont_non_recursive _ _ rest => find_use_keeps_fit scope tenv k h rest
  | Apply_cont _ | Switch _ _ =>
    match h, leaf_uses tenv k body with
    | Handler params metadata _, Some [(tenv_use, args)] =>
      leaf_uses_keep_fit tenv k body
      && use_env_keeps_fit scope tenv_use params metadata args
    | Handler _ _ _, (Some ([] | _ :: _ :: _) | None) => true
    end
  | Let_static _ _ | Let_cont_recursive _ _ | Apply _ | Invalid => true
  end.

(* The environment that [find_use] answers has the two conditions if
   the environment of the body has them and the test passes. *)
Definition find_use_known_to_fit_statement : Prop :=
  ∀ (scope tenv : Typing_env.t) (fit : bool) (k : continuation)
    (h : continuation_handler) (body : expr) (target : Typing_env.t)
    (xs : list variable) (ks : list continuation),
    known_to_fit tenv fit ->
    find_use scope tenv k h body = Some (target, xs, ks) ->
    known_to_fit target (fit && find_use_keeps_fit scope tenv k h body).

(* FlambdaSimplifyPassSpec.find_use_refines_statement, from the states
   that have [run_valid]: the handler may be rewritten under the
   environment that [find_use] answers, by a rewrite that is right
   from those states. *)
Definition find_use_refines_from_valid_statement : Prop :=
  ∀ (scope tenv : Typing_env.t) (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler handler' body : expr)
    (target : Typing_env.t) (xs : list variable) (ks : list continuation),
    find_use scope tenv k (Handler params metadata handler) body
    = Some (target, xs, ks) ->
    binders_not_free xs ks (Handler params metadata handler') = true ->
    sets_well_formed
      (Let_cont_non_recursive k (Handler params metadata handler) body)
    = true ->
    expr_refines_from_valid target handler handler' ->
    expr_refines_from_valid tenv
      (Let_cont_non_recursive k (Handler params metadata handler) body)
      (Let_cont_non_recursive k (Handler params metadata handler') body).

(* [gather_uses]: every [Let] passed, in the body and in the handlers
   gone into, and every arm at the end of them. *)
Fixpoint gather_uses_keeps_fit (tenv : Typing_env.t) (k : continuation)
    (h : continuation_handler) (body : expr) : bool :=
  match body with
  | Let x declared defining_expr rest =>
    match let_env tenv x declared defining_expr with
    | Some tenv' =>
      let_env_keeps_fit tenv x declared defining_expr
      && gather_uses_keeps_fit tenv' k h rest
    | None => true
    end
  | Let_cont_non_recursive _ h' rest =>
    gather_uses_keeps_fit tenv k h rest
    && (if cont_free_in_handler_b k h'
        then
          match h' with
          | Handler params' metadata' body' =>
            match handler_env tenv params' metadata' with
            | Some tenv_handler => gather_uses_keeps_fit tenv_handler k h body'
            | None => true
            end
          end
        else true)
  | Apply_cont _ | Switch _ _ => leaf_uses_keep_fit tenv k body
  | Let_static _ _ | Let_cont_recursive _ _ | Apply _ | Invalid => true
  end.

(* The environments gathered have the two conditions if the
   environment of the body has them and the test passes. *)
Definition gather_uses_known_to_fit_statement : Prop :=
  ∀ (tenv : Typing_env.t) (fit : bool) (k : continuation)
    (h : continuation_handler) (body : expr)
    (uses : list (Typing_env.t * list simple)) (xs : list variable)
    (ks : list continuation) (use : Typing_env.t * list simple),
    known_to_fit tenv fit ->
    gather_uses tenv k h body = Some (uses, xs, ks) ->
    use ∈ uses ->
    known_to_fit (fst use) (fit && gather_uses_keeps_fit tenv k h body).

(* And each says no less than the environment of the body: a
   valuation that satisfies it satisfies that one. The environments
   on the way are made from that of the body by defining variables,
   by giving a type to a variable just defined and by making two
   canonical elements aliases, and each of the three can be undone
   where the tests pass. So one valuation serves the environment of
   an application and an environment made from that of the body, as
   the contract of a join asks of a use and the source. *)
Definition gather_uses_weaken_statement : Prop :=
  ∀ (tenv : Typing_env.t) (k : continuation) (h : continuation_handler)
    (body : expr) (uses : list (Typing_env.t * list simple))
    (xs : list variable) (ks : list continuation)
    (use : Typing_env.t * list simple) (statics : static_env)
    (valuation : env),
    known_to_fit tenv true ->
    gather_uses tenv k h body = Some (uses, xs, ks) ->
    gather_uses_keeps_fit tenv k h body = true ->
    use ∈ uses ->
    typing_env_satisfied_strictly statics valuation (fst use) ->
    typing_env_satisfied_strictly statics valuation tenv.

(* FlambdaSimplifyPassSpec.gather_uses_refines_statement, from the
   states that have [run_valid]: the handler may be rewritten under an
   environment that every application gathered admits, by a rewrite
   that is right from those states. The sets of what the handler
   becomes are to be well formed: where the continuation is applied
   in the handler of another, the handler as rewritten is run there. *)
Definition gather_uses_refines_from_valid_statement : Prop :=
  ∀ (tenv : Typing_env.t) (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler handler' body : expr)
    (target : Typing_env.t) (uses : list (Typing_env.t * list simple))
    (xs : list variable) (ks : list continuation),
    gather_uses tenv k (Handler params metadata handler) body
    = Some (uses, xs, ks) ->
    binders_not_free xs ks (Handler params metadata handler') = true ->
    sets_well_formed
      (Let_cont_non_recursive k (Handler params metadata handler) body)
    = true ->
    sets_well_formed handler' = true ->
    (∀ use : Typing_env.t * list simple,
      use ∈ uses ->
      application_admits params metadata (fst use) (snd use) target) ->
    expr_refines_from_valid target handler handler' ->
    expr_refines_from_valid tenv
      (Let_cont_non_recursive k (Handler params metadata handler) body)
      (Let_cont_non_recursive k (Handler params metadata handler') body).

(* [find_gathered_use]: the test of the gathering, and that of
   [use_env] for the one application. *)
Definition find_gathered_use_keeps_fit (tenv : Typing_env.t)
    (k : continuation) (h : continuation_handler) (body : expr) : bool :=
  match h, gather_uses tenv k h body with
  | Handler params metadata _, Some ([(tenv_use, args)], _, _) =>
    gather_uses_keeps_fit tenv k h body
    && use_env_keeps_fit tenv tenv_use params metadata args
  | Handler _ _ _, (Some (([] | _ :: _ :: _), _, _) | None) => true
  end.

(* [find_one_use]. *)
Definition find_one_use_keeps_fit (tenv : Typing_env.t) (k : continuation)
    (h : continuation_handler) (body : expr) : bool :=
  match find_use tenv tenv k h body with
  | Some _ => find_use_keeps_fit tenv tenv k h body
  | None => find_gathered_use_keeps_fit tenv k h body
  end.

(* The environment that [find_one_use] answers has the two conditions
   if the environment of the body has them and the test passes. *)
Definition find_one_use_known_to_fit_statement : Prop :=
  ∀ (tenv : Typing_env.t) (fit : bool) (k : continuation)
    (h : continuation_handler) (body : expr) (target : Typing_env.t)
    (xs : list variable) (ks : list continuation),
    known_to_fit tenv fit ->
    find_one_use tenv k h body = Some (target, xs, ks) ->
    known_to_fit target (fit && find_one_use_keeps_fit tenv k h body).

(* ======================================================================== *)
(* The function                                                             *)
(* ======================================================================== *)

(* What is left of a [Switch] is treated as the first function treats
   it: a [Switch] whose scrutinee has a constant for canonical element
   is the [Apply_cont] of its arm, and the arguments of an [Apply_cont]
   are replaced by their canonical elements. *)
Definition after_pruning (tenv : Typing_env.t) (e : expr) : expr :=
  match e with
  | Switch scrutinee arms => simplify_switch tenv scrutinee arms
  | Apply_cont ac => Apply_cont (simplify_apply_cont tenv ac)
  | Let _ _ _ _ | Let_static _ _ | Let_cont_non_recursive _ _ _
  | Let_cont_recursive _ _ | Apply _ | Invalid =>
    e
  end.

(* A [Switch]: [prune] is asked where meets may be made in the
   environment and the scrutinee may occur in terms; where it is not
   asked, or does not answer, the [Switch] is treated as the first
   function treats it. *)
Definition simplify_switch_pruning (prune : switch_rewrite)
    (tenv : Typing_env.t) (fit : bool) (scrutinee : simple)
    (arms : list (Z * apply_cont)) : expr :=
  match
    if fit && types_fit_b tenv
       && Typing_env.mem_simple (Some Name_mode.normal) tenv scrutinee
    then prune tenv scrutinee arms
    else None
  with
  | Some e => after_pruning tenv e
  | None => simplify_switch tenv scrutinee arms
  end.

(* [simplify_expr] of FlambdaSimplifyPassSpec.v, with [prune] and the
   boolean. *)
Fixpoint simplify_expr_pruning (prune : switch_rewrite) (tenv : Typing_env.t)
    (fit : bool) (e : expr) : expr :=
  match e with
  | Let x declared defining_expr body =>
    let resolved := resolve_value_slot tenv declared defining_expr in
    match simplify_let tenv x resolved with
    | Step_invalid => Invalid
    | Step_let defining_expr' tenv' =>
      Let x declared defining_expr'
        (simplify_expr_pruning prune tenv'
           (fit && let_keeps_fit tenv x resolved) body)
    | Step_stop =>
      match declare_variable tenv x declared with
      | Some tenv' =>
        Let x declared defining_expr
          (simplify_expr_pruning prune tenv' fit body)
      | None => e
      end
    end
  | Let_static bindings body =>
    let tenv' := add_static_bindings tenv bindings in
    let fit' := fit && statics_keep_fit tenv bindings in
    Let_static
      (map (simplify_static_binding_pruning prune tenv' fit') bindings)
      (simplify_expr_pruning prune tenv' fit' body)
  | Let_cont_non_recursive k handler body =>
    let body' := simplify_expr_pruning prune tenv fit body in
    Let_cont_non_recursive k
      (match handler with
       | Handler params metadata handler_body =>
         match find_one_use tenv k handler body' with
         | Some (target, xs, ks) =>
           let rewritten :=
             Handler params metadata
               (simplify_expr_pruning prune target
                  (fit && find_one_use_keeps_fit tenv k handler body')
                  handler_body)
           in
           if binders_not_free xs ks rewritten then rewritten
           else simplify_handler_pruning prune false tenv fit handler
         | None => simplify_handler_pruning prune false tenv fit handler
         end
       end)
      body'
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map (fun '(k, h) => (k, simplify_handler_pruning prune true tenv fit h))
         handlers)
      (simplify_expr_pruning prune tenv fit body)
  | Apply ap => simplify_apply tenv ap
  | Apply_cont ac => Apply_cont (simplify_apply_cont tenv ac)
  | Switch scrutinee arms =>
    simplify_switch_pruning prune tenv fit scrutinee arms
  | Invalid => Invalid
  end

(* A handler, where [tenv] is the environment of its [Let_cont] and
   [fit] its boolean. *)
with simplify_handler_pruning (prune : switch_rewrite) (recursive : bool)
    (tenv : Typing_env.t) (fit : bool) (h : continuation_handler)
    : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    match handler_env tenv params metadata with
    | Some tenv_handler =>
      let handler' := simplify_expr_pruning prune tenv_handler fit handler in
      if negb recursive
         || Bool.eqb (binds_statics handler) (binds_statics handler')
      then Handler params metadata handler'
      else h
    | None => h
    end
  end

(* The code that a static binding binds, where [tenv] is the
   environment of the body of the [Let_static] and [fit] its
   boolean. *)
with simplify_static_binding_pruning (prune : switch_rewrite)
    (tenv : Typing_env.t) (fit : bool) (binding : static_binding)
    : static_binding :=
  match binding with
  | Bind_code id c => Bind_code id (simplify_code_pruning prune tenv fit c)
  | Bind_deleted_code _ | Bind_symbol _ _ => binding
  end

with simplify_code_pruning (prune : switch_rewrite) (tenv : Typing_env.t)
    (fit : bool) (c : code) : code :=
  match c with
  | Code params my_closure ret exn metadata body =>
    match code_env_in tenv params my_closure metadata with
    | Some tenv_code =>
      let body' := simplify_expr_pruning prune tenv_code fit body in
      if negb (binds_statics body) && negb (binds_statics body')
      then Code params my_closure ret exn metadata body'
      else c
    | None => c
    end
  end.

(* From the environment that knows nothing. *)
Definition simplify_by_types_pruning (prune : switch_rewrite) (e : expr)
    : expr :=
  simplify_expr_pruning prune Typing_env.create true e.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* What the function answers has well-formed sets, if the term has. *)
Definition simplify_expr_pruning_sets_statement : Prop :=
  ∀ (prune : switch_rewrite) (tenv : Typing_env.t) (fit : bool) (e : expr),
    switch_rewrite_sound prune ->
    known_to_fit tenv fit ->
    sets_well_formed e = true ->
    sets_well_formed (simplify_expr_pruning prune tenv fit e) = true.

(* It refines the term from the states that have [run_valid] and
   satisfy the environment. *)
Definition simplify_expr_pruning_refines_statement : Prop :=
  ∀ (prune : switch_rewrite) (tenv : Typing_env.t) (fit : bool) (e : expr),
    switch_rewrite_sound prune ->
    known_to_fit tenv fit ->
    expr_refines_from_valid tenv e (simplify_expr_pruning prune tenv fit e).

(* From the environment that knows nothing: a unit whose sets are well
   formed refines what the function makes of it, the two run on their
   own. *)
Definition simplify_by_types_pruning_refines_statement : Prop :=
  ∀ (prune : switch_rewrite) (e : expr),
    switch_rewrite_sound prune ->
    sets_well_formed e = true ->
    program_refines e (simplify_by_types_pruning prune e).

(* With a [prune] that answers nothing it is the first function,
   whatever the boolean. *)
Definition simplify_expr_no_pruning_statement : Prop :=
  ∀ (tenv : Typing_env.t) (fit : bool) (e : expr),
    simplify_expr_pruning no_pruning tenv fit e = simplify_expr tenv e.

(* ======================================================================== *)
(* With the meet that the contract is proved of                             *)
(* ======================================================================== *)

(* The function, with the pruning of FlambdaSimplifySwitchSpec.v at
   the transcribed meet that the contract is proved of
   ([meet_to_prove], FlambdaMeetSpec.v), a join and some fuel. Where
   the fuel runs out in a meet nothing is pruned. *)
Definition simplify_by_types_pruning_with (join : join_scoped_envs)
    (fuel : nat) (e : expr) : expr :=
  simplify_by_types_pruning
    (FlambdaSimplifySwitchSpec.simplify_switch (meet_to_prove join fuel)) e.

(* It refines a unit whose sets are well formed, for a join that has
   the contract of a join and whatever the fuel. *)
Definition simplify_by_types_pruning_with_refines_statement : Prop :=
  ∀ (join : join_scoped_envs) (fuel : nat) (e : expr),
    join_scoped_envs_sound join ->
    sets_well_formed e = true ->
    program_refines e (simplify_by_types_pruning_with join fuel e).

(* What the validation can run: the join that forgets
   (FlambdaMeetExamplesSpec.v), which has the contract of a join, and
   the test of the sets made first. A unit whose sets fail the test is
   left as it is. *)
Definition prune_by_types (fuel : nat) (e : expr) : expr :=
  if sets_well_formed e
  then
    simplify_by_types_pruning_with FlambdaMeetExamplesSpec.forgetful_join
      fuel e
  else e.

(* It has no premise. *)
Definition prune_by_types_refines_statement : Prop :=
  ∀ (fuel : nat) (e : expr), program_refines e (prune_by_types fuel e).

(* ======================================================================== *)
(* An example                                                               *)
(* ======================================================================== *)

(* [let i = 5 in switch i with 0 -> k0 | 1 -> k1], with the function
   that the validation can run, at the fuel of the examples of
   FlambdaSimplifySwitchSpec.v: the scrutinee is the constant, the
   meet of its type with the number of each arm is bottom, and no arm
   is left. The first function leaves the [Switch], which has no arm
   for the constant. *)
Definition example_pruning_no_arm_term (k0 k1 : continuation) : expr :=
  Let example_i naked_immediate_kind (Simple (Const (Naked_immediate 5)))
    (Switch (Var example_i)
       [(0, Apply_cont_expr k0 [] None); (1, Apply_cont_expr k1 [] None)]).

Definition example_pruning_no_arm_statement : Prop :=
  ∀ k0 k1 : continuation,
    prune_by_types 10 (example_pruning_no_arm_term k0 k1)
    = Let example_i naked_immediate_kind
        (Simple (Const (Naked_immediate 5))) Invalid
    ∧ simplify_by_types (example_pruning_no_arm_term k0 k1)
      = example_pruning_no_arm_term k0 k1.

(* [let x = opaque in let i = (x < 7) in switch i with 0 -> k0 | 1 ->
   k1 | 2 -> k2], the comparison being of tagged immediates. It is not
   decided; [i] is given the type of the two booleans; the meet of that
   type with 2 is bottom, and the arm for 2 is removed. The first
   function leaves the three arms. *)
Definition example_pruning_comparison_term (arms : list (Z * apply_cont))
    : expr :=
  Let example_x Value (example_opaque Value (Tagged_immediate 5))
    (Let example_i naked_immediate_kind
       (Prim
          (Binary
             (Int_comp Standard_tagged_immediate
                (Yielding_bool (Int_lt Signed))))
          [Var example_x; Const (Tagged_immediate 7)])
       (Switch (Var example_i) arms)).

Definition example_pruning_comparison_statement : Prop :=
  ∀ k0 k1 k2 : continuation,
    let arm (i : Z) (k : continuation) := (i, Apply_cont_expr k [] None) in
    prune_by_types 10
      (example_pruning_comparison_term [arm 0 k0; arm 1 k1; arm 2 k2])
    = example_pruning_comparison_term [arm 0 k0; arm 1 k1]
    ∧ simplify_by_types
        (example_pruning_comparison_term [arm 0 k0; arm 1 k1; arm 2 k2])
      = example_pruning_comparison_term [arm 0 k0; arm 1 k1; arm 2 k2].
