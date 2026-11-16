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


(* The rewrites from the types (FlambdaSimplifyPassSpec.simplify_by_types),
   without the test of the environment at every [Let]: the same function,
   computed another way.

   This file is to be reviewed: the two statements. What the functions
   compute is FlambdaSimplifyPassSpec.simplify_by_types, by
   [simplify_by_types_carried_eq_statement]: the equality lets the
   validation of middle_end/flambda2/tests/semantics/ run them in
   place of that function, and their text then matters only to the
   time that the validation takes.

   Why. FlambdaSimplifyPassSpec.simplify_let, [declare_variable] and
   [bind_params_to_args], and FlambdaSimplifyCodeEnvSpec.
   define_variables_checked, test that the environment passes
   FlambdaTypingEnvInvariantSpec.typing_env_well_formed before they
   define a variable in it: the test is a premise of the statements,
   made so that the function needs none. It reads the whole
   environment, comparing its levels and its binding times with each
   other, and it is made again at each [Let] and each parameter; so a
   function of n [Let]s in a row costs n tests of an environment that
   grows to n names. None of these tests is Simplify's.

   How. The functions here carry a boolean beside the environment,
   as the pruning does (FlambdaSimplifyPruningSpec.v, [known_to_fit]):
   where it is true, the environment has the invariant
   ([typing_env_invariant], which the test is part of) and
   [aliases_kinded]. The test is then known to pass, and is not made
   ([well_formed_or_known]). The boolean is true of the environment
   that knows nothing. It stays true where a variable is defined that
   the environment has not got; where an alias is added or a type
   given, if the premises of the statements by which those keep the
   two conditions pass as tests ([alias_keeps_fit],
   [equation_keeps_fit], [statics_keep_fit], [leaf_uses_keep_fit] of
   FlambdaSimplifyPruningSpec.v: each looks names up, and none
   compares the names of the environment with each other); and in the
   body of code and of a handler. Where it is false the test is made
   as before, and the boolean stays false from there down. So a term
   on which one of those tests fails costs what it cost, and those
   tests.

   Each function is the one of FlambdaSimplifyPassSpec.v,
   FlambdaSimplifyCodeEnvSpec.v or FlambdaSimplifyHandlersSpec.v whose
   name it has, with the boolean. Two things are written otherwise.
   [simplify_let_carried] asks FlambdaSimplifyPassSpec.simplify_named
   once, and answers the step with whether it keeps the two
   conditions ([let_step_of], [outcome_keeps_fit]: the cases of
   [simplify_let] and of FlambdaSimplifyPruningSpec.let_keeps_fit).
   And the functions that answer an environment answer its boolean
   with it ([gather_uses_carried], one with each application).

   Not here. FlambdaTypingEnvInvariantSpec.types_well_formed_b, which
   FlambdaSimplifyPassSpec.v tests where a prover of tagged immediates
   or the prover of physical equality is asked, reads every type of
   the environment, and is made as before; and a name is looked up by
   a search in a list. So the cost of n [Let]s in a row still grows
   as the square of n.

   The statements:
   - [simplify_expr_carried] under an environment that the boolean is
     right about is [simplify_expr] under that environment
     ([simplify_expr_carried_eq_statement]);
   - [simplify_by_types_carried] is [simplify_by_types], of every
     term ([simplify_by_types_carried_eq_statement]).

   The proofs are in FlambdaSimplifyCarriedProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaValueSpec FlambdaPrimitiveModel FlambdaSemanticsSpec
  FlambdaNormaliseSpec FlambdaUnusedContsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaTypingEnv
  FlambdaTypingEnvSpec FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvExamplesSpec FlambdaSimplifyPrimitivesSpec
  FlambdaSimplifyApplySpec FlambdaSimplifyStaticsSpec
  FlambdaSimplifyCodeEnvSpec FlambdaSimplifyHandlersSpec
  FlambdaSimplifyPassSpec FlambdaSimplifyPruningSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* The test, where the boolean does not answer it                           *)
(* ======================================================================== *)

(* Whether the environment passes the test of well-formedness: it does
   where [fit] is true, and the test is made where it is false. *)
Definition well_formed_or_known (fit : bool) (tenv : Typing_env.t) : bool :=
  fit || typing_env_well_formed tenv.

(* ======================================================================== *)
(* A [Let]                                                                  *)
(* ======================================================================== *)

(* The step of a [Let] whose defining expression has the answer
   [answer]: the cases of FlambdaSimplifyPassSpec.simplify_let. *)
Definition let_step_of (tenv : Typing_env.t) (x : variable)
    (defining_expr : named) (answer : outcome) : let_step :=
  match answer with
  | Becomes_invalid => Step_invalid
  | Becomes_simple s => bind_simple tenv x s
  | Stays_with_type ty =>
    if type_in_scope_b tenv ty
    then
      Step_let (canonical_named tenv defining_expr)
        (Typing_env.replace_equation
           (Typing_env.add_variable_definition tenv x
              (FlambdaTypeGrammar.kind ty) Name_mode.normal)
           (Name_var x) ty)
    else Step_stop
  | Stays_with_kind kind =>
    Step_let (canonical_named tenv defining_expr)
      (Typing_env.add_variable_definition tenv x kind Name_mode.normal)
  | Not_known => Step_stop
  end.

(* Whether the environment of that step is known to keep the two
   conditions: the cases of FlambdaSimplifyPruningSpec.let_keeps_fit. *)
Definition outcome_keeps_fit (tenv : Typing_env.t) (x : variable)
    (answer : outcome) : bool :=
  match answer with
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

(* [simplify_let], and whether the environment that it answers keeps
   the two conditions. *)
Definition simplify_let_carried (fit : bool) (tenv : Typing_env.t)
    (x : variable) (defining_expr : named) : let_step * bool :=
  if well_formed_or_known fit tenv
     && negb (Typing_env.mem None tenv (Name_var x))
  then
    let answer := simplify_named tenv defining_expr in
    (let_step_of tenv x defining_expr answer, outcome_keeps_fit tenv x answer)
  else (Step_stop, false).

(* [declare_variable]. *)
Definition declare_variable_carried (fit : bool) (tenv : Typing_env.t)
    (x : variable) (declared : flambda_kind) : option Typing_env.t :=
  if well_formed_or_known fit tenv
     && negb (Typing_env.mem None tenv (Name_var x))
  then
    Some
      (Typing_env.add_variable_definition tenv x declared Name_mode.normal)
  else None.

(* [let_env], with the boolean of the environment that it answers. *)
Definition let_env_carried (fit : bool) (tenv : Typing_env.t) (x : variable)
    (declared : flambda_kind) (defining_expr : named)
    : option (Typing_env.t * bool) :=
  match
    simplify_let_carried fit tenv x
      (resolve_value_slot tenv declared defining_expr)
  with
  | (Step_let _ tenv', keeps) => Some (tenv', fit && keeps)
  | (Step_invalid, _) => None
  | (Step_stop, _) =>
    match declare_variable_carried fit tenv x declared with
    | Some tenv' => Some (tenv', fit)
    | None => None
    end
  end.

(* ======================================================================== *)
(* The environments of code and of a handler                                *)
(* ======================================================================== *)

(* FlambdaSimplifyCodeEnvSpec.define_variables_checked. A variable
   that is defined keeps the two conditions, so the boolean is that of
   every environment on the way. *)
Fixpoint define_variables_carried (fit : bool) (tenv : Typing_env.t)
    (definitions : list (variable * flambda_kind)) : option Typing_env.t :=
  match definitions with
  | [] => Some tenv
  | (x, kind) :: definitions =>
    if
      well_formed_or_known fit tenv
      && negb (Typing_env.mem None tenv (Name_var x))
    then
      define_variables_carried fit
        (Typing_env.add_variable_definition tenv x kind Name_mode.normal)
        definitions
    else None
  end.

(* FlambdaSimplifyCodeEnvSpec.code_env_in. *)
Definition code_env_in_carried (fit : bool) (tenv : Typing_env.t)
    (params : list variable) (my_closure : variable)
    (metadata : code_metadata) : option Typing_env.t :=
  let kinds := unarize (cm_params_arity metadata) in
  if Nat.eqb (length kinds) (length params)
  then
    define_variables_carried fit (Typing_env.closure_env tenv)
      ((my_closure, Value) :: combine params kinds)
  else None.

(* FlambdaSimplifyHandlersSpec.handler_env. *)
Definition handler_env_carried (fit : bool) (tenv : Typing_env.t)
    (params : list variable) (metadata : handler_metadata)
    : option Typing_env.t :=
  let kinds := hm_params_kinds metadata in
  if Nat.eqb (length kinds) (length params)
  then define_variables_carried fit tenv (combine params kinds)
  else None.

(* ======================================================================== *)
(* Where a continuation is applied                                          *)
(* ======================================================================== *)

(* [bind_params_to_args], with the boolean of the environment that it
   answers: an alias that is added keeps the two conditions if it
   passes the test of [alias_keeps_fit]. *)
Fixpoint bind_params_to_args_carried (fit : bool) (aliased : simple -> bool)
    (tenv : Typing_env.t) (bindings : list (variable * flambda_kind * simple))
    : option (Typing_env.t * bool) :=
  match bindings with
  | [] => Some (tenv, fit)
  | (x, kind, canonical) :: bindings =>
    if
      well_formed_or_known fit tenv
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
          bind_params_to_args_carried
            (fit && alias_keeps_fit defined (Var x) canonical kind) aliased
            (Typing_env.add_alias_result_t result) bindings
        | Some Unknown | Some Bottom | None =>
          bind_params_to_args_carried fit aliased defined bindings
        end
      else bind_params_to_args_carried fit aliased defined bindings
    else None
  end.

(* [use_env]. *)
Definition use_env_carried (fit : bool) (scope tenv : Typing_env.t)
    (params : list variable) (metadata : handler_metadata)
    (args : list simple) : option (Typing_env.t * bool) :=
  let kinds := hm_params_kinds metadata in
  match simplify_simples tenv args with
  | Some answers =>
    if
      Nat.eqb (length kinds) (length params)
      && Nat.eqb (length answers) (length params)
    then
      bind_params_to_args_carried fit
        (fun canonical =>
           simple_in_terms_b scope canonical && not_among params canonical)
        tenv
        (combine (combine params kinds) (map snd answers))
    else None
  | None => None
  end.

(* [find_use], with the boolean of the environment that it answers.
   [fit] is the boolean of [tenv]. The environment of an arm keeps the
   two conditions if the alias that the arm adds passes its test
   ([leaf_uses_keep_fit]). *)
Fixpoint find_use_carried (fit : bool) (scope tenv : Typing_env.t)
    (k : continuation) (h : continuation_handler) (body : expr)
    : option (Typing_env.t * bool * list variable * list continuation) :=
  match body with
  | Let x declared defining_expr rest =>
    if var_free_in_handler_b x h then None
    else
      match let_env_carried fit tenv x declared defining_expr with
      | Some (tenv', fit') =>
        match find_use_carried fit' scope tenv' k h rest with
        | Some (target, fit_target, xs, ks) =>
          Some (target, fit_target, x :: xs, ks)
        | None => None
        end
      | None => None
      end
  | Let_cont_non_recursive k' h' rest =>
    if
      negb (Pos.eqb k k')
      && negb (cont_free_in_handler_b k h')
      && negb (cont_free_in_handler_b k' h)
    then
      match find_use_carried fit scope tenv k h rest with
      | Some (target, fit_target, xs, ks) =>
        Some (target, fit_target, xs, k' :: ks)
      | None => None
      end
    else None
  | Apply_cont _ | Switch _ _ =>
    match h, leaf_uses tenv k body with
    | Handler params metadata _, Some [(tenv_use, args)] =>
      match
        use_env_carried (fit && leaf_uses_keep_fit tenv k body) scope
          tenv_use params metadata args
      with
      | Some (target, fit_target) => Some (target, fit_target, [], [])
      | None => None
      end
    | Handler _ _ _, (Some ([] | _ :: _ :: _) | None) => None
    end
  | Let_static _ _ | Let_cont_recursive _ _ | Apply _ | Invalid => None
  end.

(* FlambdaSimplifyPassSpec.gather_uses, each application with the
   boolean of its environment. [fit] is the boolean of [tenv]. *)
Fixpoint gather_uses_carried (fit : bool) (tenv : Typing_env.t)
    (k : continuation) (h : continuation_handler) (body : expr)
    : option
        (list (Typing_env.t * bool * list simple) * list variable
         * list continuation) :=
  match body with
  | Let x declared defining_expr rest =>
    if var_free_in_handler_b x h then None
    else
      match let_env_carried fit tenv x declared defining_expr with
      | Some (tenv', fit') =>
        match gather_uses_carried fit' tenv' k h rest with
        | Some (uses, xs, ks) => Some (uses, x :: xs, ks)
        | None => None
        end
      | None => None
      end
  | Let_cont_non_recursive k' h' rest =>
    if negb (Pos.eqb k k') && negb (cont_free_in_handler_b k' h)
    then
      match gather_uses_carried fit tenv k h rest with
      | Some (uses, xs, ks) =>
        if cont_free_in_handler_b k h'
        then
          match h' with
          | Handler params' metadata' body' =>
            if
              negb (cont_free_in_handler_b k h)
              && forallb (fun x => negb (var_free_in_handler_b x h)) params'
            then
              match handler_env_carried fit tenv params' metadata' with
              | Some tenv_handler =>
                match gather_uses_carried fit tenv_handler k h body' with
                | Some (uses', xs', ks') =>
                  Some
                    (uses' ++ uses, params' ++ xs' ++ xs,
                     k :: k' :: ks' ++ ks)
                | None => None
                end
              | None => None
              end
            else None
          end
        else Some (uses, xs, k' :: ks)
      | None => None
      end
    else None
  | Apply_cont _ | Switch _ _ =>
    match leaf_uses tenv k body with
    | Some uses =>
      Some
        (map
           (fun use =>
              (fst use, fit && leaf_uses_keep_fit tenv k body, snd use))
           uses,
         [], [])
    | None => None
    end
  | Let_static _ _ | Let_cont_recursive _ _ | Apply _ | Invalid => None
  end.

(* FlambdaSimplifyPassSpec.find_gathered_use, with the boolean of the
   environment that it answers. *)
Definition find_gathered_use_carried (fit : bool) (tenv : Typing_env.t)
    (k : continuation) (h : continuation_handler) (body : expr)
    : option (Typing_env.t * bool * list variable * list continuation) :=
  match h, gather_uses_carried fit tenv k h body with
  | Handler params metadata _, Some ([(tenv_use, fit_use, args)], xs, ks) =>
    match use_env_carried fit_use tenv tenv_use params metadata args with
    | Some (target, fit_target) => Some (target, fit_target, xs, ks)
    | None => None
    end
  | Handler _ _ _, (Some (([] | _ :: _ :: _), _, _) | None) => None
  end.

(* FlambdaSimplifyPassSpec.find_one_use. *)
Definition find_one_use_carried (fit : bool) (tenv : Typing_env.t)
    (k : continuation) (h : continuation_handler) (body : expr)
    : option (Typing_env.t * bool * list variable * list continuation) :=
  match find_use_carried fit tenv tenv k h body with
  | Some found => Some found
  | None => find_gathered_use_carried fit tenv k h body
  end.

(* ======================================================================== *)
(* A term                                                                   *)
(* ======================================================================== *)

(* [simplify_expr], where [fit] is the boolean of [tenv]. *)
Fixpoint simplify_expr_carried (fit : bool) (tenv : Typing_env.t) (e : expr)
    : expr :=
  match e with
  | Let x declared defining_expr body =>
    match
      simplify_let_carried fit tenv x
        (resolve_value_slot tenv declared defining_expr)
    with
    | (Step_invalid, _) => Invalid
    | (Step_let defining_expr' tenv', keeps) =>
      Let x declared defining_expr'
        (simplify_expr_carried (fit && keeps) tenv' body)
    | (Step_stop, _) =>
      match declare_variable_carried fit tenv x declared with
      | Some tenv' =>
        Let x declared defining_expr (simplify_expr_carried fit tenv' body)
      | None => e
      end
    end
  | Let_static bindings body =>
    let tenv' := add_static_bindings tenv bindings in
    let fit' := fit && statics_keep_fit tenv bindings in
    Let_static (map (simplify_static_binding_carried fit' tenv') bindings)
      (simplify_expr_carried fit' tenv' body)
  | Let_cont_non_recursive k handler body =>
    let body' := simplify_expr_carried fit tenv body in
    Let_cont_non_recursive k
      (match handler with
       | Handler params metadata handler_body =>
         match find_one_use_carried fit tenv k handler body' with
         | Some (target, fit_target, xs, ks) =>
           let rewritten :=
             Handler params metadata
               (simplify_expr_carried fit_target target handler_body)
           in
           if binders_not_free xs ks rewritten then rewritten
           else simplify_handler_carried false fit tenv handler
         | None => simplify_handler_carried false fit tenv handler
         end
       end)
      body'
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map (fun '(k, h) => (k, simplify_handler_carried true fit tenv h))
         handlers)
      (simplify_expr_carried fit tenv body)
  | Apply ap => simplify_apply tenv ap
  | Apply_cont ac => Apply_cont (simplify_apply_cont tenv ac)
  | Switch scrutinee arms => simplify_switch tenv scrutinee arms
  | Invalid => Invalid
  end

(* [simplify_handler]. *)
with simplify_handler_carried (recursive : bool) (fit : bool)
    (tenv : Typing_env.t) (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    match handler_env_carried fit tenv params metadata with
    | Some tenv_handler =>
      let handler' := simplify_expr_carried fit tenv_handler handler in
      if negb recursive
         || Bool.eqb (binds_statics handler) (binds_statics handler')
      then Handler params metadata handler'
      else h
    | None => h
    end
  end

(* [simplify_static_binding]. *)
with simplify_static_binding_carried (fit : bool) (tenv : Typing_env.t)
    (binding : static_binding) : static_binding :=
  match binding with
  | Bind_code id c => Bind_code id (simplify_code_carried fit tenv c)
  | Bind_deleted_code _ | Bind_symbol _ _ => binding
  end

(* [simplify_code]. *)
with simplify_code_carried (fit : bool) (tenv : Typing_env.t) (c : code)
    : code :=
  match c with
  | Code params my_closure ret exn metadata body =>
    match code_env_in_carried fit tenv params my_closure metadata with
    | Some tenv_code =>
      let body' := simplify_expr_carried fit tenv_code body in
      if negb (binds_statics body) && negb (binds_statics body')
      then Code params my_closure ret exn metadata body'
      else c
    | None => c
    end
  end.

(* From the environment that knows nothing, which has the two
   conditions. *)
Definition simplify_by_types_carried (e : expr) : expr :=
  simplify_expr_carried true Typing_env.create e.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* Under an environment that the boolean is right about, it is the
   function of FlambdaSimplifyPassSpec.v. *)
Definition simplify_expr_carried_eq_statement : Prop :=
  ∀ (fit : bool) (tenv : Typing_env.t) (e : expr),
    known_to_fit tenv fit ->
    simplify_expr_carried fit tenv e = simplify_expr tenv e.

(* From the environment that knows nothing, of every term. *)
Definition simplify_by_types_carried_eq_statement : Prop :=
  ∀ e : expr, simplify_by_types_carried e = simplify_by_types e.
