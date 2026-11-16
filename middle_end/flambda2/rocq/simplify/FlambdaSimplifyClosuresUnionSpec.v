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


(* The handler of a continuation that is applied at several places,
   rewritten with a parameter known to be one of the closures that the
   applications pass.

   This file is to be reviewed: every definition and every statement.
   No pass uses [closures_union_env] yet, and it is not wired into the
   validation of middle_end/flambda2/tests/semantics/: nothing here
   changes what that validation computes. The computed examples at the
   end of this file are all that calls it.

   What it is for. Where a continuation is applied at several places,
   Simplify rewrites its handler under the join of the environments of
   the applications (simplify/join_points.ml, under
   -flambda2-join-points): a parameter has there the join of the types
   of its arguments. FlambdaSimplifyPassSpec.v gathers the applications
   of a continuation ([gather_uses]) and states that the handler may be
   rewritten under any environment that each of them admits
   ([gather_uses_refines_statement]); for several applications it
   computes no such environment. This file computes one, for one kind
   of knowledge: a parameter to which every application passes a
   closure that the environment of that application knows to be a
   closure.

   It is the model's own. It is not the join of the compiler and
   nothing is claimed of that join, nor that Simplify gives a parameter
   the type that this gives it. There is no compiler function to check
   the definitions against: they are to be checked against what the
   statements say of them.

   The environment ([closures_union_env tenv params metadata uses],
   where [tenv] is the environment of the [Let_cont] and [uses] are
   applications of its continuation, each with the environment at the
   application). The parameters are defined in order at the kinds
   that the handler gives them, after the tests of
   FlambdaSimplifyHandlersSpec.handler_env, and a parameter of kind
   value is given a type where all of this holds:
   - there is an application at least;
   - at every application, the type of the argument, expanded under
     the environment of that application, is a type of closures that
     are not null and nothing else ([closures_known]): a type that
     may be null, an unknown type, a variant is refused, and nothing
     is cut out of one;
   - where two applications have a case for one function slot, the
     two cases are the same, whole: the index, the entry and the
     extension, by the test FlambdaTypeEqualSpec.type_eqb
     ([closures_agree]). Two closures of different code at one slot
     are refused: that would be a join of two cases, and none is made;
   - every variable of the type may occur in terms in [tenv]
     (FlambdaSimplifyPassSpec.type_in_scope_b): what an application
     knows of a variable bound between the [Let_cont] and the
     application cannot be said where the handler is.
   The type is then the type of closures whose cases are the cases of
   all the applications ([closures_union], [closures_union_all]): a
   value has it exactly when it has the type of closures of one of
   them ([closures_union_satisfied_statement]). Its allocation mode is
   [Heap_or_local], which says nothing. A parameter that fails one of
   the conditions is defined at its kind and no more, as
   [handler_env] defines it; with no application the environment is
   that of [handler_env] ([closures_union_env_no_use_statement]).

   What is stated.
   - The type expanded is right ([closures_known_sound_statement]), and
     the union is exact ([closures_union_satisfied_statement]).
   - The environment holds where an application enters the handler
     ([closures_union_env_satisfied_statement]): in a state that
     satisfies the environment of the application and the environment
     of the [Let_cont], with the parameters bound to the values of the
     arguments. It is asked that a variable that may occur in terms at
     the [Let_cont] may occur in terms at the application; the two
     statements below have it of the applications that they are about.
   - A [Let_cont] whose body is an [Apply_cont] or a [Switch]
     ([let_cont_handler_closures_union_statement]): where [leaf_uses]
     answers the applications of the body and [closures_union_env]
     answers an environment for them, the handler may be replaced by
     one that refines it under that environment. Nothing more is
     asked of the environment of the [Let_cont]: no premise that it
     is known to fit is needed here.
   - The applications that [gather_uses] gathers, under binders: each
     admits the environment
     ([gathered_closures_union_admitted_statement]), so the handler
     may be replaced by one that refines it under the environment and
     mentions none of the binders passed on the way to the
     applications ([gathered_closures_union_refines_statement];
     [binders_not_free], as in [gather_uses_refines_statement]).
     These two have premises that the statement before has not:
     [known_to_fit tenv true] (FlambdaSimplifyPruningSpec.v: the
     invariant of the environment of the [Let_cont], which is its
     test of well-formedness with the agreement of its levels, and
     that its aliases are kinded), and the test
     [gather_uses_keeps_fit] of the binders passed. The reason:
     "admits" (FlambdaSimplifyPassSpec.application_admits) is of every
     state that satisfies the environment of the application, and the
     environment here is made from that of the [Let_cont]; that a
     state which satisfies the first satisfies the second is
     FlambdaSimplifyPruningSpec.gather_uses_weaken_statement, which
     has those two premises. A function that carries the boolean of
     the pruning function has the first where its boolean is true;
     one that does not needs the transfer of
     [gather_uses_refines_statement] proved again with the
     satisfaction of the environment of the [Let_cont] carried down to
     the applications, which is not here.
   - The environment commutes with renaming
     ([closures_union_env_rename_statement]).
   - Six computed examples, each of what the prover of code ids
     (FlambdaProvers.Provers.prove_code_ids) answers for the
     parameter under the environment made. It proves the code of
     both for two closures of two slots, and the one piece of code
     for two closures that have the same case and for a closure that
     holds a variable bound before the [Let_cont]. It answers that it
     does not know, the parameter being defined at its kind and no
     more, for two closures of different code at one slot, for a
     closure and a value of which nothing is known, and for a closure
     that holds a variable bound after the [Let_cont].

   Where it stops. An application is a use that [leaf_uses] or
   [gather_uses] answers: FlambdaSimplifyUsesSpec.v and
   FlambdaSimplifyPassSpec.v say what they leave out (a continuation
   that a call returns or raises to, an application under a
   [Let_static] or a recursive [Let_cont]). Only closures are joined:
   a parameter that is passed something else at one application is
   defined at its kind and no more, whatever the others pass.

   Not here: that the environment made is known to fit
   ([known_to_fit]) where the environment of the [Let_cont] is, which
   a function that carries the boolean into the handler needs; and
   any pass that uses the environment.

   Nothing is assumed beyond the premises of each statement.

   The proofs are in FlambdaSimplifyClosuresUnionProofs.v, which needs
   no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaValueSpec FlambdaPrimitiveModel FlambdaSemanticsSpec
  FlambdaNominalSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypeEqualSpec FlambdaBindingTime
  FlambdaTypingEnv FlambdaTypingEnvSpec FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvExamplesSpec FlambdaTypingEnvRefinementSpec
  FlambdaTypeRenameSpec FlambdaEnvRenameSpec
  FlambdaMeetGrammar FlambdaExpandHead FlambdaProvers
  FlambdaSimplifyPrimitivesSpec FlambdaSimplifyApplySpec
  FlambdaSimplifyHandlersSpec FlambdaSimplifyPassSpec
  FlambdaSimplifyPassRenamingSpec FlambdaSimplifyPruningSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* A type that is known to be of closures                                   *)
(* ======================================================================== *)

(* The closures of a head of kind value: the head is one of closures,
   and is not null. *)
Definition closures_of_head (head : head_of_kind_value)
    : option row_like_for_closures :=
  match head with
  | Head_of_kind_value (Ok (Closures closures _)) Not_null => Some closures
  | Head_of_kind_value
      (Ok
         (Variant _ _ _ _ _ _ | Mutable_block _ | Boxed_number _ _ _
         | String _)
      | Unknown | Bottom)
      Not_null
  | Head_of_kind_value _ (Maybe_null _) =>
    None
  end.

(* The closures that a type stands for under an environment: the type
   expanded (FlambdaExpandHead.expand_head: the type that the
   environment has for the canonical element of an alias, or the type
   itself) has a head of closures that are not null. [None] of every
   other type, and where the expansion answers nothing. *)
Definition closures_known (tenv : Typing_env.t) (ty : flambda_type)
    : option row_like_for_closures :=
  match expand_head tenv ty with
  | Some expanded =>
    match Expanded_type.descr expanded with
    | Ok (Expanded_type.Head_value head) => closures_of_head head
    | Ok
        (Expanded_type.Head_naked_immediate _
        | Expanded_type.Head_naked_number _ _ | Expanded_type.Head_rec_info _
        | Expanded_type.Head_region _)
    | Unknown | Bottom =>
      None
    end
  | None => None
  end.

(* The type of the closures given, which are not null. The allocation
   mode says nothing. *)
Definition closures_type (closures : row_like_for_closures) : flambda_type :=
  create_closures Alloc_mode.Heap_or_local closures.

(* A value of the type, under a valuation that satisfies the
   environment with no name exempt, has the type of the closures
   answered. *)
Definition closures_known_sound_statement : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (ty : flambda_type) (closures : row_like_for_closures) (v : value),
    typing_env_satisfied_strictly statics valuation tenv ->
    type_satisfied statics valuation v ty ->
    closures_known tenv ty = Some closures ->
    type_satisfied statics valuation v (closures_type closures).

(* ======================================================================== *)
(* The union                                                                *)
(* ======================================================================== *)

(* The cases, by function slot. *)
Definition cases_of (closures : row_like_for_closures)
    : finite_map row_like_closures_case :=
  match closures with
  | Row_like_for_closures known_closures => known_closures
  end.

(* The type of the closures of one case. *)
Definition one_case_type (slot : function_slot)
    (case : row_like_closures_case) : flambda_type :=
  closures_type (Row_like_for_closures [(slot, case)]).

(* Where both have a case for a function slot, the two cases are the
   same: the types of the closures of each alone are the same type
   (FlambdaTypeEqualSpec.type_eqb, which compares the index, the entry
   and the extension of the case). *)
Definition closures_agree (closures closures' : row_like_for_closures)
    : bool :=
  forallb
    (fun slot =>
       match lookup slot (cases_of closures), lookup slot (cases_of closures')
       with
       | Some case, Some case' =>
         type_eqb (one_case_type slot case) (one_case_type slot case')
       | Some _, None | None, _ => true
       end)
    (domain (cases_of closures)).

(* The cases of both, where they agree. A slot that both have keeps the
   case of the first. *)
Definition closures_union (closures closures' : row_like_for_closures)
    : option row_like_for_closures :=
  if closures_agree closures closures'
  then
    Some
      (Row_like_for_closures
         (union_with (fun case _ => case) (cases_of closures)
            (cases_of closures')))
  else None.

(* The same of a list that is not empty. *)
Fixpoint closures_union_all (first : row_like_for_closures)
    (rest : list row_like_for_closures) : option row_like_for_closures :=
  match rest with
  | [] => Some first
  | closures :: rest =>
    match closures_union first closures with
    | Some union => closures_union_all union rest
    | None => None
    end
  end.

(* A value has the type of the union exactly when it has the type of
   one of the two. *)
Definition closures_union_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env)
    (closures closures' union : row_like_for_closures) (v : value),
    closures_union closures closures' = Some union ->
    (type_satisfied statics vars v (closures_type union)
     <-> type_satisfied statics vars v (closures_type closures)
         ∨ type_satisfied statics vars v (closures_type closures')).

(* ======================================================================== *)
(* The types of the parameters                                              *)
(* ======================================================================== *)

(* The closures that an application passes at a position: the type
   that the environment of the application has for the argument there
   ([simplify_simple]) is known to be of closures. *)
Definition argument_closures (index : nat) (use : Typing_env.t * list simple)
    : option row_like_for_closures :=
  match nth_error (snd use) index with
  | Some arg =>
    match simplify_simple (fst use) arg with
    | Some (ty, _) => closures_known (fst use) ty
    | None => None
    end
  | None => None
  end.

(* The type of the closures that the applications pass at a position.
   [None] where there is no application, where one of them passes
   something that is not known to be a closure, or where two of them
   do not agree. *)
Definition union_of_arguments (uses : list (Typing_env.t * list simple))
    (index : nat) : option flambda_type :=
  match map_opt (argument_closures index) uses with
  | Some (first :: rest) =>
    match closures_union_all first rest with
    | Some union => Some (closures_type union)
    | None => None
    end
  | Some [] | None => None
  end.

(* The type that the parameter at a position is given, if it is given
   one: the parameter is of kind value, and every variable of the type
   may occur in terms under [scope], the environment of the
   [Let_cont]. *)
Definition parameter_union (scope : Typing_env.t)
    (uses : list (Typing_env.t * list simple)) (index : nat)
    (kind : flambda_kind) : option flambda_type :=
  match kind with
  | Value =>
    match union_of_arguments uses index with
    | Some ty => if type_in_scope_b scope ty then Some ty else None
    | None => None
    end
  | Naked_number _ | Rec_info | Region => None
  end.

(* The same for the parameters from a position on, by their kinds. *)
Fixpoint parameter_unions (scope : Typing_env.t)
    (uses : list (Typing_env.t * list simple)) (index : nat)
    (kinds : list flambda_kind) : list (option flambda_type) :=
  match kinds with
  | [] => []
  | kind :: kinds =>
    parameter_union scope uses index kind
    :: parameter_unions scope uses (S index) kinds
  end.

(* ======================================================================== *)
(* The environment                                                          *)
(* ======================================================================== *)

(* Variables are defined in order at their kinds, with the normal name
   mode, each after the two tests of
   FlambdaSimplifyCodeEnvSpec.define_variables_checked (the environment
   is well formed, and does not have the variable); a variable that
   comes with a type is given it. [None] where a test fails. *)
Fixpoint define_typed_checked (tenv : Typing_env.t)
    (definitions : list (variable * flambda_kind * option flambda_type))
    : option Typing_env.t :=
  match definitions with
  | [] => Some tenv
  | (x, kind, ty) :: definitions =>
    if
      typing_env_well_formed tenv
      && negb (Typing_env.mem None tenv (Name_var x))
    then
      let defined :=
        Typing_env.add_variable_definition tenv x kind Name_mode.normal
      in
      define_typed_checked
        (match ty with
         | Some ty => Typing_env.replace_equation defined (Name_var x) ty
         | None => defined
         end)
        definitions
    else None
  end.

(* The environment under which the handler is rewritten, where it is
   defined under [tenv] and [uses] are applications of its
   continuation: the parameters are defined at the kinds that the
   handler gives them, and each is given the type of the closures that
   the applications pass for it, where there is one. [None] where the
   handler has not as many kinds as parameters, or a test of
   [define_typed_checked] fails. *)
Definition closures_union_env (tenv : Typing_env.t) (params : list variable)
    (metadata : handler_metadata) (uses : list (Typing_env.t * list simple))
    : option Typing_env.t :=
  let kinds := hm_params_kinds metadata in
  if Nat.eqb (length kinds) (length params)
  then
    define_typed_checked tenv
      (combine (combine params kinds) (parameter_unions tenv uses 0 kinds))
  else None.

(* With no application it is the environment of the handler. *)
Definition closures_union_env_no_use_statement : Prop :=
  ∀ (tenv : Typing_env.t) (params : list variable)
    (metadata : handler_metadata),
    closures_union_env tenv params metadata []
    = handler_env tenv params metadata.

(* ======================================================================== *)
(* The environment holds where an application enters the handler            *)
(* ======================================================================== *)

(* In a state at one of the applications, which satisfies the
   environment of that application and the environment of the
   [Let_cont], the handler is entered in a state that satisfies the
   environment made: the variables of the application with the
   parameters bound to the values of the arguments, which are of the
   kinds of the handler. A variable that may occur in terms at the
   [Let_cont] is to be one that may occur in terms at the
   application. *)
Definition closures_union_env_satisfied_statement : Prop :=
  ∀ (tenv : Typing_env.t) (params : list variable)
    (metadata : handler_metadata) (uses : list (Typing_env.t * list simple))
    (target : Typing_env.t) (use : Typing_env.t * list simple)
    (statics : static_env) (vars : env) (vs : list value)
    (handler_vars : env),
    closures_union_env tenv params metadata uses = Some target ->
    use ∈ uses ->
    (∀ y : variable,
      in_scope_in_terms tenv y -> in_scope_in_terms (fst use) y) ->
    env_satisfies_strictly tenv statics vars ->
    env_satisfies_strictly (fst use) statics vars ->
    eval_simples vars (snd use) = Some vs ->
    map kind_of_value vs = hm_params_kinds metadata ->
    bind_params params vs vars = Some handler_vars ->
    env_satisfies_strictly target statics handler_vars.

(* ======================================================================== *)
(* The handler rewritten                                                    *)
(* ======================================================================== *)

(* The body of the [Let_cont] is an [Apply_cont] or a [Switch]: the
   handler may be replaced by one that refines it under the
   environment made of the applications of the body. *)
Definition let_cont_handler_closures_union_statement : Prop :=
  ∀ (tenv : Typing_env.t) (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler handler' body : expr)
    (uses : list (Typing_env.t * list simple)) (target : Typing_env.t),
    leaf_uses tenv k body = Some uses ->
    closures_union_env tenv params metadata uses = Some target ->
    expr_refines_under target handler handler' ->
    expr_refines_under tenv
      (Let_cont_non_recursive k (Handler params metadata handler) body)
      (Let_cont_non_recursive k (Handler params metadata handler') body).

(* The applications gathered, under binders: each admits the
   environment, where the environment of the [Let_cont] is known to
   fit ([known_to_fit]: its invariant, and that its aliases are
   kinded) and the test of the binders passed holds
   (FlambdaSimplifyPruningSpec.v). *)
Definition gathered_closures_union_admitted_statement : Prop :=
  ∀ (tenv : Typing_env.t) (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler body : expr)
    (uses : list (Typing_env.t * list simple)) (xs : list variable)
    (ks : list continuation) (target : Typing_env.t)
    (use : Typing_env.t * list simple),
    known_to_fit tenv true ->
    gather_uses tenv k (Handler params metadata handler) body
    = Some (uses, xs, ks) ->
    gather_uses_keeps_fit tenv k (Handler params metadata handler) body
    = true ->
    closures_union_env tenv params metadata uses = Some target ->
    use ∈ uses ->
    application_admits params metadata (fst use) (snd use) target.

(* So the handler may be replaced by one that refines it under the
   environment, if what it becomes mentions none of the binders
   passed. *)
Definition gathered_closures_union_refines_statement : Prop :=
  ∀ (tenv : Typing_env.t) (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler handler' body : expr)
    (uses : list (Typing_env.t * list simple)) (xs : list variable)
    (ks : list continuation) (target : Typing_env.t),
    known_to_fit tenv true ->
    gather_uses tenv k (Handler params metadata handler) body
    = Some (uses, xs, ks) ->
    gather_uses_keeps_fit tenv k (Handler params metadata handler) body
    = true ->
    binders_not_free xs ks (Handler params metadata handler') = true ->
    closures_union_env tenv params metadata uses = Some target ->
    expr_refines_under target handler handler' ->
    expr_refines_under tenv
      (Let_cont_non_recursive k (Handler params metadata handler) body)
      (Let_cont_non_recursive k (Handler params metadata handler') body).

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

(* The environment made from the renamed environment, the renamed
   parameters and the renamed applications is the environment,
   renamed. The metadata contains no variable. *)
Definition closures_union_env_rename_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (params : list variable)
    (metadata : handler_metadata) (uses : list (Typing_env.t * list simple)),
    closures_union_env (rename_typing_env r tenv)
      (map (rename_variable r) params) metadata
      (map (rename_application r) uses)
    = option_map (rename_typing_env r)
        (closures_union_env tenv params metadata uses).

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

(* What the prover of code ids answers for the parameter of a handler
   under the environment made, computed. The variables are those of
   FlambdaTypingEnvExamplesSpec.v; a variable of which nothing is to be
   known is bound to [Opaque_identity] of a constant
   (FlambdaSimplifyPassSpec.example_opaque). *)

(* The environment after some [Let]s. *)
Fixpoint example_union_lets (tenv : Typing_env.t)
    (lets : list (variable * flambda_kind * named)) : option Typing_env.t :=
  match lets with
  | [] => Some tenv
  | (x, kind, defining_expr) :: lets =>
    match let_env tenv x kind defining_expr with
    | Some tenv' => example_union_lets tenv' lets
    | None => None
    end
  end.

(* [lets] are bound from the environment that knows nothing; then
   comes the [Let_cont] of the continuation 1, whose handler has the
   one parameter [example_p], of kind value, and whose body is [body].
   The answer is that of the prover for [example_p], under the
   environment made of the applications gathered in [body]. *)
Definition example_union_code_ids
    (lets : list (variable * flambda_kind * named)) (body : expr)
    : option (Proof_of_property.t (list code_id)) :=
  let k : continuation := 1%positive in
  let metadata := make_handler_metadata [Value] in
  match example_union_lets Typing_env.create lets with
  | Some tenv =>
    match gather_uses tenv k (Handler [example_p] metadata Invalid) body with
    | Some (uses, _, _) =>
      match closures_union_env tenv [example_p] metadata uses with
      | Some target =>
        Provers.prove_code_ids target (alias_type_of Value (Var example_p))
      | None => None
      end
    | None => None
    end
  | None => None
  end.

(* The closure at a slot of a set of one function. *)
Definition example_union_closure (slot : function_slot) (code : code_id)
    (value_slots : list (value_slot * simple)) : named :=
  Closure_of_set slot [(slot, code)] value_slots.

(* [apply_cont 1 s]. *)
Definition example_union_apply (s : simple) : expr :=
  Apply_cont (Apply_cont_expr 1%positive [s] None).

(* [let x = first in let y = second in let c = opaque 0 in let_cont 1
   (p) = .. in switch c | 0 -> apply_cont 1 x | 1 -> apply_cont 1 y]:
   nothing is known of the naked immediate [c], and the continuation is
   applied to [x] and to [y]. *)
Definition example_union_of_two (first second : named)
    : option (Proof_of_property.t (list code_id)) :=
  example_union_code_ids
    [(example_x, Value, first); (example_y, Value, second);
     (example_c, naked_immediate_kind,
      example_opaque naked_immediate_kind (Naked_immediate 0))]
    (Switch (Var example_c)
       [(0, Apply_cont_expr 1%positive [Var example_x] None);
        (1, Apply_cont_expr 1%positive [Var example_y] None)]).

(* Two closures at two slots, of the code 1 and of the code 2: the
   parameter is a closure of one of the two. *)
Definition example_union_two_slots_statement : Prop :=
  example_union_of_two (example_union_closure 1%positive 1%positive [])
    (example_union_closure 2%positive 2%positive [])
  = Some (Proof_of_property.Proved [2%positive; 1%positive]).

(* Two closures made alike, at one slot and of one code: the two cases
   are the same. *)
Definition example_union_same_case_statement : Prop :=
  example_union_of_two (example_union_closure 1%positive 1%positive [])
    (example_union_closure 1%positive 1%positive [])
  = Some (Proof_of_property.Proved [1%positive]).

(* Two sets of one function with no name for its slot, of different
   code: the two have the same slot and different cases, and the
   parameter is defined at its kind and no more. *)
Definition example_union_one_slot_two_codes_statement : Prop :=
  example_union_of_two (Set_of_closures 1%positive [])
    (Set_of_closures 2%positive [])
  = Some Proof_of_property.Unknown.

(* A closure and a value of which nothing is known: the parameter is
   defined at its kind and no more. *)
Definition example_union_not_a_closure_statement : Prop :=
  example_union_of_two (example_union_closure 1%positive 1%positive [])
    (example_opaque Value (Tagged_immediate 0))
  = Some Proof_of_property.Unknown.

(* [let y = opaque 0 in let_cont 1 (p) = .. in let x = closure {w = y}
   in apply_cont 1 x]: the closure is made after the [Let_cont], and
   its type speaks of [y], which is bound before it. *)
Definition example_union_slot_in_scope_statement : Prop :=
  example_union_code_ids
    [(example_y, Value, example_opaque Value (Tagged_immediate 0))]
    (Let example_x Value
       (example_union_closure 1%positive 1%positive
          [(example_w, Var example_y)])
       (example_union_apply (Var example_x)))
  = Some (Proof_of_property.Proved [1%positive]).

(* [let_cont 1 (p) = .. in let y = opaque 0 in let x = closure {w = y}
   in apply_cont 1 x]: [y] is bound after the [Let_cont], and the
   parameter is defined at its kind and no more. *)
Definition example_union_slot_out_of_scope_statement : Prop :=
  example_union_code_ids []
    (Let example_y Value (example_opaque Value (Tagged_immediate 0))
       (Let example_x Value
          (example_union_closure 1%positive 1%positive
             [(example_w, Var example_y)])
          (example_union_apply (Var example_x))))
  = Some Proof_of_property.Unknown.
