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


(* Simplifying a call from the type of its callee: an indirect call
   becomes a direct call.

   This file is to be reviewed: the definitions and the statements,
   against middle_end/flambda2/simplify/simplify_apply_expr.ml
   ([simplify_apply_shared], [simplify_function_call],
   [simplify_direct_function_call]) and, for the type of a closure
   that is made, simplify_set_of_closures.ml. Refinement relative to a
   typing environment is FlambdaTypingEnvRefinementSpec.v; the type of
   an argument and the function that computes it are
   FlambdaSimplifyPrimitivesSpec.v; the provers and what their answers
   mean are FlambdaProvers.v and FlambdaProversSpec.v; what a call
   does is FlambdaSemanticsSpec.v ([call_function]).

   What Simplify does at an [Apply] of an OCaml function.
   - [simplify_apply_shared] replaces the callee and the arguments by
     their canonical elements ([Simplify_simple.simplify_simple]), and
     makes the call [Invalid] where the type of an argument is not of
     the kind that the arity of the call gives it.
   - [simplify_function_call] asks
     [Provers.meet_single_closures_entry] of the type of the callee.
     - Where it answers an entry, and the environment has the
       metadata of the code id of the function type of the entry
       ([find_code_metadata_exn]; if it has not, the call is treated
       as one of whose callee nothing is known), that code is the
       code that the call runs, and [simplify_direct_function_call]
       makes the call a direct call
       ([Call_kind.direct_function_call]). Where the call names code
       already (a direct call, or an indirect call of known arity
       that lists code ids), the code ids are met with the one of
       the type in the code age relation: the call is [Invalid] if
       the meet is bottom; the code id of the meet is taken if it is
       one code id whose metadata the environment has, and the code
       id of the type otherwise. Then the call of a tupled function
       is adapted to the calling convention of its code, the
       arities of the call and of the code are compared, an
       over-application is split, a partial application is made a
       closure, and it is decided whether to inline.
     - Where it answers that the type has no closure, the call
       becomes [Invalid].
     - Where it answers neither, the call is left; but an indirect
       call of unknown arity whose callee is, by
       [Provers.meet_code_ids], among pieces of code that are not
       tupled and all have the arity of the call is marked as an
       indirect call of known arity with those code ids.

   What is here: the canonical elements, and the step from an entry
   to the direct call, for a call that is indirect. Not here: the
   rejection of a call on the kinds of its arguments; the metadata of
   the code (the model's environments have none: the code id answered
   is used as it is); the code age relation, and so the calls that
   name code already; the calling convention of tupled functions;
   the arities, over-application and partial application; inlining;
   the mark of an indirect call of known arity, a call kind that the
   model has not; the answer that the type has no closure.

   The statements.
   - From the type to the code ([callee_code_proved_statement]). In a
     state that satisfies the environment strictly, if
     [Provers.prove_single_closures_entry] proves an entry of the type
     of the callee, the value of the callee is a closure whose code is
     the code id of the function type answered.
   - So the indirect call is equivalent to the direct call of that
     code ([apply_direct_proved_statement]). That an indirect call of
     a closure of known code is the direct call of the code is
     [apply_indirect_known_statement] of
     FlambdaApplySpec.v; this is the step from the type to
     "known".
   - The callee and the arguments may be replaced by their canonical
     elements ([apply_canonical_statement]).
   - One function that does both, [simplify_apply], for the function
     that assembles the rewrites from the types
     (FlambdaSimplifyPassSpec.v, which leaves a call as it is today).
     What it answers is equivalent to the call under the environment,
     with no premise ([simplify_apply_equivalent_under_statement]),
     and it commutes with renaming ([simplify_apply_rename_statement]).
   - The type of a closure that is made ([closure_of_set_type],
     [set_of_closures_type]): the closure at its function slot of a
     set that has exactly the functions of the defining expression,
     each of the code named, and its value slots, each of the type of
     its [Simple]. The value of the defining expression has that type
     ([closure_of_set_typed_statement],
     [set_of_closures_typed_statement]), so the body of its [Let] may
     be rewritten under the environment in which the variable has it
     ([let_closure_of_set_statement], [let_set_of_closures_statement]).
     Its variables are those of the types of the value slots, and it
     is renamed by renaming them
     ([closure_of_set_type_free_variables_statement],
     [closure_of_set_type_rename_statement]): what the function that
     assembles the rewrites needs of a type that it gives a variable.
     Simplify gives more: the types of the closures of the set (each
     an alias of the name that the closure is bound to), and the mode
     of the allocation.
   - Three small terms, from OCaml functions, rewritten with these
     from the environment that knows only the parameter: a local
     function called through its variable
     ([example_local_function_statement]), a closure read from a pair
     ([example_closure_from_pair_statement]), and a function that is
     the argument of an inlined call
     ([example_function_argument_statement]). In each the prover
     proves the entry of the type that the closure was given, and the
     call that the function answers is the direct call. The terms are
     written by hand after what the compiler makes of the functions;
     none was taken from a run of it.

   The prover asked. Simplify asks the meet shortcut
   [Provers.meet_single_closures_entry]; the statements here are of
   [Provers.prove_single_closures_entry], which reads the type in the
   same way and answers [Proved] on fewer types: not on a type that
   has values that are not closures (a closure or null). With
   [Proved], every value of the type is a closure of the code, and
   the equivalence follows from [apply_indirect_known_statement]
   alone, whatever a call does of a callee that is not a closure.
   With the shortcut's answers (an entry, or no closure) one needs
   besides that a call whose callee is not a closure is undefined.
   That is so of the machine as it is, and will not be once a partial
   application is a value that may be called (the change of the
   machine for partial applications and over-applications, which is
   being made). So the two statements for the shortcut, of which one
   gives [Invalid], are left for after that change: they need that a
   partial application has no type on which the shortcut answers.

   The number of arguments. Nothing here looks at the arities: the
   equivalence is of the call as it is, with as many arguments as it
   has. Where the machine gives a call with fewer or more arguments
   than the code has parameters no step (today), it gives none to
   either call; where it runs them as a partial application or an
   over-application (after the change above), a direct call of a
   closure of the code is the generic application that the indirect
   one is, and the statement of FlambdaApplySpec.v that is
   used here is kept for any number of arguments.

   Code ids. The code answered is the very code of the closure
   ([single_closures_entry_holds], FlambdaProversSpec.v, which has the
   CR): the model has no code age relation. So a call that is direct
   already, whose code id Simplify meets with the one of the type as
   said above, keeps its code id here.

   What is assumed.
   - Strict satisfaction ([env_satisfies_strictly]), as for every
     statement that reads a type through a prover
     (FlambdaSimplifyPrimitivesSpec.v says why).
   - Nothing of kinds, and nothing of the well-formedness of types:
     the prover's statement is of every type.

   The proofs are in FlambdaSimplifyApplyProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec FlambdaNominalSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypeRenameSpec FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaTypingEnv
  FlambdaTypingEnvSpec FlambdaTypingEnvInvariantSpec FlambdaEnvRenameSpec
  FlambdaTypingEnvExamplesSpec FlambdaTypingEnvRefinementSpec
  FlambdaProvers FlambdaProversSpec FlambdaSimplifyPrimitivesSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* From the type of the callee to its code                                  *)
(* ======================================================================== *)

(* In a state that satisfies the environment, a callee of whose type
   the prover proves an entry is a closure of the code of the function
   type answered. *)
Definition callee_code_proved_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars : env)
    (callee : simple) (ty : flambda_type) (slot : function_slot)
    (alloc_mode : Alloc_mode.t) (entry : closures_entry)
    (function_type : function_type) (v : value),
    env_satisfies_strictly tenv statics vars ->
    argument_type tenv callee ty ->
    Provers.prove_single_closures_entry tenv ty
    = Some
        (Proof_of_property.Proved (slot, alloc_mode, entry, function_type)) ->
    eval_simple vars callee = Some v ->
    ∃ value_slots : list (value_slot * value),
      closure_of statics v
      = Some (function_type_code_id function_type, value_slots).

(* ======================================================================== *)
(* The call                                                                 *)
(* ======================================================================== *)

(* A call with another call kind. *)
Definition with_call_kind (ap : apply_expr) (kind : call_kind) : apply_expr :=
  Apply_expr (ap_callee ap) kind (ap_args ap) (ap_continuation ap)
    (ap_exn_continuation ap) (ap_exn_extra_args ap) (ap_args_arity ap)
    (ap_return_arity ap).

(* A call with another callee and other arguments. *)
Definition with_callee_and_args (ap : apply_expr) (callee : simple)
    (args : list simple) : apply_expr :=
  Apply_expr callee (ap_call_kind ap) args (ap_continuation ap)
    (ap_exn_continuation ap) (ap_exn_extra_args ap) (ap_args_arity ap)
    (ap_return_arity ap).

(* [Simplify_apply_expr.simplify_direct_function_call], the call kind:
   an indirect call of a callee of whose type the prover proves an
   entry is the direct call of the code of the function type
   answered. *)
Definition apply_direct_proved_statement : Prop :=
  ∀ (tenv : Typing_env.t) (ap : apply_expr) (ty : flambda_type)
    (slot : function_slot) (alloc_mode : Alloc_mode.t)
    (entry : closures_entry) (function_type : function_type),
    ap_call_kind ap = Indirect_unknown_arity ->
    argument_type tenv (ap_callee ap) ty ->
    Provers.prove_single_closures_entry tenv ty
    = Some
        (Proof_of_property.Proved (slot, alloc_mode, entry, function_type)) ->
    expr_equivalent_under tenv (Apply ap)
      (Apply
         (with_call_kind ap (Direct (function_type_code_id function_type)))).

(* [Simplify_apply_expr.simplify_apply_shared]: the callee and the
   arguments may be replaced by their canonical elements. *)
Definition apply_canonical_statement : Prop :=
  ∀ (tenv : Typing_env.t) (ap : apply_expr) (callee : simple)
    (args : list simple),
    canonical_in_terms tenv (ap_callee ap) callee ->
    Forall2 (canonical_in_terms tenv) (ap_args ap) args ->
    expr_equivalent_under tenv (Apply ap)
      (Apply (with_callee_and_args ap callee args)).

(* ======================================================================== *)
(* The function                                                             *)
(* ======================================================================== *)

(* The call kind that the type of the callee gives a call. A direct
   call keeps its code id. *)
(* CR mshinwell: Simplify asks the meet shortcut
   [Provers.meet_single_closures_entry] here, and this asks
   [Provers.prove_single_closures_entry] ("The prover asked", in the
   header). So a call whose callee may be null is left indirect, and
   no call becomes [Invalid]. The choice was made so that the
   statement does not rest on what a call does of a callee that is
   not a closure, which changes when a partial application becomes a
   value that may be called. *)
Definition direct_call_kind (tenv : Typing_env.t) (ap : apply_expr)
    : call_kind :=
  match ap_call_kind ap with
  | Direct code => Direct code
  | Indirect_unknown_arity =>
    match simplify_simple tenv (ap_callee ap) with
    | Some (ty, _) =>
      match Provers.prove_single_closures_entry tenv ty with
      | Some (Proof_of_property.Proved (_, _, _, function_type)) =>
        Direct (function_type_code_id function_type)
      | Some Proof_of_property.Unknown | None => Indirect_unknown_arity
      end
    | None => Indirect_unknown_arity
    end
  end.

(* The callee and the arguments replaced by their canonical elements,
   where the environment knows them all. *)
Definition canonical_apply (tenv : Typing_env.t) (ap : apply_expr)
    : apply_expr :=
  match
    simplify_simple tenv (ap_callee ap), simplify_simples tenv (ap_args ap)
  with
  | Some (_, callee), Some answers =>
    with_callee_and_args ap callee (map snd answers)
  | _, _ => ap
  end.

(* A call, rewritten from the types. *)
Definition simplify_apply (tenv : Typing_env.t) (ap : apply_expr) : expr :=
  Apply (canonical_apply tenv (with_call_kind ap (direct_call_kind tenv ap))).

(* What the function answers is equivalent to the call, in every state
   that satisfies the environment; so it refines it. *)
Definition simplify_apply_equivalent_under_statement : Prop :=
  ∀ (tenv : Typing_env.t) (ap : apply_expr),
    expr_equivalent_under tenv (Apply ap) (simplify_apply tenv ap).

Definition simplify_apply_refines_under_statement : Prop :=
  ∀ (tenv : Typing_env.t) (ap : apply_expr),
    expr_refines_under tenv (Apply ap) (simplify_apply tenv ap).

(* The function commutes with renaming. *)
Definition simplify_apply_rename_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (ap : apply_expr),
    simplify_apply (rename_typing_env r tenv) (rename_apply r ap)
    = rename_expr r (simplify_apply tenv ap).

(* ======================================================================== *)
(* A closure that is made                                                   *)
(* ======================================================================== *)

(* The type that Simplify gives the variable of a closure of a set that
   is made, without what it says of the other closures of the set: the
   closure at [slot] of a set that has exactly the functions
   [function_decls], each of the code named, and the value slots
   [slots], each of the type that [tys] has for it. *)
Definition closure_of_set_type (slot : function_slot)
    (function_decls : list (function_slot * code_id))
    (slots : list value_slot) (tys : list flambda_type) : flambda_type :=
  exactly_this_closure slot
    (map_values (fun code => Or_unknown.Known (Function_type code))
       function_decls)
    [] (combine slots tys) Alloc_mode.Heap_or_local.

(* The same of a set of one function. *)
Definition set_of_closures_type (code : code_id) (slots : list value_slot)
    (tys : list flambda_type) : flambda_type :=
  closure_of_set_type unnamed_function_slot [(unnamed_function_slot, code)]
    slots tys.

(* The value of the defining expression has that type, the types of
   the value slots being those of their [Simple]s. *)
Definition closure_of_set_typed_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars : env)
    (slot : function_slot) (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * simple)) (tys : list flambda_type)
    (v : value),
    env_satisfies tenv statics vars ->
    Forall2 (argument_type tenv) (map snd value_slots) tys ->
    eval_named statics vars (Closure_of_set slot function_decls value_slots)
    = Some v ->
    type_satisfied statics vars v
      (closure_of_set_type slot function_decls (map fst value_slots) tys).

Definition set_of_closures_typed_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars : env)
    (code : code_id) (value_slots : list (value_slot * simple))
    (tys : list flambda_type) (v : value),
    env_satisfies tenv statics vars ->
    Forall2 (argument_type tenv) (map snd value_slots) tys ->
    eval_named statics vars (Set_of_closures code value_slots) = Some v ->
    type_satisfied statics vars v
      (set_of_closures_type code (map fst value_slots) tys).

(* The variables of the type are those of the types of the value
   slots. *)
Definition closure_of_set_type_free_variables_statement : Prop :=
  ∀ (slot : function_slot) (function_decls : list (function_slot * code_id))
    (slots : list value_slot) (tys : list flambda_type) (y : variable),
    y ∈ free_variables (closure_of_set_type slot function_decls slots tys) ->
    ∃ ty : flambda_type, ty ∈ tys ∧ y ∈ free_variables ty.

(* The type renamed is the type of the renamed types of the value
   slots. *)
Definition closure_of_set_type_rename_statement : Prop :=
  ∀ (r : renaming) (slot : function_slot)
    (function_decls : list (function_slot * code_id))
    (slots : list value_slot) (tys : list flambda_type),
    rename_type r (closure_of_set_type slot function_decls slots tys)
    = closure_of_set_type slot function_decls slots (map (rename_type r) tys).

(* So the body of the [Let] is rewritten under the environment in
   which the variable is defined with that type. *)
Definition let_closure_of_set_statement : Prop :=
  ∀ (tenv : Typing_env.t) (x : variable) (declared : flambda_kind)
    (slot : function_slot) (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * simple)) (tys : list flambda_type)
    (body body' : expr),
    typing_env_well_formed tenv = true ->
    Typing_env.mem None tenv (Name_var x) = false ->
    Forall2 (argument_type tenv) (map snd value_slots) tys ->
    let closure := Closure_of_set slot function_decls value_slots in
    expr_refines_under
      (Typing_env.replace_equation
         (Typing_env.add_variable_definition tenv x Value Name_mode.normal)
         (Name_var x)
         (closure_of_set_type slot function_decls (map fst value_slots) tys))
      body body' ->
    expr_refines_under tenv (Let x declared closure body)
      (Let x declared closure body').

Definition let_set_of_closures_statement : Prop :=
  ∀ (tenv : Typing_env.t) (x : variable) (declared : flambda_kind)
    (code : code_id) (value_slots : list (value_slot * simple))
    (tys : list flambda_type) (body body' : expr),
    typing_env_well_formed tenv = true ->
    Typing_env.mem None tenv (Name_var x) = false ->
    Forall2 (argument_type tenv) (map snd value_slots) tys ->
    let closure := Set_of_closures code value_slots in
    expr_refines_under
      (Typing_env.replace_equation
         (Typing_env.add_variable_definition tenv x Value Name_mode.normal)
         (Name_var x) (set_of_closures_type code (map fst value_slots) tys))
      body body' ->
    expr_refines_under tenv (Let x declared closure body)
      (Let x declared closure body').

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

(* The argument, the closure, the pair and the variable that is
   called; the code of the closure and its value slot. *)
Definition call_x : variable := 1%positive.
Definition call_g : variable := 2%positive.
Definition call_p : variable := 3%positive.
Definition call_h : variable := 4%positive.
Definition call_code : code_id := 1%positive.
Definition call_slot : value_slot := 1%positive.

(* The environment in which [x] is defined at the kind value and
   nothing more is known of it, as for the parameter of a function. *)
Definition call_env : Typing_env.t :=
  Typing_env.add_definitions_of_params Typing_env.create [(call_x, Value)].

(* A call with the argument [x], that takes one value and returns
   one. *)
Definition call_of (callee : simple) (kind : call_kind)
    (k exn : continuation) : expr :=
  Apply
    (Apply_expr callee kind [Var call_x] k exn [] [Singleton Value]
       [Singleton Value]).

(* A closure of the code, whose value slot holds [x]. *)
Definition call_closure : named :=
  Set_of_closures call_code [(call_slot, Var call_x)].

(* [let f x = let g y = y + x in g x]: a local function called through
   its variable. The variable is given the type of the closure, the
   prover proves the entry of that type, and the call is direct. *)
Definition example_local_function_statement : Prop :=
  ∀ k exn : continuation,
    expr_refines_under call_env
      (Let call_g Value call_closure (call_of (Var call_g)
        Indirect_unknown_arity k exn))
      (Let call_g Value call_closure
         (call_of (Var call_g) (Direct call_code) k exn)).

(* [let f x = let g y = y + x in let p = (g, x) in (fst p) x]: a
   closure read from a pair. The load is the field of the pair
   (FlambdaSimplifyProjectionsSpec.v), its variable is made equal to
   [g], and the call is a direct call of [g]. *)
Definition example_closure_from_pair_statement : Prop :=
  ∀ k exn : continuation,
    let pair :=
      Prim (Variadic (Make_block (Block_kind_values 0)))
        [Var call_g; Var call_x]
    in
    let load :=
      Prim
        (Unary
           (Block_load (Block_access_values (Some 0) (Some 2) Any_value)
              Immutable 0))
        [Var call_p]
    in
    expr_refines_under call_env
      (Let call_g Value call_closure
         (Let call_p Value pair
            (Let call_h Value load (call_of (Var call_h) Indirect_unknown_arity
              k exn))))
      (Let call_g Value call_closure
         (Let call_p Value pair
            (Let call_h Value (Simple (Var call_g))
               (call_of (Var call_g) (Direct call_code) k exn)))).

(* [let apply h x = h x] inlined at a call [apply g x], [g] being the
   local function above: the parameter [h] is bound to [g] by a [Let]
   (the inlined body of FlambdaApplySpec.v). The variable
   is made equal to [g], and the call is a direct call of [g]. *)
Definition example_function_argument_statement : Prop :=
  ∀ k exn : continuation,
    expr_refines_under call_env
      (Let call_g Value call_closure
         (Let call_h Value (Simple (Var call_g))
            (call_of (Var call_h) Indirect_unknown_arity k exn)))
      (Let call_g Value call_closure
         (Let call_h Value (Simple (Var call_g))
            (call_of (Var call_g) (Direct call_code) k exn))).
