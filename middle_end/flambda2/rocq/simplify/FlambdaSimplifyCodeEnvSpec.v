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

(* The environment in which Simplify rewrites the body of code that a
   [Let_static] binds, with the types of the symbols.

   This file is to be reviewed: the definitions and the statements,
   against middle_end/flambda2/types/env/typing_env.ml ([closure_env]),
   middle_end/flambda2/simplify/env/downwards_env.ml
   ([enter_set_of_closures]) and
   middle_end/flambda2/simplify/simplify_set_of_closures.ml
   ([dacc_inside_function]). It follows FlambdaSimplifyStaticsSpec.v,
   which has the types that Simplify gives the symbols of a
   [Let_static].

   Simplify rewrites the body of a function in the environment of the
   place where its set of closures is, with the variables out of
   scope: [Typing_env.closure_env] ([DE.enter_set_of_closures]). The
   symbols keep their types there. A variable bound outside stays
   known to the types and may not occur in a term: a rewrite cannot
   put it in the term, and what the types say through it, that a field
   is a constant or a symbol say, can still be used. In that
   environment the parameters are defined at their kinds with no
   types, and [my_closure] at the kind value ([dacc_inside_function]).

   The function that puts the rewrites together
   (FlambdaSimplifyPassSpec.v) rewrites the body of code where a
   [Let_static] binds the code, under this environment
   ([code_env_in]): [Typing_env.closure_env] of the environment of the
   body of the [Let_static]
   (FlambdaSimplifyStaticsSpec.add_static_bindings), then [my_closure]
   and the parameters. So inside a function a load from a symbol bound
   to a block is answered.

   Where the model and the compiler part:
   - The compiler rewrites the body of code that is not a stub where
     a set of closures uses the code, which is in the group that binds
     the code or after it, and not where the code is bound. The
     environment there has the symbols of that group, with the types
     that FlambdaSimplifyStaticsSpec.v gives them, and may know more.
     The code of a stub it rewrites where the code is bound, unless
     stubs are to be simplified as other functions
     ([Simplify_set_of_closures.simplify_static_stub_function], called
     by [Simplify_static_const.simplify_static_consts]), in the
     environment as it is then: the symbols that the group binds to
     closures are defined, with no types; of the others, those bound
     before the code have their types, and those bound after are not
     defined. The model's code is not marked as a stub; its term has
     the code bound once and the model rewrites it once, under what is
     known just after the group. For a stub that is more than the
     compiler knows, by the types of the symbols that the group binds
     after the code to constants that are not closures.
   - The compiler defines the parameters and then [my_closure]; here
     [my_closure] is first, as the machine binds it
     (FlambdaSemanticsSpec.step, at an [Apply]). The compiler's code
     has region and depth variables too, which it defines after; the
     model's code has none.
   - [my_closure] gets no type here. The compiler gives it the type of
     the closure; that is for the statements about the types of
     closures.
   - [code_env_in] tests, before it defines each variable, that the
     environment passes the test of well-formedness
     (FlambdaTypingEnvInvariantSpec.typing_env_well_formed) and does
     not have the variable. The compiler has no such tests: its
     variables are fresh and its environments well formed by
     construction. Where a test fails there is no environment, and the
     code is to be left as it is. The tests are what the proofs need of
     the environment; nothing else is asked of it, so the statements
     are of every environment.

   The statements:
   - where [code_env_in] answers, the environment is
     [Typing_env.closure_env] with [my_closure] and the parameters
     defined ([code_env_in_statement]);
   - [Typing_env.closure_env] of an environment that a state satisfies
     is satisfied by the static environment of the state and no
     variables ([closure_env_env_satisfies_strictly_statement]);
   - so the environment of the code is satisfied at a call, in a
     static environment that satisfies the environment outside with
     some variables ([code_env_in_satisfied_statement]);
   - the environment of the body of a [Let_static] does not depend on
     the code that the group binds
     ([add_static_bindings_code_statement]);
   - the body of code that a [Let_static] binds may be rewritten under
     the environment of the code made from the environment of the body
     of the [Let_static] ([let_static_code_typed_statement]). Neither
     body is to bind something static: a call of code whose body does
     is undefined (FlambdaSemanticsSpec.call_function), so no rewrite
     of a body that can run is lost;
   - [code_env_in] commutes with renaming
     ([code_env_in_rename_statement]).
   - Two examples. In the body of code, the field of a static block
     that the group of the code binds, loaded from its symbol, is the
     constant that the field was given
     ([example_load_in_code_statement]). A call of a symbol that the
     group binds to a closure is the direct call of its code, in the
     body of the [Let_static] and in the code, where the function
     calls itself ([example_call_of_static_closure_statement]).

   Why the types of the symbols of the group hold in the code: the
   code runs at a call, a call is after the [Let_static] has been
   executed, and what a symbol is bound to never changes
   (FlambdaSemanticsSpec.v). The variables that those types name have
   the values that they had at the [Let_static]; the code cannot name
   them.

   Not here: the type of [my_closure].

   Nothing else is assumed of the term or of the environment than
   what the statements say: [code_env_in] answers, so that its tests
   have passed, and neither body of the code binds something static.

   The proofs are in FlambdaSimplifyCodeEnvProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaValueSpec FlambdaPrimitiveModel FlambdaSemanticsSpec
  FlambdaNominalSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaTypingEnv
  FlambdaTypingEnvSpec FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvExamplesSpec FlambdaTypingEnvRefinementSpec
  FlambdaTypeRenameSpec FlambdaEnvRenameSpec FlambdaSimplifyStaticsSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* The environment of the body of code                                      *)
(* ======================================================================== *)

(* Variables are defined in order at their kinds, with the normal name
   mode and no types, each after the two tests: the environment is well
   formed, and does not have the variable. [None] where a test
   fails. *)
Fixpoint define_variables_checked (tenv : Typing_env.t)
    (definitions : list (variable * flambda_kind)) : option Typing_env.t :=
  match definitions with
  | [] => Some tenv
  | (x, kind) :: definitions =>
    if
      typing_env_well_formed tenv
      && negb (Typing_env.mem None tenv (Name_var x))
    then
      define_variables_checked
        (Typing_env.add_variable_definition tenv x kind Name_mode.normal)
        definitions
    else None
  end.

(* The environment under which the body of code is rewritten, where
   the code is bound under [tenv]: the variables of [tenv] go out of
   scope ([Typing_env.closure_env]); then [my_closure] is defined, a
   value, and the parameters, at the kinds of the arity of the code.
   This is what the machine binds at a call (FlambdaSemanticsSpec.step,
   at an [Apply]): the callee and the arguments, which a call that has
   a step gives the kinds of the arity. [None] where the arity has not
   as many kinds as there are parameters, or a test of
   [define_variables_checked] fails: [my_closure] and the parameters
   are then not different variables that [tenv] does not have, or an
   environment on the way is not well formed. *)
Definition code_env_in (tenv : Typing_env.t) (params : list variable)
    (my_closure : variable) (metadata : code_metadata)
    : option Typing_env.t :=
  let kinds := unarize (cm_params_arity metadata) in
  if Nat.eqb (length kinds) (length params)
  then
    define_variables_checked (Typing_env.closure_env tenv)
      ((my_closure, Value) :: combine params kinds)
  else None.

(* Where it answers, the environment is this one. *)
Definition code_env_in_statement : Prop :=
  ∀ (tenv : Typing_env.t) (params : list variable) (my_closure : variable)
    (metadata : code_metadata) (tenv_code : Typing_env.t),
    code_env_in tenv params my_closure metadata = Some tenv_code ->
    tenv_code
    = Typing_env.add_definitions_of_params
        (Typing_env.add_variable_definition (Typing_env.closure_env tenv)
           my_closure Value Name_mode.normal)
        (combine params (unarize (cm_params_arity metadata))).

(* ======================================================================== *)
(* It is satisfied where the code runs                                      *)
(* ======================================================================== *)

(* With the variables out of scope, the environment asks nothing of
   the variables of the machine: it is satisfied with none, by the
   static environment that satisfied it with some. The test is that
   of [code_env_in] before it defines [my_closure]. *)
Definition closure_env_env_satisfies_strictly_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars : env),
    typing_env_well_formed (Typing_env.closure_env tenv) = true ->
    env_satisfies_strictly tenv statics vars ->
    env_satisfies_strictly (Typing_env.closure_env tenv) statics [].

(* The environment of the code is satisfied at a call: under the
   parameters bound to arguments of the kinds of the arity and
   [my_closure] to a value of the kind value, in a static environment
   that satisfies [tenv] with some variables. *)
Definition code_env_in_satisfied_statement : Prop :=
  ∀ (tenv : Typing_env.t) (params : list variable) (my_closure : variable)
    (metadata : code_metadata) (tenv_code : Typing_env.t)
    (statics : static_env) (vars : env) (vs : list value) (callee : value)
    (callee_vars : env),
    code_env_in tenv params my_closure metadata = Some tenv_code ->
    env_satisfies_strictly tenv statics vars ->
    map kind_of_value vs = unarize (cm_params_arity metadata) ->
    kind_of_value callee = Value ->
    bind_params params vs [(my_closure, callee)] = Some callee_vars ->
    env_satisfies_strictly tenv_code statics callee_vars.

(* ======================================================================== *)
(* The body of code that a [Let_static] binds                               *)
(* ======================================================================== *)

(* The environment of the body of a [Let_static] is made from the
   symbols that the group binds and their constants: it is the same
   whatever the code. *)
Definition add_static_bindings_code_statement : Prop :=
  ∀ (tenv : Typing_env.t) (before : list static_binding) (id : code_id)
    (c c' : code) (after : list static_binding),
    add_static_bindings tenv (before ++ Bind_code id c :: after)
    = add_static_bindings tenv (before ++ Bind_code id c' :: after).

(* The body of code that a [Let_static] binds is rewritten under the
   environment of the code made from the environment of the body of
   the [Let_static]: the symbols that the group binds have their types
   in the code. *)
Definition let_static_code_typed_statement : Prop :=
  ∀ (tenv : Typing_env.t) (before : list static_binding) (id : code_id)
    (params : list variable) (my_closure : variable)
    (ret exn : continuation) (metadata : code_metadata)
    (code_body code_body' : expr) (after : list static_binding)
    (body : expr) (tenv_code : Typing_env.t),
    code_env_in
      (add_static_bindings tenv
         (before
          ++ Bind_code id (Code params my_closure ret exn metadata code_body)
          :: after))
      params my_closure metadata
    = Some tenv_code ->
    binds_statics code_body = false ->
    binds_statics code_body' = false ->
    expr_refines_under tenv_code code_body code_body' ->
    expr_refines_under tenv
      (Let_static
         (before
          ++ Bind_code id (Code params my_closure ret exn metadata code_body)
          :: after)
         body)
      (Let_static
         (before
          ++ Bind_code id
               (Code params my_closure ret exn metadata code_body')
          :: after)
         body).

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

(* The environment of the code made from the renamed environment and
   the renamed variables is the environment of the code, renamed. The
   metadata contains no variable. *)
Definition code_env_in_rename_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (params : list variable)
    (my_closure : variable) (metadata : code_metadata),
    code_env_in (rename_typing_env r tenv) (map (rename_variable r) params)
      (rename_variable r my_closure) metadata
    = option_map (rename_typing_env r)
        (code_env_in tenv params my_closure metadata).

(* ======================================================================== *)
(* An example                                                               *)
(* ======================================================================== *)

(* The variables, the symbol and the code id are those of
   FlambdaTypingEnvExamplesSpec.v.

   [let_static s = Block 0 (7) and code f (x) = (let q = Block_load s 0
   in apply_cont ret q) in body], where [x] is a parameter of kind
   value: in the code the symbol [s] has the type of the blocks of tag
   0 whose field is 7; of that type the prover answers that the field
   0 is 7; [q] is made equal to 7, and replaced by it. *)
Definition example_load_in_code_statement : Prop :=
  ∀ (ret exn : continuation) (body : expr),
    let bindings (code_body : expr) : list static_binding :=
      [Bind_symbol example_s
         (Static_const_block 0 [Const (Tagged_immediate 7)]);
       Bind_code example_f
         (Code [example_x] example_my_closure ret exn
            (make_code_metadata [Singleton Value] [Singleton Value])
            code_body)]
    in
    expr_refines_under Typing_env.create
      (Let_static
         (bindings
            (Let example_q Value
               (Prim
                  (Unary
                     (Block_load
                        (Block_access_values (Some 0) (Some 1) Any_value)
                        Immutable 0))
                  [Symbol example_s])
               (Apply_cont (Apply_cont_expr ret [Var example_q] None))))
         body)
      (Let_static
         (bindings
            (Let example_q Value (Simple (Const (Tagged_immediate 7)))
               (Apply_cont
                  (Apply_cont_expr ret [Const (Tagged_immediate 7)] None))))
         body).

(* A call of [callee] with one argument, that takes one value and
   returns one. *)
Definition example_call (callee : simple) (kind : call_kind) (arg : simple)
    (k exn : continuation) : expr :=
  Apply
    (Apply_expr callee kind [arg] k exn [] [Singleton Value]
       [Singleton Value]).

(* [let rec f x = f x] at toplevel, and a call of it:
   [let_static s = the closure of the code f and code f (x) = (apply s
   x) in apply s 7], the two calls indirect. The symbol [s] is given
   the type of the closures of the code [f]; of that type the prover
   proves the entry
   (FlambdaSimplifyApplySpec.simplify_apply_refines_under_statement);
   the call in the body is the direct call of the code, and so is the
   call in the code. *)
Definition example_call_of_static_closure_statement : Prop :=
  ∀ (k exn ret exn' : continuation),
    let bindings (kind : call_kind) : list static_binding :=
      [Bind_symbol example_s (Static_const_set_of_closures example_f []);
       Bind_code example_f
         (Code [example_x] example_my_closure ret exn'
            (make_code_metadata [Singleton Value] [Singleton Value])
            (example_call (Symbol example_s) kind (Var example_x) ret exn'))]
    in
    expr_refines_under Typing_env.create
      (Let_static (bindings Indirect)
         (example_call (Symbol example_s) Indirect
            (Const (Tagged_immediate 7)) k exn))
      (Let_static (bindings (Direct example_f))
         (example_call (Symbol example_s) (Direct example_f)
            (Const (Tagged_immediate 7)) k exn)).
