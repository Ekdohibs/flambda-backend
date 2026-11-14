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


(* Refinement of terms relative to a typing environment.

   This file is to be reviewed: the definitions and the statements. The
   typing environment is FlambdaTypingEnv.v; what it is for the machine
   to satisfy one is FlambdaTypingEnvSpec.v ([env_satisfies_strictly]);
   the invariant is FlambdaTypingEnvInvariantSpec.v; refinement of
   states is ../../rocq/FlambdaSemanticsSpec.v.

   Simplify rewrites a term by what its typing environment says of the
   names in scope. Such a rewrite is not right under every environment
   of the machine, only under those that the typing environment
   describes: [expr_refines_under tenv source target] is refinement
   under every state of the machine whose variables and static
   environment satisfy [tenv] strictly. It is to replace
   [expr_refines_in] of FlambdaTypesSpec.v, which is stated of the
   first, naive model of types; that file stays until what uses it has
   moved.

   Strict satisfaction is the notion that the statements about types
   are made of (meet, join, the provers), so a rewrite that reads a
   type is justified under it. After variant unboxing the environments
   that Simplify builds are satisfied only where poison is exempt
   (decision D26), and the statements here do not apply there; the two
   about canonical elements do not read a type and hold of that weaker
   notion too ([type_simple_in_term_canonical_statement] has it as its
   premise).

   The statements:
   - how the notion compares with refinement under all environments,
     which it is under the environment that knows nothing, and that
     nothing is asked under an environment that is bottom;
   - a [Simple] may be replaced by the canonical element that the
     environment answers for it at the normal name mode, in the
     arguments of an [Apply_cont] and in a [Let], and a [Switch] on a
     [Simple] whose canonical element is a constant is its arm;
   - in the machine, what defining a variable, defining parameters,
     giving a variable in scope a type and making two [Simple]s equal
     do to satisfaction: the steps by which Simplify's environment
     follows the term; and that the value of a [Simple] of the term is
     of the kind at which the environment has the [Simple];
   - going under a [Let]: the body is rewritten under the environment
     with the variable defined, at the kind of the value that the
     machine binds it to, and with a type if every such value has it,
     or made equal to the [Simple] that it is bound to;
   - going into the body of a [Let_cont] changes nothing; going into a
     handler defines its parameters, and the same of one handler of a
     recursive [Let_cont];
   - three small terms rewritten with these, from the environment that
     knows nothing: a constant propagated through two [Let]s, a
     [Switch] on a variable bound to a constant, and a loop that is
     left on its second turn with a constant, propagated into a handler
     inside the handler of the loop. They are there to show that the
     premises of the statements can be met and that the statements fit
     together; what they say of the terms is plain.

   Kinds. A [Let] gives the kind of its variable and a handler the
   kinds of its parameters (../../rocq/FlambdaModel.v), and the machine
   tests them where it binds: a [Let] has no step unless the value of
   its defining expression is of the kind that the [Let] gives, and a
   handler none unless the values that it is given are of the kinds of
   its parameters (../../rocq/FlambdaWellKindedSpec.v,
   [let_binds_at_kind_statement], [handler_binds_at_kinds_statement]).
   Most statements here were made before the machine tested, and do
   not read those kinds; they hold as they are, and ask more than they
   need to. The body of a [Let] is rewritten under the environment
   with the variable defined at the kind of the value that the
   defining expression gives, which is a premise; a handler under the
   environments at every choice of kinds for its parameters. Simplify
   defines a variable at the kind that its binder gives
   ([Typing_env.add_variable_definition] is a fatal error otherwise).
   For a [Let] that is [let_declared_refines_under_statement], which
   has no premise on the defining expression: the test of the machine
   gives the kind. For a handler it is one of the choices; the
   statement at the kinds of the parameters alone is to come.

   The variable of a [Let] and the parameters of a handler are not to
   be bound in the environment: a binder that shadowed would invalidate
   what is known. That is Simplify's rule that bound names are fresh.

   The proofs are in FlambdaTypingEnvRefinementProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaEnvAliasesSpec FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvInvariantSpec FlambdaTypingEnvKindedSpec
  FlambdaTypingEnvExamplesSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* Refinement relative to a typing environment                              *)
(* ======================================================================== *)

Definition expr_refines_under (tenv : Typing_env.t) (source target : expr)
    : Prop :=
  ∀ (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
    (rd : read_record),
    env_satisfies_strictly tenv statics vars ->
    refines (State source vars conts statics hp rd)
      (State target vars conts statics hp rd).

Definition expr_equivalent_under (tenv : Typing_env.t) (e e' : expr) : Prop :=
  ∀ (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
    (rd : read_record),
    env_satisfies_strictly tenv statics vars ->
    equivalent (State e vars conts statics hp rd)
      (State e' vars conts statics hp rd).

(* Refinement under all environments is refinement under any typing
   environment. *)
Definition expr_refines_under_weaken_statement : Prop :=
  ∀ (tenv : Typing_env.t) (source target : expr),
    expr_refines source target -> expr_refines_under tenv source target.

(* Under the environment that knows nothing it is refinement under all
   environments: where Simplify starts. *)
Definition expr_refines_under_create_statement : Prop :=
  ∀ source target : expr,
    expr_refines_under Typing_env.create source target
    <-> expr_refines source target.

Definition expr_equivalent_under_refines_statement : Prop :=
  ∀ (tenv : Typing_env.t) (e e' : expr),
    expr_equivalent_under tenv e e' -> expr_refines_under tenv e e'.

Definition expr_refines_under_trans_statement : Prop :=
  ∀ (tenv : Typing_env.t) (e e' e'' : expr),
    expr_refines_under tenv e e' ->
    expr_refines_under tenv e' e'' ->
    expr_refines_under tenv e e''.

(* Nothing has to be preserved under an environment that is bottom:
   nothing satisfies it. *)
Definition bottom_refines_under_statement : Prop :=
  ∀ (tenv : Typing_env.t) (e e' : expr),
    Typing_env.is_bottom tenv = true -> expr_equivalent_under tenv e e'.

(* ======================================================================== *)
(* Canonical elements                                                       *)
(* ======================================================================== *)

(* The canonical element that the environment answers for a [Simple] of
   a term, at the normal name mode ([Simplify_simple.simplify_simple]). *)
Definition canonical_in_terms (tenv : Typing_env.t) (s canonical : simple)
    : Prop :=
  Typing_env.mem_simple (Some Name_mode.normal) tenv s = true
  ∧ ∃ ty : flambda_type,
      Typing_env.type_simple_in_term_exn tenv (Some Name_mode.normal) s
      = Some (Some (ty, canonical)).

Definition apply_cont_canonical_statement : Prop :=
  ∀ (tenv : Typing_env.t) (k : continuation) (args args' : list simple)
    (action : option trap_action),
    Forall2 (canonical_in_terms tenv) args args' ->
    expr_equivalent_under tenv (Apply_cont (Apply_cont_expr k args action))
      (Apply_cont (Apply_cont_expr k args' action)).

Definition let_simple_canonical_statement : Prop :=
  ∀ (tenv : Typing_env.t) (x : variable) (kind : flambda_kind)
    (s canonical : simple) (body : expr),
    canonical_in_terms tenv s canonical ->
    expr_equivalent_under tenv (Let x kind (Simple s) body)
      (Let x kind (Simple canonical) body).

(* A [Switch] whose scrutinee is known to be a constant is its arm for
   that constant. *)
Definition switch_known_statement : Prop :=
  ∀ (tenv : Typing_env.t) (scrutinee : simple) (i : Z)
    (arms : list (Z * apply_cont)) (ac : apply_cont),
    canonical_in_terms tenv scrutinee (Const (Naked_immediate i)) ->
    find_arm i arms = Some ac ->
    expr_equivalent_under tenv (Switch scrutinee arms) (Apply_cont ac).

(* ======================================================================== *)
(* The environment follows the term                                         *)
(* ======================================================================== *)

(* A variable is defined, and the machine binds it to a value of the
   kind. *)
Definition add_variable_definition_env_satisfies_strictly_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars : env) (x : variable)
    (kind : flambda_kind) (v : value),
    typing_env_well_formed tenv = true ->
    Typing_env.mem None tenv (Name_var x) = false ->
    kind_of_value v = kind ->
    env_satisfies_strictly tenv statics vars ->
    env_satisfies_strictly
      (Typing_env.add_variable_definition tenv x kind Name_mode.normal)
      statics ((x, v) :: vars).

(* Parameters are defined, and the machine binds them, each to a value
   of its kind. *)
Definition add_definitions_of_params_env_satisfies_strictly_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars vars' : env)
    (params : list (variable * flambda_kind)) (vs : list value),
    typing_env_invariant tenv ->
    distinct (map fst params) = true ->
    (∀ x : variable,
      x ∈ map fst params -> Typing_env.mem None tenv (Name_var x) = false) ->
    Forall2 (fun param v => kind_of_value v = snd param) params vs ->
    bind_params (map fst params) vs vars = Some vars' ->
    env_satisfies_strictly tenv statics vars ->
    env_satisfies_strictly (Typing_env.add_definitions_of_params tenv params)
      statics vars'.

(* A variable in scope is given a type that the value that the machine
   has for it has. The type speaks only of variables in scope, so that
   it says the same of the machine's values and of a valuation's. *)
Definition replace_equation_env_satisfies_strictly_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars : env) (x : variable)
    (ty : flambda_type),
    in_scope_in_terms tenv x ->
    (∀ y : variable, y ∈ free_variables ty -> in_scope_in_terms tenv y) ->
    (∀ v : value, lookup x vars = Some v -> type_satisfied statics vars v ty)
    ->
    env_satisfies_strictly tenv statics vars ->
    env_satisfies_strictly (Typing_env.replace_equation tenv (Name_var x) ty)
      statics vars.

(* Two [Simple]s that have the same value in the machine are made
   equal. Their variables are in scope, so that a valuation has the
   machine's values for them. (The compiler gives
   [Typing_env.add_alias] two canonical elements. The statement does
   not ask that they are: it is about the case where [add_alias]
   answers an environment, and so asks less.) *)
Definition add_alias_env_satisfies_strictly_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars : env)
    (canonical_element1 canonical_element2 : simple)
    (result : Typing_env.add_alias_result),
    (∀ x : variable,
      canonical_element1 = Var x ∨ canonical_element2 = Var x ->
      in_scope_in_terms tenv x) ->
    Typing_env.add_alias tenv canonical_element1 canonical_element2
    = Some (Ok result) ->
    eval_simple vars canonical_element1 = eval_simple vars canonical_element2
    ->
    env_satisfies_strictly tenv statics vars ->
    env_satisfies_strictly (Typing_env.add_alias_result_t result) statics
      vars.

(* The value that the machine has for a [Simple] of the term is of the
   kind at which the environment has the [Simple]. This holds where
   poison is exempt too. *)
Definition simple_of_kind_value_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars : env) (s : simple)
    (kind : flambda_kind) (v : value),
    env_satisfies tenv statics vars ->
    (∀ x : variable, s = Var x -> in_scope_in_terms tenv x) ->
    simple_of_kind tenv s kind ->
    eval_simple vars s = Some v ->
    kind_of_value v = kind.

(* ======================================================================== *)
(* Going under binders                                                      *)
(* ======================================================================== *)

(* Under a [Let] the body is rewritten with the variable defined, at
   the kind of the value that the defining expression gives. The
   machine runs the defining expression on its heap
   (FlambdaSemanticsSpec.exec_named), and a primitive may read a cell
   there: so "the value that it gives" is the value of that run, on
   whatever heap and with whatever read record, in a state that
   satisfies the environment. The kind that the [Let] gives its
   variable, [declared], is not read here. (Simplify defines the
   variable at that kind: [let_declared_refines_under_statement],
   below, is that case.) *)
Definition let_refines_under_statement : Prop :=
  ∀ (tenv : Typing_env.t) (x : variable) (declared : flambda_kind)
    (defining_expr : named) (body body' : expr) (kind : flambda_kind),
    typing_env_well_formed tenv = true ->
    Typing_env.mem None tenv (Name_var x) = false ->
    (∀ (statics : static_env) (hp hp' : heap) (rd rd' : read_record)
       (vars : env) (v : value),
      env_satisfies_strictly tenv statics vars ->
      exec_named statics hp rd vars defining_expr = Some (v, hp', rd') ->
      kind_of_value v = kind) ->
    expr_refines_under
      (Typing_env.add_variable_definition tenv x kind Name_mode.normal) body
      body' ->
    expr_refines_under tenv (Let x declared defining_expr body)
      (Let x declared defining_expr body').

(* At the kind that the [Let] declares, with no premise on the defining
   expression: the machine has no step at a [Let] unless the value is
   of that kind (../../rocq/FlambdaWellKindedSpec.v). This is what
   Simplify does where it knows nothing of the defining expression. *)
Definition let_declared_refines_under_statement : Prop :=
  ∀ (tenv : Typing_env.t) (x : variable) (declared : flambda_kind)
    (defining_expr : named) (body body' : expr),
    typing_env_well_formed tenv = true ->
    Typing_env.mem None tenv (Name_var x) = false ->
    expr_refines_under
      (Typing_env.add_variable_definition tenv x declared Name_mode.normal)
      body body' ->
    expr_refines_under tenv (Let x declared defining_expr body)
      (Let x declared defining_expr body').

(* The same with a type, if every value that the defining expression
   gives has it, on whatever heap it is run; the variable is defined
   at the kind of the type. The type speaks of variables in scope
   where the [Let] is. *)
Definition let_typed_refines_under_statement : Prop :=
  ∀ (tenv : Typing_env.t) (x : variable) (declared : flambda_kind)
    (defining_expr : named) (body body' : expr) (ty : flambda_type),
    typing_env_well_formed tenv = true ->
    Typing_env.mem None tenv (Name_var x) = false ->
    (∀ y : variable, y ∈ free_variables ty -> in_scope_in_terms tenv y) ->
    (∀ (statics : static_env) (hp hp' : heap) (rd rd' : read_record)
       (vars : env) (v : value),
      env_satisfies_strictly tenv statics vars ->
      exec_named statics hp rd vars defining_expr = Some (v, hp', rd') ->
      type_satisfied statics vars v ty) ->
    expr_refines_under
      (Typing_env.replace_equation
         (Typing_env.add_variable_definition tenv x
            (FlambdaTypeGrammar.kind ty) Name_mode.normal)
         (Name_var x) ty)
      body body' ->
    expr_refines_under tenv (Let x declared defining_expr body)
      (Let x declared defining_expr body').

(* Where it is bound to a [Simple], it is made equal to it: the
   variable is defined at the kind at which the environment has the
   [Simple], and the aliases record that the two are equal.
   ([Meet_env.add_equation] then gives the name demoted the alias type
   of the canonical element: that step is
   [replace_equation_env_satisfies_strictly].) Simplify gives
   [Typing_env.add_alias] the canonical element of the [Simple]; the
   statement does not ask that the [Simple] is canonical, only that
   [add_alias] answers an environment. The kind that the [Let] gives,
   [declared], is not read here either: the variable is defined at the
   kind at which the environment has the [Simple]. *)
Definition let_simple_refines_under_statement : Prop :=
  ∀ (tenv : Typing_env.t) (x : variable) (declared : flambda_kind)
    (s : simple) (body body' : expr) (kind : flambda_kind)
    (result : Typing_env.add_alias_result),
    typing_env_well_formed tenv = true ->
    Typing_env.mem None tenv (Name_var x) = false ->
    (∀ y : variable, s = Var y -> in_scope_in_terms tenv y) ->
    simple_of_kind tenv s kind ->
    Typing_env.add_alias
      (Typing_env.add_variable_definition tenv x kind Name_mode.normal)
      (Var x) s
    = Some (Ok result) ->
    expr_refines_under (Typing_env.add_alias_result_t result) body body' ->
    expr_refines_under tenv (Let x declared (Simple s) body)
      (Let x declared (Simple s) body').

(* The body of a [Let_cont] runs where the [Let_cont] is. *)
Definition let_cont_body_refines_under_statement : Prop :=
  ∀ (tenv : Typing_env.t) (k : continuation) (handler : continuation_handler)
    (body body' : expr),
    expr_refines_under tenv body body' ->
    expr_refines_under tenv (Let_cont_non_recursive k handler body)
      (Let_cont_non_recursive k handler body').

Definition let_cont_recursive_body_refines_under_statement : Prop :=
  ∀ (tenv : Typing_env.t)
    (handlers : list (continuation * continuation_handler))
    (body body' : expr),
    expr_refines_under tenv body body' ->
    expr_refines_under tenv (Let_cont_recursive handlers body)
      (Let_cont_recursive handlers body').

(* A handler runs later, any number of times, under the variables of
   the place of its definition, whose values do not change, with its
   parameters bound to the arguments of a use, and under a static
   environment that may have grown. What was known where the [Let_cont]
   is is known in it, with its parameters defined; nothing is known of
   a parameter but its kind, which is that of the argument, whatever it
   is: the kinds that the handler gives its parameters, in [metadata],
   are not read here, though the machine now applies a handler only to
   values of those kinds. (The types of the arguments at the uses are
   joined by the join of environments, which is not here.) *)
Definition let_cont_handler_refines_under_statement : Prop :=
  ∀ (tenv : Typing_env.t) (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler handler' body : expr),
    typing_env_invariant tenv ->
    distinct params = true ->
    (∀ x : variable,
      x ∈ params -> Typing_env.mem None tenv (Name_var x) = false) ->
    (∀ kinds : list flambda_kind,
      length kinds = length params ->
      expr_refines_under
        (Typing_env.add_definitions_of_params tenv (combine params kinds))
        handler handler') ->
    expr_refines_under tenv
      (Let_cont_non_recursive k (Handler params metadata handler) body)
      (Let_cont_non_recursive k (Handler params metadata handler') body).

(* The same of one handler of a recursive [Let_cont]. What is known
   where the [Let_cont] is is known in the handler at every run of it:
   the variables of that place do not change. Nothing is known of a
   parameter but its kind, at the first run or at a later one. A
   handler of a recursive [Let_cont] may not bind anything static, so
   the two handlers are to agree on whether they do. *)
Definition let_cont_recursive_handler_refines_under_statement : Prop :=
  ∀ (tenv : Typing_env.t)
    (before : list (continuation * continuation_handler)) (k : continuation)
    (params : list variable) (metadata : handler_metadata)
    (handler handler' : expr)
    (after : list (continuation * continuation_handler)) (body : expr),
    typing_env_invariant tenv ->
    distinct params = true ->
    (∀ x : variable,
      x ∈ params -> Typing_env.mem None tenv (Name_var x) = false) ->
    binds_statics handler = binds_statics handler' ->
    (∀ kinds : list flambda_kind,
      length kinds = length params ->
      expr_refines_under
        (Typing_env.add_definitions_of_params tenv (combine params kinds))
        handler handler') ->
    expr_refines_under tenv
      (Let_cont_recursive
         (before ++ (k, Handler params metadata handler) :: after) body)
      (Let_cont_recursive
         (before ++ (k, Handler params metadata handler') :: after) body).

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

(* The variables are those of FlambdaTypingEnvExamplesSpec.v. Each term
   is closed but for its continuations, so Simplify starts on it with
   the environment that knows nothing. *)

(* [let p = 7 in let x = p in apply_cont k x]. At the first [Let], [p]
   is defined and made equal to the constant, which is the canonical
   element of the two. At the second, [p] is replaced by its canonical
   element; then [x] is defined and made equal to the constant too. In
   the body [x] is replaced by its canonical element. (The two [Let]s
   are then unused: removing them is [unused_let_statement],
   ../../rocq/FlambdaNominalSpec.v.) *)
Definition example_constant_propagated_statement : Prop :=
  ∀ k : continuation,
    expr_refines
      (Let example_p Value (Simple (Const (Tagged_immediate 7)))
         (Let example_x Value (Simple (Var example_p))
            (Apply_cont (Apply_cont_expr k [Var example_x] None))))
      (Let example_p Value (Simple (Const (Tagged_immediate 7)))
         (Let example_x Value (Simple (Const (Tagged_immediate 7)))
            (Apply_cont
               (Apply_cont_expr k [Const (Tagged_immediate 7)] None)))).

(* [let i = 1 in switch i with 0 -> k0 | 1 -> k1]: the scrutinee is
   known to be the constant, so the [Switch] is its arm. *)
Definition example_switch_known_statement : Prop :=
  ∀ k0 k1 : continuation,
    expr_refines
      (Let example_i (Naked_number Kind_naked_immediate)
         (Simple (Const (Naked_immediate 1)))
         (Switch (Var example_i)
            [(0, Apply_cont_expr k0 [] None);
             (1, Apply_cont_expr k1 [] None)]))
      (Let example_i (Naked_number Kind_naked_immediate)
         (Simple (Const (Naked_immediate 1)))
         (Apply_cont (Apply_cont_expr k1 [] None))).

(* A loop:

     let rec k p =
       let x = 7 in
       let_cont j () = apply_cont exit [result] in
       switch p with 0 -> apply_cont k 1 | 1 -> apply_cont j ()
     in
     apply_cont k 0

   Where the three continuations are different it goes round once and
   leaves through [exit] on the second turn, with [result]. With
   [result] the variable [x], the handler of the loop is rewritten
   under the environment with [p] defined, at any kind, on every turn;
   in it [x] is made equal to the constant; and under that the handler
   of [j] is rewritten, [x] replaced by its canonical element. *)
Definition example_loop_handler (k j exit : continuation) (result : simple)
    : continuation_handler :=
  Handler [example_p]
    (make_handler_metadata [Naked_number Kind_naked_immediate])
    (Let example_x Value (Simple (Const (Tagged_immediate 7)))
       (Let_cont_non_recursive j
          (Handler [] (make_handler_metadata [])
             (Apply_cont (Apply_cont_expr exit [result] None)))
          (Switch (Var example_p)
             [(0, Apply_cont_expr k [Const (Naked_immediate 1)] None);
              (1, Apply_cont_expr j [] None)]))).

Definition example_loop_statement : Prop :=
  ∀ k j exit : continuation,
    expr_refines
      (Let_cont_recursive
         [(k, example_loop_handler k j exit (Var example_x))]
         (Apply_cont (Apply_cont_expr k [Const (Naked_immediate 0)] None)))
      (Let_cont_recursive
         [(k, example_loop_handler k j exit (Const (Tagged_immediate 7)))]
         (Apply_cont (Apply_cont_expr k [Const (Naked_immediate 0)] None))).
