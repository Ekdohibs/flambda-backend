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

(* Rewriting inside a handler or inside the body of code, where the
   rewrite is sound only from the states that it can be run from.

   This file is to be reviewed: [kept_by_step], [handler_refines_from],
   [code_refines_from], [code_refines_from_called] and the statements.

   FlambdaContextSpec.v says that a rewrite which is a refinement under
   all environments may be applied under any binder. Most rewrites of
   the simplifier are not: they are sound because of what is known of
   the state in which the rewritten expression runs. Here that is a
   predicate on states [inv] that a step keeps ([kept_by_step]), so that
   a run from a state that has it stays among the states that have it.
   Examples are a condition on the static environment that its
   extensions keep (what a symbol is bound to), that every number of
   the state is in range (FlambdaRangeSpec.state_in_range), and their
   conjunction.

   A handler, and the body of code, do not run where they are written:
   they are kept, and run later, any number of times, under the
   variables of that place with the parameters bound. So the premise on
   the two expressions is about every state in which they can come to
   run:

   - a handler runs under the variables of its definition with its
     parameters bound to some values, under any continuations and any
     static environment. The application of a handler has no step unless
     the values are of the kinds that the handler gives its parameters
     (FlambdaSemanticsSpec.apply_cont_value), so that fact is given too
     ([handler_refines_from]);
   - the body of code runs at a call, under its parameters and
     [my_closure] alone, bound to the arguments and the callee, with its
     two continuations bound and under any static environment. A call
     has no step unless the arguments are of the kinds of the arity of
     the code (FlambdaSemanticsSpec.call_function), and the callee is a
     closure, so those two facts are given too ([code_refines_from]).
     They are about the values: nothing is asked of the kinds that the
     term gives.

   In such a state the premise asks two things. The first expression,
   if its state has [inv], refines the second: this is the rewrite, and
   it is asked only from the states that have [inv]. And the two states
   have [inv] together. The second is not there because a run could
   leave the states that have [inv]: a step keeps it, and the
   statements ask it of both whole states. It is what the proof needs.
   The proof compares the runs of the two whole states through states
   that neither need reach, the first expression under the handlers and
   code of the rewritten term, and the second under those of the
   original; [inv] of such a state is got from [inv] of the state with
   the other expression under the same environments, which one of the
   runs does reach. A predicate that does not look at the expression
   of a state, such as a condition on the static environment, meets the
   second for free.

   The kinds of the values of the parameters of a handler are given, as
   those of the arguments of a call are. The machine tests the kind of a
   value where a handler binds it, as it does at a call (the review
   comment before FlambdaSemanticsSpec.step says how certain the first
   is). So [handler_refines_from] takes the kinds that the handler gives
   its parameters, and hands the premise that the values are of those
   kinds. Both are about the values, in the state where the expression
   comes to run.

   The statements are of states, not of expressions under all
   environments: the two whole states are asked to have [inv] at the
   start.

   - [let_cont_handler_refines_from_statement]: the handler of a
     non-recursive [Let_cont].
   - [let_cont_recursive_handler_refines_from_statement]: one handler
     of a recursive [Let_cont]. As in FlambdaContextSpec.v, the two
     handlers must agree on whether they bind something static.
   - [let_static_code_refines_from_statement]: the body of code bound by
     a [Let_static], with the same condition.
   - [let_static_code_refines_from_called_statement]: the same, with
     less asked of the two bodies. They are asked the two things only
     in the states in which, besides, the callee is a closure whose
     code is the code id that the [Let_static] binds the body to
     ([code_refines_from_called]). A call finds the body that it runs
     by the code id of its callee
     (FlambdaSemanticsSpec.shape_of_closure_call, which looks the code
     up at the code id that FlambdaValueSpec.closure_of gives), so the
     body bound to a code id comes to run in no other state. Two
     bodies that have the premise of [code_refines_from] have this
     one, for every code id: so this statement gives the one above.
   - [let_static_code_refines_where_statement]: the last, for a
     condition [ok] on the static environment that extensions keep. The
     body need only refine at a call: under its parameters bound to
     values of the kinds of its arity and [my_closure] to a value of
     kind [Value], and a static environment of which [ok] holds. This
     is FlambdaStaticSpec.let_static_code_refines_statement with the
     premise weakened to where the code runs.
   - That the premises can be met: for the predicate that holds of
     every state they are a refinement in every state in which the
     expressions run ([handler_refines_from_always_statement],
     [code_refines_from_always_statement]). And for a predicate that
     looks at the expression, that of the runs in range, which a step
     keeps (FlambdaRangeSpec.step_in_range_statement): two handlers
     that pass the check of numbers have it together
     ([handler_refines_from_in_range_statement]), so in a run in range
     a handler may be replaced by one that passes the check and that
     it refines from the states in range
     ([let_cont_handler_refines_in_range_statement]).

   Nothing is trusted here. The proofs are in
   FlambdaContextWhereProofs.v, which needs no review; they are
   instances of one lemma of FlambdaContextProofs.v, by which a run
   under some environments is carried to environments that hold
   rewritten handlers and code. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaValueSpec
  FlambdaSemanticsSpec FlambdaWellFormedSpec FlambdaRangeSpec.
Import ListNotations.

(* ======================================================================== *)
(* Refinement from the states that have a predicate                         *)
(* ======================================================================== *)

(* A step from a state that has [inv] is to a state that has it. *)
Definition kept_by_step (inv : state -> Prop) : Prop :=
  ∀ s s' : state, inv s -> step s = Step s' -> inv s'.

(* A handler with parameters [params], which it gives the kinds [kinds],
   defined under the variables [def_vars], runs under those variables
   with its parameters bound to values of those kinds. In every such
   state the two expressions have [inv] together, and the first, where
   it has [inv], refines the second.

   That the two have [inv] together is asked for the proof. It goes
   through states that no run of the two whole states need reach (the
   first expression under the rewritten handlers and code, the second
   under the original ones), where that a step keeps [inv] gives
   nothing, and it gets [inv] of such a state from [inv] of the state
   with the other expression. It is not that a run could leave the
   states that have [inv]. *)
Definition handler_refines_from (inv : state -> Prop)
    (params : list variable) (kinds : list flambda_kind) (def_vars : env)
    (source target : expr) : Prop :=
  ∀ (vs : list value) (handler_vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap),
    map kind_of_value vs = kinds ->
    bind_params params vs def_vars = Some handler_vars ->
    (inv (State source handler_vars conts statics hp)
     <-> inv (State target handler_vars conts statics hp))
    ∧ (inv (State source handler_vars conts statics hp) ->
       refines (State source handler_vars conts statics hp)
         (State target handler_vars conts statics hp)).

(* The body of code with parameters [params], closure variable
   [my_closure], continuations [ret] and [exn] and metadata [metadata]
   runs at a call: under its parameters bound to the arguments, which
   are of the kinds of its arity, and [my_closure] to the callee, which
   is of kind [Value]; its two continuations are bound to what the call
   returns and raises to. In every such state the two bodies have [inv]
   together, and the first, where it has [inv], refines the second. *)
Definition code_refines_from (inv : state -> Prop) (params : list variable)
    (my_closure : variable) (ret exn : continuation)
    (metadata : code_metadata) (source target : expr) : Prop :=
  ∀ (vs : list value) (callee : value) (callee_vars : env)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap),
    bind_params params vs [(my_closure, callee)] = Some callee_vars ->
    map kind_of_value vs = unarize (cm_params_arity metadata) ->
    kind_of_value callee = Value ->
    (inv
       (State source callee_vars [(ret, return_to); (exn, exn_to)] statics hp)
     <-> inv
           (State target callee_vars [(ret, return_to); (exn, exn_to)]
              statics hp))
    ∧ (inv
         (State source callee_vars [(ret, return_to); (exn, exn_to)] statics hp)
       ->
       refines
         (State source callee_vars [(ret, return_to); (exn, exn_to)] statics hp)
         (State target callee_vars [(ret, return_to); (exn, exn_to)]
            statics hp)).

(* The same for the body of the code that is bound to the code id [id],
   asked of fewer states: those in which, besides, the callee is a
   closure whose code is [id], whatever its value slots are. The static
   environment is that of the state in which the body runs. *)
Definition code_refines_from_called (inv : state -> Prop) (id : code_id)
    (params : list variable) (my_closure : variable)
    (ret exn : continuation) (metadata : code_metadata)
    (source target : expr) : Prop :=
  ∀ (vs : list value) (callee : value) (callee_vars : env)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
    (value_slots : list (value_slot * value)),
    bind_params params vs [(my_closure, callee)] = Some callee_vars ->
    map kind_of_value vs = unarize (cm_params_arity metadata) ->
    kind_of_value callee = Value ->
    closure_of statics callee = Some (id, value_slots) ->
    (inv
       (State source callee_vars [(ret, return_to); (exn, exn_to)] statics hp)
     <-> inv
           (State target callee_vars [(ret, return_to); (exn, exn_to)]
              statics hp))
    ∧ (inv
         (State source callee_vars [(ret, return_to); (exn, exn_to)] statics hp)
       ->
       refines
         (State source callee_vars [(ret, return_to); (exn, exn_to)] statics hp)
         (State target callee_vars [(ret, return_to); (exn, exn_to)]
            statics hp)).

(* ======================================================================== *)
(* A handler replaced by one that refines it where it runs                  *)
(* ======================================================================== *)

Definition let_cont_handler_refines_from_statement : Prop :=
  ∀ (inv : state -> Prop) (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler handler' body : expr)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap),
    kept_by_step inv ->
    inv
      (State
         (Let_cont_non_recursive k (Handler params metadata handler) body)
         vars conts statics hp) ->
    inv
      (State
         (Let_cont_non_recursive k (Handler params metadata handler') body)
         vars conts statics hp) ->
    handler_refines_from inv params (hm_params_kinds metadata) vars handler
      handler' ->
    refines
      (State
         (Let_cont_non_recursive k (Handler params metadata handler) body)
         vars conts statics hp)
      (State
         (Let_cont_non_recursive k (Handler params metadata handler') body)
         vars conts statics hp).

Definition let_cont_recursive_handler_refines_from_statement : Prop :=
  ∀ (inv : state -> Prop)
    (before : list (continuation * continuation_handler)) (k : continuation)
    (params : list variable) (metadata : handler_metadata)
    (handler handler' : expr)
    (after : list (continuation * continuation_handler)) (body : expr)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap),
    kept_by_step inv ->
    inv
      (State
         (Let_cont_recursive
            (before ++ (k, Handler params metadata handler) :: after) body)
         vars conts statics hp) ->
    inv
      (State
         (Let_cont_recursive
            (before ++ (k, Handler params metadata handler') :: after) body)
         vars conts statics hp) ->
    binds_statics handler = binds_statics handler' ->
    handler_refines_from inv params (hm_params_kinds metadata) vars handler
      handler' ->
    refines
      (State
         (Let_cont_recursive
            (before ++ (k, Handler params metadata handler) :: after) body)
         vars conts statics hp)
      (State
         (Let_cont_recursive
            (before ++ (k, Handler params metadata handler') :: after) body)
         vars conts statics hp).

(* ======================================================================== *)
(* The body of code replaced by one that refines it where it runs           *)
(* ======================================================================== *)

Definition let_static_code_refines_from_statement : Prop :=
  ∀ (inv : state -> Prop) (before : list static_binding) (id : code_id)
    (params : list variable) (my_closure : variable)
    (ret exn : continuation) (metadata : code_metadata)
    (code_body code_body' : expr) (after : list static_binding)
    (body : expr) (vars : env) (conts : cont_env) (statics : static_env)
      (hp : heap),
    kept_by_step inv ->
    inv
      (State
         (Let_static
            (before
             ++ Bind_code id
                  (Code params my_closure ret exn metadata code_body)
             :: after)
            body)
         vars conts statics hp) ->
    inv
      (State
         (Let_static
            (before
             ++ Bind_code id
                  (Code params my_closure ret exn metadata code_body')
             :: after)
            body)
         vars conts statics hp) ->
    binds_statics code_body = binds_statics code_body' ->
    code_refines_from inv params my_closure ret exn metadata code_body
      code_body' ->
    refines
      (State
         (Let_static
            (before
             ++ Bind_code id
                  (Code params my_closure ret exn metadata code_body)
             :: after)
            body)
         vars conts statics hp)
      (State
         (Let_static
            (before
             ++ Bind_code id
                  (Code params my_closure ret exn metadata code_body')
             :: after)
            body)
         vars conts statics hp).

(* The same, where the two bodies are asked the two things only in the
   states in which the callee is, besides, a closure whose code is
   [id], the code id that the [Let_static] binds the body to. *)
Definition let_static_code_refines_from_called_statement : Prop :=
  ∀ (inv : state -> Prop) (before : list static_binding) (id : code_id)
    (params : list variable) (my_closure : variable)
    (ret exn : continuation) (metadata : code_metadata)
    (code_body code_body' : expr) (after : list static_binding)
    (body : expr) (vars : env) (conts : cont_env) (statics : static_env)
      (hp : heap),
    kept_by_step inv ->
    inv
      (State
         (Let_static
            (before
             ++ Bind_code id
                  (Code params my_closure ret exn metadata code_body)
             :: after)
            body)
         vars conts statics hp) ->
    inv
      (State
         (Let_static
            (before
             ++ Bind_code id
                  (Code params my_closure ret exn metadata code_body')
             :: after)
            body)
         vars conts statics hp) ->
    binds_statics code_body = binds_statics code_body' ->
    code_refines_from_called inv id params my_closure ret exn metadata
      code_body code_body' ->
    refines
      (State
         (Let_static
            (before
             ++ Bind_code id
                  (Code params my_closure ret exn metadata code_body)
             :: after)
            body)
         vars conts statics hp)
      (State
         (Let_static
            (before
             ++ Bind_code id
                  (Code params my_closure ret exn metadata code_body')
             :: after)
            body)
         vars conts statics hp).

(* For a condition on the static environment that its extensions keep:
   the body refines at every call, under any continuations and any
   static environment of which the condition holds. *)
Definition let_static_code_refines_where_statement : Prop :=
  ∀ (ok : static_env -> Prop) (before : list static_binding) (id : code_id)
    (params : list variable) (my_closure : variable)
    (ret exn : continuation) (metadata : code_metadata)
    (code_body code_body' : expr) (after : list static_binding)
    (body : expr) (vars : env) (conts : cont_env) (statics : static_env)
      (hp : heap),
    (∀ statics statics' : static_env,
      ok statics -> statics_extend statics statics' -> ok statics') ->
    ok statics ->
    binds_statics code_body = binds_statics code_body' ->
    (∀ (vs : list value) (callee : value) (callee_vars : env)
       (conts' : cont_env) (statics' : static_env) (hp' : heap),
      ok statics' ->
      map kind_of_value vs = unarize (cm_params_arity metadata) ->
      kind_of_value callee = Value ->
      bind_params params vs [(my_closure, callee)] = Some callee_vars ->
      refines (State code_body callee_vars conts' statics' hp')
        (State code_body' callee_vars conts' statics' hp')) ->
    refines
      (State
         (Let_static
            (before
             ++ Bind_code id
                  (Code params my_closure ret exn metadata code_body)
             :: after)
            body)
         vars conts statics hp)
      (State
         (Let_static
            (before
             ++ Bind_code id
                  (Code params my_closure ret exn metadata code_body')
             :: after)
            body)
         vars conts statics hp).

(* ======================================================================== *)
(* The premises can be met                                                  *)
(* ======================================================================== *)

(* For the predicate that every state has, the premises are a refinement
   in every state in which the expressions run. *)
Definition handler_refines_from_always_statement : Prop :=
  ∀ (params : list variable) (kinds : list flambda_kind) (def_vars : env)
    (source target : expr),
    handler_refines_from (fun _ => True) params kinds def_vars source target
    <-> ∀ (vs : list value) (handler_vars : env) (conts : cont_env)
          (statics : static_env) (hp : heap),
          map kind_of_value vs = kinds ->
          bind_params params vs def_vars = Some handler_vars ->
          refines (State source handler_vars conts statics hp)
            (State target handler_vars conts statics hp).

Definition code_refines_from_always_statement : Prop :=
  ∀ (params : list variable) (my_closure : variable)
    (ret exn : continuation) (metadata : code_metadata)
    (source target : expr),
    expr_refines source target ->
    code_refines_from (fun _ => True) params my_closure ret exn metadata
      source target.

(* For the runs in range, a predicate that looks at the expression: two
   handlers that pass the check of numbers, of which the first refines
   the second from the states in range. *)
Definition handler_refines_from_in_range_statement : Prop :=
  ∀ (params : list variable) (kinds : list flambda_kind) (def_vars : env)
    (source target : expr),
    numbers_in_range source = true ->
    numbers_in_range target = true ->
    (∀ (vs : list value) (handler_vars : env) (conts : cont_env)
       (statics : static_env) (hp : heap),
      map kind_of_value vs = kinds ->
      bind_params params vs def_vars = Some handler_vars ->
      state_in_range (State source handler_vars conts statics hp) ->
      refines (State source handler_vars conts statics hp)
        (State target handler_vars conts statics hp)) ->
    handler_refines_from state_in_range params kinds def_vars source target.

(* So, from a state in range, a handler may be replaced by one that
   passes the check and that it refines from the states in range. *)
Definition let_cont_handler_refines_in_range_statement : Prop :=
  ∀ (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler handler' body : expr)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap),
    state_in_range
      (State
         (Let_cont_non_recursive k (Handler params metadata handler) body)
         vars conts statics hp) ->
    numbers_in_range handler' = true ->
    (∀ (vs : list value) (handler_vars : env) (conts' : cont_env)
       (statics' : static_env) (hp' : heap),
      map kind_of_value vs = hm_params_kinds metadata ->
      bind_params params vs vars = Some handler_vars ->
      state_in_range (State handler handler_vars conts' statics' hp') ->
      refines (State handler handler_vars conts' statics' hp')
        (State handler' handler_vars conts' statics' hp')) ->
    refines
      (State
         (Let_cont_non_recursive k (Handler params metadata handler) body)
         vars conts statics hp)
      (State
         (Let_cont_non_recursive k (Handler params metadata handler') body)
         vars conts statics hp).
