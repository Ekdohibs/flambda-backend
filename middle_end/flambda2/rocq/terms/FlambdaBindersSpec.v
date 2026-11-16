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

(* What each binder of the terms of FlambdaModel.v binds: the free
   variables and the free continuations of a term, and the three
   conditions on binders that a well-formed unit meets.

   This file is to be reviewed: the scopes of the binders against the
   compiler's name abstractions; the free names of what has no binder
   against [Named.free_names], [Static_const_or_code.free_names] and
   [Static_const_group.free_names] (terms/flambda.ml) and
   [Code0.free_names] (terms/code0.ml); and the three checks against
   rows 3, 5 and 6 of the table of well-formedness
   (requests/C0-well-formed.md).

   The compiler has no function that computes the free names of an
   expression: [Flambda.Expr], [Let_expr], [Let_cont_expr],
   [Continuation_handler] and [Function_params_and_body] have no
   [free_names]. A binder there is a name abstraction
   (nominal/name_abstraction.ml) of the names bound and a body, read by
   [pattern_match], and the free names of an abstraction are those of
   its body less the names bound ([Name_abstraction.free_names]).
   Whoever builds a term supplies the free names of its body
   ([free_names_of_body] of [Let_expr.create] and of
   [Function_params_and_body.create], [free_names_of_handler] of
   [Continuation_handler.create]); Simplify accumulates them as it
   rebuilds a term, and code keeps the summary it was created with
   ([Code0.free_names]). The functions here compute what such a summary
   is to be, for the fragment of the terms that the model has.

   The names here are the variables and the continuations: the two
   sorts that a renaming renames. Symbols, code ids, value slots and
   function slots occur in terms too, and the compiler's
   [Name_occurrences.t] records them; they are outside the functions
   here, which are the free names of a term projected on these two
   sorts. (That every symbol and code id mentioned is bound is row 4 of
   well-formedness, not this file's.)

   The binders of the fragment, and where what they bind is in scope:
   - [Let x d body]: [x], in [body]; not in [d];
   - [Handler params handler]: [params], in [handler];
   - [Let_cont_non_recursive k h body]: [k], in [body]; not in [h];
   - [Let_cont_recursive handlers body]: the continuations of
     [handlers], in [body] and in every handler of [handlers];
   - [Code params my_closure return_continuation exn_continuation
     body]: [params] and [my_closure], and the two continuations, in
     [body];
   - [Let_static bindings body]: no variable and no continuation.
   Nothing else in the fragment binds one. In particular [let_closures]
   binds each of its variables by a [Let] of its own. (The compiler's
   code also binds [my_depth] and its regions, and a recursive
   [Let_cont] has parameters that its handlers share; the model has
   neither.)

   The free names are lists, read as sets (FlambdaMapSpec.v): a name may
   be repeated, and only [among] is to be asked of them. The compiler's
   [Name_occurrences.t] also counts occurrences and tells their modes;
   the model has neither.

   The free names of a term include those of the code bound in it: the
   free names of code are those of its body less what the code binds.
   This is the compiler's account of code too: its
   [Function_params_and_body] is a name abstraction of the parameters,
   [my_closure] and the two continuations over the body, and the
   summary that code keeps is to be the free names of that abstraction.
   It is why code can be asked to be closed. The human has ruled that
   code is closed (decision D31): [code_closed] is that, as a check.

   There is one function for each sort of name and each sort of term,
   and nothing else in the model is to compute free names or scope. The
   statements that a bound name may be renamed
   (FlambdaCanonicalFreeNamesSpec.v) and the rules of alpha-equivalence
   (FlambdaAlphaSpec.v) are written with these. The older accounts that
   are still there, to be restated by these and then removed:
   FlambdaWellFormedSpec.free_continuations and
   [handler_free_continuations], and
   FlambdaNormaliseSpec.expr_in_scope, which are checks that run; and
   the relations [var_free_in_expr], [cont_free_in_expr] and those
   beside them in FlambdaNominalSpec.v, which
   FlambdaApplySpec.inlinable and the proofs are written with, and the
   three statements about a binding that is not used, which keep them.
   None of those looks into code. On a term whose code is closed they
   say the same as the functions here (FlambdaFreeNamesSpec.v).

   The three checks are conditions of FlambdaWellFormedSpec.v, which
   takes them as conjuncts of [well_formed_unit]. They are here, and
   not there, because the invariant of a state that goes with them
   (FlambdaScopeSpec.v) is defined from them and is below that file.

   Two of them ask more than the compiler checks: see the comments at
   [binders_distinct] and [code_closed].

   The proofs are in FlambdaBindersProofs.v, which needs no review. *)

From Stdlib Require Import PArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec.
Import ListNotations.

(* ======================================================================== *)
(* Free variables                                                           *)
(* ======================================================================== *)

Definition free_variables_of_simple (s : simple) : list variable :=
  match s with
  | Var x => [x]
  | Symbol _ | Const _ => []
  end.

Definition free_variables_of_simples (args : list simple) : list variable :=
  flat_map free_variables_of_simple args.

Definition free_variables_of_value_slots
    (value_slots : list (value_slot * simple)) : list variable :=
  flat_map (fun slot => free_variables_of_simple (snd slot)) value_slots.

Definition free_variables_of_named (defining_expr : named)
    : list variable :=
  match defining_expr with
  | Simple s => free_variables_of_simple s
  | Prim _ args => free_variables_of_simples args
  | Set_of_closures _ value_slots | Closure_of_set _ _ value_slots =>
    free_variables_of_value_slots value_slots
  | Project_value_slot _ closure | Project_value_slot_from _ _ closure
  | Project_function_slot _ _ closure =>
    free_variables_of_simple closure
  end.

Definition free_variables_of_static_const (const : static_const)
    : list variable :=
  match const with
  | Static_const_set_of_closures _ value_slots
  | Static_const_closure_of_set _ _ _ value_slots =>
    free_variables_of_value_slots value_slots
  | Static_const_block _ fields => free_variables_of_simples fields
  | Static_const_immutable_value_array elements =>
    free_variables_of_simples elements
  | Static_const_boxed_number contents => free_variables_of_simple contents
  | Static_const_immutable_string _ => []
  end.

Definition free_variables_of_apply_cont (ac : apply_cont) : list variable :=
  free_variables_of_simples (ac_args ac).

Definition free_variables_of_apply (ap : apply_expr) : list variable :=
  free_variables_of_simple (ap_callee ap)
  ++ free_variables_of_simples (ap_args ap)
  ++ free_variables_of_simples (exn_extra_simples ap).

(* A [Let] binds its variable in its body, a handler its parameters, and
   code its parameters and [my_closure]. *)
Fixpoint free_variables_of_expr (e : expr) : list variable :=
  match e with
  | Let x _ defining_expr body =>
    free_variables_of_named defining_expr
    ++ without [x] (free_variables_of_expr body)
  | Let_static bindings body =>
    flat_map free_variables_of_binding bindings
    ++ free_variables_of_expr body
  | Let_cont_non_recursive _ h body =>
    free_variables_of_handler h ++ free_variables_of_expr body
  | Let_cont_recursive handlers body =>
    flat_map (fun '(_, h) => free_variables_of_handler h) handlers
    ++ free_variables_of_expr body
  | Apply ap => free_variables_of_apply ap
  | Apply_cont ac => free_variables_of_apply_cont ac
  | Switch scrutinee arms =>
    free_variables_of_simple scrutinee
    ++ flat_map (fun arm => free_variables_of_apply_cont (snd arm)) arms
  | Invalid => []
  end
with free_variables_of_handler (h : continuation_handler) : list variable :=
  match h with
  | Handler params _ handler => without params (free_variables_of_expr handler)
  end
with free_variables_of_binding (binding : static_binding) : list variable :=
  match binding with
  | Bind_code _ c => free_variables_of_code c
  | Bind_deleted_code _ => []
  | Bind_symbol _ const => free_variables_of_static_const const
  end
with free_variables_of_code (c : code) : list variable :=
  match c with
  | Code params my_closure _ _ _ body =>
    without (my_closure :: params) (free_variables_of_expr body)
  end.

(* ======================================================================== *)
(* Free continuations                                                       *)
(* ======================================================================== *)

(* A non-recursive [Let_cont] binds its continuation in its body; a
   recursive one binds its continuations in its body and in its handlers;
   code binds its return and exception continuations. *)
Fixpoint free_continuations_of_expr (e : expr) : list continuation :=
  match e with
  | Let _ _ _ body => free_continuations_of_expr body
  | Let_static bindings body =>
    flat_map free_continuations_of_binding bindings
    ++ free_continuations_of_expr body
  | Let_cont_non_recursive k h body =>
    free_continuations_of_handler h
    ++ without [k] (free_continuations_of_expr body)
  | Let_cont_recursive handlers body =>
    without (map fst handlers)
      (flat_map (fun '(_, h) => free_continuations_of_handler h) handlers
       ++ free_continuations_of_expr body)
  | Apply ap => [ap_continuation ap; ap_exn_continuation ap]
  | Apply_cont ac => apply_cont_continuations ac
  | Switch _ arms =>
    flat_map (fun arm => apply_cont_continuations (snd arm)) arms
  | Invalid => []
  end
with free_continuations_of_handler (h : continuation_handler)
    : list continuation :=
  match h with
  | Handler _ _ handler => free_continuations_of_expr handler
  end
with free_continuations_of_binding (binding : static_binding)
    : list continuation :=
  match binding with
  | Bind_code _ c => free_continuations_of_code c
  | Bind_deleted_code _ | Bind_symbol _ _ => []
  end
with free_continuations_of_code (c : code) : list continuation :=
  match c with
  | Code _ _ return_continuation exn_continuation _ body =>
    without [return_continuation; exn_continuation]
      (free_continuations_of_expr body)
  end.

(* ======================================================================== *)
(* Scope                                                                    *)
(* ======================================================================== *)

(* Every free variable of the expression is one of [vs], and every free
   continuation one of [ks]. *)
Definition names_among (vs : list variable) (ks : list continuation)
    (e : expr) : bool :=
  all_among (free_variables_of_expr e) vs
  && all_among (free_continuations_of_expr e) ks.

Definition handler_names_among (vs : list variable)
    (ks : list continuation) (h : continuation_handler) : bool :=
  all_among (free_variables_of_handler h) vs
  && all_among (free_continuations_of_handler h) ks.

(* Row 3 of well-formedness. A unit, with its return and exception
   continuations: it has no free variable, and no free continuation but
   those two. So every variable and every continuation that the unit
   uses is bound around the use, by one of the binders above. *)
Definition names_in_scope (unit_return unit_exn : continuation) (e : expr)
    : bool :=
  names_among [] [unit_return; unit_exn] e.

(* ======================================================================== *)
(* The names of one binder are distinct                                     *)
(* ======================================================================== *)

(* Here the model is stricter than the compiler (decision D32).
   - The continuations of a recursive [Let_cont] are distinct, and there
     is at least one. The compiler always checks both
     ([Bound_continuations.create]).
   - The parameters of a handler are distinct, and so are those of code
     together with [my_closure]. The compiler checks this only when its
     invariant checks are on ([Bound_parameters.create] and
     [check_no_duplicates], [Bound_for_function.create]), and then
     compares whole parameters, so that two parameters with one variable
     and different kinds pass. Discrepancies N1-3 and R2-16.
   - The return and exception continuations of code differ. Nothing in
     the compiler checks it. Discrepancy N1-3.
   A term that breaks one of these is not one the compiler builds: its
   binders are name abstractions, opened at fresh names. The machine
   runs such a term (a later parameter shadows an earlier one, the first
   of two handlers of one continuation counts), and the comparison of
   FlambdaCompareSpec.v then accepts terms that no renaming relates. *)

(* Row 5 of well-formedness. The names that one binder binds are
   distinct, at every binder of the expression, those of handlers and of
   code included; and a recursive [Let_cont] has a handler. *)
Fixpoint binders_distinct (e : expr) : bool :=
  match e with
  | Let _ _ _ body => binders_distinct body
  | Let_static bindings body =>
    forallb binding_binders_distinct bindings && binders_distinct body
  | Let_cont_non_recursive _ h body =>
    handler_binders_distinct h && binders_distinct body
  | Let_cont_recursive handlers body =>
    negb (Keyed.is_empty handlers)
    && distinct (map fst handlers)
    && forallb (fun '(_, h) => handler_binders_distinct h) handlers
    && binders_distinct body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_binders_distinct (h : continuation_handler) : bool :=
  match h with
  | Handler params _ handler => distinct params && binders_distinct handler
  end
with binding_binders_distinct (binding : static_binding) : bool :=
  match binding with
  | Bind_code _ c => code_binders_distinct c
  | Bind_deleted_code _ | Bind_symbol _ _ => true
  end
with code_binders_distinct (c : code) : bool :=
  match c with
  | Code params my_closure return_continuation exn_continuation _ body =>
    distinct (my_closure :: params)
    && negb (Pos.eqb return_continuation exn_continuation)
    && binders_distinct body
  end.

(* ======================================================================== *)
(* Code is closed                                                           *)
(* ======================================================================== *)

(* Here the model is stricter than the compiler (decisions D31 and D32).
   The compiler refuses code with a free variable or a free continuation
   where the code is created ([Code0.create], by
   [check_free_names_of_params_and_body]), but what it tests is the free
   names that the caller of [create] supplies, not free names that it
   computes from the body. So that code is closed is, in the compiler, a
   claim of whoever creates the code. Discrepancies N1-4 and C0-5. *)

(* Row 6 of well-formedness. Every piece of code bound in the
   expression, at any depth, is closed: it has no free variable and no
   free continuation, so its body mentions no variable but its
   parameters and [my_closure], and no continuation but its return and
   exception continuations ([closed_code_statement]). The code bound
   inside the body of code is asked the same. *)
Fixpoint code_closed (e : expr) : bool :=
  match e with
  | Let _ _ _ body => code_closed body
  | Let_static bindings body =>
    forallb binding_code_closed bindings && code_closed body
  | Let_cont_non_recursive _ h body =>
    handler_code_closed h && code_closed body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_code_closed h) handlers
    && code_closed body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_code_closed (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => code_closed handler
  end
with binding_code_closed (binding : static_binding) : bool :=
  match binding with
  | Bind_code _ c => code_is_closed c
  | Bind_deleted_code _ | Bind_symbol _ _ => true
  end
with code_is_closed (c : code) : bool :=
  match c with
  | Code _ _ _ _ _ body =>
    Keyed.is_empty (free_variables_of_code c)
    && Keyed.is_empty (free_continuations_of_code c)
    && code_closed body
  end.

(* ======================================================================== *)
(* What the checks say                                                      *)
(* ======================================================================== *)

(* Closed code, said of its body. *)
Definition closed_code_statement : Prop :=
  ∀ c : code,
    code_is_closed c = true
    <-> names_among (code_my_closure c :: code_params c)
          [code_return_continuation c; code_exn_continuation c]
          (code_body c)
        = true
        ∧ code_closed (code_body c) = true.

(* Closed code adds nothing to the free names of the term that binds
   it: the free variables of a group of static bindings whose code is
   closed are those of its static constants, and it has no free
   continuation. *)
Definition closed_code_free_names_statement : Prop :=
  ∀ bindings : list static_binding,
    forallb binding_code_closed bindings = true ->
    flat_map free_variables_of_binding bindings
    = flat_map
        (fun binding =>
           match binding with
           | Bind_symbol _ const => free_variables_of_static_const const
           | Bind_code _ _ | Bind_deleted_code _ => []
           end)
        bindings
    ∧ flat_map free_continuations_of_binding bindings = [].

(* What row 3 asks, said of each name: no variable is free in the unit,
   and a continuation that is free in it is one of the two given. *)
Definition unit_free_names_statement : Prop :=
  ∀ (unit_return unit_exn : continuation) (e : expr),
    names_in_scope unit_return unit_exn e = true
    <-> (∀ x : variable, among x (free_variables_of_expr e) = false)
        ∧ (∀ k : continuation,
            among k (free_continuations_of_expr e) = true ->
            k = unit_return ∨ k = unit_exn).
