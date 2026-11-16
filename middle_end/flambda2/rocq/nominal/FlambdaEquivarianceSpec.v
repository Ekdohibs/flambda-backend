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

(* Equivariance: renamings act on everything a state is made of, and
   execution commutes with them.

   This file is to be reviewed: the action on what a state holds
   ([rename_cont_value] and the definitions after it), and the
   statements.

   FlambdaNominalSpec.v has the renamings and their action on terms
   ([rename_expr]), code included: the compiler renames code as it
   renames everything else ([apply_renaming_static_const_or_code],
   [Code0.apply_renaming], [apply_renaming_function_params_and_body]).
   Here the action is extended to what a state holds:
   - a value holds no variable and no continuation (closures hold code
     ids and values), so a renaming does nothing to a value, and no
     action on values is written;
   - an environment has its variables renamed
     (FlambdaAlphaSpec.rename_env);
   - a continuation closure has its handler, its environments and the
     handlers defined with it renamed, and a way out of the term is
     renamed, as it is observed; a continuation with the extra
     arguments of a call kept ([Extra_args]) is renamed inside, the
     arguments being values, which hold no variables or continuations;
     what is left of an over-application has its two continuations
     renamed, and the values it keeps hold no names;
   - the static environment has its code renamed. Symbols, code ids and
     slots are fixed by the action;
   - a state, the result of a step and a behaviour are renamed
     componentwise. The continuation of an exit is renamed.

   The statements:
   - the action on states is that of a group;
   - a step, a run and the behaviour of a state commute with it
     ([step_equivariant_statement], [run_equivariant_statement],
     [behaves_equivariant_statement]). They hold of every state and have
     no premise: nothing is asked of the code, which need not be
     closed. The renaming has to go into code for this: a call takes the
     parameters, [my_closure] and the two continuations out of the code
     and binds them in the environments
     (FlambdaSemanticsSpec.call_function);
   - so equivalence and refinement, of states and of expressions, are
     preserved by renaming, and a renaming that fixes the continuations
     through which a state leaves gives an equivalent state
     ([renamed_state_equivalent_statement]);
   - two rewrites of FlambdaNormaliseSpec.v commute with renaming: one
     that looks at one construct at a time ([fold_constants]), and
     inlining, which itself renames ([inline_calls]).

   What of the compiler this is. The action is that of the compiler's
   renamings that are permutations of variables and of continuations
   within one compilation unit: those made by [Renaming.add_variable],
   [add_fresh_variable], [add_continuation], [add_fresh_continuation]
   and [compose] from [Renaming.empty], which is what the freshening of
   a binder uses (the freshening of a static binder permutes no symbol
   and no code id: [Bound_pattern.renaming], at [Static]). It is not
   that of a renaming with an import map or with maps of bound names
   ([Renaming.create_import_map], used when the terms and types of
   another unit are read). Such a renaming can replace symbols and code
   ids, takes the value slots that are not used out of a set of
   closures ([Set_of_closures.apply_renaming]), may only be the first
   argument of [compose], and has no inverse: it is not a permutation,
   and the statements here about the identity, composition, inverses
   and execution say nothing of it. It is another operation, which this
   file does not model.

   Other differences from the compiler: [apply_renaming] takes the term
   first there and the renaming first here, so that it can be mapped
   over a list. The compiler delays a renaming at each name abstraction
   and composes the delayed ones; here it is applied at once, so a term
   here and the compiler's agree in what [pattern_match] shows of them
   up to the fresh names it chooses, and not as trees.

   A renaming is a list of swaps, and two lists may be the same
   permutation: the laws of a group hold of the action, not of the
   lists.

   Nothing is trusted here. The proofs are in FlambdaEquivarianceProofs.v,
   which needs no review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaNominalSpec FlambdaAlphaSpec FlambdaApplySpec
  FlambdaNormaliseSpec.
Import ListNotations.

(* ======================================================================== *)
(* The action on states                                                     *)
(* ======================================================================== *)

(* A continuation closure is renamed throughout; a way out of the term is
   a continuation, and is renamed. The extra arguments that a
   continuation is kept with are values, which hold no variables or
   continuations. What is left of an over-application is renamed in its
   two continuations. *)
Fixpoint rename_cont_value (r : renaming) (c : cont_value) : cont_value :=
  match c with
  | Closure h vars conts recursive =>
    Closure (rename_handler r h) (rename_env r vars)
      (map (fun '(k, c) => (rename_continuation r k, rename_cont_value r c))
         conts)
      (rename_handlers r recursive)
  | Outer k => Outer (rename_continuation r k)
  | Extra_args c extra => Extra_args (rename_cont_value r c) extra
  | Over_application remaining remaining_arity return_arity return_to exn_to =>
    Over_application remaining remaining_arity return_arity
      (rename_cont_value r return_to) (rename_cont_value r exn_to)
  end.

Definition rename_cont_env (r : renaming) (conts : cont_env) : cont_env :=
  map (fun '(k, c) => (rename_continuation r k, rename_cont_value r c)) conts.

(* The code of the static environment is renamed. The statically
   allocated values hold values, which hold no names. *)
Definition rename_static_env (r : renaming) (statics : static_env)
    : static_env :=
  Static_env (map_values (rename_code r) (static_code statics))
    (static_values statics).

Definition rename_state (r : renaming) (s : state) : state :=
  State (rename_expr r (state_expr s)) (rename_env r (state_vars s))
    (rename_cont_env r (state_conts s))
    (rename_static_env r (state_statics s)) (state_heap s) (state_reads s).

(* An exit is observed by its continuation, which is renamed, and its
   values, which hold no names. *)
Definition rename_result (r : renaming) (result : step_result)
    : step_result :=
  match result with
  | Step s => Step (rename_state r s)
  | Exit k vs hp rd => Exit (rename_continuation r k) vs hp rd
  | Undefined => Undefined
  end.

Definition rename_behaviour (r : renaming) (b : behaviour) : behaviour :=
  match b with
  | Exits k vs hp => Exits (rename_continuation r k) vs hp
  | Goes_wrong => Goes_wrong
  | Diverges => Diverges
  end.

Definition rename_state_identity_statement : Prop :=
  ∀ s : state, rename_state identity_renaming s = s.

Definition rename_state_compose_statement : Prop :=
  ∀ (second first : renaming) (s : state),
    rename_state (compose_renaming second first) s
    = rename_state second (rename_state first s).

Definition rename_state_inverse_statement : Prop :=
  ∀ (r : renaming) (s : state),
    rename_state (inverse_renaming r) (rename_state r s) = s
    ∧ rename_state r (rename_state (inverse_renaming r) s) = s.

(* ======================================================================== *)
(* Execution is equivariant                                                 *)
(* ======================================================================== *)

Definition step_equivariant_statement : Prop :=
  ∀ (r : renaming) (s : state),
    step (rename_state r s) = rename_result r (step s).

Definition run_equivariant_statement : Prop :=
  ∀ (r : renaming) (fuel : nat) (s : state),
    run fuel (rename_state r s) = rename_result r (run fuel s).

Definition behaves_equivariant_statement : Prop :=
  ∀ (r : renaming) (s : state) (b : behaviour),
    behaves (rename_state r s) (rename_behaviour r b) <-> behaves s b.

Definition equivalent_equivariant_statement : Prop :=
  ∀ (r : renaming) (s s' : state),
    equivalent s s' -> equivalent (rename_state r s) (rename_state r s').

Definition refines_equivariant_statement : Prop :=
  ∀ (r : renaming) (source target : state),
    refines source target ->
    refines (rename_state r source) (rename_state r target).

(* A rewrite that is correct with one choice of names is correct with
   every other. *)
Definition expr_equivalent_equivariant_statement : Prop :=
  ∀ (r : renaming) (e e' : expr),
    expr_equivalent e e' ->
    expr_equivalent (rename_expr r e) (rename_expr r e').

Definition expr_refines_equivariant_statement : Prop :=
  ∀ (r : renaming) (source target : expr),
    expr_refines source target ->
    expr_refines (rename_expr r source) (rename_expr r target).

(* What is observed of a state is the continuations through which it
   leaves. A renaming that fixes them does not change its behaviour. *)
Definition renamed_state_equivalent_statement : Prop :=
  ∀ (r : renaming) (s : state),
    (∀ (k : continuation) (vs : list value) (hp : heap),
      behaves s (Exits k vs hp) -> rename_continuation r k = k) ->
    equivalent s (rename_state r s).

(* ======================================================================== *)
(* Two rewrites are equivariant                                             *)
(* ======================================================================== *)

(* A rewrite of one construct at a time. *)
Definition fold_constants_equivariant_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    fold_constants (rename_expr r e) = rename_expr r (fold_constants e).

(* Inlining, which renames the continuations of the code it inlines. The
   code known is renamed with the expression. The code whose calls are
   left is named by code ids, which no renaming changes. *)
Definition inline_calls_equivariant_statement : Prop :=
  ∀ (r : renaming) (kept : list code_id) (known : list (code_id * code))
    (e : expr),
    inline_calls kept (map_values (rename_code r) known) (rename_expr r e)
    = rename_expr r (inline_calls kept known e).
