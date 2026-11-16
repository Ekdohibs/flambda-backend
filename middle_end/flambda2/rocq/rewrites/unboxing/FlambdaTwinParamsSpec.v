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


(* A parameter of a continuation that holds a tagged immediate, and
   that its handler has bound again at its head, gets a twin that
   holds the number, where every application can give one: a
   normalisation that refines.

   This file is to be reviewed: [twin_expr], [twin_ok], [twin_at] and
   [twin_untag_at] and what they are made of, which are extracted and
   run by the validation of middle_end/flambda2/tests/semantics/, and
   the two statements. A round of the second phase of the validation
   applies the pass to the term before Simplify once it has settled
   the term, just after FlambdaRenameApartSpec.rename_apart_avoiding
   (FlambdaValidateSpec.with_twins, which says why only then), which
   FlambdaShadowParamsSpec.shadow_params comes just before. The
   positions that the pass is run with ([twin_positions]) are
   extracted and run too, and nothing is believed of them: the
   statements are about any positions.

   Simplify unboxes the parameters of a continuation
   (simplify/unboxing/): where the type of a parameter says that it is
   a tagged immediate, the continuation gets a parameter more for the
   number, each application gets the number as an argument more, and
   the tagged parameter goes only if nothing else uses it
   ([Unbox_continuation_params.make_decisions],
   [Apply_cont_rewrite.make_rewrite]). So for a flag that a loop both
   tests and keeps,

     let any = ref false in
     for i = 0 to n do any := !any || i > limit done;
     !any

   the continuation where the arms of [||] meet has, after Simplify,
   the flag as a value, which is what the loop returns, and the flag
   as a number, which is what it tests.
   FlambdaUntagParamsSpec.untag_params makes a parameter the number
   where it is only untagged or passed on, and leaves it where
   something else uses it. This pass makes the second parameter.

   It makes no name. The term that it is given has, at the head of
   the handler, a [Let] of a variable to the parameter
   (FlambdaShadowParamsSpec.shadow_params binds the parameter again
   to itself, and FlambdaRenameApartSpec.rename_apart_avoiding gives
   that binder a name of its own):

     k (p) = let p' = p in body

   The rewrite ([twin_expr], for the positions given, which say for
   the name of a continuation which of its parameters get a twin)
   takes that [Let] away, makes [p'] the parameter at the position
   and [p] one more parameter, after those that the continuation had:

     k (p', p) = body

   The twins of a handler are in the order of their positions, and
   have the kinds that the handler gave the parameters
   ([twin_params_list], [twin_metadata]). An application of the
   continuation gives each twin the argument that it gives at the
   position, a second time ([twin_args]). So [p'] and [p] hold the
   same value, and until one of them is bound again ([twinned], the
   pairs in scope):
   - an argument at a position that has a twin, if it is such a [p'],
     is given the second time as its [p] ([twin_simple]);
   - a [Let] of the untagging of such a [p'] is a [Let] of the
     untagging of its [p] ([twin_named]).
   Nothing else changes: every other use of [p'] stays.

   What the rewrite asks ([twin_ok]):
   - the handler of a continuation whose positions have twins begins
     with the [Let]s of those parameters, in their order ([head_ok]);
     a [Let] of a variable to another variable may stand before each,
     and stays, if it binds and reads none of those parameters and
     none of the variables that take their places. Such a [Let] is
     there where the head of a handler binds two parameters again and
     only the second gets a twin: of
       k (p, q) = let p' = p in let q' = q in body
     the rewrite with a twin for [q] alone makes
       k (p, q', q) = let p' = p in body;
   - the new parameters of every handler are distinct
     ([handler_twins_fit]);
   - no call returns to a continuation that has a twin, and none
     raises to one: a call gives a continuation values that the term
     does not show.
   Scopes are as in FlambdaUntagParamsSpec.v: the positions go by the
   name of a continuation, and [active] says which continuations in
   scope have twins. The handlers of recursive continuations are gone
   into; the bodies of code are not:
   FlambdaNormaliseSpec.rewrite_code takes a pass there.

   [twin_at] is the term rewritten where the test passes for the
   whole term, and the term as it is otherwise. The result refines
   the term ([twin_at_refines_statement]) under all environments,
   whatever the positions. Why it holds: a parameter and its twin
   hold the one value that the application gave twice. The variable
   of a [Let] that went is a parameter in the result, and holds
   there, from the entry of the handler, what the [Let] gives it in
   the term; nothing reads it before the term has run that [Let]
   (the test asks it of the [Let]s that stay at the head), and the
   two runs have the same values in every other variable. The term
   takes one step more for each [Let] that went. It is stated as a
   refinement, which
   is what a rewrite of the term before owes: where the kind that a
   [Let] taken away gave its variable is not that of the value, the
   term is undefined and the result may run.

   The twins hold the value still. [twin_untag_at] is [twin_expr]
   followed by FlambdaUntagParamsSpec.untag_expr at the positions of
   the twins ([twin_sites]), which makes each twin the number, an
   argument for it the naked constant or the variable whose tagging
   the argument is, and the untagging of a twin the twin. It is the
   two rewrites where both tests pass ([twin_ok], and
   FlambdaUntagParamsSpec.untag_ok on the term with the twins), and
   the term as it is otherwise: no twin stays that is not untagged.
   The result refines the term ([twin_untag_at_refines_statement]),
   whatever the positions: by the statement above and
   FlambdaUntagParamsSpec.untag_at_refines_statement.

   The positions that the pass is run with ([twin_positions]): the
   parameters of kind [Value] of the handlers that are not exception
   handlers, outside code, that the [Let]s at the head of their
   handler bind again ([twin_candidates]), less those that the test
   of the rewrite refuses ([twin_blame]), and less those whose twin
   FlambdaUntagParamsSpec.blame refuses in the term with the twins
   ([twin_origin] gives the position of a twin), until none is
   refused ([twin_settle]). Nothing is claimed of what is left.
   Where no handler has such a [Let] the pass gives the term back.

   What it is not: it does not decide as Simplify decides (the header
   of FlambdaUntagParamsSpec.v says how the two differ); and it gives
   a twin only what that file's rewrite can untag: a tagged constant,
   a parameter that has a twin, or a variable that a [Let] in scope
   binds to the tagging of a variable. Nothing is said here of which
   pairs of terms a validation that applies the pass accepts.

   Renaming: the functions here compare names by equality only and
   make none; the name of a twin is the parameter's own, and the name
   that takes its place is that of a [Let] of the term.
   FlambdaTwinParamsNamesProofs.v has the lemmas that the pass
   commutes with renaming and keeps scope; the statements are in
   FlambdaValidatorEquivarianceSpec.v and FlambdaStaticsScopeSpec.v.

   The proofs are in FlambdaTwinParamsProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool Arith.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaPrimitiveModel FlambdaSemanticsSpec
  FlambdaNormaliseSpec FlambdaUntagParamsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The rewrite                                                              *)
(* ======================================================================== *)

(* The variable that a defining expression is another name of. *)
Definition alias_of (d : named) : option variable :=
  match d with
  | Simple (Var p) => Some p
  | _ => None
  end.

(* The variables in scope that were bound again to a parameter at the
   head of a handler and are parameters now, each with the twin of
   its parameter, which is in scope too and holds what it holds. *)
Definition twinned : Type := list (variable * variable).

(* The twin for a variable that has one. *)
Definition twin_simple (pairs : twinned) (s : simple) : simple :=
  match s with
  | Var x =>
    match lookup x pairs with
    | Some p => Var p
    | None => s
    end
  | _ => s
  end.

(* The arguments of an application of a continuation whose positions
   [ts] have twins: the arguments, then those at the positions again,
   in their order, each as its twin if it has one. *)
Definition twin_args (ts : list nat) (pairs : twinned) (args : list simple)
    : list simple :=
  args ++ map (twin_simple pairs) (moved_from ts 0 args).

Definition twin_apply_cont (active : positions) (pairs : twinned)
    (ac : apply_cont) : apply_cont :=
  {| ac_continuation := ac_continuation ac;
     ac_args :=
       twin_args (positions_of active (ac_continuation ac)) pairs
         (ac_args ac);
     ac_trap_action := ac_trap_action ac |}.

(* The untagging of a variable that has a twin is the untagging of the
   twin. *)
Definition twin_named (pairs : twinned) (d : named) : named :=
  match untag_of d with
  | Some v =>
    match lookup v pairs with
    | Some p => Prim (Unary Untag_immediate) [Var p]
    | None => d
    end
  | None => d
  end.

(* The variables of the [Let]s at the head of an expression that bind
   again the parameters given, one after the other in their order;
   a [Let] of a variable to another variable may stand before each. *)
Fixpoint twin_names (pending : list variable) (e : expr) {struct e}
    : list variable :=
  match pending, e with
  | p :: rest, Let x _ d body =>
    match alias_of d with
    | Some q =>
      if Pos.eqb q p then x :: twin_names rest body
      else twin_names pending body
    | None => []
    end
  | _, _ => []
  end.

(* The parameters with the names given, in their order, for those at
   the positions [ts]; [i] is the position of the first parameter. *)
Fixpoint renamed_from (ts : list nat) (i : nat) (params names : list variable)
    : list variable :=
  match params with
  | [] => []
  | p :: params =>
    if at_position ts i
    then
      match names with
      | x :: names => x :: renamed_from ts (S i) params names
      | [] => p :: renamed_from ts (S i) params []
      end
    else p :: renamed_from ts (S i) params names
  end.

(* The parameters of a handler whose positions [ts] have twins: at
   those positions the variables of the [Let]s at its head, and after
   the others the parameters that were there, in their order. *)
Definition twin_params_list (ts : list nat) (params : list variable)
    (handler : expr) : list variable :=
  renamed_from ts 0 params (twin_names (moved_from ts 0 params) handler)
  ++ moved_from ts 0 params.

Definition twin_metadata (ts : list nat) (metadata : handler_metadata)
    : handler_metadata :=
  {| hm_params_kinds :=
       hm_params_kinds metadata
       ++ moved_from ts 0 (hm_params_kinds metadata);
     hm_is_exn_handler := hm_is_exn_handler metadata |}.

(* [pending] is the parameters whose [Let]s at the head of the handler
   are still to come: each such [Let] goes, and its variable is from
   there the name of what the parameter was, with the parameter for
   its twin. A [Let] of a variable to another variable that stands
   before one of them stays. *)
Fixpoint twin_expr (sites active : positions) (pairs : twinned)
    (pending : list variable) (e : expr) {struct e} : expr :=
  match e with
  | Let x kind d body =>
    match pending, alias_of d with
    | p :: rest, Some q =>
      if Pos.eqb q p
      then twin_expr sites active ((x, p) :: tags_without [x] pairs) rest body
      else
        Let x kind d
          (twin_expr sites active (tags_without [x] pairs) pending body)
    | _, _ =>
      Let x kind (twin_named pairs d)
        (twin_expr sites active (tags_without [x] pairs) [] body)
    end
  | Let_static bindings body =>
    Let_static bindings (twin_expr sites active pairs [] body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k
      (twin_handler sites active pairs (positions_of sites k) h)
      (twin_expr sites ((k, positions_of sites k) :: active) pairs [] body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map
         (fun '(k, h) =>
            (k,
             twin_handler sites (active_under sites handlers active) pairs
               (positions_of sites k) h))
         handlers)
      (twin_expr sites (active_under sites handlers active) pairs [] body)
  | Apply _ => e
  | Apply_cont ac => Apply_cont (twin_apply_cont active pairs ac)
  | Switch scrutinee arms =>
    Switch scrutinee
      (map (fun '(i, ac) => (i, twin_apply_cont active pairs ac)) arms)
  | Invalid => e
  end
with twin_handler (sites active : positions) (pairs : twinned)
    (ts : list nat) (h : continuation_handler) {struct h}
    : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler (twin_params_list ts params handler) (twin_metadata ts metadata)
      (twin_expr sites active (tags_without params pairs)
         (moved_from ts 0 params) handler)
  end.

(* ======================================================================== *)
(* The test                                                                 *)
(* ======================================================================== *)

(* The expression begins with a [Let] of a variable to each of the
   parameters given, in their order. Before each there may be [Let]s
   of a variable to another variable, which bind and read none of the
   variables to avoid. *)
Fixpoint head_ok (avoid pending : list variable) (e : expr) {struct e}
    : bool :=
  match pending with
  | [] => true
  | p :: rest =>
    match e with
    | Let x _ d body =>
      match alias_of d with
      | Some q =>
        if Pos.eqb q p then head_ok avoid rest body
        else
          negb (among x avoid) && negb (among q avoid)
          && head_ok avoid pending body
      | None => false
      end
    | _ => false
    end
  end.

(* The handler begins with the [Let]s of its parameters at the
   positions [ts], and its new parameters are distinct. The [Let]s
   that stand between those bind and read none of the parameters at
   those positions and none of the variables that take their
   places. *)
Definition handler_twins_fit (ts : list nat) (params : list variable)
    (handler : expr) : bool :=
  distinct (twin_params_list ts params handler)
  && head_ok
       (twin_names (moved_from ts 0 params) handler
        ++ moved_from ts 0 params)
       (moved_from ts 0 params) handler.

Fixpoint twin_ok (sites active : positions) (e : expr) {struct e} : bool :=
  match e with
  | Let _ _ _ body | Let_static _ body => twin_ok sites active body
  | Let_cont_non_recursive k h body =>
    twin_handler_ok sites active (positions_of sites k) h
    && twin_ok sites ((k, positions_of sites k) :: active) body
  | Let_cont_recursive handlers body =>
    forallb
      (fun '(k, h) =>
         twin_handler_ok sites (active_under sites handlers active)
           (positions_of sites k) h)
      handlers
    && twin_ok sites (active_under sites handlers active) body
  | Apply ap =>
    no_positions active (ap_continuation ap)
    && no_positions active (ap_exn_continuation ap)
  | Apply_cont _ | Switch _ _ | Invalid => true
  end
with twin_handler_ok (sites active : positions) (ts : list nat)
    (h : continuation_handler) {struct h} : bool :=
  match h with
  | Handler params _ handler =>
    handler_twins_fit ts params handler && twin_ok sites active handler
  end.

(* The rewrite, for the positions given: the term rewritten where the
   test passes, and the term as it is otherwise. *)
Definition twin_at (sites : positions) (e : expr) : expr :=
  if twin_ok sites [] e then twin_expr sites [] [] [] e else e.

(* The result refines the term, under all environments, whatever the
   positions. *)
Definition twin_at_refines_statement : Prop :=
  ∀ (sites : positions) (e : expr), expr_refines e (twin_at sites e).

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

(* The positions of the twins in the result: for each continuation,
   as many as it has twins, after the parameters that it had. *)
Fixpoint twin_sites (sites : positions) (e : expr) {struct e} : positions :=
  match e with
  | Let _ _ _ body | Let_static _ body => twin_sites sites body
  | Let_cont_non_recursive k h body =>
    handler_twin_sites sites k h ++ twin_sites sites body
  | Let_cont_recursive handlers body =>
    flat_map (fun '(k, h) => handler_twin_sites sites k h) handlers
    ++ twin_sites sites body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => []
  end
with handler_twin_sites (sites : positions) (k : continuation)
    (h : continuation_handler) {struct h} : positions :=
  match h with
  | Handler params _ handler =>
    (k,
     seq (length params)
       (length (moved_from (positions_of sites k) 0 params)))
    :: twin_sites sites handler
  end.

(* The twins untagged: the rewrite, then
   FlambdaUntagParamsSpec.untag_expr at the positions of the twins,
   where the tests of both pass; the term as it is otherwise. *)
Definition twin_untag_at (sites : positions) (e : expr) : expr :=
  if twin_ok sites [] e
     && untag_ok (twin_sites sites e) [] [] [] (twin_expr sites [] [] [] e)
  then untag_expr (twin_sites sites e) [] [] [] (twin_expr sites [] [] [] e)
  else e.

Definition twin_untag_at_refines_statement : Prop :=
  ∀ (sites : positions) (e : expr), expr_refines e (twin_untag_at sites e).

(* ======================================================================== *)
(* The positions that the pass is run with                                  *)
(* ======================================================================== *)

(* What follows the [Let] that binds the parameter again, if the
   expression comes to it through [Let]s of a variable to another
   variable. *)
Fixpoint after_alias (p : variable) (e : expr) {struct e} : option expr :=
  match e with
  | Let _ _ d body =>
    match alias_of d with
    | Some q => if Pos.eqb q p then Some body else after_alias p body
    | None => None
    end
  | _ => None
  end.

(* The positions of the parameters of kind [Value] that the [Let]s at
   the head of the handler bind again, one after the other in the
   order of the parameters; [i] is the position of the first
   parameter. *)
Fixpoint head_positions (i : nat) (params : list variable)
    (kinds : list flambda_kind) (e : expr) : list nat :=
  match params, kinds with
  | p :: params, kind :: kinds =>
    match (if kind_eqb kind Value then after_alias p e else None) with
    | Some rest => i :: head_positions (S i) params kinds rest
    | None => head_positions (S i) params kinds e
    end
  | _, _ => []
  end.

Fixpoint twin_candidates (e : expr) {struct e} : positions :=
  match e with
  | Let _ _ _ body | Let_static _ body => twin_candidates body
  | Let_cont_non_recursive k h body =>
    handler_twin_candidates k h ++ twin_candidates body
  | Let_cont_recursive handlers body =>
    flat_map (fun '(k, h) => handler_twin_candidates k h) handlers
    ++ twin_candidates body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => []
  end
with handler_twin_candidates (k : continuation) (h : continuation_handler)
    {struct h} : positions :=
  match h with
  | Handler params metadata handler =>
    (k,
     if hm_is_exn_handler metadata then []
     else head_positions 0 params (hm_params_kinds metadata) handler)
    :: twin_candidates handler
  end.

(* The positions that the test of the rewrite refuses: all those of a
   continuation that a call returns or raises to, or whose handler
   does not fit. *)
Fixpoint twin_blame (sites active : positions) (e : expr) {struct e}
    : list (continuation * nat) :=
  match e with
  | Let _ _ _ body | Let_static _ body => twin_blame sites active body
  | Let_cont_non_recursive k h body =>
    twin_handler_blame sites active k h
    ++ twin_blame sites ((k, positions_of sites k) :: active) body
  | Let_cont_recursive handlers body =>
    flat_map
      (fun '(k, h) =>
         twin_handler_blame sites (active_under sites handlers active) k h)
      handlers
    ++ twin_blame sites (active_under sites handlers active) body
  | Apply ap =>
    all_of active (ap_continuation ap)
    ++ all_of active (ap_exn_continuation ap)
  | Apply_cont _ | Switch _ _ | Invalid => []
  end
with twin_handler_blame (sites active : positions) (k : continuation)
    (h : continuation_handler) {struct h} : list (continuation * nat) :=
  match h with
  | Handler params _ handler =>
    (if handler_twins_fit (positions_of sites k) params handler then []
     else map (fun u => (k, u)) (positions_of sites k))
    ++ twin_blame sites active handler
  end.

(* The position that the twin at a position of the result is the twin
   of: [new] has, for a continuation, the positions of its twins, the
   first of which is the number of the parameters that it had. *)
Definition twin_origin (sites new : positions) (o : continuation * nat)
    : list (continuation * nat) :=
  let '(k, u) := o in
  match positions_of new k with
  | n :: _ =>
    match nth_error (positions_of sites k) (u - n) with
    | Some i => [(k, i)]
    | None => []
    end
  | [] => []
  end.

(* Positions are taken away until the test of the rewrite refuses
   none, and FlambdaUntagParamsSpec.blame refuses none of the twins
   in the result, or no fuel is left. *)
Fixpoint twin_settle (fuel : nat) (sites : positions) (e : expr)
    : positions :=
  match fuel with
  | O => sites
  | S fuel =>
    match twin_blame sites [] e with
    | [] =>
      match
        blame (twin_sites sites e) [] [] [] (twin_expr sites [] [] [] e)
      with
      | [] => sites
      | refused =>
        twin_settle fuel
          (prune (flat_map (twin_origin sites (twin_sites sites e)) refused)
             sites)
          e
      end
    | refused => twin_settle fuel (prune refused sites) e
    end
  end.

Definition twin_positions (e : expr) : positions :=
  twin_settle (S (count_positions (twin_candidates e))) (twin_candidates e) e.

(* The pass, with the positions that it finds; the term as it is where
   no handler has such a [Let]. *)
Definition twin_params (e : expr) : expr :=
  match count_positions (twin_candidates e) with
  | O => e
  | S _ => twin_untag_at (twin_positions e) e
  end.
