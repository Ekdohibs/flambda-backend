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


(* Parameters of continuations that hold a boxed number and are only
   unboxed or passed on are unboxed where they are given, for a set of
   positions at once: a normalisation that refines.

   This file is to be reviewed: [unbox_together_knowing] and what it
   is made of, which are extracted and run by the validation of
   middle_end/flambda2/tests/semantics/, and the statement. A round
   of the first phase of the validation applies the pass to the term
   before Simplify, after FlambdaUntagParamsSpec.untag_params
   (FlambdaValidateSpec.normalise_expr_params_knowing). The positions
   that the pass is run with ([together_positions]) are extracted and
   run too, and nothing is believed of them: [together_at] tests the
   positions that it is given.

   Simplify unboxes the parameters of a continuation
   (simplify/unboxing/): where the type of a parameter says that it
   is a boxed number, the continuation gets a parameter more for the
   number, each application gets the number as an argument more, and
   the boxed parameter goes if nothing uses it
   ([Optimistic_unboxing_decision.make_optimistic_number_decision];
   FlambdaUntagParamsSpec.v names the functions that add the
   parameter and the arguments). The type of an argument that is a
   parameter of another continuation is known where that one was
   unboxed. So an accumulator that the source keeps in a reference
   and sets on some turns of a loop only,

     let acc = ref 0L in
     for i = 1 to n do
       if i land 1 = 0 then acc := add !acc (of_int i)
     done;
     !acc

   which the front end makes a parameter of the loop and of the join
   after the test in its body, is a naked number in both after
   Simplify: the loop is given the boxed constant or the parameter of
   the join, and the join the parameter of the loop or a box that an
   arm made.

   FlambdaUnboxVariantsSpec.unbox_variants_knowing and
   FlambdaUnboxLoopsSpec.unbox_loops_knowing, which the rounds of the
   term before apply, unbox a parameter of one continuation where
   every application gives something that the term shows. Here
   neither continuation can be done alone: each is given the
   parameter of the other. This pass decides for a set of positions
   at once, as FlambdaUntagParamsSpec.v does for tagged immediates.
   It is made as that one is, and uses its definitions: the
   positions ([positions], [positions_of], [at_position]), the order
   of the parameters and of the arguments ([untag_params_list],
   [kept_from], [moved_from]), the scopes ([active_under],
   [untagged_under]), the test of a handler ([handler_fits]), and
   what the search for the positions is made of ([candidates],
   [origins], [prune] and their fellows). What differs is the read,
   and what an application may give.

   For one sort of box ([boxed]) and the positions given
   ([together_expr]):

   - the handler of a continuation named has those parameters after
     the others, in the reverse of the order that they had, and its
     kinds say that they are naked numbers of the sort
     ([together_handler], [together_metadata]);
   - an application of such a continuation has its arguments in that
     order ([together_args]). At those positions an argument that is
     a parameter that is unboxed stays; any other is replaced by the
     number that the term shows to be in it ([unbox_arg],
     [shown_number]): the constant, for a symbol or a variable that
     is known to be the box, of the sort, of a constant; what was
     boxed, for a variable that a [Let] in scope binds to the boxing
     of it. That is what FlambdaUnboxVariantsSpec.certain_projection
     gives for the unboxing with what is known where the application
     stands ([known], which the pass keeps as it goes down the term,
     by FlambdaUnboxParamsSpec.facts_under_let and its fellows); it
     is taken where it mentions no variable that is unboxed;
   - a [Let] of the unboxing of such a parameter, at the sort, is a
     [Let] of the parameter ([together_named]; the sorts are told
     apart by the kind of the number, [unbox_of]). The rounds follow
     the alias.

   What the pass asks ([together_ok]), given which continuations in
   scope have unboxed positions ([active]) and which variables in
   scope are unboxed parameters ([unboxed]):
   - an argument at a position that is unboxed is an unboxed
     parameter or has a number that the term shows ([unboxed_arg]);
   - a variable that is an unboxed parameter occurs nowhere but as
     such an argument and in the [Let] of its unboxing at the sort:
     not in another defining expression, not in a static binding, not
     in a call, not as another argument, not as the scrutinee of a
     [Switch];
   - no call returns to a continuation that has an unboxed position,
     and none raises to one: a call gives a continuation values that
     the term does not show;
   - the parameters of every handler are distinct, and the handler of
     a continuation that has an unboxed position is not an exception
     handler (FlambdaUntagParamsSpec.handler_fits).
   The scopes are those of FlambdaUntagParamsSpec.v. The handlers of
   recursive continuations are gone into; the bodies of code are not.

   [together_at] is the term rewritten where the test passes for the
   whole term, with nothing in scope unboxed, and the term as it is
   otherwise. [unbox_together_of] is that with the positions that
   the search finds ([together_positions]): every parameter of kind
   [Value] of every continuation that the term binds outside code,
   less those that the test would refuse ([together_blame]), until
   none is refused ([together_settle]). Nothing is claimed of them.
   [unbox_together_knowing] does it for each sort of box of which
   the term has, outside code, the unboxing of a variable, in turn
   ([unboxed_sorts], [sorts_once]); a term that has none is given
   back as it is, and nothing is searched.

   The result refines the term wherever what is known holds
   ([unbox_together_knowing_symbols_refines_statement], which is
   FlambdaUnboxInScopeSpec.refines_where_known of the pass): in
   every state whose static environment binds the symbols as they
   are known to be, what is known being of symbols alone. Why it
   holds: every variable that is an unboxed parameter holds, in the
   result, a value of the kind of the number, and that value is
   what the unboxing of the sort has on the value that the variable
   holds in the term, if that unboxing has a value, under the
   static environment and under any that binds more; nothing more
   is said of it where the unboxing has none. Each such variable
   was given another one, or the number that the term shows for a
   box. The two runs are in step, and the two terms give their
   continuations the same values, as no such variable is given to
   a continuation of the environment. It is stated as a refinement,
   which is what a rewrite of the term before owes; where the term
   is undefined the result may run.

   What it is not: it does not decide as Simplify decides. Simplify
   unboxes where the type of the parameter says a boxed number, and
   the type of an argument may say so where the term does not show
   the box (a parameter of the code, the result of a call, a field
   that was loaded); this pass asks for an unboxed parameter or a
   box that the term shows. And Simplify keeps the boxed parameter
   beside the naked one where something else uses it; this pass
   leaves such a parameter boxed. Where the two differ the two terms
   may stay different. Nothing is said here of which pairs of terms
   a validation that applies the pass accepts: what this file states
   is the refinement.

   Renaming: the functions here compare names by equality only and
   make none. FlambdaUnboxTogetherNamesProofs.v has the lemmas that
   the pass commutes with renaming and keeps scope; the statements
   are [unbox_together_knowing_equivariant_statement] in
   FlambdaUnboxVariantsRenameSpec.v and
   [unbox_together_knowing_symbols_keeps_scope_statement] in
   FlambdaUnboxInScopeRenameSpec.v.

   The proofs are in FlambdaUnboxTogetherProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool Arith.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaPrimitiveModel FlambdaSemanticsSpec
  FlambdaNormaliseSpec FlambdaUnboxParamsSpec FlambdaUnboxVariantsSpec
  FlambdaUnboxInScopeSpec FlambdaUntagParamsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The rewrite                                                              *)
(* ======================================================================== *)

(* The number that the term shows to be in a box, for an argument that
   is the box: a constant of the kind, or the number that a [Let] in
   scope boxed, where it is none of the variables that are unboxed. *)
Definition shown_number (boxed : boxable_number) (unboxed : list variable)
    (known : facts) (s : simple) : option simple :=
  match
    certain_projection known (Unbox_number boxed) (unboxed_kind boxed) s
  with
  | Some number =>
    if none_in unboxed (fun p => var_in_simple p number) then Some number
    else None
  | None => None
  end.

(* What a position that is unboxed is given for an argument: the
   argument, if it is a parameter that is unboxed; otherwise the number
   that the term shows to be in the box. *)
Definition unboxed_simple (boxed : boxable_number) (unboxed : list variable)
    (known : facts) (s : simple) : option simple :=
  match s with
  | Var x =>
    if among x unboxed then Some s else shown_number boxed unboxed known s
  | Symbol _ | Const _ => shown_number boxed unboxed known s
  end.

Definition unbox_arg (boxed : boxable_number) (unboxed : list variable)
    (known : facts) (s : simple) : simple :=
  match unboxed_simple boxed unboxed known s with
  | Some s' => s'
  | None => s
  end.

(* The arguments of an application of a continuation whose positions
   [us] are unboxed: the others first, then those, the last first. *)
Definition together_args (boxed : boxable_number) (us : list nat)
    (unboxed : list variable) (known : facts) (args : list simple)
    : list simple :=
  kept_from us 0 args
  ++ rev (map (unbox_arg boxed unboxed known) (moved_from us 0 args)).

Definition together_apply_cont (boxed : boxable_number) (active : positions)
    (unboxed : list variable) (known : facts) (ac : apply_cont)
    : apply_cont :=
  {| ac_continuation := ac_continuation ac;
     ac_args :=
       together_args boxed (positions_of active (ac_continuation ac))
         unboxed known (ac_args ac);
     ac_trap_action := ac_trap_action ac |}.

(* The variable that a defining expression unboxes, if it is the
   unboxing of a variable at the sort of box given. *)
Definition unbox_of (boxed : boxable_number) (d : named) : option variable :=
  match d with
  | Prim (Unary (Unbox_number boxed')) [Var p] =>
    if kind_eqb (unboxed_kind boxed') (unboxed_kind boxed) then Some p
    else None
  | _ => None
  end.

(* The unboxing of a variable that is unboxed is that variable. *)
Definition together_named (boxed : boxable_number) (unboxed : list variable)
    (d : named) : named :=
  match unbox_of boxed d with
  | Some p => if among p unboxed then Simple (Var p) else d
  | None => d
  end.

Definition together_metadata (boxed : boxable_number) (us : list nat)
    (metadata : handler_metadata) : handler_metadata :=
  {| hm_params_kinds :=
       kept_from us 0 (hm_params_kinds metadata)
       ++ map (fun _ => unboxed_kind boxed)
            (rev (moved_from us 0 (hm_params_kinds metadata)));
     hm_is_exn_handler := hm_is_exn_handler metadata |}.

Fixpoint together_expr (boxed : boxable_number) (sites active : positions)
    (unboxed : list variable) (known : facts) (e : expr) {struct e} : expr :=
  match e with
  | Let x kind d body =>
    Let x kind (together_named boxed unboxed d)
      (together_expr boxed sites active (without [x] unboxed)
         (facts_under_let known x d) body)
  | Let_static bindings body =>
    Let_static bindings
      (together_expr boxed sites active unboxed
         (facts_under_statics known bindings) body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k
      (together_handler boxed sites active unboxed known
         (positions_of sites k) h)
      (together_expr boxed sites ((k, positions_of sites k) :: active)
         unboxed known body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map
         (fun '(k, h) =>
            (k,
             together_handler boxed sites
               (active_under sites handlers active) unboxed known
               (positions_of sites k) h))
         handlers)
      (together_expr boxed sites (active_under sites handlers active)
         unboxed known body)
  | Apply _ => e
  | Apply_cont ac =>
    Apply_cont (together_apply_cont boxed active unboxed known ac)
  | Switch scrutinee arms =>
    Switch scrutinee
      (map
         (fun '(i, ac) =>
            (i, together_apply_cont boxed active unboxed known ac))
         arms)
  | Invalid => e
  end
with together_handler (boxed : boxable_number) (sites active : positions)
    (unboxed : list variable) (known : facts) (us : list nat)
    (h : continuation_handler) {struct h} : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler (untag_params_list us params) (together_metadata boxed us metadata)
      (together_expr boxed sites active (untagged_under us params unboxed)
         (facts_under_params known params) handler)
  end.

(* ======================================================================== *)
(* The test                                                                 *)
(* ======================================================================== *)

Definition unboxed_arg (boxed : boxable_number) (unboxed : list variable)
    (known : facts) (s : simple) : bool :=
  match unboxed_simple boxed unboxed known s with
  | Some _ => true
  | None => false
  end.

Definition together_args_ok (boxed : boxable_number)
    (unboxed : list variable) (known : facts) (us : list nat)
    (args : list simple) : bool :=
  none_in unboxed (fun p => var_in_simples p (kept_from us 0 args))
  && forallb (unboxed_arg boxed unboxed known) (moved_from us 0 args).

Definition together_apply_cont_ok (boxed : boxable_number)
    (active : positions) (unboxed : list variable) (known : facts)
    (ac : apply_cont) : bool :=
  together_args_ok boxed unboxed known
    (positions_of active (ac_continuation ac)) (ac_args ac).

Definition reads_unboxed (boxed : boxable_number) (unboxed : list variable)
    (d : named) : bool :=
  match unbox_of boxed d with
  | Some p => among p unboxed
  | None => false
  end.

Definition together_named_ok (boxed : boxable_number)
    (unboxed : list variable) (d : named) : bool :=
  reads_unboxed boxed unboxed d
  || none_in unboxed (fun p => var_in_named p d).

Fixpoint together_ok (boxed : boxable_number) (sites active : positions)
    (unboxed : list variable) (known : facts) (e : expr) {struct e} : bool :=
  match e with
  | Let x _ d body =>
    together_named_ok boxed unboxed d
    && together_ok boxed sites active (without [x] unboxed)
         (facts_under_let known x d) body
  | Let_static bindings body =>
    none_in unboxed (fun p => existsb (var_in_static_binding p) bindings)
    && together_ok boxed sites active unboxed
         (facts_under_statics known bindings) body
  | Let_cont_non_recursive k h body =>
    together_handler_ok boxed sites active unboxed known
      (positions_of sites k) h
    && together_ok boxed sites ((k, positions_of sites k) :: active) unboxed
         known body
  | Let_cont_recursive handlers body =>
    forallb
      (fun '(k, h) =>
         together_handler_ok boxed sites (active_under sites handlers active)
           unboxed known (positions_of sites k) h)
      handlers
    && together_ok boxed sites (active_under sites handlers active) unboxed
         known body
  | Apply ap =>
    none_in unboxed
      (fun p =>
         var_in_simple p (ap_callee ap) || var_in_simples p (ap_args ap)
         || var_in_simples p (exn_extra_simples ap))
    && no_positions active (ap_continuation ap)
    && no_positions active (ap_exn_continuation ap)
  | Apply_cont ac => together_apply_cont_ok boxed active unboxed known ac
  | Switch scrutinee arms =>
    none_in unboxed (fun p => var_in_simple p scrutinee)
    && forallb
         (fun '(_, ac) => together_apply_cont_ok boxed active unboxed known ac)
         arms
  | Invalid => true
  end
with together_handler_ok (boxed : boxable_number) (sites active : positions)
    (unboxed : list variable) (known : facts) (us : list nat)
    (h : continuation_handler) {struct h} : bool :=
  match h with
  | Handler params metadata handler =>
    handler_fits us params metadata
    && together_ok boxed sites active (untagged_under us params unboxed)
         (facts_under_params known params) handler
  end.

(* The pass, for the positions given: the term rewritten where the
   test passes, with nothing unboxed in scope. *)
Definition together_at (boxed : boxable_number) (sites : positions)
    (known : facts) (e : expr) : expr :=
  if together_ok boxed sites [] [] known e
  then together_expr boxed sites [] [] known e
  else e.

(* ======================================================================== *)
(* The positions that the pass is run with                                  *)
(* ======================================================================== *)

Fixpoint together_args_blame (boxed : boxable_number) (origin : origin_map)
    (known : facts) (k : continuation) (us : list nat) (i : nat)
    (args : list simple) : list (continuation * nat) :=
  match args with
  | [] => []
  | a :: args =>
    (if at_position us i
     then if unboxed_arg boxed (map fst origin) known a then [] else [(k, i)]
     else origins origin (fun x => var_in_simple x a))
    ++ together_args_blame boxed origin known k us (S i) args
  end.

Definition together_apply_cont_blame (boxed : boxable_number)
    (active : positions) (origin : origin_map) (known : facts)
    (ac : apply_cont) : list (continuation * nat) :=
  together_args_blame boxed origin known (ac_continuation ac)
    (positions_of active (ac_continuation ac)) 0 (ac_args ac).

Fixpoint together_blame (boxed : boxable_number) (sites active : positions)
    (origin : origin_map) (known : facts) (e : expr) {struct e}
    : list (continuation * nat) :=
  match e with
  | Let x _ d body =>
    (if reads_unboxed boxed (map fst origin) d then []
     else origins origin (fun y => var_in_named y d))
    ++ together_blame boxed sites active (unbind x origin)
         (facts_under_let known x d) body
  | Let_static bindings body =>
    origins origin (fun y => existsb (var_in_static_binding y) bindings)
    ++ together_blame boxed sites active origin
         (facts_under_statics known bindings) body
  | Let_cont_non_recursive k h body =>
    together_handler_blame boxed sites active origin known k h
    ++ together_blame boxed sites ((k, positions_of sites k) :: active) origin
         known body
  | Let_cont_recursive handlers body =>
    flat_map
      (fun '(k, h) =>
         together_handler_blame boxed sites
           (active_under sites handlers active) origin known k h)
      handlers
    ++ together_blame boxed sites (active_under sites handlers active) origin
         known body
  | Apply ap =>
    origins origin
      (fun y =>
         var_in_simple y (ap_callee ap) || var_in_simples y (ap_args ap)
         || var_in_simples y (exn_extra_simples ap))
    ++ all_of active (ap_continuation ap)
    ++ all_of active (ap_exn_continuation ap)
  | Apply_cont ac => together_apply_cont_blame boxed active origin known ac
  | Switch scrutinee arms =>
    origins origin (fun y => var_in_simple y scrutinee)
    ++ flat_map
         (fun '(_, ac) =>
            together_apply_cont_blame boxed active origin known ac)
         arms
  | Invalid => []
  end
with together_handler_blame (boxed : boxable_number)
    (sites active : positions) (origin : origin_map) (known : facts)
    (k : continuation) (h : continuation_handler) {struct h}
    : list (continuation * nat) :=
  match h with
  | Handler params metadata handler =>
    (if handler_fits (positions_of sites k) params metadata then []
     else map (fun u => (k, u)) (positions_of sites k))
    ++ together_blame boxed sites active
         (origins_under k (positions_of sites k) params origin)
         (facts_under_params known params) handler
  end.

Fixpoint together_settle (boxed : boxable_number) (fuel : nat)
    (sites : positions) (known : facts) (e : expr) : positions :=
  match fuel with
  | O => sites
  | S fuel =>
    match together_blame boxed sites [] [] known e with
    | [] => sites
    | refused => together_settle boxed fuel (prune refused sites) known e
    end
  end.

Definition together_positions (boxed : boxable_number) (known : facts)
    (e : expr) : positions :=
  together_settle boxed (S (count_positions (candidates e))) (candidates e)
    known e.

(* The pass for one sort of box, with the positions that it finds. *)
Definition unbox_together_of (boxed : boxable_number) (known : facts)
    (e : expr) : expr :=
  together_at boxed (together_positions boxed known e) known e.

(* The sorts of box that a term unboxes outside code, in the order in
   which it has the unboxings. *)
Fixpoint unboxed_sorts (e : expr) {struct e} : list boxable_number :=
  match e with
  | Let _ _ d body =>
    match d with
    | Prim (Unary (Unbox_number boxed)) [Var _] => boxed :: unboxed_sorts body
    | _ => unboxed_sorts body
    end
  | Let_static _ body => unboxed_sorts body
  | Let_cont_non_recursive _ h body =>
    handler_unboxed_sorts h ++ unboxed_sorts body
  | Let_cont_recursive handlers body =>
    flat_map (fun '(_, h) => handler_unboxed_sorts h) handlers
    ++ unboxed_sorts body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => []
  end
with handler_unboxed_sorts (h : continuation_handler) {struct h}
    : list boxable_number :=
  match h with
  | Handler _ _ handler => unboxed_sorts handler
  end.

(* Each sort once, the first time that it is there. *)
Fixpoint sorts_once (seen sorts : list boxable_number)
    : list boxable_number :=
  match sorts with
  | [] => []
  | boxed :: sorts =>
    if existsb
         (fun boxed' => kind_eqb (unboxed_kind boxed') (unboxed_kind boxed))
         seen
    then sorts_once seen sorts
    else boxed :: sorts_once (boxed :: seen) sorts
  end.

Fixpoint unbox_together_all (boxes : list boxable_number) (known : facts)
    (e : expr) : expr :=
  match boxes with
  | [] => e
  | boxed :: boxes =>
    unbox_together_all boxes known (unbox_together_of boxed known e)
  end.

(* The pass: for each sort of box that the term unboxes, in turn. *)
Definition unbox_together_knowing (known : facts) (e : expr) : expr :=
  unbox_together_all (sorts_once [] (unboxed_sorts e)) known e.

Definition unbox_together_knowing_symbols_refines_statement : Prop :=
  refines_where_known unbox_together_knowing.
