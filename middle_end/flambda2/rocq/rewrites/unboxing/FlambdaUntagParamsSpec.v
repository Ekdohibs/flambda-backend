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


(* Parameters of continuations that hold a tagged immediate and are only
   untagged or passed on are untagged where they are given: a
   normalisation that refines.

   This file is to be reviewed: [untag_at] and what it is made of,
   which are extracted and run by the validation of
   middle_end/flambda2/tests/semantics/, and the statement. A round
   of the first phase of the validation applies the pass to the term
   before Simplify (FlambdaValidateSpec.normalise_expr_params_knowing).
   The positions that the pass is run with ([untag_positions]) are
   extracted and run too, and nothing is believed of them: the
   statement is about any positions.

   Simplify unboxes the parameters of a continuation
   (simplify/unboxing/): where the type of a parameter says that it is
   a tagged immediate, the continuation gets a parameter more for the
   number, each application gets the number as an argument more, and
   the tagged parameter goes if nothing uses it
   ([Unbox_continuation_params.make_decisions],
   [Apply_cont_rewrite.make_rewrite]). The decision for a continuation
   is taken from the types of the arguments that it is given, and the
   type of an argument that is a parameter of another continuation is
   known where that one was unboxed. So a flag that the source keeps
   in a reference, sets in a loop and tests after it,

     let seen = ref false in
     for i = 0 to n do if i < limit then seen := true done;
     if !seen then ... else ...

   which the front end makes a parameter of three continuations (the
   loop, the join after the test in its body, and the continuation
   after the loop), is a naked immediate in all three after Simplify:
   each is given a constant or the parameter of another of the three,
   and the one read is the untagging before the last test.

   FlambdaUnboxVariantsSpec.unbox_variant_params, which the rounds of
   the term before apply, unboxes a parameter of one continuation
   where every application gives something that the term shows. Here
   no continuation of the three can be done alone: the loop is given
   the parameter of the join, and the join the parameter of the loop.
   This pass decides for a set of positions at once.

   The positions ([positions]) say, for the name of a continuation,
   which of its parameters are untagged. With them ([untag_expr]):

   - the handler of a continuation named has those parameters after
     the others, in the reverse of the order that they had, and its
     kinds say that they are naked immediates ([untag_handler]). That
     is where Simplify has them: the new parameters come after those
     that the continuation had, the last one made first
     ([Continuation_extra_params_and_args.add]);
   - an application of such a continuation has its arguments in that
     order, and at those positions a tagged constant is the naked
     constant of the same number ([untag_args]); an argument there
     that is a variable is one of the parameters that are untagged,
     and stays, or one that a [Let] in scope binds to the tagging of
     a variable, and is that variable ([untag_simple]);
   - a [Let] of the untagging of such a parameter is a [Let] of the
     parameter ([untag_named]). The rounds follow the alias.

   What the pass asks ([untag_ok]), given which continuations in scope
   are untagged ([active]), which variables in scope are untagged
   parameters, and which variables in scope a [Let] binds to the
   tagging of which variable ([tags]):
   - an argument at a position that is untagged is a tagged constant,
     a variable that is an untagged parameter, or a variable that
     such a [Let] binds ([untagged_arg]);
   - a variable that is an untagged parameter occurs nowhere but as
     such an argument and in the [Let] of its untagging: not in
     another defining expression, not in a static binding, not in a
     call, not as another argument, not as the scrutinee of a
     [Switch];
   - no call returns to a continuation that has an untagged position,
     and none raises to one: a call gives a continuation values that
     the term does not show;
   - the parameters of every handler are distinct, and the handler of
     a continuation that has an untagged position is not an exception
     handler ([handler_fits]).
   Scopes: the positions go by the name of a continuation. A
   [Let_cont] of a name that the positions do not have hides, in its
   scope, what an outer binding of that name had ([active] is a map in
   which the first binding of a name is the one in scope); a variable
   bound again is no longer an untagged parameter
   (FlambdaMapSpec.without), and what was known of the tagging that
   it names or reads is dropped ([tags_without]): the variable that
   is put for an argument is in scope where the argument is, and
   holds there what the [Let] read. The handlers of recursive
   continuations are gone into, as a loop is where such parameters
   are; the bodies of code are not: FlambdaNormaliseSpec.rewrite_code
   takes a pass there, and a continuation of the code has nothing to
   do with one of the same name outside it.

   [untag_at] is the term rewritten where the test passes for the
   whole term, with nothing in scope untagged, and the term as it is
   otherwise.

   The result refines the term ([untag_at_refines_statement]) under
   all environments, whatever the positions. Why it holds: every
   variable that is an untagged parameter holds a tagged immediate in
   the term and the naked immediate of the same number in the result,
   as each was given a constant, another such variable, or in the
   term the tagging of a number and in the result that number; the
   two runs are in step, and the two terms give their continuations
   the same values, as no such variable is given to a continuation of
   the environment. It is stated as a refinement, which is what a
   rewrite of the term before owes; where the term is undefined (the
   untagging of a tagged immediate that is out of range) the result
   may run.

   The positions that the pass is run with ([untag_positions]): every
   parameter of kind [Value] of every continuation that the term
   binds outside code ([candidates]), less those that the test would
   refuse ([blame]), until none is refused ([settle_positions]). A
   position is refused with the others of its continuation where a
   call returns or raises to the continuation or its handler does not
   fit, alone where an application gives it something else than a
   tagged constant, an untagged parameter or the tagging of a
   variable, and for a variable that occurs where it may not, the
   position of that variable. Nothing is claimed of what is left: not
   that the test accepts it, and not that it is the largest set that
   the test accepts. The test is run again on it by [untag_at], which
   leaves the term as it is where the test refuses.

   What it is not: it does not decide as Simplify decides. Simplify
   untags where the type of the parameter says a tagged immediate,
   which the type of an argument may say whatever defines it, and
   then Simplify makes the number where the argument is given; this
   pass asks for a constant, an untagged parameter, or a variable
   that a [Let] in scope binds to the tagging of a variable, and
   makes no name. And Simplify keeps the tagged parameter beside the
   naked one where something else uses it; this pass leaves such a
   parameter tagged. Where the two differ the two terms may stay
   different. Nothing is said here of which pairs of terms a
   validation that applies the pass accepts: what this file states is
   the refinement.

   Renaming: the functions here compare names by equality only and
   make none. FlambdaUntagParamsNamesProofs.v has the lemmas that the
   pass commutes with renaming and keeps scope; the statements are in
   FlambdaValidatorEquivarianceSpec.v and FlambdaStaticsScopeSpec.v.

   The proofs are in FlambdaUntagParamsProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool Arith.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaPrimitiveModel FlambdaSemanticsSpec
  FlambdaNormaliseSpec.
Import ListNotations.

(* ======================================================================== *)
(* The positions                                                            *)
(* ======================================================================== *)

(* For the name of a continuation, the positions of its parameters
   that are untagged, counted from 0. *)
Definition positions : Type := list (continuation * list nat).

Definition positions_of (sites : positions) (k : continuation) : list nat :=
  match lookup k sites with
  | Some us => us
  | None => []
  end.

Definition at_position (us : list nat) (i : nat) : bool :=
  existsb (Nat.eqb i) us.

(* The elements of a list that are not at the positions given, and
   those that are, in the order of the list; [i] is the position of the
   first element. *)
Fixpoint kept_from {A : Type} (us : list nat) (i : nat) (l : list A)
    : list A :=
  match l with
  | [] => []
  | a :: l =>
    if at_position us i then kept_from us (S i) l
    else a :: kept_from us (S i) l
  end.

Fixpoint moved_from {A : Type} (us : list nat) (i : nat) (l : list A)
    : list A :=
  match l with
  | [] => []
  | a :: l =>
    if at_position us i then a :: moved_from us (S i) l
    else moved_from us (S i) l
  end.

(* ======================================================================== *)
(* The rewrite                                                              *)
(* ======================================================================== *)

(* The kind of an untagged parameter. *)
Definition untagged_kind : flambda_kind := Naked_number Kind_naked_immediate.

(* The variables in scope that a [Let] binds to the tagging of another
   variable, each with that other, which is in scope too and is the
   one that the [Let] read. *)
Definition tags : Type := list (variable * variable).

(* The variable that a defining expression tags, if it is the tagging
   of a variable. *)
Definition tag_of (d : named) : option variable :=
  match d with
  | Prim (Unary Tag_immediate) [Var n] => Some n
  | _ => None
  end.

(* The tags that mention none of the variables given, which are bound
   again. *)
Definition tags_without (xs : list variable) (tagged : tags) : tags :=
  filter (fun '(v, n) => negb (among v xs) && negb (among n xs)) tagged.

(* The tags under a [Let] of [x] to [d]. *)
Definition tags_under_let (x : variable) (d : named) (tagged : tags) : tags :=
  match tag_of d with
  | Some n =>
    if Pos.eqb n x then tags_without [x] tagged
    else (x, n) :: tags_without [x] tagged
  | None => tags_without [x] tagged
  end.

(* The naked constant of a tagged one, and the variable that a
   variable is the tagging of. *)
Definition untag_simple (tagged : tags) (s : simple) : simple :=
  match s with
  | Const (Tagged_immediate i) => Const (Naked_immediate i)
  | Var v =>
    match lookup v tagged with
    | Some n => Var n
    | None => s
    end
  | _ => s
  end.

(* The arguments of an application of a continuation whose positions
   [us] are untagged: the others first, then those, the last first. *)
Definition untag_args (us : list nat) (tagged : tags) (args : list simple)
    : list simple :=
  kept_from us 0 args
  ++ rev (map (untag_simple tagged) (moved_from us 0 args)).

Definition untag_apply_cont (active : positions) (tagged : tags)
    (ac : apply_cont) : apply_cont :=
  {| ac_continuation := ac_continuation ac;
     ac_args :=
       untag_args (positions_of active (ac_continuation ac)) tagged
         (ac_args ac);
     ac_trap_action := ac_trap_action ac |}.

(* The variable that a defining expression untags, if it is the
   untagging of a variable. *)
Definition untag_of (d : named) : option variable :=
  match d with
  | Prim (Unary Untag_immediate) [Var p] => Some p
  | _ => None
  end.

(* The untagging of a variable that is untagged is that variable. *)
Definition untag_named (untagged : list variable) (d : named) : named :=
  match untag_of d with
  | Some p => if among p untagged then Simple (Var p) else d
  | None => d
  end.

(* The parameters and the kinds of a handler whose positions [us] are
   untagged. *)
Definition untag_params_list (us : list nat) (params : list variable)
    : list variable :=
  kept_from us 0 params ++ rev (moved_from us 0 params).

Definition untag_metadata (us : list nat) (metadata : handler_metadata)
    : handler_metadata :=
  {| hm_params_kinds :=
       kept_from us 0 (hm_params_kinds metadata)
       ++ map (fun _ => untagged_kind)
            (rev (moved_from us 0 (hm_params_kinds metadata)));
     hm_is_exn_handler := hm_is_exn_handler metadata |}.

(* The untagged variables in a handler: its untagged parameters, and
   those of outside that its parameters do not hide. *)
Definition untagged_under (us : list nat) (params untagged : list variable)
    : list variable :=
  moved_from us 0 params ++ without params untagged.

(* What is in scope in the handlers and in the body of a recursive
   [Let_cont]. *)
Definition active_under (sites : positions)
    (handlers : list (continuation * continuation_handler))
    (active : positions) : positions :=
  map (fun '(k, _) => (k, positions_of sites k)) handlers ++ active.

Fixpoint untag_expr (sites active : positions) (untagged : list variable)
    (tagged : tags) (e : expr) {struct e} : expr :=
  match e with
  | Let x kind d body =>
    Let x kind (untag_named untagged d)
      (untag_expr sites active (without [x] untagged)
         (tags_under_let x d tagged) body)
  | Let_static bindings body =>
    Let_static bindings (untag_expr sites active untagged tagged body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k
      (untag_handler sites active untagged tagged (positions_of sites k) h)
      (untag_expr sites ((k, positions_of sites k) :: active) untagged tagged
         body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map
         (fun '(k, h) =>
            (k,
             untag_handler sites (active_under sites handlers active) untagged
               tagged (positions_of sites k) h))
         handlers)
      (untag_expr sites (active_under sites handlers active) untagged tagged
         body)
  | Apply _ => e
  | Apply_cont ac => Apply_cont (untag_apply_cont active tagged ac)
  | Switch scrutinee arms =>
    Switch scrutinee
      (map (fun '(i, ac) => (i, untag_apply_cont active tagged ac)) arms)
  | Invalid => e
  end
with untag_handler (sites active : positions) (untagged : list variable)
    (tagged : tags) (us : list nat) (h : continuation_handler) {struct h}
    : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler (untag_params_list us params) (untag_metadata us metadata)
      (untag_expr sites active (untagged_under us params untagged)
         (tags_without params tagged) handler)
  end.

(* ======================================================================== *)
(* The test                                                                 *)
(* ======================================================================== *)

(* No untagged variable is one that the test says may occur. *)
Definition none_in (untagged : list variable) (mentions : variable -> bool)
    : bool :=
  forallb (fun p => negb (mentions p)) untagged.

(* What an untagged position may be given. *)
Definition untagged_arg (untagged : list variable) (tagged : tags)
    (s : simple) : bool :=
  match s with
  | Const (Tagged_immediate _) => true
  | Var x =>
    among x untagged
    || match lookup x tagged with
       | Some _ => true
       | None => false
       end
  | _ => false
  end.

Definition args_ok (untagged : list variable) (tagged : tags)
    (us : list nat) (args : list simple) : bool :=
  none_in untagged (fun p => var_in_simples p (kept_from us 0 args))
  && forallb (untagged_arg untagged tagged) (moved_from us 0 args).

Definition apply_cont_ok (active : positions) (untagged : list variable)
    (tagged : tags) (ac : apply_cont) : bool :=
  args_ok untagged tagged (positions_of active (ac_continuation ac))
    (ac_args ac).

(* A defining expression is the untagging of an untagged variable, or
   mentions none. *)
Definition reads_untagged (untagged : list variable) (d : named) : bool :=
  match untag_of d with
  | Some p => among p untagged
  | None => false
  end.

Definition named_ok (untagged : list variable) (d : named) : bool :=
  reads_untagged untagged d
  || none_in untagged (fun p => var_in_named p d).

Definition no_positions (active : positions) (k : continuation) : bool :=
  match positions_of active k with
  | [] => true
  | _ :: _ => false
  end.

(* The parameters of every handler are distinct, and a handler that
   has an untagged position is not an exception handler. *)
Definition handler_fits (us : list nat) (params : list variable)
    (metadata : handler_metadata) : bool :=
  distinct params
  && match us with
     | [] => true
     | _ :: _ => negb (hm_is_exn_handler metadata)
     end.

Fixpoint untag_ok (sites active : positions) (untagged : list variable)
    (tagged : tags) (e : expr) {struct e} : bool :=
  match e with
  | Let x _ d body =>
    named_ok untagged d
    && untag_ok sites active (without [x] untagged)
         (tags_under_let x d tagged) body
  | Let_static bindings body =>
    none_in untagged (fun p => existsb (var_in_static_binding p) bindings)
    && untag_ok sites active untagged tagged body
  | Let_cont_non_recursive k h body =>
    handler_ok sites active untagged tagged (positions_of sites k) h
    && untag_ok sites ((k, positions_of sites k) :: active) untagged tagged
         body
  | Let_cont_recursive handlers body =>
    forallb
      (fun '(k, h) =>
         handler_ok sites (active_under sites handlers active) untagged
           tagged (positions_of sites k) h)
      handlers
    && untag_ok sites (active_under sites handlers active) untagged tagged
         body
  | Apply ap =>
    none_in untagged
      (fun p =>
         var_in_simple p (ap_callee ap) || var_in_simples p (ap_args ap)
         || var_in_simples p (exn_extra_simples ap))
    && no_positions active (ap_continuation ap)
    && no_positions active (ap_exn_continuation ap)
  | Apply_cont ac => apply_cont_ok active untagged tagged ac
  | Switch scrutinee arms =>
    none_in untagged (fun p => var_in_simple p scrutinee)
    && forallb (fun '(_, ac) => apply_cont_ok active untagged tagged ac) arms
  | Invalid => true
  end
with handler_ok (sites active : positions) (untagged : list variable)
    (tagged : tags) (us : list nat) (h : continuation_handler) {struct h}
    : bool :=
  match h with
  | Handler params metadata handler =>
    handler_fits us params metadata
    && untag_ok sites active (untagged_under us params untagged)
         (tags_without params tagged) handler
  end.

(* The pass, for the positions given: the term rewritten where the
   test passes, with nothing untagged in scope. *)
Definition untag_at (sites : positions) (e : expr) : expr :=
  if untag_ok sites [] [] [] e then untag_expr sites [] [] [] e else e.

(* The result refines the term, under all environments, whatever the
   positions. *)
Definition untag_at_refines_statement : Prop :=
  ∀ (sites : positions) (e : expr), expr_refines e (untag_at sites e).

(* ======================================================================== *)
(* The positions that the pass is run with                                  *)
(* ======================================================================== *)

(* The positions of the parameters of kind [Value]. *)
Fixpoint value_positions (i : nat) (kinds : list flambda_kind) : list nat :=
  match kinds with
  | [] => []
  | kind :: kinds =>
    if kind_eqb kind Value then i :: value_positions (S i) kinds
    else value_positions (S i) kinds
  end.

(* Every parameter of kind [Value] of every continuation that a term
   binds outside code. *)
Fixpoint candidates (e : expr) {struct e} : positions :=
  match e with
  | Let _ _ _ body | Let_static _ body => candidates body
  | Let_cont_non_recursive k h body =>
    handler_candidates k h ++ candidates body
  | Let_cont_recursive handlers body =>
    flat_map (fun '(k, h) => handler_candidates k h) handlers
    ++ candidates body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => []
  end
with handler_candidates (k : continuation) (h : continuation_handler)
    {struct h} : positions :=
  match h with
  | Handler _ metadata handler =>
    (k, value_positions 0 (hm_params_kinds metadata)) :: candidates handler
  end.

(* A position: a continuation and the number of a parameter. For each
   untagged variable in scope, the position that it is the parameter
   of. *)
Definition origin_map : Type := list (variable * (continuation * nat)).

(* The positions of the variables that the test says occur. *)
Definition origins (origin : origin_map) (mentions : variable -> bool)
    : list (continuation * nat) :=
  flat_map (fun '(x, o) => if mentions x then [o] else []) origin.

(* All the positions of a continuation. *)
Definition all_of (active : positions) (k : continuation)
    : list (continuation * nat) :=
  map (fun u => (k, u)) (positions_of active k).

(* The origins of the untagged parameters of a handler. *)
Fixpoint params_origins (k : continuation) (us : list nat) (i : nat)
    (params : list variable) : origin_map :=
  match params with
  | [] => []
  | p :: params =>
    if at_position us i then (p, (k, i)) :: params_origins k us (S i) params
    else params_origins k us (S i) params
  end.

Definition origins_under (k : continuation) (us : list nat)
    (params : list variable) (origin : origin_map) : origin_map :=
  params_origins k us 0 params
  ++ filter (fun '(x, _) => negb (among x params)) origin.

(* The positions that an application refuses: a position that is not
   given a tagged constant, an untagged variable or a variable of the
   tags, and that of an untagged variable given at another position. *)
Fixpoint args_blame (origin : origin_map) (tagged : tags) (k : continuation)
    (us : list nat) (i : nat) (args : list simple)
    : list (continuation * nat) :=
  match args with
  | [] => []
  | a :: args =>
    (if at_position us i
     then if untagged_arg (map fst origin) tagged a then [] else [(k, i)]
     else origins origin (fun x => var_in_simple x a))
    ++ args_blame origin tagged k us (S i) args
  end.

Definition apply_cont_blame (active : positions) (origin : origin_map)
    (tagged : tags) (ac : apply_cont) : list (continuation * nat) :=
  args_blame origin tagged (ac_continuation ac)
    (positions_of active (ac_continuation ac)) 0 (ac_args ac).

(* The positions that the test refuses in a term. *)
Fixpoint blame (sites active : positions) (origin : origin_map)
    (tagged : tags) (e : expr) {struct e} : list (continuation * nat) :=
  match e with
  | Let x _ d body =>
    (if reads_untagged (map fst origin) d then []
     else origins origin (fun y => var_in_named y d))
    ++ blame sites active (unbind x origin) (tags_under_let x d tagged) body
  | Let_static bindings body =>
    origins origin
      (fun y => existsb (var_in_static_binding y) bindings)
    ++ blame sites active origin tagged body
  | Let_cont_non_recursive k h body =>
    handler_blame sites active origin tagged k h
    ++ blame sites ((k, positions_of sites k) :: active) origin tagged body
  | Let_cont_recursive handlers body =>
    flat_map
      (fun '(k, h) =>
         handler_blame sites (active_under sites handlers active) origin
           tagged k h)
      handlers
    ++ blame sites (active_under sites handlers active) origin tagged body
  | Apply ap =>
    origins origin
      (fun y =>
         var_in_simple y (ap_callee ap) || var_in_simples y (ap_args ap)
         || var_in_simples y (exn_extra_simples ap))
    ++ all_of active (ap_continuation ap)
    ++ all_of active (ap_exn_continuation ap)
  | Apply_cont ac => apply_cont_blame active origin tagged ac
  | Switch scrutinee arms =>
    origins origin (fun y => var_in_simple y scrutinee)
    ++ flat_map (fun '(_, ac) => apply_cont_blame active origin tagged ac)
         arms
  | Invalid => []
  end
with handler_blame (sites active : positions) (origin : origin_map)
    (tagged : tags) (k : continuation) (h : continuation_handler) {struct h}
    : list (continuation * nat) :=
  match h with
  | Handler params metadata handler =>
    (if handler_fits (positions_of sites k) params metadata then []
     else map (fun u => (k, u)) (positions_of sites k))
    ++ blame sites active
         (origins_under k (positions_of sites k) params origin)
         (tags_without params tagged) handler
  end.

(* The positions without those refused. *)
Definition blamed (refused : list (continuation * nat)) (k : continuation)
    (u : nat) : bool :=
  existsb (fun '(k', u') => Pos.eqb k' k && Nat.eqb u' u) refused.

Definition prune (refused : list (continuation * nat)) (sites : positions)
    : positions :=
  map
    (fun '(k, us) => (k, filter (fun u => negb (blamed refused k u)) us))
    sites.

(* Positions are taken away until none is refused, or no fuel is
   left. *)
Fixpoint settle_positions (fuel : nat) (sites : positions) (e : expr)
    : positions :=
  match fuel with
  | O => sites
  | S fuel =>
    match blame sites [] [] [] e with
    | [] => sites
    | refused => settle_positions fuel (prune refused sites) e
    end
  end.

Definition count_positions (sites : positions) : nat :=
  list_sum (map (fun '(_, us) => length us) sites).

Definition untag_positions (e : expr) : positions :=
  settle_positions (S (count_positions (candidates e))) (candidates e) e.

(* The pass, with the positions that it finds. *)
Definition untag_params (e : expr) : expr := untag_at (untag_positions e) e.
