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

(* Examples for the checks of exception handlers and of trap actions in
   FlambdaWellFormedSpec.v ([unit_exn_handlers_well_formed],
   [traps_well_formed]): terms that each accepts and terms that it
   refuses. Then the proofs of FlambdaTrapSpec.v: the machine with a
   stack of handlers agrees with the machine that goes by names on a
   unit whose binders are distinct and that follows the discipline of
   trap actions, and not without the discipline; which [Apply_cont] is
   a raise; and the machine with the stack is equivariant. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaBindersSpec FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaNominalSpec FlambdaNominalProofs
  FlambdaAlphaSpec FlambdaAlphaProofs FlambdaEquivarianceSpec
  FlambdaEquivarianceProofs FlambdaWellFormedSpec FlambdaTrapSpec.
Import ListNotations.

(* So that the checks are not satisfied, or refused, for nothing: each
   example is a small term with the shape of what the front end builds
   for [try ... with], or of a mistake in it. The unit's exception
   continuation is [error]; [leave] is a continuation that the term does
   not bind, a way out. *)
Section Examples.

Local Open Scope positive_scope.

Definition error : continuation := 1.
Definition leave : continuation := 2.
Definition handler : continuation := 3.
Definition body : continuation := 4.
Definition inner : continuation := 5.
Definition loop : continuation := 6.
Definition x : variable := 1.

(* A handler with no parameter that is not an exception handler, and an
   exception handler with a parameter for the exception. *)
Definition plain (e : expr) : continuation_handler :=
  Handler [] (make_handler_metadata []) e.

Definition catches (e : expr) : continuation_handler :=
  Handler [x] (make_exn_handler_metadata [Value]) e.

Definition goto (k : continuation) (action : option trap_action)
    : expr :=
  Apply_cont (Apply_cont_expr k [] action).

Definition raise_to (k : continuation) : expr :=
  Apply_cont
    (Apply_cont_expr k [Const (Tagged_immediate 0)]
       (Some (Pop k (Some No_trace)))).

(* [try body with x -> leave]: the handler is installed on the way into
   the body, and [in_body] is what the body does. *)
Definition try_with (in_body : expr) : expr :=
  Let_cont_non_recursive handler (catches (goto leave None))
    (Let_cont_non_recursive body (plain in_body)
       (goto body (Some (Push handler)))).

Definition call_with (exn : continuation)
    (extra : list (simple * flambda_kind)) : expr :=
  Apply
    (Apply_expr (Var x) Indirect_unknown_arity
      [] leave exn extra [] [Singleton Value]).

Definition call (exn : continuation) : expr :=
  call_with exn [(Const (Tagged_immediate 1), Value)].

Definition code_of (e : expr) : code :=
  Code [] x leave error (make_code_metadata [] [Singleton Value]) e.

(* The body raises to the handler. *)
Example raise_in_try :
  traps_well_formed error (try_with (raise_to handler)) = true.
Proof. reflexivity. Qed.

(* The body leaves, removing the handler on the way. *)
Example leave_try :
  traps_well_formed error
    (try_with (goto leave (Some (Pop handler None))))
  = true.
Proof. reflexivity. Qed.

(* The body leaves with the handler still installed. *)
Example leave_try_without_pop :
  traps_well_formed error (try_with (goto leave None)) = false.
Proof. reflexivity. Qed.

(* The body raises to the exception continuation of the unit, over the
   handler. The machine of the semantics goes to [error]; the backend's
   code would go to the handler. *)
Example raise_over_handler :
  traps_well_formed error (try_with (raise_to error)) = false.
Proof. reflexivity. Qed.

(* Outside a [try], a raise to the exception continuation of the unit. *)
Example raise_to_unit : traps_well_formed error (raise_to error) = true.
Proof. reflexivity. Qed.

(* A trap action that removes a handler that is not installed. *)
Example pop_not_installed :
  traps_well_formed error
    (Let_cont_non_recursive handler (catches (goto leave None))
       (goto leave (Some (Pop handler None))))
  = false.
Proof. reflexivity. Qed.

(* A call in the body, with the handler, and then the unit's exception
   continuation, as its exception continuation. Extra arguments make no
   difference. *)
Example call_in_try :
  traps_well_formed error
    (try_with
       (Let_cont_non_recursive leave
          (plain (goto 7 (Some (Pop handler None))))
          (call handler)))
  = true.
Proof. reflexivity. Qed.

Example call_over_handler :
  traps_well_formed error
    (try_with
       (Let_cont_non_recursive leave
          (plain (goto 7 (Some (Pop handler None))))
          (call error)))
  = false.
Proof. reflexivity. Qed.

(* A loop in the body, entered and repeated with the handler installed,
   and one that installs the handler again each time round. *)
Example loop_in_try :
  traps_well_formed error
    (try_with
       (Let_cont_recursive [(loop, plain (goto loop None))]
          (goto loop None)))
  = true.
Proof. reflexivity. Qed.

Example loop_pushing :
  traps_well_formed error
    (Let_cont_non_recursive handler (catches (goto leave None))
       (Let_cont_recursive
          [(loop, plain (goto loop (Some (Push handler))))]
          (goto loop None)))
  = false.
Proof. reflexivity. Qed.

(* A continuation defined outside the [try] and entered from inside it:
   the stack with which it is entered is that of its application, not
   that of its definition. *)
Example defined_outside_entered_inside :
  traps_well_formed error
    (Let_cont_non_recursive handler (catches (goto leave None))
       (Let_cont_non_recursive inner
          (plain (goto leave (Some (Pop handler None))))
          (Let_cont_non_recursive body (plain (goto inner None))
             (goto body (Some (Push handler))))))
  = true.
Proof. reflexivity. Qed.

(* Code: a raise to its exception continuation, and a return with a
   handler still installed. *)
Example code_raises :
  code_traps_well_formed (code_of (raise_to error)) = true.
Proof. reflexivity. Qed.

Example code_returns_in_try :
  code_traps_well_formed (code_of (try_with (goto leave None))) = false.
Proof. reflexivity. Qed.

(* Exception handlers apart from the others: a handler that is also
   applied as an ordinary continuation, one with no parameter, and a
   recursive one. *)
Example handler_well_used :
  unit_exn_handlers_well_formed error (try_with (raise_to handler)) = true.
Proof. reflexivity. Qed.

Example handler_applied :
  unit_exn_handlers_well_formed error
    (try_with
       (Apply_cont
          (Apply_cont_expr handler [Const (Tagged_immediate 0)] None)))
  = false.
Proof. reflexivity. Qed.

Example handler_without_parameter :
  unit_exn_handlers_well_formed error
    (Let_cont_non_recursive handler
       (Handler [] (make_exn_handler_metadata []) (goto leave None))
       (goto leave (Some (Push handler))))
  = false.
Proof. reflexivity. Qed.

Example recursive_handler :
  unit_exn_handlers_well_formed error
    (Let_cont_recursive [(handler, catches (goto leave None))]
       (goto leave (Some (Push handler))))
  = false.
Proof. reflexivity. Qed.

(* What a handler says it is, and how it is used. A handler that is
   installed and raised to and does not say that it is an exception
   handler; one that says so and is never installed, which is allowed;
   one of a recursive [Let_cont] that says so and is not used at all;
   and one of a recursive [Let_cont] that does not say so and is
   installed. *)
Example handler_not_marked :
  unit_exn_handlers_well_formed error
    (Let_cont_non_recursive handler
       (Handler [x] (make_handler_metadata [Value]) (goto leave None))
       (Let_cont_non_recursive body (plain (raise_to handler))
          (goto body (Some (Push handler)))))
  = false.
Proof. reflexivity. Qed.

Example handler_marked_and_unused :
  unit_exn_handlers_well_formed error
    (Let_cont_non_recursive handler (catches (goto leave None))
       (goto leave None))
  = true.
Proof. reflexivity. Qed.

Example recursive_handler_marked :
  unit_exn_handlers_well_formed error
    (Let_cont_recursive [(handler, catches (goto leave None))]
       (goto leave None))
  = false.
Proof. reflexivity. Qed.

Example recursive_handler_not_marked :
  unit_exn_handlers_well_formed error
    (Let_cont_recursive
       [(handler,
         Handler [x] (make_handler_metadata [Value]) (goto leave None))]
       (goto leave (Some (Push handler))))
  = false.
Proof. reflexivity. Qed.

(* The continuations that a unit, or code, does not bind. A way out that
   is not the exception continuation, installed as an exception handler
   and raised to: the stack is in order, and the term is refused as it
   uses the return of the unit, or of the code, as a handler. *)
Definition return_as_handler : expr :=
  Let_cont_non_recursive body (plain (raise_to leave))
    (goto body (Some (Push leave))).

Example unit_return_as_handler_stack :
  traps_well_formed error return_as_handler = true.
Proof. reflexivity. Qed.

Example unit_return_as_handler :
  unit_exn_handlers_well_formed error return_as_handler = false.
Proof. reflexivity. Qed.

Example code_return_as_handler :
  code_exn_handlers_well_formed (code_of return_as_handler) = false.
Proof. reflexivity. Qed.

(* The exception continuation of a unit, or of code, is for calls and
   raises: here it is removed on the way to a continuation that installs
   it again. The stack is in order; the term is refused as it names the
   exception continuation in a trap action that is not that of a
   raise. *)
Definition pop_and_push_of_exn : expr :=
  Let_cont_non_recursive body (plain (goto leave (Some (Push error))))
    (goto body (Some (Pop error None))).

Example pop_and_push_of_exn_stack :
  traps_well_formed error pop_and_push_of_exn = true.
Proof. reflexivity. Qed.

Example pop_and_push_of_unit_exn :
  unit_exn_handlers_well_formed error pop_and_push_of_exn = false.
Proof. reflexivity. Qed.

Example pop_and_push_of_code_exn :
  code_exn_handlers_well_formed (code_of pop_and_push_of_exn) = false.
Proof. reflexivity. Qed.

(* Code whose return and exception continuations are one name: a raise
   to it would return. The check of trap actions does not see it; the
   check of binders, which a well-formed unit is asked first, refuses
   it. *)
Definition code_of_one_continuation : code :=
  Code [] x leave leave (make_code_metadata [] [Singleton Value])
    (raise_to leave).

Example code_of_one_continuation_stack :
  code_traps_well_formed code_of_one_continuation = true.
Proof. reflexivity. Qed.

Example code_of_one_continuation_binders :
  FlambdaBindersSpec.code_binders_distinct code_of_one_continuation = false.
Proof. reflexivity. Qed.

(* A recursive group that binds one name twice. The machine enters the
   first handler of the name. The check of trap actions finds the name
   entered only after it has passed that handler, tests the second, and
   does not see that the first removes a handler that is not installed.
   The check of binders refuses the group. *)
Definition group_of_one_name_twice : expr :=
  Let_cont_recursive
    [(loop, plain (goto leave (Some (Pop handler None))));
     (inner, plain (goto loop None));
     (loop, plain (goto leave None))]
    (goto inner None).

Example group_of_one_name_twice_stack :
  traps_well_formed error group_of_one_name_twice = true.
Proof. reflexivity. Qed.

Example group_of_one_name_twice_binders :
  FlambdaBindersSpec.binders_distinct group_of_one_name_twice = false.
Proof. reflexivity. Qed.

(* A call that gives an extra argument to its exception continuation:
   to a handler of the term, and to the exception continuation of the
   unit or of the code, which takes the exception alone. The stack is in
   order in each. *)
Definition call_in_try_body : expr :=
  try_with
    (Let_cont_non_recursive leave
       (plain (goto 7 (Some (Pop handler None))))
       (call handler)).

Example extra_args_to_handler :
  unit_exn_handlers_well_formed error call_in_try_body = true.
Proof. reflexivity. Qed.

Example extra_args_to_unit_stack :
  traps_well_formed error (call error) = true.
Proof. reflexivity. Qed.

Example extra_args_to_unit :
  unit_exn_handlers_well_formed error (call error) = false.
Proof. reflexivity. Qed.

Example no_extra_args_to_unit :
  unit_exn_handlers_well_formed error (call_with error []) = true.
Proof. reflexivity. Qed.

Example extra_args_to_code_exn :
  code_exn_handlers_well_formed (code_of (call error)) = false.
Proof. reflexivity. Qed.

End Examples.

(* ======================================================================== *)
(* Without the discipline the two machines differ                           *)
(* ======================================================================== *)

Theorem stack_differs : stack_differs_statement.
Proof.
  unfold stack_differs_statement. split; [ reflexivity | ].
  split; [ reflexivity | ]. split.
  - exists 10%nat; eexists. vm_compute. reflexivity.
  - exists 10%nat. vm_compute. reflexivity.
Qed.

(* And without distinct binders they differ, on a unit that follows the
   discipline. *)
Theorem stack_needs_distinct_binders :
  stack_needs_distinct_binders_statement.
Proof.
  unfold stack_needs_distinct_binders_statement.
  split; [ reflexivity | ]. split; [ reflexivity | ]. split.
  - exists 10%nat; eexists. vm_compute. reflexivity.
  - exists 10%nat. vm_compute. reflexivity.
Qed.

(* ======================================================================== *)
(* The check, with what the check of binders gives                          *)
(* ======================================================================== *)

(* [trap_demands] is for a term whose binders are distinct
   (FlambdaBindersSpec.binders_distinct), and the proof uses two things
   of that: the continuations of a recursive [Let_cont] are distinct,
   and the return and exception continuations of code are two names.
   [full_demands] is [trap_demands] with those two tests made where
   they are used, so that what it says of a term it says of each part
   of the term that the machine comes to. The simulation is written
   with it. On a term whose binders are distinct the two are one
   ([full_demands_eq], below). *)
Fixpoint full_demands (ts : trap_stack) (e : expr) : option demands :=
  match e with
  | Let _ _ _ body => full_demands ts body
  | Let_static bindings body =>
    if forallb binding_full_well_formed bindings
    then full_demands ts body
    else None
  | Let_cont_non_recursive k h body =>
    if among k ts then None
    else
      let* ds := full_demands ts body in
      let* outer := leave_scope [k] ds in
      match entry_of k ds with
      | Not_entered => Some outer
      | Entered_with entered =>
        (* [k] is not in scope in its handler: what the handler asks is
           asked of the continuations around the [Let_cont]. *)
        if among k entered then None
        else
          let* ds' := handler_full_demands entered h in
          Some (outer ++ ds')
      | Entered_with_two_stacks => None
      end
  | Let_cont_recursive handlers body =>
    let ks := map fst handlers in
    if (existsb (fun k => among k ts) ks || negb (distinct ks))%bool
    then None
    else
      let* ds := full_demands ts body in
      (* Each pass checks the handlers that are known to be entered and
         have not been checked; as many passes as handlers reach them
         all, which the test after the passes confirms. *)
      let pass (state : group_state) : option group_state :=
        fold_left
          (fun state '(k, h) =>
             let* '(checked, ds) := state in
             if among k checked then Some (checked, ds)
             else
               match entry_of k ds with
               | Not_entered => Some (checked, ds)
               | Entered_with entered =>
                 let* ds' := handler_full_demands entered h in
                 Some (k :: checked, ds ++ ds')
               | Entered_with_two_stacks => None
               end)
          (Some state) handlers
      in
      let* '(checked, ds) := iterate (length handlers) pass ([], ds) in
      if forallb
           (fun k =>
              match entry_of k ds with
              | Not_entered => true
              | Entered_with _ => among k checked
              | Entered_with_two_stacks => false
              end)
           ks
      then leave_scope ks ds
      else None
  | Apply ap => apply_demands ts ap
  | Apply_cont ac => apply_cont_demands ts ac
  | Switch _ arms => arms_demands ts arms
  | Invalid => Some []
  end
with handler_full_demands (ts : trap_stack) (h : continuation_handler)
    : option demands :=
  match h with
  | Handler _ _ handler => full_demands ts handler
  end
with binding_full_well_formed (binding : static_binding) : bool :=
  match binding with
  | Bind_code _ c => code_full_well_formed c
  | Bind_deleted_code _ | Bind_symbol _ _ => true
  end
with code_full_well_formed (c : code) : bool :=
  match c with
  | Code _ _ ret exn _ body =>
    negb (Pos.eqb ret exn)
    && match full_demands [exn] body with
       | Some ds => code_demands_met ret exn ds
       | None => false
       end
  end.

Definition full_well_formed (unit_exn : continuation) (e : expr) : bool :=
  match full_demands [unit_exn] e with
  | Some ds =>
    forallb
      (fun d =>
         trap_stack_eqb (snd d)
           (if Pos.eqb (fst d) unit_exn then [] else [unit_exn]))
      ds
  | None => false
  end.

(* ======================================================================== *)
(* What the check of trap actions says of a term                            *)
(* ======================================================================== *)

Lemma trap_stack_eqb_eq (ts ts' : trap_stack) :
  trap_stack_eqb ts ts' = true <-> ts = ts'.
Proof.
  revert ts'. induction ts as [ | k ts IH]; intros [ | k' ts']; simpl.
  - split; reflexivity.
  - split; discriminate.
  - split; discriminate.
  - rewrite andb_true_iff, Pos.eqb_eq, IH.
    split; [ intros [-> ->]; reflexivity | intros [= -> ->]; auto ].
Qed.

Lemma among_false (k : positive) (ks : list positive) :
  among k ks = false <-> ~ In k ks.
Proof.
  rewrite <- (among_In k ks). destruct (among k ks); split; intros H;
    try reflexivity; try discriminate; try (intros [=]); contradiction.
Qed.

(* The demands on [k] are in the list that [entry_of] looks at. *)
Lemma demand_in_entries (k : continuation) (ds : demands)
    (d : continuation * trap_stack) :
  In d ds -> fst d = k ->
  In (snd d) (map snd (filter (fun d => Pos.eqb (fst d) k) ds)).
Proof.
  intros Hin Hk. apply in_map, filter_In. split; [ exact Hin | ].
  apply Pos.eqb_eq. exact Hk.
Qed.

Lemma entry_of_entered (k : continuation) (ds : demands) (ts : trap_stack) :
  entry_of k ds = Entered_with ts ->
  ∀ d : continuation * trap_stack, In d ds -> fst d = k -> snd d = ts.
Proof.
  unfold entry_of. intros Hentry d Hin Hk.
  pose proof (demand_in_entries k ds d Hin Hk) as Hmem.
  destruct (map snd (filter (fun d => Pos.eqb (fst d) k) ds))
    as [ | ts0 others]; [ contradiction | ].
  destruct (forallb (trap_stack_eqb ts0) others) eqn:Hall; [ | discriminate ].
  injection Hentry as <-.
  destruct Hmem as [<- | Hmem]; [ reflexivity | ].
  rewrite forallb_forall in Hall. symmetry.
  apply trap_stack_eqb_eq, Hall, Hmem.
Qed.

Lemma entry_of_not_entered (k : continuation) (ds : demands) :
  entry_of k ds = Not_entered ->
  ∀ d : continuation * trap_stack, In d ds -> fst d <> k.
Proof.
  unfold entry_of. intros Hentry d Hin Hk.
  pose proof (demand_in_entries k ds d Hin Hk) as Hmem.
  destruct (map snd (filter (fun d => Pos.eqb (fst d) k) ds))
    as [ | ts0 others]; [ contradiction | ].
  destruct (forallb (trap_stack_eqb ts0) others); discriminate.
Qed.

(* A demand that leaves the scope of [ks] is not on one of them, and its
   stack holds none of them. *)
Lemma leave_scope_in (ks : list continuation) (ds outer : demands) :
  leave_scope ks ds = Some outer ->
  ∀ d : continuation * trap_stack,
    In d ds -> among (fst d) ks = false ->
    In d outer ∧ ∀ k : continuation, In k ks -> ~ In k (snd d).
Proof.
  unfold leave_scope. intros Hleave d Hin Hk.
  set (outer0 := filter (fun d => negb (among (fst d) ks)) ds) in *.
  destruct
    (forallb (fun d => forallb (fun k => negb (among k (snd d))) ks) outer0)
    eqn:Hall; [ | discriminate ].
  injection Hleave as <-.
  assert (Hin' : In d outer0).
  { apply filter_In. split; [ exact Hin | ]. apply negb_true_iff. exact Hk. }
  split; [ exact Hin' | ].
  intros k Hks. rewrite forallb_forall in Hall. specialize (Hall d Hin').
  rewrite forallb_forall in Hall. apply among_false, negb_true_iff, Hall, Hks.
Qed.

Lemma arms_demands_find (ts : trap_stack) (arms : list (Z * apply_cont)) :
  ∀ (ds : demands) (i : Z) (ac : apply_cont),
    arms_demands ts arms = Some ds ->
    find_arm i arms = Some ac ->
    ∃ ds_ac : demands,
      apply_cont_demands ts ac = Some ds_ac ∧ incl ds_ac ds.
Proof.
  induction arms as [ | [j ac'] arms IH]; intros ds i ac; simpl;
    [ discriminate | ].
  destruct (apply_cont_demands ts ac') as [ds1 | ] eqn:H1; [ | discriminate ].
  destruct (arms_demands ts arms) as [ds2 | ] eqn:H2; [ | discriminate ].
  intros [= <-]. destruct (Z.eqb i j).
  - intros [= <-]. exists ds1.
    split; [ exact H1 | apply incl_appl, incl_refl ].
  - intros Hfind.
    destruct (IH ds2 i ac eq_refl Hfind) as (ds_ac & Hac & Hincl).
    exists ds_ac. split; [ exact Hac | apply incl_appr, Hincl ].
Qed.

(* The handlers of a recursive [Let_cont], as [full_demands] goes through
   them: one handler, and one pass over them all. *)
Definition group_step (state : option group_state)
    (kh : continuation * continuation_handler) : option group_state :=
  let '(k, h) := kh in
  let* '(checked, ds) := state in
  if among k checked then Some (checked, ds)
  else
    match entry_of k ds with
    | Not_entered => Some (checked, ds)
    | Entered_with entered =>
      let* ds' := handler_full_demands entered h in
      Some (k :: checked, ds ++ ds')
    | Entered_with_two_stacks => None
    end.

Definition group_pass (handlers : list (continuation * continuation_handler))
    (state : group_state) : option group_state :=
  fold_left group_step (Some state) handlers.

Lemma full_demands_recursive (ts : trap_stack)
    (handlers : list (continuation * continuation_handler)) (body : expr) :
  full_demands ts (Let_cont_recursive handlers body)
  = let ks := map fst handlers in
    if (existsb (fun k => among k ts) ks || negb (distinct ks))%bool
    then None
    else
      let* ds := full_demands ts body in
      let* '(checked, ds) :=
        iterate (length handlers) (group_pass handlers) ([], ds)
      in
      if forallb
           (fun k =>
              match entry_of k ds with
              | Not_entered => true
              | Entered_with _ => among k checked
              | Entered_with_two_stacks => false
              end)
           ks
      then leave_scope ks ds
      else None.
Proof. reflexivity. Qed.

Lemma entry_of_entered_in (k : continuation) (ds : demands)
    (ts : trap_stack) :
  entry_of k ds = Entered_with ts -> In (k, ts) ds.
Proof.
  unfold entry_of. intros Hentry.
  destruct (filter (fun d => Pos.eqb (fst d) k) ds) as [ | [k0 ts0] others]
    eqn:Hfilter; simpl in Hentry; [ discriminate | ].
  destruct (forallb (trap_stack_eqb ts0) (map snd others)); [ | discriminate ].
  injection Hentry as <-.
  assert (Hin : In (k0, ts0) (filter (fun d => Pos.eqb (fst d) k) ds))
    by (rewrite Hfilter; left; reflexivity).
  apply filter_In in Hin as [Hin Hk]. apply Pos.eqb_eq in Hk. simpl in Hk.
  subst k0. exact Hin.
Qed.

Lemma lookup_of_distinct {A : Type} (k : positive) (a : A)
    (l : list (positive * A)) :
  distinct (map fst l) = true -> In (k, a) l -> lookup k l = Some a.
Proof.
  induction l as [ | [k' a'] l IH]; simpl; [ contradiction | ].
  intros Hd Hin. apply andb_true_iff in Hd as [Hfresh Hd].
  destruct Hin as [[= -> ->] | Hin].
  - rewrite Pos.eqb_refl. reflexivity.
  - destruct (Pos.eqb_spec k k') as [-> | Hne]; [ | exact (IH Hd Hin) ].
    apply negb_true_iff, among_false in Hfresh. exfalso. apply Hfresh.
    apply (in_map fst) in Hin. exact Hin.
Qed.

(* What holds of the handlers checked so far: each was checked under a
   stack that a demand has for it, and what it demands is among the
   demands. *)
Definition group_good (handlers : list (continuation * continuation_handler))
    (state : group_state) : Prop :=
  ∀ k : continuation,
    In k (fst state) ->
    ∃ (h : continuation_handler) (ts : trap_stack) (ds_h : demands),
      In (k, h) handlers
      ∧ In (k, ts) (snd state)
      ∧ handler_full_demands ts h = Some ds_h
      ∧ incl ds_h (snd state).

Lemma group_fold_none (hs : list (continuation * continuation_handler)) :
  fold_left group_step None hs = None.
Proof.
  induction hs as [ | [k h] hs IH]; simpl; [ reflexivity | exact IH ].
Qed.

Lemma group_fold_good (handlers : list (continuation * continuation_handler))
    (hs : list (continuation * continuation_handler)) :
  incl hs handlers ->
  ∀ state state' : group_state,
    group_good handlers state ->
    fold_left group_step (Some state) hs = Some state' ->
    group_good handlers state' ∧ incl (snd state) (snd state').
Proof.
  induction hs as [ | [k h] hs IH]; intros Hincl [checked ds] state' Hgood;
    simpl.
  - intros [= <-]. split; [ exact Hgood | apply incl_refl ].
  - assert (Hhs : incl hs handlers)
      by (intros kh Hin; apply Hincl; right; exact Hin).
    destruct (among k checked); [ exact (IH Hhs _ _ Hgood) | ].
    destruct (entry_of k ds) as [ | entered | ] eqn:Hentry.
    + exact (IH Hhs _ _ Hgood).
    + destruct (handler_full_demands entered h) as [ds' | ] eqn:Hh;
        [ | rewrite group_fold_none; discriminate ].
      intros Hfold.
      assert (Hgood' : group_good handlers (k :: checked, ds ++ ds')).
      { intros k' [<- | Hin]; simpl.
        - exists h, entered, ds'. split; [ apply Hincl; left; reflexivity | ].
          split; [ apply in_or_app; left;
                   exact (entry_of_entered_in k ds entered Hentry) | ].
          split; [ exact Hh | apply incl_appr, incl_refl ].
        - destruct (Hgood k' Hin) as (h' & ts & ds_h & Hkh & Hd & Hds & Hsub).
          exists h', ts, ds_h. split; [ exact Hkh | ].
          split; [ apply in_or_app; left; exact Hd | ].
          split; [ exact Hds | apply incl_appl, Hsub ]. }
      destruct (IH Hhs _ _ Hgood' Hfold) as [Hgood'' Hsub].
      split; [ exact Hgood'' | ].
      intros d Hd. apply Hsub. simpl. apply in_or_app. left. exact Hd.
    + rewrite group_fold_none. discriminate.
Qed.

Lemma group_iterate_good
    (handlers : list (continuation * continuation_handler)) (n : nat) :
  ∀ state state' : group_state,
    group_good handlers state ->
    iterate n (group_pass handlers) state = Some state' ->
    group_good handlers state' ∧ incl (snd state) (snd state').
Proof.
  induction n as [ | n IH]; intros state state' Hgood; simpl.
  - intros [= <-]. split; [ exact Hgood | apply incl_refl ].
  - destruct (group_pass handlers state) as [state1 | ] eqn:Hpass;
      [ | discriminate ].
    intros Hiter.
    destruct (group_fold_good handlers handlers (incl_refl _) state state1
                Hgood Hpass) as [Hgood1 Hsub1].
    destruct (IH state1 state' Hgood1 Hiter) as [Hgood' Hsub'].
    split; [ exact Hgood' | exact (incl_tran Hsub1 Hsub') ].
Qed.

(* What the check of a recursive [Let_cont] says of the demands it ends
   with: those of the body are among them; a demand on a handler of the
   group is for a stack under which that handler was checked, and what
   the handler demands is among them; and the others leave the scope. *)
Lemma full_demands_recursive_inv (ts : trap_stack)
    (handlers : list (continuation * continuation_handler)) (body : expr)
    (ds : demands) :
  full_demands ts (Let_cont_recursive handlers body) = Some ds ->
  (∀ k : continuation, In k (map fst handlers) -> ~ In k ts)
  ∧ ∃ ds_body ds_group : demands,
      full_demands ts body = Some ds_body
      ∧ incl ds_body ds_group
      ∧ leave_scope (map fst handlers) ds_group = Some ds
      ∧ ∀ d : continuation * trap_stack,
          In d ds_group -> In (fst d) (map fst handlers) ->
          ∃ (h : continuation_handler) (ds_h : demands),
            lookup (fst d) handlers = Some h
            ∧ handler_full_demands (snd d) h = Some ds_h
            ∧ incl ds_h ds_group.
Proof.
  rewrite full_demands_recursive. cbv zeta.
  destruct (existsb (fun k => among k ts) (map fst handlers)) eqn:Hfresh;
    [ discriminate | ].
  destruct (distinct (map fst handlers)) eqn:Hdistinct; [ | discriminate ].
  simpl.
  destruct (full_demands ts body) as [ds_body | ] eqn:Hbody;
    [ | discriminate ].
  destruct (iterate (length handlers) (group_pass handlers) ([], ds_body))
    as [[checked ds_group] | ] eqn:Hiter; [ | discriminate ].
  destruct
    (forallb
       (fun k =>
          match entry_of k ds_group with
          | Not_entered => true
          | Entered_with _ => among k checked
          | Entered_with_two_stacks => false
          end)
       (map fst handlers)) eqn:Hall; [ | discriminate ].
  intros Hleave. split.
  - intros k Hk. apply among_false.
    destruct (among k ts) eqn:Hamong; [ | reflexivity ].
    assert (Hex : existsb (fun k => among k ts) (map fst handlers) = true)
      by (apply existsb_exists; exists k; split; assumption).
    rewrite Hex in Hfresh. discriminate.
  - assert (Hgood0 : group_good handlers ([], ds_body))
      by (intros k []).
    destruct (group_iterate_good handlers (length handlers) _ _ Hgood0 Hiter)
      as [Hgood Hsub].
    exists ds_body, ds_group. split; [ reflexivity | ].
    split; [ exact Hsub | ]. split; [ exact Hleave | ].
    intros d Hd Hk. rewrite forallb_forall in Hall.
    specialize (Hall (fst d) Hk).
    destruct (entry_of (fst d) ds_group) as [ | entered | ] eqn:Hentry.
    + exfalso.
      exact (entry_of_not_entered (fst d) ds_group Hentry d Hd eq_refl).
    + apply among_In in Hall.
      destruct (Hgood (fst d) Hall) as (h & ts' & ds_h & Hkh & Hd' & Hh & Hin).
      simpl in Hd', Hin.
      pose proof (entry_of_entered (fst d) ds_group entered Hentry d Hd
                    eq_refl) as Hts.
      pose proof (entry_of_entered (fst d) ds_group entered Hentry
                    (fst d, ts') Hd' eq_refl) as Hts'.
      simpl in Hts'. subst ts'. rewrite Hts.
      exists h, ds_h. split; [ | split; [ exact Hh | exact Hin ] ].
      exact (lookup_of_distinct (fst d) h handlers Hdistinct Hkh).
    + discriminate.
Qed.

(* The same for a non-recursive [Let_cont]. *)
Lemma full_demands_non_recursive_inv (ts : trap_stack) (k : continuation)
    (h : continuation_handler) (body : expr) (ds : demands) :
  full_demands ts (Let_cont_non_recursive k h body) = Some ds ->
  ~ In k ts
  ∧ ∃ ds_body outer : demands,
      full_demands ts body = Some ds_body
      ∧ leave_scope [k] ds_body = Some outer
      ∧ incl outer ds
      ∧ ∀ d : continuation * trap_stack,
          In d ds_body -> fst d = k ->
          ~ In k (snd d)
          ∧ ∃ ds_h : demands,
              handler_full_demands (snd d) h = Some ds_h ∧ incl ds_h ds.
Proof.
  simpl. destruct (among k ts) eqn:Hfresh; [ discriminate | ].
  destruct (full_demands ts body) as [ds_body | ] eqn:Hbody;
    [ | discriminate ].
  destruct (leave_scope [k] ds_body) as [outer | ] eqn:Hleave;
    [ | discriminate ].
  intros Hds. split; [ apply among_false; exact Hfresh | ].
  exists ds_body, outer. split; [ reflexivity | ]. split; [ exact Hleave | ].
  destruct (entry_of k ds_body) as [ | entered | ] eqn:Hentry.
  - injection Hds as <-. split; [ apply incl_refl | ].
    intros d Hd Hk. exfalso.
    exact (entry_of_not_entered k ds_body Hentry d Hd Hk).
  - destruct (among k entered) eqn:Hk_entered; [ discriminate | ].
    destruct (handler_full_demands entered h) as [ds_h | ] eqn:Hh;
      [ | discriminate ].
    injection Hds as <-. split; [ apply incl_appl, incl_refl | ].
    intros d Hd Hk.
    rewrite (entry_of_entered k ds_body entered Hentry d Hd Hk).
    split; [ apply among_false; exact Hk_entered | ].
    exists ds_h. split; [ exact Hh | apply incl_appr, incl_refl ].
  - discriminate.
Qed.

(* ======================================================================== *)
(* On a term whose binders are distinct, the two checks are one             *)
(* ======================================================================== *)

Definition lax_group_step (state : option group_state)
    (kh : continuation * continuation_handler) : option group_state :=
  let '(k, h) := kh in
  let* '(checked, ds) := state in
  if among k checked then Some (checked, ds)
  else
    match entry_of k ds with
    | Not_entered => Some (checked, ds)
    | Entered_with entered =>
      let* ds' := handler_trap_demands entered h in
      Some (k :: checked, ds ++ ds')
    | Entered_with_two_stacks => None
    end.

Definition lax_group_pass
    (handlers : list (continuation * continuation_handler))
    (state : group_state) : option group_state :=
  fold_left lax_group_step (Some state) handlers.

Lemma trap_demands_recursive (ts : trap_stack)
    (handlers : list (continuation * continuation_handler)) (body : expr) :
  trap_demands ts (Let_cont_recursive handlers body)
  = let ks := map fst handlers in
    if existsb (fun k => among k ts) ks then None
    else
      let* ds := trap_demands ts body in
      let* '(checked, ds) :=
        iterate (length handlers) (lax_group_pass handlers) ([], ds)
      in
      if forallb
           (fun k =>
              match entry_of k ds with
              | Not_entered => true
              | Entered_with _ => among k checked
              | Entered_with_two_stacks => false
              end)
           ks
      then leave_scope ks ds
      else None.
Proof. reflexivity. Qed.

Lemma iterate_same {A : Type} (f g : A -> option A) (n : nat) :
  (∀ x : A, f x = g x) -> ∀ x : A, iterate n f x = iterate n g x.
Proof.
  intros Hfg. induction n as [ | n IH]; intros x; simpl; [ reflexivity | ].
  rewrite (Hfg x). destruct (g x) as [y | ]; [ apply IH | reflexivity ].
Qed.

Fixpoint full_demands_eq (e : expr) :
  binders_distinct e = true ->
  ∀ ts : trap_stack, full_demands ts e = trap_demands ts e
with handler_full_demands_eq (h : continuation_handler) :
  handler_binders_distinct h = true ->
  ∀ ts : trap_stack, handler_full_demands ts h = handler_trap_demands ts h
with binding_full_well_formed_eq (binding : static_binding) :
  binding_binders_distinct binding = true ->
  binding_full_well_formed binding = binding_traps_well_formed binding
with code_full_well_formed_eq (c : code) :
  code_binders_distinct c = true ->
  code_full_well_formed c = code_traps_well_formed c.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros Hd ts; try reflexivity.
    + exact (full_demands_eq body Hd ts).
    + simpl in Hd |- *. apply andb_true_iff in Hd as [Hbindings Hbody].
      rewrite (full_demands_eq body Hbody ts).
      assert (Heq : forallb binding_full_well_formed bindings
                    = forallb binding_traps_well_formed bindings).
      { induction bindings as [ | binding bindings IH]; simpl;
          [ reflexivity | ].
        simpl in Hbindings. apply andb_true_iff in Hbindings as [Hb Hbs].
        rewrite (binding_full_well_formed_eq binding Hb), (IH Hbs).
        reflexivity. }
      rewrite Heq. reflexivity.
    + simpl in Hd |- *. apply andb_true_iff in Hd as [Hh Hbody].
      rewrite (full_demands_eq body Hbody ts).
      destruct (among k ts); [ reflexivity | ].
      destruct (trap_demands ts body) as [ds | ]; [ | reflexivity ].
      destruct (leave_scope [k] ds) as [outer | ]; [ | reflexivity ].
      destruct (entry_of k ds) as [ | entered | ]; try reflexivity.
      destruct (among k entered); [ reflexivity | ].
      rewrite (handler_full_demands_eq h Hh entered). reflexivity.
    + rewrite full_demands_recursive, trap_demands_recursive. cbv zeta.
      simpl in Hd.
      apply andb_true_iff in Hd as [Hd Hbody].
      apply andb_true_iff in Hd as [Hd Hhandlers].
      apply andb_true_iff in Hd as [_ Hdistinct].
      rewrite Hdistinct, (full_demands_eq body Hbody ts). simpl negb.
      rewrite orb_false_r.
      assert (Hstep : ∀ state : option group_state,
                fold_left group_step state handlers
                = fold_left lax_group_step state handlers).
      { clear Hdistinct.
        induction handlers as [ | [k h] handlers IH]; intros state; simpl;
          [ reflexivity | ].
        simpl in Hhandlers. apply andb_true_iff in Hhandlers as [Hh Hhs].
        rewrite (IH Hhs). f_equal.
        destruct state as [[checked ds] | ]; [ | reflexivity ].
        destruct (among k checked); [ reflexivity | ].
        destruct (entry_of k ds) as [ | entered | ]; try reflexivity.
        rewrite (handler_full_demands_eq h Hh entered). reflexivity. }
      destruct (existsb (fun k => among k ts) (map fst handlers));
        [ reflexivity | ].
      destruct (trap_demands ts body) as [ds | ]; [ | reflexivity ].
      rewrite (iterate_same (group_pass handlers) (lax_group_pass handlers)
                 (length handlers) (fun state => Hstep (Some state))
                 ([], ds)).
      reflexivity.
  - destruct h as [params metadata handler]. simpl. intros Hd ts.
    apply andb_true_iff in Hd as [_ Hd].
    exact (full_demands_eq handler Hd ts).
  - destruct binding as [id c | id | s const]; simpl; intros Hd;
      try reflexivity.
    exact (code_full_well_formed_eq c Hd).
  - destruct c as [params my_closure ret exn metadata body]. simpl.
    intros Hd.
    apply andb_true_iff in Hd as [Hd Hbody].
    apply andb_true_iff in Hd as [_ Hne].
    rewrite Hne, (full_demands_eq body Hbody [exn]). reflexivity.
Qed.

Lemma full_well_formed_eq (unit_exn : continuation) (e : expr) :
  binders_distinct e = true ->
  full_well_formed unit_exn e = traps_well_formed unit_exn e.
Proof.
  intros Hd. unfold full_well_formed, traps_well_formed.
  rewrite (full_demands_eq e Hd [unit_exn]). reflexivity.
Qed.

(* ======================================================================== *)
(* The invariant                                                            *)
(* ======================================================================== *)

(* The proof is a simulation, one step for one step. The two machines
   run the same expression under the same variables and static
   environment; their continuations differ where a call was made, since
   the body of code runs with its return and exception continuations
   bound to those of the caller on the one machine, and bound to nothing
   on the other. *)

(* The handlers installed since the call in progress was made, or since
   the unit started. *)
Fixpoint handlers_of (st : stack) : list cont_value :=
  match st with
  | Trap c :: st => c :: handlers_of st
  | [] | Call _ _ _ _ :: _ | Call_then _ _ _ _ _ _ _ :: _ => []
  end.

(* What is running: the unit, or the body of code. *)
Inductive frame_kind : Type :=
  | Unit_frame (unit_exn : continuation)
  | Code_frame (ret exn : continuation).

(* The continuations with which the machine with the stack starts what is
   running. *)
Definition bottom_conts (f : frame_kind) : cont_env :=
  match f with
  | Unit_frame _ => []
  | Code_frame ret exn => [(ret, Outer ret); (exn, Outer exn)]
  end.

(* What the check asks of a demand on a continuation that what is running
   does not bind: the last test of [full_well_formed], and
   [code_demands_met]. *)
Definition bottom_ok (f : frame_kind) (k : continuation) (ts : trap_stack)
    : Prop :=
  match f with
  | Unit_frame unit_exn => ts = if Pos.eqb k unit_exn then [] else [unit_exn]
  | Code_frame ret exn =>
    ret <> exn ∧ if Pos.eqb k exn then ts = [] else k = ret ∧ ts = [exn]
  end.

(* Going to [k] when the handlers installed are those that [ts] names is
   in order, for [n] steps: if [k] stands for a handler, the names of
   [ts] stand for the same in its environment as here, it passes the
   check under [ts], and what it demands is in order; if [k] stands for
   nothing, the demand is one that the check allows. *)
Fixpoint target_ok (n : nat) (f : frame_kind) (conts : cont_env)
    (k : continuation) (ts : trap_stack) : Prop :=
  match n with
  | O => True
  | S n =>
    match resolve k conts with
    | Closure h vars def_conts recursive =>
      let conts_h := rec_closures recursive vars def_conts ++ def_conts in
      (∀ k' : continuation,
        In k' ts -> resolve k' conts_h = resolve k' conts)
      ∧ ∃ ds : demands,
          handler_full_demands ts h = Some ds
          ∧ ∀ d : continuation * trap_stack,
              In d ds -> target_ok n f conts_h (fst d) (snd d)
    | Outer k0 => k0 = k ∧ bottom_ok f k ts
    | Extra_args _ _ | Over_application _ _ _ _ _ => False
    end
  end.

Lemma target_ok_S (n : nat) (f : frame_kind) (conts : cont_env)
    (k : continuation) (ts : trap_stack) :
  target_ok (S n) f conts k ts
  = match resolve k conts with
    | Closure h vars def_conts recursive =>
      let conts_h := rec_closures recursive vars def_conts ++ def_conts in
      (∀ k' : continuation,
        In k' ts -> resolve k' conts_h = resolve k' conts)
      ∧ ∃ ds : demands,
          handler_full_demands ts h = Some ds
          ∧ ∀ d : continuation * trap_stack,
              In d ds -> target_ok n f conts_h (fst d) (snd d)
    | Outer k0 => k0 = k ∧ bottom_ok f k ts
    | Extra_args _ _ | Over_application _ _ _ _ _ => False
    end.
Proof. reflexivity. Qed.

#[local] Arguments target_ok : simpl never.

Lemma target_ok_weaken (n : nat) :
  ∀ (f : frame_kind) (conts : cont_env) (k : continuation)
    (ts : trap_stack),
    target_ok (S n) f conts k ts -> target_ok n f conts k ts.
Proof.
  induction n as [ | n IH]; intros f conts k ts H; [ exact I | ].
  rewrite target_ok_S in H. rewrite target_ok_S.
  destruct (resolve k conts)
    as [h vars def_conts recursive | k0 | c extra
       | remaining remaining_arity return_arity return_to exn_to];
    [ | exact H | exact H | exact H ].
  destruct H as (Hagree & ds & Hh & Hds). split; [ exact Hagree | ].
  exists ds. split; [ exact Hh | ]. intros d Hd. apply IH. exact (Hds d Hd).
Qed.

Lemma target_ok_le (n m : nat) (f : frame_kind) (conts : cont_env)
    (k : continuation) (ts : trap_stack) :
  m <= n -> target_ok n f conts k ts -> target_ok m f conts k ts.
Proof.
  intros Hle. induction Hle as [ | n Hle IH]; [ exact (fun H => H) | ].
  intros H. apply IH, target_ok_weaken. exact H.
Qed.

(* It depends on the continuations only through what [k] and the names
   of [ts] stand for. *)
Lemma target_ok_same_resolve (n : nat) (f : frame_kind)
    (conts conts' : cont_env) (k : continuation) (ts : trap_stack) :
  resolve k conts' = resolve k conts ->
  (∀ k' : continuation, In k' ts -> resolve k' conts' = resolve k' conts) ->
  target_ok n f conts k ts -> target_ok n f conts' k ts.
Proof.
  destruct n as [ | n]; [ intros _ _ _; exact I | ].
  rewrite !target_ok_S. intros Hk Hts. rewrite Hk.
  destruct (resolve k conts)
    as [h vars def_conts recursive | k0 | c extra
       | remaining remaining_arity return_arity return_to exn_to];
    [ | exact (fun H => H) | exact (fun H => H) | exact (fun H => H) ].
  intros (Hagree & ds & Hh & Hds). split.
  - intros k' Hin. rewrite (Hts k' Hin). exact (Hagree k' Hin).
  - exists ds. split; assumption.
Qed.

(* The continuations of the two machines, in what is running: they start
   as [bottom] on the machine that goes by names and as [bottom_conts] on
   the other, and have the same handlers bound since, each closed over
   continuations that are so related. *)
Inductive env_rel (f : frame_kind) (bottom : cont_env)
    : cont_env -> cont_env -> Prop :=
  | Env_bottom : env_rel f bottom bottom (bottom_conts f)
  | Env_cons (k : continuation) (h : continuation_handler) (vars : env)
      (def_n def_s : cont_env)
      (recursive : list (continuation * continuation_handler))
      (conts_n conts_s : cont_env) :
      env_rel f bottom def_n def_s ->
      env_rel f bottom conts_n conts_s ->
      env_rel f bottom ((k, Closure h vars def_n recursive) :: conts_n)
        ((k, Closure h vars def_s recursive) :: conts_s).

Lemma env_rel_resolve (f : frame_kind) (bottom conts_n conts_s : cont_env)
    (k : continuation) :
  env_rel f bottom conts_n conts_s ->
  (∃ (h : continuation_handler) (vars : env) (def_n def_s : cont_env)
     (recursive : list (continuation * continuation_handler)),
    resolve k conts_n = Closure h vars def_n recursive
    ∧ resolve k conts_s = Closure h vars def_s recursive
    ∧ env_rel f bottom def_n def_s)
  ∨ (resolve k conts_n = resolve k bottom
     ∧ resolve k conts_s = resolve k (bottom_conts f)).
Proof.
  intros Henv.
  induction Henv
    as [ | k0 h vars def_n def_s recursive conts_n conts_s Hdef _ Hconts IH].
  - right. split; reflexivity.
  - destruct (Pos.eqb_spec k k0) as [-> | Hne].
    + left. exists h, vars, def_n, def_s, recursive.
      rewrite !resolve_head. split; [ reflexivity | ].
      split; [ reflexivity | exact Hdef ].
    + rewrite !(resolve_other k k0 _ _ Hne). exact IH.
Qed.

Lemma env_rel_rec (f : frame_kind) (bottom def_n def_s : cont_env)
    (recursive : list (continuation * continuation_handler)) (vars : env) :
  env_rel f bottom def_n def_s ->
  env_rel f bottom (rec_closures recursive vars def_n ++ def_n)
    (rec_closures recursive vars def_s ++ def_s).
Proof.
  intros Hdef. unfold rec_closures.
  generalize recursive at 1 3. intros all.
  induction recursive as [ | [k h] recursive IH]; simpl;
    [ exact Hdef | ].
  apply Env_cons; [ exact Hdef | exact IH ].
Qed.

(* A call in progress is a call of code, or the first call of an
   over-application, which keeps the arguments that remain. The two
   differ in the entry that the stack has, and in what the return
   continuation of the code is bound to on the machine that goes by
   names: what the return continuation of the call stands for, or what
   is left of the over-application. *)
Inductive call_frame : Type :=
  | Plain_call
  | First_call (remaining : list value)
      (remaining_arity return_arity : arity).

Definition call_entry (frame : call_frame) (ret exn : continuation)
    (return_to : cont_value) (extra : list value) : stack_entry :=
  match frame with
  | Plain_call => Call ret exn return_to extra
  | First_call remaining remaining_arity return_arity =>
    Call_then ret exn remaining remaining_arity return_arity return_to extra
  end.

Definition call_return (frame : call_frame) (return_to exn_to : cont_value)
    : cont_value :=
  match frame with
  | Plain_call => return_to
  | First_call remaining remaining_arity return_arity =>
    Over_application remaining remaining_arity return_arity return_to exn_to
  end.

Lemma handlers_of_call_entry (frame : call_frame) (ret exn : continuation)
    (return_to : cont_value) (extra : list value) (rest : stack) :
  handlers_of (call_entry frame ret exn return_to extra :: rest) = [].
Proof. destruct frame; reflexivity. Qed.

(* Leaving the body by its exception continuation, the handler it started
   with removed, is a raise in the caller with the extra arguments. *)
Lemma unwind_call_exn (frame : call_frame) (ret exn : continuation)
    (return_to : cont_value) (extra : list value) (rest : stack)
    (vs : list value) (statics : static_env) (hp : heap) (rd : read_record) :
  unwind (call_entry frame ret exn return_to extra :: rest) (Leaving_by exn)
    vs statics hp rd
  = unwind rest Raising (vs ++ extra) statics hp rd.
Proof. destruct frame; simpl; rewrite Pos.eqb_refl; reflexivity. Qed.

(* The stack, and how the machine that goes by names started what is
   running. Below a call are the continuations of the caller, on the two
   machines: the return continuation of the call, and its exception
   continuation, which is the handler on top of the stack there, are in
   order under the handlers that were installed at the call. *)
Inductive stack_sim (n : nat) : frame_kind -> stack -> cont_env -> Prop :=
  | Sim_unit (unit_exn : continuation) :
      stack_sim n (Unit_frame unit_exn) [] []
  | Sim_trap (f : frame_kind) (c : cont_value) (st : stack)
      (bottom : cont_env) :
      stack_sim n f st bottom -> stack_sim n f (Trap c :: st) bottom
  | Sim_call (frame : call_frame) (ret exn : continuation)
      (extra : list value) (rest : stack) (f : frame_kind)
      (bottom conts_n conts_s : cont_env) (k_ret k_exn : continuation)
      (ts : trap_stack) :
      target_ok n f conts_s k_ret (k_exn :: ts) ->
      target_ok n f conts_s k_exn ts ->
      map (fun k => resolve k conts_s) (k_exn :: ts) = handlers_of rest ->
      env_rel f bottom conts_n conts_s ->
      stack_sim n f rest bottom ->
      stack_sim n (Code_frame ret exn)
        (call_entry frame ret exn (resolve k_ret conts_s) extra :: rest)
        [(ret,
          call_return frame (resolve k_ret conts_n)
            (with_extra_args (resolve k_exn conts_n) extra));
         (exn, with_extra_args (resolve k_exn conts_n) extra)].

Lemma stack_sim_weaken (n : nat) (f : frame_kind) (st : stack)
    (bottom : cont_env) :
  stack_sim (S n) f st bottom -> stack_sim n f st bottom.
Proof.
  intros Hsim.
  induction Hsim
    as [ unit_exn | f c st bottom _ IH
       | frame ret exn extra rest f bottom conts_n conts_s k_ret k_exn ts
           Hret Hexn Hcorr Henv _ IH ].
  - apply Sim_unit.
  - apply Sim_trap. exact IH.
  - apply (Sim_call n frame ret exn extra rest f bottom conts_n conts_s
             k_ret k_exn ts);
      try assumption; apply target_ok_weaken; assumption.
Qed.

(* All the code that is bound passes the check. *)
Definition code_ok (statics : static_env) : Prop :=
  ∀ (id : code_id) (c : code),
    lookup id (static_code statics) = Some c ->
    code_full_well_formed c = true.

(* A state of the machine that goes by names, and one of the machine with
   the stack, that are in step, with [n] steps to go. *)
Definition inv (n : nat) (s_n s_s : state) (st : stack) : Prop :=
  state_expr s_n = state_expr s_s
  ∧ state_vars s_n = state_vars s_s
  ∧ state_statics s_n = state_statics s_s
  ∧ state_heap s_n = state_heap s_s
  ∧ state_reads s_n = state_reads s_s
  ∧ ∃ (f : frame_kind) (bottom : cont_env) (ts : trap_stack) (ds : demands),
      env_rel f bottom (state_conts s_n) (state_conts s_s)
      ∧ full_demands ts (state_expr s_s) = Some ds
      ∧ map (fun k => resolve k (state_conts s_s)) ts = handlers_of st
      ∧ (∀ d : continuation * trap_stack,
          In d ds -> target_ok n f (state_conts s_s) (fst d) (snd d))
      ∧ stack_sim n f st bottom
      ∧ code_ok (state_statics s_s).

Definition related (n : nat) (r : step_result) (r' : stack_result) : Prop :=
  match r, r' with
  | Step s_n, Stack_step s_s st => inv n s_n s_s st
  | Exit k vs hp rd, Stack_exit k' vs' hp' => k = k' ∧ vs = vs' ∧ hp = hp'
  | Undefined, Stack_undefined => True
  | Step _, (Stack_exit _ _ _ | Stack_undefined)
  | Exit _ _ _ _, (Stack_step _ _ | Stack_undefined)
  | Undefined, (Stack_step _ _ | Stack_exit _ _ _) =>
    False
  end.

(* ======================================================================== *)
(* Going to a continuation                                                  *)
(* ======================================================================== *)

Lemma handlers_of_cons (st : stack) (c : cont_value) (cs : list cont_value) :
  handlers_of st = c :: cs ->
  ∃ rest : stack, st = Trap c :: rest ∧ handlers_of rest = cs.
Proof.
  destruct st
    as [ | [c' | ret exn return_to extra
           | ret exn remaining remaining_arity return_arity return_to extra]
           rest];
    simpl; try discriminate.
  intros [= -> <-]. exists rest. split; reflexivity.
Qed.

Lemma apply_with_extra_args (c : cont_value) (extra vs : list value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  apply_cont_value (with_extra_args c extra) vs statics hp rd
  = apply_cont_value c (vs ++ extra) statics hp rd.
Proof.
  destruct extra; simpl; [ rewrite app_nil_r | ]; reflexivity.
Qed.

(* A stack of the unit has no call in progress. *)
Lemma stack_sim_unit (n : nat) (f : frame_kind) (st : stack)
    (bottom : cont_env) :
  stack_sim n f st bottom ->
  ∀ unit_exn : continuation,
    f = Unit_frame unit_exn ->
    bottom = []
    ∧ ∀ (k : continuation) (vs : list value) (statics : static_env) (hp : heap)
      (rd : read_record),
        length (handlers_of st) <= 1 ->
        unwind st (Leaving_by k) vs statics hp rd = Stack_exit k vs hp.
Proof.
  intros Hsim.
  induction Hsim
    as [ unit_exn | f c st bottom Hsim IH
       | frame ret exn extra rest f bottom conts_n conts_s k_ret k_exn ts
           Hret Hexn Hcorr Henv _ _ ];
    intros u Hf; [ | | discriminate ].
  - split; [ reflexivity | ]. intros k vs statics hp rd _. reflexivity.
  - destruct (IH u Hf) as [-> Hexit]. split; [ reflexivity | ].
    intros k vs statics hp rd Hlen. simpl in Hlen.
    destruct st as [ | entry st]; [ reflexivity | ].
    subst f. inversion Hsim; subst; simpl in Hlen; lia.
Qed.

Lemma call_entry_not_trap (frame : call_frame) (ret exn : continuation)
    (return_to c : cont_value) (extra : list value) :
  call_entry frame ret exn return_to extra <> Trap c.
Proof. destruct frame; discriminate. Qed.

Lemma stack_sim_trap_inv (n : nat) (f : frame_kind) (c : cont_value)
    (st : stack) (bottom : cont_env) :
  stack_sim n f (Trap c :: st) bottom -> stack_sim n f st bottom.
Proof.
  intros Hsim. remember (Trap c :: st) as st0 eqn:Heq.
  destruct Hsim
    as [ unit_exn | f c0 st1 bottom Hsim
       | frame ret exn extra rest f bottom conts_n conts_s k_ret k_exn ts
           _ _ _ _ _ ].
  - discriminate.
  - injection Heq as <- <-. exact Hsim.
  - injection Heq as Heq _. exfalso.
    exact (call_entry_not_trap _ _ _ _ _ _ Heq).
Qed.

(* A stack of the body of code with no handler installed has the call on
   top. *)
Lemma stack_sim_call_inv (n : nat) (ret exn : continuation) (st : stack)
    (bottom : cont_env) :
  stack_sim n (Code_frame ret exn) st bottom ->
  handlers_of st = [] ->
  ∃ (frame : call_frame) (extra : list value) (rest : stack)
    (f : frame_kind) (bottom' conts_n conts_s : cont_env)
    (k_ret k_exn : continuation) (ts : trap_stack),
    st = call_entry frame ret exn (resolve k_ret conts_s) extra :: rest
    ∧ bottom
      = [(ret,
          call_return frame (resolve k_ret conts_n)
            (with_extra_args (resolve k_exn conts_n) extra));
         (exn, with_extra_args (resolve k_exn conts_n) extra)]
    ∧ target_ok n f conts_s k_ret (k_exn :: ts)
    ∧ target_ok n f conts_s k_exn ts
    ∧ map (fun k => resolve k conts_s) (k_exn :: ts) = handlers_of rest
    ∧ env_rel f bottom' conts_n conts_s
    ∧ stack_sim n f rest bottom'.
Proof.
  intros Hsim Hnone.
  remember (Code_frame ret exn) as f0 eqn:Hf.
  destruct Hsim
    as [ unit_exn | f c st bottom Hsim
       | frame ret0 exn0 extra rest f bottom conts_n conts_s k_ret k_exn ts
           Hret Hexn Hcorr Henv Hsim ].
  - discriminate.
  - discriminate.
  - injection Hf as <- <-.
    exists frame, extra, rest, f, bottom, conts_n, conts_s, k_ret, k_exn, ts.
    auto 8.
Qed.

Lemma resolve_bottom_code (ret exn k : continuation) :
  resolve k [(ret, Outer ret); (exn, Outer exn)] = Outer k.
Proof.
  unfold resolve. simpl.
  destruct (Pos.eqb_spec k ret) as [-> | _]; [ reflexivity | ].
  destruct (Pos.eqb_spec k exn) as [-> | _]; reflexivity.
Qed.

(* Going to a continuation that is in order comes to the same on the two
   machines, whatever the stack has to be gone through. *)
(* The body of code that a call starts, on the two machines: the two are
   in step, whatever call it is that is in progress. *)
Lemma callee_inv (statics : static_env) (hp : heap) (rd : read_record) (n : nat)
    (c : code)
    (callee_vars : env) (st : stack) (bottom : cont_env) :
  code_ok statics ->
  code_full_well_formed c = true ->
  stack_sim n
    (Code_frame (code_return_continuation c) (code_exn_continuation c)) st
    bottom ->
  handlers_of st = [] ->
  inv n (State (code_body c) callee_vars bottom statics hp rd)
    (State (code_body c) callee_vars
       [(code_return_continuation c, Outer (code_return_continuation c));
        (code_exn_continuation c, Outer (code_exn_continuation c))]
       statics hp rd)
    (Trap (Outer (code_exn_continuation c)) :: st).
Proof.
  intros Hcode Hc Hsim Hnone.
  destruct c as [params my_closure ret exn metadata body].
  simpl in Hc, Hsim |- *. apply andb_true_iff in Hc as [Hne Hc].
  apply negb_true_iff, Pos.eqb_neq in Hne.
  revert Hc.
  destruct (full_demands [exn] body) as [ds_c | ] eqn:Hbody;
    intros Hc; [ | discriminate ].
  unfold inv. simpl_state.
  split; [ reflexivity | ]. split; [ reflexivity | ].
  split; [ reflexivity | ]. split; [ reflexivity | ].
  split; [ reflexivity | ].
  exists (Code_frame ret exn), bottom, [exn], ds_c.
  split; [ apply Env_bottom | ]. split; [ exact Hbody | ].
  split; [ simpl; rewrite resolve_bottom_code, Hnone; reflexivity | ].
  split.
  { intros d Hd. destruct n as [ | m]; [ exact I | ].
    rewrite target_ok_S, resolve_bottom_code.
    split; [ reflexivity | ].
    unfold code_demands_met in Hc. rewrite forallb_forall in Hc.
    specialize (Hc d Hd). destruct d as [k0 ts0]. simpl in Hc |- *.
    split; [ exact Hne | ]. revert Hc.
    destruct (Pos.eqb k0 exn); intros Hc;
      [ apply trap_stack_eqb_eq; exact Hc | ].
    apply andb_true_iff in Hc as [Hk Hts].
    split; [ apply Pos.eqb_eq; exact Hk
           | apply trap_stack_eqb_eq; exact Hts ]. }
  split; [ apply Sim_trap; exact Hsim | exact Hcode ].
Qed.

(* A call of one shape, made on the two machines from states that are
   in step, under the stack [rest]: what it comes to is in step. What
   each machine does with a partial application is given, and is
   assumed to be in step. *)
Lemma run_call_related (statics : static_env) (hp : heap) (rd : read_record)
    (n : nat)
    (f : frame_kind)
    (bottom conts_n conts_s : cont_env) (k_ret k_exn : continuation)
    (ts : trap_stack) (extra : list value) (rest : stack)
    (kind : call_kind) (shape : option call_shape) (return_arity : arity)
    (partial_n : value -> step_result) (partial_s : value -> stack_result) :
  code_ok statics ->
  shape_calls statics kind shape ->
  target_ok n f conts_s k_ret (k_exn :: ts) ->
  target_ok n f conts_s k_exn ts ->
  map (fun k => resolve k conts_s) (k_exn :: ts) = handlers_of rest ->
  env_rel f bottom conts_n conts_s ->
  stack_sim n f rest bottom ->
  (∀ partial : value, related n (partial_n partial) (partial_s partial)) ->
  related n
    (run_call partial_n shape return_arity (resolve k_ret conts_n)
       (with_extra_args (resolve k_exn conts_n) extra) statics hp rd)
    (stack_run_call partial_s shape return_arity (resolve k_ret conts_s)
       extra statics hp rd rest).
Proof.
  intros Hcode Hcalls Hret Hexn Hcorr Henv Hsim Hpartial.
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ];
    simpl in Hcalls; [ | | apply Hpartial | exact I ].
  - (* A full call. *)
    destruct Hcalls as (called & slots & _ & Hlookup & _).
    unfold run_call, stack_run_call. cbv zeta.
    destruct
      (call_code_cases closure c vs args_arity return_arity statics hp rd)
      as [Hundef | (callee_vars & Hstep)];
      [ rewrite !Hundef; exact I | rewrite !Hstep ].
    refine (callee_inv statics _ rd_ n c callee_vars
              (call_entry Plain_call (code_return_continuation c)
                 (code_exn_continuation c) (resolve k_ret conts_s) extra
               :: rest)
              [(code_return_continuation c,
                call_return Plain_call (resolve k_ret conts_n)
                  (with_extra_args (resolve k_exn conts_n) extra));
               (code_exn_continuation c,
                with_extra_args (resolve k_exn conts_n) extra)]
              Hcode (Hcode called c Hlookup) _ _);
      [ exact (Sim_call n _ _ _ extra rest f bottom conts_n conts_s k_ret
                 k_exn ts Hret Hexn Hcorr Henv Hsim)
      | apply handlers_of_call_entry ].
  - (* The first call of an over-application. *)
    destruct Hcalls as (called & slots & _ & Hlookup & _).
    unfold run_call, stack_run_call. cbv zeta.
    destruct (call_code_cases closure c vs args_arity (code_result_arity c)
                statics hp rd)
      as [Hundef | (callee_vars & Hstep)];
      [ rewrite !Hundef; exact I | rewrite !Hstep ].
    refine (callee_inv statics _ rd_ n c callee_vars
              (call_entry (First_call remaining remaining_arity return_arity)
                 (code_return_continuation c) (code_exn_continuation c)
                 (resolve k_ret conts_s) extra
               :: rest)
              [(code_return_continuation c,
                call_return
                  (First_call remaining remaining_arity return_arity)
                  (resolve k_ret conts_n)
                  (with_extra_args (resolve k_exn conts_n) extra));
               (code_exn_continuation c,
                with_extra_args (resolve k_exn conts_n) extra)]
              Hcode (Hcode called c Hlookup) _ _);
      [ exact (Sim_call n _ _ _ extra rest f bottom conts_n conts_s k_ret
                 k_exn ts Hret Hexn Hcorr Henv Hsim)
      | apply handlers_of_call_entry ].
Qed.

(* Going to a continuation that is in order comes to the same on the two
   machines, whatever the stack has to be gone through. *)
Lemma enter_related (statics : static_env) (hp : heap) (rd : read_record) :
  code_ok statics ->
  ∀ (len : nat) (st : stack),
    length st < len ->
    ∀ (n : nat) (f : frame_kind) (bottom conts_n conts_s : cont_env)
      (k : continuation) (ts : trap_stack) (vs : list value),
      stack_sim (S n) f st bottom ->
      env_rel f bottom conts_n conts_s ->
      target_ok (S n) f conts_s k ts ->
      map (fun k => resolve k conts_s) ts = handlers_of st ->
      related n (apply_cont_value (resolve k conts_n) vs statics hp rd)
        (stack_enter (resolve k conts_s) vs statics hp rd st).
Proof.
  intros Hcode len. induction len as [ | len IH]; intros st Hlen; [ lia | ].
  intros n f bottom conts_n conts_s k ts vs Hsim Henv Htarget Hcorr.
  rewrite target_ok_S in Htarget.
  destruct (env_rel_resolve f bottom conts_n conts_s k Henv)
    as [(h & vars & def_n & def_s & recursive & Hn & Hs & Hdef) | [Hn Hs]];
    rewrite Hn, Hs; rewrite Hs in Htarget.
  - (* A handler. *)
    destruct Htarget as (Hagree & ds & Hh & Hds).
    destruct h as [params metadata handler]. unfold stack_enter. simpl.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | exact I ].
    destruct (bind_params params vs vars) as [handler_vars | ]; [ | exact I ].
    simpl. unfold inv. simpl_state.
    split; [ reflexivity | ]. split; [ reflexivity | ].
    split; [ reflexivity | ]. split; [ reflexivity | ].
    split; [ reflexivity | ].
    exists f, bottom, ts, ds.
    split; [ apply env_rel_rec; exact Hdef | ].
    split; [ exact Hh | ].
    split; [ rewrite <- Hcorr; apply map_ext_in; exact Hagree | ].
    split; [ exact Hds | ].
    split; [ apply stack_sim_weaken; exact Hsim | exact Hcode ].
  - (* What is running is left. *)
    destruct f as [unit_exn | ret exn].
    + (* The unit ends. *)
      destruct (stack_sim_unit (S n) _ st bottom Hsim unit_exn eq_refl)
        as [-> Hexit].
      unfold resolve in Htarget |- *. simpl in Htarget |- *.
      destruct Htarget as [_ Hbottom].
      unfold stack_enter. simpl. rewrite Hexit; [ repeat split; reflexivity | ].
      rewrite <- Hcorr, length_map, Hbottom.
      destruct (Pos.eqb k unit_exn); simpl; lia.
    + (* The body of code is left. *)
      simpl bottom_conts in *. rewrite resolve_bottom_code in Htarget |- *.
      destruct Htarget as [_ [Hne Hbottom]]. revert Hbottom.
      destruct (Pos.eqb_spec k exn) as [-> | Hk]; intros Hbottom.
      * (* By its exception continuation: a raise in the caller. *)
        subst ts. simpl in Hcorr.
        destruct (stack_sim_call_inv (S n) ret exn st bottom Hsim
                    (eq_sym Hcorr))
          as (frame & extra & rest & f & bottom' & conts_n' & conts_s'
              & k_ret & k_exn & ts & -> & -> & _ & Hexn & Hcorr' & Henv'
              & Hsim').
        rewrite (resolve_other exn ret)
          by (intros Heq; exact (Hne (eq_sym Heq))).
        rewrite resolve_head, apply_with_extra_args.
        unfold stack_enter. cbn [apply_cont_value].
        rewrite unwind_call_exn.
        simpl in Hcorr'.
        destruct (handlers_of_cons rest _ _ (eq_sym Hcorr'))
          as (rest' & -> & Hrest').
        apply stack_sim_trap_inv in Hsim'.
        refine (IH rest' _ n f bottom' conts_n' conts_s' k_exn ts
                  (vs ++ extra) Hsim' Henv' Hexn (eq_sym Hrest')).
        simpl in Hlen. lia.
      * (* By its return continuation. *)
        destruct Hbottom as [-> ->]. simpl in Hcorr.
        destruct (handlers_of_cons st _ _ (eq_sym Hcorr))
          as (st' & -> & Hst').
        apply stack_sim_trap_inv in Hsim.
        destruct (stack_sim_call_inv (S n) ret exn st' bottom Hsim Hst')
          as (frame & extra & rest & f & bottom' & conts_n' & conts_s'
              & k_ret & k_exn & ts & -> & -> & Hret & Hexn & Hcorr' & Henv'
              & Hsim').
        rewrite resolve_head. unfold stack_enter.
        destruct frame as [ | remaining remaining_arity return_arity];
          simpl; rewrite Pos.eqb_refl.
        -- (* A return to the caller. *)
           refine (IH rest _ n f bottom' conts_n' conts_s' k_ret
                     (k_exn :: ts) vs Hsim' Henv' Hret Hcorr').
           simpl in Hlen. lia.
        -- (* The first call of an over-application returns: its result
              is called with the arguments that remain. *)
           destruct vs as [ | v [ | v' vs]]; [ exact I | | exact I ].
           apply (run_call_related statics _ rd_ n f bottom' conts_n' conts_s'
                    k_ret k_exn ts extra rest Indirect_unknown_arity);
             [ exact Hcode
             | exact (shape_of_call_calls statics v Indirect_unknown_arity
                        remaining remaining_arity return_arity)
             | apply target_ok_weaken; exact Hret
             | apply target_ok_weaken; exact Hexn | exact Hcorr'
             | exact Henv' | apply stack_sim_weaken; exact Hsim' | ].
           intros partial.
           refine (IH rest _ n f bottom' conts_n' conts_s' k_ret
                     (k_exn :: ts) [partial] Hsim' Henv' Hret Hcorr').
           simpl in Hlen. lia.
Qed.

(* ======================================================================== *)
(* One step                                                                 *)
(* ======================================================================== *)

Lemma eval_static_bindings_code_ok (vars : env)
    (bindings : list static_binding) :
  ∀ statics statics' : static_env,
    forallb binding_full_well_formed bindings = true ->
    code_ok statics ->
    eval_static_bindings vars bindings statics = Some statics' ->
    code_ok statics'.
Proof.
  induction bindings as [ | [id c | id | sym const] bindings IH];
    intros statics statics'; simpl.
  - intros _ Hok [= <-]. exact Hok.
  - intros H Hok. apply andb_true_iff in H as [Hc Hrest].
    destruct (lookup id (static_code statics)); [ discriminate | ].
    apply (IH _ _ Hrest). intros id' c'. simpl.
    destruct (Pos.eqb id' id); [ intros [= <-]; exact Hc | apply Hok ].
  - intros H Hok. exact (IH _ _ H Hok).
  - intros H Hok.
    destruct (lookup sym (static_values statics)); [ discriminate | ].
    destruct (eval_static_const vars const); simpl; [ | discriminate ].
    apply (IH _ _ H). exact Hok.
Qed.

Lemma lookup_in_keys {A : Type} (k : positive) (a : A)
    (l : list (positive * A)) :
  lookup k l = Some a -> In k (map fst l).
Proof.
  induction l as [ | [k' a'] l IH]; simpl; [ discriminate | ].
  destruct (Pos.eqb_spec k k') as [-> | _];
    [ intros _; left; reflexivity | intros H; right; exact (IH H) ].
Qed.

(* A step of the machine that goes by names and a step of the machine
   with the stack, from states that are in step, come to the same. *)
Lemma step_related (n : nat) (s_n s_s : state) (st : stack) :
  inv (S n) s_n s_s st -> related n (step s_n) (stack_step s_s st).
Proof.
  intros (He & Hv & Hstatics & Hheap & Hreads & f & bottom & ts & ds & Henv
          & Hdemands & Hcorr & Htargets & Hsim & Hcode).
  rewrite (state_eta s_n), He, Hv, Hstatics, Hheap, Hreads.
  unfold stack_step.
  assert (Hstep : step s_s
                  = step
                      (State (state_expr s_s) (state_vars s_s)
                         (state_conts s_s) (state_statics s_s)
                           (state_heap s_s) (state_reads s_s)))
    by (rewrite <- state_eta; reflexivity).
  rewrite Hstep. clear Hstep.
  assert (Hweak : ∀ d : continuation * trap_stack,
            In d ds -> target_ok n f (state_conts s_s) (fst d) (snd d))
    by (intros d Hd; apply target_ok_weaken, Htargets, Hd).
  pose proof (stack_sim_weaken n f st bottom Hsim) as Hsim_n.
  set (vars := state_vars s_s) in *. set (statics := state_statics s_s) in *.
  set (conts_n := state_conts s_n) in *.
  set (conts_s := state_conts s_s) in *.
  revert Hdemands.
  destruct (state_expr s_s)
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    intros Hdemands; cbv iota beta.
  - (* Let *)
    rewrite !step_let_exec.
    destruct (exec_named _ _ rd_ _ d) as [[[v hp'] rd'] | ]; [ | exact I ].
    destruct (kind_eqb (kind_of_value v) kind); [ | exact I ].
    simpl. unfold inv. simpl_state.
    split; [ reflexivity | ]. split; [ reflexivity | ].
    split; [ reflexivity | ]. split; [ reflexivity | ].
    split; [ reflexivity | ].
    exists f, bottom, ts, ds.
    split; [ exact Henv | ]. split; [ exact Hdemands | ].
    split; [ exact Hcorr | ]. split; [ exact Hweak | ].
    split; [ exact Hsim_n | exact Hcode ].
  - (* Let_static *)
    rewrite !step_let_static. simpl in Hdemands. revert Hdemands.
    destruct (forallb binding_full_well_formed bindings) eqn:Hbindings;
      intros Hdemands; [ | discriminate ].
    destruct (eval_static_bindings vars bindings statics) as [statics' | ]
      eqn:Heval; [ | exact I ].
    simpl. unfold inv. simpl_state.
    split; [ reflexivity | ]. split; [ reflexivity | ].
    split; [ reflexivity | ]. split; [ reflexivity | ].
    split; [ reflexivity | ].
    exists f, bottom, ts, ds.
    split; [ exact Henv | ]. split; [ exact Hdemands | ].
    split; [ exact Hcorr | ]. split; [ exact Hweak | ].
    split; [ exact Hsim_n | ].
    exact (eval_static_bindings_code_ok vars bindings statics statics'
             Hbindings Hcode Heval).
  - (* Let_cont_non_recursive *)
    rewrite !step_let_cont_non_recursive. simpl. unfold inv. simpl_state.
    split; [ reflexivity | ]. split; [ reflexivity | ].
    split; [ reflexivity | ]. split; [ reflexivity | ].
    split; [ reflexivity | ].
    destruct (full_demands_non_recursive_inv ts k h body ds Hdemands)
      as (Hfresh & ds_body & outer & Hbody & Hleave & Houter & Hk).
    exists f, bottom, ts, ds_body.
    split; [ apply Env_cons; exact Henv | ]. split; [ exact Hbody | ].
    split.
    { rewrite <- Hcorr. apply map_ext_in. intros k' Hin.
      apply resolve_other. intros ->. exact (Hfresh Hin). }
    split; [ | split; [ exact Hsim_n | exact Hcode ] ].
    intros d Hd. destruct (Pos.eqb_spec (fst d) k) as [Heq | Hne].
    + destruct (Hk d Hd Heq) as (Hnotin & ds_h & Hh & Hsub). rewrite Heq.
      destruct n as [ | m]; [ exact I | ].
      rewrite target_ok_S, resolve_head. cbv zeta. simpl. split.
      * intros k' Hin. symmetry. apply resolve_other. intros ->.
        exact (Hnotin Hin).
      * exists ds_h. split; [ exact Hh | ]. intros d' Hd'.
        apply (target_ok_le (S (S m)) m); [ lia | ].
        apply Htargets, Hsub, Hd'.
    + destruct (leave_scope_in [k] ds_body outer Hleave d Hd)
        as [Hin Hnot].
      { apply among_false. intros [Heq | []]. exact (Hne (eq_sym Heq)). }
      apply (target_ok_same_resolve n f conts_s).
      * apply resolve_other. exact Hne.
      * intros k' Hin'. apply resolve_other. intros ->.
        exact (Hnot k (or_introl eq_refl) Hin').
      * apply Hweak, Houter, Hin.
  - (* Let_cont_recursive *)
    rewrite !step_let_cont_recursive.
    destruct (handlers_bind_statics handlers); [ exact I | ].
    simpl. unfold inv. simpl_state.
    split; [ reflexivity | ]. split; [ reflexivity | ].
    split; [ reflexivity | ]. split; [ reflexivity | ].
    split; [ reflexivity | ].
    destruct (full_demands_recursive_inv ts handlers body ds Hdemands)
      as (Hfresh & ds_body & ds_group & Hbody & Hsub & Hleave & Hgroup).
    assert (Hres : ∀ k' : continuation,
              ~ In k' (map fst handlers) ->
              resolve k' (rec_closures handlers vars conts_s ++ conts_s)
              = resolve k' conts_s).
    { intros k' Hnot. rewrite resolve_rec_closures.
      destruct (lookup k' handlers) as [h | ] eqn:Hlookup; [ | reflexivity ].
      exfalso. exact (Hnot (lookup_in_keys k' h handlers Hlookup)). }
    exists f, bottom, ts, ds_body.
    split; [ apply env_rel_rec; exact Henv | ]. split; [ exact Hbody | ].
    split.
    { rewrite <- Hcorr. apply map_ext_in. intros k' Hin. apply Hres.
      intros Hk'. exact (Hfresh k' Hk' Hin). }
    split; [ | split; [ exact Hsim_n | exact Hcode ] ].
    assert (Hall : ∀ m : nat,
              m <= n ->
              ∀ d : continuation * trap_stack,
                In d ds_group ->
                target_ok m f
                  (rec_closures handlers vars conts_s ++ conts_s) (fst d)
                  (snd d)).
    { induction m as [ | m IHm]; intros Hm d Hd; [ exact I | ].
      destruct (in_dec Pos.eq_dec (fst d) (map fst handlers))
        as [Hmember | Hother].
      - destruct (Hgroup d Hd Hmember) as (h & ds_h & Hlookup & Hh & Hin).
        rewrite target_ok_S, resolve_rec_closures, Hlookup. cbv zeta.
        split; [ intros k' _; reflexivity | ].
        exists ds_h. split; [ exact Hh | ]. intros d' Hd'.
        apply IHm; [ lia | apply Hin, Hd' ].
      - destruct (leave_scope_in (map fst handlers) ds_group ds Hleave d Hd)
          as [Hin Hnot]; [ apply among_false; exact Hother | ].
        apply (target_ok_same_resolve (S m) f conts_s).
        + apply Hres. exact Hother.
        + intros k' Hin'. apply Hres. intros Hk'. exact (Hnot k' Hk' Hin').
        + apply (target_ok_le (S n) (S m)); [ lia | ]. apply Htargets, Hin. }
    intros d Hd. apply (Hall n (le_n n)), Hsub, Hd.
  - (* Apply *)
    rewrite step_apply. unfold stack_call.
    destruct (eval_simple vars (ap_callee ap)) as [callee | ]; [ | exact I ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | exact I ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
      [ | exact I ].
    simpl in Hdemands. unfold apply_demands in Hdemands. revert Hdemands.
    destruct ts as [ | top rest_ts]; [ discriminate | ].
    destruct (Pos.eqb_spec top (ap_exn_continuation ap)) as [-> | _];
      [ | discriminate ].
    intros [= <-]. unfold call_function. cbv zeta.
    apply (run_call_related statics _ rd_ n f bottom conts_n conts_s
             (ap_continuation ap) (ap_exn_continuation ap) rest_ts extra st
             (ap_call_kind ap));
      [ exact Hcode | apply shape_of_call_calls
      | apply (Hweak (ap_continuation ap,
                      ap_exn_continuation ap :: rest_ts));
        left; reflexivity
      | apply (Hweak (ap_exn_continuation ap, rest_ts));
        right; left; reflexivity
      | exact Hcorr | exact Henv | exact Hsim_n | ].
    (* A partial application is given to the return continuation of the
       call. *)
    intros partial.
    refine (enter_related statics _ rd_ Hcode (S (length st)) st
              (Nat.lt_succ_diag_r _) n f bottom conts_n conts_s
              (ap_continuation ap) (ap_exn_continuation ap :: rest_ts)
              [partial] Hsim Henv _ Hcorr).
    apply (Htargets (ap_continuation ap,
                     ap_exn_continuation ap :: rest_ts)).
    left. reflexivity.
  - (* Apply_cont *)
    rewrite step_apply_cont. unfold stack_apply_cont.
    destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | exact I ].
    unfold apply_continuation. cbv zeta.
    simpl in Hdemands. unfold apply_cont_demands in Hdemands.
    revert Hdemands.
    destruct (ac_trap_action ac) as [[handler | handler kind] | ]
      eqn:Haction.
    + (* Push *)
      intros [= <-].
      apply (enter_related statics _ rd_ Hcode (S (S (length st)))
               (Trap (resolve handler conts_s) :: st))
        with (f := f) (bottom := bottom) (ts := handler :: ts).
      * simpl. lia.
      * apply Sim_trap. exact Hsim.
      * exact Henv.
      * apply (Htargets (ac_continuation ac, handler :: ts)).
        right. left. reflexivity.
      * simpl. rewrite Hcorr. reflexivity.
    + (* Pop *)
      destruct ts as [ | top rest_ts]; [ discriminate | ].
      destruct (Pos.eqb_spec top handler) as [-> | _]; [ | discriminate ].
      intros [= <-]. simpl in Hcorr.
      destruct (handlers_of_cons st _ _ (eq_sym Hcorr))
        as (rest & -> & Hrest).
      apply stack_sim_trap_inv in Hsim.
      assert (Hgoal :
                related n
                  (apply_cont_value (resolve (ac_continuation ac) conts_n) vs
                     statics (state_heap s_s) (state_reads s_s))
                  (stack_enter (resolve (ac_continuation ac) conts_s) vs
                     statics (state_heap s_s) (state_reads s_s) rest)).
      { apply (enter_related statics _ rd_ Hcode (S (length rest)) rest)
          with (f := f) (bottom := bottom) (ts := rest_ts).
        - lia.
        - exact Hsim.
        - exact Henv.
        - apply (Htargets (ac_continuation ac, rest_ts)). left. reflexivity.
        - exact (eq_sym Hrest). }
      destruct (is_raise ac) eqn:Hraise; [ | exact Hgoal ].
      unfold is_raise in Hraise. rewrite Haction in Hraise.
      apply Pos.eqb_eq in Hraise. rewrite <- Hraise. exact Hgoal.
    + (* No trap action *)
      intros [= <-].
      apply (enter_related statics _ rd_ Hcode (S (length st)) st)
        with (f := f) (bottom := bottom) (ts := ts).
      * lia.
      * exact Hsim.
      * exact Henv.
      * apply (Htargets (ac_continuation ac, ts)). left. reflexivity.
      * exact Hcorr.
  - (* Switch *)
    rewrite !step_switch.
    destruct (eval_simple vars scrutinee) as [v | ]; [ | exact I ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | exact I ].
    destruct (find_arm i arms) as [ac | ] eqn:Hfind; [ | exact I ].
    simpl in Hdemands.
    destruct (arms_demands_find ts arms ds i ac Hdemands Hfind)
      as (ds_ac & Hac & Hincl).
    simpl. unfold inv. simpl_state.
    split; [ reflexivity | ]. split; [ reflexivity | ].
    split; [ reflexivity | ]. split; [ reflexivity | ].
    split; [ reflexivity | ].
    exists f, bottom, ts, ds_ac.
    split; [ exact Henv | ]. split; [ exact Hac | ].
    split; [ exact Hcorr | ].
    split; [ intros d Hd; apply Hweak, Hincl, Hd | ].
    split; [ exact Hsim_n | exact Hcode ].
  - (* Invalid *)
    rewrite !step_invalid. exact I.
Qed.

(* ======================================================================== *)
(* Runs                                                                     *)
(* ======================================================================== *)

Lemma run_related :
  ∀ (fuel : nat) (s_n s_s : state) (st : stack),
    inv fuel s_n s_s st ->
    related 0 (run fuel s_n) (stack_run fuel s_s st).
Proof.
  induction fuel as [ | fuel IH]; intros s_n s_s st Hinv; simpl.
  - destruct Hinv as (He & Hv & Hstatics & Hheap & Hreads & f & bottom & ts
                      & ds & Henv & Hdemands & Hcorr & _ & Hsim & Hcode).
    split; [ exact He | ]. split; [ exact Hv | ]. split; [ exact Hstatics | ].
    split; [ exact Hheap | ]. split; [ exact Hreads | ].
    exists f, bottom, ts, ds. split; [ exact Henv | ].
    split; [ exact Hdemands | ]. split; [ exact Hcorr | ].
    split; [ intros d _; exact I | ]. split; [ exact Hsim | exact Hcode ].
  - pose proof (step_related fuel s_n s_s st Hinv) as Hstep. revert Hstep.
    destruct (step s_n) as [s_n' | k vs | ],
      (stack_step s_s st) as [s_s' st' | k' vs' | ];
      simpl; intros Hstep; try contradiction.
    + exact (IH s_n' s_s' st' Hstep).
    + exact Hstep.
    + exact I.
Qed.

Lemma initial_inv_from (statics : static_env) (unit_exn : continuation)
    (e : expr) (n : nat) :
  static_code statics = [] ->
  full_well_formed unit_exn e = true ->
  inv n (State e [] [] statics empty_heap no_reads)
    (State e [] [] statics empty_heap no_reads)
    (initial_stack unit_exn).
Proof.
  intros Hcode. unfold full_well_formed.
  destruct (full_demands [unit_exn] e) as [ds | ] eqn:Hdemands;
    [ | discriminate ].
  intros Hwf.
  unfold inv. simpl_state.
  split; [ reflexivity | ]. split; [ reflexivity | ].
  split; [ reflexivity | ]. split; [ reflexivity | ].
  split; [ reflexivity | ].
  exists (Unit_frame unit_exn), [], [unit_exn], ds.
  split; [ apply Env_bottom | ]. split; [ exact Hdemands | ].
  split; [ reflexivity | ]. split.
  - intros d Hd. destruct n as [ | n]; [ exact I | ].
    rewrite target_ok_S. unfold resolve. simpl. split; [ reflexivity | ].
    rewrite forallb_forall in Hwf. apply trap_stack_eqb_eq.
    exact (Hwf d Hd).
  - split; [ apply Sim_trap, Sim_unit | ]. intros id c Hlookup.
    rewrite Hcode in Hlookup. discriminate.
Qed.

Theorem stack_agrees_from : stack_agrees_from_statement.
Proof.
  intros statics unit_exn e b Hcode Hd Hwf.
  rewrite <- (full_well_formed_eq unit_exn e Hd) in Hwf.
  assert (Hrun : ∀ fuel : nat,
            related 0 (run fuel (State e [] [] statics empty_heap no_reads))
              (stack_run fuel (State e [] [] statics empty_heap no_reads)
                 (initial_stack unit_exn))).
  { intros fuel. apply run_related, initial_inv_from; assumption. }
  destruct b as [k vs | | ]; simpl.
  - split.
    + intros [fuel H]. exists fuel. specialize (Hrun fuel).
      rewrite H in Hrun. revert Hrun.
      destruct (run fuel _) as [s | k' vs' hp' rd' | ]; simpl;
        try contradiction.
      intros [-> [-> ->]]. exists rd'. reflexivity.
    + intros [fuel [rd H]]. exists fuel. specialize (Hrun fuel).
      rewrite H in Hrun. revert Hrun.
      destruct (stack_run fuel _ _) as [s st | k' vs' | ]; simpl;
        try contradiction.
      intros [-> [-> ->]]. reflexivity.
  - split; intros [fuel H]; exists fuel; specialize (Hrun fuel);
      rewrite H in Hrun; revert Hrun.
    + destruct (run fuel _) as [s | k' vs' | ]; simpl; try contradiction.
      reflexivity.
    + destruct (stack_run fuel _ _) as [s st | k' vs' | ]; simpl;
        try contradiction.
      reflexivity.
  - split; intros H fuel; specialize (Hrun fuel).
    + destruct (H fuel) as (s' & st' & Hs). rewrite Hs in Hrun. revert Hrun.
      destruct (run fuel _) as [s | k' vs' | ]; simpl; try contradiction.
      intros _. exists s. reflexivity.
    + destruct (H fuel) as (s' & Hs). rewrite Hs in Hrun. revert Hrun.
      destruct (stack_run fuel _ _) as [s st | k' vs' | ]; simpl;
        try contradiction.
      intros _. exists s, st. reflexivity.
Qed.

Theorem stack_agrees : stack_agrees_statement.
Proof.
  intros unit_exn e b.
  exact (stack_agrees_from empty_static_env unit_exn e b eq_refl).
Qed.

(* ======================================================================== *)
(* Calls with fewer or more arguments than parameters, on the two machines  *)
(* ======================================================================== *)

(* The theorem says that the machines agree on these calls too. So that
   it does not hold of them for nothing, two units that make such calls
   run to the same exit on each machine. *)

(* [code 1] takes one argument and returns a closure of [code 2], which
   returns its argument. The unit calls a closure of [code 1] with two
   arguments, 1 and 2: the first call returns the closure, which is
   called with 2. *)
Definition over_application_unit : expr :=
  let unit_exn : continuation := 1%positive in
  let leave : continuation := 2%positive in
  let after : continuation := 3%positive in
  let ret : continuation := 4%positive in
  let exn : continuation := 5%positive in
  let f : variable := 1%positive in
  let g : variable := 2%positive in
  let x : variable := 3%positive in
  let my_closure : variable := 4%positive in
  let one : arity := [Singleton Value] in
  Let_static
    [Bind_code 1%positive
       (Code [x] my_closure ret exn (make_code_metadata one one)
          (Let g Value (Set_of_closures 2%positive [])
             (Apply_cont (Apply_cont_expr ret [Var g] None))));
     Bind_code 2%positive
       (Code [x] my_closure ret exn (make_code_metadata one one)
          (Apply_cont (Apply_cont_expr ret [Var x] None)))]
    (Let f Value (Set_of_closures 1%positive [])
       (Let_cont_non_recursive after
          (Handler [x] (make_handler_metadata [Value])
             (Apply_cont (Apply_cont_expr leave [Var x] None)))
          (Apply
             (Apply_expr (Var f) Indirect_unknown_arity
                [Const (Tagged_immediate 1%Z); Const (Tagged_immediate 2%Z)]
                after unit_exn [] (one ++ one) one)))).

Example over_application_runs :
  let s := State over_application_unit [] [] empty_static_env empty_heap
    no_reads in
  run 12 s
    = Exit 2%positive [Value_const (Tagged_immediate 2%Z)] empty_heap no_reads
  ∧ stack_run 12 s (initial_stack 1%positive)
    = Stack_exit 2%positive [Value_const (Tagged_immediate 2%Z)] empty_heap.
Proof. split; vm_compute; reflexivity. Qed.

(* [code 1] takes two arguments and returns the second. The unit calls a
   closure of it with one argument, 1, which makes a partial
   application, and calls that with 2. *)
Definition partial_application_unit : expr :=
  let unit_exn : continuation := 1%positive in
  let leave : continuation := 2%positive in
  let after : continuation := 3%positive in
  let last : continuation := 6%positive in
  let ret : continuation := 4%positive in
  let exn : continuation := 5%positive in
  let f : variable := 1%positive in
  let p : variable := 2%positive in
  let x : variable := 3%positive in
  let y : variable := 5%positive in
  let my_closure : variable := 4%positive in
  let one : arity := [Singleton Value] in
  Let_static
    [Bind_code 1%positive
       (Code [x; y] my_closure ret exn (make_code_metadata (one ++ one) one)
          (Apply_cont (Apply_cont_expr ret [Var y] None)))]
    (Let f Value (Set_of_closures 1%positive [])
       (Let_cont_non_recursive last
          (Handler [x] (make_handler_metadata [Value])
             (Apply_cont (Apply_cont_expr leave [Var x] None)))
          (Let_cont_non_recursive after
             (Handler [p] (make_handler_metadata [Value])
                (Apply
                   (Apply_expr (Var p) Indirect_unknown_arity
                      [Const (Tagged_immediate 2%Z)] last unit_exn [] one
                      one)))
             (Apply
                (Apply_expr (Var f) Indirect_unknown_arity
                   [Const (Tagged_immediate 1%Z)] after unit_exn [] one
                   one))))).

Example partial_application_runs :
  let s := State partial_application_unit [] [] empty_static_env empty_heap
    no_reads in
  run 12 s
    = Exit 2%positive [Value_const (Tagged_immediate 2%Z)] empty_heap no_reads
  ∧ stack_run 12 s (initial_stack 1%positive)
    = Stack_exit 2%positive [Value_const (Tagged_immediate 2%Z)] empty_heap.
Proof. split; vm_compute; reflexivity. Qed.

Example calls_units_well_formed :
  binders_distinct over_application_unit = true
  ∧ traps_well_formed 1%positive over_application_unit = true
  ∧ binders_distinct partial_application_unit = true
  ∧ traps_well_formed 1%positive partial_application_unit = true.
Proof. repeat split; reflexivity. Qed.

(* ======================================================================== *)
(* Which [Apply_cont] is a raise                                            *)
(* ======================================================================== *)

Theorem raises_not_as_marked : raises_not_as_marked_statement.
Proof. split; reflexivity. Qed.

Lemma existsb_false_In {A : Type} (f : A -> bool) (l : list A) (a : A) :
  existsb f l = false -> In a l -> f a = false.
Proof.
  intros Hnone Hin. destruct (f a) eqn:Ha; [ | reflexivity ].
  assert (Hsome : existsb f l = true)
    by (apply existsb_exists; exists a; split; assumption).
  congruence.
Qed.

Lemma in_without (k : positive) (bound names : list positive) :
  In k names -> ~ In k bound -> In k (without bound names).
Proof.
  intros Hin Hnot. apply among_In. rewrite among_without.
  apply andb_true_iff. split.
  - apply among_In. exact Hin.
  - apply negb_true_iff, among_false. exact Hnot.
Qed.

(* What is carried down the term: a handler in scope is not applied as
   an ordinary continuation, and a free continuation that is not such a
   handler is not used as an exception handler. Both go by names, as the
   check does. *)
Lemma raise_as_marked_ok (exn_handlers : list continuation)
    (ac : apply_cont) :
  (∀ k : continuation, In k exn_handlers -> applies k ac = false) ->
  (∀ k : continuation,
    In k (apply_cont_continuations ac) -> ~ In k exn_handlers ->
    traps k ac = false) ->
  raise_as_marked exn_handlers ac = true.
Proof.
  intros Happlies Htraps. unfold raise_as_marked.
  destruct (among (ac_continuation ac) exn_handlers) eqn:Hin.
  - apply among_In in Hin. specialize (Happlies _ Hin).
    unfold applies in Happlies. rewrite Pos.eqb_refl in Happlies.
    simpl in Happlies. apply negb_false_iff in Happlies.
    rewrite Happlies. reflexivity.
  - apply among_false in Hin.
    specialize (Htraps (ac_continuation ac) (or_introl eq_refl) Hin).
    unfold traps in Htraps. unfold is_raise.
    destruct (ac_trap_action ac) as [[k | k kind] | ]; simpl in Htraps |- *;
      try reflexivity.
    rewrite Htraps. reflexivity.
Qed.

Fixpoint raises_as_marked_aux (e : expr) :
  ∀ exn_handlers : list continuation,
    exn_handlers_well_formed e = true ->
    (∀ k : continuation, In k exn_handlers -> applies_cont k e = false) ->
    (∀ k : continuation,
      In k (free_continuations e) -> ~ In k exn_handlers ->
      raises_to k e = false) ->
    raises_as_marked exn_handlers e = true
with handler_raises_as_marked_aux (h : continuation_handler) :
  ∀ exn_handlers : list continuation,
    handler_exn_handlers_well_formed h = true ->
    (∀ k : continuation,
      In k exn_handlers -> handler_applies_cont k h = false) ->
    (∀ k : continuation,
      In k (handler_free_continuations h) -> ~ In k exn_handlers ->
      handler_raises_to k h = false) ->
    handler_raises_as_marked exn_handlers h = true
with binding_raises_as_marked_aux (binding : static_binding) :
  binding_exn_handlers_well_formed binding = true ->
  binding_raises_as_marked binding = true
with code_raises_as_marked_aux (c : code) :
  code_exn_handlers_well_formed c = true -> code_raises_as_marked c = true.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros exn_handlers Hwf Happ Hraise; simpl in Hwf, Happ, Hraise |- *.
    + (* Let *)
      exact (raises_as_marked_aux body exn_handlers Hwf Happ Hraise).
    + (* Let_static *)
      apply andb_true_iff in Hwf as [Hbindings Hbody].
      apply andb_true_iff. split;
        [ | exact (raises_as_marked_aux body exn_handlers Hbody Happ
                     Hraise) ].
      clear -Hbindings binding_raises_as_marked_aux.
      induction bindings as [ | binding bindings IH]; simpl;
        [ reflexivity | ].
      simpl in Hbindings. apply andb_true_iff in Hbindings as [Hb Hbs].
      rewrite (binding_raises_as_marked_aux binding Hb), (IH Hbs).
      reflexivity.
    + (* Let_cont_non_recursive *)
      apply andb_true_iff in Hwf as [Hwf Hbody].
      apply andb_true_iff in Hwf as [Hk Hh].
      apply andb_true_iff. split.
      * apply (handler_raises_as_marked_aux h exn_handlers Hh).
        -- intros k' Hin. specialize (Happ k' Hin).
           apply orb_false_iff in Happ. exact (proj1 Happ).
        -- intros k' Hfree Hnot.
           assert (Hin : In k'
                     (handler_free_continuations h
                      ++ without [k] (free_continuations body)))
             by (apply in_or_app; left; exact Hfree).
           specialize (Hraise k' Hin Hnot).
           apply orb_false_iff in Hraise. exact (proj1 Hraise).
      * assert (Houter : ∀ k' : continuation,
                  In k' (free_continuations body) -> k' <> k ->
                  ~ In k' exn_handlers -> raises_to k' body = false).
        { intros k' Hfree Hne Hnot.
          assert (Hin : In k'
                    (handler_free_continuations h
                     ++ without [k] (free_continuations body))).
          { apply in_or_app. right. apply in_without; [ exact Hfree | ].
            intros [Heq | []]. apply Hne. symmetry. exact Heq. }
          specialize (Hraise k' Hin Hnot).
          apply orb_false_iff in Hraise. exact (proj2 Hraise). }
        destruct (handler_is_exn_handler h).
        -- apply andb_true_iff in Hk as [Hk _].
           apply negb_true_iff in Hk.
           apply (raises_as_marked_aux body (k :: exn_handlers) Hbody).
           ++ intros k' [<- | Hin]; [ exact Hk | ].
              specialize (Happ k' Hin).
              apply orb_false_iff in Happ. exact (proj2 Happ).
           ++ intros k' Hfree Hnot. apply Houter; [ exact Hfree | | ].
              ** intros ->. apply Hnot. left. reflexivity.
              ** intros Hin. apply Hnot. right. exact Hin.
        -- apply negb_true_iff in Hk.
           apply (raises_as_marked_aux body exn_handlers Hbody).
           ++ intros k' Hin. specialize (Happ k' Hin).
              apply orb_false_iff in Happ. exact (proj2 Happ).
           ++ intros k' Hfree Hnot.
              destruct (Pos.eq_dec k' k) as [-> | Hne]; [ exact Hk | ].
              exact (Houter k' Hfree Hne Hnot).
    + (* Let_cont_recursive *)
      apply andb_true_iff in Hwf as [Hgroup Hbody].
      rewrite forallb_forall in Hgroup.
      assert (Hbound : ∀ k' : continuation,
                In k' (map fst handlers) ->
                raises_to k' body = false
                ∧ existsb (fun '(_, h') => handler_raises_to k' h') handlers
                  = false).
      { intros k' Hin. apply in_map_iff in Hin as ([k0 h0] & Heq & Hin).
        simpl in Heq. subst k0. specialize (Hgroup _ Hin). simpl in Hgroup.
        apply andb_true_iff in Hgroup as [Hgroup _].
        apply andb_true_iff in Hgroup as [Hgroup Hothers].
        apply andb_true_iff in Hgroup as [_ Hinbody].
        split; apply negb_true_iff; assumption. }
      assert (Hfree_outer : ∀ k' : continuation,
                In k'
                  (flat_map (fun '(_, h) => handler_free_continuations h)
                     handlers
                   ++ free_continuations body) ->
                ~ In k' (map fst handlers) -> ~ In k' exn_handlers ->
                existsb (fun '(_, h) => handler_raises_to k' h) handlers
                = false
                ∧ raises_to k' body = false).
      { intros k' Hfree Hunbound Hnot. apply orb_false_iff.
        apply (Hraise k'); [ | exact Hnot ].
        apply in_without; assumption. }
      apply andb_true_iff. split.
      * assert (Hhandlers : ∀ (k0 : continuation) (h0 : continuation_handler),
                  In (k0, h0) handlers ->
                  handler_exn_handlers_well_formed h0 = true
                  ∧ (∀ k' : continuation,
                      In k' exn_handlers ->
                      handler_applies_cont k' h0 = false)
                  ∧ (∀ k' : continuation,
                      In k' (handler_free_continuations h0) ->
                      ~ In k' exn_handlers ->
                      handler_raises_to k' h0 = false)).
        { intros k0 h0 Hin. split; [ | split ].
          - specialize (Hgroup _ Hin). simpl in Hgroup.
            apply andb_true_iff in Hgroup. exact (proj2 Hgroup).
          - intros k' Hexn. specialize (Happ k' Hexn).
            apply orb_false_iff in Happ.
            exact (existsb_false_In _ _ (k0, h0) (proj1 Happ) Hin).
          - intros k' Hfree Hnot.
            assert (Hnone : existsb
                      (fun '(_, h) => handler_raises_to k' h) handlers
                    = false).
            { destruct (in_dec Pos.eq_dec k' (map fst handlers))
                as [Hbound' | Hunbound].
              - exact (proj2 (Hbound k' Hbound')).
              - apply (Hfree_outer k'); [ | exact Hunbound | exact Hnot ].
                apply in_or_app. left. apply in_flat_map.
                exists (k0, h0). split; assumption. }
            exact (existsb_false_In _ _ (k0, h0) Hnone Hin). }
        clear -Hhandlers handler_raises_as_marked_aux.
        induction handlers as [ | [k0 h0] handlers IH]; simpl;
          [ reflexivity | ].
        apply andb_true_iff. split.
        -- destruct (Hhandlers k0 h0 (or_introl eq_refl))
             as (Hwf0 & Happ0 & Hraise0).
           exact (handler_raises_as_marked_aux h0 exn_handlers Hwf0 Happ0
                    Hraise0).
        -- apply IH. intros k1 h1 Hin. apply (Hhandlers k1 h1). right.
           exact Hin.
      * apply (raises_as_marked_aux body exn_handlers Hbody).
        -- intros k' Hin. specialize (Happ k' Hin).
           apply orb_false_iff in Happ. exact (proj2 Happ).
        -- intros k' Hfree Hnot.
           destruct (in_dec Pos.eq_dec k' (map fst handlers))
             as [Hbound' | Hunbound].
           ++ exact (proj1 (Hbound k' Hbound')).
           ++ apply (Hfree_outer k'); [ | exact Hunbound | exact Hnot ].
              apply in_or_app. right. exact Hfree.
    + (* Apply *)
      reflexivity.
    + (* Apply_cont *)
      exact (raise_as_marked_ok exn_handlers ac Happ Hraise).
    + (* Switch *)
      apply forallb_forall. intros arm Hin.
      apply raise_as_marked_ok.
      * intros k Hk. specialize (Happ k Hk).
        exact (existsb_false_In _ _ arm Happ Hin).
      * intros k Hk Hnot.
        assert (Hfree : In k
                  (flat_map
                     (fun arm => apply_cont_continuations (snd arm)) arms))
          by (apply in_flat_map; exists arm; split; assumption).
        specialize (Hraise k Hfree Hnot).
        exact (existsb_false_In _ _ arm Hraise Hin).
    + (* Invalid *)
      reflexivity.
  - destruct h as [params metadata handler]. simpl.
    exact (raises_as_marked_aux handler).
  - destruct binding as [id c | id | s const]; simpl; intros Hwf;
      try reflexivity.
    exact (code_raises_as_marked_aux c Hwf).
  - destruct c as [params my_closure ret exn metadata body]. simpl.
    intros Hwf.
    apply andb_true_iff in Hwf as [Hwf Hbody].
    apply andb_true_iff in Hwf as [Hways Happ].
    apply negb_true_iff in Happ.
    unfold ways_out_well_formed in Hways. rewrite forallb_forall in Hways.
    apply (raises_as_marked_aux body [exn] Hbody).
    + intros k [<- | []]. exact Happ.
    + intros k Hfree Hnot. specialize (Hways k Hfree).
      apply andb_true_iff in Hways as [Hways _].
      destruct (Pos.eqb_spec k exn) as [Heq | _];
        [ exfalso; apply Hnot; left; symmetry; exact Heq | ].
      apply negb_true_iff. exact Hways.
Qed.

Theorem raises_as_marked : raises_as_marked_statement.
Proof.
  intros unit_exn e Hwf. unfold unit_exn_handlers_well_formed in Hwf.
  apply andb_true_iff in Hwf as [Hwf Hbody].
  apply andb_true_iff in Hwf as [Hways Happ].
  apply negb_true_iff in Happ.
  unfold ways_out_well_formed in Hways. rewrite forallb_forall in Hways.
  apply (raises_as_marked_aux e [unit_exn] Hbody).
  - intros k [<- | []]. exact Happ.
  - intros k Hfree Hnot. specialize (Hways k Hfree).
    apply andb_true_iff in Hways as [Hways _].
    destruct (Pos.eqb_spec k unit_exn) as [Heq | _];
      [ exfalso; apply Hnot; left; symmetry; exact Heq | ].
    apply negb_true_iff. exact Hways.
Qed.

(* ======================================================================== *)
(* The machine with the stack is equivariant                                *)
(* ======================================================================== *)

Theorem initial_stack_equivariant : initial_stack_equivariant_statement.
Proof. intros r unit_exn. reflexivity. Qed.

Definition rename_leaving (r : renaming) (how : leaving) : leaving :=
  match how with
  | Leaving_by k => Leaving_by (rename_continuation r k)
  | Raising => Raising
  end.

(* [unwind] goes down the stack two entries at a time where a call
   returns, so the induction is on a bound of the length. *)
Lemma stack_run_call_equivariant (r : renaming)
    (partial_s partial_s' : value -> stack_result)
    (shape : option call_shape) (return_arity : arity)
    (return_to : cont_value) (extra : list value) (statics : static_env)
      (hp : heap) (rd : read_record)
    (st : stack) :
  (∀ partial : value,
    partial_s' partial = rename_stack_result r (partial_s partial)) ->
  stack_run_call partial_s' (option_map (rename_shape r) shape) return_arity
    (rename_cont_value r return_to) extra (rename_static_env r statics) hp rd
    (rename_stack r st)
  = rename_stack_result r
      (stack_run_call partial_s shape return_arity return_to extra statics hp rd
         st).
Proof.
  intros Hpartial.
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ];
    simpl; [ | | apply Hpartial | reflexivity ].
  - replace (code_return_continuation (rename_code r c))
      with (rename_continuation r (code_return_continuation c))
      by (destruct c; reflexivity).
    replace (code_exn_continuation (rename_code r c))
      with (rename_continuation r (code_exn_continuation c))
      by (destruct c; reflexivity).
    pose proof
      (call_code_equivariant r closure c vs args_arity return_arity
         (Outer (code_return_continuation c))
         (Outer (code_exn_continuation c)) statics) as Hcall.
    simpl in Hcall. rewrite Hcall.
    destruct
      (call_code closure c vs args_arity return_arity
         (Outer (code_return_continuation c))
         (Outer (code_exn_continuation c)) statics hp rd);
      reflexivity.
  - replace (code_return_continuation (rename_code r c))
      with (rename_continuation r (code_return_continuation c))
      by (destruct c; reflexivity).
    replace (code_exn_continuation (rename_code r c))
      with (rename_continuation r (code_exn_continuation c))
      by (destruct c; reflexivity).
    replace (code_result_arity (rename_code r c)) with (code_result_arity c)
      by (destruct c; reflexivity).
    pose proof
      (call_code_equivariant r closure c vs args_arity (code_result_arity c)
         (Outer (code_return_continuation c))
         (Outer (code_exn_continuation c)) statics) as Hcall.
    simpl in Hcall. rewrite Hcall.
    destruct
      (call_code closure c vs args_arity (code_result_arity c)
         (Outer (code_return_continuation c))
         (Outer (code_exn_continuation c)) statics hp rd);
      reflexivity.
Qed.

(* [unwind] goes down the stack two entries at a time where a call
   returns, so the induction is on a bound of the length. *)
Lemma unwind_equivariant (r : renaming) (statics : static_env) (hp : heap)
    (rd : read_record) :
  ∀ (n : nat) (st : stack) (how : leaving) (vs : list value),
    length st <= n ->
    unwind (rename_stack r st) (rename_leaving r how) vs
      (rename_static_env r statics) hp rd
    = rename_stack_result r (unwind st how vs statics hp rd).
Proof.
  intros n. revert hp rd.
  induction n as [ | n IH]; intros hp rd st how vs Hlen.
  - destruct st; [ | simpl in Hlen; lia ]. destruct how; reflexivity.
  - destruct how as [k | ].
    + destruct st
        as [ | [c | ret exn return_to extra
               | ret exn remaining remaining_arity return_arity return_to
                   extra] rest];
        [ reflexivity | | | ].
      * destruct rest
          as [ | [c' | ret exn return_to extra
                 | ret exn remaining remaining_arity return_arity return_to
                     extra] rest];
          [ reflexivity | reflexivity | | ].
        -- simpl. rewrite rename_continuation_eqb.
           destruct (Pos.eqb k ret); [ | reflexivity ].
           rewrite apply_cont_value_equivariant.
           destruct (apply_cont_value return_to vs statics hp rd)
             as [s | k' vs' hp' rd' | ];
             simpl; [ reflexivity | | reflexivity ].
           apply (IH _ _ rest (Leaving_by k') vs'). simpl in Hlen. lia.
        -- cbn [rename_stack map rename_stack_entry rename_leaving unwind].
           rewrite rename_continuation_eqb.
           destruct (Pos.eqb k ret); [ | reflexivity ].
           destruct vs as [ | v [ | v' vs]]; try reflexivity.
           rewrite shape_of_call_equivariant.
           apply stack_run_call_equivariant. intros partial.
           rewrite apply_cont_value_equivariant.
           destruct (apply_cont_value return_to [partial] statics hp rd)
             as [s | k' vs' hp' rd' | ];
             simpl; [ reflexivity | | reflexivity ].
           apply (IH _ _ rest (Leaving_by k') vs'). simpl in Hlen. lia.
      * simpl. rewrite rename_continuation_eqb.
        destruct (Pos.eqb k exn); [ | reflexivity ].
        apply (IH _ _ rest Raising (vs ++ extra)). simpl in Hlen. lia.
      * simpl. rewrite rename_continuation_eqb.
        destruct (Pos.eqb k exn); [ | reflexivity ].
        apply (IH _ _ rest Raising (vs ++ extra)). simpl in Hlen. lia.
    + destruct st
        as [ | [c | ret exn return_to extra
               | ret exn remaining remaining_arity return_arity return_to
                   extra] rest];
        [ reflexivity | | reflexivity | reflexivity ].
      simpl. rewrite apply_cont_value_equivariant.
      destruct (apply_cont_value c vs statics hp rd) as [s | k' vs' hp' rd' | ];
        simpl;
        [ reflexivity | | reflexivity ].
      apply (IH _ _ rest (Leaving_by k') vs'). simpl in Hlen. lia.
Qed.

Lemma stack_enter_equivariant (r : renaming) (c : cont_value)
    (vs : list value) (statics : static_env) (hp : heap) (rd : read_record)
      (st : stack) :
  stack_enter (rename_cont_value r c) vs (rename_static_env r statics) hp rd
    (rename_stack r st)
  = rename_stack_result r (stack_enter c vs statics hp rd st).
Proof.
  unfold stack_enter. rewrite apply_cont_value_equivariant.
  destruct (apply_cont_value c vs statics hp rd) as [s | k vs' | ]; simpl;
    [ reflexivity | | reflexivity ].
  exact (unwind_equivariant r statics _ rd_ (length st) st (Leaving_by k) vs'
           (le_n _)).
Qed.

Lemma is_raise_equivariant (r : renaming) (ac : apply_cont) :
  is_raise (rename_apply_cont r ac) = is_raise ac.
Proof.
  unfold is_raise. simpl.
  destruct (ac_trap_action ac) as [[k | k kind] | ]; simpl;
    try reflexivity.
  apply rename_continuation_eqb.
Qed.

Lemma stack_call_equivariant (r : renaming) (ap : apply_expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record)
      (st : stack) :
  stack_call (rename_apply r ap) (rename_env r vars)
    (rename_cont_env r conts) (rename_static_env r statics) hp rd
    (rename_stack r st)
  = rename_stack_result r (stack_call ap vars conts statics hp rd st).
Proof.
  unfold stack_call. rewrite exn_extra_simples_rename. simpl.
  rewrite eval_simple_equivariant, !eval_simples_equivariant.
  destruct (eval_simple vars (ap_callee ap)) as [callee | ];
    [ | reflexivity ].
  destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | reflexivity ].
  destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
    [ | reflexivity ].
  rewrite shape_of_call_equivariant, resolve_equivariant.
  apply stack_run_call_equivariant. intros partial.
  apply stack_enter_equivariant.
Qed.

Lemma stack_apply_cont_equivariant (r : renaming) (ac : apply_cont)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record)
      (st : stack) :
  stack_apply_cont (rename_apply_cont r ac) (rename_env r vars)
    (rename_cont_env r conts) (rename_static_env r statics) hp rd
    (rename_stack r st)
  = rename_stack_result r (stack_apply_cont ac vars conts statics hp rd st).
Proof.
  unfold stack_apply_cont. rewrite is_raise_equivariant. simpl.
  rewrite eval_simples_equivariant.
  destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | reflexivity ].
  rewrite resolve_equivariant.
  destruct (ac_trap_action ac) as [[k | k kind] | ]; simpl.
  - rewrite resolve_equivariant.
    exact (stack_enter_equivariant r _ vs statics _ rd_
             (Trap (resolve k conts) :: st)).
  - destruct (is_raise ac).
    + exact (unwind_equivariant r statics _ rd_ (length st) st Raising vs
               (le_n _)).
    + destruct st
        as [ | [c | ret exn return_to extra
               | ret exn remaining remaining_arity return_arity return_to
                   extra] rest];
        [ reflexivity | | reflexivity | reflexivity ].
      apply stack_enter_equivariant.
  - apply stack_enter_equivariant.
Qed.

Theorem stack_step_equivariant : stack_step_equivariant_statement.
Proof.
  intros r s st.
  pose proof (step_equivariant r s) as Hstep. revert Hstep.
  unfold stack_step. rewrite (state_eta s). unfold rename_state.
  simpl_state.
  destruct (state_expr s)
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    intros Hstep;
    try (cbn [rename_expr]; cbn [rename_expr] in Hstep; rewrite Hstep;
         match goal with
         | |- context [step ?s'] => destruct (step s')
         end;
         reflexivity).
  - apply stack_call_equivariant.
  - apply stack_apply_cont_equivariant.
Qed.

Theorem stack_run_equivariant : stack_run_equivariant_statement.
Proof.
  intros r fuel. induction fuel as [ | fuel IH]; intros s st; simpl;
    [ reflexivity | ].
  rewrite stack_step_equivariant.
  destruct (stack_step s st) as [s' st' | k vs | ]; simpl;
    [ apply IH | | ]; reflexivity.
Qed.

Theorem stack_behaves_equivariant : stack_behaves_equivariant_statement.
Proof.
  intros r s st [k vs | | ]; simpl.
  - split; intros [fuel Hrun]; exists fuel; revert Hrun;
      rewrite stack_run_equivariant.
    + destruct (stack_run fuel s st) as [s' st' | k' vs' | ]; simpl;
        try discriminate.
      intros [= Hk -> ->]. apply rename_continuation_injective in Hk.
      subst k'. reflexivity.
    + intros ->. reflexivity.
  - split; intros [fuel Hrun]; exists fuel; revert Hrun;
      rewrite stack_run_equivariant;
      destruct (stack_run fuel s st); simpl; intros Hrun;
      try discriminate; reflexivity.
  - split; intros Hdiverges fuel; specialize (Hdiverges fuel);
      revert Hdiverges; rewrite stack_run_equivariant;
      destruct (stack_run fuel s st) as [s' st' | k vs | ]; simpl;
      intros (t & st'' & Ht); try discriminate; eauto.
Qed.
