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


(* Proofs of FlambdaStaticsInScopeSpec.v. Nothing here needs review.

   The run of a unit keeps two invariants. On terms ([terms_inv]): the
   expression of the state passes the check under what the static
   environment binds, and each continuation that it can use leads to a
   state that does, once what the expression is still to bind at
   toplevel is bound ([cont_good]). On values ([state_statics_bound]):
   everything that the state holds mentions only what is bound. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaMapProofs.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaPrimitiveSpec FlambdaPrimitiveProofs FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaWellFormedSpec FlambdaWellFormedProofs
  FlambdaStaticsInScopeSpec.
Import ListNotations.

(* [all_among] is the test, here. FlambdaMapProofs.v proves its statement
   under the same name. *)
Lemma all_among_spec (names allowed : list positive) :
  all_among names allowed = true
  <-> (∀ name : positive,
        among name names = true -> among name allowed = true).
Proof. exact (FlambdaMapProofs.all_among names allowed). Qed.

(* ======================================================================== *)
(* One scope inside another                                                 *)
(* ======================================================================== *)

Definition sub (names names' : list positive) : Prop :=
  ∀ name : positive, among name names = true -> among name names' = true.

Lemma sub_refl (names : list positive) : sub names names.
Proof. intros name H. exact H. Qed.

Lemma sub_trans (names names' names'' : list positive) :
  sub names names' -> sub names' names'' -> sub names names''.
Proof. intros H H' name Hn. exact (H' name (H name Hn)). Qed.

Lemma sub_app_l (names names' : list positive) : sub names (names ++ names').
Proof. intros name H. rewrite among_app, H. reflexivity. Qed.

Lemma sub_app_r (names names' : list positive) : sub names' (names ++ names').
Proof. intros name H. rewrite among_app, H. apply orb_true_r. Qed.

Lemma sub_app (names names' all : list positive) :
  sub names all -> sub names' all -> sub (names ++ names') all.
Proof.
  intros H H' name Hn. rewrite among_app in Hn.
  apply orb_true_iff in Hn as [Hn | Hn]; [ exact (H name Hn) | ].
  exact (H' name Hn).
Qed.

Lemma sub_app_mono (first names names' : list positive) :
  sub names names' -> sub (first ++ names) (first ++ names').
Proof.
  intros H. apply sub_app; [ apply sub_app_l | ].
  exact (sub_trans _ _ _ H (sub_app_r first names')).
Qed.

Lemma sub_cons_mono (first : positive) (names names' : list positive) :
  sub names names' -> sub (first :: names) (first :: names').
Proof. exact (sub_app_mono [first] names names'). Qed.

Lemma sub_nil (names : list positive) : sub [] names.
Proof. intros name H. discriminate H. Qed.

Lemma among_sub (name : positive) (names names' : list positive) :
  sub names names' -> among name names = true -> among name names' = true.
Proof. intros H. apply H. Qed.

Lemma all_among_sub (names allowed allowed' : list positive) :
  sub allowed allowed' ->
  all_among names allowed = true ->
  all_among names allowed' = true.
Proof.
  intros H Hall. apply all_among_spec. intros name Hn.
  exact (H name (proj1 (all_among_spec names allowed) Hall name Hn)).
Qed.

Lemma all_among_is_sub (names allowed : list positive) :
  all_among names allowed = true <-> sub names allowed.
Proof. apply all_among_spec. Qed.

Lemma all_among_app (names names' allowed : list positive) :
  all_among (names ++ names') allowed
  = all_among names allowed && all_among names' allowed.
Proof. unfold all_among. apply forallb_app. Qed.

Lemma among_false_app (name : positive) (names names' : list positive) :
  among name (names ++ names') = false ->
  among name names = false ∧ among name names' = false.
Proof. rewrite among_app. apply orb_false_iff. Qed.

Lemma among_flat_map_false {A : Type} (f : A -> list positive)
    (name : positive) (l : list A) (a : A) :
  among name (flat_map f l) = false -> a ∈ l -> among name (f a) = false.
Proof.
  intros Hnone Hin. destruct (among name (f a)) eqn:Hname; [ | reflexivity ].
  rewrite <- Hnone. symmetry. apply among_In, in_flat_map. exists a.
  split; [ exact Hin | apply among_In, Hname ].
Qed.

Lemma among_false_cons (name first : positive) (names : list positive) :
  among name (first :: names) = false ->
  name <> first ∧ among name names = false.
Proof.
  rewrite among_cons. intros H. apply orb_false_iff in H as [Hne H].
  split; [ | exact H ]. intros ->. rewrite Pos.eqb_refl in Hne.
  discriminate Hne.
Qed.

(* ======================================================================== *)
(* The check under a larger scope                                           *)
(* ======================================================================== *)

Lemma simple_scope_sub (syms syms' : list symbol) (s : simple) :
  sub syms syms' ->
  simple_statics_in_scope syms s = true ->
  simple_statics_in_scope syms' s = true.
Proof. intros H. destruct s; simpl; auto. Qed.

Lemma simples_scope_sub (syms syms' : list symbol) (args : list simple) :
  sub syms syms' ->
  simples_statics_in_scope syms args = true ->
  simples_statics_in_scope syms' args = true.
Proof.
  intros H. unfold simples_statics_in_scope. rewrite !forallb_forall.
  intros Hall s Hs. exact (simple_scope_sub _ _ s H (Hall s Hs)).
Qed.

Lemma value_slots_scope_sub (syms syms' : list symbol)
    (value_slots : list (value_slot * simple)) :
  sub syms syms' ->
  value_slots_statics_in_scope syms value_slots = true ->
  value_slots_statics_in_scope syms' value_slots = true.
Proof.
  intros H. unfold value_slots_statics_in_scope. rewrite !forallb_forall.
  intros Hall slot Hs. exact (simple_scope_sub _ _ _ H (Hall slot Hs)).
Qed.

Lemma named_scope_sub (syms syms' : list symbol) (ids ids' : list code_id)
    (d : named) :
  sub syms syms' ->
  sub ids ids' ->
  named_statics_in_scope syms ids d = true ->
  named_statics_in_scope syms' ids' d = true.
Proof.
  intros Hs Hi.
  destruct d as [s | p args | code slots | slot decls slots | slot closure
                | from slot closure | from to closure];
    simpl; rewrite ?andb_true_iff.
  - apply simple_scope_sub, Hs.
  - apply simples_scope_sub, Hs.
  - intros [Hc Hslots]. split; [ exact (Hi code Hc) | ].
    exact (value_slots_scope_sub _ _ _ Hs Hslots).
  - intros [Hc Hslots]. split; [ exact (all_among_sub _ _ _ Hi Hc) | ].
    exact (value_slots_scope_sub _ _ _ Hs Hslots).
  - apply simple_scope_sub, Hs.
  - apply simple_scope_sub, Hs.
  - apply simple_scope_sub, Hs.
Qed.

Lemma static_const_scope_sub (syms syms' : list symbol)
    (ids ids' : list code_id) (const : static_const) :
  sub syms syms' ->
  sub ids ids' ->
  static_const_statics_in_scope syms ids const = true ->
  static_const_statics_in_scope syms' ids' const = true.
Proof.
  intros Hs Hi.
  destruct const as [code slots | slot symbols decls slots | tag fields
                    | contents | bytes | elements];
    simpl; rewrite ?andb_true_iff.
  - intros [Hc Hslots]. split; [ exact (Hi code Hc) | ].
    exact (value_slots_scope_sub _ _ _ Hs Hslots).
  - intros [[Hsyms Hc] Hslots].
    split; [ split | exact (value_slots_scope_sub _ _ _ Hs Hslots) ].
    + exact (all_among_sub _ _ _ Hs Hsyms).
    + exact (all_among_sub _ _ _ Hi Hc).
  - apply simples_scope_sub, Hs.
  - apply simple_scope_sub, Hs.
  - auto.
  - apply simples_scope_sub, Hs.
Qed.

Lemma apply_scope_sub (syms syms' : list symbol) (ids ids' : list code_id)
    (ap : apply_expr) :
  sub syms syms' ->
  sub ids ids' ->
  apply_statics_in_scope syms ids ap = true ->
  apply_statics_in_scope syms' ids' ap = true.
Proof.
  intros Hs Hi. unfold apply_statics_in_scope. rewrite !andb_true_iff.
  intros [[[Hcallee Hkind] Hargs] Hextra].
  repeat split.
  - exact (simple_scope_sub _ _ _ Hs Hcallee).
  - destruct (ap_call_kind ap) as [code | | [code_ids | ]];
      [ exact (Hi code Hkind) | auto | | auto ].
    apply forallb_forall. intros id Hin. apply Hi.
    exact (proj1 (forallb_forall _ _) Hkind id Hin).
  - exact (simples_scope_sub _ _ _ Hs Hargs).
  - exact (simples_scope_sub _ _ _ Hs Hextra).
Qed.

Lemma group_consts_scope_sub (bindings : list static_binding) :
  ∀ (before before' all all' : list symbol) (ids ids' : list code_id),
    sub before before' ->
    sub all all' ->
    sub ids ids' ->
    group_consts_in_scope before all ids bindings = true ->
    group_consts_in_scope before' all' ids' bindings = true.
Proof.
  induction bindings as [ | [id c | id | s const] bindings IH];
    intros before before' all all' ids ids' Hb Ha Hi; simpl; auto.
  - apply IH; assumption.
  - apply IH; assumption.
  - destruct (is_closure_const const); rewrite !andb_true_iff;
      intros [Hconst Hrest]; split.
    + exact (static_const_scope_sub _ _ _ _ _ Ha Hi Hconst).
    + exact (IH _ _ _ _ _ _ Hb Ha Hi Hrest).
    + exact (static_const_scope_sub _ _ _ _ _ Hb Hi Hconst).
    + exact (IH _ _ _ _ _ _ (sub_cons_mono s _ _ Hb) Ha Hi Hrest).
Qed.

(* The check passes under a larger scope. *)
Fixpoint expr_scope_sub (e : expr) {struct e} :
  ∀ (syms syms' : list symbol) (ids ids' : list code_id),
    sub syms syms' ->
    sub ids ids' ->
    expr_statics_in_scope syms ids e = true ->
    expr_statics_in_scope syms' ids' e = true.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    intros syms syms' ids ids' Hs Hi; simpl.
  - rewrite !andb_true_iff. intros [Hd Hbody].
    split; [ exact (named_scope_sub _ _ _ _ _ Hs Hi Hd) | ].
    exact (expr_scope_sub body _ _ _ _ Hs Hi Hbody).
  - rewrite !andb_true_iff. intros [[Hconsts Hcode] Hbody].
    pose proof (sub_app_mono (flat_map symbols_of_binding bindings) _ _ Hs)
      as Hall.
    pose proof
      (sub_app_mono (flat_map declared_code_ids_of_binding bindings) _ _ Hi)
      as Hids.
    split; [ split | ].
    + refine (group_consts_scope_sub bindings _ _ _ _ _ _ _ Hall Hids
                Hconsts).
      apply sub_app_mono, Hs.
    + clear Hconsts Hbody.
      assert (Hlist :
        ∀ (all all' : list symbol) (cids cids' : list code_id),
          sub all all' ->
          sub cids cids' ->
          forallb (binding_statics_in_scope all cids) bindings = true ->
          forallb (binding_statics_in_scope all' cids') bindings = true).
      { intros all all' cids cids' Ha Hc. clear Hcode Hall Hids.
        induction bindings as [ | binding bindings IH]; simpl; [ auto | ].
        rewrite !andb_true_iff. intros [Hb Hrest].
        split; [ | exact (IH Hrest) ].
        destruct binding as [id [params my_closure ret exn metadata cbody]
                            | id | s const];
          simpl in Hb |- *; auto.
        exact (expr_scope_sub cbody _ _ _ _ Ha Hc Hb). }
      exact (Hlist _ _ _ _ Hall Hids Hcode).
    + exact (expr_scope_sub body _ _ _ _ Hall Hids Hbody).
  - rewrite !andb_true_iff. intros [Hh Hbody].
    split; [ | exact (expr_scope_sub body _ _ _ _ Hs Hi Hbody) ].
    destruct h as [params metadata handler]. simpl in Hh |- *.
    refine (expr_scope_sub handler _ _ _ _ _ _ Hh);
      apply sub_app_mono; assumption.
  - rewrite !andb_true_iff. intros [Hhs Hbody].
    split; [ | exact (expr_scope_sub body _ _ _ _ Hs Hi Hbody) ].
    clear Hbody.
    induction handlers as [ | [k' [params metadata handler]] handlers IH];
      simpl in Hhs |- *; [ reflexivity | ].
    apply andb_true_iff in Hhs as [Hh Hrest].
    apply andb_true_iff. split; [ | exact (IH Hrest) ].
    refine (expr_scope_sub handler _ _ _ _ _ _ Hh);
      apply sub_app_mono; assumption.
  - apply apply_scope_sub; assumption.
  - apply simples_scope_sub, Hs.
  - rewrite !andb_true_iff. intros [Hscrutinee Harms].
    split; [ exact (simple_scope_sub _ _ _ Hs Hscrutinee) | ].
    revert Harms. rewrite !forallb_forall. intros Harms arm Hin.
    exact (simples_scope_sub _ _ _ Hs (Harms arm Hin)).
  - auto.
Qed.

(* ======================================================================== *)
(* What a group binds                                                       *)
(* ======================================================================== *)

(* After a group, the symbols bound are those of the group and those
   bound before; likewise the code ids. *)
Lemma eval_static_bindings_bound (vars : env)
    (bindings : list static_binding) :
  ∀ (statics statics' : static_env) (name : positive),
    eval_static_bindings vars bindings statics = Some statics' ->
    among name (bound_symbols statics')
    = among name (flat_map symbols_of_binding bindings)
      || among name (bound_symbols statics)
    ∧ among name (bound_code_ids statics')
      = among name (flat_map code_ids_of_binding bindings)
        || among name (bound_code_ids statics).
Proof.
  induction bindings as [ | [id c | id | s const] bindings IH];
    intros statics statics' name; simpl.
  - intros [= <-]. auto.
  - destruct (lookup id (static_code statics)); [ discriminate | ].
    intros Heval. destruct (IH _ _ name Heval) as [Hs Hc].
    rewrite Hs, Hc. unfold bound_symbols, bound_code_ids, among. simpl.
    split; [ reflexivity | ].
    destruct (Pos.eqb name id); simpl;
      [ apply orb_true_r | reflexivity ].
  - apply IH.
  - destruct (lookup s (static_values statics)); [ discriminate | ].
    destruct (eval_static_const vars const) as [allocated | ]; simpl;
      [ | discriminate ].
    intros Heval. destruct (IH _ _ name Heval) as [Hs Hc].
    rewrite Hs, Hc. unfold bound_symbols, bound_code_ids, among. simpl.
    split; [ | reflexivity ].
    destruct (Pos.eqb name s); simpl;
      [ apply orb_true_r | reflexivity ].
Qed.

(* The code that is bound after a group was bound before or is bound by
   the group. *)
Lemma eval_static_bindings_code (vars : env)
    (bindings : list static_binding) :
  ∀ (statics statics' : static_env) (id : code_id) (c : code),
    eval_static_bindings vars bindings statics = Some statics' ->
    (id, c) ∈ static_code statics' ->
    (id, c) ∈ static_code statics ∨ Bind_code id c ∈ bindings.
Proof.
  induction bindings as [ | [id' c' | id' | s const] bindings IH];
    intros statics statics' id c; simpl.
  - intros [= <-]. auto.
  - destruct (lookup id' (static_code statics)); [ discriminate | ].
    intros Heval Hin. destruct (IH _ _ id c Heval Hin) as [Hold | Hnew].
    + simpl in Hold. destruct Hold as [[= <- <-] | Hold]; auto.
    + auto.
  - intros Heval Hin. destruct (IH _ _ id c Heval Hin); auto.
  - destruct (lookup s (static_values statics)); [ discriminate | ].
    destruct (eval_static_const vars const) as [allocated | ]; simpl;
      [ | discriminate ].
    intros Heval Hin. destruct (IH _ _ id c Heval Hin); auto.
Qed.

(* A code id that a group brings into scope is bound by it or declared
   deleted by it. *)
Lemma declared_code_ids_split (bindings : list static_binding)
    (name : positive) :
  among name (flat_map declared_code_ids_of_binding bindings)
  = among name (flat_map code_ids_of_binding bindings)
    || among name (flat_map deleted_code_ids_of_binding bindings).
Proof.
  unfold among in *.
  induction bindings as [ | [id c | id | s const] bindings IH]; simpl.
  - reflexivity.
  - rewrite IH. apply orb_assoc.
  - rewrite IH.
    destruct (Pos.eqb name id); simpl;
      [ symmetry; apply orb_true_r | reflexivity ].
  - exact IH.
Qed.

(* ======================================================================== *)
(* The invariant of a run, on terms                                         *)
(* ======================================================================== *)

Section Terms.

Variable unit_exn : continuation.

(* The code ids that the unit declares deleted. *)
Variable deleted : list code_id.

(* The code ids in scope in a state: those bound, and those declared
   deleted. *)
Definition ids_of (statics : static_env) : list code_id :=
  bound_code_ids statics ++ deleted.

(* What is asked of a handler that a continuation keeps: its static
   bindings are at toplevel, given whether it is ([top]); what it
   mentions is in scope; the code ids that it declares deleted are those
   of the unit; and it does not bind the exception continuation of the
   unit. *)
Definition handler_fine (top : bool) (syms : list symbol)
    (ids : list code_id) (h : continuation_handler) : Prop :=
  handler_statics_at_unit_toplevel unit_exn top h = true
  ∧ handler_statics_in_scope syms ids h = true
  ∧ all_among (handler_static_names deleted_code_ids_of_binding h) deleted
    = true
  ∧ among unit_exn (handler_bound_continuations h) = false.

Lemma handler_fine_sub (top : bool) (syms syms' : list symbol)
    (ids ids' : list code_id) (h : continuation_handler) :
  sub syms syms' ->
  sub ids ids' ->
  handler_fine top syms ids h ->
  handler_fine top syms' ids' h.
Proof.
  intros Hs Hi (P & S & D & U). refine (conj P (conj _ (conj D U))).
  destruct h as [params metadata handler]. simpl in S |- *.
  exact (expr_scope_sub handler _ _ _ _ Hs Hi S).
Qed.

(* A handler that is not at toplevel binds nothing static. *)
Lemma handler_nowhere_names (names_of : static_binding -> list positive)
    (h : continuation_handler) :
  handler_statics_at_unit_toplevel unit_exn false h = true ->
  handler_static_names names_of h = [].
Proof.
  destruct h as [params metadata handler]. simpl. intros H.
  apply (statics_nowhere unit_exn names_of handler H).
Qed.

Lemma handler_fine_top (syms : list symbol) (ids : list code_id)
    (h : continuation_handler) :
  handler_fine false syms ids h -> handler_fine true syms ids h.
Proof.
  intros (P & S & D & U). refine (conj _ (conj S (conj D U))).
  destruct h as [params metadata handler]. simpl in P |- *.
  apply (statics_nowhere unit_exn symbols_of_binding handler P).
Qed.

(* A continuation is good under a scope: when it is applied in a state
   whose static environment binds that scope, the state that it leads
   to has the invariant. A handler that is at toplevel may bind static
   names, and a continuation that it then uses may mention them; so the
   continuations that a closure keeps need be good only under the scope
   with, added, what the handler of the closure binds. Only the
   continuations that the handlers of the closure can use are asked
   for. *)
Inductive cont_good : list symbol -> list code_id -> cont_value -> Prop :=
  | Good_outer (syms : list symbol) (ids : list code_id) (k : continuation) :
    cont_good syms ids (Outer k)
  | Good_extra (syms : list symbol) (ids : list code_id) (c : cont_value)
      (extra : list value) :
    cont_good syms ids c -> cont_good syms ids (Extra_args c extra)
  | Good_over (syms : list symbol) (ids : list code_id)
      (remaining : list value) (remaining_arity return_arity : arity)
      (return_to exn_to : cont_value) :
    cont_good syms ids return_to ->
    cont_good syms ids exn_to ->
    cont_good syms ids
      (Over_application remaining remaining_arity return_arity return_to
         exn_to)
  | Good_closure (syms : list symbol) (ids : list code_id)
      (h : continuation_handler) (vars : env) (conts : cont_env)
      (recursive : list (continuation * continuation_handler)) :
    handler_fine true syms ids h ->
    Forall (fun kh => handler_fine false syms ids (snd kh)) recursive ->
    (recursive <> [] ->
     handler_statics_at_unit_toplevel unit_exn false h = true) ->
    lookup unit_exn conts = None ->
    among unit_exn (map fst recursive) = false ->
    (∀ k : continuation,
      k ∈ handler_free_continuations h
          ++ flat_map (fun kh => handler_free_continuations (snd kh))
               recursive ->
      among k (map fst recursive) = false ->
      cont_good (handler_static_names symbols_of_binding h ++ syms)
        (handler_static_names declared_code_ids_of_binding h ++ ids)
        (resolve k conts)) ->
    cont_good syms ids (Closure h vars conts recursive).

Lemma cont_good_sub (syms : list symbol) (ids : list code_id)
    (c : cont_value) :
  cont_good syms ids c ->
  ∀ (syms' : list symbol) (ids' : list code_id),
    sub syms syms' -> sub ids ids' -> cont_good syms' ids' c.
Proof.
  induction 1
    as [ syms ids k | syms ids c extra _ IH
       | syms ids remaining remaining_arity return_arity return_to exn_to
           _ IHreturn _ IHexn
       | syms ids h vars conts recursive Hh Hrec Hnowhere Hunit Hnames _ IH];
    intros syms' ids' Hs Hi.
  - apply Good_outer.
  - apply Good_extra, IH; assumption.
  - apply Good_over; [ apply IHreturn | apply IHexn ]; assumption.
  - apply Good_closure; try assumption.
    + exact (handler_fine_sub _ _ _ _ _ _ Hs Hi Hh).
    + revert Hrec. apply Forall_impl. intros kh.
      apply handler_fine_sub; assumption.
    + intros k Hk Hnot. apply (IH k Hk Hnot); apply sub_app_mono; assumption.
Qed.

Lemma with_extra_args_good (syms : list symbol) (ids : list code_id)
    (c : cont_value) (extra : list value) :
  cont_good syms ids c -> cont_good syms ids (with_extra_args c extra).
Proof.
  intros H. destruct extra; simpl; [ exact H | apply Good_extra, H ].
Qed.

(* What is asked of the code that the static environment binds. *)
Definition code_fine (syms : list symbol) (ids : list code_id) (c : code)
    : Prop :=
  code_statics_at_unit_toplevel unit_exn c = true
  ∧ code_statics_in_scope syms ids c = true
  ∧ among unit_exn (code_bound_continuations c) = false.

Lemma code_fine_sub (syms syms' : list symbol) (ids ids' : list code_id)
    (c : code) :
  sub syms syms' -> sub ids ids' -> code_fine syms ids c ->
  code_fine syms' ids' c.
Proof.
  intros Hs Hi (P & S & U). refine (conj P (conj _ U)).
  destruct c as [params my_closure ret exn metadata body].
  simpl in S |- *.
  exact (expr_scope_sub body _ _ _ _ Hs Hi S).
Qed.

Definition code_all_fine (statics : static_env) : Prop :=
  Forall
    (fun binding =>
       code_fine (bound_symbols statics) (ids_of statics) (snd binding))
    (static_code statics).

(* The invariant: the expression of the state has its static bindings
   at toplevel and mentions only what is in scope, which is what the
   static environment binds, with the code ids declared deleted; the
   continuations that it can use are good under that scope with, added,
   what the expression is still to bind; the exception continuation of
   the unit is bound by nothing; and the code bound is fine. *)
Definition terms_inv (s : state) : Prop :=
  statics_at_unit_toplevel unit_exn true (state_expr s) = true
  ∧ expr_statics_in_scope (bound_symbols (state_statics s))
      (ids_of (state_statics s)) (state_expr s)
    = true
  ∧ all_among (static_names deleted_code_ids_of_binding (state_expr s))
      deleted
    = true
  ∧ among unit_exn (bound_continuations (state_expr s)) = false
  ∧ lookup unit_exn (state_conts s) = None
  ∧ (∀ k : continuation,
      k ∈ free_continuations (state_expr s) ->
      cont_good
        (static_names symbols_of_binding (state_expr s)
         ++ bound_symbols (state_statics s))
        (static_names declared_code_ids_of_binding (state_expr s)
         ++ ids_of (state_statics s))
        (resolve k (state_conts s)))
  ∧ code_all_fine (state_statics s).

(* A call of code. *)
Lemma call_code_terms (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity) (return_to exn_to : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) (s' : state) :
  code_all_fine statics ->
  code_fine (bound_symbols statics) (ids_of statics) c ->
  cont_good (bound_symbols statics) (ids_of statics) return_to ->
  cont_good (bound_symbols statics) (ids_of statics) exn_to ->
  call_code closure c vs args_arity return_arity return_to exn_to statics hp rd
  = Step s' ->
  terms_inv s'.
Proof.
  intros Hcode (P & S & U) Hreturn Hexn Hstep.
  apply call_code_step in Hstep as (callee_vars & _ & _ & _ & _ & _ & ->).
  destruct c as [params my_closure ret exn metadata body].
  simpl in P, S, U |- *.
  apply orb_false_iff in U as [Hret U].
  apply orb_false_iff in U as [Hexn_ne U].
  apply Pos.eqb_neq in Hret, Hexn_ne.
  destruct (statics_nowhere unit_exn symbols_of_binding body P)
    as (Ptop & _ & Nsyms).
  destruct (statics_nowhere unit_exn declared_code_ids_of_binding body P)
    as (_ & _ & Nids).
  destruct (statics_nowhere unit_exn deleted_code_ids_of_binding body P)
    as (_ & _ & Ndeleted).
  unfold terms_inv. simpl_state. rewrite Nsyms, Nids, Ndeleted, !app_nil_l.
  refine (conj Ptop (conj _ (conj eq_refl (conj U (conj _ (conj _ Hcode)))))).
  - exact S.
  - rewrite (lookup_other _ _ _ _ Hret), (lookup_other _ _ _ _ Hexn_ne).
    reflexivity.
  - intros k _. unfold resolve. rewrite !lookup_cons.
    destruct (Pos.eqb k ret); [ exact Hreturn | ].
    destruct (Pos.eqb k exn); [ exact Hexn | apply Good_outer ].
Qed.

Lemma run_call_terms (return_partial : value -> step_result)
    (kind : call_kind) (shape : option call_shape) (return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
      (rd : read_record)
      (s' : state) :
  code_all_fine statics ->
  shape_calls statics kind shape ->
  cont_good (bound_symbols statics) (ids_of statics) return_to ->
  cont_good (bound_symbols statics) (ids_of statics) exn_to ->
  (∀ partial : value, return_partial partial = Step s' -> terms_inv s') ->
  run_call return_partial shape return_arity return_to exn_to statics hp rd
  = Step s' ->
  terms_inv s'.
Proof.
  intros Hcode Hcalls Hreturn Hexn Hpartial.
  assert (Hfine :
    ∀ (called : code_id) (c : code),
      lookup called (static_code statics) = Some c ->
      code_fine (bound_symbols statics) (ids_of statics) c).
  { intros called c Hlookup. apply lookup_in in Hlookup.
    unfold code_all_fine in Hcode. rewrite Forall_forall in Hcode.
    exact (Hcode _ Hlookup). }
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ];
    simpl in Hcalls |- *.
  - destruct Hcalls as (called & slots & _ & Hlookup & _).
    exact (call_code_terms closure c vs _ _ _ _ statics _ rd_ s' Hcode
             (Hfine called c Hlookup) Hreturn Hexn).
  - destruct Hcalls as (called & slots & _ & Hlookup & _).
    apply (call_code_terms closure c vs _ _ _ _ statics _ rd_ s' Hcode
             (Hfine called c Hlookup));
      [ apply Good_over; assumption | exact Hexn ].
  - apply Hpartial.
  - discriminate.
Qed.

(* A sibling of a recursive handler: its closure is good where the
   closure of the handler is. *)
Lemma sibling_good (syms : list symbol) (ids : list code_id)
    (h : continuation_handler) (vars : env) (conts : cont_env)
    (recursive : list (continuation * continuation_handler))
    (k : continuation) (h' : continuation_handler) :
  cont_good syms ids (Closure h vars conts recursive) ->
  lookup k recursive = Some h' ->
  cont_good syms ids (Closure h' vars conts recursive).
Proof.
  intros Hgood Hlookup.
  inversion Hgood
    as [ | | | syms0 ids0 h0 vars0 conts0 recursive0 Hh Hrec Hnowhere Hunit
             Hnames Hconts];
    subst.
  assert (Hne : recursive <> []).
  { intros ->. discriminate Hlookup. }
  assert (Hin : (k, h') ∈ recursive) by exact (lookup_in _ _ _ Hlookup).
  pose proof (proj1 (Forall_forall _ _) Hrec _ Hin) as Hh'. simpl in Hh'.
  apply Good_closure; try assumption.
  - apply handler_fine_top, Hh'.
  - intros _. exact (proj1 Hh').
  - intros k' Hk' Hnot.
    assert (Hin' :
      k' ∈ handler_free_continuations h
           ++ flat_map (fun kh => handler_free_continuations (snd kh))
                recursive).
    { apply in_or_app. right. apply in_app_or in Hk' as [Hk' | Hk'];
        [ | exact Hk' ].
      apply in_flat_map. exists (k, h'). auto. }
    pose proof (Hconts k' Hin' Hnot) as Hgood'.
    rewrite (handler_nowhere_names symbols_of_binding h (Hnowhere Hne)),
      (handler_nowhere_names declared_code_ids_of_binding h (Hnowhere Hne))
      in Hgood'.
    rewrite (handler_nowhere_names symbols_of_binding h' (proj1 Hh')),
      (handler_nowhere_names declared_code_ids_of_binding h' (proj1 Hh')).
    exact Hgood'.
Qed.

(* The application of a good continuation. *)
Lemma apply_cont_value_terms (c : cont_value) :
  ∀ (vs : list value) (statics : static_env) (hp : heap) (rd : read_record)
    (s' : state),
    code_all_fine statics ->
    cont_good (bound_symbols statics) (ids_of statics) c ->
    apply_cont_value c vs statics hp rd = Step s' ->
    terms_inv s'.
Proof.
  induction c
    as [[params metadata handler] def_vars def_conts recursive | k
       | c IHc extra
       | remaining remaining_arity return_arity return_to exn_to IHreturn _]
    using cont_value_over_ind;
    intros vs statics hp rd s' Hcode Hgood Hstep.
  - simpl in Hstep.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | discriminate ].
    destruct (bind_params params vs def_vars); [ | discriminate ].
    injection Hstep as <-.
    inversion Hgood
      as [ | | | syms0 ids0 h0 vars0 conts0 recursive0 Hh Hrec Hnowhere Hunit
               Hnames Hconts];
      subst.
    destruct Hh as (P & S & D & U). simpl in P, S, D, U.
    unfold terms_inv. simpl_state.
    refine (conj P (conj S (conj D (conj U (conj _ (conj _ Hcode)))))).
    + unfold rec_closures.
      rewrite lookup_rec_closures_aux.
      destruct (lookup unit_exn recursive) as [h' | ] eqn:Hlookup;
        [ | exact Hunit ].
      apply lookup_in in Hlookup.
      assert (Hin : unit_exn ∈ map fst recursive).
      { apply in_map_iff. exists (unit_exn, h'). auto. }
      apply among_In in Hin. rewrite Hin in Hnames. discriminate Hnames.
    + intros k Hk. rewrite resolve_rec_closures.
      destruct (lookup k recursive) as [h' | ] eqn:Hlookup.
      * refine (cont_good_sub _ _ _
                  (sibling_good _ _ _ _ _ _ k h' Hgood Hlookup) _ _ _ _);
          apply sub_app_r.
      * apply Hconts; [ apply in_or_app; left; exact Hk | ].
        destruct (among k (map fst recursive)) eqn:Hamong; [ | reflexivity ].
        apply among_In in Hamong.
        exfalso. exact (lookup_none_not_in _ _ Hlookup Hamong).
  - discriminate.
  - inversion Hgood; subst.
    exact (IHc (vs ++ extra) statics hp rd_ s' Hcode ltac:(assumption) Hstep).
  - rewrite apply_cont_value_over_application in Hstep.
    destruct vs as [ | v [ | v' vs]]; [ discriminate | | discriminate ].
    inversion Hgood as [ | | syms0 ids0 r ra rta rt et Hreturn Hexn | ]; subst.
    unfold call_function in Hstep.
    apply (run_call_terms _ Indirect_unknown_arity _ _ _ _ statics _ rd_ s'
             Hcode (shape_of_call_calls _ _ _ _ _ _) Hreturn Hexn)
      in Hstep; [ exact Hstep | ].
    intros partial Hpartial.
    exact (IHreturn [partial] statics hp rd_ s' Hcode Hreturn Hpartial).
Qed.

Lemma call_function_terms (callee : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
      (rd : read_record)
      (s' : state) :
  code_all_fine statics ->
  cont_good (bound_symbols statics) (ids_of statics) return_to ->
  cont_good (bound_symbols statics) (ids_of statics) exn_to ->
  call_function callee kind vs args_arity return_arity return_to exn_to
    statics hp rd
  = Step s' ->
  terms_inv s'.
Proof.
  intros Hcode Hreturn Hexn. unfold call_function.
  apply (run_call_terms _ kind _ _ _ _ statics _ rd_ s' Hcode
           (shape_of_call_calls _ _ _ _ _ _) Hreturn Hexn).
  intros partial.
  exact (apply_cont_value_terms return_to _ statics _ rd_ s' Hcode Hreturn).
Qed.

(* The unfoldings of the checks at one constructor, and nothing else. *)
Local Ltac at_constructor H :=
  cbn [statics_at_unit_toplevel handler_statics_at_unit_toplevel
       binding_statics_at_unit_toplevel code_statics_at_unit_toplevel
       expr_statics_in_scope handler_statics_in_scope
       binding_statics_in_scope code_statics_in_scope
       static_names handler_static_names
       free_continuations handler_free_continuations
       bound_continuations handler_bound_continuations
       binding_bound_continuations code_bound_continuations
       andb app] in H.

(* A step keeps the invariant. *)
Lemma terms_inv_step (s s' : state) :
  terms_inv s -> step s = Step s' -> terms_inv s'.
Proof.
  intros (P & S & D & U & L & K & C) Hstep. rewrite (state_eta s) in Hstep.
  set (vars := state_vars s) in *. set (conts := state_conts s) in *.
  set (statics := state_statics s) in *.
  destruct (state_expr s)
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    autorewrite with step in Hstep.
  - (* Let *)
    destruct (exec_named _ _ rd_ _ d) as [[[v hp'] rd'] | ]; [ | discriminate ].
    destruct (kind_eqb (kind_of_value v) kind); [ | discriminate ].
    injection Hstep as <-. unfold terms_inv. simpl_state.
    at_constructor P. at_constructor S. at_constructor D. at_constructor U.
    at_constructor K.
    apply andb_true_iff in S as [_ S].
    exact (conj P (conj S (conj D (conj U (conj L (conj K C)))))).
  - (* Let_static *)
    destruct (eval_static_bindings vars bindings statics) as [statics' | ]
      eqn:Heval; [ | discriminate ].
    injection Hstep as <-. unfold terms_inv. simpl_state.
    at_constructor P. at_constructor S. at_constructor D. at_constructor U.
    at_constructor K.
    apply andb_true_iff in P as [Pbindings Pbody].
    apply andb_true_iff in S as [S Sbody].
    apply andb_true_iff in S as [_ Scode].
    rewrite all_among_app in D. apply andb_true_iff in D as [Dgroup Dbody].
    apply among_false_app in U as [Ugroup Ubody].
    assert (Hsyms :
      sub (flat_map symbols_of_binding bindings ++ bound_symbols statics)
        (bound_symbols statics')).
    { intros name Hn.
      rewrite
        (proj1 (eval_static_bindings_bound vars bindings statics statics'
                  name Heval)).
      rewrite among_app in Hn. exact Hn. }
    assert (Hids :
      sub (flat_map declared_code_ids_of_binding bindings ++ ids_of statics)
        (ids_of statics')).
    { intros name Hn. unfold ids_of in Hn |- *.
      rewrite !among_app, declared_code_ids_split in Hn.
      rewrite among_app,
        (proj2 (eval_static_bindings_bound vars bindings statics statics'
                  name Heval)).
      apply orb_true_iff in Hn as [Hn | Hn];
        apply orb_true_iff in Hn as [Hn | Hn].
      - rewrite Hn. reflexivity.
      - rewrite (proj1 (all_among_spec _ _) Dgroup name Hn).
        apply orb_true_r.
      - rewrite Hn, orb_true_r. reflexivity.
      - rewrite Hn. apply orb_true_r. }
    refine (conj Pbody (conj _ (conj Dbody (conj Ubody (conj L (conj _ _)))))).
    + exact (expr_scope_sub body _ _ _ _ Hsyms Hids Sbody).
    + intros k Hk. refine (cont_good_sub _ _ _ (K k Hk) _ _ _ _).
      * apply sub_app; [ apply sub_app | ].
        -- exact (sub_trans _ _ _ (sub_trans _ _ _ (sub_app_l _ _) Hsyms)
                    (sub_app_r _ _)).
        -- apply sub_app_l.
        -- exact (sub_trans _ _ _ (sub_trans _ _ _ (sub_app_r _ _) Hsyms)
                    (sub_app_r _ _)).
      * apply sub_app; [ apply sub_app | ].
        -- exact (sub_trans _ _ _ (sub_trans _ _ _ (sub_app_l _ _) Hids)
                    (sub_app_r _ _)).
        -- apply sub_app_l.
        -- exact (sub_trans _ _ _ (sub_trans _ _ _ (sub_app_r _ _) Hids)
                    (sub_app_r _ _)).
    + unfold code_all_fine. apply Forall_forall. intros [id c] Hin.
      cbn [snd].
      destruct (eval_static_bindings_code vars bindings statics statics' id c
                  Heval Hin) as [Hold | Hnew].
      * pose proof (proj1 (Forall_forall _ _) C _ Hold) as Hfine.
        cbn [snd] in Hfine.
        refine (code_fine_sub _ _ _ _ _ _ _ Hfine).
        -- exact (sub_trans _ _ _ (sub_app_r _ _) Hsyms).
        -- exact (sub_trans _ _ _ (sub_app_r _ _) Hids).
      * pose proof (proj1 (forallb_forall _ _) Pbindings _ Hnew) as Pc.
        pose proof (proj1 (forallb_forall _ _) Scode _ Hnew) as Sc.
        pose proof (among_flat_map_false _ _ _ _ Ugroup Hnew) as Uc.
        cbn [binding_statics_at_unit_toplevel binding_statics_in_scope
             binding_bound_continuations] in Pc, Sc, Uc.
        exact (code_fine_sub _ _ _ _ _ Hsyms Hids (conj Pc (conj Sc Uc))).
  - (* Let_cont_non_recursive *)
    injection Hstep as <-. unfold terms_inv. simpl_state.
    destruct h as [params metadata handler].
    at_constructor P. at_constructor S. at_constructor D. at_constructor U.
    at_constructor K.
    apply andb_true_iff in P as [Ph Pbody].
    apply andb_true_iff in S as [Sh Sbody].
    rewrite all_among_app in D. apply andb_true_iff in D as [Dh Dbody].
    apply among_false_cons in U as [Hk U].
    apply among_false_app in U as [Uh Ubody].
    pose proof (to_toplevel unit_exn symbols_of_binding _ handler Ph)
      as Phtop.
    refine
      (conj Pbody (conj Sbody (conj Dbody (conj Ubody (conj _ (conj _ C)))))).
    + rewrite (lookup_other _ _ _ _ Hk). exact L.
    + intros k' Hk'. destruct (Pos.eqb_spec k' k) as [-> | Hne].
      * rewrite resolve_head. apply Good_closure.
        -- exact (conj Phtop (conj Sh (conj Dh Uh))).
        -- apply Forall_nil.
        -- intros Hnil. destruct (Hnil eq_refl).
        -- exact L.
        -- reflexivity.
        -- intros k'' Hk'' _. cbn [flat_map] in Hk''.
           rewrite app_nil_r in Hk''.
           cbn [handler_static_names].
           refine (cont_good_sub _ _ _ (K k'' _) _ _ _ _).
           ++ apply in_or_app. left. exact Hk''.
           ++ rewrite <- app_assoc. apply sub_refl.
           ++ rewrite <- app_assoc. apply sub_refl.
      * rewrite (resolve_other _ _ _ _ Hne).
        assert (Hin :
          k' ∈ free_continuations handler
               ++ without [k] (free_continuations body)).
        { apply in_or_app. right. apply among_In.
          rewrite FlambdaMapProofs.among_without.
          apply andb_true_iff. split; [ apply among_In, Hk' | ].
          unfold among. cbn [existsb].
          rewrite (proj2 (Pos.eqb_neq k' k) Hne). reflexivity. }
        pose proof (K k' Hin) as Hgood.
        destruct (handler_at_unit_toplevel unit_exn k
                    (Handler params metadata handler) body) eqn:Hat.
        -- unfold handler_at_unit_toplevel in Hat.
           apply andb_true_iff in Hat as [_ Hat].
           pose proof
             (proj1 (all_among_spec _ _) Hat k' (proj2 (among_In _ _) Hk'))
             as Hk2.
           unfold among in Hk2. cbn [existsb] in Hk2.
           rewrite (proj2 (Pos.eqb_neq k' k) Hne), orb_false_r in Hk2.
           cbn [orb] in Hk2. apply Pos.eqb_eq in Hk2. subst k'.
           unfold resolve. rewrite L. apply Good_outer.
        -- destruct (statics_nowhere unit_exn symbols_of_binding handler Ph)
             as (_ & _ & Ns).
           destruct
             (statics_nowhere unit_exn declared_code_ids_of_binding handler
                Ph)
             as (_ & _ & Ni).
           rewrite Ns, Ni, !app_nil_l in Hgood. exact Hgood.
  - (* Let_cont_recursive *)
    destruct (handlers_bind_statics handlers); [ discriminate | ].
    injection Hstep as <-. unfold terms_inv. simpl_state.
    at_constructor P. at_constructor S. at_constructor D. at_constructor U.
    at_constructor K.
    apply andb_true_iff in P as [Phandlers Pbody].
    apply andb_true_iff in S as [Shandlers Sbody].
    destruct (handlers_nowhere_facts unit_exn deleted_code_ids_of_binding
                handlers Phandlers) as [_ Nd].
    destruct (handlers_nowhere_facts unit_exn symbols_of_binding handlers
                Phandlers) as [_ Ns].
    destruct (handlers_nowhere_facts unit_exn declared_code_ids_of_binding
                handlers Phandlers) as [_ Ni].
    rewrite Nd, app_nil_l in D.
    apply among_false_app in U as [Unames U].
    apply among_false_app in U as [Uhandlers Ubody].
    assert (Hfine :
      Forall
        (fun kh =>
           handler_fine false
             (static_names symbols_of_binding body ++ bound_symbols statics)
             (static_names declared_code_ids_of_binding body
              ++ ids_of statics)
             (snd kh))
        handlers).
    { apply Forall_forall. intros [k' h'] Hin. cbn [snd].
      pose proof (proj1 (forallb_forall _ _) Phandlers _ Hin) as Ph'.
      pose proof (proj1 (forallb_forall _ _) Shandlers _ Hin) as Sh'.
      pose proof (among_flat_map_false _ _ _ _ Uhandlers Hin) as Uh'.
      cbn beta iota in Ph', Sh', Uh'.
      refine (conj Ph' (conj _ (conj _ Uh'))).
      - exact Sh'.
      - rewrite (handler_nowhere_names deleted_code_ids_of_binding h' Ph').
        reflexivity. }
    refine (conj Pbody (conj Sbody (conj D (conj Ubody (conj _ (conj _ C)))))).
    + rewrite lookup_rec_closures_app.
      destruct (lookup unit_exn handlers) as [h' | ] eqn:Hlookup;
        [ | exact L ].
      apply lookup_in in Hlookup.
      assert (Hin : unit_exn ∈ map fst handlers).
      { apply in_map_iff. exists (unit_exn, h'). auto. }
      apply among_In in Hin. rewrite Hin in Unames. discriminate Unames.
    + intros k Hk. rewrite resolve_rec_closures.
      destruct (lookup k handlers) as [h' | ] eqn:Hlookup.
      * pose proof (proj1 (Forall_forall _ _) Hfine _ (lookup_in _ _ _ Hlookup))
          as Hh'.
        cbn [snd] in Hh'.
        apply Good_closure.
        -- apply handler_fine_top. exact Hh'.
        -- exact Hfine.
        -- intros _. exact (proj1 Hh').
        -- exact L.
        -- exact Unames.
        -- intros k'' Hk'' Hnot.
           rewrite (handler_nowhere_names symbols_of_binding h' (proj1 Hh')),
             (handler_nowhere_names declared_code_ids_of_binding h'
                (proj1 Hh')),
             !app_nil_l.
           rewrite Ns, Ni, !app_nil_l in K. apply K.
           apply among_In. rewrite FlambdaMapProofs.among_without, Hnot.
           rewrite andb_true_r. apply among_In, in_or_app. left.
           apply in_app_or in Hk'' as [Hk'' | Hk''].
           ++ apply in_flat_map. exists (k, h').
              split; [ exact (lookup_in _ _ _ Hlookup) | exact Hk'' ].
           ++ apply in_flat_map in Hk'' as ([kx hx] & Hx & Hk'').
              apply in_flat_map. exists (kx, hx). auto.
      * rewrite Ns, Ni, !app_nil_l in K. apply K.
        apply among_In. rewrite FlambdaMapProofs.among_without.
        apply andb_true_iff. split.
        -- apply among_In, in_or_app. right. exact Hk.
        -- destruct (among k (map fst handlers)) eqn:Hamong; [ | reflexivity ].
           apply among_In in Hamong.
           exfalso. exact (lookup_none_not_in _ _ Hlookup Hamong).
  - (* Apply *)
    destruct (eval_simple vars (ap_callee ap)) as [callee | ];
      [ | discriminate ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | discriminate ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
      [ | discriminate ].
    at_constructor K.
    apply (call_function_terms _ _ _ _ _ _ _ statics _ rd_ s' C) in Hstep;
      [ exact Hstep | | ].
    + apply K. left. reflexivity.
    + apply with_extra_args_good, K. right. left. reflexivity.
  - (* Apply_cont *)
    destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | discriminate ].
    unfold apply_continuation in Hstep. at_constructor K.
    refine (apply_cont_value_terms _ vs statics _ rd_ s' C (K _ _) Hstep).
    left. reflexivity.
  - (* Switch *)
    destruct (eval_simple vars scrutinee) as [v | ]; [ | discriminate ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | discriminate ].
    destruct (find_arm i arms) as [ac | ] eqn:Harm; [ | discriminate ].
    injection Hstep as <-. unfold terms_inv. simpl_state.
    apply find_arm_in in Harm.
    at_constructor S. at_constructor K.
    apply andb_true_iff in S as [_ Sarms].
    refine
      (conj eq_refl (conj _ (conj eq_refl (conj eq_refl (conj L (conj _ C)))))).
    + exact (proj1 (forallb_forall _ _) Sarms _ Harm).
    + intros k Hk. apply K. apply in_flat_map. exists (i, ac).
      split; [ exact Harm | exact Hk ].
  - discriminate.
Qed.

Lemma terms_inv_run (n : nat) :
  ∀ s s' : state, terms_inv s -> run n s = Step s' -> terms_inv s'.
Proof.
  induction n as [ | n IH]; intros s s' Hinv; simpl.
  - intros [= <-]. exact Hinv.
  - destruct (step s) as [next | k vs | ] eqn:Hstep; try discriminate.
    exact (IH next s' (terms_inv_step s next Hinv Hstep)).
Qed.

End Terms.

(* ======================================================================== *)
(* Values under a larger scope                                              *)
(* ======================================================================== *)

Fixpoint value_among_sub (syms syms' : list symbol) (ids ids' : list code_id)
    (Hs : sub syms syms') (Hi : sub ids ids') (v : value) {struct v} :
  value_statics_among syms ids v = true ->
  value_statics_among syms' ids' v = true.
Proof.
  destruct v
    as [c | code slots | slot decls slots | s | tag fields | elements | c
       | l | callee args arity];
    simpl; auto.
  - rewrite !andb_true_iff. intros [Hc Hslots].
    split; [ exact (Hi code Hc) | ].
    induction slots as [ | [slot v] slots IH]; simpl in Hslots |- *;
      [ reflexivity | ].
    apply andb_true_iff in Hslots as [Hv Hrest]. apply andb_true_iff.
    split; [ exact (value_among_sub _ _ _ _ Hs Hi v Hv) | exact (IH Hrest) ].
  - rewrite !andb_true_iff. intros [Hc Hslots].
    split; [ exact (all_among_sub _ _ _ Hi Hc) | ].
    induction slots as [ | [slot' v] slots IH]; simpl in Hslots |- *;
      [ reflexivity | ].
    apply andb_true_iff in Hslots as [Hv Hrest]. apply andb_true_iff.
    split; [ exact (value_among_sub _ _ _ _ Hs Hi v Hv) | exact (IH Hrest) ].
  - intros Hfields.
    induction fields as [ | v fields IH]; simpl in Hfields |- *;
      [ reflexivity | ].
    apply andb_true_iff in Hfields as [Hv Hrest]. apply andb_true_iff.
    split; [ exact (value_among_sub _ _ _ _ Hs Hi v Hv) | exact (IH Hrest) ].
  - intros Helements.
    induction elements as [ | v elements IH]; simpl in Helements |- *;
      [ reflexivity | ].
    apply andb_true_iff in Helements as [Hv Hrest]. apply andb_true_iff.
    split; [ exact (value_among_sub _ _ _ _ Hs Hi v Hv) | exact (IH Hrest) ].
  - rewrite !andb_true_iff. intros [Hcallee Hargs].
    split; [ exact (value_among_sub _ _ _ _ Hs Hi callee Hcallee) | ].
    induction args as [ | v args IH]; simpl in Hargs |- *; [ reflexivity | ].
    apply andb_true_iff in Hargs as [Hv Hrest]. apply andb_true_iff.
    split; [ exact (value_among_sub _ _ _ _ Hs Hi v Hv) | exact (IH Hrest) ].
Qed.

Lemma slots_among_sub (syms syms' : list symbol) (ids ids' : list code_id)
    (slots : list (value_slot * value)) :
  sub syms syms' ->
  sub ids ids' ->
  forallb (fun '(_, v) => value_statics_among syms ids v) slots = true ->
  forallb (fun '(_, v) => value_statics_among syms' ids' v) slots = true.
Proof.
  intros Hs Hi. rewrite !forallb_forall. intros H [slot v] Hin.
  exact (value_among_sub _ _ _ _ Hs Hi v (H (slot, v) Hin)).
Qed.

Lemma values_among_sub (syms syms' : list symbol) (ids ids' : list code_id)
    (vs : list value) :
  sub syms syms' ->
  sub ids ids' ->
  forallb (value_statics_among syms ids) vs = true ->
  forallb (value_statics_among syms' ids') vs = true.
Proof.
  intros Hs Hi. rewrite !forallb_forall. intros H v Hin.
  exact (value_among_sub _ _ _ _ Hs Hi v (H v Hin)).
Qed.

Lemma allocated_among_sub (syms syms' : list symbol) (ids ids' : list code_id)
    (allocated : statically_allocated_value) :
  sub syms syms' ->
  sub ids ids' ->
  allocated_statics_among syms ids allocated = true ->
  allocated_statics_among syms' ids' allocated = true.
Proof.
  intros Hs Hi.
  destruct allocated
    as [code slots | slot symbols decls slots | tag fields | contents
       | bytes | elements | c];
    simpl; rewrite ?andb_true_iff.
  - intros [Hc Hslots]. split; [ exact (Hi code Hc) | ].
    exact (slots_among_sub _ _ _ _ _ Hs Hi Hslots).
  - intros [[Hsyms Hc] Hslots].
    split; [ split | exact (slots_among_sub _ _ _ _ _ Hs Hi Hslots) ].
    + exact (all_among_sub _ _ _ Hs Hsyms).
    + exact (all_among_sub _ _ _ Hi Hc).
  - apply values_among_sub; assumption.
  - apply value_among_sub; assumption.
  - auto.
  - apply values_among_sub; assumption.
  - auto.
Qed.

(* ======================================================================== *)
(* What evaluation makes                                                    *)
(* ======================================================================== *)

Section Evaluation.

Variable syms : list symbol.
Variable ids : list code_id.

Let P (v : value) : Prop := value_statics_among syms ids v = true.

Lemma eval_simple_among (vars : env) (s : simple) (v : value) :
  env_values_all P vars ->
  simple_statics_in_scope syms s = true ->
  eval_simple vars s = Some v ->
  P v.
Proof.
  intros Hvars. destruct s as [x | sym | c]; simpl.
  - intros _ Hlookup. exact (env_values_all_lookup P vars x v Hvars Hlookup).
  - intros Hsym [= <-]. exact Hsym.
  - intros _ [= <-]. reflexivity.
Qed.

Lemma eval_simples_among (vars : env) (args : list simple) :
  env_values_all P vars ->
  ∀ vs : list value,
    simples_statics_in_scope syms args = true ->
    eval_simples vars args = Some vs ->
    forallb (value_statics_among syms ids) vs = true.
Proof.
  intros Hvars. unfold simples_statics_in_scope.
  induction args as [ | s args IH]; intros vs; simpl.
  - intros _ [= <-]. reflexivity.
  - intros Hav. apply andb_true_iff in Hav as [Hs Hargs].
    destruct (eval_simple vars s) as [v | ] eqn:Heval; [ | discriminate ].
    destruct (eval_simples vars args) as [rest | ]; [ | discriminate ].
    simpl. intros [= <-]. simpl.
    rewrite (eval_simple_among vars s v Hvars Hs Heval), (IH rest Hargs);
      reflexivity.
Qed.

Lemma eval_value_slots_among (vars : env)
    (slots : list (value_slot * simple)) :
  env_values_all P vars ->
  ∀ evaluated : list (value_slot * value),
    value_slots_statics_in_scope syms slots = true ->
    eval_value_slots vars slots = Some evaluated ->
    forallb (fun '(_, v) => value_statics_among syms ids v) evaluated = true.
Proof.
  intros Hvars. unfold value_slots_statics_in_scope.
  induction slots as [ | [slot s] slots IH]; intros evaluated; simpl.
  - intros _ [= <-]. reflexivity.
  - intros Hav. apply andb_true_iff in Hav as [Hs Hslots].
    destruct (eval_simple vars s) as [v | ] eqn:Heval; [ | discriminate ].
    destruct (eval_value_slots vars slots) as [rest | ]; [ | discriminate ].
    simpl. intros [= <-]. simpl.
    rewrite (eval_simple_among vars s v Hvars Hs Heval), (IH rest Hslots);
      reflexivity.
Qed.

(* What the static environment binds its symbols to mentions only what
   is in scope. *)
Definition statics_among (statics : static_env) : Prop :=
  Forall
    (fun binding => allocated_statics_among syms ids (snd binding) = true)
    (static_values statics).

Lemma statics_among_lookup (statics : static_env) (s : symbol)
    (allocated : statically_allocated_value) :
  statics_among statics ->
  lookup s (static_values statics) = Some allocated ->
  allocated_statics_among syms ids allocated = true.
Proof.
  intros Hstatics Hlookup. apply lookup_in in Hlookup.
  exact (proj1 (Forall_forall _ _) Hstatics _ Hlookup).
Qed.

Lemma lookup_decl_among (slot : function_slot)
    (decls : list (function_slot * code_id)) (code : code_id) :
  all_among (map snd decls) ids = true ->
  lookup slot decls = Some code ->
  among code ids = true.
Proof.
  intros Hall Hlookup. apply (proj1 (all_among_spec _ _) Hall).
  apply among_In, in_map_iff. exists (slot, code).
  split; [ reflexivity | exact (lookup_in _ _ _ Hlookup) ].
Qed.

Lemma closure_in_set_among (statics : static_env) (v : value)
    (slot : function_slot) (decls : list (function_slot * code_id))
    (slots : list (value_slot * value)) :
  statics_among statics ->
  P v ->
  closure_in_set statics v = Some (slot, decls, slots) ->
  all_among (map snd decls) ids = true
  ∧ forallb (fun '(_, v) => value_statics_among syms ids v) slots = true.
Proof.
  intros Hstatics Hv. unfold P in Hv.
  destruct v
    as [c | code vslots | vslot vdecls vslots | s | tag fields
       | elements | c | l | callee args arity];
    simpl in Hv |- *; try discriminate.
  - intros [= <- <- <-]. apply andb_true_iff in Hv as [Hc Hslots].
    split; [ | exact Hslots ]. simpl. rewrite Hc. reflexivity.
  - destruct (lookup vslot vdecls); [ | discriminate ].
    simpl. intros [= <- <- <-]. apply andb_true_iff in Hv. exact Hv.
  - destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hlookup;
      [ | discriminate ].
    pose proof (statics_among_lookup statics s allocated Hstatics Hlookup)
      as Hallocated.
    destruct allocated
      as [code aslots | aslot symbols adecls aslots | tag fields | contents
         | bytes | elements | ];
      simpl in Hallocated; try discriminate.
    + intros [= <- <- <-]. apply andb_true_iff in Hallocated as [Hc Hslots].
      split; [ | exact Hslots ]. simpl. rewrite Hc. reflexivity.
    + destruct (lookup aslot adecls); [ | discriminate ].
      simpl. intros [= <- <- <-].
      apply andb_true_iff in Hallocated as [Hrest Hslots].
      apply andb_true_iff in Hrest as [_ Hc]. auto.
Qed.

Lemma closure_of_among (statics : static_env) (v : value) (code : code_id)
    (slots : list (value_slot * value)) :
  statics_among statics ->
  P v ->
  closure_of statics v = Some (code, slots) ->
  among code ids = true
  ∧ forallb (fun '(_, v) => value_statics_among syms ids v) slots = true.
Proof.
  intros Hstatics Hv Hclosure.
  apply (proj1 (closure_in_set_of statics v code slots)) in Hclosure
    as (slot & decls & Hset & Hlookup).
  destruct (closure_in_set_among statics v slot decls slots Hstatics Hv Hset)
    as [Hdecls Hslots].
  split; [ exact (lookup_decl_among slot decls code Hdecls Hlookup) | ].
  exact Hslots.
Qed.

Lemma lookup_slot_among (slot : value_slot)
    (slots : list (value_slot * value)) (v : value) :
  forallb (fun '(_, v) => value_statics_among syms ids v) slots = true ->
  lookup slot slots = Some v ->
  P v.
Proof.
  intros Hslots Hlookup. apply lookup_in in Hlookup.
  exact (proj1 (forallb_forall _ _) Hslots (slot, v) Hlookup).
Qed.

Lemma project_function_slot_among (statics : static_env)
    (move_from move_to : function_slot) (v w : value) :
  statics_among statics ->
  P v ->
  project_function_slot statics move_from move_to v = Some w ->
  P w.
Proof.
  intros Hstatics Hv. unfold P in Hv |- *. unfold project_function_slot.
  assert (Hunnamed :
    (if Pos.eqb move_from unnamed_function_slot
     then if Pos.eqb move_to unnamed_function_slot then Some v else None
     else None)
    = Some w ->
    value_statics_among syms ids w = true).
  { destruct (Pos.eqb move_from unnamed_function_slot); [ | discriminate ].
    destruct (Pos.eqb move_to unnamed_function_slot); [ | discriminate ].
    intros [= <-]. exact Hv. }
  destruct v
    as [c | code vslots | vslot vdecls vslots | s | tag fields
       | elements | c | l | callee args arity];
    try discriminate.
  - exact Hunnamed.
  - destruct (Pos.eqb vslot move_from); [ | discriminate ].
    destruct (lookup vslot vdecls); [ | discriminate ].
    destruct (lookup move_to vdecls); [ | discriminate ].
    simpl. intros [= <-]. exact Hv.
  - destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hlookup;
      [ | discriminate ].
    pose proof (statics_among_lookup statics s allocated Hstatics Hlookup)
      as Hallocated.
    destruct allocated
      as [code aslots | aslot symbols adecls aslots | tag fields | contents
         | bytes | elements | ];
      try discriminate.
    + exact Hunnamed.
    + destruct (Pos.eqb aslot move_from); [ | discriminate ].
      destruct (lookup aslot adecls); [ | discriminate ].
      destruct (lookup move_to adecls); [ | discriminate ].
      destruct (lookup move_to symbols) as [s' | ] eqn:Hsym;
        [ | discriminate ].
      simpl. intros [= <-]. simpl. simpl in Hallocated.
      apply andb_true_iff in Hallocated as [Hrest _].
      apply andb_true_iff in Hrest as [Hsymbols _].
      apply (proj1 (all_among_spec _ _) Hsymbols).
      apply among_In, in_map_iff. exists (move_to, s').
      split; [ reflexivity | exact (lookup_in _ _ _ Hsym) ].
Qed.

Lemma eval_prim_among (statics : static_env) (p : prim) (vs : list value)
    (v : value) :
  statics_among statics ->
  forallb (value_statics_among syms ids) vs = true ->
  eval_prim statics p vs = Some v ->
  P v.
Proof.
  intros Hstatics Hvs.
  apply (eval_prim_preserves P).
  - intros c. split; reflexivity.
  - intros tag fields. unfold P. simpl. apply forallb_forall.
  - intros elements. unfold P. simpl. apply forallb_forall.
  - intros s tag fields _ Hlookup.
    pose proof (statics_among_lookup statics s _ Hstatics Hlookup)
      as Hfields.
    simpl in Hfields. apply forallb_forall. exact Hfields.
  - intros s elements _ Hlookup.
    pose proof (statics_among_lookup statics s _ Hstatics Hlookup)
      as Helements.
    simpl in Helements. apply forallb_forall. exact Helements.
  - apply forallb_forall. exact Hvs.
Qed.

Lemma eval_named_among (statics : static_env) (vars : env) (d : named)
    (v : value) :
  statics_among statics ->
  env_values_all P vars ->
  named_statics_in_scope syms ids d = true ->
  eval_named statics vars d = Some v ->
  P v.
Proof.
  intros Hstatics Hvars.
  destruct d as [s | p args | code slots | slot decls slots | slot closure
                | from slot closure | from to closure];
    simpl.
  - apply eval_simple_among, Hvars.
  - intros Hargs.
    destruct (eval_simples vars args) as [vs | ] eqn:Heval; [ | discriminate ].
    simpl.
    exact (eval_prim_among statics p vs v Hstatics
             (eval_simples_among vars args Hvars vs Hargs Heval)).
  - intros H. apply andb_true_iff in H as [Hc Hslots].
    destruct (eval_value_slots vars slots) as [evaluated | ] eqn:Heval;
      [ | discriminate ].
    simpl. intros [= <-]. unfold P. simpl. rewrite Hc.
    exact (eval_value_slots_among vars slots Hvars evaluated Hslots Heval).
  - intros H. apply andb_true_iff in H as [Hc Hslots].
    destruct (lookup slot decls); [ | discriminate ]. simpl.
    destruct (eval_value_slots vars slots) as [evaluated | ] eqn:Heval;
      [ | discriminate ].
    simpl. intros [= <-]. unfold P. simpl. rewrite Hc.
    exact (eval_value_slots_among vars slots Hvars evaluated Hslots Heval).
  - intros Hclosure.
    destruct (eval_simple vars closure) as [w | ] eqn:Heval;
      [ | discriminate ].
    simpl.
    destruct (closure_of statics w) as [[code wslots] | ] eqn:Hof;
      [ | discriminate ].
    simpl.
    destruct (closure_of_among statics w code wslots Hstatics
                (eval_simple_among vars closure w Hvars Hclosure Heval) Hof)
      as [_ Hslots].
    exact (lookup_slot_among slot wslots v Hslots).
  - intros Hclosure.
    destruct (eval_simple vars closure) as [w | ] eqn:Heval;
      [ | discriminate ].
    simpl. unfold project_value_slot.
    destruct (closure_in_set statics w) as [[[at_slot decls] wslots] | ]
      eqn:Hset; [ | discriminate ].
    simpl.
    destruct (Pos.eqb at_slot from); [ | discriminate ].
    destruct (closure_in_set_among statics w at_slot decls wslots Hstatics
                (eval_simple_among vars closure w Hvars Hclosure Heval) Hset)
      as [_ Hslots].
    exact (lookup_slot_among slot wslots v Hslots).
  - intros Hclosure.
    destruct (eval_simple vars closure) as [w | ] eqn:Heval;
      [ | discriminate ].
    simpl.
    exact (project_function_slot_among statics from to w v Hstatics
             (eval_simple_among vars closure w Hvars Hclosure Heval)).
Qed.

(* The same of [exec_named], with the values that the heap holds. *)
Lemma exec_named_among (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (vars : env)
    (d : named) (v : value) :
  statics_among statics ->
  env_values_all P vars ->
  heap_values_all P hp ->
  named_statics_in_scope syms ids d = true ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  P v ∧ heap_values_all P hp'.
Proof.
  intros Hstatics Hvars Hheap Hd.
  apply (exec_named_values_all P statics hp hp' rd rd' vars d v).
  - intros c. split; reflexivity.
  - intros tag fields. unfold P. simpl. apply forallb_forall.
  - intros elements. unfold P. simpl. apply forallb_forall.
  - intros s tag fields _ Hlookup.
    pose proof (statics_among_lookup statics s _ Hstatics Hlookup)
      as Hfields.
    simpl in Hfields. apply forallb_forall. exact Hfields.
  - intros s elements _ Hlookup.
    pose proof (statics_among_lookup statics s _ Hstatics Hlookup)
      as Helements.
    simpl in Helements. apply forallb_forall. exact Helements.
  - intros l. reflexivity.
  - exact Hheap.
  - intros p args vs -> Hvs. apply forallb_forall.
    exact (eval_simples_among vars args Hvars vs Hd Hvs).
  - intros _ w. exact (eval_named_among statics vars d w Hstatics Hvars Hd).
Qed.

Lemma eval_static_const_among (vars : env) (const : static_const)
    (allocated : statically_allocated_value) :
  env_values_all P vars ->
  static_const_statics_in_scope syms ids const = true ->
  eval_static_const vars const = Some allocated ->
  allocated_statics_among syms ids allocated = true.
Proof.
  intros Hvars.
  destruct const as [code slots | slot symbols decls slots | tag fields
                    | contents | bytes | elements];
    simpl.
  - intros H. apply andb_true_iff in H as [Hc Hslots].
    destruct (eval_value_slots vars slots) as [evaluated | ] eqn:Heval;
      [ | discriminate ].
    simpl. intros [= <-]. simpl. rewrite Hc.
    exact (eval_value_slots_among vars slots Hvars evaluated Hslots Heval).
  - intros H. apply andb_true_iff in H as [H Hslots].
    destruct (lookup slot decls); [ | discriminate ]. simpl.
    destruct (eval_value_slots vars slots) as [evaluated | ] eqn:Heval;
      [ | discriminate ].
    simpl. intros [= <-]. simpl. rewrite H.
    exact (eval_value_slots_among vars slots Hvars evaluated Hslots Heval).
  - intros Hfields.
    destruct (eval_simples vars fields) as [vs | ] eqn:Heval;
      [ | discriminate ].
    simpl. intros [= <-]. simpl.
    exact (eval_simples_among vars fields Hvars vs Hfields Heval).
  - intros Hcontents.
    destruct (eval_simple vars contents) as [v | ] eqn:Heval;
      [ | discriminate ].
    simpl. intros [= <-]. simpl.
    exact (eval_simple_among vars contents v Hvars Hcontents Heval).
  - intros _ [= <-]. reflexivity.
  - intros Helements.
    destruct (eval_simples vars elements) as [vs | ] eqn:Heval;
      [ | discriminate ].
    simpl. intros [= <-]. simpl.
    exact (eval_simples_among vars elements Hvars vs Helements Heval).
Qed.

End Evaluation.

(* ======================================================================== *)
(* What a group binds its symbols to                                        *)
(* ======================================================================== *)

(* What a symbol is bound to after a group: what it was bound to before,
   or the value of a constant of the group. *)
Lemma eval_static_bindings_values (vars : env)
    (bindings : list static_binding) :
  ∀ (statics statics' : static_env) (s : symbol)
    (allocated : statically_allocated_value),
    eval_static_bindings vars bindings statics = Some statics' ->
    (s, allocated) ∈ static_values statics' ->
    (s, allocated) ∈ static_values statics
    ∨ ∃ const : static_const,
        Bind_symbol s const ∈ bindings
        ∧ eval_static_const vars const = Some allocated.
Proof.
  induction bindings as [ | [id' c' | id' | s' const'] bindings IH];
    intros statics statics' s allocated; simpl.
  - intros [= <-]. auto.
  - destruct (lookup id' (static_code statics)); [ discriminate | ].
    intros Heval Hin.
    destruct (IH _ _ s allocated Heval Hin) as [Hold | (const & Hc & He)];
      [ auto | ].
    right. exists const. auto.
  - intros Heval Hin.
    destruct (IH _ _ s allocated Heval Hin) as [Hold | (const & Hc & He)];
      [ auto | ].
    right. exists const. auto.
  - destruct (lookup s' (static_values statics)); [ discriminate | ].
    destruct (eval_static_const vars const') as [allocated' | ] eqn:Hconst;
      simpl; [ | discriminate ].
    intros Heval Hin.
    destruct (IH _ _ s allocated Heval Hin) as [Hold | (const & Hc & He)].
    + simpl in Hold. destruct Hold as [[= <- <-] | Hold]; [ | auto ].
      right. exists const'. auto.
    + right. exists const. auto.
Qed.

(* Every constant of a group mentions only what is in scope after the
   group. *)
Lemma group_consts_all (all : list symbol) (ids : list code_id)
    (bindings : list static_binding) :
  ∀ before : list symbol,
    sub before all ->
    sub (flat_map symbols_of_binding bindings) all ->
    group_consts_in_scope before all ids bindings = true ->
    ∀ (s : symbol) (const : static_const),
      Bind_symbol s const ∈ bindings ->
      static_const_statics_in_scope all ids const = true.
Proof.
  induction bindings as [ | [id' c' | id' | s' const'] bindings IH];
    intros before Hbefore Hgroup; simpl; intros Hscope s const Hin.
  - destruct Hin.
  - destruct Hin as [[=] | Hin].
    exact (IH before Hbefore Hgroup Hscope _ _ Hin).
  - destruct Hin as [[=] | Hin].
    exact (IH before Hbefore Hgroup Hscope _ _ Hin).
  - simpl in Hgroup.
    assert (Hs' : among s' all = true).
    { apply Hgroup. unfold among. simpl. rewrite Pos.eqb_refl. reflexivity. }
    assert (Hrest : sub (flat_map symbols_of_binding bindings) all).
    { intros name Hn. apply Hgroup. unfold among in Hn |- *. simpl.
      rewrite Hn. apply orb_true_r. }
    destruct (is_closure_const const');
      apply andb_true_iff in Hscope as [Hconst Hscope].
    + destruct Hin as [[= <- <-] | Hin]; [ exact Hconst | ].
      exact (IH before Hbefore Hrest Hscope _ _ Hin).
    + destruct Hin as [[= <- <-] | Hin].
      * exact (static_const_scope_sub _ _ _ _ _ Hbefore (sub_refl _) Hconst).
      * refine (IH (s' :: before) _ Hrest Hscope _ _ Hin).
        intros name Hn. unfold among in Hn. simpl in Hn.
        apply orb_true_iff in Hn as [Hn | Hn]; [ | exact (Hbefore name Hn) ].
        apply Pos.eqb_eq in Hn. subst name. exact Hs'.
Qed.

(* The symbols that a group binds to closures are among those that it
   binds. *)
Lemma closure_symbols_sub (bindings : list static_binding) :
  sub (flat_map closure_symbols_of_binding bindings)
    (flat_map symbols_of_binding bindings).
Proof.
  intros name. unfold among.
  induction bindings as [ | [id c | id | s const] bindings IH]; simpl; auto.
  destruct (is_closure_const const); simpl.
  - intros Hn. apply orb_true_iff in Hn as [Hn | Hn];
      [ rewrite Hn; reflexivity | rewrite (IH Hn); apply orb_true_r ].
  - intros Hn. rewrite (IH Hn). apply orb_true_r.
Qed.

(* ======================================================================== *)
(* The invariant of a run, on values                                        *)
(* ======================================================================== *)

Lemma env_values_all_impl (P P' : value -> Prop) (vars : env) :
  (∀ v : value, P v -> P' v) ->
  env_values_all P vars -> env_values_all P' vars.
Proof.
  intros Himpl. unfold env_values_all. apply Forall_impl.
  intros binding. apply Himpl.
Qed.

Lemma forallb_Forall_among (syms : list symbol) (ids : list code_id)
    (vs : list value) :
  forallb (value_statics_among syms ids) vs = true
  <-> Forall (fun v => value_statics_among syms ids v = true) vs.
Proof. rewrite forallb_forall, Forall_forall. reflexivity. Qed.

Section Values.

Variable unit_exn : continuation.
Variable deleted : list code_id.

(* A step keeps what the state holds bound, given the invariant on
   terms. *)
Lemma values_step (s : state) :
  terms_inv unit_exn deleted s ->
  state_statics_bound deleted s ->
  match step s with
  | Step s' => state_statics_bound deleted s'
  | Exit _ vs _ _ => Forall (value_statics_bound deleted (state_statics s)) vs
  | Undefined => True
  end.
Proof.
  intros (P & S & D & U & L & K & C) Hvalues.
  pose proof Hvalues as (Hvars & Hconts & Hstatics & Hheap).
  apply (step_values_all (value_statics_bound deleted)
           (allocated_statics_bound deleted) s);
    [ | exact Hvalues | | | | | ].
  - intros statics callee args arity. unfold value_statics_bound. simpl.
    rewrite andb_true_iff, forallb_Forall_among. reflexivity.
  - (* the fields of a tuple: of a block, or of a symbol bound to one *)
    intros size tuple fields Htuple Hfields.
    apply (tuple_fields_of_blocks _ (state_statics s) size tuple fields);
      [ | exact Hfields ].
    intros tag block Hblock. apply forallb_Forall_among.
    destruct (block_of_cases _ _ _ _ Hblock) as [-> | (sym & -> & Hs)].
    + exact Htuple.
    + exact (statics_among_lookup _ _ (state_statics s) sym _ Hstatics Hs).
  - intros x kind d body v hp' rd' He Hexec. rewrite He in S. simpl in S.
    apply andb_true_iff in S as [S _].
    exact (exec_named_among _ _ (state_statics s) (state_heap s) hp' _ _
             (state_vars s) d v Hstatics Hvars Hheap S Hexec).
  - intros bindings body statics' He Heval. rewrite He in S, D.
    cbn [expr_statics_in_scope static_names] in S, D.
    apply andb_true_iff in S as [S _].
    apply andb_true_iff in S as [Sgroup _].
    rewrite all_among_app in D. apply andb_true_iff in D as [Dgroup _].
    set (statics := state_statics s) in *. set (vars := state_vars s) in *.
    assert (Hsyms :
      sub (flat_map symbols_of_binding bindings ++ bound_symbols statics)
        (bound_symbols statics')).
    { intros name Hn.
      rewrite
        (proj1 (eval_static_bindings_bound vars bindings statics statics'
                  name Heval)).
      rewrite among_app in Hn. exact Hn. }
    assert (Hids :
      sub
        (flat_map declared_code_ids_of_binding bindings
         ++ ids_of deleted statics)
        (ids_of deleted statics')).
    { intros name Hn. unfold ids_of in Hn |- *.
      rewrite !among_app, declared_code_ids_split in Hn.
      rewrite among_app,
        (proj2 (eval_static_bindings_bound vars bindings statics statics'
                  name Heval)).
      apply orb_true_iff in Hn as [Hn | Hn];
        apply orb_true_iff in Hn as [Hn | Hn].
      - rewrite Hn. reflexivity.
      - rewrite (proj1 (all_among_spec _ _) Dgroup name Hn).
        apply orb_true_r.
      - rewrite Hn, orb_true_r. reflexivity.
      - rewrite Hn. apply orb_true_r. }
    assert (Hsyms_old : sub (bound_symbols statics) (bound_symbols statics'))
      by exact (sub_trans _ _ _ (sub_app_r _ _) Hsyms).
    assert (Hids_old : sub (ids_of deleted statics) (ids_of deleted statics'))
      by exact (sub_trans _ _ _ (sub_app_r _ _) Hids).
    split.
    + intros v. apply value_among_sub; assumption.
    + apply Forall_forall. intros [sym allocated] Hin. cbn [snd].
      unfold allocated_statics_bound.
      destruct (eval_static_bindings_values vars bindings statics statics'
                  sym allocated Heval Hin) as [Hold | (const & Hc & He')].
      * pose proof (proj1 (Forall_forall _ _) Hstatics _ Hold) as Hall.
        cbn [snd] in Hall.
        exact (allocated_among_sub _ _ _ _ _ Hsyms_old Hids_old Hall).
      * refine (allocated_among_sub _ _ _ _ _ Hsyms Hids _).
        refine (eval_static_const_among _ _ vars const allocated _ _ He').
        -- revert Hvars. apply env_values_all_impl. intros v.
           apply value_among_sub; [ apply sub_app_r | apply sub_app_r ].
        -- refine (group_consts_all _ _ bindings _ _ _ Sgroup sym const Hc).
           ++ apply sub_app; [ | apply sub_app_r ].
              exact (sub_trans _ _ _ (closure_symbols_sub bindings)
                       (sub_app_l _ _)).
           ++ apply sub_app_l.
  - intros ap callee vs extra He Hcallee Hvs Hextra. rewrite He in S.
    cbn [expr_statics_in_scope] in S. unfold apply_statics_in_scope in S.
    apply andb_true_iff in S as [S Sextra].
    apply andb_true_iff in S as [S Sargs].
    apply andb_true_iff in S as [Scallee _].
    refine (conj _ (conj _ _)).
    + exact (eval_simple_among _ _ _ _ _ Hvars Scallee Hcallee).
    + apply forallb_Forall_among.
      exact (eval_simples_among _ _ _ _ Hvars vs Sargs Hvs).
    + apply forallb_Forall_among.
      exact (eval_simples_among _ _ _ _ Hvars extra Sextra Hextra).
  - intros ac vs He Hvs. rewrite He in S. cbn [expr_statics_in_scope] in S.
    apply forallb_Forall_among.
    exact (eval_simples_among _ _ _ _ Hvars vs S Hvs).
Qed.

End Values.

(* ======================================================================== *)
(* No step looks up what is not bound                                       *)
(* ======================================================================== *)

Lemma bound_not_unbound (deleted : list code_id) (statics : static_env)
    (v : value) :
  value_statics_bound deleted statics v -> unbound_symbol statics v = false.
Proof.
  unfold value_statics_bound. destruct v; simpl; intros H; try reflexivity.
  unfold symbol_bound. unfold bound_symbols in H.
  rewrite <- is_bound_among in H. unfold is_bound in H.
  destruct (lookup s (static_values statics));
    [ reflexivity | discriminate H ].
Qed.

Lemma closure_called_bound (deleted : list code_id) (statics : static_env)
    (callee : value) :
  value_statics_bound deleted statics callee ->
  value_statics_bound deleted statics (closure_called callee).
Proof.
  unfold value_statics_bound, closure_called. destruct callee; simpl; auto.
  intros H. apply andb_true_iff in H as [H _]. exact H.
Qed.

Lemma calls_bound (deleted : list code_id) (statics : static_env)
    (callee : value) :
  Forall
    (fun binding => allocated_statics_bound deleted statics (snd binding))
    (static_values statics) ->
  value_statics_bound deleted statics callee ->
  calls_unbound deleted statics callee = false.
Proof.
  intros Hstatics Hcallee. apply closure_called_bound in Hcallee.
  unfold calls_unbound. rewrite (bound_not_unbound _ _ _ Hcallee). simpl.
  destruct (closure_of statics (closure_called callee)) as [[called slots] | ]
    eqn:Hof; [ | reflexivity ].
  destruct (closure_of_among _ _ statics _ called slots Hstatics Hcallee Hof)
    as [Hc _].
  rewrite among_app in Hc.
  replace (code_id_bound statics called)
    with (among called (bound_code_ids statics)).
  - rewrite Hc. reflexivity.
  - unfold bound_code_ids. rewrite <- is_bound_among. reflexivity.
Qed.

Lemma called_by_cont_bound (P : value -> Prop) (c : cont_value) :
  ∀ (vs : list value) (v : value) (remaining : list value),
    cont_values_all P c -> Forall P vs ->
    called_by_cont c vs = Some (v, remaining) ->
    P v ∧ Forall P remaining.
Proof.
  induction c
    as [ h vars conts recursive | k | c IH extra
       | kept remaining_arity return_arity return_to _ exn_to _];
    intros vs v remaining; simpl; try discriminate.
  - intros [Hextra Hc] Hvs. apply IH; [ exact Hc | ].
    apply Forall_app. auto.
  - intros (Hkept & _) Hvs.
    destruct vs as [ | v' [ | v'' vs]]; try discriminate.
    intros [= <- <-]. split; [ exact (Forall_inv Hvs) | exact Hkept ].
Qed.

(* The tuple that a call reads is one of the values that the call has in
   hand: an argument, or an argument that a partial application kept. *)
Lemma reads_bound_tuple (deleted : list code_id) (statics : static_env)
    (callee : value) (vs : list value) :
  value_statics_bound deleted statics callee ->
  Forall (value_statics_bound deleted statics) vs ->
  reads_unbound_tuple statics callee vs = false.
Proof.
  intros Hcallee Hvs. unfold reads_unbound_tuple, tuple_read.
  destruct (closure_of statics (closure_called callee)) as [[called slots] | ];
    [ | reflexivity ].
  destruct (lookup called (static_code statics)) as [c | ]; [ | reflexivity ].
  destruct (code_is_tupled c); [ | reflexivity ].
  assert (Hall :
    Forall (value_statics_bound deleted statics) (arguments_called callee vs)).
  { unfold arguments_called. destruct callee; simpl; try exact Hvs.
    apply Forall_app. split; [ | exact Hvs ].
    unfold value_statics_bound in Hcallee. simpl in Hcallee.
    apply andb_true_iff in Hcallee as [_ Hkept].
    apply forallb_Forall_among. exact Hkept. }
  destruct (arguments_called callee vs) as [ | tuple rest]; [ reflexivity | ].
  simpl. exact (bound_not_unbound deleted statics tuple (Forall_inv Hall)).
Qed.

Lemma existsb_unbound_false (deleted : list code_id) (statics : static_env)
    (vs : list value) :
  Forall (value_statics_bound deleted statics) vs ->
  existsb (unbound_symbol statics) vs = false.
Proof.
  induction 1 as [ | v vs Hv _ IH]; simpl; [ reflexivity | ].
  rewrite (bound_not_unbound _ _ _ Hv), IH. reflexivity.
Qed.

Section Fault.

Variable unit_exn : continuation.
Variable deleted : list code_id.

Lemma no_scope_fault (s : state) :
  terms_inv unit_exn deleted s ->
  state_statics_bound deleted s ->
  statics_scope_fault deleted s = false.
Proof.
  intros (_ & S & _) (Hvars & Hconts & Hstatics & _).
  unfold statics_scope_fault.
  destruct (state_expr s)
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    try reflexivity.
  - cbn [expr_statics_in_scope] in S. apply andb_true_iff in S as [S _].
    destruct d as [simple | p args | code slots | slot decls slots
                  | slot closure | from slot closure | from to closure];
      try reflexivity; simpl in S.
    + destruct (eval_simples (state_vars s) args) as [vs | ] eqn:Heval;
        [ | reflexivity ].
      apply (existsb_unbound_false deleted), forallb_Forall_among.
      exact (eval_simples_among _ _ _ _ Hvars vs S Heval).
    + destruct (eval_simple (state_vars s) closure) as [v | ] eqn:Heval;
        [ | reflexivity ].
      apply (bound_not_unbound deleted).
      exact (eval_simple_among _ _ _ _ _ Hvars S Heval).
    + destruct (eval_simple (state_vars s) closure) as [v | ] eqn:Heval;
        [ | reflexivity ].
      apply (bound_not_unbound deleted).
      exact (eval_simple_among _ _ _ _ _ Hvars S Heval).
    + destruct (eval_simple (state_vars s) closure) as [v | ] eqn:Heval;
        [ | reflexivity ].
      apply (bound_not_unbound deleted).
      exact (eval_simple_among _ _ _ _ _ Hvars S Heval).
  - cbn [expr_statics_in_scope] in S. unfold apply_statics_in_scope in S.
    apply andb_true_iff in S as [S _].
    apply andb_true_iff in S as [S Sargs].
    apply andb_true_iff in S as [Scallee _].
    destruct (eval_simple (state_vars s) (ap_callee ap)) as [callee | ]
      eqn:Heval; [ | reflexivity ].
    pose proof (eval_simple_among _ _ _ _ _ Hvars Scallee Heval) as Hcallee.
    rewrite (calls_bound deleted _ callee Hstatics Hcallee). simpl.
    destruct (eval_simples (state_vars s) (ap_args ap)) as [vs | ] eqn:Hargs;
      [ | reflexivity ].
    apply (reads_bound_tuple deleted); [ exact Hcallee | ].
    apply forallb_Forall_among.
    exact (eval_simples_among _ _ _ _ Hvars vs Sargs Hargs).
  - cbn [expr_statics_in_scope] in S.
    destruct (eval_simples (state_vars s) (ac_args ac)) as [vs | ] eqn:Heval;
      [ | reflexivity ].
    destruct (called_by_cont (resolve (ac_continuation ac) (state_conts s)) vs)
      as [[v remaining] | ] eqn:Hcalled; [ | reflexivity ].
    destruct (called_by_cont_bound _ _ vs v remaining
                (resolve_values_all _ _ _ Hconts)
                (proj1 (forallb_Forall_among _ _ _)
                   (eval_simples_among _ _ _ _ Hvars vs S Heval))
                Hcalled)
      as [Hv Hremaining].
    rewrite (calls_bound deleted _ v Hstatics Hv). simpl.
    exact (reads_bound_tuple deleted _ v remaining Hv Hremaining).
Qed.

End Fault.

(* ======================================================================== *)
(* The run of a unit                                                        *)
(* ======================================================================== *)

Lemma all_among_refl (names : list positive) : all_among names names = true.
Proof. apply all_among_spec. intros name H. exact H. Qed.

(* The state that a unit is run from has both invariants. *)
Lemma initial_state_inv (imports : list symbol) (u : unit_interface)
    (e : expr) (statics : static_env) (hp : heap) (rd : read_record) :
  well_formed_unit u e = true ->
  statics_in_scope imports e = true ->
  interface_conts_not_rebound u e = true ->
  imports_bound imports statics ->
  heap_values_all (fun v => value_statics_among imports [] v = true) hp ->
  terms_inv (unit_exn u) (deleted_code_ids e) (State e [] [] statics hp rd)
  ∧ state_statics_bound (deleted_code_ids e) (State e [] [] statics hp rd).
Proof.
  intros Hwf Hscope Hconts (Hcode & Hbound & Hvalues) Hheap.
  assert (Hsyms : sub imports (bound_symbols statics)).
  { intros name Hn. unfold bound_symbols. rewrite <- is_bound_among.
    rewrite <- Hn, <- Hbound. reflexivity. }
  unfold interface_conts_not_rebound in Hconts.
  apply andb_true_iff in Hconts as [_ Hexn].
  apply negb_true_iff in Hexn.
  split.
  - unfold terms_inv. simpl_state.
    refine (conj _ (conj _ (conj _ (conj Hexn (conj eq_refl (conj _ _)))))).
    + exact (well_formed_unit_statics_at_unit_toplevel u e Hwf).
    + exact (expr_scope_sub e _ _ _ _ Hsyms (sub_nil _) Hscope).
    + apply all_among_refl.
    + intros k _. apply Good_outer.
    + unfold code_all_fine. rewrite Hcode. apply Forall_nil.
  - unfold state_statics_bound, state_values_all. simpl_state.
    refine (conj (Forall_nil _) (conj I (conj _ _))).
    + revert Hvalues. apply Forall_impl. intros binding.
      apply allocated_among_sub; [ exact Hsyms | apply sub_nil ].
    + intros l c v Hin Hv.
      exact (value_among_sub _ _ _ _ Hsyms (sub_nil _) v (Hheap l c v Hin Hv)).
Qed.

Lemma inv_run (unit_exn : continuation) (deleted : list code_id) (n : nat) :
  ∀ s s' : state,
    terms_inv unit_exn deleted s ->
    state_statics_bound deleted s ->
    run n s = Step s' ->
    terms_inv unit_exn deleted s' ∧ state_statics_bound deleted s'.
Proof.
  induction n as [ | n IH]; intros s s' Hterms Hvalues; simpl.
  - intros [= <-]. auto.
  - pose proof (values_step unit_exn deleted s Hterms Hvalues) as Hstep.
    destruct (step s) as [next | k vs | ] eqn:Hs; try discriminate.
    exact (IH next s' (terms_inv_step unit_exn deleted s next Hterms Hs)
             Hstep).
Qed.

Theorem statics_bound_in_run : statics_bound_in_run_statement.
Proof.
  intros imports u e statics hp rd n s Hwf Hscope Hconts Himports Hheap Hrun.
  destruct
    (initial_state_inv imports u e statics hp rd Hwf Hscope Hconts Himports
       Hheap)
    as [Hterms Hvalues].
  destruct (inv_run (unit_exn u) (deleted_code_ids e) n _ s Hterms Hvalues
              Hrun) as [Hterms' Hvalues'].
  refine (conj Hvalues' (conj _ _)).
  - exact (no_scope_fault (unit_exn u) (deleted_code_ids e) s Hterms'
             Hvalues').
  - pose proof (values_step (unit_exn u) (deleted_code_ids e) s Hterms'
                  Hvalues') as Hstep.
    destruct (step s); auto.
Qed.

(* ======================================================================== *)
(* A code id that is declared deleted is not bound                          *)
(* ======================================================================== *)

(* Of three lists of names, the bound, the deleted and the declared:
   the first two are among the third, and where the third has no name
   twice, no name is in both of the first two. *)
Definition apart (bound deleted declared : list positive) : Prop :=
  incl bound declared
  ∧ incl deleted declared
  ∧ (NoDup declared -> ∀ name : positive, In name deleted -> ~ In name bound).

Lemma apart_nil : apart [] [] [].
Proof.
  refine (conj (incl_refl _) (conj (incl_refl _) _)). intros _ name [].
Qed.

Lemma apart_app (bound deleted declared bound' deleted' declared'
    : list positive) :
  apart bound deleted declared ->
  apart bound' deleted' declared' ->
  apart (bound ++ bound') (deleted ++ deleted') (declared ++ declared').
Proof.
  intros (Hb & Hd & Ha) (Hb' & Hd' & Ha').
  refine (conj (incl_app_app Hb Hb') (conj (incl_app_app Hd Hd') _)).
  intros Hnodup name Hdeleted Hbound.
  pose proof (NoDup_app_remove_r _ _ Hnodup) as Hleft.
  pose proof (NoDup_app_remove_l _ _ Hnodup) as Hright.
  assert (Hdisjoint : ∀ n : positive, In n declared -> ~ In n declared').
  { intros n Hn Hn'. revert Hnodup Hn Hn'. clear.
    induction declared as [ | first declared IH]; simpl;
      [ intros _ [] | ].
    intros Hnodup [-> | Hn] Hn'.
    - apply NoDup_cons_iff in Hnodup as [Hnot _]. apply Hnot.
      apply in_or_app. right. exact Hn'.
    - apply NoDup_cons_iff in Hnodup as [_ Hnodup].
      exact (IH Hnodup Hn Hn'). }
  apply in_app_or in Hdeleted as [Hdeleted | Hdeleted];
    apply in_app_or in Hbound as [Hbound | Hbound].
  - exact (Ha Hleft name Hdeleted Hbound).
  - exact (Hdisjoint name (Hd name Hdeleted) (Hb' name Hbound)).
  - exact (Hdisjoint name (Hb name Hbound) (Hd' name Hdeleted)).
  - exact (Ha' Hright name Hdeleted Hbound).
Qed.

Lemma binding_apart (binding : static_binding) :
  apart (code_ids_of_binding binding) (deleted_code_ids_of_binding binding)
    (declared_code_ids_of_binding binding).
Proof.
  destruct binding as [id c | id | s const]; simpl.
  - refine (conj (incl_refl _) (conj (incl_nil_l _) _)). intros _ name [].
  - refine (conj (incl_nil_l _) (conj (incl_refl _) _)). intros _ name _ [].
  - apply apart_nil.
Qed.

Lemma bindings_apart (bindings : list static_binding) :
  apart (flat_map code_ids_of_binding bindings)
    (flat_map deleted_code_ids_of_binding bindings)
    (flat_map declared_code_ids_of_binding bindings).
Proof.
  induction bindings as [ | binding bindings IH]; simpl;
    [ apply apart_nil | ].
  exact (apart_app _ _ _ _ _ _ (binding_apart binding) IH).
Qed.

Fixpoint static_names_apart_lists (e : expr) {struct e} :
  apart (static_names code_ids_of_binding e)
    (static_names deleted_code_ids_of_binding e)
    (static_names declared_code_ids_of_binding e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [static_names handler_static_names]; try apply apart_nil.
  - apply static_names_apart_lists.
  - exact (apart_app _ _ _ _ _ _ (bindings_apart bindings)
             (static_names_apart_lists body)).
  - exact (apart_app _ _ _ _ _ _ (static_names_apart_lists handler)
             (static_names_apart_lists body)).
  - refine (apart_app _ _ _ _ _ _ _ (static_names_apart_lists body)).
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      cbn [flat_map handler_static_names]; [ apply apart_nil | ].
    exact (apart_app _ _ _ _ _ _ (static_names_apart_lists handler) IH).
Qed.

(* A unit whose static names are apart binds no code id that it declares
   deleted. *)
Lemma deleted_not_bound (imports : list symbol) (e : expr) (id : code_id) :
  static_names_apart imports e = true ->
  among id (deleted_code_ids e) = true ->
  among id (static_names code_ids_of_binding e) = false.
Proof.
  unfold static_names_apart, deleted_code_ids. intros Hapart Hdeleted.
  apply andb_true_iff in Hapart as [_ Hdistinct].
  apply distinct_NoDup in Hdistinct. apply among_In in Hdeleted.
  destruct (static_names_apart_lists e) as (_ & _ & Hlists).
  destruct (among id (static_names code_ids_of_binding e)) eqn:Hbound;
    [ | reflexivity ].
  apply among_In in Hbound.
  destruct (Hlists Hdistinct id Hdeleted Hbound).
Qed.

(* ======================================================================== *)
(* A test inside a test                                                     *)
(* ======================================================================== *)

(* The shape that Simplify gives a match of three cases whose arms are
   lifted: the handler of the inner test is defined above the group that
   binds the symbols of the arms, and mentions two of them. It is not at
   toplevel, since its body also applies the continuation of the whole
   match; the symbols are in scope in it because the body of its
   [Let_cont] binds them. The arms here are blocks, where Simplify has
   closures. The
   continuations are [1] and [2] for the unit, [3] for the match and [4]
   for the inner test; the symbol [1] is what the unit returns, and
   [4], [2] and [3] are those of the arms. *)
Definition nested_test_interface : unit_interface :=
  Unit_interface 1%positive 2%positive.

Definition nested_test_arm (i : Z) : static_const :=
  Static_const_block 0 [Const (Tagged_immediate i)].

Definition nested_test_unit (scrutinee : Z) : expr :=
  Let_cont_non_recursive 3%positive
    (Handler [1%positive] (make_handler_metadata [Value])
       (Let_static
          [Bind_symbol 1%positive (Static_const_block 0 [Var 1%positive])]
          (Apply_cont (Apply_cont_expr 1%positive [Symbol 1%positive] None))))
    (Let 2%positive (Naked_number Kind_naked_immediate)
       (Simple (Const (Naked_immediate scrutinee)))
       (Let_cont_non_recursive 4%positive
          (Handler [] (make_handler_metadata [])
             (Switch (Var 2%positive)
                [(1%Z, Apply_cont_expr 3%positive [Symbol 2%positive] None);
                 (2%Z, Apply_cont_expr 3%positive [Symbol 3%positive] None)]))
          (Let_static
             [Bind_symbol 4%positive (nested_test_arm 0);
              Bind_symbol 2%positive (nested_test_arm 1);
              Bind_symbol 3%positive (nested_test_arm 2)]
             (Switch (Var 2%positive)
                [(0%Z, Apply_cont_expr 3%positive [Symbol 4%positive] None);
                 (1%Z, Apply_cont_expr 4%positive [] None);
                 (2%Z, Apply_cont_expr 4%positive [] None)])))).

Example nested_test_passes :
  well_formed_unit nested_test_interface (nested_test_unit 1) = true
  ∧ statics_in_scope [] (nested_test_unit 1) = true
  ∧ static_names_apart [] (nested_test_unit 1) = true
  ∧ interface_conts_not_rebound nested_test_interface (nested_test_unit 1)
    = true.
Proof. repeat split; vm_compute; reflexivity. Qed.

(* It goes through the inner test and returns. *)
Example nested_test_returns :
  run 12
    (State (nested_test_unit 1) [] [] (Static_env [] []) empty_heap no_reads)
  = Exit 1%positive [Value_symbol 1%positive] empty_heap no_reads.
Proof. vm_compute. reflexivity. Qed.

(* ======================================================================== *)
(* Handlers that mention what the body binds                                *)
(* ======================================================================== *)

Definition handlers_interface : unit_interface :=
  Unit_interface 1%positive 2%positive.

(* A recursive handler that mentions a symbol which the body of its
   [Let_cont] binds, and returns it. *)
Definition recursive_handler_unit : expr :=
  Let_cont_recursive
    [(3%positive,
      Handler [] (make_handler_metadata [])
        (Apply_cont (Apply_cont_expr 1%positive [Symbol 1%positive] None)))]
    (Let_static [Bind_symbol 1%positive (nested_test_arm 0)]
       (Apply_cont (Apply_cont_expr 3%positive [] None))).

Example recursive_handler_passes :
  well_formed_unit handlers_interface recursive_handler_unit = true
  ∧ statics_in_scope [] recursive_handler_unit = true
  ∧ static_names_apart [] recursive_handler_unit = true
  ∧ interface_conts_not_rebound handlers_interface recursive_handler_unit
    = true.
Proof. repeat split; vm_compute; reflexivity. Qed.

Example recursive_handler_returns :
  run 4
    (State recursive_handler_unit [] [] (Static_env [] []) empty_heap no_reads)
  = Exit 1%positive [Value_symbol 1%positive] empty_heap no_reads.
Proof. vm_compute. reflexivity. Qed.

(* A handler, of the continuation [3], that mentions a symbol which the
   body of its [Let_cont] binds in the handler of another continuation,
   [4], and not on its spine. The handler of [3] is not at toplevel: the
   body of its [Let_cont] can also return. The handler of [4] is, so it
   may bind the symbol, and it always runs before [3] is applied: this
   is what row 1 gives, and what the check relies on. *)
Definition sibling_handler_unit (scrutinee : Z) : expr :=
  Let 1%positive (Naked_number Kind_naked_immediate)
    (Simple (Const (Naked_immediate scrutinee)))
    (Let_cont_non_recursive 3%positive
       (Handler [] (make_handler_metadata [])
          (Apply_cont (Apply_cont_expr 1%positive [Symbol 1%positive] None)))
       (Let_cont_non_recursive 4%positive
          (Handler [] (make_handler_metadata [])
             (Let_static [Bind_symbol 1%positive (nested_test_arm 0)]
                (Switch (Var 1%positive)
                   [(0%Z, Apply_cont_expr 3%positive [] None);
                    (1%Z,
                     Apply_cont_expr 1%positive [Const (Tagged_immediate 7)]
                       None)])))
          (Apply_cont (Apply_cont_expr 4%positive [] None)))).

Example sibling_handler_passes :
  well_formed_unit handlers_interface (sibling_handler_unit 0) = true
  ∧ statics_in_scope [] (sibling_handler_unit 0) = true
  ∧ static_names_apart [] (sibling_handler_unit 0) = true
  ∧ interface_conts_not_rebound handlers_interface (sibling_handler_unit 0)
    = true.
Proof. repeat split; vm_compute; reflexivity. Qed.

Example sibling_handler_returns :
  run 8
    (State (sibling_handler_unit 0) [] [] (Static_env [] []) empty_heap
       no_reads)
  = Exit 1%positive [Value_symbol 1%positive] empty_heap no_reads.
Proof. vm_compute. reflexivity. Qed.

(* A unit that leaves by its exception continuation before the handler
   that binds a symbol has run. The handler of [3] mentions the symbol;
   the handler of [4] binds it and applies [3]; the body raises. Row 1
   allows the raise, and the unit passes the checks: no handler is
   entered, and it exits without the symbol bound. *)
Definition raise_past_binding_unit : expr :=
  Let_cont_non_recursive 3%positive
    (Handler [] (make_handler_metadata [])
       (Apply_cont (Apply_cont_expr 1%positive [Symbol 1%positive] None)))
    (Let_cont_non_recursive 4%positive
       (Handler [] (make_handler_metadata [])
          (Let_static [Bind_symbol 1%positive (nested_test_arm 0)]
             (Apply_cont (Apply_cont_expr 3%positive [] None))))
       (Apply_cont
          (Apply_cont_expr 2%positive [Const (Tagged_immediate 0)]
             (Some (Pop 2%positive None))))).

Example raise_past_binding_passes :
  well_formed_unit handlers_interface raise_past_binding_unit = true
  ∧ statics_in_scope [] raise_past_binding_unit = true
  ∧ static_names_apart [] raise_past_binding_unit = true
  ∧ interface_conts_not_rebound handlers_interface raise_past_binding_unit
    = true.
Proof. repeat split; vm_compute; reflexivity. Qed.

Example raise_past_binding_exits :
  run 3
    (State raise_past_binding_unit [] [] (Static_env [] []) empty_heap no_reads)
  = Exit 2%positive [Value_const (Tagged_immediate 0)] empty_heap no_reads.
Proof. vm_compute. reflexivity. Qed.

(* ======================================================================== *)
(* A unit whose body goes round a handler that binds a symbol               *)
(* ======================================================================== *)

(* The variable of the test is [2], and [1] is the variable that the
   handler of [3] binds to the field that it reads of the symbol [1]. *)
Definition round_unit : expr :=
  Let 2%positive (Naked_number Kind_naked_immediate)
    (Simple (Const (Naked_immediate 1)))
    (Let_cont_non_recursive 3%positive
       (Handler [] (make_handler_metadata [])
          (Let 1%positive Value
             (Prim
                (Unary
                   (Block_load (Block_access_values None None Any_value)
                      Immutable 0))
                [Symbol 1%positive])
             (Apply_cont (Apply_cont_expr 1%positive [Var 1%positive] None))))
       (Let_cont_non_recursive 4%positive
          (Handler [] (make_handler_metadata [])
             (Let_static [Bind_symbol 1%positive (nested_test_arm 0)]
                (Apply_cont (Apply_cont_expr 3%positive [] None))))
          (Switch (Var 2%positive)
             [(0%Z, Apply_cont_expr 4%positive [] None);
              (1%Z, Apply_cont_expr 3%positive [] None)]))).

(* The state of its run that reads the symbol. *)
Definition round_state : state :=
  match run 5 (State round_unit [] [] (Static_env [] []) empty_heap no_reads)
    with
  | Step s => s
  | Exit _ _ _ _ | Undefined => State Invalid [] [] (Static_env [] [])
    empty_heap no_reads
  end.

Theorem branch_round_binding_reads_unbound :
  branch_round_binding_reads_unbound_statement.
Proof.
  exists handlers_interface, round_unit, 5%nat, round_state.
  repeat split; vm_compute; reflexivity.
Qed.

(* ======================================================================== *)
(* A unit that binds the name of its exception continuation again           *)
(* ======================================================================== *)

(* The continuations: the return continuation of the unit, its exception
   continuation, and [k_outer] and [k]. The variable, the symbol and the
   parameter of the exception handler are [1], [1] and [2]. *)
Definition rebound_interface : unit_interface :=
  Unit_interface 1%positive 2%positive.

Definition rebound_apply_k_outer : expr :=
  Apply_cont (Apply_cont_expr 3%positive [] None).

Definition rebound_unit : expr :=
  Let_cont_non_recursive 3%positive
    (Handler [] (make_handler_metadata [])
       (Let 1%positive Value
          (Prim
             (Unary
                (Block_load (Block_access_values None None Any_value)
                   Immutable 0))
             [Symbol 1%positive])
          (Apply_cont (Apply_cont_expr 1%positive [Var 1%positive] None))))
    (Let_cont_non_recursive 2%positive
       (Handler [2%positive] (make_exn_handler_metadata [Value])
          rebound_apply_k_outer)
       (Let_cont_non_recursive 4%positive
          (Handler [] (make_handler_metadata [])
             (Let_static
                [Bind_symbol 1%positive
                   (Static_const_block 0 [Const (Tagged_immediate 0)])]
                rebound_apply_k_outer))
          (Apply_cont
             (Apply_cont_expr 2%positive [Const (Tagged_immediate 0)]
                (Some (Pop 2%positive None)))))).

(* The state of its run that reads the symbol. *)
Definition rebound_state : state :=
  match run 5 (State rebound_unit [] [] (Static_env [] []) empty_heap no_reads)
    with
  | Step s => s
  | Exit _ _ _ _ | Undefined => State Invalid [] [] (Static_env [] [])
    empty_heap no_reads
  end.

Theorem rebound_unit_exn_reads_unbound :
  rebound_unit_exn_reads_unbound_statement.
Proof.
  exists rebound_interface, rebound_unit, 5%nat, rebound_state.
  repeat split; vm_compute; reflexivity.
Qed.
