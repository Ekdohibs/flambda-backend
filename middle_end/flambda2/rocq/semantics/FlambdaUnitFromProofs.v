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

(* The run of a well-formed unit from a static environment that binds
   something. Nothing here needs review: these are the two facts of
   FlambdaWellFormedSpec.v about the run of a unit from the empty
   static environment ([well_formed_never_faults_statement],
   [well_formed_names_bound_statement]), for a run that starts from a
   static environment that holds no code and binds no name that the
   unit binds (FlambdaImportsSpec.static_names_free_in), such as the
   one of the imports of a unit (FlambdaImportsSpec.imports_statics),
   and from any heap. The proofs are those of
   FlambdaWellFormedProofs.v with another first state: the invariants
   there are of any state.

   They are for the proofs of the passes that are stated of a unit run
   on its own and that take the two facts of the run of their result:
   the liftings to symbols and the two hoistings.

   The first fact is also had of a run that starts from a continuation
   environment of what may follow a unit
   (FlambdaObserversSpec.observers), and of the run of what follows,
   from the static environment that the unit left: what may follow a
   unit binds nothing static. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaBindersSpec FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaSimulationProofs FlambdaScopeSpec
  FlambdaScopeProofs FlambdaRunUnderSpec
  FlambdaWellFormedSpec FlambdaWellFormedProofs FlambdaStaticsInScopeSpec
  FlambdaImportsSpec FlambdaObserversSpec.
From OxCamlFlambda2 Require FlambdaStaticsInScopeProofs.
Import ListNotations.

(* ======================================================================== *)
(* What the test of the names gives                                         *)
(* ======================================================================== *)

Lemma free_in_symbols (statics : static_env) (e : expr) :
  static_names_free_in statics e = true ->
  ∀ n : positive,
    In n (static_names symbols_of_binding e) -> symbol_bound statics n = false.
Proof.
  unfold static_names_free_in. intros H n Hn.
  apply andb_true_iff in H as [Hsymbols _].
  rewrite forallb_forall in Hsymbols. specialize (Hsymbols n Hn).
  apply negb_true_iff in Hsymbols. exact Hsymbols.
Qed.

Lemma free_in_code_ids (statics : static_env) (e : expr) :
  static_names_free_in statics e = true ->
  ∀ n : positive,
    In n (static_names code_ids_of_binding e) ->
    code_id_bound statics n = false.
Proof.
  unfold static_names_free_in. intros H n Hn.
  apply andb_true_iff in H as [_ Hcode].
  rewrite forallb_forall in Hcode. specialize (Hcode n Hn).
  apply negb_true_iff in Hcode. exact Hcode.
Qed.

(* The empty static environment binds nothing. *)
Lemma free_in_empty (e : expr) :
  static_names_free_in empty_static_env e = true.
Proof.
  unfold static_names_free_in. apply andb_true_iff.
  split; apply forallb_forall; intros n _; reflexivity.
Qed.

(* ======================================================================== *)
(* What may follow a unit binds nothing static                              *)
(* ======================================================================== *)

(* A term with no static binding in it passes the test that every
   static binding is at toplevel, wherever it stands. *)
Fixpoint no_static_nowhere (e : expr) {struct e} :
  binds_no_static e = true -> statics_outside_loops_and_code false e = true.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    simpl; try (intros _; reflexivity).
  - apply no_static_nowhere.
  - discriminate.
  - destruct h as [params metadata handler]. simpl. intros H.
    apply andb_true_iff in H as [Hh Hb].
    rewrite (no_static_nowhere handler Hh), (no_static_nowhere body Hb).
    reflexivity.
  - intros H. apply andb_true_iff in H as [Hhs Hb].
    rewrite (no_static_nowhere body Hb), andb_true_r.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    simpl in Hhs. apply andb_true_iff in Hhs as [Hh Hrest].
    rewrite (no_static_nowhere handler Hh). exact (IH Hrest).
Qed.

(* It binds no name, by whatever the names of a binding are. *)
Lemma no_static_names (names_of : static_binding -> list positive)
    (e : expr) :
  binds_no_static e = true -> static_names names_of e = [].
Proof.
  intros H.
  exact (proj2 (proj2
           (statics_nowhere names_of e
              (no_static_nowhere e H)))).
Qed.

(* So it gives nothing to the handlers of a [Let_cont] around it. *)
Lemma no_static_handler_scope (names_of : static_binding -> list positive)
    (k : continuation) (e : expr) :
  binds_no_static e = true -> handler_scope names_of k e = [].
Proof.
  intros H.
  apply FlambdaStaticsInScopeProofs.handler_scope_nothing, no_static_names, H.
Qed.

Lemma no_static_recursive_scope (names_of : static_binding -> list positive)
    (group : list continuation) (e : expr) :
  binds_no_static e = true -> recursive_scope names_of group e = [].
Proof.
  intros H.
  apply FlambdaStaticsInScopeProofs.recursive_scope_nothing, no_static_names, H.
Qed.

(* What may follow a unit satisfies what the invariant of
   FlambdaWellFormedProofs.v asks of a continuation, and binds no
   name. *)
Lemma observer_ok (names_of : static_binding -> list positive)
    (slots : list value_slot) (c : cont_value) :
  observer slots c ->
  cont_ok names_of c ∧ cont_names names_of c = [].
Proof.
  destruct c
    as [ [params metadata body] vars conts recursive | k | c extra
       | remaining remaining_arity return_arity return_to exn_to ];
    simpl; try contradiction.
  - destruct vars; [ | contradiction ].
    destruct conts; [ | contradiction ].
    destruct recursive; [ | contradiction ].
    intros (Hno & _ & _).
    destruct
      (statics_nowhere names_of body
         (no_static_nowhere body Hno))
      as (P & _ & N).
    simpl. rewrite N. split; [ | reflexivity ].
    split; [ exact P | ]. split; [ constructor | ].
    split; [ intros n [] | ]. split; [ reflexivity | exact I ].
  - intros _. split; [ exact I | reflexivity ].
Qed.

Lemma observers_ok (names_of : static_binding -> list positive)
    (slots : list value_slot) (conts : cont_env) :
  observers slots conts ->
  conts_ok names_of conts ∧ conts_names names_of conts = [].
Proof.
  unfold observers, conts_ok, conts_names.
  induction conts as [ | [k c] conts IH]; simpl;
    [ intros _; split; [ exact I | reflexivity ] | ].
  intros H. inversion H as [ | kc rest Hc Hrest]; subst.
  destruct (observer_ok names_of slots c Hc) as [Cc Nc].
  destruct (IH Hrest) as [Cs Ns].
  rewrite Nc, Ns. split; [ exact (conj Cc Cs) | reflexivity ].
Qed.

(* Where what may follow a unit is applied and makes a step, the step
   is to its handler, which binds nothing static, with no continuation
   in scope, under the static environment and the heap given. *)
Lemma observer_applied_step (slots : list value_slot) (c : cont_value)
    (vs : list value) (statics : static_env) (hp : heap) (rd : read_record)
      (s' : state) :
  observer slots c ->
  apply_cont_value c vs statics hp rd = Step s' ->
  ∃ (body : expr) (vars : env),
    binds_no_static body = true ∧ s' = State body vars [] statics hp rd.
Proof.
  destruct c
    as [ [params metadata body] vars conts recursive | k | c extra
       | remaining remaining_arity return_arity return_to exn_to ];
    simpl; try contradiction.
  - destruct vars; [ | contradiction ].
    destruct conts; [ | contradiction ].
    destruct recursive; [ | contradiction ].
    intros (Hno & _ & _). simpl.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | discriminate ].
    destruct (bind_params params vs []) as [bound | ]; [ | discriminate ].
    intros [= <-]. exists body, bound. split; [ exact Hno | reflexivity ].
  - intros _. discriminate.
Qed.

(* What may follow a unit keeps no value. *)
Lemma observers_values_all (P : value -> Prop) (slots : list value_slot)
    (conts : cont_env) :
  observers slots conts -> conts_values_all P conts.
Proof.
  unfold observers, conts_values_all.
  induction conts as [ | [k c] conts IH]; simpl;
    [ intros _; exact I | ].
  intros H. inversion H as [ | kc rest Hc Hrest]; subst.
  split; [ | exact (IH Hrest) ].
  revert Hc.
  destruct c
    as [ [params metadata body] vars kept recursive | k0 | c0 extra
       | remaining remaining_arity return_arity return_to exn_to ];
    simpl; try contradiction; [ | intros _; exact I ].
  destruct vars; [ | contradiction ].
  destruct kept; [ | contradiction ].
  intros _. split; [ constructor | exact I ].
Qed.

(* ======================================================================== *)
(* No static binding is executed twice                                      *)
(* ======================================================================== *)

(* The invariant of FlambdaWellFormedProofs.v holds of the first state,
   for either sort of name, whatever may follow the unit. *)
Lemma inv_under (names_of : static_binding -> list positive)
    (is_bound : static_env -> positive -> bool) (slots : list value_slot)
    (e : expr) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) :
  (∀ n : positive,
    In n (static_names names_of e) -> is_bound statics n = false) ->
  static_code statics = [] ->
  observers slots conts ->
  statics_outside_loops_and_code true e = true ->
  distinct (static_names names_of e) = true ->
  inv names_of is_bound (State e [] conts statics hp rd).
Proof.
  intros Hfree Hno_code Hobs P D.
  destruct (observers_ok names_of slots conts Hobs) as [C N].
  refine
    (conj P (conj (proj1 (distinct_NoDup _) D) (conj C (conj _ (conj _ _))))).
  - intros n _. simpl. rewrite N. intros [].
  - intros n [Hn | Hn]; [ exact (Hfree n Hn) | ].
    simpl in Hn. rewrite N in Hn. destruct Hn.
  - intros id c Hlookup. simpl in Hlookup. rewrite Hno_code in Hlookup.
    discriminate Hlookup.
Qed.

Lemma inv_from (names_of : static_binding -> list positive)
    (is_bound : static_env -> positive -> bool) (e : expr)
    (statics : static_env) (hp : heap) (rd : read_record) :
  (∀ n : positive,
    In n (static_names names_of e) -> is_bound statics n = false) ->
  static_code statics = [] ->
  statics_outside_loops_and_code true e = true ->
  distinct (static_names names_of e) = true ->
  inv names_of is_bound (State e [] [] statics hp rd).
Proof.
  intros Hfree Hno_code.
  exact (inv_under names_of is_bound [] e [] statics hp rd Hfree
           Hno_code (Forall_nil _)).
Qed.

(* A state that satisfies the invariant for the two sorts of name makes
   no static fault. *)
Lemma inv_never_faults (s : state) :
  inv code_ids_of_binding code_id_bound s ->
  inv symbols_of_binding symbol_bound s ->
  static_fault s = false.
Proof.
  intros Hcode Hsymbols.
  rewrite (state_eta s) in Hcode, Hsymbols. unfold static_fault.
  destruct (state_expr s)
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    try reflexivity.
  - rewrite (inv_no_rebinding _ _ _ _ _ _ _ _ rd_ Hcode),
      (inv_no_rebinding _ _ _ _ _ _ _ _ rd_ Hsymbols).
    reflexivity.
  - exact (inv_no_recursive_statics _ _ _ _ _ _ _ _ rd_ Hcode).
  - exact (inv_no_call_binding_statics _ _ _ _ _ _ _ rd_ Hcode).
Qed.

Lemma well_formed_never_faults_under (u : unit_interface)
    (slots : list value_slot) (e : expr) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) (n : nat)
      (s : state) :
  well_formed_unit u e = true ->
  static_code statics = [] ->
  static_names_free_in statics e = true ->
  observers slots conts ->
  run n (State e [] conts statics hp rd) = Step s ->
  static_fault s = false.
Proof.
  intros Hwf Hno_code Hfree Hobs Hrun.
  pose proof (well_formed_unit_statics_outside_loops_and_code u e Hwf) as P.
  pose proof (well_formed_unit_code_ids_distinct u e Hwf) as Dcode.
  pose proof (well_formed_unit_symbols_distinct u e Hwf) as Dsymbols.
  apply inv_never_faults.
  - exact
      (inv_run code_ids_of_binding code_id_bound
         code_bound_after n _ s
         (inv_under code_ids_of_binding code_id_bound slots e
            conts statics hp rd (free_in_code_ids statics e Hfree) Hno_code Hobs
            P Dcode)
         Hrun).
  - exact
      (inv_run symbols_of_binding symbol_bound
         symbol_bound_after n _ s
         (inv_under symbols_of_binding symbol_bound slots e
            conts statics hp rd (free_in_symbols statics e Hfree) Hno_code Hobs
            P Dsymbols)
         Hrun).
Qed.

Lemma well_formed_never_faults_from (u : unit_interface) (e : expr)
    (statics : static_env) (hp : heap) (rd : read_record) (n : nat)
      (s : state) :
  well_formed_unit u e = true ->
  static_code statics = [] ->
  static_names_free_in statics e = true ->
  run n (State e [] [] statics hp rd) = Step s ->
  static_fault s = false.
Proof.
  intros Hwf Hno_code Hfree.
  exact (well_formed_never_faults_under u [] e [] statics hp rd n s Hwf Hno_code
           Hfree (Forall_nil _)).
Qed.

(* What follows a unit, run from the static environment that the unit
   has left, makes no static fault either: it binds nothing static, and
   the code that it may call is that of the unit. *)
Lemma inv_after (names_of : static_binding -> list positive)
    (is_bound : static_env -> positive -> bool) (last : state) (body : expr)
    (vars : env) (hp : heap) (rd : read_record) :
  inv names_of is_bound last ->
  binds_no_static body = true ->
  inv names_of is_bound
    (State body vars [] (state_statics last) hp rd).
Proof.
  intros (_ & _ & _ & _ & _ & Hcode) Hno.
  destruct
    (statics_nowhere names_of body
       (no_static_nowhere body Hno))
    as (P & _ & N).
  refine (conj P (conj _ (conj I (conj _ (conj _ Hcode))))); simpl;
    rewrite N.
  - constructor.
  - intros n [].
  - intros n [[] | []].
Qed.

Lemma never_faults_after (u : unit_interface) (e : expr)
    (statics : static_env) (hp : heap) (rd : read_record) (n : nat)
      (last : state)
    (body : expr) (vars : env) (hp' : heap) (rd' : read_record) (m : nat)
      (s : state) :
  well_formed_unit u e = true ->
  static_code statics = [] ->
  static_names_free_in statics e = true ->
  run n (State e [] [] statics hp rd) = Step last ->
  binds_no_static body = true ->
  run m (State body vars [] (state_statics last) hp' rd') = Step s ->
  static_fault s = false.
Proof.
  intros Hwf Hno_code Hfree Hlast Hno Hrun.
  pose proof (well_formed_unit_statics_outside_loops_and_code u e Hwf) as P.
  pose proof (well_formed_unit_code_ids_distinct u e Hwf) as Dcode.
  pose proof (well_formed_unit_symbols_distinct u e Hwf) as Dsymbols.
  apply inv_never_faults.
  - refine
      (inv_run code_ids_of_binding code_id_bound
         code_bound_after m _ s
         (inv_after code_ids_of_binding code_id_bound last body
            vars hp' rd' _ Hno)
         Hrun).
    exact
      (inv_run code_ids_of_binding code_id_bound
         code_bound_after n _ last
         (inv_from code_ids_of_binding code_id_bound e statics
            hp rd (free_in_code_ids statics e Hfree) Hno_code P Dcode)
         Hlast).
  - refine
      (inv_run symbols_of_binding symbol_bound
         symbol_bound_after m _ s
         (inv_after symbols_of_binding symbol_bound last body
            vars hp' rd' _ Hno)
         Hrun).
    exact
      (inv_run symbols_of_binding symbol_bound
         symbol_bound_after n _ last
         (inv_from symbols_of_binding symbol_bound e statics hp rd
            (free_in_symbols statics e Hfree) Hno_code P Dsymbols)
         Hlast).
Qed.

(* ======================================================================== *)
(* No variable is read that has no value                                    *)
(* ======================================================================== *)

Lemma well_formed_names_bound_from (u : unit_interface) (e : expr)
    (statics : static_env) (hp : heap) (rd : read_record) (n : nat) :
  well_formed_unit u e = true ->
  static_code statics = [] ->
  match run n (State e [] [] statics hp rd) with
  | Step s => reads_unbound_variable s = false
  | Exit k _ _ _ => k = unit_return u ∨ k = unit_exn u
  | Undefined => True
  end.
Proof.
  intros Hwf Hno_code.
  assert (Hclosed : code_all_closed statics).
  { unfold code_all_closed. rewrite Hno_code. apply Forall_nil. }
  pose proof
    (run_in_scope [unit_return u; unit_exn u] n _
       (state_in_scope_intro [unit_return u; unit_exn u] e [] [] statics hp rd
          (well_formed_unit_names_in_scope u e Hwf)
          (well_formed_unit_code_closed u e Hwf) I Hclosed))
    as Hrun.
  destruct (run n (State e [] [] statics hp rd)) as [s | k vs | ];
    [ exact (in_scope_reads_bound _ s Hrun) | | exact I ].
  apply among_In in Hrun.
  destruct Hrun as [<- | [<- | []]]; [ left | right ]; reflexivity.
Qed.

(* ======================================================================== *)
(* A simulation gives what an exit is compared by                           *)
(* ======================================================================== *)

(* Two states in a simulation in which a step of the first is answered
   by one or more of the second (FlambdaSimulationProofs.v), where the
   values of two exits that answer each other have the same constants
   and the same locations: the second comes to what the first comes to,
   up to what an exit is compared by
   (FlambdaObserversSpec.observed_refines). The two exits have the same
   heap. *)
Lemma stuttering_observed (R : state -> state -> Prop)
    (Q : state -> list value -> list value -> Prop) :
  (∀ (last : state) (vs vs' : list value),
    Q last vs vs' -> same_scalars vs vs') ->
  (∀ s s' : state, R s s' -> answered R Q (step s) s') ->
  ∀ s s' : state, R s s' -> observed_refines (Step s) (Step s').
Proof.
  intros Hsame R_step s s' Hrel. split.
  - intros k vs hp [n [rd Hn]].
    destruct (stuttering_exit R Q R_step n s s' k vs hp rd_ Hrel Hn)
      as (m & last & vs' & Hm & Hlast & HQ).
    exists vs', hp.
    split;
      [ exists (m + 1), rd; rewrite run_add, Hm; simpl; rewrite Hlast;
        reflexivity
      | ].
    split; [ exact (Hsame last vs vs' HQ) | reflexivity ].
  - intros b Hterm Hb. simpl in Hb |- *.
    destruct (behaves_ended s' b Hterm Hb) as [m Hended].
    destruct (stuttering_ends R Q R_step m s s' b Hrel Hended)
      as [n [Hwrong | (k & vs & hp & rd & Hexit)]].
    + left. exists n. exact Hwrong.
    + right. exists k, vs, hp, n, rd. exact Hexit.
Qed.

(* The same of a simulation that leaves the heaps of two exits to its
   parameter (the second section of FlambdaSimulationProofs.v), where
   two heaps that answer each other bind the same locations, in the
   same order. *)
Lemma stuttering_heaps_observed (R : state -> state -> Prop)
    (Q : state -> list value -> heap -> list value -> heap -> Prop) :
  (∀ (last : state) (vs : list value) (hp : heap) (vs' : list value)
     (hp' : heap),
    Q last vs hp vs' hp' -> same_scalars vs vs' ∧ map fst hp = map fst hp') ->
  (∀ s s' : state, R s s' -> answered_heaps R Q (step s) s') ->
  ∀ s s' : state, R s s' -> observed_refines (Step s) (Step s').
Proof.
  intros Hsame R_step s s' Hrel. split.
  - intros k vs hp [n [rd Hn]].
    destruct (stuttering_heaps_exit R Q R_step n s s' k vs hp rd Hrel Hn)
      as (m & last & vs' & hp' & rd' & Hm & Hlast & HQ).
    exists vs', hp'.
    split;
      [ exists (m + 1), rd'; rewrite run_add, Hm; simpl; rewrite Hlast;
        reflexivity
      | ].
    exact (Hsame last vs hp vs' hp' HQ).
  - intros b Hterm Hb. simpl in Hb |- *.
    destruct (behaves_ended s' b Hterm Hb) as [m Hended].
    destruct (stuttering_heaps_ends R Q R_step m s s' b Hrel Hended)
      as [n [Hwrong | (k & vs & hp & rd & Hexit)]].
    + left. exists n. exact Hwrong.
    + right. exists k, vs, hp, n, rd. exact Hexit.
Qed.

(* Two states of which the second refines the first come to results
   that differ in nothing that an exit is compared by: their exits are
   equal. *)
Lemma refines_observed (s s' : state) :
  refines s s' -> observed_refines (Step s) (Step s').
Proof.
  intros [Hexits Hends]. split.
  - intros k vs hp Hsource. exists vs, hp.
    split; [ exact (Hexits k vs hp Hsource) | ].
    split; [ | reflexivity ].
    unfold same_scalars. clear.
    induction vs as [ | v vs IH]; constructor; [ intros _; reflexivity | ].
    exact IH.
  - intros b Hnot Htarget.
    destruct (Hends b Hnot Htarget) as [Hsource | Hwrong];
      [ | left; exact Hwrong ].
    destruct b as [k vs hp | | ];
      [ right; exists k, vs, hp; exact Hsource | left; exact Hsource
      | contradiction Hnot; reflexivity ].
Qed.
