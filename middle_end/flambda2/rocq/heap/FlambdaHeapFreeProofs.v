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

(* Proofs of the statements of FlambdaHeapFreeSpec.v. Nothing here
   needs review.

   That a unit which passes the test leaves the heap as it found it is
   an invariant of its run ([state_free]): what is still to be run
   passes the test, and so do the handlers that the continuations
   hold and the code that the static environment binds. A step keeps
   it, and changes the heap only at a [Let]
   (FlambdaSemanticsSpec.step_heap_statement), where the defining
   expression is then no primitive of the heap and leaves the heap
   (FlambdaSemanticsSpec.exec_named_heap_statement). No fact about
   values is used. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaHeapSpec FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaNominalSpec FlambdaHeapFreeSpec
  FlambdaEquivarianceTactic.
Import ListNotations.

(* ======================================================================== *)
(* The invariant                                                            *)
(* ======================================================================== *)

Definition code_free (statics : static_env) : Prop :=
  ∀ (id : code_id) (c : code),
    lookup id (static_code statics) = Some c -> code_free_of_heap c = true.

#[local] Set Warnings "-register-all".
Inductive cont_free : cont_value -> Prop :=
  | Outer_free (k : continuation) : cont_free (Outer k)
  | Closure_free (h : continuation_handler) (vars : env) (conts : cont_env)
      (recursive : list (continuation * continuation_handler)) :
    handler_free_of_heap h = true ->
    Forall (fun kc => cont_free (snd kc)) conts ->
    forallb (fun '(_, h') => handler_free_of_heap h') recursive = true ->
    cont_free (Closure h vars conts recursive)
  | Extra_args_free (c : cont_value) (extra : list value) :
    cont_free c -> cont_free (Extra_args c extra)
  | Over_application_free (remaining : list value)
      (remaining_arity return_arity : arity)
      (return_to exn_to : cont_value) :
    cont_free return_to ->
    cont_free exn_to ->
    cont_free
      (Over_application remaining remaining_arity return_arity return_to
         exn_to).

Definition conts_free (conts : cont_env) : Prop :=
  Forall (fun kc => cont_free (snd kc)) conts.

Definition state_free (st : state) : Prop :=
  expr_free_of_heap (state_expr st) = true
  ∧ conts_free (state_conts st)
  ∧ code_free (state_statics st).

Lemma statics_free_of_heap_code (statics : static_env) :
  statics_free_of_heap statics = true -> code_free statics.
Proof.
  unfold statics_free_of_heap. rewrite forallb_forall.
  intros Hall id c Hc. exact (Hall (id, c) (lookup_in id c _ Hc)).
Qed.

Lemma eval_static_bindings_free (vars : env) :
  ∀ (bindings : list static_binding) (statics extended : static_env),
    forallb binding_free_of_heap bindings = true ->
    code_free statics ->
    eval_static_bindings vars bindings statics = Some extended ->
    code_free extended.
Proof.
  induction bindings as [ | binding bindings IH]; intros statics extended;
    simpl.
  - intros _ Hstatics [= <-]. exact Hstatics.
  - intros Hbindings Hcode.
    apply andb_true_iff in Hbindings as [Hbinding Hbindings].
    destruct binding as [id' c | id' | sym const]; simpl in Hbinding.
    + destruct (lookup id' (static_code statics)); [ discriminate | ].
      apply (IH _ _ Hbindings).
      intros id'' c'. simpl.
      destruct (Pos.eqb id'' id');
        [ intros [= <-]; exact Hbinding | apply Hcode ].
    + exact (IH _ _ Hbindings Hcode).
    + destruct (lookup sym (static_values statics)); [ discriminate | ].
      destruct (eval_static_const vars const) as [allocated | ];
        [ | discriminate ].
      simpl. apply (IH _ _ Hbindings). exact Hcode.
Qed.

Lemma lookup_free (k : continuation) (conts : cont_env) (c : cont_value) :
  conts_free conts -> lookup k conts = Some c -> cont_free c.
Proof.
  induction 1 as [ | [k' c'] conts Hc _ IH]; simpl; [ discriminate | ].
  destruct (Pos.eqb k k'); [ intros [= <-]; exact Hc | exact IH ].
Qed.

Lemma resolve_free (k : continuation) (conts : cont_env) :
  conts_free conts -> cont_free (resolve k conts).
Proof.
  intros Hconts. unfold resolve.
  destruct (lookup k conts) as [c | ] eqn:Hk;
    [ exact (lookup_free k conts c Hconts Hk) | apply Outer_free ].
Qed.

Lemma rec_closures_free (vars : env) (conts : cont_env)
    (recursive : list (continuation * continuation_handler)) :
  forallb (fun '(_, h) => handler_free_of_heap h) recursive = true ->
  conts_free conts ->
  conts_free (rec_closures recursive vars conts ++ conts).
Proof.
  intros Hrecursive Hconts. apply Forall_app. split; [ | exact Hconts ].
  unfold rec_closures. apply Forall_forall. intros [k c] Hin.
  apply in_map_iff in Hin as ([k' h] & [= <- <-] & Hin). simpl.
  apply Closure_free; [ | exact Hconts | exact Hrecursive ].
  rewrite forallb_forall in Hrecursive. exact (Hrecursive (k', h) Hin).
Qed.

Lemma with_extra_args_free (c : cont_value) (extra : list value) :
  cont_free c -> cont_free (with_extra_args c extra).
Proof.
  intros Hc. destruct extra as [ | v extra]; [ exact Hc | ].
  apply Extra_args_free. exact Hc.
Qed.

(* ======================================================================== *)
(* A step keeps it                                                          *)
(* ======================================================================== *)

Lemma call_code_free (closure : value) (called : code_id) (c : code)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
    (rd : read_record) (st : state) :
  code_free statics ->
  lookup called (static_code statics) = Some c ->
  cont_free return_to ->
  cont_free exn_to ->
  call_code closure c vs args_arity return_arity return_to exn_to statics hp rd
  = Step st ->
  state_free st.
Proof.
  intros Hstatics Hcode Hreturn Hexn Hstep.
  apply call_code_step in Hstep as (bound & _ & _ & _ & _ & _ & ->).
  split; [ | split; [ | exact Hstatics ] ]; simpl.
  - pose proof (Hstatics called c Hcode) as Hc. destruct c. exact Hc.
  - constructor; [ exact Hreturn | ].
    constructor; [ exact Hexn | constructor ].
Qed.

Lemma run_call_free (return_partial : value -> step_result) (kind : call_kind)
    (shape : option call_shape) (return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
    (rd : read_record) (st : state) :
  code_free statics ->
  shape_calls statics kind shape ->
  cont_free return_to ->
  cont_free exn_to ->
  (∀ partial : value, return_partial partial = Step st -> state_free st) ->
  run_call return_partial shape return_arity return_to exn_to statics hp rd
  = Step st ->
  state_free st.
Proof.
  intros Hstatics Hcalls Hreturn Hexn Hpartial.
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ];
    simpl in Hcalls |- *.
  - destruct Hcalls as (called & slots & _ & Hcode & _).
    exact (call_code_free closure called c vs _ _ _ _ statics _ _ st Hstatics
             Hcode Hreturn Hexn).
  - destruct Hcalls as (called & slots & _ & Hcode & _).
    apply (call_code_free closure called c vs _ _ _ _ statics _ _ st Hstatics
             Hcode);
      [ apply Over_application_free; assumption | exact Hexn ].
  - exact (Hpartial partial).
  - discriminate.
Qed.

Lemma apply_cont_value_free (c : cont_value) :
  cont_free c ->
  ∀ (vs : list value) (statics : static_env) (hp : heap) (rd : read_record)
    (st : state),
    code_free statics ->
    apply_cont_value c vs statics hp rd = Step st ->
    state_free st.
Proof.
  induction 1
    as [ k | [params metadata handler] dvars dconts recursive Hh Hdconts
             Hrecursive
       | c extra Hc IH
       | remaining remaining_arity return_arity return_to exn_to Hreturn
           IHreturn Hexn _ ];
    intros vs statics hp rd st Hstatics.
  - discriminate.
  - simpl.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | discriminate ].
    destruct (bind_params params vs dvars) as [bound | ]; [ | discriminate ].
    intros [= <-].
    split; [ exact Hh | ]. split; [ | exact Hstatics ].
    apply rec_closures_free; assumption.
  - simpl. apply IH. exact Hstatics.
  - rewrite apply_cont_value_over_application.
    destruct vs as [ | v [ | v' vs]]; [ discriminate | | discriminate ].
    unfold call_function.
    apply (run_call_free _ Indirect _ _ _ _ statics _ _ st Hstatics
             (shape_of_call_calls _ _ _ _ _ _));
      [ exact Hreturn | exact Hexn | ].
    intros partial. apply IHreturn. exact Hstatics.
Qed.

Lemma free_step_components (e : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) (st : state) :
  expr_free_of_heap e = true ->
  conts_free conts ->
  code_free statics ->
  step (State e vars conts statics hp rd) = Step st ->
  state_free st.
Proof.
  intros He Hconts Hstatics.
  destruct e
    as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    simpl in He.
  - (* Let *)
    apply andb_true_iff in He as [_ Hbody]. rewrite step_let_exec.
    destruct (exec_named statics hp rd vars d) as [[[v hp'] rd'] | ];
      [ | discriminate ].
    destruct (kind_eqb (kind_of_value v) xkind); [ | discriminate ].
    intros [= <-]. split; [ exact Hbody | split; assumption ].
  - (* Let_static *)
    apply andb_true_iff in He as [Hbindings Hbody]. rewrite step_let_static.
    destruct (eval_static_bindings vars bindings statics) as [extended | ]
      eqn:Hextended;
      [ | discriminate ].
    intros [= <-].
    split; [ exact Hbody | ]. split; [ exact Hconts | ].
    exact (eval_static_bindings_free vars bindings statics extended Hbindings
             Hstatics Hextended).
  - (* Let_cont_non_recursive *)
    apply andb_true_iff in He as [Hh Hbody].
    rewrite step_let_cont_non_recursive. intros [= <-].
    split; [ exact Hbody | ]. split; [ | exact Hstatics ].
    constructor; [ | exact Hconts ]. simpl.
    apply Closure_free; [ exact Hh | exact Hconts | reflexivity ].
  - (* Let_cont_recursive *)
    apply andb_true_iff in He as [Hhandlers Hbody].
    rewrite step_let_cont_recursive.
    destruct (handlers_bind_statics handlers); [ discriminate | ].
    intros [= <-].
    split; [ exact Hbody | ]. split; [ | exact Hstatics ].
    apply rec_closures_free; assumption.
  - (* Apply *)
    rewrite step_apply.
    destruct (eval_simple vars (ap_callee ap)) as [callee | ];
      [ | discriminate ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | discriminate ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
      [ | discriminate ].
    unfold call_function.
    assert (Hreturn : cont_free (resolve (ap_continuation ap) conts))
      by exact (resolve_free _ conts Hconts).
    apply (run_call_free _ (ap_call_kind ap) _ _ _ _ statics _ _ st Hstatics
             (shape_of_call_calls _ _ _ _ _ _));
      [ exact Hreturn
      | exact (with_extra_args_free _ extra (resolve_free _ conts Hconts)) | ].
    intros partial.
    exact (apply_cont_value_free _ Hreturn _ _ _ _ _ Hstatics).
  - (* Apply_cont *)
    rewrite step_apply_cont.
    destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | discriminate ].
    unfold apply_continuation.
    exact (apply_cont_value_free _
             (resolve_free (ac_continuation ac) conts Hconts) _ _ _ _ _
             Hstatics).
  - (* Switch *)
    rewrite step_switch.
    destruct (eval_simple vars scrutinee) as [v | ]; [ | discriminate ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | discriminate ].
    destruct (find_arm i arms) as [ac | ]; [ | discriminate ].
    intros [= <-]. split; [ reflexivity | split; assumption ].
  - (* Invalid *)
    rewrite step_invalid. discriminate.
Qed.

(* A step keeps the invariant, and the heap. *)
Lemma free_step (st st' : state) :
  state_free st ->
  step st = Step st' ->
  state_free st' ∧ state_heap st' = state_heap st.
Proof.
  intros (He & Hconts & Hstatics) Hstep. split.
  - revert Hstep. rewrite (state_eta st).
    exact (free_step_components _ _ _ _ _ _ st' He Hconts Hstatics).
  - destruct (step_heap st st' Hstep)
      as [Hsame | (x & kind & d & body & v & Hexpr & Hexec)];
      [ exact Hsame | ].
    rewrite Hexpr in He. simpl in He.
    apply andb_true_iff in He as [Hd _]. apply negb_true_iff in Hd.
    refine (exec_named_heap _ _ _ _ _ _ _ _ _ Hexec).
    intros q args ->. discriminate Hd.
Qed.

(* So an exit of the run has the heap that the run started from. *)
Lemma free_run_exit (n : nat) :
  ∀ (st : state) (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record),
    state_free st -> run n st = Exit k vs hp rd -> hp = state_heap st.
Proof.
  induction n as [ | n IH]; intros st k vs hp rd Hfree; simpl;
    [ discriminate | ].
  destruct (step st) as [t | k' vs' hp' rd' | ] eqn:Hstep;
    [ | | discriminate ].
  - destruct (free_step st t Hfree Hstep) as [Hfree' Hheap].
    intros Hexit. rewrite <- Hheap. exact (IH t k vs hp rd Hfree' Hexit).
  - intros [= <- <- <- <-]. exact (step_exit_heap st k' vs' hp' rd' Hstep).
Qed.

Theorem free_of_heap_exits_empty : free_of_heap_exits_empty_statement.
Proof.
  intros e statics k vs hp He Hstatics [n [rd Hn]].
  refine (free_run_exit n _ k vs hp rd _ Hn).
  split; [ exact He | ]. split; [ constructor | ].
  exact (statics_free_of_heap_code statics Hstatics).
Qed.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

(* Proved from the definitions by [equivariance]
   (FlambdaEquivarianceTactic.v). *)
Theorem expr_free_of_heap_rename : expr_free_of_heap_rename_statement.
Proof. equivariance. Qed.
