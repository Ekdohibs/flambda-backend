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

(* Proofs of the statements of FlambdaWellFormedSpec.v. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaBindersSpec FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaScopeSpec FlambdaScopeProofs
  FlambdaWellFormedSpec.
Import ListNotations.

Lemma statics_extend_refl (statics : static_env) :
  statics_extend statics statics.
Proof. split; auto. Qed.

Lemma statics_extend_trans (statics statics' statics'' : static_env) :
  statics_extend statics statics' ->
  statics_extend statics' statics'' ->
  statics_extend statics statics''.
Proof. intros [Hcode Hvalues] [Hcode' Hvalues']. split; auto. Qed.

Lemma eval_static_bindings_extend (vars : env)
    (bindings : list static_binding) :
  ∀ statics extended : static_env,
    eval_static_bindings vars bindings statics = Some extended ->
    statics_extend statics extended.
Proof.
  induction bindings as [ | binding bindings IH]; intros statics extended;
    simpl.
  - intros [= <-]. apply statics_extend_refl.
  - destruct binding as [id c | id | sym const].
    + destruct (lookup id (static_code statics)) eqn:Hbound;
        [ discriminate | ].
      intros Heval. refine (statics_extend_trans _ _ _ _ (IH _ _ Heval)).
      split; simpl; [ | auto ].
      intros id' c' Hid'. exact (lookup_fresh id' id c' c _ Hbound Hid').
    + apply IH.
    + destruct (lookup sym (static_values statics)) eqn:Hbound;
        [ discriminate | ].
      destruct (eval_static_const vars const) as [allocated | ]; simpl;
        [ | discriminate ].
      intros Heval. refine (statics_extend_trans _ _ _ _ (IH _ _ Heval)).
      split; simpl; [ auto | ].
      intros s a Hs. exact (lookup_fresh s sym a allocated _ Hbound Hs).
Qed.

Theorem step_extends_statics : step_extends_statics_statement.
Proof.
  intros s s' Hstep. rewrite (state_eta s) in Hstep.
  set (vars := state_vars s) in *. set (conts := state_conts s) in *.
  set (statics := state_statics s) in *.
  destruct (state_expr s)
    as [x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    autorewrite with step in Hstep.
  - destruct (exec_named _ _ rd_ _ d) as [[[v hp'] rd'] | ]; [ | discriminate ].
    destruct (kind_eqb (kind_of_value v) kind); [ | discriminate ].
    injection Hstep as <-. apply statics_extend_refl.
  - destruct (eval_static_bindings vars bindings statics) as [extended | ]
      eqn:Heval; [ | discriminate ].
    injection Hstep as <-.
    exact (eval_static_bindings_extend vars bindings statics extended Heval).
  - injection Hstep as <-. apply statics_extend_refl.
  - destruct (handlers_bind_statics handlers); [ discriminate | ].
    injection Hstep as <-. apply statics_extend_refl.
  - destruct (eval_simple vars (ap_callee ap)) as [callee | ];
      [ | discriminate ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | discriminate ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
      [ | discriminate ].
    rewrite (call_function_statics _ _ _ _ _ _ _ _ _ rd_ _ Hstep).
    apply statics_extend_refl.
  - destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | discriminate ].
    unfold apply_continuation in Hstep.
    rewrite (apply_cont_value_statics _ _ _ _ rd_ _ Hstep).
    apply statics_extend_refl.
  - destruct (eval_simple vars scrutinee) as [v | ]; [ | discriminate ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | discriminate ].
    destruct (find_arm i arms); [ | discriminate ].
    injection Hstep as <-. apply statics_extend_refl.
  - discriminate.
Qed.

Theorem run_extends_statics : run_extends_statics_statement.
Proof.
  intros n. induction n as [ | n IH]; intros s s'; simpl.
  - intros [= <-]. apply statics_extend_refl.
  - destruct (step s) as [next | k vs | ] eqn:Hstep; try discriminate.
    intros Hrun.
    exact (statics_extend_trans _ _ _ (step_extends_statics s next Hstep)
      (IH next s' Hrun)).
Qed.

(* ======================================================================== *)
(* A well-formed unit does nothing that is forbidden                        *)
(* ======================================================================== *)

(* The names that a state may still bind are those of its expression and
   of the handlers of the continuations it can reach. The invariant
   ([inv]) is that they are distinct, that none is bound, and that the
   names of a handler are not among those of the continuations its
   closure holds: a continuation is not in scope in its own handler, so
   once a handler runs, nothing can run it again. *)

Lemma nodup_app_l {A : Type} (l l' : list A) : NoDup (l ++ l') -> NoDup l.
Proof.
  induction l as [ | a l IH]; simpl; intros H; [ apply NoDup_nil | ].
  inversion H as [ | ? ? Hnotin Hrest]. subst.
  apply NoDup_cons; [ | exact (IH Hrest) ].
  intros Hin. apply Hnotin, in_or_app. left. exact Hin.
Qed.

Lemma nodup_app_r {A : Type} (l l' : list A) : NoDup (l ++ l') -> NoDup l'.
Proof.
  induction l as [ | a l IH]; simpl; intros H; [ exact H | ].
  inversion H as [ | ? ? Hnotin Hrest]. exact (IH Hrest).
Qed.

Lemma nodup_app_disjoint {A : Type} (l l' : list A) :
  NoDup (l ++ l') -> ∀ a : A, In a l -> In a l' -> False.
Proof.
  induction l as [ | b l IH]; simpl; intros H a Hin Hin'; [ exact Hin | ].
  inversion H as [ | ? ? Hnotin Hrest]. subst.
  destruct Hin as [-> | Hin]; [ | exact (IH Hrest a Hin Hin') ].
  apply Hnotin, in_or_app. right. exact Hin'.
Qed.

Section Names.

Variable unit_exn : continuation.

(* One sort of static name: code ids, or symbols. *)
Variable names_of : static_binding -> list positive.
Variable is_bound : static_env -> positive -> bool.

Hypothesis bound_after :
  ∀ (vars : env) (bindings : list static_binding)
    (statics statics' : static_env) (n : positive),
    eval_static_bindings vars bindings statics = Some statics' ->
    is_bound statics' n = true ->
    is_bound statics n = true ∨ In n (flat_map names_of bindings).

(* An expression that is not at toplevel has no [Let_static]. *)
Fixpoint statics_nowhere (e : expr) :
  statics_at_toplevel unit_exn false e = true ->
  statics_at_toplevel unit_exn true e = true
  ∧ binds_statics e = false
  ∧ static_names names_of e = [].
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    simpl; try (intros _; repeat split; reflexivity).
  - apply statics_nowhere.
  - discriminate.
  - destruct h as [params metadata handler]. simpl. intros H.
    apply andb_true_iff in H as [Hh Hb].
    destruct (statics_nowhere handler Hh) as (Ph & Bh & Nh).
    destruct (statics_nowhere body Hb) as (Pb & Bb & Nb).
    rewrite Pb, Bh, Bb, Nh, Nb.
    match goal with
    | |- context [statics_at_toplevel unit_exn ?toplevel handler] =>
      destruct toplevel
    end;
      [ rewrite Ph | rewrite Hh ]; repeat split; reflexivity.
  - intros H. apply andb_true_iff in H as [Hhs Hb].
    destruct (statics_nowhere body Hb) as (Pb & Bb & Nb).
    rewrite Hhs, Pb, Bb, Nb.
    assert (Hnone :
      existsb (fun '(_, h) => handler_binds_statics h) handlers = false
      ∧ flat_map (fun '(_, h) => handler_static_names names_of h) handlers
        = []).
    { clear Hb Pb Bb Nb.
      induction handlers as [ | [k [params metadata handler]] handlers IH];
        simpl; [ auto | ].
      simpl in Hhs. apply andb_true_iff in Hhs as [Hh Hrest].
      destruct (statics_nowhere handler Hh) as (_ & Bh & Nh).
      destruct (IH Hrest) as [Bs Ns].
      rewrite Bh, Nh, Bs, Ns. auto. }
    destruct Hnone as [Bs Ns]. rewrite Bs, Ns. repeat split; reflexivity.
Qed.

Lemma to_toplevel (toplevel : bool) (e : expr) :
  statics_at_toplevel unit_exn toplevel e = true ->
  statics_at_toplevel unit_exn true e = true.
Proof.
  destruct toplevel; [ auto | ]. intros H. apply (statics_nowhere e H).
Qed.

Definition handlers_nowhere
    (handlers : list (continuation * continuation_handler)) : Prop :=
  forallb (fun '(_, h) => handler_statics_at_toplevel unit_exn false h)
    handlers
  = true.

Lemma handlers_nowhere_facts
    (handlers : list (continuation * continuation_handler)) :
  handlers_nowhere handlers ->
  handlers_bind_statics handlers = false
  ∧ flat_map (fun '(_, h) => handler_static_names names_of h) handlers = [].
Proof.
  unfold handlers_nowhere, handlers_bind_statics.
  induction handlers as [ | [k [params metadata handler]] handlers IH];
    simpl; [ auto | ].
  intros H. apply andb_true_iff in H as [Hh Hrest].
  destruct (statics_nowhere handler Hh) as (_ & Bh & Nh).
  destruct (IH Hrest) as [Bs Ns]. rewrite Bh, Nh, Bs, Ns. auto.
Qed.

(* The names that the handlers of a continuation, and of those it holds,
   may bind. *)
Fixpoint cont_names (c : cont_value) : list positive :=
  match c with
  | Outer _ => []
  | Closure h _ conts _ =>
    handler_static_names names_of h
    ++ flat_map (fun kc => cont_names (snd kc)) conts
  | Extra_args c _ => cont_names c
  | Over_application _ _ _ return_to exn_to =>
    cont_names return_to ++ cont_names exn_to
  end.

Definition conts_names (conts : cont_env) : list positive :=
  flat_map (fun kc => cont_names (snd kc)) conts.

Fixpoint cont_ok (c : cont_value) : Prop :=
  match c with
  | Outer _ => True
  | Closure h _ conts recursive =>
    handler_statics_at_toplevel unit_exn true h = true
    ∧ NoDup (handler_static_names names_of h)
    ∧ (∀ n : positive,
        In n (handler_static_names names_of h) ->
        ~ In n (flat_map (fun kc => cont_names (snd kc)) conts))
    ∧ handlers_nowhere recursive
    ∧ fold_right (fun kc acc => cont_ok (snd kc) ∧ acc) True conts
  | Extra_args c _ => cont_ok c
  | Over_application _ _ _ return_to exn_to =>
    cont_ok return_to ∧ cont_ok exn_to
  end.

(* Extra arguments change nothing: what holds of a continuation holds of
   the handler or the way out that it is in the end. *)
Lemma cont_target_ok (c : cont_value) :
  cont_ok c ->
  cont_ok (cont_target c) ∧ cont_names (cont_target c) = cont_names c.
Proof.
  induction c
    as [h def_vars def_conts recursive | k | c IH extra
       | remaining remaining_arity return_arity return_to _ exn_to _];
    simpl; [ auto | auto | exact IH | auto ].
Qed.

Lemma with_extra_args_ok (c : cont_value) (extra : list value) :
  cont_ok c ->
  cont_ok (with_extra_args c extra)
  ∧ cont_names (with_extra_args c extra) = cont_names c.
Proof. destruct extra; simpl; auto. Qed.

Definition conts_ok (conts : cont_env) : Prop :=
  fold_right (fun kc acc => cont_ok (snd kc) ∧ acc) True conts.

Lemma conts_ok_app (conts conts' : cont_env) :
  conts_ok conts -> conts_ok conts' -> conts_ok (conts ++ conts').
Proof.
  unfold conts_ok. induction conts as [ | kc conts IH]; simpl; [ auto | ].
  intros [Hc Hrest] H'. split; [ exact Hc | exact (IH Hrest H') ].
Qed.

Lemma conts_names_app (conts conts' : cont_env) :
  conts_names (conts ++ conts') = conts_names conts ++ conts_names conts'.
Proof. unfold conts_names. apply flat_map_app. Qed.

Lemma resolve_ok (k : continuation) (conts : cont_env) :
  conts_ok conts ->
  cont_ok (resolve k conts)
  ∧ ∀ n : positive,
      In n (cont_names (resolve k conts)) -> In n (conts_names conts).
Proof.
  unfold resolve, conts_ok, conts_names.
  induction conts as [ | [k' c] conts IH]; simpl.
  - intros _. split; [ exact I | intros n [] ].
  - intros [Hc Hrest]. destruct (Pos.eqb k k').
    + split; [ exact Hc | ]. intros n Hn. apply in_or_app. left. exact Hn.
    + destruct (IH Hrest) as [Hok Hnames]. split; [ exact Hok | ].
      intros n Hn. apply in_or_app. right. exact (Hnames n Hn).
Qed.

(* The closures of recursive handlers: they bind nothing themselves. *)
Lemma rec_closures_ok_aux (vars : env) (conts : cont_env)
    (all handlers : list (continuation * continuation_handler)) :
  handlers_nowhere all ->
  conts_ok conts ->
  handlers_nowhere handlers ->
  conts_ok (map (fun '(k, h) => (k, Closure h vars conts all)) handlers)
  ∧ ∀ n : positive,
      In n
        (conts_names
           (map (fun '(k, h) => (k, Closure h vars conts all)) handlers)) ->
      In n (conts_names conts).
Proof.
  intros Hall Hconts. unfold handlers_nowhere, conts_ok, conts_names.
  induction handlers as [ | [k [params metadata handler]] handlers IH];
    simpl.
  - intros _. split; [ exact I | intros n [] ].
  - intros H. apply andb_true_iff in H as [Hh Hrest].
    destruct (statics_nowhere handler Hh) as (Ph & _ & Nh).
    destruct (IH Hrest) as [Hok Hnames]. rewrite Nh. simpl. split.
    + split; [ | exact Hok ].
      split; [ exact Ph | ]. split; [ apply NoDup_nil | ].
      split; [ intros n [] | ]. split; [ exact Hall | exact Hconts ].
    + intros n Hn. apply in_app_or in Hn.
      destruct Hn as [Hn | Hn]; [ exact Hn | exact (Hnames n Hn) ].
Qed.

Lemma rec_closures_ok (vars : env) (conts : cont_env)
    (handlers : list (continuation * continuation_handler)) :
  handlers_nowhere handlers ->
  conts_ok conts ->
  conts_ok (rec_closures handlers vars conts)
  ∧ ∀ n : positive,
      In n (conts_names (rec_closures handlers vars conts)) ->
      In n (conts_names conts).
Proof.
  intros Hhandlers Hconts.
  exact (rec_closures_ok_aux vars conts handlers handlers Hhandlers Hconts
    Hhandlers).
Qed.

Definition code_ok (statics : static_env) : Prop :=
  ∀ (id : code_id) (c : code),
    lookup id (static_code statics) = Some c ->
    code_statics_at_toplevel unit_exn c = true.

Lemma eval_static_bindings_code_ok (vars : env)
    (bindings : list static_binding) :
  ∀ statics statics' : static_env,
    forallb (binding_statics_at_toplevel unit_exn) bindings = true ->
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

Definition inv (s : state) : Prop :=
  statics_at_toplevel unit_exn true (state_expr s) = true
  ∧ NoDup (static_names names_of (state_expr s))
  ∧ conts_ok (state_conts s)
  ∧ (∀ n : positive,
      In n (static_names names_of (state_expr s)) ->
      ~ In n (conts_names (state_conts s)))
  ∧ (∀ n : positive,
      In n (static_names names_of (state_expr s))
      ∨ In n (conts_names (state_conts s)) ->
      is_bound (state_statics s) n = false)
  ∧ code_ok (state_statics s).

(* What is asked of a continuation for the state that it leads to to
   satisfy the invariant. *)
Definition cont_inv (statics : static_env) (c : cont_value) : Prop :=
  cont_ok c
  ∧ ∀ n : positive, In n (cont_names c) -> is_bound statics n = false.

Lemma cont_inv_over (statics : static_env) (remaining : list value)
    (remaining_arity return_arity : arity) (return_to exn_to : cont_value) :
  cont_inv statics return_to ->
  cont_inv statics exn_to ->
  cont_inv statics
    (Over_application remaining remaining_arity return_arity return_to
       exn_to).
Proof.
  intros [C1 N1] [C2 N2]. split; [ exact (conj C1 C2) | ].
  intros n Hn. simpl in Hn. apply in_app_or in Hn.
  destruct Hn as [Hn | Hn]; [ exact (N1 n Hn) | exact (N2 n Hn) ].
Qed.

Lemma call_code_inv (closure : value) (called : code_id) (c : code)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
      (rd : read_record)
      (s' : state) :
  code_ok statics ->
  lookup called (static_code statics) = Some c ->
  cont_inv statics return_to ->
  cont_inv statics exn_to ->
  call_code closure c vs args_arity return_arity return_to exn_to statics hp rd
  = Step s' ->
  inv s'.
Proof.
  intros S Hlookup [C1 N1] [C2 N2] Hstep.
  apply call_code_step in Hstep as (callee_vars & _ & _ & _ & _ & _ & ->).
  unfold inv. simpl_state.
  pose proof (S called c Hlookup) as Hc.
  destruct c as [params my_closure ret exn metadata body]. simpl in Hc |- *.
  destruct (statics_nowhere body Hc) as (Pbody & _ & Nbody).
  rewrite Nbody.
  refine
    (conj Pbody
       (conj (NoDup_nil _)
          (conj (conj C1 (conj C2 I)) (conj _ (conj _ S))))).
  - intros n [].
  - intros n [[] | Hn].
    unfold conts_names in Hn. simpl in Hn. rewrite app_nil_r in Hn.
    apply in_app_or in Hn.
    destruct Hn as [Hn | Hn]; [ exact (N1 n Hn) | exact (N2 n Hn) ].
Qed.

Lemma run_call_inv (return_partial : value -> step_result)
    (kind : call_kind) (shape : option call_shape) (return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
      (rd : read_record)
      (s' : state) :
  code_ok statics ->
  shape_calls statics kind shape ->
  cont_inv statics return_to ->
  cont_inv statics exn_to ->
  (∀ partial : value, return_partial partial = Step s' -> inv s') ->
  run_call return_partial shape return_arity return_to exn_to statics hp rd
  = Step s' ->
  inv s'.
Proof.
  intros S Hcalls Hreturn Hexn Hpartial.
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ];
    simpl in Hcalls |- *.
  - destruct Hcalls as (called & slots & _ & Hlookup & _).
    exact (call_code_inv closure called c vs _ _ _ _ statics _ rd_ s' S Hlookup
             Hreturn Hexn).
  - destruct Hcalls as (called & slots & _ & Hlookup & _).
    apply
      (call_code_inv closure called c vs _ _ _ _ statics _ rd_ s' S Hlookup);
      [ apply cont_inv_over; assumption | exact Hexn ].
  - apply Hpartial.
  - discriminate.
Qed.

Lemma apply_cont_value_inv (c : cont_value) :
  ∀ (vs : list value) (statics : static_env) (hp : heap) (rd : read_record)
    (s' : state),
    code_ok statics ->
    cont_inv statics c ->
    apply_cont_value c vs statics hp rd = Step s' ->
    inv s'.
Proof.
  induction c
    as [[params metadata handler] def_vars def_conts recursive | k
       | c IHc extra
       | remaining remaining_arity return_arity return_to exn_to IHreturn _]
    using cont_value_over_ind;
    intros vs statics hp rd s' S [Cr Nr] Hstep.
  - simpl in Hstep.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | discriminate ].
    destruct (bind_params params vs def_vars); [ | discriminate ].
    injection Hstep as <-. unfold inv. simpl_state. simpl in Cr, Nr.
    destruct Cr as (Ph & Dh & Xh & Hrec & Cdef).
    destruct (rec_closures_ok def_vars def_conts recursive Hrec Cdef)
      as [Crec Nrec].
    refine
      (conj Ph
         (conj Dh (conj (conts_ok_app _ _ Crec Cdef) (conj _ (conj _ S))))).
    + intros n Hn Hin. rewrite conts_names_app in Hin.
      apply in_app_or in Hin. destruct Hin as [Hin | Hin];
        [ apply Nrec in Hin | ]; exact (Xh n Hn Hin).
    + intros n Hn. apply Nr.
      destruct Hn as [Hn | Hn]; apply in_or_app; [ left; exact Hn | right ].
      rewrite conts_names_app in Hn. apply in_app_or in Hn.
      destruct Hn as [Hn | Hn]; [ exact (Nrec n Hn) | exact Hn ].
  - discriminate.
  - exact (IHc (vs ++ extra) statics hp rd_ s' S (conj Cr Nr) Hstep).
  - rewrite apply_cont_value_over_application in Hstep.
    destruct vs as [ | v [ | v' vs]]; [ discriminate | | discriminate ].
    simpl in Cr, Nr. destruct Cr as [C1 C2].
    assert (Hreturn : cont_inv statics return_to).
    { split; [ exact C1 | ].
      intros n Hn. apply Nr, in_or_app. left. exact Hn. }
    assert (Hexn : cont_inv statics exn_to).
    { split; [ exact C2 | ].
      intros n Hn. apply Nr, in_or_app. right. exact Hn. }
    unfold call_function in Hstep.
    apply (run_call_inv _ Indirect _ _ _ _ statics _ rd_ s' S
             (shape_of_call_calls _ _ _ _ _ _) Hreturn Hexn)
      in Hstep; [ exact Hstep | ].
    intros partial Hpartial.
    exact (IHreturn [partial] statics hp rd_ s' S Hreturn Hpartial).
Qed.

Lemma call_function_inv (callee : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
      (rd : read_record)
      (s' : state) :
  code_ok statics ->
  cont_inv statics return_to ->
  cont_inv statics exn_to ->
  call_function callee kind vs args_arity return_arity return_to exn_to
    statics hp rd
  = Step s' ->
  inv s'.
Proof.
  intros S Hreturn Hexn. unfold call_function.
  apply (run_call_inv _ kind _ _ _ _ statics _ rd_ s' S
           (shape_of_call_calls _ _ _ _ _ _) Hreturn Hexn).
  intros partial. exact
    (apply_cont_value_inv return_to _ statics _ rd_ s' S Hreturn).
Qed.

Lemma inv_step (s s' : state) : inv s -> step s = Step s' -> inv s'.
Proof.
  intros (P & D & C & X & F & S) Hstep. rewrite (state_eta s) in Hstep.
  set (vars := state_vars s) in *. set (conts := state_conts s) in *.
  set (statics := state_statics s) in *.
  destruct (state_expr s)
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    autorewrite with step in Hstep.
  - (* Let *)
    destruct (exec_named _ _ rd_ _ d) as [[[v hp'] rd'] | ]; [ | discriminate ].
    destruct (kind_eqb (kind_of_value v) kind); [ | discriminate ].
    injection Hstep as <-. unfold inv. simpl_state. simpl in P, D, X, F.
    exact (conj P (conj D (conj C (conj X (conj F S))))).
  - (* Let_static *)
    destruct (eval_static_bindings vars bindings statics) as [statics' | ]
      eqn:Heval; [ | discriminate ].
    injection Hstep as <-. unfold inv. simpl_state. simpl in P, D, X, F.
    apply andb_true_iff in P as [Pbindings Pbody].
    refine
      (conj Pbody (conj (nodup_app_r _ _ D) (conj C (conj _ (conj _ _))))).
    + intros n Hn. apply X, in_or_app. right. exact Hn.
    + intros n Hn. destruct (is_bound statics' n) eqn:Hbound;
        [ exfalso | reflexivity ].
      destruct (bound_after vars bindings statics statics' n Heval Hbound)
        as [Hold | Hnew].
      * assert (Hfresh : is_bound statics n = false).
        { apply F. destruct Hn as [Hn | Hn]; [ left | right; exact Hn ].
          apply in_or_app. right. exact Hn. }
        congruence.
      * destruct Hn as [Hn | Hn].
        -- exact (nodup_app_disjoint _ _ D n Hnew Hn).
        -- apply (X n); [ apply in_or_app; left; exact Hnew | exact Hn ].
    + exact (eval_static_bindings_code_ok vars bindings statics statics'
        Pbindings S Heval).
  - (* Let_cont_non_recursive *)
    injection Hstep as <-. unfold inv. simpl_state.
    destruct h as [params metadata handler]. simpl in P, D, X, F.
    apply andb_true_iff in P as [Ph Pbody]. apply to_toplevel in Ph.
    refine
      (conj Pbody
         (conj (nodup_app_r _ _ D) (conj (conj _ C) (conj _ (conj _ S))))).
    + simpl.
      refine (conj Ph (conj (nodup_app_l _ _ D) (conj _ (conj eq_refl C)))).
      intros n Hn. apply X, in_or_app. left. exact Hn.
    + intros n Hn Hin. unfold conts_names in Hin. simpl in Hin.
      apply in_app_or in Hin. destruct Hin as [Hin | Hin];
        [ apply in_app_or in Hin; destruct Hin as [Hin | Hin] | ].
      * exact (nodup_app_disjoint _ _ D n Hin Hn).
      * apply (X n); [ apply in_or_app; right; exact Hn | exact Hin ].
      * apply (X n); [ apply in_or_app; right; exact Hn | exact Hin ].
    + intros n Hn. apply F. destruct Hn as [Hn | Hn].
      * left. apply in_or_app. right. exact Hn.
      * unfold conts_names in Hn. simpl in Hn.
        apply in_app_or in Hn. destruct Hn as [Hn | Hn];
          [ apply in_app_or in Hn; destruct Hn as [Hn | Hn] | ].
        -- left. apply in_or_app. left. exact Hn.
        -- right. exact Hn.
        -- right. exact Hn.
  - (* Let_cont_recursive *)
    destruct (handlers_bind_statics handlers); [ discriminate | ].
    injection Hstep as <-. unfold inv. simpl_state. simpl in P, D, X, F.
    apply andb_true_iff in P as [Phandlers Pbody].
    destruct (handlers_nowhere_facts handlers Phandlers) as [_ Hnone].
    rewrite Hnone in D, X, F. simpl in D, X, F.
    destruct (rec_closures_ok vars conts handlers Phandlers C)
      as [Crec Nrec].
    refine
      (conj Pbody
         (conj D (conj (conts_ok_app _ _ Crec C) (conj _ (conj _ S))))).
    + intros n Hn Hin. rewrite conts_names_app in Hin.
      apply in_app_or in Hin. destruct Hin as [Hin | Hin];
        [ apply Nrec in Hin | ]; exact (X n Hn Hin).
    + intros n Hn. apply F. destruct Hn as [Hn | Hn]; [ left; exact Hn | ].
      right. rewrite conts_names_app in Hn. apply in_app_or in Hn.
      destruct Hn as [Hn | Hn]; [ exact (Nrec n Hn) | exact Hn ].
  - (* Apply *)
    destruct (eval_simple vars (ap_callee ap)) as [callee | ];
      [ | discriminate ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | discriminate ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
      [ | discriminate ].
    destruct (resolve_ok (ap_continuation ap) conts C) as [C1 N1].
    destruct (resolve_ok (ap_exn_continuation ap) conts C) as [C2 N2].
    destruct (with_extra_args_ok _ extra C2) as [C2' N2'].
    rewrite <- N2' in N2.
    apply (call_function_inv _ _ _ _ _ _ _ statics _ rd_ s' S) in Hstep;
      [ exact Hstep | | ];
      (split; [ assumption | ]; intros n Hn; apply F; right; auto).
  - (* Apply_cont *)
    destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | discriminate ].
    unfold apply_continuation in Hstep.
    destruct (resolve_ok (ac_continuation ac) conts C) as [Cr Nr].
    exact (apply_cont_value_inv _ vs statics _ rd_ s' S
             (conj Cr (fun n Hn => F n (or_intror (Nr n Hn)))) Hstep).
  - (* Switch *)
    destruct (eval_simple vars scrutinee) as [v | ]; [ | discriminate ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | discriminate ].
    destruct (find_arm i arms); [ | discriminate ].
    injection Hstep as <-. unfold inv. simpl_state.
    refine
      (conj eq_refl (conj (NoDup_nil _) (conj C (conj _ (conj _ S))))).
    + intros n [].
    + intros n [[] | Hn]. apply F. right. exact Hn.
  - discriminate.
Qed.

Lemma inv_run (n : nat) :
  ∀ s s' : state, inv s -> run n s = Step s' -> inv s'.
Proof.
  induction n as [ | n IH]; intros s s' Hinv; simpl.
  - intros [= <-]. exact Hinv.
  - destruct (step s) as [next | k vs | ] eqn:Hstep; try discriminate.
    exact (IH next s' (inv_step s next Hinv Hstep)).
Qed.

(* What the invariant rules out. *)
Lemma inv_no_rebinding (bindings : list static_binding) (body : expr)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) :
  inv (State (Let_static bindings body) vars conts statics hp rd) ->
  rebinds names_of (is_bound statics) bindings = false.
Proof.
  intros (_ & D & _ & _ & F & _). simpl in D, F. unfold rebinds.
  apply orb_false_iff. split.
  - apply negb_false_iff, distinct_NoDup. exact (nodup_app_l _ _ D).
  - destruct (existsb (is_bound statics) (flat_map names_of bindings))
      eqn:Hexists; [ | reflexivity ].
    apply existsb_exists in Hexists as [n [Hin Hbound]].
    rewrite (F n (or_introl (in_or_app _ _ n (or_introl Hin)))) in Hbound.
    discriminate.
Qed.

Lemma inv_no_recursive_statics
    (handlers : list (continuation * continuation_handler)) (body : expr)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) :
  inv (State (Let_cont_recursive handlers body) vars conts statics hp rd) ->
  handlers_bind_statics handlers = false.
Proof.
  intros (P & _). simpl in P. apply andb_true_iff in P as [Phandlers _].
  apply (handlers_nowhere_facts handlers Phandlers).
Qed.

Lemma inv_no_call_binding_statics (ap : apply_expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record) :
  inv (State (Apply ap) vars conts statics hp rd) ->
  calls_code_binding_statics ap vars statics = false.
Proof.
  intros (_ & _ & _ & _ & _ & S). unfold calls_code_binding_statics.
  destruct (eval_simple vars (ap_callee ap)) as [callee | ];
    [ | reflexivity ].
  cbv zeta.
  destruct (closure_of statics _) as [[called slots] | ];
    [ | reflexivity ].
  destruct (lookup called (static_code statics)) as [c | ] eqn:Hlookup;
    [ | reflexivity ].
  pose proof (S called c Hlookup) as Hc.
  destruct c as [params my_closure ret exn metadata body]. simpl in Hc |- *.
  apply (statics_nowhere body Hc).
Qed.

End Names.

Lemma code_bound_after (vars : env) (bindings : list static_binding) :
  ∀ (statics statics' : static_env) (n : positive),
    eval_static_bindings vars bindings statics = Some statics' ->
    code_id_bound statics' n = true ->
    code_id_bound statics n = true
    ∨ In n (flat_map code_ids_of_binding bindings).
Proof.
  unfold code_id_bound.
  induction bindings as [ | [id c | id | sym const] bindings IH];
    intros statics statics' n; simpl.
  - intros [= <-]. auto.
  - destruct (lookup id (static_code statics)); [ discriminate | ].
    intros Heval Hbound. destruct (IH _ _ n Heval Hbound) as [Hold | Hnew];
      [ | auto ].
    simpl in Hold. destruct (Pos.eqb_spec n id) as [-> | _]; auto.
  - apply IH.
  - destruct (lookup sym (static_values statics)); [ discriminate | ].
    destruct (eval_static_const vars const); simpl; [ | discriminate ].
    intros Heval Hbound. exact (IH _ _ n Heval Hbound).
Qed.

Lemma symbol_bound_after (vars : env) (bindings : list static_binding) :
  ∀ (statics statics' : static_env) (n : positive),
    eval_static_bindings vars bindings statics = Some statics' ->
    symbol_bound statics' n = true ->
    symbol_bound statics n = true
    ∨ In n (flat_map symbols_of_binding bindings).
Proof.
  unfold symbol_bound.
  induction bindings as [ | [id c | id | sym const] bindings IH];
    intros statics statics' n; simpl.
  - intros [= <-]. auto.
  - destruct (lookup id (static_code statics)); [ discriminate | ].
    intros Heval Hbound. exact (IH _ _ n Heval Hbound).
  - apply IH.
  - destruct (lookup sym (static_values statics)); [ discriminate | ].
    destruct (eval_static_const vars const); simpl; [ | discriminate ].
    intros Heval Hbound. destruct (IH _ _ n Heval Hbound) as [Hold | Hnew];
      [ | auto ].
    simpl in Hold. destruct (Pos.eqb_spec n sym) as [-> | _]; auto.
Qed.

(* A well-formed unit, from empty environments, satisfies the invariant
   for either sort of name. *)
Lemma well_formed_inv (unit_exn : continuation)
    (names_of : static_binding -> list positive)
    (is_bound : static_env -> positive -> bool) (e : expr) :
  (∀ n : positive, is_bound empty_static_env n = false) ->
  statics_at_toplevel unit_exn true e = true ->
  distinct (static_names names_of e) = true ->
  inv unit_exn names_of is_bound
    (State e [] [] empty_static_env empty_heap no_reads).
Proof.
  intros Hempty P D.
  refine
    (conj P (conj (proj1 (distinct_NoDup _) D) (conj I (conj _ (conj _ _))))).
  - intros n _ [].
  - intros n _. apply Hempty.
  - intros id c [=].
Qed.

(* ======================================================================== *)
(* The conditions of a well-formed unit, one at a time                      *)
(* ======================================================================== *)

(* A proof that needs a condition of [well_formed_unit] takes it by one
   of these lemmas and does not unfold the definition, so that a
   condition can be added without a change to the proofs that do not
   need the new one. A proof that a given unit is well-formed goes by
   [well_formed_unit_intro], which asks each condition by its name. *)
Local Ltac condition :=
  unfold well_formed_unit; rewrite !andb_true_iff; tauto.

Lemma well_formed_unit_statics_at_toplevel (u : unit_interface) (e : expr) :
  well_formed_unit u e = true ->
  statics_at_toplevel (unit_exn u) true e = true.
Proof. condition. Qed.

Lemma well_formed_unit_code_ids_distinct (u : unit_interface) (e : expr) :
  well_formed_unit u e = true ->
  distinct (static_names code_ids_of_binding e) = true.
Proof. condition. Qed.

Lemma well_formed_unit_symbols_distinct (u : unit_interface) (e : expr) :
  well_formed_unit u e = true ->
  distinct (static_names symbols_of_binding e) = true.
Proof. condition. Qed.

Lemma well_formed_unit_names_in_scope (u : unit_interface) (e : expr) :
  well_formed_unit u e = true ->
  names_in_scope (unit_return u) (unit_exn u) e = true.
Proof. condition. Qed.

Lemma well_formed_unit_binders_distinct (u : unit_interface) (e : expr) :
  well_formed_unit u e = true -> binders_distinct e = true.
Proof. condition. Qed.

Lemma well_formed_unit_code_closed (u : unit_interface) (e : expr) :
  well_formed_unit u e = true -> code_closed e = true.
Proof. condition. Qed.

Lemma well_formed_unit_sets_well_formed (u : unit_interface) (e : expr) :
  well_formed_unit u e = true -> sets_well_formed e = true.
Proof. condition. Qed.

Lemma well_formed_unit_numbers_in_range (u : unit_interface) (e : expr) :
  well_formed_unit u e = true -> numbers_in_range e = true.
Proof. condition. Qed.

Lemma well_formed_unit_exn_handlers_well_formed (u : unit_interface)
    (e : expr) :
  well_formed_unit u e = true ->
  unit_exn_handlers_well_formed (unit_exn u) e = true.
Proof. condition. Qed.

Lemma well_formed_unit_traps_well_formed (u : unit_interface) (e : expr) :
  well_formed_unit u e = true -> traps_well_formed (unit_exn u) e = true.
Proof. condition. Qed.

Lemma well_formed_unit_intro (u : unit_interface) (e : expr) :
  statics_at_toplevel (unit_exn u) true e = true ->
  distinct (static_names code_ids_of_binding e) = true ->
  distinct (static_names symbols_of_binding e) = true ->
  names_in_scope (unit_return u) (unit_exn u) e = true ->
  binders_distinct e = true ->
  code_closed e = true ->
  sets_well_formed e = true ->
  numbers_in_range e = true ->
  unit_exn_handlers_well_formed (unit_exn u) e = true ->
  traps_well_formed (unit_exn u) e = true ->
  well_formed_unit u e = true.
Proof. condition. Qed.

Theorem well_formed_never_faults : well_formed_never_faults_statement.
Proof.
  intros u e n s Hwf Hrun.
  pose proof (well_formed_unit_statics_at_toplevel u e Hwf) as P.
  pose proof (well_formed_unit_code_ids_distinct u e Hwf) as Dcode.
  pose proof (well_formed_unit_symbols_distinct u e Hwf) as Dsymbols.
  pose proof (inv_run (unit_exn u) code_ids_of_binding code_id_bound
    code_bound_after n _ s
    (well_formed_inv (unit_exn u) code_ids_of_binding code_id_bound e
       (fun _ => eq_refl) P Dcode)
    Hrun) as Hcode.
  pose proof (inv_run (unit_exn u) symbols_of_binding symbol_bound
    symbol_bound_after n _ s
    (well_formed_inv (unit_exn u) symbols_of_binding symbol_bound e
       (fun _ => eq_refl) P Dsymbols)
    Hrun) as Hsymbols.
  rewrite (state_eta s) in Hcode, Hsymbols. unfold static_fault.
  destruct (state_expr s)
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    try reflexivity.
  - rewrite (inv_no_rebinding (unit_exn u) _ _ _ _ _ _ _ _ rd_ Hcode),
      (inv_no_rebinding (unit_exn u) _ _ _ _ _ _ _ _ rd_ Hsymbols).
    reflexivity.
  - exact (inv_no_recursive_statics (unit_exn u) _ _ _ _ _ _ _ _ rd_ Hcode).
  - exact (inv_no_call_binding_statics (unit_exn u) _ _ _ _ _ _ _ rd_ Hcode).
Qed.

Theorem well_formed_names_bound : well_formed_names_bound_statement.
Proof.
  intros u e n Hwf.
  exact
    (unit_run (unit_return u) (unit_exn u) e n
       (well_formed_unit_names_in_scope u e Hwf)
       (well_formed_unit_code_closed u e Hwf)).
Qed.
